"""动作覆盖审计 —— 量化 RL 实际学习空间（训练效果差的第一诊断）。

问题文档明确指出："你现在应该先检查一个指标：raw_actor_action 和 executed_action
的差距到底有多大？" 如果差距很大，就是典型的 policy-environment mismatch：
SAC 在优化 actor 输出，但环境真实转移由"被规则修正后的动作"决定 → 训练梯度失真。

本脚本固定策略，逐步记录三段动作并分维度量化覆盖：

    raw_action      actor 原始输出（PSF/Lyapunov 之前）
    safe_action     scheduler.schedule 返回（经 PSF + Lyapunov 投影）
    executed_action info["executed_action"]（再经 env 内 sanitizer / 解析推进 /
                    推进平滑 / 指向兜底 / TX floor / 功率/热/队列投影）

输出：
  - 每段 L2 gap：|raw−safe|（PSF/Lyapunov）、|safe−executed|（env 规则）、|raw−executed|（总）
  - 关键维度（prop=0 / cpu=1 / tx=2 / pointing=8）的平均改写量与"被改写步占比"
  - 各覆盖触发率：PSF 介入、解析推进接管、指向兜底、TX floor、指向模式被改
  - 结论：哪些维度的 RL 决策实际被规则托底（→ 该维 actor 学不动）

诊断用，不需要好模型；未训练/早期 checkpoint 也能看清覆盖结构。
用法：
    python experiments/action_override_audit.py --model checkpoints_optimized/best_optimized.pt --device cuda
    python experiments/action_override_audit.py --random   # 无 checkpoint，用未训练策略看覆盖结构
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from config import DRL_CONFIG, TRAIN_CONFIG
from environment.satellite_env import VLEOSatelliteEnv
from environment.wrappers import DilatedFrameStackWrapper
from scheduler.integrated_scheduler import IntegratedScheduler
from utils.action_space import pointing_mode_from_unit

# 关键可学习维度（其余 3-7 为任务优先级权重，覆盖较少，单独汇总）。
KEY_DIMS = {0: "alpha_prop", 1: "alpha_cpu", 2: "alpha_tx", 8: "pointing_mode"}


def _align(a, n):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    if a.size < n:
        return np.pad(a, (0, n - a.size))
    return a[:n]


def run(args) -> dict:
    from evaluate_optimized import _resolve_device
    device = _resolve_device(args.device)
    k = int(DRL_CONFIG.get("frame_stack", 8))

    scheduler = IntegratedScheduler(device=device, enable_lyapunov=True, use_psf=True)
    loaded = False
    if args.model and os.path.exists(args.model):
        scheduler.load(args.model)
        loaded = True
    elif not args.random:
        raise FileNotFoundError(
            f"未找到 checkpoint: {args.model}。无 checkpoint 想看覆盖结构请加 --random。")

    # 累计器
    raw_safe_l2, safe_exec_l2, raw_exec_l2 = [], [], []
    dim_abs = {d: [] for d in KEY_DIMS}          # 每维 |raw−executed| 绝对改写量
    dim_modified = {d: 0 for d in KEY_DIMS}      # 每维被改写步数
    weight_dims_abs = []                          # 3-7 优先级权重整体改写量
    n_steps = 0
    fire = {
        "psf_intervened": 0,
        "analytic_propulsion_enabled": 0,
        "analytic_propulsion_applied": 0,
        "mission_pointing_fallback_applied": 0,
        "tx_floor_applied": 0,
        "pointing_mode_changed": 0,
    }

    for ep in range(args.episodes):
        base_env = VLEOSatelliteEnv(seed=args.seed + ep)
        env = DilatedFrameStackWrapper(base_env, k=k)
        scheduler.reset_all_safety_stats()
        state = env.reset()
        done = False
        while not done:
            in_window = (env._contact.get("in_window", False)
                         if env._contact is not None else False)
            prop_can_update = True
            if hasattr(env, "step_count") and hasattr(env, "N_PROP_SMOOTH"):
                prop_can_update = (env.step_count % env.N_PROP_SMOOTH == 0)
            action, was_projected, raw_action, psf_meta = scheduler.schedule(
                state, env.energy_queue.value, env.orbit_queue.value,
                env.data_queue.length, env.comm_queue.value,
                in_window=in_window, evaluate=True,
                h=env.altitude_m, soc=env.battery.soc, time_s=env.time_s,
                prop_can_update=prop_can_update,
                orbital_phase=env.orbit_sim.phase,
                tx_capacity_mbps=float((env._contact or {}).get("max_capacity_mbps", 0.0)),
                available_power_w=getattr(env, "available_power_w", None),
                env=env)
            state, reward, done, info = env.step(action, enforce_prop_smoothing=False)

            n = 9
            raw = _align(raw_action, n)
            safe = _align(action, n)
            ex = _align(info.get("executed_action", action), n)

            raw_safe_l2.append(float(np.linalg.norm(raw - safe)))
            safe_exec_l2.append(float(np.linalg.norm(safe - ex)))
            raw_exec_l2.append(float(np.linalg.norm(raw - ex)))

            for d in KEY_DIMS:
                delta = abs(float(raw[d]) - float(ex[d]))
                dim_abs[d].append(delta)
                if delta > 1e-3:
                    dim_modified[d] += 1
            weight_dims_abs.append(float(np.linalg.norm(raw[3:8] - ex[3:8])))

            # 触发率（部分来自 info meta）。
            if bool(was_projected) or float(psf_meta.get("total_modification_l2", 0.0)) > 1e-6:
                fire["psf_intervened"] += 1
            if bool(info.get("analytic_propulsion_controller_enabled", False)):
                fire["analytic_propulsion_enabled"] += 1
            if bool(info.get("analytic_propulsion_applied", False)):
                fire["analytic_propulsion_applied"] += 1
            if bool(info.get("mission_pointing_fallback_applied", False)):
                fire["mission_pointing_fallback_applied"] += 1
            # TX floor：窗口内 raw[2] 明显低于 executed[2] 且 executed 贴近 floor。
            if in_window and (float(ex[2]) - float(raw[2])) > 1e-2:
                fire["tx_floor_applied"] += 1
            if pointing_mode_from_unit(float(raw[8])) != pointing_mode_from_unit(float(ex[8])):
                fire["pointing_mode_changed"] += 1
            n_steps += 1

    def _m(x):
        return float(np.mean(x)) if x else 0.0

    steps = max(n_steps, 1)
    report = {
        "__meta__": {
            "timestamp": datetime.now().isoformat(),
            "model": args.model if loaded else "(untrained/random policy)",
            "loaded_checkpoint": loaded,
            "device": device,
            "episodes": args.episodes,
            "total_steps": n_steps,
            "note": "raw=actor输出, safe=PSF/Lyapunov后, executed=env全部规则后",
        },
        "l2_gap_mean": {
            "raw_to_safe_psf_lyapunov": _m(raw_safe_l2),
            "safe_to_executed_env_rules": _m(safe_exec_l2),
            "raw_to_executed_total": _m(raw_exec_l2),
        },
        "per_dim": {
            KEY_DIMS[d]: {
                "mean_abs_modification": _m(dim_abs[d]),
                "fraction_steps_modified": dim_modified[d] / steps,
            } for d in KEY_DIMS
        },
        "priority_weights_3to7_mean_abs_l2": _m(weight_dims_abs),
        "override_firing_rate": {k: v / steps for k, v in fire.items()},
    }

    # ── 结论：哪些维度 RL 实际学不动（被改写步占比 > 阈值）──
    threshold = 0.30
    heavily_overridden = [
        KEY_DIMS[d] for d in KEY_DIMS
        if dim_modified[d] / steps > threshold
    ]
    report["verdict"] = {
        "heavily_overridden_dims": heavily_overridden,
        "threshold_fraction": threshold,
        "policy_environment_mismatch_risk": (
            "HIGH" if report["l2_gap_mean"]["raw_to_executed_total"] > 0.3
            else "MEDIUM" if report["l2_gap_mean"]["raw_to_executed_total"] > 0.1
            else "LOW"
        ),
    }

    # ── 打印 ──
    print(f"\n{'=' * 78}\n  动作覆盖审计 ({report['__meta__']['model']}, {n_steps} steps)\n{'=' * 78}")
    g = report["l2_gap_mean"]
    print(f"  L2 gap:  raw→safe(PSF/Lya)={g['raw_to_safe_psf_lyapunov']:.4f}  "
          f"safe→exec(env规则)={g['safe_to_executed_env_rules']:.4f}  "
          f"raw→exec(总)={g['raw_to_executed_total']:.4f}")
    print(f"\n  {'维度':<16}{'平均改写量':>12}{'被改写步占比':>14}")
    for name, d in report["per_dim"].items():
        print(f"  {name:<16}{d['mean_abs_modification']:>12.4f}{d['fraction_steps_modified']:>13.1%}")
    print(f"\n  覆盖触发率:")
    for kf, v in report["override_firing_rate"].items():
        print(f"    {kf:<36}{v:>7.1%}")
    print(f"\n  结论: 重度覆盖维度 = {heavily_overridden or '无'}  "
          f"| mismatch 风险 = {report['verdict']['policy_environment_mismatch_risk']}")

    os.makedirs("results", exist_ok=True)
    out = args.output or f"results/action_override_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {out}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="动作覆盖审计 (raw vs executed gap)")
    parser.add_argument("--model", default=os.path.join(
        TRAIN_CONFIG.get("optimized_checkpoint_dir", "checkpoints_optimized/"),
        "best_optimized.pt"))
    parser.add_argument("--random", action="store_true",
                        help="无 checkpoint，用未训练策略仅看覆盖结构")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=int(TRAIN_CONFIG.get("seed", 42)) + 7000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(args)

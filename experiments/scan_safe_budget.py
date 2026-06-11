"""小规模诊断扫描：对比 fallback OFF / 旧激进 fallback ON / 新 SAFE_BUDGET_FALLBACK。

不做大规模盲训：在 frozen checkpoint 上 eval-time 切换兜底逻辑，快速画出
safety / window_util / downlink / proc-dl 的权衡前沿，决定值得带哪套配置去重训。

扫描轴：
  - mode ∈ {fallback_off, aggressive_on, safe_budget}
  - safe_budget 再扫 soft_min_soc ∈ {0.60, 0.65, 0.70}，hard_min_soc 固定 0.50
输出表格列：
  episode_safety_rate, worst_seed_safety, survival_rate, crash_count,
  comm_window_util, downlink_mb, delivered_value, proc_dl_ratio,
  fallback_trigger_rate, fallback_to_downlink_rate

成功标准：episode_safety>=0.90 且 worst_seed>=0.80 且 window_util>=0.20
          且 downlink 明显高于 fallback_off 且 proc_dl<=2.0(或明显低于 aggressive_on) 且 crash≈0。

用法:
  python experiments/scan_safe_budget.py --model checkpoints_optimized/best_optimized.pt \
      --device cuda --seeds 42,43,44 --episodes 2
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

import argparse
import json
import numpy as np

from environment.satellite_env import VLEOSatelliteEnv
from environment.wrappers import DilatedFrameStackWrapper
from scheduler.integrated_scheduler import IntegratedScheduler
from config import (TRAIN_CONFIG, DRL_CONFIG, HARD_RULES_CONFIG,
                    SAFE_BUDGET_FALLBACK_CONFIG)
from utils.action_space import POINTING_DOWNLINK


def _available_power_w(env) -> float:
    base = env.env if hasattr(env, "env") else env
    return float(getattr(base, "_last_available_power_w", 0.0))


def _mission_stage(info: dict) -> str:
    if float(info.get("failure_state", 0.0)) > 0.5:
        return "failure"
    if float(info.get("unsafe_state", 0.0)) > 0.5:
        return "unsafe"
    if float(info.get("warning_state", 0.0)) > 0.5:
        return "warning"
    return "normal"


def rollout_episode(env, scheduler):
    base_env = env.env if hasattr(env, "env") else env
    state = env.reset()
    done = False
    ep_tput = ep_tx = ep_value = 0.0
    win_utils = []
    trig = trig_dl = steps = 0
    ep_safe = True
    crashed = False
    while not done:
        in_window = (base_env._contact.get("in_window", False)
                     if base_env._contact is not None else False)
        prop_can_update = True
        if hasattr(base_env, "step_count") and hasattr(base_env, "N_PROP_SMOOTH"):
            prop_can_update = (base_env.step_count % base_env.N_PROP_SMOOTH == 0)
        action, _, _, _ = scheduler.schedule(
            state, base_env.energy_queue.value, base_env.orbit_queue.value,
            base_env.data_queue.length, base_env.comm_queue.value,
            in_window, evaluate=True,
            h=base_env.altitude_m, soc=base_env.battery.soc,
            time_s=base_env.time_s, prop_can_update=prop_can_update,
            orbital_phase=base_env.orbit_sim.phase,
            tx_capacity_mbps=float((base_env._contact or {}).get("max_capacity_mbps", 0.0)),
            available_power_w=_available_power_w(env), env=base_env)
        state, reward, done, info = env.step(action, enforce_prop_smoothing=False)

        steps += 1
        ep_tput += float(info.get("processed_mb", 0.0))
        dl = float(info.get("delivered_mb", info.get("actual_tx_mb", 0.0)))
        ep_tx += dl
        ep_value += float(info.get("delivered_value", 0.0))
        cap_mb = float(info.get("tx_capacity_mbps", 0.0)) * TRAIN_CONFIG["time_slot_s"] / 8.0
        if bool(info.get("in_window", info.get("in_comm_window", False))) and cap_mb > 1e-9:
            win_utils.append(dl / cap_mb)
        if bool(info.get("mission_pointing_fallback_applied", False)):
            trig += 1
            if int(info.get("mission_pointing_mode_after", -1)) == int(POINTING_DOWNLINK):
                trig_dl += 1
        stage = _mission_stage(info)
        if stage != "normal":
            ep_safe = False
        if stage == "failure":
            crashed = True
    survived = not crashed and steps >= int(TRAIN_CONFIG.get("max_episode_steps", 2160)) * 0.95
    return {
        "ep_safe": ep_safe, "survived": survived, "crashed": crashed,
        "win_util": float(np.mean(win_utils)) if win_utils else 0.0,
        "downlink": ep_tx, "delivered": ep_value,
        "proc_dl": ep_tput / max(ep_tx, 1e-9),
        "trig_rate": trig / max(steps, 1), "trig_dl_rate": trig_dl / max(steps, 1),
    }


def run_config(model, device, seeds, episodes):
    per_seed_safe = []
    eps = []
    for sd in seeds:
        env = DilatedFrameStackWrapper(VLEOSatelliteEnv(seed=sd),
                                       k=DRL_CONFIG.get("frame_stack", 8))
        scheduler = IntegratedScheduler(device=device, enable_lyapunov=True, use_psf=True)
        scheduler.load(model)
        scheduler.reset_all_safety_stats()
        seed_eps = [rollout_episode(env, scheduler) for _ in range(episodes)]
        eps.extend(seed_eps)
        per_seed_safe.append(np.mean([e["ep_safe"] for e in seed_eps]))

    def m(key):
        return float(np.mean([e[key] for e in eps]))
    return {
        "episode_safety_rate": float(np.mean([e["ep_safe"] for e in eps])),
        "worst_seed_safety": float(np.min(per_seed_safe)),
        "survival_rate": float(np.mean([e["survived"] for e in eps])),
        "crash_count": int(np.sum([e["crashed"] for e in eps])),
        "comm_window_util": m("win_util"),
        "downlink_mb": m("downlink"),
        "delivered_value": m("delivered"),
        "proc_dl_ratio": m("proc_dl"),
        "fallback_trigger_rate": m("trig_rate"),
        "fallback_to_downlink_rate": m("trig_dl_rate"),
    }


def set_mode(mode: str, soft: float = 0.65, hard: float = 0.50, aggr_min_soc: float = 0.55):
    """切换兜底配置（eval-time）。"""
    if mode == "fallback_off":
        HARD_RULES_CONFIG["enable_mission_pointing_fallback"] = False
        SAFE_BUDGET_FALLBACK_CONFIG["enabled"] = False
    elif mode == "aggressive_on":
        HARD_RULES_CONFIG["enable_mission_pointing_fallback"] = True
        HARD_RULES_CONFIG["mission_pointing_min_soc"] = aggr_min_soc
        SAFE_BUDGET_FALLBACK_CONFIG["enabled"] = False
    elif mode == "safe_budget":
        HARD_RULES_CONFIG["enable_mission_pointing_fallback"] = False
        SAFE_BUDGET_FALLBACK_CONFIG["enabled"] = True
        SAFE_BUDGET_FALLBACK_CONFIG["soft_min_soc"] = soft
        SAFE_BUDGET_FALLBACK_CONFIG["hard_min_soc"] = hard
    else:
        raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints_optimized/best_optimized.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--soft_sweep", default="0.60,0.65,0.70")
    ap.add_argument("--hard_min_soc", type=float, default=0.50)
    ap.add_argument("--output", default="results/scan_safe_budget.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    softs = [float(s) for s in args.soft_sweep.split(",") if s.strip()]

    configs = [("fallback_off", None), ("aggressive_on", None)]
    configs += [("safe_budget", s) for s in softs]

    rows = []
    for mode, soft in configs:
        label = mode if soft is None else f"safe_budget(soft={soft:.2f},hard={args.hard_min_soc:.2f})"
        set_mode(mode, soft=soft or 0.65, hard=args.hard_min_soc)
        print(f"[scan] running {label} ...")
        res = run_config(args.model, args.device, seeds, args.episodes)
        res["config"] = label
        rows.append(res)

    # 还原 eval 配置
    set_mode("fallback_off")

    cols = ["episode_safety_rate", "worst_seed_safety", "survival_rate", "crash_count",
            "comm_window_util", "downlink_mb", "delivered_value", "proc_dl_ratio",
            "fallback_trigger_rate", "fallback_to_downlink_rate"]
    print("\n" + "=" * 140)
    hdr = f"{'config':>34} " + " ".join(f"{c[:11]:>12}" for c in cols)
    print(hdr)
    print("-" * 140)
    for r in rows:
        line = f"{r['config']:>34} "
        for c in cols:
            v = r[c]
            line += f"{v:>12.3f} " if isinstance(v, float) else f"{v:>12} "
        print(line)
    print("=" * 140)

    # 成功判定
    off = next(r for r in rows if r["config"] == "fallback_off")
    print("\n成功标准: ep_safe>=0.90 & worst_seed>=0.80 & window>=0.20 & "
          f"downlink>off({off['downlink_mb']:.0f}) & proc_dl<=2.0 & crash≈0")
    winners = [r for r in rows if r["config"].startswith("safe_budget")
               and r["episode_safety_rate"] >= 0.90 and r["worst_seed_safety"] >= 0.80
               and r["comm_window_util"] >= 0.20 and r["downlink_mb"] > off["downlink_mb"]
               and r["crash_count"] == 0]
    if winners:
        winners.sort(key=lambda r: (r["comm_window_util"], -r["proc_dl_ratio"]), reverse=True)
        b = winners[0]
        print(f">>> 达标配置: {b['config']}  "
              f"(ep_safe={b['episode_safety_rate']:.2f}, window={b['comm_window_util']:.3f}, "
              f"downlink={b['downlink_mb']:.0f}, proc_dl={b['proc_dl_ratio']:.2f}) → 建议带此配置重训")
    else:
        print(">>> 无 eval-time 达标配置；需调 SAFE_BUDGET 参数或在重训中让策略共适应。")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] saved: {args.output}")


if __name__ == "__main__":
    main()

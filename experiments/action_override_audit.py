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
import csv
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
from utils.action_space import pointing_mode_from_unit, IDX_POINTING, GROUPED_ACTION_DIM

# 关键可学习维度（其余 3-13 为 class logits + 任务优先级权重 + drop，覆盖较少，单独汇总）。
KEY_DIMS = {0: "alpha_prop", 1: "alpha_cpu", 2: "alpha_tx", IDX_POINTING: "pointing_mode"}

AUDIT_COLUMNS = [
    "method", "seed", "episode", "step",
    "raw_action", "shielded_action", "executed_action",
    "raw_to_shield_l2", "shield_to_exec_l2", "raw_to_exec_l2",
    "prop_override", "cpu_override", "tx_override", "pointing_override",
    "safe_budget_trigger", "projection_trigger", "psf_trigger",
    "credit_gate_trigger", "fallback_reason", "in_contact",
    "high_value_available", "delivered_value",
]


def _align(a, n):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    if a.size < n:
        return np.pad(a, (0, n - a.size))
    return a[:n]


def _action_json(action) -> str:
    values = [round(float(x), 6) for x in np.asarray(action, dtype=np.float32).reshape(-1)]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def run(args) -> dict:
    from evaluate_optimized import _resolve_device
    device = _resolve_device(args.device)
    k = int(DRL_CONFIG.get("frame_stack", 8))

    scheduler = IntegratedScheduler(device=device, enable_lyapunov=True, use_psf=True)
    loaded = False
    if args.random:
        # 明确要求 random/untrained 审计时不能因为默认 checkpoint 存在而误加载模型。
        loaded = False
    elif args.model and os.path.exists(args.model):
        scheduler.load(args.model)
        loaded = True
    else:
        raise FileNotFoundError(
            f"未找到 checkpoint: {args.model}。无 checkpoint 想看覆盖结构请加 --random。")

    # 累计器
    raw_safe_l2, safe_exec_l2, raw_exec_l2 = [], [], []
    dim_abs = {d: [] for d in KEY_DIMS}          # 每维 |raw−executed| 绝对改写量
    dim_modified = {d: 0 for d in KEY_DIMS}      # 每维被改写步数
    weight_dims_abs = []                          # 3-13 class logits+优先级权重+drop 整体改写量
    n_steps = 0
    fire = {
        "psf_intervened": 0,
        "analytic_propulsion_enabled": 0,
        "analytic_propulsion_applied": 0,
        "mission_pointing_fallback_applied": 0,
        "safe_budget_fallback_applied": 0,
        "credit_gate_triggered": 0,
        "tx_floor_applied": 0,
        "pointing_mode_changed": 0,
    }
    # ── Phase 4: 分维度细分计数器 ────────────────────────────────────────
    # 按接触状态（in_contact vs out_of_contact）和高价值可用性分组统计 override rate
    contact_seg = {
        "in_contact_steps": 0, "out_contact_steps": 0,
        "in_contact_any_override": 0, "out_contact_any_override": 0,
        "in_contact_prop": 0, "out_contact_prop": 0,
        "in_contact_cpu": 0, "out_contact_cpu": 0,
        "in_contact_tx": 0, "out_contact_tx": 0,
        "in_contact_pointing": 0, "out_contact_pointing": 0,
        "in_contact_psf": 0, "out_contact_psf": 0,
        "in_contact_safe_budget": 0, "out_contact_safe_budget": 0,
        "in_contact_credit_gate": 0, "out_contact_credit_gate": 0,
    }
    hv_seg = {
        "hv_avail_steps": 0, "hv_unavail_steps": 0,
        "hv_avail_any_override": 0, "hv_unavail_any_override": 0,
        "hv_avail_prop": 0, "hv_unavail_prop": 0,
        "hv_avail_cpu": 0, "hv_unavail_cpu": 0,
        "hv_avail_tx": 0, "hv_unavail_tx": 0,
        "hv_avail_pointing": 0, "hv_unavail_pointing": 0,
        "hv_avail_psf": 0, "hv_unavail_psf": 0,
        "hv_avail_safe_budget": 0, "hv_unavail_safe_budget": 0,
        "hv_avail_credit_gate": 0, "hv_unavail_credit_gate": 0,
    }
    step_records = []
    method_name = args.method_name or ("Ours" if loaded else "Random/untrained")

    for ep in range(args.episodes):
        episode_seed = int(args.seed + ep)
        base_env = VLEOSatelliteEnv(seed=episode_seed)
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

            n = GROUPED_ACTION_DIM
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
            weight_dims_abs.append(float(np.linalg.norm(raw[3:IDX_POINTING] - ex[3:IDX_POINTING])))

            # 触发率（部分来自 info meta）。
            if bool(was_projected) or float(psf_meta.get("total_modification_l2", 0.0)) > 1e-6:
                fire["psf_intervened"] += 1
            if bool(info.get("analytic_propulsion_controller_enabled", False)):
                fire["analytic_propulsion_enabled"] += 1
            if bool(info.get("analytic_propulsion_applied", False)):
                fire["analytic_propulsion_applied"] += 1
            if bool(info.get("mission_pointing_fallback_applied", False)):
                fire["mission_pointing_fallback_applied"] += 1
            if bool(info.get("safe_budget_fallback_applied", False)):
                fire["safe_budget_fallback_applied"] += 1
            if bool(info.get("credit_gate_triggered", False)):
                fire["credit_gate_triggered"] += 1
            # TX floor：窗口内 raw[2] 明显低于 executed[2] 且 executed 贴近 floor。
            if in_window and (float(ex[2]) - float(raw[2])) > 1e-2:
                fire["tx_floor_applied"] += 1
            if pointing_mode_from_unit(float(raw[IDX_POINTING])) != pointing_mode_from_unit(float(ex[IDX_POINTING])):
                fire["pointing_mode_changed"] += 1

            # ── Phase 4: 接触状态 × 高价值可用性 细分 ──────────────────────
            _prop_ov = abs(float(raw[0]) - float(ex[0])) > 1e-3
            _cpu_ov  = abs(float(raw[1]) - float(ex[1])) > 1e-3
            _tx_ov   = abs(float(raw[2]) - float(ex[2])) > 1e-3
            _pt_ov   = (pointing_mode_from_unit(float(raw[IDX_POINTING]))
                        != pointing_mode_from_unit(float(ex[IDX_POINTING])))
            _psf_ov  = bool(was_projected) or float(psf_meta.get("total_modification_l2", 0.0)) > 1e-6
            _sb_ov   = bool(info.get("safe_budget_fallback_applied", False))
            _cg_ov   = bool(info.get("credit_gate_triggered", False))
            _any_ov  = _prop_ov or _cpu_ov or _tx_ov or _pt_ov

            _hv_avail = bool(
                float(info.get("cpu_requested_high", 0.0)) > 1e-9
                or float(info.get("tx_requested_high", 0.0)) > 1e-9
                or float(info.get("delivered_high_value", 0.0)) > 1e-9
            )

            _ic_key = "in_contact" if in_window else "out_contact"
            contact_seg[f"{_ic_key}_steps"] += 1
            if _any_ov:
                contact_seg[f"{_ic_key}_any_override"] += 1
            for _k, _v in [("prop", _prop_ov), ("cpu", _cpu_ov),
                            ("tx", _tx_ov), ("pointing", _pt_ov),
                            ("psf", _psf_ov), ("safe_budget", _sb_ov),
                            ("credit_gate", _cg_ov)]:
                if _v:
                    contact_seg[f"{_ic_key}_{_k}"] += 1

            _hv_key = "hv_avail" if _hv_avail else "hv_unavail"
            hv_seg[f"{_hv_key}_steps"] += 1
            if _any_ov:
                hv_seg[f"{_hv_key}_any_override"] += 1
            for _k, _v in [("prop", _prop_ov), ("cpu", _cpu_ov),
                            ("tx", _tx_ov), ("pointing", _pt_ov),
                            ("psf", _psf_ov), ("safe_budget", _sb_ov),
                            ("credit_gate", _cg_ov)]:
                if _v:
                    hv_seg[f"{_hv_key}_{_k}"] += 1

            raw_to_shield_l2 = float(np.linalg.norm(raw - safe))
            shield_to_exec_l2 = float(np.linalg.norm(safe - ex))
            raw_to_exec_l2 = float(np.linalg.norm(raw - ex))
            psf_trigger = bool(was_projected) or float(psf_meta.get("total_modification_l2", 0.0)) > 1e-6
            fallback_reasons = []
            for key in (
                "analytic_propulsion_applied",
                "mission_pointing_fallback_applied",
                "safe_budget_fallback_applied",
                "future_contact_cpu_gate_applied",
                "in_window_tx_floor_applied",
                "credit_gate_triggered",
            ):
                if bool(info.get(key, False)):
                    fallback_reasons.append(key)
            explicit_reason = info.get("fallback_reason", "")
            if explicit_reason:
                fallback_reasons.append(str(explicit_reason))
            step_records.append({
                "method": method_name,
                "seed": episode_seed,
                "episode": ep,
                "step": n_steps,
                "raw_action": _action_json(raw),
                "shielded_action": _action_json(safe),
                "executed_action": _action_json(ex),
                "raw_to_shield_l2": raw_to_shield_l2,
                "shield_to_exec_l2": shield_to_exec_l2,
                "raw_to_exec_l2": raw_to_exec_l2,
                "prop_override": abs(float(raw[0]) - float(ex[0])) > 1e-3,
                "cpu_override": abs(float(raw[1]) - float(ex[1])) > 1e-3,
                "tx_override": abs(float(raw[2]) - float(ex[2])) > 1e-3,
                "pointing_override": (
                    pointing_mode_from_unit(float(raw[IDX_POINTING]))
                    != pointing_mode_from_unit(float(ex[IDX_POINTING]))
                ),
                "safe_budget_trigger": bool(info.get("safe_budget_fallback_applied", False)),
                "projection_trigger": bool(was_projected),
                "psf_trigger": psf_trigger,
                "credit_gate_trigger": bool(info.get("credit_gate_triggered", False)),
                "fallback_reason": ";".join(fallback_reasons),
                "in_contact": bool(in_window),
                "high_value_available": bool(
                    float(info.get("cpu_requested_high", 0.0)) > 1e-9
                    or float(info.get("tx_requested_high", 0.0)) > 1e-9
                    or float(info.get("delivered_high_value", 0.0)) > 1e-9
                ),
                "delivered_value": float(info.get("delivered_value", 0.0)),
            })
            n_steps += 1

    def _m(x):
        return float(np.mean(x)) if x else 0.0

    steps = max(n_steps, 1)
    report = {
        "__meta__": {
            "timestamp": datetime.now().isoformat(),
            "model": args.model if loaded else "(untrained/random policy)",
            "method": method_name,
            "loaded_checkpoint": loaded,
            "device": device,
            "episodes": args.episodes,
            "total_steps": n_steps,
            "step_record_count": len(step_records),
            "note": "raw=actor output; shielded=PSF/Lyapunov output; executed=environment rules output",
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
    if args.include_step_json:
        report["step_records"] = step_records

    # ── Phase 4: 接触状态 × 高价值可用性 细分汇总 ─────────────────────────
    def _safe_rate(num: int, den: int) -> float:
        return float(num) / float(den) if den > 0 else float("nan")

    ic = contact_seg["in_contact_steps"]
    oc = contact_seg["out_contact_steps"]
    ha = hv_seg["hv_avail_steps"]
    hu = hv_seg["hv_unavail_steps"]

    report["contact_window_breakdown"] = {
        "in_contact_steps": ic,
        "out_contact_steps": oc,
        "in_contact_fraction": _safe_rate(ic, steps),
        "in_contact_any_override_rate": _safe_rate(contact_seg["in_contact_any_override"], ic),
        "out_contact_any_override_rate": _safe_rate(contact_seg["out_contact_any_override"], oc),
        "per_dim": {
            k: {
                "in_contact_rate":  _safe_rate(contact_seg.get(f"in_contact_{k}", 0), ic),
                "out_contact_rate": _safe_rate(contact_seg.get(f"out_contact_{k}", 0), oc),
            }
            for k in ("prop", "cpu", "tx", "pointing", "psf", "safe_budget", "credit_gate")
        },
    }
    report["high_value_breakdown"] = {
        "high_value_available_steps": ha,
        "high_value_unavailable_steps": hu,
        "high_value_available_fraction": _safe_rate(ha, steps),
        "high_value_available_any_override_rate": _safe_rate(hv_seg["hv_avail_any_override"], ha),
        "no_high_value_available_any_override_rate": _safe_rate(hv_seg["hv_unavail_any_override"], hu),
        "per_dim": {
            k: {
                "high_value_avail_rate":   _safe_rate(hv_seg.get(f"hv_avail_{k}", 0), ha),
                "no_high_value_avail_rate": _safe_rate(hv_seg.get(f"hv_unavail_{k}", 0), hu),
            }
            for k in ("prop", "cpu", "tx", "pointing", "psf", "safe_budget", "credit_gate")
        },
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
    csv_out = args.csv_output or os.path.splitext(out)[0] + ".csv"
    md_out  = os.path.splitext(out)[0] + "_summary.md"
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(step_records)
    report["__meta__"]["action_audit_csv"] = csv_out
    report["__meta__"]["action_audit_md"]  = md_out
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Phase 4: Markdown summary ─────────────────────────────────────────
    g = report["l2_gap_mean"]
    ovr = report["override_firing_rate"]
    cwb = report["contact_window_breakdown"]
    hvb = report["high_value_breakdown"]
    vrd = report["verdict"]
    pd  = report["per_dim"]

    def _pct(v):
        return f"{v:.1%}" if (v == v) else "n/a"

    md_lines = [
        f"# Action Override Audit — {report['__meta__']['method']}",
        f"",
        f"**Model**: `{report['__meta__']['model']}`  ",
        f"**Episodes**: {report['__meta__']['episodes']}  ",
        f"**Total steps**: {report['__meta__']['total_steps']}  ",
        f"",
        f"## L2 Gap Summary",
        f"",
        f"| Segment | Mean L2 |",
        f"|---------|---------|",
        f"| raw → safe (PSF/Lyapunov) | {g['raw_to_safe_psf_lyapunov']:.4f} |",
        f"| safe → executed (env rules) | {g['safe_to_executed_env_rules']:.4f} |",
        f"| **raw → executed (total)** | **{g['raw_to_executed_total']:.4f}** |",
        f"",
        f"**Policy-environment mismatch risk**: {vrd['policy_environment_mismatch_risk']}",
        f"",
        f"## Per-Dimension Override Rate",
        f"",
        f"| Dimension | Mean |Δ| | Steps Modified |",
        f"|-----------|------|--------------|",
    ]
    for name, d in pd.items():
        md_lines.append(
            f"| {name} | {d['mean_abs_modification']:.4f} | {_pct(d['fraction_steps_modified'])} |"
        )
    md_lines += [
        f"",
        f"## Override Trigger Rates (all steps)",
        f"",
        f"| Trigger | Rate |",
        f"|---------|------|",
    ]
    for k, v in ovr.items():
        md_lines.append(f"| {k} | {_pct(v)} |")
    md_lines += [
        f"",
        f"## Contact Window Breakdown",
        f"",
        f"| Condition | Any Override |",
        f"|-----------|-------------|",
        f"| In-contact ({cwb['in_contact_steps']} steps) | {_pct(cwb['in_contact_any_override_rate'])} |",
        f"| Out-of-contact ({cwb['out_contact_steps']} steps) | {_pct(cwb['out_contact_any_override_rate'])} |",
        f"",
        f"| Dim | In-contact | Out-of-contact |",
        f"|-----|-----------|----------------|",
    ]
    for k, v in cwb["per_dim"].items():
        md_lines.append(f"| {k} | {_pct(v['in_contact_rate'])} | {_pct(v['out_contact_rate'])} |")
    md_lines += [
        f"",
        f"## High-Value Available Breakdown",
        f"",
        f"| Condition | Any Override |",
        f"|-----------|-------------|",
        f"| High-value available ({hvb['high_value_available_steps']} steps) | {_pct(hvb['high_value_available_any_override_rate'])} |",
        f"| No high-value available ({hvb['high_value_unavailable_steps']} steps) | {_pct(hvb['no_high_value_available_any_override_rate'])} |",
        f"",
        f"| Dim | HV-avail | No-HV-avail |",
        f"|-----|----------|-------------|",
    ]
    for k, v in hvb["per_dim"].items():
        md_lines.append(
            f"| {k} | {_pct(v['high_value_avail_rate'])} | {_pct(v['no_high_value_avail_rate'])} |"
        )
    md_lines += [
        f"",
        f"## Verdict",
        f"",
        f"- Heavily overridden dimensions (>{int(vrd['threshold_fraction']*100)}%): "
        f"**{', '.join(vrd['heavily_overridden_dims']) or '(none)'}**",
        f"- RL影响CPU/TX/task-priority/high-value-delivery: "
        f"CPU override={_pct(ovr.get('analytic_propulsion_applied', 0))}, "
        f"TX floor={_pct(ovr.get('tx_floor_applied', 0))}, "
        f"pointing change={_pct(ovr.get('pointing_mode_changed', 0))}",
        f"- Safety shell主要接管: "
        f"PSF={_pct(ovr.get('psf_intervened', 0))}, "
        f"SAFE_BUDGET={_pct(ovr.get('safe_budget_fallback_applied', 0))}, "
        f"credit_gate={_pct(ovr.get('credit_gate_triggered', 0))}, "
        f"analytic_prop={_pct(ovr.get('analytic_propulsion_applied', 0))}",
        f"",
    ]

    with open(md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n  结果已保存: {out}")
    print(f"  action audit CSV: {csv_out}")
    print(f"  action audit Markdown summary: {md_out}")
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
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--method-name", default=None,
                        help="写入逐步 audit 的方法名，默认 loaded checkpoint 为 Ours")
    parser.add_argument("--include-step-json", action="store_true",
                        help="同时把逐步明细嵌入 JSON；默认只写 CSV，避免 JSON 过大")
    args = parser.parse_args()
    run(args)

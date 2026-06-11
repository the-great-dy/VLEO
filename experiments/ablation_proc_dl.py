"""定向降 proc/dl 的 config/gate 级 ablation（不重训）。

在锁定交付 checkpoint(best_optimized.pt)上 eval-time 切换 SAFE_BUDGET 的 data_pressure
CPU 节流强度 / future-contact gate，找"保吞吐(downlink≥6000, delivered≥7000)同时把
proc/dl 明显压下来"的配置。所有变体共享 canonical eval 路径（进程内 rollout，与
scan_safe_budget 同口径）。

变体（全部 SAFE_BUDGET enabled, soft_min_soc=0.60, hard_min_soc=0.50）：
  1 current            : 无 soft CPU 节流（=锁定交付基线 shell）
  2 dp_throttle@1.75   : data_pressure≥1.75 → alpha_cpu≤0.5
  3 dp_throttle@1.5    : data_pressure≥1.50 → alpha_cpu≤0.3
  4 dp_throttle@1.25   : data_pressure≥1.25 → alpha_cpu≤0.2
  5 strong_proc_penalty: data_pressure≥1.5 → 禁处理(alpha_cpu≤0) 且 hard 前移到 1.5
  6 future_contact_gate: 叠加 env 内 future-contact CPU gate（按未来可下传容量限处理）

选择规则：ep_safe≥0.90 且 window≥0.30 且 downlink≥6000 且 delivered≥7000
          且 proc/dl 相比 2.79 明显下降；不允许靠牺牲大量吞吐换 proc/dl。

用法（先快筛，再用胜出变体上 20-seed 确认）：
  python experiments/ablation_proc_dl.py --seeds 42,43,44 --episodes 2     # 快筛
  python experiments/ablation_proc_dl.py --seeds <20seeds> --episodes 5 --only dp_throttle@1.5
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _ROOT not in sys.path:
    sys.path.append(_ROOT)

import argparse
import json
import numpy as np

from environment.satellite_env import VLEOSatelliteEnv
from environment.wrappers import DilatedFrameStackWrapper
from scheduler.integrated_scheduler import IntegratedScheduler
from config import (TRAIN_CONFIG, DRL_CONFIG, HARD_RULES_CONFIG,
                    SAFE_BUDGET_FALLBACK_CONFIG, TASK_CONFIG,
                    CHECKPOINT_SELECTION_CONFIG)

# 锁定交付基线 proc/dl（"明显下降"的参照）
BASELINE_PROC_DL = 2.79
BASELINE_DOWNLINK = 6816.0
BASELINE_DELIVERED = 7587.0

# 变体定义：每个是 (label, overrides) — overrides 写入 SAFE_BUDGET/TASK 配置
VARIANTS = {
    "current":             {"cpu_throttle_pressure": 2.0,  "process_cap_alpha_soft": 1.0,  "data_pressure_hard": 2.0, "fc_gate": False},
    "dp_throttle@1.75":    {"cpu_throttle_pressure": 1.75, "process_cap_alpha_soft": 0.5,  "data_pressure_hard": 2.0, "fc_gate": False},
    "dp_throttle@1.5":     {"cpu_throttle_pressure": 1.5,  "process_cap_alpha_soft": 0.3,  "data_pressure_hard": 2.0, "fc_gate": False},
    "dp_throttle@1.25":    {"cpu_throttle_pressure": 1.25, "process_cap_alpha_soft": 0.2,  "data_pressure_hard": 2.0, "fc_gate": False},
    "strong_proc_penalty": {"cpu_throttle_pressure": 1.5,  "process_cap_alpha_soft": 0.0,  "data_pressure_hard": 1.5, "fc_gate": False},
    "future_contact_gate": {"cpu_throttle_pressure": 2.0,  "process_cap_alpha_soft": 1.0,  "data_pressure_hard": 2.0, "fc_gate": True},
}


def _available_power_w(env):
    base = env.env if hasattr(env, "env") else env
    return float(getattr(base, "_last_available_power_w", 0.0))


def _stage(info):
    if float(info.get("failure_state", 0.0)) > 0.5: return "failure"
    if float(info.get("unsafe_state", 0.0)) > 0.5: return "unsafe"
    if float(info.get("warning_state", 0.0)) > 0.5: return "warning"
    return "normal"


def _apply_variant(ov):
    SAFE_BUDGET_FALLBACK_CONFIG["enabled"] = True
    SAFE_BUDGET_FALLBACK_CONFIG["soft_min_soc"] = 0.60
    SAFE_BUDGET_FALLBACK_CONFIG["hard_min_soc"] = 0.50
    SAFE_BUDGET_FALLBACK_CONFIG["cpu_throttle_pressure"] = ov["cpu_throttle_pressure"]
    SAFE_BUDGET_FALLBACK_CONFIG["process_cap_alpha_soft"] = ov["process_cap_alpha_soft"]
    SAFE_BUDGET_FALLBACK_CONFIG["data_pressure_hard"] = ov["data_pressure_hard"]
    HARD_RULES_CONFIG["enable_mission_pointing_fallback"] = False
    TASK_CONFIG["enable_future_contact_cpu_gate"] = bool(ov["fc_gate"])


def rollout(env, scheduler):
    base = env.env if hasattr(env, "env") else env
    state = env.reset()
    done = False
    ep_tput = ep_tx = ep_value = 0.0
    ep_expired = ep_hi_del = ep_hi_exp = 0.0
    win_utils, backlogs = [], []
    steps = 0
    ep_safe = True
    crashed = False
    while not done:
        in_window = (base._contact.get("in_window", False) if base._contact is not None else False)
        prop_can_update = True
        if hasattr(base, "step_count") and hasattr(base, "N_PROP_SMOOTH"):
            prop_can_update = (base.step_count % base.N_PROP_SMOOTH == 0)
        action, _, _, _ = scheduler.schedule(
            state, base.energy_queue.value, base.orbit_queue.value,
            base.data_queue.length, base.comm_queue.value, in_window, evaluate=True,
            h=base.altitude_m, soc=base.battery.soc, time_s=base.time_s,
            prop_can_update=prop_can_update, orbital_phase=base.orbit_sim.phase,
            tx_capacity_mbps=float((base._contact or {}).get("max_capacity_mbps", 0.0)),
            available_power_w=_available_power_w(env), env=base)
        state, reward, done, info = env.step(action, enforce_prop_smoothing=False)
        steps += 1
        ep_tput += float(info.get("processed_mb", 0.0))
        dl = float(info.get("delivered_mb", info.get("actual_tx_mb", 0.0)))
        ep_tx += dl
        ep_value += float(info.get("delivered_value", 0.0))
        ep_expired += float(info.get("expired_value", info.get("expired_processed_value", 0.0)))
        ep_hi_del += float(info.get("delivered_high_value", info.get("high_value_delivered_value", 0.0)))
        ep_hi_exp += float(info.get("expired_high_value", info.get("expired_high_value_value", 0.0)))
        backlogs.append(float(base.data_queue.length) + float(base.comm_queue.value))
        cap_mb = float(info.get("tx_capacity_mbps", 0.0)) * TRAIN_CONFIG["time_slot_s"] / 8.0
        if bool(info.get("in_window", info.get("in_comm_window", False))) and cap_mb > 1e-9:
            win_utils.append(dl / cap_mb)
        st = _stage(info)
        if st != "normal": ep_safe = False
        if st == "failure": crashed = True
    survived = not crashed and steps >= int(TRAIN_CONFIG.get("max_episode_steps", 2160)) * 0.95
    hi_den = ep_hi_del + ep_hi_exp
    return {
        "ep_safe": ep_safe, "survived": survived, "crashed": crashed,
        "win_util": float(np.mean(win_utils)) if win_utils else 0.0,
        "downlink": ep_tx, "delivered": ep_value, "proc_dl": ep_tput / max(ep_tx, 1e-9),
        "backlog_mean": float(np.mean(backlogs)) if backlogs else 0.0,
        "backlog_max": float(np.max(backlogs)) if backlogs else 0.0,
        "expired": ep_expired,
        "hi_del": (ep_hi_del / hi_den) if hi_den > 1e-9 else float("nan"),
    }


def run_variant(label, ov, model, device, seeds, episodes):
    _apply_variant(ov)
    per_seed_safe, eps = [], []
    interv, proj = [], []
    for sd in seeds:
        env = DilatedFrameStackWrapper(VLEOSatelliteEnv(seed=sd), k=DRL_CONFIG.get("frame_stack", 8))
        scheduler = IntegratedScheduler(device=device, enable_lyapunov=True, use_psf=True)
        scheduler.load(model)
        scheduler.reset_all_safety_stats()
        seed_eps = [rollout(env, scheduler) for _ in range(episodes)]
        eps.extend(seed_eps)
        per_seed_safe.append(np.mean([e["ep_safe"] for e in seed_eps]))
        ss = scheduler.get_safety_stats()
        interv.append(float(ss.get("intervention_rate", 0.0)))
        proj.append(float(ss.get("lyapunov_proj_rate", 0.0)) + float(ss.get("psf_filter_rate", 0.0)))

    def m(k): return float(np.mean([e[k] for e in eps]))
    def mn(k): return float(np.nanmean([e[k] for e in eps]))
    return {
        "config": label,
        "episode_safety_rate": float(np.mean([e["ep_safe"] for e in eps])),
        "worst_seed_safety": float(np.min(per_seed_safe)),
        "survival_rate": float(np.mean([e["survived"] for e in eps])),
        "crash_count": int(np.sum([e["crashed"] for e in eps])),
        "comm_window_util": m("win_util"),
        "downlink_mb": m("downlink"),
        "delivered_value": m("delivered"),
        "proc_dl_ratio": m("proc_dl"),
        "backlog_mean": m("backlog_mean"),
        "backlog_max": m("backlog_max"),
        "expired_undelivered": m("expired"),
        "high_value_delivery": mn("hi_del"),
        "intervention_rate": float(np.mean(interv)),
        "projection_rate": float(np.mean(proj)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints_optimized/best_optimized.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--only", default=None, help="只跑指定变体（逗号分隔 label）")
    ap.add_argument("--output", default="results/ablation_proc_dl.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    labels = list(VARIANTS) if not args.only else [s.strip() for s in args.only.split(",")]

    rows = []
    for label in labels:
        print(f"[ablation] running {label} ...")
        rows.append(run_variant(label, VARIANTS[label], args.model, args.device, seeds, args.episodes))

    # 还原默认（锁定交付基线 shell）
    _apply_variant(VARIANTS["current"])
    TASK_CONFIG["enable_future_contact_cpu_gate"] = True  # config 默认值

    cols = ["episode_safety_rate", "worst_seed_safety", "survival_rate", "crash_count",
            "comm_window_util", "downlink_mb", "delivered_value", "proc_dl_ratio",
            "backlog_mean", "backlog_max", "expired_undelivered", "high_value_delivery",
            "intervention_rate", "projection_rate"]
    print("\n" + "=" * 200)
    print(f"{'config':>20} " + " ".join(f"{c[:12]:>13}" for c in cols))
    print("-" * 200)
    for r in rows:
        line = f"{r['config']:>20} "
        for c in cols:
            v = r[c]
            line += (f"{v:>13.3f} " if isinstance(v, float) else f"{v:>13} ")
        print(line)
    print("=" * 200)

    # 选择规则
    print(f"\n选择规则: ep_safe>=0.90 & window>=0.30 & downlink>=6000 & delivered>=7000 "
          f"& proc_dl 明显<{BASELINE_PROC_DL}（不靠牺牲吞吐）")
    winners = [r for r in rows
               if r["episode_safety_rate"] >= 0.90 and r["comm_window_util"] >= 0.30
               and r["downlink_mb"] >= 6000 and r["delivered_value"] >= 7000
               and r["proc_dl_ratio"] < BASELINE_PROC_DL - 0.15]
    if winners:
        winners.sort(key=lambda r: r["proc_dl_ratio"])
        b = winners[0]
        print(f">>> 候选胜出: {b['config']} (proc_dl={b['proc_dl_ratio']:.2f}, window={b['comm_window_util']:.3f}, "
              f"downlink={b['downlink_mb']:.0f}, delivered={b['delivered_value']:.0f}) → 建议上 20-seed 确认")
    else:
        print(">>> 无变体在保吞吐前提下明显降 proc/dl；保留当前交付基线，proc/dl≤2.0 需换思路。")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"[OK] saved: {args.output}")


if __name__ == "__main__":
    main()

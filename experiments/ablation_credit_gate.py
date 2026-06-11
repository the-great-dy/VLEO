"""credit-bucket 处理门 ablation（config/gate 级，不重训）。

在锁定交付 checkpoint(best_optimized.pt) + SAFE_BUDGET 上 eval-time 开 credit-bucket
流控门，找"保吞吐(downlink≥6000, delivered≥7000, window≥0.30)同时把 proc/dl 明显降到
≤2.5（再试 2.2/2.0）、expired 明显下降"的配置。

leaky-bucket 机制（env._apply_safe_budget_fallback + step 末更新）：
  credit = initial + gain·累计下传 − 累计处理；credit<=0→禁处理/禁成像，credit<=soft→节流。
  渐近把 episode proc/dl 钉到 ~target，且 initial 缓冲避免前期无窗口锁死。

变体：
  1 baseline               : 无 credit gate（=锁定交付基线 shell）
  2 credit target=2.5
  3 credit target=2.2
  4 credit target=2.0
  5 credit target=2.5 + 大 initial(2.5×)
  6 credit target=2.2 + 大 initial(2.5×)

保留标准：ep_safe≥0.90 & worst_seed≥0.80 & window≥0.30 & downlink≥6000 & delivered≥7000
          & proc/dl 明显<2.79(优先≤2.5) & expired 明显下降；不靠牺牲吞吐换 proc/dl。

用法：
  python experiments/ablation_credit_gate.py --seeds 42,43,44 --episodes 2   # 快筛
  python experiments/ablation_credit_gate.py --seeds <20seeds> --episodes 5 --only "credit target=2.5"
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
from config import TRAIN_CONFIG, DRL_CONFIG, HARD_RULES_CONFIG, SAFE_BUDGET_FALLBACK_CONFIG, TASK_CONFIG

BASELINE_PROC_DL = 2.79

# 变体：credit gate overrides（target 同时设 gain 与 soft/hard ratio 限）
def _cv(target, initial):
    return {"enable_credit_gate": True, "target_proc_dl_ratio": target,
            "credit_gain_per_downlink": target, "soft_ratio_limit": target,
            "hard_ratio_limit": target + 0.5, "initial_credit_factor": initial}

VARIANTS = {
    "baseline":                {"enable_credit_gate": False},
    "credit target=2.5":       _cv(2.5, 1.5),
    "credit target=2.2":       _cv(2.2, 1.5),
    "credit target=2.0":       _cv(2.0, 1.5),
    "credit t=2.5 bigInit":    _cv(2.5, 2.5),
    "credit t=2.2 bigInit":    _cv(2.2, 2.5),
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
    # 固定 SAFE_BUDGET 交付壳设置
    SAFE_BUDGET_FALLBACK_CONFIG["enabled"] = True
    SAFE_BUDGET_FALLBACK_CONFIG["soft_min_soc"] = 0.60
    SAFE_BUDGET_FALLBACK_CONFIG["hard_min_soc"] = 0.50
    # CPU 节流 lever 复位 no-op（本 ablation 只测 credit gate）
    SAFE_BUDGET_FALLBACK_CONFIG["cpu_throttle_pressure"] = 2.0
    SAFE_BUDGET_FALLBACK_CONFIG["process_cap_alpha_soft"] = 1.0
    SAFE_BUDGET_FALLBACK_CONFIG["data_pressure_hard"] = 2.0
    HARD_RULES_CONFIG["enable_mission_pointing_fallback"] = False
    # credit gate 默认关，再按变体覆盖
    SAFE_BUDGET_FALLBACK_CONFIG["enable_credit_gate"] = False
    for k, v in ov.items():
        SAFE_BUDGET_FALLBACK_CONFIG[k] = v


def rollout(env, scheduler):
    base = env.env if hasattr(env, "env") else env
    state = env.reset()
    done = False
    ep_tput = ep_tx = ep_value = 0.0
    ep_expired = ep_hi_del = ep_hi_exp = 0.0
    win_utils, backlogs = [], []
    gate_trig = img_blk = proc_blk = dl_prio = 0
    last_run_ratio = last_credit_mean = 0.0
    last_credit_min = float("nan")
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
        ep_hi_del += float(info.get("delivered_high_value", 0.0))
        ep_hi_exp += float(info.get("expired_high_value", 0.0))
        backlogs.append(float(base.data_queue.length) + float(base.comm_queue.value))
        cap_mb = float(info.get("tx_capacity_mbps", 0.0)) * TRAIN_CONFIG["time_slot_s"] / 8.0
        if bool(info.get("in_window", info.get("in_comm_window", False))) and cap_mb > 1e-9:
            win_utils.append(dl / cap_mb)
        gate_trig += int(bool(info.get("credit_gate_triggered", False)))
        img_blk += int(bool(info.get("credit_image_blocked", False)))
        proc_blk += int(bool(info.get("credit_process_blocked", False)))
        dl_prio += int(bool(info.get("credit_downlink_prioritized", False)))
        if "running_proc_dl" in info: last_run_ratio = float(info["running_proc_dl"])
        if "processing_credit_mean" in info: last_credit_mean = float(info["processing_credit_mean"])
        if "processing_credit_min" in info: last_credit_min = float(info["processing_credit_min"])
        st = _stage(info)
        if st != "normal": ep_safe = False
        if st == "failure": crashed = True
    survived = not crashed and steps >= int(TRAIN_CONFIG.get("max_episode_steps", 2160)) * 0.95
    hi_den = ep_hi_del + ep_hi_exp
    return {
        "ep_safe": ep_safe, "survived": survived, "crashed": crashed,
        "win_util": float(np.mean(win_utils)) if win_utils else 0.0,
        "downlink": ep_tx, "delivered": ep_value, "proc_dl": ep_tput / max(ep_tx, 1e-9),
        "running_proc_dl": last_run_ratio,
        "backlog_mean": float(np.mean(backlogs)) if backlogs else 0.0,
        "backlog_max": float(np.max(backlogs)) if backlogs else 0.0,
        "expired": ep_expired, "hi_del": (ep_hi_del / hi_den) if hi_den > 1e-9 else float("nan"),
        "credit_mean": last_credit_mean, "credit_min": last_credit_min,
        "gate_trig_rate": gate_trig / max(steps, 1),
        "img_blk_rate": img_blk / max(steps, 1),
        "proc_blk_rate": proc_blk / max(steps, 1),
        "dl_prio_rate": dl_prio / max(steps, 1),
    }


def run_variant(label, ov, model, device, seeds, episodes):
    _apply_variant(ov)
    per_seed_safe, eps, interv, proj = [], [], [], []
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
        "comm_window_util": m("win_util"), "downlink_mb": m("downlink"),
        "delivered_value": m("delivered"), "proc_dl_ratio": m("proc_dl"),
        "running_proc_dl": m("running_proc_dl"),
        "backlog_mean": m("backlog_mean"), "backlog_max": m("backlog_max"),
        "expired": m("expired"), "high_value_delivery": mn("hi_del"),
        "credit_mean": m("credit_mean"), "credit_min": mn("credit_min"),
        "gate_trigger_rate": m("gate_trig_rate"), "image_blocked": m("img_blk_rate"),
        "process_blocked": m("proc_blk_rate"), "downlink_prioritized": m("dl_prio_rate"),
        "intervention_rate": float(np.mean(interv)), "projection_rate": float(np.mean(proj)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints_optimized/best_optimized.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--only", default=None)
    ap.add_argument("--output", default="results/ablation_credit_gate.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    labels = list(VARIANTS) if not args.only else [s.strip() for s in args.only.split(",")]

    rows = []
    for label in labels:
        print(f"[credit-ablation] running {label} ...")
        rows.append(run_variant(label, VARIANTS[label], args.model, args.device, seeds, args.episodes))

    _apply_variant(VARIANTS["baseline"])  # 还原默认（credit gate 关）

    cols = ["episode_safety_rate", "worst_seed_safety", "survival_rate", "crash_count",
            "comm_window_util", "downlink_mb", "delivered_value", "proc_dl_ratio", "running_proc_dl",
            "expired", "backlog_mean", "backlog_max", "high_value_delivery",
            "credit_mean", "credit_min", "gate_trigger_rate", "image_blocked",
            "process_blocked", "downlink_prioritized", "intervention_rate", "projection_rate"]
    print("\n" + "=" * 270)
    print(f"{'config':>20} " + " ".join(f"{c[:11]:>12}" for c in cols))
    print("-" * 270)
    for r in rows:
        line = f"{r['config']:>20} "
        for c in cols:
            v = r[c]
            line += (f"{v:>12.3f} " if isinstance(v, float) else f"{v:>12} ")
        print(line)
    print("=" * 270)

    base = next((r for r in rows if r["config"] == "baseline"), None)
    print(f"\n保留标准: ep_safe≥0.90 & worst≥0.80 & window≥0.30 & downlink≥6000 & delivered≥7000 "
          f"& proc_dl 明显<{BASELINE_PROC_DL}(优先≤2.5) & expired 明显下降(不靠牺牲吞吐)")
    winners = [r for r in rows if r["config"] != "baseline"
               and r["episode_safety_rate"] >= 0.90 and r["worst_seed_safety"] >= 0.80
               and r["comm_window_util"] >= 0.30 and r["downlink_mb"] >= 6000
               and r["delivered_value"] >= 7000 and r["proc_dl_ratio"] <= 2.5
               and (base is None or r["expired"] < base["expired"])]
    if winners:
        winners.sort(key=lambda r: r["proc_dl_ratio"])
        b = winners[0]
        print(f">>> 候选胜出: {b['config']} (proc_dl={b['proc_dl_ratio']:.2f}, window={b['comm_window_util']:.3f}, "
              f"downlink={b['downlink_mb']:.0f}, delivered={b['delivered_value']:.0f}, "
              f"expired={b['expired']:.0f}) → 上 20-seed 确认")
    else:
        print(">>> 无变体在保吞吐前提下达标；保留交付基线，记录 credit gate 趋势。")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"[OK] saved: {args.output}")


if __name__ == "__main__":
    main()

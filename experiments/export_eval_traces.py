"""
导出评估 episode 逐步轨迹的工具。

导出评估 episode 的逐步微观统计帧，用于论文中的物理过程图：
日照/阴影、SOC、热状态、raw/processed queue、Q_E/Q_H、通信窗口、下传和风险状态。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from config import DRL_CONFIG, TRAIN_CONFIG
from environment.satellite_env import VLEOSatelliteEnv
from environment.wrappers import DilatedFrameStackWrapper
from scheduler.integrated_scheduler import IntegratedScheduler
from evaluate_optimized import _resolve_device


DEFAULT_OPTIMIZED_CHECKPOINT = os.path.join(
    TRAIN_CONFIG.get("optimized_checkpoint_dir", "checkpoints_optimized/"),
    "best_optimized.pt",
)


TRACE_COLUMNS = [
    "episode", "step", "time_s", "sunlit", "in_window",
    "altitude_km", "soc", "battery_capacity_wh",
    "thermal_temperature_c", "thermal_margin_norm", "thermal_stage",
    "raw_queue_mb", "processed_queue_mb",
    "energy_virtual_queue", "orbit_virtual_queue", "comm_virtual_queue",
    "P_solar_w", "P_total_w", "P_propulsion_w", "P_cpu_w", "P_tx_w",
    "tx_capacity_mbps", "actual_tx_mb", "delivered_value",
    "average_aoi_steps", "voi_degradation_rate",
    "deadline_urgency", "expiring_value", "scene_name",
    "risk_stage", "overall_safe",
    "alpha_prop", "alpha_cpu", "alpha_tx",
    # 指向/任务可行性诊断 (定位 "对日保守 → 不成像/不下传")
    "can_image", "can_downlink",
    "pointing_mode", "pointing_mode_requested", "pointing_fallback_reason",
]


def export_eval_traces(args) -> str:
    device = _resolve_device(args.device)
    scheduler = IntegratedScheduler(device=device, enable_lyapunov=True, use_psf=True)
    scheduler.load(args.checkpoint)

    rows = []
    for ep in range(args.episodes):
        base_env = VLEOSatelliteEnv(seed=args.seed + ep)
        env = DilatedFrameStackWrapper(
            base_env, k=int(DRL_CONFIG.get("frame_stack", 8)))
        state = env.reset()
        done = False
        step = 0
        while not done:
            in_window = bool((base_env._contact or {}).get("in_window", False))
            prop_can_update = (base_env.step_count % base_env.N_PROP_SMOOTH == 0)
            action, _, _, _ = scheduler.schedule(
                state,
                base_env.energy_queue.value,
                base_env.orbit_queue.value,
                base_env.data_queue.length,
                base_env.comm_queue.value,
                in_window=in_window,
                evaluate=True,
                h=base_env.altitude_m,
                soc=base_env.battery.soc,
                time_s=base_env.time_s,
                prop_can_update=prop_can_update,
                orbital_phase=base_env.orbit_sim.phase,
                tx_capacity_mbps=float((base_env._contact or {}).get("max_capacity_mbps", 0.0)),
                available_power_w=getattr(base_env, "available_power_w", None),
                env=base_env,
            )
            state, _, done, info = env.step(action, enforce_prop_smoothing=False)
            executed = info.get("executed_action", action)
            rows.append({
                "episode": ep,
                "step": step,
                "time_s": float(info.get("time_s", base_env.time_s)),
                "sunlit": int(bool(info.get("sunlit", False))),
                "in_window": int(bool(info.get("in_window", False))),
                "altitude_km": float(info.get("altitude_km", 0.0)),
                "soc": float(info.get("soc", 0.0)),
                "battery_capacity_wh": float(info.get("battery_capacity_wh", 0.0)),
                "thermal_temperature_c": float(info.get("thermal_temperature_c", 0.0)),
                "thermal_margin_norm": float(info.get("thermal_margin_norm", 1.0)),
                "thermal_stage": str(info.get("thermal_stage", "normal")),
                "raw_queue_mb": float(info.get("raw_queue_mb", 0.0)),
                "processed_queue_mb": float(info.get("processed_queue_mb", 0.0)),
                "energy_virtual_queue": float(info.get("energy_virtual_queue", 0.0)),
                "orbit_virtual_queue": float(info.get("orbit_virtual_queue", 0.0)),
                "comm_virtual_queue": float(info.get("comm_virtual_queue", 0.0)),
                "P_solar_w": float(info.get("P_solar_w", 0.0)),
                "P_total_w": float(info.get("P_total_w", 0.0)),
                "P_propulsion_w": float(info.get("P_propulsion_w", 0.0)),
                "P_cpu_w": float(info.get("P_cpu_w", 0.0)),
                "P_tx_w": float(info.get("P_tx_w", 0.0)),
                "tx_capacity_mbps": float(info.get("tx_capacity_mbps", 0.0)),
                "actual_tx_mb": float(info.get("actual_tx_mb", 0.0)),
                "delivered_value": float(info.get("delivered_value", 0.0)),
                "average_aoi_steps": float(info.get("average_aoi_steps", 0.0)),
                "voi_degradation_rate": float(info.get("voi_degradation_rate", 0.0)),
                "deadline_urgency": float(info.get("deadline_urgency", 0.0)),
                "expiring_value": float(info.get("expiring_value", 0.0)),
                "scene_name": str(info.get("scene_name", "")),
                "risk_stage": str(info.get("risk_stage", "normal")),
                "overall_safe": float(info.get("overall_safe", 1.0)),
                "alpha_prop": float(executed[0]),
                "alpha_cpu": float(executed[1]),
                "alpha_tx": float(executed[2]),
                "can_image": int(bool(info.get("can_image", True))),
                "can_downlink": int(bool(info.get("can_downlink", True))),
                "pointing_mode": int(info.get("pointing_mode", -1)),
                "pointing_mode_requested": int(info.get("mission_pointing_mode_before", -1)),
                "pointing_fallback_reason": str(info.get("mission_pointing_fallback_reason", "")),
            })
            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                done = True

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRACE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] trace saved: {args.output} ({len(rows)} rows)")
    return args.output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export per-step evaluation traces")
    parser.add_argument("--checkpoint", default=DEFAULT_OPTIMIZED_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=int(TRAIN_CONFIG.get("seed", 42)) + 20000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/eval_traces_{ts}.csv"
    export_eval_traces(args)

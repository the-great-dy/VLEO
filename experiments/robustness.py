"""
扰动和真实 trace 条件下的鲁棒性评估入口。
鲁棒性测试：在不同扰动条件下验证方法的稳定性

测试维度（对应论文 Section V.C）：
  1. 初始轨道高度扰动  （含 150km 安全边界与 122km 再入边界附近）
  2. 大气密度扰动      （标称值 ±30%）
  3. 太阳能板退化      （效率 50%~100%）
  4. 任务到达率变化    （0.5x ~ 3x 标准速率）
  5. 电池容量退化      （70%~100%）
  6. 真实 trace 驱动    （TLE/MSISE/通信窗口/链路容量序列）

评估口径：
  1. 先 reset() 再执行 condition_fn，确保初始高度、SOC 等扰动真正生效
  2. LS-PSF CMDP 使用 DilatedFrameStackWrapper，与训练保持一致
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    # 仅在脚本直跑时追加，避免导入期全局污染 sys.path
    sys.path.append(_PROJECT_ROOT)

import numpy as np
import json
import argparse
import csv
from datetime import datetime

from environment.satellite_env import VLEOSatelliteEnv
from environment.wrappers import DilatedFrameStackWrapper
from environment.orbital_dynamics import OrbitalDynamics
from environment.energy_model import BatteryModel, SolarPanelModel
from environment.task_value_model import TaskBatch
from scheduler.integrated_scheduler import IntegratedScheduler
from baselines.mpc_baseline import MPCBaseline
from baselines.robust_mpc_baseline import RobustMPCBaseline
from baselines.oracle_mpc_baseline import OracleMPCBaseline
from baselines.dpp_baseline import DriftPlusPenaltyBaseline
from baselines.heuristic_baseline import HeuristicBaseline
from baselines.value_baselines import StaticRuleBaseline
from utils.paper_metrics import add_paper_metrics
from config import (
    TRAIN_CONFIG, ORBITAL_CONFIG, ENERGY_CONFIG, DRAG_CONFIG,
    DRL_CONFIG, TASK_CONFIG, QUEUE_CONFIG,
)

ALTITUDE_SAFE_KM = float(ORBITAL_CONFIG["altitude_min_km"])
BATTERY_SAFE_SOC = float(ENERGY_CONFIG["battery_min_soc"])

DEFAULT_OPTIMIZED_CHECKPOINT = os.path.join(
    TRAIN_CONFIG.get("optimized_checkpoint_dir", "checkpoints_optimized/"),
    "best_optimized.pt",
)
OURS_NAME = "LS-PSF CMDP (Ours)"


def _to_float(value):
    try:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_binary_float(value):
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "t", "yes", "y", "sunlit"}:
            return 1.0
        if s in {"0", "false", "f", "no", "n", "eclipse"}:
            return 0.0
    v = _to_float(value)
    if v is None:
        return None
    return 1.0 if v > 0 else 0.0


def _load_trace_rows(trace_csv: str):
    """
    加载外生扰动 trace CSV（真实轨迹/气象驱动）。

    支持列名（任选其一）：
      - 大气扰动: rho_scale / density_scale / atmospheric_scale / rho_factor
      - 光照扰动: solar_scale / solar_factor / illumination_scale
      - 光照状态: sunlit / sunlit_flag / eclipse_flag
      - 轨道位置: altitude_km / lat_deg / lon_deg
      - 通信窗口: in_window / tx_capacity_mbps
      - 任务强度: data_scale / arrival_scale
    """
    if not os.path.exists(trace_csv):
        raise FileNotFoundError(f"未找到 trace CSV: {trace_csv}")

    # trace 只覆盖外生扰动，不直接写入动作或 reward，避免把策略结果提前编码进数据。
    rows = []
    with open(trace_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("trace CSV 缺少表头")

        fields = {str(name).strip().lower(): name for name in reader.fieldnames if name}

        rho_keys = ["rho_scale", "density_scale", "atmospheric_scale", "rho_factor"]
        solar_keys = ["solar_scale", "solar_factor", "illumination_scale", "sunlight_scale"]
        sunlit_keys = ["sunlit", "sunlit_flag", "eclipse_flag", "in_sunlight"]
        altitude_keys = ["altitude_km", "height_km", "alt_km"]
        lat_keys = ["lat_deg", "latitude_deg", "latitude"]
        lon_keys = ["lon_deg", "longitude_deg", "longitude"]
        window_keys = ["in_window", "contact", "contact_flag", "visible"]
        capacity_keys = ["tx_capacity_mbps", "capacity_mbps", "max_capacity_mbps", "link_capacity_mbps"]
        data_scale_keys = ["data_scale", "arrival_scale", "arrival_factor", "task_scale"]

        for raw in reader:
            item = {}

            rho_val = None
            for key in rho_keys:
                col = fields.get(key)
                if col is None:
                    continue
                rho_val = _to_float(raw.get(col))
                if rho_val is not None:
                    break

            solar_val = None
            for key in solar_keys:
                col = fields.get(key)
                if col is None:
                    continue
                solar_val = _to_float(raw.get(col))
                if solar_val is not None:
                    break

            if solar_val is None:
                for key in sunlit_keys:
                    col = fields.get(key)
                    if col is None:
                        continue
                    parsed = _to_binary_float(raw.get(col))
                    if parsed is not None:
                        solar_val = 1.0 - parsed if key == "eclipse_flag" else parsed
                        break

            sunlit_val = None
            for key in sunlit_keys:
                col = fields.get(key)
                if col is None:
                    continue
                parsed = _to_binary_float(raw.get(col))
                if parsed is not None:
                    sunlit_val = 1.0 - parsed if key == "eclipse_flag" else parsed
                    break

            if rho_val is not None:
                item["rho_scale"] = float(np.clip(rho_val, 0.1, 5.0))
            if solar_val is not None:
                item["solar_scale"] = float(np.clip(solar_val, 0.0, 2.0))
            if sunlit_val is not None:
                item["sunlit"] = int(sunlit_val > 0.0)

            altitude_lower_km = float(ORBITAL_CONFIG.get("altitude_crash_km", 122.0))
            for out_key, keys, lower, upper in [
                ("altitude_km", altitude_keys, altitude_lower_km, 600.0),
                ("lat_deg", lat_keys, -90.0, 90.0),
                ("lon_deg", lon_keys, -180.0, 180.0),
                ("tx_capacity_mbps", capacity_keys, 0.0, 5000.0),
                ("data_scale", data_scale_keys, 0.0, 10.0),
            ]:
                for key in keys:
                    col = fields.get(key)
                    if col is None:
                        continue
                    val = _to_float(raw.get(col))
                    if val is not None:
                        item[out_key] = float(np.clip(val, lower, upper))
                        break

            for key in window_keys:
                col = fields.get(key)
                if col is None:
                    continue
                val = _to_binary_float(raw.get(col))
                if val is not None:
                    item["in_window"] = bool(val > 0.0)
                    break

            if item:
                rows.append(item)

    if not rows:
        raise ValueError(
            "trace CSV 未解析出有效扰动列，请提供 rho_scale/density_scale 或 solar_scale/sunlit 列"
        )

    return rows


def _trace_field_summary(trace_rows) -> list:
    keys = set()
    for row in trace_rows or []:
        keys.update(row.keys())
    return sorted(keys)


def _trace_row_at(trace_rows, step_idx: int, cycle: bool = False):
    # cycle=True 适合短 trace 循环复用；默认则用最后一行延长到 episode 结束。
    if not trace_rows:
        return None
    if cycle:
        return trace_rows[step_idx % len(trace_rows)]
    idx = min(step_idx, len(trace_rows) - 1)
    return trace_rows[idx]


def _apply_trace_item(base_env, trace_item: dict,
                      base_rho_ref: float,
                      base_solar_eta: float,
                      trace_altitude_mode: str = "ignore"):
    """
    将 trace 的外生物理量注入当前控制步。

    altitude_km 默认只记录不强制覆盖；若设置 --trace_altitude_mode force，
    则按 trace 回放高度，适合用真实 VLEO 轨迹做压力测试。
    """
    base_env._contact_override = None
    if not trace_item:
        return

    if "rho_scale" in trace_item:
        base_env.orbit_dyn.atm.rho_ref = base_rho_ref * float(trace_item["rho_scale"])

    if "solar_scale" in trace_item:
        eta = base_solar_eta * float(trace_item["solar_scale"])
        base_env.solar.eta = float(np.clip(eta, 0.0, 1.5))

    if "data_scale" in trace_item:
        base_env._data_arrival_scale = float(np.clip(trace_item["data_scale"], 0.0, 10.0))

    if trace_altitude_mode == "force" and "altitude_km" in trace_item:
        h_m = float(trace_item["altitude_km"]) * 1e3
        lower_bound_m = float(getattr(base_env, "_h_crash", 0.0))
        base_env.altitude_m = float(np.clip(h_m, lower_bound_m, base_env._h_max))
        # 只在 episode 起点同步一次初始队列；逐步回放 trace 高度时不能反复 reset，
        # 否则会清掉轨道虚拟队列历史，低估真实 trace 下的轨道安全压力。
        if getattr(base_env, "step_count", 0) == 0:
            base_env.orbit_queue.reset(base_env.altitude_m)

    contact_override = {}
    if "in_window" in trace_item:
        contact_override["in_window"] = bool(trace_item["in_window"])
    if "tx_capacity_mbps" in trace_item:
        contact_override["max_capacity_mbps"] = float(max(0.0, trace_item["tx_capacity_mbps"]))
    if contact_override:
        if contact_override.get("in_window") is False and "max_capacity_mbps" not in contact_override:
            contact_override["max_capacity_mbps"] = 0.0
        base_env._contact_override = contact_override
    base_env._contact = base_env._get_contact_info()


def _eval_under_condition(scheduler_fn, condition_fn,
                          n_episodes: int = None,
                          seed_offset: int = 100,
                          use_wrapper: bool = False,
                          max_steps: int = None,
                          trace_rows=None,
                          trace_cycle: bool = False,
                          trace_altitude_mode: str = "ignore") -> dict:
    """
    在特定扰动条件下评估调度器

    扰动注入时机：
      env.reset() → condition_fn(env) → 重新生成 obs → 运行。
      这样可以确保高度、SOC 等初始条件扰动不会被 reset 覆盖。

    对于"大气密度"、"太阳能效率"等物理参数扰动，顺序无所谓
    （reset 不重置这些参数）。但为了统一且安全，全部放到 reset 后面。
    """
    n_episodes = int(TRAIN_CONFIG.get("eval_episodes", 30) if n_episodes is None else n_episodes)
    effective_steps = int(max_steps) if max_steps is not None else int(TRAIN_CONFIG["max_episode_steps"])
    old_arrival_rate = float(QUEUE_CONFIG.get("data_arrival_rate_mbs", 2.0))

    rewards, processed_mbs, downlink_mbs, delivered_values, safes, survivals = [], [], [], [], [], []
    orbit_safe_rates, energy_safe_rates, thermal_safe_rates = [], [], []
    raw_safe_rates, processed_safe_rates, overall_safe_rates = [], [], []
    stage_rate_sums = {"normal": [], "warning": [], "unsafe": [], "failure": []}
    deadline_rates, expired_rates, drop_rates, aoi_steps = [], [], [], []
    value_weighted_deadline_rates, value_weighted_aoi_steps, voi_loss_rates = [], [], []
    high_value_delivery_rates, processed_final_utils = [], []
    window_utils, tx_active_contact_flags = [], []
    k = DRL_CONFIG.get("frame_stack", 8)

    for ep in range(n_episodes):
        # 每个扰动条件下仍然随机化 episode seed，避免鲁棒性结论依赖单一初始状态。
        base_env = VLEOSatelliteEnv(seed=seed_offset + ep)

        if use_wrapper:
            env = DilatedFrameStackWrapper(base_env, k=k)
        else:
            env = base_env

        # 先 reset，再注入扰动，最后重新获取观测。
        state = env.reset()

        # 注入扰动（此时 env 内部状态已初始化，可以安全修改）
        condition_fn(base_env)

        # trace 作为外生变量输入：保留仿真引擎，按时间序列覆盖太阳/大气扰动
        # 这里使用“条件扰动后的值”作为基线，trace 在其上做倍率扰动。
        base_rho_ref = float(base_env.orbit_dyn.atm.rho_ref)
        base_solar_eta = float(base_env.solar.eta)
        # 多 episode 评估时顺着 trace 往后取不同时间段，避免每个 episode
        # 都只复用 CSV 开头的 90 分钟窗口。
        trace_offset = ep * effective_steps
        first_trace_item = _trace_row_at(trace_rows, trace_offset, cycle=trace_cycle)
        _apply_trace_item(
            base_env, first_trace_item,
            base_rho_ref, base_solar_eta,
            trace_altitude_mode=trace_altitude_mode)

        # 扰动可能修改高度，需同步刷新通信窗口状态
        base_env._contact = base_env._get_contact_info()

        # 扰动可能修改了 altitude_m / battery.soc 等状态，
        # 需要重新生成观测向量以反映扰动后的真实状态
        if use_wrapper:
            # wrapper 的 _get_obs 会从 base_env 重新读状态
            # 但 wrapper 内部的 history deque 里全是旧 obs，
            # 需要用扰动后的新 obs 刷新整个历史
            new_raw_obs = base_env._get_observation()
            env._history.clear()
            for _ in range(env._max_offset + 1):
                env._history.append(new_raw_obs.copy())
            state = env._get_obs()
        else:
            state = base_env._get_observation()

        ep_reward = ep_processed = ep_downlink = ep_value = 0.0
        ep_high_delivered = ep_high_expired = ep_high_dropped = 0.0
        ep_final_processed_util = 0.0
        safe_counts = {"orbit": 0, "energy": 0, "thermal": 0, "raw": 0, "proc": 0, "overall": 0}
        stage_counts = {"normal": 0, "warning": 0, "unsafe": 0, "failure": 0}
        is_safe = True
        survived = True
        done = False
        step_count = 0
        while not done:
            # trace 扰动按 step 注入，覆盖当前步的大气密度/太阳能缩放，动作仍由 scheduler 生成。
            trace_item = _trace_row_at(
                trace_rows, trace_offset + step_count, cycle=trace_cycle)
            _apply_trace_item(
                base_env, trace_item,
                base_rho_ref, base_solar_eta,
                trace_altitude_mode=trace_altitude_mode)
            if trace_item is not None:
                # trace 会改变当前步观测中的窗口/高度/阻力，需要让调度器看到覆盖后的状态。
                if use_wrapper:
                    new_raw_obs = base_env._get_observation()
                    env._history[-1] = new_raw_obs.copy()
                    state = env._get_obs()
                else:
                    state = base_env._get_observation()

            action = scheduler_fn(state, env)
            if use_wrapper:
                state, reward, done, info = env.step(
                    action, enforce_prop_smoothing=False)
            else:
                state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_processed += float(info.get(
                "processed_mb",
                info.get("service_rate_mbs", 0.0) * TRAIN_CONFIG["time_slot_s"]))
            ep_downlink += float(info.get("delivered_mb", info.get("actual_tx_mb", 0.0)))
            ep_value += float(info.get("delivered_value", 0.0))
            ep_high_delivered += float(info.get("delivered_high_value", 0.0))
            ep_high_expired += float(info.get("expired_high_value", 0.0))
            ep_high_dropped += float(info.get("dropped_high_value", 0.0))
            ep_final_processed_util = float(info.get("processed_queue_utilization", 0.0))
            capacity_mb = float(info.get("tx_capacity_mbps", 0.0)) * TRAIN_CONFIG["time_slot_s"] / 8.0
            if bool(info.get("in_window", False)) and capacity_mb > 1e-9:
                window_utils.append(float(info.get("delivered_mb", info.get("actual_tx_mb", 0.0))) / capacity_mb)
                tx_active_contact_flags.append(float(
                    info.get("delivered_mb", info.get("actual_tx_mb", 0.0)) > 1e-9
                ))
            step_count += 1

            if max_steps is not None and step_count >= max_steps:
                done = True

            if bool(info.get("terminated", False)):
                survived = False
            orbit_safe = bool(info.get("orbit_safe", info.get("altitude_km", 300) >= ALTITUDE_SAFE_KM))
            energy_safe = bool(info.get("energy_safe", info.get("soc", 1.0) >= BATTERY_SAFE_SOC))
            thermal_safe = bool(info.get("thermal_safe", True))
            raw_safe = bool(info.get("raw_queue_safe", info.get("raw_queue_overflow_mb", 0.0) <= 1e-9))
            proc_safe = bool(info.get("processed_queue_safe", info.get("processed_queue_overflow_mb", 0.0) <= 1e-9))
            overall_safe = bool(info.get("overall_safe", orbit_safe and energy_safe and thermal_safe and raw_safe and proc_safe))
            safe_counts["orbit"] += int(orbit_safe)
            safe_counts["energy"] += int(energy_safe)
            safe_counts["thermal"] += int(thermal_safe)
            safe_counts["raw"] += int(raw_safe)
            safe_counts["proc"] += int(proc_safe)
            safe_counts["overall"] += int(overall_safe)

            if not overall_safe:
                is_safe = False
            stage = str(info.get("risk_stage", "normal"))
            if stage not in stage_counts:
                stage = "failure" if bool(info.get("crashed", False)) else "normal"
            stage_counts[stage] += 1
        rewards.append(ep_reward)
        processed_mbs.append(ep_processed)
        downlink_mbs.append(ep_downlink)
        delivered_values.append(ep_value)
        high_den = ep_high_delivered + ep_high_expired + ep_high_dropped
        high_value_delivery_rates.append(float(ep_high_delivered / max(high_den, 1e-9)))
        processed_final_utils.append(float(ep_final_processed_util))
        safes.append(float(is_safe))
        survivals.append(float(survived))
        orbit_safe_rates.append(safe_counts["orbit"] / max(step_count, 1))
        energy_safe_rates.append(safe_counts["energy"] / max(step_count, 1))
        thermal_safe_rates.append(safe_counts["thermal"] / max(step_count, 1))
        raw_safe_rates.append(safe_counts["raw"] / max(step_count, 1))
        processed_safe_rates.append(safe_counts["proc"] / max(step_count, 1))
        overall_safe_rates.append(safe_counts["overall"] / max(step_count, 1))
        for stage_name, values in stage_rate_sums.items():
            values.append(stage_counts[stage_name] / max(step_count, 1))
        task_summary = getattr(base_env, "task_tracker", None).summary() if hasattr(base_env, "task_tracker") else {}
        deadline_rates.append(float(task_summary.get("deadline_success_rate", 0.0)))
        value_weighted_deadline_rates.append(float(task_summary.get(
            "value_weighted_deadline_success_rate",
            task_summary.get("deadline_success_rate", 0.0),
        )))
        expired_rates.append(float(task_summary.get("expired_value_rate", 0.0)))
        drop_rates.append(float(task_summary.get("dropped_value_rate", 0.0)))
        aoi_steps.append(float(task_summary.get(
            "average_aoi_steps", task_summary.get("avg_delivery_delay_steps", 0.0))))
        value_weighted_aoi_steps.append(float(task_summary.get(
            "value_weighted_aoi_steps",
            task_summary.get("average_aoi_steps", task_summary.get("avg_delivery_delay_steps", 0.0)),
        )))
        voi_loss_rates.append(float(task_summary.get("voi_loss_rate", 0.0)))

    QUEUE_CONFIG["data_arrival_rate_mbs"] = old_arrival_rate
    processed_mean = float(np.mean(processed_mbs))
    downlink_mean = float(np.mean(downlink_mbs))

    return add_paper_metrics({
        "reward_mean":       float(np.mean(rewards)),
        "reward_std":        float(np.std(rewards)),
        "delivered_value_mean": float(np.mean(delivered_values)),
        "delivered_value_std":  float(np.std(delivered_values)),
        "processed_mean_mb":  processed_mean,
        "processed_std_mb":   float(np.std(processed_mbs)),
        "downlink_mean_mb":   downlink_mean,
        "downlink_std_mb":    float(np.std(downlink_mbs)),
        "global_proc_downlink_ratio": float(np.sum(processed_mbs) / max(np.sum(downlink_mbs), 1e-9)),
        "mean_episode_proc_downlink_ratio": float(np.mean([
            processed / max(downlink, 1e-9)
            for processed, downlink in zip(processed_mbs, downlink_mbs)
        ])) if processed_mbs else 0.0,
        "proc_downlink_ratio": float(np.sum(processed_mbs) / max(np.sum(downlink_mbs), 1e-9)),
        "episode_proc_dl_ratio": float(np.mean([
            processed / max(downlink, 1e-9)
            for processed, downlink in zip(processed_mbs, downlink_mbs)
        ])) if processed_mbs else 0.0,
        "proc_dl_ratio":      float(np.sum(processed_mbs) / max(np.sum(downlink_mbs), 1e-9)),
        "comm_window_utilization": float(np.mean(window_utils)) if window_utils else 0.0,
        "processed_queue_final_utilization": float(np.mean(processed_final_utils)) if processed_final_utils else 0.0,
        "tx_active_in_contact_ratio": float(np.mean(tx_active_contact_flags)) if tx_active_contact_flags else 0.0,
        "high_value_delivery_rate": (
            float(np.mean(high_value_delivery_rates)) if high_value_delivery_rates else 0.0
        ),
        "high_value_delivery_ratio": (
            float(np.mean(high_value_delivery_rates)) if high_value_delivery_rates else 0.0
        ),
        "actual_tx_mean_mb":  downlink_mean,
        "throughput_mean":    processed_mean,  # 兼容旧结果字段：这里表示处理量，不表示下行回传。
        "survival_rate":      float(np.mean(survivals)),
        "crash_count":        int(np.sum(1.0 - np.asarray(survivals, dtype=float))),
        "orbit_safe_rate":    float(np.mean(orbit_safe_rates)),
        "energy_safe_rate":   float(np.mean(energy_safe_rates)),
        "thermal_safe_rate":  float(np.mean(thermal_safe_rates)),
        "raw_queue_safe_rate": float(np.mean(raw_safe_rates)),
        "processed_queue_safe_rate": float(np.mean(processed_safe_rates)),
        "overall_safe_rate":  float(np.mean(overall_safe_rates)),
        "step_safety_rate":   float(np.mean(overall_safe_rates)),
        "normal_state_rate":  float(np.mean(stage_rate_sums["normal"])) if stage_rate_sums["normal"] else 0.0,
        "warning_state_rate": float(np.mean(stage_rate_sums["warning"])) if stage_rate_sums["warning"] else 0.0,
        "unsafe_state_rate":  float(np.mean(stage_rate_sums["unsafe"])) if stage_rate_sums["unsafe"] else 0.0,
        "failure_state_rate": float(np.mean(stage_rate_sums["failure"])) if stage_rate_sums["failure"] else 0.0,
        "episode_safety_rate": float(np.mean(safes)),
        "safety_rate":        float(np.mean(safes)),
        "deadline_success_rate": float(np.mean(deadline_rates)),
        "value_weighted_deadline_success_rate": float(np.mean(value_weighted_deadline_rates)),
        "expired_value_rate": float(np.mean(expired_rates)),
        "voi_degradation_rate": float(np.mean(expired_rates)),
        "average_aoi_steps": float(np.mean(aoi_steps)) if aoi_steps else 0.0,
        "value_weighted_aoi_steps": float(np.mean(value_weighted_aoi_steps)) if value_weighted_aoi_steps else 0.0,
        "dropped_value_rate": float(np.mean(drop_rates)),
        "voi_loss_rate": float(np.mean(voi_loss_rates)) if voi_loss_rates else 0.0,
    })


def run_robustness(args):
    # 鲁棒性实验会复用同一个 checkpoint，在不同外部扰动下比较性能和安全率。
    print("=" * 65)
    print("  鲁棒性测试：不同扰动条件下的性能验证")
    print("=" * 65)

    full_steps = int(TRAIN_CONFIG["max_episode_steps"])
    effective_steps = int(args.max_steps) if args.max_steps is not None else full_steps
    # 半回合以上视为正式评估；更短的仅作快速烟雾检查。
    smoke_step_threshold = max(120, int(0.5 * full_steps))
    is_smoke = (effective_steps < smoke_step_threshold) or (int(args.n_episodes) < 5)

    print(f"  评估参数: episodes={args.n_episodes}, max_steps={effective_steps}"
          f" (full={full_steps})")
    if is_smoke:
        print("  [提示] 当前为快速/烟雾评估参数，"
              f"结果更适合调试（smoke 阈值={smoke_step_threshold} 步）。")

    trace_rows = None
    if args.trace_csv:
        trace_rows = _load_trace_rows(args.trace_csv)
        trace_fields = _trace_field_summary(trace_rows)
        print(f"  trace驱动: {args.trace_csv} (rows={len(trace_rows)}, "
              f"cycle={args.trace_cycle}, fields={trace_fields})")

    # ── 加载待测方法 ──────────────────────────────────────────────
    schedulers = {}

    if os.path.exists(args.checkpoint):
        ours = IntegratedScheduler(device=args.device, enable_lyapunov=True)
        ours.load(args.checkpoint)

        def _ours_fn(s, e):
            in_win = (e._contact.get("in_window", False)
                      if getattr(e, '_contact', None) else False)
            prop_can_update = True
            if hasattr(e, "step_count") and hasattr(e, "N_PROP_SMOOTH"):
                prop_can_update = (e.step_count % e.N_PROP_SMOOTH == 0)
            return ours.schedule(
                s, e.energy_queue.value, e.orbit_queue.value,
                e.data_queue.length, e.comm_queue.value,
                in_window=in_win, evaluate=True,
                h=e.altitude_m, soc=e.battery.soc, time_s=e.time_s,
                prop_can_update=prop_can_update,
                orbital_phase=e.orbit_sim.phase,
                tx_capacity_mbps=float((e._contact or {}).get("max_capacity_mbps", 0.0)),
                available_power_w=getattr(e, "available_power_w", None),
                env=e)[0]

        schedulers[OURS_NAME] = (_ours_fn, True)
    else:
        print(f"[警告] 未找到 {args.checkpoint}，跳过 {OURS_NAME}")

    mpc = MPCBaseline()

    def _mpc_fn(s, e):
        s_raw = s[0] if s.ndim == 2 else s
        return mpc.schedule(
            s_raw, e.battery.soc, e.altitude_m,
            e.orbit_sim.is_sunlit(e.time_s),
            e.solar.output_power(e.orbit_sim.sunlit_fraction(e.time_s)),
            time_s=e.time_s,
            env=e)

    schedulers["MPC"] = (_mpc_fn, False)

    robust_mpc = RobustMPCBaseline(horizon=args.robust_mpc_horizon)

    def _robust_mpc_fn(s, e):
        s_raw = s[0] if s.ndim == 2 else s
        return robust_mpc.schedule(
            s_raw, e.battery.soc, e.altitude_m,
            e.orbit_sim.is_sunlit(e.time_s),
            e.solar.output_power(e.orbit_sim.sunlit_fraction(e.time_s)),
            time_s=e.time_s,
            env=e)

    schedulers["Robust MPC"] = (_robust_mpc_fn, False)

    oracle_mpc = OracleMPCBaseline(
        horizon=args.oracle_mpc_horizon,
        beam_width=args.oracle_mpc_beam_width,
    )

    def _oracle_mpc_fn(s, e):
        s_raw = s[0] if s.ndim == 2 else s
        return oracle_mpc.schedule(s_raw, e)

    schedulers["Omniscient MPC (Oracle)"] = (_oracle_mpc_fn, False)

    dpp = DriftPlusPenaltyBaseline(V=args.dpp_V)

    def _dpp_fn(s, e):
        s_raw = s[0] if s.ndim == 2 else s
        return dpp.schedule(s_raw, e)

    schedulers["DPP"] = (_dpp_fn, False)

    heuristic = HeuristicBaseline()

    def _heu_fn(s, e):
        return heuristic.schedule(s[0] if s.ndim == 2 else s)

    schedulers["启发式"] = (_heu_fn, False)

    static = StaticRuleBaseline()

    def _static_fn(s, e):
        s_raw = s[0] if s.ndim == 2 else s
        return static.schedule(s_raw, e)

    schedulers["Static Rule"] = (_static_fn, False)

    # ── 定义扰动条件 ──────────────────────────────────────────────
    # 论文中按三类不确定性组织：轨道、能源、工作负载。condition 名称保持旧格式，
    # 额外在 metadata 中保存分组，避免破坏已有画图脚本。
    test_conditions = []
    condition_groups = {}

    def add_condition(group: str, name: str, fn) -> None:
        test_conditions.append((name, fn))
        condition_groups[name] = group

    extreme_mode = (args.profile == "extreme")

    if extreme_mode:
        # 代表性极端集：保留边界与拐点，减少总时长
        height_levels = [140, 150, 180, 300, 440]
        density_levels = [0.5, 1.0, 1.6]
        solar_levels = [0.30, 0.65, 1.0]
        arrival_levels = [0.5, 1.0, 2.0, 4.0]
        battery_levels = [0.50, 0.70, 1.0]
    else:
        height_levels = [140, 150, 180, 260, 350, 440]
        density_levels = [0.7, 0.85, 1.0, 1.15, 1.3]
        solar_levels = [0.50, 0.65, 0.80, 0.90, 1.0]
        arrival_levels = [0.5, 1.0, 1.5, 2.0, 3.0]
        battery_levels = [0.70, 0.80, 0.90, 1.0]

    def _apply_extreme_state(env, soc_cap=None, preload_data_ratio=None):
        """极端档附加状态注入：压低初始 SOC + 预加载任务队列。"""
        if soc_cap is not None:
            new_soc = min(float(env.battery.soc), float(soc_cap))
            lower_soc = float(getattr(env.battery, "soc_crash", 0.0))
            env.battery.soc = float(np.clip(new_soc, lower_soc, env.battery.soc_max))
            env.energy_queue.reset(env.battery.energy_margin_wh)

        if preload_data_ratio is not None:
            old_length = float(env.data_queue.length)
            preload = float(preload_data_ratio) * float(env.data_queue.max_length)
            env.data_queue.length = float(np.clip(
                max(old_length, preload),
                0.0,
                float(env.data_queue.max_length),
            ))
            if hasattr(env.data_queue, "prev_length"):
                env.data_queue.prev_length = env.data_queue.length
            added_mb = max(0.0, env.data_queue.length - old_length)
            if added_mb > 1e-9 and hasattr(env, "task_tracker"):
                # 预加载 raw queue 必须同步任务价值批次，否则后续处理/下传会变成无价值数据。
                value_density = float(TASK_CONFIG.get("base_value_per_mb", 1.0))
                env.task_tracker.raw_batches.append(TaskBatch(
                    mb=added_mb,
                    value=added_mb * value_density,
                    priority=1.0,
                    quality=1.0,
                    deadline_steps=int(TASK_CONFIG.get("deadline_max_steps", 360)),
                    created_step=int(getattr(env, "step_count", 0) or 0),
                ))
                env.task_tracker.total_generated_mb += added_mb
                env.task_tracker.total_generated_value += added_mb * value_density

    force_trace_altitude = bool(trace_rows) and args.trace_altitude_mode == "force"
    if force_trace_altitude:
        print("  [提示] trace_altitude_mode=force：跳过会被 trace 覆盖的初始高度条件。")
    else:
        for h_km in height_levels:
            def make_h(h):
                def cond(env):
                    # 直接设置高度；reset() 已经执行完毕。
                    env.altitude_m = h * 1e3
                    # 同步更新轨道相关子系统
                    env.orbit_queue.reset(env.altitude_m)
                    if extreme_mode and h <= 180:
                        if h <= 150:
                            _apply_extreme_state(env, soc_cap=0.18, preload_data_ratio=0.50)
                        else:
                            _apply_extreme_state(env, soc_cap=0.22, preload_data_ratio=0.30)
                return cond
            add_condition("orbital_uncertainty", f"初始高度 {h_km}km", make_h(h_km))

    for factor in density_levels:
        def make_drag(f):
            def cond(env):
                env.orbit_dyn.atm.rho_ref = DRAG_CONFIG["rho_ref"] * f
                if extreme_mode and f >= 1.6:
                    _apply_extreme_state(env, soc_cap=0.22, preload_data_ratio=0.25)
            return cond
        add_condition("orbital_uncertainty", f"大气密度 ×{factor:.2f}", make_drag(factor))

    for eff in solar_levels:
        def make_solar(e):
            def cond(env):
                env.solar.eta = ENERGY_CONFIG["solar_efficiency"] * e
                if extreme_mode and e <= 0.30:
                    _apply_extreme_state(env, soc_cap=0.20, preload_data_ratio=0.30)
            return cond
        add_condition("energy_uncertainty", f"太阳能效率 {eff*100:.0f}%", make_solar(eff))

    old_base_arrival_rate = float(QUEUE_CONFIG.get("data_arrival_rate_mbs", 2.0))
    for rate_factor in arrival_levels:
        def make_rate(f):
            def cond(env):
                QUEUE_CONFIG["data_arrival_rate_mbs"] = old_base_arrival_rate * f
                if extreme_mode:
                    if f >= 4.0:
                        _apply_extreme_state(env, soc_cap=0.20, preload_data_ratio=0.45)
                    elif f >= 2.0:
                        _apply_extreme_state(env, soc_cap=0.25, preload_data_ratio=0.30)
            return cond
        add_condition("workload_uncertainty", f"任务到达率 ×{rate_factor:.1f}", make_rate(rate_factor))

    for cap_factor in battery_levels:
        def make_batt(f):
            def cond(env):
                env.battery.capacity_wh = ENERGY_CONFIG["battery_capacity_wh"] * f
                if extreme_mode:
                    if f <= 0.50:
                        _apply_extreme_state(env, soc_cap=0.20, preload_data_ratio=0.35)
                    elif f <= 0.70:
                        _apply_extreme_state(env, soc_cap=0.25, preload_data_ratio=0.25)
            return cond
        add_condition("energy_uncertainty", f"电池容量 {cap_factor*100:.0f}%", make_batt(cap_factor))

    print(f"  扰动配置: profile={args.profile}, conditions={len(test_conditions)}")

    # ── 执行测试 ──────────────────────────────────────────────────
    all_results = {}

    for cond_name, cond_fn in test_conditions:
        group = condition_groups.get(cond_name, "uncategorized")
        print(f"\n[条件] {group}/{cond_name}")
        cond_results = {}
        for sched_name, (sched_fn, use_wrap) in schedulers.items():
            r = _eval_under_condition(
                sched_fn, cond_fn,
                args.n_episodes,
                use_wrapper=use_wrap,
                max_steps=args.max_steps,
                trace_rows=trace_rows,
                trace_cycle=args.trace_cycle,
                trace_altitude_mode=args.trace_altitude_mode)
            cond_results[sched_name] = r
            print(f"  {sched_name:<12} 奖励:{r['reward_mean']:>8.1f}"
                  f"  价值:{r['delivered_value_mean']:>7.1f}"
                  f"  回传:{r['downlink_mean_mb']:>7.1f}MB"
                  f"  处理:{r['processed_mean_mb']:>7.1f}MB"
                  f"  综合安全:{r['overall_safe_rate']:.1%}")
        all_results[cond_name] = cond_results

    # ── 打印汇总 ─────────────────────────────────────────────────
    if OURS_NAME in schedulers and "Static Rule" in schedulers:
        print("\n" + "=" * 65)
        print(f"  鲁棒性汇总：{OURS_NAME} vs Static Rule 性能提升")
        print("  " + "-" * 63)
        for cond_name, cond_r in all_results.items():
            our_r = cond_r.get(OURS_NAME, {}).get("reward_mean", 0)
            base_r = cond_r.get("Static Rule", {}).get("reward_mean", 1)
            our_value = cond_r.get(OURS_NAME, {}).get("delivered_value_mean", 0)
            base_value = cond_r.get("Static Rule", {}).get("delivered_value_mean", 0)
            our_downlink = cond_r.get(OURS_NAME, {}).get("downlink_mean_mb", 0)
            base_downlink = cond_r.get("Static Rule", {}).get("downlink_mean_mb", 0)
            our_s = cond_r.get(OURS_NAME, {}).get("overall_safe_rate", 0)
            if base_r != 0:
                impr = (our_r - base_r) / abs(base_r) * 100
                value_text = "基线为0"
                if abs(base_value) > 1e-9:
                    value_text = f"{(our_value - base_value) / abs(base_value) * 100:>+7.1f}%"
                downlink_impr = None
                if abs(base_downlink) > 1e-9:
                    downlink_impr = (our_downlink - base_downlink) / abs(base_downlink) * 100
                downlink_text = (f"{downlink_impr:>+7.1f}%"
                                 if downlink_impr is not None else "基线为0")
                print(f"  {cond_name:<22}  "
                      f"奖励提升: {impr:>+7.1f}%  "
                      f"价值提升: {value_text}  "
                      f"回传提升: {downlink_text}  "
                      f"安全率: {our_s:.1%}")
        print("=" * 65)

    # ── 保存结果 ──────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/robustness_{ts}.json"
    out_obj = {
        "__meta__": {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": args.checkpoint,
            "device": args.device,
            "n_episodes": int(args.n_episodes),
            "max_steps": effective_steps,
            "full_episode_steps": full_steps,
            "is_smoke": bool(is_smoke),
            "profile": args.profile,
            "trace_csv": args.trace_csv,
            "trace_rows": int(len(trace_rows)) if trace_rows else 0,
            "trace_cycle": bool(args.trace_cycle),
            "trace_fields": _trace_field_summary(trace_rows),
            "trace_altitude_mode": args.trace_altitude_mode,
            "uncertainty_taxonomy": {
                "orbital_uncertainty": ["initial altitude", "atmospheric density"],
                "energy_uncertainty": ["solar efficiency", "battery capacity"],
                "workload_uncertainty": ["task arrival rate", "trace-driven workload"],
            },
            "condition_groups": dict(condition_groups),
            "mpc_taxonomy": {
                "MPC": "myopic current-observation MPC",
                "Robust MPC": "myopic scenario-robust MPC",
                "Omniscient MPC (Oracle)": (
                    "non-deployable future-rollout upper-bound proxy using copied environment"
                ),
            },
        }
    }
    out_obj.update(all_results)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {out_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="鲁棒性测试")
    parser.add_argument("--checkpoint",
                        default=DEFAULT_OPTIMIZED_CHECKPOINT)
    parser.add_argument("--n_episodes", type=int, default=int(TRAIN_CONFIG.get("eval_episodes", 30)))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--robust_mpc_horizon", type=int, default=8,
                        help="Robust MPC 的预测窗口")
    parser.add_argument("--oracle_mpc_horizon", type=int, default=12,
                        help="Omniscient MPC 复制环境 rollout 的预测窗口")
    parser.add_argument("--oracle_mpc_beam_width", type=int, default=8,
                        help="Omniscient MPC beam search 保留宽度")
    parser.add_argument("--dpp_V", type=float, default=8.0,
                        help="DPP 基线的吞吐权重 V")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="可选：每个 episode 仅评估前 max_steps 步（用于快速烟雾检查）")
    parser.add_argument("--profile", choices=["standard", "extreme"], default="standard",
                        help="扰动档位：standard(默认) 或 extreme(更苛刻，拉开差异)")
    parser.add_argument("--trace_csv", default=None,
                        help="可选：真实外生扰动序列 CSV（列如 rho_scale、solar_scale 或 sunlit）")
    parser.add_argument("--trace_cycle", action="store_true",
                        help="trace 长度不足时循环复用（默认使用末行保持）")
    parser.add_argument("--trace_altitude_mode", choices=["ignore", "force"], default="ignore",
                        help="是否用 trace altitude_km 强制覆盖环境高度；默认只用于记录/密度扰动")
    args = parser.parse_args()
    run_robustness(args)

"""
compare_all.py
全方法对比评估

对比方法（当前脚本实际评估）：
  LS-PSF CMDP、MPC、Robust MPC、DPP、Greedy Value、EDF、LLF、
  Heuristic、Value-aware Heuristic、Static Rule。
  Ours 的 CPU throttle / work-conserving 开关只作为可选诊断表输出，不进入论文主表。
  另含 Omniscient MPC，用环境复制 rollout 提供上帝视角上界代理。

说明：
  SAC、SAC+PSF、SAC+Lyapunov、SAC-Lagrangian 需要独立训练 checkpoint，不在本脚本中凭空声明。
  若要报告这些学习型 baseline，请使用 experiments/ablation.py --train_independent_models
  产出独立模型后再纳入表格。

评估口径：
  1. 所有方法都在当前环境上评估，确保论文主表口径一致
  2. LS-PSF CMDP 使用 DilatedFrameStackWrapper，与训练保持一致
  3. 基线方法只读取当前原始 obs，避免误用帧堆叠序列
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    # 降低导入劫持风险：不要把项目路径插到最高优先级
    sys.path.append(_PROJECT_ROOT)
import numpy as np
import json
import argparse
from contextlib import contextmanager
from datetime import datetime
import torch
from environment.satellite_env import VLEOSatelliteEnv
from environment.wrappers import DilatedFrameStackWrapper
from scheduler.integrated_scheduler import IntegratedScheduler
from baselines.mpc_baseline import MPCBaseline
from baselines.robust_mpc_baseline import RobustMPCBaseline
from baselines.oracle_mpc_baseline import OracleMPCBaseline
from baselines.dpp_baseline import DriftPlusPenaltyBaseline
from baselines.heuristic_baseline import HeuristicBaseline, ValueAwareHeuristicBaseline
from baselines.value_baselines import GreedyValueBaseline, EDFBaseline, LLFBaseline, StaticRuleBaseline
from safety.lyapunov_projection import LyapunovActionProjection
from utils.paper_metrics import add_paper_metrics, compact_paper_table_row
from config import TRAIN_CONFIG, DRL_CONFIG, ORBITAL_CONFIG, ENERGY_CONFIG, TASK_CONFIG

ALTITUDE_SAFE_KM = float(ORBITAL_CONFIG["altitude_min_km"])
BATTERY_SAFE_SOC = float(ENERGY_CONFIG["battery_min_soc"])

DEFAULT_OPTIMIZED_CHECKPOINT = os.path.join(
    TRAIN_CONFIG.get("optimized_checkpoint_dir", "checkpoints_optimized/"),
    "best_optimized.pt",
)

OURS_NAME = "LS-PSF CMDP (Ours)"


@contextmanager
def _temporary_task_config(overrides: dict | None):
    """Temporarily override task-scheduling config during one evaluation."""
    overrides = dict(overrides or {})
    old_values = {key: TASK_CONFIG.get(key, None) for key in overrides}
    missing = {key for key in overrides if key not in TASK_CONFIG}
    try:
        TASK_CONFIG.update(overrides)
        yield
    finally:
        for key, old_value in old_values.items():
            if key in missing:
                TASK_CONFIG.pop(key, None)
            else:
                TASK_CONFIG[key] = old_value


def _resolve_device(device_arg: str) -> str:
    req = (device_arg or "auto").lower()
    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if req == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if req == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return req


def _available_power_w(env) -> float | None:
    """读取环境的动态可用功率，供动作边界裁剪层使用。"""
    try:
        return float(getattr(env, "available_power_w"))
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════
# 当前环境评估（LS-PSF CMDP + 基线）
# ══════════════════════════════════════════════════════════════════════
def evaluate_on_env(scheduler_fn, n_episodes: int = None,
                   seed_offset: int = 200,
                   use_wrapper: str = "none",
                   max_steps: int = None) -> dict:
    """
    在 VLEOSatelliteEnv 上评估。
    Args:
        scheduler_fn: callable(state, env) → action
        use_wrapper:  "none" | "dilated"
    """
    n_episodes = int(TRAIN_CONFIG.get("eval_episodes", 30) if n_episodes is None else n_episodes)
    lyapunov_opt = LyapunovActionProjection()
    rewards, throughputs, tx_mbs, delivered_values, safes, survivals = [], [], [], [], [], []
    proc_dl_ratios, high_value_delivery_rates = [], []
    orbit_safe_rates, energy_safe_rates, thermal_safe_rates = [], [], []
    raw_safe_rates, processed_safe_rates, overall_safe_rates = [], [], []
    stage_rate_sums = {"normal": [], "warning": [], "unsafe": [], "failure": []}
    deadline_rates, expired_rates, drop_rates, aoi_steps = [], [], [], []
    value_weighted_deadline_rates, value_weighted_aoi_steps, voi_loss_rates = [], [], []
    orbit_viols, energy_viols, thermal_viols, raw_viols, processed_viols, lyapunov_finals = [], [], [], [], [], []
    prop_powers, cpu_powers, tx_powers, energy_whs, episode_energy_per_value, window_utils = [], [], [], [], [], []
    processed_final_utils, processed_queue_utils, processed_future_contact_ratios, tx_active_contact_flags = [], [], [], []

    k = DRL_CONFIG.get("frame_stack", 8)

    for ep in range(n_episodes):
        # 每个 episode 换一个 seed，保证比较结果不是单一初始相位/队列状态的偶然结果。
        base_env = VLEOSatelliteEnv(seed=seed_offset + ep)

        if use_wrapper == "dilated":
            env = DilatedFrameStackWrapper(base_env, k=k)
        else:
            env = base_env

        state = env.reset()
        ep_reward = ep_tput = ep_tx = ep_value = 0.0
        ep_energy_wh = 0.0
        ep_high_delivered = ep_high_expired = ep_high_dropped = 0.0
        ep_final_processed_util = 0.0
        orbit_v = energy_v = thermal_v = raw_v = proc_v = 0
        safe_counts = {"orbit": 0, "energy": 0, "thermal": 0, "raw": 0, "proc": 0, "overall": 0}
        stage_counts = {"normal": 0, "warning": 0, "unsafe": 0, "failure": 0}
        done = False
        step_count = 0
        survived = True

        while not done:
            # scheduler_fn 统一屏蔽各方法接口差异，评估循环只负责执行动作和统计指标。
            action = scheduler_fn(state, env)
            if use_wrapper == "dilated":
                state, reward, done, info = env.step(
                    action, enforce_prop_smoothing=False)
            else:
                state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_tput += info.get(
                "processed_mb",
                info.get("service_rate_mbs", 0) * TRAIN_CONFIG["time_slot_s"],
            )
            ep_tx += info.get("delivered_mb", info.get("actual_tx_mb", 0))
            ep_value += info.get("delivered_value", 0.0)
            ep_high_delivered += float(info.get("delivered_high_value", 0.0))
            ep_high_expired += float(info.get("expired_high_value", 0.0))
            ep_high_dropped += float(info.get("dropped_high_value", 0.0))
            ep_final_processed_util = float(info.get("processed_queue_utilization", 0.0))
            processed_queue_utils.append(ep_final_processed_util)
            processed_future_contact_ratios.append(float(
                info.get("processed_queue_future_contact_ratio_raw", info.get("processed_queue_future_contact_ratio", 0.0))))
            prop_powers.append(float(info.get("P_propulsion_w", 0.0)))
            cpu_powers.append(float(info.get("P_cpu_w", 0.0)))
            tx_powers.append(float(info.get("P_tx_w", 0.0)))
            step_energy_wh = float(info.get("P_total_w", 0.0)) * TRAIN_CONFIG["time_slot_s"] / 3600.0
            energy_whs.append(step_energy_wh)
            ep_energy_wh += step_energy_wh
            capacity_mb = float(info.get("tx_capacity_mbps", 0.0)) * TRAIN_CONFIG["time_slot_s"] / 8.0
            if bool(info.get("in_window", False)) and capacity_mb > 1e-9:
                window_utils.append(float(info.get("delivered_mb", info.get("actual_tx_mb", 0.0))) / capacity_mb)
                tx_active_contact_flags.append(float(
                    info.get("delivered_mb", info.get("actual_tx_mb", 0.0)) > 1e-9
                ))
            step_count += 1
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
            if not orbit_safe:
                orbit_v += 1
            if not energy_safe:
                energy_v += 1
            if not thermal_safe:
                thermal_v += 1
            if not raw_safe:
                raw_v += 1
            if not proc_safe:
                proc_v += 1
            stage = str(info.get("risk_stage", "normal"))
            if stage not in stage_counts:
                stage = "failure" if bool(info.get("crashed", False)) else "normal"
            stage_counts[stage] += 1
            if max_steps is not None and step_count >= max_steps:
                done = True

        rewards.append(ep_reward)
        throughputs.append(ep_tput)
        tx_mbs.append(ep_tx)
        delivered_values.append(ep_value)
        proc_dl_ratios.append(float(ep_tput / max(ep_tx, 1e-9)))
        episode_energy_per_value.append(float(ep_energy_wh / max(ep_value, 1e-9)))
        high_den = ep_high_delivered + ep_high_expired + ep_high_dropped
        high_value_delivery_rates.append(float(ep_high_delivered / max(high_den, 1e-9)))
        processed_final_utils.append(float(ep_final_processed_util))
        safes.append(float(orbit_v == 0 and energy_v == 0 and thermal_v == 0 and raw_v == 0 and proc_v == 0))
        survivals.append(float(survived))
        orbit_safe_rates.append(safe_counts["orbit"] / max(step_count, 1))
        energy_safe_rates.append(safe_counts["energy"] / max(step_count, 1))
        thermal_safe_rates.append(safe_counts["thermal"] / max(step_count, 1))
        raw_safe_rates.append(safe_counts["raw"] / max(step_count, 1))
        processed_safe_rates.append(safe_counts["proc"] / max(step_count, 1))
        overall_safe_rates.append(safe_counts["overall"] / max(step_count, 1))
        for stage_name, values in stage_rate_sums.items():
            values.append(stage_counts[stage_name] / max(step_count, 1))
        orbit_viols.append(orbit_v)
        energy_viols.append(energy_v)
        thermal_viols.append(thermal_v)
        raw_viols.append(raw_v)
        processed_viols.append(proc_v)
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

        # Lyapunov 终值（通过 wrapper 或直接访问 base_env）
        _env = base_env if use_wrapper == "none" else env
        lyapunov_finals.append(lyapunov_opt.lyapunov_function(
            _env.energy_queue.value, _env.orbit_queue.value,
            _env.data_queue.length, _env.comm_queue.value))

    return add_paper_metrics({
        # processed/downlink 同时保留：前者是星上处理量，后者才是论文主目标的有效回传量。
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "delivered_value_mean": float(np.mean(delivered_values)),
        "delivered_value_std": float(np.std(delivered_values)),
        "processed_mean": float(np.mean(throughputs)),
        "downlink_mean": float(np.mean(tx_mbs)),
        "global_proc_downlink_ratio": float(np.sum(throughputs) / max(np.sum(tx_mbs), 1e-9)),
        "mean_episode_proc_downlink_ratio": float(np.mean(proc_dl_ratios)) if proc_dl_ratios else 0.0,
        "proc_downlink_ratio": float(np.sum(throughputs) / max(np.sum(tx_mbs), 1e-9)),
        "episode_proc_dl_ratio": float(np.mean(proc_dl_ratios)) if proc_dl_ratios else 0.0,
        "proc_dl_ratio": float(np.sum(throughputs) / max(np.sum(tx_mbs), 1e-9)),
        "high_value_delivery_rate": (
            float(np.mean(high_value_delivery_rates)) if high_value_delivery_rates else 0.0
        ),
        "high_value_delivery_ratio": (
            float(np.mean(high_value_delivery_rates)) if high_value_delivery_rates else 0.0
        ),
        "throughput_mean": float(np.mean(throughputs)),
        "tx_mb_mean": float(np.mean(tx_mbs)),
        "survival_rate": float(np.mean(survivals)),
        "crash_count": int(np.sum(1.0 - np.asarray(survivals, dtype=float))),
        "episode_safety_rate": float(np.mean(safes)),
        "orbit_safe_rate": float(np.mean(orbit_safe_rates)),
        "energy_safe_rate": float(np.mean(energy_safe_rates)),
        "thermal_safe_rate": float(np.mean(thermal_safe_rates)),
        "raw_queue_safe_rate": float(np.mean(raw_safe_rates)),
        "processed_queue_safe_rate": float(np.mean(processed_safe_rates)),
        "overall_safe_rate": float(np.mean(overall_safe_rates)),
        "step_safety_rate": float(np.mean(overall_safe_rates)),
        "normal_state_rate": float(np.mean(stage_rate_sums["normal"])) if stage_rate_sums["normal"] else 0.0,
        "warning_state_rate": float(np.mean(stage_rate_sums["warning"])) if stage_rate_sums["warning"] else 0.0,
        "unsafe_state_rate": float(np.mean(stage_rate_sums["unsafe"])) if stage_rate_sums["unsafe"] else 0.0,
        "failure_state_rate": float(np.mean(stage_rate_sums["failure"])) if stage_rate_sums["failure"] else 0.0,
        "safety_rate": float(np.mean(safes)),
        "deadline_success_rate": float(np.mean(deadline_rates)),
        "value_weighted_deadline_success_rate": float(np.mean(value_weighted_deadline_rates)),
        "expired_value_rate": float(np.mean(expired_rates)),
        "voi_degradation_rate": float(np.mean(expired_rates)),
        "average_aoi_steps": float(np.mean(aoi_steps)) if aoi_steps else 0.0,
        "value_weighted_aoi_steps": float(np.mean(value_weighted_aoi_steps)) if value_weighted_aoi_steps else 0.0,
        "dropped_value_rate": float(np.mean(drop_rates)),
        "voi_loss_rate": float(np.mean(voi_loss_rates)) if voi_loss_rates else 0.0,
        "mean_prop_power": float(np.mean(prop_powers)) if prop_powers else 0.0,
        "mean_cpu_power": float(np.mean(cpu_powers)) if cpu_powers else 0.0,
        "mean_tx_power": float(np.mean(tx_powers)) if tx_powers else 0.0,
        "energy_efficiency": float(np.sum(delivered_values) / max(np.sum(energy_whs), 1e-9)),
        "energy_per_value": float(np.mean(episode_energy_per_value)) if episode_energy_per_value else 0.0,
        "energy_per_delivered_value_episode": float(np.mean(episode_energy_per_value)) if episode_energy_per_value else 0.0,
        "comm_window_utilization": float(np.mean(window_utils)) if window_utils else 0.0,
        "processed_queue_final_utilization": float(np.mean(processed_final_utils)) if processed_final_utils else 0.0,
        "processed_queue_peak_utilization": float(np.max(processed_queue_utils)) if processed_queue_utils else 0.0,
        "processed_queue_p95_utilization": float(np.percentile(processed_queue_utils, 95)) if processed_queue_utils else 0.0,
        "processed_queue_future_contact_ratio": float(np.mean(processed_future_contact_ratios)) if processed_future_contact_ratios else 0.0,
        "processed_queue_future_contact_ratio_p95": float(np.percentile(processed_future_contact_ratios, 95)) if processed_future_contact_ratios else 0.0,
        "processed_queue_future_contact_ratio_peak": float(np.max(processed_future_contact_ratios)) if processed_future_contact_ratios else 0.0,
        "tx_active_in_contact_ratio": float(np.mean(tx_active_contact_flags)) if tx_active_contact_flags else 0.0,
        "orbit_viol_mean": float(np.mean(orbit_viols)),
        "energy_viol_mean": float(np.mean(energy_viols)),
        "energy_violation_rate": float(np.mean(np.asarray(energy_viols, dtype=float) > 0.0)) if energy_viols else 0.0,
        "energy_unsafe_rate": float(np.mean(np.asarray(energy_safe_rates, dtype=float) < 1.0)) if energy_safe_rates else 0.0,
        "thermal_viol_mean": float(np.mean(thermal_viols)),
        "raw_queue_viol_mean": float(np.mean(raw_viols)),
        "processed_queue_viol_mean": float(np.mean(processed_viols)),
        "lyapunov_final_mean": float(np.mean(lyapunov_finals)),
    })


def _get_raw_state(state):
    """从可能的帧堆叠状态中提取最新一帧的原始观测"""
    # DilatedFrameStackWrapper 的顺序是 [当前, 更早, ...]，所以最新帧是 index 0。
    if state.ndim == 2:
        return state[0]
    return state


def _learned_scheduler_fn(scheduler: IntegratedScheduler):
    """统一学习型方法的评估入口，确保上下文和训练阶段一致。"""
    def _fn(state, env):
        in_window = (env._contact.get("in_window", False)
                     if env._contact is not None else False)
        prop_can_update = True
        if hasattr(env, "step_count") and hasattr(env, "N_PROP_SMOOTH"):
            prop_can_update = (env.step_count % env.N_PROP_SMOOTH == 0)
        action, _, _, _ = scheduler.schedule(
            state,
            env.energy_queue.value, env.orbit_queue.value,
            env.data_queue.length, env.comm_queue.value,
            in_window=in_window, evaluate=True,
            h=env.altitude_m, soc=env.battery.soc,
            time_s=env.time_s,
            prop_can_update=prop_can_update,
            orbital_phase=env.orbit_sim.phase,
            tx_capacity_mbps=float((env._contact or {}).get("max_capacity_mbps", 0.0)),
            available_power_w=_available_power_w(env),
            env=env)
        return action
    return _fn


def _same_checkpoint_path(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    try:
        return os.path.abspath(left) == os.path.abspath(right)
    except Exception:
        return False


def _first_metric(stats: dict, keys: tuple[str, ...]) -> float:
    for key in keys:
        if key in stats:
            return float(stats.get(key) or 0.0)
    return 0.0


def _delivery_summary(stats: dict | None) -> dict:
    stats = stats if isinstance(stats, dict) else {}
    delivered_value = _first_metric(stats, (
        "delivered_value_mean",
        "delivered_value_total",
        "Delivered VoI",
    ))
    downlink_mb = _first_metric(stats, (
        "downlink_mean",
        "downlink_mean_mb",
        "tx_mb_mean",
        "tx_mb_total",
        "Downlink MB",
    ))
    processed_mb = _first_metric(stats, (
        "processed_mean",
        "processed_mean_mb",
        "throughput_mean",
        "throughput_mean_mb",
        "throughput_total",
        "Processed MB",
    ))
    return {
        "delivered_value": float(delivered_value),
        "downlink_mb": float(downlink_mb),
        "processed_mb": float(processed_mb),
        "nonzero_delivery": bool(max(delivered_value, downlink_mb) > 1e-9),
    }


def _paper_table_delivery_check(
    results: dict,
    *,
    allow_zero_delivery: bool,
    main_method_name: str = OURS_NAME,
) -> dict:
    """Guard against invalid formal tables with no delivery, especially for Ours."""
    max_value = 0.0
    max_downlink = 0.0
    max_processed = 0.0
    method_summaries = {}
    for method_name, stats in results.items():
        if not isinstance(stats, dict):
            continue
        summary = _delivery_summary(stats)
        method_summaries[str(method_name)] = summary
        max_value = max(max_value, summary["delivered_value"])
        max_downlink = max(max_downlink, summary["downlink_mb"])
        max_processed = max(max_processed, summary["processed_mb"])
    nonzero_delivery = max(max_value, max_downlink) > 1e-9
    main_summary = method_summaries.get(str(main_method_name), _delivery_summary(None))
    main_method_present = str(main_method_name) in method_summaries
    main_method_nonzero = bool(main_summary["nonzero_delivery"])
    check = {
        "nonzero_delivery": bool(nonzero_delivery),
        "main_method_present": bool(main_method_present),
        "main_method_nonzero_delivery": bool(main_method_nonzero),
        "main_method_name": str(main_method_name),
        "main_method_delivered_value": float(main_summary["delivered_value"]),
        "main_method_downlink_mb": float(main_summary["downlink_mb"]),
        "main_method_processed_mb": float(main_summary["processed_mb"]),
        "max_delivered_value": float(max_value),
        "max_downlink_mb": float(max_downlink),
        "max_processed_mb": float(max_processed),
        "allow_zero_delivery": bool(allow_zero_delivery),
        "method_delivery_summaries": method_summaries,
    }
    if not nonzero_delivery and not allow_zero_delivery:
        raise RuntimeError(
            "compare_all 检测到所有方法的 delivered_value 和 downlink 都为 0，"
            "这通常说明通信窗口、下传 pipeline 或评估步长有问题，不能作为论文主表。"
            "如果只是短步数 smoke/debug，请显式添加 --allow_zero_delivery；"
            "正式评估请使用足够长的 episode 并检查 in_window/contact/downlink 指标。"
        )
    if main_method_present and not main_method_nonzero and not allow_zero_delivery:
        raise RuntimeError(
            f"compare_all 检测到主方法 {main_method_name} 的 delivered_value 和 downlink 都为 0，"
            "即使其他 baseline 有下传，也不能把该结果作为有效论文主表。"
            "请检查主方法 checkpoint、通信窗口、动作输出和 downlink pipeline；"
            "如果只是短步数 smoke/debug，请显式添加 --allow_zero_delivery。"
        )
    return check


def _evaluate_learned_checkpoint(results: dict, name: str,
                                 checkpoint_path: str | None,
                                 args,
                                 *,
                                 enable_lyapunov: bool,
                                 use_psf: bool,
                                 task_config_overrides: dict | None = None,
                                 require_independent_checkpoint: bool = True) -> bool:
    if not checkpoint_path:
        return False
    if not os.path.exists(checkpoint_path):
        print(f"  [跳过] 未找到 {name} checkpoint: {checkpoint_path}")
        return False
    if (
        require_independent_checkpoint
        and name != OURS_NAME
        and _same_checkpoint_path(checkpoint_path, args.checkpoint)
        and not bool(getattr(args, "allow_posthoc_learning_baselines", False))
    ):
        raise ValueError(
            f"{name} 使用了与主方法相同的 checkpoint: {checkpoint_path}。"
            "正式学习型 baseline 必须独立训练；"
            "若只是补充诊断，请显式添加 --allow_posthoc_learning_baselines。"
        )

    with _temporary_task_config(task_config_overrides):
        scheduler = IntegratedScheduler(
            device=args.device,
            enable_lyapunov=enable_lyapunov,
            use_psf=use_psf,
        )
        if not enable_lyapunov:
            scheduler.agent.set_lyapunov_penalty_coeff(0.0)
        # 论文对比以“方法标签”定义安全链路；加载权重时不让 checkpoint metadata 覆盖当前开关。
        scheduler.load(checkpoint_path, restore_safety_config=False)
        if not enable_lyapunov:
            scheduler.agent.set_lyapunov_penalty_coeff(0.0)
        results[name] = evaluate_on_env(
            _learned_scheduler_fn(scheduler),
            args.n_episodes,
            use_wrapper="dilated",
            max_steps=args.max_steps,
        )
        if task_config_overrides:
            results[name]["task_config_overrides"] = dict(task_config_overrides)
    print(f"  {name}: {results[name]}")
    return True


def run_compare_all(args):
    results = {}
    diagnostic_results = {}
    args.device = _resolve_device(args.device)

    # ── 1. LS-PSF CMDP 主方法 ────────────────────────────────────
    ours_loaded = _evaluate_learned_checkpoint(
        results, OURS_NAME, args.checkpoint, args,
        enable_lyapunov=True, use_psf=True,
        require_independent_checkpoint=False)
    if not ours_loaded and not bool(getattr(args, "allow_missing_ours", False)):
        raise FileNotFoundError(
            f"未找到主方法 checkpoint: {args.checkpoint}。"
            "compare_all 默认不生成缺少 Ours 的论文主表；"
            "请先训练得到 checkpoints_optimized/best_optimized.pt，"
            "或仅做 baseline smoke 时显式加 --allow_missing_ours。"
        )
    if (
        ours_loaded
        and bool(getattr(args, "include_deployment_ablations", False))
        and not bool(getattr(args, "skip_config_ablations", False))
    ):
        _evaluate_learned_checkpoint(
            diagnostic_results, "Ours + CPU throttle (deployment)", args.checkpoint, args,
            enable_lyapunov=True, use_psf=True,
            task_config_overrides={"enable_cpu_throttle": True},
            require_independent_checkpoint=False)
        _evaluate_learned_checkpoint(
            diagnostic_results, "Ours w/o Work-Conserving", args.checkpoint, args,
            enable_lyapunov=True, use_psf=True,
            task_config_overrides={"work_conserving_reallocation": False},
            require_independent_checkpoint=False)
    _evaluate_learned_checkpoint(
        results, "SAC w/o Safety", getattr(args, "sac_checkpoint", None), args,
        enable_lyapunov=False, use_psf=False)
    _evaluate_learned_checkpoint(
        results, "SAC-Lagrangian", getattr(args, "sac_lagrangian_checkpoint", None), args,
        enable_lyapunov=False, use_psf=False)
    _evaluate_learned_checkpoint(
        results, "SAC + PSF", getattr(args, "sac_psf_checkpoint", None), args,
        enable_lyapunov=False, use_psf=True)
    _evaluate_learned_checkpoint(
        results, "SAC + Lyapunov", getattr(args, "sac_lya_checkpoint", None), args,
        enable_lyapunov=True, use_psf=False)

    # ── 3. MPC 基线 (当前环境) ─────────────────────────────────────
    mpc = MPCBaseline(horizon=args.mpc_horizon)

    def mpc_fn(state, env):
        # MPC 不使用帧堆叠历史，只读取当前原始状态和环境物理量做短视野预测。
        s = _get_raw_state(state)
        return mpc.schedule(
            s, env.battery.soc, env.altitude_m,
            env.orbit_sim.is_sunlit(env.time_s),
            env.solar.output_power(env.orbit_sim.sunlit_fraction(env.time_s)),
            time_s=env.time_s,
            env=env)

    results["MPC"] = evaluate_on_env(
        mpc_fn, args.n_episodes, use_wrapper="none", max_steps=args.max_steps)
    print(f"  MPC: {results['MPC']}")

    # ── 4. Robust MPC 基线 (当前环境) ──────────────────────────────
    robust_mpc = RobustMPCBaseline(horizon=args.robust_mpc_horizon)

    def robust_mpc_fn(state, env):
        s = _get_raw_state(state)
        return robust_mpc.schedule(
            s, env.battery.soc, env.altitude_m,
            env.orbit_sim.is_sunlit(env.time_s),
            env.solar.output_power(env.orbit_sim.sunlit_fraction(env.time_s)),
            time_s=env.time_s,
            env=env)

    results["Robust MPC"] = evaluate_on_env(
        robust_mpc_fn, args.n_episodes, use_wrapper="none", max_steps=args.max_steps)
    print(f"  Robust MPC: {results['Robust MPC']}")

    oracle_mpc = OracleMPCBaseline(
        horizon=args.oracle_mpc_horizon,
        beam_width=args.oracle_mpc_beam_width,
    )

    def oracle_mpc_fn(state, env):
        return oracle_mpc.schedule(_get_raw_state(state), env)

    results["Omniscient MPC (Oracle)"] = evaluate_on_env(
        oracle_mpc_fn, args.n_episodes, use_wrapper="none", max_steps=args.max_steps)
    results["Omniscient MPC (Oracle)"]["oracle_metadata"] = oracle_mpc.metadata
    print(f"  Omniscient MPC (Oracle): {results['Omniscient MPC (Oracle)']}")

    # ── 5. DPP 基线 (当前环境) ─────────────────────────────────────
    dpp = DriftPlusPenaltyBaseline(V=args.dpp_V)

    def dpp_fn(state, env):
        # DPP 直接用四队列和当前通信窗口做一步漂移加惩罚优化，是更强的传统非学习基线。
        return dpp.schedule(_get_raw_state(state), env)

    results["DPP"] = evaluate_on_env(
        dpp_fn, args.n_episodes, use_wrapper="none", max_steps=args.max_steps)
    print(f"  DPP: {results['DPP']}")

    # ── 6. Greedy Value / EDF 时效任务基线 ───────────────────────
    greedy_value = GreedyValueBaseline()

    def greedy_value_fn(state, env):
        return greedy_value.schedule(_get_raw_state(state), env)

    results["Greedy Value"] = evaluate_on_env(
        greedy_value_fn, args.n_episodes, use_wrapper="none", max_steps=args.max_steps)
    print(f"  Greedy Value: {results['Greedy Value']}")

    edf = EDFBaseline()

    def edf_fn(state, env):
        return edf.schedule(_get_raw_state(state), env)

    results["EDF"] = evaluate_on_env(
        edf_fn, args.n_episodes, use_wrapper="none", max_steps=args.max_steps)
    print(f"  EDF: {results['EDF']}")

    llf = LLFBaseline()

    def llf_fn(state, env):
        return llf.schedule(_get_raw_state(state), env)

    results["LLF"] = evaluate_on_env(
        llf_fn, args.n_episodes, use_wrapper="none", max_steps=args.max_steps)
    print(f"  LLF: {results['LLF']}")

    # ── 7. 启发式基线 (当前环境) ───────────────────────────────────
    heu = HeuristicBaseline()

    def heu_fn(state, env):
        # 启发式规则只依赖当前观测，不接触训练过的网络参数。
        return heu.schedule(_get_raw_state(state))

    results["启发式"] = evaluate_on_env(
        heu_fn, args.n_episodes, use_wrapper="none", max_steps=args.max_steps)
    print(f"  启发式: {results['启发式']}")

    value_aware_heu = ValueAwareHeuristicBaseline()

    def value_aware_heu_fn(state, env):
        # 强规则基线：读取 High/Mid/Low 队列压力，优先 High，再 Mid，再 Low，并只主动丢动态 Low。
        return value_aware_heu.schedule(_get_raw_state(state))

    results["Value-aware Heuristic"] = evaluate_on_env(
        value_aware_heu_fn, args.n_episodes, use_wrapper="none", max_steps=args.max_steps)
    print(f"  Value-aware Heuristic: {results['Value-aware Heuristic']}")

    # ── 8. 静态规则基线 (当前环境) ────────────────────────────────
    static = StaticRuleBaseline()

    def static_fn(state, env):
        # 静态规则基线使用 当前状态和环境上下文，不依赖旧调度器。
        return static.schedule(_get_raw_state(state), env)

    results["Static Rule"] = evaluate_on_env(
        static_fn, args.n_episodes, use_wrapper="none", max_steps=args.max_steps)
    print(f"  Static Rule: {results['Static Rule']}")

    delivery_check = _paper_table_delivery_check(
        results,
        allow_zero_delivery=bool(getattr(args, "allow_zero_delivery", False)),
    )
    optional_learning_checkpoints = {
        "SAC w/o Safety": getattr(args, "sac_checkpoint", None),
        "SAC-Lagrangian": getattr(args, "sac_lagrangian_checkpoint", None),
        "SAC + PSF": getattr(args, "sac_psf_checkpoint", None),
        "SAC + Lyapunov": getattr(args, "sac_lya_checkpoint", None),
    }
    missing_optional_learning = [
        name for name, path in optional_learning_checkpoints.items()
        if not path or not os.path.exists(path)
    ]

    # ── 保存结果 ──────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    fname = f"results/compare_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    meta = {
        "result_type": "formal_main_comparison",
        "mpc_taxonomy": {
            "MPC": "myopic model-predictive baseline using current observations and local physics forecast",
            "Robust MPC": "myopic MPC with scenario robustness",
            "Omniscient MPC (Oracle)": (
                "non-deployable upper-bound proxy; copies env and rolls out future stochastic/contact trace"
            ),
        },
        "method_roles": {
            OURS_NAME: "main_method",
            "SAC w/o Safety": "independently_trained_learning_baseline",
            "SAC-Lagrangian": "independently_trained_learning_baseline",
            "SAC + PSF": "independently_trained_learning_baseline",
            "SAC + Lyapunov": "independently_trained_learning_baseline",
            "Omniscient MPC (Oracle)": "upper_bound_not_deployable_baseline",
            "Ours + CPU throttle (deployment)": "diagnostic_deployment_variant_not_main_table",
            "Ours w/o Work-Conserving": "diagnostic_config_ablation_not_main_table",
        },
        "paper_table": {
            name: compact_paper_table_row(stats)
            for name, stats in results.items()
        },
        "diagnostic_table": {
            name: compact_paper_table_row(stats)
            for name, stats in diagnostic_results.items()
        },
        "paper_table_valid": bool(
            ours_loaded
            and delivery_check["nonzero_delivery"]
            and delivery_check["main_method_nonzero_delivery"]
        ),
        "required_main_method": OURS_NAME,
        "missing_required_methods": [] if ours_loaded else [OURS_NAME],
        "missing_optional_learning_baselines": missing_optional_learning,
        "allow_missing_ours": bool(getattr(args, "allow_missing_ours", False)),
        "allow_zero_delivery": bool(getattr(args, "allow_zero_delivery", False)),
        "allow_posthoc_learning_baselines": bool(getattr(args, "allow_posthoc_learning_baselines", False)),
        "delivery_validity_check": delivery_check,
        "main_table_methods": list(results.keys()),
        "diagnostic_methods": list(diagnostic_results.keys()),
    }
    with open(fname, "w", encoding="utf-8") as f:
        json.dump({
            "__meta__": meta,
            "comparable_results": results,
            "diagnostic_results": diagnostic_results,
        },
                  f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {fname}")

    # ── 打印对比表 ────────────────────────────────────────────────
    print(f"\n{'方法':<28} {'CSR':>8} {'EpSafe':>8} {'Surv':>8} "
          f"{'VoI':>10} {'Downlink':>10} {'Proc/DL':>8} {'Window':>8} {'Interv':>8}")
    print("-" * 108)
    for name, stats in results.items():
        row = compact_paper_table_row(stats)
        print(f"  {name:<26} "
              f"{row['Constraint Satisfaction Rate']:>8.1%} "
              f"{row['Episode Safety Rate']:>8.1%} "
              f"{row['Survival Rate']:>8.1%} "
              f"{row['Delivered VoI']:>10.1f} "
              f"{row['Downlink MB']:>10.1f} "
              f"{row['Proc/DL Ratio']:>8.2f} "
              f"{row['Window Utilization']:>8.1%} "
              f"{row['Intervention Rate']:>8.1%}")

    # ── 相对提升分析 ─────────────────────────────────────────────
    ref_name = "Static Rule"
    if ref_name in results:
        ref = float(results[ref_name].get("delivered_value_mean", 0.0))
        if abs(ref) <= 1e-6:
            print(f"\n  相对「{ref_name}」的交付价值提升：基线为 0，跳过百分比计算")
        else:
            print(f"\n  相对「{ref_name}」的交付价值提升：")
            for name, stats in results.items():
                if name == ref_name:
                    continue
                cur = float(stats.get("delivered_value_mean", 0.0))
                pct = (cur - ref) / abs(ref) * 100
                print(f"    {name:<28} {pct:+.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VLEO 全方法对比评估")
    parser.add_argument("--checkpoint",
                        default=DEFAULT_OPTIMIZED_CHECKPOINT,
                        help="LS-PSF CMDP 主方法的 checkpoint")
    parser.add_argument("--sac_checkpoint", default=None,
                        help="可选：SAC w/o Safety 独立训练 checkpoint")
    parser.add_argument("--sac_lagrangian_checkpoint", default=None,
                        help="可选：SAC-Lagrangian 独立训练 checkpoint")
    parser.add_argument("--sac_psf_checkpoint", default=None,
                        help="可选：SAC + PSF 独立训练 checkpoint")
    parser.add_argument("--sac_lya_checkpoint", default=None,
                        help="可选：SAC + Lyapunov 独立训练 checkpoint")
    parser.add_argument("--n_episodes", type=int, default=int(TRAIN_CONFIG.get("eval_episodes", 30)))
    parser.add_argument("--mpc_horizon", type=int, default=6)
    parser.add_argument("--robust_mpc_horizon", type=int, default=8,
                        help="Robust MPC 的预测窗口")
    parser.add_argument("--oracle_mpc_horizon", type=int, default=12,
                        help="Omniscient MPC 复制环境 rollout 的预测窗口")
    parser.add_argument("--oracle_mpc_beam_width", type=int, default=8,
                        help="Omniscient MPC beam search 保留宽度")
    parser.add_argument("--dpp_V", type=float, default=8.0,
                        help="DPP 基线的吞吐权重 V")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="可选：每个 episode 仅评估前 max_steps 步（用于快速烟雾检查）")
    parser.add_argument("--skip_config_ablations", action="store_true",
                        help="兼容旧参数：跳过 Ours 的 CPU throttle / work-conserving 诊断")
    parser.add_argument("--include_deployment_ablations", action="store_true",
                        help="附加输出 Ours 的 CPU throttle / work-conserving 诊断；不进入论文主表")
    parser.add_argument("--allow_missing_ours", action="store_true",
                        help="仅用于 smoke/baseline-only 调试：允许主方法 checkpoint 缺失并标记结果不可作为论文主表")
    parser.add_argument("--allow_zero_delivery", action="store_true",
                        help="仅用于 smoke/debug：允许所有方法 delivered_value/downlink 为 0，并标记结果不可作为论文主表")
    parser.add_argument("--allow_posthoc_learning_baselines", action="store_true",
                        help="仅用于诊断：允许学习型 baseline 复用主方法 checkpoint；正式论文对比不要使用")
    args = parser.parse_args()
    run_compare_all(args)

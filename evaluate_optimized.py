"""
训练后模型评估、对比统计和星载可部署性估算入口。
模型评估 & 对比工具

评估口径：
  1. 使用 DilatedFrameStackWrapper，与 LS-PSF CMDP 训练入口保持一致
  2. 评估前清空安全层统计，避免 projection_rate / psf_rate 被历史污染
  3. 均值、方差、分位数都带空数组保护
  4. 报告中区分 processed_mb 与真实对地下传 downlink_mb
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    # 仅在脚本直跑时追加，避免导入期全局污染 sys.path
    sys.path.append(_PROJECT_ROOT)
import numpy as np
import argparse
import json
import time
from datetime import datetime
import torch
import torch.nn as nn
from environment.satellite_env import VLEOSatelliteEnv
from environment.wrappers import DilatedFrameStackWrapper
from scheduler.integrated_scheduler import IntegratedScheduler
from utils.paper_metrics import add_paper_metrics
from config import (
    TRAIN_CONFIG, DRL_CONFIG, REWARD_CONFIG, OBJECTIVE_VERSION,
    ORBITAL_CONFIG, ENERGY_CONFIG, THERMAL_CONFIG,
)

ALTITUDE_SAFE_KM = float(ORBITAL_CONFIG["altitude_min_km"])
BATTERY_SAFE_SOC = float(ENERGY_CONFIG["battery_min_soc"])


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


def _safe_mean(arr):
    """空数组保护，避免 np.mean([]) → NaN"""
    if len(arr) == 0:
        return 0.0
    return float(np.mean(arr))


def _safe_std(arr):
    if len(arr) == 0:
        return 0.0
    return float(np.std(arr))


def _safe_percentile(arr, p):
    if len(arr) == 0:
        return 0.0
    return float(np.percentile(arr, p))


def _available_power_w(env) -> float | None:
    """读取环境的动态可用功率，供动作边界裁剪层使用。"""
    try:
        return float(getattr(env, "available_power_w"))
    except Exception:
        return None


def _count_params(module: nn.Module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return int(total), int(trainable)


def _objective_summary() -> dict:
    # 评估报告里固定论文版 CMDP 口径，避免把约束代价误读成任务 reward。
    return {
        "primary_target": "maximize semantic scene-aware VoI delivered to ground under LS-PSF CMDP safety constraints",
        "primary_reward_term": "w_delivered_value * delivered_value + w_deadline_success * on_time_delivered_value",
        "task_value_model": (
            "orbital phase maps to semantic scenes; scene profile determines arrival "
            "rate, priority, quality/cloud penalty, AoI/VoI decay horizon and base value multiplier"
        ),
        "observation_schema": (
            f"{int(DRL_CONFIG.get('state_dim', 40))}-D state includes grouped High/Mid/Low raw and processed queues, "
            "expiring value histograms, current scene class, upcoming value-pressure intensity, "
            "future contact capacity, thermal margin, next-window range flag and CPU backpressure feedback"
        ),
        "action_schema": (
            f"{int(DRL_CONFIG.get('action_dim', 10))}-D action = physical power allocation "
            "[prop,cpu,tx] + compact CPU/TX value and urgency axes + Low-drop strength"
        ),
        "network_input_preprocessing": "SAC Actor/Critic receive RunningMeanStd-normalized observations; evaluation freezes the statistics",
        "link_capacity_model": "ground-station capacity uses discrete AMC/MCS levels selected by SNR, then applies low-elevation Doppler/path penalty",
        "auxiliary_training_reward_terms": [
            "w_deadline_success * on_time_delivered_value",
        ],
        "reward_weights": {
            "w_delivered_value": float(REWARD_CONFIG.get("w_delivered_value", 0.0)),
            "w_deadline_success": float(REWARD_CONFIG.get("w_deadline_success", 0.0)),
        },
        "risk_boundaries": {
            "normal": "h>=180km and SOC>=15%",
            "warning": "150km<=h<180km or 5%<SOC<15%",
            "unsafe": "122km<h<150km",
            "failure": "h<=122km or SOC<=5%",
            "altitude_warning_km": float(ORBITAL_CONFIG.get("altitude_warning_km", 180.0)),
            "altitude_min_km": float(ORBITAL_CONFIG.get("altitude_min_km", 150.0)),
            "altitude_crash_km": float(ORBITAL_CONFIG.get("altitude_crash_km", 122.0)),
            "battery_min_soc": float(ENERGY_CONFIG.get("battery_min_soc", 0.15)),
            "battery_crash_soc": float(ENERGY_CONFIG.get("battery_crash_soc", 0.05)),
            "thermal_warning_temp_c": float(THERMAL_CONFIG.get("warning_temp_c", 45.0)),
            "thermal_max_temp_c": float(THERMAL_CONFIG.get("max_temp_c", 55.0)),
        },
        "metric_semantics": {
            "processed_mean_mb": "actual onboard processed data volume",
            "downlink_mean_mb": "actual delivered-to-ground data volume",
            "delivered_value_mean": "deadline-aware task value received by the ground",
            "scene_class": "semantic class derived from orbital phase, e.g. cloud/ocean/routine/urban/disaster/military",
            "upcoming_task_intensity": "lookahead value-pressure from scene arrival, value multiplier and deadline urgency",
            "overall_safe_rate": "step-wise satisfaction of orbit, energy, raw-queue and processed-queue constraints",
            "normal_state_rate": "fraction of steps with h>=180km and SOC>=15%",
            "warning_state_rate": "fraction of steps in 150<=h<180km or 5%<SOC<15%",
            "unsafe_state_rate": "fraction of steps in 122<h<150km",
            "failure_state_rate": "fraction of steps after h<=122km or SOC<=5% terminal condition",
            "safety_rate": "episode-wise safety rate; an episode is unsafe if any safety constraint is violated",
            "average_aoi_steps": "average Age of Information for delivered data",
            "voi_degradation_rate": "Value-of-Information degradation proxy from expired semantic value",
            "thermal_safe_rate": "fraction of steps below the configured thermal safety limit",
        },
    }


def _resolve_objective_summary(metadata: dict) -> dict:
    # 优先读取 checkpoint 自带的目标说明；如果 checkpoint 训练目标和当前代码不一致，报告里必须显式标红。
    current_version = OBJECTIVE_VERSION
    if metadata.get("objective_summary"):
        summary = dict(metadata["objective_summary"])
        summary["source"] = "checkpoint_metadata"
        if metadata.get("reward_weights"):
            summary["reward_weights"] = metadata["reward_weights"]
        if metadata.get("objective_version"):
            summary["objective_version"] = metadata["objective_version"]
        if summary.get("objective_version") != current_version:
            summary["current_code_objective_version"] = current_version
            summary["warning"] = (
                "checkpoint objective metadata does not match the current timely-value "
                "delivery objective; retrain before using this checkpoint as a paper result"
            )
        return summary

    summary = _objective_summary()
    summary["source"] = "current_code_default"
    summary["objective_version"] = current_version
    summary["warning"] = (
        "checkpoint lacks objective metadata; if it was trained before the "
        "reward refactor, it likely used an older reward that may not match the current "
        "timely-value delivery setup"
    )
    return summary


def _estimate_actor_flops(actor: nn.Module, example_state: np.ndarray) -> int:
    """
    轻量 FLOPs 估算（前向单次）：
      1) 统计所有 Linear 层乘加
      2) 额外补充 Multi-Head Attention 的 QK^T 与 AV 两次矩阵乘
    说明：这是部署侧常用的近似估算，足够用于模型对比与量级判断。
    """
    if example_state.ndim == 2:
        state = torch.from_numpy(example_state).float().unsqueeze(0)
    elif example_state.ndim == 3:
        state = torch.from_numpy(example_state).float()
    else:
        raise ValueError(f"Unexpected state shape for FLOPs estimate: {example_state.shape}")

    device = next(actor.parameters()).device
    state = state.to(device)

    flops = {"total": 0}
    hooks = []

    def linear_hook(module, inputs, outputs):
        if len(inputs) == 0 or not torch.is_tensor(inputs[0]):
            return
        x = inputs[0]
        if module.in_features <= 0:
            return
        n = x.numel() // module.in_features
        flops["total"] += int(2 * n * module.in_features * module.out_features)

    def mha_hook(module, inputs, outputs):
        if len(inputs) == 0 or not torch.is_tensor(inputs[0]):
            return
        q = inputs[0]
        if q.dim() != 3:
            return
        if module.batch_first:
            bsz, seq_len, embed_dim = q.shape
        else:
            seq_len, bsz, embed_dim = q.shape
        n_heads = max(1, int(module.num_heads))
        head_dim = max(1, embed_dim // n_heads)
        # 两次矩阵乘：QK^T 和 Attention*V
        flops["total"] += int(4 * bsz * n_heads * seq_len * seq_len * head_dim)

    for m in actor.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.MultiheadAttention):
            hooks.append(m.register_forward_hook(mha_hook))

    was_training = actor.training
    actor.eval()
    with torch.no_grad():
        actor(state)
    if was_training:
        actor.train()

    for h in hooks:
        h.remove()

    return int(flops["total"])


def benchmark_onboard_feasibility(checkpoint_path: str,
                                  n_calls: int = 300,
                                  warmup: int = 50) -> dict:
    """星载可部署性基准：CPU 单次调度延迟 + 参数量 + FLOPs"""
    env = DilatedFrameStackWrapper(
        VLEOSatelliteEnv(seed=123),
        k=DRL_CONFIG.get("frame_stack", 8))
    scheduler = IntegratedScheduler(
        device="cpu", enable_lyapunov=True, use_psf=True)
    scheduler.load(checkpoint_path)
    scheduler.reset_all_safety_stats()

    # 本基准只评估“单次调度决策”的时间与模型规模，用来说明板载实时性，不参与模型优劣结论。
    state = env.reset()
    ref_state = np.array(state, copy=True)

    def run_one_step(current_state):
        # 基准测试也要完整经过调度器与安全层，才能反映真实部署时的单次决策开销。
        in_window = (env._contact.get("in_window", False)
                     if env._contact is not None else False)
        action, _, _, _ = scheduler.schedule(
            current_state,
            env.energy_queue.value,
            env.orbit_queue.value,
            env.data_queue.length,
            env.comm_queue.value,
            in_window,
            evaluate=True,
            h=env.altitude_m,
            soc=env.battery.soc,
            time_s=env.time_s,
            orbital_phase=env.orbit_sim.phase,
            tx_capacity_mbps=float((env._contact or {}).get("max_capacity_mbps", 0.0)),
            available_power_w=_available_power_w(env),
            env=env)
        next_state, _, done, _ = env.step(
            action, enforce_prop_smoothing=False)
        # done 后的 reset 不属于单次 onboard 推理/执行开销，不能混入延迟统计。
        return next_state, bool(done)

    # 预热：避免首次调用初始化开销干扰统计
    # 先做预热，避免首次调用的额外初始化开销干扰正式延迟统计。
    for _ in range(max(0, warmup)):
        state, done = run_one_step(state)
        if done:
            state = env.reset()

    # 正式计时阶段：完整走一遍“状态 -> 调度器 -> 安全层 -> 动作”链路。
    latencies_ms = []
    for _ in range(max(1, n_calls)):
        t0 = time.perf_counter()
        state, done = run_one_step(state)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)
        if done:
            state = env.reset()

    actor = scheduler.agent.actor
    critic = scheduler.agent.critic
    actor_params_total, actor_params_train = _count_params(actor)
    critic_params_total, critic_params_train = _count_params(critic)
    actor_flops = _estimate_actor_flops(actor, ref_state)

    total_params = actor_params_total + critic_params_total
    return {
        "benchmark_device": "cpu",
        "benchmark_calls": int(max(1, n_calls)),
        "benchmark_warmup": int(max(0, warmup)),
        "latency_ms_mean": _safe_mean(latencies_ms),
        "latency_ms_std": _safe_std(latencies_ms),
        "latency_ms_p50": _safe_percentile(latencies_ms, 50),
        "latency_ms_p95": _safe_percentile(latencies_ms, 95),
        "latency_ms_p99": _safe_percentile(latencies_ms, 99),
        "latency_ms_min": float(np.min(latencies_ms)) if latencies_ms else 0.0,
        "latency_ms_max": float(np.max(latencies_ms)) if latencies_ms else 0.0,
        "actor_params_total": actor_params_total,
        "actor_params_trainable": actor_params_train,
        "critic_params_total": critic_params_total,
        "critic_params_trainable": critic_params_train,
        "model_params_total": int(total_params),
        "model_size_mb_fp32": float(total_params * 4 / (1024 ** 2)),
        "model_size_mb_fp16": float(total_params * 2 / (1024 ** 2)),
        "actor_flops_estimate": int(actor_flops),
    }


def print_onboard_benchmark(name: str, stats: dict):
    print(f"\n[星载可部署性分析 - {name}]")
    print("-" * 60)
    print(f"  设备: CPU")
    print(f"  调度时延(ms): mean={stats['latency_ms_mean']:.4f}, "
          f"p50={stats['latency_ms_p50']:.4f}, p95={stats['latency_ms_p95']:.4f}, "
          f"p99={stats['latency_ms_p99']:.4f}")
    print(f"  参数量: actor={stats['actor_params_total']:,}, "
          f"critic={stats['critic_params_total']:,}, total={stats['model_params_total']:,}")
    print(f"  模型体积: fp32={stats['model_size_mb_fp32']:.2f}MB, "
          f"fp16={stats['model_size_mb_fp16']:.2f}MB")
    print(f"  Actor FLOPs(估算/前向单次): {stats['actor_flops_estimate']:,}")


def evaluate_model(checkpoint_path: str, n_episodes: int = None,
                   device: str = "cuda",
                   force_enable_lyapunov: bool = None,
                   force_use_psf: bool = None,
                   force_use_inference_mpc: bool = None,
                   eval_seed: int = None) -> dict:
    # 独立评估要尽量和训练内评估同口径：同样的状态堆叠、同样的安全层配置、同样的推进器更新约束。
    n_episodes = int(TRAIN_CONFIG.get("eval_episodes", 30) if n_episodes is None else n_episodes)
    device = _resolve_device(device)
    eval_seed = int(TRAIN_CONFIG.get("seed", 42) if eval_seed is None else eval_seed)
    # 使用 DilatedFrameStackWrapper，与训练阶段保持同一观测链路。
    env = DilatedFrameStackWrapper(
        VLEOSatelliteEnv(seed=eval_seed),
        k=DRL_CONFIG.get("frame_stack", 8))
    use_mpc_init = bool(force_use_inference_mpc) if force_use_inference_mpc is not None else False
    scheduler = IntegratedScheduler(
        device=device, enable_lyapunov=True, use_psf=True,
        use_inference_mpc=use_mpc_init)
    metadata = scheduler.load(checkpoint_path)
    if force_enable_lyapunov is not None:
        # 命令行显式关闭/开启时，优先级高于 checkpoint 里保存的训练配置。
        scheduler.enable_lyapunov = bool(force_enable_lyapunov)
        metadata["enable_lyapunov_forced"] = bool(force_enable_lyapunov)
    if force_use_psf is not None:
        scheduler.use_psf = bool(force_use_psf)
        scheduler.psf = None
        metadata["use_psf_forced"] = bool(force_use_psf)
    if force_use_inference_mpc is not None:
        # 评估时显式打开/关闭 MPC，绕过 checkpoint metadata。
        scheduler.use_inference_mpc = bool(force_use_inference_mpc)
        if scheduler.use_inference_mpc and scheduler.inference_mpc is None:
            from drl.inference_mpc import InferenceMPCPlanner
            scheduler.inference_mpc = InferenceMPCPlanner(
                predictor=scheduler.safety_predictor)
        elif not scheduler.use_inference_mpc:
            scheduler.inference_mpc = None
        metadata["use_inference_mpc_forced"] = bool(force_use_inference_mpc)

    # 评估前重置统计
    # 每次单独评估前都清空安全层统计，避免多次评估之间互相污染投影率/拦截率。
    scheduler.reset_all_safety_stats()

    rewards, reward_per_steps, throughputs, tx_mbs_list, delivered_values = [], [], [], [], []
    safety_rates, survival_rates, energy_ratios, episode_lengths = [], [], [], []
    orbit_safe_rates, energy_safe_rates, thermal_safe_rates = [], [], []
    raw_queue_safe_rates, processed_queue_safe_rates, overall_safe_rates = [], [], []
    stage_rate_sums = {"normal": [], "warning": [], "unsafe": [], "failure": []}
    deadline_rates, expired_rates, drop_rates, delay_steps = [], [], [], []
    value_weighted_deadline_rates, value_weighted_aoi_steps, voi_loss_rates = [], [], []
    raw_overflow_mbs, processed_overflow_mbs = [], []
    constraint_violations = {"energy": 0, "orbit": 0, "thermal": 0, "raw_queue": 0, "processed_queue": 0, "total": 0}
    prop_powers, cpu_powers, tx_powers, energy_whs, window_utils = [], [], [], [], []
    processed_final_utils, tx_active_contact_flags = [], []
    processed_queue_utils, processed_future_contact_ratios = [], []
    future_cpu_gate_applied_flags = []
    cpu_gate_ratio_before_values, cpu_gate_ratio_after_values = [], []
    cpu_gate_alpha_before_values, cpu_gate_alpha_after_values = [], []
    cpu_gate_requested_values, cpu_gate_allowed_values = [], []
    cpu_gate_mod_l2_values = []
    episode_proc_dl_ratios, episode_energy_per_value, useful_processing_ratios = [], [], []
    processed_value_totals, processed_voi_basis_value_totals = [], []
    projected_flags = []
    action_mods = []
    delivered_high_values, delivered_mid_values, delivered_low_values = [], [], []
    expired_high_values, dropped_high_values, active_dropped_high_values = [], [], []
    cpu_req_high, cpu_req_mid, cpu_req_low = [], [], []
    tx_req_high, tx_req_mid, tx_req_low = [], [], []
    energy_per_value_steps = []

    for ep in range(n_episodes):
        # 每个 episode 单独累计 reward、处理量、下传量和违规次数，最后再统一做均值统计。
        state = env.reset()
        ep_r = ep_tput = ep_tx = ep_value = ep_solar = ep_steps = 0.0
        ep_energy_wh = 0.0
        ep_processed_value = 0.0
        ep_processed_voi_basis_value = 0.0
        ep_raw_overflow = ep_processed_overflow = 0.0
        ep_final_processed_util = 0.0
        safe_counts = {"orbit": 0, "energy": 0, "thermal": 0, "raw_queue": 0, "processed_queue": 0, "overall": 0}
        stage_counts = {"normal": 0, "warning": 0, "unsafe": 0, "failure": 0}
        violations = {"energy": 0, "orbit": 0, "thermal": 0, "raw_queue": 0, "processed_queue": 0}
        survived = True
        done = False

        while not done:
            # 这些上下文会影响动作后处理是否允许改推进器、是否处于通信窗口，因此评估时也要真实传入。
            in_window = (env._contact.get("in_window", False)
                         if env._contact is not None else False)
            prop_can_update = True
            if hasattr(env, "step_count") and hasattr(env, "N_PROP_SMOOTH"):
                prop_can_update = (env.step_count % env.N_PROP_SMOOTH == 0)
            action, was_projected, _, psf_meta = scheduler.schedule(
                state, env.energy_queue.value, env.orbit_queue.value,
                env.data_queue.length, env.comm_queue.value,
                in_window, evaluate=True,
                h=env.altitude_m, soc=env.battery.soc,
                time_s=env.time_s,
                prop_can_update=prop_can_update,
                orbital_phase=env.orbit_sim.phase,
                tx_capacity_mbps=float((env._contact or {}).get("max_capacity_mbps", 0.0)),
                available_power_w=_available_power_w(env),
                env=env)
            state, reward, done, info = env.step(
                action, enforce_prop_smoothing=False)

            # reward 是训练代理信号；论文正式主指标另外用 actual_tx_mb 单独累计。
            ep_r += reward
            ep_tput += info.get(
                "processed_mb",
                info.get("service_rate_mbs", 0) * TRAIN_CONFIG["time_slot_s"],
            )
            ep_tx += info.get("delivered_mb", info.get("actual_tx_mb", 0))
            ep_value += info.get("delivered_value", 0.0)
            ep_processed_value += float(info.get("processed_value", 0.0))
            ep_processed_voi_basis_value += float(
                info.get("processed_voi_basis_value", info.get("processed_value", 0.0))
            )
            ep_raw_overflow += float(info.get("raw_queue_overflow_mb", info.get("overflow_mb", 0.0)))
            ep_processed_overflow += float(info.get("processed_queue_overflow_mb", info.get("comm_overflow_mb", 0.0)))
            ep_final_processed_util = float(info.get("processed_queue_utilization", 0.0))
            processed_queue_utils.append(ep_final_processed_util)
            processed_future_contact_ratios.append(float(
                info.get("processed_queue_future_contact_ratio_raw", info.get("processed_queue_future_contact_ratio", 0.0))))
            future_cpu_gate_applied_flags.append(float(
                1.0 if info.get("future_contact_cpu_gate_applied", False) else 0.0))
            cpu_gate_ratio_before_values.append(float(info.get("cpu_gate_ratio_before", 0.0)))
            cpu_gate_ratio_after_values.append(float(info.get("cpu_gate_ratio_after_est", 0.0)))
            cpu_gate_alpha_before_values.append(float(info.get("cpu_gate_alpha_cpu_before", 0.0)))
            cpu_gate_alpha_after_values.append(float(info.get("cpu_gate_alpha_cpu_after", 0.0)))
            cpu_gate_requested_values.append(float(info.get("cpu_gate_requested_processed_mb", 0.0)))
            cpu_gate_allowed_values.append(float(info.get("cpu_gate_allowed_processed_mb", 0.0)))
            cpu_gate_mod_l2_values.append(float(info.get("cpu_gate_mod_l2", 0.0)))
            ep_solar += info.get("P_solar_w", 0)
            ep_steps += 1
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
            projected_flags.append(float(was_projected))
            action_mods.append(float(psf_meta.get("total_modification_l2", 0.0)))
            delivered_high_values.append(float(info.get("delivered_high_value", 0.0)))
            delivered_mid_values.append(float(info.get("delivered_mid_value", 0.0)))
            delivered_low_values.append(float(info.get("delivered_low_value", 0.0)))
            expired_high_values.append(float(info.get("expired_high_value", 0.0)))
            dropped_high_values.append(float(info.get("dropped_high_value", 0.0)))
            active_dropped_high_values.append(
                float(info.get("active_dropped_raw_high_value", 0.0))
                + float(info.get("active_dropped_processed_high_value", 0.0))
            )
            cpu_req_high.append(float(info.get("cpu_requested_high", 0.0)))
            cpu_req_mid.append(float(info.get("cpu_requested_mid", 0.0)))
            cpu_req_low.append(float(info.get("cpu_requested_low", 0.0)))
            tx_req_high.append(float(info.get("tx_requested_high", 0.0)))
            tx_req_mid.append(float(info.get("tx_requested_mid", 0.0)))
            tx_req_low.append(float(info.get("tx_requested_low", 0.0)))
            energy_per_value_steps.append(
                float(info.get("P_total_w", 0.0)) * TRAIN_CONFIG["time_slot_s"] / 3600.0
                / max(float(info.get("delivered_value", 0.0)), 1e-9)
            )

            orbit_safe = bool(info.get("orbit_safe", info.get("altitude_km", 300) >= ALTITUDE_SAFE_KM))
            energy_safe = bool(info.get("energy_safe", info.get("soc", 1.0) >= BATTERY_SAFE_SOC))
            thermal_safe = bool(info.get("thermal_safe", True))
            raw_safe = bool(info.get("raw_queue_safe", info.get("raw_queue_overflow_mb", 0.0) <= 1e-9))
            proc_safe = bool(info.get("processed_queue_safe", info.get("processed_queue_overflow_mb", 0.0) <= 1e-9))
            overall_safe = bool(info.get("overall_safe", orbit_safe and energy_safe and thermal_safe and raw_safe and proc_safe))
            safe_counts["orbit"] += int(orbit_safe)
            safe_counts["energy"] += int(energy_safe)
            safe_counts["thermal"] += int(thermal_safe)
            safe_counts["raw_queue"] += int(raw_safe)
            safe_counts["processed_queue"] += int(proc_safe)
            safe_counts["overall"] += int(overall_safe)
            if not energy_safe:
                violations["energy"] += 1
            if not orbit_safe:
                violations["orbit"] += 1
            if not thermal_safe:
                violations["thermal"] += 1
            if not raw_safe:
                violations["raw_queue"] += 1
            if not proc_safe:
                violations["processed_queue"] += 1
            if info.get("terminated", False):
                survived = False
            stage = str(info.get("risk_stage", "normal"))
            if stage not in stage_counts:
                stage = "failure" if bool(info.get("crashed", False)) else "normal"
            stage_counts[stage] += 1

        rewards.append(ep_r)
        reward_per_steps.append(float(ep_r / max(ep_steps, 1)))
        throughputs.append(ep_tput)
        tx_mbs_list.append(ep_tx)
        delivered_values.append(ep_value)
        processed_value_totals.append(ep_processed_value)
        processed_voi_basis_value_totals.append(ep_processed_voi_basis_value)
        useful_processing_ratios.append(float(
            ep_value / max(ep_processed_voi_basis_value, 1e-9)
            if ep_processed_voi_basis_value > 1e-9 else 0.0
        ))
        episode_proc_dl_ratios.append(float(ep_tput / max(ep_tx, 1e-9)))
        episode_energy_per_value.append(float(ep_energy_wh / max(ep_value, 1e-9)))
        processed_final_utils.append(float(ep_final_processed_util))
        raw_overflow_mbs.append(ep_raw_overflow)
        processed_overflow_mbs.append(ep_processed_overflow)
        is_ep_safe = all(v == 0 for v in violations.values())
        safety_rates.append(float(is_ep_safe))
        survival_rates.append(float(survived))
        episode_lengths.append(int(ep_steps))
        orbit_safe_rates.append(safe_counts["orbit"] / max(ep_steps, 1))
        energy_safe_rates.append(safe_counts["energy"] / max(ep_steps, 1))
        thermal_safe_rates.append(safe_counts["thermal"] / max(ep_steps, 1))
        raw_queue_safe_rates.append(safe_counts["raw_queue"] / max(ep_steps, 1))
        processed_queue_safe_rates.append(safe_counts["processed_queue"] / max(ep_steps, 1))
        overall_safe_rates.append(safe_counts["overall"] / max(ep_steps, 1))
        for stage_name, values in stage_rate_sums.items():
            values.append(stage_counts[stage_name] / max(ep_steps, 1))
        constraint_violations["energy"] += violations["energy"]
        constraint_violations["orbit"] += violations["orbit"]
        constraint_violations["thermal"] += violations["thermal"]
        constraint_violations["raw_queue"] += violations["raw_queue"]
        constraint_violations["processed_queue"] += violations["processed_queue"]
        if not is_ep_safe:
            constraint_violations["total"] += 1
        energy_ratios.append(ep_solar / (ep_steps + 1e-6))
        task_summary = getattr(env, "task_tracker", None).summary() if hasattr(env, "task_tracker") else {}
        deadline_rates.append(float(task_summary.get("deadline_success_rate", 0.0)))
        value_weighted_deadline_rates.append(float(task_summary.get(
            "value_weighted_deadline_success_rate",
            task_summary.get("deadline_success_rate", 0.0),
        )))
        expired_rates.append(float(task_summary.get("expired_value_rate", 0.0)))
        drop_rates.append(float(task_summary.get("dropped_value_rate", 0.0)))
        delay_steps.append(float(task_summary.get("avg_delivery_delay_steps", 0.0)))
        value_weighted_aoi_steps.append(float(task_summary.get(
            "value_weighted_aoi_steps",
            task_summary.get("average_aoi_steps", task_summary.get("avg_delivery_delay_steps", 0.0)),
        )))
        voi_loss_rates.append(float(task_summary.get("voi_loss_rate", 0.0)))

    safety_stats = scheduler.get_safety_stats()
    final_proj_rate = float(safety_stats.get("lyapunov_proj_rate", 0.0))

    mean_r = _safe_mean(rewards)
    stats = {
        "timestamp": datetime.now().isoformat(),
        "n_episodes": n_episodes,
        "eval_seed": int(eval_seed),
        "device": device,
        "checkpoint_metadata": metadata,
        "objective_summary": _resolve_objective_summary(metadata),
        "reward_mean": mean_r,
        "reward_std": _safe_std(rewards),
        "reward_per_step_mean": _safe_mean(reward_per_steps),
        "reward_per_step_std": _safe_std(reward_per_steps),
        "reward_max": float(np.max(rewards)) if rewards else 0.0,
        "reward_min": float(np.min(rewards)) if rewards else 0.0,
        "processed_mean_mb": _safe_mean(throughputs),
        "processed_std_mb": _safe_std(throughputs),
        "downlink_mean_mb": _safe_mean(tx_mbs_list),
        "downlink_std_mb": _safe_std(tx_mbs_list),
        "processed_value_mean": _safe_mean(processed_value_totals),
        "processed_voi_basis_value_mean": _safe_mean(processed_voi_basis_value_totals),
        "delivered_value_mean": _safe_mean(delivered_values),
        "delivered_value_std": _safe_std(delivered_values),
        "delivered_high_value_mean": _safe_mean(delivered_high_values),
        "delivered_mid_value_mean": _safe_mean(delivered_mid_values),
        "delivered_low_value_mean": _safe_mean(delivered_low_values),
        "expired_high_value_mean": _safe_mean(expired_high_values),
        "dropped_high_value_mean": _safe_mean(dropped_high_values),
        "active_dropped_high_value_mean": _safe_mean(active_dropped_high_values),
        "cpu_requested_high_mean": _safe_mean(cpu_req_high),
        "cpu_requested_mid_mean": _safe_mean(cpu_req_mid),
        "cpu_requested_low_mean": _safe_mean(cpu_req_low),
        "tx_requested_high_mean": _safe_mean(tx_req_high),
        "tx_requested_mid_mean": _safe_mean(tx_req_mid),
        "tx_requested_low_mean": _safe_mean(tx_req_low),
        "deadline_success_rate": _safe_mean(deadline_rates),
        "value_weighted_deadline_success_rate": _safe_mean(value_weighted_deadline_rates),
        "expired_value_rate": _safe_mean(expired_rates),
        "voi_degradation_rate": _safe_mean(expired_rates),
        "average_aoi_steps": _safe_mean(delay_steps),
        "value_weighted_aoi_steps": _safe_mean(value_weighted_aoi_steps),
        "dropped_value_rate": _safe_mean(drop_rates),
        "voi_loss_rate": _safe_mean(voi_loss_rates),
        "avg_delivery_delay_steps": _safe_mean(delay_steps),
        "value_per_mb": float(_safe_mean(delivered_values) / max(_safe_mean(tx_mbs_list), 1e-9)),
        "mean_prop_power": _safe_mean(prop_powers),
        "mean_cpu_power": _safe_mean(cpu_powers),
        "mean_tx_power": _safe_mean(tx_powers),
        "energy_efficiency": float(np.sum(delivered_values) / max(np.sum(energy_whs), 1e-9)),
        "energy_per_value": _safe_mean(episode_energy_per_value),
        "energy_per_delivered_value_episode": _safe_mean(episode_energy_per_value),
        "energy_per_value_step_mean": _safe_mean(energy_per_value_steps),
        "comm_window_utilization": _safe_mean(window_utils),
        "processed_queue_final_utilization": _safe_mean(processed_final_utils),
        "processed_queue_peak_utilization": float(np.max(processed_queue_utils)) if processed_queue_utils else 0.0,
        "processed_queue_p95_utilization": _safe_percentile(processed_queue_utils, 95),
        "processed_queue_future_contact_ratio": _safe_mean(processed_future_contact_ratios),
        "processed_queue_future_contact_ratio_p95": _safe_percentile(processed_future_contact_ratios, 95),
        "processed_queue_future_contact_ratio_peak": float(np.max(processed_future_contact_ratios)) if processed_future_contact_ratios else 0.0,
        "useful_processing_ratio": _safe_mean(useful_processing_ratios),
        "episode_useful_processing_ratio": _safe_mean(useful_processing_ratios),
        "future_contact_cpu_gate_applied_rate": _safe_mean(future_cpu_gate_applied_flags),
        "cpu_gate_ratio_before_mean": _safe_mean(cpu_gate_ratio_before_values),
        "cpu_gate_ratio_after_est_mean": _safe_mean(cpu_gate_ratio_after_values),
        "cpu_gate_alpha_cpu_before_mean": _safe_mean(cpu_gate_alpha_before_values),
        "cpu_gate_alpha_cpu_after_mean": _safe_mean(cpu_gate_alpha_after_values),
        "cpu_gate_requested_processed_mb_mean": _safe_mean(cpu_gate_requested_values),
        "cpu_gate_allowed_processed_mb_mean": _safe_mean(cpu_gate_allowed_values),
        "cpu_gate_mod_l2_mean": _safe_mean(cpu_gate_mod_l2_values),
        "tx_active_in_contact_ratio": _safe_mean(tx_active_contact_flags),
        "raw_overflow_mean": _safe_mean(raw_overflow_mbs),
        "processed_overflow_mean": _safe_mean(processed_overflow_mbs),
        "global_proc_downlink_ratio": float(np.sum(throughputs) / max(np.sum(tx_mbs_list), 1e-9)),
        "mean_episode_proc_downlink_ratio": _safe_mean(episode_proc_dl_ratios),
        "proc_downlink_ratio": float(np.sum(throughputs) / max(np.sum(tx_mbs_list), 1e-9)),
        "episode_proc_dl_ratio": _safe_mean(episode_proc_dl_ratios),
        "throughput_mean_mb": _safe_mean(throughputs),
        "throughput_std_mb": _safe_std(throughputs),
        "tx_mean_mb": _safe_mean(tx_mbs_list),
        "tx_std_mb": _safe_std(tx_mbs_list),
        "survival_rate": _safe_mean(survival_rates),
        "crash_count": int(sum(1 for x in survival_rates if x <= 0.0)),
        "operational_safety_rate": float(min(
            _safe_mean(orbit_safe_rates),
            _safe_mean(energy_safe_rates),
            _safe_mean(thermal_safe_rates),
        )),
        "orbit_safe_rate": _safe_mean(orbit_safe_rates),
        "energy_safe_rate": _safe_mean(energy_safe_rates),
        "energy_violation_rate": float(max(0.0, 1.0 - _safe_mean(energy_safe_rates))),
        "energy_unsafe_rate": float(max(0.0, 1.0 - _safe_mean(energy_safe_rates))),
        "thermal_safe_rate": _safe_mean(thermal_safe_rates),
        "high_value_delivery_rate": float(
            np.sum(delivered_high_values)
            / max(
                np.sum(delivered_high_values)
                + np.sum(expired_high_values)
                + np.sum(dropped_high_values),
                1e-9,
            )
        ),
        "high_value_delivery_ratio": float(
            np.sum(delivered_high_values)
            / max(
                np.sum(delivered_high_values)
                + np.sum(expired_high_values)
                + np.sum(dropped_high_values),
                1e-9,
            )
        ),
        "raw_queue_safe_rate": _safe_mean(raw_queue_safe_rates),
        "processed_queue_safe_rate": _safe_mean(processed_queue_safe_rates),
        "overall_safe_rate": _safe_mean(overall_safe_rates),
        "step_safety_rate": _safe_mean(overall_safe_rates),
        "normal_state_rate": _safe_mean(stage_rate_sums["normal"]),
        "warning_state_rate": _safe_mean(stage_rate_sums["warning"]),
        "unsafe_state_rate": _safe_mean(stage_rate_sums["unsafe"]),
        "failure_state_rate": _safe_mean(stage_rate_sums["failure"]),
        "episode_safety_rate": _safe_mean(safety_rates),
        "safety_rate": _safe_mean(safety_rates),
        "constraint_energy_violations": int(constraint_violations["energy"]),
        "constraint_energy_violation_rate": float(
            constraint_violations["energy"] / max(sum(len_ for len_ in episode_lengths), 1)
        ),
        "constraint_orbit_violations": int(constraint_violations["orbit"]),
        "constraint_thermal_violations": int(constraint_violations["thermal"]),
        "constraint_raw_queue_violations": int(constraint_violations["raw_queue"]),
        "constraint_processed_queue_violations": int(constraint_violations["processed_queue"]),
        "constraint_total_violations": int(constraint_violations["total"]),
        "violation_percentage": float(
            constraint_violations["total"] / max(n_episodes, 1) * 100),
        "avg_renewable_power_w": _safe_mean(energy_ratios),
        "lyapunov_projection_rate": float(final_proj_rate),
        "intervention_rate": _safe_mean(projected_flags),
        "action_mod_l2_mean": _safe_mean(action_mods),
        "mean_action_modification": _safe_mean(action_mods),
        **{k: float(v) if isinstance(v, (int, float)) else v
           for k, v in safety_stats.items()},
        "coefficient_of_variation_reward": float(
            _safe_std(rewards) / (abs(mean_r) + 1e-6)),
        "coefficient_of_variation_processed": float(
            _safe_std(throughputs) / (abs(_safe_mean(throughputs)) + 1e-6)),
        "coefficient_of_variation_throughput": float(
            _safe_std(throughputs) / (abs(_safe_mean(throughputs)) + 1e-6)),
        "coefficient_of_variation_downlink": float(
            _safe_std(tx_mbs_list) / (abs(_safe_mean(tx_mbs_list)) + 1e-6)),
    }
    return add_paper_metrics(stats)


def compare_models(model1_path: str, model2_path: str = None,
                   n_episodes: int = None, device: str = "cuda",
                   force_enable_lyapunov: bool = None,
                   force_use_psf: bool = None,
                   force_use_inference_mpc: bool = None,
                   eval_seed: int = None):
    n_episodes = int(TRAIN_CONFIG.get("eval_episodes", 30) if n_episodes is None else n_episodes)
    print(f"\n{'='*70}\n  模型评估 ({n_episodes} episodes)\n{'='*70}")
    print(f"\n  模型 1: {model1_path}")
    # 命令行里 --model 是待评估方法，--baseline 才是对照方法；
    # 后续改善比例都按“模型相对 baseline”计算，避免论文报告把提升方向写反。
    stats1 = evaluate_model(
        model1_path, n_episodes, device,
        force_enable_lyapunov=force_enable_lyapunov,
        force_use_psf=force_use_psf,
        force_use_inference_mpc=force_use_inference_mpc,
        eval_seed=eval_seed)

    if model2_path:
        print(f"  模型 2: {model2_path}")
        stats2 = evaluate_model(
            model2_path, n_episodes, device,
            force_enable_lyapunov=force_enable_lyapunov,
            force_use_psf=force_use_psf,
            force_use_inference_mpc=force_use_inference_mpc,
            eval_seed=eval_seed)
        print(f"\n{'指标':<30} {'模型1':>15} {'模型2':>15} {'提升':>10}")
        print("-" * 70)
        for name, key in [("单步奖励", "reward_per_step_mean"),
                          ("平均奖励", "reward_mean"),
                          ("交付价值", "delivered_value_mean"),
                          ("处理量(MB)", "processed_mean_mb"),
                          ("有效回传(MB)", "downlink_mean_mb"),
                          ("处理/下传比", "proc_downlink_ratio"),
                          ("processed溢出(MB)", "processed_overflow_mean"),
                          ("综合安全率", "overall_safe_rate"),
                          ("正常状态率", "normal_state_rate"),
                          ("警告状态率", "warning_state_rate"),
                          ("不安全状态率", "unsafe_state_rate"),
                          ("生存率", "survival_rate"),
                          ("坠毁次数", "crash_count"),
                          ("AoI均值(steps)", "average_aoi_steps"),
                          ("VoI退化率", "voi_degradation_rate"),
                          ("热安全率", "thermal_safe_rate"),
                          ("违规%", "violation_percentage"),
                          ("Lyapunov投影率", "lyapunov_projection_rate")]:
            model_value = stats1.get(key, 0)
            baseline_value = stats2.get(key, 0)
            imp = _relative_improvement(
                model_value,
                baseline_value,
                lower_is_better=(key in {"violation_percentage", "crash_count"}),
            )
            print(f"  {name:<28} {model_value:>15.4f} {baseline_value:>15.4f} {imp:>+10.2f}%")
        return stats1, stats2

    print(f"\n{'指标':<35} {'值':>15}")
    print("-" * 50)
    for k, v in stats1.items():
        if isinstance(v, float):
            print(f"  {k:<33} {v:>15.4f}")
    return stats1, None


def _relative_improvement(model_value: float,
                          baseline_value: float,
                          lower_is_better: bool = False) -> float:
    """计算模型相对 baseline 的提升百分比；违规率这类指标越低越好。"""
    denom = max(abs(float(baseline_value)), 1e-6)
    if lower_is_better:
        return (float(baseline_value) - float(model_value)) / denom * 100.0
    return (float(model_value) - float(baseline_value)) / denom * 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--eval_episodes", type=int, default=int(TRAIN_CONFIG.get("eval_episodes", 30)))
    parser.add_argument("--eval_seed", type=int, default=int(TRAIN_CONFIG.get("seed", 42)),
                        help="评估环境随机种子；多 seed 统计请用 experiments/multi_seed.py --mode eval")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--benchmark_onboard", action="store_true",
                        help="启用星载可部署性分析（CPU时延+参数量+FLOPs估算）")
    parser.add_argument("--benchmark_calls", type=int, default=300,
                        help="时延统计的调度调用次数")
    parser.add_argument("--benchmark_warmup", type=int, default=50,
                        help="时延统计前的预热调用次数")
    parser.add_argument("--no_lyapunov", action="store_true",
                        help="评估时显式关闭 Lyapunov 层，主要用于旧消融 checkpoint")
    parser.add_argument("--no_psf", action="store_true",
                        help="评估时显式关闭 PSF，主要用于旧消融 checkpoint")
    parser.add_argument("--use_inference_mpc", action="store_true",
                        help="评估时显式启用 inference-time MPC（包裹 actor 的 shooting planner）")
    parser.add_argument("--no_inference_mpc", action="store_true",
                        help="评估时显式关闭 inference MPC（与 --use_inference_mpc 互斥）")
    parser.add_argument("--output", default="evaluation_report.json")
    args = parser.parse_args()

    force_enable_lyapunov = False if args.no_lyapunov else None
    force_use_psf = False if args.no_psf else None
    if args.use_inference_mpc and args.no_inference_mpc:
        raise SystemExit("--use_inference_mpc 与 --no_inference_mpc 不能同时指定")
    if args.use_inference_mpc:
        force_use_inference_mpc = True
    elif args.no_inference_mpc:
        force_use_inference_mpc = False
    else:
        force_use_inference_mpc = None

    stats1, stats2 = compare_models(
        args.model, args.baseline, args.eval_episodes, args.device,
        force_enable_lyapunov=force_enable_lyapunov,
        force_use_psf=force_use_psf,
        force_use_inference_mpc=force_use_inference_mpc,
        eval_seed=args.eval_seed)

    report = {"model": args.model, "stats": stats1}
    if stats2:
        report["baseline"] = args.baseline
        report["baseline_stats"] = stats2

    if args.benchmark_onboard:
        model_bench = benchmark_onboard_feasibility(
            args.model,
            n_calls=args.benchmark_calls,
            warmup=args.benchmark_warmup)
        report["onboard_benchmark"] = {"model": model_bench}
        print_onboard_benchmark("model", model_bench)

        if args.baseline:
            baseline_bench = benchmark_onboard_feasibility(
                args.baseline,
                n_calls=args.benchmark_calls,
                warmup=args.benchmark_warmup)
            report["onboard_benchmark"]["baseline"] = baseline_bench
            print_onboard_benchmark("baseline", baseline_bench)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"\n报告已保存: {args.output}")


if __name__ == "__main__":
    main()

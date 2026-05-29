"""
LS-PSF CMDP 主训练入口（纯任务 reward + 约束 Critic + 安全投影）。

职责：
  1. 训练论文主模型（默认仍保存为 checkpoints_optimized/best_optimized.pt 以兼容旧脚本）
  2. 支持课程学习、评估与断点续训
  3. 作为当前论文主线的唯一训练入口
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    # 仅在脚本直跑时追加，避免导入期全局污染 sys.path
    sys.path.append(_PROJECT_ROOT)

import argparse
import json
import multiprocessing as mp
import random
import traceback
from datetime import datetime

import numpy as np
import torch

#正常全速训练时请务必注释掉这两行：
# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

from config import (
    TRAIN_CONFIG, DRL_CONFIG, QUEUE_CONFIG, OBJECTIVE_VERSION,
    ORBITAL_CONFIG, ENERGY_CONFIG, THERMAL_CONFIG, REWARD_CONFIG,
    EXPERIMENT_PROTOCOL,
)
from algorithms.adaptive_lyapunov_dual import adaptive_lyapunov_coeff_step
from constraints.safety_cost import compute_lyapunov_safety_cost
from environment.satellite_env import VLEOSatelliteEnv
from environment.wrappers import DilatedFrameStackWrapper
from scheduler.integrated_scheduler import IntegratedScheduler
from utils.logger import TrainingLogger
from utils.paper_metrics import add_paper_metrics


def _resolve_device(device_arg: str) -> str:
    """解析设备参数，避免在无 CUDA 环境下崩溃。"""
    req = (device_arg or "auto").lower()
    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if req == "cuda" and not torch.cuda.is_available():
        print("[警告] 请求 CUDA 但当前环境不可用，自动降级为 CPU")
        return "cpu"

    if req == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        print("[警告] 请求 MPS 但当前环境不可用，自动降级为 CPU")
        return "cpu"

    return req


def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _safe_in_window(env) -> bool:
    contact = getattr(env, "_contact", None)
    if contact is None:
        return False
    return bool(contact.get("in_window", False))


def _available_power_w(env) -> float | None:
    """读取环境提供的 P_available，供动作边界层执行功率约束。"""
    try:
        return float(getattr(env, "available_power_w"))
    except Exception:
        return None


def _training_env_context(env) -> dict:
    """提取调度器需要的环境上下文，供本地/子进程环境统一使用。"""
    base_env = getattr(env, "env", env)
    contact = getattr(base_env, "_contact", None) or {}
    step_count = int(getattr(base_env, "step_count", 0) or 0)
    smooth_n = int(getattr(base_env, "N_PROP_SMOOTH", 1) or 1)
    return {
        "qe": float(base_env.energy_queue.value),
        "qh": float(base_env.orbit_queue.value),
        "qd": float(base_env.data_queue.length),
        "qc": float(base_env.comm_queue.value),
        "in_window": bool(contact.get("in_window", False)),
        "h": float(base_env.altitude_m),
        "soc": float(base_env.battery.soc),
        "time_s": float(base_env.time_s),
        "step_count": step_count,
        "prop_can_update": bool(step_count % max(smooth_n, 1) == 0),
        "orbital_phase": float(base_env.orbit_sim.phase),
        "tx_capacity_mbps": float(contact.get("max_capacity_mbps", 0.0)),
        "available_power_w": _available_power_w(base_env),
    }


def _make_training_env(seed: int, stack_len: int):
    """创建一个训练环境实例；子进程和串行模式共用同一入口。"""
    base_env = VLEOSatelliteEnv(seed=seed)
    env = DilatedFrameStackWrapper(base_env, k=stack_len)
    return env, base_env


def _env_worker(remote, seed: int, stack_len: int):
    """子进程环境 worker：持有独立环境，接收 reset/step/set_data_scale 指令。"""
    env, base_env = _make_training_env(seed, stack_len)
    try:
        while True:
            cmd, payload = remote.recv()
            if cmd == "reset":
                state = env.reset()
                remote.send(("ok", (state, _training_env_context(env))))
            elif cmd == "step":
                action = np.asarray(payload["action"], dtype=np.float32)
                state, reward, done, info = env.step(
                    action,
                    enforce_prop_smoothing=bool(payload.get("enforce_prop_smoothing", False)),
                )
                remote.send(("ok", (state, reward, done, info, _training_env_context(env))))
            elif cmd == "set_data_scale":
                base_env._data_arrival_scale = float(payload["scale"])
                remote.send(("ok", None))
            elif cmd == "set_randomization_scale":
                base_env._randomization_scale = float(payload["scale"])
                remote.send(("ok", None))
            elif cmd == "close":
                remote.send(("ok", None))
                break
            else:
                raise ValueError(f"未知环境指令: {cmd}")
    except EOFError:
        pass
    except Exception:
        remote.send(("error", traceback.format_exc()))
    finally:
        remote.close()


class SubprocessEnvPool:
    """轻量级多进程环境池，避免主进程串行执行多个环境物理仿真。"""

    def __init__(self, n_envs: int, seed: int, stack_len: int):
        self.n_envs = int(n_envs)
        self._closed = False
        ctx = mp.get_context("spawn")
        self._remotes = []
        self._processes = []
        self._last_data_scale = None
        self._pending_step_indices = None
        for idx in range(self.n_envs):
            parent_remote, child_remote = ctx.Pipe()
            proc = ctx.Process(
                target=_env_worker,
                args=(child_remote, int(seed + idx), int(stack_len)),
                daemon=True,
            )
            proc.start()
            child_remote.close()
            self._remotes.append(parent_remote)
            self._processes.append(proc)

    def _recv_checked(self, remote):
        status, payload = remote.recv()
        if status == "error":
            raise RuntimeError(f"子进程环境异常:\n{payload}")
        return payload

    def reset(self, indices: list[int] | None = None):
        indices = list(range(self.n_envs)) if indices is None else list(indices)
        for idx in indices:
            self._remotes[idx].send(("reset", {}))
        return [self._recv_checked(self._remotes[idx]) for idx in indices]

    def set_data_scale(self, scale: float):
        # data_scale 只会在课程阶段切换时变化；跳过重复 Pipe 通信可以减少多进程采样开销。
        scale = float(scale)
        if self._last_data_scale is not None and np.isclose(
            scale, self._last_data_scale, rtol=0.0, atol=1e-12
        ):
            return
        for remote in self._remotes:
            remote.send(("set_data_scale", {"scale": scale}))
        for remote in self._remotes:
            self._recv_checked(remote)
        self._last_data_scale = scale

    def set_randomization_scale(self, scale: float):
        """随课程阶段调节 domain randomization 幅度 ∈ [0, 1]。"""
        scale = float(scale)
        last = getattr(self, "_last_randomization_scale", None)
        if last is not None and np.isclose(scale, last, rtol=0.0, atol=1e-12):
            return
        for remote in self._remotes:
            remote.send(("set_randomization_scale", {"scale": scale}))
        for remote in self._remotes:
            self._recv_checked(remote)
        self._last_randomization_scale = scale

    def step(self, indices: list[int], actions: list[np.ndarray]):
        self.step_async(indices, actions)
        return self.recv_step()

    def step_async(self, indices: list[int], actions: list[np.ndarray]):
        if self._pending_step_indices is not None:
            raise RuntimeError("已有未接收的异步环境 step，请先调用 recv_step()")
        indices = list(indices)
        for idx, action in zip(indices, actions):
            self._remotes[idx].send((
                "step",
                {"action": np.asarray(action, dtype=np.float32),
                 "enforce_prop_smoothing": False},
            ))
        self._pending_step_indices = indices

    def recv_step(self):
        if self._pending_step_indices is None:
            raise RuntimeError("没有待接收的异步环境 step")
        indices = self._pending_step_indices
        self._pending_step_indices = None
        return [self._recv_checked(self._remotes[idx]) for idx in indices]

    def close(self):
        if self._closed:
            return
        self._closed = True
        for remote in self._remotes:
            try:
                remote.send(("close", {}))
            except Exception:
                pass
        for remote in self._remotes:
            try:
                self._recv_checked(remote)
            except Exception:
                pass
            try:
                remote.close()
            except Exception:
                pass
        for proc in self._processes:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _is_finite_number(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def _coerce_action_like(action, reference) -> np.ndarray:
    """Pad/truncate legacy 3-D actions to match the current action shape."""
    ref = np.asarray(reference, dtype=np.float32).reshape(-1)
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.size < ref.size:
        arr = np.pad(arr, (0, ref.size - arr.size), mode="constant")
    elif arr.size > ref.size:
        arr = arr[:ref.size]
    return arr.astype(np.float32, copy=False)


def _checkpoint_training_tag(path: str) -> str | None:
    """读取 checkpoint 的训练口径标识，用于防止误续训不兼容模型。"""
    if not path or not os.path.exists(path):
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # 老 PyTorch 没有 weights_only 时不做不安全回退，宁可跳过自动续训。
        return None
    except Exception:
        return None
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(metadata, dict):
        return None
    version = metadata.get("objective_version")
    return str(version) if version is not None else None


def _checkpoint_matches_current_objective(path: str) -> bool:
    """只有当前训练口径一致的 checkpoint 才允许自动续训/参与历史 best 比较。"""
    return _checkpoint_training_tag(path) == OBJECTIVE_VERSION


def _objective_summary() -> dict:
    # 这个摘要会被写入训练报告和 checkpoint 元数据，
    # 用来固定“正式优化目标 / 训练代理目标 / 选模标准”的表述。
    return {
        "primary_target": (
            "VoI-aware CMDP: maximize AoI/deadline-discounted delivered task value "
            "under VLEO energy, orbit, thermal, queue and contact-window constraints"
        ),
        "primary_algorithm": (
            "Decoupled Constraint-Critic SAC with separate reward Q_r and safety-cost Q_c, "
            "plus an adaptive global dual weight updated from normalized CMDP constraint violation"
        ),
        "mission_reward": (
            "ablation throughput reward"
            if str(REWARD_CONFIG.get("reward_mode", "value_aware")).lower() == "throughput"
            else "r_t = w_v * delivered_value + w_on * on_time_value; "
                 "energy, orbit, thermal, overflow, expiration and drop risks are excluded "
                 "from reward TD targets"
        ),
        "constraint_cost": (
            "c_t = queue + processed-backlog + window-waste + energy + orbit + thermal "
            "+ high-value task-loss + small efficiency costs; "
            "lambda uses EMA(c_t/norm) - threshold"
        ),
        "deployment_safety_layer": (
            "Lyapunov projection and PSF are deployment-time execution shields; intervention "
            "rates are diagnostics and do not drive reward TD or adaptive dual updates"
        ),
        "task_value_model": (
            "orbital phase maps to semantic scenes such as military, disaster, urban, "
            "routine land, ocean and cloud; each scene controls arrival rate, priority, "
            "quality/cloud penalty, AoI/VoI decay horizon and base value multiplier"
        ),
        "emergency_event_process": (
            "low-probability short-duration emergency_disaster events provide a non-stationary "
            "robustness benchmark without changing the 43-D observation schema"
        ),
        "observation_schema": (
            f"{int(DRL_CONFIG.get('state_dim', 40))}-D state includes grouped High/Mid/Low raw and processed queues, "
            "expiring value histograms, current scene class, upcoming value-pressure intensity, "
            "future contact capacity, thermal margin and CPU backpressure feedback"
        ),
        "action_schema": (
            f"{int(DRL_CONFIG.get('action_dim', 10))}-D action = physical power allocation "
            "[prop,cpu,tx] + CPU/TX High/Mid/Low logits + Low-drop strength"
        ),
        "network_input_preprocessing": "SAC Actor/Critic receive RunningMeanStd-normalized observations; evaluation freezes the statistics",
        "link_capacity_model": "ground-station capacity uses discrete AMC/MCS levels selected by SNR, then applies low-elevation Doppler/path penalty",
        "auxiliary_training_reward_terms": [
            "w_deadline_success * on_time_delivered_value",
        ],
        "risk_boundaries": {
            "normal": f"h>={ORBITAL_CONFIG.get('altitude_warning_km', 200.0):.0f}km and SOC>={ENERGY_CONFIG.get('battery_min_soc', 0.15):.0%}",
            "warning": f"{ORBITAL_CONFIG.get('altitude_min_km', 180.0):.0f}km<=h<{ORBITAL_CONFIG.get('altitude_warning_km', 200.0):.0f}km or SOC in ({ENERGY_CONFIG.get('battery_crash_soc', 0.05):.0%},{ENERGY_CONFIG.get('battery_min_soc', 0.15):.0%})",
            "unsafe": f"{ORBITAL_CONFIG.get('altitude_crash_km', 120.0):.0f}km<h<{ORBITAL_CONFIG.get('altitude_min_km', 180.0):.0f}km",
            "failure": f"h<={ORBITAL_CONFIG.get('altitude_crash_km', 120.0):.0f}km or SOC<={ENERGY_CONFIG.get('battery_crash_soc', 0.05):.0%}",
            "altitude_warning_km": float(ORBITAL_CONFIG.get("altitude_warning_km", 200.0)),
            "altitude_min_km": float(ORBITAL_CONFIG.get("altitude_min_km", 180.0)),
            "altitude_crash_km": float(ORBITAL_CONFIG.get("altitude_crash_km", 120.0)),
            "battery_min_soc": float(ENERGY_CONFIG.get("battery_min_soc", 0.15)),
            "battery_crash_soc": float(ENERGY_CONFIG.get("battery_crash_soc", 0.05)),
            "thermal_warning_temp_c": float(THERMAL_CONFIG.get("warning_temp_c", 45.0)),
            "thermal_max_temp_c": float(THERMAL_CONFIG.get("max_temp_c", 55.0)),
        },
        "notes": [
            "delivered_value/voi_delivered_value is accumulated only when processed task data reaches the ground and is weighted by AoI/VoI timeliness",
            "VoI degradation and drop-value signals are reported as constraint diagnostics, not reward terms",
            "processed queue backpressure is applied in the LS-PSF actuator projection before environment execution",
            "best-model selection uses safety-adjusted delivered value, not reward alone",
            "scene-aware value prevents low-utility cloudy/ocean data from being treated like urgent military/disaster tasks",
            "thermal state is part of the observation; thermal risk is learned through constraint cost and scheduler-side projection, not task reward",
        ],
    }


def _selection_tuple(stats: dict) -> tuple[float, ...]:
    # 最佳模型选择顺序：先满足安全/proc-dl/backlog/能耗约束，再按交付价值和窗口利用率排序。
    safety_rate = float(stats.get("safety_rate", 0.0))
    violation_pct = float(stats.get("violation_percentage", max(0.0, (1.0 - safety_rate) * 100.0)))
    delivered_value = float(stats.get("delivered_value_mean", stats.get("downlink_mean", stats.get("tx_mb_mean", 0.0))))
    high_value_delivered = float(stats.get(
        "high_value_downlink_value_mean",
        stats.get("high_value_downlink_mb_mean", 0.0),
    ))
    downlink = float(stats.get("downlink_mean", stats.get("tx_mb_mean", 0.0)))
    reward_mean = float(stats.get("reward_mean", 0.0))
    proc_dl = float(stats.get(
        "global_proc_downlink_ratio",
        stats.get("proc_downlink_ratio", np.inf),
    ))
    mean_ep_proc_dl = float(stats.get(
        "mean_episode_proc_downlink_ratio",
        stats.get("episode_proc_dl_ratio", proc_dl),
    ))
    processed_final_util = float(stats.get("processed_queue_final_utilization", 0.0))
    processed_peak_util = float(stats.get(
        "processed_queue_peak_utilization",
        processed_final_util,
    ))
    future_ratio = float(stats.get("processed_queue_future_contact_ratio", 0.0))
    future_ratio_peak = float(stats.get(
        "processed_queue_future_contact_ratio_peak",
        future_ratio,
    ))
    cpu_far_rate = float(stats.get("cpu_active_far_from_window_rate", 0.0))
    energy_violation_rate = float(stats.get(
        "energy_violation_rate",
        stats.get("energy_unsafe_rate", 0.0),
    ))
    window_util = float(stats.get("comm_window_utilization", 0.0))
    lyapunov_proj_rate = float(stats.get("lyapunov_proj_rate", stats.get("lyapunov_projected_rate_eval", 0.0)))
    was_projected_rate = float(stats.get(
        "safety_intervention_rate",
        stats.get("was_projected_rate", lyapunov_proj_rate),
    ))
    action_mod_l2_mean = float(stats.get(
        "action_mod_l2_mean",
        stats.get("psf_mean_mod_l2", 0.0),
    ))
    stability_penalty = (
        float(DRL_CONFIG.get("checkpoint_proj_penalty_mb", 0.0)) * lyapunov_proj_rate
        + float(DRL_CONFIG.get("checkpoint_projected_penalty_mb", 0.0)) * was_projected_rate
        + float(DRL_CONFIG.get("checkpoint_action_mod_penalty_mb", 0.0)) * action_mod_l2_mean
    )
    max_proc_dl = float(DRL_CONFIG.get("checkpoint_max_proc_downlink_ratio", np.inf))
    max_processed_util = float(DRL_CONFIG.get("checkpoint_max_processed_queue_final_utilization", np.inf))
    max_future_ratio = float(DRL_CONFIG.get("checkpoint_max_processed_queue_future_contact_ratio", np.inf))
    max_cpu_far = float(DRL_CONFIG.get("checkpoint_max_cpu_far_from_window_rate", np.inf))
    max_energy_violation = float(DRL_CONFIG.get("checkpoint_max_energy_violation_rate", np.inf))
    min_delivered_baseline = float(DRL_CONFIG.get("checkpoint_min_delivered_value", 0.0))
    delivered_fraction_floor = float(DRL_CONFIG.get("checkpoint_min_delivered_value_fraction", 0.0))
    if "baseline_delivered_value_mean" in stats:
        min_delivered_baseline = max(
            min_delivered_baseline,
            delivered_fraction_floor * float(stats.get("baseline_delivered_value_mean", 0.0)),
        )

    safety_feasible = safety_rate >= 1.0 - 1e-9 and violation_pct <= 1e-9
    proc_feasible = proc_dl <= max_proc_dl and mean_ep_proc_dl <= max_proc_dl
    backlog_feasible = processed_final_util <= max_processed_util and processed_peak_util <= max_processed_util
    capacity_feasible = future_ratio <= max_future_ratio and future_ratio_peak <= max_future_ratio
    cpu_feasible = cpu_far_rate <= max_cpu_far
    energy_feasible = energy_violation_rate <= max_energy_violation
    value_feasible = delivered_value >= min_delivered_baseline
    feasible = 1.0 if all((
        safety_feasible,
        proc_feasible,
        backlog_feasible,
        capacity_feasible,
        cpu_feasible,
        energy_feasible,
        value_feasible,
    )) else 0.0

    constraint_violation = (
        max(0.0, violation_pct)
        + max(0.0, proc_dl - max_proc_dl)
        + max(0.0, mean_ep_proc_dl - max_proc_dl)
        + max(0.0, processed_final_util - max_processed_util)
        + max(0.0, processed_peak_util - max_processed_util)
        + max(0.0, future_ratio - max_future_ratio)
        + max(0.0, future_ratio_peak - max_future_ratio)
        + max(0.0, cpu_far_rate - max_cpu_far)
        + max(0.0, energy_violation_rate - max_energy_violation)
        + max(0.0, min_delivered_baseline - delivered_value) / max(min_delivered_baseline, 1.0)
    )
    safety_adjusted_value = delivered_value - stability_penalty
    return (
        feasible,
        -constraint_violation,
        safety_adjusted_value,
        delivered_value,
        high_value_delivered,
        window_util,
        -proc_dl,
        -processed_peak_util,
        downlink,
        reward_mean,
    )


def evaluate(eval_env, scheduler, n_episodes: int = None,
             data_scale: float | None = None) -> dict:
    """独立评估窗口，确保安全统计不受训练历史污染。"""
    n_episodes = int(TRAIN_CONFIG.get("eval_episodes", 30) if n_episodes is None else n_episodes)
    scheduler.reset_all_safety_stats()

    base_eval_env = getattr(eval_env, "env", eval_env)
    previous_data_scale = float(getattr(base_eval_env, "_data_arrival_scale", 1.0))
    eval_data_scale = previous_data_scale if data_scale is None else float(data_scale)
    base_eval_env._data_arrival_scale = eval_data_scale
    # 关键：eval 期间关掉 per-episode 随机 ds，否则每次 reset 会覆盖固定 eval ds，破坏对比口径。
    prev_random_ds = getattr(base_eval_env, "_random_ds_enabled", False)
    base_eval_env._random_ds_enabled = False

    rewards, reward_per_steps, throughputs, tx_mbs, delivered_values, safes, survivals = [], [], [], [], [], [], []
    # 诊断用：分解 processed_value 去向（delivered / expired_processed / dropped_processed / discount_loss）
    ep_processed_values_diag, ep_expired_processed_values_diag = [], []
    ep_dropped_processed_values_diag, ep_expired_raw_values_diag = [], []
    deadline_rates, expired_rates, drop_rates, aoi_steps, overall_safe_rates = [], [], [], [], []
    value_weighted_deadline_rates, value_weighted_aoi_steps, voi_loss_rates = [], [], []
    high_value_delivery_rates, processed_final_utils, tx_active_contact_flags = [], [], []
    high_value_downlink_mbs, low_value_drop_mbs, active_low_drop_mbs = [], [], []
    high_value_downlink_values, low_value_drop_values = [], []
    passive_low_drop_mbs, low_drop_recalls = [], []
    low_processing_ratios, low_delivery_ratios = [], []
    useful_processing_ratios, cpu_active_far_from_window_flags = [], []
    processed_since_contact_values, delivered_since_contact_values = [], []
    processed_future_contact_ratios = []
    episode_proc_dl_ratios, episode_energy_per_value = [], []
    processed_queue_utils = []
    stage_rate_sums = {"normal": [], "warning": [], "unsafe": [], "failure": []}
    raw_overflow_mbs, processed_overflow_mbs = [], []
    energy_violation_flags = []
    projected_flags, safety_intervention_flags, action_mods = [], [], []
    prop_smoothing_flags, boundary_clip_flags, power_clip_flags = [], [], []
    lyapunov_projected_flags, psf_modified_flags, environment_execution_flags = [], [], []
    prop_powers, cpu_powers, tx_powers, energy_whs, window_utils = [], [], [], [], []

    try:
        for _ in range(n_episodes):
            state = eval_env.reset()
            done = False
            ep_reward = ep_tput = ep_tx = ep_value = 0.0
            ep_processed_value = 0.0
            ep_expired_processed_value = 0.0
            ep_dropped_processed_value = 0.0
            ep_expired_raw_value = 0.0
            ep_high_delivered = ep_high_expired = ep_high_dropped = 0.0
            ep_high_delivered_mb = 0.0
            ep_low_dropped_mb = ep_active_low_dropped_mb = 0.0
            ep_low_dropped_value = 0.0
            ep_raw_overflow = ep_processed_overflow = 0.0
            ep_final_processed_util = 0.0
            safe_steps = total_steps = 0
            stage_counts = {"normal": 0, "warning": 0, "unsafe": 0, "failure": 0}
            ep_safe = True
            survived = True

            while not done:
                # 评估链路必须和训练链路使用同一份上下文。
                in_window = _safe_in_window(eval_env)
                prop_can_update = True
                if hasattr(eval_env, "step_count") and hasattr(eval_env, "N_PROP_SMOOTH"):
                    prop_can_update = (eval_env.step_count % eval_env.N_PROP_SMOOTH == 0)

                action, was_projected, raw_action, psf_meta = scheduler.schedule(
                    state,
                    evaluate=True,
                    in_window=in_window,
                    h=eval_env.altitude_m,
                    soc=eval_env.battery.soc,
                    time_s=eval_env.time_s,
                    prop_can_update=prop_can_update,
                    orbital_phase=eval_env.orbit_sim.phase,
                    tx_capacity_mbps=float((eval_env._contact or {}).get("max_capacity_mbps", 0.0)),
                    available_power_w=_available_power_w(eval_env),
                    env=eval_env,
                )

                state, reward, done, info = eval_env.step(
                    action, enforce_prop_smoothing=False)
                executed_action = np.asarray(info.get("executed_action", action), dtype=np.float32)
                safe_action_for_diff = _coerce_action_like(action, executed_action)
                raw_action_for_diff = _coerce_action_like(raw_action, executed_action)
                execution_mod_l2 = float(np.linalg.norm(executed_action - safe_action_for_diff))
                total_execution_mod_l2 = float(np.linalg.norm(executed_action - raw_action_for_diff))
                environment_modified = bool(execution_mod_l2 > 1e-8)
                safety_intervention = bool(
                    psf_meta.get("safety_intervention_projected", False)
                    or environment_modified
                )
                projected_flags.append(float(was_projected or environment_modified))
                safety_intervention_flags.append(float(safety_intervention))
                prop_smoothing_flags.append(float(psf_meta.get("prop_smoothing_applied", False)))
                boundary_clip_flags.append(float(psf_meta.get("boundary_clipped", False)))
                power_clip_flags.append(float(psf_meta.get("power_clipped", False)))
                lyapunov_projected_flags.append(float(psf_meta.get("lyapunov_projected", False)))
                psf_modified_flags.append(float(
                    psf_meta.get("psf_modified", False)
                    or psf_meta.get("psf_triggered", False)
                ))
                environment_execution_flags.append(float(environment_modified))
                action_mods.append(float(max(
                    psf_meta.get("total_modification_l2", 0.0),
                    total_execution_mod_l2,
                    0.0,
                )))

                ep_reward += reward
                ep_tput += info.get(
                    "processed_mb",
                    info.get("service_rate_mbs", 0.0) * TRAIN_CONFIG["time_slot_s"],
                )
                ep_processed_value += float(info.get("processed_value", 0.0))
                ep_expired_processed_value += float(info.get("expired_processed_value", 0.0))
                ep_dropped_processed_value += float(info.get("dropped_processed_value", 0.0))
                ep_expired_raw_value += float(info.get("expired_raw_value", 0.0))
                ep_tx += info.get("delivered_mb", info.get("actual_tx_mb", 0.0))
                ep_value += info.get("delivered_value", 0.0)
                ep_high_delivered += float(info.get("delivered_high_value", 0.0))
                ep_high_delivered_mb += float(info.get("delivered_high_mb", 0.0))
                ep_high_expired += float(info.get("expired_high_value", 0.0))
                ep_high_dropped += float(info.get("dropped_high_value", 0.0))
                ep_low_dropped_mb += float(info.get("low_value_dropped_mb", info.get("dropped_low_mb", 0.0)))
                ep_active_low_dropped_mb += float(info.get("active_dropped_low_mb", 0.0))
                ep_low_dropped_value += float(info.get("low_value_dropped_value", 0.0))
                low_drop_recalls.append(float(info.get("low_drop_recall", 0.0)))
                low_processing_ratios.append(float(info.get("low_processing_ratio", 0.0)))
                low_delivery_ratios.append(float(info.get("low_delivery_ratio", 0.0)))
                ep_raw_overflow += float(info.get("raw_queue_overflow_mb", info.get("overflow_mb", 0.0)))
                ep_processed_overflow += float(info.get("processed_queue_overflow_mb", info.get("comm_overflow_mb", 0.0)))
                ep_final_processed_util = float(info.get("processed_queue_utilization", 0.0))
                processed_queue_utils.append(ep_final_processed_util)
                processed_future_contact_ratios.append(float(
                    info.get("processed_queue_future_contact_ratio_raw", info.get("processed_queue_future_contact_ratio", 0.0))))
                cpu_active_far_from_window_flags.append(float(
                    info.get("cpu_active_far_from_window_rate", 0.0)))
                processed_since_contact_values.append(float(
                    info.get("processed_since_contact_mb", 0.0)))
                delivered_since_contact_values.append(float(
                    info.get("delivered_since_contact_mb", 0.0)))
                prop_powers.append(float(info.get("P_propulsion_w", 0.0)))
                cpu_powers.append(float(info.get("P_cpu_w", 0.0)))
                tx_powers.append(float(info.get("P_tx_w", 0.0)))
                energy_whs.append(float(info.get("P_total_w", 0.0)) * TRAIN_CONFIG["time_slot_s"] / 3600.0)
                capacity_mb = float(info.get("tx_capacity_mbps", 0.0)) * TRAIN_CONFIG["time_slot_s"] / 8.0
                if bool(info.get("in_window", False)) and capacity_mb > 1e-9:
                    window_utils.append(float(info.get("delivered_mb", info.get("actual_tx_mb", 0.0))) / capacity_mb)
                    tx_active_contact_flags.append(float(
                        info.get("delivered_mb", info.get("actual_tx_mb", 0.0)) > 1e-9
                    ))
                safe_steps += int(bool(info.get("overall_safe", 1.0)))
                total_steps += 1

                if not bool(info.get("overall_safe", 1.0)):
                    ep_safe = False
                energy_safe = bool(info.get("energy_safe", info.get("soc", 1.0) >= ENERGY_CONFIG["battery_min_soc"]))
                energy_violation_flags.append(float(not energy_safe))
                if bool(info.get("terminated", False)):
                    survived = False
                    # 诊断：记录哪个子系统触发了 crash
                    crash_reason = []
                    if bool(info.get("energy_crashed", False)) or bool(info.get("soc", 1.0) <= 0.05):
                        crash_reason.append(f"energy(SOC={info.get('soc', 0):.3f})")
                    if bool(info.get("orbit_crashed", False)) or float(info.get("altitude_km", 999.0)) <= 122.0:
                        crash_reason.append(f"orbit(h={info.get('altitude_km', 0):.1f}km)")
                    if bool(info.get("thermal_crashed", False)) or float(info.get("thermal_temperature_c", 0.0)) >= 65.0:
                        crash_reason.append(f"thermal(T={info.get('thermal_temperature_c', 0):.1f}C)")
                    if not crash_reason:
                        crash_reason.append(f"unknown(SOC={info.get('soc',0):.3f},h={info.get('altitude_km',0):.1f},T={info.get('thermal_temperature_c',0):.1f})")
                    print(f"  [CRASH ep@step{total_steps}] " + ",".join(crash_reason), flush=True)
                stage = str(info.get("risk_stage", "normal"))
                if stage not in stage_counts:
                    stage = "failure" if bool(info.get("crashed", False)) else "normal"
                stage_counts[stage] += 1

            rewards.append(ep_reward)
            reward_per_steps.append(float(ep_reward / max(total_steps, 1)))
            throughputs.append(ep_tput)
            tx_mbs.append(ep_tx)
            delivered_values.append(ep_value)
            useful_processing_ratios.append(float(
                ep_value / max(ep_processed_value, 1e-9)
                if ep_processed_value > 1e-9
                else 0.0
            ))
            ep_processed_values_diag.append(float(ep_processed_value))
            ep_expired_processed_values_diag.append(float(ep_expired_processed_value))
            ep_dropped_processed_values_diag.append(float(ep_dropped_processed_value))
            ep_expired_raw_values_diag.append(float(ep_expired_raw_value))
            high_den = ep_high_delivered + ep_high_expired + ep_high_dropped
            high_value_delivery_rates.append(float(ep_high_delivered / max(high_den, 1e-9)))
            high_value_downlink_mbs.append(float(ep_high_delivered_mb))
            high_value_downlink_values.append(float(ep_high_delivered))
            low_value_drop_mbs.append(float(ep_low_dropped_mb))
            active_low_drop_mbs.append(float(ep_active_low_dropped_mb))
            passive_low_drop_mbs.append(float(max(ep_low_dropped_mb - ep_active_low_dropped_mb, 0.0)))
            low_value_drop_values.append(float(ep_low_dropped_value))
            safes.append(float(ep_safe))
            survivals.append(float(survived))
            overall_safe_rates.append(float(safe_steps / max(total_steps, 1)))
            processed_final_utils.append(float(ep_final_processed_util))
            episode_proc_dl_ratios.append(float(ep_tput / max(ep_tx, 1e-9)))
            episode_energy_per_value.append(float(
                sum(energy_whs[-total_steps:]) / max(ep_value, 1e-9)
                if total_steps > 0 else 0.0
            ))
            for stage_name, values in stage_rate_sums.items():
                values.append(float(stage_counts[stage_name] / max(total_steps, 1)))
            raw_overflow_mbs.append(ep_raw_overflow)
            processed_overflow_mbs.append(ep_processed_overflow)
            summary = getattr(eval_env, "task_tracker", None)
            task_summary = summary.summary() if summary is not None else {}
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
    finally:
        base_eval_env._data_arrival_scale = previous_data_scale
        base_eval_env._random_ds_enabled = prev_random_ds

    safety_stats = scheduler.get_safety_stats()
    eval_stats = {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_per_step_mean": float(np.mean(reward_per_steps)),
        "reward_per_step_std": float(np.std(reward_per_steps)),
        "processed_mean": float(np.mean(throughputs)),
        "processed_std": float(np.std(throughputs)),
        "downlink_mean": float(np.mean(tx_mbs)),
        "downlink_std": float(np.std(tx_mbs)),
        "delivered_value_mean": float(np.mean(delivered_values)),
        "delivered_value_std": float(np.std(delivered_values)),
        "deadline_success_rate": float(np.mean(deadline_rates)),
        "value_weighted_deadline_success_rate": float(np.mean(value_weighted_deadline_rates)),
        "expired_value_rate": float(np.mean(expired_rates)),
        "voi_degradation_rate": float(np.mean(expired_rates)),
        "average_aoi_steps": float(np.mean(aoi_steps)) if aoi_steps else 0.0,
        "value_weighted_aoi_steps": float(np.mean(value_weighted_aoi_steps)) if value_weighted_aoi_steps else 0.0,
        "dropped_value_rate": float(np.mean(drop_rates)),
        "voi_loss_rate": float(np.mean(voi_loss_rates)) if voi_loss_rates else 0.0,
        "high_value_delivery_rate": float(np.mean(high_value_delivery_rates)) if high_value_delivery_rates else 0.0,
        "high_value_delivery_ratio": float(np.mean(high_value_delivery_rates)) if high_value_delivery_rates else 0.0,
        "high_value_downlink_mb_mean": float(np.mean(high_value_downlink_mbs)) if high_value_downlink_mbs else 0.0,
        "episode_high_value_downlink_mb_mean": float(np.mean(high_value_downlink_mbs)) if high_value_downlink_mbs else 0.0,
        "high_value_downlink_value_mean": float(np.mean(high_value_downlink_values)) if high_value_downlink_values else 0.0,
        "low_value_dropped_mb_mean": float(np.mean(low_value_drop_mbs)) if low_value_drop_mbs else 0.0,
        "episode_low_value_dropped_mb_mean": float(np.mean(low_value_drop_mbs)) if low_value_drop_mbs else 0.0,
        "active_low_dropped_mb_mean": float(np.mean(active_low_drop_mbs)) if active_low_drop_mbs else 0.0,
        "active_low_drop_mb_mean": float(np.mean(active_low_drop_mbs)) if active_low_drop_mbs else 0.0,
        "passive_low_drop_mb_mean": float(np.mean(passive_low_drop_mbs)) if passive_low_drop_mbs else 0.0,
        "low_drop_recall": float(np.mean(low_drop_recalls)) if low_drop_recalls else 0.0,
        "low_processing_ratio": float(np.mean(low_processing_ratios)) if low_processing_ratios else 0.0,
        "low_delivery_ratio": float(np.mean(low_delivery_ratios)) if low_delivery_ratios else 0.0,
        "low_value_dropped_value_mean": float(np.mean(low_value_drop_values)) if low_value_drop_values else 0.0,
        "throughput_mean": float(np.mean(throughputs)),
        "tx_mb_mean": float(np.mean(tx_mbs)),
        "safety_rate": float(np.mean(safes)),
        "episode_safety_rate": float(np.mean(safes)),
        "survival_rate": float(np.mean(survivals)),
        "crash_count": int(np.sum(1.0 - np.asarray(survivals, dtype=float))),
        "overall_safe_rate": float(np.mean(overall_safe_rates)),
        "step_safety_rate": float(np.mean(overall_safe_rates)),
        "normal_state_rate": float(np.mean(stage_rate_sums["normal"])) if stage_rate_sums["normal"] else 0.0,
        "warning_state_rate": float(np.mean(stage_rate_sums["warning"])) if stage_rate_sums["warning"] else 0.0,
        "unsafe_state_rate": float(np.mean(stage_rate_sums["unsafe"])) if stage_rate_sums["unsafe"] else 0.0,
        "failure_state_rate": float(np.mean(stage_rate_sums["failure"])) if stage_rate_sums["failure"] else 0.0,
        "energy_violation_rate": float(np.mean(energy_violation_flags)) if energy_violation_flags else 0.0,
        "energy_unsafe_rate": float(np.mean(energy_violation_flags)) if energy_violation_flags else 0.0,
        "was_projected_rate": float(np.mean(projected_flags)) if projected_flags else 0.0,
        "chain_total_rate": float(np.mean(projected_flags)) if projected_flags else 0.0,
        "safety_intervention_rate": float(np.mean(safety_intervention_flags)) if safety_intervention_flags else 0.0,
        "intervention_rate": float(np.mean(safety_intervention_flags)) if safety_intervention_flags else 0.0,
        "prop_smoothing_rate": float(np.mean(prop_smoothing_flags)) if prop_smoothing_flags else 0.0,
        "boundary_clip_rate_eval": float(np.mean(boundary_clip_flags)) if boundary_clip_flags else 0.0,
        "power_clip_rate_eval": float(np.mean(power_clip_flags)) if power_clip_flags else 0.0,
        "lyapunov_projected_rate_eval": float(np.mean(lyapunov_projected_flags)) if lyapunov_projected_flags else 0.0,
        "psf_modified_rate": float(np.mean(psf_modified_flags)) if psf_modified_flags else 0.0,
        "environment_execution_rate": float(np.mean(environment_execution_flags)) if environment_execution_flags else 0.0,
        "action_mod_l2_mean": float(np.mean(action_mods)) if action_mods else 0.0,
        "mean_action_modification": float(np.mean(action_mods)) if action_mods else 0.0,
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
        "processed_since_contact_mb": float(np.mean(processed_since_contact_values)) if processed_since_contact_values else 0.0,
        "delivered_since_contact_mb": float(np.mean(delivered_since_contact_values)) if delivered_since_contact_values else 0.0,
        "tx_active_in_contact_ratio": float(np.mean(tx_active_contact_flags)) if tx_active_contact_flags else 0.0,
        "cpu_active_far_from_window_rate": float(np.mean(cpu_active_far_from_window_flags)) if cpu_active_far_from_window_flags else 0.0,
        "raw_overflow_mean": float(np.mean(raw_overflow_mbs)),
        "processed_overflow_mean": float(np.mean(processed_overflow_mbs)),
        "global_proc_downlink_ratio": float(np.sum(throughputs) / max(np.sum(tx_mbs), 1e-9)),
        "mean_episode_proc_downlink_ratio": float(np.mean(episode_proc_dl_ratios)) if episode_proc_dl_ratios else 0.0,
        "proc_downlink_ratio": float(np.sum(throughputs) / max(np.sum(tx_mbs), 1e-9)),
        "episode_proc_dl_ratio": float(np.mean(episode_proc_dl_ratios)) if episode_proc_dl_ratios else 0.0,
        "energy_per_delivered_value_episode": float(np.mean(episode_energy_per_value)) if episode_energy_per_value else 0.0,
        "useful_processing_ratio": float(np.mean(useful_processing_ratios)) if useful_processing_ratios else 0.0,
        "episode_useful_processing_ratio": float(np.mean(useful_processing_ratios)) if useful_processing_ratios else 0.0,
        # ── Phase B 诊断：processed_value 去向分解 ──
        # delivered (有效) + expired_processed (处理完过期) + dropped_processed (处理完被丢)
        # + discount_loss (timeliness × specificity)，加起来等于 processed_value
        "processed_value_mean": float(np.mean(ep_processed_values_diag)) if ep_processed_values_diag else 0.0,
        "expired_processed_value_mean": float(np.mean(ep_expired_processed_values_diag)) if ep_expired_processed_values_diag else 0.0,
        "dropped_processed_value_mean": float(np.mean(ep_dropped_processed_values_diag)) if ep_dropped_processed_values_diag else 0.0,
        "expired_raw_value_mean": float(np.mean(ep_expired_raw_values_diag)) if ep_expired_raw_values_diag else 0.0,
        "discount_loss_value_mean": float(max(0.0, (
            np.mean(ep_processed_values_diag)
            - np.mean(delivered_values)
            - np.mean(ep_expired_processed_values_diag)
            - np.mean(ep_dropped_processed_values_diag)
        ))) if ep_processed_values_diag else 0.0,
        "eval_data_arrival_scale": float(eval_data_scale),
        **{k: float(v) if isinstance(v, (int, float)) else v for k, v in safety_stats.items()},
    }
    eval_stats["checkpoint_value_score"] = float(_selection_tuple(eval_stats)[2])
    eval_stats["checkpoint_downlink_score"] = eval_stats["checkpoint_value_score"]
    return add_paper_metrics(eval_stats)


def _load_existing_best_eval(checkpoint_path: str,
                             eval_episodes: int,
                             device: str,
                             enable_lyapunov: bool,
                             use_psf: bool) -> dict | None:
    """恢复已有 best checkpoint 的评估值，避免续训时首轮评估覆盖历史最优。"""
    # 断点续训时优先恢复磁盘上历史 best 的评估结果，
    # 避免本轮第一次评估就把真正的历史最优覆盖掉。
    if not os.path.exists(checkpoint_path):
        return None

    try:
        eval_base = VLEOSatelliteEnv(seed=int(TRAIN_CONFIG.get("seed", 42)) + 2000)
        eval_env = DilatedFrameStackWrapper(
            eval_base, k=int(DRL_CONFIG.get("frame_stack", 8)))
        scheduler = IntegratedScheduler(
            device=device,
            enable_lyapunov=enable_lyapunov,
            use_psf=use_psf,
        )
        if not enable_lyapunov:
            scheduler.agent.set_lyapunov_penalty_coeff(0.0)
        # 这里是在当前训练任务里比较历史 best，必须沿用当前命令行安全层配置。
        # 否则消融续训时，旧 checkpoint metadata 会把 No_PSF/No_Lyapunov 误恢复成 Full。
        scheduler.load(checkpoint_path, restore_safety_config=False)
        if not enable_lyapunov:
            scheduler.agent.set_lyapunov_penalty_coeff(0.0)
        return evaluate(eval_env, scheduler, n_episodes=eval_episodes)
    except Exception as exc:
        print(f"[警告] 无法恢复已有 best 评估，将在后续重新选择: {exc}")
        return None


def _build_curriculum(total_steps: int):
    # 若配置里没有课程学习，就退化成单阶段全难度训练。
    if TRAIN_CONFIG.get("use_curriculum", False):
        stages = TRAIN_CONFIG.get("curriculum_stages", [])
        if stages:
            return stages
    return [{
        "stage_name": "Full",
        "steps": total_steps,
        "lyapunov_weight_scale": 1.0,
        "data_arrival_scale": 1.0,
    }]


# 课程阶段 PSF 策略：随训练推进逐步放手，让 actor 把安全约束内化进策略。
# Exploration: PSF 保留（A+ 档默认即可，soc=0.05/K=5/no_long_horizon）
# Balancing:   PSF 进一步放宽（soc_margin=0.02, K=3）
# Ramp:        只兜灾难性违规（soc_margin=0，贴 crash 才拦）
# Optimization: PSF 完全关闭，actor 只受 boundary_clip 物理硬上限保护
_LAST_PSF_STAGE = {"value": None}


def _apply_stage_psf_policy(scheduler, stage_name: str) -> None:
    """按课程阶段渐进放手 PSF。仅在阶段切换时执行一次。"""
    if _LAST_PSF_STAGE["value"] == stage_name:
        return
    _LAST_PSF_STAGE["value"] = stage_name

    psf = getattr(scheduler, "psf", None)
    if stage_name == "Optimization":
        # 第四阶段：完全关闭 PSF，actor 自己撑住安全。
        scheduler.use_psf = False
        print(f"  [PSF policy] Optimization → PSF 完全关闭（仅保留 boundary_clip）")
        return

    if psf is None:
        # 用户用 --no_psf 启动；尊重原意，不在中途打开。
        return

    if stage_name == "Exploration":
        scheduler.use_psf = True
        psf.K = 5
        psf.line_search_steps = 3
        psf.soc_safe_min = psf.soc_crash + 0.05
        psf.long_horizon_enabled = False
        print(f"  [PSF policy] Exploration → A+ 默认（soc_margin=0.05, K=5）")
    elif stage_name == "Balancing":
        scheduler.use_psf = True
        psf.K = 3
        psf.line_search_steps = 2
        psf.soc_safe_min = psf.soc_crash + 0.02
        psf.long_horizon_enabled = False
        print(f"  [PSF policy] Balancing → 放宽（soc_margin=0.02, K=3）")
    elif stage_name == "Ramp":
        scheduler.use_psf = True
        psf.K = 3
        psf.line_search_steps = 2
        psf.soc_safe_min = psf.soc_crash  # 贴 soc_crash 才拦
        psf.long_horizon_enabled = False
        print(f"  [PSF policy] Ramp → 只兜灾难（soc_margin=0, K=3）")


def _get_stage(stages, step: int):
    # 课程学习的 data_arrival_scale + randomization_scale 都跨阶段线性 ramp，
    # 避免 Phase 边界的分布硬跳变让 critic 估值崩盘。
    cum = 0
    previous = dict(stages[0])
    for idx, stg in enumerate(stages):
        span = int(stg.get("steps", 0))
        if step < cum + span:
            current = dict(stg)
            if idx > 0 and span > 0:
                progress = float(np.clip((step - cum) / span, 0.0, 1.0))
                prev_scale = float(previous.get(
                    "data_arrival_scale",
                    current.get("data_arrival_scale", 1.0),
                ))
                target_scale = float(current.get("data_arrival_scale", prev_scale))
                current["target_data_arrival_scale"] = target_scale
                current["data_arrival_scale"] = prev_scale + (target_scale - prev_scale) * progress
                # randomization_scale 同样做线性 ramp
                prev_rand = float(previous.get(
                    "randomization_scale",
                    current.get("randomization_scale", 1.0),
                ))
                target_rand = float(current.get("randomization_scale", prev_rand))
                current["target_randomization_scale"] = target_rand
                current["randomization_scale"] = prev_rand + (target_rand - prev_rand) * progress
                current["stage_progress"] = progress
            else:
                current["target_data_arrival_scale"] = float(current.get("data_arrival_scale", 1.0))
                current["target_randomization_scale"] = float(current.get("randomization_scale", 1.0))
                current["stage_progress"] = 0.0
            return current
        cum += span
        previous = stg
    final = dict(stages[-1])
    final["target_data_arrival_scale"] = float(final.get("data_arrival_scale", 1.0))
    final["target_randomization_scale"] = float(final.get("randomization_scale", 1.0))
    final["stage_progress"] = 1.0
    return final



def _compute_safety_action_penalties(
    action_mod_l2: float,
    was_projected: bool,
    reward: float,
) -> tuple[float, float, float]:
    """计算安全层动作介入惩罚。"""
    # projection_penalty 直接惩罚“安全链路改动了原始动作”这件事，
    # 解决 actor 只学会依赖边界裁剪/Lyapunov/PSF 兜底、原始动作长期不合规的问题。
    projection_penalty = 0.0
    if was_projected:
        projection_penalty = -abs(float(DRL_CONFIG.get("projection_penalty_coeff", 0.0)))

    # action_mod_penalty 按最终安全动作和原始动作的距离连续惩罚，
    # 修改越大，说明 actor 原始动作离可执行动作越远。
    mod_coeff = float(DRL_CONFIG.get("action_mod_penalty_coeff", 1.0))
    action_mod_penalty = -mod_coeff * max(float(action_mod_l2), 0.0)

    raw_penalty = projection_penalty + action_mod_penalty
    if raw_penalty >= 0.0:
        return 0.0, 0.0, 0.0

    # 用“比例上限 + 最小上限”裁剪，既防止单步惩罚炸掉训练，
    # 又避免 reward 很小时投影惩罚被 cap 到几乎没有信号。
    cap_ratio = float(DRL_CONFIG.get("safety_action_penalty_cap_ratio", 0.25))
    min_cap = float(DRL_CONFIG.get("safety_action_penalty_min_cap", 1.0))
    penalty_cap = max(min_cap, abs(float(reward)) * cap_ratio)
    total_penalty = float(np.clip(raw_penalty, -penalty_cap, 0.0))
    return float(projection_penalty), float(action_mod_penalty), total_penalty


def _adaptive_lyapunov_coeff_step(
    current_coeff: float,
    constraint_ema: float,
    constraint_value: float,
    enabled: bool = True,
) -> tuple[float, float, float]:
    """根据安全压力 EMA 更新 Actor 中的全局 Lyapunov 约束权重。"""
    update = adaptive_lyapunov_coeff_step(
        current_coeff,
        constraint_ema,
        constraint_value,
        enabled=enabled,
    )
    return update.coeff, update.constraint_value, update.constraint_ema



def train(args):
    seed = int(getattr(args, "seed", TRAIN_CONFIG.get("seed", 42)))
    set_global_seed(seed)

    # 先解析 CLI 覆盖项；如果用户没传，就退回配置文件默认值。
    total_steps = int(getattr(args, "total_steps", TRAIN_CONFIG.get("total_steps", 1500000)))
    checkpoint_dir = getattr(
        args, "checkpoint_dir",
        TRAIN_CONFIG.get("optimized_checkpoint_dir", "checkpoints_optimized/"),
    )
    log_dir = getattr(
        args, "log_dir",
        TRAIN_CONFIG.get("optimized_log_dir", "logs_optimized/"),
    )
    eval_freq = int(getattr(args, "eval_freq", TRAIN_CONFIG.get("eval_freq", 5000)))
    save_freq = int(getattr(args, "save_freq", TRAIN_CONFIG.get("save_freq", 50000)))
    keep_step_checkpoints = bool(getattr(
        args,
        "keep_step_checkpoints",
        TRAIN_CONFIG.get("keep_step_checkpoints", False),
    ))
    eval_episodes = int(getattr(args, "eval_episodes", TRAIN_CONFIG.get("eval_episodes", 30)))
    device = _resolve_device(getattr(args, "device", "auto"))

    constraint_variant = getattr(args, "constraint_variant", "ours")
    enable_lyapunov = not bool(getattr(args, "no_lyapunov", False))
    use_psf = not bool(getattr(args, "no_psf", False))
    use_inference_mpc_cli = bool(getattr(args, "use_inference_mpc", False))
    inference_mpc_warmup_override = getattr(args, "inference_mpc_warmup_steps", None)
    if inference_mpc_warmup_override is not None:
        from config import INFERENCE_MPC_CONFIG as _MPC_CFG
        _MPC_CFG["warmup_steps"] = int(inference_mpc_warmup_override)
    if constraint_variant == "plain_sac":
        enable_lyapunov = False
        use_psf = False
    elif constraint_variant == "lagrangian_sac":
        # Lagrangian SAC 基线只在 reward/TD 目标中加入约束代价，不使用外部投影或 PSF。
        enable_lyapunov = False
        use_psf = False
    if constraint_variant == "sac_psf":
        enable_lyapunov = False
        use_psf = True
    elif constraint_variant == "sac_lyapunov":
        enable_lyapunov = True
        use_psf = False

    n_envs = max(1, int(getattr(args, "n_envs", TRAIN_CONFIG.get("n_envs", 1))))
    env_backend_arg = str(getattr(args, "env_backend", TRAIN_CONFIG.get("env_backend", "auto"))).lower()
    if env_backend_arg == "auto":
        env_backend = "subproc" if n_envs > 1 else "serial"
    elif env_backend_arg in {"serial", "subproc"}:
        env_backend = env_backend_arg
    else:
        raise ValueError(f"未知 env_backend={env_backend_arg}，可选 auto/serial/subproc")

    warmup_override = getattr(args, "warmup_steps", None)
    update_freq_override = getattr(args, "update_freq", None)
    update_actor_freq_override = getattr(args, "update_actor_freq", None)

    if warmup_override is not None:
        DRL_CONFIG["warmup_steps"] = int(warmup_override)
    if update_freq_override is not None:
        DRL_CONFIG["update_freq"] = int(update_freq_override)
        TRAIN_CONFIG["update_freq"] = int(update_freq_override)
    if update_actor_freq_override is not None:
        DRL_CONFIG["update_actor_freq"] = int(update_actor_freq_override)
    adaptive_dual_override = getattr(args, "adaptive_lyapunov_coeff_enable", None)
    if adaptive_dual_override is not None:
        DRL_CONFIG["adaptive_lyapunov_coeff_enable"] = bool(adaptive_dual_override)
    network_arch_override = getattr(args, "network_arch", None)
    if network_arch_override is not None:
        DRL_CONFIG["network_arch"] = str(network_arch_override).lower()
    bc_coeff_override = getattr(args, "behavior_cloning_coeff", None)
    if bc_coeff_override is not None:
        DRL_CONFIG["behavior_cloning_coeff"] = float(bc_coeff_override)
    mission_reward_variant = str(
        getattr(args, "mission_reward_variant", REWARD_CONFIG.get("reward_mode", "value_aware"))
    ).lower()
    if mission_reward_variant in {"throughput", "delivered_mb", "non_value"}:
        REWARD_CONFIG["reward_mode"] = "throughput"
        REWARD_CONFIG["w_delivered_mb"] = float(getattr(args, "throughput_reward_weight", 1.0))
        DRL_CONFIG["value_aux_head_enable"] = False
        DRL_CONFIG["value_aux_loss_weight"] = 0.0
        DRL_CONFIG["value_aux_loss_weight_final"] = 0.0
        DRL_CONFIG["value_action_aux_loss_weight"] = 0.0
        DRL_CONFIG["value_action_aux_loss_weight_final"] = 0.0
    else:
        # 默认训练口径不把 reward_mode 写入全局配置，保持论文主 reward 只有两个权重。
        REWARD_CONFIG.pop("reward_mode", None)
        REWARD_CONFIG.pop("w_delivered_mb", None)
        mission_reward_variant = "value_aware"

    # 训练日志和 checkpoint 目录提前创建，避免首次写盘时报错。
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 70)
    print("  LS-PSF CMDP 训练")
    print("=" * 70)
    print(f"  设备: {device}")
    print(f"  总步数: {total_steps:,}")
    print(f"  评估频率: 每 {eval_freq:,} 步")
    print(f"  每次评估: {eval_episodes} episodes")
    print(f"  step中间检查点: {'开启，每 ' + format(save_freq, ',') + ' 步' if keep_step_checkpoints else '关闭，仅保留 best/latest'}")
    print(f"  checkpoint目录: {checkpoint_dir}")
    print(f"  log目录: {log_dir}")
    print(f"  并行环境数: {n_envs}")
    print(f"  环境后端: {env_backend}")
    print(f"  Lyapunov: {'ON' if enable_lyapunov else 'OFF'}")
    print(f"  PSF: {'ON' if use_psf else 'OFF'}")
    if use_inference_mpc_cli:
        from config import INFERENCE_MPC_CONFIG as _MPC_CFG
        print(f"  Inference MPC: ON  (warmup={_MPC_CFG.get('warmup_steps', 50_000)}, "
              f"H={_MPC_CFG.get('horizon_steps', 10)}, N={_MPC_CFG.get('num_candidates', 32)})")
    else:
        print(f"  Inference MPC: OFF")
    print(f"  约束变体: {constraint_variant}")
    print(f"  随机种子: {seed}")
    print(f"  warmup_steps: {int(DRL_CONFIG.get('warmup_steps', 0))}")
    print(f"  update_freq: {int(DRL_CONFIG.get('update_freq', 1))}")
    print(f"  update_actor_freq: {int(DRL_CONFIG.get('update_actor_freq', 1))}")
    print(f"  network_arch: {DRL_CONFIG.get('network_arch', 'transformer')}")
    print(f"  behavior_cloning_coeff: {float(DRL_CONFIG.get('behavior_cloning_coeff', 0.0))}")
    print(f"  mission_reward_variant: {mission_reward_variant}")

    stack_len = int(DRL_CONFIG.get("frame_stack", 8))
    base_envs = []
    envs = []
    env_pool = None
    # 训练环境和评估环境分开建，避免评估污染训练状态。
    if env_backend == "subproc":
        env_pool = SubprocessEnvPool(n_envs, seed, stack_len)
    else:
        for i in range(n_envs):
            env, base_env = _make_training_env(seed + i, stack_len)
            base_envs.append(base_env)
            envs.append(env)

    eval_base = VLEOSatelliteEnv(seed=seed + 1000)
    eval_env = DilatedFrameStackWrapper(eval_base, k=stack_len)

    scheduler = IntegratedScheduler(
        device=device,
        enable_lyapunov=enable_lyapunov,
        use_psf=use_psf,
        use_inference_mpc=use_inference_mpc_cli,
    )
    scheduler.constraint_variant = constraint_variant
    scheduler.variant_key = getattr(args, "variant_key", None)
    scheduler.variant_code = getattr(args, "variant_code", None)
    scheduler.ablation_axis = getattr(args, "ablation_axis", None)
    scheduler.training_seed = seed
    scheduler.training_total_steps = total_steps
    scheduler.mission_reward_variant = mission_reward_variant

    def _apply_variant_agent_overrides():
        # 变体级训练隔离：没有 Lyapunov 的方法不能在 actor loss 中偷偷保留 Q_c。
        if (not enable_lyapunov) or constraint_variant in {"plain_sac", "lagrangian_sac", "sac_psf"}:
            scheduler.agent.set_lyapunov_penalty_coeff(0.0)
        elif not bool(DRL_CONFIG.get("adaptive_lyapunov_coeff_enable", True)):
            scheduler.agent.set_lyapunov_penalty_coeff(float(DRL_CONFIG.get("lyapunov_penalty_coeff", 0.0)))
        scheduler.agent.value_aux_head_enable = bool(DRL_CONFIG.get("value_aux_head_enable", False))
        scheduler.agent.value_aux_loss_weight = float(DRL_CONFIG.get("value_aux_loss_weight", 0.0))
        scheduler.agent.value_aux_loss_weight_final = float(
            DRL_CONFIG.get("value_aux_loss_weight_final", 0.0))
        scheduler.agent.value_action_aux_loss_weight = float(
            DRL_CONFIG.get("value_action_aux_loss_weight", 0.0))
        scheduler.agent.value_action_aux_loss_weight_final = float(
            DRL_CONFIG.get("value_action_aux_loss_weight_final", 0.0))

    _apply_variant_agent_overrides()

    logger = TrainingLogger(
        log_dir,
        run_name="train",
        enable_tensorboard=bool(getattr(args, "tensorboard", False)),
        wandb_project=getattr(args, "wandb_project", None),
        wandb_run_name=getattr(args, "wandb_run_name", None),
        wandb_config={
            "objective_version": OBJECTIVE_VERSION,
            "constraint_variant": constraint_variant,
            "seed": seed,
            "frame_stack": stack_len,
            "n_envs": n_envs,
            "network_arch": str(DRL_CONFIG.get("network_arch", "transformer")),
            "behavior_cloning_coeff": float(DRL_CONFIG.get("behavior_cloning_coeff", 0.0)),
            "experiment_protocol": EXPERIMENT_PROTOCOL,
        },
    )

    best_path = os.path.join(checkpoint_dir, "best_optimized.pt")
    best_so_far_path = os.path.join(checkpoint_dir, "best_so_far.pt")
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    best_so_far_score = (-np.inf,) * 10
    best_per_stage: dict[str, tuple] = {}

    manual_resume = getattr(args, "resume_path", None)
    # 续训优先级：
    # 1. 用户手动指定的 resume_path
    # 2. latest.pt
    # 3. best_optimized.pt
    if manual_resume:
        if not os.path.exists(manual_resume):
            raise FileNotFoundError(f"指定的 resume_path 不存在: {manual_resume}")
        resume_path = manual_resume
        resume_tag = _checkpoint_training_tag(resume_path)
        if resume_tag != OBJECTIVE_VERSION:
            print(
                "  [警告] 手动续训 checkpoint 的 reward 口径与当前代码不同: "
                f"{resume_tag or 'unknown'} != {OBJECTIVE_VERSION}"
            )
    else:
        resume_path = None
        for candidate in (latest_path, best_path):
            if not os.path.exists(candidate):
                continue
            version = _checkpoint_training_tag(candidate)
            if version == OBJECTIVE_VERSION:
                resume_path = candidate
                break
            # 自动续训不能加载旧 reward 口径模型，否则会把错误 TD 尺度继续带进新训练。
            print(
                "  [跳过旧checkpoint] "
                f"{candidate} training_tag={version or 'unknown'}，当前={OBJECTIVE_VERSION}"
            )
    if resume_path and os.path.exists(resume_path):
        # 续训时命令行/当前训练配置优先级高于 checkpoint metadata。
        # 这样 --no_psf、--no_lyapunov 或 constraint_variant 不会被旧 latest.pt 误恢复成 Full。
        scheduler.load(resume_path, restore_safety_config=False)
        _apply_variant_agent_overrides()
        print(f"  [续训] 已加载: {resume_path}")

    global_step = int(getattr(scheduler.agent, "total_steps", 0))
    # 历史 best 的评估结果要单独恢复出来，用来和本轮后续评估公平比较。
    best_eval = None
    if os.path.exists(best_path) and _checkpoint_matches_current_objective(best_path):
        best_eval = _load_existing_best_eval(
            best_path,
            eval_episodes,
            device,
            enable_lyapunov=enable_lyapunov,
            use_psf=use_psf,
        )
    best_score = _selection_tuple(best_eval) if best_eval is not None else (-np.inf,) * 10
    if best_eval is not None:
        print(
            "  [best] 已恢复历史最优: "
            f"safe={best_eval.get('overall_safe_rate', best_eval.get('safety_rate', 0.0)):.1%}, "
            f"ep_safe={best_eval.get('episode_safety_rate', best_eval.get('safety_rate', 0.0)):.1%}, "
            f"downlink={best_eval.get('downlink_mean', best_eval.get('tx_mb_mean', 0.0)):.3f}MB, "
            f"sel={best_eval.get('checkpoint_downlink_score', _selection_tuple(best_eval)[2]):.3f}, "
            f"reward={best_eval.get('reward_mean', 0.0):.3f}"
        )

    _qe = float(QUEUE_CONFIG["energy_queue_max"])
    _qh = float(QUEUE_CONFIG["orbit_queue_max"])
    _qd = float(QUEUE_CONFIG["data_queue_max_mb"])
    _qc = float(QUEUE_CONFIG["comm_queue_max"])

    stages = _build_curriculum(total_steps)
    fail_fast_on_nan = bool(TRAIN_CONFIG.get("fail_fast_on_nan", True))
    nan_guard_max_hits = max(1, int(TRAIN_CONFIG.get("nan_guard_max_hits", 1)))
    nan_guard_hits = 0
    projected_ema = 0.0
    adaptive_constraint_ema = 0.0
    adaptive_constraint_signal_sum = 0.0
    adaptive_constraint_signal_count = 0
    adaptive_constraint_value_used = 0.0
    adaptive_constraint_violation = 0.0
    adaptive_lya_coeff = float(scheduler.agent.get_lyapunov_penalty_coeff())
    lagrangian_lambda = 0.0
    lagrangian_lr = float(getattr(args, "lagrangian_lr", 0.02))
    lagrangian_target = float(getattr(args, "lagrangian_target", 0.03))
    lagrangian_max_lambda = float(getattr(args, "lagrangian_max_lambda", 50.0))

    def _handle_nan_event(reason: str, step_hint: int):
        # NaN guard 的处理策略是：
        # 先落盘当前现场，再按配置决定是否立即中止训练。
        nonlocal nan_guard_hits
        nan_guard_hits += 1
        crash_path = os.path.join(checkpoint_dir, f"nan_guard_step_{step_hint}.pt")
        scheduler.save(crash_path)
        print(f"[NaNGuard] {reason} | hits={nan_guard_hits}/{nan_guard_max_hits}")
        print(f"[NaNGuard] 已保存现场检查点: {crash_path}")
        if fail_fast_on_nan:
            raise RuntimeError(f"NaNGuard触发: {reason}")
        if nan_guard_hits >= nan_guard_max_hits:
            # 尝试从最新检查点恢复，而不是直接崩溃。
            if os.path.exists(latest_path):
                try:
                    scheduler.load(latest_path, restore_safety_config=False)
                    nan_guard_hits = 0
                    print(f"[NaNGuard] 已从 {latest_path} 恢复，重置 NaN 计数器。")
                except Exception as _e:
                    raise RuntimeError(f"NaNGuard触发且检查点恢复失败: {reason}") from _e
            else:
                raise RuntimeError(f"NaNGuard触发: {reason}")

    latest_update_stats = {}

    def _validate_update_stats(update_stats: dict, step_hint: int) -> bool:
        if float(update_stats.get("nan_guard_triggered", 0.0)) > 0.0:
            stage = update_stats.get("nan_guard_stage", -1)
            _handle_nan_event(
                f"agent.update 检测到非有限值并跳过更新 (stage={stage})",
                step_hint,
            )
            return False
        for key in (
            "actor_loss", "critic_loss", "constraint_critic_loss",
            "constraint_actor_loss", "alpha", "actor_lr", "critic_lr",
            "lyapunov_penalty_coeff",
        ):
            if key in update_stats and (not _is_finite_number(update_stats[key])):
                _handle_nan_event(
                    f"update_stats[{key}] 非有限值: {update_stats[key]}",
                    step_hint,
                )
                return False
        return True

    def _adaptive_dual_enabled() -> bool:
        return (
            constraint_variant in {"ours", "sac_lyapunov"}
            and enable_lyapunov
            and bool(DRL_CONFIG.get("adaptive_lyapunov_coeff_enable", True))
        )

    def _queue_adaptive_constraint_signal(value: float):
        nonlocal adaptive_constraint_signal_sum, adaptive_constraint_signal_count
        nonlocal adaptive_constraint_value_used
        if not _is_finite_number(value):
            return
        adaptive_constraint_value_used = float(max(0.0, value))
        adaptive_constraint_signal_sum += adaptive_constraint_value_used
        adaptive_constraint_signal_count += 1

    def _apply_adaptive_dual_updates(update_count: int):
        nonlocal adaptive_constraint_ema, adaptive_constraint_signal_sum
        nonlocal adaptive_constraint_signal_count, adaptive_constraint_value_used
        nonlocal adaptive_constraint_violation, adaptive_lya_coeff
        if update_count <= 0:
            return
        threshold = float(DRL_CONFIG.get(
            "adaptive_lyapunov_constraint_threshold",
            DRL_CONFIG.get("adaptive_lyapunov_coeff_target_pressure", 0.02),
        ))
        if not _adaptive_dual_enabled():
            adaptive_constraint_signal_sum = 0.0
            adaptive_constraint_signal_count = 0
            adaptive_constraint_violation = float(adaptive_constraint_ema) - threshold
            return
        if adaptive_constraint_signal_count > 0:
            signal_value = adaptive_constraint_signal_sum / max(adaptive_constraint_signal_count, 1)
        else:
            signal_value = adaptive_constraint_value_used
        if not _is_finite_number(signal_value):
            signal_value = 0.0
        for _ in range(update_count):
            update = adaptive_lyapunov_coeff_step(
                scheduler.agent.get_lyapunov_penalty_coeff(),
                adaptive_constraint_ema,
                signal_value,
                enabled=True,
            )
            adaptive_lya_coeff = update.coeff
            adaptive_constraint_value_used = update.constraint_value
            adaptive_constraint_ema = update.constraint_ema
            scheduler.agent.set_lyapunov_penalty_coeff(adaptive_lya_coeff)
        adaptive_constraint_violation = float(adaptive_constraint_ema) - threshold
        adaptive_constraint_signal_sum = 0.0
        adaptive_constraint_signal_count = 0

    def _run_scheduled_updates(stored_steps: int):
        # 多环境模式先批量写 replay，再在这里连续执行该批次对应的 update。
        nonlocal latest_update_stats
        update_freq = max(1, int(DRL_CONFIG.get("update_freq", 1)))
        total_steps_now = int(getattr(scheduler.agent, "total_steps", 0))
        previous_steps = max(0, total_steps_now - stored_steps)
        update_count = (total_steps_now // update_freq) - (previous_steps // update_freq)
        if update_count > 0:
            _apply_adaptive_dual_updates(update_count)
        for update_stats in scheduler.trigger_scheduled_updates(stored_steps):
            if not update_stats:
                continue
            if not _validate_update_stats(update_stats, global_step):
                continue
            latest_update_stats = update_stats

    episode = 0

    def _episode_counters() -> dict:
        return {
            "ep_processed_mb": 0.0,
            "ep_processed_value": 0.0,
            "ep_downlink_mb": 0.0,
            "ep_delivered_value": 0.0,
            "ep_high_value_downlink_mb": 0.0,
            "ep_high_value_downlink_value": 0.0,
            "ep_low_value_dropped_mb": 0.0,
            "ep_active_low_dropped_mb": 0.0,
            "ep_passive_low_dropped_mb": 0.0,
            "ep_low_drop_recall_sum": 0.0,
            "ep_low_processing_ratio_sum": 0.0,
            "ep_low_delivery_ratio_sum": 0.0,
            "ep_low_value_dropped_value": 0.0,
            "ep_expired_value": 0.0,
            "ep_expired_processed_value": 0.0,
            "ep_expired_high_value": 0.0,
            "ep_cpu_active_far_from_window_steps": 0.0,
            "ep_processed_queue_future_contact_ratio_sum": 0.0,
        }

    def _reset_episode_counters(slot: dict) -> None:
        slot.update(_episode_counters())

    env_slots = []
    if env_backend == "subproc":
        assert env_pool is not None
        for state, context in env_pool.reset():
            # 子进程模式下，slot 只在主进程保存状态和上下文；真实环境状态留在 worker。
            env_slots.append({
                "env": None,
                "state": state,
                "context": context,
                "done": False,
                "ep_reward": 0.0,
                "ep_steps": 0,
                **_episode_counters(),
            })
    else:
        for env in envs:
            # 每个 slot 对应一个并行环境的局部状态缓存。
            env_slots.append({
                "env": env,
                "state": env.reset(),
                "context": _training_env_context(env),
                "done": False,
                "ep_reward": 0.0,
                "ep_steps": 0,
                **_episode_counters(),
            })

    def _process_transition(slot: dict, state: np.ndarray,
                            pre_context: dict, next_context: dict,
                            safe_action: np.ndarray, was_projected: bool,
                            raw_action: np.ndarray, psf_meta: dict,
                            next_state: np.ndarray, reward: float,
                            done: bool, info: dict,
                            stg: dict, lya_mult: float):
        """处理单条环境转移：计算 TD reward、写 replay、日志、评估和 episode 统计。"""
        nonlocal global_step, projected_ema, adaptive_constraint_ema
        nonlocal adaptive_constraint_value_used, adaptive_constraint_violation
        nonlocal adaptive_lya_coeff, lagrangian_lambda, episode, best_score
        nonlocal best_so_far_score, best_per_stage

        executed_action = np.asarray(
            info.get("executed_action", safe_action),
            dtype=np.float32,
        )

        qe = float(pre_context.get("qe", 0.0))
        qh = float(pre_context.get("qh", 0.0))
        qd = float(pre_context.get("qd", 0.0))
        qc = float(pre_context.get("qc", 0.0))
        qe2 = float(next_context.get("qe", qe))
        qh2 = float(next_context.get("qh", qh))
        qd2 = float(next_context.get("qd", qd))
        qc2 = float(next_context.get("qc", qc))

        safety_cost_info = dict(info)
        safety_cost_info["global_step"] = int(global_step)
        safety_cost = compute_lyapunov_safety_cost(
            previous_queues=(qe, qh, qd, qc),
            next_queues=(qe2, qh2, qd2, qc2),
            queue_maxes=(_qe, _qh, _qd, _qc),
            info=safety_cost_info,
        )
        lya_drift_raw = safety_cost.raw_cost
        lya_drift = safety_cost.training_cost
        lya_training_cost = safety_cost.training_cost
        lya_normalized_cost = safety_cost.normalized_cost
        lya_clip_saturation = safety_cost.training_cost_clip_saturation
        lya_soft_penalty = safety_cost.queue_soft_penalty
        lya_hard_penalty = safety_cost.queue_hard_penalty
        task_loss_penalty = safety_cost.task_loss_penalty
        constraint_queue_cost = safety_cost.queue_cost
        constraint_processed_backlog_cost = safety_cost.processed_backlog_cost
        constraint_window_waste_cost = safety_cost.window_waste_cost
        constraint_low_value_waste_cost = safety_cost.low_value_waste_cost
        constraint_over_processing_cost = safety_cost.over_processing_cost
        constraint_unproductive_cpu_cost = safety_cost.unproductive_cpu_cost
        constraint_energy_cost = safety_cost.energy_cost
        constraint_orbit_cost = safety_cost.orbit_cost
        constraint_thermal_cost = safety_cost.thermal_cost
        constraint_task_loss_cost = safety_cost.task_loss_cost
        constraint_efficiency_cost = safety_cost.efficiency_cost
        constraint_total_cost = safety_cost.total_cost
        adaptive_constraint_norm = safety_cost.dual_cost_norm
        adaptive_constraint_value = safety_cost.dual_violation_signal
        adaptive_constraint_threshold = float(DRL_CONFIG.get(
            "adaptive_lyapunov_constraint_threshold",
            DRL_CONFIG.get("adaptive_lyapunov_coeff_target_pressure", 0.02),
        ))

        safe_action_for_diff = _coerce_action_like(safe_action, executed_action)
        raw_action_for_diff = _coerce_action_like(raw_action, executed_action)
        execution_mod_l2 = float(np.linalg.norm(
            executed_action - safe_action_for_diff))
        total_execution_mod_l2 = float(np.linalg.norm(
            executed_action - raw_action_for_diff))
        psf_meta["environment_execution_mod_l2"] = execution_mod_l2
        psf_meta["total_modification_l2"] = max(
            float(psf_meta.get("total_modification_l2", 0.0)),
            total_execution_mod_l2,
        )
        mod = float(max(psf_meta.get("total_modification_l2", 0.0), 0.0))
        safety_projected = bool(psf_meta.get("safety_intervention_projected", was_projected))
        required_safety_correction = bool(
            safety_projected
            or psf_meta.get("psf_required_correction", False)
            or psf_meta.get("psf_no_safe_candidate", False)
            or psf_meta.get("boundary_clipped", False)
            or psf_meta.get("power_clipped", False)
            or psf_meta.get("thermal_clipped", False)
            or psf_meta.get("cpu_backpressure_required", False)
        )
        conservative_only_correction = bool(
            (not required_safety_correction)
            and (mod > 1e-8 or execution_mod_l2 > 1e-8)
        )
        bc_required_safety_weight = 1.0 if required_safety_correction else 0.0
        bc_conservative_weight = (
            float(DRL_CONFIG.get("behavior_cloning_conservative_weight_coeff", 0.25))
            * max(mod, execution_mod_l2)
        ) if conservative_only_correction else 0.0
        behavior_weight = max(bc_required_safety_weight, bc_conservative_weight)
        behavior_weight = float(np.clip(
            behavior_weight,
            0.0,
            float(DRL_CONFIG.get("behavior_cloning_max_weight", 1.0)),
        ))
        projection_penalty, action_mod_penalty, safety_action_penalty = (
            _compute_safety_action_penalties(mod, safety_projected, reward)
        )
        psf_penalty = safety_action_penalty

        ema_beta = float(np.clip(DRL_CONFIG.get("projection_ema_beta", 0.995), 0.0, 0.9999))
        projected_ema = ema_beta * projected_ema + (1.0 - ema_beta) * float(safety_projected)
        # 当前口径下课程权重只缩放约束 Critic 的 drift 样本；
        # 真正的自适应强度改为调整 Actor 中约束 Q 的全局权重。
        lya_mult_effective = float(lya_mult)
        lya_adaptive_scale = 1.0
        adaptive_lya_enabled = _adaptive_dual_enabled()
        adaptive_lya_coeff = float(scheduler.agent.get_lyapunov_penalty_coeff())
        adaptive_constraint_value_used = float(max(0.0, adaptive_constraint_value))
        if adaptive_lya_enabled:
            _queue_adaptive_constraint_signal(adaptive_constraint_value)
        adaptive_constraint_violation = (
            float(adaptive_constraint_ema) - float(adaptive_constraint_threshold)
        )

        lagrangian_cost = 0.0
        if constraint_variant == "lagrangian_sac":
            soc_now = float(info.get("soc", next_context.get("soc", 0.7)))
            h_km_now = float(info.get("altitude_km", next_context.get("h", 350e3) / 1e3))
            energy_threshold = float(ENERGY_CONFIG.get("battery_min_soc", 0.15))
            orbit_warning_km = float(ORBITAL_CONFIG.get(
                "altitude_warning_km", ORBITAL_CONFIG.get("altitude_min_km", 150.0)))
            orbit_cost = max(0.0, orbit_warning_km - h_km_now) / 5.0
            energy_cost = max(0.0, energy_threshold - soc_now) / 0.03
            lagrangian_cost = float(
                lya_soft_penalty + lya_hard_penalty + energy_cost + orbit_cost
            )
            lagrangian_lambda = float(np.clip(
                lagrangian_lambda + lagrangian_lr * (lagrangian_cost - lagrangian_target),
                0.0,
                lagrangian_max_lambda,
            ))

        terminated = bool(info.get("terminated", done))
        reward_scale = float(DRL_CONFIG.get("reward_scale", 100.0))
        lya_scale = float(DRL_CONFIG.get("lyapunov_drift_scale", 1.0))
        step_committed = False

        def _commit_env_step():
            nonlocal global_step, step_committed
            if step_committed:
                return
            slot["ep_reward"] += float(reward)
            slot["ep_steps"] += 1
            slot["ep_processed_mb"] += float(info.get("processed_mb", 0.0))
            slot["ep_processed_value"] += float(info.get("processed_value", 0.0))
            slot["ep_downlink_mb"] += float(
                info.get("delivered_mb", info.get("actual_tx_mb", 0.0)))
            slot["ep_delivered_value"] += float(info.get("delivered_value", 0.0))
            slot["ep_high_value_downlink_mb"] += float(
                info.get("high_value_downlink_mb", info.get("delivered_high_mb", 0.0)))
            slot["ep_high_value_downlink_value"] += float(
                info.get("high_value_downlink_value", info.get("delivered_high_value", 0.0)))
            slot["ep_low_value_dropped_mb"] += float(
                info.get("low_value_dropped_mb", info.get("dropped_low_mb", 0.0)))
            slot["ep_active_low_dropped_mb"] += float(
                info.get("active_dropped_low_mb", 0.0))
            slot["ep_passive_low_dropped_mb"] += float(
                info.get("passive_low_drop_mb", 0.0))
            slot["ep_low_drop_recall_sum"] += float(info.get("low_drop_recall", 0.0))
            slot["ep_low_processing_ratio_sum"] += float(info.get("low_processing_ratio", 0.0))
            slot["ep_low_delivery_ratio_sum"] += float(info.get("low_delivery_ratio", 0.0))
            slot["ep_low_value_dropped_value"] += float(
                info.get("low_value_dropped_value", 0.0))
            slot["ep_expired_value"] += float(info.get("expired_value", 0.0))
            slot["ep_expired_processed_value"] += float(
                info.get("expired_processed_value", 0.0))
            slot["ep_expired_high_value"] += float(info.get("expired_high_value", 0.0))
            slot["ep_cpu_active_far_from_window_steps"] += float(
                info.get("cpu_active_far_from_window_rate", 0.0))
            slot["ep_processed_queue_future_contact_ratio_sum"] += float(
                info.get("processed_queue_future_contact_ratio", 0.0))
            global_step += 1
            slot["state"] = next_state
            slot["context"] = next_context
            slot["done"] = bool(done)
            step_committed = True

        if constraint_variant == "lagrangian_sac" or not enable_lyapunov:
            lya_scaled = 0.0
        else:
            lya_scaled = (lya_drift / max(lya_scale, 1e-6)) * lya_mult_effective

        # ReplayBuffer 存的是最终执行动作 asafe。Reward Critic 的 TD reward 保持干净；
        # Lyapunov 漂移进入 SACAgent 内部的独立约束 Critic，安全执行动作由 BC 辅助学习。
        reward_for_td = reward
        if constraint_variant == "lagrangian_sac":
            reward_for_td -= lagrangian_lambda * lagrangian_cost
        reward_scaled = reward_for_td / max(reward_scale, 1e-6)
        deliverable_target_key = str(DRL_CONFIG.get("deliverable_critic_target_key", "processed_deliverable_value_step"))
        deliverable_reward = float((info.get("reward_breakdown", {}) or {}).get(deliverable_target_key, 0.0)) / max(reward_scale, 1e-6)
        if (not _is_finite_number(reward_scaled)) or (not _is_finite_number(lya_scaled)):
            _handle_nan_event(
                f"reward_scaled/lya_scaled 非有限值: reward_scaled={reward_scaled}, lya_scaled={lya_scaled}",
                global_step,
            )
            _commit_env_step()
            return 0

        # 【恢复原始逻辑】必须存 raw_action！
        # 如果存 executed_action，Replay Buffer 里永远没有"越界动作"，Q 网络会对越界动作产生错误的极高估计（Extrapolation Error）。
        # 这会导致 actor 疯狂输出越界动作（如 alpha_cpu=1.0），全靠底层 filter 兜底，最终学不到任何东西。
        # 必须让 Q 网络学到：如果我输出了 raw_action，环境（经过 filter 后）给我的真实回报是多少。
        # BC 辅助 (behavior_action) 仍用 executed_action：鼓励 actor 在安全方向上靠拢。
        scheduler.store_transition(
            state,
            raw_action,
            reward_scaled,
            next_state,
            done=done,
            lya_drift=lya_scaled,
            terminated=terminated,
            deliverable_reward=deliverable_reward,
            behavior_action=executed_action,
            behavior_weight=behavior_weight,
        )

        _commit_env_step()

        if global_step % int(TRAIN_CONFIG.get("log_freq", 500)) == 0:
            safety_stats = scheduler.get_safety_stats()
            reward_breakdown = info.get("reward_breakdown", {}) or {}
            psf_comm_pressure = psf_meta.get("psf_comm_pressure", None)
            delivered_high_value = float(info.get("delivered_high_value", 0.0))
            expired_high_value = float(info.get("expired_high_value", 0.0))
            dropped_high_value = float(info.get("dropped_high_value", 0.0))
            active_dropped_low_value = float(info.get("active_dropped_low_value", 0.0))
            active_dropped_total_value = float(info.get("active_dropped_total_value", 0.0))
            processed_mb_step = float(info.get("processed_mb", 0.0))
            processed_value_step = float(info.get("processed_value", 0.0))
            actual_tx_mb_step = float(info.get("actual_tx_mb", 0.0))
            delivered_mb_step = float(info.get("delivered_mb", actual_tx_mb_step))
            delivered_value_step = float(info.get("delivered_value", 0.0))
            episode_proc_dl_ratio = float(
                slot["ep_processed_mb"] / max(slot["ep_downlink_mb"], 1e-6))
            episode_useful_processing_ratio = float(
                slot["ep_delivered_value"] / max(slot["ep_processed_value"], 1e-6)
                if slot["ep_processed_value"] > 1e-9
                else 0.0
            )
            cpu_active_far_from_window_rate = float(
                slot["ep_cpu_active_far_from_window_steps"] / max(slot["ep_steps"], 1)
            )
            episode_processed_queue_future_contact_ratio = float(
                slot["ep_processed_queue_future_contact_ratio_sum"]
                / max(slot["ep_steps"], 1)
            )
            episode_low_drop_recall = float(
                slot["ep_low_drop_recall_sum"] / max(slot["ep_steps"], 1))
            episode_low_processing_ratio = float(
                slot["ep_low_processing_ratio_sum"] / max(slot["ep_steps"], 1))
            episode_low_delivery_ratio = float(
                slot["ep_low_delivery_ratio_sum"] / max(slot["ep_steps"], 1))
            step_useful_processing_ratio = float(
                delivered_value_step / max(processed_value_step, 1e-6)
                if processed_value_step > 1e-9
                else 0.0
            )
            capacity_mb_step = (
                float(info.get("tx_capacity_mbps", 0.0))
                * float(TRAIN_CONFIG.get("time_slot_s", 10))
                / 8.0
            )
            window_utilization_step = (
                delivered_mb_step / max(capacity_mb_step, 1e-6)
                if bool(info.get("in_window", False)) and capacity_mb_step > 1e-9
                else 0.0
            )
            tx_active_in_contact_step = float(
                bool(info.get("in_window", False)) and delivered_mb_step > 1e-9
            )

            high_value_den = delivered_high_value + expired_high_value + dropped_high_value
            high_value_downlink_rate = delivered_high_value / max(high_value_den, 1e-6)
            # 主动丢弃指标（is_droppable_batch 已强制仅 Low，因此 High/Medium 均应为 0）
            active_dropped_high_value = (
                float(info.get("active_dropped_raw_high_value", 0.0))
                + float(info.get("active_dropped_processed_high_value", 0.0))
            )
            active_dropped_medium_value = (
                float(info.get("active_dropped_raw_medium_value", 0.0))
                + float(info.get("active_dropped_processed_medium_value", 0.0))
            )
            # protected_violation_rate：主动丢弃中 High 的比例（应趋近 0）
            protected_violation_rate = active_dropped_high_value / max(active_dropped_total_value, 1e-6)
            # droppable_precision：丢弃中 Low+Medium 的比例（应趋近 1.0）
            droppable_precision = (active_dropped_low_value + active_dropped_medium_value) / max(active_dropped_total_value, 1e-6)
            # low_share：丢弃中纯 Low 的比例（强制 class_id==2 后应趋近 1.0）
            low_share = active_dropped_low_value / max(active_dropped_low_value + active_dropped_medium_value, 1e-6)
            # total_high_value_loss：所有高价值损失
            total_high_value_loss = expired_high_value + dropped_high_value
            low_drop_precision = droppable_precision
            active_low_discard_precision = active_dropped_low_value / max(active_dropped_total_value, 1e-6)
            cpu_tx_mismatch_rate = max(0.0, processed_mb_step - actual_tx_mb_step) / max(
                processed_mb_step, 1e-6
            )
            log_data = {
                "objective_version": OBJECTIVE_VERSION,
                "reward_schema": "mission_only",
                "reward_step": float(reward),
                "psf_penalty": float(psf_penalty),
                "projection_penalty": float(projection_penalty),
                "action_mod_penalty": float(action_mod_penalty),
                "safety_action_penalty": float(safety_action_penalty),
                "behavior_cloning_weight": float(behavior_weight),
                "bc_required_safety_weight": float(bc_required_safety_weight),
                "bc_conservative_weight": float(bc_conservative_weight),
                "bc_projection_required": float(1.0 if required_safety_correction else 0.0),
                "bc_conservative_only": float(1.0 if conservative_only_correction else 0.0),
                "reward_for_td": float(reward_for_td),
                "reward_td_excludes_safety_action_penalty": 1.0,
                "environment_execution_mod_l2": float(execution_mod_l2),
                "episode_reward": float(slot["ep_reward"]),
                "episode_steps": int(slot["ep_steps"]),
                "episode_processed_mb": float(slot["ep_processed_mb"]),
                "episode_processed_value": float(slot["ep_processed_value"]),
                "episode_downlink_mb": float(slot["ep_downlink_mb"]),
                "episode_delivered_value": float(slot["ep_delivered_value"]),
                "episode_proc_dl_ratio": episode_proc_dl_ratio,
                "episode_proc_downlink_ratio": episode_proc_dl_ratio,
                "episode_useful_processing_ratio": episode_useful_processing_ratio,
                "episode_high_value_downlink_mb": float(slot["ep_high_value_downlink_mb"]),
                "episode_high_value_downlink_value": float(slot["ep_high_value_downlink_value"]),
                "episode_low_value_dropped_mb": float(slot["ep_low_value_dropped_mb"]),
                "episode_active_low_dropped_mb": float(slot["ep_active_low_dropped_mb"]),
                "episode_active_low_drop_mb": float(slot["ep_active_low_dropped_mb"]),
                "episode_passive_low_drop_mb": float(slot["ep_passive_low_dropped_mb"]),
                "episode_low_drop_recall": episode_low_drop_recall,
                "episode_low_processing_ratio": episode_low_processing_ratio,
                "episode_low_delivery_ratio": episode_low_delivery_ratio,
                "episode_low_value_dropped_value": float(slot["ep_low_value_dropped_value"]),
                "episode_expired_value": float(slot["ep_expired_value"]),
                "episode_expired_processed_value": float(slot["ep_expired_processed_value"]),
                "episode_expired_high_value": float(slot["ep_expired_high_value"]),
                "cpu_active_far_from_window_rate": cpu_active_far_from_window_rate,
                "episode_cpu_active_far_from_window_rate": cpu_active_far_from_window_rate,
                "cpu_active_far_from_window_step": float(
                    info.get("cpu_active_far_from_window_rate", 0.0)),
                "episode_processed_queue_future_contact_ratio": episode_processed_queue_future_contact_ratio,
                "soc": float(info.get("soc", next_context.get("soc", 0.0))),
                "thermal_temperature_c": float(info.get("thermal_temperature_c", 0.0)),
                "thermal_margin_norm": float(info.get("thermal_margin_norm", 1.0)),
                "thermal_stage": str(info.get("thermal_stage", "normal")),
                "thermal_safe": float(info.get("thermal_safe", 1.0)),
                "thermal_throttle_applied": float(1.0 if (
                    psf_meta.get("thermal_clipped", False)
                    or info.get("thermal_throttle_applied", False)
                ) else 0.0),
                "altitude_km": float(info.get("altitude_km", next_context.get("h", 0.0) / 1e3)),
                "risk_stage": str(info.get("risk_stage", "normal")),
                "risk_stage_code": float(info.get("risk_stage_code", 0.0)),
                "orbit_stage": str(info.get("orbit_stage", "normal")),
                "energy_stage": str(info.get("energy_stage", "normal")),
                "warning_state": float(info.get("warning_state", 0.0)),
                "unsafe_state": float(info.get("unsafe_state", 0.0)),
                "failure_state": float(info.get("failure_state", 0.0)),
                "data_queue_mb": float(info.get("data_queue_mb", next_context.get("qd", 0.0))),
                "data_queue_util": float(info.get("data_queue_utilization", next_context.get("qd", 0.0) / max(_qd, 1e-6))),
                "overflow_mb": float(info.get("overflow_mb", 0.0)),
                "comm_queue_mb": float(info.get("comm_virtual_queue", next_context.get("qc", 0.0))),
                "comm_urgency": float(info.get("comm_urgency", 0.0)),
                "comm_urgency_raw": float(info.get("comm_urgency_raw", info.get("comm_urgency", 0.0))),
                "comm_overflow_mb": float(info.get("comm_overflow_mb", 0.0)),
                "service_rate": float(info.get("service_rate_mbs", 0.0)),
                "processed_mb": float(info.get("processed_mb", 0.0)),
                "processed_value": processed_value_step,
                "actual_tx_mb": float(info.get("actual_tx_mb", 0.0)),
                "delivered_mb": delivered_mb_step,
                "proc_dl_ratio": float(processed_mb_step / max(delivered_mb_step, 1e-6)),
                "window_utilization": float(window_utilization_step),
                "processed_queue_final_utilization": float(info.get("processed_queue_utilization", 0.0)),
                "future_contact_capacity_mb": float(
                    info.get("future_contact_capacity_mb", info.get("future_capacity_mb", 0.0))),
                "processed_queue_future_contact_ratio": float(
                    info.get("processed_queue_future_contact_ratio", 0.0)),
                "processed_queue_to_future_contact_ratio": float(
                    info.get("processed_queue_to_future_contact_ratio",
                             info.get("processed_queue_future_contact_ratio", 0.0))),
                "processed_since_contact_mb": float(info.get("processed_since_contact_mb", 0.0)),
                "delivered_since_contact_mb": float(info.get("delivered_since_contact_mb", 0.0)),
                "in_window": float(1.0 if bool(info.get("in_window", False)) else 0.0),
                "time_to_next_window_s": float(info.get("time_to_next_window_s", 0.0)),
                "tx_active_in_contact_ratio": tx_active_in_contact_step,
                "delivered_value": delivered_value_step,
                "step_useful_processing_ratio": step_useful_processing_ratio,
                "useful_processing_ratio": episode_useful_processing_ratio,
                "delivered_high_value": delivered_high_value,
                "average_aoi_steps": float(info.get("average_aoi_steps", 0.0)),
                "value_weighted_aoi_steps": float(info.get("value_weighted_aoi_steps", 0.0)),
                "voi_degradation_rate": float(info.get("voi_degradation_rate", info.get("expired_value_rate", 0.0))),
                "voi_loss_rate": float(info.get("voi_loss_rate", 0.0)),
                "deadline_success_rate": float(info.get("deadline_success_rate", 0.0)),
                "value_weighted_deadline_success_rate": float(
                    info.get("value_weighted_deadline_success_rate", info.get("deadline_success_rate", 0.0))
                ),
                "expired_value_rate": float(info.get("expired_value_rate", 0.0)),
                "dropped_value_rate": float(info.get("dropped_value_rate", 0.0)),
                "high_value_downlink_rate": float(high_value_downlink_rate),
                "high_value_delivery_ratio": float(high_value_downlink_rate),
                "low_value_discard_precision": float(low_drop_precision),
                "active_low_discard_precision": float(active_low_discard_precision),
                "total_high_value_loss": float(total_high_value_loss),
                "cpu_tx_mismatch_rate": float(cpu_tx_mismatch_rate),
                "emergency_event_active": float(info.get("emergency_event_active", 0.0)),
                "emergency_event_remaining_steps": float(info.get("emergency_event_remaining_steps", 0.0)),
                "emergency_event_triggered": float(info.get("emergency_event_triggered", 0.0)),
                "raw_queue_mb": float(info.get("raw_queue_mb", info.get("data_queue_mb", 0.0))),
                "processed_queue_mb": float(info.get("processed_queue_mb", info.get("comm_virtual_queue", 0.0))),
                "raw_high_mb": float(info.get("raw_high_mb", 0.0)),
                "raw_mid_mb": float(info.get("raw_mid_mb", 0.0)),
                "raw_low_mb": float(info.get("raw_low_mb", 0.0)),
                "processed_high_mb": float(info.get("processed_high_mb", 0.0)),
                "processed_mid_mb": float(info.get("processed_mid_mb", 0.0)),
                "processed_low_mb": float(info.get("processed_low_mb", 0.0)),
                "processed_high_mb_step": float(info.get("processed_high_mb_step", 0.0)),
                "processed_mid_mb_step": float(info.get("processed_mid_mb_step", 0.0)),
                "processed_low_mb_step": float(info.get("processed_low_mb_step", 0.0)),
                "raw_queue_overflow_mb": float(info.get("raw_queue_overflow_mb", info.get("overflow_mb", 0.0))),
                "processed_queue_overflow_mb": float(info.get("processed_queue_overflow_mb", info.get("comm_overflow_mb", 0.0))),
                "expired_processed_value": float(info.get("expired_processed_value", 0.0)),
                "expired_raw_value": float(info.get("expired_raw_value", 0.0)),
                "expired_high_value": float(info.get("expired_high_value", 0.0)),
                "dropped_raw_value": float(info.get("dropped_raw_value", 0.0)),
                "dropped_processed_value": float(info.get("dropped_processed_value", 0.0)),
                "dropped_high_value": float(info.get("dropped_high_value", 0.0)),
                "active_dropped_high_value": float(active_dropped_high_value),
                "active_dropped_medium_value": float(active_dropped_medium_value),
                "active_dropped_low_value": float(info.get("active_dropped_low_value", 0.0)),
                "active_low_drop_mb": float(info.get("active_low_drop_mb", info.get("active_dropped_low_mb", 0.0))),
                "passive_low_drop_mb": float(info.get("passive_low_drop_mb", 0.0)),
                "low_drop_recall": float(info.get("low_drop_recall", 0.0)),
                "low_processing_ratio": float(info.get("low_processing_ratio", 0.0)),
                "low_delivery_ratio": float(info.get("low_delivery_ratio", 0.0)),
                "active_dropped_low_raw_mb": float(info.get("active_dropped_low_raw_mb", 0.0)),
                "active_dropped_low_processed_mb": float(info.get("active_dropped_low_processed_mb", 0.0)),
                "active_dropped_low_processed_value": float(
                    info.get("active_dropped_processed_low_value", 0.0)),
                "active_dropped_total_value": float(active_dropped_total_value),
                "protected_violation_rate": float(protected_violation_rate),
                "droppable_precision": float(droppable_precision),
                "low_share": float(low_share),
                "capacity_driven_drop_mb": float(info.get("capacity_driven_drop_mb", 0.0)),
                "cpu_ratio_requested_high": float(info.get("cpu_requested_high", 0.0)),
                "cpu_ratio_requested_mid": float(info.get("cpu_requested_mid", 0.0)),
                "cpu_ratio_requested_low": float(info.get("cpu_requested_low", 0.0)),
                "cpu_ratio_executed_high": float(info.get("cpu_executed_share_high", 0.0)),
                "cpu_ratio_executed_mid": float(info.get("cpu_executed_share_mid", 0.0)),
                "cpu_ratio_executed_low": float(info.get("cpu_executed_share_low", 0.0)),
                "tx_ratio_requested_high": float(info.get("tx_requested_high", 0.0)),
                "tx_ratio_requested_mid": float(info.get("tx_requested_mid", 0.0)),
                "tx_ratio_requested_low": float(info.get("tx_requested_low", 0.0)),
                "tx_ratio_executed_high": float(info.get("tx_executed_share_high", 0.0)),
                "tx_ratio_executed_mid": float(info.get("tx_executed_share_mid", 0.0)),
                "tx_ratio_executed_low": float(info.get("tx_executed_share_low", 0.0)),
                "cpu_reallocation_rate": float(info.get("cpu_reallocation_rate", 0.0)),
                "tx_reallocation_rate": float(info.get("tx_reallocation_rate", 0.0)),
                "drop_low_strength": float(info.get("drop_low_strength", 0.0)),
                "cpu_throttle_applied": float(info.get("cpu_throttle_applied", 0.0)),
                "cpu_throttle_proc_util": float(info.get("cpu_throttle_proc_util", 0.0)),
                "future_contact_cpu_gate_applied": float(
                    1.0 if info.get("future_contact_cpu_gate_applied", False) else 0.0),
                "cpu_gate_ratio_before": float(info.get("cpu_gate_ratio_before", 0.0)),
                "cpu_gate_ratio_after_est": float(info.get("cpu_gate_ratio_after_est", 0.0)),
                "cpu_gate_requested_processed_mb": float(
                    info.get("cpu_gate_requested_processed_mb", 0.0)),
                "cpu_gate_allowed_processed_mb": float(
                    info.get("cpu_gate_allowed_processed_mb", 0.0)),
                "cpu_gate_alpha_cpu_before": float(
                    info.get("cpu_gate_alpha_cpu_before", 0.0)),
                "cpu_gate_alpha_cpu_after": float(
                    info.get("cpu_gate_alpha_cpu_after", 0.0)),
                "cpu_gate_mod_l2": float(info.get("cpu_gate_mod_l2", 0.0)),
                "overall_safe": float(info.get("overall_safe", 1.0)),
                "was_projected": int(was_projected),
                "safety_intervention_projected": int(safety_projected),
                "prop_smoothing_applied": float(1.0 if psf_meta.get("prop_smoothing_applied", False) else 0.0),
                "action_mod_l2": float(mod),
                "boundary_clipped": float(1.0 if psf_meta.get("boundary_clipped", False) else 0.0),
                "power_clipped": float(1.0 if psf_meta.get("power_clipped", False) else 0.0),
                "cpu_backpressure_applied": float(1.0 if (
                    psf_meta.get("cpu_backpressure_applied", False)
                    or info.get("cpu_backpressure_applied", False)
                ) else 0.0),
                "cpu_backpressure_required": float(1.0 if (
                    psf_meta.get("cpu_backpressure_required", False)
                    or info.get("cpu_backpressure_required", False)
                ) else 0.0),
                "required_cpu_backpressure_ratio": float(psf_meta.get(
                    "required_cpu_backpressure_ratio",
                    info.get("required_cpu_backpressure_ratio", 0.0),
                )),
                "processed_queue_boundary_violation": float(psf_meta.get(
                    "processed_queue_boundary_violation",
                    info.get("processed_queue_boundary_violation", 0.0),
                )),
                "cpu_backpressure_mod_l2": float(psf_meta.get(
                    "cpu_backpressure_mod_l2",
                    info.get("cpu_backpressure_mod_l2", 0.0),
                )),
                "requested_processed_mb": float(psf_meta.get(
                    "requested_processed_mb",
                    info.get("requested_processed_mb", 0.0),
                )),
                "allowed_processed_mb": float(psf_meta.get(
                    "allowed_processed_mb",
                    info.get("allowed_processed_mb", 0.0),
                )),
                "processed_queue_headroom_mb": float(psf_meta.get(
                    "processed_queue_headroom_mb",
                    info.get("processed_queue_headroom_mb", 0.0),
                )),
                "processed_queue_tx_room_mb": float(psf_meta.get(
                    "processed_queue_tx_room_mb",
                    info.get("processed_queue_tx_room_mb", 0.0),
                )),
                "available_power_w": float(psf_meta.get("available_power_w", info.get("available_power_w", 0.0))),
                "adjustable_power_budget_w": float(psf_meta.get("adjustable_power_budget_w", info.get("adjustable_power_budget_w", 0.0))),
                "psf_comm_urgency": float(
                    0.0 if psf_meta.get("psf_comm_urgency", None) is None
                    else psf_meta.get("psf_comm_urgency", 0.0)
                ),
                "psf_comm_pressure_applicable": float(psf_comm_pressure is not None),
                "psf_required_correction": float(
                    1.0 if psf_meta.get("psf_required_correction", False) else 0.0),
                "psf_no_safe_candidate": float(
                    1.0 if psf_meta.get("psf_no_safe_candidate", False) else 0.0),
                "psf_used_emergency": float(
                    1.0 if psf_meta.get("psf_used_emergency", False) else 0.0),
                "psf_candidate_count": float(psf_meta.get("psf_candidate_count", 0.0)),
                "psf_safe_candidate_count": float(psf_meta.get("psf_safe_candidate_count", 0.0)),
                "psf_formal_guarantee": float(
                    1.0 if psf_meta.get("psf_formal_guarantee", False) else 0.0),
                "lya_drift": float(lya_drift),
                "safety_cost_raw": float(lya_drift_raw),
                "safety_cost_training": float(lya_training_cost),
                "safety_cost_normalized": float(lya_normalized_cost),
                "safety_cost_clip_saturation": float(lya_clip_saturation),
                "safety_cost_constraint": float(lya_drift),
                "lya_soft_penalty": float(lya_soft_penalty),
                "lya_hard_penalty": float(lya_hard_penalty),
                "task_loss_penalty": float(task_loss_penalty),
                "constraint_queue_cost": float(constraint_queue_cost),
                "constraint_processed_backlog_cost": float(constraint_processed_backlog_cost),
                "constraint_window_waste_cost": float(constraint_window_waste_cost),
                "constraint_low_value_waste_cost": float(constraint_low_value_waste_cost),
                "constraint_over_processing_cost": float(constraint_over_processing_cost),
                "constraint_unproductive_cpu_cost": float(constraint_unproductive_cpu_cost),
                "constraint_energy_cost": float(constraint_energy_cost),
                "constraint_orbit_cost": float(constraint_orbit_cost),
                "constraint_thermal_cost": float(constraint_thermal_cost),
                "constraint_task_loss_cost": float(constraint_task_loss_cost),
                "constraint_efficiency_cost": float(constraint_efficiency_cost),
                "constraint_total_cost": float(constraint_total_cost),
                "constraint_dual_cost": float(safety_cost.dual_cost),
                "constraint_training_cost_clip": float(safety_cost.training_cost_clip),
                "constraint_over_processing_raw_cost": float(safety_cost.over_processing_raw_cost),
                "constraint_over_processing_normalized_cost": float(safety_cost.over_processing_normalized_cost),
                "constraint_over_processing_training_cost": float(safety_cost.over_processing_training_cost),
                "constraint_over_processing_clip_saturation": float(safety_cost.over_processing_clip_saturation),
                "constraint_backlog_excess_mb": float(safety_cost.backlog_excess_mb),
                "constraint_admission_excess_mb": float(safety_cost.admission_excess_mb),
                "constraint_clearable_capacity_mb": float(safety_cost.clearable_capacity_mb),
                "constraint_over_processing_ratio": float(safety_cost.over_processing_ratio),
                "lya_mult": float(lya_mult),
                "lya_mult_effective": float(lya_mult_effective),
                "lya_adaptive_scale": float(lya_adaptive_scale),
                "adaptive_dual_source": "constraint_cost",
                "adaptive_constraint_cost_norm": float(adaptive_constraint_norm),
                "adaptive_constraint_value": float(adaptive_constraint_value_used),
                "adaptive_constraint_ema": float(adaptive_constraint_ema),
                "adaptive_constraint_threshold": float(adaptive_constraint_threshold),
                "adaptive_constraint_violation": float(adaptive_constraint_violation),
                "adaptive_lyapunov_coeff": float(adaptive_lya_coeff),
                "adaptive_lyapunov_pressure": float(adaptive_constraint_value_used),
                "adaptive_lyapunov_pressure_ema": float(adaptive_constraint_ema),
                "constraint_variant": constraint_variant,
                "lagrangian_cost": float(lagrangian_cost),
                "lagrangian_lambda": float(lagrangian_lambda),
                "projected_ema": float(projected_ema),
                "stage": stg.get("stage_name", "Full"),
                **latest_update_stats,
            }
            if psf_comm_pressure is not None:
                log_data["psf_comm_pressure"] = float(
                    1.0 if psf_comm_pressure else 0.0
                )
            for k in (
                "r_delivered_value", "r_deadline_success",
                "r_deliverable_processing", "deliverable_processing_credit_value",
                "_eq_drift", "_oq_drift", "_thermal_excess_c"
            ):
                if k in reward_breakdown:
                    log_data[k] = float(reward_breakdown[k])
            for k, v in safety_stats.items():
                if isinstance(v, (int, float)):
                    log_data[k] = float(v)
            logger.log_step(global_step, log_data)

        if global_step % eval_freq == 0:
            eval_stats = evaluate(
                eval_env, scheduler, n_episodes=eval_episodes,
                data_scale=data_scale,
            )
            logger.log_eval(global_step, eval_stats)
            scheduler.save(latest_path)
            current_score = _selection_tuple(eval_stats)
            stage_name_eval = str(stg.get("stage_name", "Full"))
            # 1) Optimization stage 的官方 best（论文/对外口径）
            if data_scale >= 1.0 - 1e-9 and current_score > best_score:
                best_score = current_score
                scheduler.save(best_path)
            # 2) 任意难度下的全局最佳（避免训练崩了把好策略弄丢）
            if current_score > best_so_far_score:
                best_so_far_score = current_score
                scheduler.save(best_so_far_path)
                print(f"  [best_so_far] saved at step {global_step}, ds={data_scale:.2f}, stage={stage_name_eval}")
            # 3) 每个 stage 的局部最佳
            stage_best = best_per_stage.get(stage_name_eval, (-np.inf,) * 10)
            if current_score > stage_best:
                best_per_stage[stage_name_eval] = current_score
                stage_ckpt = os.path.join(checkpoint_dir, f"best_stage_{stage_name_eval}.pt")
                scheduler.save(stage_ckpt)

            print(
                f"  [step {global_step:>8,}/{total_steps:,}] "
                f"r_step={eval_stats.get('reward_per_step_mean', 0.0):.2f} "
                f"R={eval_stats['reward_mean']:.1f}±{eval_stats['reward_std']:.1f} "
                f"stage={stg.get('stage_name', 'Full')} "
                f"scale={eval_stats.get('eval_data_arrival_scale', 1.0):.2f} "
                f"val={eval_stats.get('delivered_value_mean', 0.0):.1f} "
                f"proc={eval_stats['processed_mean']:.1f}MB "
                f"dl={eval_stats['downlink_mean']:.1f}MB "
                f"proc/dl={eval_stats.get('proc_downlink_ratio', 0.0):.2f} "
                f"hi_dl={eval_stats.get('high_value_downlink_mb_mean', 0.0):.1f}MB "
                f"low_drop={eval_stats.get('active_low_drop_mb_mean', eval_stats.get('active_low_dropped_mb_mean', 0.0)):.1f}MB "
                f"active_low={eval_stats.get('active_low_dropped_mb_mean', 0.0):.1f}MB "
                f"ovf={eval_stats.get('processed_overflow_mean', 0.0):.1f}MB "
                f"sel={eval_stats.get('checkpoint_value_score', 0.0):.1f} "
                    f"safe={eval_stats.get('overall_safe_rate', eval_stats['safety_rate']):.1%} "
                    f"ep_safe={eval_stats.get('episode_safety_rate', eval_stats['safety_rate']):.1%} "
                    f"warn={eval_stats.get('warning_state_rate', 0.0):.1%} "
                    f"unsafe={eval_stats.get('unsafe_state_rate', 0.0):.1%} "
                    f"lya_proj={eval_stats.get('lyapunov_proj_rate', 0.0):.1%} "
                    f"psf_phys={eval_stats.get('psf_filter_rate', 0.0):.1%} "
                    f"chain_safe={eval_stats.get('safety_intervention_rate', eval_stats.get('intervention_rate', 0.0)):.1%} "
                    f"prop_lock={eval_stats.get('prop_smoothing_rate', 0.0):.1%} "
                    f"bound={eval_stats.get('boundary_clip_rate_eval', eval_stats.get('boundary_clip_rate', 0.0)):.1%} "
                    f"mod={eval_stats.get('action_mod_l2_mean', 0.0):.3f} "
                    f"chain_all={eval_stats.get('chain_total_rate', eval_stats.get('was_projected_rate', 0.0)):.1%}"
                )

        if keep_step_checkpoints and global_step % save_freq == 0:
            scheduler.save(os.path.join(checkpoint_dir, f"step_{global_step}.pt"))

        if slot["done"]:
            episode += 1
            if episode % 20 == 0:
                print(
                    f"  [ep {episode:>4}] step={global_step:>8,}/{total_steps:,} "
                    f"ep_reward={slot['ep_reward']:>9.1f} "
                    f"proc={slot['ep_processed_mb']:.1f}MB "
                    f"dl={slot['ep_downlink_mb']:.1f}MB "
                    f"proc/dl={slot['ep_processed_mb'] / max(slot['ep_downlink_mb'], 1e-6):.2f} "
                    f"useful={slot['ep_delivered_value'] / max(slot['ep_processed_value'], 1e-6) if slot['ep_processed_value'] > 1e-9 else 0.0:.2f} "
                    f"far_cpu={slot['ep_cpu_active_far_from_window_steps'] / max(slot['ep_steps'], 1):.1%} "
                    f"hi_dl={slot['ep_high_value_downlink_mb']:.1f}MB "
                    f"low_drop={slot['ep_active_low_dropped_mb']:.1f}MB "
                    f"active_low={slot['ep_active_low_dropped_mb']:.1f}MB "
                    f"exp_high_val={slot['ep_expired_high_value']:.1f}"
                )
        return 1

    def _active_slot_indices() -> list[int]:
        active_indices = []
        for i in range(len(env_slots)):
            if global_step + len(active_indices) >= total_steps:
                break
            active_indices.append(i)
        return active_indices

    def _reset_serial_done_slots() -> None:
        for slot in env_slots:
            if not slot["done"]:
                continue
            env = slot["env"]
            slot["state"] = env.reset()
            slot["context"] = _training_env_context(env)
            slot["done"] = False
            slot["ep_reward"] = 0.0
            slot["ep_steps"] = 0
            _reset_episode_counters(slot)

    def _process_step_results(active_indices: list[int],
                              batch_contexts: list[dict],
                              schedule_outputs: list[tuple],
                              step_results: list[tuple],
                              stg: dict, lya_mult: float) -> int:
        stored_steps = 0
        for idx, pre_context, schedule_output, step_result in zip(
            active_indices, batch_contexts, schedule_outputs, step_results
        ):
            safe_action, was_projected, raw_action, psf_meta = schedule_output
            next_state, reward, done, info, next_context = step_result
            stored_steps += _process_transition(
                env_slots[idx],
                env_slots[idx]["state"],
                pre_context,
                next_context,
                safe_action,
                was_projected,
                raw_action,
                psf_meta,
                next_state,
                reward,
                done,
                info,
                stg,
                lya_mult,
            ) or 0
        return int(stored_steps)

    pending_update_steps = 0

    try:
        while global_step < total_steps:
            # 课程学习每一阶段会同时调几类量：
            # 1. lya_mult：约束 Critic 里的 Lyapunov 漂移样本权重
            # 2. data_scale：环境任务到达强度
            # 3. rand_scale：domain randomization 幅度 (rho/β/storm) — 防止训练初期分布太宽
            # Actor 里的约束 Q 全局权重由 adaptive_lyapunov_coeff 独立调节。
            stg = _get_stage(stages, global_step)
            _apply_stage_psf_policy(scheduler, str(stg.get("stage_name", "")))
            lya_mult = float(stg.get("lyapunov_weight_scale", 1.0))
            data_scale = float(stg.get("data_arrival_scale", 1.0))
            rand_scale = float(stg.get("randomization_scale", 1.0))
            if env_backend == "subproc":
                assert env_pool is not None
                # 多 ds 混合训练时不强制统一 data_scale，让每个 env 在 reset() 里各自随机抽样。
                if not bool(TRAIN_CONFIG.get("train_random_ds_enabled", False)):
                    env_pool.set_data_scale(data_scale)
                if hasattr(env_pool, "set_randomization_scale"):
                    env_pool.set_randomization_scale(rand_scale)

                done_indices = [i for i, slot in enumerate(env_slots) if slot["done"]]
                if done_indices:
                    reset_results = env_pool.reset(done_indices)
                    for idx, (state, context) in zip(done_indices, reset_results):
                        env_slots[idx]["state"] = state
                        env_slots[idx]["context"] = context
                        env_slots[idx]["done"] = False
                        env_slots[idx]["ep_reward"] = 0.0
                        env_slots[idx]["ep_steps"] = 0
                        _reset_episode_counters(env_slots[idx])

                active_indices = _active_slot_indices()
                if not active_indices:
                    continue

                batch_states = np.stack(
                    [env_slots[i]["state"] for i in active_indices],
                    axis=0,
                )
                batch_contexts = [env_slots[i]["context"] for i in active_indices]
                schedule_outputs = scheduler.schedule_batch(
                    batch_states,
                    batch_contexts,
                    evaluate=False,
                )
                env_pool.step_async(
                    active_indices,
                    [out[0] for out in schedule_outputs],
                )
                # 让 worker 计算下一批环境物理仿真时，主进程同时消化上一批网络更新。
                if pending_update_steps > 0:
                    _run_scheduled_updates(pending_update_steps)
                    pending_update_steps = 0
                step_results = env_pool.recv_step()
                stored_steps = _process_step_results(
                    active_indices, batch_contexts, schedule_outputs, step_results,
                    stg, lya_mult,
                )
                pending_update_steps += stored_steps
                continue

            for base_env in base_envs:
                base_env._data_arrival_scale = data_scale
                base_env._randomization_scale = rand_scale

            _reset_serial_done_slots()
            active_indices = _active_slot_indices()
            if not active_indices:
                continue

            batch_states = np.stack(
                [env_slots[i]["state"] for i in active_indices],
                axis=0,
            )
            batch_contexts = []
            for idx in active_indices:
                env = env_slots[idx]["env"]
                pre_context = _training_env_context(env)
                # 串行模式能直接传 env，PSF 可同步当前扰动物理参数。
                pre_context["env"] = env
                env_slots[idx]["context"] = pre_context
                batch_contexts.append(pre_context)

            schedule_outputs = scheduler.schedule_batch(
                batch_states,
                batch_contexts,
                evaluate=False,
            )
            step_results = []
            for idx, schedule_output in zip(active_indices, schedule_outputs):
                env = env_slots[idx]["env"]
                next_state, reward, done, info = env.step(
                    schedule_output[0], enforce_prop_smoothing=False)
                step_results.append((
                    next_state,
                    reward,
                    done,
                    info,
                    _training_env_context(env),
                ))
            stored_steps = _process_step_results(
                active_indices, batch_contexts, schedule_outputs, step_results,
                stg, lya_mult,
            )
            if stored_steps > 0:
                _run_scheduled_updates(stored_steps)

        if pending_update_steps > 0:
            _run_scheduled_updates(pending_update_steps)
            pending_update_steps = 0
    except KeyboardInterrupt:
        if pending_update_steps > 0:
            try:
                _run_scheduled_updates(pending_update_steps)
            except Exception as exc:
                print(f"\n[暂停] 保存前执行待处理更新失败，已跳过: {exc}")
            pending_update_steps = 0
        scheduler.save(latest_path)
        logger.save()
        if env_pool is not None:
            env_pool.close()
        print("\n[暂停] 已保存 latest checkpoint，下次运行默认会从该检查点继续训练。")
        print(f"  latest: {latest_path}")
        return

    scheduler.save(latest_path)
    if (not os.path.exists(best_path)) or (not _checkpoint_matches_current_objective(best_path)):
        # 旧 reward 口径的 best 不能继续作为当前实验的“最优模型”。
        scheduler.save(best_path)

    # 训练结束后再做一次完整评估，并把结果写入最终报告。
    final_eval = evaluate(eval_env, scheduler, n_episodes=eval_episodes)
    logger.log_eval(global_step, final_eval)
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_steps": total_steps,
        "device": device,
        "seed": seed,
        "env_backend": env_backend,
        "n_envs": n_envs,
        "constraint_variant": constraint_variant,
        "network_arch": str(DRL_CONFIG.get("network_arch", "transformer")),
        "behavior_cloning_coeff": float(DRL_CONFIG.get("behavior_cloning_coeff", 0.0)),
        "mission_reward_variant": mission_reward_variant,
        "enable_lyapunov": bool(scheduler.enable_lyapunov),
        "use_psf": bool(scheduler.use_psf),
        "objective_summary": _objective_summary(),
        "experiment_protocol": EXPERIMENT_PROTOCOL,
        "best_model_selection": (
            "feasible models first, then maximize safety-adjusted delivered task value "
            "(delivered value minus Lyapunov projection/action-modification penalties), "
            "then raw delivered value, downlink and reward"
        ),
        "final_eval": final_eval,
        "best_checkpoint": best_path,
        "latest_checkpoint": latest_path,
    }
    report_path = os.path.join(log_dir, "train_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.save()
    if env_pool is not None:
        env_pool.close()

    print("\n" + "=" * 70)
    print("  训练完成")
    print("=" * 70)
    print(f"  best:   {best_path}")
    print(f"  latest: {latest_path}")
    print(f"  report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LS-PSF CMDP 训练入口")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--total_steps", type=int, default=int(TRAIN_CONFIG.get("total_steps", 1500000)))
    parser.add_argument("--checkpoint_dir",
                        default=TRAIN_CONFIG.get("optimized_checkpoint_dir", "checkpoints_optimized/"))
    parser.add_argument("--resume_path", default=None, help="显式指定续训检查点路径")
    parser.add_argument("--log_dir",
                        default=TRAIN_CONFIG.get("optimized_log_dir", "logs_optimized/"))
    parser.add_argument("--tensorboard", action="store_true",
                        help="同时把统一指标写入 TensorBoard")
    parser.add_argument("--wandb_project", default=None,
                        help="设置后将统一指标同步到 WandB project")
    parser.add_argument("--wandb_run_name", default=None,
                        help="WandB run name")
    parser.add_argument("--eval_freq", type=int, default=int(TRAIN_CONFIG.get("eval_freq", 5000)))
    parser.add_argument("--eval_episodes", type=int, default=int(TRAIN_CONFIG.get("eval_episodes", 30)))
    parser.add_argument("--save_freq", type=int, default=int(TRAIN_CONFIG.get("save_freq", 50000)))
    parser.add_argument("--keep_step_checkpoints",
                        action="store_true",
                        default=bool(TRAIN_CONFIG.get("keep_step_checkpoints", False)),
                        help="开启后按 save_freq 额外保存 step_*.pt；默认只保存 best/latest")
    parser.add_argument("--n_envs", type=int, default=int(TRAIN_CONFIG.get("n_envs", 1)))
    parser.add_argument("--env_backend",
                        choices=["auto", "serial", "subproc"],
                        default=TRAIN_CONFIG.get("env_backend", "auto"),
                        help="训练环境后端：auto在n_envs>1时使用子进程并行，serial用于调试")
    parser.add_argument("--seed", type=int, default=int(TRAIN_CONFIG.get("seed", 42)),
                        help="训练随机种子；严格多 seed 训练会为每个模型传入不同 seed")
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--update_freq", type=int, default=None)
    parser.add_argument("--update_actor_freq", type=int, default=None)
    parser.add_argument("--network_arch", choices=["transformer", "mlp"], default=None,
                        help="网络 backbone；mlp 仅用于论文消融")
    parser.add_argument("--behavior_cloning_coeff", type=float, default=None,
                        help="安全投影动作模仿损失系数；设为0可做 w/o BC 消融")
    parser.add_argument("--no_lyapunov", action="store_true")
    parser.add_argument("--no_psf", action="store_true")
    parser.add_argument("--use_inference_mpc", action="store_true",
                        help="启用推理时 short-horizon MPC（包裹 actor，按 critic 终端价值打分）")
    parser.add_argument("--inference_mpc_warmup_steps", type=int, default=None,
                        help="MPC 启用的步数门槛；到此之前 schedule() 不走 MPC（评估模式无此限制）")
    parser.add_argument("--constraint_variant",
                        choices=[
                            "ours", "plain_sac", "lagrangian_sac",
                            "sac_psf", "sac_lyapunov",
                        ],
                        default="ours",
                        help=("训练约束变体：ours=LS-PSF CMDP，plain_sac=无安全层，"
                              "lagrangian_sac=SAC-Lagrangian 基线"))
    parser.add_argument("--lagrangian_lr", type=float, default=0.02)
    parser.add_argument("--lagrangian_target", type=float, default=0.03)
    parser.add_argument("--lagrangian_max_lambda", type=float, default=50.0)
    parser.add_argument("--disable_adaptive_dual", action="store_true",
                        help="消融用：固定 Lyapunov actor coeff，不按 CMDP constraint violation 自适应更新")
    parser.add_argument("--mission_reward_variant",
                        choices=["value_aware", "throughput"],
                        default="value_aware",
                        help="训练 reward 变体；throughput 仅用于 H 消融")
    parser.add_argument("--throughput_reward_weight", type=float, default=1.0,
                        help="throughput reward 变体中的 delivered MB 权重")
    cli_args = parser.parse_args()
    cli_args.adaptive_lyapunov_coeff_enable = (
        False if cli_args.disable_adaptive_dual else None
    )
    train(cli_args)

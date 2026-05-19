"""
Simplified Scheduler for pure RL training.

This removes the bloated Predictive Safety Filter (PSF) and Lyapunov projection,
letting the agent actually learn the constraints and capabilities directly from
the environment and reward signals.
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

import numpy as np
from config import (
    DRL_CONFIG,
    ENERGY_CONFIG,
    ORBITAL_CONFIG,
    REWARD_CONFIG,
    THERMAL_CONFIG,
    TRAIN_CONFIG,
    OBJECTIVE_VERSION,
)
from environment.satellite_env import OBSERVATION_FEATURES
from algorithms.decoupled_constraint_sac import DecoupledConstraintSAC
from safety.actuator_constraints import ActuatorConstraintFilter


class IntegratedScheduler:
    """
    Simplified pure RL scheduler.
    
    1. πθ(s)：SAC 策略网络输出原始连续动作
    2. Πfeas：仅满足基础的动作盒约束和瞬时功率边界（防物理越界报错）
    
    去除了所有多余的预测安全过滤（PSF）和李雅普诺夫约束，
    将学习权完全交还给智能体本身。
    """

    def __init__(self, device: str = "auto", enable_lyapunov: bool = True, use_psf: bool = False, **kwargs):
        state_dim  = DRL_CONFIG.get("state_dim", 30)
        action_dim = DRL_CONFIG["action_dim"]

        # 仍然使用现有的 agent 类，但约束和额外的安全层将被旁路
        self.agent = DecoupledConstraintSAC(state_dim, action_dim, device)
        self.enable_lyapunov = bool(enable_lyapunov)
        self.use_psf = False
        self.psf = None
        self.psf_K = 5
        if not self.enable_lyapunov:
            self.agent.set_lyapunov_penalty_coeff(0.0)

        self._boundary_total_count = 0
        self._boundary_clip_count = 0

        self._power_weights = np.array([
            ENERGY_CONFIG["power_propulsion_max_w"],
            ENERGY_CONFIG["power_cpu_max_w"],
            ENERGY_CONFIG["power_tx_max_w"],
        ], dtype=np.float64)
        self._power_baseline_w = float(ENERGY_CONFIG["power_baseline_w"])
        self._power_total_max_w = float(ENERGY_CONFIG.get("power_total_max_w", 120.0))
        self._prop_ignition_threshold_w = float(
            ENERGY_CONFIG.get("propulsion_ignition_threshold_w", 0.0))
            
        self.actuator_filter = ActuatorConstraintFilter(
            power_weights=self._power_weights,
            baseline_w=self._power_baseline_w,
            total_limit_w=self._power_total_max_w,
            prop_ignition_threshold_w=self._prop_ignition_threshold_w,
            action_dim=action_dim,
        )

    def schedule(self, state: np.ndarray, *args, evaluate: bool = False, **kwargs) -> tuple:
        if args and "in_window" not in kwargs:
            kwargs["in_window"] = bool(args[-1])
        raw_action = self.agent.select_action(state, evaluate=evaluate)
        return self._schedule_from_raw_action(raw_action, state, **kwargs)

    def schedule_batch(self, states: np.ndarray, contexts: list[dict], evaluate: bool = False) -> list[tuple]:
        states_arr = np.asarray(states, dtype=np.float32)
        raw_actions = self.agent.select_actions(states_arr, evaluate=evaluate)
        outputs = []
        for state, raw_action, ctx in zip(states_arr, raw_actions, contexts):
            outputs.append(self._schedule_from_raw_action(raw_action, state, **ctx))
        return outputs

    def _schedule_from_raw_action(self, raw_action: np.ndarray, state: np.ndarray, **kwargs) -> tuple:
        # 只做最基础的物理可行性保证（例如功率分配不能超过上限，不然环境就崩溃了）
        feasible_action, feasibility_meta = self._apply_physical_feasibility_projection(
            raw_action,
            state,
            in_window=kwargs.get("in_window", False),
            h=kwargs.get("h", 350e3),
            prop_can_update=kwargs.get("prop_can_update", True),
            available_power_w=kwargs.get("available_power_w", None),
        )

        safe_action = feasible_action
        
        # 简单记录一下
        was_projected = bool(feasibility_meta.get("boundary_clipped", False))
        
        psf_meta = {}
        psf_meta.update(feasibility_meta)
        psf_meta["safety_chain_projected"] = was_projected
        psf_meta["safety_operator"] = "Pi_safe"
        psf_meta["implementation_safeguard_projected"] = was_projected
        psf_meta["ls_psf_projected"] = False

        safe = np.asarray(safe_action, dtype=np.float32).reshape(-1)
        raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if raw.size < safe.size:
            raw = np.pad(raw, (0, safe.size - raw.size), mode="constant")
        elif raw.size > safe.size:
            raw = raw[:safe.size]
        psf_meta["total_modification_l2"] = float(np.linalg.norm(safe - raw))

        return safe_action, was_projected, raw_action, psf_meta

    def _apply_physical_feasibility_projection(self, raw_action: np.ndarray, state: np.ndarray, *, in_window: bool, h: float, prop_can_update: bool, available_power_w: float | None) -> tuple[np.ndarray, dict]:
        action_after_prop, prop_meta = self._apply_propulsion_update_constraint(raw_action, state, prop_can_update)
        action_after_boundary, boundary_meta = self._clip_action_boundaries(
            action_after_prop,
            available_power_w=available_power_w,
            in_window=in_window,
            force_prop_priority=(float(h) <= float(ORBITAL_CONFIG.get("altitude_warning_km", 180.0)) * 1e3),
            thermal_margin_norm=self._extract_feature(state, "thermal_margin_norm"),
        )
        if not prop_meta["raw_action_finite_before_smoothing"]:
            if not boundary_meta["boundary_clipped"]:
                self._boundary_clip_count += 1
            boundary_meta["boundary_clipped"] = True
            boundary_meta["action_bound_clipped"] = True

        boundary_meta.update(prop_meta)
        boundary_meta["physical_feasibility_stage"] = "Pi_feas"
        boundary_meta["physical_feasibility_projected"] = bool(
            boundary_meta.get("boundary_clipped", False) or boundary_meta.get("prop_smoothing_applied", False)
        )
        return action_after_boundary, boundary_meta

    def _apply_propulsion_update_constraint(self, action: np.ndarray, state: np.ndarray, prop_can_update: bool) -> tuple[np.ndarray, dict]:
        previous_prop = self._extract_previous_propulsion(state)
        result = self.actuator_filter.apply_propulsion_update_lock(
            action,
            previous_alpha_prop=previous_prop,
            prop_can_update=prop_can_update,
            dtype=np.float32,
        )
        return result.action, result.meta

    @staticmethod
    def _extract_previous_propulsion(state: np.ndarray) -> float | None:
        arr = np.asarray(state, dtype=np.float32)
        if "prev_alpha_prop" not in OBSERVATION_FEATURES:
            return None
        prev_prop_idx = OBSERVATION_FEATURES.index("prev_alpha_prop")
        if arr.ndim == 2 and arr.shape[1] > prev_prop_idx:
            return float(arr[0, prev_prop_idx]) if np.isfinite(arr[0, prev_prop_idx]) else None
        elif arr.ndim == 1 and arr.shape[0] > prev_prop_idx:
            return float(arr[prev_prop_idx]) if np.isfinite(arr[prev_prop_idx]) else None
        return None

    @staticmethod
    def _extract_feature(state: np.ndarray, feature_name: str) -> float | None:
        if feature_name not in OBSERVATION_FEATURES:
            return None
        idx = OBSERVATION_FEATURES.index(feature_name)
        arr = np.asarray(state, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] > idx:
            return float(arr[0, idx]) if np.isfinite(arr[0, idx]) else None
        elif arr.ndim == 1 and arr.shape[0] > idx:
            return float(arr[idx]) if np.isfinite(arr[idx]) else None
        return None

    def _clip_action_boundaries(self, action: np.ndarray, available_power_w: float | None = None, in_window: bool = False, force_prop_priority: bool = False, thermal_margin_norm: float | None = None) -> tuple[np.ndarray, dict]:
        self._boundary_total_count += 1
        allocation = self.actuator_filter.apply_power_boundary(
            action,
            available_power_w=available_power_w,
            in_window=bool(in_window),
            force_prop_priority=bool(force_prop_priority),
            dtype=np.float64,
        )
        clipped = allocation.action.astype(np.float64, copy=True)
        meta = dict(allocation.meta)
        request_action = np.asarray(meta.get("request_action", clipped), dtype=np.float64)

        thermal_clipped = False
        thermal_cpu_cap = 1.0
        thermal_tx_cap = 1.0
        if bool(THERMAL_CONFIG.get("enabled", True)) and thermal_margin_norm is not None:
            margin = float(np.clip(thermal_margin_norm, -1.0, 1.0))
            if margin <= 0.0:
                thermal_cpu_cap = float(np.clip(THERMAL_CONFIG.get("critical_cpu_cap", 0.25), 0.0, 1.0))
                thermal_tx_cap = float(np.clip(THERMAL_CONFIG.get("critical_tx_cap", 0.0), 0.0, 1.0))
            elif margin < 0.35:
                scale = margin / 0.35
                min_scale = float(np.clip(THERMAL_CONFIG.get("warning_cpu_tx_min_scale", 0.35), 0.0, 1.0))
                cap = float(np.clip(min_scale + (1.0 - min_scale) * scale, min_scale, 1.0))
                thermal_cpu_cap = cap
                thermal_tx_cap = cap
            before_thermal = clipped.copy()
            clipped[1] = min(clipped[1], thermal_cpu_cap)
            clipped[2] = min(clipped[2], thermal_tx_cap)
            thermal_clipped = bool(np.linalg.norm(clipped - before_thermal) > 1e-9)

        boundary_clipped = bool(
            (not meta.get("raw_action_finite", True))
            or meta.get("action_bound_clipped", False)
            or meta.get("power_clipped", False)
            or thermal_clipped
            or meta.get("propulsion_deadband_applied", False)
            or meta.get("propulsion_ignition_boost_applied", False)
            or np.linalg.norm(clipped - request_action) > 1e-9
        )
        if boundary_clipped:
            self._boundary_clip_count += 1

        return clipped.astype(np.float32), {
            "boundary_clipped": boundary_clipped,
            "action_bound_clipped": bool(meta.get("action_bound_clipped", False)),
            "power_clipped": bool(meta.get("power_clipped", False)),
            "thermal_clipped": bool(thermal_clipped),
        }

    def store_transition(self, state, action, reward, next_state, done, lya_drift, terminated=None, deliverable_reward: float = 0.0, behavior_action=None, behavior_weight: float = 0.0) -> None:
        self.agent.store(state, action, reward, next_state, done, lya_drift, terminated=terminated, deliverable_reward=deliverable_reward, behavior_action=behavior_action, behavior_weight=behavior_weight)

    def trigger_update(self) -> dict:
        return self.agent.update()

    def trigger_scheduled_updates(self, stored_steps: int = 1) -> list[dict]:
        update_freq = max(1, int(DRL_CONFIG.get("update_freq", 1)))
        stored_steps = max(0, int(stored_steps))
        if stored_steps <= 0:
            return []
        total_steps = int(self.agent.total_steps)
        previous_steps = max(0, total_steps - stored_steps)
        update_count = (total_steps // update_freq) - (previous_steps // update_freq)
        updates = []
        for _ in range(max(0, update_count)):
            updates.append(self.trigger_update())
        return updates

    def learn(self, state, action, reward, next_state, done, lya_drift, terminated=None, behavior_action=None, behavior_weight: float = 0.0) -> dict:
        self.store_transition(state, action, reward, next_state, done, lya_drift, terminated=terminated, behavior_action=behavior_action, behavior_weight=behavior_weight)
        update_stats = self.trigger_scheduled_updates(stored_steps=1)
        return update_stats[-1] if update_stats else {}

    def save(self, path: str):
        metadata = {
            "objective_version": OBJECTIVE_VERSION,
            "enable_lyapunov": bool(self.enable_lyapunov),
            "use_psf": bool(self.use_psf),
            "constraint_variant": getattr(self, "constraint_variant", "ours"),
            "variant_key": getattr(self, "variant_key", None),
            "variant_code": getattr(self, "variant_code", None),
            "ablation_axis": getattr(self, "ablation_axis", None),
            "seed": getattr(self, "training_seed", TRAIN_CONFIG.get("seed", 42)),
            "total_steps": getattr(self, "training_total_steps", TRAIN_CONFIG.get("total_steps", 0)),
            "objective_summary": {
                "risk_boundaries": {
                    "altitude_crash_km": float(ORBITAL_CONFIG.get("altitude_crash_km", 122.0)),
                },
            },
            "reward_weights": {
                "w_delivered_value": float(REWARD_CONFIG.get("w_delivered_value", 1.0)),
                "w_deadline_success": float(REWARD_CONFIG.get("w_deadline_success", 0.0)),
            },
            "constraint_cost_config": {
                "projection_penalty_coeff": float(DRL_CONFIG.get("projection_penalty_coeff", 0.0)),
            },
            "value_aux_head_enable": bool(DRL_CONFIG.get("value_aux_head_enable", False)),
            "value_aux_loss_weight": float(DRL_CONFIG.get("value_aux_loss_weight", 0.0)),
            "value_aux_loss_weight_final": float(DRL_CONFIG.get("value_aux_loss_weight_final", 0.0)),
            "value_aux_high_pressure_margin": float(DRL_CONFIG.get("value_aux_high_pressure_margin", 1.10)),
            "value_aux_low_pressure_margin": float(DRL_CONFIG.get("value_aux_low_pressure_margin", 1.25)),
            "adaptive_lyapunov_constraint_norm": float(DRL_CONFIG.get("adaptive_lyapunov_constraint_norm", 10.0)),
            "adaptive_lyapunov_constraint_threshold": float(DRL_CONFIG.get("adaptive_lyapunov_constraint_threshold", 0.0)),
            "behavior_cloning_conservative_weight_coeff": float(DRL_CONFIG.get("behavior_cloning_conservative_weight_coeff", 0.0)),
        }
        self.agent.save(path, metadata=metadata)

    def load(self, path: str, restore_safety_config: bool = True):
        metadata = self.agent.load(path)
        if restore_safety_config:
            self.enable_lyapunov = bool(metadata.get("enable_lyapunov", self.enable_lyapunov))
            self.use_psf = bool(metadata.get("use_psf", self.use_psf))
            self.psf = None
            if not self.enable_lyapunov:
                self.agent.set_lyapunov_penalty_coeff(0.0)
        return metadata

    def get_safety_stats(self) -> dict:
        boundary_rate = self._boundary_clip_count / max(self._boundary_total_count, 1)
        return {
            "boundary_clip_rate": boundary_rate,
            "physical_projection_rate": boundary_rate,
            "lyapunov_proj_rate": 0.0,
            "psf_filter_rate": 0.0,
            "intervention_rate": boundary_rate,
        }

    def reset_all_safety_stats(self):
        self._boundary_total_count = 0
        self._boundary_clip_count = 0

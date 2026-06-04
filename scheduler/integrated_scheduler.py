"""LS-PSF scheduler: 物理可行 + 状态相关 Lyapunov 投影 + K 步 PSF。

安全链 (Pi_safe = Pi_PSF ∘ Pi_Lya ∘ Pi_feas)：
  raw_action
    -> Pi_feas    : 动作盒 / 推进锁定 / 功率上限 / 热限位
    -> Pi_Lya     : 状态相关 Lyapunov projection (Chow et al. 2018)
    -> Pi_PSF     : K 步 predictive safety filter (Wabersich & Zeilinger 2018)
    -> execute

各算子默认开关由 LYAPUNOV_CONFIG["enabled"] 与 PSF_CONFIG["enabled"] 控制；
__init__ 的 enable_lyapunov / use_psf 参数会覆盖配置默认值。
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

import numpy as np
import torch
from config import (
    DRL_CONFIG,
    ENERGY_CONFIG,
    INFERENCE_MPC_CONFIG,
    LYAPUNOV_CONFIG,
    ORBITAL_CONFIG,
    PSF_CONFIG,
    QUEUE_CONFIG,
    REWARD_CONFIG,
    THERMAL_CONFIG,
    TRAIN_CONFIG,
    OBJECTIVE_VERSION,
)
from environment.satellite_env import OBSERVATION_FEATURES
from algorithms.decoupled_constraint_sac import DecoupledConstraintSAC
from drl.inference_mpc import InferenceMPCPlanner
from safety.actuator_constraints import ActuatorConstraintFilter
from safety.dynamics_predictor import SafetyDynamicsPredictor
from safety.lyapunov_function import LyapunovFunction
from safety.lyapunov_projection import LyapunovProjector
from safety.psf_filter import PredictiveSafetyFilter


class IntegratedScheduler:
    """LS-PSF scheduler: Pi_feas → Pi_Lya → Pi_PSF。

    1. πθ(s)         : SAC 策略输出原始连续动作
    2. Pi_feas       : 动作盒 + 功率/热硬上限（保证环境物理不崩）
    3. Pi_Lya        : 状态相关 Lyapunov projection（半空间投影到 ΔL ≤ ε）
    4. Pi_PSF        : K 步前向 rollout + backup 控制器，线搜索找最大可行 α
    """

    def __init__(
        self,
        device: str = "auto",
        enable_lyapunov: bool | None = None,
        use_psf: bool | None = None,
        use_inference_mpc: bool | None = None,
        **kwargs,
    ):
        state_dim  = DRL_CONFIG.get("state_dim", 30)
        action_dim = DRL_CONFIG["action_dim"]
        self.action_dim = int(action_dim)

        self.agent = DecoupledConstraintSAC(state_dim, action_dim, device)
        self.enable_lyapunov = bool(
            LYAPUNOV_CONFIG.get("enabled", True) if enable_lyapunov is None
            else enable_lyapunov)
        self.use_psf = bool(
            PSF_CONFIG.get("enabled", True) if use_psf is None else use_psf)

        if not self.enable_lyapunov:
            self.agent.set_lyapunov_penalty_coeff(0.0)

        self._boundary_total_count = 0
        self._boundary_clip_count = 0
        self._lyapunov_eval_count = 0
        self._lyapunov_proj_count = 0
        self._psf_eval_count = 0
        self._psf_intervene_count = 0
        self._psf_backup_failure_count = 0

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

        # 共享一个 SafetyDynamicsPredictor 给 Lyapunov 投影 + PSF。
        # 仅在对应安全层开启时才构造算子（测试契约要求关闭时 psf/lyapunov_projector is None）。
        self.safety_predictor = SafetyDynamicsPredictor(
            power_weights=self._power_weights,
            baseline_power_w=self._power_baseline_w,
        )
        self.lyapunov_function = LyapunovFunction()
        self.lyapunov_projector = (
            LyapunovProjector(lyapunov=self.lyapunov_function, predictor=self.safety_predictor)
            if self.enable_lyapunov else None
        )
        self.psf = (
            PredictiveSafetyFilter(predictor=self.safety_predictor)
            if self.use_psf else None
        )
        self.psf_K = int(self.psf.K) if self.psf is not None else int(PSF_CONFIG.get("horizon_steps", 5))

        # Inference MPC planner（短视野规划，纯推理）。
        self.use_inference_mpc = bool(
            INFERENCE_MPC_CONFIG.get("enabled", False) if use_inference_mpc is None
            else use_inference_mpc)
        self.inference_mpc = (
            InferenceMPCPlanner(predictor=self.safety_predictor)
            if self.use_inference_mpc else None
        )
        self._mpc_eval_count = 0
        self._mpc_override_count = 0

    def schedule(self, state: np.ndarray, *args, evaluate: bool = False, **kwargs) -> tuple:
        if args and "in_window" not in kwargs:
            kwargs["in_window"] = bool(args[-1])
        raw_action = self.agent.select_action(state, evaluate=evaluate)
        # 仅在 evaluate / 较高 warmup 之后启用 inference MPC；训练前期让 actor
        # 自由探索，避免 planner 偏置过强反而把策略锁死在 actor 起点附近。
        if self.use_inference_mpc and self.inference_mpc is not None:
            warmup_done = self.agent.total_steps >= int(
                INFERENCE_MPC_CONFIG.get("warmup_steps", self.agent.warmup))
            if evaluate or warmup_done:
                raw_action = self._apply_inference_mpc(state, raw_action, **kwargs)
        return self._schedule_from_raw_action(raw_action, state, **kwargs)

    def schedule_batch(self, states: np.ndarray, contexts: list[dict], evaluate: bool = False) -> list[tuple]:
        states_arr = np.asarray(states, dtype=np.float32)
        raw_actions = self.agent.select_actions(states_arr, evaluate=evaluate)
        outputs = []
        warmup_done = self.agent.total_steps >= int(
            INFERENCE_MPC_CONFIG.get("warmup_steps", self.agent.warmup))
        for state, raw_action, ctx in zip(states_arr, raw_actions, contexts):
            if self.use_inference_mpc and self.inference_mpc is not None and (evaluate or warmup_done):
                raw_action = self._apply_inference_mpc(state, raw_action, **ctx)
            outputs.append(self._schedule_from_raw_action(raw_action, state, **ctx))
        return outputs

    def _apply_inference_mpc(
        self,
        state: np.ndarray,
        actor_mean_action: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """用短视野 shooting 优化 first action。失败/异常时回退到 actor 输出。"""
        try:
            in_window = bool(kwargs.get("in_window", False))
            tx_capacity_mbps = float(
                kwargs.get(
                    "tx_capacity_mbps",
                    (self._extract_feature(state, "tx_capacity_norm") or 0.0)
                    * QUEUE_CONFIG.get("tx_downlink_rate_max_mbs", 5.0) * 8.0,
                )
            )
            sunlit_fraction = float(
                kwargs.get(
                    "sunlit_fraction",
                    self._extract_feature(state, "solar_input_norm") or 0.5,
                )
            )
            physical_state = self._physical_state_from_obs(
                state,
                in_window=in_window,
                tx_capacity_mbps=tx_capacity_mbps,
                sunlit_fraction=sunlit_fraction,
            )
            critic_fn = self._make_critic_value_fn()
            result = self.inference_mpc.plan(
                observation=state,
                physical_state=physical_state,
                actor_mean_action=actor_mean_action,
                critic_value_fn=critic_fn,
                action_dim=self.action_dim,
            )
            self._mpc_eval_count += 1
            if result.used:
                self._mpc_override_count += 1
            return result.action
        except Exception:
            return actor_mean_action

    def _make_critic_value_fn(self):
        """构造一个 (obs, action) -> Q 的可调用对象，用 reward + deliverable critic 联合。"""
        deliverable_coeff = float(getattr(self.agent, "_deliverable_critic_actor_coeff", 0.0))
        deliverable_enabled = bool(getattr(self.agent, "_deliverable_critic_enabled", False))

        def critic_value(obs: np.ndarray, action: np.ndarray) -> float:
            with torch.no_grad():
                obs_arr = np.asarray(obs, dtype=np.float32)
                if obs_arr.ndim == 2:
                    obs_arr = obs_arr[None, ...]  # (1, T, D)
                obs_arr = self.agent._normalize_states_np(obs_arr)
                s = torch.from_numpy(obs_arr).to(self.agent.device)
                a = torch.from_numpy(
                    np.asarray(action, dtype=np.float32).reshape(1, -1)
                ).to(self.agent.device)
                q1, q2 = self.agent.critic(s, a)
                q = torch.min(q1, q2)
                if deliverable_enabled and deliverable_coeff > 0.0:
                    d1, d2 = self.agent.deliverable_critic(s, a)
                    q = q + deliverable_coeff * torch.min(d1, d2)
                return float(q.detach().cpu().numpy().reshape(-1)[0])

        return critic_value

    def _schedule_from_raw_action(self, raw_action: np.ndarray, state: np.ndarray, **kwargs) -> tuple:
        in_window = bool(kwargs.get("in_window", False))
        h = float(kwargs.get("h", 350e3))
        prop_can_update = bool(kwargs.get("prop_can_update", True))
        available_power_w = kwargs.get("available_power_w", None)
        tx_capacity_mbps = float(kwargs.get("tx_capacity_mbps",
                                            self._extract_feature(state, "tx_capacity_norm") or 0.0)
                                  * QUEUE_CONFIG.get("tx_downlink_rate_max_mbs", 5.0) * 8.0)
        sunlit_fraction = float(kwargs.get("sunlit_fraction",
                                           self._extract_feature(state, "solar_input_norm") or 0.5))

        # 1. Pi_feas
        feasible_action, feasibility_meta = self._apply_physical_feasibility_projection(
            raw_action,
            state,
            in_window=in_window,
            h=h,
            prop_can_update=prop_can_update,
            available_power_w=available_power_w,
        )

        safe_action = np.asarray(feasible_action, dtype=np.float32).reshape(-1).copy()

        psf_meta = {}
        psf_meta.update(feasibility_meta)
        psf_meta["safety_operator"] = "Pi_safe"
        psf_meta["implementation_safeguard_projected"] = bool(
            feasibility_meta.get("boundary_clipped", False))

        # 2. Pi_Lya — state-dependent Lyapunov projection
        lya_proj_applied = False
        lya_meta: dict = {}
        if self.enable_lyapunov and self.lyapunov_projector is not None:
            state_phys = self._physical_state_from_obs(
                state,
                in_window=in_window,
                tx_capacity_mbps=tx_capacity_mbps,
                sunlit_fraction=sunlit_fraction,
            )
            lya_res = self.lyapunov_projector.project(safe_action, state_phys)
            self._lyapunov_eval_count += 1
            if lya_res.intervened:
                self._lyapunov_proj_count += 1
                lya_proj_applied = True
                safe_action = np.asarray(lya_res.action, dtype=np.float32).reshape(-1)
            lya_meta = {
                "lyapunov_proj_applied": bool(lya_res.intervened),
                "lyapunov_value": float(lya_res.l_now),
                "lyapunov_next_raw": float(lya_res.l_next_raw),
                "lyapunov_next_projected": float(lya_res.l_next_projected),
                "lyapunov_slack": float(lya_res.slack),
                "lyapunov_violation": float(lya_res.violation),
                "lyapunov_grad_norm": float(lya_res.grad_norm),
                "lyapunov_iterations": int(lya_res.iterations),
            }

        # 3. Pi_PSF — K-step Predictive Safety Filter
        psf_applied = False
        psf_diag: dict = {}
        if self.use_psf and self.psf is not None:
            state_phys = self._physical_state_from_obs(
                state,
                in_window=in_window,
                tx_capacity_mbps=tx_capacity_mbps,
                sunlit_fraction=sunlit_fraction,
            )
            psf_res = self.psf.filter(safe_action, state_phys)
            self._psf_eval_count += 1
            if psf_res.intervened:
                self._psf_intervene_count += 1
                psf_applied = True
                safe_action = np.asarray(psf_res.action, dtype=np.float32).reshape(-1)
            if psf_res.intervened and not psf_res.backup_safe:
                self._psf_backup_failure_count += 1
            psf_diag = {
                "psf_applied": bool(psf_res.intervened),
                "psf_raw_safe": bool(psf_res.raw_safe),
                "psf_backup_safe": bool(psf_res.backup_safe),
                "psf_interpolation_alpha": float(psf_res.interpolation_alpha),
                "psf_horizon_used": int(psf_res.horizon_used),
                "psf_worst_altitude_m": float(psf_res.worst_altitude_m),
                "psf_worst_soc": float(psf_res.worst_soc),
                "psf_worst_processed_queue_mb": float(psf_res.worst_processed_queue_mb),
                "psf_worst_thermal_margin": float(psf_res.worst_thermal_margin),
            }

        psf_meta.update(lya_meta)
        psf_meta.update(psf_diag)
        was_projected = bool(
            feasibility_meta.get("boundary_clipped", False)
            or lya_proj_applied
            or psf_applied
        )
        psf_meta["safety_chain_projected"] = was_projected
        psf_meta["ls_psf_projected"] = bool(lya_proj_applied or psf_applied)

        safe = np.asarray(safe_action, dtype=np.float32).reshape(-1)
        raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if raw.size < safe.size:
            raw = np.pad(raw, (0, safe.size - raw.size), mode="constant")
        elif raw.size > safe.size:
            raw = raw[:safe.size]
        psf_meta["total_modification_l2"] = float(np.linalg.norm(safe - raw))

        return safe_action, was_projected, raw_action, psf_meta

    def _physical_state_from_obs(
        self,
        state: np.ndarray,
        *,
        in_window: bool,
        tx_capacity_mbps: float,
        sunlit_fraction: float,
    ) -> dict:
        """从观测向量反推 PSF/Lyapunov 需要的物理量。"""
        h_min = float(ORBITAL_CONFIG.get("altitude_min_km", 180.0)) * 1e3
        h_max = float(ORBITAL_CONFIG.get("altitude_max_km", 300.0)) * 1e3
        processed_max = float(QUEUE_CONFIG.get("comm_queue_max", 4096.0))
        raw_max = float(QUEUE_CONFIG.get("data_queue_max_mb", 4096.0))
        altitude_norm = self._extract_feature(state, "altitude_norm") or 1.0
        soc = self._extract_feature(state, "soc") or 1.0
        processed_util = self._extract_feature(state, "processed_queue_utilization") or 0.0
        raw_util = self._extract_feature(state, "raw_queue_utilization") or 0.0
        thermal_margin = self._extract_feature(state, "thermal_margin_norm")
        thermal_margin = 1.0 if thermal_margin is None else float(thermal_margin)
        future_norm = self._extract_feature(state, "future_contact_capacity_norm") or 0.0
        return {
            "altitude_m": float(altitude_norm * (h_max - h_min) + h_min),
            "soc": float(soc),
            "processed_queue_mb": float(processed_util * processed_max),
            "raw_queue_mb": float(raw_util * raw_max),
            "thermal_margin_norm": float(thermal_margin),
            "in_window": bool(in_window),
            "tx_capacity_mbps": float(tx_capacity_mbps),
            "sunlit_fraction": float(np.clip(sunlit_fraction, 0.0, 1.0)),
            "future_contact_capacity_mb": float(future_norm * processed_max),
        }

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
            # 暴露热降额上限（已在上面算好），供诊断/测试核对 cpu/tx 是否被压到上限内。
            "thermal_cpu_cap": float(thermal_cpu_cap),
            "thermal_tx_cap": float(thermal_tx_cap),
            # 透传 apply_power_boundary 已算出的诊断键，保持与
            # env._enforce_available_power 的 meta 口径一致（这些键内部已用于
            # boundary_clipped 判定，之前漏在返回 dict 里）。
            "power_priority_order": str(meta.get("power_priority_order", "prop>cpu>tx")),
            "propulsion_deadband_applied": bool(meta.get("propulsion_deadband_applied", False)),
            "raw_action_finite": bool(meta.get("raw_action_finite", True)),
        }

    def store_transition(self, state, action, reward, next_state, done, lya_drift, terminated=None, deliverable_reward: float = 0.0, behavior_action=None, behavior_weight: float = 0.0, env_id: int = 0) -> None:
        self.agent.store(state, action, reward, next_state, done, lya_drift, terminated=terminated, deliverable_reward=deliverable_reward, behavior_action=behavior_action, behavior_weight=behavior_weight, env_id=env_id)

    def reset_env_aggregator(self, env_id: int) -> None:
        """转发 episode 边界 n-step aggregator reset 到底层 agent。"""
        if hasattr(self.agent, "reset_env_aggregator"):
            self.agent.reset_env_aggregator(env_id)

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
            # 记录观测 schema，便于加载时校验 checkpoint 与当前观测定义一致。
            "observation_features": list(OBSERVATION_FEATURES),
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
            # 同步实例化/置空各安全算子，保持 None 契约。
            if self.enable_lyapunov and self.lyapunov_projector is None:
                self.lyapunov_projector = LyapunovProjector(
                    lyapunov=self.lyapunov_function,
                    predictor=self.safety_predictor,
                )
            elif not self.enable_lyapunov:
                self.lyapunov_projector = None
                self.agent.set_lyapunov_penalty_coeff(0.0)
            if self.use_psf and self.psf is None:
                self.psf = PredictiveSafetyFilter(predictor=self.safety_predictor)
            elif not self.use_psf:
                self.psf = None
        return metadata

    def get_safety_stats(self) -> dict:
        boundary_rate = self._boundary_clip_count / max(self._boundary_total_count, 1)
        lyapunov_rate = (
            self._lyapunov_proj_count / self._lyapunov_eval_count
            if self._lyapunov_eval_count > 0 else 0.0
        )
        psf_rate = (
            self._psf_intervene_count / self._psf_eval_count
            if self._psf_eval_count > 0 else 0.0
        )
        intervention_rate = (
            (self._boundary_clip_count + self._lyapunov_proj_count + self._psf_intervene_count)
            / max(self._boundary_total_count, 1)
        )
        mpc_override_rate = (
            self._mpc_override_count / self._mpc_eval_count
            if self._mpc_eval_count > 0 else 0.0
        )
        return {
            "boundary_clip_rate": boundary_rate,
            "physical_projection_rate": boundary_rate,
            "lyapunov_proj_rate": float(lyapunov_rate),
            "psf_filter_rate": float(psf_rate),
            "psf_backup_failure_rate": float(
                self._psf_backup_failure_count / max(self._psf_eval_count, 1)
                if self._psf_eval_count > 0 else 0.0
            ),
            "inference_mpc_eval_count": int(self._mpc_eval_count),
            "inference_mpc_override_rate": float(mpc_override_rate),
            "intervention_rate": float(intervention_rate),
        }

    def reset_all_safety_stats(self):
        self._boundary_total_count = 0
        self._boundary_clip_count = 0
        self._lyapunov_eval_count = 0
        self._lyapunov_proj_count = 0
        self._psf_eval_count = 0
        self._psf_intervene_count = 0
        self._psf_backup_failure_count = 0
        self._mpc_eval_count = 0
        self._mpc_override_count = 0

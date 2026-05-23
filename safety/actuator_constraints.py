"""论文面向的执行器约束滤波器。

本模块命名被动部署侧边界算子：

    a_exec = Phi_act(a_policy | S_physical)

它故意放在 reward critic 外。调度器和环境调用同一算子
以保证策略到执行器映射保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import QUEUE_CONFIG, TRAIN_CONFIG
from safety.power_manager import (
    PowerAllocationResult,
    allocate_power_strict_priority,
    apply_propulsion_deadband_to_action,
)
from utils.sanitizers import sanitize_action, sanitize_scalar


@dataclass(frozen=True)
class ActuatorConstraintResult:
    """执行器约束滤波器输出的动作和元数据。"""

    action: np.ndarray
    meta: dict


class BoundedActionSanitizer:
    """将一个策略动作约束到 [0, 1]^action_dim。"""

    def __init__(self, action_dim: int | None = None, dtype=np.float64):
        from config import DRL_CONFIG
        self.action_dim = int(action_dim or DRL_CONFIG.get("action_dim", 10))
        self.dtype = dtype

    def __call__(self, action, *, dtype=None) -> ActuatorConstraintResult:
        out_dtype = self.dtype if dtype is None else dtype
        bounded, raw_finite, in_bounds = sanitize_action(
            action,
            action_dim=self.action_dim,
            dtype=out_dtype,
        )
        return ActuatorConstraintResult(
            action=bounded,
            meta={
                "raw_action_finite": bool(raw_finite),
                "input_action_in_bounds": bool(in_bounds),
                "action_bound_clipped": bool(not in_bounds),
            },
        )


class ActuatorConstraintFilter:
    """
    执行器侧约束的共享被动安全算子。

    该类用论文术语包装低级功率管理器。它不优化奖励；
    仅将策略命令映射到可执行的执行器命令。
    """

    def __init__(
        self,
        *,
        power_weights: np.ndarray | None = None,
        baseline_w: float | None = None,
        total_limit_w: float | None = None,
        prop_ignition_threshold_w: float | None = None,
        action_dim: int | None = None,
    ):
        self.power_weights = (
            None if power_weights is None else np.asarray(power_weights, dtype=np.float64)
        )
        self.baseline_w = baseline_w
        self.total_limit_w = total_limit_w
        self.prop_ignition_threshold_w = prop_ignition_threshold_w
        self.sanitizer = BoundedActionSanitizer(action_dim=action_dim)

    def sanitize(self, action, *, dtype=np.float64) -> ActuatorConstraintResult:
        return self.sanitizer(action, dtype=dtype)

    def apply_propulsion_update_lock(
        self,
        action,
        *,
        previous_alpha_prop: float | None,
        prop_can_update: bool,
        dtype=np.float32,
    ) -> ActuatorConstraintResult:
        """当执行器更新周期被锁定时，保持推进通道。"""
        sanitized = self.sanitize(action, dtype=np.float64)
        bounded = sanitized.action.copy()

        previous_prop = previous_alpha_prop
        smoothing_applied = False
        if (not bool(prop_can_update)) and previous_prop is not None:
            previous_prop = sanitize_scalar(
                previous_prop,
                min_value=0.0,
                max_value=1.0,
            )
            if abs(float(bounded[0]) - float(previous_prop)) > 1e-9:
                bounded[0] = previous_prop
                smoothing_applied = True

        return ActuatorConstraintResult(
            action=bounded.astype(dtype),
            meta={
                "prop_smoothing_applied": bool(smoothing_applied),
                "prop_can_update": bool(prop_can_update),
                "scheduler_prev_alpha_prop": (
                    None if previous_prop is None else float(previous_prop)
                ),
                "raw_action_finite_before_smoothing": bool(
                    sanitized.meta["raw_action_finite"]),
            },
        )

    def apply_propulsion_deadband(self, action, *, dtype=np.float32) -> ActuatorConstraintResult:
        bounded, applied = apply_propulsion_deadband_to_action(
            action,
            prop_max_w=None if self.power_weights is None else float(self.power_weights[0]),
            threshold_w=self.prop_ignition_threshold_w,
            dtype=dtype,
        )
        return ActuatorConstraintResult(
            action=bounded,
            meta={"propulsion_deadband_applied": bool(applied)},
        )

    def apply_power_boundary(
        self,
        action,
        *,
        available_power_w: float | None,
        in_window: bool,
        force_prop_priority: bool,
        dtype=np.float64,
    ) -> PowerAllocationResult:
        return allocate_power_strict_priority(
            action,
            available_power_w=available_power_w,
            in_window=bool(in_window),
            force_prop_priority=bool(force_prop_priority),
            power_weights=self.power_weights,
            baseline_w=self.baseline_w,
            total_limit_w=self.total_limit_w,
            prop_ignition_threshold_w=self.prop_ignition_threshold_w,
            dtype=dtype,
        )

    def project_processed_queue_boundary(
        self,
        action,
        *,
        processed_queue_mb: float,
        processed_queue_max_mb: float | None = None,
        in_window: bool,
        tx_capacity_mbps: float,
        dt_s: float | None = None,
        future_contact_capacity_mb: float | None = None,
        future_capacity_margin: float | None = None,
        future_ratio_start: float | None = None,
        future_ratio_hard_stop: float | None = None,
        apply_projection: bool = True,
        dtype=np.float32,
    ) -> ActuatorConstraintResult:
        """
        Project CPU effort so one step cannot overfill the processed-data queue.

        This is a deployment-side hard boundary:

            alpha_cpu_safe = Proj_Aq(alpha_cpu_raw | q_processed, c_link)

        It is intentionally not a reward term. The environment can call this
        with ``apply_projection=False`` to report boundary pressure without
        changing its physical transition.
        """
        bounded = self.sanitize(action, dtype=np.float64).action.copy()
        original = bounded.copy()
        dt = float(TRAIN_CONFIG.get("time_slot_s", 10.0) if dt_s is None else dt_s)
        dt = max(dt, 0.0)

        service_rate_max = float(QUEUE_CONFIG.get(
            "data_service_rate_max_mbs",
            QUEUE_CONFIG.get("data_service_rate_max_mbps", 5.0),
        ))
        tx_rate_max = float(QUEUE_CONFIG.get("tx_downlink_rate_max_mbs", 5.0))
        q_max = float(QUEUE_CONFIG.get("comm_queue_max", 4096.0))
        if processed_queue_max_mb is not None:
            q_max = float(processed_queue_max_mb)
        q_now = float(processed_queue_mb)
        margin_mb = max(0.0, float(
            QUEUE_CONFIG.get("processed_queue_backpressure_margin_mb", 1.0)))

        alpha_cpu = float(np.clip(bounded[1], 0.0, 1.0))
        alpha_tx = float(np.clip(bounded[2], 0.0, 1.0))
        requested_processed_mb = alpha_cpu * service_rate_max * dt

        link_capacity_mb = 0.0
        tx_room_mb = 0.0
        if bool(in_window):
            link_capacity_mb = max(0.0, float(tx_capacity_mbps)) * dt / 8.0
            rf_capacity_mb = alpha_tx * tx_rate_max * dt
            tx_room_mb = min(alpha_tx * link_capacity_mb, rf_capacity_mb)

        headroom_mb = max(0.0, q_max - q_now)
        one_step_allowed_mb = max(0.0, headroom_mb + tx_room_mb - margin_mb)
        allowed_processed_mb = one_step_allowed_mb
        future_contact_allowed_mb = float("inf")
        processed_queue_future_contact_ratio = 0.0
        future_contact_boundary_violation = False
        if future_contact_capacity_mb is not None:
            future_capacity_mb = max(0.0, float(future_contact_capacity_mb))
            margin = max(0.0, float(
                1.0 if future_capacity_margin is None else future_capacity_margin))
            start_ratio = max(0.0, float(
                margin if future_ratio_start is None else future_ratio_start))
            hard_stop_ratio = max(start_ratio, float(
                margin if future_ratio_hard_stop is None else future_ratio_hard_stop))
            processed_queue_future_contact_ratio = q_now / max(future_capacity_mb, 1e-6)
            if future_capacity_mb <= 1e-6:
                future_contact_allowed_mb = 0.0
            elif processed_queue_future_contact_ratio >= hard_stop_ratio:
                future_contact_allowed_mb = 0.0
            elif processed_queue_future_contact_ratio >= start_ratio:
                future_contact_allowed_mb = max(0.0, margin * future_capacity_mb - q_now)
            # else: ratio < start_ratio → safe region, no future-contact back-pressure
            allowed_processed_mb = min(allowed_processed_mb, future_contact_allowed_mb)
            future_contact_boundary_violation = bool(
                requested_processed_mb > future_contact_allowed_mb + 1e-9)
        else:
            future_capacity_mb = 0.0

        processed_queue_boundary_violation = bool(
            requested_processed_mb > one_step_allowed_mb + 1e-9)
        required_ratio = 0.0
        should_project = bool(requested_processed_mb > allowed_processed_mb + 1e-9)
        if should_project:
            cpu_scale = allowed_processed_mb / max(requested_processed_mb, 1e-9)
            required_ratio = float(np.clip(1.0 - cpu_scale, 0.0, 1.0))
            if apply_projection:
                bounded[1] = float(np.clip(bounded[1] * cpu_scale, 0.0, 1.0))

        applied = bool(apply_projection and should_project)
        meta = {
            "cpu_backpressure_applied": bool(applied),
            "cpu_backpressure_required": bool(should_project),
            "cpu_backpressure_mod_l2": float(np.linalg.norm(bounded - original)),
            "cpu_backpressure_ratio": float(required_ratio if applied else 0.0),
            "required_cpu_backpressure_ratio": float(required_ratio),
            "requested_processed_mb": float(requested_processed_mb),
            "allowed_processed_mb": float(allowed_processed_mb),
            "one_step_allowed_processed_mb": float(one_step_allowed_mb),
            "future_contact_allowed_processed_mb": float(
                0.0 if not np.isfinite(future_contact_allowed_mb) else future_contact_allowed_mb),
            "future_contact_capacity_mb": float(future_capacity_mb),
            "processed_queue_future_contact_ratio": float(processed_queue_future_contact_ratio),
            "processed_queue_headroom_mb": float(headroom_mb),
            "processed_queue_tx_room_mb": float(tx_room_mb),
            "processed_backpressure_in_window": float(bool(in_window)),
            "processed_backpressure_link_capacity_mb": float(link_capacity_mb),
            "processed_queue_boundary_violation": bool(processed_queue_boundary_violation),
            "future_contact_boundary_violation": bool(future_contact_boundary_violation),
        }
        return ActuatorConstraintResult(action=bounded.astype(dtype), meta=meta)

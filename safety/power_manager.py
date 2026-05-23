"""共享执行器死区和严格优先级功率分配。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import ENERGY_CONFIG
from utils.sanitizers import sanitize_action, sanitize_scalar
from utils.action_space import PHYSICAL_ACTION_DIM


@dataclass(frozen=True)
class PowerAllocationResult:
    """将一个动作裁剪到可用执行器功率预算的结果。"""

    action: np.ndarray
    meta: dict


def default_power_weights(cfg: dict | None = None) -> np.ndarray:
    cfg = cfg or ENERGY_CONFIG
    return np.array([
        cfg["power_propulsion_max_w"],
        cfg["power_cpu_max_w"],
        cfg["power_tx_max_w"],
    ], dtype=np.float64)


def priority_order(in_window: bool, force_prop_priority: bool) -> tuple[int, int, int]:
    if bool(force_prop_priority):
        return (0, 2, 1) if bool(in_window) else (0, 1, 2)
    return (2, 0, 1) if bool(in_window) else (0, 1, 2)


def priority_label(in_window: bool, force_prop_priority: bool) -> str:
    if bool(force_prop_priority) and bool(in_window):
        return "prop>tx>cpu"
    if bool(in_window):
        return "tx>prop>cpu"
    return "prop>cpu>tx"


def apply_propulsion_deadband_watts(
    power_w: float,
    threshold_w: float | None = None,
) -> tuple[float, bool]:
    """关闭低于点火阈值的推进功率。"""
    threshold = float(ENERGY_CONFIG.get(
        "propulsion_ignition_threshold_w", 0.0)
        if threshold_w is None else threshold_w)
    power = sanitize_scalar(power_w, nan=0.0, posinf=0.0, neginf=0.0, min_value=0.0)
    if 0.0 < power < threshold:
        return 0.0, True
    return float(power), False


def apply_propulsion_deadband_to_action(
    action,
    *,
    prop_max_w: float | None = None,
    threshold_w: float | None = None,
    dtype=np.float32,
) -> tuple[np.ndarray, bool]:
    """对动作中的alpha_prop应用推进点火死区。"""
    clean, _, _ = sanitize_action(action, dtype=np.float64)
    prop_max = float(ENERGY_CONFIG.get("power_propulsion_max_w", 1.0)
                     if prop_max_w is None else prop_max_w)
    threshold = float(ENERGY_CONFIG.get("propulsion_ignition_threshold_w", 0.0)
                      if threshold_w is None else threshold_w)
    prop_power = float(clean[0] * prop_max)
    if 0.0 < prop_power < threshold:
        clean[0] = 0.0
        return clean.astype(dtype), True
    return clean.astype(dtype), False


def allocate_power_strict_priority(
    action,
    *,
    available_power_w: float | None,
    in_window: bool,
    force_prop_priority: bool,
    power_weights: np.ndarray | None = None,
    baseline_w: float | None = None,
    total_limit_w: float | None = None,
    prop_ignition_threshold_w: float | None = None,
    dtype=np.float64,
) -> PowerAllocationResult:
    """
    将物理通道裁剪到严格优先级可调功率预算。

    这是调度器端裁剪和环境端执行闭包都使用的单一实现。
    """
    weights = default_power_weights() if power_weights is None else np.asarray(power_weights, dtype=np.float64)
    baseline = float(ENERGY_CONFIG["power_baseline_w"] if baseline_w is None else baseline_w)
    total_limit = float(ENERGY_CONFIG.get("power_total_max_w", 120.0)
                        if total_limit_w is None else total_limit_w)
    threshold = float(ENERGY_CONFIG.get("propulsion_ignition_threshold_w", 0.0)
                      if prop_ignition_threshold_w is None else prop_ignition_threshold_w)

    raw_action = np.asarray(action, dtype=np.float64).reshape(-1)
    action_dim = max(int(raw_action.size), PHYSICAL_ACTION_DIM)
    clipped_request, raw_is_finite, input_in_bounds = sanitize_action(
        action, action_dim=action_dim, dtype=np.float64)
    actuator_request = clipped_request[:PHYSICAL_ACTION_DIM]
    if available_power_w is None:
        available_power = total_limit
    else:
        available_power = sanitize_scalar(
            available_power_w,
            nan=total_limit,
            posinf=total_limit,
            neginf=0.0,
            min_value=0.0,
            max_value=total_limit,
        )
    adjustable_budget = max(available_power - baseline, 0.0)

    requested_adjustable = float(np.dot(weights, actuator_request))
    order = priority_order(in_window, force_prop_priority)
    allocated_w = np.zeros(PHYSICAL_ACTION_DIM, dtype=np.float64)
    power_clipped = False
    prop_deadband_applied = False
    prop_ignition_boost_applied = False

    if requested_adjustable > adjustable_budget + 1e-9:
        power_clipped = True
        remaining = adjustable_budget
        requested_w = actuator_request * weights
        for idx in order:
            requested_channel_w = max(float(requested_w[idx]), 0.0)
            if idx == 0 and threshold > 0.0:
                if 0.0 < requested_channel_w < threshold:
                    if bool(force_prop_priority) and remaining >= threshold:
                        requested_channel_w = threshold
                        prop_ignition_boost_applied = True
                    else:
                        prop_deadband_applied = True
                        continue
                if remaining < threshold:
                    if requested_channel_w > 0.0:
                        prop_deadband_applied = True
                    continue
            take = min(requested_channel_w, max(remaining, 0.0))
            allocated_w[idx] = take
            remaining -= take
    else:
        allocated_w = actuator_request * weights
        if threshold > 0.0 and 0.0 < allocated_w[0] < threshold:
            lower_priority_power = float(allocated_w[1] + allocated_w[2])
            if (bool(force_prop_priority)
                    and lower_priority_power + threshold <= adjustable_budget + 1e-9):
                allocated_w[0] = threshold
                prop_ignition_boost_applied = True
            else:
                allocated_w[0] = 0.0
                prop_deadband_applied = True

    clipped_physical = np.divide(
        allocated_w,
        weights,
        out=np.zeros_like(allocated_w),
        where=weights > 0.0,
    )
    clipped = clipped_request.copy()
    clipped[:PHYSICAL_ACTION_DIM] = clipped_physical
    executed_adjustable = float(np.dot(weights, clipped[:PHYSICAL_ACTION_DIM]))
    effective_scale = (
        executed_adjustable / requested_adjustable
        if requested_adjustable > 1e-9 else 1.0
    )
    meta = {
        "action_bound_clipped": bool(not input_in_bounds),
        "raw_action_finite": bool(raw_is_finite),
        "power_clipped": bool(power_clipped),
        "power_action_scale": float(effective_scale),
        "power_clip_mode": "strict_priority",
        "propulsion_deadband_applied": bool(prop_deadband_applied),
        "propulsion_ignition_boost_applied": bool(prop_ignition_boost_applied),
        "propulsion_ignition_threshold_w": float(threshold),
        "power_priority_order": priority_label(in_window, force_prop_priority),
        "available_power_w": float(available_power),
        "adjustable_power_budget_w": float(adjustable_budget),
        "requested_adjustable_power_w": float(requested_adjustable),
        "executed_adjustable_power_w": float(executed_adjustable),
        "requested_total_power_w": float(baseline + requested_adjustable),
        "executed_total_power_w": float(baseline + executed_adjustable),
        "allocated_power_w": allocated_w.astype(np.float64),
        "request_action": clipped_request.astype(np.float64),
        "request_physical_action": actuator_request.astype(np.float64),
    }
    return PowerAllocationResult(action=clipped.astype(dtype), meta=meta)

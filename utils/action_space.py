"""用于价值感知任务调度的共享动作空间助手。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PHYSICAL_ACTION_DIM = 3
VALUE_CLASS_COUNT = 3
GROUPED_ACTION_DIM = 8
COMPACT_PRIORITY_ACTION_DIM = 8
LEGACY_GROUPED_ACTION_DIM = 10
VALUE_CLASS_NAMES = ("high", "medium", "low")


@dataclass(frozen=True)
class GroupedActionDecision:
    """环境转移使用的解码策略动作。"""

    physical: np.ndarray
    cpu_ratios: np.ndarray
    tx_ratios: np.ndarray
    drop_low_strength: float
    cpu_value_weight: float
    cpu_urgency_weight: float
    tx_value_weight: float
    tx_urgency_weight: float


def _softmax(values: np.ndarray, *, scale: float = 4.0) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64).reshape(-1)
    if logits.size != VALUE_CLASS_COUNT:
        logits = np.resize(logits, VALUE_CLASS_COUNT)
    logits = np.clip(logits, 0.0, 1.0) * float(scale)
    logits = logits - float(np.max(logits))
    exp_logits = np.exp(logits)
    denom = float(np.sum(exp_logits))
    if denom <= 1e-12 or not np.isfinite(denom):
        return np.full(VALUE_CLASS_COUNT, 1.0 / VALUE_CLASS_COUNT, dtype=np.float64)
    return (exp_logits / denom).astype(np.float64)


def _unit_to_signed(value: float) -> float:
    return float(np.clip(2.0 * float(value) - 1.0, -1.0, 1.0))


def decode_grouped_action(action, *, logit_scale: float = 4.0) -> GroupedActionDecision:
    """解码[prop, cpu_budget, tx, cpu_value, cpu_urgency, tx_value, tx_urgency, drop_low]。"""
    original = np.asarray(action, dtype=np.float64).reshape(-1)
    original_size = int(original.size)
    if original_size >= LEGACY_GROUPED_ACTION_DIM:
        legacy = np.nan_to_num(original[:LEGACY_GROUPED_ACTION_DIM], nan=0.0, posinf=1.0, neginf=0.0)
        legacy = np.clip(legacy, 0.0, 1.0)
        arr = np.array([
            legacy[0], legacy[1], legacy[2],
            legacy[3], legacy[4], legacy[6], legacy[7], legacy[9],
        ], dtype=np.float64)
    elif original_size < GROUPED_ACTION_DIM:
        arr = np.pad(original, (0, GROUPED_ACTION_DIM - original_size), mode="constant", constant_values=0.5)
        if original_size < PHYSICAL_ACTION_DIM:
            arr[:PHYSICAL_ACTION_DIM] = np.pad(
                original[:original_size],
                (0, PHYSICAL_ACTION_DIM - original_size),
                mode="constant",
            )
        arr[GROUPED_ACTION_DIM - 1] = 0.0
    else:
        arr = original[:GROUPED_ACTION_DIM].copy()
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)

    cpu_ratio_logits = np.array([arr[3], arr[4], 0.5], dtype=np.float64)
    tx_ratio_logits = np.array([arr[5], arr[6], 0.5], dtype=np.float64)

    return GroupedActionDecision(
        physical=arr[:PHYSICAL_ACTION_DIM].astype(np.float64),
        cpu_ratios=_softmax(cpu_ratio_logits, scale=logit_scale),
        tx_ratios=_softmax(tx_ratio_logits, scale=logit_scale),
        drop_low_strength=float(arr[7]),
        cpu_value_weight=_unit_to_signed(arr[3]),
        cpu_urgency_weight=_unit_to_signed(arr[4]),
        tx_value_weight=_unit_to_signed(arr[5]),
        tx_urgency_weight=_unit_to_signed(arr[6]),
    )


def default_grouped_action(physical_action) -> np.ndarray:
    """将遗留的3维物理动作扩展到优先级权重动作。"""
    physical = np.asarray(physical_action, dtype=np.float32).reshape(-1)
    if physical.size < PHYSICAL_ACTION_DIM:
        physical = np.pad(physical, (0, PHYSICAL_ACTION_DIM - physical.size))
    out = np.full(GROUPED_ACTION_DIM, 0.5, dtype=np.float32)
    out[:PHYSICAL_ACTION_DIM] = np.clip(physical[:PHYSICAL_ACTION_DIM], 0.0, 1.0)
    out[7] = 0.0
    return out

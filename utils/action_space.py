"""用于价值感知任务调度的共享动作空间助手。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PHYSICAL_ACTION_DIM = 3
VALUE_CLASS_COUNT = 3
GROUPED_ACTION_DIM = 9   # 8 + pointing_mode([SAFETY-REAL] 姿态指向: 0=IMAGE 1=DOWNLINK 2=SUN)
COMPACT_PRIORITY_ACTION_DIM = 9
LEGACY_GROUPED_ACTION_DIM = 10
VALUE_CLASS_NAMES = ("high", "medium", "low")

# 指向模式编码
POINTING_IMAGE = 0     # 对地成像:可采集原始数据
POINTING_DOWNLINK = 1  # 对站下传:窗口内可 TX
POINTING_SUN = 2       # 对日充电:太阳输入最大
POINTING_MODE_COUNT = 3


def pointing_mode_from_unit(value: float) -> int:
    """连续 [0,1] 动作离散为 3 个指向模式。"""
    v = float(np.clip(value, 0.0, 1.0))
    return int(min(POINTING_MODE_COUNT - 1, int(v * POINTING_MODE_COUNT)))


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
    pointing_mode: int = POINTING_DOWNLINK


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
            0.5,  # pointing 默认(legacy 无此维)
        ], dtype=np.float64)
    elif original_size < GROUPED_ACTION_DIM:
        arr = np.pad(original, (0, GROUPED_ACTION_DIM - original_size), mode="constant", constant_values=0.5)
        if original_size < PHYSICAL_ACTION_DIM:
            arr[:PHYSICAL_ACTION_DIM] = np.pad(
                original[:original_size],
                (0, PHYSICAL_ACTION_DIM - original_size),
                mode="constant",
            )
        if original_size <= 7:
            arr[7] = 0.0  # drop_low 默认关闭(未提供时)
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
        pointing_mode=pointing_mode_from_unit(arr[8]),
    )


def default_grouped_action(physical_action, pointing_unit: float = 0.5) -> np.ndarray:
    """将遗留的3维物理动作扩展到优先级权重动作。pointing_unit 控制指向模式([0,1]→IMAGE/DOWNLINK/SUN)。"""
    physical = np.asarray(physical_action, dtype=np.float32).reshape(-1)
    if physical.size < PHYSICAL_ACTION_DIM:
        physical = np.pad(physical, (0, PHYSICAL_ACTION_DIM - physical.size))
    out = np.full(GROUPED_ACTION_DIM, 0.5, dtype=np.float32)
    out[:PHYSICAL_ACTION_DIM] = np.clip(physical[:PHYSICAL_ACTION_DIM], 0.0, 1.0)
    out[7] = 0.0
    out[8] = float(np.clip(pointing_unit, 0.0, 1.0))
    return out


def pointing_unit_for_mode(mode: int) -> float:
    """把离散模式映射回 [0,1] 动作值(取该模式区间中点)。"""
    return (float(mode) + 0.5) / float(POINTING_MODE_COUNT)


def choose_pointing_unit_for_env(env, min_processed_mb: float = 1.0) -> float:
    """非学习基线的默认指向策略:窗口内有已处理数据→下传;昼侧→成像采集;否则→对日充电。"""
    contact = getattr(env, "_contact", None) or {}
    in_window = bool(contact.get("in_window", False))
    cq = getattr(env, "comm_queue", None)
    has_proc = bool(cq is not None and float(getattr(cq, "value", 0.0)) > min_processed_mb)
    try:
        daylit = bool(env.orbit_sim.is_sunlit(env.time_s))
    except Exception:
        daylit = True
    if in_window and has_proc:
        return pointing_unit_for_mode(POINTING_DOWNLINK)
    if daylit:
        return pointing_unit_for_mode(POINTING_IMAGE)
    return pointing_unit_for_mode(POINTING_SUN)

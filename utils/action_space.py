"""用于价值感知任务调度的共享动作空间助手。

动作布局（解耦版，action_dim = 15）：
    [0]  prop           推进功率比
    [1]  cpu_budget     CPU 处理预算比
    [2]  tx             下传功率比
    [3,4,5]  cpu_class_logits   CPU 高/中/低三类显式分配 logits
    [6,7,8]  tx_class_logits    TX  高/中/低三类显式分配 logits
    [9]  cpu_value_weight    CPU 价值权重（[0,1]→[-1,1]）
    [10] cpu_urgency_weight  CPU 紧迫权重
    [11] tx_value_weight     TX  价值权重
    [12] tx_urgency_weight   TX  紧迫权重
    [13] drop_low            低价值丢弃强度
    [14] pointing            指向模式([0,1]→IMAGE/DOWNLINK/SUN)

历史版本（已弃用）曾把 arr[3]/arr[4] 同时当作“CPU 高/中类 logit”和
“CPU value/urgency 权重”，arr[5]/arr[6] 同理用于 TX，且 low 类 logit 恒为 0.5。
该双重身份让 actor 无法形成清晰动作语义（调 urgency 实际也在改 class 分配），
现已拆成显式 3-class logits + 独立 value/urgency 参数。切换需重训（checkpoint 不兼容）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PHYSICAL_ACTION_DIM = 3
VALUE_CLASS_COUNT = 3
GROUPED_ACTION_DIM = 15
# 历史紧凑布局维度，仅用于识别/兼容旧 8/9/10 维动作（解码时按中性值补齐，不再解释旧语义）。
LEGACY_GROUPED_ACTION_DIM = 10
COMPACT_PRIORITY_ACTION_DIM = 9
VALUE_CLASS_NAMES = ("high", "medium", "low")

# ── 命名下标（所有消费者按语义索引，避免裸下标错位）─────────────────────
IDX_PROP = 0
IDX_CPU_BUDGET = 1
IDX_TX = 2
CPU_LOGITS_SLICE = slice(3, 6)
TX_LOGITS_SLICE = slice(6, 9)
IDX_CPU_VALUE_WEIGHT = 9
IDX_CPU_URGENCY_WEIGHT = 10
IDX_TX_VALUE_WEIGHT = 11
IDX_TX_URGENCY_WEIGHT = 12
IDX_DROP_LOW = 13
IDX_POINTING = 14

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


def _neutral_action() -> np.ndarray:
    """中性动作模板：logits/权重取 0.5（→ 均匀分配 / 零偏置），drop=0，pointing=0.5。"""
    arr = np.full(GROUPED_ACTION_DIM, 0.5, dtype=np.float64)
    arr[IDX_DROP_LOW] = 0.0
    return arr


def _coerce_to_grouped(action) -> np.ndarray:
    """把任意长度动作规整为 GROUPED_ACTION_DIM：足够长截断，不足按中性值补齐。"""
    original = np.asarray(action, dtype=np.float64).reshape(-1)
    n = int(original.size)
    if n >= GROUPED_ACTION_DIM:
        arr = original[:GROUPED_ACTION_DIM].copy()
    else:
        arr = _neutral_action()
        # 物理维度按实际提供量补齐，其余保持中性。
        m = min(n, GROUPED_ACTION_DIM)
        arr[:m] = original[:m]
        if n < IDX_DROP_LOW:
            arr[IDX_DROP_LOW] = 0.0  # 未提供时丢弃默认关闭
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def decode_grouped_action(action, *, logit_scale: float = 4.0) -> GroupedActionDecision:
    """解码 15 维分组动作（见模块 docstring 的布局）。"""
    arr = _coerce_to_grouped(action)
    return GroupedActionDecision(
        physical=arr[:PHYSICAL_ACTION_DIM].astype(np.float64),
        cpu_ratios=_softmax(arr[CPU_LOGITS_SLICE], scale=logit_scale),
        tx_ratios=_softmax(arr[TX_LOGITS_SLICE], scale=logit_scale),
        drop_low_strength=float(arr[IDX_DROP_LOW]),
        cpu_value_weight=_unit_to_signed(arr[IDX_CPU_VALUE_WEIGHT]),
        cpu_urgency_weight=_unit_to_signed(arr[IDX_CPU_URGENCY_WEIGHT]),
        tx_value_weight=_unit_to_signed(arr[IDX_TX_VALUE_WEIGHT]),
        tx_urgency_weight=_unit_to_signed(arr[IDX_TX_URGENCY_WEIGHT]),
        pointing_mode=pointing_mode_from_unit(arr[IDX_POINTING]),
    )


def build_grouped_action(
    physical,
    *,
    cpu_class_logits=None,
    tx_class_logits=None,
    cpu_value_weight: float = 0.5,
    cpu_urgency_weight: float = 0.5,
    tx_value_weight: float = 0.5,
    tx_urgency_weight: float = 0.5,
    drop_low_strength: float = 0.0,
    pointing_unit: float = 0.5,
) -> np.ndarray:
    """按语义字段构造 15 维分组动作（供基线/工具按意图组装，避免裸下标错位）。

    cpu_class_logits / tx_class_logits 为长度 3 的 [high, mid, low]；None 时取均匀(0.5)。
    权重参数为 [0,1] 区间（0.5 = 中性 → signed 0）。
    """
    out = _neutral_action()
    phys = np.asarray(physical, dtype=np.float64).reshape(-1)
    m = min(phys.size, PHYSICAL_ACTION_DIM)
    out[:m] = phys[:m]
    if cpu_class_logits is not None:
        cl = np.asarray(cpu_class_logits, dtype=np.float64).reshape(-1)
        cl = np.resize(cl, VALUE_CLASS_COUNT) if cl.size != VALUE_CLASS_COUNT else cl
        out[CPU_LOGITS_SLICE] = cl
    if tx_class_logits is not None:
        tl = np.asarray(tx_class_logits, dtype=np.float64).reshape(-1)
        tl = np.resize(tl, VALUE_CLASS_COUNT) if tl.size != VALUE_CLASS_COUNT else tl
        out[TX_LOGITS_SLICE] = tl
    out[IDX_CPU_VALUE_WEIGHT] = cpu_value_weight
    out[IDX_CPU_URGENCY_WEIGHT] = cpu_urgency_weight
    out[IDX_TX_VALUE_WEIGHT] = tx_value_weight
    out[IDX_TX_URGENCY_WEIGHT] = tx_urgency_weight
    out[IDX_DROP_LOW] = drop_low_strength
    out[IDX_POINTING] = pointing_unit
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def default_grouped_action(physical_action, pointing_unit: float = 0.5) -> np.ndarray:
    """将遗留的3维物理动作扩展到完整分组动作（class logits 均匀、权重中性、不丢弃）。

    pointing_unit 控制指向模式([0,1]→IMAGE/DOWNLINK/SUN)。
    """
    return build_grouped_action(physical_action, pointing_unit=pointing_unit)


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

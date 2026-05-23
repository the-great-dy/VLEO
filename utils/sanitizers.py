"""策略-物理边界的共享数值清理器。"""

from __future__ import annotations

import numpy as np


def sanitize_scalar(
    value,
    *,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """将一个数值输入转换为有限浮点数，并可选地进行裁剪。"""
    pos = nan if posinf is None else posinf
    neg = nan if neginf is None else neginf
    cleaned = float(np.nan_to_num(value, nan=nan, posinf=pos, neginf=neg))
    if min_value is not None:
        cleaned = max(float(min_value), cleaned)
    if max_value is not None:
        cleaned = min(float(max_value), cleaned)
    return float(cleaned)


def sanitize_array(
    values,
    *,
    nan: float = 0.0,
    posinf: float = 1.0,
    neginf: float = 0.0,
    dtype=np.float64,
) -> tuple[np.ndarray, bool]:
    """返回有限的numpy数组以及原始数据是否完全有限。"""
    raw = np.asarray(values, dtype=dtype)
    finite = bool(np.all(np.isfinite(raw)))
    cleaned = np.nan_to_num(raw, nan=nan, posinf=posinf, neginf=neginf)
    return cleaned.astype(dtype, copy=False), finite


def sanitize_action(
    action,
    *,
    action_dim: int | None = None,
    dtype=np.float64,
) -> tuple[np.ndarray, bool, bool]:
    """
    清理连续动作并将其裁剪到[0, 1]^action_dim。

    Returns:
        clipped_action, original_all_finite, original_in_unit_box
    """
    raw = np.asarray(action, dtype=dtype)
    flat = raw.reshape(-1)
    original_finite = bool(np.all(np.isfinite(raw)))
    target_dim = int(flat.size if action_dim is None else action_dim)
    target_dim = max(target_dim, 1)
    original_shape_ok = raw.shape == (target_dim,)
    if flat.size < target_dim:
        # 缺失的高维任务选择动作使用 0 填充；经过 softmax 后等价于中性均分。
        raw = np.pad(flat, (0, target_dim - flat.size), mode="constant")
    elif flat.size > target_dim:
        raw = flat[:target_dim]
    else:
        raw = flat
    original_in_bounds = bool(
        original_shape_ok
        and original_finite
        and np.all((raw >= 0.0) & (raw <= 1.0))
    )
    cleaned = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(cleaned, 0.0, 1.0).astype(dtype, copy=False), original_finite, original_in_bounds

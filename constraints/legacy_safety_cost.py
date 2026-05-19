"""Compatibility shim for the old legacy_safety_cost module.

历史的 5 个 proc/dl 冗余 cost 函数 (efficiency / low_value_waste /
processed_backlog / unproductive_cpu / window_waste) 已经在重构中被
``constraints.safety_cost.compute_over_processing_cost`` 统一覆盖。

本文件保留为兼容性 shim:任何旧脚本仍然可以从
``from constraints.legacy_safety_cost import compute_xxx`` 导入,
但所有函数都会返回 0.0。
"""

from __future__ import annotations

from constraints.safety_cost import (
    compute_efficiency_cost,
    compute_low_value_waste_cost,
    compute_processed_backlog_cost,
    compute_unproductive_cpu_cost,
    compute_window_waste_cost,
)

__all__ = [
    "compute_efficiency_cost",
    "compute_low_value_waste_cost",
    "compute_processed_backlog_cost",
    "compute_unproductive_cpu_cost",
    "compute_window_waste_cost",
]

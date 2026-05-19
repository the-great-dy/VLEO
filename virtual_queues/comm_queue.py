"""
通信窗口虚拟队列和窗口利用率统计。
通信窗口虚拟队列 Q_C  ── 量纲统一修正版

修正说明：
  原版存在量纲不一致问题：λ_C 是无量纲归一化比值，μ_C 是 MB 数据量，
  两者直接相加减在数学上不成立。

  修正方案：将队列状态统一定义为"未传输数据体积（MB）"
    · λ_C(t)：单位时间产生的数据增量（MB/slot），= 数据到达量 A(t)
    · μ_C(t)：单位时间内通信窗口可传输数据量（MB/slot），= min(C_k*dt/8, backlog)
    · 队列更新：Q_C[t+1] = max(Q_C[t] + λ_C(t) - μ_C(t)*1[w=1], 0)
    · 稳定条件：E[μ_C * 1[w=1]] >= E[λ_C] + ε_C

  量纲验证：Q_C [MB]，λ_C [MB/slot]，μ_C [MB/slot]，三者完全一致。
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    # 降低导入劫持风险：不要把项目路径插到最高优先级
    sys.path.append(_PROJECT_ROOT)
import numpy as np
from config import QUEUE_CONFIG
from utils.sanitizers import sanitize_scalar
from virtual_queues.base_queue import BaseVirtualQueue


class CommWindowQueue(BaseVirtualQueue):
    """
    通信窗口虚拟队列 Q_C（论文贡献点C1核心）

    物理含义：队列值代表"当前尚未利用通信窗口传回地面的数据积压量（MB）"
    · 有通信窗口且传输充分 → Q_C 减小
    · 有通信窗口但传输能力不足（alpha_tx 低）→ Q_C 增大
    · 无通信窗口 → Q_C 随新数据到达缓慢增大
    """

    def __init__(self):
        super().__init__(
            QUEUE_CONFIG.get("comm_queue_max", 500.0),
            lyapunov_weight_scale=QUEUE_CONFIG.get("comm_weight_V", 15.0),
        )  # 单位 MB

        # 统计
        self.total_tx_mb         = 0.0
        self.total_arrival_mb    = 0.0
        self.window_steps        = 0
        self.total_steps         = 0

    def reset(self):
        self._reset_value(0.0)
        self.total_tx_mb         = 0.0
        self.total_arrival_mb    = 0.0
        self.window_steps        = 0
        self.total_steps         = 0

    def update(self,
               data_arrival_mb: float,    # λ_C(t)：本时隙新产生/处理的数据量 (MB)
               tx_capacity_mb: float,     # μ_C(t)：本时隙最大可传量 (MB) = C_k*dt/8
               in_window: bool,
               alpha_tx: float = 1.0,     # 发射机功率比例 [0,1]
               rf_capacity_mb: float = None, # 本时隙发射机物理速率上限 (MB)
               dropped_mb: float = 0.0,
               actual_tx_override_mb: float | None = None,
               ) -> dict:
        """
        更新通信窗口虚拟队列

        Args:
            data_arrival_mb : 本时隙数据到达量 λ_C(t)，单位 MB
            tx_capacity_mb  : 本时隙信道可传上限 μ_C_max(t)，单位 MB（= C_k(t)*Δt/8）
            in_window       : 是否在地面站接触窗口内
            alpha_tx        : 发射机功率分配比例，决定实际使用多少信道容量
            rf_capacity_mb  : 发射机功率对应的物理下传上限，单位 MB

        Returns:
            dict 包含队列状态、漂移、实际下传量等
        """
        self._begin_update()
        self.total_steps += 1

        # ── λ_C：本时隙数据到达量（MB） ──────────────────────────
        # 队列状态是训练目标的一部分，所有外部输入先压成非负有限 MB。
        lambda_c = sanitize_scalar(
            data_arrival_mb,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
            min_value=0.0,
        )        # 单位：MB
        self.total_arrival_mb += lambda_c
        tx_capacity_mb = sanitize_scalar(
            tx_capacity_mb,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
            min_value=0.0,
        )
        alpha_tx = sanitize_scalar(
            alpha_tx,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
            min_value=0.0,
            max_value=1.0,
        )
        external_drop = sanitize_scalar(
            dropped_mb,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
            min_value=0.0,
        )
        actual_external_drop = min(external_drop, self.value)

        # ── μ_C：本时隙实际下传量（MB） ──────────────────────────
        # 只有在窗口内才能下传；受 alpha_tx 和信道容量双重约束
        mu_c = 0.0
        actual_tx_mb = 0.0
        link_limited_mb = 0.0
        rf_limited_mb = 0.0
        max_tx = 0.0
        if in_window:
            self.window_steps += 1
            # 实际下传必须同时满足地面链路容量和发射机物理速率上限。
            # tx_capacity_mb 来自几何/链路预算；rf_capacity_mb 来自 P_tx 对应的射频侧最大速率。
            link_limited_mb = alpha_tx * tx_capacity_mb
            if rf_capacity_mb is None:
                rf_limited_mb = np.inf
            else:
                rf_limited_mb = sanitize_scalar(
                    rf_capacity_mb,
                    nan=0.0,
                    posinf=np.inf,
                    neginf=0.0,
                    min_value=0.0,
                )
            max_tx = min(link_limited_mb, rf_limited_mb)          # 单位：MB
            requested_tx_mb = max_tx
            if actual_tx_override_mb is not None:
                requested_tx_mb = sanitize_scalar(
                    actual_tx_override_mb,
                    nan=0.0,
                    posinf=max_tx,
                    neginf=0.0,
                    min_value=0.0,
                )
            actual_tx_mb = min(requested_tx_mb, max_tx,
                               self.value - actual_external_drop + lambda_c)
            mu_c = actual_tx_mb
            self.total_tx_mb += actual_tx_mb

        # ── 队列更新：Q_C[t+1] = max(Q_C[t] + λ_C - μ_C * 1[w], 0) ──
        # processed queue 是有限缓存；超过上限的部分视为数据价值丢失。
        next_value_uncapped = max(
            self.value - actual_external_drop + lambda_c - mu_c, 0.0)
        overflow_mb = max(next_value_uncapped - self.max_value, 0.0)
        self._set_value(next_value_uncapped)
        queue_ratio = self.value / max(self.max_value, 1e-6)
        urgency = float(np.clip(queue_ratio, 0.0, 3.0))

        return {
            "queue_value"   : float(self.value),
            "drift"         : float(self.drift),
            "lambda_c_mb"   : float(lambda_c),      # 本步到达量 (MB)
            "mu_c_mb"       : float(mu_c),           # 本步下传量 (MB)
            "actual_tx_mb"  : float(actual_tx_mb),
            "externally_dropped_mb": float(actual_external_drop),
            "link_limited_tx_mb": float(link_limited_mb),
            "rf_limited_tx_mb": float(rf_limited_mb) if np.isfinite(rf_limited_mb) else float("inf"),
            "effective_tx_capacity_mb": float(max_tx) if np.isfinite(max_tx) else 0.0,
            "in_window"     : bool(in_window),
            "is_stable"     : queue_ratio < 0.8,
            "is_over_capacity": overflow_mb > 0.0,
            "overflow_mb"   : float(overflow_mb),
            "dropped_mb"    : float(overflow_mb),
            "urgency"       : urgency,
            "urgency_raw"   : float(queue_ratio),
            "window_ratio"  : self.window_steps / max(self.total_steps, 1),
            "total_tx_mb"   : float(self.total_tx_mb),
        }


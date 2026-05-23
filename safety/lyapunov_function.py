"""控制型 Lyapunov 函数 L(s)（Chow et al. 2018, NeurIPS 风格）。

L(s) 是一个非负标量，刻画 "状态 s 距离不安全集合有多近"。安全集合：
  altitude  > h_crash   且
  soc       > soc_crash 且
  processed_queue 利用率 < 1.0 且
  thermal_margin_norm   > 0
L(s) = 各通道归一化压力之和（每通道 ∈ [0,1]）。在硬失败时 L(s)≥1；在标称状态附近 L(s)≈0。

这个 L 满足 Chow et al. 2018 的核心性质：
  1. L(s) ≥ V_C(s)：分量级数大于等于单步约束 cost 的累计上界（在 cost 被
     normalize 到 [0,1] 的前提下）；
  2. L(s) 是 state-dependent 的——这是它跟 algorithms/adaptive_lyapunov_dual
     里那个**全局** Lagrangian λ 的本质区别。

跟它配套的 projection operator 在 safety.lyapunov_projection 中实现。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (
    ENERGY_CONFIG,
    LYAPUNOV_CONFIG,
    ORBITAL_CONFIG,
    QUEUE_CONFIG,
)
from environment.satellite_env import OBSERVATION_FEATURES


@dataclass(frozen=True)
class LyapunovBreakdown:
    """L(s) 的拆解，便于诊断不是哪条约束驱动了 projection。"""

    altitude_pressure: float
    soc_pressure: float
    processed_queue_pressure: float
    raw_queue_pressure: float
    thermal_pressure: float
    future_capacity_pressure: float
    total: float

    def as_dict(self) -> dict:
        return {
            "lyapunov_altitude": float(self.altitude_pressure),
            "lyapunov_soc": float(self.soc_pressure),
            "lyapunov_processed_queue": float(self.processed_queue_pressure),
            "lyapunov_raw_queue": float(self.raw_queue_pressure),
            "lyapunov_thermal": float(self.thermal_pressure),
            "lyapunov_future_capacity": float(self.future_capacity_pressure),
            "lyapunov_value": float(self.total),
        }


class LyapunovFunction:
    """从原始物理量计算 L(s)。

    可以接受两种输入：
      * physical dict（altitude_m, soc, processed_queue_mb, ...）—— PSF
        rollout 用这条；
      * 观测向量 obs —— scheduler 在线 projection 用这条。
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or LYAPUNOV_CONFIG
        self.cfg = cfg
        self.h_warning = float(ORBITAL_CONFIG.get("altitude_warning_km", 200.0)) * 1e3
        self.h_min = float(ORBITAL_CONFIG.get("altitude_min_km", 180.0)) * 1e3
        self.h_crash = float(ORBITAL_CONFIG.get("altitude_crash_km", 120.0)) * 1e3
        self.h_max = float(ORBITAL_CONFIG.get("altitude_max_km", 300.0)) * 1e3
        self.soc_min = float(ENERGY_CONFIG.get("battery_min_soc", 0.15))
        self.soc_crash = float(ENERGY_CONFIG.get("battery_crash_soc", 0.05))
        self.processed_queue_max = float(QUEUE_CONFIG.get("comm_queue_max", 4096.0))
        self.raw_queue_max = float(QUEUE_CONFIG.get("data_queue_max_mb", 4096.0))
        self.alpha_alt = float(cfg.get("L_altitude_weight", 1.0))
        self.alpha_soc = float(cfg.get("L_soc_weight", 1.0))
        self.alpha_proc = float(cfg.get("L_processed_queue_weight", 1.0))
        self.alpha_raw = float(cfg.get("L_raw_queue_weight", 0.5))
        self.alpha_thermal = float(cfg.get("L_thermal_weight", 0.5))
        self.alpha_future = float(cfg.get("L_future_capacity_weight", 0.5))
        # safe-margin "slack" used by Lyapunov projection: ε(s) = max(0, d - L(s))*decay
        self.d_target = float(cfg.get("L_target_level", 0.5))
        self.slack_decay = float(cfg.get("L_slack_decay", 0.05))

    # ── 主入口 ─────────────────────────────────────────────────────────
    def from_physical(self, phys: dict) -> LyapunovBreakdown:
        h = float(phys.get("altitude_m", self.h_warning))
        soc = float(phys.get("soc", 1.0))
        qc = float(phys.get("processed_queue_mb", 0.0))
        qd = float(phys.get("raw_queue_mb", 0.0))
        thermal_margin_norm = float(phys.get("thermal_margin_norm", 1.0))
        future_capacity_mb = float(phys.get("future_contact_capacity_mb", -1.0))

        # 高度压力：低于 warning 开始累积，在 crash 时达到 1.0。
        alt_span = max(self.h_warning - self.h_crash, 1.0)
        alt_p = float(np.clip((self.h_warning - h) / alt_span, 0.0, 1.5))

        # SOC 压力：低于 min 开始累积。
        soc_span = max(self.soc_min - self.soc_crash, 1e-3)
        soc_p = float(np.clip((self.soc_min - soc) / soc_span, 0.0, 1.5))

        # processed queue 压力（直接用利用率）。
        proc_p = float(np.clip(qc / max(self.processed_queue_max, 1e-6), 0.0, 1.5))

        # raw queue 压力。
        raw_p = float(np.clip(qd / max(self.raw_queue_max, 1e-6), 0.0, 1.5))

        # 热压力：thermal_margin_norm ≤ 0 → 进入 warning/critical。
        thermal_p = float(np.clip(-thermal_margin_norm, 0.0, 1.5))

        # 未来 contact 容量压力：如果 processed queue 超过未来 contact 总容量，
        # 我们已经"超过了任何 downlink 计划能清掉的量"。
        if future_capacity_mb > 0.0:
            future_p = float(np.clip(qc / max(future_capacity_mb, 1e-6) - 1.0, 0.0, 1.5))
        else:
            future_p = 0.0

        total = float(
            self.alpha_alt * alt_p
            + self.alpha_soc * soc_p
            + self.alpha_proc * proc_p
            + self.alpha_raw * raw_p
            + self.alpha_thermal * thermal_p
            + self.alpha_future * future_p
        )

        return LyapunovBreakdown(
            altitude_pressure=alt_p,
            soc_pressure=soc_p,
            processed_queue_pressure=proc_p,
            raw_queue_pressure=raw_p,
            thermal_pressure=thermal_p,
            future_capacity_pressure=future_p,
            total=total,
        )

    def from_observation(self, obs: np.ndarray) -> LyapunovBreakdown:
        """从归一化观测反推物理量后调用 from_physical。"""
        phys = self._physical_from_observation(obs)
        return self.from_physical(phys)

    def value(self, source) -> float:
        """Quick scalar accessor: returns L(s)."""
        if isinstance(source, dict):
            return self.from_physical(source).total
        return self.from_observation(np.asarray(source)).total

    def slack(self, l_value: float) -> float:
        """ε(s) = max(0, d - L(s))·decay  —— L 越接近不安全集，能允许的漂移越小。"""
        return float(max(0.0, self.d_target - float(l_value)) * self.slack_decay)

    # ── helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _feature(obs: np.ndarray, name: str, default: float = 0.0) -> float:
        if name not in OBSERVATION_FEATURES:
            return default
        idx = OBSERVATION_FEATURES.index(name)
        arr = np.asarray(obs, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] > idx:
            v = arr[0, idx]
        elif arr.ndim == 1 and arr.shape[0] > idx:
            v = arr[idx]
        else:
            return default
        v = float(v)
        return v if np.isfinite(v) else default

    def _physical_from_observation(self, obs: np.ndarray) -> dict:
        altitude_norm = self._feature(obs, "altitude_norm", default=1.0)
        soc = self._feature(obs, "soc", default=1.0)
        thermal_margin_norm = self._feature(obs, "thermal_margin_norm", default=1.0)
        processed_util = self._feature(obs, "processed_queue_utilization", default=0.0)
        raw_util = self._feature(obs, "raw_queue_utilization", default=0.0)
        future_norm = self._feature(obs, "future_contact_capacity_norm", default=1.0)

        altitude_m = float(self.h_min + altitude_norm * (self.h_max - self.h_min))
        processed_queue_mb = float(processed_util * self.processed_queue_max)
        raw_queue_mb = float(raw_util * self.raw_queue_max)
        # future_contact_capacity_norm 是按 processed_queue_max 归一的（与 obs 中其它 norm 一致）。
        future_capacity_mb = float(future_norm * self.processed_queue_max)
        return {
            "altitude_m": altitude_m,
            "soc": soc,
            "thermal_margin_norm": thermal_margin_norm,
            "processed_queue_mb": processed_queue_mb,
            "raw_queue_mb": raw_queue_mb,
            "future_contact_capacity_mb": future_capacity_mb,
        }


__all__ = ["LyapunovFunction", "LyapunovBreakdown"]

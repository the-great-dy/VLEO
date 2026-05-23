"""Predictive Safety Filter (Wabersich & Zeilinger 2018, "Linear Model
Predictive Safety Certification") 的简化实现。

定义（论文 Eq. 3）：
    a* = argmin_a ||a - a_raw||²
          s.t.  存在长度-K 的可行轨迹 {x_0=s, x_1, ..., x_K}
                ∀k: x_k ∈ X_safe
                x_K  ∈ X_terminal  (backup controller 的吸引域)
                x_{k+1} = f(x_k, u_k),  u_0 = a,  u_k = π_backup(x_k) (k≥1)

实现策略：
  1. 用 SafetyDynamicsPredictor 跑一遍 (a_raw, backup, ..., backup)；若轨迹
     全在 X_safe 内，直接返回 a_raw。
  2. 否则在 a_raw 与 a_backup 之间二分线性搜索 α∈[0,1]，找到最大的 α 使
     α·a_raw + (1-α)·a_backup 通过 K 步可行性检查。
  3. 若 α=0 仍违反 → backup 自身也救不了；返回 backup 并标记 failure。

Backup controller (state-dependent)：
  * altitude < altitude_warning → α_prop = 1（最大推力），其他 0；
  * SOC < soc_warning              → 全 0 让电池靠太阳能恢复；
  * processed_queue > 0.8          → α_cpu = 0, α_tx = 1（只清空队列）；
  * 默认                              → 全 0 (drift)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import ENERGY_CONFIG, ORBITAL_CONFIG, PSF_CONFIG, QUEUE_CONFIG
from safety.dynamics_predictor import SafetyDynamicsPredictor, PredictedState
from utils.action_space import PHYSICAL_ACTION_DIM


@dataclass(frozen=True)
class PSFResult:
    action: np.ndarray
    intervened: bool
    raw_safe: bool
    backup_safe: bool
    interpolation_alpha: float
    horizon_used: int
    worst_altitude_m: float
    worst_soc: float
    worst_processed_queue_mb: float
    worst_thermal_margin: float


def make_backup_action(
    state: dict,
    action_dim: int,
    *,
    altitude_warning_m: float,
    soc_warning: float,
    processed_queue_max_mb: float,
) -> np.ndarray:
    """状态相关的 backup 控制器，写成 [α_prop, α_cpu, α_tx, ...]。"""
    out = np.zeros(action_dim, dtype=np.float64)
    altitude = float(state.get("altitude_m", altitude_warning_m))
    soc = float(state.get("soc", 1.0))
    qc = float(state.get("processed_queue_mb", 0.0))

    if altitude < altitude_warning_m:
        out[0] = 1.0  # 高度危险 → 全推力。
    if soc < soc_warning:
        out[0] = 0.0  # SOC 危险 → 切掉所有耗电负载，靠太阳能恢复。
        if PHYSICAL_ACTION_DIM > 1:
            out[1] = 0.0
        if PHYSICAL_ACTION_DIM > 2:
            out[2] = 0.0
        return out
    if qc / max(processed_queue_max_mb, 1e-6) > 0.8 and bool(state.get("in_window", False)):
        if PHYSICAL_ACTION_DIM > 1:
            out[1] = 0.0
        if PHYSICAL_ACTION_DIM > 2:
            out[2] = 1.0  # 队列满 + 通信窗口 → 全力下传。
        return out
    return out


class PredictiveSafetyFilter:
    """K 步 PSF：扫描候选 α 找最大 α 使 (α·raw + (1-α)·backup) 通过约束。"""

    def __init__(
        self,
        predictor: SafetyDynamicsPredictor | None = None,
        *,
        K: int | None = None,
        cfg: dict | None = None,
        line_search_steps: int = 6,
    ):
        cfg = cfg or PSF_CONFIG
        self.predictor = predictor or SafetyDynamicsPredictor()
        self.K = int(K if K is not None else cfg.get("horizon_steps", 5))
        self.line_search_steps = int(max(1, line_search_steps))

        self.h_warning = float(ORBITAL_CONFIG.get("altitude_warning_km", 200.0)) * 1e3
        self.h_min = float(ORBITAL_CONFIG.get("altitude_min_km", 180.0)) * 1e3
        self.h_crash = float(ORBITAL_CONFIG.get("altitude_crash_km", 120.0)) * 1e3
        self.soc_min = float(ENERGY_CONFIG.get("battery_min_soc", 0.15))
        self.soc_crash = float(ENERGY_CONFIG.get("battery_crash_soc", 0.05))
        self.processed_queue_max_mb = float(QUEUE_CONFIG.get("comm_queue_max", 4096.0))

        # PSF 边界设置成"距硬边界一个 margin"，提前介入而不是贴线干预。
        self.h_safe_min = self.h_crash + float(cfg.get("altitude_trigger_margin_m", 15_000.0))
        self.soc_safe_min = self.soc_crash + float(cfg.get("soc_trigger_margin", 0.05))
        self.h_warn_for_backup = self.h_min  # backup 触发用 altitude_min。
        self.soc_warn_for_backup = self.soc_min

    # ── 安全集合判定 ──────────────────────────────────────────────────
    def _state_in_safe_set(self, state: PredictedState) -> bool:
        if state.altitude_m <= self.h_safe_min:
            return False
        if state.soc <= self.soc_safe_min:
            return False
        if state.processed_queue_mb >= self.processed_queue_max_mb:
            return False
        if state.thermal_margin_norm <= -0.99:
            return False
        return True

    def _trajectory_safe(self, traj: list[PredictedState]) -> tuple[bool, dict]:
        worst_alt = min((s.altitude_m for s in traj), default=float("inf"))
        worst_soc = min((s.soc for s in traj), default=1.0)
        worst_qc = max((s.processed_queue_mb for s in traj), default=0.0)
        worst_thermal = min((s.thermal_margin_norm for s in traj), default=1.0)
        ok = all(self._state_in_safe_set(s) for s in traj)
        return ok, {
            "worst_altitude_m": float(worst_alt),
            "worst_soc": float(worst_soc),
            "worst_processed_queue_mb": float(worst_qc),
            "worst_thermal_margin": float(worst_thermal),
        }

    # ── 主接口 ─────────────────────────────────────────────────────────
    def filter(self, action: np.ndarray, state_phys: dict) -> PSFResult:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        action_dim = int(action.size)
        backup = make_backup_action(
            state_phys,
            action_dim=action_dim,
            altitude_warning_m=self.h_warn_for_backup,
            soc_warning=self.soc_warn_for_backup,
            processed_queue_max_mb=self.processed_queue_max_mb,
        )

        raw_traj = self._rollout(state_phys, action, backup)
        raw_ok, raw_worst = self._trajectory_safe(raw_traj)
        if raw_ok:
            return PSFResult(
                action=action.astype(np.float32),
                intervened=False,
                raw_safe=True,
                backup_safe=True,
                interpolation_alpha=1.0,
                horizon_used=self.K,
                worst_altitude_m=raw_worst["worst_altitude_m"],
                worst_soc=raw_worst["worst_soc"],
                worst_processed_queue_mb=raw_worst["worst_processed_queue_mb"],
                worst_thermal_margin=raw_worst["worst_thermal_margin"],
            )

        # 二分搜索最大可行 α。如果 α=0 (backup) 都不行，直接返回 backup（fallback 失败）。
        backup_traj = self._rollout(state_phys, backup, backup)
        backup_ok, backup_worst = self._trajectory_safe(backup_traj)
        if not backup_ok:
            return PSFResult(
                action=backup.astype(np.float32),
                intervened=True,
                raw_safe=False,
                backup_safe=False,
                interpolation_alpha=0.0,
                horizon_used=self.K,
                worst_altitude_m=backup_worst["worst_altitude_m"],
                worst_soc=backup_worst["worst_soc"],
                worst_processed_queue_mb=backup_worst["worst_processed_queue_mb"],
                worst_thermal_margin=backup_worst["worst_thermal_margin"],
            )

        best_alpha = 0.0
        best_traj = backup_traj
        best_worst = backup_worst
        # bisection: low=0(可行 backup), high=1(不可行 raw)。
        lo, hi = 0.0, 1.0
        for _ in range(self.line_search_steps):
            mid = 0.5 * (lo + hi)
            mixed = mid * action + (1.0 - mid) * backup
            mixed[:PHYSICAL_ACTION_DIM] = np.clip(mixed[:PHYSICAL_ACTION_DIM], 0.0, 1.0)
            mixed_traj = self._rollout(state_phys, mixed, backup)
            mixed_ok, mixed_worst = self._trajectory_safe(mixed_traj)
            if mixed_ok:
                best_alpha = mid
                best_traj = mixed_traj
                best_worst = mixed_worst
                lo = mid
            else:
                hi = mid

        final_action = best_alpha * action + (1.0 - best_alpha) * backup
        final_action[:PHYSICAL_ACTION_DIM] = np.clip(
            final_action[:PHYSICAL_ACTION_DIM], 0.0, 1.0)
        return PSFResult(
            action=final_action.astype(np.float32),
            intervened=True,
            raw_safe=False,
            backup_safe=True,
            interpolation_alpha=float(best_alpha),
            horizon_used=self.K,
            worst_altitude_m=best_worst["worst_altitude_m"],
            worst_soc=best_worst["worst_soc"],
            worst_processed_queue_mb=best_worst["worst_processed_queue_mb"],
            worst_thermal_margin=best_worst["worst_thermal_margin"],
        )

    def _rollout(
        self,
        state_phys: dict,
        first_action: np.ndarray,
        backup_action: np.ndarray,
    ) -> list[PredictedState]:
        actions: list[np.ndarray] = [first_action]
        # 后续 K-1 步使用 backup（也可以做 state-dependent backup，但此处先固定）。
        for _ in range(max(0, self.K - 1)):
            actions.append(backup_action)
        return self.predictor.rollout(state_phys, actions)


__all__ = ["PredictiveSafetyFilter", "PSFResult", "make_backup_action"]

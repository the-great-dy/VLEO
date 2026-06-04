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
from utils.action_space import (PHYSICAL_ACTION_DIM, pointing_unit_for_mode,
                                 POINTING_DOWNLINK, POINTING_SUN, IDX_POINTING)


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
    thermal_warning_margin: float = 0.10,
) -> np.ndarray:
    """状态相关的 backup 控制器，写成 [α_prop, α_cpu, α_tx, ...]。

    优先级（先触发先生效）：
      thermal_margin < 0.10 → 切 CPU/TX 散热（热是 141k eval 测出的 crash 主因）
      soc < soc_warning     → 切所有耗电负载，让太阳能充电（次主因）
      altitude < warning_m  → 全推力（防 deorbit）
      processed_queue > 80% + in_window → 全力下传（清队列）
    """
    out = np.zeros(action_dim, dtype=np.float64)
    altitude = float(state.get("altitude_m", altitude_warning_m))
    soc = float(state.get("soc", 1.0))
    qc = float(state.get("processed_queue_mb", 0.0))
    thermal_margin = float(state.get("thermal_margin_norm", 1.0))

    # [SAFETY-REAL] backup 也必须设置指向模式(pointing 维),否则"切TX/对日充电"的意图在姿态门控下落空。
    def _set_point(mode):
        if action_dim > IDX_POINTING:
            out[IDX_POINTING] = pointing_unit_for_mode(mode)

    # 热保护最高优先（一旦过热会很快进 failure）。CPU/TX 全部切掉等冷却,对日散热/充电。
    if thermal_margin < thermal_warning_margin:
        out[0] = 0.0
        if PHYSICAL_ACTION_DIM > 1:
            out[1] = 0.0
        if PHYSICAL_ACTION_DIM > 2:
            out[2] = 0.0
        _set_point(POINTING_SUN)
        return out

    if altitude < altitude_warning_m:
        out[0] = 1.0  # 高度危险 → 全推力。
        _set_point(POINTING_SUN)  # 对日充电以支撑推进功率
    if soc < soc_warning:
        out[0] = 0.0  # SOC 危险 → 切掉所有耗电负载，靠太阳能恢复。
        if PHYSICAL_ACTION_DIM > 1:
            out[1] = 0.0
        if PHYSICAL_ACTION_DIM > 2:
            out[2] = 0.0
        _set_point(POINTING_SUN)  # 必须对日才能充电
        return out
    if qc / max(processed_queue_max_mb, 1e-6) > 0.8 and bool(state.get("in_window", False)):
        if PHYSICAL_ACTION_DIM > 1:
            out[1] = 0.0
        if PHYSICAL_ACTION_DIM > 2:
            out[2] = 1.0  # 队列满 + 通信窗口 → 全力下传。
        _set_point(POINTING_DOWNLINK)  # 必须对站才能真正下传
        return out
    _set_point(POINTING_SUN)  # 默认安全:对日充电
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

        # ── 长视野解析式预测（C 改动）────────────────────────────────
        # K=10 步 rollout 看 100 秒，但热常数（thermal_capacity=18000 J/K）和
        # SOC 漂移常数（500Wh / 116W ≈ 4.3 小时）都是分钟级。在 K 步之外用
        # 解析式向 long_horizon_steps 推一次（默认 540 步=轨道周期），看 thermal/SOC
        # 是否会在 contact 窗口之前就进 warning，是就把 raw_action 视为 unsafe。
        self.long_horizon_enabled = bool(cfg.get("long_horizon_enabled", True))
        self.long_horizon_steps = int(cfg.get("long_horizon_steps", 540))
        self.long_horizon_thermal_margin_floor = float(
            cfg.get("long_horizon_thermal_margin_floor", 0.10))
        self.long_horizon_soc_floor = float(
            cfg.get("long_horizon_soc_floor", self.soc_min))

        # 解析式预测用的热模型常数（从 THERMAL_CONFIG 读，避免重复定义）。
        from config import THERMAL_CONFIG as _THC
        self.thermal_capacity_j_per_k = float(_THC.get("thermal_capacity_j_per_k", 18000.0))
        self.electronics_heat_fraction = float(_THC.get("electronics_heat_fraction", 0.35))
        self.propulsion_heat_fraction = float(_THC.get("propulsion_heat_fraction", 0.04))
        self.thermal_warning_temp_c = float(_THC.get("warning_temp_c", 45.0))
        self.thermal_max_temp_c = float(_THC.get("max_temp_c", 55.0))
        # warning 与 normal 的 thermal_margin_norm 差距：约等于 (max - warning) / max。
        # warning_temp=45, max=55 → 满 margin 1.0 对应 < warning，0.0 对应 = warning。
        # 解析式按线性外推 → margin_drop_per_step ≈ heat_in_w * dt / capacity / (max-warning)。
        self.thermal_margin_per_kelvin = 1.0 / max(
            self.thermal_max_temp_c - self.thermal_warning_temp_c, 1e-3)
        # SOC 漂移：dSOC/step ≈ (P_solar - P_load) * dt / 3600 / battery_capacity_wh。
        self.battery_capacity_wh = float(self.predictor.battery_capacity_wh)
        self.dt_s = float(self.predictor.dt_s)

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

    # ── 长视野解析式预测（C 改动）────────────────────────────────────
    def _long_horizon_safety_check(
        self,
        state_phys: dict,
        first_action: np.ndarray,
    ) -> tuple[bool, dict]:
        """假设连续应用 first_action 长达 long_horizon_steps 步，估算 thermal/SOC 是否进 warning。

        Return (safe, info)。这是**单变量线性外推**，不是严格 rollout——故意保守
        点（高估漂移），让 PSF 在风险还小时就提前介入。
        """
        if not self.long_horizon_enabled or self.long_horizon_steps <= 0:
            return True, {}

        # 当前物理量
        soc = float(state_phys.get("soc", 1.0))
        thermal_margin = float(state_phys.get("thermal_margin_norm", 1.0))
        sunlit_fraction = float(state_phys.get("sunlit_fraction", 0.5))

        # 假设动作 = first_action：估算稳态净功率与产热
        a = np.asarray(first_action, dtype=np.float64).reshape(-1)
        if a.size < PHYSICAL_ACTION_DIM:
            a = np.pad(a, (0, PHYSICAL_ACTION_DIM - a.size))
        a = np.clip(a[:PHYSICAL_ACTION_DIM], 0.0, 1.0)
        p_prop = a[0] * float(self.predictor.power_weights[0])
        p_cpu = a[1] * float(self.predictor.power_weights[1])
        p_tx = a[2] * float(self.predictor.power_weights[2])
        p_load = p_prop + p_cpu + p_tx + self.predictor.baseline_power_w

        # 净功率（用 sunlit_fraction 名义太阳能；不考虑 eclipse 变化的保守估计）。
        p_solar = (self.predictor.solar_panel_power_w
                   * self.predictor.solar_efficiency * sunlit_fraction)
        p_net = p_solar - p_load

        # SOC 漂移：单步 ΔSOC ≈ p_net * dt / 3600 / capacity。
        d_soc_per_step = p_net * self.dt_s / 3600.0 / max(self.battery_capacity_wh, 1e-6)
        soc_at_horizon = soc + d_soc_per_step * self.long_horizon_steps
        soc_safe = soc_at_horizon >= self.long_horizon_soc_floor

        # 热漂移：CPU/Tx/bus 按电子热折算，推进只计 PPU/安装耦合的小比例；
        # 这里仍忽略辐射散热，作为保守上界。
        electronics_heat_w = (
            p_cpu + p_tx + self.predictor.baseline_power_w
        ) * self.electronics_heat_fraction
        propulsion_heat_w = p_prop * self.propulsion_heat_fraction
        internal_heat_w = electronics_heat_w + propulsion_heat_w
        d_temp_per_step_k = internal_heat_w * self.dt_s / max(self.thermal_capacity_j_per_k, 1e-6)
        d_margin_per_step = d_temp_per_step_k * self.thermal_margin_per_kelvin
        thermal_at_horizon = thermal_margin - d_margin_per_step * self.long_horizon_steps
        thermal_safe = thermal_at_horizon >= self.long_horizon_thermal_margin_floor

        safe = bool(soc_safe and thermal_safe)
        info = {
            "long_horizon_steps": int(self.long_horizon_steps),
            "long_horizon_soc_predicted": float(soc_at_horizon),
            "long_horizon_thermal_margin_predicted": float(thermal_at_horizon),
            "long_horizon_soc_violation": bool(not soc_safe),
            "long_horizon_thermal_violation": bool(not thermal_safe),
            "long_horizon_p_net_w": float(p_net),
            "long_horizon_electronics_heat_w": float(electronics_heat_w),
            "long_horizon_propulsion_heat_w": float(propulsion_heat_w),
            "long_horizon_internal_heat_w": float(internal_heat_w),
        }
        return safe, info

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
        # C 改动：K 步 rollout 通过后，再用解析式向 long_horizon_steps 推一次。
        # 如果长视野预测 thermal/SOC 会进 warning，把 raw 视为不安全，触发 line search
        # 寻找混合 α（不会全切 backup，毕竟 K 步内还是 OK 的）。
        long_safe, long_info = self._long_horizon_safety_check(state_phys, action)
        if raw_ok and long_safe:
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
        # 长视野不安全但 K 步通过 → 当作 raw_ok=False 处理，进入 line search。
        if raw_ok and not long_safe:
            raw_ok = False

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

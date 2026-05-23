"""轻量一步前向动力学预测器，供 Lyapunov projection 与 PSF rollout 共用。

只复用真正影响硬安全集的子系统：
  * 高度 (OrbitalDynamics.step)
  * 电池 SOC (近似 BatteryModel.step：能量平衡 + 充放电效率)
  * processed queue：当前 dt 内 CPU 处理 - TX 下传
  * thermal_margin_norm：线性 cool-down，CPU/TX 通道造成热量产生（保守估计）

它**故意不**复现 reward critic、task value、ground station 计划等环境细节——
我们只需要它给 safety set 提供保守的一步预测，而不需要复现奖励。预测器对
"安全方向"持悲观立场：drag 用名义值，TX 容量用 in_window 标志，热散用最低速率。

接口与 environment.satellite_env 的真实状态保持一致：
  state_phys = {
      altitude_m, soc, processed_queue_mb, raw_queue_mb,
      thermal_margin_norm, in_window, tx_capacity_mbps,
      sunlit_fraction, future_contact_capacity_mb,
  }
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (
    ENERGY_CONFIG,
    ORBITAL_CONFIG,
    QUEUE_CONFIG,
    THERMAL_CONFIG,
    TRAIN_CONFIG,
)
from environment.orbital_dynamics import OrbitalDynamics
from utils.action_space import PHYSICAL_ACTION_DIM


@dataclass
class PredictedState:
    """一步预测后的物理状态。"""

    altitude_m: float
    soc: float
    processed_queue_mb: float
    raw_queue_mb: float
    thermal_margin_norm: float
    in_window: bool
    tx_capacity_mbps: float
    sunlit_fraction: float
    future_contact_capacity_mb: float

    def to_dict(self) -> dict:
        return {
            "altitude_m": float(self.altitude_m),
            "soc": float(self.soc),
            "processed_queue_mb": float(self.processed_queue_mb),
            "raw_queue_mb": float(self.raw_queue_mb),
            "thermal_margin_norm": float(self.thermal_margin_norm),
            "in_window": bool(self.in_window),
            "tx_capacity_mbps": float(self.tx_capacity_mbps),
            "sunlit_fraction": float(self.sunlit_fraction),
            "future_contact_capacity_mb": float(self.future_contact_capacity_mb),
        }


class SafetyDynamicsPredictor:
    """名义动力学的一步预测器；Lyapunov projection 与 PSF 共用。"""

    def __init__(
        self,
        *,
        dt_s: float | None = None,
        orbital: OrbitalDynamics | None = None,
        power_weights: np.ndarray | None = None,
        baseline_power_w: float | None = None,
    ):
        self.dt_s = float(dt_s if dt_s is not None else TRAIN_CONFIG.get("time_slot_s", 10.0))
        self.orbital = orbital if orbital is not None else OrbitalDynamics()

        if power_weights is None:
            power_weights = np.array([
                ENERGY_CONFIG["power_propulsion_max_w"],
                ENERGY_CONFIG["power_cpu_max_w"],
                ENERGY_CONFIG["power_tx_max_w"],
            ], dtype=np.float64)
        self.power_weights = np.asarray(power_weights, dtype=np.float64).reshape(-1)
        self.baseline_power_w = float(
            baseline_power_w if baseline_power_w is not None
            else ENERGY_CONFIG.get("power_baseline_w", 30.0))

        self.solar_panel_power_w = float(ENERGY_CONFIG.get("solar_panel_power_w", 200.0))
        self.solar_efficiency = float(ENERGY_CONFIG.get("solar_efficiency", 0.30))
        self.battery_capacity_wh = float(ENERGY_CONFIG.get("battery_capacity_wh", 200.0))
        self.eta_charge = float(ENERGY_CONFIG.get("eta_charge", 0.95))
        self.eta_discharge = float(ENERGY_CONFIG.get("eta_discharge", 0.95))
        self.soc_max = float(ENERGY_CONFIG.get("battery_max_soc", 1.0))

        self.service_rate_max_mbs = float(QUEUE_CONFIG.get(
            "data_service_rate_max_mbs",
            QUEUE_CONFIG.get("data_service_rate_max_mbps", 5.0)))
        self.tx_rate_max_mbs = float(QUEUE_CONFIG.get("tx_downlink_rate_max_mbs", 5.0))
        self.processed_queue_max_mb = float(QUEUE_CONFIG.get("comm_queue_max", 4096.0))
        self.raw_queue_max_mb = float(QUEUE_CONFIG.get("data_queue_max_mb", 4096.0))

        self.thermal_warning_c = float(THERMAL_CONFIG.get("warning_temp_c", 45.0))
        self.thermal_critical_c = float(THERMAL_CONFIG.get("critical_temp_c", 60.0))
        self.thermal_cool_rate = float(THERMAL_CONFIG.get("cool_rate_c_per_s", 0.02))
        self.thermal_heat_per_w = float(THERMAL_CONFIG.get("heat_rate_c_per_ws", 1.5e-4))

    # ── 主接口 ─────────────────────────────────────────────────────────
    def step(self, state: dict, action: np.ndarray) -> PredictedState:
        """一步保守前向预测；不修改输入 dict。"""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size < PHYSICAL_ACTION_DIM:
            a = np.pad(a, (0, PHYSICAL_ACTION_DIM - a.size))
        a = np.clip(a[:PHYSICAL_ACTION_DIM], 0.0, 1.0)
        alpha_prop, alpha_cpu, alpha_tx = float(a[0]), float(a[1]), float(a[2])

        altitude_m = float(state.get("altitude_m", 250e3))
        soc = float(state.get("soc", 0.7))
        qc = float(state.get("processed_queue_mb", 0.0))
        qd = float(state.get("raw_queue_mb", 0.0))
        thermal_margin = float(state.get("thermal_margin_norm", 1.0))
        in_window = bool(state.get("in_window", False))
        tx_capacity_mbps = float(state.get("tx_capacity_mbps", 0.0))
        sunlit_fraction = float(state.get("sunlit_fraction", 0.5))
        future_capacity_mb = float(state.get("future_contact_capacity_mb", 0.0))

        p_prop = alpha_prop * float(self.power_weights[0])
        p_cpu = alpha_cpu * float(self.power_weights[1])
        p_tx = alpha_tx * float(self.power_weights[2])
        p_load = p_prop + p_cpu + p_tx + self.baseline_power_w

        # 高度：用真实 orbital_dyn.step（保留 drag/thrust 的非线性）。
        orbit_info = self.orbital.step(altitude_m, p_prop, self.dt_s)
        altitude_next = float(orbit_info["altitude_m"])

        # 电池：名义太阳能 + 净功率 → ΔSOC。
        p_solar = self.solar_panel_power_w * self.solar_efficiency * sunlit_fraction
        p_net = p_solar - p_load
        dt_h = self.dt_s / 3600.0
        if p_net >= 0.0:
            delta_wh = p_net * dt_h * self.eta_charge
        else:
            delta_wh = p_net * dt_h / self.eta_discharge
        energy_now = soc * self.battery_capacity_wh
        energy_next = max(0.0, min(self.soc_max * self.battery_capacity_wh,
                                   energy_now + delta_wh))
        soc_next = float(energy_next / max(self.battery_capacity_wh, 1e-9))

        # processed queue：CPU 入 - TX 出（仅在 in_window 时下传）。
        processed_in_mb = alpha_cpu * self.service_rate_max_mbs * self.dt_s
        link_capacity_mb = 0.0
        downlink_mb = 0.0
        if in_window:
            link_capacity_mb = max(0.0, tx_capacity_mbps) * self.dt_s / 8.0
            rf_capacity_mb = alpha_tx * self.tx_rate_max_mbs * self.dt_s
            downlink_mb = min(alpha_tx * link_capacity_mb, rf_capacity_mb)
        qc_next = float(max(0.0, qc + processed_in_mb - downlink_mb))

        # raw queue：保守 (raw 到达率未知，沿用上一步剩余的减去处理量)。
        raw_processed = min(qd, alpha_cpu * self.service_rate_max_mbs * self.dt_s)
        qd_next = float(max(0.0, qd - raw_processed))

        # 热：CPU + TX 产热，被动散热。
        heat_in = (p_cpu + p_tx) * self.thermal_heat_per_w * self.dt_s
        cool = self.thermal_cool_rate * self.dt_s
        thermal_margin_next = float(np.clip(
            thermal_margin - heat_in + cool, -1.0, 1.0))

        return PredictedState(
            altitude_m=altitude_next,
            soc=soc_next,
            processed_queue_mb=qc_next,
            raw_queue_mb=qd_next,
            thermal_margin_norm=thermal_margin_next,
            in_window=in_window,
            tx_capacity_mbps=tx_capacity_mbps,
            sunlit_fraction=sunlit_fraction,
            future_contact_capacity_mb=future_capacity_mb,
        )

    def rollout(
        self,
        state: dict,
        actions: list[np.ndarray] | np.ndarray,
    ) -> list[PredictedState]:
        """多步 rollout：用列表中第 k 个 action 推进第 k+1 个状态。"""
        traj: list[PredictedState] = []
        current = dict(state)
        for a in actions:
            nxt = self.step(current, a)
            traj.append(nxt)
            current = nxt.to_dict()
        return traj


__all__ = ["SafetyDynamicsPredictor", "PredictedState"]

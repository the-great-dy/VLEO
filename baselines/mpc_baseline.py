"""
短视野模型预测控制基线。
MPC（模型预测控制）基线
航天领域传统优化方法：在预测窗口内求解最优功率分配

原理：
  每步基于已知物理模型，向前预测 H 步，
  用贪心优化求解满足安全约束的最大时效任务价值动作。
  不需要训练，直接利用轨道/能量动力学模型。
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    # 降低导入劫持风险：不要把项目路径插到最高优先级
    sys.path.append(_PROJECT_ROOT)

import numpy as np
from config import ENERGY_CONFIG, ORBITAL_CONFIG, QUEUE_CONFIG, DRAG_CONFIG
from environment.satellite_env import OBSERVATION_FEATURES
from environment.orbital_dynamics import vleo_density


_FEATURE_INDEX = {name: idx for idx, name in enumerate(OBSERVATION_FEATURES)}


def _feature(state: np.ndarray, name: str, default: float = 0.0) -> float:
    """Read an observation feature by name to avoid stale hard-coded indices."""
    idx = _FEATURE_INDEX.get(name)
    if idx is None or state.shape[0] <= idx:
        return float(default)
    return float(state[idx])


class MPCBaseline:
    """
    模型预测控制基线
    每步向前预测 horizon 步，找到满足安全约束的最优功率分配。
    优化目标：最大化预测窗口内的时效性加权任务价值，处理量只作为窗口前准备项
    安全约束：预测窗口内 SOC 和轨道高度始终在安全范围内
    """

    def __init__(self, horizon: int = 6, dt_s: float = 10.0):
        self.horizon  = horizon      # 预测步数（6步 = 60秒）
        self.dt       = dt_s
        self.dt_h     = dt_s / 3600.0

        # 物理参数
        self.cap_wh      = ENERGY_CONFIG["battery_capacity_wh"]
        self.soc_min     = ENERGY_CONFIG["battery_min_soc"]
        self.soc_max     = ENERGY_CONFIG["battery_max_soc"]
        self.P_prop_max  = ENERGY_CONFIG["power_propulsion_max_w"]
        self.P_cpu_max   = ENERGY_CONFIG["power_cpu_max_w"]
        self.P_tx_max    = ENERGY_CONFIG["power_tx_max_w"]
        self.cpu_rate_max_mbs = float(QUEUE_CONFIG.get(
            "data_service_rate_max_mbs",
            QUEUE_CONFIG.get("data_service_rate_max_mbps", 5.0),
        ))
        self.rf_rate_max_mbs = float(QUEUE_CONFIG.get("tx_downlink_rate_max_mbs", 5.0))
        self.tx_capacity_norm_mbps = float(QUEUE_CONFIG.get(
            "tx_capacity_norm_mbps",
            self.rf_rate_max_mbs * 8.0,
        ))
        self.P_base      = ENERGY_CONFIG["power_baseline_w"]
        self.h_min       = ORBITAL_CONFIG["altitude_min_km"] * 1e3
        self.h_crash     = ORBITAL_CONFIG.get("altitude_crash_km", 122.0) * 1e3
        self.mu          = ORBITAL_CONFIG["mu"]
        self.R_e         = ORBITAL_CONFIG["earth_radius_km"] * 1e3
        self.rho_ref     = DRAG_CONFIG["rho_ref"]
        self.H_scale     = DRAG_CONFIG["H_scale_km"] * 1e3
        self.ref_alt     = DRAG_CONFIG["ref_altitude_km"] * 1e3
        self.drag_cd     = DRAG_CONFIG["Cd"]
        self.drag_area_m2 = DRAG_CONFIG["area_m2"]
        # 卫星质量从配置读取，确保 MPC 与环境动力学使用同一物理参数。
        self.satellite_mass_kg = DRAG_CONFIG.get("mass_kg", 50.0)
        # 与 env 保持同一 drag 物理 (PDF Section 8.1)：v_rel 而非 v_orbit。
        self.enable_atmospheric_corotation = bool(
            DRAG_CONFIG.get("enable_atmospheric_corotation", True))
        self.inclination_rad = float(np.deg2rad(
            ORBITAL_CONFIG.get("inclination_deg", 51.6)))
        self._omega_earth = 7.2921159e-5
        # eclipse_fraction = 阴影时长占轨道周期的比例。env 在 reset 时按 β 角随机抽样
        # (enable_eclipse_beta_randomization=True)，每个 episode 阴影时长 0~50%。
        # MPC 6-step lookahead 必须用当前 episode 的实际值，否则在高 β 角 episode 上
        # 会假设不存在的阴影段，错估可用太阳能。sync_from_env 会从 env.orbit_sim 读取最新值。
        self.eclipse_fraction = float(
            ORBITAL_CONFIG["eclipse_duration_min"]) / float(
            ORBITAL_CONFIG["orbital_period_min"])

        # 离散动作空间（粗搜索）
        # 这里使用离散粗网格而不是连续优化，目的是保证基线稳定、可复现、计算开销可控。
        alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.action_candidates = [
            np.array([ap, ac, at])
            for ap in alphas
            for ac in alphas
            for at in alphas
        ]

    def sync_from_env(self, env) -> None:
        """
        同步当前环境物理参数。

        鲁棒性测试和真实 trace 会动态修改大气密度、太阳能效率、电池容量等参数。
        MPC 基线如果仍使用初始化常数，会在扰动场景中得到不公平或失真的预测结果。
        """
        if env is None:
            return

        orbit_dyn = getattr(env, "orbit_dyn", None)
        atmosphere = getattr(orbit_dyn, "atm", None)
        orbit_sim = getattr(env, "orbit_sim", None)
        if orbit_sim is not None:
            # 同步当前 episode 的 eclipse_fraction（由 env reset 按 β 角随机抽样）
            self._set_nonnegative(
                "eclipse_fraction",
                getattr(orbit_sim, "eclipse_fraction", self.eclipse_fraction))
        if orbit_dyn is not None:
            self._set_positive("mu", getattr(orbit_dyn, "mu", self.mu))
            self._set_positive("R_e", getattr(orbit_dyn, "R_e", self.R_e))
            self._set_positive("drag_cd", getattr(orbit_dyn, "Cd", self.drag_cd))
            self._set_positive("drag_area_m2", getattr(orbit_dyn, "A", self.drag_area_m2))
            self._set_positive("satellite_mass_kg", getattr(orbit_dyn, "m", self.satellite_mass_kg))
            self._set_positive("h_min", getattr(orbit_dyn, "h_min", self.h_min))
            self._set_positive("h_crash", getattr(orbit_dyn, "h_crash", self.h_crash))
        if atmosphere is not None:
            self._set_positive("rho_ref", getattr(atmosphere, "rho_ref", self.rho_ref))
            self._set_positive("H_scale", getattr(atmosphere, "H_scale", self.H_scale))
            self._set_positive("ref_alt", getattr(atmosphere, "ref_alt", self.ref_alt))

        battery = getattr(env, "battery", None)
        if battery is not None:
            self._set_positive("cap_wh", getattr(battery, "capacity_wh", self.cap_wh))
            self._set_nonnegative("soc_min", getattr(battery, "soc_min", self.soc_min))
            self._set_positive("soc_max", getattr(battery, "soc_max", self.soc_max))

        power_sys = getattr(env, "power_sys", None)
        if power_sys is not None:
            self._set_nonnegative("P_prop_max", getattr(power_sys, "P_prop_max", self.P_prop_max))
            self._set_nonnegative("P_cpu_max", getattr(power_sys, "P_cpu_max", self.P_cpu_max))
            self._set_nonnegative("P_tx_max", getattr(power_sys, "P_tx_max", self.P_tx_max))
            self._set_nonnegative("P_base", getattr(power_sys, "P_baseline", self.P_base))
            if hasattr(power_sys, "throughput_rate"):
                self._set_nonnegative("cpu_rate_max_mbs", power_sys.throughput_rate(self.P_cpu_max))
            if hasattr(power_sys, "tx_downlink_rate"):
                self._set_nonnegative("rf_rate_max_mbs", power_sys.tx_downlink_rate(self.P_tx_max))
                self._set_nonnegative("tx_capacity_norm_mbps", self.rf_rate_max_mbs * 8.0)

        self._set_positive("dt", getattr(env, "dt", self.dt))
        self.dt_h = self.dt / 3600.0

    def _set_positive(self, attr_name: str, value) -> None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if np.isfinite(value) and value > 0.0:
            setattr(self, attr_name, value)

    def _set_nonnegative(self, attr_name: str, value) -> None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if np.isfinite(value) and value >= 0.0:
            setattr(self, attr_name, value)

    # ── 物理预测 ──────────────────────────────────────────────────────
    def _predict_soc(self, soc: float, P_solar: float,
                     P_load: float) -> float:
        P_net = P_solar - P_load
        delta_wh = P_net * self.dt_h * (0.95 if P_net >= 0 else 1/0.95)
        new_soc  = soc + delta_wh / self.cap_wh
        return float(np.clip(new_soc, 0.0, self.soc_max))

    def _predict_altitude(self, altitude_m: float,
                          P_prop: float) -> float:
        r   = self.R_e + altitude_m
        v_orbit = np.sqrt(self.mu / r)
        # PDF Section 8.1: 阻力作用于 v_rel = v_orbit - ω_E·r·cos(i)。
        if self.enable_atmospheric_corotation:
            v_rel = max(v_orbit - self._omega_earth * r * np.cos(self.inclination_rad), 0.0)
        else:
            v_rel = v_orbit
        n   = np.sqrt(self.mu / r**3)
        rho = self._density(altitude_m)
        F_d = 0.5 * self.drag_cd * self.drag_area_m2 * rho * v_rel * v_rel
        thrust = P_prop * 0.65 / (1000 * 9.80665)
        # 使用与环境一致的卫星质量进行高度预测。
        dh     = (2 * (thrust - F_d) / (self.satellite_mass_kg * n)) * self.dt
        return float(np.clip(altitude_m + dh, self.h_crash, 450e3))

    def _density(self, altitude_m: float, density_scale: float = 1.0) -> float:
        # 与 env 一致使用 US Std Atm 1976 分段指数模型，避免 PSF/MPC 在低高度
        # 把密度低估 ~3 个数量级。density_scale 由 robust rollout 注入扰动。
        rho = float(density_scale) * vleo_density(
            altitude_m, self.rho_ref, self.ref_alt, self.H_scale)
        return float(max(rho, 1e-15))

    def _value_score(self, state: np.ndarray,
                     P_cpu: float, P_tx: float) -> float:
        """估计一步内的任务价值收益，兼顾处理准备和窗口内下传。"""
        state = np.asarray(state, dtype=np.float32)
        in_window = bool(_feature(state, "in_comm_window", 0.0) > 0.5)
        tx_capacity_mbps = (
            _feature(state, "tx_capacity_norm", 0.0) * self.tx_capacity_norm_mbps
        )
        processed_value_norm = _feature(state, "total_processed_value_norm", 0.0)
        priority = _feature(state, "topk_priority_norm", 1.0)
        quality = _feature(state, "topk_quality_norm", 1.0)
        deadline_urgency = _feature(state, "deadline_urgency", 0.0)
        scene_class = _feature(state, "current_scene_class_norm", 0.0)
        value_density = max(
            0.1,
            priority * quality * (1.0 + 0.5 * deadline_urgency) * (1.0 + 0.2 * scene_class),
        )

        processed_mb = (
            P_cpu / max(self.P_cpu_max, 1e-6)
        ) * self.cpu_rate_max_mbs * self.dt
        delivered_mb = 0.0
        if in_window:
            capacity_mb = max(0.0, tx_capacity_mbps * self.dt / 8.0)
            alpha_tx = float(np.clip(P_tx / max(self.P_tx_max, 1e-6), 0.0, 1.0))
            link_limited_mb = alpha_tx * capacity_mb
            rf_limited_mb = alpha_tx * self.rf_rate_max_mbs * self.dt
            # 与环境保持同一物理口径：地面链路容量和发射机 RF 速率都必须满足。
            delivered_mb = min(link_limited_mb, rf_limited_mb)
        downlink_pressure = 1.0 + 0.25 * max(0.0, processed_value_norm)
        return float(value_density * delivered_mb * downlink_pressure + 0.25 * value_density * processed_mb)

    # ── MPC 决策 ─────────────────────────────────────────────────────
    def schedule(self, state: np.ndarray,
                 soc: float, altitude_m: float,
                 sunlit: bool, P_solar: float,
                 time_s: float = 0.0,
                 env=None) -> np.ndarray:  # 全局物理时间用于预测日照/阴影相位
        """
        Args:
            state:      12维观测向量（MPC不直接使用，保持接口一致）
            soc:        当前电池SOC [0,1]
            altitude_m: 当前轨道高度 (m)
            sunlit:     是否在日照区
            P_solar:    当前太阳能功率 (W)
            time_s:     真实全局物理时间 (s)
        Returns:
            action: [alpha_prop, alpha_cpu, alpha_tx]
        """
        best_action  = np.array([0.5, 0.1, 0.1])
        best_score   = -np.inf
        self.sync_from_env(env)

        # 对每个候选动作都做短视野物理预测，只选择全程安全且吞吐得分最高的动作。
        for action in self.action_candidates:
            score, feasible = self._evaluate_action(
                action, soc, altitude_m, sunlit, P_solar, time_s, state)
            if feasible and score > best_score:
                best_score  = score
                best_action = action.copy()

        return best_action

    def _evaluate_action(self, action: np.ndarray,
                         soc0: float, alt0: float,
                         sunlit0: bool, P_solar0: float,
                         time_s: float,
                         state: np.ndarray) -> tuple:
        """
        向前预测 horizon 步，返回 (累计吞吐量, 是否全程安全)
        """
        ap, ac, at = action
        P_prop = ap * self.P_prop_max
        P_cpu  = ac * self.P_cpu_max
        P_tx   = at * self.P_tx_max
        P_load = P_prop + P_cpu + P_tx + self.P_base

        # MPC 的“可行”要求预测窗口内 SOC 和高度都不越界；一旦越界直接判该动作不可行。
        soc    = soc0
        alt    = alt0
        total_value = 0.0
        sunlit     = sunlit0
        P_solar    = P_solar0

        # 根据全局时间推算预测窗口内的日照/阴影切换。
        # 用当前 episode 的真实 eclipse_fraction（sync_from_env 中从 orbit_sim 同步），
        # 而不是硬编码 35min。env 的 β 角随机化下 eclipse_fraction 在 0~0.50 之间变化。
        current_global_step = int(time_s / self.dt)
        orbital_period_s = float(ORBITAL_CONFIG["orbital_period_min"]) * 60.0
        _ORBITAL_STEPS = max(1, int(orbital_period_s / max(self.dt, 1e-6)))
        _SUNLIT_STEPS = max(0, int(_ORBITAL_STEPS * (1.0 - float(self.eclipse_fraction))))

        for step in range(self.horizon):
            # 安全检查
            if soc < self.soc_min + 0.02:
                return total_value, False
            if alt < self.h_min + 2e3:
                return total_value, False

            # 推进不足时轨道快速衰减，直接判不安全
            decay_est = 2 * 0.5 * self.drag_cd * self.drag_area_m2 * \
                self._density(alt) * \
                (self.mu / (self.R_e+alt)) / \
                (self.satellite_mass_kg * np.sqrt(self.mu/(self.R_e+alt)**3)) * self.dt
            if abs(decay_est) > 500 and P_prop < 5:
                return total_value, False

            # 按轨道相位切换日照/阴影，避免短视野内误判可用太阳能。
            _phase = (current_global_step + step) % _ORBITAL_STEPS
            if _phase >= _SUNLIT_STEPS:
                sunlit  = False
                P_solar = 0.0
            else:
                sunlit  = True
                P_solar = P_solar0

            # 更新状态
            soc = self._predict_soc(soc, P_solar, P_load)
            alt = self._predict_altitude(alt, P_prop)
            total_value += self._value_score(state, P_cpu, P_tx)

        return total_value, True

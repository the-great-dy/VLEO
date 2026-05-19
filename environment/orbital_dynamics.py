"""
VLEO 轨道动力学、阻力衰减和轨道相位模型。
VLEO卫星轨道动力学模型

相位模型：在 Env 中维护一个累加的轨道相位 θ ∈ [0, 2π)。
每步积分：θ += n(h) × dt，其中 n(h) = √(μ/r³) 是当前高度的角速度。
日照判断基于 θ 而不是 time_s % T(h)，避免高度变化导致相位不连续。
"""
import sys as _sys, os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in _sys.path:
    # 降低导入劫持风险：不要把项目路径插到最高优先级
    _sys.path.append(_PROJECT_ROOT)

import numpy as np
from config import ORBITAL_CONFIG, DRAG_CONFIG, ENERGY_CONFIG
from utils.sanitizers import sanitize_scalar


class AtmosphericModel:
    """指数型大气密度模型"""
    def __init__(self):
        self.rho_ref = DRAG_CONFIG["rho_ref"]
        self.H_scale = DRAG_CONFIG["H_scale_km"] * 1e3
        self.ref_alt = DRAG_CONFIG["ref_altitude_km"] * 1e3

    def density(self, altitude_m: float) -> float:
        delta_h = altitude_m - self.ref_alt
        return max(self.rho_ref * np.exp(-delta_h / self.H_scale), 1e-15)

    def density_gradient(self, altitude_m: float) -> float:
        return -self.density(altitude_m) / self.H_scale


class OrbitalDynamics:
    """VLEO卫星轨道动力学"""
    def __init__(self):
        self.mu = ORBITAL_CONFIG["mu"]
        self.R_e = ORBITAL_CONFIG["earth_radius_km"] * 1e3
        self.Cd = DRAG_CONFIG["Cd"]
        self.A = DRAG_CONFIG["area_m2"]
        self.m = DRAG_CONFIG["mass_kg"]
        self.atm = AtmosphericModel()
        self.h_warning = ORBITAL_CONFIG.get("altitude_warning_km", 180.0) * 1e3
        self.h_min = ORBITAL_CONFIG["altitude_min_km"] * 1e3
        self.h_crash = ORBITAL_CONFIG.get("altitude_crash_km", 122.0) * 1e3
        self.h_max = ORBITAL_CONFIG["altitude_max_km"] * 1e3
        self.h_nominal = ORBITAL_CONFIG["altitude_nominal_km"] * 1e3
        self.propulsion_ignition_threshold_w = float(
            ENERGY_CONFIG.get("propulsion_ignition_threshold_w", 0.0))
        self.propulsion_efficiency = float(
            ENERGY_CONFIG.get("propulsion_efficiency", 0.65))
        self.propulsion_isp_s = float(
            ENERGY_CONFIG.get("propulsion_isp_s", 1000.0))

    def classify_altitude(self, altitude_m: float) -> tuple[str, int]:
        """Return orbit risk stage: normal, warning, unsafe, or failure."""
        if altitude_m <= self.h_crash:
            return "failure", 3
        if altitude_m < self.h_min:
            return "unsafe", 2
        if altitude_m < self.h_warning:
            return "warning", 1
        return "normal", 0

    def orbital_velocity(self, altitude_m: float) -> float:
        return np.sqrt(self.mu / (self.R_e + altitude_m))

    def mean_motion(self, altitude_m: float) -> float:
        r = self.R_e + altitude_m
        return np.sqrt(self.mu / r**3)

    def drag_force(self, altitude_m: float) -> float:
        rho = self.atm.density(altitude_m)
        v = self.orbital_velocity(altitude_m)
        return 0.5 * self.Cd * self.A * rho * v**2

    def altitude_decay_rate(self, altitude_m: float) -> float:
        return -2.0 * self.drag_force(altitude_m) / (self.m * self.mean_motion(altitude_m))

    def propulsion_thrust(self, power_w: float, Isp: float | None = None) -> float:
        power = sanitize_scalar(
            power_w,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
            min_value=0.0,
        )
        if power < self.propulsion_ignition_threshold_w:
            return 0.0
        isp_s = self.propulsion_isp_s if Isp is None else max(float(Isp), 1e-9)
        return power * self.propulsion_efficiency / (isp_s * 9.80665)

    def step(self, altitude_m: float, power_propulsion_w: float, dt_s: float) -> dict:
        # 单步轨道高度变化由大气阻力衰减和推进补偿共同决定，输出 dh 便于日志与监控。
        decay = self.altitude_decay_rate(altitude_m)
        thrust = self.propulsion_thrust(power_propulsion_w)
        n = self.mean_motion(altitude_m)
        dh = (decay + 2.0 * thrust / (self.m * n)) * dt_s
        # Do not clip to the safety boundary: 150 km is an unsafe boundary,
        # while 122 km is the terminal re-entry boundary that must remain reachable.
        new_altitude = float(np.clip(altitude_m + dh, 0.0, self.h_max))
        orbit_stage, orbit_stage_code = self.classify_altitude(new_altitude)
        return {
            "altitude_m": new_altitude,
            "drag_force_N": self.drag_force(altitude_m),
            "decay_rate_ms": decay,
            "thrust_N": thrust,
            "propulsion_ignition_active": bool(
                power_propulsion_w >= self.propulsion_ignition_threshold_w),
            "dh_m": dh,
            "is_safe": new_altitude >= self.h_min,
            "is_crashed": new_altitude <= self.h_crash,
            "is_warning": orbit_stage == "warning",
            "safety_stage": orbit_stage,
            "safety_stage_code": orbit_stage_code,
        }


class OrbitalPeriodSimulator:
    """
    轨道周期模拟器（积分法相位模型）

    维护累加相位 θ ∈ [0, 2π)，每步 θ += n(h) × dt。
    日照判断基于 θ 所在区间，并保留 time_s 参数作为未初始化时的兼容路径。
    """

    def __init__(self):
        self.mu = ORBITAL_CONFIG["mu"]
        self.R_e = ORBITAL_CONFIG["earth_radius_km"] * 1e3
        self.h_nominal = ORBITAL_CONFIG["altitude_nominal_km"] * 1e3

        # 日照/阴影比例
        self.eclipse_fraction = (
            ORBITAL_CONFIG["eclipse_duration_min"]
            / ORBITAL_CONFIG["orbital_period_min"]
        )
        # 日照区覆盖的相位角
        self._sunlit_phase = 2.0 * np.pi * (1.0 - self.eclipse_fraction)

        # 标称周期
        self._nominal_period_s = self._compute_period(self.h_nominal)

        # 累加轨道相位（由 Env 调用 advance_phase 更新）
        self._phase = 0.0
        self._phase_initialized = False   # 标记是否已切换到积分相位模式

    def _compute_period(self, altitude_m: float) -> float:
        # 使用当前高度对应的角速度积分相位，避免 time_s % T(h) 在高度变化时造成相位跳变。
        r = self.R_e + altitude_m
        return 2.0 * np.pi * np.sqrt(r**3 / self.mu)

    @property
    def period_s(self) -> float:
        return self._nominal_period_s

    def period_at(self, altitude_m: float) -> float:
        return self._compute_period(altitude_m)

    @property
    def phase(self) -> float:
        """当前轨道相位 [0, 2π)"""
        return self._phase

    def reset_phase(self, initial_phase: float = None):
        """重置轨道相位（Env.reset 时调用）"""
        if initial_phase is not None:
            self._phase = initial_phase % (2.0 * np.pi)
        else:
            self._phase = 0.0
        self._phase_initialized = True   # 标记已初始化

    def advance_phase(self, altitude_m: float, dt_s: float):
        """
        积分法推进轨道相位。
        θ += n(h) × dt，其中 n(h) = √(μ/r³)。
        每个 Env.step() 调用一次。
        """
        r = self.R_e + altitude_m
        n = np.sqrt(self.mu / r**3)   # 当前角速度 (rad/s)
        self._phase = (self._phase + n * dt_s) % (2.0 * np.pi)

    def is_sunlit(self, time_s: float = None, altitude_m: float = None) -> bool:
        """
        判断是否在日照区。
        优先使用积分相位；如果 Env 没调用 reset_phase/advance_phase，
        则 fallback 到 time_s % T 模式。
        """
        if self._phase_initialized:
            # 积分模式：phase < sunlit_phase → 日照
            return self._phase < self._sunlit_phase
        # 兼容路径：仅在未使用积分相位时按 time_s % T(h) 判断。
        if time_s is not None:
            T = self._compute_period(altitude_m) if altitude_m else self._nominal_period_s
            return (time_s % T) < T * (1.0 - self.eclipse_fraction)
        return True

    def time_to_next_eclipse(self, time_s: float = None,
                             altitude_m: float = None) -> float:
        """距离下次进入阴影区的时间（秒）"""
        T = self._compute_period(altitude_m) if altitude_m else self._nominal_period_s
        if self._phase < self._sunlit_phase:
            # 当前日照，距阴影还有多远
            remaining_angle = self._sunlit_phase - self._phase
        else:
            # 当前阴影，要等整个阴影+下一个日照
            remaining_angle = (2.0 * np.pi - self._phase) + self._sunlit_phase
        # 角度 → 时间
        r = self.R_e + (altitude_m if altitude_m else self.h_nominal)
        n = np.sqrt(self.mu / r**3)
        return remaining_angle / n

    def time_to_next_sunlit(self, time_s: float = None,
                            altitude_m: float = None) -> float:
        if self._phase < self._sunlit_phase:
            return 0.0  # 当前已日照
        remaining_angle = 2.0 * np.pi - self._phase
        r = self.R_e + (altitude_m if altitude_m else self.h_nominal)
        n = np.sqrt(self.mu / r**3)
        return remaining_angle / n

    def sunlit_fraction(self, time_s: float = None,
                        altitude_m: float = None) -> float:
        if not self.is_sunlit(time_s, altitude_m):
            return 0.0
        # 正弦模型：日照区内从 0→1→0
        # 日照强度用半个正弦近似，表示从进入日照到日照峰值再到出日照的连续变化。
        angle = np.pi * self._phase / self._sunlit_phase
        return max(np.sin(angle), 0.0)

    # 保留属性接口，供旧代码读取标称阴影/日照时长。
    @property
    def eclipse_s(self) -> float:
        return self._nominal_period_s * self.eclipse_fraction

    @property
    def sunlit_s(self) -> float:
        return self._nominal_period_s * (1.0 - self.eclipse_fraction)

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


# ─────────────────────────────────────────────
# 物理常数与 PDF Section 5 / 8 派生模型
# ─────────────────────────────────────────────
_OMEGA_EARTH_RAD_S = 7.2921159e-5  # 地球自转角速度 (rad/s)

# 日间隆起 (Diurnal Bulge) 强度：rho_M / rho_m 与高度的关系。
# 数据来源：Harris-Priester F10.7≈150 标准表（PDF Section 5.3）。
# 低于 120km 大气均匀混合，无显著昼夜差异；450km 处 ρ_M/ρ_m ≈ 3~4x。
_DIURNAL_BULGE_RATIO_ANCHORS = (
    # (altitude_km, rho_M / rho_m)
    (100.0, 1.00),
    (120.0, 1.05),
    (150.0, 1.20),
    (200.0, 1.32),
    (250.0, 1.55),
    (300.0, 2.00),
    (350.0, 2.50),
    (400.0, 3.00),
    (450.0, 3.50),
    (500.0, 4.00),
)
_DIURNAL_BULGE_H_KM = np.array(
    [row[0] for row in _DIURNAL_BULGE_RATIO_ANCHORS], dtype=np.float64)
_DIURNAL_BULGE_RATIO = np.array(
    [row[1] for row in _DIURNAL_BULGE_RATIO_ANCHORS], dtype=np.float64)


def diurnal_bulge_amplitude(altitude_m: float) -> float:
    """日间隆起半幅 α：rho(Ψ) = rho_base * (1 + α·cos Ψ)，α ∈ [0, 1)。

    ρ_M / ρ_m = (1+α)/(1-α)，从 Harris-Priester 表插值。
    高度越高 α 越大；100km 以下 α≈0（大气均匀混合）。
    """
    h_km = float(altitude_m) / 1000.0
    if h_km <= _DIURNAL_BULGE_H_KM[0]:
        ratio = float(_DIURNAL_BULGE_RATIO[0])
    elif h_km >= _DIURNAL_BULGE_H_KM[-1]:
        ratio = float(_DIURNAL_BULGE_RATIO[-1])
    else:
        ratio = float(np.interp(h_km, _DIURNAL_BULGE_H_KM, _DIURNAL_BULGE_RATIO))
    return float((ratio - 1.0) / (ratio + 1.0))


# ─────────────────────────────────────────────
# VLEO 分段指数大气密度模型 (Vallado/Wertz 校准表)
#
# 单一指数 (H=50km) 在 VLEO 范围严重失真：scale height 在 100km 仅约 6km、
# 150km 约 22km、200km 约 35km、350km 约 50km。固定 H=50km 模型在 120km
# 处会把密度低估 ~2~3 个数量级，PSF/MPC 据此预测的高度衰减会乐观得离谱。
#
# 这里采用 Vallado《Fundamentals of Astrodynamics》& Wertz《SMAD》分段指数表
# (12 段，100~500km，每段独立 base 密度 + 局部 scale height)，源自 MSIS/CIRA72
# 统计拟合。每段独立校准，段边界允许小幅密度不连续 (Vallado 原表性质，<10%)。
#
# 表参考密度对应中等太阳活跃期 (F10.7 ≈ 150)。用户的 DRAG_CONFIG["rho_ref"] 与
# 表中 350km 标准值 6.660e-12 的比值决定全局校准倍数 (相当于实际太阳活跃水平)：
#   - 默认 4.89e-11 / 6.660e-12 ≈ 7.34x → 高太阳活跃期 (F10.7 ≈ 200~250)
#   - robustness 实验中 atm.rho_ref *= scale 等价于直接调整太阳活跃倍数
#
# 不变量保留：用户 DRAG_CONFIG["H_scale_km"] 用作 ref_alt 所在段的局部 scale
# height，在 [ref_alt, ref_alt+H_scale] 区间用 rho_ref * exp(-(h-ref_alt)/H_scale)，
# 严格满足 density(ref_alt + H_scale) = rho_ref/e (单元测试不变量)。
#
# 进一步精度需要日夜隆起、地磁暴响应：见 Harris-Priester / JB2008 / DTM2020
# (这里没有实时空间天气输入，故未集成)。
# ─────────────────────────────────────────────
# 每行: (高度下边界 km, 段内 base 密度 kg/m³, 段内 scale height km)。
# 密度模型: rho(z_m) = rho_base * exp(-(z_km - h_lower_km) / H_local_km)
#           对于 h_lower_km <= z_km < next_h_lower_km。
_VALLADO_VLEO_TABLE = (
    (100.0,  5.297e-7,   5.877),
    (110.0,  9.661e-8,   7.263),
    (120.0,  2.438e-8,   9.473),
    (130.0,  8.484e-9,  12.636),
    (140.0,  3.845e-9,  16.149),
    (150.0,  1.730e-9,  25.500),
    (200.0,  2.410e-10, 37.500),
    (250.0,  5.970e-11, 44.800),
    (300.0,  1.870e-11, 50.300),
    (350.0,  6.660e-12, 54.800),  # 标准 350km 密度；用户 ref_alt 与之对齐
    (400.0,  2.620e-12, 58.200),
    (450.0,  1.050e-12, 61.200),
)
_VALLADO_REF_ALT_KM = 350.0
_VALLADO_REF_RHO_KG_M3 = 6.660e-12   # 表中 350km 标准密度，用作 rho_ref 校准基准

_VALLADO_H_LOWER_M = np.array(
    [row[0] * 1e3 for row in _VALLADO_VLEO_TABLE], dtype=np.float64)
_VALLADO_RHO_BASE_KG_M3 = np.array(
    [row[1] for row in _VALLADO_VLEO_TABLE], dtype=np.float64)
_VALLADO_H_LOCAL_M = np.array(
    [row[2] * 1e3 for row in _VALLADO_VLEO_TABLE], dtype=np.float64)


def _vleo_segment_idx(altitude_m: float) -> int:
    """返回包含 altitude_m 的 Vallado 段索引 (最大 i 满足 h_lower[i] <= h)。"""
    h = float(altitude_m)
    if h <= _VALLADO_H_LOWER_M[0]:
        return 0
    if h >= _VALLADO_H_LOWER_M[-1]:
        return len(_VALLADO_H_LOWER_M) - 1
    idx = int(np.searchsorted(_VALLADO_H_LOWER_M, h, side='right')) - 1
    return max(0, min(len(_VALLADO_H_LOWER_M) - 1, idx))


def vleo_density(altitude_m: float, rho_ref: float,
                 ref_alt_m: float, H_scale_m: float) -> float:
    """Vallado/Wertz 分段指数 VLEO 大气密度。

    在 [ref_alt, ref_alt+H_scale_m] 区间，使用用户指定的 H_scale_m 作为局部 scale
    height，密度 = rho_ref * exp(-(h-ref_alt)/H_scale_m)。这保证：
      - density(ref_alt) = rho_ref
      - density(ref_alt + H_scale_m) = rho_ref / e (H_scale 单元测试不变量)

    其他高度按 Vallado 分段指数表计算，密度乘以全局校准倍数 (rho_ref / 6.660e-12)
    以反映实际太阳活跃水平。rho_ref 缩放对所有高度线性生效，兼容
    robustness 实验中 `atm.rho_ref *= scale` 的扰动注入语义。

    每段独立校准，段边界允许 <10% 密度不连续 (Vallado 原表性质)。
    """
    h = float(altitude_m)
    ref_alt = float(ref_alt_m)
    H_user = float(max(H_scale_m, 1.0))

    # ref_alt 段覆盖区间 [ref_alt, ref_alt + H_user]：用户 H_scale 主导，保留 1/e 不变量。
    if ref_alt <= h <= ref_alt + H_user:
        return max(float(rho_ref) * float(np.exp(-(h - ref_alt) / H_user)), 1e-15)

    # 其余高度走 Vallado 表，整体按 rho_ref 与表标准值的比例缩放。
    cal_scale = float(rho_ref) / _VALLADO_REF_RHO_KG_M3
    idx = _vleo_segment_idx(h)
    h_lower = float(_VALLADO_H_LOWER_M[idx])
    rho_base = float(_VALLADO_RHO_BASE_KG_M3[idx]) * cal_scale
    H_local = float(_VALLADO_H_LOCAL_M[idx])
    return max(rho_base * float(np.exp(-(h - h_lower) / max(H_local, 1.0))), 1e-15)


def vleo_local_scale_height(altitude_m: float,
                            ref_alt_m: float, H_scale_m: float) -> float:
    """vleo_density 在指定高度处的局部 scale height (m)。"""
    h = float(altitude_m)
    ref_alt = float(ref_alt_m)
    H_user = float(max(H_scale_m, 1.0))
    if ref_alt <= h <= ref_alt + H_user:
        return H_user
    idx = _vleo_segment_idx(h)
    return float(_VALLADO_H_LOCAL_M[idx])


class AtmosphericModel:
    """VLEO Vallado/Wertz 分段指数大气模型，集成日间隆起与地磁暴瞬态调制。

    密度构成 (PDF Section 5 / 8.2)：
        rho(h, Ψ) = vleo_density(h, rho_ref_effective) * (1 + α(h)·cos Ψ)
        rho_ref_effective = rho_ref_base * storm_multiplier(t)

    - rho_ref_base 是用户在 DRAG_CONFIG 设置的太阳活跃水平校准
    - storm_multiplier 是当前地磁暴/F10.7 突变引起的瞬态膨胀因子 (默认 1.0)
    - α(h)·cos Ψ 是 Harris-Priester 日间隆起调制，Ψ=0 对应 bulge 峰值
    """

    def __init__(self):
        self._rho_ref = float(DRAG_CONFIG["rho_ref"])
        self.H_scale = float(DRAG_CONFIG["H_scale_km"]) * 1e3
        self.ref_alt = float(DRAG_CONFIG["ref_altitude_km"]) * 1e3
        self.enable_diurnal_bulge = bool(
            DRAG_CONFIG.get("enable_diurnal_bulge", True))
        # 单位为 rho_ref 倍数；由 env 风暴事件循环调制 (>1 表示风暴期热层膨胀)。
        self._storm_multiplier = 1.0

    @property
    def rho_ref(self) -> float:
        return self._rho_ref

    @rho_ref.setter
    def rho_ref(self, value: float) -> None:
        # 保留 robustness 测试 / experiments 里 `atm.rho_ref *= scale` 的扰动注入语义：
        # 所有高度密度按 rho_ref 线性缩放。风暴乘子与之独立叠加。
        self._rho_ref = float(value)

    @property
    def storm_multiplier(self) -> float:
        return self._storm_multiplier

    @storm_multiplier.setter
    def storm_multiplier(self, value: float) -> None:
        # 限幅 [1, 5]，覆盖 Starlink 2022 +190% 极值；同时避免病态值进入 drag 公式。
        self._storm_multiplier = float(np.clip(value, 1.0, 5.0))

    def _effective_rho_ref(self) -> float:
        return self._rho_ref * self._storm_multiplier

    def density(self, altitude_m: float,
                diurnal_angle_rad: float | None = None) -> float:
        """局部大气密度。

        diurnal_angle_rad: 卫星位置相对日间 bulge 中心的角度 Ψ。
            None → 返回轨道平均密度 (cos Ψ 在 [0,2π] 上均值=0，等价于不调制)。
            提供时按 Harris-Priester (1 + α(h)·cos Ψ) 调制。
        """
        rho_base = vleo_density(
            altitude_m, self._effective_rho_ref(), self.ref_alt, self.H_scale)
        if not self.enable_diurnal_bulge or diurnal_angle_rad is None:
            return rho_base
        alpha = diurnal_bulge_amplitude(altitude_m)
        modulation = 1.0 + alpha * float(np.cos(float(diurnal_angle_rad)))
        return max(rho_base * modulation, 1e-15)

    def density_gradient(self, altitude_m: float,
                         diurnal_angle_rad: float | None = None) -> float:
        H_local = vleo_local_scale_height(altitude_m, self.ref_alt, self.H_scale)
        return -self.density(altitude_m,
                             diurnal_angle_rad=diurnal_angle_rad) / max(H_local, 1.0)


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
        # 大气共转：a_drag ∝ |v_rel|², v_rel = v_orbit - ω_E·r·cos(i) (PDF Section 8.1)。
        # 关闭时退回 v_orbit (历史行为)。
        self.enable_atmospheric_corotation = bool(
            DRAG_CONFIG.get("enable_atmospheric_corotation", True))
        self.inclination_rad = float(np.deg2rad(
            ORBITAL_CONFIG.get("inclination_deg", 51.6)))

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

    def relative_velocity(self, altitude_m: float) -> float:
        """卫星相对共转大气的速度大小 (PDF Section 8.1)。

        v_rel = v_orbit - ω_E·r·cos(i)。51.6° 倾角下 v_orbit 减 ~3.8%，
        进入 drag 公式后 |v_rel|² 比 v_orbit² 低 ~7.5%。
        """
        v_orbit = self.orbital_velocity(altitude_m)
        if not self.enable_atmospheric_corotation:
            return float(v_orbit)
        r = self.R_e + float(altitude_m)
        v_atm_along = _OMEGA_EARTH_RAD_S * r * float(np.cos(self.inclination_rad))
        return float(max(v_orbit - v_atm_along, 0.0))

    def drag_force(self, altitude_m: float,
                   diurnal_angle_rad: float | None = None) -> float:
        rho = self.atm.density(altitude_m, diurnal_angle_rad=diurnal_angle_rad)
        v_rel = self.relative_velocity(altitude_m)
        return 0.5 * self.Cd * self.A * rho * v_rel * v_rel

    def altitude_decay_rate(self, altitude_m: float,
                            diurnal_angle_rad: float | None = None) -> float:
        return -2.0 * self.drag_force(altitude_m,
                                      diurnal_angle_rad=diurnal_angle_rad) / (
            self.m * self.mean_motion(altitude_m))

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

    def step(self, altitude_m: float, power_propulsion_w: float, dt_s: float,
             diurnal_angle_rad: float | None = None) -> dict:
        # 单步轨道高度变化由大气阻力衰减和推进补偿共同决定，输出 dh 便于日志与监控。
        decay = self.altitude_decay_rate(altitude_m, diurnal_angle_rad=diurnal_angle_rad)
        thrust = self.propulsion_thrust(power_propulsion_w)
        n = self.mean_motion(altitude_m)
        dh = (decay + 2.0 * thrust / (self.m * n)) * dt_s
        # Do not clip to the safety boundary: 150 km is an unsafe boundary,
        # while 122 km is the terminal re-entry boundary that must remain reachable.
        new_altitude = float(np.clip(altitude_m + dh, 0.0, self.h_max))
        orbit_stage, orbit_stage_code = self.classify_altitude(new_altitude)
        return {
            "altitude_m": new_altitude,
            "drag_force_N": self.drag_force(altitude_m,
                                            diurnal_angle_rad=diurnal_angle_rad),
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

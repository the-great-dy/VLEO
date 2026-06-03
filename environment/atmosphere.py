"""
VLEO 大气密度模型子系统（可切换模型 + 空间天气显式驱动）。

本模块把大气建模从 orbital_dynamics 中独立出来，提供：
  1. 三档可切换大气模型（DRAG_CONFIG["atmosphere_model"]）：
       - "exponential"  单一 scale height 指数模型（debug / 消融）
       - "vallado"      Vallado/Wertz 分段指数（默认，保留全部既有物理与单元测试不变量）
       - "nrlmsise00"   NRLMSISE-00 经验模型（pymsis 后端，F10.7 + Ap 原生驱动）
  2. SpaceWeatherState：F10.7（daily / 81-day avg）、Ap（含风暴瞬态）、epoch/doy。
       每 episode reset 时由 env 抽样并 push 给当前大气模型。
  3. 统一的 F10.7/Ap → forcing 换算：
       - 解析模型经 f107_to_rho_scale / ap_to_storm_multiplier 把指数折算为 ρ 缩放
         与风暴乘子（锚定 F10.7=150 + quiet Ap 精确复现既有基线）。
       - NRLMSISE-00 直接消费 F10.7/Ap；其内部已建模昼夜，故解析 cos Ψ 调制自动关闭。

向后兼容：`AtmosphericModel` 作为默认 Vallado 模型的别名导出，且
`vleo_density` / `vleo_local_scale_height` / `diurnal_bulge_amplitude` 仍可从此处
（以及经 orbital_dynamics 重新导出）取用。`rho_ref` 注入语义（robustness 实验中
`atm.rho_ref *= scale`）保持不变。
"""
from __future__ import annotations

import sys as _sys, os as _os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in _sys.path:
    _sys.path.append(_PROJECT_ROOT)

import numpy as np
from config import ORBITAL_CONFIG, DRAG_CONFIG


# ─────────────────────────────────────────────
# 物理常数
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
# 表中 350km 标准值 6.660e-12 的比值决定全局校准倍数 (相当于实际太阳活跃水平)。
#
# 不变量保留：用户 DRAG_CONFIG["H_scale_km"] 用作 ref_alt 所在段的局部 scale
# height，在 [ref_alt, ref_alt+H_scale] 区间用 rho_ref * exp(-(h-ref_alt)/H_scale)，
# 严格满足 density(ref_alt + H_scale) = rho_ref/e (单元测试不变量)。
# ─────────────────────────────────────────────
# 每行: (高度下边界 km, 段内 base 密度 kg/m³, 段内 scale height km)。
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
_EXP_ARG_MIN = -745.0
_EXP_ARG_MAX = 80.0


def _bounded_exp(value: float) -> float:
    """对指数参数做有限裁剪，避免非法高度把密度计算推到 overflow。"""
    return float(np.exp(np.clip(float(value), _EXP_ARG_MIN, _EXP_ARG_MAX)))


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
    """
    h = float(altitude_m)
    ref_alt = float(ref_alt_m)
    H_user = float(max(H_scale_m, 1.0))

    # ref_alt 段覆盖区间 [ref_alt, ref_alt + H_user]：用户 H_scale 主导，保留 1/e 不变量。
    if ref_alt <= h <= ref_alt + H_user:
        return max(float(rho_ref) * _bounded_exp(-(h - ref_alt) / H_user), 1e-15)

    # 其余高度走 Vallado 表，整体按 rho_ref 与表标准值的比例缩放。
    cal_scale = float(rho_ref) / _VALLADO_REF_RHO_KG_M3
    idx = _vleo_segment_idx(h)
    h_lower = float(_VALLADO_H_LOWER_M[idx])
    rho_base = float(_VALLADO_RHO_BASE_KG_M3[idx]) * cal_scale
    H_local = float(_VALLADO_H_LOCAL_M[idx])
    return max(rho_base * _bounded_exp(-(h - h_lower) / max(H_local, 1.0)), 1e-15)


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


# ─────────────────────────────────────────────
# 空间天气状态 + F10.7/Ap → forcing 换算
# ─────────────────────────────────────────────
@dataclass
class SpaceWeatherState:
    """一个 episode 的空间天气驱动量。env.reset 时抽样并 push 给大气模型。

    - f107_daily / f107_81avg : 太阳 10.7cm 射电流量 (sfu)；6h episode 内视为恒定。
    - ap                      : 当前地磁活动指数 (含风暴瞬态)；由 env 每步更新。
    - epoch_doy               : 年内日序 (1~366)，供 NRLMSISE-00 季节/昼夜几何。
    - raan_deg / inclination_deg : 供 NRLMSISE-00 合成亚卫星点经纬度。
    """
    f107_daily: float = 150.0
    f107_81avg: float = 150.0
    ap: float = 4.0
    epoch_doy: float = 80.0
    raan_deg: float = 0.0
    inclination_deg: float = 51.6


def f107_to_rho_scale(f107: float,
                      f107_nominal: float | None = None,
                      log_slope_per_sfu: float | None = None) -> float:
    """F10.7 (sfu) → 350km 密度全局缩放倍数 (解析模型用)。

    log(rho_scale) = k·(F10.7 − F10.7_nominal)。锚定 F10.7=F10.7_nominal → rho_scale=1.0，
    即精确复现 DRAG_CONFIG["rho_ref"] 标定的既有基线。默认 k 取自 config，使
    F10.7 ∈ [70, 250] 映射到旧 log 范围 [-0.7, 0.7] (rho_scale ∈ [0.50, 2.01])。
    """
    if f107_nominal is None:
        f107_nominal = float(DRAG_CONFIG.get("f107_nominal", 150.0))
    if log_slope_per_sfu is None:
        log_slope_per_sfu = float(DRAG_CONFIG.get("f107_log_slope_per_sfu", 0.007))
    log_scale = float(log_slope_per_sfu) * (float(f107) - float(f107_nominal))
    return float(np.exp(log_scale))


def ap_to_storm_multiplier(ap: float,
                           ap_quiet: float | None = None,
                           ap_peak: float | None = None,
                           mult_max: float | None = None) -> float:
    """地磁指数 Ap → 解析模型的密度风暴乘子 ∈ [1.0, mult_max]。

    线性映射：ap_quiet → 1.0，ap_peak → mult_max。锚定 quiet Ap → 乘子 1.0，
    即静日精确复现既有基线；ap_peak (默认 ap_storm_range 上界) → mult_max (≈2.5，
    Starlink 2022 G 级量级)。Ap 是真正的驱动量，乘子由其折算。
    """
    if ap_quiet is None:
        ap_quiet = float(DRAG_CONFIG.get("ap_quiet_range", (3.0, 15.0))[0])
    if ap_peak is None:
        ap_peak = float(DRAG_CONFIG.get("ap_storm_range", (50.0, 400.0))[1])
    if mult_max is None:
        mult_max = float(DRAG_CONFIG.get("storm_peak_multiplier_range", (1.3, 2.5))[1])
    span = max(float(ap_peak) - float(ap_quiet), 1e-6)
    frac = float(np.clip((float(ap) - float(ap_quiet)) / span, 0.0, 1.0))
    return float(1.0 + (float(mult_max) - 1.0) * frac)


def subsatellite_point(phase_rad: float, inclination_deg: float,
                       raan_deg: float, epoch_doy: float) -> dict:
    """由轨道相位 + 倾角 + RAAN + epoch 合成亚卫星点几何 (NRLMSISE-00 输入)。

    近似（圆轨道、相位即纬度幅角 u）：
        φ   = arcsin(sin i · sin u)                          地心纬度
        λ   = RAAN + atan2(cos i · sin u, cos u) − θ_GMST     经度 (deg, [-180,180])
        LST = (λ/15 + 12 + 24·doy_frac_of_day) mod 24         地方太阳时 (h)
    GMST 用 doy 的近似日序推进；绝对精度不重要——F10.7/Ap 已显式传入，
    epoch 只决定季节 (doy) 与昼夜相位 (LST) 几何。
    """
    i = np.deg2rad(float(inclination_deg))
    u = float(phase_rad)
    lat = float(np.rad2deg(np.arcsin(np.clip(np.sin(i) * np.sin(u), -1.0, 1.0))))
    lon_inertial = float(raan_deg) + float(np.rad2deg(
        np.arctan2(np.cos(i) * np.sin(u), np.cos(u))))
    # 近似 GMST：以 doy 的整日部分推进地球自转角（仅用于 LST 相位，不需历元精度）。
    doy = float(epoch_doy)
    gmst_deg = (360.0 * (doy % 1.0)) % 360.0
    lon = ((lon_inertial - gmst_deg + 180.0) % 360.0) - 180.0
    lst_h = (lon / 15.0 + 12.0) % 24.0
    return {"lat_deg": lat, "lon_deg": lon, "lst_h": lst_h, "doy": doy}


# ─────────────────────────────────────────────
# 大气模型基类 + 三档实现
# ─────────────────────────────────────────────
class BaseAtmosphere(ABC):
    """大气模型抽象基类。

    统一契约：density(altitude_m, diurnal_angle_rad) → kg/m³。
    所有模型共享 rho_ref / storm_multiplier 注入接口（向后兼容 robustness 实验与
    既有调度器/预测器），并持有可选的 SpaceWeatherState（NRLMSISE-00 必需）。
    """

    def __init__(self):
        self._rho_ref = float(DRAG_CONFIG["rho_ref"])
        self.H_scale = float(DRAG_CONFIG["H_scale_km"]) * 1e3
        self.ref_alt = float(DRAG_CONFIG["ref_altitude_km"]) * 1e3
        self.enable_diurnal_bulge = bool(
            DRAG_CONFIG.get("enable_diurnal_bulge", True))
        self._storm_multiplier = 1.0
        self._sw: SpaceWeatherState | None = None

    # ── rho_ref：保留 robustness 测试 / experiments 里 `atm.rho_ref *= scale` 的语义 ──
    @property
    def rho_ref(self) -> float:
        return self._rho_ref

    @rho_ref.setter
    def rho_ref(self, value: float) -> None:
        self._rho_ref = float(value)

    # ── storm_multiplier：解析模型的密度风暴瞬态乘子 (由 Ap 折算) ──
    @property
    def storm_multiplier(self) -> float:
        return self._storm_multiplier

    @storm_multiplier.setter
    def storm_multiplier(self, value: float) -> None:
        # 限幅 [1, 5]，覆盖 Starlink 2022 +190% 极值；同时避免病态值进入 drag 公式。
        self._storm_multiplier = float(np.clip(value, 1.0, 5.0))

    def set_space_weather(self, sw: SpaceWeatherState | None) -> None:
        """env.reset 时 push 当前 episode 的空间天气状态。"""
        self._sw = sw

    @property
    def space_weather(self) -> SpaceWeatherState | None:
        return self._sw

    def _effective_rho_ref(self) -> float:
        return self._rho_ref * self._storm_multiplier

    @abstractmethod
    def density(self, altitude_m: float,
                diurnal_angle_rad: float | None = None) -> float:
        ...

    def density_gradient(self, altitude_m: float,
                         diurnal_angle_rad: float | None = None) -> float:
        H_local = self._local_scale_height(altitude_m)
        return -self.density(altitude_m,
                             diurnal_angle_rad=diurnal_angle_rad) / max(H_local, 1.0)

    def _local_scale_height(self, altitude_m: float) -> float:
        return max(self.H_scale, 1.0)


class ExponentialAtmosphere(BaseAtmosphere):
    """单一 scale height 指数模型 (debug / backbone 消融)。

    rho(h, Ψ) = rho_ref_eff · exp(-(h-ref_alt)/H) · (1 + α(h)·cos Ψ)
    H 取 DRAG_CONFIG["H_scale_km"]。物理上在 VLEO 全段失真，仅供对照。
    """

    def density(self, altitude_m: float,
                diurnal_angle_rad: float | None = None) -> float:
        h = float(altitude_m)
        rho_base = self._effective_rho_ref() * float(
            _bounded_exp(-(h - self.ref_alt) / max(self.H_scale, 1.0)))
        rho_base = max(rho_base, 1e-15)
        if not self.enable_diurnal_bulge or diurnal_angle_rad is None:
            return rho_base
        alpha = diurnal_bulge_amplitude(h)
        return max(rho_base * (1.0 + alpha * float(np.cos(float(diurnal_angle_rad)))),
                   1e-15)

    def _local_scale_height(self, altitude_m: float) -> float:
        return max(self.H_scale, 1.0)


class PiecewiseExpAtmosphere(BaseAtmosphere):
    """VLEO Vallado/Wertz 分段指数大气模型，集成日间隆起与地磁暴瞬态调制。

    密度构成 (PDF Section 5 / 8.2)：
        rho(h, Ψ) = vleo_density(h, rho_ref_effective) * (1 + α(h)·cos Ψ)
        rho_ref_effective = rho_ref_base * storm_multiplier(t)

    - rho_ref_base 是用户在 DRAG_CONFIG 设置 / 由 F10.7 折算的太阳活跃水平校准
    - storm_multiplier 是当前地磁活动 (Ap) 折算的瞬态膨胀因子 (默认 1.0)
    - α(h)·cos Ψ 是 Harris-Priester 日间隆起调制，Ψ=0 对应 bulge 峰值
    """

    def density(self, altitude_m: float,
                diurnal_angle_rad: float | None = None) -> float:
        rho_base = vleo_density(
            altitude_m, self._effective_rho_ref(), self.ref_alt, self.H_scale)
        if not self.enable_diurnal_bulge or diurnal_angle_rad is None:
            return rho_base
        alpha = diurnal_bulge_amplitude(altitude_m)
        modulation = 1.0 + alpha * float(np.cos(float(diurnal_angle_rad)))
        return max(rho_base * modulation, 1e-15)

    def _local_scale_height(self, altitude_m: float) -> float:
        return vleo_local_scale_height(altitude_m, self.ref_alt, self.H_scale)


class NRLMSISE00Atmosphere(BaseAtmosphere):
    """NRLMSISE-00 经验大气模型 (pymsis 后端)，F10.7 + Ap 原生驱动。

    昼夜 (diurnal) 与季节变化由 NRLMSISE-00 内部经局部太阳时 + doy 建模，
    因此忽略外部 diurnal_angle_rad 与解析 storm_multiplier（由 env 在选择本模型时
    自动关闭解析日间调制，避免重复计算）。F10.7/Ap 经 SpaceWeatherState 显式传入。

    pymsis 为可选依赖：未安装时构造即抛 ImportError（默认 vallado 模型无需此依赖）。
    """

    def __init__(self):
        super().__init__()
        try:
            import pymsis  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise ImportError(
                "atmosphere_model='nrlmsise00' 需要 pymsis：pip install pymsis。"
                "默认 'vallado' 模型无需该依赖。"
            ) from exc
        # calculate 在不同 pymsis 版本中位置不同：新版顶层 pymsis.calculate，
        # 旧版 pymsis.msis.calculate。两者都试，取到为准。
        calc = getattr(pymsis, "calculate", None)
        if calc is None:
            from pymsis import msis as _msis  # type: ignore[import-untyped]
            calc = _msis.calculate
        self._calc = calc
        # NRLMSISE-00 自带昼夜建模，关闭解析 cos Ψ 调制避免重复计算。
        self.enable_diurnal_bulge = False
        self._cal_floor_kg_m3 = 1e-15

    def _query(self, altitude_m: float) -> float:
        sw = self._sw or SpaceWeatherState()
        geo = subsatellite_point(
            phase_rad=float(getattr(self, "_phase_rad", 0.0)),
            inclination_deg=sw.inclination_deg,
            raan_deg=sw.raan_deg,
            epoch_doy=sw.epoch_doy,
        )
        # pymsis 需要 numpy datetime64；用 doy 合成同年日期 (绝对年份不影响，F10.7/Ap 显式传入)。
        day_idx = int(np.clip(sw.epoch_doy, 1, 365)) - 1
        date = np.datetime64("2020-01-01") + np.timedelta64(day_idx, "D")
        # 全部以长度 1 的数组传入，aps 用文档要求的 (ndates, 7) 形状，避免标量/1D 形状歧义。
        aps = np.array([[float(sw.ap)] * 7], dtype=np.float64)
        out = self._calc(
            np.array([date]),
            np.array([float(geo["lon_deg"])]),
            np.array([float(geo["lat_deg"])]),
            np.array([float(altitude_m) / 1000.0]),  # pymsis 高度单位 km
            np.array([float(sw.f107_daily)]),
            np.array([float(sw.f107_81avg)]),
            aps,
        )
        rho = float(np.asarray(out).ravel()[0])  # index 0 = 总质量密度 kg/m³
        if not np.isfinite(rho) or rho <= 0.0:
            return self._cal_floor_kg_m3
        return max(rho, self._cal_floor_kg_m3)

    def set_phase(self, phase_rad: float) -> None:
        """env 每步把当前轨道相位推进来，用于合成亚卫星点。"""
        self._phase_rad = float(phase_rad)

    def density(self, altitude_m: float,
                diurnal_angle_rad: float | None = None) -> float:
        # diurnal_angle_rad 被忽略：NRLMSISE-00 内部已处理昼夜。
        return self._query(altitude_m)

    def _local_scale_height(self, altitude_m: float) -> float:
        # 有限差分估计局部 scale height：H = -rho / (d rho / dh)。
        h = float(altitude_m)
        dh = 2000.0
        rho0 = self._query(h)
        rho1 = self._query(h + dh)
        if rho1 <= 0.0 or rho0 <= 0.0 or rho1 >= rho0:
            return max(self.H_scale, 1.0)
        return float(dh / max(np.log(rho0 / rho1), 1e-6))


# 向后兼容：旧代码 / 测试以 `AtmosphericModel` 引用默认 Vallado 模型。
AtmosphericModel = PiecewiseExpAtmosphere


_ATMOSPHERE_REGISTRY = {
    "exponential": ExponentialAtmosphere,
    "vallado": PiecewiseExpAtmosphere,
    "nrlmsise00": NRLMSISE00Atmosphere,
}


def make_atmosphere(name: str | None = None) -> BaseAtmosphere:
    """大气模型工厂。name 缺省时读 DRAG_CONFIG["atmosphere_model"]，再缺省为 'vallado'。"""
    if name is None:
        name = str(DRAG_CONFIG.get("atmosphere_model", "vallado"))
    key = str(name).strip().lower()
    if key not in _ATMOSPHERE_REGISTRY:
        raise ValueError(
            f"未知 atmosphere_model={name!r}；可选 {sorted(_ATMOSPHERE_REGISTRY)}。")
    return _ATMOSPHERE_REGISTRY[key]()

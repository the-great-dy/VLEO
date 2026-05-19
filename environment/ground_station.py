"""
地面站可见性、链路容量和通信窗口模型。
地面站可见性与通信窗口模型

核心约定：
  1. elevation_angle 使用 arctan2 公式，并用 gamma_max 显式排除地平线以下目标
  2. channel_capacity_mbps 使用法余弦公式计算斜距和 SNR
  3. 链路容量按 AMC/MCS 离散档位输出，不使用连续 Shannon 容量作为最终 Mbps
  4. 低仰角通信容量会额外考虑多普勒/大气路径惩罚
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    # 降低导入劫持风险：不要把项目路径插到最高优先级
    sys.path.append(_PROJECT_ROOT)

import numpy as np
from config import GROUND_STATION_CONFIG, ORBITAL_CONFIG


class GroundStation:
    """单个地面站"""

    def __init__(self, lat_deg: float, lon_deg: float,
                 name: str = "GS",
                 min_elevation_deg: float = 5.0,
                 bandwidth_mhz: float = 100.0,
                 tx_power_dbw: float = 40.0,
                 atmospheric_refraction_enabled: bool | None = None):
        self.lat    = np.radians(lat_deg)
        self.lon    = np.radians(lon_deg)
        self.name   = name
        self.min_el = np.radians(min_elevation_deg)
        self.B      = bandwidth_mhz * 1e6
        self.tx_dbw = tx_power_dbw
        self.atmospheric_refraction_enabled = bool(
            GROUND_STATION_CONFIG.get("atmospheric_refraction_enabled", False)
            if atmospheric_refraction_enabled is None
            else atmospheric_refraction_enabled
        )

    def elevation_angle(self, satellite_lat: float,
                        satellite_lon: float,
                        altitude_m: float) -> float:
        """
        计算卫星对本地面站的仰角（弧度）
        使用 arctan2 公式，负值表示地平线以下（不可见）

        公式推导（球面三角）：
          rho   = R_e / (R_e + h)          归一化地球半径
          gamma = 地心角（站-地心-星）
          el    = arctan2(cos(gamma) - rho, sin(gamma))
        """
        # 仰角先由地心角推导，可见性判断必须同时满足地平线和最小仰角约束。
        R_e = ORBITAL_CONFIG["earth_radius_km"] * 1e3
        r   = R_e + altitude_m
        rho = R_e / r

        # 地心角 gamma
        cos_gamma = (np.sin(self.lat) * np.sin(satellite_lat) +
                     np.cos(self.lat) * np.cos(satellite_lat) *
                     np.cos(satellite_lon - self.lon))
        cos_gamma = float(np.clip(cos_gamma, -1.0, 1.0))
        gamma     = np.arccos(cos_gamma)

        # 最大可见地心角（超过则卫星在地平线以下）
        gamma_max = np.arccos(rho)   # arccos(R_e / (R_e+h))

        if gamma >= gamma_max:
            return -0.1   # 明确返回负值 = 不可见

        # 仰角（arctan2 公式，正值 = 可见）
        el = np.arctan2(cos_gamma - rho, np.sin(gamma))
        return float(self._apply_atmospheric_refraction(el))

    def _apply_atmospheric_refraction(self, elevation_rad: float) -> float:
        """对低仰角做轻量折射修正，避免窗口过早被地平线截断。"""
        if not self.atmospheric_refraction_enabled:
            return float(elevation_rad)

        elevation_deg = float(np.degrees(elevation_rad))
        if elevation_deg <= -1.0:
            return float(elevation_rad)

        # Bennett 近似：只用于低仰角的工程修正，不作为高精度大气模型。
        refraction_arcmin = 1.02 / np.tan(np.radians(elevation_deg + 10.3 / (elevation_deg + 5.11)))
        refraction_deg = float(np.clip(refraction_arcmin / 60.0, 0.0, 1.0))
        return float(elevation_rad + np.radians(refraction_deg))

    def is_visible(self, satellite_lat: float,
                   satellite_lon: float,
                   altitude_m: float) -> bool:
        el = self.elevation_angle(satellite_lat, satellite_lon, altitude_m)
        return el >= self.min_el

    def channel_capacity_mbps(self, elevation_rad: float,
                               altitude_m: float,
                               gamma: float = None) -> float:
        """
        简化链路容量 (Mbps)

        正确的 gamma 反推公式（由 el 推 gamma）：
          在三角形（地心O, 地面站G, 卫星S）中：
          角G = 90° + el，角O = gamma，则：
          gamma = arccos(rho * cos(el)) - el
          其中 rho = R_e / (R_e + h)

        链路预算参数（小卫星S频段下行典型值）：
          卫星发射功率：5 W = 7 dBW
          卫星天线增益：5 dBi（贴片天线）
          地面站接收天线增益：30 dBi（0.6m 口径碟形天线）
          有效发射 EIRP = 7 + 5 = 12 dBW
          有效接收增益加入 tx_dbw（40 dBW = EIRP + 地面接收增益）
          系统噪声温度：100 K
        """
        # 低于最小仰角直接视为无链路，避免低仰角下给出不现实容量。
        if elevation_rad < self.min_el:
            return 0.0

        R_e    = ORBITAL_CONFIG["earth_radius_km"] * 1e3
        r      = R_e + altitude_m
        rho    = R_e / r
        cos_el = np.cos(elevation_rad)
        sin_el = np.sin(elevation_rad)

        # 正确 gamma 反推：gamma = arccos(rho * cos_el) - el
        arg   = float(np.clip(rho * cos_el, -1.0, 1.0))
        gamma_val = np.arccos(arg) - elevation_rad
        if gamma_val < 0:
            return 0.0

        # 斜距使用法余弦公式，避免低仰角几何下的数值不稳定。
        cos_gamma_val = np.cos(gamma_val)
        slant_m = np.sqrt(R_e**2 + r**2 - 2.0*R_e*r*cos_gamma_val)
        slant_km = max(slant_m / 1e3, 100.0)

        # 自由空间路径损耗（2 GHz）
        freq_hz = 2e9
        c       = 3e8
        fspl_db = 20 * np.log10(4 * np.pi * slant_km * 1e3 * freq_hz / c)

        # 噪声功率：用系统噪声温度 T_sys=100K 代替 NF 模型
        k_boltz   = 1.38e-23
        T_sys     = 100.0            # 系统噪声温度 (K)
        noise_dbw = 10 * np.log10(k_boltz * T_sys * self.B)

        # 有效发射功率（卫星EIRP + 地面接收天线增益）
        effective_eirp = self.tx_dbw   # 初始化时设为 40 dBW

        rx_dbw     = effective_eirp - fspl_db
        snr_db     = rx_dbw - noise_dbw
        snr_linear = max(10 ** (snr_db / 10), 0.0)

        if bool(GROUND_STATION_CONFIG.get("amc_enabled", True)):
            # 实际链路按 MCS 档位切换，不会随 SNR 连续平滑变化。
            # Shannon 公式只用于理解链路预算，不作为最终可用 Mbps。
            capacity_bps = self._amc_capacity_mbps(snr_db) * 1e6
        else:
            capacity_bps = self.B * np.log2(1.0 + snr_linear)

        # ── 多普勒频移与低仰角链路惩罚 ─────────────────────────────
        # VLEO @ 300km，轨道速度 ≈ 7.8 km/s，对地面站产生极大多普勒频偏。
        # 低仰角时（5°~10°）径向速度分量最大，频偏可达 ±200 kHz（S频段2GHz）。
        # 实际影响：接收机AFC追踪延迟 + 雨衰（低仰角大气路径长）导致有效容量下降。
        # 修正因子参考：Del Portillo et al. (2019), ITU-R S.1257
        #
        # 仰角越低 → 多普勒+大气衰减越严重 → 有效容量系数越小
        # el=90°（天顶）: penalty≈0（满容量）
        # el=10°（低仰角）: penalty≈0.30（容量降至70%）
        # el=5°（最低仰角）: penalty≈0.45（容量降至55%）
        el_deg = np.degrees(elevation_rad)
        doppler_penalty = 0.45 * np.exp(-(el_deg - 5.0) / 20.0)
        doppler_penalty = float(np.clip(doppler_penalty, 0.0, 0.45))
        capacity_effective_bps = capacity_bps * (1.0 - doppler_penalty)
        max_capacity_mbps = float(GROUND_STATION_CONFIG.get(
            "max_channel_capacity_mbps", 0.0))
        if max_capacity_mbps > 0.0:
            capacity_effective_bps = min(
                capacity_effective_bps,
                max_capacity_mbps * 1e6,
            )

        return float(capacity_effective_bps / 1e6)

    @staticmethod
    def _amc_spectral_efficiency(snr_db: float) -> float:
        """根据 SNR 选择离散调制编码档位的频谱效率。"""
        thresholds = list(GROUND_STATION_CONFIG.get(
            "amc_snr_thresholds_db", [-3.0, 3.0, 8.0, 13.0, 18.0]))
        efficiencies = list(GROUND_STATION_CONFIG.get(
            "amc_spectral_efficiencies", [0.25, 0.5, 1.0, 2.0, 3.0]))
        if not thresholds or not efficiencies:
            return 0.0

        level = 0.0
        for threshold, efficiency in zip(thresholds, efficiencies):
            if float(snr_db) >= float(threshold):
                level = float(efficiency)
            else:
                break
        return max(level, 0.0)

    @staticmethod
    def _amc_capacity_mbps(snr_db: float) -> float:
        """
        根据 SNR 选择离散 MCS 容量档位。

        配置中的 capacity_levels 长度应比 thresholds 多 1：
        低于第一个门限为 0 档，之后依次切到 QPSK/16QAM/64QAM 等工程档位。
        若未配置容量表，则回退到 bandwidth * spectral_efficiency 的离散结果。
        """
        thresholds = [
            float(x) for x in GROUND_STATION_CONFIG.get(
                "amc_snr_thresholds_db", [-3.0, 3.0, 8.0, 13.0, 18.0])
        ]
        levels = GROUND_STATION_CONFIG.get("amc_capacity_levels_mbps", None)
        if levels is None:
            bandwidth_mhz = float(GROUND_STATION_CONFIG.get("bandwidth_mhz", 100.0))
            return bandwidth_mhz * GroundStation._amc_spectral_efficiency(snr_db)

        capacity_levels = [float(x) for x in levels]
        if not capacity_levels:
            return 0.0
        level_idx = 0
        for threshold in thresholds:
            if float(snr_db) >= threshold:
                level_idx += 1
            else:
                break
        level_idx = min(level_idx, len(capacity_levels) - 1)
        return max(float(capacity_levels[level_idx]), 0.0)


class GroundStationNetwork:
    """地面站网络"""

    DEFAULT_STATIONS = GROUND_STATION_CONFIG["stations"]

    def __init__(self, station_configs: list = None,
                 min_elevation_deg: float = 5.0,
                 atmospheric_refraction_enabled: bool | None = None):
        configs = station_configs or self.DEFAULT_STATIONS
        self.atmospheric_refraction_enabled = bool(
            GROUND_STATION_CONFIG.get("atmospheric_refraction_enabled", False)
            if atmospheric_refraction_enabled is None
            else atmospheric_refraction_enabled
        )
        self.stations = [
            GroundStation(
                lat_deg=cfg["lat"], lon_deg=cfg["lon"],
                name=cfg.get("name", "GS"),
                min_elevation_deg=min_elevation_deg,
                atmospheric_refraction_enabled=self.atmospheric_refraction_enabled,
            )
            for cfg in configs
        ]
        self.n_stations = len(self.stations)

    @staticmethod
    def _raan_drift_rate_rad_s(altitude_m: float, inclination_deg: float) -> float:
        """基于 J2 的 RAAN 长期漂移率，单位 rad/s。"""
        if not bool(ORBITAL_CONFIG.get("enable_j2", False)):
            return 0.0

        j2 = float(ORBITAL_CONFIG.get("J2", 0.0))
        if j2 <= 0.0:
            return 0.0

        mu = float(ORBITAL_CONFIG["mu"])
        R_e = float(ORBITAL_CONFIG["earth_radius_km"]) * 1e3
        a = R_e + float(altitude_m)
        n = np.sqrt(mu / a**3)
        inclination = np.radians(inclination_deg)
        return float(-1.5 * j2 * (R_e / a) ** 2 * n * np.cos(inclination))

    @staticmethod
    def satellite_position(time_s: float, altitude_m: float,
                           inclination_deg: float | None = None,
                           raan_deg: float | None = None) -> tuple:
        """圆轨道卫星位置（纬度、经度），单位弧度"""
        mu      = ORBITAL_CONFIG["mu"]
        R_e     = ORBITAL_CONFIG["earth_radius_km"] * 1e3
        if inclination_deg is None:
            inclination_deg = float(ORBITAL_CONFIG.get("inclination_deg", 51.6))
        if raan_deg is None:
            raan_deg = float(ORBITAL_CONFIG.get("raan_deg", 0.0))
        r       = R_e + altitude_m
        omega   = np.sqrt(mu / r**3)
        theta   = (omega * time_s) % (2 * np.pi)

        incl    = np.radians(inclination_deg)
        raan    = np.radians(raan_deg) + GroundStationNetwork._raan_drift_rate_rad_s(
            altitude_m, inclination_deg) * time_s

        lat     = np.arcsin(np.sin(incl) * np.sin(theta))
        earth_rot = 7.2921e-5
        lon     = (raan + np.arctan2(np.cos(incl) * np.sin(theta),
                                      np.cos(theta))
                   - earth_rot * time_s) % (2 * np.pi)
        if lon > np.pi:
            lon -= 2 * np.pi

        return float(lat), float(lon)

    def get_contact_info(self, time_s: float,
                         altitude_m: float) -> dict:
        """计算当前时刻通信状态"""
        # 多站网络取当前可见站中容量最大的那一个，作为本时隙可用下传能力。
        sat_lat, sat_lon = self.satellite_position(
            time_s,
            altitude_m,
            inclination_deg=ORBITAL_CONFIG.get("inclination_deg", 51.6),
            raan_deg=ORBITAL_CONFIG.get("raan_deg", 0.0),
        )

        max_cap  = 0.0
        best_el  = 0.0
        in_window = False

        for gs in self.stations:
            el = gs.elevation_angle(sat_lat, sat_lon, altitude_m)
            if el >= gs.min_el:
                in_window = True
                cap = gs.channel_capacity_mbps(el, altitude_m)
                if cap > max_cap:
                    max_cap = cap
                    best_el = el

        time_to_next = self._predict_next_window(
            time_s, altitude_m, currently_in=in_window)
        window_remaining = (self._predict_window_end(time_s, altitude_m)
                            if in_window else 0.0)

        return {
            "in_window":             bool(in_window),
            "max_capacity_mbps":     float(max_cap),
            "best_elevation_rad":    float(best_el),
            "best_elevation_deg":    float(np.degrees(best_el)),
            "time_to_next_window_s": float(time_to_next),
            "window_remaining_s":    float(window_remaining),
            "sat_lat_deg":           float(np.degrees(sat_lat)),
            "sat_lon_deg":           float(np.degrees(sat_lon)),
        }

    def _predict_next_window(self, time_s, altitude_m,
                              currently_in=False,
                              scan_step_s=10.0,
                              max_scan_s=5400.0) -> float:
        """
        扫描未来接触窗口的开始时间。

        若当前已在窗口内，先找到本窗口结束点，再继续向后找下一次进入窗口的时刻。
        该近似以 scan_step_s 为时间分辨率，主要服务于状态特征 t_to_window。
        """
        if currently_in:
            t = time_s + scan_step_s
            for _ in range(int(max_scan_s / scan_step_s)):
                lat, lon = self.satellite_position(t, altitude_m)
                if not any(gs.is_visible(lat, lon, altitude_m)
                           for gs in self.stations):
                    t_out = t
                    break
                t += scan_step_s
            else:
                return max_scan_s
            t = t_out + scan_step_s
            for _ in range(int(max_scan_s / scan_step_s)):
                lat, lon = self.satellite_position(t, altitude_m)
                if any(gs.is_visible(lat, lon, altitude_m)
                       for gs in self.stations):
                    return t - time_s
                t += scan_step_s
            return max_scan_s
        else:
            t = time_s + scan_step_s
            for _ in range(int(max_scan_s / scan_step_s)):
                lat, lon = self.satellite_position(t, altitude_m)
                if any(gs.is_visible(lat, lon, altitude_m)
                       for gs in self.stations):
                    return t - time_s
                t += scan_step_s
            return max_scan_s

    def _predict_window_end(self, time_s, altitude_m,
                             scan_step_s=10.0,
                             max_scan_s=600.0) -> float:
        """扫描当前通信窗口剩余时间，若扫描范围内一直可见则返回 max_scan_s。"""
        t = time_s + scan_step_s
        for _ in range(int(max_scan_s / scan_step_s)):
            lat, lon = self.satellite_position(t, altitude_m)
            if not any(gs.is_visible(lat, lon, altitude_m)
                       for gs in self.stations):
                return t - time_s
            t += scan_step_s
        return max_scan_s

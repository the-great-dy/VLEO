"""
VLEO 卫星调度环境（含通信窗口约束）。

当前建模要点：
  1. 轨道相位使用积分法 advance_phase，避免 time_s % T(h) 引起相位跳变
  2. 推进器按 N_PROP_SMOOTH 节奏更新，安全覆盖触发时可临时突破平滑
  3. 观测为任务价值交付状态，覆盖轨道、能源、通信、分组队列、任务价值、热状态、场景语义、历史动作和安全压力
  4. contact 在 time_s 更新之后计算，确保通信窗口和观测处于同一时刻
  5. 主链路为 raw_queue → 星上计算处理 → processed_queue → 通信窗口下传 → 地面获得任务价值
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    # 降低导入劫持风险：不要把项目路径插到最高优先级
    sys.path.append(_PROJECT_ROOT)

import numpy as np
from config import (ORBITAL_CONFIG, DRAG_CONFIG, ENERGY_CONFIG, THERMAL_CONFIG,
                    PROPULSION_CONTROLLER_CONFIG, PROPELLANT_CONFIG, ATTITUDE_CONFIG,
                    QUEUE_CONFIG, TASK_CONFIG,
                    DRL_CONFIG, PROCESSING_CREDIT_CONFIG,
                    TRAIN_CONFIG, REWARD_CONFIG, GROUND_STATION_CONFIG,
                    HARD_RULES_CONFIG, SAFE_BUDGET_FALLBACK_CONFIG)
try:
    from config import ACTUATOR_GATE_CONFIG  # type: ignore
except ImportError:
    ACTUATOR_GATE_CONFIG = {"cpu_gate_soft_mode": False}
from environment.orbital_dynamics import (
    OrbitalDynamics, OrbitalPeriodSimulator, eclipse_fraction_from_beta,
    SpaceWeatherState, f107_to_rho_scale, ap_to_storm_multiplier)
from environment.energy_model import SolarPanelModel, BatteryModel, PowerSubsystem
from environment.ground_station import GroundStationNetwork
from environment.task_value_model import TaskValueTracker
from constraints.safety_cost import (
    compute_lyapunov_safety_cost,
    compute_state_safety_penalty,
)
from objectives.mission_reward import compute_mission_reward
from safety.actuator_constraints import (
    ActuatorConstraintFilter,
    BoundedActionSanitizer,
)
from utils.action_space import (decode_grouped_action, pointing_mode_from_unit,
                                 pointing_unit_for_mode, POINTING_IMAGE,
                                 POINTING_DOWNLINK, POINTING_SUN, IDX_POINTING)
from virtual_queues.energy_queue import (EnergyVirtualQueue,
                                          OrbitVirtualQueue, DataTaskQueue)
from virtual_queues.comm_queue import CommWindowQueue


# 观测向量的唯一顺序表。网络、评估脚本和可视化脚本必须按这个顺序解释状态。
OBSERVATION_FEATURES = [
    "altitude_norm",               # 0  当前高度归一化
    "drag_strength_norm",          # 1  阻力强度归一化
    "altitude_safety_margin_norm", # 2  高度安全裕度
    "soc",                         # 3  电池 SOC
    "solar_input_norm",            # 4  太阳能输入
    "last_total_power_norm",       # 5  上一时隙总功耗
    "in_comm_window",              # 6  当前是否在通信窗口
    "time_to_next_window_norm",     # 7  距下一窗口时间
    "window_remaining_norm",       # 8  当前窗口剩余时间
    "tx_capacity_norm",            # 9  当前链路容量
    "raw_queue_utilization",       # 10 原始数据队列大小
    "processed_queue_utilization", # 11 已处理队列大小
    "raw_high_queue_utilization",
    "raw_mid_queue_utilization",
    "raw_low_queue_utilization",
    "processed_high_queue_utilization",
    "processed_mid_queue_utilization",
    "processed_low_queue_utilization",
    "expiring_high_value_norm",
    "expiring_mid_value_norm",
    "expiring_low_value_norm",
    "expiring_value_norm",
    "total_processed_value_norm",
    "topk_priority_norm",
    "topk_quality_norm",
    "deadline_urgency",
    "prev_alpha_prop",
    "prev_alpha_cpu",
    "prev_alpha_tx",
    "energy_queue_pressure",
    "orbit_queue_pressure",
    "raw_queue_pressure",
    "processed_queue_future_contact_ratio",
    "prop_update_phase",
    "current_scene_class_norm",
    "upcoming_task_intensity_norm",
    "future_contact_capacity_norm",
    "cpu_backpressure_ratio",
    "next_window_in_range",
    "thermal_margin_norm",
    "processed_high_next_window_deliverable_ratio",
    "raw_high_next_window_deliverable_ratio",
    "high_value_deadline_contact_mismatch",
    "capacity_bin_0_mb_norm",
    "capacity_bin_0_time_norm",
    "capacity_bin_1_mb_norm",
    "capacity_bin_1_time_norm",
    "capacity_bin_2_mb_norm",
    "capacity_bin_2_time_norm",
    "capacity_bin_3_mb_norm",
    "capacity_bin_3_time_norm",
    "capacity_bin_4_mb_norm",
    "capacity_bin_4_time_norm",
    "capacity_bin_5_mb_norm",
    "capacity_bin_5_time_norm",
    "capacity_bin_6_mb_norm",
    "capacity_bin_6_time_norm",
    "capacity_bin_7_mb_norm",
    "capacity_bin_7_time_norm",
    "concurrent_high_same_class_mb_norm",
    "concurrent_medium_same_class_mb_norm",
    "concurrent_low_same_class_mb_norm",
    "propellant_fraction",         # [SAFETY-REAL] 剩余推进剂比例(0-1):agent 据此学习省油/寿命管理
    "pointing_mode_norm",          # [SAFETY-REAL] 当前指向模式归一化(IMAGE/DOWNLINK/SUN → 0/0.5/1.0)
    "momentum_norm",               # [SAFETY-REAL] 反作用轮归一化动量(0-1):接近 1 需去饱和
]

_GYM_AVAILABLE = False
try:
    import gymnasium  # type: ignore[import-untyped]
    from gymnasium import spaces  # type: ignore[import-untyped]
    _GYM_AVAILABLE = True
except ImportError:
    try:
        import gym  # type: ignore[import-untyped]
        from gym import spaces  # type: ignore[import-untyped]
        _GYM_AVAILABLE = True
    except ImportError:
        spaces = None  # type: ignore[assignment]


class VLEOSatelliteEnv:

    def __init__(self, seed: int = 42, gs_config: dict = None):
        self.seed = seed
        self.rng  = np.random.default_rng(seed)
        # 风暴事件用独立 rng，避免新增的 prob 抽样改动主 rng 序列、
        # 破坏既有 seed 复现性 (scene / emergency / data 都吃主 rng)。
        self._storm_rng = np.random.default_rng(seed + 0xA17_F1A2C)
        # 物理 domain randomization (太阳活跃度 + β 角阴影) 用独立 rng，
        # 同样为了避免 reset 时新增的抽样改动主 rng 序列。
        self._physics_rng = np.random.default_rng(seed + 0xC51A_BEEF)
        # base rho_ref 用于每 episode 重置时的 rho_scale 缩放基准
        self._base_rho_ref = float(DRAG_CONFIG["rho_ref"])

        # 环境由轨道、能源、通信窗口和队列四部分组成，step() 中会按物理时序逐一推进。
        self.orbit_dyn  = OrbitalDynamics()
        self.orbit_sim  = OrbitalPeriodSimulator()
        self.solar      = SolarPanelModel()
        self.battery    = BatteryModel()
        self.power_sys  = PowerSubsystem()
        self.battery.set_rng(self.rng)

        cfg = gs_config or GROUND_STATION_CONFIG
        station_configs = cfg.get("stations")
        profile_name = cfg.get("profile")
        profiles = cfg.get("profiles", {})
        if profile_name in profiles:
            station_configs = profiles[profile_name]
        self.gs_network = GroundStationNetwork(
            station_configs=station_configs,
            min_elevation_deg=cfg["min_elevation_deg"],
            atmospheric_refraction_enabled=cfg.get(
                "atmospheric_refraction_enabled",
                GROUND_STATION_CONFIG.get("atmospheric_refraction_enabled", False),
            ))

        self.energy_queue = EnergyVirtualQueue()
        self.orbit_queue  = OrbitVirtualQueue()
        self.data_queue   = DataTaskQueue()
        self.comm_queue   = CommWindowQueue()
        self.task_tracker = TaskValueTracker(TASK_CONFIG)

        # 推进器不是每个 10s 控制步都允许大幅改变，用 N_PROP_SMOOTH 模拟推进控制的执行周期。
        self.dt        = TRAIN_CONFIG["time_slot_s"]
        self.max_steps = TRAIN_CONFIG["max_episode_steps"]
        self.N_PROP_SMOOTH = 3   # 6→3: 推力调整响应快 1 倍，减少衰减期边界震荡锁定，配合紧急恢复杜绝 crash

        self.state_dim  = int(DRL_CONFIG.get("state_dim", len(OBSERVATION_FEATURES)))
        if self.state_dim != len(OBSERVATION_FEATURES):
            raise ValueError(
                f"DRL_CONFIG['state_dim']={self.state_dim} 与 当前观测维度 "
                f"{len(OBSERVATION_FEATURES)} 不一致，请同步修改 OBSERVATION_FEATURES。"
            )
        self.action_dim = int(DRL_CONFIG.get("action_dim", 10))
        self.action_sanitizer = BoundedActionSanitizer(
            action_dim=self.action_dim,
            dtype=np.float32,
        )
        self.actuator_filter = ActuatorConstraintFilter(
            baseline_w=float(ENERGY_CONFIG["power_baseline_w"]),
            total_limit_w=float(ENERGY_CONFIG.get("power_total_max_w", 120.0)),
            prop_ignition_threshold_w=float(
                ENERGY_CONFIG.get("propulsion_ignition_threshold_w", 0.0)),
            action_dim=self.action_dim,
        )

        self._h_warning = ORBITAL_CONFIG.get("altitude_warning_km", 180.0) * 1e3
        self._h_min = ORBITAL_CONFIG["altitude_min_km"] * 1e3
        self._h_crash = ORBITAL_CONFIG.get("altitude_crash_km", 122.0) * 1e3
        self._h_max = ORBITAL_CONFIG["altitude_max_km"] * 1e3
        self._max_window_time = 600.0
        self._time_to_next_window_norm_s = float(
            TASK_CONFIG.get("time_to_next_window_norm_s", 5400.0))
        self._data_arrival_scale = 1.0
        # 全程均匀随机 ds 模式（默认关闭）。关闭时由 curriculum 按阶段控制 ds（推荐做法）；
        # 开启时每 episode 从 _random_ds_range 均匀随机抽 ds，仅用于消融对比，会破坏 curriculum 效果。
        self._random_ds_enabled = bool(TRAIN_CONFIG.get("train_random_ds_enabled", False))
        self._random_ds_range = tuple(TRAIN_CONFIG.get("train_random_ds_range", (0.5, 1.0)))
        # Domain randomization curriculum 缩放因子 ∈ [0, 1]。
        # 训练循环根据课程阶段 (Exploration/Balancing/Ramp/Optimization) 写入对应值，
        # env.reset() 时把 rho_scale 范围、β 上界、storm peak/概率全部按此因子线性收缩。
        # 1.0 = 完整随机化（PDF 物理极值），0 = 完全确定性（debug 用）。
        self._randomization_scale = 1.0
        self._last_total_power_w = ENERGY_CONFIG.get("power_baseline_w", 15.0)
        self._last_available_power_w = ENERGY_CONFIG.get("power_total_max_w", 120.0)
        self._last_delivery_info = {}
        self._last_scene_context = {}
        self._last_cpu_backpressure_ratio = 0.0
        self._scene_phase_offset_fraction = 0.0
        # Per-episode 打乱后的 phase 块（reset 时重建）；init 时默认走 TASK_CONFIG 原始顺序。
        self._phase_scene_rules = list(TASK_CONFIG.get("phase_scene_rules", []))
        self._emergency_event_remaining_steps = 0
        self._emergency_event_cooldown_steps = 0
        self._last_emergency_event_active = False
        self._last_emergency_event_triggered = False
        # 地磁暴瞬态事件状态 (PDF Section 8.2 - Starlink 2022 教训)。
        # 触发后 atm.storm_multiplier 暂态上升 1.3~2.5x (三角剖面)，几百步内回归。
        self._storm_active_steps_total = 0
        self._storm_active_steps_remaining = 0
        self._storm_cooldown_remaining = 0
        self._storm_peak_multiplier = 1.0
        self._last_storm_multiplier = 1.0
        # 空间天气状态 (F10.7/Ap)：reset 时按 episode 抽样填充；__init__ 给安全默认。
        self._sw_state = None
        _ap_q0 = float(DRAG_CONFIG.get("ap_quiet_range", (3.0, 15.0))[0])
        self._episode_quiet_ap = _ap_q0
        self._storm_peak_ap = _ap_q0
        self._last_ap = _ap_q0
        self._last_future_contact_capacity_norm = 0.0
        self._comm_window_age_steps = 0
        self._comm_pass_capacity_mb = float(
            GROUND_STATION_CONFIG.get("max_downlink_mb_per_pass", 0.0))
        self._comm_pass_remaining_mb = self._comm_pass_capacity_mb
        self.thermal_temperature_c = float(
            THERMAL_CONFIG.get("initial_temp_c", 20.0))

        # 仅放宽 drag_strength_norm 维度（index=1），其余维度保持原有上界 2.0。
        self._obs_low = np.full((self.state_dim,), -1.0, dtype=np.float32)
        self._obs_high = np.full((self.state_dim,), 2.0, dtype=np.float32)
        if self.state_dim > 1:
            self._obs_high[1] = 5.0

        # 使用初始化时固定的参考阻力，避免 rho_ref 扰动在分子/分母中相互抵消
        self._drag_ref_force_hmin = float(max(self.orbit_dyn.drag_force(self._h_min), 1e-12))

        if _GYM_AVAILABLE and spaces is not None:
            self.observation_space = spaces.Box(
                low=self._obs_low, high=self._obs_high, dtype=np.float32)
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        self.altitude_m = None
        self.time_s = None
        self.step_count = None
        self.prev_action = None
        self.episode_reward = None
        self._contact = None
        self._contact_override = None
        self._processed_since_contact_mb = 0.0
        self._delivered_since_contact_mb = 0.0
        self._raw_equivalent_processed_since_contact_mb = 0.0
        self._raw_equivalent_delivered_since_contact_mb = 0.0
        self._prev_in_window_for_budget = False
        # ── step→_get_observation 缓存（干掉同一步内重复调用昂贵扫描函数）──
        self._cached_step = -1
        self._cached_deadline_contact_stats = {}
        self._cached_deliverability_features = {}
        self._cached_bins_step = -1
        self._cached_bins_value = []

    def reseed_rngs(self, seed: int) -> None:
        """把所有随机源重置到给定 seed（与 __init__ 同样的 offset）。
        评估时用它保证每次 evaluate() 都在完全相同的场景序列上打分，消除 eval
        环境被反复复用、reset() 不重播种导致的跨次评估 RNG 漂移。"""
        seed = int(seed)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._storm_rng = np.random.default_rng(seed + 0xA17_F1A2C)
        self._physics_rng = np.random.default_rng(seed + 0xC51A_BEEF)
        self.battery.set_rng(self.rng)

    def reset(self) -> np.ndarray:
        # 多 ds 混合训练：训练 env 每 episode 随机抽 data_arrival_scale，
        # 让策略对负载强度鲁棒（泛化），而不是死记单一 ds 的最优解。
        # eval 路径会把 _random_ds_enabled 置 False 并显式设定固定 ds。
        if self._random_ds_enabled:
            lo, hi = self._random_ds_range
            self._data_arrival_scale = float(self.rng.uniform(float(lo), float(hi)))
        # 每个 episode 随机化初始高度和轨道相位，避免策略只记住固定窗口/日照模式。
        # 初始高度下界取 warning_km + 20km，确保高 rho_scale 的 episode 不会在第一步就进入警告区：
        # 200km 处 nominal drag ~15mN; rho×2 → drag ~30mN, 仍可用 720W 维持; 但 warning zone 没有缓冲。
        # +20km 给出约 10km/orbit 的高度余量，让 agent 在 episode 初期有时间学到"满推维持轨道"。
        initial_altitude_min_km = max(
            float(ORBITAL_CONFIG.get("altitude_warning_km", 200.0)) + 20.0,
            float(ORBITAL_CONFIG["altitude_min_km"]) * 1.20,
        )
        self.altitude_m = self.rng.uniform(
            initial_altitude_min_km,
            ORBITAL_CONFIG["altitude_max_km"] * 0.95) * 1e3
        self.time_s = self.rng.uniform(0, self.orbit_sim.period_s)

        # [SAFETY-REAL] 有限推进剂:每个 episode 满箱起步;耗尽→推力=0→阻力主导→衰减再入。
        self._propellant_enabled = bool(PROPELLANT_CONFIG.get("enabled", False))
        self._propellant_initial_kg = float(PROPELLANT_CONFIG.get("initial_mass_kg", 30.0))
        # 统一时间压缩 C:衰减/推力响应/燃料消耗同尺度
        self._orbital_time_compression = float(PROPELLANT_CONFIG.get("orbital_time_compression", 1.0))
        self.propellant_kg = self._propellant_initial_kg
        self._out_of_fuel = False

        # [SAFETY-REAL] 姿态/指向状态:当前指向模式 + 反作用轮归一化动量
        self._attitude_enabled = bool(ATTITUDE_CONFIG.get("enabled", False))
        self._pointing_mode = POINTING_SUN     # 初始对日充电
        self._momentum = 0.0
        self._slew_active = False
        self._desat_active = False

        # [credit-bucket 处理门 2026-06-08] leaky-bucket 流控状态（默认门关；eval-time ablation 开）
        _cg = SAFE_BUDGET_FALLBACK_CONFIG
        _avg_cap = float(_cg.get("avg_contact_capacity_mb", 800.0))
        self._proc_credit = float(_cg.get("initial_credit_factor", 1.5)) * _avg_cap
        self._cum_processed_mb = 0.0
        self._cum_downlinked_mb = 0.0
        self._credit_sum = 0.0
        self._credit_min = float("inf")
        self._credit_steps = 0

        r0 = self.orbit_dyn.R_e + self.altitude_m
        n0 = np.sqrt(self.orbit_dyn.mu / r0**3)
        self.orbit_sim.reset_phase(n0 * self.time_s)

        # ── Domain randomization：每 episode 重置一次长尺度太阳/几何条件 ──
        # 物理 RNG 与主 RNG 分离，确保 scene/task/emergency 抽样的 seed 复现性不受影响。
        # _randomization_scale ∈ [0, 1] 由训练课程注入：Exploration=0.2, Balancing=0.45,
        # Ramp=0.75, Optimization=1.0。所有范围按此因子线性收缩，evaluation 期使用 1.0
        # 暴露 agent 到 PDF 物理极值（rho×2, β 75°, 风暴 2.5×）。
        r_scale = float(np.clip(self._randomization_scale, 0.0, 1.0))
        # ── 空间天气显式驱动 (F10.7 + Ap) + shape/Cd 不确定度 ──
        # F10.7 每 episode 恒定 (6h 内变化 <1%)；中心 f107_nominal，宽度按课程因子缩窄。
        # quiet Ap 基线每 episode 抽样；风暴瞬态由 _advance_storm_event_state 抬升。
        # 这套机制取代旧 log-uniform rho_scale 抽象：解析模型经 f107_to_rho_scale 折算 rho_ref，
        # NRLMSISE-00 直接消费 SpaceWeatherState。F10.7 随机化沿用 enable_solar_activity_randomization 开关。
        f107_nominal = float(DRAG_CONFIG.get("f107_nominal", 150.0))
        f107_lo, f107_hi = DRAG_CONFIG.get("f107_range", (70.0, 250.0))
        if bool(DRAG_CONFIG.get("enable_solar_activity_randomization", True)) and r_scale > 0.0:
            half_lo = (f107_nominal - float(f107_lo)) * r_scale
            half_hi = (float(f107_hi) - f107_nominal) * r_scale
            f107_daily = float(self._physics_rng.uniform(
                f107_nominal - half_lo, f107_nominal + half_hi))
            jitter = float(DRAG_CONFIG.get("f107_81day_jitter_sfu", 15.0)) * r_scale
            f107_81avg = float(np.clip(
                f107_daily + self._physics_rng.uniform(-jitter, jitter),
                float(f107_lo), float(f107_hi)))
        else:
            # 关闭 F10.7 随机化 → daily 与 81avg 都钉死在标称 (完全复现既有基线)。
            f107_daily = f107_nominal
            f107_81avg = f107_nominal
        ap_quiet_lo, ap_quiet_hi = DRAG_CONFIG.get("ap_quiet_range", (3.0, 15.0))
        self._episode_quiet_ap = float(self._physics_rng.uniform(
            float(ap_quiet_lo), float(ap_quiet_hi)))
        doy_lo, doy_hi = DRAG_CONFIG.get("nrlmsise_epoch_doy_range", (1, 365))
        epoch_doy = float(self._physics_rng.integers(int(doy_lo), int(doy_hi) + 1))
        self._sw_state = SpaceWeatherState(
            f107_daily=f107_daily,
            f107_81avg=f107_81avg,
            ap=self._episode_quiet_ap,
            epoch_doy=epoch_doy,
            raan_deg=float(ORBITAL_CONFIG.get("raan_deg", 0.0)),
            inclination_deg=float(ORBITAL_CONFIG.get("inclination_deg", 51.6)),
        )
        self.orbit_dyn.atm.set_space_weather(self._sw_state)
        # 解析模型 (exponential/vallado) 用 rho_ref 作为太阳活跃度的传导通道；
        # F10.7=nominal → rho_scale=1.0 精确复现既有基线。NRLMSISE-00 忽略 rho_ref、直接用 F10.7。
        self.orbit_dyn.atm.rho_ref = self._base_rho_ref * f107_to_rho_scale(
            f107_daily, f107_nominal=f107_nominal)
        if hasattr(self.orbit_dyn.atm, "set_phase"):
            self.orbit_dyn.atm.set_phase(float(self.orbit_sim.phase))
        # satellite shape/Cd 不确定度：每 episode 静态抽 Cd 与迎风面积，按课程因子缩放。
        # scale=0 或 enable 开关关闭 → 标称确定性 (供 robustness 干净控制 drag)。
        # drag_strength_norm 的归一化基准固定在标称，使变化作为真实信号暴露给 agent。
        cd_nom = float(DRAG_CONFIG.get("Cd", 2.2))
        area_nom = float(DRAG_CONFIG.get("area_m2", 1.0))
        if bool(DRAG_CONFIG.get("enable_shape_cd_randomization", True)) and r_scale > 0.0:
            cd_lo, cd_hi = DRAG_CONFIG.get("cd_range", (2.0, 2.4))
            cd_half = max(cd_nom - float(cd_lo), float(cd_hi) - cd_nom) * r_scale
            sampled_cd = float(np.clip(
                self._physics_rng.uniform(cd_nom - cd_half, cd_nom + cd_half),
                float(cd_lo), float(cd_hi)))
            area_frac = float(DRAG_CONFIG.get("area_uncertainty_fraction", 0.18)) * r_scale
            sampled_area = float(area_nom * (
                1.0 + self._physics_rng.uniform(-area_frac, area_frac)))
        else:
            sampled_cd = cd_nom
            sampled_area = area_nom
        self.orbit_dyn.set_drag_params(Cd=sampled_cd, area_m2=sampled_area)
        # β 角 eclipse 随机化 (PDF Section 5：太阳赤纬 + RAAN 决定季节性阴影时长)
        # β = arcsin(sin i · sin(Ω-Ω_⊙) + cos i · sin δ_⊙)；51.6° 倾角下 |β| 可达 75°。
        # 这里直接抽 β 的幅值，避开显式跟踪 RAAN/Ω_⊙/δ_⊙ 等长期变量；分布等价于
        # 联合采样 epoch (季节) 和 ascending-node-LST (相位)。
        if bool(ORBITAL_CONFIG.get("enable_eclipse_beta_randomization", True)) and r_scale > 0.0:
            beta_max_deg = float(ORBITAL_CONFIG.get("eclipse_beta_max_deg", 75.0)) * r_scale
            beta_rad = float(self._physics_rng.uniform(0.0, np.deg2rad(beta_max_deg)))
            ecl_frac = eclipse_fraction_from_beta(
                beta_rad, self.altitude_m, self.orbit_dyn.R_e)
            self.orbit_sim.set_eclipse_fraction(ecl_frac)
        else:
            # 关闭或 scale=0 时退回 config 默认阴影占比
            default_frac = float(ORBITAL_CONFIG["eclipse_duration_min"]) / float(
                ORBITAL_CONFIG["orbital_period_min"])
            self.orbit_sim.set_eclipse_fraction(default_frac)

        self.battery.reset()
        self.energy_queue.reset(self.battery.energy_margin_wh)
        self.orbit_queue.reset(self.altitude_m)
        self.data_queue.reset()
        self.comm_queue.reset()
        self.task_tracker.reset()
        self.step_count = 0
        self.prev_action = np.zeros(self.action_dim)
        self.episode_reward = 0.0
        self._last_total_power_w = ENERGY_CONFIG.get("power_baseline_w", 15.0)
        self._last_available_power_w = ENERGY_CONFIG.get("power_total_max_w", 120.0)
        self._last_delivery_info = {}
        self._last_cpu_backpressure_ratio = 0.0
        self._emergency_event_remaining_steps = 0
        self._emergency_event_cooldown_steps = 0
        self._last_emergency_event_active = False
        self._last_emergency_event_triggered = False
        self._storm_active_steps_total = 0
        self._storm_active_steps_remaining = 0
        self._storm_cooldown_remaining = 0
        self._storm_peak_multiplier = 1.0
        self._storm_peak_ap = self._episode_quiet_ap
        self._last_ap = self._episode_quiet_ap
        self._last_storm_multiplier = 1.0
        self.orbit_dyn.atm.storm_multiplier = 1.0
        self._last_future_contact_capacity_norm = 0.0
        if bool(TASK_CONFIG.get("randomize_scene_phase_offset", True)):
            max_offset = float(TASK_CONFIG.get("scene_phase_offset_max_fraction", 1.0))
            self._scene_phase_offset_fraction = float(
                self.rng.uniform(0.0, max(0.0, max_offset)) % 1.0
            )
        else:
            self._scene_phase_offset_fraction = 0.0
        # 每 episode 打乱 phase_scene_rules 块顺序，消除 "见 X 后必是 Y" 的 shortcut 学习。
        # 时长 + 场景集合不变，只换排列。
        self._phase_scene_rules = self._build_episode_phase_scene_rules()
        self._last_scene_context = self._scene_context_for_phase()
        self._contact_override = None
        self._comm_window_age_steps = 0
        self._comm_pass_remaining_mb = self._comm_pass_capacity_mb
        self.thermal_temperature_c = float(
            THERMAL_CONFIG.get("initial_temp_c", 20.0))
        # 初始 contact 使用 reset 后的当前 time_s。
        self._contact = self._apply_acquisition_latency(self._get_contact_info())
        
        self._last_future_contact_capacity_mb = 0.0
        self._last_future_contact_capacity_mb_time = -1e30
        self._last_future_contact_capacity_norm = float(self._future_contact_capacity_norm())
        self._processed_since_contact_mb = 0.0
        self._delivered_since_contact_mb = 0.0
        self._raw_equivalent_processed_since_contact_mb = 0.0
        self._raw_equivalent_delivered_since_contact_mb = 0.0
        self._prev_in_window_for_budget = False
        # potential-based shaping：episode 开始时初始化为 0 避免虚假 shaping
        self._prev_potential = 0.0
        # ── 缓存熔断：防止跨 Episode 走私陈旧状态 ──
        # 先清 → _get_observation 可能会重算并写入 step=0 的缓存 → 最终再清 bins
        self._cached_step = -1
        self._cached_deadline_contact_stats = {}
        self._cached_deliverability_features = {}
        self._cached_bins_step = -1
        self._cached_bins_value = []
        self._step_horizon_cache = {}

        obs = self._get_observation()
        # _get_observation 可能已重算 bins 并写入 step=0 缓存，清掉它让首步强制刷新
        self._cached_bins_step = -1
        self._cached_bins_value = []
        return obs

    def _apply_mission_pointing_fallback(
        self,
        action: np.ndarray,
        *,
        orbit_guard: bool,
        energy_guard: bool,
    ) -> tuple[np.ndarray, dict]:
        """在物理安全时兜底任务姿态，避免策略把成像/下传链路长期关死。"""
        adjusted = np.asarray(action, dtype=np.float32).copy()
        current_mode = pointing_mode_from_unit(float(adjusted[IDX_POINTING]) if adjusted.size > IDX_POINTING else 0.5)
        meta = {
            "mission_pointing_fallback_enabled": bool(
                HARD_RULES_CONFIG.get("enable_mission_pointing_fallback", False)),
            "mission_pointing_fallback_applied": False,
            "mission_pointing_fallback_reason": "disabled",
            "mission_pointing_mode_before": int(current_mode),
            "mission_pointing_mode_after": int(current_mode),
        }
        if (
            not meta["mission_pointing_fallback_enabled"]
            or not getattr(self, "_attitude_enabled", False)
            or adjusted.size <= 8
        ):
            return adjusted, meta

        reserve_soc = max(0.0, float(ENERGY_CONFIG.get("battery_operational_reserve_soc", 0.0)))
        thermal_margin = float(self._thermal_margin_norm())
        min_thermal_margin = float(
            HARD_RULES_CONFIG.get("mission_pointing_min_thermal_margin", 0.20))
        # SOC 门控：只有电量有富余时才允许 fallback 强制 DOWNLINK/IMAGE（占用指向、放弃充电）；
        # 否则视为非 task_operational → 交回策略（通常对日充电），保住推进所需电量预算。
        min_soc_for_task = max(
            float(self.battery.soc_min) + reserve_soc,
            float(HARD_RULES_CONFIG.get("mission_pointing_min_soc", 0.0)),
        )
        task_operational = bool(
            (not orbit_guard)
            and (not energy_guard)
            and float(self.altitude_m) > float(self._h_warning)
            and float(self.battery.soc) > min_soc_for_task
            and thermal_margin > min_thermal_margin
        )
        if not task_operational:
            meta["mission_pointing_fallback_reason"] = "safety_guard"
            if int(current_mode) != int(POINTING_SUN):
                adjusted[IDX_POINTING] = float(pointing_unit_for_mode(POINTING_SUN))
                meta["mission_pointing_fallback_applied"] = True
                meta["mission_pointing_fallback_reason"] = "safety_guard_sun"
                meta["mission_pointing_mode_after"] = int(POINTING_SUN)
            return adjusted, meta

        contact = getattr(self, "_contact_override", None) or self._contact or {}
        in_window = bool(contact.get("in_window", False))
        raw_queue_mb = max(0.0, float(getattr(self.data_queue, "length", 0.0)))
        processed_queue_mb = max(0.0, float(getattr(self.comm_queue, "value", 0.0)))
        raw_low_mb = max(0.0, float(
            HARD_RULES_CONFIG.get("mission_pointing_raw_low_mb", 1.0)))
        tx_min_queue_mb = max(0.0, float(
            HARD_RULES_CONFIG.get("in_window_floor_min_queue_mb", 5.0)))
        daylit = bool(self.orbit_sim.is_sunlit(self.time_s, self.altitude_m))

        raw_capacity_mb = max(1e-6, float(getattr(self.data_queue, "max_length", 0.0)))
        raw_util = raw_queue_mb / raw_capacity_mb
        raw_room_util = float(
            HARD_RULES_CONFIG.get("mission_pointing_raw_room_util", 0.8))

        desired_mode = None
        reason = "no_task_need"
        if in_window and (processed_queue_mb > tx_min_queue_mb or raw_queue_mb > raw_low_mb):
            desired_mode = POINTING_DOWNLINK
            reason = "contact_backlog"
        elif daylit and raw_util < raw_room_util:
            # 昼侧且 raw 队列仍有空间 → 持续成像填充管路。
            # (旧逻辑 raw_queue_mb<=raw_low_mb 只在饿死时成像,攒过 1MB 就停手,
            #  导致 daylit 大部分步落入 no_task_need=89% → actor 默认对日,管路填不满。)
            desired_mode = POINTING_IMAGE
            reason = "daylit_image"

        if desired_mode is None:
            meta["mission_pointing_fallback_reason"] = reason
            return adjusted, meta

        meta["mission_pointing_fallback_reason"] = reason
        meta["mission_pointing_mode_after"] = int(desired_mode)
        if int(desired_mode) != int(current_mode):
            adjusted[IDX_POINTING] = float(pointing_unit_for_mode(desired_mode))
            meta["mission_pointing_fallback_applied"] = True
        return adjusted, meta

    def _project_soc_through_eclipse(self, mode: int, action: np.ndarray) -> float:
        """前瞻式能量预算：估计在 `mode` 指向下穿越剩余日照(充电)+下一次阴影(放电)后的 SOC。

        task 指向(IMAGE/DOWNLINK)会因离日余弦损失降低太阳输入，并叠加 imager/TX 负载，
        从而压低投影 SOC；与 reserve 比较即可判断"现在花这点电，下次阴影会不会破线"。
        物理与 BatteryModel.step 同口径(充 ×eta_charge，放 ÷eta_discharge)。
        """
        batt = self.battery
        cap_wh = max(float(batt.capacity_wh), 1e-6)
        energy_wh = float(batt.soc) * cap_wh
        soc_cap_wh = float(batt.soc_max) * cap_wh
        eta_c = float(getattr(batt, "eta_charge", 0.95))
        eta_d = float(getattr(batt, "eta_discharge", 0.95))

        acfg = ATTITUDE_CONFIG
        offsun = float(acfg.get("solar_offsun_scale", 0.30))
        solar_scale = 1.0 if int(mode) == int(POINTING_SUN) else offsun
        p_baseline = float(ENERGY_CONFIG.get("power_baseline_w", 15.0))
        p_prop = float(np.clip(action[0], 0.0, 1.0)) * float(ENERGY_CONFIG.get("power_propulsion_max_w", 720.0)) \
            if action.size > 0 else 0.0
        p_cpu = float(np.clip(action[1], 0.0, 1.0)) * float(ENERGY_CONFIG.get("power_cpu_max_w", 25.0)) \
            if action.size > 1 else 0.0
        p_img = float(acfg.get("imager_power_w", 30.0)) if int(mode) == int(POINTING_IMAGE) else 0.0
        p_tx = (float(np.clip(action[2], 0.0, 1.0)) * float(ENERGY_CONFIG.get("power_tx_max_w", 35.0))
                if (action.size > 2 and int(mode) == int(POINTING_DOWNLINK)) else 0.0)
        p_load_task = p_baseline + p_prop + p_cpu + p_img + p_tx
        # 阴影期把 imager/TX 关掉，只剩 bus + 维轨推进（保守估计阴影放电）。
        p_load_eclipse = p_baseline + p_prop

        daylit = bool(self.orbit_sim.is_sunlit(self.time_s, self.altitude_m))
        try:
            t_to_eclipse = float(self.orbit_sim.time_to_next_eclipse(self.time_s, self.altitude_m))
            t_to_sunlit = float(self.orbit_sim.time_to_next_sunlit(self.time_s, self.altitude_m))
        except Exception:
            t_to_eclipse, t_to_sunlit = 0.0, 0.0
        eclipse_dur_s = float(ORBITAL_CONFIG.get("eclipse_duration_min", 35.0)) * 60.0

        if daylit:
            # Phase 1: 剩余日照下按 task 指向充/放电
            sun_remain_s = max(0.0, t_to_eclipse)
            p_solar = float(self.solar.output_power(self.orbit_sim.sunlit_fraction())) * solar_scale
            net = p_solar - p_load_task
            dt_h = sun_remain_s / 3600.0
            energy_wh += net * dt_h * (eta_c if net >= 0 else 1.0 / eta_d)
            energy_wh = float(np.clip(energy_wh, 0.0, soc_cap_wh))
            # Phase 2: 整段阴影放电
            energy_wh -= p_load_eclipse * (eclipse_dur_s / 3600.0) / eta_d
        else:
            # 当前已在阴影：先放电到日照，剩余阴影时长 = t_to_sunlit
            ecl_remain_s = max(0.0, t_to_sunlit)
            energy_wh -= p_load_eclipse * (ecl_remain_s / 3600.0) / eta_d
        return float(np.clip(energy_wh, 0.0, soc_cap_wh)) / cap_wh

    def _apply_safe_budget_fallback(
        self,
        action: np.ndarray,
        *,
        orbit_guard: bool,
        energy_guard: bool,
    ) -> tuple[np.ndarray, dict]:
        """前瞻式安全预算 + 数据压力门控的任务姿态兜底（替代激进 mission_pointing_fallback）。

        安全优先级（硬序）：hard safety > charge/recovery > downlink-in-contact > image/process > idle。
        诊断（diagnose_fallback_safety.py）显示旧 fallback 只看当前 SOC，把 SOC 钉在门控线上
        反复 SUN↔IMAGE churn，一遇扰动即破线，且强制成像把 proc/dl 推到 4.16。本兜底用
        投影 SOC（穿越下次阴影）+ data_pressure 抑制成像/处理，把安全裕度和 backlog 都管住。

        同时回填 mission_pointing_* 键，使 eval 聚合与训练 AP-BC（mission_pointing_bc_weight）
        仍以"安全预算执行动作"为模仿目标，让策略共适应。
        """
        cfg = SAFE_BUDGET_FALLBACK_CONFIG
        adjusted = np.asarray(action, dtype=np.float32).copy()
        cur = pointing_mode_from_unit(float(adjusted[IDX_POINTING]) if adjusted.size > IDX_POINTING else 0.5)
        meta = {
            "safe_budget_fallback_enabled": bool(cfg.get("enabled", False)),
            "safe_budget_fallback_applied": False,
            "safe_budget_reason": "disabled",
            "safe_budget_data_pressure": 0.0,
            "safe_budget_projected_soc": float(self.battery.soc),
            "safe_budget_reserve": float(cfg.get("reserve_soc", 0.45)),
            "safe_budget_alpha_cpu_capped": False,
            "safe_budget_alpha_tx_capped": False,
            # 兼容旧 mission_pointing_* 诊断/BC 键
            "mission_pointing_fallback_enabled": bool(cfg.get("enabled", False)),
            "mission_pointing_fallback_applied": False,
            "mission_pointing_fallback_reason": "disabled",
            "mission_pointing_mode_before": int(cur),
            "mission_pointing_mode_after": int(cur),
        }
        if (not cfg.get("enabled", False)
                or not getattr(self, "_attitude_enabled", False)
                or adjusted.size <= 8):
            return adjusted, meta

        soc = float(self.battery.soc)
        hard_min = float(cfg.get("hard_min_soc", 0.50))
        soft_min = float(cfg.get("soft_min_soc", 0.65))
        reserve = max(
            float(cfg.get("reserve_soc", 0.45)),
            float(self.battery.soc_min) + float(ENERGY_CONFIG.get("battery_operational_reserve_soc", 0.0)),
        )
        thermal_margin = float(self._thermal_margin_norm())
        min_thermal = float(cfg.get("min_thermal_margin",
                                    HARD_RULES_CONFIG.get("mission_pointing_min_thermal_margin", 0.20)))
        daylit = bool(self.orbit_sim.is_sunlit(self.time_s, self.altitude_m))

        # ── 充电机会前瞻：未来 lookahead 内日照时间太短 → 抬高 reserve ──
        lookahead_s = float(cfg.get("charge_lookahead_s", 1200.0))
        try:
            t_to_sunlit = float(self.orbit_sim.time_to_next_sunlit(self.time_s, self.altitude_m))
        except Exception:
            t_to_sunlit = 0.0
        sunlit_ahead_s = lookahead_s if daylit else max(0.0, lookahead_s - t_to_sunlit)
        if sunlit_ahead_s < float(cfg.get("eclipse_min_sunlit_s", 300.0)):
            reserve += float(cfg.get("eclipse_reserve_bonus", 0.10))
        meta["safe_budget_reserve"] = float(reserve)

        # ── 数据压力：下传压力用 product MB，存储压力单独记录 ──
        raw_mb = max(0.0, float(getattr(self.data_queue, "length", 0.0)))
        proc_mb = max(0.0, float(getattr(self.comm_queue, "value", 0.0)))
        try:
            compression_ratio = float(self.task_tracker._raw_to_processed_ratio())
        except Exception:
            compression_ratio = float(np.clip(
                TASK_CONFIG.get("RAW_TO_PROCESSED_RATIO", 1.0), 1e-6, 1.0))
        storage_onboard_mb = raw_mb + proc_mb
        future_downlink_product_mb = raw_mb * compression_ratio + proc_mb
        future_cap_mb = max(float(cfg.get("data_pressure_eps_mb", 1.0)),
                            float(self._future_contact_capacity_mb()))
        data_pressure = future_downlink_product_mb / future_cap_mb
        meta["safe_budget_data_pressure"] = float(data_pressure)
        meta["safe_budget_storage_pressure"] = float(storage_onboard_mb / future_cap_mb)
        meta["safe_budget_future_downlink_product_mb"] = float(future_downlink_product_mb)
        meta["safe_budget_storage_onboard_mb"] = float(storage_onboard_mb)
        dp_soft = float(cfg.get("data_pressure_soft", 1.5))
        dp_hard = float(cfg.get("data_pressure_hard", 2.0))

        contact = getattr(self, "_contact_override", None) or self._contact or {}
        in_window = bool(contact.get("in_window", False))
        tx_min_mb = float(HARD_RULES_CONFIG.get("in_window_floor_min_queue_mb", 5.0))
        raw_low_mb = float(HARD_RULES_CONFIG.get("mission_pointing_raw_low_mb", 1.0))
        raw_cap = max(1e-6, float(getattr(self.data_queue, "max_length", 0.0)))
        raw_util = raw_mb / raw_cap
        raw_room = float(HARD_RULES_CONFIG.get("mission_pointing_raw_room_util", 0.8))

        # ── 决策：硬序优先级 ──
        hard_unsafe = bool(
            orbit_guard or energy_guard
            or float(self.altitude_m) <= float(self._h_warning)
            or thermal_margin <= min_thermal
            or soc < hard_min
        )
        desired = None
        reason = "no_task_need"
        cap_cpu = False
        cap_tx = False

        if hard_unsafe:
            desired, reason = POINTING_SUN, "hard_safety_charge"
        elif in_window and (proc_mb > tx_min_mb or raw_mb > raw_low_mb):
            # 通信窗口内有 backlog：优先下传，但要过能量预算
            proj_dl = self._project_soc_through_eclipse(POINTING_DOWNLINK, adjusted)
            meta["safe_budget_projected_soc"] = float(proj_dl)
            if soc >= soft_min and proj_dl >= reserve:
                desired, reason = POINTING_DOWNLINK, "contact_backlog"
            elif soc >= hard_min and soc >= soft_min - float(cfg.get("low_power_tx_soc_margin", 0.05)):
                # SOC 裕度不足但未触硬线：只允许低功率/短时下传
                desired, reason = POINTING_DOWNLINK, "contact_lowpower"
                cap_tx = True
            else:
                desired, reason = POINTING_SUN, "guard_charge_in_window"
        elif daylit and raw_util < raw_room:
            # 候选成像：先过 data_pressure，再过能量预算
            if data_pressure >= dp_hard:
                desired, reason = POINTING_SUN, "data_pressure_hard_no_image"
                cap_cpu = True
            elif data_pressure >= dp_soft:
                desired, reason = POINTING_SUN, "data_pressure_soft_charge"
            elif soc >= soft_min:
                proj_img = self._project_soc_through_eclipse(POINTING_IMAGE, adjusted)
                meta["safe_budget_projected_soc"] = float(proj_img)
                if proj_img >= reserve:
                    desired, reason = POINTING_IMAGE, "daylit_image"
                else:
                    desired, reason = POINTING_SUN, "energy_budget_charge"
            else:
                desired, reason = POINTING_SUN, "below_soft_soc_charge"
        elif soc < soft_min:
            # 无任务需求且 SOC 偏低：主动充电建立缓冲（避免钉在门控线）
            desired, reason = POINTING_SUN, "buffer_charge"

        # ── alpha 门控（data_pressure 分级抑制处理；低功率下传）──
        # 分级 cpu cap：dp≥hard → process_cap_alpha_under_pressure(默认0=禁处理)；
        # dp≥cpu_throttle_pressure(默认=hard→无 soft 档) → process_cap_alpha_soft。
        # 默认两键 no-op，保持锁定交付基线；ablation 在 eval-time 覆盖以定向降 proc/dl。
        queue_projection_policy = str(DRL_CONFIG.get(
            "queue_projection_policy", "diagnostic_only"))
        deployment_queue_projection_enabled = (
            bool(DRL_CONFIG.get("enable_deployment_queue_projection", False))
            and queue_projection_policy in {
                "deployment_hard_boundary",
                "hard_boundary",
                "safety_algorithms_only",
            }
        )
        dedicated_cpu_backpressure_enabled = bool(
            TASK_CONFIG.get(
                "enable_future_contact_cpu_gate",
                TASK_CONFIG.get("enable_cpu_throttle", False),
            )
            or TASK_CONFIG.get("cpu_action_is_admissible_budget", False)
            or deployment_queue_projection_enabled
        )
        meta["safe_budget_cpu_pressure_deferred"] = bool(dedicated_cpu_backpressure_enabled)
        if adjusted.size > 1 and not dedicated_cpu_backpressure_enabled:
            cpu_throttle_p = float(cfg.get("cpu_throttle_pressure", dp_hard))
            cpu_cap = 1.0
            if data_pressure >= dp_hard:
                cpu_cap = float(cfg.get("process_cap_alpha_under_pressure", 0.0))
            elif data_pressure >= cpu_throttle_p:
                cpu_cap = float(cfg.get("process_cap_alpha_soft", 1.0))
            if float(adjusted[1]) > cpu_cap + 1e-9:
                adjusted[1] = cpu_cap
                meta["safe_budget_alpha_cpu_capped"] = True
                cap_cpu = True
        if cap_tx and adjusted.size > 2:
            tx_cap = float(cfg.get("low_power_tx_alpha_cap", 0.35))
            if float(adjusted[2]) > tx_cap:
                adjusted[2] = tx_cap
                meta["safe_budget_alpha_tx_capped"] = True

        # ── credit-bucket 处理门（leaky-bucket 流控，默认关）──
        # credit<=0 或 running proc/dl≥hard（过 warmup）→ 禁处理/禁成像（窗口内改下传，否则充电）；
        # credit<=soft 或 running proc/dl≥soft → 节流 alpha_cpu。credit 在 step 末按下传/处理更新。
        meta["credit_gate_triggered"] = False
        meta["credit_image_blocked"] = False
        meta["credit_process_blocked"] = False
        meta["credit_downlink_prioritized"] = False
        if cfg.get("enable_credit_gate", False) and adjusted.size > 1:
            avg_cap = float(cfg.get("avg_contact_capacity_mb", 800.0))
            credit = float(getattr(self, "_proc_credit",
                                   float(cfg.get("initial_credit_factor", 1.5)) * avg_cap))
            soft_credit = float(cfg.get("soft_credit_factor", 0.5)) * avg_cap
            cum_dl = float(getattr(self, "_cum_downlinked_mb", 0.0))
            cum_proc = float(getattr(self, "_cum_processed_mb", 0.0))
            run_ratio = cum_proc / max(cum_dl, 1e-6)
            ratio_active = cum_dl >= float(cfg.get("credit_ratio_warmup_mb", avg_cap))
            hard_block = (credit <= 0.0) or (ratio_active and run_ratio >= float(cfg.get("hard_ratio_limit", 3.0)))
            soft_block = (credit <= soft_credit) or (ratio_active and run_ratio >= float(cfg.get("soft_ratio_limit", 2.5)))
            meta["credit_value"] = float(credit)
            meta["credit_running_proc_dl"] = float(run_ratio)
            if hard_block:
                if float(adjusted[1]) > 1e-9:
                    adjusted[1] = 0.0
                    meta["safe_budget_alpha_cpu_capped"] = True
                meta["credit_process_blocked"] = True
                meta["credit_gate_triggered"] = True
                # 禁成像：窗口内有 backlog 改下传，否则充电
                if desired in (None, POINTING_IMAGE) and cur == POINTING_IMAGE or desired == POINTING_IMAGE:
                    if in_window and (proc_mb > tx_min_mb or raw_mb > raw_low_mb):
                        desired, reason = POINTING_DOWNLINK, "credit_block_downlink"
                        meta["credit_downlink_prioritized"] = True
                    else:
                        desired, reason = POINTING_SUN, "credit_block_charge"
                    meta["credit_image_blocked"] = True
            elif soft_block:
                cap = float(cfg.get("credit_throttle_alpha_cpu", 0.3))
                if float(adjusted[1]) > cap:
                    adjusted[1] = cap
                    meta["safe_budget_alpha_cpu_capped"] = True
                meta["credit_gate_triggered"] = True

        meta["safe_budget_reason"] = reason
        meta["mission_pointing_fallback_reason"] = reason
        if desired is not None:
            meta["mission_pointing_mode_after"] = int(desired)
            if int(desired) != int(cur):
                adjusted[IDX_POINTING] = float(pointing_unit_for_mode(desired))
                meta["safe_budget_fallback_applied"] = True
                meta["mission_pointing_fallback_applied"] = True
        # alpha 改写也算 fallback 介入（供 BC/诊断口径）
        if meta["safe_budget_alpha_cpu_capped"] or meta["safe_budget_alpha_tx_capped"]:
            meta["safe_budget_fallback_applied"] = True
            meta["mission_pointing_fallback_applied"] = True
        return adjusted, meta

    def _task_queue_sync_flags(self, tol: float = 1e-6) -> tuple[bool, bool]:
        """记录本步开始时物理队列和 batch 明细是否已经同步。"""
        raw_batches_mb = float(getattr(self.task_tracker, "raw_mb", 0.0))
        processed_batches_mb = float(getattr(self.task_tracker, "processed_mb", 0.0))
        raw_synced = abs(float(self.data_queue.length) - raw_batches_mb) <= tol
        processed_synced = abs(float(self.comm_queue.value) - processed_batches_mb) <= tol
        return bool(raw_synced), bool(processed_synced)

    def _assert_task_queue_sync(
        self,
        stage: str,
        *,
        check_raw: bool = True,
        check_processed: bool = True,
        tol: float = 1e-5,
    ) -> None:
        """正常数据流中确保物理队列 MB 与 batch 明细 MB 一致。"""
        if check_raw:
            raw_batches_mb = float(self.task_tracker.raw_mb)
            assert abs(float(self.data_queue.length) - raw_batches_mb) <= tol, (
                f"{stage}: data_queue.length={self.data_queue.length:.9f} "
                f"!= sum(raw_batches.mb)={raw_batches_mb:.9f}"
            )
        if check_processed:
            processed_batches_mb = float(self.task_tracker.processed_mb)
            assert abs(float(self.comm_queue.value) - processed_batches_mb) <= tol, (
                f"{stage}: comm_queue.value={self.comm_queue.value:.9f} "
                f"!= sum(processed_batches.mb)={processed_batches_mb:.9f}"
            )

    def step(self, action: np.ndarray, enforce_prop_smoothing: bool = True) -> tuple:
        """执行一个 10s 调度步并返回 Gym 风格的 observation/reward/done/info。

        主数据链路是 raw_queue -> processed_queue -> communication window -> ground。
        本函数负责把 actor 动作转换为物理功率、队列流转和本步任务价值交付；
        reward 只取本步交付价值，安全和积压压力通过 CMDP cost 写入 info。
        """
        # 环境是最后一道执行层：即使上游网络/脚本给出 NaN/Inf，也不能让非有限动作污染物理状态。
        self._step_horizon_cache = {}  # 步内 horizon_s 字典缓存（拦截 process_by_priority 高频重复调用）
        raw_queue_was_synced, processed_queue_was_synced = self._task_queue_sync_flags()
        sanitized_action = self.action_sanitizer(action, dtype=np.float32)
        action = sanitized_action.action
        input_action_finite = bool(sanitized_action.meta["raw_action_finite"])
        input_action_in_bounds = bool(sanitized_action.meta["input_action_in_bounds"])
        action, propulsion_controller_meta = self._apply_analytic_propulsion_controller(action)

        # 推进通道带平滑约束；LS-PSF 调度器会在安全层前处理这一约束，
        # 所以它传入的最终安全动作不再被环境二次平滑。
        # ── 推进器平滑 + 安全覆盖 ────────────────────────────
        prop_can_update = (self.step_count % self.N_PROP_SMOOTH == 0)
        smooth_action = action.copy()
        prop_delta = abs(action[0] - self.prev_action[0])
        # 大幅改变或贴近物理安全底线时，允许临时突破推进器平滑。
        # 否则 PSF/Lyapunov 给出的救急推进或降功率动作可能被平滑层吞掉。
        orbit_guard = (
            self.altitude_m <= self._h_min + 10e3
            and action[0] > self.prev_action[0] + 1e-8
        )
        energy_guard = (
            self.battery.soc <= self.battery.soc_min + 0.02
            and action[0] < self.prev_action[0] - 1e-8
        )
        analytic_propulsion_enabled = bool(
            propulsion_controller_meta.get("analytic_propulsion_controller_enabled", False))
        analytic_propulsion_requires_update = bool(
            analytic_propulsion_enabled
            and abs(float(action[0]) - float(self.prev_action[0])) > 1e-9
        )
        safety_override = bool(
            enforce_prop_smoothing
            and (analytic_propulsion_requires_update or prop_delta >= 0.4 or orbit_guard or energy_guard)
        )
        if analytic_propulsion_requires_update:
            prop_safety_override_reason = "analytic_propulsion"
        elif not enforce_prop_smoothing:
            prop_safety_override_reason = "scheduler_final_action"
        elif orbit_guard:
            prop_safety_override_reason = "orbit_guard"
        elif energy_guard:
            prop_safety_override_reason = "energy_guard"
        elif prop_delta >= 0.4:
            prop_safety_override_reason = "large_delta"
        else:
            prop_safety_override_reason = "none"

        if enforce_prop_smoothing and not prop_can_update and not safety_override:
            smooth_action[0] = self.prev_action[0]
        action = smooth_action

        # SAFE_BUDGET_FALLBACK 优先（前瞻能量预算 + data_pressure 门控）；
        # 未启用时回退到旧的 mission_pointing_fallback（单点 SOC 门控）。
        if bool(SAFE_BUDGET_FALLBACK_CONFIG.get("enabled", False)):
            action, mission_pointing_meta = self._apply_safe_budget_fallback(
                action,
                orbit_guard=orbit_guard,
                energy_guard=energy_guard,
            )
        else:
            action, mission_pointing_meta = self._apply_mission_pointing_fallback(
                action,
                orbit_guard=orbit_guard,
                energy_guard=energy_guard,
            )

        # ── [SAFETY-REAL] 姿态/指向:成像/下传/充电互斥 + 机动耗时耗能 + 动量去饱和 ──
        if getattr(self, "_attitude_enabled", False):
            acfg = ATTITUDE_CONFIG
            req_mode = pointing_mode_from_unit(float(action[IDX_POINTING]) if action.size > IDX_POINTING else 0.5)
            self._slew_active = (req_mode != self._pointing_mode)
            self._momentum += float(acfg.get("momentum_disturbance_per_step", 0.0))
            if self._slew_active:
                self._momentum += float(acfg.get("momentum_per_slew", 0.0))
            bleed = (acfg.get("momentum_bleed_sun", 0.05) if req_mode == POINTING_SUN
                     else acfg.get("momentum_bleed_default", 0.01))
            self._momentum = max(0.0, self._momentum - float(bleed))
            self._desat_active = self._momentum >= float(acfg.get("momentum_max", 1.0))
            if self._desat_active:
                self._momentum = max(0.0, self._momentum - 3.0 * float(acfg.get("momentum_bleed_sun", 0.05)))
                productivity_scale = 0.0          # 强制去饱和:整步无产出
            elif self._slew_active:
                productivity_scale = max(0.0, 1.0 - float(acfg.get("slew_lost_fraction", 0.3)))
            else:
                productivity_scale = 1.0
            self._attitude_productivity = float(productivity_scale)
            self._can_image = (req_mode == POINTING_IMAGE) and productivity_scale > 0.0
            self._can_downlink = (req_mode == POINTING_DOWNLINK) and productivity_scale > 0.0
            self._solar_scale = 1.0 if req_mode == POINTING_SUN else float(acfg.get("solar_offsun_scale", 0.3))
            self._imager_power_w = float(acfg.get("imager_power_w", 0.0)) if self._can_image else 0.0
            self._slew_energy_wh = float(acfg.get("slew_energy_wh", 0.0)) if self._slew_active else 0.0
            self._pointing_mode = req_mode
        else:
            self._slew_active = False; self._desat_active = False
            self._attitude_productivity = 1.0
            self._can_image = True; self._can_downlink = True
            self._solar_scale = 1.0; self._imager_power_w = 0.0; self._slew_energy_wh = 0.0

        # 1: 计算当前太阳能输入，并估算本时隙可用总功率。
        sunlit_frac = self.orbit_sim.sunlit_fraction()
        P_solar = self.solar.output_power(sunlit_frac) * float(getattr(self, "_solar_scale", 1.0))
        available_power_w = self._compute_available_power(P_solar)
        self._last_available_power_w = float(available_power_w)

        # ── Phase 1 硬规则 E：窗口期 alpha_tx 硬 floor（必须在功率分配之前执行）──
        # 窗口期 processed_queue 有货时，强制 alpha_tx ≥ floor，避免 agent 主动留链路不下传。
        # 关键时序：必须赶在 _enforce_available_power → compute_total_load → 电池放电 → rf_capacity
        # 之前抬高 alpha_tx，否则发射机功率、RF 物理限值、电池放电全部按低 alpha_tx 结算，下方
        # step 9 的 min(alpha_tx*link, rf_capacity) 会把硬规则拉起的下传量重新卡死、硬规则形同
        # 虚设（并产生 alpha_tx=0.95 但 P_tx_w/吞吐极低的物理矛盾）。in_window 取 self._contact
        # （步初窗口，与下方 _enforce_available_power / cpu_gate 同源），comm_queue.value 取步初
        # 值——功率决策只能依赖步初已知信息，不能用本步处理后的未来量。
        try:
            from config import HARD_RULES_CONFIG as _HR_CFG  # noqa: WPS433
        except Exception:
            _HR_CFG = {}
        raw_backlog_for_window_feed = (
            bool(TASK_CONFIG.get("enable_in_window_cpu_feed_floor", False))
            and float(getattr(self.data_queue, "length", 0.0))
            > float(TASK_CONFIG.get("in_window_cpu_feed_min_raw_mb", 1.0))
        )
        processed_backlog_for_tx_floor = (
            float(self.comm_queue.value) > float(_HR_CFG.get("in_window_floor_min_queue_mb", 5.0))
        )
        tx_floor_meta = {
            "in_window_tx_floor_applied": False,
            "in_window_tx_floor_alpha_before": float(np.asarray(action, dtype=np.float32)[2]),
            "in_window_tx_floor_alpha_after": float(np.asarray(action, dtype=np.float32)[2]),
        }
        if (
            bool(_HR_CFG.get("enable_in_window_tx_floor", False))
            and bool((self._contact or {}).get("in_window", False))
            and (processed_backlog_for_tx_floor or raw_backlog_for_window_feed)
            and float(action[2]) < float(_HR_CFG.get("in_window_alpha_tx_floor", 0.95))
        ):
            action = np.asarray(action, dtype=np.float32).copy()
            tx_floor_meta["in_window_tx_floor_alpha_before"] = float(action[2])
            action[2] = float(_HR_CFG.get("in_window_alpha_tx_floor", 0.95))
            tx_floor_meta["in_window_tx_floor_alpha_after"] = float(action[2])
            tx_floor_meta["in_window_tx_floor_applied"] = True

        window_cpu_feed_meta = {
            "in_window_cpu_feed_floor_applied": False,
            "in_window_cpu_feed_alpha_before": float(np.asarray(action, dtype=np.float32)[1]),
            "in_window_cpu_feed_alpha_after": float(np.asarray(action, dtype=np.float32)[1]),
        }
        if (
            bool(TASK_CONFIG.get("enable_in_window_cpu_feed_floor", False))
            and bool((self._contact or {}).get("in_window", False))
            and float(getattr(self.data_queue, "length", 0.0))
            > float(TASK_CONFIG.get("in_window_cpu_feed_min_raw_mb", 1.0))
        ):
            cpu_floor = float(np.clip(
                TASK_CONFIG.get("in_window_cpu_feed_alpha_floor", 1.0),
                0.0,
                1.0,
            ))
            if float(action[1]) < cpu_floor:
                action = np.asarray(action, dtype=np.float32).copy()
                window_cpu_feed_meta["in_window_cpu_feed_alpha_before"] = float(action[1])
                action[1] = cpu_floor
                window_cpu_feed_meta["in_window_cpu_feed_alpha_after"] = float(action[1])
                window_cpu_feed_meta["in_window_cpu_feed_floor_applied"] = True

        contact_for_cpu_gate = self._contact or {}
        action, cpu_gate_meta = self._apply_future_contact_cpu_gate(
            action,
            in_window=bool(contact_for_cpu_gate.get("in_window", False)),
            time_to_next_window_s=float(contact_for_cpu_gate.get("time_to_next_window_s", 0.0)),
            dt_s=float(self.dt),
        )
        cpu_gate_meta.update(window_cpu_feed_meta)
        cpu_gate_meta.update(tx_floor_meta)

        # 1.5: 热安全是执行层的最终权威。in-window CPU/TX floor 与 CPU gate 都可能把
        # CPU/TX 抬高，这里在功率闭环前按当前热裕度再压一次，避免“接触窗口+高热”工况下
        # floor 覆盖调度器算好的热限流（执行动作必须 ≤ 热上限）。语义与
        # scheduler.integrated_scheduler 的 Pi_feas 热降额保持一致。
        thermal_cpu_cap = 1.0
        thermal_tx_cap = 1.0
        thermal_throttle_applied = False
        thermal_mod_l2 = 0.0
        if bool(THERMAL_CONFIG.get("enabled", True)):
            margin = float(np.clip(self._thermal_margin_norm(), -1.0, 1.0))
            if margin <= 0.0:
                thermal_cpu_cap = float(np.clip(THERMAL_CONFIG.get("critical_cpu_cap", 0.25), 0.0, 1.0))
                thermal_tx_cap = float(np.clip(THERMAL_CONFIG.get("critical_tx_cap", 0.0), 0.0, 1.0))
            elif margin < 0.35:
                scale = margin / 0.35
                min_scale = float(np.clip(THERMAL_CONFIG.get("warning_cpu_tx_min_scale", 0.35), 0.0, 1.0))
                cap = float(np.clip(min_scale + (1.0 - min_scale) * scale, min_scale, 1.0))
                thermal_cpu_cap = cap
                thermal_tx_cap = cap
            if thermal_cpu_cap < 1.0 or thermal_tx_cap < 1.0:
                action_arr = np.asarray(action, dtype=np.float32).copy()
                before_thermal = action_arr.copy()
                action_arr[1] = min(float(action_arr[1]), thermal_cpu_cap)
                action_arr[2] = min(float(action_arr[2]), thermal_tx_cap)
                thermal_mod_l2 = float(np.linalg.norm(action_arr - before_thermal))
                thermal_throttle_applied = bool(thermal_mod_l2 > 1e-9)
                action = action_arr

        # 2: 环境执行层做最终功率闭环。调度器已按可用功率裁剪过一次，
        # 但基线或手工脚本仍可能在环境层触发推进平滑/越界动作，所以这里必须用
        # 最终执行动作重新闭合 P_prop + P_cpu + P_tx + P_base <= P_available。
        action, power_execution_meta = self._enforce_available_power(action, available_power_w)
        power_execution_meta.update(cpu_gate_meta)

        # 3: 根据最终执行动作分配推进、计算和通信功率。
        # [SAFETY-REAL] 姿态载荷(成像功耗 + 机动瞬时耗能)计入 P_total,使其参与功率约束/安全/热/电池/日志
        # (此前仅在电池处叠加→安全判定与功耗日志低估真实负载)。
        contact_preview = self._get_contact_info_at(
            float(self.time_s + self.dt), float(self.altitude_m))
        preview_in_window = bool(contact_preview.get("in_window", False))
        preview_tx_capacity_mbps = float(contact_preview.get("max_capacity_mbps", 0.0))
        if preview_in_window:
            preview_tx_capacity_mbps = self._cap_link_capacity_by_pass_budget(
                preview_tx_capacity_mbps,
                float(self.dt),
            )
        queue_projection_policy = str(DRL_CONFIG.get(
            "queue_projection_policy", "diagnostic_only"))
        apply_queue_projection = (
            bool(DRL_CONFIG.get("enable_deployment_queue_projection", False))
            and queue_projection_policy in {
                "deployment_hard_boundary",
                "hard_boundary",
                "safety_algorithms_only",
            }
        )
        queue_boundary_result = self.actuator_filter.project_processed_queue_boundary(
            action,
            processed_queue_mb=float(self.comm_queue.value),
            processed_queue_max_mb=float(self.comm_queue.max_value),
            in_window=preview_in_window,
            tx_capacity_mbps=preview_tx_capacity_mbps,
            dt_s=float(self.dt),
            future_contact_capacity_mb=float(cpu_gate_meta.get(
                "cpu_gate_future_contact_capacity_mb", self._future_contact_capacity_mb())),
            future_capacity_margin=float(DRL_CONFIG.get("constraint_future_capacity_margin", 0.80)),
            future_ratio_start=float(TASK_CONFIG.get("cpu_gate_start_future_ratio", 0.55)),
            future_ratio_hard_stop=float(TASK_CONFIG.get("cpu_gate_hard_stop_future_ratio", 0.90)),
            apply_projection=apply_queue_projection,
            dtype=np.float32,
        )
        if apply_queue_projection:
            action = queue_boundary_result.action
        queue_boundary_meta = dict(queue_boundary_result.meta)
        queue_boundary_meta["queue_projection_policy"] = queue_projection_policy
        queue_boundary_meta["deployment_queue_projection_enabled"] = bool(apply_queue_projection)

        power_info = self.power_sys.compute_total_load(action)
        _attitude_load_w = float(getattr(self, "_imager_power_w", 0.0))
        if getattr(self, "_slew_energy_wh", 0.0) > 0.0:
            _attitude_load_w += float(self._slew_energy_wh) * 3600.0 / max(self.dt, 1e-9)
        power_info["P_attitude_w"] = _attitude_load_w
        power_info["P_total_w"] = float(power_info["P_total_w"]) + _attitude_load_w

        thermal_constraint_meta = {
            "thermal_throttle_applied": bool(thermal_throttle_applied),
            "thermal_clip_stage": "env_execution" if thermal_throttle_applied else "scheduler_or_none",
            "thermal_cpu_cap": float(thermal_cpu_cap),
            "thermal_tx_cap": float(thermal_tx_cap),
            "thermal_mod_l2": float(thermal_mod_l2),
        }
        power_execution_meta.update(queue_boundary_meta)
        power_execution_meta.update(thermal_constraint_meta)
        power_info["propulsion_deadband_applied"] = bool(
            power_execution_meta.get("propulsion_deadband_applied", False)
            or power_info.get("propulsion_deadband_applied", False)
        )
        self._last_cpu_backpressure_ratio = float(
            queue_boundary_meta.get("required_cpu_backpressure_ratio", 0.0)
        )
        # 把 CPU gate 的越界量存到 self 上，供 _compute_reward 取用做软惩罚。
        # soft mode 下 gate 不再改写动作，这个量直接进 reward；hard mode 下记 0。
        self._last_cpu_gate_violation_mb = float(
            cpu_gate_meta.get("cpu_gate_violation_mb", 0.0)
            if bool(cpu_gate_meta.get("cpu_gate_soft_mode", False)) else 0.0
        )
        self._last_total_power_w = float(power_info["P_total_w"])
        power_constraint_safe = bool(power_info["P_total_w"] <= available_power_w + 1e-6)

        # 4: 电池 SOC 更新。P_total_w 已含姿态载荷(见步骤3),此处直接使用,避免重复叠加。
        batt_info = self.battery.step(
            P_solar, power_info["P_total_w"], self.dt)
        thermal_info = self._update_thermal_state(
            power_info["P_total_w"],
            sunlit_frac,
            propulsion_power_w=power_info["P_propulsion_w"],
            cpu_power_w=power_info["P_cpu_w"],
            tx_power_w=power_info["P_tx_w"],
        )

        # 5: 大气状态更新 + 推进-阻力高度演化。
        # 5a: 推进地磁暴瞬态状态 (PDF Section 8.2)；设置 atm.storm_multiplier，
        #     对所有高度的密度线性放大，模拟 Starlink 2022 式短临密度激增。
        self._advance_storm_event_state()
        # 5b: 计算当前卫星-bulge 几何角 Ψ (PDF Section 5)，传入 drag 公式作日间隆起调制。
        diurnal_psi = self._diurnal_angle_rad()
        # NRLMSISE-00 需要当前轨道相位合成亚卫星点 (经纬度/地方时)；解析模型无 set_phase，跳过。
        if hasattr(self.orbit_dyn.atm, "set_phase"):
            self.orbit_dyn.atm.set_phase(float(self.orbit_sim.phase))
        # [SAFETY-REAL] 有限推进剂消耗 + 油尽断推:mdot = P*eff/(Isp*g0)^2;燃料=0 → 推力=0 → 阻力主导衰减。
        prop_power_eff = float(power_info["P_propulsion_w"])
        if getattr(self, "_propellant_enabled", False):
            g0 = 9.80665
            isp_s = max(1e-6, float(ENERGY_CONFIG.get("propulsion_isp_s", 1500.0)))
            eff = max(1e-6, float(ENERGY_CONFIG.get("propulsion_efficiency", 0.65)))
            if self.propellant_kg <= 0.0:
                self.propellant_kg = 0.0
                self._out_of_fuel = True
                prop_power_eff = 0.0
            else:
                mdot = prop_power_eff * eff / (isp_s * g0) ** 2  # kg/s
                consumed = mdot * self.dt * self._orbital_time_compression  # 统一时间压缩 C
                if consumed >= self.propellant_kg:
                    prop_power_eff *= self.propellant_kg / max(consumed, 1e-12)
                    self.propellant_kg = 0.0
                    self._out_of_fuel = True
                else:
                    self.propellant_kg -= consumed
        self._propellant_fraction = float(
            self.propellant_kg / max(getattr(self, "_propellant_initial_kg", 1.0), 1e-9))
        # [SAFETY-REAL] 轨道时间压缩:把高度变化(推力上升 + 阻力衰减)按同一 C 放大,
        # 保持 thrust=drag 平衡不变,只压缩"走向坠毁/恢复"的瞬态 → 不推进则 episode 内衰减坠毁。
        _alt_before = float(self.altitude_m)
        orbit_info = self.orbit_dyn.step(
            _alt_before, prop_power_eff, self.dt,
            diurnal_angle_rad=diurnal_psi)
        _alt_delta = float(orbit_info["altitude_m"]) - _alt_before
        _compressed_alt_delta = _alt_delta * float(getattr(self, "_orbital_time_compression", 1.0))
        _max_alt_delta = float(PROPELLANT_CONFIG.get("max_altitude_delta_m_per_step", 0.0))
        _altitude_delta_limited = False
        if not np.isfinite(_compressed_alt_delta):
            _compressed_alt_delta = -abs(_max_alt_delta) if _max_alt_delta > 0.0 else 0.0
            _altitude_delta_limited = True
        elif _max_alt_delta > 0.0:
            _limited_delta = float(np.clip(_compressed_alt_delta, -_max_alt_delta, _max_alt_delta))
            _altitude_delta_limited = abs(_limited_delta - _compressed_alt_delta) > 1e-9
            _compressed_alt_delta = _limited_delta
        self.altitude_m = _alt_before + _compressed_alt_delta
        # clip:下界避免坠毁过冲到负值致 NaN;上界封死"爬到高轨无阻力白嫖"(上限处仍有阻力需推进)
        self.altitude_m = float(np.clip(
            self.altitude_m, 1.0e4, float(ORBITAL_CONFIG["altitude_max_km"]) * 1e3))
        _orbit_stage, _orbit_stage_code = self.orbit_dyn.classify_altitude(self.altitude_m)
        orbit_info.update({
            "altitude_m": self.altitude_m,
            "dh_m_uncompressed": float(_alt_delta),
            "dh_m": float(self.altitude_m - _alt_before),
            "altitude_delta_limited": bool(_altitude_delta_limited),
            "is_safe": bool(self.altitude_m >= self._h_min),
            "is_crashed": bool(self.altitude_m <= self._h_crash),
            "is_warning": bool(_orbit_stage == "warning"),
            "safety_stage": _orbit_stage,
            "safety_stage_code": _orbit_stage_code,
        })
        mission_stage, mission_stage_code = self._classify_mission_stage(
            orbit_info, batt_info, thermal_info)

        # 积分推进轨道相位
        self.orbit_sim.advance_phase(self.altitude_m, self.dt)

        # 6: 任务数据产生，并进入原始数据队列 raw_queue。
        self._advance_emergency_event_state()
        scene_context = self._scene_context_for_phase()
        self._last_scene_context = scene_context
        data_arrival = self._sample_data_arrival(scene_context)
        # [SAFETY-REAL] 成像门控:仅 IMAGE 指向 + 昼侧地表才能采集原始数据(对地成像姿态)
        if getattr(self, "_attitude_enabled", False):
            ground_daylit = bool(self.orbit_sim.is_sunlit(self.time_s))
            if (not self._can_image) or (not ground_daylit):
                data_arrival = 0.0
            else:
                data_arrival *= float(getattr(self, "_attitude_productivity", 1.0))  # 机动损时
        arrival_info = self.task_tracker.add_arrival(
            data_arrival, self.rng, self.step_count,
            scene_context=scene_context)
        # 星上计算处理 raw_queue，处理完成的数据会被转入 processed_queue。
        service_rate = self.power_sys.throughput_rate(power_info["P_cpu_w"])
        task_action = decode_grouped_action(
            action,
            logit_scale=float(TASK_CONFIG.get("action_selection_logit_scale", 4.0)),
        )
        class_stats_before_drop = self.task_tracker.class_stats(self.step_count)
        low_drop_info = self._apply_active_low_value_drop(
            task_action.drop_low_strength,
            class_stats_before_drop,
        )
        admissible_cpu_mb = float(cpu_gate_meta.get("cpu_gate_admissible_cpu_mb", 0.0))
        reserved_raw_mb = float(cpu_gate_meta.get("cpu_gate_reserved_raw_mb", 0.0))
        physical_cpu_budget_mb = float(service_rate * self.dt)
        # In admissible-budget mode, CPU gate has already translated the requested
        # admissible work into the minimum physical CPU alpha needed for that work.
        # Processing is therefore capped by the gate's effective MB budget, not by
        # executed alpha_cpu * admissible_mb (that would double-scale and under-process).
        if bool(TASK_CONFIG.get("cpu_action_is_admissible_budget", False)):
            effective_cpu_budget_mb = float(cpu_gate_meta.get(
                "cpu_gate_effective_processed_budget_mb",
                float(action[1]) * admissible_cpu_mb,
            ))
            cpu_capacity_mb = min(
                physical_cpu_budget_mb,
                effective_cpu_budget_mb,
                max(0.0, self.data_queue.length),
            )
        else:
            cpu_capacity_mb = min(physical_cpu_budget_mb, max(admissible_cpu_mb, physical_cpu_budget_mb), max(0.0, self.data_queue.length))
        power_cpu_budget_mb = physical_cpu_budget_mb
        _cpu_throttle_applied = bool(power_execution_meta.get(
            "future_contact_cpu_gate_applied", False))
        _proc_util_for_throttle = float(power_execution_meta.get(
            "cpu_gate_ratio_before",
            self.comm_queue.value / max(float(self._future_contact_capacity_mb()), 1e-6),
        ))

        process_info = self.task_tracker.process_by_priority(
            cpu_capacity_mb,
            self.step_count,
            value_weight=task_action.cpu_value_weight,
            urgency_weight=task_action.cpu_urgency_weight,
            future_capacity_fn=self._future_contact_capacity_until_step,
        )
        raw_processed_mb = float(process_info.get(
            "raw_processed_mb",
            process_info.get("processed_mb", 0.0),
        ))
        processed_output_mb = float(process_info.get(
            "processed_output_mb",
            process_info.get("processed_mb", raw_processed_mb),
        ))
        data_info = self.data_queue.update_with_removals(
            data_arrival,
            raw_processed_mb,
            dropped_mb=float(low_drop_info.get("active_dropped_low_raw_mb", 0.0)),
        )
        raw_drop_info = self.task_tracker.drop_raw(float(data_info.get("overflow_mb", 0.0)), self.step_count)
        if raw_queue_was_synced:
            self._assert_task_queue_sync(
                "raw processing update",
                check_raw=True,
                check_processed=False,
            )

        # 先推进时间，再计算 contact；这样通信窗口和下一步观测都对应同一时刻。
        self.time_s += self.dt
        self.step_count += 1

        # 7: 通信窗口（现在用更新后的 time_s）
        self._contact = self._apply_acquisition_latency(self._get_contact_info())
        in_window = self._contact["in_window"]
        tx_capacity = self._contact["max_capacity_mbps"]

        # 8: 下传前先清理过期任务，避免“零价值下传”把过期数据从 processed queue 中移走，
        # 却没有进入 expired_value 惩罚统计。
        expire_info = self.task_tracker.expire(self.step_count)
        self.data_queue.length = max(
            0.0,
            min(self.data_queue.length - float(expire_info.get("expired_raw_mb", 0.0)),
                self.data_queue.max_length),
        )
        self.comm_queue.value = max(
            0.0,
            min(self.comm_queue.value - float(expire_info.get("expired_processed_mb", 0.0)),
                self.comm_queue.max_value),
        )

        # 9: processed_queue 在通信窗口内下传，地面端只有收到后才获得任务价值。
        # 处理完成的数据先进入 processed queue，只有通信窗口内并且发射功率非零时才形成交付。
        lambda_c_mb = processed_output_mb
        link_capacity_mb = tx_capacity * self.dt / 8.0 if tx_capacity > 0 else 0.0
        rf_capacity_mbs = self.power_sys.tx_downlink_rate(power_info["P_tx_w"])
        rf_capacity_mb = rf_capacity_mbs * self.dt
        # [SAFETY-REAL] 下传门控:非 DOWNLINK 指向(或正在机动/去饱和)时链路不可用
        if getattr(self, "_attitude_enabled", False) and not getattr(self, "_can_downlink", True):
            link_capacity_mb = 0.0
            rf_capacity_mb = 0.0
        # 注：窗口期 alpha_tx 硬 floor（Phase 1 硬规则 E）已上移到步骤 1（功率分配之前）。
        # 这样 power_info["P_tx_w"]/rf_capacity_mb 都基于抬高后的 alpha_tx，下方 max_tx_mb
        # 不再被旧的低 rf_capacity_mb 卡死，且电池放电与下传功率口径一致。
        max_tx_mb = 0.0
        if in_window:
            max_tx_mb = min(float(action[2]) * link_capacity_mb, rf_capacity_mb)
            max_tx_mb *= float(getattr(self, "_attitude_productivity", 1.0))  # 机动损时降低本步下传
        processed_low_drop_mb = float(
            low_drop_info.get("active_dropped_low_processed_mb", 0.0))
        pending_processed_mb = max(
            0.0, self.comm_queue.value - processed_low_drop_mb + lambda_c_mb)
        tx_budget_mb = min(max_tx_mb, pending_processed_mb)
        delivery_info = self.task_tracker.deliver_by_priority(
            tx_budget_mb,
            self.step_count,
            value_weight=task_action.tx_value_weight,
            urgency_weight=task_action.tx_urgency_weight,
        )
        actual_tx_override_mb = float(delivery_info.get("delivered_mb", 0.0))
        cq_info = self.comm_queue.update(
            data_arrival_mb=lambda_c_mb, tx_capacity_mb=link_capacity_mb,
            in_window=in_window, alpha_tx=float(action[2]),
            rf_capacity_mb=rf_capacity_mb,
            dropped_mb=processed_low_drop_mb,
            actual_tx_override_mb=actual_tx_override_mb)
        actual_tx_mb = float(cq_info.get("actual_tx_mb", 0.0))
        # 暴露给 reward 层：r_window_underuse 需要 "物理链路容量" (与 action 无关)
        # 和 "本步下传前可发送的 backlog"，从而判断窗口期是否有货却没卡满 tx。
        cq_info["link_capacity_mb"] = float(link_capacity_mb)
        cq_info["pre_tx_pending_mb"] = float(pending_processed_mb)
        cq_info["raw_processed_mb"] = float(raw_processed_mb)
        cq_info["processed_output_mb"] = float(processed_output_mb)
        if in_window and self._max_downlink_mb_per_pass() > 0.0:
            self._comm_pass_remaining_mb = max(
                0.0, self._comm_pass_remaining_mb - actual_tx_mb)
        processed_drop_info = self.task_tracker.drop_processed(
            float(cq_info.get("dropped_mb", cq_info.get("overflow_mb", 0.0))),
            self.step_count)
        if processed_queue_was_synced:
            self._assert_task_queue_sync(
                "processed queue update",
                check_raw=False,
                check_processed=True,
            )
        current_in_window = bool(in_window)

        # 一个通信窗口结束后，进入下一个处理-下传周期前先清零预算计量。
        if self._prev_in_window_for_budget and not current_in_window:
            self._processed_since_contact_mb = 0.0
            self._delivered_since_contact_mb = 0.0
            self._raw_equivalent_processed_since_contact_mb = 0.0
            self._raw_equivalent_delivered_since_contact_mb = 0.0

        compression_ratio_for_budget = max(
            float(process_info.get("compression_ratio", TASK_CONFIG.get("RAW_TO_PROCESSED_RATIO", 1.0))),
            1e-9,
        )
        processed_product_for_budget = float(
            process_info.get("processed_product_mb", process_info.get("processed_mb", 0.0))
        )
        rf_downlinked_for_budget = float(
            delivery_info.get("rf_downlinked_mb", delivery_info.get("delivered_mb", 0.0))
        )
        raw_equiv_processed_for_budget = float(
            process_info.get(
                "raw_equivalent_processed_mb",
                processed_product_for_budget / compression_ratio_for_budget,
            )
        )
        raw_equiv_delivered_for_budget = float(
            delivery_info.get(
                "raw_equivalent_delivered_mb",
                rf_downlinked_for_budget / compression_ratio_for_budget,
            )
        )
        self._processed_since_contact_mb += processed_product_for_budget
        self._delivered_since_contact_mb += rf_downlinked_for_budget
        self._raw_equivalent_processed_since_contact_mb += raw_equiv_processed_for_budget
        self._raw_equivalent_delivered_since_contact_mb += raw_equiv_delivered_for_budget
        self._prev_in_window_for_budget = current_in_window
        task_stats = self.task_tracker.topk_stats(self.step_count)
        class_stats = self.task_tracker.class_stats(self.step_count)
        task_summary = self.task_tracker.summary()
        # reward 必须只读取“本步”的交付/过期/丢弃价值。
        # task_summary 是 episode 累计指标，字段名里也有 delivered_value 等同名键；
        # 如果直接合并，会把本步 delivered_value 覆盖成累计值，导致 reward 随时间虚高到百万级。
        task_summary_prefixed = {
            f"episode_{key}": value for key, value in task_summary.items()
        }
        self._last_delivery_info = {
            **arrival_info,
            **process_info,
            **delivery_info,
            **low_drop_info,
            **raw_drop_info,
            **processed_drop_info,
            **expire_info,
            **task_stats,
            **class_stats,
            **task_summary_prefixed,
        }

        # 9: 更新能量、轨道等安全虚拟队列。
        eq_info = self.energy_queue.update(batt_info["energy_margin_wh"])
        oq_info = self.orbit_queue.update(self.altitude_m)

        # 9.5: 先算 deadline_contact_stats 并缓存（_compute_reward 内部的
        # _deliverable_processing_credit 会读缓存，避免同一步内重复硬算 2 次）。
        time_to_next_window_s_step = float(
            self._contact.get("time_to_next_window_s", 0.0))
        deadline_contact_stats = self.task_tracker.deadline_contact_stats(
            self.step_count,
            0.0 if bool(in_window) else time_to_next_window_s_step / max(float(self.dt), 1e-6),
        )
        self._cached_step = self.step_count
        self._cached_deadline_contact_stats = deadline_contact_stats

        # 10: 计算时效性加权任务价值奖励。
        reward, breakdown = self._compute_reward(
            data_info, batt_info, orbit_info,
            eq_info, oq_info, cq_info,
            actual_tx_mb, in_window, power_info,
            delivery_info=self._last_delivery_info,
            thermal_info=thermal_info)

        executed_prop_delta = abs(float(action[0]) - float(self.prev_action[0]))

        # 11: 更新动作记录（时间和步数已在上面更新）。
        self.prev_action = action.copy()
        terminated, truncated = self._check_done(batt_info, orbit_info, thermal_info)
        done = terminated or truncated

        steps_until = self.N_PROP_SMOOTH - (self.step_count % self.N_PROP_SMOOTH)
        if steps_until == self.N_PROP_SMOOTH:
            steps_until = 0

        delivered_high_value_step = float(delivery_info.get("delivered_high_value", 0.0))
        expired_high_value_step = float(expire_info.get("expired_high_value", 0.0))
        dropped_high_value_step = float(
            raw_drop_info.get("dropped_raw_high_value", 0.0)
            + processed_drop_info.get("dropped_processed_high_value", 0.0)
            + low_drop_info.get("active_dropped_raw_high_value", 0.0)
            + low_drop_info.get("active_dropped_processed_high_value", 0.0)
        )
        high_value_delivery_ratio_step = delivered_high_value_step / max(
            delivered_high_value_step + expired_high_value_step + dropped_high_value_step,
            1e-9,
        )
        future_contact_capacity_mb_step = float(
            low_drop_info.get("future_capacity_mb", self._future_contact_capacity_mb())
        )
        deliverability_info_step = self.task_tracker.deliverability_features(
            self.step_count,
            self._future_contact_capacity_bins(),
        )
        # ── 缓存（供 _get_observation 复用，避免同一步内重复调用那些 90×19 扫描函数）──
        self._cached_step = self.step_count
        self._cached_deliverability_features = deliverability_info_step
        processed_queue_future_contact_ratio = float(
            self.comm_queue.value / max(future_contact_capacity_mb_step, 1e-6)
        )
        processed_value_step = float(process_info.get("processed_value", 0.0))
        delivered_value_step = float(delivery_info.get("delivered_value", 0.0))
        compression_ratio_step = float(process_info.get("compression_ratio", 1.0))
        processed_product_mb_step = float(
            process_info.get("processed_product_mb", processed_output_mb)
        )
        rf_downlinked_mb_step = float(
            delivery_info.get(
                "rf_downlinked_mb",
                delivery_info.get("delivered_mb", actual_tx_mb),
            )
        )
        raw_equivalent_processed_mb_step = float(
            process_info.get(
                "raw_equivalent_processed_mb",
                processed_product_mb_step / max(compression_ratio_step, 1e-9),
            )
        )
        raw_equivalent_delivered_mb_step = float(
            delivery_info.get(
                "raw_equivalent_delivered_mb",
                rf_downlinked_mb_step / max(compression_ratio_step, 1e-9),
            )
        )
        processed_queue_raw_equivalent_mb = float(
            self.comm_queue.value / max(compression_ratio_step, 1e-9)
        )
        future_contact_capacity_raw_equivalent_mb = float(
            future_contact_capacity_mb_step / max(compression_ratio_step, 1e-9)
        )
        processed_voi_basis_value_step = float(
            process_info.get("processed_voi_basis_value", processed_value_step)
        )
        useful_processing_ratio_step = (
            delivered_value_step / max(processed_voi_basis_value_step, 1e-6)
            if processed_voi_basis_value_step > 1e-9
            else 0.0
        )
        episode_processed_value = float(task_summary.get("processed_value", 0.0))
        episode_processed_voi_basis_value = float(
            task_summary.get("processed_voi_basis_value", episode_processed_value)
        )
        episode_delivered_value = float(task_summary.get("delivered_value", 0.0))
        episode_useful_processing_ratio = float(
            task_summary.get(
                "useful_processing_ratio",
                episode_delivered_value / max(episode_processed_voi_basis_value, 1e-6)
                if episode_processed_voi_basis_value > 1e-9
                else 0.0,
            )
        )
        episode_proc_dl_ratio = float(
            task_summary.get(
                "proc_dl_ratio",
                float(task_summary.get("processed_mb", 0.0))
                / max(float(task_summary.get("delivered_mb", 0.0)), 1e-6),
            )
        )
        episode_processed_product_mb = float(
            task_summary.get("processed_product_mb", task_summary.get("processed_mb", 0.0))
        )
        episode_rf_downlinked_mb = float(
            task_summary.get("rf_downlinked_mb", task_summary.get("delivered_mb", 0.0))
        )
        episode_raw_equiv_processed_mb = float(task_summary.get("raw_equivalent_processed_mb", 0.0))
        episode_raw_equiv_delivered_mb = float(task_summary.get("raw_equivalent_delivered_mb", 0.0))
        rf_product_proc_downlink_ratio_step = float(
            processed_product_mb_step / max(rf_downlinked_mb_step, 1e-6)
        )
        raw_equivalent_proc_delivery_ratio_step = float(
            raw_equivalent_processed_mb_step / max(raw_equivalent_delivered_mb_step, 1e-6)
        )
        raw_equivalent_delivery_coverage_step = float(
            raw_equivalent_delivered_mb_step / max(raw_equivalent_processed_mb_step, 1e-6)
            if raw_equivalent_processed_mb_step > 1e-9
            else 0.0
        )
        episode_rf_product_proc_downlink_ratio = float(
            episode_processed_product_mb / max(episode_rf_downlinked_mb, 1e-6)
        )
        episode_raw_equivalent_proc_delivery_ratio = float(
            episode_raw_equiv_processed_mb / max(episode_raw_equiv_delivered_mb, 1e-6)
        )
        episode_raw_equivalent_delivery_coverage = float(
            episode_raw_equiv_delivered_mb / max(episode_raw_equiv_processed_mb, 1e-6)
            if episode_raw_equiv_processed_mb > 1e-9
            else 0.0
        )
        episode_value_realization_ratio = float(
            episode_delivered_value / max(episode_processed_voi_basis_value, 1e-6)
            if episode_processed_voi_basis_value > 1e-9
            else 0.0
        )
        # deadline_contact_stats 已在 _compute_reward 前算好并缓存，直接复用
        deadline_contact_stats = self._cached_deadline_contact_stats
        # 与 r_proc_far_window shaping 同源的诊断阈值：默认走 cpu_gate 的 120s 边界，
        # 消除"gate strict 在 120s+ 但日志只统计 300s+"的 gap。需要旧的 300s 二值统计
        # 可以单独读取 cpu_active_strictly_far_rate（保留兼容）。
        far_log_threshold_s = float(TASK_CONFIG.get(
            "cpu_active_far_log_threshold_s",
            TASK_CONFIG.get("cpu_gate_far_window_lead_s", 120.0),
        ))
        cpu_active_far_from_window_rate = float(
            (not bool(in_window))
            and time_to_next_window_s_step > far_log_threshold_s
            and float(action[1]) > 0.10
            and float(process_info.get("processed_mb", 0.0)) > 1.0
        )
        cpu_active_strictly_far_rate = float(
            (not bool(in_window))
            and time_to_next_window_s_step > float(
                DRL_CONFIG.get("constraint_prepass_min_lead_s", 300.0)
            )
            and float(action[1]) > 0.10
            and float(process_info.get("processed_mb", 0.0)) > 1.0
        )
        active_low_drop_mb = float(
            low_drop_info.get("active_dropped_low_raw_mb", 0.0)
            + low_drop_info.get("active_dropped_low_processed_mb", 0.0)
        )
        passive_low_drop_mb = float(
            raw_drop_info.get("dropped_raw_low_mb", 0.0)
            + processed_drop_info.get("dropped_processed_low_mb", 0.0)
        )
        low_drop_recall = float(
            active_low_drop_mb / max(float(low_drop_info.get("droppable_backlog_mb", 0.0)), 1e-6)
        )
        low_processing_ratio = float(
            process_info.get("processed_low_mb", 0.0)
            / max(float(process_info.get("processed_mb", 0.0)), 1e-6)
        )
        low_delivery_ratio = float(
            delivery_info.get("delivered_low_mb", 0.0)
            / max(float(delivery_info.get("delivered_mb", actual_tx_mb)), 1e-6)
        )

        info = {
            "step": self.step_count, "time_s": self.time_s,
            "terminated": terminated, "truncated": truncated,
            "executed_action": action.copy(),
            "prop_can_update": prop_can_update,
            "prop_smoothing_enforced": bool(enforce_prop_smoothing),
            "safety_override": safety_override,
            "prop_safety_override_reason": prop_safety_override_reason,
            **propulsion_controller_meta,
            "input_action_in_bounds": input_action_in_bounds,
            "action_bounds_safe": True,
            "prop_delta": float(prop_delta),
            "executed_prop_delta": float(executed_prop_delta),
            "steps_until_prop_update": steps_until,
            "altitude_km": self.altitude_m / 1e3,
            "propellant_kg": float(getattr(self, "propellant_kg", 0.0)),
            "propellant_fraction": float(getattr(self, "_propellant_fraction", 1.0)),
            "out_of_fuel": float(getattr(self, "_out_of_fuel", False)),
            "pointing_mode": int(getattr(self, "_pointing_mode", POINTING_SUN)),
            "attitude_momentum": float(getattr(self, "_momentum", 0.0)),
            "attitude_slew": float(getattr(self, "_slew_active", False)),
            "attitude_desat": float(getattr(self, "_desat_active", False)),
            **mission_pointing_meta,
            # ── 指向/任务可行性诊断字段 (定位 "对日保守 → 不成像/不下传") ──
            # pointing_mode(执行后) / mission_pointing_mode_before(策略原始请求) /
            # mission_pointing_fallback_reason 已在上方;此处补三个缺失的可行性标志。
            "sunlit": float(self.orbit_sim.is_sunlit(self.time_s, self.altitude_m)),
            "can_image": float(getattr(self, "_can_image", True)),
            "can_downlink": float(getattr(self, "_can_downlink", True)),
            "soc": batt_info["soc"],
            "battery_capacity_wh": float(batt_info.get("capacity_wh", self.battery.capacity_wh)),
            "battery_cycle_degradation": float(batt_info.get("cycle_degradation", 0.0)),
            "battery_efc": float(batt_info.get("equivalent_full_cycles", 0.0)),
            "battery_capacity_loss_wh": float(batt_info.get("capacity_loss_wh", 0.0)),
            "thermal_temperature_c": float(thermal_info.get("temperature_c", self.thermal_temperature_c)),
            "thermal_margin_norm": float(thermal_info.get("thermal_margin_norm", 1.0)),
            "thermal_safe": float(thermal_info.get("is_safe", True)),
            "thermal_warning": float(thermal_info.get("is_warning", False)),
            "thermal_crashed": float(thermal_info.get("is_crashed", False)),
            "thermal_stage": str(thermal_info.get("safety_stage", "normal")),
            "thermal_internal_heat_w": float(thermal_info.get("internal_heat_w", 0.0)),
            "thermal_electronics_heat_w": float(thermal_info.get("electronics_heat_w", 0.0)),
            "thermal_propulsion_heat_w": float(thermal_info.get("propulsion_heat_w", 0.0)),
            "thermal_throttle_applied": bool(power_execution_meta.get("thermal_throttle_applied", False)),
            "thermal_cpu_cap": float(power_execution_meta.get("thermal_cpu_cap", 1.0)),
            "thermal_tx_cap": float(power_execution_meta.get("thermal_tx_cap", 1.0)),
            "raw_queue_mb": self.data_queue.length,
            "raw_queue_utilization": self.data_queue.length / max(self.data_queue.max_length, 1e-6),
            "raw_queue_overflow_mb": float(data_info.get("overflow_mb", 0.0)),
            "processed_queue_mb": self.comm_queue.value,
            "processed_queue_product_mb": self.comm_queue.value,
            "processed_queue_utilization": self.comm_queue.value / max(self.comm_queue.max_value, 1e-6),
            "processed_queue_overflow_mb": float(cq_info.get("overflow_mb", 0.0)),
            "processed_since_contact_mb": float(self._processed_since_contact_mb),
            "delivered_since_contact_mb": float(self._delivered_since_contact_mb),
            "processed_product_since_contact_mb": float(self._processed_since_contact_mb),
            "rf_downlinked_since_contact_mb": float(self._delivered_since_contact_mb),
            "raw_equivalent_processed_since_contact_mb": float(
                self._raw_equivalent_processed_since_contact_mb),
            "raw_equivalent_delivered_since_contact_mb": float(
                self._raw_equivalent_delivered_since_contact_mb),
            "raw_equivalent_since_contact_delivery_coverage": float(
                self._raw_equivalent_delivered_since_contact_mb
                / max(self._raw_equivalent_processed_since_contact_mb, 1e-6)
                if self._raw_equivalent_processed_since_contact_mb > 1e-9
                else 0.0
            ),
            "raw_high_mb": float(class_stats.get("raw_high_mb", 0.0)),
            "raw_mid_mb": float(class_stats.get("raw_medium_mb", 0.0)),
            "raw_low_mb": float(class_stats.get("raw_low_mb", 0.0)),
            "processed_high_mb": float(class_stats.get("processed_high_mb", 0.0)),
            "processed_mid_mb": float(class_stats.get("processed_medium_mb", 0.0)),
            "processed_low_mb": float(class_stats.get("processed_low_mb", 0.0)),
            "expiring_high_value": float(class_stats.get("expiring_high_value", 0.0)),
            "expiring_mid_value": float(class_stats.get("expiring_medium_value", 0.0)),
            "expiring_low_value": float(class_stats.get("expiring_low_value", 0.0)),
            "data_queue_mb": self.data_queue.length,
            "raw_processed_mb": raw_processed_mb,
            "processed_output_mb": processed_output_mb,
            "processed_product_mb": processed_product_mb_step,
            "raw_equivalent_processed_mb": raw_equivalent_processed_mb_step,
            "compression_ratio": compression_ratio_step,
            "value_retention": float(process_info.get("value_retention", 1.0)),
            "processed_mb": processed_output_mb,
            "processed_value": processed_value_step,
            "processed_voi_basis_value": processed_voi_basis_value_step,
            "processed_high_mb_step": float(process_info.get("processed_high_mb", 0.0)),
            "processed_mid_mb_step": float(process_info.get("processed_medium_mb", 0.0)),
            "processed_low_mb_step": float(process_info.get("processed_low_mb", 0.0)),
            "processed_high_value_step": float(process_info.get("processed_high_value", 0.0)),
            "processed_mid_value_step": float(process_info.get("processed_medium_value", 0.0)),
            "processed_low_value_step": float(process_info.get("processed_low_value", 0.0)),
            "processed_deliverable_value_step": float(process_info.get("processed_deliverable_value", 0.0)),
            "processed_undeliverable_value_step": float(process_info.get("processed_undeliverable_value", 0.0)),
            "cpu_skip_break_triggered": float(process_info.get("cpu_skip_break_triggered", 0.0)),
            "data_queue_utilization": self.data_queue.length / max(self.data_queue.max_length, 1e-6),
            "overflow_mb": float(data_info.get("overflow_mb", 0.0)),
            "energy_virtual_queue": eq_info["queue_value"],
            "orbit_virtual_queue": oq_info["queue_value"],
            "comm_virtual_queue": self.comm_queue.value,
            "comm_urgency": float(cq_info.get("urgency", 0.0)),
            "comm_urgency_raw": float(cq_info.get("urgency_raw", cq_info.get("urgency", 0.0))),
            "comm_overflow_mb": float(cq_info.get("overflow_mb", 0.0)),
            "dropped_raw_mb": float(raw_drop_info.get("dropped_raw_mb", 0.0)),
            "dropped_raw_value": float(raw_drop_info.get("dropped_raw_value", 0.0)),
            "dropped_processed_mb": float(processed_drop_info.get("dropped_processed_mb", 0.0)),
            "dropped_processed_value": float(processed_drop_info.get("dropped_processed_value", 0.0)),
            "active_dropped_low_raw_mb": float(low_drop_info.get("active_dropped_low_raw_mb", 0.0)),
            "active_dropped_low_processed_mb": float(low_drop_info.get("active_dropped_low_processed_mb", 0.0)),
            "active_dropped_low_mb": active_low_drop_mb,
            "active_low_drop_mb": active_low_drop_mb,
            "active_dropped_low_value": float(low_drop_info.get("active_dropped_low_value", 0.0)),
            "passive_dropped_low_raw_mb": float(raw_drop_info.get("dropped_raw_low_mb", 0.0)),
            "passive_dropped_low_processed_mb": float(
                processed_drop_info.get("dropped_processed_low_mb", 0.0)),
            "passive_low_drop_mb": passive_low_drop_mb,
            "low_drop_recall": low_drop_recall,
            "low_processing_ratio": low_processing_ratio,
            "low_delivery_ratio": low_delivery_ratio,
            "droppable_low_backlog_mb": float(low_drop_info.get("droppable_backlog_mb", 0.0)),
            "dropped_low_mb": float(
                raw_drop_info.get("dropped_raw_low_mb", 0.0)
                + processed_drop_info.get("dropped_processed_low_mb", 0.0)
                + low_drop_info.get("active_dropped_low_raw_mb", 0.0)
                + low_drop_info.get("active_dropped_low_processed_mb", 0.0)),
            "low_value_dropped_mb": float(
                raw_drop_info.get("dropped_raw_low_mb", 0.0)
                + processed_drop_info.get("dropped_processed_low_mb", 0.0)
                + low_drop_info.get("active_dropped_low_raw_mb", 0.0)
                + low_drop_info.get("active_dropped_low_processed_mb", 0.0)),
            "low_value_dropped_value": float(
                raw_drop_info.get("dropped_raw_low_value", 0.0)
                + processed_drop_info.get("dropped_processed_low_value", 0.0)
                + low_drop_info.get("active_dropped_low_value", 0.0)),
            "dropped_value": float(
                raw_drop_info.get("dropped_raw_value", 0.0)
                + processed_drop_info.get("dropped_processed_value", 0.0)
                + low_drop_info.get("active_dropped_low_value", 0.0)),
            "dropped_high_value": dropped_high_value_step,
            "expired_mb": float(expire_info.get("expired_mb", 0.0)),
            "expired_value": float(expire_info.get("expired_value", 0.0)),
            "expired_raw_mb": float(expire_info.get("expired_raw_mb", 0.0)),
            "expired_processed_mb": float(expire_info.get("expired_processed_mb", 0.0)),
            "expired_raw_value": float(expire_info.get("expired_raw_value", 0.0)),
            "expired_processed_value": float(expire_info.get("expired_processed_value", 0.0)),
            "expired_high_value": expired_high_value_step,
            "delivered_mb": float(delivery_info.get("delivered_mb", actual_tx_mb)),
            "rf_downlinked_mb": rf_downlinked_mb_step,
            "raw_equivalent_delivered_mb": raw_equivalent_delivered_mb_step,
            "rf_product_proc_downlink_ratio": rf_product_proc_downlink_ratio_step,
            "raw_equivalent_proc_delivery_ratio": raw_equivalent_proc_delivery_ratio_step,
            "raw_equivalent_delivery_coverage": raw_equivalent_delivery_coverage_step,
            "delivered_high_mb": float(delivery_info.get("delivered_high_mb", 0.0)),
            "high_value_downlink_mb": float(delivery_info.get("delivered_high_mb", 0.0)),
            "delivered_mid_mb": float(delivery_info.get("delivered_medium_mb", 0.0)),
            "delivered_low_mb": float(delivery_info.get("delivered_low_mb", 0.0)),
            "delivered_value": float(delivery_info.get("delivered_value", 0.0)),
            "delivered_high_value": delivered_high_value_step,
            "high_value_downlink_value": delivered_high_value_step,
            "delivered_mid_value": float(delivery_info.get("delivered_medium_value", 0.0)),
            "delivered_low_value": float(delivery_info.get("delivered_low_value", 0.0)),
            "deadline_success_value": float(delivery_info.get("on_time_delivered_value", 0.0)),
            "avg_delivery_delay_steps": float(delivery_info.get("avg_delivery_delay_steps", 0.0)),
            "aoi_steps": float(delivery_info.get("aoi_steps", delivery_info.get("avg_delivery_delay_steps", 0.0))),
            "average_aoi_steps": float(task_summary.get("average_aoi_steps", task_summary.get("avg_delivery_delay_steps", 0.0))),
            "useful_processing_ratio": useful_processing_ratio_step,
            "episode_processed_mb": float(task_summary.get("processed_mb", 0.0)),
            "value_realization_ratio": useful_processing_ratio_step,
            "episode_processed_product_mb": episode_processed_product_mb,
            "episode_raw_equivalent_processed_mb": episode_raw_equiv_processed_mb,
            "episode_processed_value": episode_processed_value,
            "episode_processed_voi_basis_value": episode_processed_voi_basis_value,
            "episode_delivered_mb": float(task_summary.get("delivered_mb", 0.0)),
            "episode_rf_downlinked_mb": episode_rf_downlinked_mb,
            "episode_raw_equivalent_delivered_mb": episode_raw_equiv_delivered_mb,
            "episode_delivered_value": episode_delivered_value,
            "episode_generated_value": float(task_summary.get("generated_value", 0.0)),
            "episode_proc_dl_ratio": episode_proc_dl_ratio,
            "episode_rf_product_proc_downlink_ratio": episode_rf_product_proc_downlink_ratio,
            "episode_raw_equivalent_proc_delivery_ratio": episode_raw_equivalent_proc_delivery_ratio,
            "episode_raw_equivalent_delivery_coverage": episode_raw_equivalent_delivery_coverage,
            "episode_useful_processing_ratio": episode_useful_processing_ratio,
            "episode_value_realization_ratio": episode_value_realization_ratio,
            "scene_name": str(scene_context.get("scene_name", "generic")),
            "scene_class_code": float(scene_context.get("scene_class_code", 0.0)),
            "scene_arrival_multiplier": float(scene_context.get("arrival_multiplier", 1.0)),
            "scene_phase_fraction": float(scene_context.get("phase_fraction", 0.0)),
            "scene_latitude_proxy": float(scene_context.get("latitude_proxy", 0.0)),
            "scene_cloud_cover": float(arrival_info.get("scene_cloud_cover", 0.0)),
            "data_arrival_mb": float(data_arrival),
            "emergency_event_active": float(scene_context.get("emergency_event_active", False)),
            "emergency_event_triggered": float(scene_context.get("emergency_event_triggered", False)),
            "emergency_event_remaining_steps": float(scene_context.get("emergency_event_remaining_steps", 0.0)),
            "generated_value_density": float(arrival_info.get("generated_value_density", 0.0)),
            "generated_priority": float(arrival_info.get("generated_priority", 0.0)),
            "generated_quality": float(arrival_info.get("generated_quality", 0.0)),
            "generated_deadline_steps": float(arrival_info.get("generated_deadline_steps", 0.0)),
            "top_task_priority": float(task_stats.get("top_task_priority", 0.0)),
            "top_task_quality": float(task_stats.get("top_task_quality", 0.0)),
            "deadline_urgency": float(task_stats.get("deadline_urgency", 0.0)),
            "expiring_value": float(task_stats.get("expiring_value", 0.0)),
            "cpu_ratio_high": float(task_action.cpu_ratios[0]),
            "cpu_ratio_mid": float(task_action.cpu_ratios[1]),
            "cpu_ratio_low": float(task_action.cpu_ratios[2]),
            "tx_ratio_high": float(task_action.tx_ratios[0]),
            "tx_ratio_mid": float(task_action.tx_ratios[1]),
            "tx_ratio_low": float(task_action.tx_ratios[2]),
            "cpu_value_weight": float(task_action.cpu_value_weight),
            "cpu_urgency_weight": float(task_action.cpu_urgency_weight),
            "tx_value_weight": float(task_action.tx_value_weight),
            "tx_urgency_weight": float(task_action.tx_urgency_weight),
            "drop_low_strength": float(task_action.drop_low_strength),
            # 问题3修复：分开requested和executed allocation
            "cpu_requested_high": float(task_action.cpu_ratios[0]),
            "cpu_requested_mid": float(task_action.cpu_ratios[1]),
            "cpu_requested_low": float(task_action.cpu_ratios[2]),
            "cpu_executed_share_high": float(process_info.get("processed_high_mb", 0.0)) / max(float(process_info.get("processed_mb", 0.0)), 1e-6),
            "cpu_executed_share_mid": float(process_info.get("processed_medium_mb", 0.0)) / max(float(process_info.get("processed_mb", 0.0)), 1e-6),
            "cpu_executed_share_low": float(process_info.get("processed_low_mb", 0.0)) / max(float(process_info.get("processed_mb", 0.0)), 1e-6),
            "tx_requested_high": float(task_action.tx_ratios[0]),
            "tx_requested_mid": float(task_action.tx_ratios[1]),
            "tx_requested_low": float(task_action.tx_ratios[2]),
            "tx_executed_share_high": float(delivery_info.get("delivered_high_mb", 0.0)) / max(float(delivery_info.get("delivered_mb", 0.0)), 1e-6),
            "tx_executed_share_mid": float(delivery_info.get("delivered_medium_mb", 0.0)) / max(float(delivery_info.get("delivered_mb", 0.0)), 1e-6),
            "tx_executed_share_low": float(delivery_info.get("delivered_low_mb", 0.0)) / max(float(delivery_info.get("delivered_mb", 0.0)), 1e-6),
            "future_capacity_mb": future_contact_capacity_mb_step,
            "future_contact_capacity_mb": future_contact_capacity_mb_step,
            "processed_queue_future_contact_ratio": processed_queue_future_contact_ratio,
            "processed_queue_future_contact_ratio_raw": processed_queue_future_contact_ratio,
            "processed_queue_future_contact_ratio_raw_equiv": processed_queue_future_contact_ratio,
            "processed_queue_to_future_contact_ratio": processed_queue_future_contact_ratio,
            "processed_queue_raw_equivalent_mb": processed_queue_raw_equivalent_mb,
            "future_contact_capacity_raw_equivalent_mb": future_contact_capacity_raw_equivalent_mb,
            "processed_high_next_window_deliverable_ratio": float(
                deadline_contact_stats.get("processed_high_next_window_deliverable_ratio", 0.0)),
            "raw_high_next_window_deliverable_ratio": float(
                deadline_contact_stats.get("raw_high_next_window_deliverable_ratio", 0.0)),
            "high_value_deadline_contact_mismatch": float(
                deadline_contact_stats.get("high_value_deadline_contact_mismatch", 0.0)),
            **deliverability_info_step,
            "raw_high_next_window_deliverable_mb": float(
                deadline_contact_stats.get("raw_high_next_window_deliverable_mb", 0.0)),
            "raw_high_next_window_deliverable_raw_equiv_mb": float(
                deadline_contact_stats.get("raw_high_next_window_deliverable_raw_equiv_mb", 0.0)),
            "processed_high_next_window_deliverable_mb": float(
                deadline_contact_stats.get("processed_high_next_window_deliverable_mb", 0.0)),
            "processed_high_next_window_deliverable_raw_equiv_mb": float(
                deadline_contact_stats.get("processed_high_next_window_deliverable_raw_equiv_mb", 0.0)),
            "raw_high_backlog_mb": float(
                deadline_contact_stats.get("raw_high_backlog_mb", 0.0)),
            "processed_high_product_backlog_mb": float(
                deadline_contact_stats.get("processed_high_product_backlog_mb", 0.0)),
            "processed_high_raw_equiv_backlog_mb": float(
                deadline_contact_stats.get("processed_high_raw_equiv_backlog_mb", 0.0)),
            "total_high_raw_equiv_backlog_mb": float(
                deadline_contact_stats.get("total_high_raw_equiv_backlog_mb", 0.0)),
            "high_value_backlog_mb": float(
                deadline_contact_stats.get("high_value_backlog_mb", 0.0)),
            "high_value_backlog_value": float(
                deadline_contact_stats.get("high_value_backlog_value", 0.0)),
            "protected_demand_mb": float(low_drop_info.get("protected_demand_mb", 0.0)),
            "low_capacity_slack_mb": float(low_drop_info.get("low_capacity_slack_mb", 0.0)),
            "low_excess_mb": float(low_drop_info.get("low_excess_mb", 0.0)),
            "future_contact_shortage": float(low_drop_info.get("future_contact_shortage", 0.0)),
            "resource_pressure": float(low_drop_info.get("resource_pressure", 0.0)),
            "active_drop_budget_mb": float(low_drop_info.get("active_drop_budget_mb", 0.0)),
            "capacity_driven_drop_mb": float(low_drop_info.get("capacity_driven_drop_mb", 0.0)),
            "queue_driven_drop_mb": float(low_drop_info.get("queue_driven_drop_mb", 0.0)),
            "low_share_driven_drop_mb": float(low_drop_info.get("low_share_driven_drop_mb", 0.0)),
            "policy_driven_drop_mb": float(low_drop_info.get("policy_driven_drop_mb", 0.0)),
            "active_dropped_total_value": float(low_drop_info.get("active_dropped_total_value", 0.0)),
            "active_dropped_raw_high_value": float(low_drop_info.get("active_dropped_raw_high_value", 0.0)),
            "active_dropped_raw_medium_value": float(low_drop_info.get("active_dropped_raw_medium_value", 0.0)),
            "active_dropped_raw_low_value": float(low_drop_info.get("active_dropped_raw_low_value", 0.0)),
            "active_dropped_processed_high_value": float(low_drop_info.get("active_dropped_processed_high_value", 0.0)),
            "active_dropped_processed_medium_value": float(low_drop_info.get("active_dropped_processed_medium_value", 0.0)),
            "active_dropped_processed_low_value": float(low_drop_info.get("active_dropped_processed_low_value", 0.0)),
            "cpu_unused_before_reallocation_mb": float(process_info.get("cpu_unused_before_reallocation_mb", 0.0)),
            "cpu_reallocated_mb": float(process_info.get("cpu_reallocated_mb", 0.0)),
            "cpu_reallocated_to_high_mb": float(process_info.get("cpu_reallocated_to_high_mb", 0.0)),
            "cpu_reallocated_to_mid_mb": float(process_info.get("cpu_reallocated_to_medium_mb", 0.0)),
            "cpu_reallocated_to_low_mb": float(process_info.get("cpu_reallocated_to_low_mb", 0.0)),
            "tx_unused_before_reallocation_mb": float(delivery_info.get("tx_unused_before_reallocation_mb", 0.0)),
            "tx_reallocated_mb": float(delivery_info.get("tx_reallocated_mb", 0.0)),
            "tx_reallocated_to_high_mb": float(delivery_info.get("tx_reallocated_to_high_mb", 0.0)),
            "tx_reallocated_to_mid_mb": float(delivery_info.get("tx_reallocated_to_medium_mb", 0.0)),
            "tx_reallocated_to_low_mb": float(delivery_info.get("tx_reallocated_to_low_mb", 0.0)),
            "cpu_reallocation_rate": float(process_info.get("cpu_reallocated_mb", 0.0)) / max(power_cpu_budget_mb, 1e-6),
            "tx_reallocation_rate": float(delivery_info.get("tx_reallocated_mb", 0.0)) / max(tx_budget_mb, 1e-6),
            "alpha_cpu": float(action[1]),
            "alpha_tx": float(action[2]),
            "cpu_active_far_from_window_rate": cpu_active_far_from_window_rate,
            "cpu_active_far_from_window": cpu_active_far_from_window_rate,
            "cpu_active_strictly_far_rate": cpu_active_strictly_far_rate,
            "cpu_capacity_mb": float(cpu_capacity_mb),
            "cpu_physical_capacity_mb": float(power_cpu_budget_mb),
            "cpu_admissible_mb": float(admissible_cpu_mb),
            "cpu_reserved_raw_mb": float(reserved_raw_mb),
            "comm_queue_max": float(self.comm_queue.max_value),
            "cpu_throttle_applied": float(_cpu_throttle_applied),
            "cpu_throttle_proc_util": float(_proc_util_for_throttle),
            "value_per_mb": float(task_summary.get("value_per_mb", 0.0)),
            "value_per_rf_downlinked_mb": float(task_summary.get("value_per_rf_downlinked_mb", 0.0)),
            "value_per_raw_equivalent_mb": float(task_summary.get("value_per_raw_equivalent_mb", 0.0)),
            "deadline_success_rate": float(task_summary.get("deadline_success_rate", 0.0)),
            "value_weighted_deadline_success_rate": float(
                task_summary.get(
                    "value_weighted_deadline_success_rate",
                    task_summary.get("deadline_success_rate", 0.0),
                )
            ),
            "expired_value_rate": float(task_summary.get("expired_value_rate", 0.0)),
            "voi_degradation_rate": float(task_summary.get("voi_degradation_rate", task_summary.get("expired_value_rate", 0.0))),
            "voi_loss_rate": float(task_summary.get("voi_loss_rate", 0.0)),
            "value_weighted_aoi_steps": float(
                task_summary.get(
                    "value_weighted_aoi_steps",
                    task_summary.get("average_aoi_steps", 0.0),
                )
            ),
            "voi_delivered_value": float(delivery_info.get("voi_delivered_value", delivery_info.get("delivered_value", 0.0))),
            "dropped_value_rate": float(task_summary.get("dropped_value_rate", 0.0)),
            "high_value_delivery_ratio": float(high_value_delivery_ratio_step),
            "raw_queue_safe": float(data_info.get("overflow_mb", 0.0) <= 1e-9),
            "processed_queue_safe": float(cq_info.get("overflow_mb", 0.0) <= 1e-9),
            "orbit_safe": float(orbit_info.get("is_safe", True)),
            "energy_safe": float(batt_info.get("is_safe", True)),
            "orbit_warning": float(orbit_info.get("is_warning", False)),
            "energy_warning": float(batt_info.get("is_warning", False)),
            "orbit_stage": str(orbit_info.get("safety_stage", "normal")),
            "energy_stage": str(batt_info.get("safety_stage", "normal")),
            "orbit_crashed": float(orbit_info.get("is_crashed", False)),
            "energy_crashed": float(batt_info.get("is_crashed", False)),
            "crashed": float(
                orbit_info.get("is_crashed", False)
                or batt_info.get("is_crashed", False)
                or thermal_info.get("is_crashed", False)),
            "risk_stage": mission_stage,
            "risk_stage_code": float(mission_stage_code),
            "nominal_state": float(mission_stage == "normal"),
            "warning_state": float(mission_stage == "warning"),
            "unsafe_state": float(mission_stage == "unsafe"),
            "failure_state": float(mission_stage == "failure"),
            "power_constraint_safe": float(power_constraint_safe),
            "available_power_w": float(available_power_w),
            "adjustable_power_budget_w": float(max(available_power_w - power_info["P_baseline_w"], 0.0)),
            **power_execution_meta,
            "overall_safe": float(
                orbit_info.get("is_safe", True)
                and batt_info.get("is_safe", True)
                and thermal_info.get("is_safe", True)
                and data_info.get("overflow_mb", 0.0) <= 1e-9
                and cq_info.get("overflow_mb", 0.0) <= 1e-9
                and power_constraint_safe),
            "P_solar_w": P_solar,
            "sunlit": self.orbit_sim.is_sunlit(),
            "in_window": in_window, "tx_capacity_mbps": tx_capacity,
            "time_to_next_window_s": time_to_next_window_s_step,
            "next_window_in_range": bool(self._contact.get("time_to_next_window_s", 5400.0) < 5400.0 - 1e-6 or in_window),
            "future_contact_capacity_norm": float(self._future_contact_capacity_norm()),
            "actual_tx_mb": actual_tx_mb,
            "service_rate_mbs": service_rate,
            "physical_service_rate_mbs": service_rate,
            "link_tx_capacity_mb": float(link_capacity_mb),
            "rf_tx_capacity_mbs": float(rf_capacity_mbs),
            "rf_tx_capacity_mb": float(rf_capacity_mb),
            "effective_tx_capacity_mb": float(cq_info.get("effective_tx_capacity_mb", 0.0)),
            "comm_window_age_steps": int(self._comm_window_age_steps),
            "comm_pass_remaining_mb": float(self._comm_pass_remaining_mb),
            "comm_pass_capacity_mb": float(self._comm_pass_capacity_mb),
            "acquisition_latency_active": bool(self._contact.get("acquisition_latency_active", False)),
            "acquisition_latency_scale": float(self._contact.get("acquisition_latency_scale", 1.0)),
            "window_ratio": cq_info["window_ratio"],
            "reward_breakdown": breakdown,
            **power_info,
        }
        diagnostic_safety_cost = compute_lyapunov_safety_cost(
            previous_queues=(
                self.energy_queue.prev_value,
                self.orbit_queue.prev_value,
                self.data_queue.prev_value,
                self.comm_queue.prev_value,
            ),
            next_queues=(
                self.energy_queue.value,
                self.orbit_queue.value,
                self.data_queue.length,
                self.comm_queue.value,
            ),
            queue_maxes=(
                self.energy_queue.max_value,
                self.orbit_queue.max_value,
                self.data_queue.max_length,
                self.comm_queue.max_value,
            ),
            info={**info, "_thermal_excess_c": float(breakdown.get("_thermal_excess_c", 0.0))},
        )
        info["costs"] = {
            "state_safety_cost": compute_state_safety_penalty(info),
            "queue_cost": float(diagnostic_safety_cost.queue_cost),
            "processed_backlog_cost": float(diagnostic_safety_cost.processed_backlog_cost),
            "window_waste_cost": float(diagnostic_safety_cost.window_waste_cost),
            "low_value_waste_cost": float(diagnostic_safety_cost.low_value_waste_cost),
            "over_processing_cost": float(diagnostic_safety_cost.over_processing_cost),
            "unproductive_cpu_cost": float(diagnostic_safety_cost.unproductive_cpu_cost),
            "energy_cost": float(diagnostic_safety_cost.energy_cost),
            "orbit_cost": float(diagnostic_safety_cost.orbit_cost),
            "thermal_cost": float(diagnostic_safety_cost.thermal_cost),
            "task_loss_cost": float(diagnostic_safety_cost.task_loss_cost),
            "efficiency_cost": float(diagnostic_safety_cost.efficiency_cost),
            "total_cost": float(diagnostic_safety_cost.total_cost),
            "raw_cost": float(diagnostic_safety_cost.raw_cost),
            "training_cost": float(diagnostic_safety_cost.training_cost),
            "normalized_cost": float(diagnostic_safety_cost.normalized_cost),
            "training_cost_clip": float(diagnostic_safety_cost.training_cost_clip),
            "training_cost_clip_saturation": float(diagnostic_safety_cost.training_cost_clip_saturation),
            "dual_cost": float(diagnostic_safety_cost.dual_cost),
            "dual_violation_signal": float(diagnostic_safety_cost.dual_violation_signal),
            "over_processing_raw_cost": float(diagnostic_safety_cost.over_processing_raw_cost),
            "over_processing_normalized_cost": float(diagnostic_safety_cost.over_processing_normalized_cost),
            "over_processing_training_cost": float(diagnostic_safety_cost.over_processing_training_cost),
            "over_processing_clip_saturation": float(diagnostic_safety_cost.over_processing_clip_saturation),
            "backlog_excess_mb": float(diagnostic_safety_cost.backlog_excess_mb),
            "admission_excess_mb": float(diagnostic_safety_cost.admission_excess_mb),
            "clearable_capacity_mb": float(diagnostic_safety_cost.clearable_capacity_mb),
            "over_processing_ratio": float(diagnostic_safety_cost.over_processing_ratio),
            "raw_queue_overflow_mb": float(info["raw_queue_overflow_mb"]),
            "processed_queue_overflow_mb": float(info["processed_queue_overflow_mb"]),
            "expired_high_value": float(info.get("expired_high_value", 0.0)),
            "dropped_high_value": float(info.get("dropped_high_value", 0.0)),
            "risk_stage_code": float(info["risk_stage_code"]),
            "thermal_safe": float(info["thermal_safe"]),
            "thermal_excess_c": float(breakdown.get("_thermal_excess_c", 0.0)),
            "power_constraint_safe": float(info["power_constraint_safe"]),
        }
        self.episode_reward += reward

        # ── credit-bucket 处理门：step 末按本步下传/处理更新 credit + 写诊断 ──
        _cg = SAFE_BUDGET_FALLBACK_CONFIG
        if _cg.get("enable_credit_gate", False):
            avg_cap = float(_cg.get("avg_contact_capacity_mb", 800.0))
            gain = float(_cg.get("credit_gain_per_downlink", _cg.get("target_proc_dl_ratio", 2.5)))
            max_credit = float(_cg.get("max_credit_factor", 3.0)) * avg_cap
            step_proc = float(info.get("processed_mb", 0.0))
            step_dl = float(info.get("delivered_mb", 0.0))
            self._cum_processed_mb = float(getattr(self, "_cum_processed_mb", 0.0)) + step_proc
            self._cum_downlinked_mb = float(getattr(self, "_cum_downlinked_mb", 0.0)) + step_dl
            credit = float(getattr(self, "_proc_credit", float(_cg.get("initial_credit_factor", 1.5)) * avg_cap))
            credit = min(max_credit, credit + gain * step_dl - step_proc)
            credit = max(-max_credit, credit)
            self._proc_credit = credit
            self._credit_sum = float(getattr(self, "_credit_sum", 0.0)) + credit
            self._credit_min = min(float(getattr(self, "_credit_min", float("inf"))), credit)
            self._credit_steps = int(getattr(self, "_credit_steps", 0)) + 1
            info["processing_credit"] = credit
            info["processing_credit_mean"] = self._credit_sum / max(self._credit_steps, 1)
            info["processing_credit_min"] = self._credit_min
            info["running_proc_dl"] = self._cum_processed_mb / max(self._cum_downlinked_mb, 1e-6)
            info["cumulative_processed_mb"] = self._cum_processed_mb
            info["cumulative_downlinked_mb"] = self._cum_downlinked_mb

        return self._get_observation(), reward, done, info

    def _classify_mission_stage(self, orbit_info: dict, batt_info: dict,
                                thermal_info: dict | None = None) -> tuple[str, int]:
        """
        四层任务风险状态 (与 ORBITAL_CONFIG 中的物理阈值对齐):
        normal  : h >= altitude_warning_km (200km) 且 SOC >= battery_min_soc (15%)
        warning : altitude_min_km (180km) <= h < altitude_warning_km (200km) 或 battery_crash_soc < SOC < battery_min_soc
        unsafe  : altitude_crash_km (120km) < h < altitude_min_km (180km)
        failure : h <= altitude_crash_km (120km) 或 SOC <= battery_crash_soc (5%)
        注意: overall_safe 使用 is_safe = h >= altitude_min_km (180km)，非 altitude_warning_km。
        """
        thermal_info = thermal_info or {}
        if (
            orbit_info.get("is_crashed", False)
            or batt_info.get("is_crashed", False)
            or thermal_info.get("is_crashed", False)
        ):
            return "failure", 3
        if str(thermal_info.get("safety_stage", "normal")) == "critical":
            return "failure", 3
        if str(orbit_info.get("safety_stage", "normal")) == "unsafe":
            return "unsafe", 2
        if str(thermal_info.get("safety_stage", "normal")) == "unsafe":
            return "unsafe", 2
        if (
            str(orbit_info.get("safety_stage", "normal")) == "warning"
            or str(batt_info.get("safety_stage", "normal")) == "warning"
            or str(thermal_info.get("safety_stage", "normal")) == "warning"
        ):
            return "warning", 1
        return "normal", 0

    @property
    def observation_features(self) -> tuple[str, ...]:
        """返回观测向量标签，便于可视化和论文表格保持同一套状态定义。"""
        return tuple(OBSERVATION_FEATURES)

    @property
    def available_power_w(self) -> float:
        """当前时刻可用于动作边界裁剪的总功率估计。"""
        p_solar = self.solar.output_power(self.orbit_sim.sunlit_fraction())
        return self._compute_available_power(p_solar)

    def _get_observation(self) -> np.ndarray:
        # 状态按 OBSERVATION_FEATURES 的固定顺序组织，调整特征时必须同步网络索引和测试。
        h_norm = (self.altitude_m - self._h_min) / (self._h_max - self._h_min)
        altitude_margin = np.clip((self.altitude_m - self._h_min) / 50e3, 0.0, 2.0)
        soc = self.battery.soc
        q_raw = self.data_queue.length / QUEUE_CONFIG["data_queue_max_mb"]
        _q_raw_max = max(float(QUEUE_CONFIG["data_queue_max_mb"]), 1e-6)
        q_raw_delta = float(np.clip(
            (self.data_queue.length - self.data_queue.prev_length) / _q_raw_max,
            -1.0, 1.0,
        ))
        q_energy = self.energy_queue.value / QUEUE_CONFIG["energy_queue_max"]
        q_orbit = self.orbit_queue.value / QUEUE_CONFIG["orbit_queue_max"]
        _dynamic_period = self.orbit_sim.period_at(self.altitude_m)
        sunlit_frac = self.orbit_sim.sunlit_fraction()
        # drag 观测要反映当前实时的日间隆起 + 风暴乘子，agent 才能据此感知/规划。
        drag = self.orbit_dyn.drag_force(
            self.altitude_m, diurnal_angle_rad=self._diurnal_angle_rad())
        drag_norm = float(np.clip(
            drag / (self._drag_ref_force_hmin + 1e-8),
            0.0,
            5.0,
        ))

        contact = self._contact or {}
        in_win = float(contact.get("in_window", False))
        time_to_next_window = float(contact.get("time_to_next_window_s", 5400.0))
        t_to_win = np.clip(
            time_to_next_window / max(self._time_to_next_window_norm_s, 1e-6),
            0.0,
            1.0,
        )
        next_window_in_range = float(
            bool(in_win) or time_to_next_window < 5400.0 - 1e-6
        )
        win_rem = np.clip(contact.get("window_remaining_s", 0) / self._max_window_time, 0.0, 1.0)
        q_proc = self.comm_queue.value / QUEUE_CONFIG.get("comm_queue_max", 200.0)
        tx_capacity_scale_mbps = float(QUEUE_CONFIG.get(
            "tx_capacity_norm_mbps",
            QUEUE_CONFIG.get("tx_downlink_rate_max_mbs", 25.0) * 8.0,
        ))
        tx_capacity_norm = np.clip(
            contact.get("max_capacity_mbps", 0.0) / max(tx_capacity_scale_mbps, 1e-6),
            0.0,
            2.0,
        )
        task_stats = self.task_tracker.topk_stats(self.step_count or 0)
        class_stats = self.task_tracker.class_stats(self.step_count or 0)
        # ── 从缓存读取（step() 已经算过这些 90×19 扫描函数了）────────────
        if self._cached_step == (self.step_count or 0):
            deadline_contact_stats = self._cached_deadline_contact_stats
            deliverability_features = self._cached_deliverability_features
        else:
            # 缓存未命中：reset() 后首次调用或测试中单独调用 _get_observation()
            contact_steps = time_to_next_window / max(float(self.dt), 1e-6)
            deadline_contact_stats = self.task_tracker.deadline_contact_stats(
                self.step_count or 0,
                0.0 if bool(in_win) else contact_steps,
            )
            deliverability_features = self.task_tracker.deliverability_features(
                self.step_count or 0,
                self._future_contact_capacity_bins(),
            )
        value_norm = max(float(TASK_CONFIG.get("value_norm", 500.0)), 1e-6)
        raw_max = max(float(QUEUE_CONFIG["data_queue_max_mb"]), 1e-6)
        proc_max = max(float(QUEUE_CONFIG.get("comm_queue_max", 200.0)), 1e-6)
        raw_high = np.clip(class_stats.get("raw_high_mb", 0.0) / raw_max, 0.0, 2.0)
        raw_mid = np.clip(class_stats.get("raw_medium_mb", 0.0) / raw_max, 0.0, 2.0)
        raw_low = np.clip(class_stats.get("raw_low_mb", 0.0) / raw_max, 0.0, 2.0)
        proc_high = np.clip(class_stats.get("processed_high_mb", 0.0) / proc_max, 0.0, 2.0)
        proc_mid = np.clip(class_stats.get("processed_medium_mb", 0.0) / proc_max, 0.0, 2.0)
        proc_low = np.clip(class_stats.get("processed_low_mb", 0.0) / proc_max, 0.0, 2.0)
        expiring_high = np.clip(
            class_stats.get("expiring_high_value", 0.0) / value_norm, 0.0, 2.0)
        expiring_mid = np.clip(
            class_stats.get("expiring_medium_value", 0.0) / value_norm, 0.0, 2.0)
        expiring_low = np.clip(
            class_stats.get("expiring_low_value", 0.0) / value_norm, 0.0, 2.0)
        expiring_value = np.clip(task_stats.get("expiring_value", 0.0) / value_norm, 0.0, 2.0)
        processed_value = np.clip(
            task_stats.get("processed_backlog_value", 0.0) / value_norm,
            0.0,
            2.0,
        )
        processed_future_contact_ratio = np.clip(
            self.comm_queue.value / max(
                min(self._future_contact_capacity_mb(), float(self.comm_queue.max_value)), 1e-6),
            0.0,
            2.0,
        )
        priority = np.clip(task_stats.get("top_task_priority", 0.0) / max(TASK_CONFIG.get("priority_max", 1.5), 1e-6), 0.0, 2.0)
        quality = np.clip(task_stats.get("top_task_quality", 0.0) / max(TASK_CONFIG.get("quality_max", 1.2), 1e-6), 0.0, 2.0)
        deadline_urgency = np.clip(task_stats.get("deadline_urgency", 0.0), 0.0, 1.0)
        prop_phase = (self.step_count % self.N_PROP_SMOOTH) / self.N_PROP_SMOOTH
        current_scene = self._scene_context_for_phase()
        lookahead_steps = int(TASK_CONFIG.get("scene_lookahead_steps", 6))
        upcoming_scene = self._scene_context_for_phase(lookahead_steps=lookahead_steps)
        current_scene_class = np.clip(current_scene.get("scene_class_code", 0.0), 0.0, 1.0)
        upcoming_intensity = self._normalized_scene_intensity(upcoming_scene)
        future_contact_capacity = self._last_future_contact_capacity_norm
        cpu_backpressure_ratio = np.clip(self._last_cpu_backpressure_ratio, 0.0, 1.0)
        thermal_margin = self._thermal_margin_norm()
        proc_high_deliverable = np.clip(
            deadline_contact_stats.get(
                "processed_high_next_window_deliverable_ratio", 0.0),
            0.0,
            1.0,
        )
        raw_high_deliverable = np.clip(
            deadline_contact_stats.get(
                "raw_high_next_window_deliverable_ratio", 0.0),
            0.0,
            1.0,
        )
        high_deadline_mismatch = np.clip(
            deadline_contact_stats.get("high_value_deadline_contact_mismatch", 0.0),
            0.0,
            1.0,
        )
        total_power_norm = np.clip(
            self._last_total_power_w / max(ENERGY_CONFIG.get("power_total_max_w", 120.0), 1e-6),
            0.0,
            2.0,
        )

        obs = np.array([
            h_norm, drag_norm, altitude_margin,
            soc, sunlit_frac, total_power_norm,
            in_win, t_to_win, win_rem, tx_capacity_norm,
            q_raw, q_proc,
            raw_high, raw_mid, raw_low,
            proc_high, proc_mid, proc_low,
            expiring_high, expiring_mid, expiring_low,
            expiring_value,
            processed_value, priority, quality, deadline_urgency,
            self.prev_action[0], self.prev_action[1], self.prev_action[2],
            q_energy, q_orbit, q_raw_delta, processed_future_contact_ratio,
            prop_phase, current_scene_class, upcoming_intensity,
            future_contact_capacity, cpu_backpressure_ratio,
            next_window_in_range,
            thermal_margin,
            proc_high_deliverable, raw_high_deliverable,
            high_deadline_mismatch,
            deliverability_features["capacity_bin_0_mb_norm"],
            deliverability_features["capacity_bin_0_time_norm"],
            deliverability_features["capacity_bin_1_mb_norm"],
            deliverability_features["capacity_bin_1_time_norm"],
            deliverability_features["capacity_bin_2_mb_norm"],
            deliverability_features["capacity_bin_2_time_norm"],
            deliverability_features["capacity_bin_3_mb_norm"],
            deliverability_features["capacity_bin_3_time_norm"],
            deliverability_features["capacity_bin_4_mb_norm"],
            deliverability_features["capacity_bin_4_time_norm"],
            deliverability_features["capacity_bin_5_mb_norm"],
            deliverability_features["capacity_bin_5_time_norm"],
            deliverability_features["capacity_bin_6_mb_norm"],
            deliverability_features["capacity_bin_6_time_norm"],
            deliverability_features["capacity_bin_7_mb_norm"],
            deliverability_features["capacity_bin_7_time_norm"],
            deliverability_features["concurrent_high_same_class_mb_norm"],
            deliverability_features["concurrent_medium_same_class_mb_norm"],
            deliverability_features["concurrent_low_same_class_mb_norm"],
            float(getattr(self, "propellant_kg", 0.0)
                  / max(getattr(self, "_propellant_initial_kg", 1.0), 1e-9)),
            float(getattr(self, "_pointing_mode", POINTING_SUN)) / float(max(POINTING_SUN, 1)),
            float(getattr(self, "_momentum", 0.0)
                  / max(float(ATTITUDE_CONFIG.get("momentum_max", 1.0)), 1e-9)),
        ], dtype=np.float32)
        return np.clip(obs, self._obs_low, self._obs_high)

    def _apply_active_low_value_drop(self, drop_strength: float,
                                     class_stats: dict) -> dict:
        """按综合压力触发主动丢弃，避免仅在高队列占用时才生效。"""
        strength = float(np.clip(drop_strength, 0.0, 1.0))
        if strength <= 1e-9:
            return self.task_tracker.drop_low_value(0.0, self.step_count, {})

        future_capacity_mb = self._future_contact_capacity_mb()
        
        raw_high_mb = float(class_stats.get("raw_high_mb", 0.0))
        proc_high_mb = float(class_stats.get("processed_high_mb", 0.0))
        raw_mid_mb = float(class_stats.get("raw_medium_mb", 0.0))
        proc_mid_mb = float(class_stats.get("processed_medium_mb", 0.0))
        raw_low_mb = float(class_stats.get("raw_low_mb", 0.0))
        proc_low_mb = float(class_stats.get("processed_low_mb", 0.0))
        
        # RAW_TO_PROCESSED_RATIO < 1 时 raw_xxx_mb（未压缩）与 proc_xxx_mb（已压缩）
        # 量纲不同：future_capacity_mb 是 RF product MB（压缩后），protected_demand_mb
        # 必须也用压缩后 MB，否则 low_capacity_slack_mb 永远为 0（raw_high 远大于 RF 容量）。
        _cmp_ratio = float(TASK_CONFIG.get("RAW_TO_PROCESSED_RATIO", 1.0))
        _inv_ratio = 1.0 / max(_cmp_ratio, 1e-9)
        # low_share 使用"压缩后等效 MB"统一量纲（raw × ratio + proc）
        low_backlog_comp_mb = raw_low_mb * _cmp_ratio + proc_low_mb
        total_backlog_comp_mb = max(
            (raw_high_mb + raw_mid_mb + raw_low_mb) * _cmp_ratio
            + proc_high_mb + proc_mid_mb + proc_low_mb,
            1e-9,
        )
        low_share = low_backlog_comp_mb / total_backlog_comp_mb
        expected_processing_ratio = float(TASK_CONFIG.get("low_drop_expected_processing_ratio", 0.6))
        mid_protection_ratio = float(TASK_CONFIG.get("low_drop_mid_protection_ratio", 0.35))

        # protected_demand_mb 与 future_capacity_mb 统一为压缩后 MB
        protected_demand_mb = (
            proc_high_mb
            + raw_high_mb * _cmp_ratio * expected_processing_ratio
            + mid_protection_ratio * (proc_mid_mb + raw_mid_mb * _cmp_ratio * expected_processing_ratio)
        )

        low_capacity_slack_mb = max(0.0, future_capacity_mb - protected_demand_mb)
        # 转回 raw MB，供后续与 droppable_backlog_mb（raw MB）比较
        low_capacity_slack_raw_mb = low_capacity_slack_mb * _inv_ratio

        raw_util = self.data_queue.length / max(self.data_queue.max_length, 1e-6)
        proc_util = self.comm_queue.value / max(self.comm_queue.max_value, 1e-6)
        queue_pressure = max(raw_util, proc_util)

        # 预估未来容量缺口（raw MB 口径，与 droppable_backlog_mb 一致）
        low_backlog_raw_mb = raw_low_mb + proc_low_mb * _inv_ratio
        static_low_excess_mb = max(0.0, low_backlog_raw_mb - low_capacity_slack_raw_mb)
        future_contact_shortage = static_low_excess_mb / max(low_backlog_raw_mb, 1e-9)

        # 综合资源压力（队列满载 or 容量告急）
        resource_pressure = float(np.clip(max(queue_pressure, future_contact_shortage), 0.0, 1.0))

        # 使用综合资源压力，圈定【真正允许丢弃】的动态任务集合（仅纯 Low）
        droppable_stats = self.task_tracker.droppable_backlog(
            self.step_count,
            {"resource_pressure": resource_pressure},
        )
        droppable_backlog_mb = droppable_stats["droppable_backlog_mb"]

        # 基于真实可丢弃集合（raw MB），重新计算实际容量缺口（raw MB 口径）
        low_excess_mb = max(0.0, droppable_backlog_mb - low_capacity_slack_raw_mb)
        
        capacity_driven_drop_mb = low_excess_mb
        queue_pressure_threshold = float(
            TASK_CONFIG.get("low_drop_resource_pressure_threshold", 0.03))
        queue_driven_drop_mb = droppable_backlog_mb * max(
            0.0, queue_pressure - queue_pressure_threshold)
        low_share_target = float(TASK_CONFIG.get("low_drop_share_target", 0.05))
        low_share_driven_drop_mb = droppable_backlog_mb * max(
            0.0, low_share - low_share_target)
        policy_driven_drop_mb = droppable_backlog_mb * float(
            TASK_CONFIG.get("active_low_drop_floor_ratio", 0.05))
        target_drop_mb = strength * max(
            capacity_driven_drop_mb,
            queue_driven_drop_mb,
            low_share_driven_drop_mb,
            policy_driven_drop_mb,
        )
        
        max_drop_mb = float(TASK_CONFIG.get("low_value_drop_max_mbs", 8.0)) * float(self.dt)
        drop_mb = min(droppable_backlog_mb, target_drop_mb, max_drop_mb)
        
        drop_context = {
            "future_capacity_mb": future_capacity_mb,
            "protected_demand_mb": protected_demand_mb,
            "droppable_backlog_mb": droppable_backlog_mb,
            "low_capacity_slack_mb": low_capacity_slack_mb,
            "low_excess_mb": low_excess_mb,
            "future_contact_shortage": future_contact_shortage,
            "resource_pressure": resource_pressure,
            "capacity_driven_drop_mb": capacity_driven_drop_mb,
            "queue_driven_drop_mb": queue_driven_drop_mb,
            "low_share": low_share,
            "low_share_target": low_share_target,
            "low_share_driven_drop_mb": low_share_driven_drop_mb,
            "policy_driven_drop_mb": policy_driven_drop_mb,
        }
        
        out = self.task_tracker.drop_low_value(drop_mb, self.step_count, drop_context)
        out.update(drop_context)
        out["active_drop_budget_mb"] = float(drop_mb)
        return out

    def _compute_available_power(self, p_solar_w: float) -> float:
        """
        估算本时隙可用总功率 P_available。

        太阳能输入可直接使用；电池只允许动用高于 SOC_min 的安全裕度，避免为了短时吞吐
        透支安全底线。最终再受电源管理器 power_total_max_w 限制。
        """
        dt_h = max(self.dt / 3600.0, 1e-9)
        baseline_w = float(ENERGY_CONFIG.get("power_baseline_w", 0.0))
        reserve_soc = max(0.0, float(ENERGY_CONFIG.get("battery_operational_reserve_soc", 0.0)))
        operational_floor_soc = float(self.battery.soc_min) + reserve_soc
        operational_margin_wh = (
            (float(self.battery.soc) - operational_floor_soc)
            * float(self.battery.capacity_wh)
        )
        safe_battery_wh = max(float(operational_margin_wh), 0.0)
        battery_burst_w = safe_battery_wh * self.battery.eta_discharge / dt_h
        total_limit_w = float(ENERGY_CONFIG.get("power_total_max_w", 120.0))
        available_w = max(baseline_w, float(p_solar_w) + battery_burst_w)
        return float(np.clip(available_w, 0.0, total_limit_w))

    def _analytic_propulsion_alpha(self) -> dict:
        """根据高度、阻力和 SOC 计算推进器解析控制量。"""
        raw_cfg = PROPULSION_CONTROLLER_CONFIG
        enabled = bool(raw_cfg.get("enabled", False))
        prop_max_w = max(1e-6, float(ENERGY_CONFIG["power_propulsion_max_w"]))
        ignition_alpha = float(np.clip(
            raw_cfg.get(
                "min_ignited_alpha",
                float(ENERGY_CONFIG.get("propulsion_ignition_threshold_w", 0.0)) / prop_max_w,
            ),
            0.0,
            1.0,
        ))
        if not enabled or self.altitude_m is None:
            return {
                "enabled": bool(enabled),
                "alpha": 0.0,
                "hover_power_w": 0.0,
                "hover_alpha": 0.0,
                "reason": "disabled" if not enabled else "altitude_uninitialized",
                "min_ignited_alpha": ignition_alpha,
            }

        altitude_km = float(self.altitude_m) / 1e3
        target_km = float(raw_cfg.get(
            "target_altitude_km",
            ORBITAL_CONFIG.get("altitude_nominal_km", 250.0),
        ))
        warning_full_km = float(raw_cfg.get(
            "warning_full_power_km",
            ORBITAL_CONFIG.get("altitude_warning_km", 200.0),
        ))
        min_full_km = float(raw_cfg.get(
            "min_altitude_full_power_km",
            ORBITAL_CONFIG.get("altitude_min_km", 180.0),
        ))
        coast_above_km = float(raw_cfg.get("coast_above_km", target_km + 20.0))
        max_alpha = float(np.clip(raw_cfg.get("max_alpha", 1.0), 0.0, 1.0))

        try:
            drag_force_n = float(self.orbit_dyn.drag_force(
                float(self.altitude_m),
                diurnal_angle_rad=self._diurnal_angle_rad(),
            ))
            isp_s = max(1e-6, float(ENERGY_CONFIG.get("propulsion_isp_s", 1500.0)))
            efficiency = max(1e-6, float(ENERGY_CONFIG.get("propulsion_efficiency", 0.65)))
            hover_power_w = max(0.0, drag_force_n * isp_s * 9.80665 / efficiency)
        except Exception:
            hover_power_w = 0.0

        hover_alpha = float(np.clip(
            float(raw_cfg.get("hover_margin", 1.15)) * hover_power_w / prop_max_w,
            0.0,
            max_alpha,
        ))

        if altitude_km <= min_full_km:
            alpha = max_alpha
            reason = "min_altitude_full_power"
        elif altitude_km <= warning_full_km:
            alpha = max_alpha
            reason = "warning_full_power"
        elif altitude_km >= coast_above_km:
            alpha = 0.0
            reason = "coast_above_band"
        elif altitude_km < target_km:
            recovery_span = max(target_km - warning_full_km, 1e-6)
            recovery_alpha = max_alpha * np.clip(
                (target_km - altitude_km) / recovery_span,
                0.0,
                1.0,
            )
            alpha = max(hover_alpha, float(recovery_alpha))
            reason = "recover_to_target"
        else:
            taper_span = max(coast_above_km - target_km, 1e-6)
            taper = np.clip((coast_above_km - altitude_km) / taper_span, 0.0, 1.0)
            alpha = hover_alpha * float(taper)
            reason = "hover_taper"

        try:
            sunlit_fraction = float(self.orbit_sim.sunlit_fraction())
        except Exception:
            sunlit_fraction = 1.0
        if 0.0 < alpha < max_alpha and altitude_km > warning_full_km:
            sunlit_fraction = float(np.clip(sunlit_fraction, 0.0, 1.0))
            sun_scale = float(raw_cfg.get("sunlit_power_scale", 1.0))
            eclipse_scale = float(raw_cfg.get("eclipse_power_scale", 0.75))
            alpha *= sun_scale * sunlit_fraction + eclipse_scale * (1.0 - sunlit_fraction)

        soc = float(getattr(self.battery, "soc", 1.0))
        if altitude_km > warning_full_km:
            critical_soc = float(raw_cfg.get("critical_soc_threshold", 0.12))
            low_soc = float(raw_cfg.get("low_soc_threshold", 0.18))
            if soc <= critical_soc:
                alpha *= float(raw_cfg.get("critical_soc_power_scale", 0.0))
                reason = f"{reason}_critical_soc"
            elif soc <= low_soc:
                alpha *= float(raw_cfg.get("low_soc_power_scale", 0.60))
                reason = f"{reason}_low_soc"

        alpha = float(np.clip(alpha, 0.0, max_alpha))
        # Hall 推进器有点火门限：低于门限的非零命令不会产生推力。
        # 低于目标高度时把小命令抬到门限；目标以上则直接滑行，避免无效耗电。
        if 0.0 < alpha < ignition_alpha:
            if altitude_km < target_km:
                alpha = ignition_alpha
                reason = f"{reason}_ignition_floor"
            else:
                alpha = 0.0
                reason = f"{reason}_deadband_coast"

        return {
            "enabled": True,
            "alpha": float(alpha),
            "hover_power_w": float(hover_power_w),
            "hover_alpha": float(hover_alpha),
            "reason": reason,
            "min_ignited_alpha": float(ignition_alpha),
        }

    def _apply_analytic_propulsion_controller(self, action: np.ndarray) -> tuple[np.ndarray, dict]:
        """按配置应用解析推进兜底。

        guard_only=True 时，健康状态下保留 agent 的推进动作；只有轨道或 SOC 接近
        硬安全边界时才允许解析控制器接管，避免训练后的策略被环境长期改写。
        """
        controlled = np.asarray(action, dtype=np.float64).copy()
        raw_alpha = float(np.clip(controlled[0], 0.0, 1.0))
        controller = self._analytic_propulsion_alpha()
        enabled = bool(controller.get("enabled", False))
        guard_only = bool(PROPULSION_CONTROLLER_CONFIG.get("guard_only", True))
        altitude_guard = False
        soc_guard = False
        if enabled and guard_only:
            altitude_margin_m = max(
                0.0,
                float(PROPULSION_CONTROLLER_CONFIG.get("guard_altitude_margin_km", 5.0)) * 1e3,
            )
            soc_margin = max(
                0.0,
                float(PROPULSION_CONTROLLER_CONFIG.get("guard_soc_margin", 0.02)),
            )
            altitude_guard = bool(
                self.altitude_m is not None
                and float(self.altitude_m) <= float(self._h_min) + altitude_margin_m
            )
            soc_guard = bool(
                float(getattr(self.battery, "soc", 1.0))
                <= float(getattr(self.battery, "soc_min", 0.0)) + soc_margin
            )
            if not (altitude_guard or soc_guard):
                controller = dict(controller)
                controller["enabled"] = False
                controller["reason"] = "guard_not_active"
                enabled = False
        analytic_alpha = raw_alpha
        if enabled:
            analytic_alpha = float(np.clip(controller.get("alpha", raw_alpha), 0.0, 1.0))
            controlled[0] = analytic_alpha
        return controlled, {
            "analytic_propulsion_controller_enabled": enabled,
            "analytic_propulsion_applied": bool(enabled and abs(analytic_alpha - raw_alpha) > 1e-9),
            "analytic_propulsion_guard_only": bool(guard_only),
            "analytic_propulsion_guard_active": bool(altitude_guard or soc_guard),
            "analytic_propulsion_altitude_guard": bool(altitude_guard),
            "analytic_propulsion_soc_guard": bool(soc_guard),
            "raw_alpha_prop": float(raw_alpha),
            "analytic_alpha_prop": float(analytic_alpha),
            "analytic_propulsion_hover_power_w": float(controller.get("hover_power_w", 0.0)),
            "analytic_propulsion_hover_alpha": float(controller.get("hover_alpha", 0.0)),
            "analytic_propulsion_min_ignited_alpha": float(controller.get("min_ignited_alpha", 0.0)),
            "analytic_propulsion_reason": str(controller.get("reason", "disabled")),
        }

    def _admissible_cpu_budget_mb(self) -> dict:
        # admissible = margin * effective_cap - processed_queue
        # Use the same near-term cap (2 × max_pass = 1600 MB) as _reward_config_for_step so
        # the gate fires at the same queue level where deliver_prob starts dropping.
        # Previously capping at queue_max (4096 MB) meant the gate only bit at ~3891 MB,
        # but the reward was already penalising at 1600 MB — the two signals were inconsistent
        # and the agent never received a gate signal until the queue was completely saturated.
        # Current behavior is controlled by cpu_gate_near_term_passes (default: one pass).
        future_capacity_mb = float(self._future_contact_capacity_mb())
        near_term_passes = max(0.0, float(TASK_CONFIG.get("cpu_gate_near_term_passes", 1.0)))
        near_term_dl_cap_mb = near_term_passes * float(
            GROUND_STATION_CONFIG.get("max_downlink_mb_per_pass", 800.0)
        )
        if near_term_dl_cap_mb <= 0.0:
            near_term_dl_cap_mb = float(self.comm_queue.max_value)
        effective_cap_mb = min(future_capacity_mb, near_term_dl_cap_mb, float(self.comm_queue.max_value))
        admissible_mb = max(
            0.0,
            float(TASK_CONFIG.get("deliverability_capacity_margin", 0.95)) * effective_cap_mb
            - float(self.comm_queue.value),
        )
        return {
            "admissible_cpu_mb": float(admissible_mb),
            "future_capacity_mb": float(future_capacity_mb),
            "effective_future_capacity_mb": float(effective_cap_mb),
            "reserved_raw_mb": 0.0,
        }

    def _high_value_cpu_gate_escape_budget_mb(
        self,
        *,
        future_capacity_mb: float,
        in_window: bool,
        time_to_next_window_s: float,
        dt_s: float,
        service_rate_max_mbs: float,
    ) -> dict:
        """Allow deliverable raw high-value data to bypass the total-buffer CPU gate."""
        meta = {
            "enabled": bool(TASK_CONFIG.get("enable_high_value_cpu_gate_escape", False)),
            "budget_mb": 0.0,
            "raw_high_mb": 0.0,
            "processed_high_mb": 0.0,
            "raw_high_deliverable_ratio": 0.0,
            "deadline_mismatch": 0.0,
            "high_capacity_headroom_mb": 0.0,
            "near_window": False,
        }
        if not meta["enabled"]:
            return meta

        class_stats = self.task_tracker.class_stats(self.step_count)
        raw_high_mb = max(0.0, float(class_stats.get("raw_high_mb", 0.0)))
        processed_high_mb = max(0.0, float(class_stats.get("processed_high_mb", 0.0)))
        meta["raw_high_mb"] = raw_high_mb
        meta["processed_high_mb"] = processed_high_mb

        min_raw_mb = max(0.0, float(TASK_CONFIG.get("high_value_cpu_escape_min_raw_mb", 1.0)))
        if raw_high_mb <= min_raw_mb:
            return meta

        lead_s = max(0.0, float(TASK_CONFIG.get("high_value_cpu_escape_lead_s", 900.0)))
        near_window = bool(in_window or float(time_to_next_window_s) <= lead_s)
        meta["near_window"] = near_window
        if not near_window:
            return meta

        steps_to_next_window = 0.0 if in_window else float(time_to_next_window_s) / max(float(dt_s), 1e-6)
        deadline_stats = self.task_tracker.deadline_contact_stats(
            self.step_count,
            steps_to_next_window,
        )
        raw_deliverable_ratio = float(np.clip(
            deadline_stats.get("raw_high_next_window_deliverable_ratio", 0.0),
            0.0,
            1.0,
        ))
        mismatch = float(np.clip(
            deadline_stats.get("high_value_deadline_contact_mismatch", 0.0),
            0.0,
            1.0,
        ))
        meta["raw_high_deliverable_ratio"] = raw_deliverable_ratio
        meta["deadline_mismatch"] = mismatch

        min_deliverable_ratio = float(TASK_CONFIG.get(
            "high_value_cpu_escape_min_deliverable_ratio", 0.5))
        max_mismatch = float(TASK_CONFIG.get("high_value_cpu_escape_max_mismatch", 0.4))
        if raw_deliverable_ratio < min_deliverable_ratio or mismatch > max_mismatch:
            return meta

        capacity_fraction = float(np.clip(
            TASK_CONFIG.get("high_value_cpu_escape_capacity_fraction", 1.0),
            0.0,
            1.0,
        ))
        capacity_margin = float(np.clip(
            TASK_CONFIG.get("high_value_cpu_escape_capacity_margin", 0.95),
            0.0,
            1.0,
        ))
        high_capacity_mb = max(0.0, float(future_capacity_mb) * capacity_fraction * capacity_margin)
        high_headroom_mb = max(0.0, high_capacity_mb - processed_high_mb)
        meta["high_capacity_headroom_mb"] = high_headroom_mb
        if high_headroom_mb <= 1e-9:
            return meta

        raw_deliverable_mb = max(0.0, float(
            deadline_stats.get("raw_high_next_window_deliverable_mb", raw_high_mb)
        ))
        physical_step_mb = max(0.0, float(service_rate_max_mbs) * max(0.0, float(dt_s)))
        budget_mb = min(raw_high_mb, raw_deliverable_mb, high_headroom_mb, physical_step_mb)
        meta["budget_mb"] = float(max(0.0, budget_mb))
        return meta

    def _apply_future_contact_cpu_gate(
        self,
        action: np.ndarray,
        *,
        in_window: bool,
        time_to_next_window_s: float,
        dt_s: float,
    ) -> tuple[np.ndarray, dict]:
        gated = np.asarray(action, dtype=np.float64).copy()
        original = gated.copy()
        alpha_before = float(np.clip(gated[1], 0.0, 1.0))
        gated[1] = alpha_before
        dt = max(0.0, float(dt_s))
        service_rate_max = max(0.0, float(QUEUE_CONFIG.get(
            "data_service_rate_max_mbs",
            QUEUE_CONFIG.get("data_service_rate_max_mbps", 5.0),
        )))
        budget_meta = self._admissible_cpu_budget_mb()
        raw_future_capacity_mb = max(0.0, float(budget_meta.get("future_capacity_mb", 0.0)))
        future_capacity_mb = max(0.0, float(
            budget_meta.get("effective_future_capacity_mb", raw_future_capacity_mb)
        ))
        processed_queue_mb = max(0.0, float(self.comm_queue.value))
        ratio_before = processed_queue_mb / max(future_capacity_mb, 1e-6)
        admissible_mode = bool(TASK_CONFIG.get("cpu_action_is_admissible_budget", False))
        admissible_mb = float(budget_meta.get("admissible_cpu_mb", 0.0))
        high_escape_meta = self._high_value_cpu_gate_escape_budget_mb(
            future_capacity_mb=future_capacity_mb,
            in_window=in_window,
            time_to_next_window_s=time_to_next_window_s,
            dt_s=dt,
            service_rate_max_mbs=service_rate_max,
        )
        high_escape_budget_mb = float(high_escape_meta.get("budget_mb", 0.0))
        admissible_with_escape_mb = max(admissible_mb, high_escape_budget_mb)
        if admissible_mode:
            requested_processed_mb = alpha_before * max(0.0, admissible_with_escape_mb)
        else:
            requested_processed_mb = alpha_before * service_rate_max * dt
        allowed_processed_mb = requested_processed_mb
        force_zero_cpu_power = False
        high_escape_applied = False
        enabled = bool(TASK_CONFIG.get(
            "enable_future_contact_cpu_gate",
            TASK_CONFIG.get("enable_cpu_throttle", True),
        ))

        if enabled and admissible_mode and admissible_with_escape_mb <= 1e-9 and alpha_before > 1e-9:
            # admissible=0 表示本步没有任何可交付处理额度。若继续保留 alpha_cpu，
            # 计算功耗会按满 CPU 结算，但实际 processed_mb=0，正是 proc>>dl 的无用耗电。
            allowed_processed_mb = 0.0
            force_zero_cpu_power = True
        elif enabled and requested_processed_mb > 1e-9:
            start_ratio = max(0.0, float(TASK_CONFIG.get("cpu_gate_start_future_ratio", 0.55)))
            target_ratio = max(start_ratio, float(TASK_CONFIG.get("cpu_gate_target_future_ratio", 0.75)))
            far_target_ratio = float(np.clip(
                TASK_CONFIG.get("cpu_gate_far_window_target_ratio", start_ratio),
                0.0,
                target_ratio,
            ))
            hard_stop_ratio = max(target_ratio, float(TASK_CONFIG.get("cpu_gate_hard_stop_future_ratio", 0.90)))
            far_window_lead_s = max(0.0, float(TASK_CONFIG.get("cpu_gate_far_window_lead_s", 120.0)))
            far_from_window = bool(
                not in_window and float(time_to_next_window_s) > far_window_lead_s)
            if future_capacity_mb <= 1e-6 and not in_window:
                allowed_processed_mb = 0.0
                force_zero_cpu_power = True
            elif future_capacity_mb > 1e-6:
                effective_target_ratio = far_target_ratio if far_from_window else target_ratio
                if ratio_before >= hard_stop_ratio:
                    allowed_processed_mb = 0.0
                    force_zero_cpu_power = True
                elif ratio_before >= start_ratio or far_from_window:
                    allowed_processed_mb = max(
                        0.0,
                        effective_target_ratio * future_capacity_mb - processed_queue_mb,
                    )
                    allowed_processed_mb = min(allowed_processed_mb, requested_processed_mb)
                    force_zero_cpu_power = bool(
                        allowed_processed_mb <= 1e-9 and alpha_before > 1e-9
                    )

        # ── soft mode：不改写动作，只把"本应被截掉的 MB"作为 violation 暴露给 reward。
        # 旧 hard mode：直接缩放 alpha_cpu。这是导致 agent 永远学不到"远窗口少处理"
        # 的根因 —— 它的不安全动作每次都被 gate 悄悄修正，policy gradient 收不到信号。
        high_escape_requested_mb = min(requested_processed_mb, high_escape_budget_mb)
        if enabled and high_escape_requested_mb > allowed_processed_mb + 1e-9:
            allowed_processed_mb = high_escape_requested_mb
            force_zero_cpu_power = False
            high_escape_applied = True

        soft_mode = bool(ACTUATOR_GATE_CONFIG.get("cpu_gate_soft_mode", False))
        violation_mb = max(0.0, requested_processed_mb - allowed_processed_mb)
        alpha_after = alpha_before
        modified = False
        if (
            (requested_processed_mb > allowed_processed_mb + 1e-9 or force_zero_cpu_power)
            and not soft_mode
        ):
            if force_zero_cpu_power:
                cpu_scale = 0.0
            else:
                cpu_scale = float(np.clip(
                    allowed_processed_mb / max(requested_processed_mb, 1e-9),
                    0.0,
                    1.0,
                ))
            floor_alpha = float(np.clip(TASK_CONFIG.get("cpu_gate_floor_alpha", 0.0), 0.0, 1.0))
            if alpha_before > 1e-9:
                alpha_after = min(alpha_before, max(floor_alpha, alpha_before * cpu_scale))
            else:
                alpha_after = 0.0
            gated[1] = float(np.clip(alpha_after, 0.0, 1.0))
            modified = True

        if admissible_mode:
            if enabled and not soft_mode:
                effective_processed_budget_mb = min(
                    max(0.0, allowed_processed_mb),
                    max(0.0, requested_processed_mb),
                    max(0.0, float(getattr(self.data_queue, "length", 0.0))),
                )
                physical_max_mb = max(1e-9, service_rate_max * dt)
                power_sized_alpha = float(np.clip(
                    effective_processed_budget_mb / physical_max_mb,
                    0.0,
                    1.0,
                ))
                if abs(power_sized_alpha - float(gated[1])) > 1e-9:
                    gated[1] = power_sized_alpha
                    alpha_after = power_sized_alpha
                    modified = True
            else:
                effective_processed_budget_mb = max(0.0, requested_processed_mb)
            requested_after_mb = effective_processed_budget_mb
        else:
            effective_processed_budget_mb = gated[1] * service_rate_max * dt
            requested_after_mb = gated[1] * service_rate_max * dt
        ratio_after_est = (processed_queue_mb + requested_after_mb) / max(future_capacity_mb, 1e-6)
        applied = bool(modified and abs(float(gated[1]) - alpha_before) > 1e-9)
        meta = {
            "future_contact_cpu_gate_applied": bool(applied),
            "cpu_gate_soft_mode": bool(soft_mode),
            "cpu_gate_violation_mb": float(violation_mb),
            "cpu_gate_ratio_before": float(ratio_before),
            "cpu_gate_ratio_after_est": float(ratio_after_est),
            "cpu_gate_requested_processed_mb": float(requested_processed_mb),
            "cpu_gate_allowed_processed_mb": float(allowed_processed_mb),
            "cpu_gate_effective_processed_budget_mb": float(effective_processed_budget_mb),
            "cpu_gate_alpha_cpu_before": float(alpha_before),
            "cpu_gate_alpha_cpu_after": float(gated[1]),
            "cpu_gate_mod_l2": float(np.linalg.norm(gated - original)),
            "cpu_gate_future_contact_capacity_mb": float(future_capacity_mb),
            "cpu_gate_raw_future_contact_capacity_mb": float(raw_future_capacity_mb),
            "cpu_gate_processed_queue_mb": float(processed_queue_mb),
            "cpu_gate_admissible_cpu_mb": float(admissible_mb),
            "cpu_gate_admissible_with_high_escape_mb": float(admissible_with_escape_mb),
            "cpu_gate_reserved_raw_mb": float(budget_meta.get("reserved_raw_mb", 0.0)),
            "cpu_gate_high_value_escape_enabled": bool(high_escape_meta.get("enabled", False)),
            "cpu_gate_high_value_escape_applied": bool(high_escape_applied),
            "cpu_gate_high_value_escape_budget_mb": float(high_escape_budget_mb),
            "cpu_gate_high_value_escape_raw_mb": float(high_escape_meta.get("raw_high_mb", 0.0)),
            "cpu_gate_high_value_escape_processed_mb": float(
                high_escape_meta.get("processed_high_mb", 0.0)),
            "cpu_gate_high_value_escape_deliverable_ratio": float(
                high_escape_meta.get("raw_high_deliverable_ratio", 0.0)),
            "cpu_gate_high_value_escape_mismatch": float(
                high_escape_meta.get("deadline_mismatch", 0.0)),
            "cpu_gate_high_value_escape_headroom_mb": float(
                high_escape_meta.get("high_capacity_headroom_mb", 0.0)),
            "cpu_gate_high_value_escape_near_window": bool(
                high_escape_meta.get("near_window", False)),
        }
        return gated.astype(np.float64), meta

    def _enforce_available_power(self, action: np.ndarray,
                                 available_power_w: float) -> tuple[np.ndarray, dict]:
        """
        对环境最终执行动作做动态功率闭环。

        调度器只能看到 step() 前的可用功率；环境兜底、基线动作或手工脚本
        都可能带来越界动作。因此环境必须在真正计算功耗前再兜底一次，保证 replay buffer 里的
        executed_action 与实际功耗、约束统计是同一个动作。
        """
        baseline_w = float(ENERGY_CONFIG["power_baseline_w"])
        in_window = bool((self._contact or {}).get("in_window", False))
        orbit_needs_recovery = bool(self.altitude_m is not None and self.altitude_m <= self._h_warning)
        allocation = self.actuator_filter.apply_power_boundary(
            action,
            available_power_w=available_power_w,
            in_window=in_window,
            force_prop_priority=orbit_needs_recovery,
            dtype=np.float64,
        )
        clipped = allocation.action.astype(np.float64, copy=False)
        meta = allocation.meta

        # 保持 float64 传给功率模型，避免 float32 舍入把刚好贴边的功率又推过预算。
        return clipped.astype(np.float64), {
            "power_execution_clipped": bool(
                meta.get("action_bound_clipped", False)
                or meta.get("power_clipped", False)
                or meta.get("propulsion_deadband_applied", False)
                or meta.get("propulsion_ignition_boost_applied", False)
            ),
            "power_action_scale": float(meta.get("power_action_scale", 1.0)),
            "power_clip_mode": str(meta.get("power_clip_mode", "strict_priority")),
            "propulsion_deadband_applied": bool(meta.get("propulsion_deadband_applied", False)),
            "propulsion_ignition_boost_applied": bool(
                meta.get("propulsion_ignition_boost_applied", False)),
            "propulsion_ignition_threshold_w": float(
                meta.get("propulsion_ignition_threshold_w", 0.0)),
            "power_priority_order": str(meta.get("power_priority_order", "prop>cpu>tx")),
            "raw_action_finite": bool(meta.get("raw_action_finite", True)),
            "action_bound_clipped": bool(meta.get("action_bound_clipped", False)),
            "requested_adjustable_power_w": float(meta.get("requested_adjustable_power_w", 0.0)),
            "executed_adjustable_power_w": float(meta.get("executed_adjustable_power_w", 0.0)),
            "requested_total_power_w": float(meta.get("requested_total_power_w", baseline_w)),
            "executed_total_power_w": float(meta.get("executed_total_power_w", baseline_w)),
        }

    def _thermal_margin_norm(self) -> float:
        """热安全裕度归一化：1 表示接近初始冷态，0 表示达到热安全上限。"""
        warning = float(THERMAL_CONFIG.get("warning_temp_c", 45.0))
        max_temp = float(THERMAL_CONFIG.get("max_temp_c", 55.0))
        initial = float(THERMAL_CONFIG.get("initial_temp_c", 20.0))
        span = max(max_temp - initial, 1e-6)
        margin = (max_temp - float(self.thermal_temperature_c)) / span
        # warning 以上仍保留连续信号，方便策略提前学习热裕度衰减。
        return float(np.clip(margin, -1.0 if self.thermal_temperature_c > warning else 0.0, 1.0))

    def _update_thermal_state(self, total_power_w: float,
                              sunlit_fraction: float,
                              propulsion_power_w: float | None = None,
                              cpu_power_w: float | None = None,
                              tx_power_w: float | None = None) -> dict:
        """一阶热状态：内部耗散、太阳吸收和辐射散热共同作用。"""
        if not bool(THERMAL_CONFIG.get("enabled", True)):
            return {
                "temperature_c": float(self.thermal_temperature_c),
                "thermal_margin_norm": 1.0,
                "is_safe": True,
                "is_warning": False,
                "safety_stage": "disabled",
            }

        ambient = float(THERMAL_CONFIG.get("ambient_temp_c", -20.0))
        thermal_capacity = max(
            float(THERMAL_CONFIG.get("thermal_capacity_j_per_k", 18000.0)),
            1e-6,
        )
        electronics_heat_fraction = float(
            np.clip(THERMAL_CONFIG.get("electronics_heat_fraction", 0.35), 0.0, 1.0)
        )
        propulsion_heat_fraction = float(
            np.clip(THERMAL_CONFIG.get("propulsion_heat_fraction", 0.04), 0.0, 1.0)
        )
        sunlit_absorbing_area_m2 = max(
            float(THERMAL_CONFIG.get("sunlit_absorbing_area_m2", 0.08)),
            0.0,
        )
        solar_absorptivity = float(
            np.clip(THERMAL_CONFIG.get("solar_absorptivity", 0.20), 0.0, 1.0)
        )
        radiator_area_m2 = max(
            float(THERMAL_CONFIG.get("radiator_area_m2", 0.18)),
            0.0,
        )
        radiator_emissivity = float(
            np.clip(THERMAL_CONFIG.get("radiator_emissivity", 0.82), 0.0, 1.0)
        )
        solar_flux_w_m2 = max(float(THERMAL_CONFIG.get("solar_flux_w_m2", 1361.0)), 0.0)
        sigma_sb = 5.670374419e-8

        if (
            propulsion_power_w is None
            and cpu_power_w is None
            and tx_power_w is None
        ):
            # 兼容旧单元测试/外部调用：未提供分量时仍按旧口径估算舱内热。
            electronics_thermal_power_w = max(float(total_power_w), 0.0)
            propulsion_thermal_power_w = 0.0
        else:
            propulsion_thermal_power_w = max(float(propulsion_power_w or 0.0), 0.0)
            cpu_thermal_power_w = max(float(cpu_power_w or 0.0), 0.0)
            tx_thermal_power_w = max(float(tx_power_w or 0.0), 0.0)
            baseline_thermal_power_w = max(
                float(total_power_w)
                - propulsion_thermal_power_w
                - cpu_thermal_power_w
                - tx_thermal_power_w,
                0.0,
            )
            # 推进喷流功率不等同于舱内热；CPU/Tx/bus 仍按电子设备热效率折算。
            electronics_thermal_power_w = (
                cpu_thermal_power_w + tx_thermal_power_w + baseline_thermal_power_w
            )
        electronics_heat_w = electronics_thermal_power_w * electronics_heat_fraction
        propulsion_heat_w = propulsion_thermal_power_w * propulsion_heat_fraction
        internal_heat_w = electronics_heat_w + propulsion_heat_w
        sunlit_fraction = float(np.clip(sunlit_fraction, 0.0, 1.0))
        solar_heat_w = (
            solar_flux_w_m2
            * sunlit_absorbing_area_m2
            * solar_absorptivity
            * sunlit_fraction
        )
        temperature_k = float(self.thermal_temperature_c) + 273.15
        ambient_k = ambient + 273.15
        radiative_cooling_w = radiator_emissivity * sigma_sb * radiator_area_m2 * (
            temperature_k**4 - ambient_k**4
        )
        net_heat_w = internal_heat_w + solar_heat_w - radiative_cooling_w
        self.thermal_temperature_c = float(
            self.thermal_temperature_c + self.dt * net_heat_w / thermal_capacity
        )

        warning = float(THERMAL_CONFIG.get("warning_temp_c", 45.0))
        max_temp = float(THERMAL_CONFIG.get("max_temp_c", 55.0))
        critical = float(THERMAL_CONFIG.get("critical_temp_c", 65.0))
        if self.thermal_temperature_c >= critical:
            stage = "critical"
        elif self.thermal_temperature_c > max_temp:
            stage = "unsafe"
        elif self.thermal_temperature_c >= warning:
            stage = "warning"
        else:
            stage = "normal"
        return {
            "temperature_c": float(self.thermal_temperature_c),
            "thermal_margin_norm": float(self._thermal_margin_norm()),
            "is_safe": bool(self.thermal_temperature_c <= max_temp),
            "is_warning": bool(stage == "warning"),
            "is_crashed": bool(stage == "critical"),
            "safety_stage": stage,
            "internal_heat_w": float(internal_heat_w),
            "electronics_heat_w": float(electronics_heat_w),
            "propulsion_heat_w": float(propulsion_heat_w),
            "electronics_thermal_power_w": float(electronics_thermal_power_w),
            "propulsion_thermal_power_w": float(propulsion_thermal_power_w),
            "warning_temp_c": warning,
            "max_temp_c": max_temp,
            "critical_temp_c": critical,
        }

    def _max_downlink_mb_per_pass(self) -> float:
        return max(0.0, float(
            GROUND_STATION_CONFIG.get("max_downlink_mb_per_pass", 0.0)))

    def _cap_link_capacity_by_pass_budget(self, capacity_mbps: float,
                                          duration_s: float,
                                          remaining_mb: float | None = None) -> float:
        """
        将单步链路容量限制在当前过顶剩余接收预算内。

        地面站几何只给瞬时链路质量；真实过顶还受捕获、排程、接收缓存和后端链路约束。
        这里用每次过顶的总 MB 上限避免高仰角阶段被模型误解成无限容量窗口。
        """
        cap = max(0.0, float(capacity_mbps))
        pass_limit_mb = self._max_downlink_mb_per_pass()
        if pass_limit_mb <= 0.0:
            return cap
        remaining = self._comm_pass_remaining_mb if remaining_mb is None else remaining_mb
        remaining = max(0.0, float(remaining))
        if duration_s <= 0.0:
            return 0.0
        budget_mbps = remaining * 8.0 / max(float(duration_s), 1e-9)
        return float(min(cap, budget_mbps))

    def _get_contact_info_at(self, time_s: float, altitude_m: float) -> dict:
        """按指定时刻/高度计算通信窗口，并应用外部 trace 覆盖。"""
        contact = self.gs_network.get_contact_info(time_s, altitude_m)
        override = self._contact_override
        if override:
            contact = dict(contact)
            contact.update(override)
            if not contact.get("in_window", False) and "max_capacity_mbps" not in override:
                contact["max_capacity_mbps"] = 0.0
        return contact

    def _get_contact_info(self) -> dict:
        """
        获取当前通信窗口信息。

        默认使用地面站几何模型；鲁棒性 trace 实验可临时写入 _contact_override，
        用真实/外部窗口标志和链路容量覆盖同一时间步的可见性。
        """
        return self._get_contact_info_at(self.time_s, self.altitude_m)

    def _instant_contact_capacity_mbps(self, time_s: float, altitude_m: float) -> float:
        """只计算指定时刻的最大链路容量，不做窗口起止扫描，供前瞻观测高频调用。"""
        sat_lat, sat_lon = self.gs_network.satellite_position(time_s, altitude_m)
        # 向量化：一次算全部站取最大容量（等价于原 per-station 循环）。
        return float(self.gs_network._contact_at(sat_lat, sat_lon, altitude_m)[1])

    def _future_contact_capacity_until_step(self, deadline_step: int) -> float:
        target_steps = max(0, int(deadline_step) - int(self.step_count or 0))
        horizon_s = float(target_steps) * float(self.dt)
        try:
            return self._future_contact_capacity_mb(horizon_s=horizon_s)
        except TypeError:
            return self._future_contact_capacity_mb()

    def _future_contact_capacity_bins(self) -> list[tuple[float, float]]:
        # ── 跨步缓存：卫星 10s 内轨道几何几乎不变，每步重算 90×19 重型三角函数纯浪费。
        # 缓存 ≤6 步（60s = 1 个 scan_step_s），超过则强制刷新（捕捉窗口切换）──
        _bins_step = getattr(self, "_cached_bins_step", -1)
        _delta = (self.step_count or 0) - _bins_step
        if _bins_step >= 0 and 0 <= _delta < 6:
            if _delta == 0:
                return self._cached_bins_value
            # 容量(MB)沿用缓存(60s 内轨道几何近似不变)，但 steps_to_bin 时间轴
            # 必须随真实时间流逝递减 _delta 步——否则远期窗口在缓存有效期内时间轴
            # 冻结、第 6 步刷新时突跳，喂给策略网络的 capacity_bin_*_time_norm 特征
            # 出现周期性锯齿抖动，破坏状态平稳性。
            return [(max(0.0, off - _delta), cap)
                    for off, cap in self._cached_bins_value]
        bin_count = max(0, int(TASK_CONFIG.get("deliverability_bin_count", 8)))
        scan_step_s = max(
            float(TRAIN_CONFIG.get("time_slot_s", 10.0)),
            float(TASK_CONFIG.get("future_contact_scan_step_s", 60.0)),
        )
        horizon_s = max(0.0, float(TASK_CONFIG.get("future_contact_lookahead_s", 0.0)))
        if bin_count <= 0 or horizon_s <= 0.0:
            return []
        pass_limit_mb = self._max_downlink_mb_per_pass()
        in_predicted_pass = bool((self._contact or {}).get("in_window", False))
        dt_f = max(float(self.dt), 1e-6)

        # ── 一次性向量化：90 时间点 × 19 站 ──────────────────────────
        # floor 语义：与 _future_contact_capacity_mb 保持一致（只取 horizon 之内的扫描点）。
        K = int(np.floor((horizon_s + 1e-9) / scan_step_s))
        if K <= 0:
            self._cached_bins_step = self.step_count or 0
            self._cached_bins_value = []
            return []
        times_s = np.arange(1, K + 1, dtype=np.float64) * scan_step_s
        # 全部时间点的各站容量 (K, 19)
        all_caps = self._contact_capacities_at_times(times_s)
        # 每步取各站最大容量 (K,)
        max_caps = np.max(all_caps, axis=1)
        # step_capacity: 每步 MB
        step_cap = max_caps * scan_step_s / 8.0

        # 过顶预算：向量化（干掉 K 次 Python for 循环）
        current_pass_used_mb = (
            max(0.0, pass_limit_mb - self._comm_pass_remaining_mb)
            if pass_limit_mb > 0.0 and in_predicted_pass else 0.0
        )
        has_cap = step_cap > 0.0
        prev_has_cap = np.empty(K, dtype=bool)
        prev_has_cap[0] = in_predicted_pass
        if K > 1:
            prev_has_cap[1:] = has_cap[:-1]
        new_pass = has_cap & ~prev_has_cap
        cumsum_full = np.cumsum(step_cap)
        # 新 pass 起点：减去 step_cap 保留当前步值（不是抹零）
        pass_start_values = np.where(new_pass, cumsum_full - step_cap, 0.0)
        base_offset = np.maximum.accumulate(pass_start_values)
        cumsum_within = cumsum_full - base_offset
        # 延续 pass：叠加当前窗口已用预算（不加会导致 cumsum_within 低估已用量）
        if in_predicted_pass and has_cap[0]:
            end_idx_arr = np.flatnonzero(~has_cap)
            end_idx = end_idx_arr[0] if end_idx_arr.size > 0 else K
            cumsum_within[:end_idx] += current_pass_used_mb
        if pass_limit_mb > 0.0:
            overflow = np.maximum(0.0, cumsum_within - pass_limit_mb)
            step_cap = np.maximum(0.0, step_cap - overflow)

        # 过滤出非零 cap 的 bins
        mask = step_cap > 1e-9
        if not mask.any():
            self._cached_bins_step = self.step_count or 0
            self._cached_bins_value = []
            return []
        idx = np.flatnonzero(mask)[:bin_count]
        res = [(float(times_s[i] / dt_f), float(step_cap[i])) for i in idx]
        self._cached_bins_step = self.step_count or 0
        self._cached_bins_value = res
        return res

    def _contact_capacities_at_times(self, time_array):
        """向量化：给定 (K,) 时间偏移数组，返回 (K, N_stations) 各站链路容量 Mbps。
        直接复用 ground_station 的统一向量化实现(单一真相源)，避免在这里重复实现
        链路预算/AMC/折射——之前的本地副本硬编码了地球半径、忽略 amc_enabled 开关、
        且 amc_capacity_levels=None 时会崩，与 _capacities_vec 存在漂移风险。"""
        t = np.asarray(time_array, dtype=np.float64)
        return self.gs_network.link_capacities_over_times(
            self.time_s + t, self.altitude_m)

    def _future_contact_capacity_mb(self, horizon_s: float | None = None) -> float:
        """
        估计未来窗口可下传容量积分（MB级别）。
        """
        scan_step_s = max(
            float(TRAIN_CONFIG.get("time_slot_s", 10.0)),
            float(TASK_CONFIG.get("future_contact_scan_step_s", 60.0)),
        )
        horizon_override = horizon_s is not None
        horizon_s = max(0.0, float(
            TASK_CONFIG.get("future_contact_lookahead_s", 0.0) if horizon_s is None else horizon_s
        ))
        # time_s 尚未初始化（reset 之前直接调 _compute_reward 等单元测试路径）时，没有
        # 几何/时间基准可算未来容量，返回 0（与 _deliverable_processing_credit 的 None 防御一致）。
        if horizon_s <= 0.0 or self.time_s is None:
            return 0.0
        # ── 步内字典缓存：process_by_priority 遍历 ~100 个任务时，相同 deadline → 相同 horizon_s
        # → 缓存命中率极高（拦截 ~95% 重复调用），直接返回不再跑 90×19 广播 ──
        if horizon_override:
            cache_key = round(horizon_s, 2)
            _sc = getattr(self, "_step_horizon_cache", None)
            if _sc is not None and cache_key in _sc:
                return _sc[cache_key]
        if not horizon_override:
            delta_t = float(self.time_s - getattr(self, "_last_future_contact_capacity_mb_time", -1e30))
            if 0.0 <= delta_t < scan_step_s / 2:
                return float(self._last_future_contact_capacity_mb)
            self._last_future_contact_capacity_mb_time = float(self.time_s)

        # ── 一次性向量化：90 时间点 × 19 站 ──────────────────────────
        pass_limit_mb = self._max_downlink_mb_per_pass()
        in_predicted_pass = bool((self._contact or {}).get("in_window", False))
        current_pass_used_mb = (
            max(0.0, pass_limit_mb - self._comm_pass_remaining_mb)
            if pass_limit_mb > 0.0 and in_predicted_pass else 0.0
        )
        # floor 语义：只统计 deadline(horizon_s)之前的扫描点，与被向量化替换掉的旧
        # 标量循环 `while elapsed <= horizon_s + 1e-9` 等价。若用 ceil，会把 deadline
        # 之后的第一个 60s 扫描点也算进来，系统性高估紧迫任务(deadline 临近)的未来
        # 可下传容量——而该路径正是 process_by_priority 每任务高频调用的。
        K = int(np.floor((horizon_s + 1e-9) / scan_step_s))
        if K <= 0:
            capacity_mb = 0.0
        else:
            times_s = np.arange(1, K + 1, dtype=np.float64) * scan_step_s
            all_caps = self._contact_capacities_at_times(times_s)
            max_caps = np.max(all_caps, axis=1)
            step_cap = max_caps * scan_step_s / 8.0
            # ── 过顶预算：向量化（干掉 K 次 Python for 循环）──
            has_cap = step_cap > 0.0
            prev_has_cap = np.empty(K, dtype=bool)
            prev_has_cap[0] = in_predicted_pass
            if K > 1:
                prev_has_cap[1:] = has_cap[:-1]
            new_pass = has_cap & ~prev_has_cap
            cumsum_full = np.cumsum(step_cap)
            # 新 pass 起点：减去 step_cap 保留当前步值（不是抹零）
            pass_start_values = np.where(new_pass, cumsum_full - step_cap, 0.0)
            base_offset = np.maximum.accumulate(pass_start_values)
            cumsum_within = cumsum_full - base_offset
            # 延续 pass：叠加当前窗口已用预算
            if in_predicted_pass and has_cap[0]:
                end_idx_arr = np.flatnonzero(~has_cap)
                end_idx = end_idx_arr[0] if end_idx_arr.size > 0 else K
                cumsum_within[:end_idx] += current_pass_used_mb
            if pass_limit_mb > 0.0:
                overflow = np.maximum(0.0, cumsum_within - pass_limit_mb)
                step_cap = np.maximum(0.0, step_cap - overflow)
            capacity_mb = float(np.sum(step_cap))

        if not horizon_override:
            self._last_future_contact_capacity_mb = float(capacity_mb)
            return self._last_future_contact_capacity_mb
        # 写入步内字典缓存（供 process_by_priority 高频调用复用）
        _sc = getattr(self, "_step_horizon_cache", None)
        if _sc is not None:
            _sc[round(horizon_s, 2)] = float(capacity_mb)
        return float(capacity_mb)

    def _future_contact_capacity_norm(self) -> float:
        """估计未来窗口可下传容量积分（归一化）。"""
        capacity_mb = self._future_contact_capacity_mb()
        horizon_s = max(0.0, float(TASK_CONFIG.get("future_contact_lookahead_s", 0.0)))
        if horizon_s <= 0.0:
            return 0.0
        pass_limit_mb = float(self._max_downlink_mb_per_pass())
        orbital_period_s = max(
            1.0,
            float(ORBITAL_CONFIG.get("orbital_period_min", 90.0)) * 60.0,
        )
        expected_passes = max(1.0, horizon_s / orbital_period_s)
        if pass_limit_mb > 0.0:
            norm_mb = pass_limit_mb * expected_passes
        else:
            norm_mbps = float(QUEUE_CONFIG.get(
                "tx_capacity_norm_mbps",
                QUEUE_CONFIG.get("tx_downlink_rate_max_mbs", 12.5) * 8.0,
            ))
            norm_mb = norm_mbps * horizon_s / 8.0
        norm_mb = max(norm_mb, 1e-6)
        self._last_future_contact_capacity_norm = float(np.clip(capacity_mb / norm_mb, 0.0, 2.0))
        return self._last_future_contact_capacity_norm

    def _apply_acquisition_latency(self, contact: dict) -> dict:
        """
        对刚进入通信窗口的链路容量施加建链/捕获延迟。

        地面站几何只回答“是否可见”，但真实数传需要捕获、同步和稳态跟踪。
        这里让窗口前几个 step 的容量从低值平滑爬升，避免策略学成刚擦边进窗口就满功率下传。
        """
        adjusted = dict(contact or {})
        in_window = bool(adjusted.get("in_window", False))
        if not in_window:
            self._comm_window_age_steps = 0
            self._comm_pass_remaining_mb = self._comm_pass_capacity_mb
            adjusted["acquisition_latency_active"] = False
            adjusted["acquisition_latency_scale"] = 1.0
            adjusted["comm_window_age_steps"] = 0
            adjusted["comm_pass_remaining_mb"] = float(self._comm_pass_remaining_mb)
            adjusted["comm_pass_capacity_mb"] = float(self._comm_pass_capacity_mb)
            return adjusted

        if self._comm_window_age_steps == 0:
            self._comm_pass_remaining_mb = self._comm_pass_capacity_mb
        self._comm_window_age_steps += 1
        latency_steps = max(0, int(GROUND_STATION_CONFIG.get("acquisition_latency_steps", 0)))
        min_scale = float(np.clip(
            GROUND_STATION_CONFIG.get("acquisition_latency_min_scale", 1.0),
            0.0,
            1.0,
        ))
        scale = 1.0
        active = False
        if latency_steps > 0 and self._comm_window_age_steps <= latency_steps:
            progress = (self._comm_window_age_steps - 1) / max(latency_steps, 1)
            scale = min_scale + (1.0 - min_scale) * progress
            active = True
            adjusted["max_capacity_mbps"] = float(
                adjusted.get("max_capacity_mbps", 0.0)
            ) * scale

        adjusted["max_capacity_mbps"] = self._cap_link_capacity_by_pass_budget(
            float(adjusted.get("max_capacity_mbps", 0.0)),
            float(self.dt),
        )
        adjusted["comm_pass_remaining_mb"] = float(self._comm_pass_remaining_mb)
        adjusted["comm_pass_capacity_mb"] = float(self._comm_pass_capacity_mb)
        adjusted["acquisition_latency_active"] = bool(active)
        adjusted["acquisition_latency_scale"] = float(scale)
        adjusted["comm_window_age_steps"] = int(self._comm_window_age_steps)
        return adjusted

    def _scene_context_for_phase(self, phase: float | None = None,
                                 lookahead_steps: int = 0) -> dict:
        """把轨道相位映射为当前地理/任务语义画像。"""
        use_current_event = phase is None and int(lookahead_steps) == 0
        if phase is None:
            phase = float(self.orbit_sim.phase)
        if lookahead_steps:
            altitude_m = (
                float(self.altitude_m)
                if self.altitude_m is not None
                else float(ORBITAL_CONFIG["altitude_nominal_km"]) * 1e3
            )
            r = self.orbit_dyn.R_e + altitude_m
            n = np.sqrt(self.orbit_dyn.mu / max(r, 1e-9)**3)
            phase = phase + n * self.dt * int(lookahead_steps)
        phase = float(phase % (2.0 * np.pi))
        semantic_phase = (
            phase + float(self._scene_phase_offset_fraction) * 2.0 * np.pi
        ) % (2.0 * np.pi)
        phase_fraction = semantic_phase / (2.0 * np.pi)
        latitude_proxy = float(np.sin(semantic_phase))

        scene_name = self._scene_name_for_phase_fraction(phase_fraction, latitude_proxy)
        profiles = TASK_CONFIG.get("scene_profiles", {})
        profile = dict(profiles.get(scene_name, profiles.get("routine_land", {})))
        if use_current_event and bool(self._last_emergency_event_active):
            return self._emergency_scene_context(
                scene_name,
                phase_fraction,
                latitude_proxy,
            )
        return {
            "scene_name": scene_name,
            "scene_class_code": float(profile.get("class_code", 0.0)),
            "arrival_multiplier": float(profile.get("arrival_multiplier", 1.0)),
            "phase_fraction": float(phase_fraction),
            "scene_phase_offset_fraction": float(self._scene_phase_offset_fraction),
            "latitude_proxy": latitude_proxy,
            "profile": profile,
            "emergency_event_active": False,
            "emergency_event_triggered": False,
            "emergency_event_remaining_steps": 0.0,
        }

    def _emergency_scene_context(self, base_scene_name: str,
                                 phase_fraction: float,
                                 latitude_proxy: float) -> dict:
        profiles = TASK_CONFIG.get("scene_profiles", {})
        emergency_scene = str(TASK_CONFIG.get(
            "emergency_event_scene", "emergency_disaster"))
        profile = dict(profiles.get(
            emergency_scene,
            profiles.get("disaster", profiles.get("routine_land", {})),
        ))
        return {
            "scene_name": emergency_scene,
            "base_scene_name": str(base_scene_name),
            "scene_class_code": float(profile.get("class_code", 1.0)),
            "arrival_multiplier": float(profile.get("arrival_multiplier", 1.0)),
            "phase_fraction": float(phase_fraction),
            "scene_phase_offset_fraction": float(self._scene_phase_offset_fraction),
            "latitude_proxy": float(latitude_proxy),
            "profile": profile,
            "emergency_event_active": True,
            "emergency_event_triggered": bool(self._last_emergency_event_triggered),
            "emergency_event_remaining_steps": float(
                self._emergency_event_remaining_steps),
        }

    def _diurnal_angle_rad(self) -> float:
        """卫星位置与日间 bulge 中心的夹角 Ψ (PDF Section 5)。

        简化模型：orbit_sim 中 phase=0 是进入日照、phase=sunlit_phase/2 是亚午点，
        bulge 中心约 = 亚午点 + 30° 滞后 (热惯性，pdf 默认 2h ≈ 30° 局部时角)。
        Ψ ∈ [-π, π]；Ψ=0 处于 bulge 峰值，|Ψ|=π 处于夜侧密度谷。
        """
        phase = float(self.orbit_sim.phase)
        sunlit_phase = float(getattr(self.orbit_sim, "_sunlit_phase", np.pi))
        lag_rad = float(DRAG_CONFIG.get("diurnal_bulge_lag_rad", np.pi / 6.0))
        bulge_center = 0.5 * sunlit_phase + lag_rad
        delta = (phase - bulge_center + np.pi) % (2.0 * np.pi) - np.pi
        return float(delta)

    def _advance_storm_event_state(self) -> None:
        """推进地磁暴事件状态机，更新当前 Ap (PDF Section 8.2)。

        Ap 是真正的驱动量：静日 = episode quiet 基线；风暴期由三角剖面抬升到峰值 Ap
        (前 20% 时长线性爬升，后 80% 指数衰减回 quiet)。当前 Ap 同时喂给两条路径：
          - 解析模型 (exponential/vallado)：经 ap_to_storm_multiplier 折算 atm.storm_multiplier
          - NRLMSISE-00：写入 SpaceWeatherState.ap，模型原生消费
        触发概率默认每步 ~5e-5 (~0.1 次/episode)；peak Ap ∈ ap_storm_range。
        """
        atm = self.orbit_dyn.atm
        quiet_ap = float(getattr(self, "_episode_quiet_ap",
                                 DRAG_CONFIG.get("ap_quiet_range", (3.0, 15.0))[0]))

        def _apply_ap(ap_value: float) -> None:
            ap_value = float(ap_value)
            if self._sw_state is not None:
                self._sw_state.ap = ap_value
            atm.storm_multiplier = ap_to_storm_multiplier(ap_value, ap_quiet=quiet_ap)
            self._last_storm_multiplier = float(atm.storm_multiplier)
            self._last_ap = ap_value

        if not bool(DRAG_CONFIG.get("enable_storm_events", True)):
            _apply_ap(quiet_ap)
            return
        if self._storm_active_steps_remaining > 0:
            total = max(int(self._storm_active_steps_total), 1)
            elapsed = total - int(self._storm_active_steps_remaining)
            progress = float(elapsed) / float(total)
            ramp_frac = 0.2
            if progress < ramp_frac:
                shape = progress / ramp_frac
            else:
                shape = float(np.exp(-3.0 * (progress - ramp_frac) / max(1.0 - ramp_frac, 1e-6)))
            _apply_ap(quiet_ap + (float(self._storm_peak_ap) - quiet_ap) * shape)
            self._storm_active_steps_remaining -= 1
            if self._storm_active_steps_remaining <= 0:
                self._storm_cooldown_remaining = int(
                    DRAG_CONFIG.get("storm_cooldown_steps", 600))
            return
        if self._storm_cooldown_remaining > 0:
            self._storm_cooldown_remaining -= 1
            _apply_ap(quiet_ap)
            return
        # 尝试触发新风暴 (用独立 rng，避免改动主 rng 序列)。
        # 课程缩放：触发概率 + peak Ap 上界 同时按 _randomization_scale 缩窄。
        # Exploration (scale=0.2)：prob *= 0.2，peak Ap 仅轻微高于 quiet；
        # Optimization (scale=1.0)：完整 prob，peak Ap 达 ap_storm_range 上界 (G 级风暴)。
        r_scale = float(np.clip(self._randomization_scale, 0.0, 1.0))
        prob = float(DRAG_CONFIG.get("storm_probability_per_step", 5e-5)) * r_scale
        if self._storm_rng.random() < prob:
            dur_range = DRAG_CONFIG.get("storm_duration_steps_range", (30, 180))
            ap_storm_lo, ap_storm_hi = DRAG_CONFIG.get("ap_storm_range", (50.0, 400.0))
            sampled_peak_ap = float(self._storm_rng.uniform(
                float(ap_storm_lo), float(ap_storm_hi)))
            # 课程缩放：peak Ap 在 quiet 与抽样值之间按 r_scale 线性插值。
            self._storm_peak_ap = quiet_ap + (sampled_peak_ap - quiet_ap) * r_scale
            dur_min = max(1, int(dur_range[0]))
            dur_max = max(dur_min, int(dur_range[1]))
            self._storm_active_steps_total = int(
                self._storm_rng.integers(dur_min, dur_max + 1))
            self._storm_active_steps_remaining = self._storm_active_steps_total
        _apply_ap(quiet_ap)

    def _advance_emergency_event_state(self) -> bool:
        """推进突发灾害事件状态；返回当前 step 是否处于事件覆盖中。"""
        self._last_emergency_event_triggered = False
        if not bool(TASK_CONFIG.get("emergency_event_enable", True)):
            self._emergency_event_remaining_steps = 0
            self._emergency_event_cooldown_steps = 0
            self._last_emergency_event_active = False
            return False

        if self._emergency_event_remaining_steps <= 0:
            if self._emergency_event_cooldown_steps > 0:
                self._emergency_event_cooldown_steps -= 1
            else:
                probability = float(np.clip(
                    TASK_CONFIG.get("emergency_event_probability_per_step", 0.0),
                    0.0,
                    1.0,
                ))
                if self.rng.random() < probability:
                    duration_bounds = TASK_CONFIG.get(
                        "emergency_event_duration_steps", (18, 48))
                    duration_min = max(1, int(duration_bounds[0]))
                    duration_max = max(duration_min, int(duration_bounds[1]))
                    self._emergency_event_remaining_steps = int(
                        self.rng.integers(duration_min, duration_max + 1))
                    self._last_emergency_event_triggered = True

        active = self._emergency_event_remaining_steps > 0
        if active:
            self._emergency_event_remaining_steps -= 1
            if self._emergency_event_remaining_steps <= 0:
                self._emergency_event_cooldown_steps = max(
                    0,
                    int(TASK_CONFIG.get("emergency_event_cooldown_steps", 0)),
                )
        self._last_emergency_event_active = bool(active)
        return bool(active)

    def _scene_name_for_phase_fraction(self, phase_fraction: float,
                                       latitude_proxy: float) -> str:
        phase_fraction = float(phase_fraction % 1.0)
        # 读取 per-episode 打乱后的 rules；fallback 到 TASK_CONFIG (env 实例化前的早期调用)。
        rules = getattr(self, "_phase_scene_rules", None) \
            or TASK_CONFIG.get("phase_scene_rules", [])
        for rule in rules:
            start = float(rule.get("start", 0.0))
            end = float(rule.get("end", 0.0))
            if start <= end:
                matched = start <= phase_fraction < end
            else:
                matched = phase_fraction >= start or phase_fraction < end
            if matched:
                return str(rule.get("scene", "routine_land"))

        if abs(latitude_proxy) > 0.85:
            return "polar_cloud"
        if -0.2 <= latitude_proxy <= 0.2:
            return "open_ocean"
        return "routine_land"

    def _build_episode_phase_scene_rules(self) -> list:
        """每 episode 在 reset 时基于 TASK_CONFIG.phase_scene_rules 构造当前 episode 用的
        phase 块序列。三层随机化（与 TASK_CONFIG 一致仅保留物理上不变的"轨道平均场景占比"）：

        1. 每个场景的"块数"随机化（military 可能 1 个 4% 大块，也可能 2~3 个小块）
        2. 每块的"时长"用 Dirichlet 在该场景总额内随机切分（块长度不再统一）
        3. 所有块的"出现顺序"随机打乱

        保留：各场景的总占比与 TASK_CONFIG 严格一致（地球地理决定，海洋永远 ~49% 等）。
        效果：彻底消除 agent 从 "scene 顺序 / 块时长 / 块数" 走捷径的可能。
        """
        base_rules = list(TASK_CONFIG.get("phase_scene_rules", []))
        if not base_rules:
            return base_rules

        # 提取各场景的总占比 + 原始块数（用作随机化基准）
        scene_totals = {}
        scene_base_block_count = {}
        for r in base_rules:
            s = float(r.get("start", 0.0))
            e = float(r.get("end", 0.0))
            dur = (e - s) if e >= s else ((1.0 - s) + e)
            scene = str(r.get("scene", "routine_land"))
            scene_totals[scene] = scene_totals.get(scene, 0.0) + dur
            scene_base_block_count[scene] = scene_base_block_count.get(scene, 0) + 1

        # 关闭随机化：按原始 base_rules 重建（保留 wrap-around 拆段）
        if not bool(TASK_CONFIG.get("randomize_scene_rule_order", True)):
            rules = []
            cum = 0.0
            for r in base_rules:
                s = float(r.get("start", 0.0))
                e = float(r.get("end", 0.0))
                dur = (e - s) if e >= s else ((1.0 - s) + e)
                scene = str(r.get("scene", "routine_land"))
                rules.append({"start": cum, "end": cum + dur, "scene": scene})
                cum += dur
            if rules:
                rules[-1]["end"] = 1.0
            return rules

        # 1) 每场景随机化块数：base_count + Δ，Δ ∈ {-1, 0, +1, +2}（至少 1 块）
        scene_block_counts = {}
        for scene, base_count in scene_base_block_count.items():
            delta = int(self._physics_rng.integers(-1, 3))  # -1, 0, 1, 2
            scene_block_counts[scene] = max(1, base_count + delta)

        # 2) 每场景用 Dirichlet 切分块时长。α=2.5 适度集中（既有变化又避免极端小块）。
        all_blocks = []
        min_block_frac = 0.005  # 块时长下限 0.5% (≈11 step at 2160-step episode)
        for scene, total in scene_totals.items():
            n = scene_block_counts[scene]
            if n == 1 or total <= 1e-9:
                all_blocks.append((float(total), scene))
                continue
            alpha = np.full(n, 2.5)
            shares = self._physics_rng.dirichlet(alpha)
            # 钳位：每块 ≥ min_block_frac（在该场景内部归一化）
            shares = np.maximum(shares, min_block_frac / max(total, min_block_frac))
            shares = shares / shares.sum()
            for s in shares:
                all_blocks.append((float(s * total), scene))

        # 3) 打乱所有块顺序
        perm = list(range(len(all_blocks)))
        self._physics_rng.shuffle(perm)
        shuffled = [all_blocks[i] for i in perm]

        # 4) 按 cumulative 重建 (start, end)
        rules = []
        cum = 0.0
        for dur, scene in shuffled:
            rules.append({"start": cum, "end": cum + dur, "scene": scene})
            cum += dur
        if rules:
            rules[-1]["end"] = 1.0
        return rules

    def _max_scene_arrival_multiplier(self) -> float:
        profiles = TASK_CONFIG.get("scene_profiles", {})
        if not profiles:
            return 1.0
        return max(
            float(profile.get("arrival_multiplier", 1.0))
            for profile in profiles.values()
        )

    def _scene_intensity_score(self, scene_context: dict) -> float:
        """估计场景的前瞻任务压力：到达量、价值倍率、语义等级和 deadline 紧迫度共同决定。"""
        profile = dict(scene_context.get("profile", {}))
        arrival = float(profile.get(
            "arrival_multiplier",
            scene_context.get("arrival_multiplier", 1.0),
        ))
        value_multiplier = float(profile.get("base_value_multiplier", 1.0))
        scene_class = float(profile.get(
            "class_code",
            scene_context.get("scene_class_code", 0.0),
        ))
        deadline_range = profile.get(
            "deadline_range_steps",
            (TASK_CONFIG.get("deadline_min_steps", 60), TASK_CONFIG.get("deadline_max_steps", 360)),
        )
        deadline_mid = 0.5 * (float(deadline_range[0]) + float(deadline_range[1]))
        urgency = float(TASK_CONFIG.get("deadline_max_steps", 360)) / max(deadline_mid, 1.0)
        return float(arrival * value_multiplier * (0.25 + scene_class) * urgency)

    def _max_scene_intensity_score(self) -> float:
        profiles = TASK_CONFIG.get("scene_profiles", {})
        if not profiles:
            return 1.0
        scores = []
        for name, profile in profiles.items():
            scores.append(self._scene_intensity_score({
                "scene_name": name,
                "scene_class_code": float(profile.get("class_code", 0.0)),
                "arrival_multiplier": float(profile.get("arrival_multiplier", 1.0)),
                "profile": profile,
            }))
        return max(max(scores), 1e-6)

    def _normalized_scene_intensity(self, scene_context: dict) -> float:
        return float(np.clip(
            self._scene_intensity_score(scene_context) / self._max_scene_intensity_score(),
            0.0,
            1.0,
        ))

    def _arrival_rate_for_scene(self, scene_context: dict | None = None) -> float:
        base = QUEUE_CONFIG["data_arrival_rate_mbs"] * self.dt
        context = scene_context or self._scene_context_for_phase()
        scene_mult = float(context.get("arrival_multiplier", 1.0))
        period_phase = float(context.get(
            "phase_fraction",
            float(self.orbit_sim.phase) / (2.0 * np.pi),
        ))
        hotspot_strength = float(TASK_CONFIG.get("orbital_hotspot_strength", 0.35))
        hot = 1.0 + hotspot_strength * np.exp(-((period_phase * 90 - 30)**2) / (2 * 20**2))
        return float(base * hot * scene_mult * self._data_arrival_scale)

    def _sample_data_arrival(self, scene_context: dict | None = None) -> float:
        # 数据到达量由轨道相位语义主导，轻量相位热点只模拟载荷观测机会的周期性变化。
        scaled_rate = self._arrival_rate_for_scene(scene_context)
        return float(self.rng.poisson(max(scaled_rate, 0.0)))

    def _compute_reward(self, data_info, batt_info, orbit_info,
                        eq_info, oq_info, cq_info,
                        actual_tx_mb, in_window, power_info,
                        delivery_info=None,
                        thermal_info=None) -> tuple:
        w = self._reward_config_for_step()
        delivery_info = delivery_info or {}
        # 当前口径下 reward 只计算论文目标 r_t；安全风险统一进入 constraint cost c_t。
        delivered_value = float(delivery_info.get("delivered_value", actual_tx_mb))
        on_time_value = float(delivery_info.get("on_time_delivered_value", delivered_value))
        expired_value = float(delivery_info.get("expired_value", 0.0))
        expired_high_value = float(delivery_info.get("expired_high_value", 0.0))
        dropped_value = float(delivery_info.get("dropped_value", 0.0))
        if dropped_value <= 0.0:
            dropped_value = float(delivery_info.get("dropped_raw_value", 0.0)) + \
                float(delivery_info.get("dropped_processed_value", 0.0)) + \
                float(delivery_info.get("active_dropped_low_value", 0.0))

        # dropped_mb：所有丢弃动作的 MB 总量（用于按 MB 计算 w_drop_mb_penalty）
        dropped_mb = (
            float(delivery_info.get("dropped_raw_mb", 0.0))
            + float(delivery_info.get("dropped_processed_mb", 0.0))
            + float(delivery_info.get("active_dropped_low_raw_mb", 0.0))
            + float(delivery_info.get("active_dropped_low_processed_mb", 0.0))
        )

        processed_util = max(0.0, float(
            cq_info.get("urgency_raw", cq_info.get("urgency", 0.0)) or 0.0))

        # ── potential-based reward shaping（Ng 1999）──────────────────────────
        # r'_t = r_t + γ·Φ(s_{t+1}) - Φ(s_t)
        # 不改变最优策略，但让"远窗口处理行为"能即时感受到未来容量变化，
        # 解决 γ^540 ≈ 0.005 的长时域信号衰减问题。
        # 注：此时 processed queue 已更新（本步处理/下传已完成），即为 s_{t+1} 的队列状态。
        gamma = float(DRL_CONFIG.get("gamma", 0.995))
        shaping_coeff = float(DRL_CONFIG.get("reward_shaping_coeff", 0.1))
        phi_next = self._potential()
        phi_prev = float(getattr(self, "_prev_potential", phi_next))
        self._prev_potential = phi_next
        reward_shaping = shaping_coeff * (gamma * phi_next - phi_prev)

        # ── 时序感知 shaping 所需的三个状态量 ───────────────────────────────
        # 1. time_to_next_window_norm: 0=正在窗口, 1=最远未来窗口
        contact_for_shaping = self._contact or {}
        ttnw_s_step = float(contact_for_shaping.get("time_to_next_window_s", 0.0))
        in_window_now = bool(contact_for_shaping.get("in_window", False))
        time_to_next_window_norm = 0.0 if in_window_now else float(np.clip(
            ttnw_s_step / max(self._time_to_next_window_norm_s, 1e-6), 0.0, 1.0))

        # 2. processed_value: 本步处理掉的总价值（不分 high/mid/low），用于
        #    按 (1 - deliver_prob) 折扣的 prospective shaping。
        #    delivery_info 已经 merge 了 process_info（见 self._last_delivery_info 构造）
        processed_value_step = float(delivery_info.get("processed_value", 0.0))

        # 3. prospective_deliver_prob: 简单估计 "本步处理 + 已有 backlog 是否能
        #    在未来 contact capacity 内被全部下传"。
        #    queue_total = 当前 processed queue + 本步刚处理的 MB
        #    prob = clip(future_capacity / queue_total, 0, 1)
        #    含义：未来窗口能不能装下"现在已经在排队的所有东西"。装得下 prob=1，
        #    远远装不下 prob→0，agent 处理这条数据被打的折扣就越狠。
        processed_mb_step = float(delivery_info.get("processed_mb", data_info.get("serviced", 0.0)))
        future_cap_mb_shaping = float(w.get("_future_contact_capacity_mb", 0.0))
        queue_after_proc = float(self.comm_queue.value) + processed_mb_step
        prospective_deliver_prob = 1.0
        if processed_mb_step > 1e-9 and future_cap_mb_shaping >= 0.0:
            if queue_after_proc <= 1e-9:
                prospective_deliver_prob = 1.0
            else:
                prospective_deliver_prob = float(np.clip(
                    future_cap_mb_shaping / queue_after_proc, 0.0, 1.0))

        # 4. actuator_violation_mb: CPU gate soft-mode 下 "本应被截掉" 的 MB。
        actuator_violation_mb_step = float(
            getattr(self, "_last_cpu_gate_violation_mb", 0.0))

        reward = compute_mission_reward(
            delivered_value=delivered_value,
            on_time_delivered_value=on_time_value,
            expired_value=expired_value,
            expired_high_value=expired_high_value,
            dropped_value=dropped_value,
            dropped_mb=dropped_mb,
            transmitted_mb=actual_tx_mb,
            processed_mb=processed_mb_step,
            total_power_w=float(power_info.get("P_total_w", 0.0)),
            propulsion_power_w=float(power_info.get("P_propulsion_w", 0.0)),
            dt_s=float(self.dt),
            cfg=w,
            deliverable_processing_credit_value=(
                self._deliverable_processing_credit(delivery_info)
                if bool(DRL_CONFIG.get("enable_deliverable_processing_reward", False))
                else 0.0
            ),
            processed_value=processed_value_step,
            processed_deliverable_value=float(delivery_info.get("processed_deliverable_value", 0.0)),
            processed_undeliverable_value=float(delivery_info.get("processed_undeliverable_value", 0.0)),
            time_to_next_window_norm=time_to_next_window_norm,
            prospective_deliver_prob=prospective_deliver_prob,
            actuator_violation_mb=actuator_violation_mb_step,
            # A2: 类加权 reward 所需的分类拆分
            delivered_high_value=float(delivery_info.get("delivered_high_value", 0.0)),
            delivered_mid_value=float(delivery_info.get("delivered_medium_value", 0.0)),
            delivered_low_value=float(delivery_info.get("delivered_low_value", 0.0)),
            # 窗口期 TX 闲置惩罚所需输入
            in_window=bool(in_window),
            link_capacity_mb=float(cq_info.get("link_capacity_mb", 0.0)),
            pre_tx_pending_mb=float(cq_info.get("pre_tx_pending_mb", 0.0)),
        )

        mission_stage, _ = self._classify_mission_stage(orbit_info, batt_info)
        thermal_info = thermal_info or {}
        breakdown = dict(reward.components)
        breakdown.update({
            "risk_stage_code": {"normal": 0.0, "warning": 1.0, "unsafe": 2.0, "failure": 3.0}[mission_stage],
            "delivered_value": reward.delivered_value,
            "on_time_delivered_value": reward.on_time_delivered_value,
            "expired_value": reward.expired_value,
            "dropped_value": reward.dropped_value,
            "dropped_mb": dropped_mb,
            "r_shaping": reward_shaping,
            "phi_prev": phi_prev,
            "phi_next": phi_next,
            "_thermal_excess_c": max(
                0.0,
                float(thermal_info.get("temperature_c", 0.0))
                - float(thermal_info.get(
                    "warning_temp_c",
                    THERMAL_CONFIG.get("warning_temp_c", 45.0),
                )),
            ),
            "_processed_queue_util": processed_util,
            "_eq_drift": eq_info["drift"], "_oq_drift": oq_info["drift"],
            "reward_objective": breakdown.get("reward_objective", "value_aware"),
        })
        total_with_shaping = float(reward.total) + reward_shaping
        return float(total_with_shaping), breakdown

    def _reward_config_for_step(self) -> dict:
        cfg = dict(REWARD_CONFIG)

        # ── 注入远窗口处理 shaping 所需的窗口状态 ───────────────────────────
        # r_proc_far_window 依赖当前 time_to_next_window 与 in_window 标志做连续 ramp。
        # 不使用绑定到 step 的固定阈值（如 cpu_active_far_from_window 用的 300s），
        # 直接给 reward 函数原始时间值，由 mission_reward 内部连续映射。
        contact_now = self._contact or {}
        cfg["_time_to_next_window_s"] = float(contact_now.get("time_to_next_window_s", 0.0))
        cfg["_in_comm_window"] = bool(contact_now.get("in_window", False))

        # ── 注入容量门控 headroom 参数（容量门控分段处理惩罚所需）──────────────
        cfg["_processed_queue_mb"] = float(self.comm_queue.value)
        # 用 "近期实际可下传量" 作为 prospective_deliver_prob 的分母上限，而不是 queue_max。
        # 单次过顶最多下传 max_downlink_mb_per_pass(=800)，queue 容量 4096 >> 单次过顶，
        # 之前用 queue_max 当 cap 让 prob ≈ 1.0 永远成立，agent 收不到"队列太满 → 处理无效"
        # 的信号。用 2× pass_cap 作为分母 cap：queue 超过 ~1600MB 后 prob 开始下降，
        # 给 r_processing_opportunity_cost 提供有效梯度。
        future_cap_raw = float(self._future_contact_capacity_mb())
        near_term_dl_cap_mb = 2.0 * float(
            GROUND_STATION_CONFIG.get("max_downlink_mb_per_pass", 0.0))
        if near_term_dl_cap_mb <= 0.0:
            near_term_dl_cap_mb = float(self.comm_queue.max_value)
        cfg["_future_contact_capacity_mb"] = min(future_cap_raw, near_term_dl_cap_mb)

        if not bool(DRL_CONFIG.get("enable_deliverable_processing_reward", False)):
            return cfg
        warmup = max(0, int(PROCESSING_CREDIT_CONFIG.get(
            "deliverable_processing_credit_warmup_steps", 0)))
        anneal = max(0, int(PROCESSING_CREDIT_CONFIG.get(
            "deliverable_processing_credit_anneal_steps", 0)))
        initial = max(0.0, float(PROCESSING_CREDIT_CONFIG.get(
            "w_deliverable_processing_initial", 0.0)))
        final = max(0.0, float(PROCESSING_CREDIT_CONFIG.get(
            "w_deliverable_processing_final", initial)))
        current_step = 0 if self.step_count is None else int(self.step_count)
        if current_step <= warmup:
            weight = initial
        elif anneal <= 0:
            weight = final
        else:
            progress = float(np.clip((current_step - warmup) / max(anneal, 1), 0.0, 1.0))
            weight = initial + (final - initial) * progress
        cfg.update({
            "w_deliverable_processing": weight,
            "deliverable_processing_credit_cap_fraction": max(0.0, float(
                PROCESSING_CREDIT_CONFIG.get("deliverable_processing_credit_cap_fraction", 0.20))),
        })
        return cfg

    def _potential(self) -> float:
        """势函数 Φ(s) = min(processed_queue_value, future_contact_capacity_value)。

        用于 potential-based reward shaping（Ng 1999），不改变最优策略，
        但能让"远窗口处理行为"即时感受到未来下传收益的变化，解决长时域信号稀疏问题。
        Φ 上升 → agent 处理且 future capacity 跟得上 → shaping 正向
        Φ 被 min 钉住 → processed queue 已超 future capacity → 继续处理 shaping 为 0
        """
        processed_value = float(self.task_tracker.processed_value)
        future_cap_mb = float(self._future_contact_capacity_mb())
        # 把 MB 容量转换到 value 尺度：用当前 processed_queue 的平均价值密度估算
        proc_mb = max(float(self.comm_queue.value), 1e-6)
        value_density = processed_value / proc_mb  # value/MB
        future_cap_value = future_cap_mb * value_density
        return float(min(processed_value, future_cap_value))

    def _deliverable_processing_credit(self, info: dict) -> float:
        processed_high = max(0.0, float(info.get("processed_high_value", 0.0)))
        processed_mid = max(0.0, float(info.get("processed_medium_value", 0.0)))
        if "future_capacity_mb" in info:
            future_capacity_mb = max(0.0, float(info.get("future_capacity_mb", 0.0)))
        elif "future_contact_capacity_mb" in info:
            future_capacity_mb = max(0.0, float(info.get("future_contact_capacity_mb", 0.0)))
        elif self.time_s is None:
            return 0.0
        else:
            future_capacity_mb = max(0.0, float(self._future_contact_capacity_mb()))
        if future_capacity_mb <= 1e-6:
            return 0.0

        contact = self._contact or {}
        in_window = bool(contact.get("in_window", False))
        time_to_next_window_s = max(0.0, float(contact.get("time_to_next_window_s", 0.0)))

        # Smooth capacity decay instead of hard cutoff at max_future_ratio
        processed_ratio = float(np.clip(
            self.comm_queue.value / max(future_capacity_mb, 1e-6), 0.0, 1.0))
        capacity_gate = 1.0 - processed_ratio

        # 从缓存读取（step() 已在 _compute_reward 前算好并缓存）
        if getattr(self, "_cached_step", -1) == self.step_count:
            deadline_stats = self._cached_deadline_contact_stats
        else:
            deadline_stats = self.task_tracker.deadline_contact_stats(
                self.step_count,
                0.0 if in_window else time_to_next_window_s / max(float(self.dt), 1e-6),
            )
        high_deliverable = float(np.clip(
            deadline_stats.get("raw_high_next_window_deliverable_ratio", 0.0), 0.0, 1.0))
        processed_high_deliverable = float(np.clip(
            deadline_stats.get("processed_high_next_window_deliverable_ratio", 0.0), 0.0, 1.0))
        mismatch = float(np.clip(
            deadline_stats.get("high_value_deadline_contact_mismatch", 0.0), 0.0, 1.0))
        
        # Smooth deliverability score instead of min_high_gate
        high_gate = max(0.0, max(high_deliverable, processed_high_deliverable) * (1.0 - mismatch))
        mid_gate = max(
            0.0,
            min(
                high_gate,
                float(PROCESSING_CREDIT_CONFIG.get(
                    "deliverable_processing_mid_gate_floor", 0.25)) * capacity_gate,
            ),
        )
        mid_weight = max(0.0, float(PROCESSING_CREDIT_CONFIG.get(
            "deliverable_processing_mid_value_weight", 0.20)))
        credit = (processed_high * high_gate + mid_weight * processed_mid * mid_gate) * capacity_gate
        return float(max(0.0, credit))

    def _check_done(self, batt_info, orbit_info, thermal_info=None) -> tuple:
        # 警告/不安全状态可恢复；只有 122km 再入、SOC<=5% 深度过放或严重过热才终止。
        thermal_info = thermal_info or {}
        if (
            batt_info.get("is_crashed", False)
            or orbit_info.get("is_crashed", False)
            or thermal_info.get("is_crashed", False)
            # [SAFETY-REAL] 油尽不直接判死:它使推力=0,随后高度衰减触发上面的 orbit is_crashed
        ):
            return True, False
        if self.step_count >= self.max_steps:
            return False, True
        return False, False

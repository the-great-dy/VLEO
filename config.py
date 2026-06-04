"""
全局实验配置入口。

本模块集中定义轨道、能量、任务队列、地面站、SAC/CMDP 训练和评估参数。
其他模块应从这里读取超参数，避免在算法、环境或实验脚本中散落硬编码常量。
"""

# ─────────────────────────────────────────────
# 轨道参数
# ─────────────────────────────────────────────
ORBITAL_CONFIG = {
    # ── 操作高度阈值对齐到真实物理临界点 (thrust @ 720W = 31.83mN, area=1.0m²) ──
    # 临界点是 thrust/drag 比的物理边界，每个阈值对应一个明确的工程含义：
    #   - 200km: t/d = 2.14 → 安全巡航下界 (推力是 drag 2倍以上，余量充足)
    #   - 180km: t/d = 0.96 → drag ≈ thrust 临界点 (再低则无法仅靠推进维持)
    #   - 150km: t/d = 0.30 → drag 完全主导，eclipse 不可恢复
    #   - 120km: 物理再入终止线 (PDF VLEO 下边界)
    # 对应真实 VLEO 操作剖面：GOCE 230-275km nominal、SLATS 240-271km，
    # 167km tech demo 对应我们的 warning 边缘。
    "altitude_warning_km": 200.0,    # 安全巡航下界；低于此值进入 marginal zone
    "altitude_min_km": 180.0,        # 不安全边界 (t/d < 1)；低于此值即使满推也下降
    "altitude_crash_km": 120.0,      # PDF VLEO 下边界 = 不可恢复再入终止线
    "altitude_nominal_km": 250.0,    # 标称工作点 (200~300km cruise 区中段)
    "altitude_max_km": 300.0,        # PDF VLEO 主应用区上界
    "inclination_deg": 51.6,         # 默认轨道倾角；地面站可见性与J2摄动共用
    "raan_deg": 0.0,                 # 初始升交点赤经
    "orbital_period_min": 90.0,      # 轨道周期 (分钟)
    "eclipse_duration_min": 35.0,    # 阴影区持续时间 (分钟)
    "sunlit_duration_min": 55.0,     # 日照区持续时间 (分钟)
    "earth_radius_km": 6371.0,       # 地球半径 (km)
    "mu": 3.986e14,                  # 地球引力常数 (m^3/s^2)
    "J2": 1.08262668e-3,             # 地球二阶带谐项，用于地面轨迹/窗口预测
    "enable_j2": True,               # 启用J2导致的RAAN长期漂移
    # ── (新 PDF) Section 5：β 角 eclipse domain randomization ──
    # 每 episode reset 按 β = arcsin(sin i·sin(Ω-Ω_⊙) + cos i·sin δ_⊙) 抽样阴影占比，
    # 让 agent 学到季节性能量盈亏（β≈0：~35min 满阴影；|β|>β_crit≈70°：全日照）。
    # 上界 75° 是 51.6° 倾角 + δ_⊙±23.45° 的物理极值。
    # 训练时由 env._randomization_scale 线性缩窄，Exploration 阶段约 15° 上限
    # （eclipse 始终接近基线），Optimization 阶段使用全 75°（含全日照极端情况）。
    "enable_eclipse_beta_randomization": True,
    "eclipse_beta_max_deg": 75.0,
}

# ─────────────────────────────────────────────
# 大气阻力参数
# ─────────────────────────────────────────────
DRAG_CONFIG = {
    "Cd": 2.2,                       # 阻力系数 (VLEO 范围 2.0~2.4，PDF Section 8.1) — 物理常数
    # 设计模型：「SLATS - 大型科学载荷 (SAR / 光学 / 相控阵 / 独立热控)」
    # 物理参数全部真实，能源/电池/质量按"减去载荷子系统所贡献的那部分"调整：
    #   - mass: 383 (SLATS 整星) - 83 (SAR + 载荷支撑结构) = 300kg
    #   - area: 1.0m² (真实 VLEO 平台几何不变)
    #   - prop_max: 900W (载荷不再占功率配额，prop 可分配到接近 solar 上限)
    #   - solar: 800W (略低于真实 SLATS ~1000W，无大载荷不需要峰值发电)
    #   - battery: 500Wh (真实 ~1000Wh - 载荷 eclipse 需求)
    # 操作包线：sustainable cruise 200-300km，180km burst（眼罩内不可持续），
    # 150-180km warning zone 有部分可恢复性 (180-190km t/d > 1)，<150km 临界失控。
    # 警告区/不安全区在此设计下都有训练意义。
    "area_m2": 1.0,                  # GOCE arrow-shape 等效迎风面积 (真实 VLEO 标准)
    "mass_kg": 300.0,                # SLATS 整星 383kg 减去 SAR 载荷 (~83kg)
    # rho_ref 是 ref_altitude (=350km, 单元测试 invariant 锚点) 处的标定密度。
    # 旧 4.89e-11 对应 F10.7≈200~250 高太阳活跃期，叠加 Vallado 低高度密度让 250km
    # 处 drag 达推力的 12 倍 → 任何高度都死。现在校到 Vallado 标准 nominal solar
    # (F10.7=150)，250km 处 drag ≈ 1.7×推力，270km 平衡，agent 有真实操作空间。
    # 高太阳活跃期通过 enable_solar_activity_randomization × rho_scale 暴露给 agent。
    "rho_ref": 6.66e-12,             # Vallado 标准 @ 350km (F10.7=150 nominal solar)
    "H_scale_km": 50.0,              # ref_altitude 段局部 scale height；ref_alt=350km 在操作区外，此值仅作单元测试不变量
    "ref_altitude_km": 350.0,        # 大气密度校准锚点 (km)；non-operating，保留以维持 H_scale 测试 invariant
    # ── (新 PDF) Section 5：长尺度太阳活跃度 domain randomization ──
    # 每 episode reset 时按 log-uniform 抽样 rho_scale，模拟 F10.7 在 11 年太阳周期内
    # 70~250 的变化（350km 密度差 ~8~10x）。完整范围 [-0.7, 0.7] → rho_scale ∈ [0.50, 2.01]。
    # 训练时由 env._randomization_scale (跟课程阶段绑定) 线性缩窄此范围，
    # 在 Exploration 阶段约 ±0.14（rho ~ 0.87~1.15），在 Optimization 阶段使用全幅。
    # robustness.py 中如需精确控制 rho_ref，应先关闭此开关后再注入 trace_scale。
    "enable_solar_activity_randomization": True,
    "solar_activity_log_rho_scale_range": (-0.7, 0.7),
    # ── PDF Section 8.1：大气共转 (Earth-fixed atmosphere co-rotation) ──
    # drag 公式用 v_rel = v_orbit - ω_E·r·cos(i) 而非 v_orbit；51.6° 倾角下 drag 减 ~7.5%
    "enable_atmospheric_corotation": True,
    # ── PDF Section 5：Harris-Priester 日间隆起 (Diurnal Bulge) ──
    # rho(h, Ψ) = rho_base(h) * (1 + α(h)·cos Ψ)；350km 处 α≈0.43 → ρ_M/ρ_m ≈ 2.5x
    "enable_diurnal_bulge": True,
    "diurnal_bulge_lag_rad": 0.5236, # ≈ π/6 = 30°，热层热惯性带来的 ~2h 当地时角滞后
    # ── PDF Section 8.2：地磁暴瞬态密度激增 (Starlink 2022 教训) ──
    # 模拟 G1~G3 级地磁暴下的短临 ρ 突变；完整：peak 2.5x (≈Starlink 2022 +190%)，
    # prob 5e-5 (~0.1 次/episode)。训练时由 env._randomization_scale 线性收缩 peak 上界
    # + 触发概率，Exploration 阶段几乎不触发，Optimization 阶段使用全 2.5x。
    "enable_storm_events": True,
    "storm_probability_per_step": 5e-5,           # 每步触发概率，~0.1 次/2160-step episode
    "storm_duration_steps_range": (30, 180),      # 5~30 min (dt=10s)
    "storm_peak_multiplier_range": (1.3, 2.5),    # 峰值密度乘子；2.5 ≈ Starlink 2022 G1 期 +190%
    "storm_cooldown_steps": 600,                  # 1h 冷却防止连发
    # ── 大气模型切换 + F10.7/Ap 显式驱动 + shape/Cd 不确定度 (domain randomization) ──
    # 全部为隐藏 DR：不进观测向量，state_dim 保持不变，现有 checkpoint 兼容。
    # 默认 'vallado' + F10.7=150 + quiet Ap + 无风暴 → 现有物理完全不变。
    "atmosphere_model": "vallado",      # 'exponential' | 'vallado'(默认) | 'nrlmsise00'(需 pymsis)
    # F10.7 太阳射电流量 (sfu)：每 episode 恒定 (6h 内变化 <1%)。
    # 锚定 f107_nominal=150 → rho_scale=1.0 (复现既有 rho_ref 基线)；
    # log_slope=0.007/sfu 使 [70,250] 映射到旧 log 范围 [-0.7,0.7] (rho_scale∈[0.50,2.01])。
    "f107_nominal": 150.0,
    "f107_range": (70.0, 250.0),        # 11 年太阳周期 F10.7 极值范围
    "f107_log_slope_per_sfu": 0.007,
    "f107_81day_jitter_sfu": 15.0,      # 81-day avg 相对 daily 的抽样抖动幅度
    # Ap 地磁指数：每 episode quiet 基线 + 风暴期三角瞬态抬升 (取代旧直接乘 rho)。
    # quiet → 解析 storm_multiplier=1.0 (静日复现基线)；storm 上界 → ≈storm_peak_multiplier_range 上界。
    "ap_quiet_range": (3.0, 15.0),      # 静日 Ap (Kp 0~3 量级)
    "ap_storm_range": (50.0, 400.0),    # 风暴峰值 Ap (G1~G5 量级)；上界映射到 storm 乘子上界
    # NRLMSISE-00 季节/昼夜几何：每 episode 抽 epoch 年内日序 (绝对年份无关，F10.7/Ap 显式传入)。
    "nrlmsise_epoch_doy_range": (1, 365),
    # satellite shape / Cd 不确定度：每 episode 静态抽样，按 env._randomization_scale 课程缩放。
    # enable 开关默认 True（训练用）；置 False 把 Cd/area 钉死在标称，供 robustness
    # "密度 ×factor" 等需要干净控制 drag 的实验关闭 shape/Cd 噪声（与 solar/β 开关一致）。
    "enable_shape_cd_randomization": True,
    "cd_range": (2.0, 2.4),             # VLEO Cd 真实范围 (PDF Section 8.1)
    "area_uncertainty_fraction": 0.18,  # 迎风面积 ±18% (shape/姿态不确定度)，基准 area_m2
    # ── J2 摄动 ──
    # ground_station.py 已用 J2 做长期 RAAN 漂移预测地面可见窗口；6-hour episode 内
    # RAAN 漂移 ~1.7°、当地时角变化 ~6.7 min，对密度采样几何均可忽略，
    # 故 orbit/altitude 传播不再引入 J2 修正项。
    # ── 已删除项（PDF 中有但 RL 训练不必要） ──
    # - Sentman 变 C_D：归一化后偏差全程 ±3%，agent 行为与静态 Cd=2.2 等价
    # - Bates-Walker T_i：仅用于 Sentman，孤立删除
    # - 热层风 V_wind：方向假设武断，风暴 drag 不确定性已被 storm_multiplier 抓住
}

# ─────────────────────────────────────────────
# 能量系统参数
# ─────────────────────────────────────────────
ENERGY_CONFIG = {
    # ── 电池容量推导：真实 SLATS ~1000Wh，主要为 eclipse 期 (35min) 内 payload+bus+prop
    # 全负荷供能。减去载荷 (SAR ~250W) 和减化 bus (~75W 替代 ~200W)，eclipse 需求降一半。
    # 500Wh = 400Wh usable，覆盖 200km cruise eclipse drain (~270Wh) + 余量。
    "battery_capacity_wh": 500.0,    # 真实 SLATS ~1000Wh 减去载荷 eclipse 配额
    "battery_min_soc": 0.15,         # 电量警告/安全边界 (15%)；低于此值需主动节能
    "battery_crash_soc": 0.05,       # 能源终止失败SOC (5%)
    "battery_max_soc": 0.95,         # 最高SOC (95%)
    # 运行保电缓冲：低于 min_soc+reserve 时，只保留基础载荷/太阳能供电，不再给
    # CPU/TX/非必要推进分配可调功率，避免长训后段 SOC 贴线穿越。
    "battery_operational_reserve_soc": 0.03,
    # ── 推导：真实 SLATS solar ~1000W → 800W (去除载荷峰值发电需求)
    "solar_panel_power_w": 800.0,    # 太阳板峰值发电；移除大载荷后无需 1000W
    "solar_efficiency": 0.28,        # 太阳能转换效率（物理常数）
    # ── 推进：保持真实 SLATS IES Hall thruster 规格 720W。
    # Hall thruster 的最大功率是硬件本身的规格 (推进器尺寸 + PPU 容量决定)，跟卫星上
    # 有没有 SAR 载荷无关。移除载荷 ≠ 推进器自动变大。所以 prop_max 保持真实 720W。
    "power_propulsion_max_w": 720.0, # 真实 SLATS IES Hall thruster nominal 功率
    "propulsion_ignition_threshold_w": 120.0,  # ~17% P_prop_max
    "propulsion_efficiency": 0.65,   # Hall thruster 电-喷流效率（物理常数）
    # Isp=1500s 物理常数不缩 (真实 Hall thruster, SLATS IES 1850s, GOCE T5 3500-4500s)：
    # thrust = 720 × 0.65 / (1500 × 9.80665) = 31.8 mN，配 area=1.0m² (真实 GOCE arrow)
    # 给出 safe cruise 200-300km、burst 180-200km、warning zone <180km 不可恢复
    # (匹配真实 VLEO: GOCE 230-275km nominal, SLATS 167km 仅 tech demo 边缘)。
    "propulsion_isp_s": 1500.0,      # 真实 Hall thruster 比冲 (s)
    # housekeeping 子系统简化绝对值 (无 SAR/光学/相控阵)：
    "power_cpu_max_w": 25.0,         # 星上计算最大任务功率（简化）
    "power_tx_max_w": 35.0,          # 数传发射机最大任务功率（简化，无相控阵）
    "power_baseline_w": 15.0,        # 任务相关基础功耗（简化 bus + ADCS）
    # power_total_max: 覆盖 prop 满负载 + 简化 bus 75W + 25W 余量
    "power_total_max_w": 820.0,      # = prop_max(720) + cpu+tx+base(75) + 25 余量
    "battery_cycle_degradation_enabled": True, # 按等效完整循环累计电池容量老化
    "battery_capacity_loss_per_efc": 2e-4, # 每个等效完整循环损失的容量比例
    "battery_degradation_max_fraction": 0.20, # 单次仿真中最多暴露 20% 可用容量衰退
}

# ── 有限推进剂(氙气)模型 [SAFETY-REAL] ─────────────────────────────
# 真实 VLEO: ~40kg 氙气维持 ~55 个月,耗尽后 ~2 周再入。这里把"月级寿命"压缩进
# 一个 ~6h episode(consumption_scale),让"省燃料 vs 抢数据"的权衡在 episode 内见生死。
# 质量流 mdot = thrust/(Isp*g0) = P_prop*eff/(Isp*g0)^2;燃料=0 → 推力=0 → 阻力主导 → 衰减再入。
PROPELLANT_CONFIG = {
    "enabled": True,
    "initial_mass_kg": 30.0,         # 标定自真实 VLEO 氙气量级(40kg/55mo),按 300kg 整星缩放
    # 统一"轨道时间压缩"C:同时作用于①轨道衰减②推力高度响应③燃料消耗,保证物理一致
    # (稳态 thrust=drag 不随 C 变,只压缩"走向死亡"的瞬态)。标定到:不推进则 episode 内衰减坠毁;
    # 持续维持高轨则 episode 内烧光燃料→断推→衰减坠毁;只有"按需省推 + 高效投递"能活得久。
    "orbital_time_compression": 2800.0,
    # 时间压缩会放大低高度阻力导致的单步高度变化；限幅保留 unsafe→failure 的过渡样本，
    # 避免从 140km 一步跳过 crash 边界，污染安全 critic。
    "max_altitude_delta_m_per_step": 5000.0,
    "reserve_fraction": 0.0,         # 低于该比例视为不可用(可选安全余量)
}

# ── 姿态/指向模型 [SAFETY-REAL] ────────────────────────────────────
# EO 卫星机体同一时刻只能朝一个方向:成像(对地)/下传(对站)/充电(对日)三者互斥。
# IMAGE: 采集原始数据(需昼侧)、耗成像载荷功率、面板偏日(余弦损失);
# DOWNLINK: 窗口内可 TX、面板偏日; SUN: 太阳满输入、不能成像/下传、动量去饱和最快。
# 模式切换=机动:损失部分步时 + 耗能 + 累积动量;动量饱和需强制去饱和(损失一步生产)。
ATTITUDE_CONFIG = {
    "enabled": True,
    "solar_offsun_scale": 0.30,             # 非 SUN 模式太阳输入余弦损失系数
    "imager_power_w": 30.0,                 # IMAGE 模式成像载荷功耗(光学相机)
    "slew_lost_fraction": 0.30,             # 模式切换损失的步内有效时间比例(机动耗时)
    "slew_energy_wh": 0.5,                  # 单次机动耗能(反作用轮)
    "momentum_per_slew": 0.20,              # 单次机动累积的归一化动量
    "momentum_disturbance_per_step": 0.02,  # VLEO 气动/重力梯度扰动每步累积(>默认去饱和,使工作期动量持续累积)
    "momentum_bleed_sun": 0.08,             # SUN 模式每步磁力矩去饱和量(净 -0.06/步,需专门对日去饱和)
    "momentum_bleed_default": 0.01,         # 成像/下传时去饱和慢(净 +0.01/步 → ~100步饱和需去饱和)
    "momentum_max": 1.0,                    # 动量饱和阈值;到达则强制去饱和(该步不能成像/下传)
}

# ─────────────────────────────────────────────
# 解析式推进控制器参数
# ─────────────────────────────────────────────
PROPULSION_CONTROLLER_CONFIG = {
    "enabled": True,                 # 解析推进默认开启：避免 actor 把推进学成长期过烧或完全不烧
    "guard_only": False,             # True 时只在轨道/SOC 临界兜底，供 ablation/debug 使用
    "guard_altitude_margin_km": 5.0, # altitude_min 附近才允许解析推进接管
    "guard_soc_margin": 0.02,        # battery_min 附近才允许解析降推接管
    "target_altitude_km": ORBITAL_CONFIG["altitude_nominal_km"],
    "warning_full_power_km": ORBITAL_CONFIG["altitude_warning_km"],
    "min_altitude_full_power_km": ORBITAL_CONFIG["altitude_min_km"],
    "coast_above_km": 270.0,         # 高于该高度直接滑行，避免 actor 学成长期过推
    "hover_margin": 1.15,            # 维持高度时在阻力补偿功率上留少量裕度
    "sunlit_power_scale": 1.0,
    "eclipse_power_scale": 0.75,     # 阴影期稍降推，优先保护 SOC
    "low_soc_threshold": 0.18,
    "low_soc_power_scale": 0.60,
    "critical_soc_threshold": 0.12,
    "critical_soc_power_scale": 0.0,
    "max_alpha": 1.0,
    "min_ignited_alpha": (
        ENERGY_CONFIG["propulsion_ignition_threshold_w"]
        / ENERGY_CONFIG["power_propulsion_max_w"]
    ),
}

# ─────────────────────────────────────────────
# 热管理参数
# ─────────────────────────────────────────────
THERMAL_CONFIG = {
    "enabled": True,                 # 启用一阶热状态，避免 CPU/Tx 并发满载长期不受热约束
    "initial_temp_c": 20.0,          # 初始星载电子设备温度
    "ambient_temp_c": -20.0,         # 简化等效散热环境温度
    "warning_temp_c": 60.0,          # 45→60：对齐 NASA SMAD / Li-ion 标准。原 45 太保守
    "max_temp_c": 75.0,              # 55→75：性能衰退点，对齐部件 Tj 容限
    "critical_temp_c": 85.0,         # 90→85：对齐工业级 IC 失效温度（85°C 是真实物理 hard limit）
    "thermal_capacity_j_per_k": 18000.0, # 等效热容，决定负载/日照热输入的温升速度
    "electronics_heat_fraction": 0.90,   # [REALISM] 0.35→0.90:电子功耗几乎全转为舱内热(扣 TX 辐射 RF);0.35 不物理,使热从不绑定
    # Hall 推进器大部分功率随喷流离开，只有 PPU/安装耦合损耗进入星体热平衡；
    # 不把推进喷流功率按 electronics_heat_fraction 计入舱内热。
    "propulsion_heat_fraction": 0.04,
    "sunlit_absorbing_area_m2": 0.08,    # 参与热输入的等效受照面积
    "solar_absorptivity": 0.20,          # 外表面对太阳辐照的吸收率
    "radiator_area_m2": 0.22,            # 0.18→0.22：原 18W 等效散热不足，2160 步 episode 下温度爬到 85-90°C
    "radiator_emissivity": 0.92,         # 0.82→0.92：配合 area 提升让平衡温度落回 75°C 以下
    "solar_flux_w_m2": 1361.0,           # 近地太阳常数
    # 调参依据：141k eval thermal_violations=6477，actor 即使在 critical 区也能
    # 跑 25% CPU 继续产热。把 critical_cpu_cap 降到 0.10、warning_min_scale 降到
    # 0.20，让物理层在热警告时硬性拉死 CPU，给热系统更多冷却时间。
    "warning_cpu_tx_min_scale": 0.20,# 默认 0.35 → 0.20（热警告区 CPU/Tx 保留更少）
    "critical_cpu_cap": 0.10,        # 默认 0.25 → 0.10（critical 区几乎切 CPU）
    "critical_tx_cap": 0.0,          # critical 区禁 TX，保留不变
}

# ─────────────────────────────────────────────
# 虚拟队列参数
# ─────────────────────────────────────────────
QUEUE_CONFIG = {
    # 能量虚拟队列
    "energy_queue_max": 100.0,       # 能量虚拟队列上限
    "energy_weight_V": 20.0,         # 李雅普诺夫权重V (能量)

    # 轨道高度虚拟队列
    "orbit_queue_max": 100.0,        # 轨道虚拟队列上限
    "orbit_weight_V": 30.0,          # 李雅普诺夫权重V (轨道)

    # 数据任务队列
    "data_queue_max_mb": 16384.0,    # raw工作队列上限 (MB) [16GB任务缓存]
    "data_arrival_rate_mbs": 5.0,    # 数据到达率基准 (MB/s)，scale=1为受限载荷负载
    "data_service_rate_max_mbs": 8.0, # 星上处理最大速率 (MB/s)
    "data_service_rate_max_mbps": 8.0, # 兼容旧字段名；历史名称实际按 MB/s 使用
    "tx_downlink_rate_max_mbs": 12.5, # 发射机物理下传上限 (MB/s)，约等于100 Mbps
    "tx_capacity_norm_mbps": 100.0,  # 观测归一化用链路容量尺度，与受限发射机同量级
    # ── 通信窗口虚拟队列 ──────────────────────
    "comm_queue_max": 4096.0,          # processed工作队列上限 (MB) [4GB物理上限，agent通过CMDP cost学会控制]
    "comm_weight_V":  15.0,             # 李雅普诺夫权重
    "processed_queue_backpressure_margin_mb": 128.0, # 执行层背压安全余量，避免 CPU 把 processed queue 顶满溢出
}

# ─────────────────────────────────────────────
# 时敏任务价值模型
# ─────────────────────────────────────────────
TASK_CONFIG = {
    "base_value_per_mb": 1.0,        # 单位数据基础任务价值
    "priority_min": 0.1,
    "priority_max": 10.0,
    "quality_min": 0.2,
    "quality_max": 1.5,
    "intrinsic_value_min": 0.01,
    "intrinsic_value_max": 100.0,
    "deadline_min_steps": 60,        # 最短 deadline: 10min (60*10s)
    "deadline_max_steps": 360,       # 最长 deadline: 60min
    "urgent_deadline_steps": 30,     # 5min 内到期视为紧急任务
    "deadline_decay_floor": 0.15,    # 0.05→0.15：超期保留 15% 价值（真实 EO 应急决策仍可用），与 grace_steps 配合给 agent 缓冲
    "deadline_decay_power": 1.0,     # 兼容旧线性时效参数
    "overdue_grace_steps": 15,       # 3→15：超期后 150s(15*10s) 缓冲窗口（原 30s 太短）。
                                     # 注意：这不是一轨道周期(~90min=540 steps)；做成与 deadline(60-360 steps)
                                     # 同量级的短缓冲是刻意的——grace 不应超过任务 deadline 本身。
    "overdue_decay_rate": 4.0,       # 兼容旧字段
    "freshness_floor": 0.0,
    "freshness_default_power": 1.4,
    "specificity_gamma": 1.0,
    "specificity_scale_mb": 256.0,
    "deliverability_reservation_by_class": (0.75, 0.35, 0.0),
    "deliverability_capacity_margin": 0.95,
    "deliverability_bin_count": 8,
    "deliverability_bin_norm_mb": 512.0,
    "deliverability_time_bin_norm_steps": 540.0,
    "top_k": 5,
    "value_norm": 5000.0,            # 状态归一化用价值尺度，随MB级负载提升同步放大
    "scene_lookahead_steps": 6,      # 观测中提前 1min 估计即将进入的任务场景强度
    "orbital_hotspot_strength": 0.35, # 保留轻量轨道相位热点，主要负载仍由场景画像决定
    "randomize_scene_phase_offset": True, # 每个 episode 随机平移场景相位，避免策略死记固定区域顺序
    "scene_phase_offset_max_fraction": 1.0,
    # 仅相位平移不够：phase_scene_rules 的循环顺序仍是固定的，agent 可能学到
    # "看到 X 之后必是 Y" 的 shortcut。开启后每 episode 在保留各场景时长占比
    # （military 仍 4%、urban 仍 3% 等）和场景集合不变的前提下，**打乱块的排列顺序**。
    # agent 必须基于 current_scene_class_norm 即时反应，而非记忆固定循环。
    "randomize_scene_rule_order": True,
    "future_contact_lookahead_s": 5400.0, # 前瞻 1 个约 90min 轨道周期的可下传容量
    "future_contact_scan_step_s": 60.0,
    "time_to_next_window_norm_s": 5400.0,
    # 任务价值分组：状态只暴露压缩直方图，动作按 High/Mid/Low 分配 CPU/TX 预算。
    "class_high_value_density": 2.0,
    "class_medium_value_density": 0.75,
    "class_high_residual_value_density": 3.0,
    "class_medium_residual_value_density": 1.20,
    "protect_nominal_high_value": True,
    "class_high_priority": 2.0,
    "class_medium_priority": 0.8,
    "action_selection_logit_scale": 4.0,
    "work_conserving_reallocation": True,
    "cpu_work_conserving_reallocation": False,
    "tx_work_conserving_reallocation": False,
    "low_value_drop_max_mbs": 0.8,
    "low_drop_expected_processing_ratio": 0.4,
    "low_drop_mid_protection_ratio": 0.25,
    "low_residual_value_density_threshold": 1.20,
    "low_drop_resource_pressure_threshold": 0.40,  # 0.12→0.40：drop 门槛从极严到正常（40% 竞争压力下允许主动丢低价值）
    "low_drop_deadline_urgency_protection": 0.95,
    # 大改：active_low_drop_floor_ratio 0.01 → 0.0
    # 原值会让每步无条件丢 1% droppable backlog，无视 agent 的 drop_strength，
    # 造成 13GB/episode 不可控的 drop，污染 useful 指标和 drop 相关 penalty。
    # 改为 0 之后，drop 完全由 agent 的 drop_strength + 3 个真实压力 driver
    # (capacity / queue_pressure / low_share) 共同决定，agent 才真正"拥有"
    # drop 这个动作。
    "active_low_drop_floor_ratio": 0.0,
    "low_drop_share_target": 0.15,  # 0.05→0.15：target 低价值占比从 5% 拉到 15%（不要求清空，但留 15% headroom）
    # CPU 动作语义：actor 可请求处理额度；环境执行层会按 admissible 可交付 MB
    # 把 alpha_cpu 裁成完成这些工作所需的最小物理功率，避免小额度时满功率空烧。
    "cpu_action_is_admissible_budget": True,
    "enable_future_contact_cpu_gate": True,
    "cpu_gate_start_future_ratio": 0.55,
    "cpu_gate_target_future_ratio": 0.75,
    # 真实短训反馈：ds=1.0 时 safety/energy/proc-dl/win/hi 已过线，但 useful≈0.17，
    # discount_loss 很高。远离窗口时只保留小 processed 缓冲，减少提前处理后的 VoI 等待折损。
    "cpu_gate_far_window_target_ratio": 0.25,
    "cpu_gate_hard_stop_future_ratio": 0.90,
    # Limit the processed buffer to near-term passes, not the whole lookahead horizon.
    # Half a pass leaves a small TX buffer and lets in-window CPU fill the rest.
    "cpu_gate_near_term_passes": 0.5,
    "cpu_gate_far_window_lead_s": 60.0,  # Phase 2 硬规则 H: 120 → 60，处理更靠近窗口减少时效衰减
    "cpu_gate_floor_alpha": 0.0,
    # During an active contact, do not let an empty processed queue starve TX while
    # raw data is waiting; CPU gate still caps the actual processed MB.
    "enable_in_window_cpu_feed_floor": True,
    "in_window_cpu_feed_alpha_floor": 1.0,
    "in_window_cpu_feed_min_raw_mb": 1.0,
    # If the total processed buffer is full of lower-value data, still allow raw
    # high-value data that can catch the next pass to use high-specific headroom.
    # 诊断（diag_high_value.py, ds=1 10ep baseline hi_del=0.191）显示高价值几乎全部在
    # raw 队列等处理时过期（expired_raw_high ≫ expired_proc_high），escape 仅在窗口前
    # 900s 内触发（~30% 步），不够。放宽触发条件，让高价值原始数据更早/更多被预处理：
    #   - lead_s 900→2700（窗口前 45min≈半轨道就允许处理高价值，覆盖其 deadline 跨轨道情形）
    #   - min_deliverable_ratio 0.50→0.25（哪怕只 25% 能赶上下个窗口也处理，胜过全部过期=0%）
    #   - max_mismatch 0.40→0.70（容忍更大的 deadline-窗口错配）
    # proc/dl 当前 1.07、阈值 2.0，useful 0.67，有充足余量吸收多处理的高价值。
    "enable_high_value_cpu_gate_escape": True,
    "high_value_cpu_escape_lead_s": 2700.0,
    "high_value_cpu_escape_min_raw_mb": 1.0,
    "high_value_cpu_escape_min_deliverable_ratio": 0.25,
    "high_value_cpu_escape_max_mismatch": 0.70,
    "high_value_cpu_escape_capacity_fraction": 1.0,
    "high_value_cpu_escape_capacity_margin": 0.95,
    # 兼容旧字段；admissible-budget 模式下不会静默裁剪策略动作。
    "enable_cpu_throttle": True,
    "cpu_throttle_start_utilization": 0.30,
    "cpu_throttle_floor_ratio": 0.20,
    # 轨道相位语义画像：默认只作为合成压力测试先验，不作为实证任务价值标定。
    # 若用于论文主实验，应在结果中声明 synthetic_scene_prior，或替换为外部标定/用户研究配置。
    "scene_model_source": "synthetic_scene_prior",
    "scene_profiles_are_empirical": False,
    "scene_profiles": {
        # base_value_multiplier 与 arrival_multiplier 同时调整：
        # 1) 旧 max/min ≈ 250x，过宽，让单个高价值场景就能压死全 episode 的 value 信号。
        # 2) 高价值场景以前 arrival_multiplier 也最大，等于同时贡献「最多 MB + 最贵」，
        #    叠加 44% 场景占比 → 价值流量被高价值场景垄断。现在拉低高价值的 arrival，
        #    保留 ordering（仍 military > urban > routine > ocean）但不再线性放大。
        # 3) hump 场景 freshness_peak_fraction 从 0.25~0.35 后移到 0.45~0.50，
        #    deadline 同步放宽，让 agent 有 10~20min 真实窗口完成 process+downlink。
        "cloud_ocean": {
            "class_code": 0.05,
            "arrival_multiplier": 0.20,
            "base_value_multiplier": 0.06,
            "priority_range": (0.08, 0.18),
            "quality_range": (0.20, 0.50),
            "deadline_range_steps": (240, 420),
            "cloud_cover_range": (0.75, 1.00),
            "cloud_penalty": 0.85,
            "freshness_profile": "early",
            "freshness_power": 1.8,
        },
        "open_ocean": {
            "class_code": 0.20,
            "arrival_multiplier": 0.40,
            "base_value_multiplier": 0.18,
            "priority_range": (0.18, 0.45),
            "quality_range": (0.55, 0.90),
            "deadline_range_steps": (180, 360),
            "cloud_cover_range": (0.20, 0.65),
            "cloud_penalty": 0.40,
            "freshness_profile": "early",
            "freshness_power": 1.3,
        },
        "routine_land": {
            "class_code": 0.45,
            "arrival_multiplier": 0.90,
            "base_value_multiplier": 0.65,
            "priority_range": (0.55, 1.10),
            "quality_range": (0.70, 1.05),
            "deadline_range_steps": (120, 300),
            "cloud_cover_range": (0.05, 0.35),
            "cloud_penalty": 0.30,
            "freshness_profile": "late",
            "freshness_power": 0.8,
        },
        "urban": {
            "class_code": 0.65,
            "arrival_multiplier": 1.00,
            "base_value_multiplier": 1.20,
            "priority_range": (1.20, 2.20),
            "quality_range": (0.80, 1.15),
            "deadline_range_steps": (90, 240),
            "cloud_cover_range": (0.00, 0.30),
            "cloud_penalty": 0.25,
            "freshness_profile": "late",
            "freshness_power": 0.55,
        },
        "disaster": {
            "class_code": 0.85,
            "arrival_multiplier": 1.10,
            "base_value_multiplier": 2.20,
            "priority_range": (2.50, 4.50),
            "quality_range": (0.85, 1.20),
            "deadline_range_steps": (150, 300),
            "cloud_cover_range": (0.00, 0.35),
            "cloud_penalty": 0.35,
            "freshness_profile": "hump",
            "freshness_peak_fraction": 0.50,
            "freshness_late_floor": 0.20,
        },
        "military": {
            "class_code": 1.00,
            "arrival_multiplier": 1.20,
            "base_value_multiplier": 3.20,
            "priority_range": (3.20, 6.00),
            "quality_range": (0.90, 1.20),
            "deadline_range_steps": (150, 330),
            "cloud_cover_range": (0.00, 0.25),
            "cloud_penalty": 0.25,
            "freshness_profile": "hump",
            "freshness_peak_fraction": 0.50,
            "freshness_late_floor": 0.25,
        },
        "emergency_disaster": {
            "class_code": 1.00,
            "arrival_multiplier": 1.60,
            "base_value_multiplier": 4.50,
            "priority_range": (5.50, 10.00),
            "quality_range": (0.90, 1.20),
            "deadline_range_steps": (120, 240),
            "cloud_cover_range": (0.00, 0.30),
            "cloud_penalty": 0.30,
            "freshness_profile": "hump",
            "freshness_peak_fraction": 0.45,
            "freshness_late_floor": 0.15,
        },
        "polar_cloud": {
            "class_code": 0.12,
            "arrival_multiplier": 0.25,
            "base_value_multiplier": 0.08,
            "priority_range": (0.10, 0.30),
            "quality_range": (0.25, 0.60),
            "deadline_range_steps": (180, 360),
            "cloud_cover_range": (0.65, 1.00),
            "cloud_penalty": 0.80,
            "freshness_profile": "early",
            "freshness_power": 1.6,
        },
    },
    # 高价值场景占比从 ~44% 调整到 ~10%，更贴近真实地球观测分布：
    # 海洋 ~50%、常规陆地/极地云 ~33%、城市 6%、灾害 3%、军事 4%（含 0.20 相位锚点）。
    # 测试需要 phase=0.20→military、phase=0.90→cloud_ocean，下面保留这两个相位锚点。
    "phase_scene_rules": [
        {"start": 0.00, "end": 0.18, "scene": "open_ocean"},
        {"start": 0.18, "end": 0.22, "scene": "military"},
        {"start": 0.22, "end": 0.34, "scene": "open_ocean"},
        {"start": 0.34, "end": 0.50, "scene": "routine_land"},
        {"start": 0.50, "end": 0.60, "scene": "polar_cloud"},
        {"start": 0.60, "end": 0.74, "scene": "open_ocean"},
        {"start": 0.74, "end": 0.77, "scene": "disaster"},
        {"start": 0.77, "end": 0.85, "scene": "routine_land"},
        {"start": 0.85, "end": 0.88, "scene": "urban"},
        {"start": 0.88, "end": 0.95, "scene": "cloud_ocean"},
        {"start": 0.95, "end": 1.00, "scene": "open_ocean"},
    ],
    # 随机突发灾害事件：低概率覆盖当前场景一小段时间，形成高价值、短 deadline 的应急任务。
    # 旧参数 (p=0.001, dur=18~48, cd=180) 在 2160-step episode 里平均触发 ~2 次、每次叠加 10x
    # base_value_multiplier，是 exp_high_val 飙升的主要来源之一。
    # 现在: 触发概率 -> 0.0003 (≈0.65 次/episode)、单次更短、冷却更长，让 emergency 真正成为稀有事件。
    "emergency_event_enable": True,
    "emergency_event_scene": "emergency_disaster",
    "emergency_event_probability_per_step": 0.0003,
    "emergency_event_duration_steps": (12, 30),
    "emergency_event_cooldown_steps": 360,
}

# ─────────────────────────────────────────────
# 李雅普诺夫投影参数
# ─────────────────────────────────────────────
LYAPUNOV_CONFIG = {
    "V_penalty": 50.0,               # drift-plus-penalty 中交付收益的权重
    "lipschitz_bound": 10.0,         # compute_lyapunov_bound 使用的保守漂移上界
    # ── state-dependent Lyapunov projection (Chow et al. 2018 NeurIPS) ──
    # L(s) 各通道权重；所有通道归一到 [0,1]，硬失败时 L 自然 ≥ 1。
    "L_altitude_weight": 0.5,          # 高度从不出事（orbit_safe_rate=1.0），降权
    "L_soc_weight": 1.3,               # SOC 是 crash 次因（energy_safe_rate=0.78），升权
    # 调参依据：141k eval seed=42 的 lyapunov_proj_rate=38.8%，actor 一直被投影；
    # 但 crash 元凶是 thermal/energy 不是 queue：thermal_violations=6477, energy=5511,
    # queue 完全没事。所以下调 queue 通道、上调 thermal/SOC 通道。
    "L_processed_queue_weight": 0.4,   # 默认 1.0 → 0.4（queue 完全没出事）
    "L_raw_queue_weight": 0.2,         # 默认 0.5 → 0.2
    "L_thermal_weight": 0.8,           # 默认 0.5 → 0.8（撤销上一轮 0.3 的错误降权，反而要升权）
    "L_future_capacity_weight": 0.3,
    # 投影松弛 ε(s) = max(0, d - L(s)) * decay。
    # L < d 时允许少量上升（让 actor 不被压死在零边界）；越靠近不安全集越紧。
    "L_target_level": 1.0,             # 默认 0.5 → 1.0（slack 更宽）
    "L_slack_decay": 0.10,             # 默认 0.05 → 0.10
    # 投影器超参。
    "projection_finite_diff_eps": 1e-2,
    "projection_max_iter": 3,
    "projection_feasibility_tol": 1e-3,
    "enabled": True,                 # IntegratedScheduler.enable_lyapunov 的默认值
}

# ─────────────────────────────────────────────
# 深度强化学习参数
# ─────────────────────────────────────────────
# 当前训练口径标识。
# 修改 reward/状态/训练语义后要同步更新，防止主训练入口误续训不兼容 checkpoint。
OBJECTIVE_VERSION = "delivered_voi_cmdp_rlfirst_delivered_plus_td_rawexec_diag_rootcause"

DRL_CONFIG = {
    "algorithm": "SAC",              # 基础算法 SAC，配合约束 Critic (CMDP Lagrangian) + Lyapunov 投影 + PSF 安全层
    "state_dim": 65,                 # 63 + pointing_mode_norm + momentum_norm(姿态/指向状态);物理 + 任务价值直方图 + 可达性前瞻 + 燃料 + 姿态
    "action_dim": 15,                # 解耦布局[推进, CPU预算, TX, CPU类logits×3(高/中/低), TX类logits×3(高/中/低),
                                     #          CPU价值权重, CPU紧迫权重, TX价值权重, TX紧迫权重, 低价值丢弃, 指向模式(IMAGE/DOWNLINK/SUN)]
                                     # 旧 9 维把 class logit 与 value/urgency 权重混用同一下标，已拆开（见 utils/action_space.py）
    "hidden_dim": 512,               # 隐藏层维度（增强网络容量）
    "network_arch": "transformer",   # transformer | mlp；MLP 只用于 backbone 消融
    "lr_actor": 1.0e-4,              # 从零训练的 base LR。⚠warm-start 续训会从 checkpoint 恢复 optimizer+cosine
                                     # 调度器状态（见 agent.load 行 1005-1027），实际 LR 由恢复的调度器接管、本值对续训不生效：
                                     # RECOVER_backup3_320k 续训起点≈7.7e-5（T_max=125k cosine，已过 31%），随训练缓降。
    # 调参依据：141k eval reward CV=1.77, std=41k, reward_min=-135k 极端坏 episode 频出，
    # 训练不稳的强信号。Critic 学习率降到 2/3，让 Q 估值更平滑；actor 保持不变
    # 避免拖慢策略适应。
    "lr_critic": 1.0e-4,             # 同 lr_actor：from-scratch base LR；warm-start 续训由恢复的调度器接管。
    "lr_alpha": 1e-4,                # 熵系数学习率（保持探索自适应能力）
    "gamma": 0.99,                   # 0.995→0.99：远期信号放大配合 critic 发散导致 Q 失稳；先稳住再考虑抬高
    "reward_shaping_coeff": 0.0,     # 大改：暂时关掉 potential-based shaping。
                                     # Ng 1999 理论上不改最优策略，但 Φ 在过渡期可能给出
                                     # 反方向梯度（queue 增长时 phi_next>phi_prev 给正 shaping，
                                     # 鼓励 agent 继续填队列）。和现在的"少处理"目标冲突。
                                     # 等 useful>0.3 之后再尝试升回 0.1。
    "tau": 0.005,                    # 目标网络软更新系数（更慢的更新，更稳定）
    "batch_size": 512,               # 批量大小（增加批量以改进梯度质量）
    "buffer_size": int(6e5),         # 经验回放缓冲区(>=总训练步数→等价全历史;2e6 配 65维×8帧会一次性占数GB致OOM)
    "warmup_steps": 4000,            # 随机探索预热步数
    # A+ 档训练加速：n_envs=8 下原 update_freq=4 意味着 2 个 env step 就更新一次，
    # 4 critic+actor 全过一遍。改 8 让 GPU 反向传播负载减半，样本不变。
    "update_freq": 8,                # 4 → 8
    # 【学习率调度】使用余弦退火确保训练后期充分收敛
    "lr_schedule_type": "cosine",    # 学习率调度类型: constant | cosine | exponential
    "lr_min_scale": 0.01,            # 最小学习率比例（最终 lr = lr_initial * lr_min_scale）
    "update_actor_freq": 1,          # Actor/alpha每次critic更新都更新；优先保证策略能跟上队列反馈
    "gradient_clip": 1.0,            # 10.0→1.0：actor_loss 涨到 5000+ 时旧阈值形同虚设，硬刹车防发散
    # A+ 档训练加速：4070 Ti SUPER 全程 FP32 浪费算力。打开 AMP 配合
    # nan_guard_enable + gradient_clip 风险可控，1.5-2x 加速。
    "use_amp": True,                # False → True
    "state_normalization": True,     # SAC 网络输入使用 RunningMeanStd 动态归一化，环境观测本身保持物理语义
    "state_norm_epsilon": 1e-4,
    "state_norm_clip": 5.0,
    "nan_guard_enable": True,        # 启用更新阶段非有限值守卫
    "alpha_log_clip": 10.0,          # log_alpha 裁剪范围 [-clip, clip]
    "alpha_min": 0.05,               # 0.01→0.05：上一轮 alpha 跌到 0.17 并仍在下行，下限抬高保持探索
    # 缩放参数（用于稳定训练与约束强度可解释性）
    # - reward_scale: 将环境奖励缩放到 O(1)
    # - lyapunov_drift_scale: Lyapunov 漂移缩放（默认 1.0，不再额外 ÷100）
    "reward_scale": 50.0,            # 100→50：相对约束代价的量纲过强，actor 易被奖励侧拉偏
    # TD reward 口径（Day2 reward-TD 消融，见 docs/问题.docx B-2）。一律不含安全壳动作惩罚
    # （projection_penalty/action_mod_penalty/r_actuator_violation），只在“任务目标”口径间切换：
    #   "primary"        → primary_mission_reward（仅交付价值 r_value，用于消融/回归）
    #   "env_total"      → 环境返回的完整 reward（含全部 shaping）
    #   "delivered_plus" → r_delivered_value + r_deadline_success + r_expired_penalty + r_window_underuse
    "td_reward_mode": "delivered_plus",
    "lyapunov_drift_scale": 1.0,
    # A+ 档：原 20 在 raw_cost ~37 时严重 saturating，约束信号被削平。
    # 抬到 40 让 thermal/energy 大幅违规能完整传到 critic。
    "lyapunov_drift_clip": 40.0,
    # 安全层动作惩罚：让 actor 学会“原始动作也尽量可行”，而不是长期依赖投影兜底
    "projection_penalty_coeff": 4.0,
    "action_mod_penalty_coeff": 1.0,
    "safety_action_penalty_cap_ratio": 0.25,
    "safety_action_penalty_min_cap": 3.0,
    # Actor 辅助模仿最终安全执行动作，避免长期依赖安全投影兜底。
    "behavior_cloning_coeff": 0.05,
    "behavior_cloning_max_weight": 1.0,
    "behavior_cloning_conservative_weight_coeff": 0.25,
    # [2026-06-03] 指向兜底改写动作时的 BC 权重:把执行动作(含纠正后指向)作为强模仿目标，
    # 让 actor 逐步自己学会任务指向，而非长期依赖兜底脚手架。0=关闭该增强。
    "mission_pointing_bc_weight": 0.8,
    # raw high 在 ds=1 中主要死于 raw 队列等待处理；这里给 actor 一个窄 BC 目标：
    # 当 raw high 可赶上下个窗口时，主动提高 alpha_cpu 和 CPU high/urgency logits。
    "enable_high_value_cpu_behavior_cloning": True,
    "high_value_cpu_bc_min_raw_mb": 1.0,
    "high_value_cpu_bc_min_deliverable_ratio": 0.25,
    "high_value_cpu_bc_max_mismatch": 0.70,
    "high_value_cpu_bc_lead_s": 2700.0,
    "high_value_cpu_bc_alpha_target": 0.95,
    "high_value_cpu_bc_high_logit_target": 1.0,
    # 8-D compact action uses index 4 both as CPU urgency and as the medium-class
    # softmax axis in diagnostics.  For raw-high rescue, keep it low so decoded
    # CPU allocation is truly high-first instead of high+medium.
    "high_value_cpu_bc_urgency_logit_target": 0.0,
    "high_value_cpu_bc_raw_norm_mb": 400.0,
    "high_value_cpu_bc_base_weight": 0.85,
    "high_value_cpu_bc_min_weight": 0.35,
    "high_value_cpu_bc_max_weight": 1.0,
    # 价值辅助头默认关闭：其标签来自状态规则，只能用于诊断/预训练消融。
    "value_aux_head_enable": False,
    "value_aux_num_classes": 3,          # [high_first, balanced, low_drop]
    "value_aux_loss_weight": 0.0,
    "value_aux_loss_weight_final": 0.0,
    "value_aux_weight_decay_steps": 900000,
    "value_action_aux_loss_weight": 0.0,
    "value_action_aux_loss_weight_final": 0.0,
    "value_aux_processed_future_contact_threshold": 0.75,
    # 默认训练口径：reward critic 学远期交付，deliverable critic 学近端处理投资回报，constraint critic 学安全代价。
    "deliverable_critic_enable": True,
    # 调参依据：141k eval comm_window_utilization=32%，tx_active_in_contact=50%——
    # actor 在窗口里没在下传。加大 deliverable critic 在 actor loss 中的权重，
    # 让"近端可下传"信号在策略梯度里更显眼。
    "deliverable_critic_actor_coeff": 0.5,          # 1.0→0.5：actor_loss 主要来源，回收一半权重
    "deliverable_critic_target_key": "processed_deliverable_value_step",
    "lyapunov_penalty_coeff": 0.3,
    "adaptive_lyapunov_coeff_enable": True,
    # threshold: 允许的平均归一化约束代价上界。
    # norm 从 10 缩到 3 后 dual_signal 相对 raw_cost 约 3.3x，threshold 同步放大到 0.30
    # 以维持相同的"约束容忍度"（等效 raw_cost 允许值 = 0.30 × 3 = 0.9，约等于原 0.10×10=1.0）。
    "adaptive_lyapunov_constraint_threshold": 0.30,  # 0.10 → 0.30（与 norm 10→3 同步调整）
    "adaptive_lyapunov_coeff_target_pressure": 0.10,
    "adaptive_lyapunov_coeff_lr": 0.003,  # 0.01→0.003：降速 3.3x，给 actor 时间适应约束
    "adaptive_lyapunov_coeff_ema_beta": 0.97,   # 0.99→0.97：EMA 略快响应（dual 现在由 EMA 驱动，需足够灵敏又平滑）
    "adaptive_lyapunov_coeff_kd": 0.15,         # 0.05→0.15：增强微分阻尼 3x，防止 dual 快速爬升压崩 actor
    "adaptive_lyapunov_coeff_min": 0.3,
    # Run17 修正：dual 撞 2.0 顶后 actor_loss 60→1600 diverge。
    # 降到 1.2 让安全权重温和，配合降速 lr 和增强 kd 确保稳定。
    "adaptive_lyapunov_coeff_max": 1.2,  # 2.0→1.2：防止 dual 过高压崩 actor
    # Dual update uses normalized CMDP cost c_t / norm, not PSF/projection rate.
    # D 改动：关掉队列/orbit 后 raw_cost 最大值大约从 25 缩到 ~6（thermal+energy+over_processing+state_penalty+task_loss）。
    # 把 norm 同步从 10 → 3，dual signal 占比保持 ~50%，让 λ 真的能 ramp up。
    "adaptive_lyapunov_constraint_norm": 3.0,    # 默认 10.0 → 3.0
    "adaptive_lyapunov_constraint_signal_max": 3.0,
    "projection_ema_beta": 0.995,
    # checkpoint 选择时把安全层介入代价折算为 MB 惩罚，避免只按 reward/下传挑到“靠投影兜底”的模型
    "checkpoint_proj_penalty_mb": 1800.0,
    "checkpoint_projected_penalty_mb": 1200.0,
    "checkpoint_action_mod_penalty_mb": 250.0,
    "checkpoint_max_proc_downlink_ratio": 2.0,
    "checkpoint_max_processed_queue_final_utilization": 0.85,
    "checkpoint_max_processed_queue_future_contact_ratio": 0.95,
    "checkpoint_max_cpu_far_from_window_rate": 0.25,
    "checkpoint_max_energy_violation_rate": 0.0,
    # energy_viol=0 只说明没有跌破安全线；best checkpoint 还必须把电量换成足够 VoI。
    # 旧 ds=1.0 可行运行约 0.09 Wh/VoI，这里留一点裕度，拒绝明显“高下传但高耗电”的模型。
    "checkpoint_max_energy_per_value": 0.12,
    "checkpoint_min_useful_processing_ratio": 0.30,
    "checkpoint_min_comm_window_utilization": 0.70,
    "checkpoint_min_high_value_delivery_ratio": 0.30,
    # 队列风险惩罚（注入到 Lyapunov 漂移信号）：
    # - 软惩罚：利用率超过阈值后按二次项增长，促使策略提前降风险
    # - 硬惩罚：真实 overflow（丢包/积压越界）按比例强惩罚
    # D 改动：141k eval raw_queue_safe=1.0 / processed_queue_safe=1.0，队列从不出事，
    # 队列代价反而压住了 thermal/energy 的信号。关掉队列软硬代价，让 dual signal
    # 由 thermal/energy/over_processing 主导。
    "lya_soft_util_threshold": 0.75,
    "lya_soft_util_penalty_coeff": 0.0,   # 默认 0.5 → 0.0（队列从不出事，关掉）
    "lya_soft_penalty_clip": 1.0,
    "lya_hard_overflow_penalty_coeff": 0.0,  # 默认 3.0 → 0.0（队列从不 overflow，关掉）
    "lya_hard_penalty_clip": 2.0,
    # 【重构后】CMDP cost 只保留 4 大主项,移除互相打架的旧 cost。
    # 恢复温和的 over_processing_cost，配合较低的 adaptive_lyapunov_coeff_max 防止 collapse
    # ── 顶刊 Issue#1: 干净 CMDP 约束语义（默认 False，保持现有训练/checkpoint 可比性）──
    # True 时，constraint critic 与 adaptive dual 的代价 = 仅物理安全+队列稳定
    #   (orbit + energy + thermal + queue + 硬状态安全 + Lyapunov drift)，
    # 把 QoS 项 (task_loss + over_processing + low_value_waste + unproductive_cpu)
    # 从 safety critic 中剔除，改作 reward shaping / 次级指标。
    # 论文主公式只把 constraint_total 作为 CMDP cost。切到 True 需重训（口径变化）。
    "clean_constraint_cost_enabled": True,
    "enable_capacity_aware_cost_v2": True,
    "enable_deliverable_processing_reward": False,
    "queue_projection_policy": "safety_algorithms_only",
    "enable_deployment_queue_projection": True,
    # A+ 档：原 8.0/15.0/3.0 让 over_processing 占了总 cost 96%，把 thermal/SOC 信号
    # 完全压扁。调回温和水平，让 stage_costs 提到 0.5/3.0/10.0 后能在 dual 里
    # 真正起作用。actor 仍能感受到"处理多于可下传"的信号，只是不再独霸。
    "constraint_over_processing_coeff": 1.0,        # 1.5→1.0：只保留容量超处理主约束，避免压过交付奖励
    "constraint_over_processing_clip": 6.0,         # 9.0→6.0
    "constraint_over_processing_ratio_weight": 1.0, # 1.2→1.0：降低 proc/dl 约束斜率，减少与 gate 的重复拉扯
    "constraint_capacity_norm_mb": 400.0,
    "constraint_capacity_norm": 400.0,  # 兼容旧脚本
    "constraint_future_capacity_margin": 0.70,      # 保持
    "constraint_efficiency_processed_value_credit": 0.0,
    # ── 物理状态硬安全约束(阶段化代价 + 热超限 + 能量边界 + 轨道边界)。
    # A+ 档：放手 PSF 同时让 thermal/SOC 自己有发声权。原 0.08/0.8/3.0 在
    # over_processing_cost=40 面前只占 1%，actor 完全感受不到过热代价。
    "constraint_stage_costs": {"warning": 0.5, "unsafe": 3.0, "failure": 10.0},
    "constraint_auxiliary_violation_cost": 0.6,
    # 调参依据：141k eval crash 30/30，热和能量是元凶。原 coeff=0.25 让 cost
    # 信号过弱，actor 学不到"过热要付出代价"。翻倍 coeff，让 constraint critic
    # 对热/能量警告反应更敏锐。
    "constraint_thermal_excess": {"coeff": 1.0, "norm_c": 10.0},  # 0.50→1.0：诊断显示热崩是 100% crash 元凶，强化训练信号
    "constraint_energy_margin_coeff": 0.50,                        # 默认 0.25 → 0.50
    "constraint_energy_margin_clip": 1.5,                          # 默认 1.0 → 1.5
    # D 改动：orbit_safe_rate=1.0，轨道从不出事。关掉这条，让 thermal/energy 信号纯净。
    "constraint_orbit_margin_coeff": 0.0,   # 默认 0.25 → 0.0
    "constraint_orbit_margin_clip": 1.0,
    # cpu_active_strictly_far_rate 的"很远"判定阈值（仅用于诊断日志，不进 reward）。
    # 主 reward shaping (r_proc_far_window) 用 proc_far_window_lead_s = 120s 作为起点，
    # 这里 300s 作为更严的二级警告：agent 半轨道内不该有 CPU 活动。
    "constraint_prepass_min_lead_s": 300.0,
    # ── 高价值任务过期/丢弃约束(平滑启用)。
    # 调参依据：141k eval 显示 cpu_requested_mid=0.495 > high=0.362，actor 学到了
    # 给 mid 而不是 high 让路（mid 截止期更长容易交付）。过期高价值 35/episode，
    # 高价值交付率仅 11%。把惩罚系数 2x、clip 1.5x，让"丢一个 high"比"丢一个 mid"
    # 在 cost critic 里量级显著拉开。
    "constraint_high_value_loss_coeff": 2.5,   # 默认 1.0 → 2.5
    "constraint_task_loss_value_norm": 5000.0,
    "constraint_task_loss_clip": 3.0,           # 默认 2.0 → 3.0
    "constraint_task_loss_warmup_steps": 30000,
    "constraint_task_loss_anneal_steps": 120000,
    "constraint_task_loss_min_scale": 0.0,
    # Historical aliases kept for old checkpoints/scripts (compatibility only).
    "constraint_warning_cost": 0.5,
    "constraint_unsafe_cost": 3.0,
    "constraint_failure_cost": 10.0,
    "constraint_thermal_warning_cost": 0.20,   # 默认 0.08 → 0.20（热警告代价 2.5x）
    "constraint_thermal_excess_coeff": 1.0,    # 0.50→1.0：与上面 dict 保持一致
    "constraint_thermal_excess_norm_c": 10.0,
    "constraint_power_violation_cost": 0.25,
    # ── 已废弃的旧 cost 项参数(全部置 0,只为兼容旧 import / 旧 checkpoint)。
    # 不要在新实验里调它们;它们的语义已经被 over_processing_cost 与 reward 覆盖。
    "lya_processed_backlog_coeff": 0.0,
    "lya_processed_backlog_threshold": 0.08,
    "lya_processed_backlog_clip": 0.0,
    "constraint_processed_backlog_coeff": 0.0,
    "constraint_processed_backlog_threshold": 0.08,
    "constraint_processed_backlog_clip": 0.0,
    "constraint_low_value_waste_coeff": 0.0,    # 低价值浪费交给 class-aware gate / drop 规则，不再作为单独 cost
    "constraint_low_value_waste_clip": 2.0,     # 3.0→2.0
    "constraint_low_value_waste_norm_mb": 5.0,
    "constraint_unproductive_cpu_coeff": 0.0,   # 远窗口 CPU 浪费已由 admissible CPU gate 处理，避免双重惩罚
    "constraint_unproductive_cpu_clip": 1.0,    # 2.0→1.0
    "constraint_unproductive_cpu_far_window_s": 300.0,
    "constraint_window_waste_coeff": 0.6,       # 0.1→0.6：6× 强化——4b 显示 win_util 崩到 0.22，必须强 pull TX
    "constraint_window_waste_clip": 2.0,        # 0.8→2.0
    "constraint_efficiency_cost_coeff": 0.0,
    "constraint_efficiency_cost_clip": 0.0,
    # 调参依据：141k 日志 alpha=0.054 已经很低（actor 接近确定性），但 delivered_value
    # 仍在涨说明还有探索空间。把目标熵从 -8 (-action_dim·1.0) 抬到 -4 (-8·0.5)，
    # 让 alpha 维持稍高水平、actor 保留更多探索。
    "target_entropy_scale": 0.8,     # 0.5→0.8：上一轮 alpha 跌至 0.17 且仍在下行，提高目标熵避免过早塌缩
    # 帧堆叠长度：Transformer 时序输入窗口（steps），8步×10s=80秒历史
    # compare_all.py / ablation.py / robustness.py 均通过此值获取 stack_len
    "frame_stack": 8,
    # Dilated 时序采样偏移（单位：step），需与位置编码保持一致
    "dilated_offsets": [0, 1, 3, 9, 27, 90, 270, 540],
    # Transformer 分支只读取随时间演化且对调度有前瞻意义的观测字段。
    "transformer_temporal_features": [
        "solar_input_norm",
        "in_comm_window",
        "time_to_next_window_norm",
        "window_remaining_norm",
        "tx_capacity_norm",
        "processed_queue_utilization",
        "processed_high_queue_utilization",
        "processed_mid_queue_utilization",
        "processed_low_queue_utilization",
        "processed_queue_future_contact_ratio",
        "deadline_urgency",
        "expiring_high_value_norm",
        "expiring_mid_value_norm",
        "expiring_low_value_norm",
        "prop_update_phase",
        "current_scene_class_norm",
        "upcoming_task_intensity_norm",
        "future_contact_capacity_norm",
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
        "concurrent_high_same_class_mb_norm",
        "concurrent_medium_same_class_mb_norm",
        "concurrent_low_same_class_mb_norm",
    ],
}

# ─────────────────────────────────────────────
# 预测安全滤波器参数
# ─────────────────────────────────────────────
INFERENCE_MPC_CONFIG = {
    # 推理时 short-horizon shooting planner（包裹 SAC actor）。
    # 不改训练，只在 schedule() 时介入。
    #
    # score_mode="critic" (默认): 主信号 = critic(obs, candidate)，value-aware；
    #   rollout 仅用于"硬安全过滤"（reject 踩到 crash 边界的候选）。
    #   这是修掉 141k 步 A/B 显示的 "MPC 拖累 delivered_value" 问题后的版本。
    # score_mode="delivered_mb" (旧): 用累计下传 MB 打分，value-blind，保留作回归。
    "enabled": False,                # 默认关；通过 IntegratedScheduler(use_inference_mpc=True) 打开
    "score_mode": "critic",          # critic | delivered_mb
    "num_candidates": 32,            # 候选 action 数（含 actor mean）
    "horizon_steps": 10,             # rollout 视野；100 秒（dt=10s）足够覆盖一个 contact 子段
    "noise_std_physical": 0.20,      # 物理三维（prop/cpu/tx）扰动 std
    "noise_std_priority": 0.10,      # 优先级权重维扰动 std
    # critic 模式专用：
    "override_margin": 0.05,         # best critic 比 actor 高出此阈值才接管，防止噪声驱动过度 override
    # 调参依据：141k eval 显示 crash 30/30，元凶 thermal+energy。MPC rollout 时
    # 应该把可能踩进热/SOC 警告区的候选直接 reject，而不是等贴 crash 才动。
    "reject_altitude_m": 130_000.0,  # 高度其实从不出事，阈值保留默认即可
    "reject_soc": 0.20,              # 默认 0.10 → 0.20（贴 soc_min 就拒）
    "reject_queue_util": 0.95,       # 默认 0.99 → 0.95（更早拦截）
    "reject_thermal_margin": 0.05,   # 默认 -0.95 → 0.05（thermal 进入警告区即拒，硬变化）
    # delivered_mb 模式专用（旧）：
    "delivered_weight": 1.0,         # 内部 reward 中的 delivered_mb 系数
    "constraint_weight": 1.0,        # 约束违反惩罚系数
    "terminal_weight": 1.0,          # 终端 critic Q 系数
    # 共用：
    "include_safe_anchor": True,     # 是否额外加入"全推力 / 全下传"两个极端 anchor
    "force_use_mpc_output": False,   # True → 即使 best_idx=0 也标记 used=True（便于 ablation）
    "gamma": 0.99,                   # MPC 内部折扣；不必跟 DRL gamma 一致
    "warmup_steps": 50_000,          # 训练步数门槛：到此之前不启用 MPC（让 actor 先学起来）
}

N_STEP_CONFIG = {
    # n-step TD targets。target_Q = R_n + γ^n * Q(s_{t+n}, a_{t+n})
    # 标准 SAC 扩展（Hessel et al. Rainbow / Bellemare distributional 等都用）。
    # off-policy bias 在 n ≤ 10 时实践中可控（无需 Retrace）。
    # 调参依据：n=5 只覆盖 50 秒，γ=0.995 在 540 步外信号衰减到 ~6.7%（有效视野约 540 步）。
    # 把 n 升到 10（100 秒）后，"远窗口处理 → 拿 reward"信号传播链从 episode 起点到
    # 当前步所需的 TD bootstrap 次数减半，信号传播更快。n 再大就要 Retrace 防 off-policy bias，先不动。
    "enabled": True,
    "n": 10,                         # 默认 5 → 10
    "discard_short_episode_tail": False,  # episode 末尾不足 n 时也用现有累积 reward（自然 truncation）
}

PSF_CONFIG = {
    # PSF 是最终物理兜底层，触发阈值要早于硬失败边界，但晚于常规可学习区间。
    # 高度 165km / SOC 20% 会在接近安全边界前做短视野 rollout，
    # 避免只在贴近硬边界时才介入。
    "enabled": True,                 # IntegratedScheduler.use_psf 的默认值
    # 调参依据：141k eval 显示 psf_filter_rate=0% 但 crash 30/30。重要发现：
    # orbit_safe_rate=1.0（高度从不出事），crash 元凶是 thermal (6477 violations)
    # 和 energy (5511 violations)。PSF 主要应该在 SOC 低 / 热高时介入，
    # 高度阈值保持默认即可（不需要 30km 那么宽）。K=10 (100s) 给 PSF 看到
    # 一段轨道的能力，对推理延迟影响适中（边缘部署关注）。
    # A+ 档：PSF 瘦身 + 放手让 actor 自己学。原 horizon=10 + long_horizon=540
    # + line_search=6 + robust 检查，单步 PSF 评估很重。
    "horizon_steps": 5,              # 10 → 5（短视野 rollout 减半）
    "line_search_steps": 3,          # 6 → 3（二分搜索减半）
    "altitude_trigger_margin_m": 15_000.0,  # 撤回上轮的 30_000，恢复默认（高度不是问题）
    # A+ 档：原 0.15 让 SOC<0.30 就拦，agent 90% 时间被 PSF 接管学不会自己管电量。
    # 降到 0.05 让真正贴 soc_min=0.15 才拦，剩下时间让 reward/dual 自己学。
    "soc_trigger_margin": 0.15,             # C 修复：0.05 → 0.15 恢复安全底线(更早拦截能量崩，配合过推惩罚双保险)
    # ── C 改动：解析式长视野预测（让 PSF 看到 K 步以外的慢漂移）──────
    # K=10 步 = 100 秒，但 thermal_capacity=18000J/K + 116W 缺口意味着
    # 温度上升常数是 ~分钟级，SOC 漂移更是 ~小时级。把 first_action 沿用
    # long_horizon_steps 步（默认 540=一个轨道周期）做线性外推，看 thermal/SOC
    # 是否会在窗口前进 warning，是就把 raw 视为不安全。
    # A+ 档：关掉 540 步线性外推 —— 这是 PSF 88% 触发的主因，"将来可能漂移"
    # 全被拦了。改靠 thermal/SOC 的 reward + dual 信号让 actor 自己提前避免。
    "long_horizon_enabled": False,              # True → False
    "long_horizon_thermal_margin_floor": 0.10,  # 预测 540 步后 thermal_margin 不能低于 0.10
    "long_horizon_soc_floor": 0.20,             # 预测 540 步后 SOC 不能低于 0.20（贴 soc_min=0.15+margin）
    # 只有明显处于正常区间时才跳过 rollout。高度阈值与 180km warning 上界对齐。
    "passthrough_altitude_margin_m": 30_000.0,
    "passthrough_soc_margin": 0.08,
    "robust_altitude_margin_m": 30_000.0,
    "long_horizon_steps": 540,
    "long_horizon_altitude_margin_m": 60_000.0,
    "long_horizon_violation_margin_m": 5_000.0,
    # A+ 档：训练阶段关掉鲁棒扰动检查（让 PSF 更"乐观"，actor 多见违规自己学）。
    # eval 时可以临时把这几项调回来做鲁棒性测试。
    "robust_density_perturb_range": 0.0,        # 0.50 → 0.0
    "robust_solar_power_scale": 1.0,            # 0.80 → 1.0
    "robust_battery_capacity_scale": 1.0,       # 0.85 → 1.0
    "robust_propulsion_thrust_scale": 1.0,      # 0.85 → 1.0
}

# ─────────────────────────────────────────────
# 训练参数
# ─────────────────────────────────────────────
TRAIN_CONFIG = {
    "total_steps": 540000,           # 从零课程训练的完整预算（curriculum 5 阶段 + 末段长 ds=1.0）。
                                     # 这是论文复现/泛化的主训练路径：run_all_experiments.py → train.py（无 --resume_path）。
                                     # 可选 warm-start 微调：--resume_path <ckpt> + --total_steps(base_steps + 微调步)。
    # A+ 档：原 20k freq × 5 ep × 2160 步 = 单次 eval 最多 10800 步，1M 训练里
    # 触发 50 次 eval = 等于又跑了 50w 步。改 50k freq × 3 ep 省掉一大半。
    "eval_freq": 50000,              # 长跑（540k）控制 eval 开销；正式 best 验证另用 evaluate_optimized.py 多 episode
    "eval_episodes": 3,              # 训练内 eval 用 3 ep 控制开销；终评再用 5+
    "save_freq": 50000,              # 模型保存频率
    "keep_step_checkpoints": False,   # 默认只保留 best/latest，避免生成大量中间模型文件
    "log_freq": 500,                 # 日志记录频率
    "max_episode_steps": 2160,       # 每episode最大步数 (=4个90min轨道, dt=10s)，覆盖跨轨道资源规划
    "update_freq": 4,                # 每4步采样后触发一次网络更新（DRL_CONFIG["update_freq"]=8 是每次更新的 gradient steps，含义不同）
    "time_slot_s": 10,               # 时间片长度 (秒)
    "seed": 42,
    # 顶刊 Issue#6: evaluation seeds ≥20（原 5 太少）。experiments/multi_seed.py 默认读取这里；
    # 每方法总评估 episode ≥ eval_episodes(终评建议≥5) × len(eval_seeds) ≥ 100。
    "eval_seeds": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                   52, 53, 54, 55, 56, 57, 58, 59, 60, 61],
    # 顶刊 Issue#6: 训练随机种子（≥5）用于报告 train-time 方差，不只是 eval 方差。
    # multi_seed.py --mode train-eval --seeds 取此列表（每个 seed 独立训练一个模型）。
    "train_seeds": [42, 43, 44, 45, 46],
    "n_envs": 6,                     # 多环境采样默认；对齐 6 物理核 CPU（Ryzen 5 9600X），
                                     # 留 1 核给学习/系统，避免 >物理核数 的超订争抢。
                                     # 训练量锚定全局环境步，改 n_envs 不影响 total_steps/update_freq/UTD。
    "env_backend": "auto",           # auto: n_envs>1 时启用子进程环境；serial: 调试用串行环境
    "optimized_checkpoint_dir": "checkpoints_optimized/",
    "optimized_log_dir": "logs_optimized/",
    "fail_fast_on_nan": False,       # 允许少量NaN告警，避免一次抖动中断训练
    "nan_guard_max_hits": 10,        # NaN告警累计上限（达到后停止）
    
    # 多阶段课程学习：从低负载平滑提升到完整负载；训练入口会对阶段边界做线性 ramp，
    # 避免任务到达率硬跳变导致策略崩溃。
    "use_curriculum": True,   # 从零训练必开：ds 0.5→0.7→0.85→1.0→长 ds=1.0 的渐进课程。
                              # 既当冷启动脚手架（配合在线 AP-BC 模仿安全投影动作），又让策略见过低负载（泛化），
                              # 末段长时间 ds=1.0 保证目标场景收敛且不遗忘。这是生成 RUN15 级策略的正确配方。
    # 【关键修复 — 真正的崩溃根因】train_random_ds_enabled=True 会让每个 env 每 episode 在
    # [0.5,1.0] 里"均匀随机"抽 ds，从而 *绕过并架空* 上面精心设计的 curriculum（见 train.py 行 2207：
    # 开启时不再按阶段 set_data_scale）。Run17/540k 就是 use_curriculum=False + 全程均匀随机 ds，
    # 在 ds<1.0 样本上反复更新把 ds=1.0 安全策略冲掉 → safe 99%→0%、crash、reward -156k 遗忘崩溃。
    # 正确做法：泛化靠 curriculum（渐进多 ds），不是破坏性的全程均匀随机。故默认关闭此项。
    "train_random_ds_enabled": False,
    "train_random_ds_range": (0.5, 1.0),  # 仅在显式做"全程均匀多 ds"消融时才置 True
    # randomization_scale 控制 env._randomization_scale，影响 rho/β/storm 三项随机化幅度。
    # 阶段间用与 data_arrival_scale 同样的线性 ramp 平滑过渡，避免分布硬跳变。
    # Exploration: 0.20 → rho×[0.87,1.15], β≤15°, storm prob 1e-5/peak~1.54
    # Balancing:   0.45 → rho×[0.73,1.37], β≤34°, storm prob 2.2e-5/peak~1.84
    # Ramp:        0.75 → rho×[0.59,1.69], β≤56°, storm prob 3.8e-5/peak~2.20
    # Optimization:1.00 → 完整 PDF 物理极值
    "curriculum_stages": [
        # Run 12 修复：curriculum 阶段间 linear ramp 会让 Final 永远到不了 constant ds=1.0
        # 改成：Final 之前快速 bridge 到 1.0，然后真正在 ds=1.0 训练 300k 步
        {
            "stage_name": "Adapt_50",
            "steps": 30000,
            "lyapunov_weight_scale": 0.75,
            "data_arrival_scale": 0.50,
            "randomization_scale": 0.45,
            "description": "warm-start 入口",
        },
        {
            "stage_name": "Adapt_70",
            "steps": 50000,
            "lyapunov_weight_scale": 0.80,
            "data_arrival_scale": 0.70,
            "randomization_scale": 0.55,
            "description": "ramp 0.5→0.7",
        },
        {
            "stage_name": "Adapt_85",
            "steps": 60000,
            "lyapunov_weight_scale": 0.90,
            "data_arrival_scale": 0.85,
            "randomization_scale": 0.70,
            "description": "ramp 0.7→0.85",
        },
        {
            "stage_name": "Bridge_100",
            "steps": 20000,
            "lyapunov_weight_scale": 0.95,
            "data_arrival_scale": 1.0,
            "randomization_scale": 0.90,
            "description": "快速 ramp 0.85→1.0",
        },
        {
            "stage_name": "Final",
            "steps": 300000,
            "lyapunov_weight_scale": 1.0,
            "data_arrival_scale": 1.0,
            "randomization_scale": 1.0,
            "description": "ds=1.0 constant（无 ramp，因为前一阶段 target 也是 1.0）",
        }
    ],
    
    # 【学习率预热】前期缓慢增加学习率
    "use_lr_warmup": True,
    "lr_warmup_steps": 4000,
    "lr_warmup_init_scale": 0.1,
    
    # 【学习率调度】余弦衰减确保后期充分收敛
    "lr_schedule": "cosine",         # constant | cosine | exponential
    "lr_min_ratio": 0.001,           # 最小学习率 = lr_init * lr_min_ratio
    
    # 【探索参数衰减】
    "alpha_schedule": "exponential_decay",
    "alpha_init": 0.2,               # 初始熵系数
    "alpha_final": 0.01,             # 最终熵系数
    "alpha_decay_steps": 1000000,
}

# ─────────────────────────────────────────────
# 实验协议记录
# ─────────────────────────────────────────────
EXPERIMENT_PROTOCOL = {
    "hyperparameter_search": {
        "enabled": True,
        "method": "predefined_grid_then_fixed_protocol",
        "selection_metric": "safety_adjusted_delivered_value",
        "requires_multi_seed_report": True,
        "search_space": {
            "reward_scale": [50.0, 100.0, 200.0],
            "lyapunov_drift_clip": [6.0, 12.0, 20.0],
            "projection_penalty_coeff": [1.0, 2.0, 4.0],
            "constraint_over_processing_coeff": [0.0, 1.0, 2.0],
            "constraint_future_capacity_margin": [0.75, 0.80],
            "w_deliverable_processing_final": [0.0],
        },
    },
    "scene_model_source": TASK_CONFIG["scene_model_source"],
    "scene_profiles_are_empirical": TASK_CONFIG["scene_profiles_are_empirical"],
    "scene_model_note": (
        "Default scene profiles are synthetic stress-test priors; paper claims "
        "must report this or replace TASK_CONFIG scene profiles with calibrated inputs."
    ),
}

# ─────────────────────────────────────────────
# 奖励函数权重
# ─────────────────────────────────────────────
# reward 结构:
#   r_t = w_v * delivered_value
#         + w_on * on_time_delivered_value
#         + r_processing_penalty  (容量门控分段惩罚，见 mission_reward.py)
#         + r_drop_penalty + r_drop_mb_penalty
#         + r_energy_penalty
#
# 处理惩罚设计原则（容量门控）：
#   headroom = processing_capacity_margin * future_capacity_mb - processed_queue_mb
#   headroom 内：w_processing_penalty_useful  * MB  （小惩罚，仅反映电量）
#   超出 headroom：w_processing_penalty_overflow * MB  （强惩罚，约等于失去半个 VoI）
#
# 权重校准目标（按典型 step 数量级：delivered_value~10, energy_wh~0.05, processed_mb~5）：
#   正信号 r_delivered_value ≈ +5.0 * 10 = +50（窗口内）
#   单步 penalty 总和必须 << 正信号期望，否则长 horizon Q 累积成大负值，agent 放弃下传。
# 之前 overflow=-1.0、energy=-5.0、drop_mb=-0.1 同时叠加，per-step 惩罚 30~100，压死正信号
# → r_step 持续 -7~-9、val 转负、proc/dl 飙到 59。这次把 penalty 全线下调 10x，
#   并降低 CMDP over_processing_coeff 让它和 reward overflow penalty 不再重复打。
PAPER_REWARD_CONFIG = {
    # A2 class-weighted reward：让 critic 对 high 类的梯度独立于稀疏采样频次。
    # value_density 已经放大了类间差 ~200x，但 replay buffer 里 high 样本只占 ~10%，
    # 显式 class 权重让 high reward 在 actor loss 里再多一份"梯度强度"。
    "enable_class_weighted_reward": True,
    # 诊断（ds=1 10ep baseline hi_del=0.191）：高价值几乎全在 raw 队列等处理时过期，
    # 纯放宽 escape gate 对策略零效果（actor cpu_req_high=0.279 偏低，gate 开大也不处理）。
    # 原训练时 gate 挡住高价值预处理 → actor 学到“请求高价值 CPU 无用”→ 低 alpha_cpu。
    # 现 gate 放宽给了行动空间，需续训让策略利用它；同时把高价值交付奖励权重 3.0→4.5
    # 增强“多交付 high”的策略梯度拉力（配合既有 w_expired_penalty=-1.5 的推力）。
    "class_high_reward_weight": 4.5,
    "class_mid_reward_weight": 1.5,
    "class_low_reward_weight": 0.5,
    # 调参依据：A2 把 r_value 改成 w_v·(3 v_h + 1.5 v_m + 0.5 v_l)，相比原来
    # w_v·delivered_value 在典型 high 主导的步上会膨胀 ~3x。把 w_v 从 5.0 降到 2.0
    # 让总量级跟原来近似（高 class 步的 r_value 维持 ~30 而不是飙到 ~65），
    # critic Q 学习不被突变冲击。
    "w_delivered_value": 2.0,           # 默认 5.0 → 2.0（A2 类加权配套）
    "w_deadline_success": 0.5,

    # w_processing_deliverable_value: 0.3 -> 0.0
    # 关闭正向处理奖励：只要 deliver_prob > 0，处理就有正收益，agent 始终倾向满负荷 CPU。
    # 去掉后处理只有负信号 (opportunity_cost)，不再有"处理越多越好"的梯度。
    # w_processing_opportunity_cost: 0.5 -> 1.0 (ds=1.0 finetune)
    # RUN15 在 ds=1.0 下 proc/dl 仍偏高，opportunity_cost ≈ -2.5/step 相对 r_value ~30 太弱；
    # 2x 加大让 "处理下不去的高价值数据" 真疼，逼 agent 学会节约 CPU。
    "w_processing_deliverable_value": 0.0,
    "w_processing_opportunity_cost": 0.0,  # 处理是否可交付交给 gate/over-processing cost，reward 不再额外扣

    # 普通能耗进入 cost critic；reward 只在超过每步预算时给很小的软代价。
    # 调参依据：141k eval 显示 solar=345W, 总负载=461W (含 prop=411W)，长期亏空
    # 116W → 22% 时间 SOC 在 warning。把超预算惩罚 2x，迫使 actor 学会在
    # 高度有 headroom 时关推进省电。energy_budget 略调小到 0.18，让边界更紧。
    # w_energy_penalty: 0.0 → -0.05 (ds=1.0 finetune)
    # 该项在 mission_reward.py 中直接加到 reward 上，因此负数才表示能耗惩罚。
    # 它只提供弱信号，避免 agent 在没有交付收益时仍倾向无谓 CPU/TX/推进。
    # [RECOVER 2026-06-03] 姿态兜底修复 raw_queue 断流后，恢复极弱能耗信号；
    # 比 -0.05 小一个数量级，避免再次压过稀疏交付奖励。
    "w_energy_penalty": -0.005,
    "w_energy_over_budget_penalty": 0.0,   # 能源风险统一进入 CMDP cost，reward 保持交付主信号
    "energy_budget_wh_per_step": 0.18,     # 默认 0.22 → 0.18
    # ── C 修复(过推惩罚)：推进功率超过 prop_overburn_threshold_w 的部分线性扣分。
    # Evidence C：维持 250km 仅需 ~83W、点火门限 120W，但 agent 学成烧 ~411W 平均推进
    # → 热崩(>405W 散不掉)+能崩(>309W 净放电)，同源。阈值 150W 给轨道维持留足余量，只罚"多余"推进。
    # 量级：411W 时 excess≈261W × 0.02 ≈ -5.2/step，持续过推一条 episode ~-1.1万(占 reward~18%)，
    # 足以改变行为又不主导。首次 eval 后按 e_viol / 高度安全率校准此权重。
    "w_prop_overburn_penalty": 0.003,       # 弱过推惩罚：解析推进负责维轨，reward 只抑制明显多烧
    "prop_overburn_threshold_w": 150.0,

    # 旧惩罚项关闭，避免长期负反馈把 Q 学成“什么都不做”。
    "w_processing_penalty_useful": 0.0,
    "w_processing_penalty_overflow": 0.0,
    "processing_capacity_margin": 0.95,
    "w_drop_penalty": 0.0,
    "w_drop_mb_penalty": 0.0,
    # w_expired_penalty: 0.0 -> -0.3 -> -1.0
    # 注意符号约定：mission_reward.py 直接乘 (r_expired_penalty = w * expired_value)，
    # 未在代码里加负号 (与 w_processing_opportunity_cost 不同)。权重必须为负才是惩罚。
    # 参考 w_energy_over_budget_penalty = -0.5 的同款约定。
    # -0.3 信号太弱，exp_high_val 仍高达 ~200K，-1.0 让过期惩罚与 r_delivered_value 量级相当，
    # 迫使 agent 优先高价值 + 及时下传而不是无限积压。
    # 141k eval 复盘：expired_high_value=35/ep，actor 仍偏好 mid（cpu_mid=0.495>high=0.362）。
    # -1.0 还是不够痛，再 1.5x。配合 w_delivered_value=5.0，"丢 1 个高价值"≈ -1.0 × value_per_unit
    # 跟"交付 1 个"5.0×value 相抵，让 actor 真正算清账。
    "w_expired_penalty": -1.5,                  # 默认 -1.0 → -1.5
    # ── B 修复(只修奖励归因)：过期罚分只作用在"可控的高价值过期"上，不再惩罚结构上
    # 无法交付的低价值(海洋)过期(ds=1.0 下占 ~83%、与策略无关)。True=只罚高价值过期(新口径)；
    # False=回到旧的全量 expired_value(raw+proc) 行为。w_expired_penalty 仍保持 -1.5。
    "expired_penalty_high_value_only": True,
    "w_prospective_expiry_shaping": 0.0,
    "w_actuator_violation_penalty": 0.0,
    # ── 远窗口处理连续 shaping (修复 120s gate / 300s far_cpu 指标的 gap) ──
    # 之前 r_proc_far_window 是死代码（写死 0.0），far_cpu 只是日志指标，
    # agent 收不到"不要在远窗口处理"的 reward 信号 → 远窗口处理活跃度 ~46%。
    # 现在：处理 1MB 远窗口数据，按 (t_to_window - lead_s)/(sat_s - lead_s) 线性
    # 增加 penalty，没有 cliff。typical step 处理 20MB、远 strength=1.0 → -0.5 reward
    # （相对 r_step ~32 的 1.5% 量级，足够引导但不主导）。
    # w_proc_far_window_penalty: 0.025 → 0.06 (ds=1.0 finetune)
    # 远窗口处理 20MB × strength=1.0 → 之前只 -0.5/step，相对 r_value~30 的 1.5%；
    # 2.4x 加大到 -1.2/step（4%），让 "远窗口别处理" 真正进 actor gradient。
    "w_proc_far_window_penalty": 0.01,       # 弱时机 shaping：远窗口处理 20MB 约 -0.2
    "proc_far_window_lead_s": 60.0,          # Phase 2 H: 与 cpu_gate_far_window_lead_s 同步 (120 → 60)
    "proc_far_window_saturation_s": 600.0,   # 远 480s 后饱和（约半轨道）

    # ── 窗口期 TX 闲置惩罚 (ds=1.0 finetune 新增) ──
    # 之前没有显式 "窗口期吃满 tx" 的正向梯度，agent 没动力卡满 alpha_tx。
    # 逻辑：in_window 且 processed_queue_mb >= min_queue_mb 时，
    #   idle_mb = max(0, max_tx_mb * target_ratio - actual_tx_mb)
    #   penalty = w * idle_mb
    # typical: max_tx_mb=80, target=0.85 → 目标 68MB；若 agent 只下 30MB，
    # idle=38 → -0.04 × 38 = -1.5/step (in_window 时累积 ~30step ≈ -45/window)。
    # 量级介于 r_value 和 opportunity_cost 之间，足够把 alpha_tx 推向 1.0。
    # [2026-06-03] 0.01 → 0.03：诊断 trace 显示 477 窗口步中 82% 零下传、窗口 alpha_tx 仅 0.34。
    # 唯一"奖励窗口内下传"的信号，针对"窗口不下传"病加大；w_energy_penalty 保持极小以防回到躺平。
    "w_window_underuse_penalty": 0.03,         # 弱窗口利用 shaping：有货不下传时给可学习梯度
    "window_underuse_min_queue_mb": 5.0,       # processed_queue 太空时不罚（没货可下）
    "window_underuse_target_ratio": 0.85,      # 目标利用 85% 链路容量
}

# ────────────────────────────────────────────────────────────────────
# CPU gate soft mode 开关。
#
# 实测教训：直接打开 soft mode 会让 agent 沿用旧策略输出 alpha_cpu≈1，
# 实际处理量爆炸，r_step 从 -9 跌到 -110，整个训练失稳。
#
# 当前策略（阶段 1）：False — 保留 hard gate 做稳定脚手架。
# 在新的 4 项 sparse reward 下，agent 学到的策略是"何时让 gate 都不需要介入"
# （即自己输出小 alpha_cpu）。等 useful>0.3 / proc/dl<2 之后切换到 True，
# 在 sparse reward 已经塑形完成的基础上"撤掉脚手架"，验证策略迁移性。
# ────────────────────────────────────────────────────────────────────
ACTUATOR_GATE_CONFIG = {
    # Phase 1 硬规则 F：关掉 soft mode，hard gate 在远窗口强制 clip α_cpu。
    # 之前用 soft mode 让 actor 自学 → far_cpu=53% 半轨道 CPU 都在白处理。
    # 硬切回 False 直接让 53%→<10%，agent 行为完全由 gate 决定，可解释。
    "cpu_gate_soft_mode": False,  # True → False (Phase 1 硬规则)
}

# ────────────────────────────────────────────────────────────────────
# Phase 1 硬规则配置（A/D/E）
#
# 这些不是 reward shaping，是 env/scheduler 层的 *机制*，把"显然正确的行为"
# 用代码兜底，让 RL 只学真正的 trade-off。理由（见 docs/findings.md）：
#   - reward 调权要 30~60 分钟训练才看到效果，硬规则即刻生效
#   - 硬规则可解释、可回退、不会像 Run17 那样训出毛病
# ────────────────────────────────────────────────────────────────────
HARD_RULES_CONFIG = {
    # 规则 A: process_by_priority 中 deliver_prob < min 的批次直接跳过。
    # RUN15 诊断：useful_processing_ratio=11%，89% 处理白干。
    # 阈值 0.3 = "至少 30% 概率能在 deadline 前送出去" 才允许处理。
    "min_deliver_prob_for_processing": 0.30,
    "enable_deliver_prob_gate": True,

    # 规则 B: 处理排序加 class 优先级 floor (high → mid → low)，actor 的 value_weight
    # 只在同类内决定细排序。RUN15 诊断：low_processing_ratio=58.7%。
    "enable_class_priority_floor": True,

    # 规则 D: deliver_by_priority 预留 70% tx 给 high 类，剩 30% 自由。
    # RUN15 诊断：high_value_delivery_ratio=29.6%。
    "tx_high_reserve_fraction": 0.70,
    "enable_tx_high_reserve": True,

    # 规则 E: env.step() 窗口期 alpha_tx 硬 floor，processed_queue 有货时强制吃满链路。
    # RUN15 诊断：comm_window_utilization=67.2%。
    "in_window_alpha_tx_floor": 0.95,
    "in_window_floor_min_queue_mb": 5.0,
    "enable_in_window_tx_floor": True,

    # 规则 F: 任务姿态兜底。训练日志显示后期策略会塌缩到 SUN/DOWNLINK，
    # 导致昼侧不成像、raw_queue 长期为 0，随后 CPU gate 又把 CPU 执行动作压成 0。
    # 该规则只在物理安全且热/电有余量时生效；危险状态仍完全交给安全层保命。
    "enable_mission_pointing_fallback": True,
    "mission_pointing_raw_low_mb": 1.0,
    "mission_pointing_min_thermal_margin": 0.20,
    # [2026-06-03] 昼侧持续成像的 raw 队列利用率上限:raw_util < 此值且昼侧 → 强制 IMAGE。
    # 取代旧的 "raw<=1MB 饿死才成像" 门槛(导致 daylit 89% 落入 no_task_need)。
    "mission_pointing_raw_room_util": 0.8,

    # 规则 C: 分层 EDF。class_priority_first sort key 变成 (class, tight, -score)。
    # 让 "deadline 紧的 high" 插到 "deadline 松的 high" 之前救命，但
    # "deadline 紧的 low" 仍排在 "deadline 松的 high" 之后——不破坏 class 优先级。
    # Phase 1 结果：hi_del 30%→31% 没动，根因 high 在 raw_queue 等过期。
    "enable_layered_edf": True,
    "edf_tight_deadline_steps": 10,    # deadline_remaining ≤ 10 步（100s）视为 tight

    # 规则 G: class-aware deliverability gate。低价值任务要求 deliver_prob 更高才处理。
    # 缘由：deliver_prob × value = 期望交付价值。同一 deliver_prob 下高价值期望产出大，
    # 值得"赌"；低价值期望产出小，CPU 花同样的电不划算。
    # Phase 1 useful_processing_ratio=17%（目标>30%），主要因为低价值占 processed_value 比重大。
    "enable_class_aware_gate": True,
    "min_deliver_prob_high": 0.30,     # high 宽松——deliver_prob 30% 就值得搏
    "min_deliver_prob_medium": 0.50,
    "min_deliver_prob_low": 0.70,      # low 严格——70%+ 才处理
}

# 训练引导项（processing credit）:
# w_deliverable_processing_initial/final 均置 0 ——不再给 processing 加正向分。
# _deliverable_processing_credit 的计算值仍保留并作为 observation 特征暴露给 actor，
# agent 能"看见"当前处理是否有意义，但不因处理本身得分，避免强化"满 CPU"倾向。
PROCESSING_CREDIT_CONFIG = {
    # 完全关闭 processing 正向分，保留参数只用于消融/诊断。
    "w_deliverable_processing_initial": 0.0,
    "w_deliverable_processing_final": 0.0,
    "deliverable_processing_credit_warmup_steps": 20_000,
    "deliverable_processing_credit_anneal_steps": 80_000,
    "deliverable_processing_credit_cap_fraction": 0.08,
    "deliverable_processing_near_window_s": 120.0,
    "deliverable_processing_max_future_ratio": 0.45,
    "deliverable_processing_min_high_gate": 0.75,
    "deliverable_processing_mid_value_weight": 0.0,
    "deliverable_processing_mid_gate_floor": 0.25,
}

REWARD_CONFIG = PAPER_REWARD_CONFIG.copy()

# ─────────────────────────────────────────────
# 地面站配置
# ─────────────────────────────────────────────
GROUND_STATION_REGIONAL = [
    {"lat": 39.9, "lon": 116.4, "name": "GS-Beijing"},
    {"lat": 31.2, "lon": 121.5, "name": "GS-Shanghai"},
    {"lat": 23.1, "lon": 113.3, "name": "GS-Guangzhou"},
    {"lat": 1.35, "lon": 103.82, "name": "GS-Singapore"},
    {"lat": 35.68, "lon": 139.76, "name": "GS-Tokyo"},
]

GROUND_STATION_GLOBAL = [
    {"lat": 39.9, "lon": 116.4, "name": "GS-Beijing"},
    {"lat": 31.2, "lon": 121.5, "name": "GS-Shanghai"},
    {"lat": 23.1, "lon": 113.3, "name": "GS-Guangzhou"},
    {"lat": 78.23, "lon": 15.40, "name": "GS-Svalbard"},
    {"lat": -72.01, "lon": 2.54, "name": "GS-Troll"},
    {"lat": 68.32, "lon": -133.55, "name": "GS-Inuvik"},
    {"lat": 64.98, "lon": -147.51, "name": "GS-Alaska"},
    {"lat": 67.89, "lon": 21.06, "name": "GS-Kiruna"},
    {"lat": 27.76, "lon": -15.63, "name": "GS-Maspalomas"},
    {"lat": 37.94, "lon": -75.47, "name": "GS-Wallops"},
    {"lat": 19.01, "lon": -155.67, "name": "GS-Hawaii"},
    {"lat": -33.45, "lon": -70.67, "name": "GS-Santiago"},
    {"lat": -53.16, "lon": -70.91, "name": "GS-PuntaArenas"},
    {"lat": -31.80, "lon": 115.89, "name": "GS-Perth"},
    {"lat": 1.35, "lon": 103.82, "name": "GS-Singapore"},
    {"lat": 35.68, "lon": 139.76, "name": "GS-Tokyo"},
    {"lat": -25.89, "lon": 27.71, "name": "GS-Hartebeesthoek"},
    {"lat": 5.24, "lon": -52.77, "name": "GS-Kourou"},
    {"lat": 50.00, "lon": 5.15, "name": "GS-Redu"},
]

GROUND_STATION_PROFILES = {
    "regional": GROUND_STATION_REGIONAL,
    "global": GROUND_STATION_GLOBAL,
}

GROUND_STATION_CONFIG = {
    "profile": "global",            # 训练/评估默认使用全局站网，打开所有配置的地面站
    "profiles": GROUND_STATION_PROFILES,
    "stations": GROUND_STATION_GLOBAL,
    "min_elevation_deg": 5.0,        # 最低可见仰角
    "bandwidth_mhz":     100.0,      # 通信带宽
    "atmospheric_refraction_enabled": True, # 低仰角可见性使用简化大气折射修正
    "acquisition_latency_steps": 2,  # 进入窗口后的建链/捕获延迟步数
    "acquisition_latency_min_scale": 0.25, # 建链初期容量折减下限
    # 自适应调制编码（AMC）：根据 SNR 离散选择 MCS 档位，而不是连续 Shannon 容量。
    "amc_enabled": True,
    "amc_snr_thresholds_db": [-3.0, 3.0, 8.0, 13.0, 18.0],
    "amc_spectral_efficiencies": [0.25, 0.5, 1.0, 2.0, 3.0],
    "amc_capacity_levels_mbps": [0.0, 10.0, 50.0, 120.0, 150.0, 220.0],
    "max_channel_capacity_mbps": 300.0, # 单站瞬时链路容量封顶，避免顶端仰角无限放大
    "max_downlink_mb_per_pass": 800.0,  # 单次过顶地面端可接收/调度的总容量上限
    # [REALISM] 链路真实性修正
    "fixed_link_loss_db": 4.0,          # 固定链路余量损耗(实现损耗+馈线+极化+指向),从 EIRP 扣除
    "coding_efficiency": 0.80,          # FEC/成帧/协议开销 → 有效 goodput < 原始信道速率
}

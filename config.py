"""
全局实验配置入口。

本模块集中定义轨道、能量、任务队列、地面站、SAC/CMDP 训练和评估参数。
其他模块应从这里读取超参数，避免在算法、环境或实验脚本中散落硬编码常量。
"""

# ─────────────────────────────────────────────
# 轨道参数
# ─────────────────────────────────────────────
ORBITAL_CONFIG = {
    "altitude_warning_km": 180.0,    # 轨道警告区上界 (km)；150~180km 需主动恢复高度
    "altitude_min_km": 150.0,        # 不安全轨道边界 (km)；低于此值为严重再入风险
    "altitude_crash_km": 122.0,      # 不可恢复再入/坠毁终止边界 (km)
    "altitude_nominal_km": 350.0,    # 标称轨道高度 (km)
    "altitude_max_km": 450.0,        # 最高轨道高度 (km)
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
    "Cd": 2.2,                       # 阻力系数 (VLEO 范围 2.0~2.4，PDF Section 8.1)
    "area_m2": 2.5,                  # 卫星迎风面积 (m^2) [中型VLEO遥感平台的紧凑迎风面积]
    "mass_kg": 383.0,                # 卫星质量 (kg) [JAXA SLATS/TSUBAME量级，贴近VLEO阻力补偿平台]
    "rho_ref": 4.89e-11,             # 参考大气密度 @ 350km (kg/m^3)；相对 Vallado 标准 6.660e-12 ≈ 7.3x，约 F10.7=200~250 高太阳活跃期
    "H_scale_km": 50.0,              # ref_altitude 段局部 scale height (km)；表内其他段独立校准
    "ref_altitude_km": 350.0,        # 参考高度 (km)
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
    "battery_capacity_wh": 300.0,    # 任务级可调能源预算 (Wh)，用于制造能源调度压力
    "battery_min_soc": 0.15,         # 电量警告/安全边界 (15%)；低于此值需主动节能
    "battery_crash_soc": 0.05,       # 能源终止失败SOC (5%)
    "battery_max_soc": 0.95,         # 最高SOC (95%)
    "solar_panel_power_w": 120.0,    # 分配给任务调度器的峰值发电预算 (W)
    "solar_efficiency": 0.28,        # 太阳能转换效率；峰值功率已是发电输出，此项保留作面积模型参考
    # 各子系统功耗 (W)
    "power_propulsion_max_w": 90.0,  # 推进系统最大任务功率
    "propulsion_ignition_threshold_w": 30.0, # 电推启动门限；低于该功率视为未点火
    "propulsion_efficiency": 0.65,   # 电推电功率到喷流功率的等效效率
    "propulsion_isp_s": 1000.0,      # 电推比冲 (s)
    "power_cpu_max_w": 25.0,         # 星上计算最大任务功率
    "power_tx_max_w": 35.0,          # 数传发射机最大任务功率
    "power_baseline_w": 15.0,        # 任务相关基础功耗
    "power_total_max_w": 120.0,      # 任务级电源管理总功率上限 (W)
    "battery_cycle_degradation_enabled": True, # 按等效完整循环累计电池容量老化
    "battery_capacity_loss_per_efc": 2e-4, # 每个等效完整循环损失的容量比例
    "battery_degradation_max_fraction": 0.20, # 单次仿真中最多暴露 20% 可用容量衰退
}

# ─────────────────────────────────────────────
# 热管理参数
# ─────────────────────────────────────────────
THERMAL_CONFIG = {
    "enabled": True,                 # 启用一阶热状态，避免 CPU/Tx 并发满载长期不受热约束
    "initial_temp_c": 20.0,          # 初始星载电子设备温度
    "ambient_temp_c": -20.0,         # 简化等效散热环境温度
    "warning_temp_c": 45.0,          # 进入热降额区
    "max_temp_c": 55.0,              # 热安全上限，超过后 overall_safe=False
    "critical_temp_c": 65.0,         # 极端过热区，用于强制关断 Tx/压低 CPU
    "thermal_capacity_j_per_k": 18000.0, # 等效热容，决定负载/日照热输入的温升速度
    "electronics_heat_fraction": 0.35,   # 可调负载转化为舱内热的比例
    "sunlit_absorbing_area_m2": 0.08,    # 参与热输入的等效受照面积
    "solar_absorptivity": 0.20,          # 外表面对太阳辐照的吸收率
    "radiator_area_m2": 0.18,            # 等效散热面积
    "radiator_emissivity": 0.82,         # 红外发射率
    "solar_flux_w_m2": 1361.0,           # 近地太阳常数
    "warning_cpu_tx_min_scale": 0.35,# 热警告区 CPU/Tx 最低保留比例
    "critical_cpu_cap": 0.25,        # 严重过热时 CPU 动作上限
    "critical_tx_cap": 0.0,          # 严重过热时禁止下传
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
    "deadline_decay_floor": 0.05,    # 交付时效主要由场景 freshness 曲线决定，仅保留很小的过期尾巴
    "deadline_decay_power": 1.0,     # 兼容旧线性时效参数
    "overdue_grace_steps": 3,        # 短尾巴只用于避免刚超时一步奖励断崖
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
    "low_drop_resource_pressure_threshold": 0.12,
    "low_drop_deadline_urgency_protection": 0.95,
    # 大改：active_low_drop_floor_ratio 0.01 → 0.0
    # 原值会让每步无条件丢 1% droppable backlog，无视 agent 的 drop_strength，
    # 造成 13GB/episode 不可控的 drop，污染 useful 指标和 drop 相关 penalty。
    # 改为 0 之后，drop 完全由 agent 的 drop_strength + 3 个真实压力 driver
    # (capacity / queue_pressure / low_share) 共同决定，agent 才真正"拥有"
    # drop 这个动作。
    "active_low_drop_floor_ratio": 0.0,
    "low_drop_share_target": 0.05,
    # CPU 动作语义：alpha_cpu 表示“当前 admissible 处理额度中使用多少比例”，不是直接功率强度。
    "cpu_action_is_admissible_budget": True,
    "enable_future_contact_cpu_gate": True,
    "cpu_gate_start_future_ratio": 0.55,
    "cpu_gate_target_future_ratio": 0.75,
    "cpu_gate_hard_stop_future_ratio": 0.90,
    "cpu_gate_far_window_lead_s": 120.0,
    "cpu_gate_floor_alpha": 0.0,
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
}

# ─────────────────────────────────────────────
# 深度强化学习参数
# ─────────────────────────────────────────────
# 当前训练口径标识。
# 修改 reward/状态/训练语义后要同步更新，防止主训练入口误续训不兼容 checkpoint。
OBJECTIVE_VERSION = "delivered_voi_cmdp_admissible_cpu_deliverability_v4"

DRL_CONFIG = {
    "algorithm": "SAC",              # 算法: SAC (纯正的 SAC，无CMDP/PSF等)
    "state_dim": 62,                 # 状态空间维度：物理状态 + 任务价值/紧急度分组直方图 + 可达性前瞻
    "action_dim": 8,                 # [推进, CPU可接纳预算比例, TX, CPU价值权重, CPU紧迫权重, TX价值权重, TX紧迫权重, 低价值主动丢弃]
    "hidden_dim": 512,               # 隐藏层维度（增强网络容量）
    "network_arch": "transformer",   # transformer | mlp；MLP 只用于 backbone 消融
    "lr_actor": 2.5e-4,              # Actor学习率
    "lr_critic": 2.5e-4,             # Critic学习率
    "lr_alpha": 1e-4,                # 熵系数学习率（保持探索自适应能力）
    "gamma": 0.995,                  # 折扣因子（0.99→0.995：远窗口信号强度 ~14x，让 Q 能传回 540 步外的处理动作）
    "reward_shaping_coeff": 0.0,     # 大改：暂时关掉 potential-based shaping。
                                     # Ng 1999 理论上不改最优策略，但 Φ 在过渡期可能给出
                                     # 反方向梯度（queue 增长时 phi_next>phi_prev 给正 shaping，
                                     # 鼓励 agent 继续填队列）。和现在的"少处理"目标冲突。
                                     # 等 useful>0.3 之后再尝试升回 0.1。
    "tau": 0.005,                    # 目标网络软更新系数（更慢的更新，更稳定）
    "batch_size": 512,               # 批量大小（增加批量以改进梯度质量）
    "buffer_size": int(2e6),         # 经验回放缓冲区大小
    "warmup_steps": 4000,            # 随机探索预热步数
    "update_freq": 4,                # 每4步更新一次网络；多环境采样下减少反向传播阻塞
    # 【学习率调度】使用余弦退火确保训练后期充分收敛
    "lr_schedule_type": "cosine",    # 学习率调度类型: constant | cosine | exponential
    "lr_min_scale": 0.01,            # 最小学习率比例（最终 lr = lr_initial * lr_min_scale）
    "update_actor_freq": 1,          # Actor/alpha每次critic更新都更新；优先保证策略能跟上队列反馈
    "gradient_clip": 10.0,           # 梯度裁剪阈值
    "use_amp": False,               # 数值稳定优先：默认关闭AMP，避免可能的数值不稳定（如NaN）导致训练中断，全程跑FP32
    "state_normalization": True,     # SAC 网络输入使用 RunningMeanStd 动态归一化，环境观测本身保持物理语义
    "state_norm_epsilon": 1e-4,
    "state_norm_clip": 5.0,
    "nan_guard_enable": True,        # 启用更新阶段非有限值守卫
    "alpha_log_clip": 10.0,          # log_alpha 裁剪范围 [-clip, clip]
    "alpha_min": 0.01,               # 熵系数下限，避免后期探索过早塌缩导致策略固化
    # 缩放参数（用于稳定训练与约束强度可解释性）
    # - reward_scale: 将环境奖励缩放到 O(1)
    # - lyapunov_drift_scale: Lyapunov 漂移缩放（默认 1.0，不再额外 ÷100）
    "reward_scale": 100.0,
    "lyapunov_drift_scale": 1.0,
    "lyapunov_drift_clip": 20.0,
    # 安全层动作惩罚：让 actor 学会“原始动作也尽量可行”，而不是长期依赖投影兜底
    "projection_penalty_coeff": 4.0,
    "action_mod_penalty_coeff": 1.0,
    "safety_action_penalty_cap_ratio": 0.25,
    "safety_action_penalty_min_cap": 3.0,
    # Actor 辅助模仿最终安全执行动作，避免长期依赖安全投影兜底。
    "behavior_cloning_coeff": 0.05,
    "behavior_cloning_max_weight": 1.0,
    "behavior_cloning_conservative_weight_coeff": 0.25,
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
    "deliverable_critic_actor_coeff": 0.5,
    "deliverable_critic_target_key": "processed_deliverable_value_step",
    "lyapunov_penalty_coeff": 0.3,
    "adaptive_lyapunov_coeff_enable": True,
    # threshold: 允许的平均归一化约束代价。设为 0.10 表示 proc/dl 远超容量时才引发。
    "adaptive_lyapunov_constraint_threshold": 0.10,
    "adaptive_lyapunov_coeff_target_pressure": 0.10,
    "adaptive_lyapunov_coeff_lr": 0.01,
    "adaptive_lyapunov_coeff_ema_beta": 0.99,
    "adaptive_lyapunov_coeff_min": 0.3,
    "adaptive_lyapunov_coeff_max": 1.5,
    # Dual update uses normalized CMDP cost c_t / norm, not PSF/projection rate.
    "adaptive_lyapunov_constraint_norm": 10.0,
    "adaptive_lyapunov_constraint_signal_max": 3.0,
    "projection_ema_beta": 0.995,
    # checkpoint 选择时把安全层介入代价折算为 MB 惩罚，避免只按 reward/下传挑到“靠投影兜底”的模型
    "checkpoint_proj_penalty_mb": 1800.0,
    "checkpoint_projected_penalty_mb": 1200.0,
    "checkpoint_action_mod_penalty_mb": 250.0,
    "checkpoint_max_proc_downlink_ratio": 4.0,
    "checkpoint_max_processed_queue_final_utilization": 0.85,
    "checkpoint_max_processed_queue_future_contact_ratio": 0.95,
    "checkpoint_max_cpu_far_from_window_rate": 0.25,
    "checkpoint_max_energy_violation_rate": 0.02,
    # 队列风险惩罚（注入到 Lyapunov 漂移信号）：
    # - 软惩罚：利用率超过阈值后按二次项增长，促使策略提前降风险
    # - 硬惩罚：真实 overflow（丢包/积压越界）按比例强惩罚
    "lya_soft_util_threshold": 0.75,
    "lya_soft_util_penalty_coeff": 0.5,
    "lya_soft_penalty_clip": 1.0,
    "lya_hard_overflow_penalty_coeff": 3.0,
    "lya_hard_penalty_clip": 2.0,
    # 【重构后】CMDP cost 只保留 4 大主项,移除互相打架的旧 cost。
    # 恢复温和的 over_processing_cost，配合较低的 adaptive_lyapunov_coeff_max 防止 collapse
    "enable_capacity_aware_cost_v2": True,
    "enable_deliverable_processing_reward": False,
    "queue_projection_policy": "safety_algorithms_only",
    "enable_deployment_queue_projection": True,
    "constraint_over_processing_coeff": 4.0,
    "constraint_over_processing_clip": 10.0,
    "constraint_over_processing_ratio_weight": 1.0,
    "constraint_capacity_norm_mb": 400.0,
    "constraint_capacity_norm": 400.0,  # 兼容旧脚本
    "constraint_future_capacity_margin": 0.60,
    "constraint_efficiency_processed_value_credit": 0.0,
    # ── 物理状态硬安全约束(阶段化代价 + 热超限 + 能量边界 + 轨道边界)。
    "constraint_stage_costs": {"warning": 0.08, "unsafe": 0.8, "failure": 3.0},
    "constraint_auxiliary_violation_cost": 0.25,
    "constraint_thermal_excess": {"coeff": 0.25, "norm_c": 10.0},
    "constraint_energy_margin_coeff": 0.25,
    "constraint_energy_margin_clip": 1.0,
    "constraint_orbit_margin_coeff": 0.25,
    "constraint_orbit_margin_clip": 1.0,
    # ── 高价值任务过期/丢弃约束(平滑启用)。
    "constraint_high_value_loss_coeff": 1.0,
    "constraint_task_loss_value_norm": 5000.0,
    "constraint_task_loss_clip": 2.0,
    "constraint_task_loss_warmup_steps": 30000,
    "constraint_task_loss_anneal_steps": 120000,
    "constraint_task_loss_min_scale": 0.0,
    # Historical aliases kept for old checkpoints/scripts (compatibility only).
    "constraint_warning_cost": 0.08,
    "constraint_unsafe_cost": 0.8,
    "constraint_failure_cost": 3.0,
    "constraint_thermal_warning_cost": 0.08,
    "constraint_thermal_excess_coeff": 0.25,
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
    "constraint_low_value_waste_coeff": 0.0,
    "constraint_low_value_waste_clip": 0.0,
    "constraint_unproductive_cpu_coeff": 0.0,
    "constraint_unproductive_cpu_clip": 0.0,
    "constraint_window_waste_coeff": 0.0,
    "constraint_window_waste_clip": 0.0,
    "constraint_efficiency_cost_coeff": 0.0,
    "constraint_efficiency_cost_clip": 0.0,
    "target_entropy_scale": 1.0,     # 目标熵缩放
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
PSF_CONFIG = {
    # PSF 是最终物理兜底层，触发阈值要早于硬失败边界，但晚于常规可学习区间。
    # 高度 165km / SOC 20% 会在接近安全边界前做短视野 rollout，
    # 避免只在贴近硬边界时才介入。
    "altitude_trigger_margin_m": 15_000.0,
    "soc_trigger_margin": 0.05,
    # 只有明显处于正常区间时才跳过 rollout。高度阈值与 180km warning 上界对齐。
    "passthrough_altitude_margin_m": 30_000.0,
    "passthrough_soc_margin": 0.08,
    "robust_altitude_margin_m": 30_000.0,
    "long_horizon_steps": 540,
    "long_horizon_altitude_margin_m": 60_000.0,
    "long_horizon_violation_margin_m": 5_000.0,
    "robust_density_perturb_range": 0.50,
    "robust_solar_power_scale": 0.80,
    "robust_battery_capacity_scale": 0.85,
    "robust_propulsion_thrust_scale": 0.85,
}

# ─────────────────────────────────────────────
# 训练参数
# ─────────────────────────────────────────────
TRAIN_CONFIG = {
    "total_steps": 1000000,          # 主训练步数：1M，优先把算力留给多seed/强基线/真实trace
    "eval_freq": 20000,              # 评估频率 (步)；降低评估开销以加快主训练
    "eval_episodes": 5,           # 正式评估默认 episode 数；快速调试请用各脚本的 smoke/max_steps 参数
    "save_freq": 50000,              # 模型保存频率
    "keep_step_checkpoints": False,   # 默认只保留 best/latest，避免生成大量中间模型文件
    "log_freq": 500,                 # 日志记录频率
    "max_episode_steps": 2160,       # 每episode最大步数 (=4个90min轨道, dt=10s)，覆盖跨轨道资源规划
    "update_freq": 4,                # 每4步更新一次网络 (与 DRL_CONFIG 保持一致)
    "time_slot_s": 10,               # 时间片长度 (秒)
    "seed": 42,
    "eval_seeds": [42, 43, 44, 45, 46], # 多随机种子默认种子；experiments/multi_seed.py 默认读取这里
    "n_envs": 8,                     # 多环境采样默认（提升样本多样性与学习效率）
    "env_backend": "auto",           # auto: n_envs>1 时启用子进程环境；serial: 调试用串行环境
    "optimized_checkpoint_dir": "checkpoints_optimized/",
    "optimized_log_dir": "logs_optimized/",
    "fail_fast_on_nan": False,       # 允许少量NaN告警，避免一次抖动中断训练
    "nan_guard_max_hits": 10,        # NaN告警累计上限（达到后停止）
    
    # 多阶段课程学习：从低负载平滑提升到完整负载；训练入口会对阶段边界做线性 ramp，
    # 避免任务到达率硬跳变导致策略崩溃。
    "use_curriculum": True,
    # randomization_scale 控制 env._randomization_scale，影响 rho/β/storm 三项随机化幅度。
    # 阶段间用与 data_arrival_scale 同样的线性 ramp 平滑过渡，避免分布硬跳变。
    # Exploration: 0.20 → rho×[0.87,1.15], β≤15°, storm prob 1e-5/peak~1.54
    # Balancing:   0.45 → rho×[0.73,1.37], β≤34°, storm prob 2.2e-5/peak~1.84
    # Ramp:        0.75 → rho×[0.59,1.69], β≤56°, storm prob 3.8e-5/peak~2.20
    # Optimization:1.00 → 完整 PDF 物理极值
    "curriculum_stages": [
        {
            "stage_name": "Exploration",
            "steps": 50000,
            "lyapunov_weight_scale": 0.80,
            "data_arrival_scale": 0.25,
            "randomization_scale": 0.20,
            "description": "弱难度，让 agent 学基础策略；随机化几乎关闭",
        },
        {
            "stage_name": "Balancing",
            "steps": 150000,
            "lyapunov_weight_scale": 0.75,
            "data_arrival_scale": 0.55,
            "randomization_scale": 0.45,
            "description": "中等难度；引入温和随机化",
        },
        {
            "stage_name": "Ramp",
            "steps": 300000,
            "lyapunov_weight_scale": 0.9,
            "data_arrival_scale": 0.75,
            "randomization_scale": 0.75,
            "description": "接近完整负载 + 大幅随机化",
        },
        {
            "stage_name": "Optimization",
            "steps": 500000,
            "lyapunov_weight_scale": 1.0,
            "data_arrival_scale": 1.0,
            "randomization_scale": 1.0,
            "description": "完整难度 + 完整 PDF 物理随机化（含 Starlink 级风暴和全日照）",
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
    "w_delivered_value": 5.0,
    "w_deadline_success": 0.5,

    # w_processing_deliverable_value: 0.3 -> 0.0
    # 关闭正向处理奖励：只要 deliver_prob > 0，处理就有正收益，agent 始终倾向满负荷 CPU。
    # 去掉后处理只有负信号 (opportunity_cost)，不再有"处理越多越好"的梯度。
    # w_processing_opportunity_cost: 0.3 -> 0.5
    # 更强的不可投递惩罚，确保 deliver_prob < 1.0 时负梯度足够驱动 alpha_cpu 下降。
    "w_processing_deliverable_value": 0.0,
    "w_processing_opportunity_cost": 0.5,

    # 普通能耗进入 cost critic；reward 只在超过每步预算时给很小的软代价。
    "w_energy_penalty": 0.0,
    "w_energy_over_budget_penalty": -0.5,
    "energy_budget_wh_per_step": 0.22,

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
    "w_expired_penalty": -1.0,
    "w_prospective_expiry_shaping": 0.0,
    "w_actuator_violation_penalty": 0.0,
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
    "cpu_gate_soft_mode": True,
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
}

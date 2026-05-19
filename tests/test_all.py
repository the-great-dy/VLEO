"""
tests/test_all.py  —  完整单元测试套件（自包含版本）

覆盖: 轨道动力学、能量模型、虚拟队列、李雅普诺夫优化器、环境接口。
每个测试类均自行注入 sys.path，消除所有随机性依赖，
确保无论在哪个目录运行都能找到模块。

运行方式:
    python tests/test_all.py    # 从项目根目录
    python test_all.py          # 从 tests/ 目录
"""

import unittest
import numpy as np
import sys
import os
import tempfile

# ── 路径注入 (无论从哪里运行都能找到项目根) ───────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
if __package__ in (None, ""):
    for _p in [_ROOT_DIR, _THIS_DIR]:
        if _p not in sys.path:
            # 测试脚本同样避免最高优先级注入，减少模块遮蔽风险
            sys.path.append(_p)


# ═══════════════════════════════════════════════════════════════════════
# 0. 训练配置
# ═══════════════════════════════════════════════════════════════════════
class TestTrainingConfig(unittest.TestCase):

    def test_update_freq_consistent(self):
        """DRL_CONFIG 和 TRAIN_CONFIG 的更新频率必须保持一致"""
        from config import DRL_CONFIG, TRAIN_CONFIG
        self.assertEqual(DRL_CONFIG["update_freq"], TRAIN_CONFIG["update_freq"])

    def test_fast_training_defaults(self):
        """默认训练配置应启用多环境采样，并避免每步都反向传播"""
        from config import DRL_CONFIG, TRAIN_CONFIG
        self.assertGreaterEqual(TRAIN_CONFIG["n_envs"], 2)
        self.assertGreaterEqual(DRL_CONFIG["update_freq"], 2)

    def test_formal_eval_episodes_default(self):
        """正式评估默认 episode 数不能太少，否则难以支撑随机初值结论。"""
        from config import TRAIN_CONFIG
        self.assertGreaterEqual(TRAIN_CONFIG["eval_episodes"], 5)

    def test_curriculum_ramps_to_full_load(self):
        """课程学习的数据难度应从低负载平滑升到完整负载"""
        from config import TRAIN_CONFIG
        stages = TRAIN_CONFIG["curriculum_stages"]
        scales = [float(stage["data_arrival_scale"]) for stage in stages]
        lya_scales = [float(stage["lyapunov_weight_scale"]) for stage in stages]
        stage_steps = [int(stage["steps"]) for stage in stages]
        self.assertLessEqual(scales[0], 0.5)
        self.assertEqual(scales, sorted(scales))
        self.assertAlmostEqual(scales[-1], 1.0)
        self.assertLessEqual(stage_steps[0], 50000)
        self.assertEqual(sum(stage_steps), int(TRAIN_CONFIG["total_steps"]))
        self.assertGreaterEqual(lya_scales[0], 0.40)
        self.assertGreaterEqual(lya_scales[1], 0.70)
        self.assertAlmostEqual(lya_scales[-1], 1.0)

    def test_episode_length_covers_multiple_orbits(self):
        """episode 不能只覆盖一个轨道周期，否则难以学到跨轨道资源规划。"""
        from config import ORBITAL_CONFIG, TRAIN_CONFIG

        orbit_steps = int(round(
            float(ORBITAL_CONFIG["orbital_period_min"]) * 60.0
            / float(TRAIN_CONFIG["time_slot_s"])
        ))
        self.assertGreaterEqual(int(TRAIN_CONFIG["max_episode_steps"]), 4 * orbit_steps)

    def test_experiment_protocol_records_search_settings(self):
        """实验协议应显式记录超参数搜索和场景先验口径。"""
        from config import EXPERIMENT_PROTOCOL

        search = EXPERIMENT_PROTOCOL["hyperparameter_search"]
        self.assertTrue(bool(search.get("enabled", False)))
        self.assertEqual(search.get("selection_metric"), "safety_adjusted_delivered_value")
        self.assertEqual(EXPERIMENT_PROTOCOL.get("scene_model_source"), "synthetic_scene_prior")
        self.assertFalse(bool(EXPERIMENT_PROTOCOL.get("scene_profiles_are_empirical", True)))

    def test_mid_vleo_physical_defaults(self):
        """默认物理参数应落在中型VLEO对地观测星的合理量级。"""
        from config import DRAG_CONFIG, ENERGY_CONFIG, QUEUE_CONFIG, ORBITAL_CONFIG

        self.assertGreaterEqual(DRAG_CONFIG["mass_kg"], 150.0)
        self.assertLessEqual(DRAG_CONFIG["mass_kg"], 450.0)
        self.assertEqual(ORBITAL_CONFIG["altitude_warning_km"], 180.0)
        self.assertEqual(ORBITAL_CONFIG["altitude_min_km"], 150.0)
        self.assertEqual(ORBITAL_CONFIG["altitude_crash_km"], 122.0)
        self.assertEqual(ENERGY_CONFIG["solar_panel_power_w"], 120.0)
        self.assertEqual(ENERGY_CONFIG["battery_capacity_wh"], 300.0)
        self.assertAlmostEqual(ENERGY_CONFIG["battery_min_soc"], 0.15)
        self.assertAlmostEqual(ENERGY_CONFIG["battery_crash_soc"], 0.05)
        self.assertEqual(ENERGY_CONFIG["power_propulsion_max_w"], 90.0)
        self.assertEqual(ENERGY_CONFIG["power_cpu_max_w"], 25.0)
        self.assertEqual(ENERGY_CONFIG["power_tx_max_w"], 35.0)
        self.assertEqual(ENERGY_CONFIG["power_baseline_w"], 15.0)
        self.assertEqual(ENERGY_CONFIG["power_total_max_w"], 120.0)
        self.assertEqual(QUEUE_CONFIG["data_queue_max_mb"], 16 * 1024.0)
        self.assertEqual(QUEUE_CONFIG["comm_queue_max"], 4 * 1024.0)
        self.assertAlmostEqual(QUEUE_CONFIG["data_arrival_rate_mbs"], 5.0)
        self.assertAlmostEqual(QUEUE_CONFIG["data_service_rate_max_mbs"], 8.0)
        self.assertAlmostEqual(QUEUE_CONFIG["tx_downlink_rate_max_mbs"] * 8.0, 100.0)
        self.assertAlmostEqual(QUEUE_CONFIG["tx_capacity_norm_mbps"], 100.0)
        self.assertGreater(
            QUEUE_CONFIG["data_service_rate_max_mbs"],
            QUEUE_CONFIG["data_arrival_rate_mbs"],
        )


# ═══════════════════════════════════════════════════════════════════════
# 1. 大气密度模型
# ═══════════════════════════════════════════════════════════════════════
class TestAtmosphericModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from environment.orbital_dynamics import AtmosphericModel
        cls.atm = AtmosphericModel()

    def test_density_positive(self):
        """各高度密度必须为正"""
        for h_km in [150, 250, 350, 450]:
            with self.subTest(h_km=h_km):
                self.assertGreater(self.atm.density(h_km * 1e3), 0)

    def test_density_decreases_with_altitude(self):
        """密度随高度增加而指数下降"""
        self.assertGreater(self.atm.density(200e3), self.atm.density(400e3))

    def test_density_exponential_scale(self):
        """验证指数标高: rho(h+H)/rho(h) = 1/e"""
        h0 = 350e3
        H = self.atm.H_scale
        ratio = self.atm.density(h0 + H) / self.atm.density(h0)
        self.assertAlmostEqual(ratio, np.exp(-1), places=5)


# ═══════════════════════════════════════════════════════════════════════
# 2. 轨道动力学
# ═══════════════════════════════════════════════════════════════════════
class TestOrbitalDynamics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from environment.orbital_dynamics import OrbitalDynamics
        cls.orb = OrbitalDynamics()

    def test_orbital_velocity_350km(self):
        """350km 圆轨道速度约 7.7 km/s"""
        v = self.orb.orbital_velocity(350e3)
        self.assertAlmostEqual(v / 1e3, 7.7, delta=0.2)

    def test_drag_force_positive(self):
        """阻力必须为正"""
        self.assertGreater(self.orb.drag_force(350e3), 0)

    def test_altitude_decay_negative_without_thrust(self):
        """无推力时 dh/dt < 0（轨道自然衰减）"""
        self.assertLess(self.orb.altitude_decay_rate(350e3), 0)

    def test_thrust_slows_decay(self):
        """有推力时轨道高度高于无推力情况"""
        h = 350e3
        no_thrust  = self.orb.step(h, 0.0,  60.0)
        with_thrust = self.orb.step(h, 40.0, 60.0)
        self.assertGreater(with_thrust["altitude_m"], no_thrust["altitude_m"])

    def test_propulsion_has_ignition_threshold(self):
        """低于电推进启动门限时不应产生虚假的微小连续推力。"""
        from config import ENERGY_CONFIG

        h = 350e3
        threshold_w = float(ENERGY_CONFIG["propulsion_ignition_threshold_w"])
        no_thrust = self.orb.step(h, 0.0, 60.0)
        below_threshold = self.orb.step(h, threshold_w * 0.5, 60.0)
        above_threshold = self.orb.step(h, threshold_w * 1.2, 60.0)

        self.assertAlmostEqual(below_threshold["thrust_N"], 0.0, places=12)
        self.assertAlmostEqual(
            below_threshold["altitude_m"], no_thrust["altitude_m"], places=9)
        self.assertGreater(above_threshold["thrust_N"], 0.0)

    def test_step_required_keys(self):
        """step() 返回值包含所有必要键"""
        result = self.orb.step(350e3, 20.0, 10.0)
        for key in ["altitude_m", "drag_force_N", "decay_rate_ms",
                    "thrust_N", "dh_m", "is_safe", "is_crashed",
                    "is_warning", "safety_stage", "safety_stage_code"]:
            self.assertIn(key, result)

    def test_safe_flag_above_hmin(self):
        """高于最低安全高度时 is_safe=True"""
        result = self.orb.step(350e3, 30.0, 10.0)
        self.assertTrue(result["is_safe"])
        self.assertFalse(result["is_crashed"])
        self.assertEqual(result["safety_stage"], "normal")

    def test_warning_altitude_between_150_and_180km(self):
        """150~180km 是警告区，不是严重不安全或坠毁。"""
        result = self.orb.step(170e3, 0.0, 1.0)
        self.assertTrue(result["is_safe"])
        self.assertTrue(result["is_warning"])
        self.assertFalse(result["is_crashed"])
        self.assertEqual(result["safety_stage"], "warning")

    def test_unsafe_altitude_not_immediate_crash(self):
        """150km 以下但高于 122km 时是不安全状态，不是物理坠毁"""
        result = self.orb.step(140e3, 0.0, 1.0)
        self.assertFalse(result["is_safe"])
        self.assertFalse(result["is_crashed"])
        self.assertEqual(result["safety_stage"], "unsafe")

    def test_reentry_altitude_triggers_crash(self):
        """122km 以下触发物理再入/坠毁终态"""
        result = self.orb.step(121e3, 0.0, 1.0)
        self.assertTrue(result["is_crashed"])
        self.assertEqual(result["safety_stage"], "failure")


# ═══════════════════════════════════════════════════════════════════════
# 3. 轨道周期模拟器
# ═══════════════════════════════════════════════════════════════════════
class TestOrbitalPeriodSimulator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from environment.orbital_dynamics import OrbitalPeriodSimulator
        cls.sim = OrbitalPeriodSimulator()

    def test_sunlit_at_t0(self):
        """t=0 在日照区"""
        self.assertTrue(self.sim.is_sunlit(0.0))

    def test_eclipse_after_sunlit_ends(self):
        """日照结束 1 秒后进入阴影"""
        self.assertFalse(self.sim.is_sunlit(self.sim.sunlit_s + 1))

    def test_period_is_cyclic(self):
        """整数倍轨道周期后状态一致"""
        self.assertEqual(self.sim.is_sunlit(0.0),
                         self.sim.is_sunlit(self.sim.period_s))

    def test_sunlit_fraction_in_unit_interval(self):
        """日照强度归一化因子在 [0, 1]"""
        for t in np.linspace(0, self.sim.period_s, 100):
            with self.subTest(t=round(t, 1)):
                frac = self.sim.sunlit_fraction(t)
                self.assertGreaterEqual(frac, 0.0)
                self.assertLessEqual(frac, 1.0)


# ═══════════════════════════════════════════════════════════════════════
# 4. 电池模型
# ═══════════════════════════════════════════════════════════════════════
class TestBatteryModel(unittest.TestCase):

    def setUp(self):
        from environment.energy_model import BatteryModel
        self.batt = BatteryModel()
        self.batt.soc = 0.80

    def test_charging_raises_soc(self):
        """太阳能 > 负载时 SOC 增加"""
        before = self.batt.soc
        self.batt.step(80.0, 10.0, 60.0)
        self.assertGreater(self.batt.soc, before)

    def test_discharging_lowers_soc(self):
        """负载 > 太阳能时 SOC 减小"""
        before = self.batt.soc
        self.batt.step(0.0, 50.0, 60.0)
        self.assertLess(self.batt.soc, before)

    def test_soc_capped_at_max(self):
        """SOC 不超过 soc_max"""
        self.batt.soc = 0.94
        self.batt.step(80.0, 0.0, 3600.0)
        self.assertLessEqual(self.batt.soc, self.batt.soc_max + 1e-9)

    def test_energy_margin_positive_when_safe(self):
        """SOC 高于安全线时裕度 > 0"""
        self.batt.soc = 0.80
        self.assertGreater(self.batt.energy_margin_wh, 0)

    def test_unsafe_soc_triggers_flag(self):
        """5%~15% SOC 是能源警告区，不是终止失败。"""
        self.batt.soc = 0.10
        result = self.batt.step(0.0, 1.0, 1.0)
        self.assertFalse(result["is_safe"])
        self.assertFalse(result["is_crashed"])
        self.assertTrue(result["is_warning"])
        self.assertEqual(result["safety_stage"], "warning")

    def test_crash_soc_triggers_flag(self):
        """SOC 低于终止失败线时 is_crashed=True"""
        self.batt.soc = 0.04
        result = self.batt.step(0.0, 1.0, 1.0)
        self.assertTrue(result["is_crashed"])
        self.assertEqual(result["safety_stage"], "failure")

    def test_reset_produces_varied_soc(self):
        """多次 reset() 产生不同 SOC"""
        socs = set()
        for _ in range(10):
            self.batt.reset()
            socs.add(round(self.batt.soc, 3))
        self.assertGreater(len(socs), 3)

    def test_cycle_degradation_reduces_capacity_and_resets(self):
        """频繁充放电应累计等效循环并降低可用容量，reset 后恢复新 episode。"""
        self.batt.capacity_loss_per_efc = 0.02
        before_capacity = self.batt.capacity_wh

        info = self.batt.step(0.0, 180.0, 3600.0)

        self.assertGreater(info["equivalent_full_cycles"], 0.0)
        self.assertGreater(info["capacity_loss_wh"], 0.0)
        self.assertLess(self.batt.capacity_wh, before_capacity)

        self.batt.reset(initial_soc=0.8)
        self.assertAlmostEqual(self.batt.capacity_wh, self.batt.nominal_capacity_wh)
        self.assertAlmostEqual(self.batt.equivalent_full_cycles, 0.0)


# ═══════════════════════════════════════════════════════════════════════
# 5. 功率子系统
# ═══════════════════════════════════════════════════════════════════════
class TestPowerSubsystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from environment.energy_model import PowerSubsystem
        cls.ps = PowerSubsystem()

    def test_zero_action_baseline_only(self):
        """零动作时总功耗 = 基础功耗"""
        info = self.ps.compute_total_load(np.array([0.0, 0.0, 0.0]))
        self.assertAlmostEqual(info["P_total_w"], self.ps.P_baseline)

    def test_full_action_max_power(self):
        """全功率动作时总功耗应被限制在 power_total_max_w（或达到理论最大值）"""
        info = self.ps.compute_total_load(np.array([1.0, 1.0, 1.0]))
        
        # 导入配置获取功率上限
        from config import ENERGY_CONFIG
        P_max_total = ENERGY_CONFIG.get("power_total_max_w", 120.0)
        theoretical_max = (self.ps.P_prop_max + self.ps.P_cpu_max +
                          self.ps.P_tx_max + self.ps.P_baseline)
        
        # 实际功耗应为 min(理论最大值, 功率上限)
        expected = min(theoretical_max, P_max_total)
        self.assertAlmostEqual(info["P_total_w"], expected)

    def test_zero_power_zero_throughput(self):
        """零功率吞吐率为 0"""
        self.assertEqual(self.ps.throughput_rate(0.0, 0.0), 0.0)

    def test_positive_power_positive_throughput(self):
        """有功率时吞吐率 > 0"""
        self.assertGreater(self.ps.throughput_rate(10.0, 10.0), 0.0)

    def test_propulsion_deadband_removes_sub_ignition_power(self):
        """推进动作低于启动门限时，功率模型应直接关断推进通道。"""
        from config import ENERGY_CONFIG

        threshold_alpha = (
            ENERGY_CONFIG["propulsion_ignition_threshold_w"]
            / ENERGY_CONFIG["power_propulsion_max_w"]
        )
        info = self.ps.compute_total_load(
            np.array([threshold_alpha * 0.5, 0.0, 0.0]))

        self.assertAlmostEqual(info["P_propulsion_w"], 0.0)
        self.assertAlmostEqual(info["P_total_w"], self.ps.P_baseline)
        self.assertTrue(bool(info["propulsion_deadband_applied"]))

    def test_configured_peak_rates_are_used(self):
        """CPU处理速率和发射机速率必须读取配置，避免隐藏的5MB/s瓶颈。"""
        from config import QUEUE_CONFIG

        self.assertAlmostEqual(
            self.ps.throughput_rate(self.ps.P_cpu_max),
            QUEUE_CONFIG["data_service_rate_max_mbs"],
        )
        self.assertAlmostEqual(
            self.ps.tx_downlink_rate(self.ps.P_tx_max),
            QUEUE_CONFIG["tx_downlink_rate_max_mbs"],
        )

    def test_solar_panel_power_is_generated_peak_power(self):
        """太阳能峰值配置已是标称发电输出功率，效率扰动只按相对比例生效。"""
        from config import ENERGY_CONFIG
        from environment.energy_model import SolarPanelModel

        solar = SolarPanelModel()
        self.assertAlmostEqual(
            solar.output_power(1.0),
            ENERGY_CONFIG["solar_panel_power_w"],
        )
        solar.eta *= 0.5
        self.assertAlmostEqual(
            solar.output_power(1.0),
            ENERGY_CONFIG["solar_panel_power_w"] * 0.5,
        )


# ═══════════════════════════════════════════════════════════════════════
# 6. 能量虚拟队列
# ═══════════════════════════════════════════════════════════════════════
class TestEnergyVirtualQueue(unittest.TestCase):

    def setUp(self):
        from virtual_queues.energy_queue import EnergyVirtualQueue
        self.queue = EnergyVirtualQueue()
        self.queue.reset(initial_energy_margin=10.0)

    def test_decreases_with_positive_margin(self):
        """正裕度时队列减小"""
        self.queue.value = 5.0
        result = self.queue.update(10.0)
        self.assertLessEqual(result["queue_value"], 5.0)

    def test_increases_with_negative_margin(self):
        """赤字时队列增加"""
        self.queue.value = 5.0
        result = self.queue.update(-5.0)
        self.assertGreater(result["queue_value"], 5.0)

    def test_nonnegative(self):
        """队列值不为负"""
        self.queue.value = 0.0
        result = self.queue.update(100.0)
        self.assertGreaterEqual(result["queue_value"], 0.0)

    def test_urgency_in_unit_interval(self):
        """紧急程度在 [0, 1]"""
        for v in [0.0, 25.0, 50.0, 100.0]:
            self.queue.value = v
            result = self.queue.update(0.0)
            self.assertGreaterEqual(result["urgency"], 0.0)
            self.assertLessEqual(result["urgency"], 1.0)

    def test_drift_negative_on_decrease(self):
        """队列缩小时 drift < 0"""
        self.queue.value = 20.0
        result = self.queue.update(50.0)
        self.assertLess(result["drift"], 0.0)

    def test_drift_positive_on_increase(self):
        """队列增大时 drift > 0"""
        self.queue.value = 10.0
        result = self.queue.update(-20.0)
        self.assertGreater(result["drift"], 0.0)


# ═══════════════════════════════════════════════════════════════════════
# 7. 轨道高度虚拟队列
# ═══════════════════════════════════════════════════════════════════════
class TestOrbitVirtualQueue(unittest.TestCase):

    def setUp(self):
        # 直接从 energy_queue 导入，不经过 stub 文件
        from virtual_queues.energy_queue import OrbitVirtualQueue
        self.queue = OrbitVirtualQueue()
        self.queue.reset(initial_altitude_m=350e3)

    def test_stable_at_safe_altitude(self):
        """安全高度时队列保持稳定或缩小"""
        self.queue.value = 2.0
        result = self.queue.update(350e3)
        self.assertLessEqual(result["queue_value"], 2.0)

    def test_grows_below_hmin(self):
        """低于安全线时队列增长 (148km < h_min=150km)"""
        self.queue.value = 5.0
        result = self.queue.update(148e3)
        self.assertGreater(result["queue_value"], 5.0)

    def test_unchanged_at_hmin(self):
        """恰好在安全线时 margin=0，队列不变"""
        self.queue.value = 3.0
        result = self.queue.update(150e3)
        self.assertAlmostEqual(result["queue_value"], 3.0, places=5)

    def test_reset_safe_gives_zero(self):
        """安全高度 reset → 初始队列 = 0"""
        self.queue.reset(350e3)
        self.assertEqual(self.queue.value, 0.0)

    def test_reset_unsafe_gives_positive(self):
        """低于安全线 reset → 初始队列 > 0"""
        self.queue.reset(145e3)
        self.assertGreater(self.queue.value, 0.0)


# ═══════════════════════════════════════════════════════════════════════
# 8. 数据任务队列
# ═══════════════════════════════════════════════════════════════════════
class TestDataTaskQueue(unittest.TestCase):

    def setUp(self):
        from virtual_queues.energy_queue import DataTaskQueue
        self.queue = DataTaskQueue()
        self.queue.reset()

    def test_grows_when_arrival_exceeds_service(self):
        """到达 > 服务时队列增长"""
        self.queue.length = 0.0
        result = self.queue.step(10.0, 2.0)
        self.assertGreater(result["queue_length"], 0.0)

    def test_shrinks_when_service_exceeds_arrival(self):
        """服务 > 到达时队列缩小"""
        self.queue.length = 50.0
        result = self.queue.step(1.0, 20.0)
        self.assertLess(result["queue_length"], 50.0)

    def test_nonnegative(self):
        """队列长度不为负"""
        self.queue.length = 5.0
        result = self.queue.step(0.0, 100.0)
        self.assertGreaterEqual(result["queue_length"], 0.0)

    def test_service_capped_by_available_data(self):
        """实际服务量不超过「队列 + 到达量」"""
        self.queue.length = 3.0
        result = self.queue.step(2.0, 100.0)
        self.assertLessEqual(result["serviced"], 5.0 + 1e-9)


# ═══════════════════════════════════════════════════════════════════════
# 9. 李雅普诺夫优化器
# ═══════════════════════════════════════════════════════════════════════
@unittest.skip(
    "safety.lyapunov_projection module removed in CMDP refactor — "
    "Lyapunov action projection is now handled by the adaptive dual variable in training."
)
class TestLyapunovActionProjection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from safety.lyapunov_projection import LyapunovActionProjection
        cls.opt = LyapunovActionProjection()

    def test_function_nonnegative(self):
        """李雅普诺夫函数值 >= 0"""
        self.assertGreaterEqual(self.opt.lyapunov_function(5.0, 3.0, 10.0), 0.0)

    def test_function_zero_at_origin(self):
        """原点处 L = 0"""
        self.assertAlmostEqual(self.opt.lyapunov_function(0.0, 0.0, 0.0), 0.0)

    def test_drift_negative_on_decrease(self):
        """队列全部缩小时漂移为负"""
        drift = self.opt.compute_drift(10.0, 5.0, 8.0, 4.0, 20.0, 10.0)
        self.assertLess(drift, 0.0)

    def test_drift_positive_on_increase(self):
        """队列全部增大时漂移为正"""
        drift = self.opt.compute_drift(2.0, 10.0, 2.0, 8.0, 5.0, 20.0)
        self.assertGreater(drift, 0.0)

    def test_energy_critical_reduces_cpu(self):
        """能量告急时投影降低 CPU 功率"""
        raw  = np.array([1.0, 0.9, 1.0])
        proj = self.opt.safety_projection(raw, Q_E=90.0, Q_H=0.0)
        self.assertLess(proj[1], raw[1])

    def test_orbit_critical_raises_propulsion(self):
        """轨道告急时投影提升推进功率"""
        raw  = np.array([0.1, 0.8, 0.8])
        proj = self.opt.safety_projection(raw, Q_E=0.0, Q_H=90.0)
        self.assertGreater(proj[0], raw[0])

    def test_projected_action_in_unit_cube(self):
        """投影后动作始终在 [0,1]^3"""
        raw = np.array([0.5, 0.5, 0.5])
        for Q_E, Q_H in [(0, 0), (80, 0), (0, 80), (80, 80)]:
            with self.subTest(Q_E=Q_E, Q_H=Q_H):
                proj = self.opt.safety_projection(raw, Q_E, Q_H)
                self.assertTrue(np.all(proj >= 0.0) and np.all(proj <= 1.0))


# ═══════════════════════════════════════════════════════════════════════
# 10. 主链路烟雾测试（optimized 路径）
# ═══════════════════════════════════════════════════════════════════════
class TestPipelineSmoke(unittest.TestCase):

    def setUp(self):
        from environment.satellite_env import VLEOSatelliteEnv
        from environment.wrappers import DilatedFrameStackWrapper
        from scheduler.integrated_scheduler import IntegratedScheduler
        from config import DRL_CONFIG

        self.base_env = VLEOSatelliteEnv(seed=123)
        self.stack_k = int(DRL_CONFIG.get("frame_stack", 8))
        self.state_dim = int(DRL_CONFIG.get("state_dim", 30))
        self.env = DilatedFrameStackWrapper(self.base_env, k=self.stack_k)
        self.scheduler = IntegratedScheduler(device="cpu", enable_lyapunov=True, use_psf=True)

    def test_observation_dim(self):
        obs = self.base_env.reset()
        self.assertEqual(obs.shape, (self.state_dim,))

    def test_dilated_wrapper_shape(self):
        state = self.env.reset()
        self.assertEqual(state.shape, (self.stack_k, self.state_dim))

    def test_scheduler_output_valid(self):
        state = self.env.reset()
        in_window = bool(self.env._contact.get("in_window", False)) if self.env._contact else False
        action, was_projected, raw_action, psf_meta = self.scheduler.schedule(
            state,
            self.env.energy_queue.value,
            self.env.orbit_queue.value,
            self.env.data_queue.length,
            self.env.comm_queue.value,
            in_window=in_window,
            evaluate=True,
            h=self.env.altitude_m,
            soc=self.env.battery.soc,
            time_s=self.env.time_s,
            orbital_phase=self.env.orbit_sim.phase,
            env=self.env,
        )

        self.assertEqual(action.shape, (self.scheduler.agent.action_dim,))
        self.assertEqual(raw_action.shape, (self.scheduler.agent.action_dim,))
        self.assertTrue(np.all(action >= 0.0) and np.all(action <= 1.0))
        self.assertIsInstance(was_projected, bool)
        self.assertIn("total_modification_l2", psf_meta)
        self.assertEqual(psf_meta.get("safety_operator"), "Pi_safe")
        self.assertIn("physical_feasibility_projected", psf_meta)
        self.assertIn("ls_psf_projected", psf_meta)
        self.assertIn("implementation_safeguard_projected", psf_meta)

    def test_scheduler_uses_paper_algorithm_entrypoint(self):
        from algorithms.decoupled_constraint_sac import DecoupledConstraintSAC
        from drl.agent import SACAgent

        self.assertIsInstance(self.scheduler.agent, DecoupledConstraintSAC)
        self.assertIsNot(
            DecoupledConstraintSAC.compute_actor_objective,
            SACAgent.compute_actor_objective,
        )
        self.assertIsNot(
            DecoupledConstraintSAC.compute_td_targets,
            SACAgent.compute_td_targets,
        )

    def test_scheduler_can_defer_batch_updates(self):
        """多环境批量入池后，应按 update_freq 集中触发对应次数的网络更新"""
        from config import DRL_CONFIG

        update_freq = max(1, int(DRL_CONFIG.get("update_freq", 1)))
        stored_steps = update_freq * 2
        state = np.zeros((self.stack_k, self.state_dim), dtype=np.float32)
        action = np.zeros(3, dtype=np.float32)

        for _ in range(stored_steps):
            self.scheduler.store_transition(
                state, action, 0.0, state, done=False, lya_drift=0.0,
                terminated=False,
            )

        calls = []
        original_update = self.scheduler.agent.update

        def fake_update():
            calls.append(1)
            return {"critic_loss": 0.0}

        self.scheduler.agent.update = fake_update
        try:
            updates = self.scheduler.trigger_scheduled_updates(stored_steps)
        finally:
            self.scheduler.agent.update = original_update

        self.assertEqual(len(calls), 2)
        self.assertEqual(len(updates), 2)

    def test_evaluate_temporarily_uses_requested_data_scale(self):
        """周期评估可按当前课程难度运行，且结束后恢复环境原配置"""
        import train

        state = self.env.reset()
        original_scale = 0.73
        self.base_env._data_arrival_scale = original_scale
        self.scheduler.schedule = lambda *args, **kwargs: (
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            False,
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            {"total_modification_l2": 0.0},
        )
        stats = train.evaluate(self.env, self.scheduler, n_episodes=1, data_scale=0.25)

        self.assertAlmostEqual(stats["eval_data_arrival_scale"], 0.25)
        self.assertAlmostEqual(self.base_env._data_arrival_scale, original_scale)

    def test_evaluate_action_temporarily_disables_dropout(self):
        """评估动作必须在 eval 模式下算，避免 Dropout 影响 checkpoint 选择。"""
        state = self.env.reset()
        self.scheduler.agent.actor.train()
        seen_training_flags = []
        original_sample = self.scheduler.agent.actor.sample

        def wrapped_sample(tensor_state):
            seen_training_flags.append(self.scheduler.agent.actor.training)
            return original_sample(tensor_state)

        self.scheduler.agent.actor.sample = wrapped_sample
        try:
            self.scheduler.agent.select_action(state, evaluate=True)
        finally:
            self.scheduler.agent.actor.sample = original_sample

        self.assertEqual(seen_training_flags, [False])
        self.assertTrue(self.scheduler.agent.actor.training)

    def test_state_normalizer_updates_on_store_but_not_eval(self):
        """RunningMeanStd 只在训练采样入池时更新，评估推理必须冻结统计量。"""
        state = self.env.reset()
        next_state, _, _, _ = self.env.step(np.zeros(3, dtype=np.float32))
        before = float(self.scheduler.agent.state_rms.count)

        self.scheduler.store_transition(
            state,
            np.zeros(3, dtype=np.float32),
            0.0,
            next_state,
            done=False,
            lya_drift=0.0,
            terminated=False,
        )
        after_store = float(self.scheduler.agent.state_rms.count)
        self.scheduler.agent.select_action(next_state, evaluate=True)

        self.assertGreater(after_store, before)
        self.assertAlmostEqual(float(self.scheduler.agent.state_rms.count), after_store)

    def test_state_normalizer_changes_network_input_scale(self):
        """送入网络前的状态应经过 RunningMeanStd 标准化。"""
        state = self.env.reset()
        agent = self.scheduler.agent
        agent.state_rms.mean = np.ones((self.state_dim,), dtype=np.float64) * 0.5
        agent.state_rms.var = np.ones((self.state_dim,), dtype=np.float64) * 0.25
        normalized = agent._normalize_states_np(state[None, ...])

        self.assertEqual(normalized.shape, (1, self.stack_k, self.state_dim))
        self.assertFalse(np.allclose(normalized[0], state))

    @unittest.skip("simplified scheduler no longer instantiates the old PSF predictor")
    def test_scheduler_synchronizes_psf_physics_from_env(self):
        """鲁棒/trace 扰动后，PSF 预测器必须同步当前环境物理参数。"""
        state = self.env.reset()
        self.env.orbit_dyn.atm.rho_ref *= 3.0
        self.env.solar.eta *= 0.4

        self.scheduler.schedule(
            state,
            self.env.energy_queue.value,
            self.env.orbit_queue.value,
            self.env.data_queue.length,
            self.env.comm_queue.value,
            in_window=False,
            evaluate=True,
            h=self.env.altitude_m,
            soc=self.env.battery.soc,
            time_s=self.env.time_s,
            orbital_phase=self.env.orbit_sim.phase,
            env=self.env,
        )

        self.assertAlmostEqual(
            self.scheduler.psf.predictor.rho_ref,
            self.env.orbit_dyn.atm.rho_ref,
        )
        self.assertAlmostEqual(
            self.scheduler.psf.predictor.eta_solar,
            self.env.solar.eta,
        )


# ═══════════════════════════════════════════════════════════════════════
# 12. PSF 物理兜底语义（队列高压时只要物理安全就不应拦截）
# ═══════════════════════════════════════════════════════════════════════
class TestCheckpointMetadata(unittest.TestCase):

    def test_scheduler_metadata_roundtrip(self):
        from scheduler.integrated_scheduler import IntegratedScheduler
        from config import OBJECTIVE_VERSION

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "meta.pt")

            scheduler = IntegratedScheduler(
                device="cpu", enable_lyapunov=False, use_psf=False)
            scheduler.save(ckpt)

            restored = IntegratedScheduler(
                device="cpu", enable_lyapunov=True, use_psf=True)
            metadata = restored.load(ckpt)

            self.assertFalse(restored.enable_lyapunov)
            self.assertFalse(restored.use_psf)
            self.assertIsNone(restored.psf)
            self.assertEqual(metadata.get("enable_lyapunov"), False)
            self.assertEqual(metadata.get("use_psf"), False)
            self.assertEqual(metadata.get("objective_version"), OBJECTIVE_VERSION)
            self.assertNotRegex(OBJECTIVE_VERSION, r"_v\d+\b")
            self.assertEqual(
                metadata["objective_summary"]["risk_boundaries"]["altitude_crash_km"],
                122.0,
            )
            self.assertEqual(
                set(metadata["reward_weights"].keys()),
                {"w_delivered_value", "w_deadline_success"},
            )
            self.assertIn("constraint_variant", metadata)
            self.assertIn("variant_key", metadata)
            self.assertIn("variant_code", metadata)
            self.assertIn("seed", metadata)
            self.assertIn("total_steps", metadata)
            self.assertIn("ablation_axis", metadata)
            self.assertIn("value_aux_head_enable", metadata)
            self.assertIn("value_aux_loss_weight", metadata)
            self.assertIn("value_aux_loss_weight_final", metadata)
            self.assertIn("value_aux_high_pressure_margin", metadata)
            self.assertIn("value_aux_low_pressure_margin", metadata)
            self.assertIn("adaptive_lyapunov_constraint_norm", metadata)
            self.assertIn("adaptive_lyapunov_constraint_threshold", metadata)
            self.assertIn("behavior_cloning_conservative_weight_coeff", metadata)
            self.assertIn("projection_penalty_coeff", metadata["constraint_cost_config"])

    def test_scheduler_load_can_preserve_ablation_switches(self):
        """消融实验加载权重时，不应被 checkpoint metadata 覆盖安全开关。"""
        from scheduler.integrated_scheduler import IntegratedScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "meta.pt")

            full_scheduler = IntegratedScheduler(
                device="cpu", enable_lyapunov=True, use_psf=True)
            full_scheduler.agent.set_lyapunov_penalty_coeff(1.234)
            full_scheduler.save(ckpt)

            ablation_scheduler = IntegratedScheduler(
                device="cpu", enable_lyapunov=False, use_psf=False)
            ablation_scheduler.agent.set_lyapunov_penalty_coeff(0.0)
            metadata = ablation_scheduler.load(ckpt, restore_safety_config=False)

            self.assertFalse(ablation_scheduler.enable_lyapunov)
            self.assertFalse(ablation_scheduler.use_psf)
            self.assertIsNone(ablation_scheduler.psf)
            self.assertAlmostEqual(
                ablation_scheduler.agent.get_lyapunov_penalty_coeff(),
                0.0,
            )
            self.assertEqual(metadata.get("enable_lyapunov"), True)
            self.assertEqual(metadata.get("use_psf"), False)

    def test_checkpoint_restores_adaptive_lyapunov_coeff(self):
        """自适应后的全局 Lyapunov 权重要随 checkpoint 一起恢复。"""
        from scheduler.integrated_scheduler import IntegratedScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "lya_coeff.pt")
            scheduler = IntegratedScheduler(
                device="cpu", enable_lyapunov=True, use_psf=False)
            scheduler.agent.set_lyapunov_penalty_coeff(1.234)
            scheduler.save(ckpt)

            restored = IntegratedScheduler(
                device="cpu", enable_lyapunov=True, use_psf=False)
            restored.load(ckpt)

        self.assertAlmostEqual(
            restored.agent.get_lyapunov_penalty_coeff(),
            1.234,
            places=6,
        )


class TestRewardSemantics(unittest.TestCase):

    def test_timeliness_weight_has_smooth_overdue_tail(self):
        from environment.task_value_model import TaskBatch
        from config import DRL_CONFIG, TASK_CONFIG

        batch = TaskBatch(
            mb=1.0,
            value=1.0,
            priority=1.0,
            quality=1.0,
            deadline_steps=10,
            created_step=0,
        )
        floor = float(TASK_CONFIG["deadline_decay_floor"])
        grace = int(TASK_CONFIG["overdue_grace_steps"])
        rate = float(TASK_CONFIG["overdue_decay_rate"])

        at_deadline = batch.timeliness_weight(
            10, floor=floor, overdue_grace_steps=grace, overdue_decay_rate=rate)
        just_overdue = batch.timeliness_weight(
            11, floor=floor, overdue_grace_steps=grace, overdue_decay_rate=rate)
        too_late = batch.timeliness_weight(
            10 + grace + 1, floor=floor,
            overdue_grace_steps=grace, overdue_decay_rate=rate)

        self.assertAlmostEqual(at_deadline, floor)
        self.assertGreater(just_overdue, 0.0)
        self.assertLess(just_overdue, at_deadline)
        self.assertAlmostEqual(too_late, 0.0)

    def test_overdue_tasks_do_not_jump_to_front_of_queue(self):
        from environment.task_value_model import TaskValueTracker, TaskBatch
        from config import TASK_CONFIG

        tracker = TaskValueTracker(TASK_CONFIG)
        tracker.raw_batches = [
            TaskBatch(
                mb=10.0,
                value=100.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=5,
                created_step=0,
                scene_name="overdue_high",
            ),
            TaskBatch(
                mb=10.0,
                value=20.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=40,
                created_step=35,
                scene_name="fresh_low",
            ),
        ]

        tracker.process(1.0, now_step=40)

        self.assertEqual(tracker.processed_batches[0].scene_name, "fresh_low")

    def test_expired_high_value_uses_nominal_value_class(self):
        from copy import deepcopy

        from environment.task_value_model import TaskValueTracker, TaskBatch
        from config import TASK_CONFIG

        cfg = deepcopy(TASK_CONFIG)
        cfg["overdue_grace_steps"] = 0
        tracker = TaskValueTracker(cfg)
        tracker.raw_batches = [
            TaskBatch(
                mb=10.0,
                value=30.0,
                priority=3.0,
                quality=1.0,
                deadline_steps=1,
                created_step=0,
                scene_name="expired_high",
            )
        ]

        expire_info = tracker.expire(now_step=2)

        self.assertAlmostEqual(float(expire_info["expired_high_value"]), 30.0, places=6)

    def test_orbital_phase_maps_to_semantic_scene_profiles(self):
        from environment.satellite_env import VLEOSatelliteEnv

        env = VLEOSatelliteEnv(seed=31)
        env.reset()
        env._scene_phase_offset_fraction = 0.0
        military = env._scene_context_for_phase(phase=0.20 * 2.0 * np.pi)
        cloud = env._scene_context_for_phase(phase=0.90 * 2.0 * np.pi)

        self.assertEqual(military["scene_name"], "military")
        self.assertEqual(cloud["scene_name"], "cloud_ocean")
        self.assertGreater(military["scene_class_code"], cloud["scene_class_code"])
        self.assertGreater(military["arrival_multiplier"], cloud["arrival_multiplier"])

    def test_scene_profile_drives_generated_task_value(self):
        from environment.satellite_env import VLEOSatelliteEnv
        from environment.task_value_model import TaskValueTracker
        from config import TASK_CONFIG

        env = VLEOSatelliteEnv(seed=32)
        env.reset()
        env._scene_phase_offset_fraction = 0.0
        military_scene = env._scene_context_for_phase(phase=0.20 * 2.0 * np.pi)
        cloud_scene = env._scene_context_for_phase(phase=0.90 * 2.0 * np.pi)

        high_tracker = TaskValueTracker(TASK_CONFIG)
        low_tracker = TaskValueTracker(TASK_CONFIG)
        high = high_tracker.add_arrival(
            100.0, np.random.default_rng(123), 0,
            scene_context=military_scene)
        low = low_tracker.add_arrival(
            100.0, np.random.default_rng(123), 0,
            scene_context=cloud_scene)

        self.assertGreater(high["generated_value"], low["generated_value"] * 10.0)
        self.assertLess(high["generated_deadline_steps"], low["generated_deadline_steps"])
        self.assertEqual(high["scene_name"], "military")
        self.assertEqual(low["scene_name"], "cloud_ocean")

    def test_emergency_disaster_event_generates_urgent_high_priority_tasks(self):
        from environment.satellite_env import VLEOSatelliteEnv
        from environment.task_value_model import TaskValueTracker
        from config import TASK_CONFIG

        keys = {
            "emergency_event_enable",
            "emergency_event_probability_per_step",
            "emergency_event_duration_steps",
            "emergency_event_cooldown_steps",
        }
        old = {key: TASK_CONFIG.get(key) for key in keys}
        try:
            TASK_CONFIG.update({
                "emergency_event_enable": True,
                "emergency_event_probability_per_step": 1.0,
                "emergency_event_duration_steps": (18, 18),
                "emergency_event_cooldown_steps": 0,
            })
            env = VLEOSatelliteEnv(seed=33)
            env.reset()
            env._scene_phase_offset_fraction = 0.0

            active = env._advance_emergency_event_state()
            emergency_scene = env._scene_context_for_phase()
            disaster_scene = env._scene_context_for_phase(phase=0.30 * 2.0 * np.pi)

            emergency_tracker = TaskValueTracker(TASK_CONFIG)
            disaster_tracker = TaskValueTracker(TASK_CONFIG)
            emergency = emergency_tracker.add_arrival(
                100.0, np.random.default_rng(321), 0,
                scene_context=emergency_scene)
            ordinary = disaster_tracker.add_arrival(
                100.0, np.random.default_rng(321), 0,
                scene_context=disaster_scene)
        finally:
            for key, value in old.items():
                if value is None:
                    TASK_CONFIG.pop(key, None)
                else:
                    TASK_CONFIG[key] = value

        self.assertTrue(active)
        self.assertEqual(emergency_scene["scene_name"], "emergency_disaster")
        self.assertTrue(emergency_scene["emergency_event_triggered"])
        self.assertGreater(emergency["generated_priority"], ordinary["generated_priority"])
        self.assertLess(
            emergency["generated_deadline_steps"],
            ordinary["generated_deadline_steps"],
        )

    def test_reward_uses_delivered_value_as_primary_signal(self):
        from environment.satellite_env import VLEOSatelliteEnv
        from config import REWARD_CONFIG

        env = VLEOSatelliteEnv(seed=7)
        reward, breakdown = env._compute_reward(
            data_info={"serviced": 5.0, "overflow_mb": 0.0},
            batt_info={"energy_margin_wh": 10.0, "is_safe": True},
            orbit_info={"altitude_m": env._h_min + 50e3, "is_safe": True},
            eq_info={"drift": 0.0},
            oq_info={"drift": 0.0},
            cq_info={"urgency": 0.0, "urgency_raw": 0.0, "overflow_mb": 0.0},
            actual_tx_mb=12.0,
            in_window=True,
            power_info={"P_total_w": 20.0},
            delivery_info={
                "delivered_value": 18.0,
                "on_time_delivered_value": 18.0,
                "expired_value": 0.0,
                "dropped_value": 0.0,
            },
        )

        self.assertAlmostEqual(
            breakdown["r_delivered_value"],
            REWARD_CONFIG["w_delivered_value"] * 18.0,
        )
        self.assertAlmostEqual(
            breakdown["r_deadline_success"],
            REWARD_CONFIG["w_deadline_success"] * 18.0,
        )
        self.assertNotIn("r_processed", breakdown)
        self.assertNotIn("r_energy_cost", breakdown)
        self.assertNotIn("r_delay_cost", breakdown)
        self.assertNotIn("r_drop_value", breakdown)
        self.assertNotIn("r_energy", breakdown)
        self.assertNotIn("r_orbit", breakdown)
        self.assertNotIn("r_thermal", breakdown)
        self.assertNotIn("r_tx", breakdown)
        self.assertGreater(reward, 0.0)

    def test_throughput_reward_variant_ignores_value_for_ablation(self):
        from objectives.mission_reward import compute_mission_reward

        reward = compute_mission_reward(
            delivered_value=999.0,
            on_time_delivered_value=999.0,
            expired_value=0.0,
            dropped_value=0.0,
            transmitted_mb=3.5,
            processed_mb=12.0,
            total_power_w=20.0,
            dt_s=1.0,
            cfg={"reward_mode": "throughput", "w_delivered_mb": 2.0},
        )

        self.assertAlmostEqual(reward.total, 7.0)
        self.assertAlmostEqual(reward.components["r_delivered_value"], 0.0)
        self.assertAlmostEqual(reward.components["r_deadline_success"], 0.0)
        self.assertAlmostEqual(reward.components["r_delivered_mb"], 7.0)
        self.assertEqual(reward.components["reward_objective"], "throughput")

    def test_value_aware_reward_uses_capacity_gated_processing_penalty(self):
        """重构后：处理惩罚由容量门控决定，不再是固定值；processed_mb 越多、headroom 越小惩罚越重。"""
        from objectives.mission_reward import compute_mission_reward

        cfg_with_capacity = {
            "w_delivered_value": 1.0,
            "w_deadline_success": 0.2,
            "_processed_queue_mb": 0.0,       # 队列为空
            "_future_contact_capacity_mb": 200.0,  # 充足的未来容量
            "processing_capacity_margin": 0.70,
            "w_processing_penalty_useful": -0.01,
            "w_processing_penalty_overflow": -1.0,
            "w_energy_penalty": -5.0,
            "w_drop_penalty": -0.5,
            "w_drop_mb_penalty": -0.1,
        }
        cfg_no_capacity = dict(cfg_with_capacity,
                               _future_contact_capacity_mb=0.0)  # 无未来容量

        common = dict(
            delivered_value=25.0,
            on_time_delivered_value=20.0,
            expired_value=0.0,
            dropped_value=0.0,
            transmitted_mb=4.0,
            total_power_w=90.0,
            dt_s=10.0,
        )

        # headroom 充足时处理成本低（仅小惩罚）
        reward_small = compute_mission_reward(processed_mb=5.0, cfg=cfg_with_capacity, **common)
        # 无容量时处理会触发 overflow 强惩罚
        reward_overflow = compute_mission_reward(processed_mb=5.0, cfg=cfg_no_capacity, **common)
        # headroom 充足时多处理比少处理惩罚略多（但远低于 overflow）
        reward_large_in_headroom = compute_mission_reward(processed_mb=50.0, cfg=cfg_with_capacity, **common)

        self.assertLess(reward_overflow.total, reward_small.total)           # overflow 更贵
        self.assertLess(reward_large_in_headroom.total, reward_small.total)  # 处理越多成本越高
        self.assertGreater(reward_overflow.components["processed_into_overflow_mb"], 0.0)
        self.assertAlmostEqual(reward_small.components["processed_into_overflow_mb"], 0.0, places=6)
        self.assertNotIn("r_processed", reward_small.components)
        self.assertEqual(reward_small.components["reward_objective"], "value_aware_deliverability_gated")

    def test_paper_reward_config_contains_only_objective_weights(self):
        """论文版 reward 配置只保留论文目标里明确定义的权重（容量门控重构后）。"""
        from config import PROCESSING_CREDIT_CONFIG, REWARD_CONFIG

        # 重构后：固定的 w_processing_penalty 被容量门控的分段惩罚取代；
        # 新增了 w_drop_mb_penalty（按 MB 计算的丢弃代价）和门控配置项。
        self.assertIn("w_delivered_value", REWARD_CONFIG)
        self.assertIn("w_deadline_success", REWARD_CONFIG)
        self.assertIn("w_energy_penalty", REWARD_CONFIG)
        self.assertIn("w_drop_penalty", REWARD_CONFIG)
        self.assertIn("w_drop_mb_penalty", REWARD_CONFIG)
        self.assertIn("w_processing_penalty_useful", REWARD_CONFIG)
        self.assertIn("w_processing_penalty_overflow", REWARD_CONFIG)
        self.assertIn("processing_capacity_margin", REWARD_CONFIG)
        # 固定的 w_processing_penalty 已被分段惩罚取代，不应再出现
        self.assertNotIn("w_processing_penalty", REWARD_CONFIG)
        self.assertNotIn("w_deliverable_processing", REWARD_CONFIG)
        self.assertNotIn("w_deliverable_processing_initial", REWARD_CONFIG)
        self.assertIn("w_deliverable_processing_initial", PROCESSING_CREDIT_CONFIG)
        # processing credit 正向奖励已关闭
        self.assertEqual(float(PROCESSING_CREDIT_CONFIG["w_deliverable_processing_initial"]), 0.0)
        self.assertEqual(float(PROCESSING_CREDIT_CONFIG["w_deliverable_processing_final"]), 0.0)

    def test_processing_credit_schedule_is_separate_from_paper_reward(self):
        from config import PROCESSING_CREDIT_CONFIG
        from environment.satellite_env import VLEOSatelliteEnv

        env = VLEOSatelliteEnv(seed=11)
        warmup = int(PROCESSING_CREDIT_CONFIG["deliverable_processing_credit_warmup_steps"])
        anneal = int(PROCESSING_CREDIT_CONFIG["deliverable_processing_credit_anneal_steps"])

        env.step_count = 0
        early_cfg = env._reward_config_for_step()
        env.step_count = warmup + anneal // 2
        mid_cfg = env._reward_config_for_step()
        env.step_count = warmup + anneal
        late_cfg = env._reward_config_for_step()

        self.assertNotIn("w_deliverable_processing", early_cfg)
        self.assertNotIn("w_deliverable_processing", mid_cfg)
        self.assertNotIn("w_deliverable_processing", late_cfg)

    def test_deliverable_processing_credit_requires_near_window_capacity_and_high_gate(self):
        from environment.satellite_env import VLEOSatelliteEnv

        env = VLEOSatelliteEnv(seed=12)
        env.comm_queue.value = 0.0
        env.task_tracker.deadline_contact_stats = lambda *_args, **_kwargs: {
            "raw_high_next_window_deliverable_ratio": 0.8,
            "processed_high_next_window_deliverable_ratio": 0.6,
            "high_value_deadline_contact_mismatch": 0.1,
        }
        info = {
            "processed_high_value": 100.0,
            "processed_medium_value": 100.0,
            "processed_low_value": 1e6,
            "future_capacity_mb": 1000.0,
        }

        env._contact = {"in_window": False, "time_to_next_window_s": 3600.0}
        credit_far = env._deliverable_processing_credit(info)
        self.assertEqual(credit_far, 0.0)

        env._contact = {"in_window": True, "time_to_next_window_s": 0.0}
        credit_with_low = env._deliverable_processing_credit(info)
        credit_without_low = env._deliverable_processing_credit({
            "processed_high_value": 100.0,
            "processed_medium_value": 100.0,
            "future_capacity_mb": 1000.0,
        })
        self.assertGreater(credit_with_low, 0.0)
        self.assertAlmostEqual(credit_with_low, credit_without_low, places=7)

        env._contact = {"in_window": True, "time_to_next_window_s": 0.0}
        env.comm_queue.value = 1000.0
        self.assertEqual(env._deliverable_processing_credit(info), 0.0)

    def test_processing_credit_component_respects_w_deliverable_processing_weight(self):
        """r_deliverable_processing 分量完全由 w_deliverable_processing 权重控制；
        config 中该权重为 0 时，credit 分量为 0（无论 deliverable_processing_credit_value 多大）。"""
        from objectives.mission_reward import compute_mission_reward

        # w_deliverable_processing=0.0（config 中已关闭），credit 分量应为 0
        reward_no_credit = compute_mission_reward(
            delivered_value=0.0,
            on_time_delivered_value=0.0,
            expired_value=0.0,
            dropped_value=0.0,
            transmitted_mb=0.0,
            processed_mb=0.0,
            total_power_w=0.0,
            dt_s=10.0,
            cfg={
                "w_delivered_value": 1.0,
                "w_deadline_success": 0.2,
                "w_deliverable_processing": 0.0,  # 已关闭
            },
            deliverable_processing_credit_value=1000.0,
        )
        self.assertAlmostEqual(
            reward_no_credit.components["r_deliverable_processing"], 0.0, places=7)

        # w_deliverable_processing=0.5 时，credit 分量应非零
        reward_with_credit = compute_mission_reward(
            delivered_value=1.0,
            on_time_delivered_value=1.0,
            expired_value=0.0,
            dropped_value=0.0,
            transmitted_mb=1.0,
            processed_mb=0.0,
            total_power_w=0.0,
            dt_s=10.0,
            cfg={
                "w_delivered_value": 1.0,
                "w_deadline_success": 0.2,
                "w_deliverable_processing": 0.5,
            },
            deliverable_processing_credit_value=10.0,
        )
        self.assertAlmostEqual(
            reward_with_credit.components["r_deliverable_processing"], 5.0, places=7)

    def test_reward_excludes_battery_cycle_degradation_from_td_target(self):
        """电池老化属于安全/寿命约束，不再污染 reward Q 的 TD target。"""
        from environment.satellite_env import VLEOSatelliteEnv

        env = VLEOSatelliteEnv(seed=9)
        _, breakdown = env._compute_reward(
            data_info={"serviced": 0.0, "overflow_mb": 0.0},
            batt_info={
                "energy_margin_wh": 10.0,
                "is_safe": True,
                "capacity_loss_wh": 0.25,
            },
            orbit_info={"altitude_m": env._h_min + 50e3, "is_safe": True},
            eq_info={"drift": 0.0},
            oq_info={"drift": 0.0},
            cq_info={"urgency": 0.0, "urgency_raw": 0.0, "overflow_mb": 0.0},
            actual_tx_mb=0.0,
            in_window=False,
            power_info={"P_total_w": 20.0},
            delivery_info={
                "delivered_value": 0.0,
                "on_time_delivered_value": 0.0,
                "expired_value": 0.0,
                "dropped_value": 0.0,
            },
        )

        self.assertNotIn("r_battery_degradation", breakdown)

    def test_terminal_failure_enters_constraint_cost_not_reward(self):
        """坠毁/深度过放等终止风险应进入 c_t，而不是 reward TD target。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from constraints.safety_cost import compute_state_safety_penalty

        env = VLEOSatelliteEnv(seed=9)
        _, breakdown = env._compute_reward(
            data_info={"serviced": 0.0, "overflow_mb": 0.0},
            batt_info={"energy_margin_wh": -5.0, "is_safe": False, "is_crashed": True},
            orbit_info={"altitude_m": env._h_min + 50e3, "is_safe": True},
            eq_info={"drift": 0.0},
            oq_info={"drift": 0.0},
            cq_info={"urgency": 0.0, "urgency_raw": 0.0, "overflow_mb": 0.0},
            actual_tx_mb=0.0,
            in_window=False,
            power_info={"P_total_w": 20.0},
            delivery_info={
                "delivered_value": 0.0,
                "on_time_delivered_value": 0.0,
                "expired_value": 0.0,
                "dropped_value": 0.0,
            },
        )

        self.assertNotIn("r_catastrophic", breakdown)
        self.assertGreater(
            compute_state_safety_penalty({"risk_stage": "failure", "energy_crashed": True}),
            0.0,
        )

    def test_thermal_failure_is_not_double_counted(self):
        from constraints.safety_cost import compute_state_safety_penalty

        cost = compute_state_safety_penalty(
            {
                "risk_stage": "failure",
                "crashed": True,
                "thermal_crashed": True,
                "thermal_stage": "failure",
                "thermal_safe": False,
            },
            cfg={
                "constraint_stage_costs": {"failure": 3.0},
                "constraint_failure_cost": 3.0,
                "constraint_thermal_warning_cost": 0.08,
            },
        )

        self.assertAlmostEqual(cost, 3.0, places=6)

    def test_sac_td_targets_keep_lyapunov_constraint_separate(self):
        """Reward Q 的 TD 目标不能直接混入 Lyapunov 漂移。"""
        import torch
        from drl.agent import SACAgent

        agent = object.__new__(SACAgent)
        agent.gamma = 0.9
        reward = torch.tensor([[1.0]])
        done = torch.tensor([[0.0]])
        lya = torch.tensor([[5.0]])
        reward_next_q = torch.tensor([[2.0]])
        constraint_next_q = torch.tensor([[3.0]])

        target_q, target_c = SACAgent._compute_td_targets(
            agent, reward, done, lya, reward_next_q, constraint_next_q)

        self.assertAlmostEqual(float(target_q.item()), 2.8, places=6)
        self.assertAlmostEqual(float(target_c.item()), 7.7, places=6)

    def test_adaptive_lyapunov_coeff_tracks_pressure_and_clip(self):
        """全局 Lyapunov 权重应随安全压力升降，并服从上下界。"""
        from config import DRL_CONFIG
        from train import _adaptive_lyapunov_coeff_step

        keys = {
            "adaptive_lyapunov_coeff_min",
            "adaptive_lyapunov_coeff_max",
            "adaptive_lyapunov_constraint_threshold",
            "adaptive_lyapunov_coeff_target_pressure",
            "adaptive_lyapunov_coeff_lr",
            "adaptive_lyapunov_coeff_ema_beta",
        }
        old = {key: DRL_CONFIG.get(key) for key in keys}
        try:
            DRL_CONFIG.update({
                "adaptive_lyapunov_coeff_min": 0.1,
                "adaptive_lyapunov_coeff_max": 0.6,
                "adaptive_lyapunov_constraint_threshold": 0.2,
                "adaptive_lyapunov_coeff_target_pressure": 0.2,
                "adaptive_lyapunov_coeff_lr": 1.0,
                "adaptive_lyapunov_coeff_ema_beta": 0.0,
            })

            high, raw_high, ema_high = _adaptive_lyapunov_coeff_step(
                0.5, 0.0, 0.4, enabled=True)
            low, raw_low, ema_low = _adaptive_lyapunov_coeff_step(
                0.2, 0.0, 0.0, enabled=True)
            disabled, _, _ = _adaptive_lyapunov_coeff_step(
                0.5, 0.0, 1.0, enabled=False)
        finally:
            for key, value in old.items():
                if value is None:
                    DRL_CONFIG.pop(key, None)
                else:
                    DRL_CONFIG[key] = value

        self.assertAlmostEqual(raw_high, 0.4)
        self.assertAlmostEqual(ema_high, 0.4)
        self.assertAlmostEqual(high, 0.6)
        self.assertAlmostEqual(raw_low, 0.0)
        self.assertAlmostEqual(ema_low, 0.0)
        self.assertAlmostEqual(low, 0.1)
        self.assertAlmostEqual(disabled, 0.5)

    def test_adaptive_dual_uses_constraint_cost_not_projection_rate(self):
        import inspect
        import train

        source = inspect.getsource(train.train)

        self.assertIn("adaptive_constraint_value", source)
        self.assertIn("adaptive_constraint_violation", source)
        self.assertIn('"adaptive_dual_source": "constraint_cost"', source)
        self.assertNotIn("max(lya_proj_rate_now, projected_ema)", source)

    def test_reward_td_excludes_safety_action_penalties(self):
        import inspect
        import train

        source = inspect.getsource(train.train)
        start = source.index("reward_for_td = reward")
        end = source.index("reward_scaled = reward_for_td", start)
        reward_window = source[start:end]

        self.assertNotIn("projection_penalty", reward_window)
        self.assertNotIn("action_mod_penalty", reward_window)
        self.assertIn("reward_td_excludes_safety_action_penalty", source)

    def test_unified_safety_cost_matches_lyapunov_drift_and_queue_risk(self):
        """约束 Critic 的 c_t 应由归一化 Lyapunov 漂移和队列风险组成。"""
        from constraints.safety_cost import compute_lyapunov_safety_cost

        cfg = {
            "lya_soft_util_threshold": 0.75,
            "lya_soft_util_penalty_coeff": 0.5,
            "lya_soft_penalty_clip": 1.0,
            "lya_hard_overflow_penalty_coeff": 3.0,
            "lya_hard_penalty_clip": 2.0,
            "lyapunov_drift_clip": 3.0,
        }
        cost = compute_lyapunov_safety_cost(
            previous_queues=(0.0, 0.0, 75.0, 20.0),
            next_queues=(0.0, 0.0, 90.0, 90.0),
            queue_maxes=(100.0, 100.0, 100.0, 100.0),
            info={"overflow_mb": 10.0, "comm_overflow_mb": 0.0},
            cfg=cfg,
        )

        expected_drift = 0.5 * ((0.90 ** 2 - 0.75 ** 2) + (0.90 ** 2 - 0.20 ** 2))
        self.assertAlmostEqual(cost.lyapunov_drift, expected_drift, places=6)
        self.assertAlmostEqual(cost.positive_lyapunov_drift, expected_drift, places=6)
        self.assertGreater(cost.queue_soft_penalty, 0.0)
        self.assertGreater(cost.queue_hard_penalty, 0.0)
        self.assertEqual(cost.state_safety_penalty, 0.0)
        self.assertAlmostEqual(
            cost.soft_constraint_cost,
            cost.positive_lyapunov_drift + cost.queue_soft_penalty,
            places=6,
        )
        self.assertAlmostEqual(
            cost.hard_violation_cost,
            cost.queue_hard_penalty,
            places=6,
        )
        self.assertAlmostEqual(
            cost.raw_cost,
            cost.soft_constraint_cost + cost.hard_violation_cost,
            places=6,
        )
        self.assertAlmostEqual(cost.dual_cost, cost.raw_cost, places=6)
        self.assertGreaterEqual(cost.dual_violation_signal, 0.0)
        self.assertEqual(cost.clipped_cost, min(cost.raw_cost, 3.0))

    def test_queue_hard_cost_reads_raw_and_processed_overflow_aliases(self):
        from constraints.safety_cost import compute_queue_risk_penalties

        cfg = {
            "lya_soft_util_threshold": 0.95,
            "lya_soft_util_penalty_coeff": 0.0,
            "lya_soft_penalty_clip": 1.0,
            "lya_hard_overflow_penalty_coeff": 4.0,
            "lya_hard_penalty_clip": 100.0,
        }
        _, legacy_hard = compute_queue_risk_penalties(
            qd=100.0,
            qc=100.0,
            qd_max=100.0,
            qc_max=100.0,
            info={"overflow_mb": 50.0, "comm_overflow_mb": 25.0},
            cfg=cfg,
            include_processed_backlog=False,
        )
        _, alias_hard = compute_queue_risk_penalties(
            qd=100.0,
            qc=100.0,
            qd_max=100.0,
            qc_max=100.0,
            info={"raw_queue_overflow_mb": 50.0, "processed_queue_overflow_mb": 25.0},
            cfg=cfg,
            include_processed_backlog=False,
        )
        _, severe_alias_hard = compute_queue_risk_penalties(
            qd=100.0,
            qc=100.0,
            qd_max=100.0,
            qc_max=100.0,
            info={"raw_queue_overflow_mb": 50.0, "processed_queue_overflow_mb": 100.0},
            cfg=cfg,
            include_processed_backlog=False,
        )

        self.assertAlmostEqual(alias_hard, legacy_hard, places=6)
        self.assertGreater(severe_alias_hard, alias_hard)

    def test_thermal_excess_enters_constraint_cost_not_reward(self):
        from constraints.safety_cost import compute_lyapunov_safety_cost

        cost = compute_lyapunov_safety_cost(
            previous_queues=(0.0, 0.0, 0.0, 0.0),
            next_queues=(0.0, 0.0, 0.0, 0.0),
            queue_maxes=(100.0, 100.0, 100.0, 100.0),
            info={"thermal_temperature_c": 55.0, "thermal_stage": "warning"},
            cfg={
                "constraint_thermal_excess": {"coeff": 0.25, "norm_c": 10.0},
                "constraint_stage_costs": {
                    "warning": 0.0,
                    "unsafe": 0.0,
                    "failure": 0.0,
                },
                "constraint_auxiliary_violation_cost": 0.0,
                "constraint_thermal_warning_cost": 0.08,
                "constraint_warning_cost": 0.0,
                "constraint_unsafe_cost": 0.0,
                "constraint_failure_cost": 0.0,
                "constraint_power_violation_cost": 0.0,
            },
        )

        self.assertGreater(cost.thermal_excess_penalty, 0.0)
        self.assertGreater(cost.raw_cost, cost.positive_lyapunov_drift)

    def test_constraint_cost_exposes_value_aware_components(self):
        from constraints.safety_cost import compute_lyapunov_safety_cost

        cost = compute_lyapunov_safety_cost(
            previous_queues=(0.0, 0.0, 10.0, 10.0),
            next_queues=(0.0, 0.0, 20.0, 80.0),
            queue_maxes=(100.0, 100.0, 100.0, 100.0),
            info={
                "soc": 0.10,
                "altitude_km": 160.0,
                "P_total_w": 90.0,
                "delivered_value": 0.0,
                "expired_high_value": 5000.0,
                "dropped_high_value": 0.0,
            },
            cfg={
                "lya_soft_util_threshold": 0.50,
                "lya_soft_util_penalty_coeff": 0.5,
                "lya_soft_penalty_clip": 2.0,
                "lya_hard_overflow_penalty_coeff": 3.0,
                "lya_hard_penalty_clip": 2.0,
                "lyapunov_drift_clip": 10.0,
                "constraint_high_value_loss_coeff": 1.0,
                "constraint_task_loss_value_norm": 5000.0,
                "constraint_task_loss_clip": 2.0,
                "constraint_energy_margin_coeff": 0.25,
                "constraint_orbit_margin_coeff": 0.25,
                "constraint_efficiency_cost_coeff": 0.02,
                "constraint_efficiency_power_norm_w": 120.0,
            },
        )

        self.assertGreater(cost.queue_cost, 0.0)
        self.assertGreater(cost.energy_cost, 0.0)
        self.assertGreater(cost.orbit_cost, 0.0)
        self.assertGreater(cost.task_loss_cost, 0.0)
        self.assertEqual(cost.efficiency_cost, 0.0)
        self.assertAlmostEqual(cost.total_cost, cost.raw_cost, places=6)

    def test_processed_backlog_penalty_is_covered_by_over_processing_constraint(self):
        from config import DRL_CONFIG
        from constraints.safety_cost import compute_queue_risk_penalties

        for key in [
            "lya_processed_backlog_threshold",
            "lya_processed_backlog_coeff",
            "lya_processed_backlog_clip",
            "constraint_processed_backlog_threshold",
            "constraint_processed_backlog_coeff",
            "constraint_processed_backlog_clip",
        ]:
            self.assertIn(key, DRL_CONFIG)

        soft, hard = compute_queue_risk_penalties(
            qd=0.0,
            qc=60.0,
            qd_max=100.0,
            qc_max=100.0,
            info={},
            cfg={
                "lya_soft_util_threshold": 0.95,
                "lya_soft_util_penalty_coeff": 0.0,
                "lya_soft_penalty_clip": 1.0,
                "lya_hard_overflow_penalty_coeff": 0.0,
                "lya_hard_penalty_clip": 1.0,
                "lya_processed_backlog_threshold": 0.35,
                "lya_processed_backlog_coeff": 0.3,
                "lya_processed_backlog_clip": 0.5,
            },
        )

        self.assertAlmostEqual(soft, 0.0, places=6)
        self.assertAlmostEqual(hard, 0.0, places=6)

    def test_learning_first_constraint_defaults_are_sufficiently_sensitive(self):
        from config import DRL_CONFIG, PROCESSING_CREDIT_CONFIG, TASK_CONFIG

        self.assertEqual(DRL_CONFIG["state_dim"], 62)
        self.assertTrue(bool(DRL_CONFIG["enable_capacity_aware_cost_v2"]))
        self.assertFalse(bool(DRL_CONFIG["enable_deliverable_processing_reward"]))
        self.assertEqual(DRL_CONFIG["queue_projection_policy"], "safety_algorithms_only")
        self.assertTrue(bool(DRL_CONFIG["enable_deployment_queue_projection"]))

        self.assertEqual(DRL_CONFIG["constraint_efficiency_processed_value_credit"], 0.0)
        self.assertEqual(DRL_CONFIG["constraint_efficiency_cost_coeff"], 0.0)
        self.assertEqual(DRL_CONFIG["constraint_window_waste_coeff"], 0.0)
        self.assertEqual(DRL_CONFIG["constraint_processed_backlog_coeff"], 0.0)
        self.assertEqual(DRL_CONFIG["constraint_low_value_waste_coeff"], 0.0)
        self.assertEqual(DRL_CONFIG["constraint_unproductive_cpu_coeff"], 0.0)
        self.assertLessEqual(DRL_CONFIG["constraint_over_processing_coeff"], 2.0)
        self.assertLessEqual(DRL_CONFIG["constraint_over_processing_clip"], 10.0)
        self.assertAlmostEqual(DRL_CONFIG["constraint_capacity_norm_mb"], 400.0)
        self.assertAlmostEqual(DRL_CONFIG["constraint_capacity_norm"], 400.0)
        self.assertLessEqual(DRL_CONFIG["constraint_over_processing_ratio_weight"], 1.0)
        self.assertAlmostEqual(DRL_CONFIG["constraint_future_capacity_margin"], 0.80)

        self.assertEqual(DRL_CONFIG["value_action_aux_loss_weight"], 0.0)
        self.assertEqual(DRL_CONFIG["value_action_aux_loss_weight_final"], 0.0)
        self.assertAlmostEqual(
            DRL_CONFIG["value_aux_processed_future_contact_threshold"],
            0.75,
        )
        self.assertAlmostEqual(
            PROCESSING_CREDIT_CONFIG["w_deliverable_processing_initial"], 0.0)
        self.assertAlmostEqual(
            PROCESSING_CREDIT_CONFIG["w_deliverable_processing_final"], 0.0)
        self.assertAlmostEqual(
            PROCESSING_CREDIT_CONFIG["deliverable_processing_credit_cap_fraction"], 0.08)
        self.assertAlmostEqual(
            PROCESSING_CREDIT_CONFIG["deliverable_processing_near_window_s"], 120.0)
        self.assertAlmostEqual(
            PROCESSING_CREDIT_CONFIG["deliverable_processing_max_future_ratio"], 0.45)
        self.assertAlmostEqual(
            PROCESSING_CREDIT_CONFIG["deliverable_processing_min_high_gate"], 0.75)
        self.assertEqual(
            PROCESSING_CREDIT_CONFIG["deliverable_processing_credit_warmup_steps"], 20000)
        self.assertEqual(
            PROCESSING_CREDIT_CONFIG["deliverable_processing_credit_anneal_steps"], 80000)
        self.assertEqual(
            PROCESSING_CREDIT_CONFIG["deliverable_processing_mid_value_weight"], 0.0)

        self.assertAlmostEqual(TASK_CONFIG["time_to_next_window_norm_s"], 5400.0)
        self.assertAlmostEqual(TASK_CONFIG["class_high_residual_value_density"], 3.0)
        self.assertAlmostEqual(TASK_CONFIG["class_medium_residual_value_density"], 1.20)
        self.assertAlmostEqual(TASK_CONFIG["low_residual_value_density_threshold"], 1.20)
        self.assertLessEqual(TASK_CONFIG["low_value_drop_max_mbs"], 0.8)
        self.assertLessEqual(TASK_CONFIG["active_low_drop_floor_ratio"], 0.01)
        self.assertGreaterEqual(TASK_CONFIG["low_drop_resource_pressure_threshold"], 0.10)

        self.assertAlmostEqual(DRL_CONFIG["lyapunov_drift_clip"], 20.0)
        self.assertAlmostEqual(DRL_CONFIG["lyapunov_penalty_coeff"], 0.0)
        self.assertFalse(bool(DRL_CONFIG["adaptive_lyapunov_coeff_enable"]))
        self.assertAlmostEqual(DRL_CONFIG["adaptive_lyapunov_constraint_threshold"], 0.10)
        self.assertEqual(
            DRL_CONFIG["adaptive_lyapunov_coeff_target_pressure"],
            DRL_CONFIG["adaptive_lyapunov_constraint_threshold"],
        )
        self.assertGreaterEqual(DRL_CONFIG["adaptive_lyapunov_coeff_min"], 0.20)
        self.assertLessEqual(DRL_CONFIG["adaptive_lyapunov_coeff_max"], 3.0)
        self.assertAlmostEqual(DRL_CONFIG["adaptive_lyapunov_constraint_norm"], 10.0)
        self.assertAlmostEqual(DRL_CONFIG["adaptive_lyapunov_constraint_signal_max"], 3.0)

    def test_window_waste_cost_is_legacy_diagnostic_stub(self):
        from constraints.safety_cost import compute_lyapunov_safety_cost

        cost = compute_lyapunov_safety_cost(
            previous_queues=(0.0, 0.0, 30.0, 60.0),
            next_queues=(0.0, 0.0, 30.0, 60.0),
            queue_maxes=(100.0, 100.0, 100.0, 100.0),
            info={
                "in_window": True,
                "processed_queue_utilization": 0.6,
                "processed_queue_mb": 60.0,
                "delivered_mb": 1.0,
                "link_tx_capacity_mb": 20.0,
                "effective_tx_capacity_mb": 20.0,
                "rf_tx_capacity_mb": 20.0,
            },
            cfg={
                "lyapunov_drift_clip": 10.0,
                "constraint_window_waste_coeff": 1.5,
                "constraint_window_waste_clip": 1.5,
                "constraint_window_waste_backlog_threshold": 0.1,
                "constraint_window_waste_target_tx_util": 0.70,
            },
        )
        self.assertEqual(cost.window_waste_cost, 0.0)
        self.assertAlmostEqual(
            cost.soft_constraint_cost,
            cost.positive_lyapunov_drift
            + cost.queue_soft_penalty
            + cost.thermal_cost
            + cost.energy_cost
            + cost.orbit_cost
            + cost.over_processing_cost,
            places=6,
        )

    def test_low_value_is_diagnostic_and_over_processing_enters_v3_soft_cost(self):
        from constraints.safety_cost import compute_lyapunov_safety_cost

        cost = compute_lyapunov_safety_cost(
            previous_queues=(0.0, 0.0, 0.0, 40.0),
            next_queues=(0.0, 0.0, 0.0, 40.0),
            queue_maxes=(100.0, 100.0, 100.0, 100.0),
            info={
                "processed_low_mb_step": 20.0,
                "delivered_low_mb": 5.0,
                "raw_low_mb": 50.0,
                "processed_low_mb": 30.0,
                "processed_queue_mb": 40.0,
                "processed_mb": 200.0,
                "processed_since_contact_mb": 2000.0,
                "delivered_since_contact_mb": 0.0,
                "future_contact_capacity_mb": 100.0,
                "comm_queue_max": 1000.0,
                "in_window": False,
                "time_to_next_window_s": 2000.0,
                "alpha_cpu": 0.7,
                "raw_high_mb": 0.0,
                "raw_mid_mb": 0.0,
                "processed_high_mb": 0.0,
                "processed_mid_mb": 0.0,
            },
            cfg={
                "enable_capacity_aware_cost_v2": True,
                "lyapunov_drift_clip": 10.0,
                "constraint_low_value_waste_coeff": 2.0,
                "constraint_low_value_waste_clip": 2.0,
                "constraint_low_value_waste_norm_mb": 100.0,
                "constraint_over_processing_coeff": 3.0,
                "constraint_over_processing_clip": 4.0,
                "constraint_capacity_norm_mb": 800.0,
                "constraint_future_capacity_margin": 0.90,
                "constraint_unproductive_cpu_coeff": 0.0,
                "constraint_unproductive_cpu_clip": 0.0,
                "constraint_prepass_min_lead_s": 300.0,
                "constraint_prepass_lead_margin": 1.3,
            },
        )

        self.assertEqual(cost.low_value_waste_cost, 0.0)
        self.assertGreater(cost.over_processing_cost, 0.0)
        self.assertEqual(cost.unproductive_cpu_cost, 0.0)
        self.assertAlmostEqual(
            cost.soft_constraint_cost,
            cost.positive_lyapunov_drift
            + cost.queue_soft_penalty
            + cost.thermal_cost
            + cost.energy_cost
            + cost.orbit_cost
            + cost.over_processing_cost,
            places=6,
        )
        self.assertGreaterEqual(cost.soft_constraint_cost, cost.over_processing_cost)

    def test_active_low_drop_legacy_cost_is_stubbed(self):
        from constraints.legacy_safety_cost import compute_low_value_waste_cost

        cost = compute_low_value_waste_cost(
            info={
                "processed_low_mb_step": 0.0,
                "delivered_low_mb": 0.0,
                "raw_low_mb": 0.0,
                "processed_low_mb": 0.0,
                "active_low_drop_mb": 20.0,
            },
            cfg={
                "constraint_low_value_waste_coeff": 2.0,
                "constraint_low_value_waste_clip": 2.0,
                "constraint_low_value_waste_norm_mb": 100.0,
            },
        )

        self.assertEqual(cost, 0.0)

    def test_over_processing_cost_uses_episode_cumulative_admission(self):
        """即使 processed queue 被清理，累计处理量过多也应进入 CMDP 约束。"""
        from constraints.safety_cost import compute_over_processing_cost

        cost = compute_over_processing_cost(
            info={
                "processed_queue_mb": 0.0,
                "future_contact_capacity_mb": 100.0,
                "episode_processed_mb": 1000.0,
                "episode_delivered_mb": 100.0,
            },
            cfg={
                "constraint_over_processing_coeff": 5.0,
                "constraint_over_processing_clip": 4.0,
                "constraint_capacity_norm_mb": 800.0,
                "constraint_future_capacity_margin": 0.90,
            },
        )
        higher_cost = compute_over_processing_cost(
            info={
                "processed_queue_mb": 0.0,
                "future_contact_capacity_mb": 100.0,
                "episode_processed_mb": 2000.0,
                "episode_delivered_mb": 100.0,
            },
            cfg={
                "constraint_over_processing_coeff": 5.0,
                "constraint_over_processing_clip": 20.0,
                "constraint_capacity_norm_mb": 500.0,
                "constraint_future_capacity_margin": 0.90,
                "constraint_over_processing_ratio_weight": 1.0,
            },
        )

        self.assertGreater(cost, 0.0)
        self.assertGreater(higher_cost, cost)

    def test_over_processing_cost_is_monotonic_with_capacity_ratio(self):
        from constraints.safety_cost import compute_over_processing_cost

        cfg = {
            "constraint_over_processing_coeff": 10.0,
            "constraint_over_processing_clip": 100.0,
            "constraint_capacity_norm_mb": 100.0,
            "constraint_future_capacity_margin": 0.80,
            "constraint_over_processing_ratio_weight": 3.0,
        }

        def _cost_for_ratio(ratio: float) -> float:
            return compute_over_processing_cost(
                info={
                    "processed_queue_mb": 100.0 * ratio,
                    "future_contact_capacity_mb": 100.0,
                    "episode_processed_mb": 100.0 * ratio,
                    "episode_delivered_mb": 0.0,
                },
                cfg=cfg,
            )

        c12 = _cost_for_ratio(1.2)
        c20 = _cost_for_ratio(2.0)
        c30 = _cost_for_ratio(3.0)

        self.assertGreater(c12, 0.0)
        self.assertGreater(c20, c12)
        self.assertGreater(c30, c20)

    def test_unproductive_cpu_legacy_cost_is_stubbed(self):
        from constraints.legacy_safety_cost import compute_unproductive_cpu_cost

        cfg = {
            "constraint_unproductive_cpu_coeff": 1.5,
            "constraint_unproductive_cpu_clip": 2.0,
            "constraint_prepass_min_lead_s": 300.0,
            "constraint_prepass_lead_margin": 1.3,
            "constraint_unproductive_cpu_processed_ratio_threshold": 0.75,
            "constraint_unproductive_cpu_far_horizon_s": 5400.0,
            "constraint_unproductive_cpu_deadline_mismatch_weight": 0.75,
        }
        far_cost = compute_unproductive_cpu_cost(
            info={
                "in_window": False,
                "time_to_next_window_s": 3600.0,
                "alpha_cpu": 0.8,
                "processed_mb": 40.0,
                "processed_queue_mb": 900.0,
                "processed_queue_future_contact_ratio": 1.2,
                "raw_high_mb": 120.0,
                "raw_mid_mb": 60.0,
                "processed_high_mb": 20.0,
                "processed_mid_mb": 10.0,
                "future_contact_capacity_mb": 500.0,
                "processed_high_next_window_deliverable_ratio": 0.2,
                "raw_high_next_window_deliverable_ratio": 0.3,
                "high_value_deadline_contact_mismatch": 0.8,
            },
            cfg=cfg,
        )
        near_cost = compute_unproductive_cpu_cost(
            info={
                "in_window": False,
                "time_to_next_window_s": 120.0,
                "alpha_cpu": 0.8,
                "processed_mb": 40.0,
                "processed_queue_mb": 900.0,
                "processed_queue_future_contact_ratio": 1.2,
                "raw_high_mb": 120.0,
                "raw_mid_mb": 60.0,
                "processed_high_mb": 20.0,
                "processed_mid_mb": 10.0,
                "future_contact_capacity_mb": 500.0,
                "processed_high_next_window_deliverable_ratio": 0.2,
                "raw_high_next_window_deliverable_ratio": 0.3,
                "high_value_deadline_contact_mismatch": 0.8,
            },
            cfg=cfg,
        )

        self.assertEqual(far_cost, 0.0)
        self.assertEqual(near_cost, 0.0)

    def test_processed_backlog_legacy_cost_is_stubbed(self):
        from constraints.legacy_safety_cost import compute_processed_backlog_cost

        cost = compute_processed_backlog_cost(
            0.7,
            cfg={
                "constraint_processed_backlog_threshold": 0.08,
                "constraint_processed_backlog_coeff": 4.0,
            },
        )

        self.assertEqual(cost, 0.0)

    def test_value_aware_heuristic_baseline_uses_grouped_action_schema(self):
        from baselines.heuristic_baseline import ValueAwareHeuristicBaseline
        from environment.satellite_env import OBSERVATION_FEATURES
        from utils.action_space import decode_grouped_action
        from config import DRL_CONFIG

        idx = {name: i for i, name in enumerate(OBSERVATION_FEATURES)}
        state = np.zeros((int(DRL_CONFIG["state_dim"]),), dtype=np.float32)
        state[idx["altitude_norm"]] = 1.0
        state[idx["soc"]] = 0.8
        state[idx["solar_input_norm"]] = 1.0
        state[idx["future_contact_capacity_norm"]] = 0.1
        state[idx["raw_queue_utilization"]] = 0.8
        state[idx["processed_queue_utilization"]] = 0.7
        state[idx["raw_high_queue_utilization"]] = 0.4
        state[idx["raw_low_queue_utilization"]] = 0.5
        state[idx["processed_high_queue_utilization"]] = 0.4
        state[idx["processed_low_queue_utilization"]] = 0.5

        action = ValueAwareHeuristicBaseline().schedule(state)
        decoded = decode_grouped_action(action)

        self.assertEqual(action.shape, (int(DRL_CONFIG["action_dim"]),))
        self.assertGreater(decoded.cpu_value_weight, -0.5)
        self.assertGreater(action[7], 0.0)

    def test_llf_baseline_outputs_grouped_priority_action(self):
        from types import SimpleNamespace
        from baselines.value_baselines import LLFBaseline
        from utils.action_space import decode_grouped_action

        class DummyTracker:
            def topk_stats(self, step_count):
                return {"deadline_urgency": 0.9, "expiring_value": 1000.0}

        env = SimpleNamespace(
            _contact={"in_window": True},
            data_queue=SimpleNamespace(length=80.0, max_length=100.0),
            comm_queue=SimpleNamespace(value=60.0, max_value=100.0),
            battery=SimpleNamespace(soc=0.8),
            altitude_m=350e3,
            task_tracker=DummyTracker(),
            step_count=0,
        )

        action = LLFBaseline().schedule(np.zeros(40, dtype=np.float32), env)
        decoded = decode_grouped_action(action)

        self.assertEqual(action.shape[0], 8)
        self.assertGreaterEqual(decoded.cpu_value_weight, 0.0)
        self.assertGreaterEqual(decoded.tx_value_weight, 0.0)

    def test_compare_all_separates_value_aware_heuristic_and_diagnostics(self):
        import inspect

        import experiments.compare_all as compare_all
        from baselines.heuristic_baseline import ValueAwareHeuristicBaseline

        source = inspect.getsource(compare_all.run_compare_all)
        eval_source = inspect.getsource(compare_all.evaluate_on_env)

        self.assertIs(compare_all.ValueAwareHeuristicBaseline, ValueAwareHeuristicBaseline)
        self.assertIn("Value-aware Heuristic", source)
        self.assertIn("LLF", source)
        self.assertIn("proc_dl_ratio", eval_source)
        self.assertIn("high_value_delivery_rate", eval_source)
        self.assertIn("Ours + CPU throttle (deployment)", source)
        self.assertIn("Ours w/o Work-Conserving", source)
        self.assertIn("diagnostic_results", source)
        self.assertIn("include_deployment_ablations", source)
        self.assertIn("allow_missing_ours", source)

    def test_compare_all_rejects_zero_delivery_formal_table(self):
        from experiments.compare_all import OURS_NAME, _paper_table_delivery_check

        with self.assertRaises(RuntimeError):
            _paper_table_delivery_check(
                {"A": {"delivered_value_mean": 0.0, "downlink_mean": 0.0, "processed_mean": 5.0}},
                allow_zero_delivery=False,
            )
        check = _paper_table_delivery_check(
            {"A": {"delivered_value_mean": 0.0, "downlink_mean": 0.0, "processed_mean": 5.0}},
            allow_zero_delivery=True,
        )
        self.assertFalse(check["nonzero_delivery"])
        self.assertEqual(check["max_processed_mb"], 5.0)

        with self.assertRaises(RuntimeError):
            _paper_table_delivery_check(
                {
                    OURS_NAME: {
                        "delivered_value_mean": 0.0,
                        "downlink_mean": 0.0,
                        "processed_mean": 12.0,
                    },
                    "Baseline": {
                        "delivered_value_mean": 5.0,
                        "downlink_mean": 2.0,
                        "processed_mean": 7.0,
                    },
                },
                allow_zero_delivery=False,
            )
        ours_check = _paper_table_delivery_check(
            {
                OURS_NAME: {
                    "delivered_value_mean": 0.0,
                    "downlink_mean": 0.0,
                    "processed_mean": 12.0,
                },
                "Baseline": {
                    "delivered_value_mean": 5.0,
                    "downlink_mean": 2.0,
                    "processed_mean": 7.0,
                },
            },
            allow_zero_delivery=True,
        )
        self.assertTrue(ours_check["nonzero_delivery"])
        self.assertFalse(ours_check["main_method_nonzero_delivery"])
        self.assertEqual(ours_check["main_method_processed_mb"], 12.0)

    def test_paper_metrics_include_voi_scheduling_diagnostics(self):
        from utils.paper_metrics import compact_paper_table_row

        row = compact_paper_table_row({
            "overall_safe_rate": 0.9,
            "survival_rate": 1.0,
            "delivered_value_mean": 12.0,
            "downlink_mean": 3.0,
            "processed_mean": 9.0,
            "proc_dl_ratio": 3.0,
            "global_proc_downlink_ratio": 2.5,
            "mean_episode_proc_downlink_ratio": 3.5,
            "chain_total_rate": 0.11,
            "boundary_clip_rate_eval": 0.12,
            "lyapunov_projected_rate_eval": 0.13,
            "psf_modified_rate": 0.14,
            "comm_window_utilization": 0.4,
            "high_value_delivery_rate": 0.7,
            "value_weighted_deadline_success_rate": 0.8,
            "value_weighted_aoi_steps": 4.5,
            "voi_loss_rate": 0.2,
            "processed_queue_final_utilization": 0.6,
            "tx_active_in_contact_ratio": 0.75,
            "high_value_delivery_ratio": 0.9,
        })

        self.assertEqual(row["Proc/DL Ratio"], 2.5)
        self.assertEqual(row["Global Proc/DL Ratio"], 2.5)
        self.assertEqual(row["Mean Episode Proc/DL Ratio"], 3.5)
        self.assertEqual(row["Total Action Modification Rate"], 0.11)
        self.assertEqual(row["Physical Projection Rate"], 0.12)
        self.assertEqual(row["Lyapunov Projection Rate"], 0.13)
        self.assertEqual(row["PSF Intervention Rate"], 0.14)
        self.assertEqual(row["Window Utilization"], 0.4)
        self.assertEqual(row["High-value Delivery Rate"], 0.7)
        self.assertEqual(row["Value-weighted Deadline Success"], 0.8)
        self.assertEqual(row["Value-weighted AoI"], 4.5)
        self.assertEqual(row["VoI Loss Rate"], 0.2)
        self.assertEqual(row["Processed Queue Final Utilization"], 0.6)
        self.assertEqual(row["TX Active in Contact Ratio"], 0.75)
        self.assertEqual(row["High-value Delivery Ratio"], 0.9)

    def test_compare_all_detects_same_checkpoint_path(self):
        from experiments.compare_all import _same_checkpoint_path

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "best.pt")
            self.assertTrue(_same_checkpoint_path(ckpt, os.path.abspath(ckpt)))

    def test_reward_ignores_processed_queue_pressure(self):
        from environment.satellite_env import VLEOSatelliteEnv

        env = VLEOSatelliteEnv(seed=8)
        _, breakdown = env._compute_reward(
            data_info={"serviced": 40.0, "overflow_mb": 0.0},
            batt_info={"energy_margin_wh": 10.0, "is_safe": True},
            orbit_info={"altitude_m": env._h_min + 50e3, "is_safe": True},
            eq_info={"drift": 0.0},
            oq_info={"drift": 0.0},
            cq_info={"urgency": 0.9, "urgency_raw": 0.9, "overflow_mb": 0.0},
            actual_tx_mb=0.0,
            in_window=False,
            power_info={"P_total_w": 20.0},
            delivery_info={
                "delivered_value": 0.0,
                "on_time_delivered_value": 0.0,
                "expired_value": 0.0,
                "dropped_value": 0.0,
            },
        )

        self.assertNotIn("r_processed", breakdown)
        self.assertNotIn("_processed_shaping_gate", breakdown)

    def test_task_queue_uses_deadline_weighted_score_to_avoid_hol_blocking(self):
        """近截止任务应能越过远截止高名义价值任务，避免队头阻塞。"""
        from environment.task_value_model import TaskValueTracker, TaskBatch
        from config import TASK_CONFIG

        tracker = TaskValueTracker(TASK_CONFIG)
        tracker.raw_batches = [
            TaskBatch(
                mb=100.0,
                value=1000.0,
                priority=3.0,
                quality=1.0,
                deadline_steps=10000,
                created_step=0,
                scene_name="far_military",
            ),
            TaskBatch(
                mb=10.0,
                value=20.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=5,
                created_step=0,
                scene_name="urgent_ocean",
            ),
        ]

        tracker.process(5.0, now_step=0)

        self.assertEqual(tracker.processed_batches[0].scene_name, "urgent_ocean")

    def test_active_low_drop_uses_residual_value_density_not_deadline_promotion(self):
        """低密度但紧急升类任务不应被 active low-drop 误删。"""
        from copy import deepcopy

        from environment.task_value_model import TaskValueTracker, TaskBatch
        from config import TASK_CONFIG

        cfg = deepcopy(TASK_CONFIG)
        tracker = TaskValueTracker(cfg)
        tracker.raw_batches = [
            TaskBatch(
                mb=5.0,
                value=0.5,  # density=0.1，名义低价值
                priority=0.1,
                quality=1.0,
                deadline_steps=101,
                created_step=0,
                scene_name="urgent_promoted",
            ),
            TaskBatch(
                mb=5.0,
                value=0.5,
                priority=0.1,
                quality=1.0,
                deadline_steps=300,
                created_step=0,
                scene_name="true_low",
            ),
        ]

        # now_step=100 时，第一批 remaining=1，会因 deadline pressure 升到 Mid。
        self.assertEqual(tracker.task_class_id(tracker.raw_batches[0], now_step=100), 2)
        self.assertEqual(tracker.task_class_id(tracker.raw_batches[1], now_step=100), 2)
        drop_info = tracker.drop_low_value(5.0, now_step=100, drop_context={"resource_pressure": 1.0})

        self.assertAlmostEqual(float(drop_info["active_dropped_low_raw_mb"]), 5.0, places=6)
        self.assertAlmostEqual(float(drop_info["active_dropped_raw_high_value"]), 0.0, places=6)
        self.assertTrue(any(b.scene_name == "urgent_promoted" for b in tracker.raw_batches))
        self.assertFalse(any(b.scene_name == "true_low" for b in tracker.raw_batches))

    def test_active_low_drop_respects_residual_value_density_threshold(self):
        """主动丢弃只应删除剩余价值密度足够低的动态 Low 数据。"""
        from copy import deepcopy

        from environment.task_value_model import TaskValueTracker, TaskBatch
        from config import TASK_CONFIG

        cfg = deepcopy(TASK_CONFIG)
        cfg["low_residual_value_density_threshold"] = 1.20
        tracker = TaskValueTracker(cfg)
        tracker.raw_batches = [
            TaskBatch(
                mb=10.0,
                value=13.0,
                priority=0.1,
                quality=1.0,
                deadline_steps=300,
                created_step=0,
                scene_name="residual_still_useful",
            ),
            TaskBatch(
                mb=10.0,
                value=1.0,
                priority=0.1,
                quality=1.0,
                deadline_steps=300,
                created_step=0,
                scene_name="residual_low",
            ),
        ]

        drop_info = tracker.drop_low_value(
            10.0,
            now_step=0,
            drop_context={"resource_pressure": 1.0},
        )

        self.assertAlmostEqual(float(drop_info["active_dropped_low_raw_mb"]), 10.0, places=6)
        self.assertTrue(any(b.scene_name == "residual_still_useful" for b in tracker.raw_batches))
        self.assertFalse(any(b.scene_name == "residual_low" for b in tracker.raw_batches))

    def test_task_classification_uses_residual_density(self):
        """分类应当由剩余价值密度决定，而不是单看原始 value_density。"""
        from environment.task_value_model import TaskValueTracker, TaskBatch
        from config import TASK_CONFIG

        tracker = TaskValueTracker(TASK_CONFIG)
        batch = TaskBatch(
            mb=10.0,
            value=13.0,
            priority=0.2,
            quality=1.0,
            deadline_steps=5,
            created_step=0,
            scene_name="deadline_sensitive",
        )

        self.assertEqual(tracker.task_class_id(batch, now_step=0), 1)
        self.assertEqual(tracker.task_class_id(batch, now_step=4), 2)

    def test_grouped_budget_supports_work_conserving_reallocation(self):
        """某类预算未用完时，应可回流到其他有任务的类别。"""
        from copy import deepcopy

        from environment.task_value_model import TaskValueTracker, TaskBatch
        from config import TASK_CONFIG

        cfg = deepcopy(TASK_CONFIG)
        cfg["work_conserving_reallocation"] = True
        cfg["cpu_work_conserving_reallocation"] = True
        tracker = TaskValueTracker(cfg)
        tracker.raw_batches = [
            TaskBatch(
                mb=10.0,
                value=15.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=200,
                created_step=0,
                scene_name="mid_only_backlog",
            )
        ]

        result = tracker.process_by_class([10.0, 0.0, 0.0], now_step=0)
        self.assertAlmostEqual(float(result["processed_mb"]), 10.0, places=6)
        self.assertAlmostEqual(float(result["processed_medium_mb"]), 10.0, places=6)

    def test_cpu_and_tx_work_conserving_reallocation_are_split(self):
        from copy import deepcopy

        from environment.task_value_model import TaskValueTracker, TaskBatch
        from config import TASK_CONFIG

        cfg = deepcopy(TASK_CONFIG)
        self.assertFalse(bool(cfg["cpu_work_conserving_reallocation"]))
        self.assertFalse(bool(cfg["tx_work_conserving_reallocation"]))

        tracker = TaskValueTracker(cfg)
        tracker.raw_batches = [
            TaskBatch(
                mb=10.0,
                value=15.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=200,
                created_step=0,
                scene_name="mid_only_raw",
            )
        ]
        cpu_result = tracker.process_by_class([10.0, 0.0, 0.0], now_step=0)
        self.assertAlmostEqual(float(cpu_result["processed_mb"]), 0.0, places=6)
        self.assertAlmostEqual(float(cpu_result["cpu_reallocated_mb"]), 0.0, places=6)

        tracker.processed_batches = [
            TaskBatch(
                mb=10.0,
                value=15.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=200,
                created_step=0,
                scene_name="mid_only_processed",
            )
        ]
        tx_result = tracker.deliver_by_class([10.0, 0.0, 0.0], now_step=0)
        self.assertAlmostEqual(float(tx_result["delivered_mb"]), 0.0, places=6)
        self.assertAlmostEqual(float(tx_result["tx_reallocated_mb"]), 0.0, places=6)

        cfg["tx_work_conserving_reallocation"] = True
        tracker = TaskValueTracker(cfg)
        tracker.processed_batches = [
            TaskBatch(
                mb=10.0,
                value=10.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=200,
                created_step=0,
                scene_name="mid_only_processed",
            )
        ]
        compat_tx_result = tracker.deliver_by_class([10.0, 0.0, 0.0], now_step=0)
        self.assertAlmostEqual(float(compat_tx_result["delivered_mb"]), 10.0, places=6)
        self.assertAlmostEqual(float(compat_tx_result["tx_reallocated_mb"]), 10.0, places=6)

    def test_future_contact_cpu_gate_is_enabled_by_default(self):
        from config import TASK_CONFIG

        self.assertTrue(bool(TASK_CONFIG.get("enable_future_contact_cpu_gate", False)))
        self.assertTrue(bool(TASK_CONFIG.get("enable_cpu_throttle", False)))

    def test_future_contact_cpu_gate_closes_power_and_processing(self):
        from copy import deepcopy

        from config import TASK_CONFIG
        from environment.satellite_env import VLEOSatelliteEnv
        from environment.task_value_model import TaskBatch

        old_task_cfg = deepcopy(TASK_CONFIG)
        try:
            TASK_CONFIG.update({
                "enable_future_contact_cpu_gate": True,
                "enable_cpu_throttle": True,
                "cpu_gate_start_future_ratio": 0.55,
                "cpu_gate_target_future_ratio": 0.75,
                "cpu_gate_hard_stop_future_ratio": 0.90,
                "cpu_gate_far_window_lead_s": 120.0,
                "cpu_gate_floor_alpha": 0.0,
            })
            env = VLEOSatelliteEnv(seed=37)
            env.reset()
            env._data_arrival_scale = 0.0
            env._contact = {"in_window": False, "time_to_next_window_s": 3600.0}
            env._contact_override = {"in_window": False, "time_to_next_window_s": 3600.0}
            env._future_contact_capacity_mb = lambda: 100.0
            env.comm_queue.value = 90.0
            env.data_queue.length = 200.0
            env.task_tracker.raw_batches.append(TaskBatch(
                mb=200.0,
                value=800.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=500,
                created_step=env.step_count,
            ))

            _, _, _, gated_info = env.step(
                np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                enforce_prop_smoothing=False,
            )

            env_no_gate = VLEOSatelliteEnv(seed=37)
            env_no_gate.reset()
            env_no_gate._data_arrival_scale = 0.0
            env_no_gate._contact = {"in_window": False, "time_to_next_window_s": 3600.0}
            env_no_gate._contact_override = {"in_window": False, "time_to_next_window_s": 3600.0}
            env_no_gate._future_contact_capacity_mb = lambda: 100.0
            env_no_gate.comm_queue.value = 90.0
            env_no_gate.data_queue.length = 200.0
            env_no_gate.task_tracker.raw_batches.append(TaskBatch(
                mb=200.0,
                value=800.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=500,
                created_step=env_no_gate.step_count,
            ))
            TASK_CONFIG["enable_future_contact_cpu_gate"] = False
            TASK_CONFIG["enable_cpu_throttle"] = False
            _, _, _, ungated_info = env_no_gate.step(
                np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                enforce_prop_smoothing=False,
            )
        finally:
            TASK_CONFIG.clear()
            TASK_CONFIG.update(old_task_cfg)

        self.assertTrue(bool(gated_info["future_contact_cpu_gate_applied"]))
        self.assertAlmostEqual(float(gated_info["cpu_gate_alpha_cpu_before"]), 1.0, places=6)
        self.assertAlmostEqual(float(gated_info["executed_action"][1]), 0.0, places=6)
        self.assertLess(float(gated_info["P_cpu_w"]), float(ungated_info["P_cpu_w"]))
        self.assertLess(float(gated_info["P_total_w"]), float(ungated_info["P_total_w"]))
        self.assertLess(float(gated_info["service_rate_mbs"]), float(ungated_info["service_rate_mbs"]))
        self.assertLess(float(gated_info["processed_mb"]), float(ungated_info["processed_mb"]))

    def test_step_reward_uses_step_delivery_not_episode_summary(self):
        """reward 只能使用本步交付价值，不能被 episode 累计 summary 覆盖。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from environment.task_value_model import TaskBatch
        from config import REWARD_CONFIG

        env = VLEOSatelliteEnv(seed=15)
        env.reset()
        env._data_arrival_scale = 0.0
        env._contact_override = {"in_window": True, "max_capacity_mbps": 8000.0}
        env.comm_queue.value = 10.0
        env.task_tracker.total_generated_mb = 1_000.0
        env.task_tracker.total_generated_value = 1_000_000.0
        env.task_tracker.total_delivered_mb = 500.0
        env.task_tracker.total_delivered_value = 1_000_000.0
        env.task_tracker.processed_batches.append(TaskBatch(
            mb=10.0,
            value=20.0,
            priority=1.0,
            quality=1.0,
            deadline_steps=500,
            created_step=env.step_count,
        ))

        _, reward, _, info = env.step(np.array([0.0, 0.0, 0.1], dtype=np.float32))
        breakdown = info["reward_breakdown"]

        self.assertGreater(info["episode_delivered_value"], info["delivered_value"])
        self.assertAlmostEqual(breakdown["delivered_value"], info["delivered_value"])
        self.assertLess(
            breakdown["r_delivered_value"],
            REWARD_CONFIG["w_delivered_value"] * 100.0,
        )
        self.assertLess(reward, 200.0)

    def test_step_reports_task_value_and_buffer_safety(self):
        from environment.satellite_env import VLEOSatelliteEnv

        env = VLEOSatelliteEnv(seed=9)
        env.reset()
        _, _, _, info = env.step(np.array([0.3, 0.8, 0.8], dtype=np.float32))
        for key in [
            "raw_queue_mb", "processed_queue_mb", "delivered_value",
            "deadline_success_rate", "expired_value_rate",
            "average_aoi_steps", "value_weighted_aoi_steps",
            "value_weighted_deadline_success_rate", "voi_degradation_rate",
            "voi_loss_rate", "high_value_delivery_ratio",
            "raw_queue_safe", "processed_queue_safe", "power_constraint_safe",
            "available_power_w", "overall_safe", "orbit_crashed",
            "energy_crashed", "thermal_crashed", "crashed", "risk_stage", "risk_stage_code",
            "nominal_state", "warning_state", "unsafe_state", "failure_state",
            "orbit_stage", "energy_stage", "thermal_temperature_c",
            "thermal_margin_norm", "thermal_safe", "thermal_stage",
            "high_value_downlink_mb", "high_value_downlink_value",
            "active_dropped_low_mb", "dropped_low_mb",
            "low_value_dropped_mb", "low_value_dropped_value",
            "processed_value", "processed_high_mb_step",
            "processed_mid_mb_step", "processed_low_mb_step",
            "expired_raw_value", "expired_processed_value",
            "dropped_raw_value", "dropped_processed_value",
            "future_contact_capacity_mb", "processed_queue_future_contact_ratio",
            "processed_since_contact_mb", "delivered_since_contact_mb",
            "episode_processed_mb", "episode_processed_value",
            "episode_delivered_mb", "episode_delivered_value", "episode_proc_dl_ratio",
            "episode_useful_processing_ratio", "useful_processing_ratio",
            "cpu_active_far_from_window_rate",
            "costs",
        ]:
            self.assertIn(key, info)
        self.assertIn("state_safety_cost", info["costs"])

    def test_environment_reports_warning_without_termination(self):
        """170km 属于警告区：episode 不终止，risk_stage=warning。"""
        from environment.satellite_env import VLEOSatelliteEnv

        env = VLEOSatelliteEnv(seed=21)
        env.reset()
        env.altitude_m = 170e3
        env.orbit_queue.reset(env.altitude_m)
        env.battery.soc = 0.8
        _, _, done, info = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))

        self.assertFalse(done)
        self.assertEqual(info["risk_stage"], "warning")
        self.assertEqual(info["orbit_stage"], "warning")
        self.assertTrue(bool(info["orbit_safe"]))

    def test_environment_reports_unsafe_without_crash(self):
        """140km 属于不安全区：安全率下降，但不按坠毁终止。"""
        from environment.satellite_env import VLEOSatelliteEnv

        env = VLEOSatelliteEnv(seed=22)
        env.reset()
        env.altitude_m = 140e3
        env.orbit_queue.reset(env.altitude_m)
        env.battery.soc = 0.8
        _, _, done, info = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))

        self.assertFalse(done)
        self.assertEqual(info["risk_stage"], "unsafe")
        self.assertFalse(bool(info["orbit_safe"]))
        self.assertFalse(bool(info["crashed"]))

    def test_environment_crashed_includes_thermal_crash(self):
        """严重过热应同时进入 thermal_crashed 和总 crashed 统计。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from config import THERMAL_CONFIG

        env = VLEOSatelliteEnv(seed=23)
        env.reset()
        env.thermal_temperature_c = float(THERMAL_CONFIG.get("critical_temp_c", 65.0)) + 20.0

        _, _, done, info = env.step(np.zeros(10, dtype=np.float32))

        self.assertTrue(done)
        self.assertTrue(bool(info["thermal_crashed"]))
        self.assertTrue(bool(info["crashed"]))
        self.assertEqual(info["risk_stage"], "failure")

    def test_environment_power_closure_after_propulsion_smoothing(self):
        """推进器平滑改变动作后，环境仍要把最终执行动作压回可用功率内。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from config import ENERGY_CONFIG

        env = VLEOSatelliteEnv(seed=12)
        env.reset()
        # 强制在阴影区运行，并只给电池留一小段安全裕度，制造有限可用功率预算。
        env.orbit_sim.reset_phase(env.orbit_sim._sunlit_phase + 0.1)
        target_available_w = ENERGY_CONFIG["power_baseline_w"] + 25.0
        safe_margin_wh = target_available_w * (env.dt / 3600.0) / env.battery.eta_discharge
        env.battery.soc = env.battery.soc_min + safe_margin_wh / env.battery.capacity_wh
        env.energy_queue.reset(env.battery.energy_margin_wh)
        env.step_count = 1
        env.prev_action = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        _, _, _, info = env.step(np.array([0.7, 1.0, 1.0], dtype=np.float32))

        self.assertTrue(bool(info["power_execution_clipped"]))
        self.assertTrue(bool(info["power_constraint_safe"]))
        self.assertLessEqual(info["P_total_w"], info["available_power_w"] + 1e-6)
        self.assertGreater(info["requested_total_power_w"], info["available_power_w"])
        self.assertLess(info["executed_action"][0], 1.0)

    def test_propulsion_safety_override_breaks_smoothing_near_orbit_floor(self):
        """轨道贴近底线时，小幅救急推进也不能被 N_PROP_SMOOTH 吞掉。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from config import ENERGY_CONFIG

        env = VLEOSatelliteEnv(seed=17)
        env.reset()
        env.altitude_m = env._h_min + 1_000.0
        env.step_count = 1
        env.prev_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        _, _, _, info = env.step(np.array([0.2, 0.0, 0.0], dtype=np.float32))

        self.assertFalse(bool(info["prop_can_update"]))
        self.assertTrue(bool(info["safety_override"]))
        self.assertEqual(info["prop_safety_override_reason"], "orbit_guard")
        self.assertAlmostEqual(
            float(info["executed_action"][0]),
            ENERGY_CONFIG["propulsion_ignition_threshold_w"]
            / ENERGY_CONFIG["power_propulsion_max_w"],
            places=6,
        )
        self.assertTrue(bool(info["propulsion_ignition_boost_applied"]))

    def test_environment_sanitizes_nonfinite_actions(self):
        """环境执行层必须兜底 NaN/Inf 动作，避免污染物理状态和 replay。"""
        from environment.satellite_env import VLEOSatelliteEnv

        env = VLEOSatelliteEnv(seed=18)
        env.reset()

        _, _, _, info = env.step(np.array([np.nan, np.inf, -np.inf], dtype=np.float32))

        self.assertFalse(bool(info["input_action_in_bounds"]))
        self.assertTrue(np.all(np.isfinite(info["executed_action"])))
        self.assertTrue(np.isfinite(info["P_total_w"]))

    def test_tx_capacity_observation_uses_configured_scale(self):
        """通信容量观测归一化必须随新下传尺度同步，避免状态长期饱和。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from config import QUEUE_CONFIG

        env = VLEOSatelliteEnv(seed=22)
        env.reset()
        norm_mbps = float(QUEUE_CONFIG["tx_capacity_norm_mbps"])
        env._contact_override = {
            "in_window": True,
            "max_capacity_mbps": norm_mbps,
        }
        env._contact = env._get_contact_info()
        obs = env._get_observation()

        self.assertAlmostEqual(float(obs[9]), 1.0, places=6)

    def test_environment_can_skip_second_prop_smoothing_for_scheduler_final_action(self):
        """optimized 安全链路输出的最终动作不应再被环境推进平滑吞掉。"""
        from environment.satellite_env import VLEOSatelliteEnv

        env = VLEOSatelliteEnv(seed=20)
        env.reset()
        env.altitude_m = env._h_min + 50_000.0
        env.step_count = 1
        env.prev_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        _, _, _, info = env.step(
            np.array([0.2, 0.0, 0.0], dtype=np.float32),
            enforce_prop_smoothing=False,
        )

        self.assertFalse(bool(info["prop_can_update"]))
        self.assertFalse(bool(info["prop_smoothing_enforced"]))
        self.assertAlmostEqual(float(info["executed_action"][0]), 0.0, places=6)
        self.assertTrue(bool(info["propulsion_deadband_applied"]))

    def test_scheduler_applies_prop_smoothing_before_safety_layers(self):
        """调度器先处理推进器更新节奏，再让 Lyapunov/PSF 基于可执行原始动作修正。"""
        from scheduler.integrated_scheduler import IntegratedScheduler
        from config import DRL_CONFIG
        from environment.satellite_env import OBSERVATION_FEATURES

        scheduler = IntegratedScheduler(device="cpu", enable_lyapunov=False, use_psf=False)
        scheduler.agent.select_action = lambda state, evaluate=False: np.array(
            [0.7, 0.2, 0.2], dtype=np.float32)
        state_dim = int(DRL_CONFIG.get("state_dim", len(OBSERVATION_FEATURES)))
        state = np.zeros((DRL_CONFIG.get("frame_stack", 8), state_dim), dtype=np.float32)
        state[0, OBSERVATION_FEATURES.index("prev_alpha_prop")] = 0.15

        action, was_projected, raw_action, meta = scheduler.schedule(
            state,
            Q_E=0.0, Q_H=0.0, Q_D=0.0, Q_C=0.0,
            in_window=False,
            evaluate=True,
            prop_can_update=False,
            available_power_w=150.0,
        )

        self.assertAlmostEqual(float(raw_action[0]), 0.7, places=6)
        self.assertAlmostEqual(float(action[0]), 0.0, places=6)
        self.assertTrue(was_projected)
        self.assertTrue(bool(meta["prop_smoothing_applied"]))
        self.assertTrue(bool(meta["propulsion_deadband_applied"]))
        self.assertTrue(bool(meta["actuator_constraint_applied"]))
        self.assertTrue(bool(meta["safety_intervention_projected"]))

    def test_boundary_clip_is_safety_intervention_not_actuator_lock(self):
        """功率/边界裁剪应计入安全介入，但不应和推进器锁定混淆。"""
        from scheduler.integrated_scheduler import IntegratedScheduler
        from config import DRL_CONFIG, ENERGY_CONFIG
        from environment.satellite_env import OBSERVATION_FEATURES

        scheduler = IntegratedScheduler(device="cpu", enable_lyapunov=False, use_psf=False)
        state_dim = int(DRL_CONFIG.get("state_dim", len(OBSERVATION_FEATURES)))
        raw_state = np.zeros((DRL_CONFIG.get("frame_stack", 8), state_dim), dtype=np.float32)
        raw_action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        action, was_projected, _, meta = scheduler._schedule_from_raw_action(
            raw_action,
            raw_state,
            Q_E=0.0,
            Q_H=0.0,
            Q_D=0.0,
            Q_C=0.0,
            in_window=False,
            prop_can_update=True,
            available_power_w=ENERGY_CONFIG["power_baseline_w"] + 10.0,
        )

        self.assertTrue(was_projected)
        self.assertTrue(bool(meta["boundary_clipped"]))
        self.assertTrue(bool(meta["safety_intervention_projected"]))
        self.assertFalse(bool(meta["actuator_constraint_applied"]))
        self.assertLessEqual(
            float(np.dot(action[:3], [
                ENERGY_CONFIG["power_propulsion_max_w"],
                ENERGY_CONFIG["power_cpu_max_w"],
                ENERGY_CONFIG["power_tx_max_w"],
            ])),
            10.0 + 1e-6,
        )

    def test_downlink_is_limited_by_rf_transmitter_rate(self):
        """强链路窗口下，实际下传量仍不能超过发射机功率对应的物理速率。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from environment.task_value_model import TaskBatch

        env = VLEOSatelliteEnv(seed=13)
        env.reset()
        env._contact_override = {"in_window": True, "max_capacity_mbps": 8000.0}
        env.comm_queue.value = 100.0
        env.task_tracker.processed_batches.append(TaskBatch(
            mb=100.0,
            value=100.0,
            priority=1.0,
            quality=1.0,
            deadline_steps=500,
            created_step=env.step_count,
        ))

        _, _, _, info = env.step(np.array([0.0, 0.0, 0.1], dtype=np.float32))

        self.assertGreater(info["link_tx_capacity_mb"], info["rf_tx_capacity_mb"])
        self.assertLessEqual(info["actual_tx_mb"], info["rf_tx_capacity_mb"] + 1e-6)

    def test_downlink_is_limited_by_per_pass_receive_budget(self):
        """单次过顶应有接收容量上限，避免高仰角瞬时容量被当成无限窗口。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from environment.task_value_model import TaskBatch

        env = VLEOSatelliteEnv(seed=34)
        env.reset()
        env._comm_pass_capacity_mb = 20.0
        env._comm_pass_remaining_mb = 20.0
        env._contact_override = {"in_window": True, "max_capacity_mbps": 8000.0}
        env.comm_queue.value = 100.0
        env.task_tracker.processed_batches.append(TaskBatch(
            mb=100.0,
            value=100.0,
            priority=1.0,
            quality=1.0,
            deadline_steps=500,
            created_step=env.step_count,
        ))

        _, _, _, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))

        self.assertLessEqual(info["link_tx_capacity_mb"], 20.0 + 1e-6)
        self.assertLessEqual(info["actual_tx_mb"], 20.0 + 1e-6)
        self.assertLess(info["comm_pass_remaining_mb"], 20.0)

    def test_cpu_backpressure_deployment_projection_is_disabled_by_default(self):
        """主实验默认不让 processed-queue 部署保护硬压 CPU。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from environment.task_value_model import TaskBatch
        from scheduler.integrated_scheduler import IntegratedScheduler

        env = VLEOSatelliteEnv(seed=20)
        state = env.reset()
        env._data_arrival_scale = 0.0
        env._contact_override = {"in_window": False}
        env.data_queue.length = 100.0
        env.comm_queue.value = env.comm_queue.max_value
        env.task_tracker.raw_batches.append(TaskBatch(
            mb=100.0,
            value=100.0,
            priority=1.0,
            quality=1.0,
            deadline_steps=500,
            created_step=env.step_count,
        ))

        scheduler = IntegratedScheduler(device="cpu", enable_lyapunov=False, use_psf=False)
        safe_action, _, _, meta = scheduler._schedule_from_raw_action(
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            state,
            in_window=False,
            h=env.altitude_m,
            prop_can_update=True,
            available_power_w=env._last_available_power_w,
        )
        _, _, _, info = env.step(safe_action, enforce_prop_smoothing=False)

        self.assertFalse(bool(meta["cpu_backpressure_applied"]))
        self.assertFalse(bool(meta["queue_boundary_projected"]))
        self.assertAlmostEqual(float(safe_action[1]), 1.0, places=6)
        self.assertFalse(bool(info["cpu_backpressure_applied"]))

    def test_cpu_backpressure_deployment_projection_can_be_enabled(self):
        """部署保护显式打开时仍可压低 CPU，用于保守实验或线上兜底。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from environment.task_value_model import TaskBatch
        from scheduler.integrated_scheduler import IntegratedScheduler
        from config import DRL_CONFIG

        old_enabled = DRL_CONFIG.get("enable_deployment_queue_projection", False)
        old_policy = DRL_CONFIG.get("queue_projection_policy", "off")
        try:
            DRL_CONFIG["enable_deployment_queue_projection"] = True
            DRL_CONFIG["queue_projection_policy"] = "all"
            env = VLEOSatelliteEnv(seed=20)
            state = env.reset()
            env._data_arrival_scale = 0.0
            env._contact_override = {"in_window": False}
            env.data_queue.length = 100.0
            env.comm_queue.value = env.comm_queue.max_value
            env.task_tracker.raw_batches.append(TaskBatch(
                mb=100.0,
                value=100.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=500,
                created_step=env.step_count,
            ))

            scheduler = IntegratedScheduler(device="cpu", enable_lyapunov=False, use_psf=False)
            safe_action, _, _, meta = scheduler._schedule_from_raw_action(
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                state,
                in_window=False,
                h=env.altitude_m,
                prop_can_update=True,
                available_power_w=env._last_available_power_w,
            )
        finally:
            DRL_CONFIG["enable_deployment_queue_projection"] = old_enabled
            DRL_CONFIG["queue_projection_policy"] = old_policy

        self.assertTrue(bool(meta["cpu_backpressure_applied"]))
        self.assertTrue(bool(meta["queue_boundary_projected"]))
        self.assertLess(float(safe_action[1]), 1.0)

    def test_environment_power_closure_preserves_tx_priority_in_window(self):
        """环境最终功率闭环也必须使用严格优先级，不能把 Tx 等比例打碎。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from config import ENERGY_CONFIG

        env = VLEOSatelliteEnv(seed=33)
        env.reset()
        env.altitude_m = env._h_warning + 50e3
        env._contact = {"in_window": True}
        action = np.array([0.5, 0.1, 0.8], dtype=np.float32)
        available = ENERGY_CONFIG["power_baseline_w"] + 30.0

        clipped, meta = env._enforce_available_power(action, available)

        self.assertEqual(meta["power_priority_order"], "tx>prop>cpu")
        self.assertAlmostEqual(float(clipped[2]), 0.8, places=6)
        self.assertAlmostEqual(float(clipped[0]), 0.0, places=6)
        self.assertAlmostEqual(
            float(clipped[1]),
            (30.0 - 0.8 * ENERGY_CONFIG["power_tx_max_w"])
            / ENERGY_CONFIG["power_cpu_max_w"],
            places=6,
        )
        self.assertTrue(bool(meta["propulsion_deadband_applied"]))

    def test_expired_processed_tasks_are_removed_before_downlink(self):
        """超过宽限期的 processed 数据应计入 expired_value，而不是被零价值下传移除。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from environment.task_value_model import TaskBatch
        from config import TASK_CONFIG

        env = VLEOSatelliteEnv(seed=14)
        env.reset()
        env._contact_override = {"in_window": True, "max_capacity_mbps": 8000.0}
        env._data_arrival_scale = 0.0
        env.step_count = int(TASK_CONFIG.get("overdue_grace_steps", 0)) + 2
        env.comm_queue.value = 10.0
        env.task_tracker.total_generated_value = 20.0
        env.task_tracker.total_generated_mb = 10.0
        env.task_tracker.processed_batches.append(TaskBatch(
            mb=10.0,
            value=20.0,
            priority=1.0,
            quality=1.0,
            deadline_steps=1,
            created_step=0,
        ))

        _, _, _, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))

        self.assertAlmostEqual(info["actual_tx_mb"], 0.0)
        self.assertAlmostEqual(info["delivered_value"], 0.0)
        self.assertGreaterEqual(info["expired_value"], 20.0)
        self.assertGreater(info["expired_value_rate"], 0.0)

    def test_slightly_overdue_processed_tasks_deliver_with_decay(self):
        """刚超 deadline 的 processed 数据在短宽限期内可折价交付，避免 reward 断崖。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from environment.task_value_model import TaskBatch

        env = VLEOSatelliteEnv(seed=24)
        env.reset()
        env._data_arrival_scale = 0.0
        env._contact_override = {"in_window": True, "max_capacity_mbps": 8000.0}
        env.step_count = 2
        env.comm_queue.value = 10.0
        env.task_tracker.total_generated_value = 20.0
        env.task_tracker.total_generated_mb = 10.0
        env.task_tracker.processed_batches.append(TaskBatch(
            mb=10.0,
            value=20.0,
            priority=1.0,
            quality=1.0,
            deadline_steps=2,
            created_step=0,
        ))

        _, _, _, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))

        self.assertGreater(info["actual_tx_mb"], 0.0)
        self.assertGreater(info["delivered_value"], 0.0)
        self.assertLess(info["delivered_value"], 20.0)
        self.assertAlmostEqual(info["expired_value"], 0.0)

    def test_observation_schema_matches_state_dim(self):
        from environment.satellite_env import VLEOSatelliteEnv, OBSERVATION_FEATURES
        from config import DRL_CONFIG

        env = VLEOSatelliteEnv(seed=10)
        obs = env.reset()
        self.assertEqual(len(OBSERVATION_FEATURES), int(DRL_CONFIG["state_dim"]))
        self.assertEqual(obs.shape[0], len(env.observation_features))
        self.assertIn("raw_queue_utilization", env.observation_features)
        self.assertIn("total_processed_value_norm", env.observation_features)
        self.assertIn("processed_queue_future_contact_ratio", env.observation_features)
        self.assertIn("future_contact_capacity_norm", env.observation_features)
        self.assertIn("next_window_in_range", env.observation_features)
        self.assertIn("cpu_backpressure_ratio", env.observation_features)
        self.assertIn("thermal_margin_norm", env.observation_features)
        self.assertIn("processed_high_next_window_deliverable_ratio", env.observation_features)
        self.assertIn("raw_high_next_window_deliverable_ratio", env.observation_features)
        self.assertIn("high_value_deadline_contact_mismatch", env.observation_features)

    def test_emergency_event_preserves_observation_schema(self):
        from environment.satellite_env import VLEOSatelliteEnv
        from config import DRL_CONFIG, TASK_CONFIG

        keys = {
            "emergency_event_enable",
            "emergency_event_probability_per_step",
            "emergency_event_duration_steps",
            "emergency_event_cooldown_steps",
        }
        old = {key: TASK_CONFIG.get(key) for key in keys}
        try:
            TASK_CONFIG.update({
                "emergency_event_enable": True,
                "emergency_event_probability_per_step": 1.0,
                "emergency_event_duration_steps": (18, 18),
                "emergency_event_cooldown_steps": 0,
            })
            env = VLEOSatelliteEnv(seed=34)
            env.reset()
            obs, _, _, info = env.step(
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                enforce_prop_smoothing=False,
            )
        finally:
            for key, value in old.items():
                if value is None:
                    TASK_CONFIG.pop(key, None)
                else:
                    TASK_CONFIG[key] = value

        self.assertEqual(obs.shape[0], len(env.observation_features))
        self.assertEqual(obs.shape[0], int(DRL_CONFIG["state_dim"]))
        self.assertEqual(info["scene_name"], "emergency_disaster")
        self.assertEqual(info["emergency_event_active"], 1.0)

    def test_heuristic_baseline_accepts_current_observation_schema(self):
        """启发式基线必须接受当前 43 维观测，避免旧阈值漏检。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from baselines.heuristic_baseline import HeuristicBaseline
        from config import DRL_CONFIG

        env = VLEOSatelliteEnv(seed=37)
        obs = env.reset()
        action = HeuristicBaseline().schedule(obs)

        self.assertEqual(action.shape, (int(DRL_CONFIG["action_dim"]),))
        self.assertTrue(np.all(np.isfinite(action)))
        self.assertTrue(np.all((action >= 0.0) & (action <= 1.0)))

    def test_scheduler_thermal_projection_limits_cpu_and_tx_when_hot(self):
        """热状态进入警告/过热区时，LS-PSF 边界层应在环境前降额 CPU/Tx。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from config import THERMAL_CONFIG
        from scheduler.integrated_scheduler import IntegratedScheduler

        env = VLEOSatelliteEnv(seed=36)
        env.reset()
        env.thermal_temperature_c = float(THERMAL_CONFIG["critical_temp_c"]) + 1.0

        state = env._get_observation()
        scheduler = IntegratedScheduler(device="cpu", enable_lyapunov=False, use_psf=False)
        safe_action, _, _, meta = scheduler._schedule_from_raw_action(
            np.array([0.0, 1.0, 1.0], dtype=np.float32),
            state,
            env.energy_queue.value,
            env.orbit_queue.value,
            env.data_queue.length,
            env.comm_queue.value,
            in_window=False,
            h=env.altitude_m,
            soc=env.battery.soc,
            time_s=env.time_s,
            prop_can_update=True,
            tx_capacity_mbps=0.0,
            available_power_w=env._last_available_power_w,
        )
        _, _, _, info = env.step(safe_action, enforce_prop_smoothing=False)

        self.assertTrue(bool(meta["thermal_clipped"]))
        self.assertLessEqual(float(safe_action[1]), float(meta["thermal_cpu_cap"]) + 1e-6)
        self.assertLessEqual(float(safe_action[2]), float(meta["thermal_tx_cap"]) + 1e-6)
        self.assertFalse(bool(info["thermal_throttle_applied"]))
        self.assertFalse(bool(info["thermal_safe"]))

    def test_thermal_state_uses_solar_and_radiative_balance(self):
        """热模型应对日照和辐射散热有方向一致的响应。"""
        from environment.satellite_env import VLEOSatelliteEnv

        env_sunlit = VLEOSatelliteEnv(seed=38)
        env_sunlit.reset()
        env_sunlit.thermal_temperature_c = 20.0
        hot = env_sunlit._update_thermal_state(total_power_w=60.0, sunlit_fraction=1.0)

        env_shade = VLEOSatelliteEnv(seed=39)
        env_shade.reset()
        env_shade.thermal_temperature_c = 20.0
        shade = env_shade._update_thermal_state(total_power_w=60.0, sunlit_fraction=0.0)

        self.assertGreater(float(hot["temperature_c"]), float(shade["temperature_c"]))

    def test_observation_includes_processed_backlog_value(self):
        from environment.satellite_env import VLEOSatelliteEnv
        from environment.task_value_model import TaskBatch
        from config import TASK_CONFIG

        env = VLEOSatelliteEnv(seed=25)
        env.reset()
        value_norm = float(TASK_CONFIG["value_norm"])
        env.task_tracker.processed_batches.append(TaskBatch(
            mb=100.0,
            value=0.5 * value_norm,
            priority=1.0,
            quality=1.0,
            deadline_steps=500,
            created_step=env.step_count,
        ))

        obs = env._get_observation()
        idx = env.observation_features.index("total_processed_value_norm")
        self.assertAlmostEqual(float(obs[idx]), 0.5, places=6)

    def test_observation_includes_scene_semantics(self):
        from environment.satellite_env import VLEOSatelliteEnv

        env = VLEOSatelliteEnv(seed=26)
        env.reset()
        env._scene_phase_offset_fraction = 0.0
        env.orbit_sim.reset_phase(0.20 * 2.0 * np.pi)
        obs = env._get_observation()
        scene_idx = env.observation_features.index("current_scene_class_norm")
        upcoming_idx = env.observation_features.index("upcoming_task_intensity_norm")

        self.assertAlmostEqual(float(obs[scene_idx]), 1.0, places=6)
        self.assertGreaterEqual(float(obs[upcoming_idx]), 0.0)
        self.assertLessEqual(float(obs[upcoming_idx]), 1.0)

    def test_observation_distinguishes_unknown_next_window(self):
        from environment.satellite_env import VLEOSatelliteEnv

        env = VLEOSatelliteEnv(seed=29)
        env.reset()
        env._contact = {
            "in_window": False,
            "time_to_next_window_s": 5400.0,
            "window_remaining_s": 0.0,
            "max_capacity_mbps": 0.0,
        }

        obs = env._get_observation()
        known_idx = env.observation_features.index("next_window_in_range")
        time_idx = env.observation_features.index("time_to_next_window_norm")

        self.assertAlmostEqual(float(obs[time_idx]), 1.0)
        self.assertAlmostEqual(float(obs[known_idx]), 0.0)

        env._contact["time_to_next_window_s"] = 600.0
        obs = env._get_observation()
        self.assertLess(float(obs[time_idx]), 0.2)

    def test_observation_includes_future_contact_capacity(self):
        from environment.satellite_env import VLEOSatelliteEnv

        env = VLEOSatelliteEnv(seed=30)
        obs = env.reset()
        idx = env.observation_features.index("future_contact_capacity_norm")

        self.assertGreaterEqual(float(obs[idx]), 0.0)
        self.assertLessEqual(float(obs[idx]), 2.0)

    def test_observation_includes_deadline_contact_deliverability(self):
        from environment.satellite_env import VLEOSatelliteEnv
        from environment.task_value_model import TaskBatch

        env = VLEOSatelliteEnv(seed=31)
        env.reset()
        env._contact = {
            "in_window": False,
            "time_to_next_window_s": 5400.0,
            "window_remaining_s": 0.0,
            "max_capacity_mbps": 0.0,
        }
        env.task_tracker.raw_batches = [
            TaskBatch(
                mb=10.0,
                value=200.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=700,
                created_step=0,
            ),
            TaskBatch(
                mb=10.0,
                value=200.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=120,
                created_step=-100,
            ),
        ]
        env.task_tracker.processed_batches = [
            TaskBatch(
                mb=10.0,
                value=200.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=700,
                created_step=-20,
            ),
            TaskBatch(
                mb=10.0,
                value=200.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=120,
                created_step=-100,
            ),
        ]

        obs = env._get_observation()
        proc_idx = env.observation_features.index(
            "processed_high_next_window_deliverable_ratio")
        raw_idx = env.observation_features.index(
            "raw_high_next_window_deliverable_ratio")
        mismatch_idx = env.observation_features.index(
            "high_value_deadline_contact_mismatch")

        self.assertGreater(float(obs[proc_idx]), 0.0)
        self.assertLess(float(obs[proc_idx]), 1.0)
        self.assertGreater(float(obs[raw_idx]), 0.0)
        self.assertLess(float(obs[raw_idx]), 1.0)
        self.assertGreater(float(obs[mismatch_idx]), 0.0)
        self.assertLess(float(obs[mismatch_idx]), 1.0)

    def test_scene_profile_changes_arrival_rate(self):
        from environment.satellite_env import VLEOSatelliteEnv

        env = VLEOSatelliteEnv(seed=27)
        env.reset()
        env._scene_phase_offset_fraction = 0.0
        military_scene = env._scene_context_for_phase(phase=0.20 * 2.0 * np.pi)
        cloud_scene = env._scene_context_for_phase(phase=0.90 * 2.0 * np.pi)

        self.assertGreater(
            env._arrival_rate_for_scene(military_scene),
            env._arrival_rate_for_scene(cloud_scene),
        )

    def test_upcoming_scene_intensity_uses_value_and_deadline(self):
        from environment.satellite_env import VLEOSatelliteEnv

        env = VLEOSatelliteEnv(seed=28)
        env.reset()
        env._scene_phase_offset_fraction = 0.0
        military_scene = env._scene_context_for_phase(phase=0.20 * 2.0 * np.pi)
        ocean_scene = env._scene_context_for_phase(phase=0.02 * 2.0 * np.pi)

        self.assertGreater(
            env._normalized_scene_intensity(military_scene),
            env._normalized_scene_intensity(ocean_scene),
        )
        self.assertLessEqual(env._normalized_scene_intensity(military_scene), 1.0)

    def test_scheduler_metadata_records_observation_schema(self):
        from scheduler.integrated_scheduler import IntegratedScheduler
        from config import DRL_CONFIG, OBJECTIVE_VERSION

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "schema.pt")
            scheduler = IntegratedScheduler(device="cpu", enable_lyapunov=True, use_psf=False)
            scheduler.save(ckpt)

            restored = IntegratedScheduler(device="cpu", enable_lyapunov=False, use_psf=False)
            metadata = restored.load(ckpt)

        self.assertEqual(metadata.get("objective_version"), OBJECTIVE_VERSION)
        self.assertEqual(metadata.get("state_dim"), DRL_CONFIG["state_dim"])
        self.assertIn("current_scene_class_norm", metadata.get("observation_features", []))
        self.assertIn("upcoming_task_intensity_norm", metadata.get("observation_features", []))
        self.assertIn("future_contact_capacity_norm", metadata.get("observation_features", []))
        self.assertIn("next_window_in_range", metadata.get("observation_features", []))

    def test_scheduler_boundary_layer_enforces_power_budget(self):
        from scheduler.integrated_scheduler import IntegratedScheduler
        from config import ENERGY_CONFIG

        scheduler = IntegratedScheduler(device="cpu", enable_lyapunov=False, use_psf=False)
        raw = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        available = ENERGY_CONFIG["power_baseline_w"] + 10.0
        safe, meta = scheduler._clip_action_boundaries(raw, available_power_w=available)
        adjustable = (
            safe[0] * ENERGY_CONFIG["power_propulsion_max_w"]
            + safe[1] * ENERGY_CONFIG["power_cpu_max_w"]
            + safe[2] * ENERGY_CONFIG["power_tx_max_w"]
        )
        self.assertTrue(meta["boundary_clipped"])
        self.assertTrue(meta["power_clipped"])
        self.assertLessEqual(adjustable, 10.0 + 1e-6)

    def test_scheduler_boundary_uses_strict_priority_in_window(self):
        from scheduler.integrated_scheduler import IntegratedScheduler
        from config import ENERGY_CONFIG

        scheduler = IntegratedScheduler(device="cpu", enable_lyapunov=False, use_psf=False)
        raw = np.array([0.5, 0.1, 0.8], dtype=np.float32)
        available = ENERGY_CONFIG["power_baseline_w"] + 30.0
        safe, meta = scheduler._clip_action_boundaries(
            raw, available_power_w=available, in_window=True)

        self.assertEqual(meta["power_priority_order"], "tx>prop>cpu")
        self.assertAlmostEqual(float(safe[2]), 0.8, places=6)
        self.assertAlmostEqual(float(safe[0]), 0.0, places=6)
        self.assertAlmostEqual(
            float(safe[1]),
            (30.0 - 0.8 * ENERGY_CONFIG["power_tx_max_w"])
            / ENERGY_CONFIG["power_cpu_max_w"],
            places=6,
        )
        self.assertTrue(bool(meta["propulsion_deadband_applied"]))

    def test_scheduler_boundary_uses_strict_priority_outside_window(self):
        from scheduler.integrated_scheduler import IntegratedScheduler
        from config import ENERGY_CONFIG

        scheduler = IntegratedScheduler(device="cpu", enable_lyapunov=False, use_psf=False)
        raw = np.array([0.5, 0.1, 0.8], dtype=np.float32)
        available = ENERGY_CONFIG["power_baseline_w"] + 30.0
        safe, meta = scheduler._clip_action_boundaries(
            raw, available_power_w=available, in_window=False)

        self.assertEqual(meta["power_priority_order"], "prop>cpu>tx")
        self.assertAlmostEqual(
            float(safe[0]),
            30.0 / ENERGY_CONFIG["power_propulsion_max_w"],
            places=6,
        )
        self.assertAlmostEqual(float(safe[1]), 0.0, places=6)
        self.assertAlmostEqual(float(safe[2]), 0.0, places=6)

    def test_scheduler_boundary_layer_sanitizes_nonfinite_actions(self):
        from scheduler.integrated_scheduler import IntegratedScheduler

        scheduler = IntegratedScheduler(device="cpu", enable_lyapunov=False, use_psf=False)
        safe, meta = scheduler._clip_action_boundaries(
            np.array([np.nan, np.inf, -np.inf], dtype=np.float32),
            available_power_w=np.nan,
        )

        self.assertTrue(meta["boundary_clipped"])
        self.assertFalse(meta["raw_action_finite"])
        self.assertTrue(np.all(np.isfinite(safe)))
        self.assertTrue(np.all((safe >= 0.0) & (safe <= 1.0)))

    def test_baseline_downlink_prediction_respects_rf_rate(self):
        """传统基线内部打分也要受发射机 RF 速率上限约束。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from baselines.dpp_baseline import DriftPlusPenaltyBaseline
        from baselines.mpc_baseline import MPCBaseline
        from config import QUEUE_CONFIG

        env = VLEOSatelliteEnv(seed=19)
        env.reset()
        env._contact_override = {"in_window": True, "max_capacity_mbps": 8000.0}
        env._contact = env._get_contact_info()

        dpp = DriftPlusPenaltyBaseline()
        p_tx_w = 0.1 * env.power_sys.P_tx_max
        dpp_tx = dpp._predict_downlink_mb(env, 0.1, available_mb=1e9, p_tx_w=p_tx_w)
        rf_cap = env.power_sys.tx_downlink_rate(p_tx_w) * env.dt
        self.assertLessEqual(dpp_tx, rf_cap + 1e-6)

        mpc = MPCBaseline()
        self.assertAlmostEqual(
            mpc.rf_rate_max_mbs,
            QUEUE_CONFIG["tx_downlink_rate_max_mbs"],
        )
        self.assertAlmostEqual(
            mpc.cpu_rate_max_mbs,
            QUEUE_CONFIG["data_service_rate_max_mbs"],
        )
        state = env._get_observation()
        value_low_power = mpc._value_score(state, P_cpu=0.0, P_tx=p_tx_w)
        value_high_power = mpc._value_score(state, P_cpu=0.0, P_tx=env.power_sys.P_tx_max)
        self.assertLess(value_low_power, value_high_power)

    def test_mpc_baseline_syncs_density_from_environment(self):
        """MPC/Robust MPC 在鲁棒测试中必须使用当前环境大气参数。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from baselines.mpc_baseline import MPCBaseline
        from baselines.robust_mpc_baseline import RobustMPCBaseline

        env = VLEOSatelliteEnv(seed=21)
        env.reset()
        altitude_m = 230e3

        nominal = MPCBaseline()
        nominal_next = nominal._predict_altitude(altitude_m, P_prop=0.0)

        env.orbit_dyn.atm.rho_ref *= 5.0
        mpc = MPCBaseline()
        mpc.schedule(
            env._get_observation(),
            env.battery.soc,
            altitude_m,
            env.orbit_sim.is_sunlit(env.time_s),
            env.solar.output_power(env.orbit_sim.sunlit_fraction()),
            time_s=env.time_s,
            env=env,
        )
        synced_next = mpc._predict_altitude(altitude_m, P_prop=0.0)

        robust_mpc = RobustMPCBaseline()
        robust_mpc.sync_from_env(env)

        self.assertAlmostEqual(mpc.rho_ref, env.orbit_dyn.atm.rho_ref)
        self.assertAlmostEqual(robust_mpc.rho_ref, env.orbit_dyn.atm.rho_ref)
        self.assertLess(synced_next, nominal_next)

    def test_mpc_altitude_prediction_clips_to_crash_boundary_not_safe_boundary(self):
        """MPC 预测不能把 150km 不安全线误当成物理高度下限。"""
        from baselines.mpc_baseline import MPCBaseline
        from baselines.robust_mpc_baseline import RobustMPCBaseline
        from config import ORBITAL_CONFIG

        h_crash = float(ORBITAL_CONFIG["altitude_crash_km"]) * 1e3
        mpc = MPCBaseline()
        robust_mpc = RobustMPCBaseline()

        self.assertAlmostEqual(mpc._predict_altitude(100e3, P_prop=0.0), h_crash)
        self.assertAlmostEqual(
            robust_mpc._predict_altitude_with_density(100e3, P_prop=0.0, density_scale=1.0),
            h_crash,
        )

    def test_mpc_value_score_reads_observation_features_by_name(self):
        from baselines.mpc_baseline import MPCBaseline
        from environment.satellite_env import OBSERVATION_FEATURES
        from config import DRL_CONFIG

        idx = {name: i for i, name in enumerate(OBSERVATION_FEATURES)}
        state = np.zeros((int(DRL_CONFIG["state_dim"]),), dtype=np.float32)
        state[idx["in_comm_window"]] = 1.0
        state[idx["tx_capacity_norm"]] = 1.0
        state[idx["total_processed_value_norm"]] = 0.5
        state[idx["topk_quality_norm"]] = 1.0
        state[idx["deadline_urgency"]] = 0.5
        state[idx["current_scene_class_norm"]] = 0.5
        state[idx["raw_low_queue_utilization"]] = 1.0

        mpc = MPCBaseline()
        state[idx["topk_priority_norm"]] = 0.2
        low_score = mpc._value_score(state, P_cpu=0.0, P_tx=mpc.P_tx_max)
        state[idx["topk_priority_norm"]] = 1.0
        high_score = mpc._value_score(state, P_cpu=0.0, P_tx=mpc.P_tx_max)

        self.assertGreater(high_score, low_score)

    def test_oracle_mpc_is_marked_as_non_deployable_upper_bound(self):
        """Oracle MPC 只能作为上帝视角上界，不能伪装成在线可部署 baseline。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from baselines.oracle_mpc_baseline import OracleMPCBaseline

        env = VLEOSatelliteEnv(seed=35)
        state = env.reset()
        oracle = OracleMPCBaseline(horizon=2, beam_width=2)
        action = oracle.schedule(state, env)

        self.assertTrue(np.all(np.isfinite(action)))
        self.assertTrue(np.all((action >= 0.0) & (action <= 1.0)))
        self.assertTrue(oracle.metadata["uses_future_environment_rollout"])
        self.assertFalse(oracle.metadata["deployable_online_policy"])

    def test_lyapunov_layer_does_not_use_queue_pressure_to_change_cpu_tx(self):
        """无窗口且 processed 已满时，调度器应保守压低 CPU，避免继续制造溢出。"""
        from scheduler.integrated_scheduler import LyapunovConstraintLayer
        from config import QUEUE_CONFIG

        layer = LyapunovConstraintLayer()
        high_raw = QUEUE_CONFIG.get("data_queue_max_mb", 500.0)
        high_comm = QUEUE_CONFIG.get("comm_queue_max", 500.0)

        action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        safe_high_raw, _ = layer.project(
            action, Q_E=0.0, Q_H=0.0, Q_D=high_raw, Q_C=high_comm,
            in_window=False,
        )
        safe_low_raw, _ = layer.project(
            action, Q_E=0.0, Q_H=0.0, Q_D=0.0, Q_C=high_comm,
            in_window=False,
        )

        np.testing.assert_allclose(safe_high_raw, action, atol=1e-6)
        np.testing.assert_allclose(safe_low_raw, action, atol=1e-6)

    def test_lyapunov_layer_triggers_in_moderate_risk_band(self):
        from scheduler.integrated_scheduler import LyapunovConstraintLayer
        from config import QUEUE_CONFIG

        layer = LyapunovConstraintLayer()
        orbit_threshold = float(QUEUE_CONFIG.get("orbit_queue_max", 500.0)) * 0.4
        action = np.array([0.0, 0.6, 0.2], dtype=np.float32)

        safe, projected = layer.project(
            action,
            Q_E=0.0,
            Q_H=orbit_threshold,
            Q_D=0.0,
            Q_C=0.0,
            in_window=False,
        )

        self.assertTrue(projected)
        self.assertGreater(float(safe[0]), float(action[0]))

    def test_time_limit_truncation_keeps_bootstrap_in_replay(self):
        """时间截断不是物理终止，ReplayBuffer 里应存 terminated=False。"""
        from drl.agent import SACAgent
        from config import DRL_CONFIG

        state_dim = int(DRL_CONFIG.get("state_dim", 30))
        agent = SACAgent(state_dim=state_dim, action_dim=3, device="cpu")
        state = np.zeros((DRL_CONFIG.get("frame_stack", 8), state_dim), dtype=np.float32)
        action = np.zeros(3, dtype=np.float32)

        agent.store(state, action, 0.0, state, d=True, lya=0.0, terminated=False)

        self.assertEqual(float(agent.buffer.dones[0, 0]), 0.0)

    def test_sac_agent_default_action_dim_uses_config_everywhere(self):
        from drl.agent import SACAgent
        from config import DRL_CONFIG

        state_dim = int(DRL_CONFIG.get("state_dim", 40))
        action_dim = int(DRL_CONFIG.get("action_dim", 10))
        agent = SACAgent(state_dim=state_dim, device="cpu")

        self.assertEqual(agent.action_dim, action_dim)
        self.assertEqual(agent.buffer.action_dim, action_dim)
        self.assertAlmostEqual(
            float(agent.target_entropy),
            -action_dim * float(DRL_CONFIG.get("target_entropy_scale", 1.0)),
        )

    def test_replay_buffer_stores_behavior_action_separately(self):
        """安全动作模仿目标应单独存放，避免把安全惩罚污染 Critic reward。"""
        from drl.agent import ReplayBuffer
        from config import DRL_CONFIG

        state_dim = int(DRL_CONFIG.get("state_dim", 30))
        frame_stack = int(DRL_CONFIG.get("frame_stack", 8))
        buf = ReplayBuffer(capacity=1, state_dim=state_dim, action_dim=3)
        state = np.zeros((frame_stack, state_dim), dtype=np.float32)
        raw_action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        safe_action = np.array([0.2, 0.3, 0.4], dtype=np.float32)

        buf.push(
            state, raw_action, 1.25, state, False, 0.0,
            behavior_action=safe_action,
            behavior_weight=0.7,
        )
        _, action, reward, _, _, _, behavior_action, behavior_weight = buf.sample(1)

        self.assertTrue(np.allclose(action[0], raw_action))
        self.assertAlmostEqual(float(reward[0, 0]), 1.25)
        self.assertTrue(np.allclose(behavior_action[0], safe_action))
        self.assertAlmostEqual(float(behavior_weight[0, 0]), 0.7)


class TestPSFQueueGuard(unittest.TestCase):

    def setUp(self):
        from safety.predictive_safety_filter import PredictiveSafetyFilter
        self.psf = PredictiveSafetyFilter(K=6, robust_check=False)

    def test_physical_trigger_thresholds_keep_recovery_margin(self):
        """PSF 触发边界应比 153km/16.5% 留出更早恢复余量。"""
        from config import ENERGY_CONFIG, ORBITAL_CONFIG, PSF_CONFIG

        h_min = float(ORBITAL_CONFIG["altitude_min_km"]) * 1e3
        soc_min = float(ENERGY_CONFIG["battery_min_soc"])

        self.assertAlmostEqual(
            self.psf.predictor.h_crit,
            h_min + float(PSF_CONFIG["altitude_trigger_margin_m"]),
        )
        self.assertAlmostEqual(
            self.psf.predictor.soc_crit,
            soc_min + float(PSF_CONFIG["soc_trigger_margin"]),
        )
        self.assertGreater(self.psf.predictor.h_crit, h_min + 3e3)
        self.assertGreater(self.psf.predictor.soc_crit, soc_min + 0.015)

    def test_data_queue_pressure_does_not_change_action(self):
        """仅队列高压但物理安全时，PSF 不应修改动作。"""
        from config import QUEUE_CONFIG

        high_raw = float(QUEUE_CONFIG.get("data_queue_max_mb", 500.0)) * 0.99
        high_comm = float(QUEUE_CONFIG.get("comm_queue_max", 500.0)) * 0.25
        raw = np.array([0.3, 0.05, 0.1], dtype=np.float32)
        safe, meta = self.psf.filter(
            raw,
            h=360e3,
            soc=0.6,
            time_s=0.0,
            Q_E=5.0,
            Q_H=5.0,
            Q_D=high_raw,
            Q_C=high_comm,
            in_window=False,
            tx_capacity_mbps=0.0,
            orbital_phase=0.0,
        )
        self.assertTrue(np.allclose(safe, raw, atol=1e-8))
        self.assertFalse(meta.get("psf_triggered", True))
        self.assertNotIn("psf_queue_checked", meta)
        self.assertNotIn("psf_queue_safe", meta)
        self.assertIn("psf_formal_guarantee", meta)
        self.assertFalse(bool(meta["psf_formal_guarantee"]))
        self.assertIn("psf_candidate_count", meta)
        self.assertEqual(int(meta["psf_candidate_count"]), 0)

    def test_comm_queue_pressure_does_not_change_action(self):
        """窗口内通信队列高压但物理安全时，PSF 不应修改动作。"""
        from config import QUEUE_CONFIG

        moderate_raw = float(QUEUE_CONFIG.get("data_queue_max_mb", 500.0)) * 0.25
        high_comm = float(QUEUE_CONFIG.get("comm_queue_max", 500.0)) * 0.99
        raw = np.array([0.25, 0.2, 0.05], dtype=np.float32)
        safe, meta = self.psf.filter(
            raw,
            h=360e3,
            soc=0.6,
            time_s=0.0,
            Q_E=5.0,
            Q_H=5.0,
            Q_D=moderate_raw,
            Q_C=high_comm,
            in_window=True,
            tx_capacity_mbps=80.0,
            orbital_phase=0.0,
        )
        self.assertTrue(np.allclose(safe, raw, atol=1e-8))
        self.assertFalse(meta.get("psf_triggered", True))
        self.assertNotIn("psf_queue_checked", meta)
        self.assertNotIn("psf_queue_safe", meta)
        self.assertNotIn("psf_queue_correction", meta)

    def test_psf_stats_report_post_hoc_candidate_search(self):
        stats = self.psf.get_stats()

        self.assertEqual(stats["psf_search_method"], "finite_candidate_rollout")
        self.assertEqual(float(stats["psf_formal_guarantee"]), 0.0)
        self.assertNotIn("psf_queue_checks_enabled", stats)
        self.assertIn("psf_no_safe_candidate_count", stats)
        self.assertGreaterEqual(float(stats["psf_horizon_seconds"]), 0.0)


class TestTraceGroundStationConfig(unittest.TestCase):

    def test_ground_station_amc_uses_discrete_mcs_levels(self):
        """链路容量应按离散 AMC 档位变化，而不是连续 Shannon 曲线。"""
        from environment.ground_station import GroundStation

        self.assertAlmostEqual(GroundStation._amc_spectral_efficiency(-4.0), 0.0)
        self.assertAlmostEqual(GroundStation._amc_spectral_efficiency(-3.0), 0.25)
        self.assertAlmostEqual(GroundStation._amc_spectral_efficiency(2.9), 0.25)
        self.assertAlmostEqual(GroundStation._amc_spectral_efficiency(3.0), 0.5)
        self.assertAlmostEqual(GroundStation._amc_spectral_efficiency(20.0), 3.0)
        self.assertAlmostEqual(GroundStation._amc_capacity_mbps(-4.0), 0.0)
        self.assertAlmostEqual(GroundStation._amc_capacity_mbps(-3.0), 10.0)
        self.assertAlmostEqual(GroundStation._amc_capacity_mbps(8.0), 120.0)
        self.assertAlmostEqual(GroundStation._amc_capacity_mbps(20.0), 220.0)

    def test_ground_station_channel_capacity_has_explicit_cap(self):
        """即使关闭 AMC 使用 Shannon 曲线，高仰角容量也应有工程封顶。"""
        from environment.ground_station import GroundStation
        from config import GROUND_STATION_CONFIG

        old_amc = GROUND_STATION_CONFIG.get("amc_enabled", True)
        old_cap = GROUND_STATION_CONFIG.get("max_channel_capacity_mbps", 0.0)
        try:
            GROUND_STATION_CONFIG["amc_enabled"] = False
            GROUND_STATION_CONFIG["max_channel_capacity_mbps"] = 100.0
            station = GroundStation(0.0, 0.0, tx_power_dbw=80.0)
            cap_mbps = station.channel_capacity_mbps(np.radians(90.0), 350e3)
        finally:
            GROUND_STATION_CONFIG["amc_enabled"] = old_amc
            GROUND_STATION_CONFIG["max_channel_capacity_mbps"] = old_cap

        self.assertLessEqual(cap_mbps, 100.0 + 1e-9)

    def test_ground_station_refraction_and_j2_helpers_are_active(self):
        """低仰角折射与 J2 漂移都应进入地面站模型。"""
        from environment.ground_station import GroundStation, GroundStationNetwork

        station = GroundStation(0.0, 0.0, atmospheric_refraction_enabled=True)
        corrected = station._apply_atmospheric_refraction(np.radians(5.0))
        drift_rate = GroundStationNetwork._raan_drift_rate_rad_s(350e3, 51.6)

        self.assertGreaterEqual(corrected, np.radians(5.0))
        self.assertNotEqual(drift_rate, 0.0)

    def test_default_trace_station_network_is_global(self):
        """真实 trace 默认站网应覆盖中高纬和南半球，避免窗口过度稀疏。"""
        from experiments.generate_trace_csv import DEFAULT_GROUND_STATIONS

        names = {station["name"] for station in DEFAULT_GROUND_STATIONS}
        lats = [float(station["lat_deg"]) for station in DEFAULT_GROUND_STATIONS]

        self.assertGreaterEqual(len(DEFAULT_GROUND_STATIONS), 12)
        self.assertIn("GS-Beijing", names)
        self.assertIn("GS-Svalbard", names)
        self.assertGreater(max(lats), 60.0)
        self.assertLess(min(lats), -50.0)

    def test_runtime_ground_station_config_has_regional_and_global_profiles(self):
        """训练默认使用全球站网，同时保留区域站网配置用于对照实验。"""
        from config import GROUND_STATION_CONFIG

        self.assertEqual(GROUND_STATION_CONFIG["profile"], "global")
        profiles = GROUND_STATION_CONFIG["profiles"]
        self.assertIn("regional", profiles)
        self.assertIn("global", profiles)

        regional = profiles["regional"]
        global_stations = profiles["global"]
        global_lats = [float(station["lat"]) for station in global_stations]

        self.assertEqual(GROUND_STATION_CONFIG["stations"], global_stations)
        self.assertGreaterEqual(len(regional), 3)
        self.assertLess(len(regional), len(global_stations))
        self.assertGreaterEqual(len(global_stations), 12)
        self.assertGreater(max(global_lats), 60.0)
        self.assertLess(min(global_lats), -50.0)
        self.assertEqual(float(GROUND_STATION_CONFIG["min_elevation_deg"]), 5.0)

    def test_global_ground_station_csv_loads(self):
        """CSV 站网可被 trace 生成器直接读取。"""
        from experiments.generate_trace_csv import _load_ground_station_csv

        csv_path = os.path.join(_ROOT_DIR, "data", "ground_stations_global.csv")
        stations = _load_ground_station_csv(csv_path, default_min_elevation_deg=5.0)

        self.assertGreaterEqual(len(stations), 12)
        self.assertTrue(all("min_elevation_deg" in station for station in stations))
        self.assertTrue(any(station["lat_deg"] > 60.0 for station in stations))
        self.assertTrue(any(station["lat_deg"] < -50.0 for station in stations))


class TestTraceRobustnessSemantics(unittest.TestCase):

    def test_force_trace_altitude_does_not_reset_orbit_queue_mid_episode(self):
        """逐步回放真实高度时，不能清空轨道虚拟队列历史。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from experiments.robustness import _apply_trace_item

        env = VLEOSatelliteEnv(seed=11)
        env.reset()
        env.step_count = 1
        env.orbit_queue.value = 12.5

        _apply_trace_item(
            env,
            {"altitude_km": 230.0},
            base_rho_ref=env.orbit_dyn.atm.rho_ref,
            base_solar_eta=env.solar.eta,
            trace_altitude_mode="force",
        )

        self.assertAlmostEqual(env.altitude_m / 1e3, 230.0)
        self.assertAlmostEqual(env.orbit_queue.value, 12.5)

    def test_force_trace_altitude_can_reach_crash_boundary(self):
        """真实 trace 低于 122km 时应落到再入终止线，而不是被 150km 吃掉。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from experiments.robustness import _apply_trace_item

        env = VLEOSatelliteEnv(seed=12)
        env.reset()
        env.step_count = 1
        env.orbit_queue.value = 9.0

        _apply_trace_item(
            env,
            {"altitude_km": 121.0},
            base_rho_ref=env.orbit_dyn.atm.rho_ref,
            base_solar_eta=env.solar.eta,
            trace_altitude_mode="force",
        )

        self.assertAlmostEqual(env.altitude_m, env._h_crash)
        self.assertAlmostEqual(env.orbit_queue.value, 9.0)

    def test_trace_loader_keeps_altitudes_below_150km(self):
        """CSV 回放解析不能把 122~150km 的不安全状态提前裁剪成安全边界。"""
        import os
        import tempfile
        from experiments.robustness import _load_trace_rows
        from config import ORBITAL_CONFIG

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("altitude_km\n130\n121\n")
            path = f.name
        try:
            rows = _load_trace_rows(path)
        finally:
            os.unlink(path)

        self.assertAlmostEqual(rows[0]["altitude_km"], 130.0)
        self.assertAlmostEqual(
            rows[1]["altitude_km"],
            float(ORBITAL_CONFIG["altitude_crash_km"]),
        )


class TestEvaluationReportMath(unittest.TestCase):

    def test_relative_improvement_direction(self):
        """模型优于 baseline 时，报告中的提升比例应为正。"""
        from evaluate_optimized import _relative_improvement

        self.assertAlmostEqual(_relative_improvement(120.0, 100.0), 20.0)
        self.assertAlmostEqual(
            _relative_improvement(5.0, 10.0, lower_is_better=True),
            50.0,
        )

    def test_paper_ablation_variants_are_single_axis(self):
        """论文消融应围绕模块归因，而不是旧的 reward/TD 补丁变体。"""
        from experiments.ablation import (
            VARIANT_SPECS, DISPLAY_ORDER, LEARNING_BASELINE_SPECS,
        )

        self.assertEqual(
            [VARIANT_SPECS[key]["code"] for key in DISPLAY_ORDER],
            ["A", "B", "C", "D", "E", "F", "G", "H"],
        )
        forbidden = {"reward_clipping", "contaminated_td"}
        self.assertFalse({
            spec["constraint_variant"] for spec in VARIANT_SPECS.values()
        } & forbidden)
        self.assertEqual(
            VARIANT_SPECS["B_Throughput_Objective"]["mission_reward_variant"],
            "throughput",
        )
        self.assertEqual(
            VARIANT_SPECS["C_No_CMDP"]["constraint_variant"],
            "plain_sac",
        )
        self.assertEqual(
            VARIANT_SPECS["D_No_Adaptive_Dual"]["adaptive_dual_enable"],
            False,
        )
        self.assertEqual(
            VARIANT_SPECS["G_MLP_Backbone"]["network_arch"],
            "mlp",
        )
        self.assertEqual(
            VARIANT_SPECS["H_No_BC"]["behavior_cloning_coeff"],
            0.0,
        )
        self.assertEqual(
            LEARNING_BASELINE_SPECS["SAC_PSF"]["constraint_variant"],
            "sac_psf",
        )
        self.assertEqual(
            LEARNING_BASELINE_SPECS["SAC_Lyapunov"]["constraint_variant"],
            "sac_lyapunov",
        )

    def test_formal_ablation_requires_all_independent_checkpoints(self):
        from types import SimpleNamespace
        from experiments.ablation import missing_independent_checkpoints, DISPLAY_ORDER

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(ablation_dir=tmpdir)
            missing = missing_independent_checkpoints(args)

        self.assertEqual(set(missing.keys()), set(DISPLAY_ORDER))

    def test_throughput_ablation_disables_value_auxiliary_head(self):
        from config import DRL_CONFIG
        from experiments.ablation import _temporary_reward_config

        old_enabled = bool(DRL_CONFIG.get("value_aux_head_enable", False))
        with _temporary_reward_config("B_Throughput_Objective"):
            self.assertFalse(bool(DRL_CONFIG.get("value_aux_head_enable", True)))
            self.assertEqual(float(DRL_CONFIG.get("value_aux_loss_weight", 1.0)), 0.0)
            self.assertEqual(float(DRL_CONFIG.get("value_aux_loss_weight_final", 1.0)), 0.0)
        self.assertEqual(bool(DRL_CONFIG.get("value_aux_head_enable", False)), old_enabled)

    def test_value_aux_pseudo_label_thresholds_are_configured(self):
        from algorithms.decoupled_constraint_sac import DecoupledConstraintSAC
        from config import DRL_CONFIG

        state_dim = int(DRL_CONFIG.get("state_dim", 40))
        action_dim = int(DRL_CONFIG.get("action_dim", 10))
        agent = DecoupledConstraintSAC(
            state_dim=state_dim,
            action_dim=action_dim,
            device="cpu",
        )

        self.assertAlmostEqual(
            agent.value_aux_high_pressure_margin,
            1.10,
        )
        self.assertAlmostEqual(
            agent.value_aux_low_pressure_margin,
            1.25,
        )
        self.assertAlmostEqual(
            agent.value_aux_expiring_high_threshold,
            0.10,
        )
        self.assertAlmostEqual(
            agent.value_aux_processed_future_contact_threshold,
            0.75,
        )

    def test_value_aux_feature_indices_follow_observation_schema(self):
        from algorithms.decoupled_constraint_sac import DecoupledConstraintSAC
        from environment.satellite_env import OBSERVATION_FEATURES

        self.assertFalse(any(name.startswith("_IDX_") for name in dir(DecoupledConstraintSAC)))
        for feature_name in [
            "in_comm_window",
            "raw_high_queue_utilization",
            "raw_mid_queue_utilization",
            "raw_low_queue_utilization",
            "processed_high_queue_utilization",
            "processed_mid_queue_utilization",
            "processed_low_queue_utilization",
            "expiring_high_value_norm",
            "expiring_low_value_norm",
            "processed_queue_future_contact_ratio",
            "future_contact_capacity_norm",
        ]:
            self.assertEqual(
                DecoupledConstraintSAC._FEATURE_INDEX[feature_name],
                OBSERVATION_FEATURES.index(feature_name),
            )

    def test_constraint_actor_loss_can_be_disabled_for_plain_variants(self):
        import torch
        from algorithms.decoupled_constraint_sac import DecoupledConstraintSAC
        from config import DRL_CONFIG

        state_dim = int(DRL_CONFIG.get("state_dim", 40))
        action_dim = int(DRL_CONFIG.get("action_dim", 10))
        frame_stack = int(DRL_CONFIG.get("frame_stack", 8))
        agent = DecoupledConstraintSAC(state_dim=state_dim, action_dim=action_dim, device="cpu")
        agent.set_lyapunov_penalty_coeff(0.0)
        agent.value_aux_head_enable = False
        states = torch.zeros((2, frame_stack, state_dim), dtype=torch.float32)
        actions = torch.zeros((2, action_dim), dtype=torch.float32)
        weights = torch.zeros((2, 1), dtype=torch.float32)

        terms = agent.compute_actor_objective(states, actions, weights, raw_states=states)

        self.assertAlmostEqual(float(terms["constraint_actor_loss"].item()), 0.0, places=7)

    def test_transformer_temporal_indices_follow_observation_schema(self):
        from drl.networks import OrbitalTransformerEncoder
        from environment.satellite_env import OBSERVATION_FEATURES
        from config import DRL_CONFIG

        encoder = OrbitalTransformerEncoder(state_dim=int(DRL_CONFIG["state_dim"]))
        temporal_names = [OBSERVATION_FEATURES[i] for i in encoder.temporal_idx]

        self.assertEqual(temporal_names, list(DRL_CONFIG["transformer_temporal_features"]))
        self.assertNotIn("prev_alpha_prop", temporal_names)
        self.assertNotIn("prev_alpha_cpu", temporal_names)
        self.assertNotIn("energy_queue_pressure", temporal_names)
        self.assertIn("processed_queue_future_contact_ratio", temporal_names)


# ═══════════════════════════════════════════════════════════════════════
# 运行入口
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        TestTrainingConfig,
        TestAtmosphericModel,
        TestOrbitalDynamics,
        TestOrbitalPeriodSimulator,
        TestBatteryModel,
        TestPowerSubsystem,
        TestEnergyVirtualQueue,
        TestOrbitVirtualQueue,
        TestDataTaskQueue,
        TestLyapunovActionProjection,
        TestPipelineSmoke,
        TestCheckpointMetadata,
        TestRewardSemantics,
        TestPSFQueueGuard,
        TestTraceGroundStationConfig,
        TestTraceRobustnessSemantics,
        TestEvaluationReportMath,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total  = result.testsRun
    failed = len(result.failures) + len(result.errors)
    print(f"\n{'='*50}")
    print(f"测试总计: {total} | 通过: {total - failed} | 失败: {failed}")
    print(f"{'='*50}")
    sys.exit(0 if failed == 0 else 1)

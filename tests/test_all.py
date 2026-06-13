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

    @staticmethod
    def _safe_checkpoint_stats(**overrides):
        """构造一个通过 anti-collapse gate 的 checkpoint 统计样例。"""
        stats = {
            "safety_rate": 1.0,
            "episode_safety_rate": 1.0,
            "survival_rate": 1.0,
            "energy_violation_rate": 0.0,
            "delivered_value_mean": 10000.0,
            "rf_downlinked_mb_mean": 1200.0,
            "raw_equivalent_delivered_mb_mean": 5000.0,
            "comm_window_utilization": 0.80,
            "global_proc_downlink_ratio": 1.5,
            "mean_episode_proc_downlink_ratio": 1.5,
            "high_value_delivery_ratio": 0.30,
            "useful_processing_ratio": 0.45,
            "safety_intervention_rate": 0.10,
            "reward_mean": 0.0,
        }
        stats.update(overrides)
        return stats

    def test_update_freq_consistent(self):
        """DRL/TRAIN 的 update_freq 语义不同（见 config 注释），各自应为正整数。"""
        from config import DRL_CONFIG, TRAIN_CONFIG
        # DRL_CONFIG["update_freq"] = 每次更新执行的 gradient steps（当前 8）；
        # TRAIN_CONFIG["update_freq"] = 每多少 env step 触发一次更新（当前 4）。
        # 二者含义不同，不要求相等，只校验各自合法。
        self.assertGreaterEqual(int(DRL_CONFIG["update_freq"]), 1)
        self.assertGreaterEqual(int(TRAIN_CONFIG["update_freq"]), 1)

    def test_fast_training_defaults(self):
        """默认训练配置应启用多环境采样，并避免每步都反向传播"""
        from config import DRL_CONFIG, TRAIN_CONFIG
        self.assertGreaterEqual(TRAIN_CONFIG["n_envs"], 2)
        self.assertGreaterEqual(DRL_CONFIG["update_freq"], 2)

    def test_default_config_is_rl_first_not_hard_rule_takeover(self):
        """主配置默认应把核心动作面交给策略，硬规则只作为显式 profile/消融。"""
        from config import (
            ACTUATOR_GATE_CONFIG,
            DRL_CONFIG,
            HARD_RULES_CONFIG,
            ORBITAL_CONFIG,
            PROPULSION_CONTROLLER_CONFIG,
            TASK_CONFIG,
        )

        self.assertTrue(bool(PROPULSION_CONTROLLER_CONFIG["guard_only"]))
        self.assertGreaterEqual(
            float(PROPULSION_CONTROLLER_CONFIG["guard_altitude_margin_km"]),
            float(ORBITAL_CONFIG["altitude_warning_km"] - ORBITAL_CONFIG["altitude_min_km"]),
        )
        self.assertTrue(bool(ACTUATOR_GATE_CONFIG["cpu_gate_soft_mode"]))
        self.assertFalse(bool(TASK_CONFIG["cpu_action_is_admissible_budget"]))
        self.assertFalse(bool(TASK_CONFIG["enable_future_contact_cpu_gate"]))
        self.assertFalse(bool(TASK_CONFIG["enable_cpu_throttle"]))
        self.assertFalse(bool(TASK_CONFIG["enable_high_value_cpu_gate_escape"]))
        self.assertFalse(bool(TASK_CONFIG["enable_in_window_cpu_feed_floor"]))
        self.assertFalse(bool(HARD_RULES_CONFIG["enable_deliver_prob_gate"]))
        self.assertFalse(bool(HARD_RULES_CONFIG["enable_class_aware_gate"]))
        self.assertFalse(bool(HARD_RULES_CONFIG["enable_class_priority_floor"]))
        self.assertFalse(bool(HARD_RULES_CONFIG["enable_tx_high_reserve"]))
        self.assertFalse(bool(HARD_RULES_CONFIG["enable_layered_edf"]))
        self.assertFalse(bool(HARD_RULES_CONFIG["enable_in_window_tx_floor"]))
        self.assertFalse(bool(HARD_RULES_CONFIG["enable_mission_pointing_fallback"]))
        self.assertGreater(float(DRL_CONFIG["behavior_cloning_coeff"]), 0.0)
        self.assertLessEqual(float(DRL_CONFIG["behavior_cloning_coeff"]), 0.10)
        self.assertFalse(bool(DRL_CONFIG["enable_high_value_cpu_behavior_cloning"]))
        self.assertEqual(DRL_CONFIG["queue_projection_policy"], "diagnostic_only")
        self.assertFalse(bool(DRL_CONFIG["enable_deployment_queue_projection"]))
        self.assertGreater(float(DRL_CONFIG["reward_shaping_coeff"]), 0.0)
        self.assertGreater(float(DRL_CONFIG["constraint_orbit_margin_coeff"]), 0.0)

    def test_formal_eval_episodes_default(self):
        """训练内 eval 用较少 episode 控制开销（当前 3）；终评在实验脚本里用 5+。"""
        from config import TRAIN_CONFIG
        self.assertGreaterEqual(TRAIN_CONFIG["eval_episodes"], 3)

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
        # 课程阶段覆盖训练前段；total_steps 可不小于课程步数之和，余量用
        # 末段满载配置（data_arrival_scale=1.0）继续训练，故用 <= 而非精确相等。
        self.assertLessEqual(sum(stage_steps), int(TRAIN_CONFIG["total_steps"]))
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

    def test_checkpoint_selection_maximizes_delivered_voi_within_safe_set(self):
        """通过硬安全与 anti-collapse gate 后，checkpoint 仍按 delivered VoI 排序。"""
        from train import _selection_tuple

        high_voi = self._safe_checkpoint_stats(
            delivered_value_mean=14000.0,
            high_value_delivery_ratio=0.10,
            useful_processing_ratio=0.20,
        )
        low_voi = self._safe_checkpoint_stats(
            delivered_value_mean=10000.0,
            high_value_delivery_ratio=0.40,
            useful_processing_ratio=0.50,
        )

        self.assertEqual(_selection_tuple(high_voi)[0], 1.0)
        self.assertEqual(_selection_tuple(low_voi)[0], 1.0)
        self.assertGreater(
            _selection_tuple(high_voi),
            _selection_tuple(low_voi),
        )

    def test_checkpoint_selection_rejects_any_energy_violation(self):
        """硬安全约束：energy_viol>0 直接落出可行集（constraint_satisfied=0）。"""
        from train import _selection_tuple
        from config import DRL_CONFIG

        stats = self._safe_checkpoint_stats(energy_violation_rate=1e-6)

        self.assertEqual(DRL_CONFIG["checkpoint_max_energy_violation_rate"], 0.0)
        self.assertEqual(_selection_tuple(stats)[0], 0.0)

    def test_checkpoint_selection_rejects_crashing_policy(self):
        """硬安全约束：发生 crash（survival<1）直接落出可行集，且不敌零 crash 模型。"""
        from train import _selection_tuple

        crashing = self._safe_checkpoint_stats(
            safety_rate=0.67,
            episode_safety_rate=0.67,
            survival_rate=0.67,
            delivered_value_mean=30000.0,  # 交付再高也不能盖过安全
        )
        safe = self._safe_checkpoint_stats(delivered_value_mean=10000.0)

        self.assertEqual(_selection_tuple(crashing)[0], 0.0)
        self.assertEqual(_selection_tuple(safe)[0], 1.0)
        # 零 crash 的低交付模型必须优于高交付但会 crash 的模型。
        self.assertGreater(_selection_tuple(safe), _selection_tuple(crashing))

    def test_checkpoint_selection_rejects_conservative_collapse_metrics(self):
        """选模必须拒绝零下传、低窗口、proc/dl 爆炸等保守坍缩 checkpoint。"""
        from train import _selection_tuple
        from config import DRL_CONFIG

        viable = self._safe_checkpoint_stats(delivered_value_mean=14000.0)
        low_window = self._safe_checkpoint_stats(comm_window_utilization=0.0)
        zero_downlink = self._safe_checkpoint_stats(
            rf_downlinked_mb_mean=0.0,
            downlink_mean=0.0,
        )
        proc_dl_exploded = self._safe_checkpoint_stats(
            global_proc_downlink_ratio=DRL_CONFIG["checkpoint_max_proc_downlink_ratio"] * 2.0,
        )
        episode_proc_dl_invalid = self._safe_checkpoint_stats(
            mean_episode_proc_downlink_ratio=float("nan"),
        )

        self.assertEqual(_selection_tuple(viable)[0], 1.0)
        for collapsed in [low_window, zero_downlink, proc_dl_exploded, episode_proc_dl_invalid]:
            self.assertEqual(_selection_tuple(collapsed)[0], 0.0)

    def test_high_value_cpu_behavior_target_boosts_cpu_request(self):
        """raw high 可交付但策略 CPU 请求偏低时，BC 目标应推高 CPU/high logits。"""
        from config import DRL_CONFIG
        from train import _high_value_cpu_behavior_target
        from utils.action_space import decode_grouped_action

        old_enabled = DRL_CONFIG.get("enable_high_value_cpu_behavior_cloning")
        try:
            DRL_CONFIG["enable_high_value_cpu_behavior_cloning"] = True
            raw_action = np.array([0.0, 0.1, 0.0, 0.2, 0.9, 0.5, 0.5, 0.0], dtype=np.float32)
            executed_action = raw_action.copy()
            info = {
                "raw_high_mb": 120.0,
                "raw_high_next_window_deliverable_ratio": 0.8,
                "high_value_deadline_contact_mismatch": 0.1,
                "time_to_next_window_s": 300.0,
                "in_window": False,
                "energy_safe": True,
                "thermal_safe": True,
            }

            target, weight, meta = _high_value_cpu_behavior_target(
                raw_action, executed_action, 0.0, info)
        finally:
            DRL_CONFIG["enable_high_value_cpu_behavior_cloning"] = old_enabled

        self.assertTrue(bool(meta["high_value_cpu_bc_applied"]))
        self.assertGreater(float(weight), 0.0)
        self.assertGreater(float(target[1]), float(raw_action[1]))
        self.assertGreaterEqual(float(target[3]), 0.95)
        self.assertLessEqual(float(target[4]), 0.10)
        decoded = decode_grouped_action(target)
        self.assertGreater(float(decoded.cpu_ratios[0]), 0.80)
        self.assertLess(float(decoded.cpu_ratios[1]), 0.05)

    def test_high_value_cpu_behavior_target_ignores_undeliverable_raw_high(self):
        """deadline/contact 错配太高时，不应把 BC 目标推向无效 CPU 处理。"""
        from train import _high_value_cpu_behavior_target

        raw_action = np.array([0.0, 0.1, 0.0, 0.2, 0.2, 0.5, 0.5, 0.0], dtype=np.float32)
        info = {
            "raw_high_mb": 120.0,
            "raw_high_next_window_deliverable_ratio": 0.1,
            "high_value_deadline_contact_mismatch": 0.95,
            "time_to_next_window_s": 300.0,
            "in_window": False,
            "energy_safe": True,
            "thermal_safe": True,
        }

        target, weight, meta = _high_value_cpu_behavior_target(
            raw_action, raw_action, 0.0, info)

        self.assertFalse(bool(meta["high_value_cpu_bc_applied"]))
        self.assertAlmostEqual(float(weight), 0.0, places=6)
        np.testing.assert_allclose(target, raw_action)

    def test_mid_vleo_physical_defaults(self):
        """默认物理参数应落在中型VLEO对地观测星的合理量级。"""
        from config import DRAG_CONFIG, ENERGY_CONFIG, QUEUE_CONFIG, ORBITAL_CONFIG

        self.assertGreaterEqual(DRAG_CONFIG["mass_kg"], 150.0)
        self.assertLessEqual(DRAG_CONFIG["mass_kg"], 450.0)
        self.assertEqual(ORBITAL_CONFIG["altitude_warning_km"], 200.0)
        self.assertEqual(ORBITAL_CONFIG["altitude_min_km"], 180.0)
        self.assertEqual(ORBITAL_CONFIG["altitude_crash_km"], 120.0)
        self.assertEqual(ENERGY_CONFIG["solar_panel_power_w"], 800.0)
        self.assertEqual(ENERGY_CONFIG["battery_capacity_wh"], 500.0)
        self.assertAlmostEqual(ENERGY_CONFIG["battery_min_soc"], 0.15)
        self.assertAlmostEqual(ENERGY_CONFIG["battery_crash_soc"], 0.05)
        self.assertEqual(ENERGY_CONFIG["power_propulsion_max_w"], 720.0)
        self.assertEqual(ENERGY_CONFIG["power_cpu_max_w"], 25.0)
        self.assertEqual(ENERGY_CONFIG["power_tx_max_w"], 35.0)
        self.assertEqual(ENERGY_CONFIG["power_baseline_w"], 15.0)
        self.assertEqual(ENERGY_CONFIG["power_total_max_w"], 820.0)
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
        """验证 ref_altitude 处指数标高 invariant: rho(ref_alt + H)/rho(ref_alt) = 1/e。
        ref_altitude_km=350km 是大气密度模型的校准锚点 (非操作区间)。"""
        from config import DRAG_CONFIG
        h0 = float(DRAG_CONFIG["ref_altitude_km"]) * 1e3
        H = self.atm.H_scale
        ratio = self.atm.density(h0 + H) / self.atm.density(h0)
        self.assertAlmostEqual(ratio, np.exp(-1), places=5)


class TestAtmosphereSwitchingAndSpaceWeather(unittest.TestCase):
    """大气模型切换 + F10.7/Ap 显式驱动 + Cd/shape 不确定度随机化。"""

    def test_registry_models_finite_positive(self):
        """三档模型在 100~450km 全段密度有限且为正。"""
        from environment.atmosphere import make_atmosphere
        for name in ("exponential", "vallado"):
            m = make_atmosphere(name)
            for h_km in (100, 150, 200, 300, 450):
                with self.subTest(model=name, h_km=h_km):
                    rho = m.density(h_km * 1e3)
                    self.assertTrue(np.isfinite(rho) and rho > 0.0)

    def test_unknown_model_raises(self):
        from environment.atmosphere import make_atmosphere
        with self.assertRaises(ValueError):
            make_atmosphere("does_not_exist")

    def test_default_is_vallado_alias(self):
        """AtmosphericModel 向后兼容别名 = 默认 Vallado 分段指数模型。"""
        from environment.atmosphere import AtmosphericModel, PiecewiseExpAtmosphere
        self.assertIs(AtmosphericModel, PiecewiseExpAtmosphere)

    def test_f107_nominal_reproduces_baseline(self):
        """F10.7=150 → rho_scale=1.0，精确复现既有 rho_ref 基线。"""
        from environment.atmosphere import f107_to_rho_scale
        self.assertAlmostEqual(f107_to_rho_scale(150.0), 1.0, places=9)

    def test_f107_monotonic_density(self):
        """更高 F10.7 → 更高密度 (经 rho_ref 传导，三个解析模型一致)。"""
        from environment.atmosphere import make_atmosphere, f107_to_rho_scale
        from config import DRAG_CONFIG
        base = float(DRAG_CONFIG["rho_ref"])
        for name in ("exponential", "vallado"):
            m = make_atmosphere(name)
            m.rho_ref = base * f107_to_rho_scale(90.0)
            lo = m.density(250e3)
            m.rho_ref = base * f107_to_rho_scale(220.0)
            hi = m.density(250e3)
            with self.subTest(model=name):
                self.assertGreater(hi, lo)

    def test_ap_storm_raises_density_monotonic(self):
        """Ap 越高 → storm 乘子越大 → 密度越高 (静日乘子≈1.0)。"""
        from environment.atmosphere import ap_to_storm_multiplier
        quiet = ap_to_storm_multiplier(4.0)
        storm = ap_to_storm_multiplier(300.0)
        self.assertAlmostEqual(quiet, 1.0, places=1)
        self.assertGreater(storm, quiet)

    def test_nrlmsise_requires_pymsis(self):
        """未装 pymsis 时选 nrlmsise00 抛清晰 ImportError；装了则密度为正。"""
        from environment.atmosphere import make_atmosphere, SpaceWeatherState
        try:
            import pymsis  # noqa: F401
        except ImportError:
            with self.assertRaises(ImportError):
                make_atmosphere("nrlmsise00")
            return
        m = make_atmosphere("nrlmsise00")
        m.set_space_weather(SpaceWeatherState(
            f107_daily=150, f107_81avg=150, ap=4, epoch_doy=80))
        m.set_phase(1.0)
        self.assertGreater(m.density(250e3), 0.0)

    def test_cd_area_dr_scaled_by_curriculum(self):
        """scale=0 → 标称确定性；scale=1 → 不同 seed 抽到区间内不同 Cd。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from config import DRAG_CONFIG
        e0 = VLEOSatelliteEnv(seed=0)
        e0._randomization_scale = 0.0
        e0.reset()
        self.assertAlmostEqual(e0.orbit_dyn.Cd, float(DRAG_CONFIG["Cd"]), places=9)
        self.assertAlmostEqual(e0.orbit_dyn.A, float(DRAG_CONFIG["area_m2"]), places=9)
        cds = []
        cd_lo, cd_hi = DRAG_CONFIG["cd_range"]
        for s in range(6):
            e = VLEOSatelliteEnv(seed=s)
            e._randomization_scale = 1.0
            e.reset()
            self.assertTrue(float(cd_lo) <= e.orbit_dyn.Cd <= float(cd_hi))
            cds.append(round(e.orbit_dyn.Cd, 4))
        self.assertGreater(len(set(cds)), 1)

    def test_disable_switches_force_nominal(self):
        """关 enable_shape_cd_randomization / enable_solar_activity_randomization
        → 即使 scale=1 也钉死标称 Cd/area/F10.7 (供 robustness 干净控制 drag)。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from config import DRAG_CONFIG
        keys = ("enable_shape_cd_randomization", "enable_solar_activity_randomization")
        saved = {k: DRAG_CONFIG.get(k) for k in keys}
        try:
            DRAG_CONFIG["enable_shape_cd_randomization"] = False
            DRAG_CONFIG["enable_solar_activity_randomization"] = False
            e = VLEOSatelliteEnv(seed=4)
            e._randomization_scale = 1.0
            e.reset()
            self.assertAlmostEqual(e.orbit_dyn.Cd, float(DRAG_CONFIG["Cd"]), places=9)
            self.assertAlmostEqual(e.orbit_dyn.A, float(DRAG_CONFIG["area_m2"]), places=9)
            self.assertAlmostEqual(
                e._sw_state.f107_daily, float(DRAG_CONFIG["f107_nominal"]), places=9)
            self.assertAlmostEqual(
                e._sw_state.f107_81avg, float(DRAG_CONFIG["f107_nominal"]), places=9)
        finally:
            for k, v in saved.items():
                DRAG_CONFIG[k] = v

    def test_space_weather_state_set_on_reset(self):
        """reset 后 atm 持有当前 episode 的 SpaceWeatherState，且 state_dim 不变。"""
        from environment.satellite_env import VLEOSatelliteEnv
        env = VLEOSatelliteEnv(seed=2)
        obs = env.reset()
        self.assertEqual(obs.shape[0], env.state_dim)
        self.assertIsNotNone(env._sw_state)
        self.assertGreater(env._sw_state.f107_daily, 0.0)


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
        # 用 220km (在操作区中下段，drag 可观)，且推力必须高于 ignition_threshold (150W)。
        # 选 P_prop=500W 给出确定的非零推力，60s 时间步累积足以看到高度差异。
        h = 220e3
        no_thrust  = self.orb.step(h, 0.0,   60.0)
        with_thrust = self.orb.step(h, 500.0, 60.0)
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

    def test_warning_altitude_between_min_and_warning(self):
        """[altitude_min_km, altitude_warning_km] = [180, 200]km 是警告区
        (t/d 在 1.0~2.0 之间，marginal cruise — 可短暂操作但需主动爬升)。"""
        from config import ORBITAL_CONFIG
        h_mid = 0.5 * (float(ORBITAL_CONFIG["altitude_min_km"])
                       + float(ORBITAL_CONFIG["altitude_warning_km"])) * 1e3
        result = self.orb.step(h_mid, 0.0, 1.0)
        self.assertTrue(result["is_safe"])
        self.assertTrue(result["is_warning"])
        self.assertFalse(result["is_crashed"])
        self.assertEqual(result["safety_stage"], "warning")

    def test_unsafe_altitude_not_immediate_crash(self):
        """[altitude_crash_km, altitude_min_km] = [120, 180]km 是不安全区
        (t/d < 1，drag 主导，即使满推也下降；尚未到物理再入)。"""
        from config import ORBITAL_CONFIG
        h_mid = 0.5 * (float(ORBITAL_CONFIG["altitude_crash_km"])
                       + float(ORBITAL_CONFIG["altitude_min_km"])) * 1e3
        result = self.orb.step(h_mid, 0.0, 1.0)
        self.assertFalse(result["is_safe"])
        self.assertFalse(result["is_crashed"])
        self.assertEqual(result["safety_stage"], "unsafe")

    def test_reentry_altitude_triggers_crash(self):
        """altitude_crash_km 以下触发物理再入/坠毁终态 (PDF VLEO 120km 下边界)。"""
        from config import ORBITAL_CONFIG
        crash_m = float(ORBITAL_CONFIG["altitude_crash_km"]) * 1e3
        # 严格低于 crash 边界 (1km 余量) 以确保触发
        result = self.orb.step(crash_m - 1e3, 0.0, 1.0)
        self.assertTrue(result["is_crashed"])
        self.assertEqual(result["safety_stage"], "failure")

    def test_extreme_altitudes_are_sanitized_without_runtime_warnings(self):
        """非法或极低高度不应触发 NaN/overflow warning，避免污染训练日志。"""
        import warnings

        for altitude_m in (-7000e3, -6371e3, -1000.0):
            with self.subTest(altitude_m=altitude_m):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", RuntimeWarning)
                    result = self.orb.step(altitude_m, 0.0, 10.0)

                self.assertEqual(caught, [])
                self.assertTrue(np.isfinite(result["altitude_m"]))
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

    def test_full_action_uses_strict_priority_split(self):
        """超预算时 compute_total_load 使用 strict_priority 切片，与 env.step 主路径
        (allocate_power_strict_priority) 语义一致；不再按比例等比例缩。

        通过手动 monkey-patch power_total_max_w 制造超预算情形，避免依赖具体配置值。
        请求 [1,1,1] = [P_prop_max, P_cpu_max, P_tx_max]，预算设为略低于这三项之和:
        - 窗口外、非紧急 (默认 prop > cpu > tx)
        - 窗口内、非紧急 (tx > prop > cpu)
        - 紧急 + 窗口内 (prop > tx > cpu)
        """
        from config import ENERGY_CONFIG
        action = np.array([1.0, 1.0, 1.0])
        # 强制超预算：把 ENERGY_CONFIG["power_total_max_w"] 砍到 (prop_max + cpu_max
        # + tx_max + baseline) 的 70%。这样 [1,1,1] 必然超出 budget，触发 strict_priority。
        # compute_total_load 直接读 ENERGY_CONFIG，monkey-patch self.ps 没用。
        full_demand = self.ps.P_prop_max + self.ps.P_cpu_max + self.ps.P_tx_max
        original_total_max = ENERGY_CONFIG["power_total_max_w"]
        ENERGY_CONFIG["power_total_max_w"] = self.ps.P_baseline + full_demand * 0.7
        budget = ENERGY_CONFIG["power_total_max_w"] - self.ps.P_baseline
        try:
            # 默认 (in_window=False, force_prop_priority=False) → prop > cpu > tx
            info = self.ps.compute_total_load(action)
            self.assertAlmostEqual(info["P_propulsion_w"],
                                   min(self.ps.P_prop_max, budget))
            self.assertAlmostEqual(
                info["P_cpu_w"],
                min(self.ps.P_cpu_max, max(0.0, budget - self.ps.P_prop_max)))
            remaining_for_tx = max(
                0.0, budget - self.ps.P_prop_max - self.ps.P_cpu_max)
            self.assertAlmostEqual(info["P_tx_w"],
                                   min(self.ps.P_tx_max, remaining_for_tx))
            self.assertAlmostEqual(info["P_total_w"],
                                   ENERGY_CONFIG["power_total_max_w"])

            # in_window=True → tx > prop > cpu
            info_win = self.ps.compute_total_load(action, in_window=True)
            self.assertAlmostEqual(info_win["P_tx_w"],
                                   min(self.ps.P_tx_max, budget))
            self.assertAlmostEqual(
                info_win["P_propulsion_w"],
                min(self.ps.P_prop_max, max(0.0, budget - self.ps.P_tx_max)))
            remaining_for_cpu = max(
                0.0, budget - self.ps.P_tx_max - self.ps.P_prop_max)
            self.assertAlmostEqual(info_win["P_cpu_w"],
                                   min(self.ps.P_cpu_max, remaining_for_cpu))

            # in_window=True + force_prop_priority=True → prop > tx > cpu
            info_emg = self.ps.compute_total_load(
                action, in_window=True, force_prop_priority=True)
            self.assertAlmostEqual(info_emg["P_propulsion_w"],
                                   min(self.ps.P_prop_max, budget))
            self.assertAlmostEqual(
                info_emg["P_tx_w"],
                min(self.ps.P_tx_max, max(0.0, budget - self.ps.P_prop_max)))
            remaining_for_cpu = max(
                0.0, budget - self.ps.P_prop_max - self.ps.P_tx_max)
            self.assertAlmostEqual(info_emg["P_cpu_w"],
                                   min(self.ps.P_cpu_max, remaining_for_cpu))
        finally:
            ENERGY_CONFIG["power_total_max_w"] = original_total_max

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
        """恰好在安全线 h_min 时 margin=0，队列不变"""
        from config import ORBITAL_CONFIG
        h_min_m = ORBITAL_CONFIG["altitude_min_km"] * 1e3
        self.queue.value = 3.0
        result = self.queue.update(h_min_m)
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
        from config import OBJECTIVE_VERSION, ORBITAL_CONFIG

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
                float(ORBITAL_CONFIG["altitude_crash_km"]),
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
            # lyapunov_penalty_coeff 是训练超参，随 checkpoint 恢复（见
            # test_checkpoint_restores_adaptive_lyapunov_coeff）；restore_safety_config=False
            # 只保留消融的安全开关（enable_lyapunov/use_psf），不影响该系数——且开关已关时
            # 系数不参与计算，恢复无害。
            self.assertAlmostEqual(
                ablation_scheduler.agent.get_lyapunov_penalty_coeff(),
                1.234,
            )
            self.assertEqual(metadata.get("enable_lyapunov"), True)
            # ckpt 由 full_scheduler(use_psf=True) 保存，metadata 记录的是保存时配置 → True；
            # ablation 自身的 use_psf 仍为 False（上面 943 已验证未被覆盖）。
            self.assertEqual(metadata.get("use_psf"), True)

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

    def test_default_reward_shaping_keeps_action_timing_visible(self):
        """默认论文 reward 应给时机错误的 CPU/TX/推进动作留下可学习信号。"""
        from config import REWARD_CONFIG

        self.assertGreater(float(REWARD_CONFIG["w_proc_far_window_penalty"]), 0.0)
        self.assertGreater(float(REWARD_CONFIG["w_window_underuse_penalty"]), 0.0)
        self.assertLess(float(REWARD_CONFIG["w_energy_penalty"]), 0.0)
        self.assertGreater(float(REWARD_CONFIG["w_prop_overburn_penalty"]), 0.0)

    def test_default_reward_penalizes_far_processing_and_window_underuse(self):
        """远离窗口处理、窗口期有货不下传应直接反映在 reward 分解中。"""
        from copy import deepcopy

        from config import REWARD_CONFIG
        from objectives.mission_reward import compute_mission_reward

        cfg = deepcopy(REWARD_CONFIG)
        cfg.update({
            "_in_comm_window": False,
            "_time_to_next_window_s": 900.0,
            "_processed_queue_mb": 120.0,
        })
        far_processing = compute_mission_reward(
            delivered_value=0.0,
            on_time_delivered_value=0.0,
            expired_value=0.0,
            dropped_value=0.0,
            transmitted_mb=0.0,
            processed_mb=20.0,
            total_power_w=120.0,
            propulsion_power_w=0.0,
            dt_s=10.0,
            cfg=cfg,
        )
        self.assertLess(float(far_processing.components["r_proc_far_window"]), 0.0)

        cfg["_in_comm_window"] = True
        window_idle = compute_mission_reward(
            delivered_value=0.0,
            on_time_delivered_value=0.0,
            expired_value=0.0,
            dropped_value=0.0,
            transmitted_mb=10.0,
            processed_mb=0.0,
            total_power_w=120.0,
            propulsion_power_w=0.0,
            dt_s=10.0,
            cfg=cfg,
            in_window=True,
            link_capacity_mb=80.0,
            pre_tx_pending_mb=100.0,
        )
        self.assertLess(float(window_idle.components["r_window_underuse"]), 0.0)

    def test_analytic_propulsion_controller_can_be_limited_to_guard_only(self):
        """非安全临界状态下，guard_only 模式不能吞掉 agent 的推进动作。"""
        from config import PROPULSION_CONTROLLER_CONFIG
        from environment.satellite_env import VLEOSatelliteEnv

        old = {
            "enabled": PROPULSION_CONTROLLER_CONFIG.get("enabled"),
            "guard_only": PROPULSION_CONTROLLER_CONFIG.get("guard_only"),
            "guard_altitude_margin_km": PROPULSION_CONTROLLER_CONFIG.get("guard_altitude_margin_km"),
            "guard_soc_margin": PROPULSION_CONTROLLER_CONFIG.get("guard_soc_margin"),
        }
        try:
            PROPULSION_CONTROLLER_CONFIG["enabled"] = True
            PROPULSION_CONTROLLER_CONFIG["guard_only"] = True
            PROPULSION_CONTROLLER_CONFIG["guard_altitude_margin_km"] = 5.0
            PROPULSION_CONTROLLER_CONFIG["guard_soc_margin"] = 0.02
            env = VLEOSatelliteEnv(seed=17)
            env.reset()
            env.altitude_m = 260e3
            env.battery.soc = 0.80

            raw_action = np.array([0.13, 0.4, 0.4], dtype=np.float32)
            controlled, meta = env._apply_analytic_propulsion_controller(raw_action)

            self.assertAlmostEqual(float(controlled[0]), float(raw_action[0]), places=6)
            self.assertFalse(bool(meta["analytic_propulsion_controller_enabled"]))
            self.assertEqual(meta["analytic_propulsion_reason"], "guard_not_active")
        finally:
            for key, value in old.items():
                if value is None:
                    PROPULSION_CONTROLLER_CONFIG.pop(key, None)
                else:
                    PROPULSION_CONTROLLER_CONFIG[key] = value

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
        from config import TASK_CONFIG

        env = VLEOSatelliteEnv(seed=31)
        env.reset()
        env._scene_phase_offset_fraction = 0.0
        # 关闭 per-episode rule 打乱：本测试验证 base TASK_CONFIG 映射正确性。
        env._phase_scene_rules = list(TASK_CONFIG["phase_scene_rules"])
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
        env._phase_scene_rules = list(TASK_CONFIG["phase_scene_rules"])
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
                # 当前 reward 为 class-weighted：需提供分类 breakdown（这里全计为 high 类），
                # 否则分类值默认 0 会让 class-weighted r_value 归零。
                "delivered_high_value": 18.0,
                "delivered_medium_value": 0.0,
                "delivered_low_value": 0.0,
                "on_time_delivered_value": 18.0,
                "expired_value": 0.0,
                "dropped_value": 0.0,
            },
        )

        # class-weighted：r_value = w_delivered_value · class_high_reward_weight · delivered_high
        self.assertAlmostEqual(
            breakdown["r_delivered_value"],
            REWARD_CONFIG["w_delivered_value"]
            * REWARD_CONFIG.get("class_high_reward_weight", 3.0) * 18.0,
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
        info = {
            "processed_high_value": 100.0,
            "processed_medium_value": 100.0,
            "processed_low_value": 1e6,
            "future_capacity_mb": 1000.0,
        }
        high_deliverable = lambda *_a, **_k: {
            "raw_high_next_window_deliverable_ratio": 0.8,
            "processed_high_next_window_deliverable_ratio": 0.6,
            "high_value_deadline_contact_mismatch": 0.1,
        }
        low_deliverable = lambda *_a, **_k: {
            "raw_high_next_window_deliverable_ratio": 0.0,
            "processed_high_next_window_deliverable_ratio": 0.0,
            "high_value_deadline_contact_mismatch": 1.0,
        }

        # 1) high gate 缺失（任务赶不上下个窗口、不可交付）→ 无 credit。
        env._contact = {"in_window": False, "time_to_next_window_s": 3600.0}
        env.comm_queue.value = 0.0
        env.task_tracker.deadline_contact_stats = low_deliverable
        self.assertEqual(env._deliverable_processing_credit(info), 0.0)

        # 2) 近窗口可交付（high gate）+ future capacity 有空间 → credit>0；
        #    low 类只是诊断量，不进 credit（with/without low 相等）。
        env._contact = {"in_window": True, "time_to_next_window_s": 0.0}
        env.comm_queue.value = 0.0
        env.task_tracker.deadline_contact_stats = high_deliverable
        credit_with_low = env._deliverable_processing_credit(info)
        credit_without_low = env._deliverable_processing_credit({
            "processed_high_value": 100.0,
            "processed_medium_value": 100.0,
            "future_capacity_mb": 1000.0,
        })
        self.assertGreater(credit_with_low, 0.0)
        self.assertAlmostEqual(credit_with_low, credit_without_low, places=7)

        # 3) processed queue 已占满未来 capacity（capacity_gate=0）→ 无 credit。
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
        deliverable_r = torch.tensor([[0.0]])
        reward_next_q = torch.tensor([[2.0]])
        deliverable_next_q = torch.tensor([[0.0]])
        constraint_next_q = torch.tensor([[3.0]])

        # 新签名：reward / deliverable / constraint 三路解耦的 TD 目标。
        target_q, target_deliverable, target_c = SACAgent._compute_td_targets(
            agent, reward, done, lya, deliverable_r, reward_next_q,
            deliverable_next_q, constraint_next_q)

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
        start = source.index("_td_mode = str(DRL_CONFIG.get(\"td_reward_mode\"")
        end = source.index("reward_scaled = reward_for_td", start)
        reward_window = source[start:end]

        self.assertNotIn("projection_penalty", reward_window)
        self.assertNotIn("action_mod_penalty", reward_window)
        self.assertIn("delivered_plus", reward_window)
        self.assertIn("reward_td_excludes_safety_action_penalty", source)

    def test_default_td_reward_mode_uses_task_shaping_not_primary_only(self):
        """默认 TD reward 不能只学 delivered value，否则 deadline/expiry/window 信号只进日志。"""
        from config import DRL_CONFIG

        self.assertEqual(DRL_CONFIG["td_reward_mode"], "delivered_plus")

    def test_delivered_plus_td_reward_keeps_task_shaping_excluding_safety_action_cost(self):
        """delivered_plus 应包含任务 shaping 和 potential shaping，但不包含安全壳动作罚项。"""
        from train import _task_td_reward_from_breakdown

        reward = _task_td_reward_from_breakdown(
            {
                "primary_mission_reward": 10.0,
                "auxiliary_shaping_reward": -4.0,
                "r_actuator_violation": -3.0,
                "r_shaping": 1.5,
            },
            env_reward=-999.0,
            mode="delivered_plus",
        )

        self.assertAlmostEqual(reward, 10.0 - 4.0 - (-3.0) + 1.5)

    def test_n_step_replay_keeps_parallel_envs_isolated(self):
        """多环境训练时，每个 env 必须有独立 n-step 队列，不能跨 env 串 reward/next_state。"""
        from drl.agent import SACAgent
        from config import DRL_CONFIG, N_STEP_CONFIG

        self.assertTrue(bool(N_STEP_CONFIG["enabled"]))
        n_step = int(N_STEP_CONFIG["n"])
        state_dim = int(DRL_CONFIG.get("state_dim", 30))
        frame_stack = int(DRL_CONFIG.get("frame_stack", 8))
        agent = SACAgent(state_dim=state_dim, action_dim=3, device="cpu")
        state = np.zeros((frame_stack, state_dim), dtype=np.float32)
        action = np.zeros(3, dtype=np.float32)

        for _ in range(n_step - 1):
            agent.store(state, action, 1.0, state, d=False, terminated=False, env_id=0)
            agent.store(state, action, 2.0, state, d=False, terminated=False, env_id=1)

        self.assertEqual(len(agent.buffer), 0)

        agent.store(state, action, 1.0, state, d=False, terminated=False, env_id=0)

        self.assertEqual(len(agent.buffer), 1)
        expected_reward = sum(float(agent.gamma) ** k for k in range(n_step))
        self.assertAlmostEqual(float(agent.buffer.rewards[0, 0]), expected_reward, places=5)

    def test_n_step_reset_flushes_time_limit_tail_before_env_reset(self):
        """time-limit truncation 可以 bootstrap，但 env reset 时不能把尾巴拼到新 episode。"""
        from drl.agent import SACAgent
        from config import DRL_CONFIG, N_STEP_CONFIG

        self.assertTrue(bool(N_STEP_CONFIG["enabled"]))
        n_step = int(N_STEP_CONFIG["n"])
        state_dim = int(DRL_CONFIG.get("state_dim", 30))
        frame_stack = int(DRL_CONFIG.get("frame_stack", 8))
        agent = SACAgent(state_dim=state_dim, action_dim=3, device="cpu")
        state = np.zeros((frame_stack, state_dim), dtype=np.float32)
        action = np.zeros(3, dtype=np.float32)

        for _ in range(n_step - 1):
            agent.store(state, action, 1.0, state, d=True, terminated=False, env_id=0)
        self.assertEqual(len(agent.buffer), 0)

        agent.reset_env_aggregator(0)
        self.assertEqual(len(agent.buffer), n_step - 1)
        expected_tail_reward = sum(float(agent.gamma) ** k for k in range(n_step - 1))
        self.assertAlmostEqual(float(agent.buffer.rewards[0, 0]), expected_tail_reward, places=5)

        agent.store(state, action, 2.0, state, d=False, terminated=False, env_id=0)

        self.assertEqual(len(agent.buffer), n_step - 1)

    def test_training_cli_exposes_td_reward_mode_override(self):
        """训练入口必须能跑 primary/env_total/delivered_plus 三组 reward-TD 对照。"""
        import inspect
        import train

        source = inspect.getsource(train)

        self.assertIn("--td_reward_mode", source)
        self.assertIn('DRL_CONFIG["td_reward_mode"]', source)
        self.assertIn("choices=[\"primary\", \"env_total\", \"delivered_plus\"]", source)

    def test_training_cli_exposes_rl_first_rule_switches(self):
        """训练入口要能逐项撤掉 hard rules，验证 actor 本体学习能力。"""
        import inspect
        import train

        source = inspect.getsource(train)
        expected_flags = [
            "--cpu_gate_soft_mode",
            "--disable_future_contact_cpu_gate",
            "--disable_in_window_cpu_feed_floor",
            "--disable_class_priority_floor",
            "--disable_deliverability_gate",
            "--disable_tx_high_reserve",
            "--disable_layered_edf",
            "--rl_first_training_profile",
            "--disable_task_scaffold_curriculum",
        ]

        for flag in expected_flags:
            self.assertIn(flag, source)

        self.assertIn("ACTUATOR_GATE_CONFIG", source)
        self.assertIn("TASK_CONFIG", source)
        self.assertIn('HARD_RULES_CONFIG["enable_class_priority_floor"]', source)

    def test_default_training_profile_keeps_core_actions_learnable(self):
        """主训练默认不应让推进、CPU、TX、指向四个核心动作都被硬规则接管。"""
        from copy import deepcopy
        from types import SimpleNamespace
        from config import (
            ACTUATOR_GATE_CONFIG,
            HARD_RULES_CONFIG,
            PROPULSION_CONTROLLER_CONFIG,
            TASK_CONFIG,
        )
        from train import _apply_training_safety_profile

        saved_prop = deepcopy(PROPULSION_CONTROLLER_CONFIG)
        saved_actuator = deepcopy(ACTUATOR_GATE_CONFIG)
        saved_task = deepcopy(TASK_CONFIG)
        saved_hard = deepcopy(HARD_RULES_CONFIG)
        try:
            _apply_training_safety_profile(
                SimpleNamespace(
                    use_hard_rule_training_profile=False,
                    rl_first_training_profile=False,
                ),
                announce=False,
            )
            self.assertTrue(PROPULSION_CONTROLLER_CONFIG["guard_only"])
            self.assertTrue(ACTUATOR_GATE_CONFIG["cpu_gate_soft_mode"])
            self.assertFalse(TASK_CONFIG["cpu_action_is_admissible_budget"])
            self.assertFalse(TASK_CONFIG["enable_future_contact_cpu_gate"])
            self.assertFalse(TASK_CONFIG["enable_cpu_throttle"])
            self.assertFalse(TASK_CONFIG["enable_high_value_cpu_gate_escape"])
            self.assertFalse(TASK_CONFIG["enable_in_window_cpu_feed_floor"])
            self.assertFalse(HARD_RULES_CONFIG["enable_deliver_prob_gate"])
            self.assertFalse(HARD_RULES_CONFIG["enable_class_aware_gate"])
            self.assertFalse(HARD_RULES_CONFIG["enable_class_priority_floor"])
            self.assertFalse(HARD_RULES_CONFIG["enable_tx_high_reserve"])
            self.assertFalse(HARD_RULES_CONFIG["enable_layered_edf"])
            self.assertFalse(HARD_RULES_CONFIG["enable_in_window_tx_floor"])
            self.assertFalse(HARD_RULES_CONFIG["enable_mission_pointing_fallback"])
        finally:
            PROPULSION_CONTROLLER_CONFIG.clear()
            PROPULSION_CONTROLLER_CONFIG.update(saved_prop)
            ACTUATOR_GATE_CONFIG.clear()
            ACTUATOR_GATE_CONFIG.update(saved_actuator)
            TASK_CONFIG.clear()
            TASK_CONFIG.update(saved_task)
            HARD_RULES_CONFIG.clear()
            HARD_RULES_CONFIG.update(saved_hard)

    def test_task_scaffold_curriculum_releases_hard_rules_by_final_stage(self):
        """前期 bootstrap 可以开任务脚手架，但 Final 阶段必须释放回 policy-first。"""
        from copy import deepcopy
        from config import HARD_RULES_CONFIG, PROPULSION_CONTROLLER_CONFIG, TASK_CONFIG
        from train import _LAST_SCAFFOLD_STAGE, _apply_stage_task_scaffold

        saved_prop = deepcopy(PROPULSION_CONTROLLER_CONFIG)
        saved_task = deepcopy(TASK_CONFIG)
        saved_hard = deepcopy(HARD_RULES_CONFIG)
        try:
            _LAST_SCAFFOLD_STAGE["value"] = None
            _apply_stage_task_scaffold("Adapt_50")
            self.assertFalse(PROPULSION_CONTROLLER_CONFIG["guard_only"])
            self.assertTrue(HARD_RULES_CONFIG["enable_in_window_tx_floor"])
            self.assertTrue(HARD_RULES_CONFIG["enable_mission_pointing_fallback"])
            self.assertFalse(HARD_RULES_CONFIG["enable_class_priority_floor"])
            self.assertFalse(TASK_CONFIG["enable_future_contact_cpu_gate"])

            _apply_stage_task_scaffold("Final")
            self.assertTrue(PROPULSION_CONTROLLER_CONFIG["guard_only"])
            self.assertFalse(HARD_RULES_CONFIG["enable_in_window_tx_floor"])
            self.assertFalse(HARD_RULES_CONFIG["enable_mission_pointing_fallback"])
        finally:
            _LAST_SCAFFOLD_STAGE["value"] = None
            PROPULSION_CONTROLLER_CONFIG.clear()
            PROPULSION_CONTROLLER_CONFIG.update(saved_prop)
            TASK_CONFIG.clear()
            TASK_CONFIG.update(saved_task)
            HARD_RULES_CONFIG.clear()
            HARD_RULES_CONFIG.update(saved_hard)

    def test_stage_psf_policy_matches_current_curriculum_names(self):
        """PSF policy 必须识别当前 Adapt_* curriculum 名称。"""
        from scheduler.integrated_scheduler import IntegratedScheduler
        from train import _LAST_PSF_STAGE, _apply_stage_psf_policy

        scheduler = IntegratedScheduler(device="cpu", use_psf=True)
        try:
            _LAST_PSF_STAGE["value"] = None
            _apply_stage_psf_policy(scheduler, "Adapt_50")
            self.assertTrue(scheduler.use_psf)
            self.assertEqual(scheduler.psf.K, 10)
            self.assertEqual(scheduler.psf.line_search_steps, 5)
            self.assertTrue(scheduler.psf.long_horizon_enabled)

            _apply_stage_psf_policy(scheduler, "Final")
            self.assertTrue(scheduler.use_psf)
            self.assertEqual(scheduler.psf.K, 5)
            self.assertEqual(scheduler.psf.line_search_steps, 3)
            self.assertFalse(scheduler.psf.long_horizon_enabled)
        finally:
            _LAST_PSF_STAGE["value"] = None

    def test_training_cli_can_restore_hard_rule_bootstrap_profile(self):
        """旧的硬规则脚手架仍应能一键恢复，便于做消融和稳定性对照。"""
        import inspect
        import train

        source = inspect.getsource(train)

        self.assertIn("--use_hard_rule_training_profile", source)
        self.assertIn('PROPULSION_CONTROLLER_CONFIG["guard_only"] = False', source)
        self.assertIn('ACTUATOR_GATE_CONFIG["cpu_gate_soft_mode"] = False', source)
        self.assertIn('TASK_CONFIG["cpu_action_is_admissible_budget"] = True', source)
        self.assertIn('TASK_CONFIG["enable_future_contact_cpu_gate"] = True', source)
        self.assertIn('TASK_CONFIG["enable_cpu_throttle"] = True', source)
        self.assertIn('TASK_CONFIG["enable_high_value_cpu_gate_escape"] = True', source)
        self.assertIn('TASK_CONFIG["enable_in_window_cpu_feed_floor"] = True', source)
        self.assertIn('HARD_RULES_CONFIG["enable_deliver_prob_gate"] = True', source)
        self.assertIn('HARD_RULES_CONFIG["enable_class_aware_gate"] = True', source)
        self.assertIn('HARD_RULES_CONFIG["enable_class_priority_floor"] = True', source)
        self.assertIn('HARD_RULES_CONFIG["enable_tx_high_reserve"] = True', source)
        self.assertIn('HARD_RULES_CONFIG["enable_layered_edf"] = True', source)
        self.assertIn('HARD_RULES_CONFIG["enable_in_window_tx_floor"] = True', source)
        self.assertIn('HARD_RULES_CONFIG["enable_mission_pointing_fallback"] = True', source)

    def test_training_cli_exposes_n_step_ablation_switches(self):
        """训练入口要能直接关闭/调整 n-step，复现实验排查数据污染问题。"""
        import inspect
        import train

        source = inspect.getsource(train)

        self.assertIn("--disable_n_step", source)
        self.assertIn("--n_step_n", source)
        self.assertIn("N_STEP_CONFIG", source)
        self.assertIn('N_STEP_CONFIG["enabled"]', source)
        self.assertIn('N_STEP_CONFIG["n"]', source)

    def test_serial_env_reset_clears_matching_n_step_aggregator(self):
        """串行 backend 与子进程 backend 一样，reset 前后都不能跨 episode 拼 n-step。"""
        import inspect
        import train

        source = inspect.getsource(train.train)
        start = source.index("def _reset_serial_done_slots")
        end = source.index("def _process_step_results", start)
        reset_window = source[start:end]

        self.assertIn("reset_env_aggregator", reset_window)
        self.assertIn('slot.get("env_id", 0)', reset_window)

    def test_training_loop_passes_env_id_into_n_step_replay(self):
        """训练主 loop 必须把 slot env_id 传到 replay，旧 learn 入口也要支持 env_id。"""
        import inspect
        import train
        from scheduler.integrated_scheduler import IntegratedScheduler

        train_source = inspect.getsource(train.train)
        learn_source = inspect.getsource(IntegratedScheduler.learn)

        self.assertIn('env_id=slot.get("env_id", 0)', train_source)
        self.assertIn("scheduler.reset_env_aggregator", train_source)
        self.assertIn("env_id: int = 0", learn_source)
        self.assertIn("env_id=env_id", learn_source)

    def test_evaluate_cli_exposes_all_safety_layer_ablation_flags(self):
        """评估入口要能一键关闭每个 hard-rule 轴，方便做安全壳归因实验。"""
        import inspect
        import evaluate_optimized

        source = inspect.getsource(evaluate_optimized.main)
        expected_flags = [
            "--use_hard_rule_shell",
            "--disable_analytic_propulsion",
            "--disable_pointing_fallback",
            "--disable_in_window_tx_floor",
            "--disable_future_contact_cpu_gate",
            "--disable_in_window_cpu_feed_floor",
            "--disable_class_priority_floor",
            "--disable_deliverability_gate",
            "--disable_tx_high_reserve",
            "--disable_layered_edf",
        ]

        for flag in expected_flags:
            self.assertIn(flag, source)

        self.assertIn("hard_rule_ablation_requested", source)
        self.assertIn("enable_hard_rule_shell", source)

        for name in [
            "disable_in_window_tx_floor",
            "disable_future_contact_cpu_gate",
            "disable_in_window_cpu_feed_floor",
            "disable_class_priority_floor",
            "disable_deliverability_gate",
            "disable_tx_high_reserve",
            "disable_layered_edf",
        ]:
            self.assertIn(f"{name}=bool(args.{name})", source)

    def test_training_logs_action_intervention_diagnostics_by_dimension(self):
        """训练日志必须暴露分维度动作改写和主要 hard-rule 触发率。"""
        import inspect
        import train

        source = inspect.getsource(train.train)

        for field in [
            "raw_executed_action_l2_prop",
            "raw_executed_action_l2_cpu",
            "raw_executed_action_l2_tx",
            "raw_executed_action_l2_pointing",
            "analytic_propulsion_applied",
            "in_window_tx_floor_applied",
            "mission_pointing_fallback_applied",
            "future_contact_cpu_gate_applied",
        ]:
            self.assertIn(field, source)

    def test_training_logs_td_reward_scale_components(self):
        """训练日志必须暴露 TD reward 尺度和 shaping 组成，便于判断 critic 是否失衡。"""
        import inspect
        import train

        source = inspect.getsource(train.train)

        for field in [
            "reward_scale",
            "task_reward_for_td",
            "reward_for_td_scaled",
            "td_primary_component",
            "td_auxiliary_component",
            "td_potential_shaping_component",
            "td_excluded_safety_action_component",
        ]:
            self.assertIn(field, source)

    def test_env_reports_in_window_tx_floor_intervention(self):
        """TX floor 是 hard rule，必须进 info 方便统计 actor 被接管比例。"""
        import inspect
        from environment.satellite_env import VLEOSatelliteEnv

        source = inspect.getsource(VLEOSatelliteEnv.step)

        self.assertIn("in_window_tx_floor_applied", source)
        self.assertIn("in_window_tx_floor_alpha_before", source)
        self.assertIn("in_window_tx_floor_alpha_after", source)

    def test_evaluate_report_aggregates_hard_rule_intervention_rates(self):
        """评估报告也要汇总 hard-rule 触发率，不能只在训练日志里可见。"""
        import inspect
        import evaluate_optimized

        source = inspect.getsource(evaluate_optimized.evaluate_model)

        for field in [
            "analytic_propulsion_applied_rate",
            "in_window_tx_floor_applied_rate",
            "mission_pointing_fallback_applied_rate",
            "future_contact_cpu_gate_applied_rate",
        ]:
            self.assertIn(field, source)

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
            # 显式声明 warning 阈值，使本测试自包含、不随 config 的 warning_temp_c 漂移：
            # 验证"温度超过 warning → thermal excess 进 constraint cost"这一语义本身。
            info={"thermal_temperature_c": 55.0, "thermal_stage": "warning",
                  "thermal_warning_temp_c": 45.0},
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
        from config import (
            DRL_CONFIG,
            PROCESSING_CREDIT_CONFIG,
            PROPULSION_CONTROLLER_CONFIG,
            REWARD_CONFIG,
            TASK_CONFIG,
        )

        self.assertEqual(DRL_CONFIG["state_dim"], 65)
        self.assertTrue(bool(DRL_CONFIG["enable_capacity_aware_cost_v2"]))
        self.assertFalse(bool(DRL_CONFIG["enable_deliverable_processing_reward"]))
        self.assertEqual(DRL_CONFIG["queue_projection_policy"], "diagnostic_only")
        self.assertFalse(bool(DRL_CONFIG["enable_deployment_queue_projection"]))

        # 注：配置已从早期 learning-first（约束系数全关）演进到 constrained 阶段，
        # 以下若干 constraint 系数已显式启用，期望值随当前 config 更新。
        self.assertEqual(DRL_CONFIG["constraint_efficiency_processed_value_credit"], 0.0)
        self.assertEqual(DRL_CONFIG["constraint_efficiency_cost_coeff"], 0.0)
        self.assertEqual(DRL_CONFIG["constraint_window_waste_coeff"], 0.6)
        self.assertEqual(DRL_CONFIG["constraint_processed_backlog_coeff"], 0.0)
        self.assertEqual(DRL_CONFIG["constraint_low_value_waste_coeff"], 0.0)
        self.assertEqual(DRL_CONFIG["constraint_unproductive_cpu_coeff"], 0.0)
        self.assertAlmostEqual(DRL_CONFIG["constraint_over_processing_coeff"], 1.0)
        self.assertLessEqual(DRL_CONFIG["constraint_over_processing_coeff"], 2.0)
        self.assertLessEqual(DRL_CONFIG["constraint_over_processing_clip"], 10.0)
        # RAW_TO_PROCESSED_RATIO=0.25 → norm = 400 × 0.25 = 100（与压缩后 MB 量级对齐）
        self.assertAlmostEqual(DRL_CONFIG["constraint_capacity_norm_mb"], 100.0)
        self.assertAlmostEqual(DRL_CONFIG["constraint_capacity_norm"], 100.0)
        self.assertLessEqual(DRL_CONFIG["constraint_over_processing_ratio_weight"], 1.0)
        self.assertAlmostEqual(DRL_CONFIG["constraint_future_capacity_margin"], 0.70)
        self.assertTrue(bool(PROPULSION_CONTROLLER_CONFIG["enabled"]))
        self.assertAlmostEqual(REWARD_CONFIG["w_processing_opportunity_cost"], 0.0)
        self.assertGreater(REWARD_CONFIG["w_proc_far_window_penalty"], 0.0)
        self.assertGreater(REWARD_CONFIG["w_window_underuse_penalty"], 0.0)
        self.assertGreater(REWARD_CONFIG["w_prop_overburn_penalty"], 0.0)
        self.assertLess(REWARD_CONFIG["w_energy_penalty"], 0.0)
        self.assertAlmostEqual(REWARD_CONFIG["w_energy_over_budget_penalty"], 0.0)

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

        # constrained 阶段：Lyapunov 惩罚与自适应系数已启用，期望值随当前 config 更新。
        self.assertAlmostEqual(DRL_CONFIG["lyapunov_drift_clip"], 40.0)
        self.assertAlmostEqual(DRL_CONFIG["lyapunov_penalty_coeff"], 0.3)
        self.assertTrue(bool(DRL_CONFIG["adaptive_lyapunov_coeff_enable"]))
        self.assertAlmostEqual(DRL_CONFIG["adaptive_lyapunov_constraint_threshold"], 0.3)
        # target_pressure 与 threshold 已解绑（当前 config 分别取值）。
        self.assertAlmostEqual(
            DRL_CONFIG["adaptive_lyapunov_coeff_target_pressure"], 0.1)
        self.assertGreaterEqual(DRL_CONFIG["adaptive_lyapunov_coeff_min"], 0.20)
        self.assertLessEqual(DRL_CONFIG["adaptive_lyapunov_coeff_max"], 3.0)
        self.assertAlmostEqual(DRL_CONFIG["adaptive_lyapunov_constraint_norm"], 3.0)
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

        # low_value waste 现已是启用的 soft constraint 分量（coeff 非 0），与
        # over_processing 一并计入 soft_constraint_cost。
        self.assertGreater(cost.low_value_waste_cost, 0.0)
        self.assertGreater(cost.over_processing_cost, 0.0)
        self.assertEqual(cost.unproductive_cpu_cost, 0.0)
        self.assertAlmostEqual(
            cost.soft_constraint_cost,
            cost.positive_lyapunov_drift
            + cost.queue_soft_penalty
            + cost.thermal_cost
            + cost.energy_cost
            + cost.orbit_cost
            + cost.over_processing_cost
            + cost.low_value_waste_cost,
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
        from utils.action_space import decode_grouped_action, IDX_DROP_LOW
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
        self.assertGreater(action[IDX_DROP_LOW], 0.0)

    def test_llf_baseline_outputs_grouped_priority_action(self):
        from types import SimpleNamespace
        from baselines.value_baselines import LLFBaseline
        from utils.action_space import decode_grouped_action
        from config import DRL_CONFIG

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

        self.assertEqual(action.shape[0], int(DRL_CONFIG["action_dim"]))
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
        self.assertIn("raw_equivalent_delivery_coverage_mean", eval_source)
        self.assertIn("value_realization_ratio_mean", eval_source)
        self.assertIn("rf_product_proc_downlink_ratio_mean", eval_source)
        self.assertIn("high_value_delivery_rate", eval_source)
        self.assertIn("Ours + CPU throttle (deployment)", source)
        self.assertIn("Ours w/o Work-Conserving", source)
        self.assertIn("diagnostic_results", source)
        self.assertIn("include_deployment_ablations", source)
        self.assertIn("allow_missing_ours", source)

    def test_evaluate_optimized_reports_efficiency_metrics(self):
        import inspect
        import evaluate_optimized

        source = inspect.getsource(evaluate_optimized.evaluate_model)

        self.assertIn("useful_processing_ratio", source)
        self.assertIn("energy_violation_rate", source)
        self.assertIn("energy_per_value", source)
        self.assertIn("high_value_delivery_ratio", source)
        self.assertIn("comm_window_utilization", source)

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
            "processed_product_mb_mean": 9.0,
            "rf_downlinked_mb_mean": 3.0,
            "raw_equivalent_processed_mb_mean": 36.0,
            "raw_equivalent_delivered_mb_mean": 12.0,
            "raw_equivalent_delivery_coverage_mean": 0.3333333333333333,
            "rf_product_proc_downlink_ratio_mean": 3.0,
            "value_realization_ratio_mean": 0.75,
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
            "energy_violation_rate": 0.02,
            "energy_efficiency": 9.5,
            "energy_per_value": 0.105,
            "primary_goal_feasible": True,
            "primary_goal_violation": 0.0,
        })

        self.assertNotIn("Proc/DL Ratio", row)
        self.assertEqual(row["Processed MB"], 9.0)
        self.assertEqual(row["RF Downlinked MB"], 3.0)
        self.assertEqual(row["Raw-equivalent Processed MB"], 36.0)
        self.assertEqual(row["Raw-equivalent Delivered MB"], 12.0)
        self.assertAlmostEqual(row["Raw-equivalent Delivery Coverage"], 1.0 / 3.0)
        self.assertEqual(row["RF Product Proc/DL Ratio"], 3.0)
        self.assertEqual(row["Value Realization Ratio"], 0.75)
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
        self.assertEqual(row["Energy Violation Rate"], 0.02)
        self.assertEqual(row["Energy Efficiency"], 9.5)
        self.assertEqual(row["Energy per VoI"], 0.105)
        self.assertEqual(row["Primary Goal Feasible"], 1.0)
        self.assertEqual(row["Primary Goal Violation"], 0.0)

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

    def test_processing_value_accounting_uses_residual_voi_without_double_discount(self):
        """处理侧 useful 分母应按处理时可恢复 VoI 记账，但 processed 批次仍保留名义价值。"""
        from environment.task_value_model import TaskValueTracker, TaskBatch
        from config import TASK_CONFIG

        tracker = TaskValueTracker(TASK_CONFIG)
        batch = TaskBatch(
            mb=10.0,
            value=100.0,
            priority=1.0,
            quality=1.0,
            deadline_steps=100,
            created_step=0,
            scene_name="aged_linear",
            freshness_profile="linear",
        )
        tracker.raw_batches = [batch]

        now_step = 50
        expected_timeliness_weight = batch.timeliness_weight(
            now_step,
            floor=float(TASK_CONFIG["deadline_decay_floor"]),
            power=float(TASK_CONFIG["deadline_decay_power"]),
            overdue_grace_steps=int(TASK_CONFIG["overdue_grace_steps"]),
            overdue_decay_rate=float(TASK_CONFIG["overdue_decay_rate"]),
        )
        expected_voi_basis = 100.0 * expected_timeliness_weight
        result = tracker.process_by_priority(10.0, now_step=now_step)

        self.assertAlmostEqual(float(result["processed_value"]), 100.0, places=6)
        self.assertAlmostEqual(
            float(result["processed_voi_basis_value"]),
            expected_voi_basis,
            places=6,
        )
        self.assertAlmostEqual(
            float(result["processed_deliverable_value"]),
            expected_voi_basis,
            places=6,
        )
        self.assertAlmostEqual(float(tracker.total_processed_value), 100.0, places=6)
        self.assertAlmostEqual(
            float(tracker.total_processed_voi_basis_value),
            expected_voi_basis,
            places=6,
        )
        self.assertAlmostEqual(float(tracker.processed_batches[0].value), 100.0, places=6)

    def test_delivery_methods_use_same_timeliness_specificity_value_path(self):
        from copy import deepcopy

        from environment.task_value_model import TaskValueTracker, TaskBatch
        from config import TASK_CONFIG

        cfg = deepcopy(TASK_CONFIG)
        cfg["RAW_TO_PROCESSED_RATIO"] = 0.25
        cfg["specificity_gamma"] = 1.0
        cfg["specificity_scale_mb"] = 100.0

        def make_tracker():
            tracker = TaskValueTracker(cfg)
            tracker.processed_batches = [
                TaskBatch(
                    mb=10.0,
                    value=100.0,
                    priority=1.0,
                    quality=1.0,
                    deadline_steps=100,
                    created_step=0,
                    nominal_class_id=0,
                    raw_equivalent_mb=40.0,
                )
            ]
            return tracker

        now_step = 0
        expected_specificity = 1.0 / (1.0 + 40.0 / 100.0)
        expected_value = 100.0 * expected_specificity

        plain = make_tracker().deliver(10.0, now_step=now_step)
        by_priority = make_tracker().deliver_by_priority(10.0, now_step=now_step)
        by_class = make_tracker().deliver_by_class([10.0, 0.0, 0.0], now_step=now_step)

        for result in (plain, by_priority, by_class):
            self.assertAlmostEqual(float(result["delivered_value"]), expected_value, places=6)
            self.assertAlmostEqual(float(result["rf_downlinked_mb"]), 10.0, places=6)
            self.assertAlmostEqual(float(result["raw_equivalent_delivered_mb"]), 40.0, places=6)

    def test_specificity_uses_processed_raw_equivalent_mb(self):
        from copy import deepcopy

        from environment.task_value_model import TaskValueTracker, TaskBatch
        from config import TASK_CONFIG

        cfg = deepcopy(TASK_CONFIG)
        cfg["RAW_TO_PROCESSED_RATIO"] = 0.25
        cfg["specificity_gamma"] = 1.0
        cfg["specificity_scale_mb"] = 10.0
        tracker = TaskValueTracker(cfg)
        tracker.processed_batches = [
            TaskBatch(
                mb=2.5,
                value=10.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=100,
                created_step=0,
                nominal_class_id=0,
                raw_equivalent_mb=10.0,
            )
        ]

        self.assertAlmostEqual(
            float(tracker._specificity_discount(0, now_step=0)),
            0.5,
            places=6,
        )

    def test_summary_reports_product_rf_and_raw_equivalent_mb(self):
        from copy import deepcopy

        from environment.task_value_model import TaskValueTracker, TaskBatch
        from config import TASK_CONFIG

        cfg = deepcopy(TASK_CONFIG)
        cfg["RAW_TO_PROCESSED_RATIO"] = 0.25
        cfg["specificity_gamma"] = 0.0
        tracker = TaskValueTracker(cfg)
        tracker.raw_batches = [
            TaskBatch(
                mb=10.0,
                value=100.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=100,
                created_step=0,
                nominal_class_id=0,
                raw_equivalent_mb=10.0,
            )
        ]

        process_result = tracker.process_by_priority(10.0, now_step=0)
        deliver_result = tracker.deliver(2.5, now_step=0)
        summary = tracker.summary()

        self.assertAlmostEqual(float(process_result["processed_product_mb"]), 2.5, places=6)
        self.assertAlmostEqual(float(process_result["raw_equivalent_processed_mb"]), 10.0, places=6)
        self.assertAlmostEqual(float(deliver_result["rf_downlinked_mb"]), 2.5, places=6)
        self.assertAlmostEqual(float(deliver_result["raw_equivalent_delivered_mb"]), 10.0, places=6)
        self.assertAlmostEqual(float(summary["processed_product_mb"]), 2.5, places=6)
        self.assertAlmostEqual(float(summary["rf_downlinked_mb"]), 2.5, places=6)
        self.assertAlmostEqual(float(summary["raw_equivalent_processed_mb"]), 10.0, places=6)
        self.assertAlmostEqual(float(summary["raw_equivalent_delivered_mb"]), 10.0, places=6)

    def test_raw_to_processed_ratio_is_clamped_to_nonzero_compression_range(self):
        from copy import deepcopy

        from environment.task_value_model import TaskValueTracker
        from config import TASK_CONFIG

        low_cfg = deepcopy(TASK_CONFIG)
        low_cfg["RAW_TO_PROCESSED_RATIO"] = 0.0
        high_cfg = deepcopy(TASK_CONFIG)
        high_cfg["RAW_TO_PROCESSED_RATIO"] = 2.0

        self.assertAlmostEqual(TaskValueTracker(low_cfg)._raw_to_processed_ratio(), 0.05, places=6)
        self.assertAlmostEqual(TaskValueTracker(high_cfg)._raw_to_processed_ratio(), 1.0, places=6)

    def test_add_arrival_freezes_raw_nominal_class_id(self):
        from copy import deepcopy

        import numpy as np

        from environment.task_value_model import TaskValueTracker
        from config import TASK_CONFIG

        cfg = deepcopy(TASK_CONFIG)
        cfg["base_value_per_mb"] = 1.0
        tracker = TaskValueTracker(cfg)
        info = tracker.add_arrival(
            10.0,
            np.random.default_rng(123),
            now_step=0,
            scene_context={
                "scene_name": "fixed_medium",
                "profile": {
                    "priority_range": (1.5, 1.5),
                    "quality_range": (1.0, 1.0),
                    "deadline_range_steps": (100, 100),
                    "cloud_cover_range": (0.0, 0.0),
                },
            },
        )

        self.assertEqual(int(info["generated_nominal_class_id"]), 1)
        self.assertEqual(tracker.raw_batches[0].nominal_class_id, 1)
        self.assertAlmostEqual(float(tracker.raw_batches[0].raw_equivalent_mb), 10.0, places=6)

    def test_processing_compresses_raw_mb_but_retains_configured_value(self):
        """CPU 消耗 raw MB；processed queue 只增加压缩后的 MB，价值按 retention 保留。"""
        from copy import deepcopy

        from environment.task_value_model import TaskValueTracker, TaskBatch
        from config import TASK_CONFIG

        cfg = deepcopy(TASK_CONFIG)
        cfg["RAW_TO_PROCESSED_RATIO"] = 0.25
        cfg["PROCESSING_VALUE_RETENTION"] = 0.8
        tracker = TaskValueTracker(cfg)
        tracker.raw_batches = [
            TaskBatch(
                mb=10.0,
                value=100.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=100,
                created_step=0,
                scene_name="compressible",
            )
        ]

        result = tracker.process_by_priority(10.0, now_step=0)

        self.assertAlmostEqual(float(result["raw_processed_mb"]), 10.0, places=6)
        self.assertAlmostEqual(float(result["processed_output_mb"]), 2.5, places=6)
        self.assertAlmostEqual(float(result["processed_mb"]), 2.5, places=6)
        self.assertAlmostEqual(float(result["processed_value"]), 80.0, places=6)
        self.assertAlmostEqual(float(result["compression_ratio"]), 0.25, places=6)
        self.assertAlmostEqual(float(result["value_retention"]), 0.8, places=6)
        self.assertAlmostEqual(float(tracker.raw_mb), 0.0, places=6)
        self.assertAlmostEqual(float(tracker.processed_mb), 2.5, places=6)
        self.assertAlmostEqual(float(tracker.processed_batches[0].value), 80.0, places=6)

    def test_processed_batches_inherit_raw_nominal_class_after_compression(self):
        from copy import deepcopy

        from environment.task_value_model import TaskValueTracker, TaskBatch
        from config import TASK_CONFIG

        cfg = deepcopy(TASK_CONFIG)
        cfg["RAW_TO_PROCESSED_RATIO"] = 0.25
        cfg["PROCESSING_VALUE_RETENTION"] = 1.0
        tracker = TaskValueTracker(cfg)
        tracker.raw_batches = [
            TaskBatch(
                mb=10.0,
                value=40.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=100,
                created_step=0,
                scene_name="raw_high",
            ),
            TaskBatch(
                mb=10.0,
                value=15.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=100,
                created_step=0,
                scene_name="raw_medium",
            ),
            TaskBatch(
                mb=10.0,
                value=5.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=100,
                created_step=0,
                scene_name="raw_low",
            ),
        ]

        result = tracker.process_by_priority(30.0, now_step=0)
        expected_processed_mb = 10.0 * float(cfg["RAW_TO_PROCESSED_RATIO"])

        self.assertAlmostEqual(float(result["raw_processed_mb"]), 30.0, places=6)
        self.assertAlmostEqual(float(result["processed_mb"]), 7.5, places=6)

        expected_classes = {
            "raw_high": 0,
            "raw_medium": 1,
            "raw_low": 2,
        }
        processed_by_scene = {batch.scene_name: batch for batch in tracker.processed_batches}
        self.assertEqual(set(processed_by_scene), set(expected_classes))
        for scene_name, expected_class_id in expected_classes.items():
            batch = processed_by_scene[scene_name]
            self.assertEqual(batch.nominal_class_id, expected_class_id)
            self.assertEqual(tracker.task_nominal_class_id(batch), expected_class_id)
            self.assertEqual(tracker.task_class_id(batch, now_step=0), expected_class_id)

        stats = tracker.class_stats(now_step=0)
        self.assertAlmostEqual(float(stats["processed_high_mb"]), expected_processed_mb, places=6)
        self.assertAlmostEqual(float(stats["processed_medium_mb"]), expected_processed_mb, places=6)
        self.assertAlmostEqual(float(stats["processed_low_mb"]), expected_processed_mb, places=6)

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

    def test_nominal_high_keeps_high_scheduling_priority_after_residual_decay(self):
        """指标按名义 high 统计时，调度保护也应继续把它当 high 处理。"""
        from copy import deepcopy

        from config import HARD_RULES_CONFIG, TASK_CONFIG
        from environment.task_value_model import TaskValueTracker, TaskBatch

        cfg = deepcopy(TASK_CONFIG)
        tracker = TaskValueTracker(cfg)
        old_hard_cfg = deepcopy(HARD_RULES_CONFIG)
        try:
            HARD_RULES_CONFIG["enable_class_priority_floor"] = True
            tracker.raw_batches = [
                TaskBatch(
                    mb=10.0,
                    value=25.0,  # nominal density=2.5 -> high, residual at now=8 -> low
                    priority=2.0,
                    quality=1.0,
                    deadline_steps=10,
                    created_step=0,
                    scene_name="stale_nominal_high",
                ),
                TaskBatch(
                    mb=10.0,
                    value=15.0,
                    priority=0.8,
                    quality=1.0,
                    deadline_steps=100,
                    created_step=8,
                    scene_name="fresh_medium",
                ),
            ]

            result = tracker.process_by_priority(10.0, now_step=8)
            expected_processed_mb = 10.0 * float(TASK_CONFIG["RAW_TO_PROCESSED_RATIO"])
        finally:
            HARD_RULES_CONFIG.clear()
            HARD_RULES_CONFIG.update(old_hard_cfg)

        self.assertEqual(tracker.processed_batches[0].scene_name, "stale_nominal_high")
        self.assertAlmostEqual(float(result["raw_processed_high_mb"]), 10.0, places=6)
        self.assertAlmostEqual(float(result["processed_high_mb"]), expected_processed_mb, places=6)

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
        expected_processed_mb = 10.0 * float(TASK_CONFIG["RAW_TO_PROCESSED_RATIO"])
        self.assertAlmostEqual(float(result["raw_processed_mb"]), 10.0, places=6)
        self.assertAlmostEqual(float(result["processed_mb"]), expected_processed_mb, places=6)
        self.assertAlmostEqual(float(result["processed_medium_mb"]), expected_processed_mb, places=6)

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

    def test_future_contact_cpu_gate_is_configurable_but_not_default_takeover(self):
        from config import TASK_CONFIG

        self.assertFalse(bool(TASK_CONFIG.get("enable_future_contact_cpu_gate", True)))
        self.assertFalse(bool(TASK_CONFIG.get("enable_cpu_throttle", True)))
        self.assertFalse(bool(TASK_CONFIG.get("cpu_action_is_admissible_budget", True)))
        self.assertGreater(float(TASK_CONFIG.get("cpu_gate_near_term_passes", 0.0)), 0.0)
        self.assertLessEqual(float(TASK_CONFIG.get("cpu_gate_near_term_passes", 999.0)), 0.5)

    def test_future_contact_cpu_gate_closes_power_and_processing(self):
        from copy import deepcopy

        from config import ACTUATOR_GATE_CONFIG, TASK_CONFIG
        from environment.satellite_env import VLEOSatelliteEnv
        from environment.task_value_model import TaskBatch

        old_actuator_cfg = deepcopy(ACTUATOR_GATE_CONFIG)
        old_task_cfg = deepcopy(TASK_CONFIG)
        try:
            ACTUATOR_GATE_CONFIG["cpu_gate_soft_mode"] = False
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
            env._future_contact_capacity_until_step = lambda *_args, **_kwargs: 10000.0
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
            env_no_gate._future_contact_capacity_until_step = lambda *_args, **_kwargs: 10000.0
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
            ACTUATOR_GATE_CONFIG.clear()
            ACTUATOR_GATE_CONFIG.update(old_actuator_cfg)
            TASK_CONFIG.clear()
            TASK_CONFIG.update(old_task_cfg)

        self.assertTrue(bool(gated_info["future_contact_cpu_gate_applied"]))
        self.assertAlmostEqual(float(gated_info["cpu_gate_alpha_cpu_before"]), 1.0, places=6)
        self.assertAlmostEqual(float(gated_info["executed_action"][1]), 0.0, places=6)
        self.assertLess(float(gated_info["P_cpu_w"]), float(ungated_info["P_cpu_w"]))
        self.assertLess(float(gated_info["P_total_w"]), float(ungated_info["P_total_w"]))
        self.assertLess(float(gated_info["service_rate_mbs"]), float(ungated_info["service_rate_mbs"]))
        self.assertLess(float(gated_info["processed_mb"]), float(ungated_info["processed_mb"]))

    def test_far_window_cpu_gate_uses_tighter_buffer_target(self):
        """远离通信窗口时只预处理小缓冲，避免 processed 队列长期等待造成 VoI 折损。"""
        from copy import deepcopy

        from config import ACTUATOR_GATE_CONFIG, TASK_CONFIG
        from environment.satellite_env import VLEOSatelliteEnv

        old_actuator_cfg = deepcopy(ACTUATOR_GATE_CONFIG)
        old_task_cfg = deepcopy(TASK_CONFIG)
        try:
            ACTUATOR_GATE_CONFIG["cpu_gate_soft_mode"] = False
            TASK_CONFIG.update({
                "cpu_action_is_admissible_budget": True,
                "enable_future_contact_cpu_gate": True,
                "cpu_gate_start_future_ratio": 0.55,
                "cpu_gate_target_future_ratio": 0.75,
                "cpu_gate_far_window_target_ratio": 0.25,
                "cpu_gate_hard_stop_future_ratio": 0.90,
                "cpu_gate_far_window_lead_s": 120.0,
                "cpu_gate_floor_alpha": 0.0,
                "deliverability_capacity_margin": 0.95,
            })
            env = VLEOSatelliteEnv(seed=38)
            env.reset()
            env._future_contact_capacity_mb = lambda: 100.0
            env.comm_queue.value = 20.0
            env.data_queue.length = 100.0

            _, meta = env._apply_future_contact_cpu_gate(
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                in_window=False,
                time_to_next_window_s=600.0,
                dt_s=float(env.dt),
            )
        finally:
            ACTUATOR_GATE_CONFIG.clear()
            ACTUATOR_GATE_CONFIG.update(old_actuator_cfg)
            TASK_CONFIG.clear()
            TASK_CONFIG.update(old_task_cfg)

        self.assertTrue(bool(meta["future_contact_cpu_gate_applied"]))
        self.assertAlmostEqual(float(meta["cpu_gate_allowed_processed_mb"]), 5.0, places=6)
        self.assertAlmostEqual(float(meta["cpu_gate_effective_processed_budget_mb"]), 5.0, places=6)
        self.assertLess(float(meta["cpu_gate_ratio_after_est"]), 0.26)

    def test_admissible_cpu_budget_uses_configured_near_term_passes(self):
        """CPU gate should not pre-process more than the configured near-term pass buffer."""
        from copy import deepcopy

        from config import GROUND_STATION_CONFIG, TASK_CONFIG
        from environment.satellite_env import VLEOSatelliteEnv

        old_task_cfg = deepcopy(TASK_CONFIG)
        old_ground_cfg = deepcopy(GROUND_STATION_CONFIG)
        try:
            GROUND_STATION_CONFIG["max_downlink_mb_per_pass"] = 800.0
            TASK_CONFIG.update({
                "deliverability_capacity_margin": 0.95,
                "cpu_gate_near_term_passes": 1.0,
            })
            env = VLEOSatelliteEnv(seed=46)
            env.reset()
            env._future_contact_capacity_mb = lambda: 10_000.0
            env.comm_queue.value = 700.0

            budget = env._admissible_cpu_budget_mb()
        finally:
            TASK_CONFIG.clear()
            TASK_CONFIG.update(old_task_cfg)
            GROUND_STATION_CONFIG.clear()
            GROUND_STATION_CONFIG.update(old_ground_cfg)

        self.assertAlmostEqual(float(budget["admissible_cpu_mb"]), 60.0, places=6)
        self.assertAlmostEqual(float(budget["effective_future_capacity_mb"]), 800.0, places=6)

    def test_cpu_gate_allows_deliverable_raw_high_escape_when_buffer_full(self):
        """总 processed 缓冲满时，可赶上下个窗口的 raw high 仍应获得 CPU 逃逸额度。"""
        from copy import deepcopy

        from config import GROUND_STATION_CONFIG, HARD_RULES_CONFIG, TASK_CONFIG
        from environment.satellite_env import VLEOSatelliteEnv
        from environment.task_value_model import TaskBatch

        old_task_cfg = deepcopy(TASK_CONFIG)
        old_ground_cfg = deepcopy(GROUND_STATION_CONFIG)
        old_hard_cfg = deepcopy(HARD_RULES_CONFIG)
        try:
            GROUND_STATION_CONFIG["max_downlink_mb_per_pass"] = 800.0
            HARD_RULES_CONFIG.update({
                "enable_tx_high_reserve": True,
                "tx_high_reserve_fraction": 0.70,
            })
            TASK_CONFIG.update({
                "cpu_action_is_admissible_budget": True,
                "enable_future_contact_cpu_gate": True,
                "cpu_gate_near_term_passes": 0.5,
                "cpu_gate_start_future_ratio": 0.55,
                "cpu_gate_target_future_ratio": 0.75,
                "cpu_gate_far_window_target_ratio": 0.25,
                "cpu_gate_hard_stop_future_ratio": 0.90,
                "cpu_gate_far_window_lead_s": 120.0,
                "deliverability_capacity_margin": 0.95,
                "enable_high_value_cpu_gate_escape": True,
                "high_value_cpu_escape_min_raw_mb": 1.0,
                "high_value_cpu_escape_min_deliverable_ratio": 0.50,
                "high_value_cpu_escape_max_mismatch": 0.40,
                "high_value_cpu_escape_capacity_margin": 0.95,
            })
            env = VLEOSatelliteEnv(seed=48)
            env.reset()
            env._future_contact_capacity_mb = lambda: 800.0
            env.comm_queue.value = 400.0
            env.data_queue.length = 80.0
            env.task_tracker.processed_batches.append(TaskBatch(
                mb=400.0,
                value=100.0,
                priority=0.1,
                quality=0.5,
                deadline_steps=500,
                created_step=env.step_count,
            ))
            env.task_tracker.raw_batches.append(TaskBatch(
                mb=80.0,
                value=800.0,
                priority=2.0,
                quality=1.0,
                deadline_steps=100,
                created_step=env.step_count,
            ))

            _, meta = env._apply_future_contact_cpu_gate(
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                in_window=False,
                time_to_next_window_s=300.0,
                dt_s=float(env.dt),
            )
        finally:
            TASK_CONFIG.clear()
            TASK_CONFIG.update(old_task_cfg)
            GROUND_STATION_CONFIG.clear()
            GROUND_STATION_CONFIG.update(old_ground_cfg)
            HARD_RULES_CONFIG.clear()
            HARD_RULES_CONFIG.update(old_hard_cfg)

        self.assertTrue(bool(meta["cpu_gate_high_value_escape_applied"]))
        self.assertGreater(float(meta["cpu_gate_allowed_processed_mb"]), 0.0)
        self.assertGreater(float(meta["cpu_gate_effective_processed_budget_mb"]), 0.0)
        self.assertAlmostEqual(float(meta["cpu_gate_high_value_escape_budget_mb"]), 80.0, places=6)

    def test_in_window_cpu_feed_floor_processes_raw_for_same_step_tx(self):
        """When a contact window is open, raw backlog should be processed and transmitted."""
        from copy import deepcopy

        from config import ACTUATOR_GATE_CONFIG, TASK_CONFIG
        from environment.satellite_env import VLEOSatelliteEnv
        from environment.task_value_model import TaskBatch
        from utils.action_space import IDX_POINTING, POINTING_DOWNLINK, pointing_unit_for_mode

        old_actuator_cfg = deepcopy(ACTUATOR_GATE_CONFIG)
        old_task_cfg = deepcopy(TASK_CONFIG)
        try:
            ACTUATOR_GATE_CONFIG["cpu_gate_soft_mode"] = False
            TASK_CONFIG.update({
                "cpu_action_is_admissible_budget": True,
                "enable_future_contact_cpu_gate": True,
                "cpu_gate_near_term_passes": 0.5,
                "deliverability_capacity_margin": 0.95,
                "enable_in_window_cpu_feed_floor": True,
                "in_window_cpu_feed_alpha_floor": 1.0,
                "in_window_cpu_feed_min_raw_mb": 1.0,
            })
            env = VLEOSatelliteEnv(seed=47)
            env.reset()
            env._data_arrival_scale = 0.0
            env._contact = {
                "in_window": True,
                "time_to_next_window_s": 0.0,
                "max_capacity_mbps": 8000.0,
            }
            env._contact_override = dict(env._contact)
            env._future_contact_capacity_mb = lambda: 800.0
            env._future_contact_capacity_until_step = lambda *_args, **_kwargs: 800.0
            env.comm_queue.value = 0.0
            env.data_queue.length = 100.0
            env.task_tracker.raw_batches.append(TaskBatch(
                mb=100.0,
                value=500.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=500,
                created_step=env.step_count,
            ))

            action = np.zeros(env.action_dim, dtype=np.float32)
            action[:8] = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            action[IDX_POINTING] = pointing_unit_for_mode(POINTING_DOWNLINK)

            _, _, _, info = env.step(action, enforce_prop_smoothing=False)
        finally:
            ACTUATOR_GATE_CONFIG.clear()
            ACTUATOR_GATE_CONFIG.update(old_actuator_cfg)
            TASK_CONFIG.clear()
            TASK_CONFIG.update(old_task_cfg)

        self.assertTrue(bool(info.get("in_window_cpu_feed_floor_applied", False)))
        self.assertGreater(float(info["processed_mb"]), 0.0)
        self.assertGreater(float(info["delivered_mb"]), 0.0)

    def test_zero_admissible_cpu_budget_closes_cpu_power(self):
        """可交付处理额度为 0 时，alpha_cpu=1 也不能白烧 CPU 电量。"""
        from copy import deepcopy

        from config import ACTUATOR_GATE_CONFIG, TASK_CONFIG
        from environment.satellite_env import VLEOSatelliteEnv
        from environment.task_value_model import TaskBatch

        old_actuator_cfg = deepcopy(ACTUATOR_GATE_CONFIG)
        old_task_cfg = deepcopy(TASK_CONFIG)
        try:
            ACTUATOR_GATE_CONFIG["cpu_gate_soft_mode"] = False
            TASK_CONFIG.update({
                "cpu_action_is_admissible_budget": True,
                "enable_future_contact_cpu_gate": True,
                "enable_cpu_throttle": True,
                "cpu_gate_floor_alpha": 0.0,
            })
            env = VLEOSatelliteEnv(seed=43)
            env.reset()
            env._data_arrival_scale = 0.0
            env._contact = {"in_window": False, "time_to_next_window_s": 3600.0}
            env._contact_override = {"in_window": False, "time_to_next_window_s": 3600.0}
            env._future_contact_capacity_mb = lambda: 0.0
            env._future_contact_capacity_until_step = lambda *_args, **_kwargs: 0.0
            env.data_queue.length = 50.0
            env.task_tracker.raw_batches.append(TaskBatch(
                mb=50.0,
                value=500.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=500,
                created_step=env.step_count,
            ))

            _, _, _, info = env.step(
                np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                enforce_prop_smoothing=False,
            )
        finally:
            ACTUATOR_GATE_CONFIG.clear()
            ACTUATOR_GATE_CONFIG.update(old_actuator_cfg)
            TASK_CONFIG.clear()
            TASK_CONFIG.update(old_task_cfg)

        self.assertTrue(bool(info["future_contact_cpu_gate_applied"]))
        self.assertAlmostEqual(float(info["cpu_gate_admissible_cpu_mb"]), 0.0, places=6)
        self.assertAlmostEqual(float(info["executed_action"][1]), 0.0, places=6)
        self.assertAlmostEqual(float(info["P_cpu_w"]), 0.0, places=6)
        self.assertAlmostEqual(float(info["processed_mb"]), 0.0, places=6)

    def test_small_admissible_cpu_budget_scales_cpu_power_to_work(self):
        """admissible 很小时，CPU 功率应按实际可处理 MB 缩放，而不是满功率空烧。"""
        from copy import deepcopy

        from config import ACTUATOR_GATE_CONFIG, QUEUE_CONFIG, TASK_CONFIG
        from environment.satellite_env import VLEOSatelliteEnv
        from environment.task_value_model import TaskBatch

        old_actuator_cfg = deepcopy(ACTUATOR_GATE_CONFIG)
        old_task_cfg = deepcopy(TASK_CONFIG)
        try:
            ACTUATOR_GATE_CONFIG["cpu_gate_soft_mode"] = False
            TASK_CONFIG.update({
                "cpu_action_is_admissible_budget": True,
                "enable_future_contact_cpu_gate": True,
                "enable_cpu_throttle": True,
                "cpu_gate_floor_alpha": 0.0,
                "deliverability_capacity_margin": 0.95,
            })
            env = VLEOSatelliteEnv(seed=44)
            env.reset()
            env._data_arrival_scale = 0.0
            env._contact = {"in_window": True, "time_to_next_window_s": 0.0}
            env._contact_override = {"in_window": True, "time_to_next_window_s": 0.0}
            env._future_contact_capacity_mb = lambda: 5.0
            env._future_contact_capacity_until_step = lambda *_args, **_kwargs: 5.0
            env.comm_queue.value = 0.0
            env.data_queue.length = 50.0
            env.task_tracker.raw_batches.append(TaskBatch(
                mb=50.0,
                value=500.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=500,
                created_step=env.step_count,
            ))

            _, _, _, info = env.step(
                np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                enforce_prop_smoothing=False,
            )
        finally:
            ACTUATOR_GATE_CONFIG.clear()
            ACTUATOR_GATE_CONFIG.update(old_actuator_cfg)
            TASK_CONFIG.clear()
            TASK_CONFIG.update(old_task_cfg)

        max_cpu_mb = float(QUEUE_CONFIG["data_service_rate_max_mbs"]) * float(env.dt)
        expected_admissible_mb = 0.95 * 5.0
        expected_alpha_cpu_power = expected_admissible_mb / max_cpu_mb

        self.assertAlmostEqual(
            float(info["cpu_gate_admissible_cpu_mb"]),
            expected_admissible_mb,
            places=6,
        )
        self.assertAlmostEqual(
            float(info["executed_action"][1]),
            expected_alpha_cpu_power,
            places=6,
        )
        self.assertLess(float(info["P_cpu_w"]), 2.0)
        self.assertLessEqual(float(info["processed_mb"]), expected_admissible_mb + 1e-6)
        self.assertAlmostEqual(float(info["cpu_capacity_mb"]), expected_admissible_mb, places=5)

    def test_cpu_gate_soft_mode_does_not_hide_hard_processing_clip(self):
        """soft mode 只暴露 violation，不应通过 effective budget 暗中裁剪处理量。"""
        from copy import deepcopy

        from config import ACTUATOR_GATE_CONFIG, TASK_CONFIG
        from environment.satellite_env import VLEOSatelliteEnv

        old_task_cfg = deepcopy(TASK_CONFIG)
        old_gate_cfg = deepcopy(ACTUATOR_GATE_CONFIG)
        try:
            TASK_CONFIG.update({
                "cpu_action_is_admissible_budget": True,
                "enable_future_contact_cpu_gate": True,
                "enable_cpu_throttle": True,
                "deliverability_capacity_margin": 0.95,
            })
            ACTUATOR_GATE_CONFIG["cpu_gate_soft_mode"] = True
            env = VLEOSatelliteEnv(seed=45)
            env.reset()
            env._future_contact_capacity_mb = lambda: 100.0
            env.comm_queue.value = 90.0
            env.data_queue.length = 50.0

            gated, meta = env._apply_future_contact_cpu_gate(
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                in_window=True,
                time_to_next_window_s=0.0,
                dt_s=float(env.dt),
            )
        finally:
            TASK_CONFIG.clear()
            TASK_CONFIG.update(old_task_cfg)
            ACTUATOR_GATE_CONFIG.clear()
            ACTUATOR_GATE_CONFIG.update(old_gate_cfg)

        self.assertTrue(bool(meta["cpu_gate_soft_mode"]))
        self.assertFalse(bool(meta["future_contact_cpu_gate_applied"]))
        self.assertAlmostEqual(float(gated[1]), 1.0, places=6)
        self.assertGreater(float(meta["cpu_gate_violation_mb"]), 0.0)
        self.assertAlmostEqual(
            float(meta["cpu_gate_effective_processed_budget_mb"]),
            float(meta["cpu_gate_requested_processed_mb"]),
            places=6,
        )

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
            "processed_product_mb", "raw_equivalent_processed_mb",
            "expired_raw_value", "expired_processed_value",
            "dropped_raw_value", "dropped_processed_value",
            "future_contact_capacity_mb", "processed_queue_future_contact_ratio",
            "processed_since_contact_mb", "delivered_since_contact_mb",
            "episode_processed_mb", "episode_processed_value",
            "episode_processed_product_mb", "episode_raw_equivalent_processed_mb",
            "episode_delivered_mb", "episode_delivered_value", "episode_proc_dl_ratio",
            "episode_rf_downlinked_mb", "episode_raw_equivalent_delivered_mb",
            "rf_downlinked_mb", "raw_equivalent_delivered_mb",
            "episode_useful_processing_ratio", "useful_processing_ratio",
            "cpu_active_far_from_window_rate",
            "costs",
        ]:
            self.assertIn(key, info)
        self.assertIn("state_safety_cost", info["costs"])

    def test_mission_pointing_fallback_images_when_daylit_and_idle(self):
        """昼侧、非窗口、无 raw backlog 时，安全状态下应兜底指向成像，避免任务链路断流。"""
        from copy import deepcopy

        from config import HARD_RULES_CONFIG
        from environment.satellite_env import VLEOSatelliteEnv
        from utils.action_space import POINTING_IMAGE, pointing_unit_for_mode, IDX_POINTING

        old_hard_cfg = deepcopy(HARD_RULES_CONFIG)
        try:
            HARD_RULES_CONFIG.update({
                "enable_mission_pointing_fallback": True,
                "mission_pointing_raw_low_mb": 1.0,
            })
            env = VLEOSatelliteEnv(seed=73)
            env.reset()
            env._data_arrival_scale = 1.0
            env._contact = {"in_window": False, "time_to_next_window_s": 1800.0}
            env._contact_override = dict(env._contact)
            env.orbit_sim.reset_phase(0.5 * env.orbit_sim._sunlit_phase)
            env.data_queue.length = 0.0
            env.comm_queue.value = 0.0
            env.battery.soc = 0.80
            env.altitude_m = env._h_warning + 40e3

            action = np.zeros(env.action_dim, dtype=np.float32)
            action[IDX_POINTING] = pointing_unit_for_mode(2)
            _, _, _, info = env.step(action, enforce_prop_smoothing=False)
        finally:
            HARD_RULES_CONFIG.clear()
            HARD_RULES_CONFIG.update(old_hard_cfg)

        self.assertTrue(bool(info["mission_pointing_fallback_applied"]))
        self.assertEqual(int(info["pointing_mode"]), POINTING_IMAGE)
        self.assertGreater(float(info["data_arrival_mb"]), 0.0)

    def test_mission_pointing_fallback_downlinks_in_contact(self):
        """窗口内有 processed backlog 时，任务兜底应指向下传而不是继续对日。"""
        from copy import deepcopy

        from config import HARD_RULES_CONFIG
        from environment.satellite_env import VLEOSatelliteEnv
        from environment.task_value_model import TaskBatch
        from utils.action_space import POINTING_DOWNLINK, pointing_unit_for_mode, IDX_POINTING

        old_hard_cfg = deepcopy(HARD_RULES_CONFIG)
        try:
            HARD_RULES_CONFIG.update({
                "enable_mission_pointing_fallback": True,
                "in_window_floor_min_queue_mb": 1.0,
            })
            env = VLEOSatelliteEnv(seed=74)
            env.reset()
            env._data_arrival_scale = 0.0
            env._contact = {
                "in_window": True,
                "time_to_next_window_s": 0.0,
                "max_capacity_mbps": 8000.0,
            }
            env._contact_override = dict(env._contact)
            env.comm_queue.value = 20.0
            env.task_tracker.processed_batches.append(TaskBatch(
                mb=20.0,
                value=80.0,
                priority=1.0,
                quality=1.0,
                deadline_steps=500,
                created_step=env.step_count,
            ))
            env.battery.soc = 0.80
            env.altitude_m = env._h_warning + 40e3

            action = np.zeros(env.action_dim, dtype=np.float32)
            action[2] = 0.05
            action[IDX_POINTING] = pointing_unit_for_mode(2)
            _, _, _, info = env.step(action, enforce_prop_smoothing=False)
        finally:
            HARD_RULES_CONFIG.clear()
            HARD_RULES_CONFIG.update(old_hard_cfg)

        self.assertTrue(bool(info["mission_pointing_fallback_applied"]))
        self.assertEqual(int(info["pointing_mode"]), POINTING_DOWNLINK)
        self.assertGreater(float(info["delivered_mb"]), 0.0)

    def test_mission_pointing_fallback_respects_low_soc_guard(self):
        """SOC 贴近安全线时，任务兜底不能把对日保电改成成像/下传。"""
        from copy import deepcopy

        from config import ENERGY_CONFIG, HARD_RULES_CONFIG
        from environment.satellite_env import VLEOSatelliteEnv
        from utils.action_space import POINTING_SUN, pointing_unit_for_mode, IDX_POINTING

        old_hard_cfg = deepcopy(HARD_RULES_CONFIG)
        try:
            HARD_RULES_CONFIG.update({
                "enable_mission_pointing_fallback": True,
                "mission_pointing_raw_low_mb": 1.0,
            })
            env = VLEOSatelliteEnv(seed=75)
            env.reset()
            env._data_arrival_scale = 1.0
            env._contact = {"in_window": False, "time_to_next_window_s": 1800.0}
            env._contact_override = dict(env._contact)
            env.orbit_sim.reset_phase(0.5 * env.orbit_sim._sunlit_phase)
            env.data_queue.length = 0.0
            env.battery.soc = (
                float(ENERGY_CONFIG["battery_min_soc"])
                + 0.5 * float(ENERGY_CONFIG.get("battery_operational_reserve_soc", 0.0))
            )
            env.altitude_m = env._h_warning + 40e3

            action = np.zeros(env.action_dim, dtype=np.float32)
            action[IDX_POINTING] = pointing_unit_for_mode(POINTING_SUN)
            _, _, _, info = env.step(action, enforce_prop_smoothing=False)
        finally:
            HARD_RULES_CONFIG.clear()
            HARD_RULES_CONFIG.update(old_hard_cfg)

        self.assertFalse(bool(info["mission_pointing_fallback_applied"]))
        self.assertEqual(int(info["pointing_mode"]), POINTING_SUN)

    def test_training_log_helper_reports_execution_diagnostics(self):
        """训练日志必须记录诊断这次塌缩所需的姿态、功率和 CPU gate 字段。"""
        import train

        info = {
            "pointing_mode": 2,
            "attitude_slew": 1.0,
            "attitude_desat": 0.0,
            "P_cpu_w": 3.5,
            "P_tx_w": 7.5,
            "P_propulsion_w": 120.0,
            "alpha_cpu": 0.25,
            "alpha_tx": 0.50,
            "cpu_capacity_mb": 4.0,
            "cpu_gate_admissible_cpu_mb": 6.0,
        }

        diagnostics = train._extract_training_execution_diagnostics(info)

        for key in train.TRAIN_LOG_DIAGNOSTIC_FIELDS:
            self.assertIn(key, diagnostics)
        self.assertEqual(diagnostics["pointing_mode"], 2)
        self.assertAlmostEqual(diagnostics["P_cpu_w"], 3.5)

    def test_metric_logger_summary_keeps_mean_and_std(self):
        """summary.json 应同时保留均值和标准差，便于复核 TD/reward 尺度。"""
        import json
        import os
        import tempfile

        from utils.metric_logger import MetricLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricLogger(tmpdir)
            logger.log_step(1, {"target_q_mean": 1.0})
            logger.log_step(2, {"target_q_mean": 3.0})
            logger.save()

            with open(os.path.join(tmpdir, "summary.json"), "r", encoding="utf-8") as f:
                summary = json.load(f)

        self.assertAlmostEqual(summary["target_q_mean"], 2.0)
        self.assertAlmostEqual(summary["target_q_mean_std"], 1.0)

    def test_objective_summary_matches_current_schema(self):
        """训练报告元数据必须同步当前观测/动作 schema，避免论文表述沿用旧维度。"""
        import train
        from config import DRL_CONFIG

        summary = train._objective_summary()

        self.assertIn(f"{int(DRL_CONFIG['state_dim'])}-D", summary["observation_schema"])
        self.assertNotIn("43-D", summary["emergency_event_process"])
        self.assertIn(f"{int(DRL_CONFIG['action_dim'])}-D", summary["action_schema"])
        self.assertIn("pointing", summary["action_schema"].lower())
        self.assertIn("IMAGE/DOWNLINK/SUN", summary["action_schema"])

    def test_environment_reports_warning_without_termination(self):
        """190km 属于警告区（warning 区为 altitude_min~warning 之间）：episode 不终止，
        risk_stage=warning。注：警告/不安全高度边界已调整，170km 现属 unsafe 区。"""
        from environment.satellite_env import VLEOSatelliteEnv

        env = VLEOSatelliteEnv(seed=21)
        env.reset()
        env.altitude_m = 190e3
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
        from utils.action_space import IDX_POINTING, POINTING_SUN, pointing_unit_for_mode

        env = VLEOSatelliteEnv(seed=12)
        env.reset()
        # 强制在阴影区运行，并只给电池留一小段安全裕度，制造有限可用功率预算。
        env.orbit_sim.reset_phase(env.orbit_sim._sunlit_phase + 0.1)
        target_available_w = ENERGY_CONFIG["power_baseline_w"] + 25.0
        safe_margin_wh = target_available_w * (env.dt / 3600.0) / env.battery.eta_discharge
        env.battery.soc = env.battery.soc_min + safe_margin_wh / env.battery.capacity_wh
        env.energy_queue.reset(env.battery.energy_margin_wh)
        env.step_count = 1
        env.prev_action = np.zeros(env.action_dim, dtype=np.float32)
        env.prev_action[:3] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        env.prev_action[IDX_POINTING] = pointing_unit_for_mode(POINTING_SUN)

        action = np.zeros(env.action_dim, dtype=np.float32)
        action[:3] = np.array([0.7, 1.0, 1.0], dtype=np.float32)
        action[IDX_POINTING] = pointing_unit_for_mode(POINTING_SUN)

        _, _, _, info = env.step(action)

        self.assertTrue(bool(info["power_execution_clipped"]))
        self.assertTrue(bool(info["power_constraint_safe"]))
        self.assertLessEqual(info["P_total_w"], info["available_power_w"] + 1e-6)
        self.assertGreater(info["requested_total_power_w"], info["available_power_w"])
        self.assertLess(info["executed_action"][0], 1.0)

    def test_low_soc_operational_reserve_blocks_adjustable_payload_power(self):
        """SOC 贴近安全线时应提前保电，只保留基础载荷，避免下一步跌破 energy safe。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from config import ENERGY_CONFIG
        from utils.action_space import IDX_POINTING, POINTING_SUN, pointing_unit_for_mode

        env = VLEOSatelliteEnv(seed=46)
        env.reset()
        env.orbit_sim.reset_phase(env.orbit_sim._sunlit_phase + 0.1)
        env.altitude_m = env._h_warning + 50e3
        reserve = float(ENERGY_CONFIG["battery_operational_reserve_soc"])
        env.battery.soc = env.battery.soc_min + 0.5 * reserve
        env.energy_queue.reset(env.battery.energy_margin_wh)
        env._contact = {"in_window": True, "time_to_next_window_s": 0.0}
        env._contact_override = {"in_window": True, "time_to_next_window_s": 0.0}

        action = np.zeros(env.action_dim, dtype=np.float32)
        action[:8] = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        action[IDX_POINTING] = pointing_unit_for_mode(POINTING_SUN)

        _, _, _, info = env.step(action, enforce_prop_smoothing=False)

        self.assertAlmostEqual(
            float(info["available_power_w"]),
            float(ENERGY_CONFIG["power_baseline_w"]),
            places=6,
        )
        self.assertAlmostEqual(float(info["P_cpu_w"]), 0.0, places=6)
        self.assertAlmostEqual(float(info["P_tx_w"]), 0.0, places=6)
        self.assertAlmostEqual(float(info["P_propulsion_w"]), 0.0, places=6)
        self.assertTrue(bool(info["power_constraint_safe"]))

    def test_propulsion_safety_override_breaks_smoothing_near_orbit_floor(self):
        """轨道贴近底线时，小于点火门限的小幅救急推进必须被升到门限点火，
        不能被 N_PROP_SMOOTH 吞掉或低于门限不点火。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from config import ENERGY_CONFIG

        env = VLEOSatelliteEnv(seed=17)
        env.reset()
        env.altitude_m = env._h_min + 1_000.0
        env.step_count = 1
        env.prev_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # 必须用小于 ignition_threshold_ratio 的 action 才能测 boost。
        # 新设计 threshold=150W / max=800W = 0.1875；旧设计是 30/90 = 0.333。
        threshold_ratio = (
            ENERGY_CONFIG["propulsion_ignition_threshold_w"]
            / ENERGY_CONFIG["power_propulsion_max_w"]
        )
        below_threshold_action = 0.5 * threshold_ratio   # 严格低于门限
        _, _, _, info = env.step(
            np.array([below_threshold_action, 0.0, 0.0], dtype=np.float32))

        self.assertFalse(bool(info["prop_can_update"]))
        self.assertTrue(bool(info["safety_override"]))
        self.assertEqual(info["prop_safety_override_reason"], "analytic_propulsion")
        self.assertAlmostEqual(
            float(info["executed_action"][0]),
            1.0,
            places=6,
        )
        self.assertFalse(bool(info["propulsion_ignition_boost_applied"]))
        self.assertTrue(bool(info["analytic_propulsion_controller_enabled"]))
        self.assertTrue(bool(info["analytic_propulsion_applied"]))

    def test_analytic_propulsion_controller_recovers_low_altitude(self):
        """解析推进控制器应接管 actor 的推进维度，低高度时不能让 0 推进动作继续执行。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from config import ENERGY_CONFIG, PROPULSION_CONTROLLER_CONFIG

        env = VLEOSatelliteEnv(seed=41)
        env.reset()
        env.altitude_m = 195e3
        env.battery.soc = 0.80
        env.step_count = 1
        env.prev_action = np.zeros(env.action_dim, dtype=np.float32)

        _, _, _, info = env.step(np.zeros(env.action_dim, dtype=np.float32))
        ignition_alpha = (
            ENERGY_CONFIG["propulsion_ignition_threshold_w"]
            / ENERGY_CONFIG["power_propulsion_max_w"]
        )

        self.assertTrue(bool(PROPULSION_CONTROLLER_CONFIG["enabled"]))
        self.assertTrue(bool(info["analytic_propulsion_controller_enabled"]))
        self.assertTrue(bool(info["analytic_propulsion_applied"]))
        self.assertEqual(info["prop_safety_override_reason"], "analytic_propulsion")
        self.assertGreaterEqual(float(info["executed_action"][0]), ignition_alpha - 1e-6)
        self.assertGreaterEqual(
            float(info["P_propulsion_w"]),
            ENERGY_CONFIG["propulsion_ignition_threshold_w"] - 1e-6,
        )

    def test_analytic_propulsion_controller_coasts_above_target_band(self):
        """高度已有余量时，推进控制器应压住 actor 的满推，避免过推拖累热/能量。"""
        from config import PROPULSION_CONTROLLER_CONFIG
        from environment.satellite_env import VLEOSatelliteEnv

        old_guard_only = PROPULSION_CONTROLLER_CONFIG.get("guard_only")
        try:
            PROPULSION_CONTROLLER_CONFIG["guard_only"] = False
            env = VLEOSatelliteEnv(seed=42)
            env.reset()
            env.altitude_m = 285e3
            env.battery.soc = 0.80
            raw_action = np.zeros(env.action_dim, dtype=np.float32)
            raw_action[0] = 1.0

            _, _, _, info = env.step(raw_action, enforce_prop_smoothing=False)
        finally:
            PROPULSION_CONTROLLER_CONFIG["guard_only"] = old_guard_only

        self.assertTrue(bool(info["analytic_propulsion_controller_enabled"]))
        self.assertTrue(bool(info["analytic_propulsion_applied"]))
        self.assertEqual(info["analytic_propulsion_reason"], "coast_above_band")
        self.assertLess(float(info["executed_action"][0]), 0.05)
        self.assertLess(float(info["P_propulsion_w"]), 1.0)

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
        from config import PROPULSION_CONTROLLER_CONFIG
        from environment.satellite_env import VLEOSatelliteEnv

        old_guard_only = PROPULSION_CONTROLLER_CONFIG.get("guard_only")
        try:
            PROPULSION_CONTROLLER_CONFIG["guard_only"] = False
            env = VLEOSatelliteEnv(seed=20)
            env.reset()
            env.altitude_m = env._h_min + 50_000.0
            env.step_count = 1
            env.prev_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            # alpha_prop=0.1 落在推进点火死区内（ignition_threshold/P_prop_max≈0.167），
            # deadband 会把它归零——验证最终动作虽跳过平滑，仍受物理死区约束。
            _, _, _, info = env.step(
                np.array([0.1, 0.0, 0.0], dtype=np.float32),
                enforce_prop_smoothing=False,
            )
        finally:
            PROPULSION_CONTROLLER_CONFIG["guard_only"] = old_guard_only

        self.assertFalse(bool(info["prop_can_update"]))
        self.assertFalse(bool(info["prop_smoothing_enforced"]))
        self.assertTrue(bool(info["analytic_propulsion_controller_enabled"]))
        self.assertTrue(bool(info["analytic_propulsion_applied"]))
        self.assertAlmostEqual(float(info["raw_alpha_prop"]), 0.1, places=6)
        self.assertGreater(float(info["executed_action"][0]), 0.1)
        self.assertFalse(bool(info["propulsion_deadband_applied"]))

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
        # 推进器平滑锁定即执行器约束（旧 actuator_constraint_applied 并入 prop_smoothing_applied）。
        self.assertTrue(bool(meta["prop_smoothing_applied"]))
        # 安全链投影（旧 safety_intervention_projected 改名为 safety_chain_projected）。
        self.assertTrue(bool(meta["safety_chain_projected"]))

    def test_scheduler_physical_state_preserves_zero_features(self):
        """0.0 是合法观测值，不能被 Python truthiness 默认成健康状态。"""
        from scheduler.integrated_scheduler import IntegratedScheduler
        from config import DRL_CONFIG, ORBITAL_CONFIG
        from environment.satellite_env import OBSERVATION_FEATURES

        scheduler = IntegratedScheduler(device="cpu", enable_lyapunov=False, use_psf=False)
        state_dim = int(DRL_CONFIG.get("state_dim", len(OBSERVATION_FEATURES)))
        state = np.zeros((DRL_CONFIG.get("frame_stack", 8), state_dim), dtype=np.float32)

        phys = scheduler._physical_state_from_obs(
            state,
            in_window=False,
            tx_capacity_mbps=0.0,
            sunlit_fraction=0.0,
        )

        self.assertAlmostEqual(
            float(phys["altitude_m"]),
            float(ORBITAL_CONFIG["altitude_min_km"]) * 1e3,
        )
        self.assertAlmostEqual(float(phys["soc"]), 0.0)
        self.assertAlmostEqual(float(phys["sunlit_fraction"]), 0.0)

    def test_scheduler_orbit_recovery_bypasses_prop_lock(self):
        """低轨应急恢复必须能绕过推进更新锁，否则旧 prop=0 会把救轨道动作吞掉。"""
        from scheduler.integrated_scheduler import IntegratedScheduler
        from config import DRL_CONFIG, ORBITAL_CONFIG, PROPULSION_CONTROLLER_CONFIG
        from environment.satellite_env import OBSERVATION_FEATURES
        from utils.action_space import IDX_POINTING, POINTING_SUN, pointing_mode_from_unit

        scheduler = IntegratedScheduler(device="cpu", enable_lyapunov=False, use_psf=False)
        action_dim = int(DRL_CONFIG.get("action_dim", 3))
        state_dim = int(DRL_CONFIG.get("state_dim", len(OBSERVATION_FEATURES)))
        state = np.zeros((DRL_CONFIG.get("frame_stack", 8), state_dim), dtype=np.float32)
        state[0, OBSERVATION_FEATURES.index("prev_alpha_prop")] = 0.0
        state[0, OBSERVATION_FEATURES.index("soc")] = 0.8
        raw_action = np.ones(action_dim, dtype=np.float32)
        raw_action[0] = 0.0
        h_warning = float(ORBITAL_CONFIG["altitude_warning_km"]) * 1e3

        action, was_projected, _, meta = scheduler._schedule_from_raw_action(
            raw_action,
            state,
            in_window=False,
            h=h_warning - 1.0,
            prop_can_update=False,
            available_power_w=None,
        )

        self.assertTrue(was_projected)
        self.assertTrue(bool(meta["prop_lock_bypassed_for_orbit_recovery"]))
        self.assertFalse(bool(meta["prop_smoothing_applied"]))
        self.assertTrue(bool(meta["orbit_recovery_override"]))
        self.assertGreaterEqual(
            float(action[0]),
            float(PROPULSION_CONTROLLER_CONFIG["emergency_recovery_alpha"]) - 1e-6,
        )
        self.assertLessEqual(
            float(action[1]),
            float(PROPULSION_CONTROLLER_CONFIG["emergency_recovery_cpu_cap"]) + 1e-6,
        )
        self.assertLessEqual(
            float(action[2]),
            float(PROPULSION_CONTROLLER_CONFIG["emergency_recovery_tx_cap"]) + 1e-6,
        )
        if action_dim > IDX_POINTING:
            self.assertEqual(pointing_mode_from_unit(float(action[IDX_POINTING])), POINTING_SUN)

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
        # boundary 功率裁剪是安全链介入，但不是推进器平滑锁定。
        self.assertTrue(bool(meta["safety_chain_projected"]))
        self.assertFalse(bool(meta["prop_smoothing_applied"]))
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
        from utils.action_space import IDX_POINTING, POINTING_DOWNLINK, pointing_unit_for_mode

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

        action = np.zeros(env.action_dim, dtype=np.float32)
        action[2] = 0.1
        action[IDX_POINTING] = pointing_unit_for_mode(POINTING_DOWNLINK)

        _, _, _, info = env.step(action)

        self.assertGreater(info["link_tx_capacity_mb"], info["rf_tx_capacity_mb"])
        self.assertLessEqual(info["actual_tx_mb"], info["rf_tx_capacity_mb"] + 1e-6)

    def test_downlink_is_limited_by_per_pass_receive_budget(self):
        """单次过顶应有接收容量上限，避免高仰角瞬时容量被当成无限窗口。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from environment.task_value_model import TaskBatch
        from utils.action_space import IDX_POINTING, POINTING_DOWNLINK, pointing_unit_for_mode

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

        action = np.zeros(env.action_dim, dtype=np.float32)
        action[2] = 1.0
        action[IDX_POINTING] = pointing_unit_for_mode(POINTING_DOWNLINK)

        _, _, _, info = env.step(action)

        self.assertLessEqual(info["link_tx_capacity_mb"], 20.0 + 1e-6)
        self.assertLessEqual(info["actual_tx_mb"], 20.0 + 1e-6)
        self.assertLess(info["comm_pass_remaining_mb"], 20.0)

    def test_cpu_backpressure_deployment_projection_is_disabled_by_default(self):
        """processed-queue 未接近上限时，部署边界投影不应压 CPU。

        注：部署侧 queue projection 已从 scheduler 移到 env 的 actuator_filter
        (project_processed_queue_boundary)，本测试直接核对该投影器：队列有充足
        headroom 时不触发 back-pressure，alpha_cpu 保持原值。"""
        from environment.satellite_env import VLEOSatelliteEnv

        env = VLEOSatelliteEnv(seed=20)
        env.reset()
        res = env.actuator_filter.project_processed_queue_boundary(
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            processed_queue_mb=0.0,          # 队列空 → headroom 充足
            processed_queue_max_mb=env.comm_queue.max_value,
            in_window=False,
            tx_capacity_mbps=0.0,
            dt_s=env.dt,
            apply_projection=True,
        )

        self.assertFalse(bool(res.meta["cpu_backpressure_applied"]))
        self.assertAlmostEqual(float(res.action[1]), 1.0, places=6)

    def test_cpu_backpressure_deployment_projection_can_be_enabled(self):
        """processed-queue 接近上限且无下传窗口时，部署边界投影应压低 CPU。

        注：部署侧 queue projection 已从 scheduler 移到 env 的 actuator_filter。
        队列满 + 窗口外（无 tx headroom）时，再处理只会溢出，投影器应把 alpha_cpu 压低。"""
        from environment.satellite_env import VLEOSatelliteEnv

        env = VLEOSatelliteEnv(seed=20)
        env.reset()
        res = env.actuator_filter.project_processed_queue_boundary(
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            processed_queue_mb=env.comm_queue.max_value,   # 队列满 → 无 headroom
            processed_queue_max_mb=env.comm_queue.max_value,
            in_window=False,                               # 窗口外 → 无 tx_room
            tx_capacity_mbps=0.0,
            dt_s=env.dt,
            apply_projection=True,
        )

        self.assertTrue(bool(res.meta["cpu_backpressure_applied"]))
        self.assertLess(float(res.action[1]), 1.0)

    def test_environment_queue_backpressure_is_diagnostic_by_default(self):
        """Default env rollout should expose queue pressure without rewriting CPU."""
        from copy import deepcopy

        from config import DRL_CONFIG
        from environment.satellite_env import VLEOSatelliteEnv
        from utils.action_space import IDX_POINTING, POINTING_SUN, pointing_unit_for_mode

        old_drl_cfg = deepcopy(DRL_CONFIG)
        try:
            DRL_CONFIG["queue_projection_policy"] = "diagnostic_only"
            DRL_CONFIG["enable_deployment_queue_projection"] = False
            env = VLEOSatelliteEnv(seed=21)
            env.reset()
            env.battery.soc = 0.95
            env.comm_queue.value = float(env.comm_queue.max_value)
            env._contact_override = {
                "in_window": False,
                "time_to_next_window_s": 10000.0,
                "max_capacity_mbps": 0.0,
            }
            action = np.zeros(env.action_dim, dtype=np.float32)
            action[1] = 1.0
            action[IDX_POINTING] = pointing_unit_for_mode(POINTING_SUN)

            _, _, _, info = env.step(action, enforce_prop_smoothing=False)
        finally:
            DRL_CONFIG.clear()
            DRL_CONFIG.update(old_drl_cfg)

        self.assertEqual(info["queue_projection_policy"], "diagnostic_only")
        self.assertFalse(bool(info["deployment_queue_projection_enabled"]))
        self.assertTrue(bool(info["cpu_backpressure_required"]))
        self.assertFalse(bool(info["cpu_backpressure_applied"]))
        self.assertAlmostEqual(float(info["executed_action"][1]), 1.0, places=5)

    def test_environment_queue_backpressure_can_be_explicit_deployment_boundary(self):
        """Explicit deployment hard-boundary mode may rewrite CPU before power accounting."""
        from copy import deepcopy

        from config import DRL_CONFIG
        from environment.satellite_env import VLEOSatelliteEnv
        from utils.action_space import IDX_POINTING, POINTING_SUN, pointing_unit_for_mode

        old_drl_cfg = deepcopy(DRL_CONFIG)
        try:
            DRL_CONFIG["queue_projection_policy"] = "deployment_hard_boundary"
            DRL_CONFIG["enable_deployment_queue_projection"] = True
            env = VLEOSatelliteEnv(seed=22)
            env.reset()
            env.battery.soc = 0.95
            env.comm_queue.value = float(env.comm_queue.max_value)
            env._contact_override = {
                "in_window": False,
                "time_to_next_window_s": 10000.0,
                "max_capacity_mbps": 0.0,
            }
            action = np.zeros(env.action_dim, dtype=np.float32)
            action[1] = 1.0
            action[IDX_POINTING] = pointing_unit_for_mode(POINTING_SUN)

            _, _, _, info = env.step(action, enforce_prop_smoothing=False)
        finally:
            DRL_CONFIG.clear()
            DRL_CONFIG.update(old_drl_cfg)

        self.assertEqual(info["queue_projection_policy"], "deployment_hard_boundary")
        self.assertTrue(bool(info["deployment_queue_projection_enabled"]))
        self.assertTrue(bool(info["cpu_backpressure_required"]))
        self.assertTrue(bool(info["cpu_backpressure_applied"]))
        self.assertLess(float(info["executed_action"][1]), 1.0)

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
        from utils.action_space import IDX_POINTING, POINTING_DOWNLINK, pointing_unit_for_mode

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

        action = np.zeros(env.action_dim, dtype=np.float32)
        action[2] = 1.0
        action[IDX_POINTING] = pointing_unit_for_mode(POINTING_DOWNLINK)

        _, _, _, info = env.step(action)

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

    def test_thermal_state_counts_propulsion_as_ppu_body_heat(self):
        """推进功率只应按 PPU/安装耦合的小比例进入舱内热，而不是全部按电子热折算。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from config import ENERGY_CONFIG, THERMAL_CONFIG

        env = VLEOSatelliteEnv(seed=40)
        env.reset()
        env.thermal_temperature_c = 20.0

        p_prop = float(ENERGY_CONFIG["power_propulsion_max_w"])
        p_base = float(ENERGY_CONFIG["power_baseline_w"])
        info = env._update_thermal_state(
            total_power_w=p_prop + p_base,
            sunlit_fraction=0.0,
            propulsion_power_w=p_prop,
            cpu_power_w=0.0,
            tx_power_w=0.0,
        )

        electronics_fraction = float(THERMAL_CONFIG["electronics_heat_fraction"])
        propulsion_fraction = float(THERMAL_CONFIG["propulsion_heat_fraction"])
        legacy_full_heat_w = (p_prop + p_base) * electronics_fraction

        self.assertAlmostEqual(
            float(info["propulsion_thermal_power_w"]),
            p_prop,
            places=6,
        )
        self.assertAlmostEqual(
            float(info["propulsion_heat_w"]),
            p_prop * propulsion_fraction,
            places=6,
        )
        self.assertAlmostEqual(
            float(info["electronics_heat_w"]),
            p_base * electronics_fraction,
            places=6,
        )
        self.assertLess(float(info["internal_heat_w"]), 0.25 * legacy_full_heat_w)

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
        from config import TASK_CONFIG

        env = VLEOSatelliteEnv(seed=26)
        env.reset()
        env._scene_phase_offset_fraction = 0.0
        env._phase_scene_rules = list(TASK_CONFIG["phase_scene_rules"])
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
        from config import TASK_CONFIG

        env = VLEOSatelliteEnv(seed=27)
        env.reset()
        env._scene_phase_offset_fraction = 0.0
        env._phase_scene_rules = list(TASK_CONFIG["phase_scene_rules"])
        military_scene = env._scene_context_for_phase(phase=0.20 * 2.0 * np.pi)
        cloud_scene = env._scene_context_for_phase(phase=0.90 * 2.0 * np.pi)

        self.assertGreater(
            env._arrival_rate_for_scene(military_scene),
            env._arrival_rate_for_scene(cloud_scene),
        )

    def test_upcoming_scene_intensity_uses_value_and_deadline(self):
        from environment.satellite_env import VLEOSatelliteEnv
        from config import TASK_CONFIG

        env = VLEOSatelliteEnv(seed=28)
        env.reset()
        env._scene_phase_offset_fraction = 0.0
        env._phase_scene_rules = list(TASK_CONFIG["phase_scene_rules"])
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
        # 给 150W 可调功率：prop 优先吃满（150W > 120W 点火阈值，不被 deadband 归零），
        # cpu/tx 无剩余功率，验证 out-window 严格优先级 prop>cpu>tx。
        available = ENERGY_CONFIG["power_baseline_w"] + 150.0
        safe, meta = scheduler._clip_action_boundaries(
            raw, available_power_w=available, in_window=False)

        self.assertEqual(meta["power_priority_order"], "prop>cpu>tx")
        self.assertAlmostEqual(
            float(safe[0]),
            150.0 / ENERGY_CONFIG["power_propulsion_max_w"],
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
        """物理安全时，Lyapunov 投影不应改变 alpha_cpu / alpha_tx。

        注：Lyapunov 安全层已从旧的队列阈值 layer 重构为状态相关投影
        (LyapunovProjector，基于 LyapunovFunction + 动力学预测)。即便 raw/processed
        队列进入 L 函数，只要物理状态(高度/SOC/热)安全，投影后动作应维持不变。"""
        from safety.lyapunov_projection import LyapunovProjector
        from config import QUEUE_CONFIG

        proj = LyapunovProjector()
        high_raw = QUEUE_CONFIG.get("data_queue_max_mb", 500.0)
        high_comm = QUEUE_CONFIG.get("comm_queue_max", 500.0)
        base = dict(altitude_m=360e3, soc=0.6, thermal_margin_norm=1.0,
                    sunlit_fraction=0.7, in_window=False,
                    future_contact_capacity_mb=500.0)
        action = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        r_high = proj.project(
            action, {**base, "processed_queue_mb": high_comm, "raw_queue_mb": high_raw})
        r_low = proj.project(
            action, {**base, "processed_queue_mb": high_comm, "raw_queue_mb": 0.0})

        np.testing.assert_allclose(r_high.action, action, atol=1e-6)
        np.testing.assert_allclose(r_low.action, action, atol=1e-6)

    def test_lyapunov_layer_triggers_in_moderate_risk_band(self):
        """安全恢复职责已分层：临界高度风险由 PSF 硬性抬推进恢复（旧的 Lyapunov
        队列阈值层已重构为状态相关投影，不再在安全层因 orbit 队列硬抬 prop）。
        这里验证当前真正负责硬恢复的 PSF：接近再入高度时强制抬高 alpha_prop。"""
        from safety.psf_filter import PredictiveSafetyFilter
        from config import ORBITAL_CONFIG

        psf = PredictiveSafetyFilter(K=6)
        h_crash = float(ORBITAL_CONFIG.get("altitude_crash_km", 120.0)) * 1e3
        action = np.array([0.0, 0.6, 0.2], dtype=np.float32)

        result = psf.filter(action, {
            "altitude_m": h_crash + 10e3,
            "soc": 0.6,
            "processed_queue_mb": 0.0,
            "thermal_margin_norm": 1.0,
            "sunlit_fraction": 0.7,
            "in_window": False,
        })

        self.assertTrue(result.intervened)
        self.assertGreater(float(result.action[0]), float(action[0]))

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
        executed_action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        safe_action = np.array([0.2, 0.3, 0.4], dtype=np.float32)

        buf.push(
            state, executed_action, 1.25, state, False, 0.0,
            behavior_action=safe_action,
            behavior_weight=0.7,
        )
        # sample 同时返回 executed action 和 raw action；critic 训练仍使用 executed action。
        (_, action, reward, _, _, _, _,
         behavior_action, behavior_weight, _, raw_action) = buf.sample(1)

        self.assertTrue(np.allclose(action[0], executed_action))
        self.assertTrue(np.allclose(raw_action[0], executed_action))
        self.assertAlmostEqual(float(reward[0, 0]), 1.25)
        self.assertTrue(np.allclose(behavior_action[0], safe_action))
        self.assertAlmostEqual(float(behavior_weight[0, 0]), 0.7)

    def test_replay_buffer_stores_raw_action_for_off_support_diagnostics(self):
        """Replay 必须保留 raw action，才能诊断 Q(s,a_raw) 与 Q(s,a_exec) 的分布错位。"""
        from drl.agent import ReplayBuffer
        from config import DRL_CONFIG

        state_dim = int(DRL_CONFIG.get("state_dim", 30))
        frame_stack = int(DRL_CONFIG.get("frame_stack", 8))
        buf = ReplayBuffer(capacity=1, state_dim=state_dim, action_dim=3)
        state = np.zeros((frame_stack, state_dim), dtype=np.float32)
        executed_action = np.array([0.2, 0.3, 0.4], dtype=np.float32)
        raw_action = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        buf.push(
            state, executed_action, 1.25, state, False, 0.0,
            raw_action=raw_action,
        )
        (*_, n_step_gamma_pow, sampled_raw_action) = buf.sample(1)

        self.assertEqual(n_step_gamma_pow.shape, (1, 1))
        self.assertTrue(np.allclose(sampled_raw_action[0], raw_action))

    def test_agent_update_reports_raw_vs_executed_q_gap(self):
        """SAC update 应报告 raw/executed critic 查询差，暴露安全层导致的 off-support 风险。"""
        import inspect
        from drl.agent import SACAgent

        source = inspect.getsource(SACAgent.update)

        self.assertIn("critic_q_raw_minus_exec_mean", source)
        self.assertIn("critic_q_raw_exec_abs_gap_mean", source)
        self.assertIn("critic_q_raw_mean", source)
        self.assertIn("critic_q_exec_mean", source)
        self.assertIn("actor_reward_q_mean", source)
        self.assertIn("actor_update_applied", source)
        self.assertIn("target_q_mean", source)
        self.assertIn("target_q_std", source)
        self.assertIn("replay_reward_mean", source)
        self.assertIn("replay_constraint_cost_mean", source)


class TestPSFQueueGuard(unittest.TestCase):

    def setUp(self):
        from safety.psf_filter import PredictiveSafetyFilter
        self.psf = PredictiveSafetyFilter(K=6)

    def test_psf_constructor_uses_configured_search_budget(self):
        from config import PSF_CONFIG
        from safety.psf_filter import PredictiveSafetyFilter

        psf = PredictiveSafetyFilter()

        self.assertEqual(psf.K, int(PSF_CONFIG["horizon_steps"]))
        self.assertEqual(psf.line_search_steps, int(PSF_CONFIG["line_search_steps"]))

    def test_psf_backup_prioritizes_orbit_recovery_over_thermal_guard(self):
        from config import DRL_CONFIG, ORBITAL_CONFIG, QUEUE_CONFIG
        from safety.psf_filter import make_backup_action
        from utils.action_space import IDX_POINTING, POINTING_SUN, pointing_mode_from_unit

        action_dim = int(DRL_CONFIG.get("action_dim", 3))
        warning_m = float(ORBITAL_CONFIG["altitude_warning_km"]) * 1e3
        backup = make_backup_action(
            {
                "altitude_m": warning_m - 1.0,
                "soc": 0.8,
                "thermal_margin_norm": -0.5,
                "processed_queue_mb": 0.0,
                "in_window": False,
            },
            action_dim=action_dim,
            altitude_warning_m=warning_m,
            soc_warning=0.20,
            processed_queue_max_mb=float(QUEUE_CONFIG.get("comm_queue_max", 4096.0)),
        )

        self.assertAlmostEqual(float(backup[0]), 1.0)
        self.assertAlmostEqual(float(backup[1]), 0.0)
        self.assertAlmostEqual(float(backup[2]), 0.0)
        if action_dim > IDX_POINTING:
            self.assertEqual(pointing_mode_from_unit(float(backup[IDX_POINTING])), POINTING_SUN)

    def test_scheduler_safety_stats_split_psf_effectiveness(self):
        from scheduler.integrated_scheduler import IntegratedScheduler

        scheduler = IntegratedScheduler(device="cpu", use_psf=True)
        stats = scheduler.get_safety_stats()

        self.assertIn("psf_filter_rate", stats)
        self.assertIn("psf_backup_failure_rate", stats)
        self.assertIn("psf_effective_success_rate", stats)
        self.assertIn("chain_total_rate", stats)
        self.assertIn("orbit_recovery_override_rate", stats)
        self.assertIn("prop_lock_orbit_bypass_rate", stats)

    def test_long_horizon_check_is_risk_band_gated(self):
        from config import ORBITAL_CONFIG
        from safety.psf_filter import PredictiveSafetyFilter

        psf = PredictiveSafetyFilter(K=6)
        action = np.zeros(3, dtype=np.float32)

        healthy_ok, healthy_info = psf._long_horizon_safety_check(
            {
                "altitude_m": 350_000.0,
                "soc": 0.95,
                "thermal_margin_norm": 0.95,
                "sunlit_fraction": 1.0,
            },
            action,
        )
        self.assertTrue(healthy_ok)
        self.assertTrue(bool(healthy_info.get("long_horizon_skipped", False)))

        h_min = float(ORBITAL_CONFIG["altitude_min_km"]) * 1e3
        risk_ok, risk_info = psf._long_horizon_safety_check(
            {
                "altitude_m": h_min + 1_000.0,
                "soc": 0.95,
                "thermal_margin_norm": 0.95,
                "sunlit_fraction": 1.0,
            },
            action,
        )
        self.assertTrue(risk_ok)
        self.assertFalse(bool(risk_info.get("long_horizon_skipped", False)))

    def test_physical_trigger_thresholds_keep_recovery_margin(self):
        """PSF 安全边界应在物理硬边界(crash)之上留出提前恢复余量。"""
        from config import ENERGY_CONFIG, ORBITAL_CONFIG, PSF_CONFIG

        h_crash = float(ORBITAL_CONFIG["altitude_crash_km"]) * 1e3
        soc_crash = float(ENERGY_CONFIG["battery_crash_soc"])

        self.assertAlmostEqual(
            self.psf.h_safe_min,
            h_crash + float(PSF_CONFIG["altitude_trigger_margin_m"]),
        )
        self.assertAlmostEqual(
            self.psf.soc_safe_min,
            soc_crash + float(PSF_CONFIG["soc_trigger_margin"]),
        )
        self.assertGreater(self.psf.h_safe_min, h_crash + 3e3)
        self.assertGreater(self.psf.soc_safe_min, soc_crash + 0.015)

    def test_data_queue_pressure_does_not_change_action(self):
        """物理安全时 PSF 不应拦截（当前 PSF 只看物理状态，不看 data 队列）。"""
        raw = np.array([0.3, 0.05, 0.1], dtype=np.float32)
        result = self.psf.filter(raw, {
            "altitude_m": 360e3,
            "soc": 0.6,
            "processed_queue_mb": 5.0,
            "thermal_margin_norm": 1.0,
            "sunlit_fraction": 0.7,
            "in_window": False,
        })
        self.assertTrue(np.allclose(result.action, raw, atol=1e-6))
        self.assertFalse(result.intervened)
        self.assertTrue(result.raw_safe)

    def test_comm_queue_pressure_does_not_change_action(self):
        """窗口内 processed 队列高压但物理安全时，PSF 不应拦截。"""
        raw = np.array([0.25, 0.2, 0.05], dtype=np.float32)
        result = self.psf.filter(raw, {
            "altitude_m": 360e3,
            "soc": 0.6,
            "processed_queue_mb": 50.0,
            "thermal_margin_norm": 1.0,
            "sunlit_fraction": 0.7,
            "in_window": True,
        })
        self.assertTrue(np.allclose(result.action, raw, atol=1e-6))
        self.assertFalse(result.intervened)

    def test_psf_uses_finite_rollout_not_formal_guarantee(self):
        """当前 PSF 是有限步 rollout + 二分线搜的近似认证，非形式化安全保证。"""
        self.assertGreaterEqual(int(self.psf.K), 1)
        self.assertGreaterEqual(int(self.psf.line_search_steps), 1)
        self.assertGreater(float(self.psf.long_horizon_steps), 0.0)


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
        """真实 trace 低于 altitude_crash_km 时应落到再入终止线，而不是被 150km 吃掉。"""
        from environment.satellite_env import VLEOSatelliteEnv
        from experiments.robustness import _apply_trace_item
        from config import ORBITAL_CONFIG

        env = VLEOSatelliteEnv(seed=12)
        env.reset()
        env.step_count = 1
        env.orbit_queue.value = 9.0

        # 用 crash_km - 1 (低于 crash 边界 1km) 触发夹断到 crash
        crash_km = float(ORBITAL_CONFIG["altitude_crash_km"])
        _apply_trace_item(
            env,
            {"altitude_km": crash_km - 1.0},
            base_rho_ref=env.orbit_dyn.atm.rho_ref,
            base_solar_eta=env.solar.eta,
            trace_altitude_mode="force",
        )

        self.assertAlmostEqual(env.altitude_m, env._h_crash)
        self.assertAlmostEqual(env.orbit_queue.value, 9.0)

    def test_trace_loader_keeps_altitudes_below_150km(self):
        """CSV 回放解析不能把 crash~150km 的不安全状态提前裁剪成安全边界 (h_min)。"""
        import os
        import tempfile
        from experiments.robustness import _load_trace_rows
        from config import ORBITAL_CONFIG

        crash_km = float(ORBITAL_CONFIG["altitude_crash_km"])
        # 第一行 130km (unsafe 区) 保留，第二行 crash-1 (crash 边界下) 被夹到 crash
        below_crash = crash_km - 1.0
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(f"altitude_km\n130\n{below_crash}\n")
            path = f.name
        try:
            rows = _load_trace_rows(path)
        finally:
            os.unlink(path)

        self.assertAlmostEqual(rows[0]["altitude_km"], 130.0)
        self.assertAlmostEqual(rows[1]["altitude_km"], crash_km)


class TestEvaluationReportMath(unittest.TestCase):

    def test_relative_improvement_direction(self):
        """模型优于 baseline 时，报告中的提升比例应为正。"""
        from evaluate_optimized import _relative_improvement

        self.assertAlmostEqual(_relative_improvement(120.0, 100.0), 20.0)
        self.assertAlmostEqual(
            _relative_improvement(5.0, 10.0, lower_is_better=True),
            50.0,
        )

    def test_compact_paper_table_includes_policy_diagnostics(self):
        """论文主表应暴露动作改写和燃料消耗，否则看不出策略是否真正生效。"""
        from utils.paper_metrics import compact_paper_table_row

        row = compact_paper_table_row({
            "delivered_value_mean": 100.0,
            "safety_adjusted_delivered_value": 87.5,
            "constraint_total_clean_mean": 0.15,
            "qos_total_mean": 0.35,
            "shield_dependence_score": 0.46,
            "mean_action_modification": 0.12,
            "raw_executed_action_l2_mean": 0.34,
            "fuel_consumed_g_mean": 5.6,
            "propellant_remaining_fraction_mean": 0.91,
            "delivered_high_value_mean": 40.0,
            "delivered_mid_value_mean": 30.0,
            "delivered_low_value_mean": 10.0,
        })

        self.assertAlmostEqual(row["Mean Action Modification"], 0.12)
        self.assertAlmostEqual(row["Raw/Executed Action L2"], 0.34)
        self.assertAlmostEqual(row["Safety-adjusted Delivered VoI"], 87.5)
        self.assertAlmostEqual(row["Clean Constraint Cost"], 0.15)
        self.assertAlmostEqual(row["Mission QoS Cost"], 0.35)
        self.assertAlmostEqual(row["Shield Dependence Score"], 0.46)
        self.assertAlmostEqual(row["Fuel Consumed (g)"], 5.6)
        self.assertAlmostEqual(row["Propellant Remaining Fraction"], 0.91)
        self.assertAlmostEqual(row["Delivered High VoI"], 40.0)
        self.assertAlmostEqual(row["Delivered Mid VoI"], 30.0)
        self.assertAlmostEqual(row["Delivered Low VoI"], 10.0)

    def test_compare_all_declares_decoupled_baselines(self):
        """解耦堆叠基线必须进入正式对照矩阵，而不是只停留在未注册文件。"""
        from types import SimpleNamespace
        from experiments.compare_all import (
            _baseline_information_conditions,
            _make_decoupled_baseline_schedulers,
        )

        factories = _make_decoupled_baseline_schedulers()
        names = [name for name, _ in factories]

        self.assertIn("DECOUPLED-Heur", names)
        self.assertIn("DECOUPLED-MPC", names)

        conditions = _baseline_information_conditions(
            SimpleNamespace(
                baseline_safety_shell=False,
                mpc_horizon=6,
                robust_mpc_horizon=8,
            ),
            {name: {} for name in names},
        )

        by_method = conditions["by_method"]
        self.assertIn("DECOUPLED-Heur", by_method)
        self.assertIn("DECOUPLED-MPC", by_method)
        self.assertIn("coupling-blind", by_method["DECOUPLED-Heur"]["notes"])
        self.assertIn("coupling-blind", by_method["DECOUPLED-MPC"]["notes"])

    def test_compare_all_declares_algorithm_and_deployment_tables(self):
        """正式对比必须区分算法本体表和统一安全壳部署表。"""
        from experiments.compare_all import _comparison_table_protocol

        protocol = _comparison_table_protocol()

        self.assertIn("algorithm_only", protocol)
        self.assertIn("deployment_shell", protocol)
        self.assertFalse(bool(protocol["algorithm_only"]["extra_safety_shell"]))
        self.assertTrue(bool(protocol["deployment_shell"]["extra_safety_shell"]))
        self.assertIn("no_extra_hard_rules", protocol["algorithm_only"]["safety_policy"])
        self.assertIn("same_safety_shell", protocol["deployment_shell"]["safety_policy"])

    def test_compare_all_declares_shell_attribution_baselines(self):
        """安全壳归因必须包含 default/random actor + same shell，量化规则壳自身分数。"""
        from types import SimpleNamespace
        from experiments.compare_all import (
            _baseline_information_conditions,
            _make_shell_attribution_baseline_schedulers,
        )

        names = [name for name, _ in _make_shell_attribution_baseline_schedulers(seed=123)]

        self.assertIn("Rule-only Shell (no learned policy)", names)
        self.assertIn("Random Actor + Safety Shell", names)

        conditions = _baseline_information_conditions(
            SimpleNamespace(baseline_safety_shell=True, mpc_horizon=6, robust_mpc_horizon=8),
            {name: {} for name in names},
        )
        by_method = conditions["by_method"]
        self.assertIn("Rule-only Shell (no learned policy)", by_method)
        self.assertIn("Random Actor + Safety Shell", by_method)
        self.assertIn("same shell", by_method["Rule-only Shell (no learned policy)"]["notes"])
        self.assertIn("same shell", by_method["Random Actor + Safety Shell"]["notes"])

    def test_mission_reward_exposes_primary_and_auxiliary_shaping(self):
        """主 reward 只应可识别为 mission value，其余工程项必须显式归入 shaping。"""
        from objectives.mission_reward import compute_mission_reward

        reward = compute_mission_reward(
            delivered_value=10.0,
            on_time_delivered_value=6.0,
            expired_value=3.0,
            expired_high_value=3.0,
            dropped_value=2.0,
            dropped_mb=1.0,
            transmitted_mb=1.0,
            processed_mb=5.0,
            total_power_w=100.0,
            propulsion_power_w=200.0,
            dt_s=10.0,
            cfg={
                "w_delivered_value": 2.0,
                "w_deadline_success": 1.0,
                "w_expired_penalty": -1.0,
                "w_energy_penalty": -0.5,
                "w_prop_overburn_penalty": 0.1,
                "prop_overburn_threshold_w": 100.0,
                "enable_class_weighted_reward": False,
            },
        )

        self.assertAlmostEqual(reward.components["primary_mission_reward"], 20.0)
        self.assertNotEqual(reward.components["auxiliary_shaping_reward"], 0.0)
        self.assertAlmostEqual(
            reward.total,
            reward.components["primary_mission_reward"]
            + reward.components["auxiliary_shaping_reward"],
        )
        self.assertEqual(reward.components["reward_contract"], "primary_plus_auxiliary_shaping")

    def test_training_critic_replay_action_uses_executed_action(self):
        """critic 的 replay action 应对齐真实环境转移，raw action 只作为安全层依赖诊断。"""
        from train import _critic_replay_action

        raw_action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        executed_action = np.array([0.2, 0.3, 0.4], dtype=np.float32)

        replay_action, meta = _critic_replay_action(raw_action, executed_action)

        np.testing.assert_allclose(replay_action, executed_action)
        self.assertAlmostEqual(meta["raw_executed_action_l2"], float(np.linalg.norm(raw_action - executed_action)))
        self.assertEqual(meta["critic_action_semantics"], "executed_action")

    def test_compare_all_declares_full_rule_ablation_axes(self):
        """Rule ablations must cover each deployment hard rule as its own axis."""
        from experiments.compare_all import _rule_ablation_specs

        specs = _rule_ablation_specs()

        expected = {
            "analytic_propulsion_controller",
            "mission_pointing_fallback",
            "in_window_tx_floor",
            "future_contact_cpu_gate",
            "in_window_cpu_feed_floor",
            "class_priority_floor",
            "deliverability_gate",
            "tx_high_reserve",
            "layered_edf",
        }
        self.assertEqual(expected, set(specs))
        for key in expected:
            self.assertIn("label", specs[key])
            self.assertIn("paper_axis", specs[key])
            self.assertIn("overrides", specs[key])
            self.assertTrue(specs[key]["overrides"])

    def test_compare_all_hard_rule_shell_is_explicit_not_default(self):
        """Deployment-rule ablations must first enable the old hard-rule shell explicitly."""
        from experiments.compare_all import (
            _hard_rule_shell_overrides,
            _merge_nested_overrides,
            _rule_ablation_specs,
        )

        shell = _hard_rule_shell_overrides()
        self.assertTrue(shell["propulsion"]["enabled"])
        self.assertFalse(shell["propulsion"]["guard_only"])
        for key in [
            "cpu_action_is_admissible_budget",
            "enable_future_contact_cpu_gate",
            "enable_cpu_throttle",
            "enable_high_value_cpu_gate_escape",
            "enable_in_window_cpu_feed_floor",
        ]:
            self.assertTrue(shell["task"][key])
        for key in [
            "enable_deliver_prob_gate",
            "enable_class_aware_gate",
            "enable_class_priority_floor",
            "enable_tx_high_reserve",
            "enable_layered_edf",
            "enable_in_window_tx_floor",
            "enable_mission_pointing_fallback",
        ]:
            self.assertTrue(shell["hard_rules"][key])

        ablated = _merge_nested_overrides(
            shell,
            _rule_ablation_specs()["deliverability_gate"]["overrides"],
        )
        self.assertFalse(ablated["hard_rules"]["enable_deliver_prob_gate"])
        self.assertFalse(ablated["hard_rules"]["enable_class_aware_gate"])
        self.assertTrue(ablated["hard_rules"]["enable_class_priority_floor"])

    def test_compare_all_global_rule_ablation_starts_from_hard_shell(self):
        """Whole-run rule disable flags should not become no-op under policy-first defaults."""
        import inspect
        import experiments.compare_all as compare_all

        source = inspect.getsource(compare_all)

        self.assertIn("--use_hard_rule_shell", source)
        self.assertIn("hard_rule_ablation_requested", source)
        self.assertIn("enable_hard_rule_shell", source)

    def test_compare_all_excludes_oracle_from_formal_paper_table(self):
        """Oracle MPC is an upper bound table entry, not a fair deployable baseline."""
        from experiments.compare_all import (
            _formal_paper_table_results,
            _upper_bound_table_results,
        )

        results = {
            "MPC": {"delivered_value_mean": 1.0},
            "Omniscient MPC (Oracle)": {"delivered_value_mean": 2.0},
        }

        self.assertIn("MPC", _formal_paper_table_results(results))
        self.assertNotIn("Omniscient MPC (Oracle)", _formal_paper_table_results(results))
        self.assertIn("Omniscient MPC (Oracle)", _upper_bound_table_results(results))

    def test_env_safety_layer_overrides_can_disable_all_rule_axes(self):
        """Evaluation overrides should disable and restore each rule axis."""
        from config import (
            ACTUATOR_GATE_CONFIG,
            HARD_RULES_CONFIG,
            PROPULSION_CONTROLLER_CONFIG,
            TASK_CONFIG,
        )
        from evaluate_optimized import env_safety_layer_overrides

        saved = {
            "prop": PROPULSION_CONTROLLER_CONFIG.get("enabled"),
            "point": HARD_RULES_CONFIG.get("enable_mission_pointing_fallback"),
            "tx_floor": HARD_RULES_CONFIG.get("enable_in_window_tx_floor"),
            "cpu_gate": TASK_CONFIG.get("enable_future_contact_cpu_gate"),
            "cpu_feed": TASK_CONFIG.get("enable_in_window_cpu_feed_floor"),
            "class_floor": HARD_RULES_CONFIG.get("enable_class_priority_floor"),
            "deliver_prob": HARD_RULES_CONFIG.get("enable_deliver_prob_gate"),
            "class_aware": HARD_RULES_CONFIG.get("enable_class_aware_gate"),
            "tx_reserve": HARD_RULES_CONFIG.get("enable_tx_high_reserve"),
            "edf": HARD_RULES_CONFIG.get("enable_layered_edf"),
            "guard_only": PROPULSION_CONTROLLER_CONFIG.get("guard_only"),
            "cpu_soft": ACTUATOR_GATE_CONFIG.get("cpu_gate_soft_mode"),
            "cpu_budget": TASK_CONFIG.get("cpu_action_is_admissible_budget"),
            "cpu_throttle": TASK_CONFIG.get("enable_cpu_throttle"),
            "high_escape": TASK_CONFIG.get("enable_high_value_cpu_gate_escape"),
        }

        with env_safety_layer_overrides(
            disable_analytic_propulsion=True,
            disable_pointing_fallback=True,
            disable_in_window_tx_floor=True,
            disable_future_contact_cpu_gate=True,
            disable_in_window_cpu_feed_floor=True,
            disable_class_priority_floor=True,
            disable_deliverability_gate=True,
            disable_tx_high_reserve=True,
            disable_layered_edf=True,
            enable_hard_rule_shell=True,
        ):
            self.assertFalse(PROPULSION_CONTROLLER_CONFIG["enabled"])
            self.assertFalse(HARD_RULES_CONFIG["enable_mission_pointing_fallback"])
            self.assertFalse(HARD_RULES_CONFIG["enable_in_window_tx_floor"])
            self.assertFalse(TASK_CONFIG["enable_future_contact_cpu_gate"])
            self.assertFalse(TASK_CONFIG["enable_in_window_cpu_feed_floor"])
            self.assertFalse(HARD_RULES_CONFIG["enable_class_priority_floor"])
            self.assertFalse(HARD_RULES_CONFIG["enable_deliver_prob_gate"])
            self.assertFalse(HARD_RULES_CONFIG["enable_class_aware_gate"])
            self.assertFalse(HARD_RULES_CONFIG["enable_tx_high_reserve"])
            self.assertFalse(HARD_RULES_CONFIG["enable_layered_edf"])

        self.assertEqual(PROPULSION_CONTROLLER_CONFIG.get("enabled"), saved["prop"])
        self.assertEqual(HARD_RULES_CONFIG.get("enable_mission_pointing_fallback"), saved["point"])
        self.assertEqual(HARD_RULES_CONFIG.get("enable_in_window_tx_floor"), saved["tx_floor"])
        self.assertEqual(TASK_CONFIG.get("enable_future_contact_cpu_gate"), saved["cpu_gate"])
        self.assertEqual(TASK_CONFIG.get("enable_in_window_cpu_feed_floor"), saved["cpu_feed"])
        self.assertEqual(HARD_RULES_CONFIG.get("enable_class_priority_floor"), saved["class_floor"])
        self.assertEqual(HARD_RULES_CONFIG.get("enable_deliver_prob_gate"), saved["deliver_prob"])
        self.assertEqual(HARD_RULES_CONFIG.get("enable_class_aware_gate"), saved["class_aware"])
        self.assertEqual(HARD_RULES_CONFIG.get("enable_tx_high_reserve"), saved["tx_reserve"])
        self.assertEqual(HARD_RULES_CONFIG.get("enable_layered_edf"), saved["edf"])
        self.assertEqual(PROPULSION_CONTROLLER_CONFIG.get("guard_only"), saved["guard_only"])
        self.assertEqual(ACTUATOR_GATE_CONFIG.get("cpu_gate_soft_mode"), saved["cpu_soft"])
        self.assertEqual(TASK_CONFIG.get("cpu_action_is_admissible_budget"), saved["cpu_budget"])
        self.assertEqual(TASK_CONFIG.get("enable_cpu_throttle"), saved["cpu_throttle"])
        self.assertEqual(TASK_CONFIG.get("enable_high_value_cpu_gate_escape"), saved["high_escape"])

    def test_env_safety_layer_overrides_can_enable_full_hard_rule_shell(self):
        """Evaluation attribution must start from an explicit old hard-rule shell."""
        from config import (
            ACTUATOR_GATE_CONFIG,
            HARD_RULES_CONFIG,
            PROPULSION_CONTROLLER_CONFIG,
            TASK_CONFIG,
        )
        from evaluate_optimized import env_safety_layer_overrides

        with env_safety_layer_overrides(enable_hard_rule_shell=True):
            self.assertTrue(PROPULSION_CONTROLLER_CONFIG["enabled"])
            self.assertFalse(PROPULSION_CONTROLLER_CONFIG["guard_only"])
            self.assertFalse(ACTUATOR_GATE_CONFIG["cpu_gate_soft_mode"])
            for key in [
                "cpu_action_is_admissible_budget",
                "enable_future_contact_cpu_gate",
                "enable_cpu_throttle",
                "enable_high_value_cpu_gate_escape",
                "enable_in_window_cpu_feed_floor",
            ]:
                self.assertTrue(TASK_CONFIG[key])
            for key in [
                "enable_mission_pointing_fallback",
                "enable_in_window_tx_floor",
                "enable_class_priority_floor",
                "enable_deliver_prob_gate",
                "enable_class_aware_gate",
                "enable_tx_high_reserve",
                "enable_layered_edf",
            ]:
                self.assertTrue(HARD_RULES_CONFIG[key])

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
            DISPLAY_ORDER[-1],
            "H_With_AP_BC",
        )
        self.assertGreater(
            VARIANT_SPECS["H_With_AP_BC"]["behavior_cloning_coeff"],
            VARIANT_SPECS["A_Full"]["behavior_cloning_coeff"],
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
        TestAtmosphereSwitchingAndSpaceWeather,
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

"""
Drift-Plus-Penalty 传统优化基线。
Drift-Plus-Penalty（DPP）传统优化基线

该基线不训练网络，而是在每个控制步枚举少量动作，选择“一步李雅普诺夫漂移小、
有效下传收益高”的动作。它比静态阈值/启发式更接近传统随机网络优化口径，
可作为冲击高水平期刊时更有说服力的非学习基线。
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    # 降低导入劫持风险：不要把项目路径插到最高优先级
    sys.path.append(_PROJECT_ROOT)

import numpy as np

from config import ENERGY_CONFIG, ORBITAL_CONFIG, QUEUE_CONFIG, TRAIN_CONFIG


class DriftPlusPenaltyBaseline:
    """
    一步 Drift-Plus-Penalty 调度器。

    目标近似为：
        min ΔL(t) - V * delivered_value(t) - beta * prepared_value(t)

    其中 L(t) 使用与项目 LyapunovActionProjection 一致的归一化四队列形式。
    """

    def __init__(self,
                 action_levels=None,
                 V: float = 8.0,
                 processed_weight: float = 0.25,
                 power_penalty: float = 0.005,
                 dt_s: float = None):
        self.action_levels = action_levels or [0.0, 0.25, 0.5, 0.75, 1.0]
        self.V = float(V)
        self.processed_weight = float(processed_weight)
        self.power_penalty = float(power_penalty)
        self.dt_s = float(dt_s if dt_s is not None else TRAIN_CONFIG["time_slot_s"])
        self.dt_h = self.dt_s / 3600.0

        self.energy_queue_max = QUEUE_CONFIG.get("energy_queue_max", 100.0)
        self.orbit_queue_max = QUEUE_CONFIG.get("orbit_queue_max", 100.0)
        self.data_queue_max = QUEUE_CONFIG.get("data_queue_max_mb", 500.0)
        self.comm_queue_max = QUEUE_CONFIG.get("comm_queue_max", 500.0)
        self.h_min_m = ORBITAL_CONFIG["altitude_min_km"] * 1e3

        self.soc_min = ENERGY_CONFIG["battery_min_soc"]
        self.soc_max = ENERGY_CONFIG["battery_max_soc"]

        self.action_candidates = [
            np.array([ap, ac, at], dtype=np.float32)
            for ap in self.action_levels
            for ac in self.action_levels
            for at in self.action_levels
        ]

    def schedule(self, state: np.ndarray, env) -> np.ndarray:
        """
        根据当前环境状态返回动作 [alpha_prop, alpha_cpu, alpha_tx]。

        参数 state 保留用于统一接口，主要物理量直接从 env 读取，避免状态归一化反推误差。
        """
        del state
        best_action = np.array([0.3, 0.2, 0.2], dtype=np.float32)
        best_score = -np.inf

        # value_density 只依赖 env.task_tracker / step_count，与候选 action 无关，
        # 而 topk_stats 对上千个活跃批次排序极慢；在候选循环外只算一次（严格等价），
        # 避免每步对 125 个候选重复调用 topk_stats（原实现使 DPP 慢约 100×）。
        value_density = self._estimate_value_density(env)

        for action in self.action_candidates:
            score = self._score_action(action, env, value_density)
            if score > best_score:
                best_score = score
                best_action = action.copy()

        return best_action

    def _score_action(self, action: np.ndarray, env,
                      value_density: float) -> float:
        # 传入真实的 in_window / 紧急状态，确保 DPP 评分使用的功率切片与 env.step
        # 实际执行的 strict_priority 切片一致，避免 "评分按比例分、执行按优先级分"
        # 的语义错位（旧版按比例 fallback 会让 DPP 选到"看起来好但实际不一样"的动作）。
        contact = getattr(env, "_contact", None) or {}
        in_window = bool(contact.get("in_window", False))
        h_warning = getattr(env, "_h_warning", None)
        altitude_m = getattr(env, "altitude_m", None)
        force_prop_priority = bool(
            altitude_m is not None and h_warning is not None
            and float(altitude_m) <= float(h_warning))
        power_info = env.power_sys.compute_total_load(
            action,
            in_window=in_window,
            force_prop_priority=force_prop_priority,
        )
        p_total = float(power_info["P_total_w"])
        p_cpu = float(power_info["P_cpu_w"])
        p_prop = float(power_info["P_propulsion_w"])
        p_tx = float(power_info["P_tx_w"])

        q_e = float(env.energy_queue.value)
        q_h = float(env.orbit_queue.value)
        q_d = float(env.data_queue.length)
        q_c = float(env.comm_queue.value)

        processed_mb = self._predict_processed_mb(env, p_cpu)
        tx_mb = self._predict_downlink_mb(env, action[2], q_c + processed_mb, p_tx)
        delivered_value = tx_mb * value_density
        prepared_value = processed_mb * value_density

        q_e_next = self._predict_energy_queue(env, q_e, p_total)
        q_h_next = self._predict_orbit_queue(env, q_h, p_prop)
        q_d_next = self._predict_data_queue(env, q_d, processed_mb)
        q_c_next = max(q_c + processed_mb - tx_mb, 0.0)

        drift = self._lyapunov(q_e_next, q_h_next, q_d_next, q_c_next) - \
            self._lyapunov(q_e, q_h, q_d, q_c)

        # 软安全边界：队列只有越界后才变大，这里额外惩罚“即将越界”的动作。
        energy_margin = self._predict_soc(env, p_total) - self.soc_min
        orbit_margin_km = (self._predict_altitude(env, p_prop) - self.h_min_m) / 1e3
        safety_penalty = 120.0 * max(0.0, 0.05 - energy_margin) ** 2
        safety_penalty += 8.0 * max(0.0, 8.0 - orbit_margin_km) ** 2

        return (
            self.V * delivered_value
            + self.processed_weight * prepared_value
            - drift
            - self.power_penalty * p_total
            - safety_penalty
        )

    def _estimate_value_density(self, env) -> float:
        tracker = getattr(env, "task_tracker", None)
        if tracker is None:
            return 1.0
        stats = tracker.topk_stats(getattr(env, "step_count", 0))
        priority = float(stats.get("top_task_priority", 1.0) or 1.0)
        quality = float(stats.get("top_task_quality", 1.0) or 1.0)
        urgency = float(stats.get("deadline_urgency", 0.0) or 0.0)
        return max(0.1, priority * quality * (1.0 + 0.5 * urgency))

    def _predict_processed_mb(self, env, p_cpu_w: float) -> float:
        expected_arrival = self._expected_data_arrival(env)
        service_capacity = env.power_sys.throughput_rate(p_cpu_w) * self.dt_s
        backlog = float(env.data_queue.length) + expected_arrival
        return float(np.clip(min(service_capacity, backlog), 0.0, backlog))

    def _predict_downlink_mb(self, env, alpha_tx: float,
                             available_mb: float,
                             p_tx_w: float) -> float:
        contact = getattr(env, "_contact", None) or {}
        in_window = bool(contact.get("in_window", False))
        if not in_window:
            return 0.0
        capacity_mbps = float(contact.get("max_capacity_mbps", 0.0))
        capacity_mb = max(0.0, capacity_mbps * self.dt_s / 8.0)
        link_limited_mb = float(alpha_tx) * capacity_mb
        # 基线内部预测必须和环境一致：实际下传同时受链路容量和发射机 RF 物理速率限制。
        if hasattr(env, "power_sys") and hasattr(env.power_sys, "tx_downlink_rate"):
            rf_limited_mb = env.power_sys.tx_downlink_rate(p_tx_w) * self.dt_s
        else:
            rf_rate_max = float(QUEUE_CONFIG.get("tx_downlink_rate_max_mbs", 5.0))
            rf_limited_mb = max(float(alpha_tx), 0.0) * rf_rate_max * self.dt_s
        return float(np.clip(min(link_limited_mb, rf_limited_mb), 0.0, available_mb))

    def _expected_data_arrival(self, env) -> float:
        base = QUEUE_CONFIG["data_arrival_rate_mbs"] * self.dt_s
        scale = float(getattr(env, "_data_arrival_scale", 1.0))
        phase = float(getattr(env.orbit_sim, "phase", 0.0))
        period_phase = phase / (2.0 * np.pi)
        hot = 1.0 + 2.0 * np.exp(-((period_phase * 90.0 - 30.0) ** 2) / (2.0 * 20.0 ** 2))
        return float(max(base * scale * hot, 0.0))

    def _predict_soc(self, env, p_total_w: float) -> float:
        sunlit_fraction = env.orbit_sim.sunlit_fraction(
            time_s=getattr(env, "time_s", None),
            altitude_m=getattr(env, "altitude_m", None),
        )
        p_solar = env.solar.output_power(sunlit_fraction)
        p_net = p_solar - p_total_w
        if p_net >= 0:
            delta_wh = p_net * self.dt_h * env.battery.eta_charge
        else:
            delta_wh = p_net * self.dt_h / env.battery.eta_discharge
        new_energy = env.battery.energy_wh + delta_wh
        return float(np.clip(new_energy / env.battery.capacity_wh, 0.0, self.soc_max))

    def _predict_energy_queue(self, env, q_e: float, p_total_w: float) -> float:
        soc_next = self._predict_soc(env, p_total_w)
        energy_margin_wh = (
            soc_next * env.battery.capacity_wh
            - env.battery.soc_min * env.battery.capacity_wh
        )
        return float(np.clip(max(q_e - energy_margin_wh, 0.0), 0.0, self.energy_queue_max))

    def _predict_altitude(self, env, p_prop_w: float) -> float:
        orbit_info = env.orbit_dyn.step(env.altitude_m, p_prop_w, self.dt_s)
        return float(orbit_info["altitude_m"])

    def _predict_orbit_queue(self, env, q_h: float, p_prop_w: float) -> float:
        altitude_next = self._predict_altitude(env, p_prop_w)
        altitude_margin_km = (altitude_next - self.h_min_m) / 1e3
        return float(np.clip(max(q_h - altitude_margin_km, 0.0), 0.0, self.orbit_queue_max))

    def _predict_data_queue(self, env, q_d: float, processed_mb: float) -> float:
        next_qd = q_d + self._expected_data_arrival(env) - processed_mb
        return float(np.clip(max(next_qd, 0.0), 0.0, self.data_queue_max))

    def _lyapunov(self, q_e: float, q_h: float, q_d: float, q_c: float) -> float:
        return 0.5 * (
            (q_e / max(self.energy_queue_max, 1e-6)) ** 2
            + (q_h / max(self.orbit_queue_max, 1e-6)) ** 2
            + (q_d / max(self.data_queue_max, 1e-6)) ** 2
            + (q_c / max(self.comm_queue_max, 1e-6)) ** 2
        )

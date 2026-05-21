"""
多扰动场景鲁棒 MPC 基线。
鲁棒 MPC 基线

与普通 MPC 不同，Robust MPC 要求同一个候选动作在多组扰动场景下都保持安全：
  - 高大气密度：轨道衰减更强
  - 低太阳能输入：能量更紧张

该基线不使用学习参数，适合作为一区论文中比普通 MPC 更强的传统优化对照。
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    # 降低导入劫持风险：不要把项目路径插到最高优先级
    sys.path.append(_PROJECT_ROOT)

import numpy as np

from baselines.mpc_baseline import MPCBaseline


class RobustMPCBaseline(MPCBaseline):
    """
    多场景鲁棒模型预测控制。

    schedule() 会枚举普通 MPC 的动作网格，并对每个动作做 scenario set 检查。
    只有所有场景均满足 SOC/高度安全约束时，动作才被视为可行。
    """

    def __init__(self,
                 horizon: int = 8,
                 dt_s: float = 10.0,
                 density_scales=None,
                 solar_scales=None,
                 risk_penalty: float = 0.15):
        super().__init__(horizon=horizon, dt_s=dt_s)
        self.density_scales = density_scales or [1.0, 1.3, 1.6]
        self.solar_scales = solar_scales or [1.0, 0.75, 0.50]
        self.risk_penalty = float(risk_penalty)

    def schedule(self, state: np.ndarray,
                 soc: float, altitude_m: float,
                 sunlit: bool, P_solar: float,
                 time_s: float = 0.0,
                 env=None) -> np.ndarray:
        best_action = np.array([0.5, 0.1, 0.1], dtype=np.float32)
        best_score = -np.inf
        self.sync_from_env(env)

        for action in self.action_candidates:
            # Robust MPC 的候选动作评分仍需要当前观测中的队列/任务价值信息，
            # 因此不能丢弃 state；它会继续传入 _value_score。
            score, feasible = self._evaluate_action_robust(
                action, soc, altitude_m, sunlit, P_solar, time_s, state)
            if feasible and score > best_score:
                best_score = score
                best_action = action.copy()

        return best_action

    def _evaluate_action_robust(self, action: np.ndarray,
                                soc0: float, alt0: float,
                                sunlit0: bool, P_solar0: float,
                                time_s: float,
                                state: np.ndarray) -> tuple:
        scenario_scores = []
        scenario_margins = []

        for density_scale in self.density_scales:
            for solar_scale in self.solar_scales:
                score, feasible, min_soc_margin, min_h_margin_km = self._evaluate_action_scenario(
                    action, soc0, alt0, sunlit0, P_solar0,
                    time_s, float(density_scale), float(solar_scale), state)
                if not feasible:
                    return score, False
                scenario_scores.append(score)
                scenario_margins.append(min(min_soc_margin * 100.0, min_h_margin_km))

        # 鲁棒目标使用最坏场景吞吐，并轻微奖励安全裕度，避免选到贴边动作。
        worst_score = float(np.min(scenario_scores)) if scenario_scores else -np.inf
        worst_margin = float(np.min(scenario_margins)) if scenario_margins else 0.0
        return worst_score + self.risk_penalty * worst_margin, True

    def _evaluate_action_scenario(self, action: np.ndarray,
                                  soc0: float, alt0: float,
                                  sunlit0: bool, P_solar0: float,
                                  time_s: float,
                                  density_scale: float,
                                  solar_scale: float,
                                  state: np.ndarray) -> tuple:
        ap, ac, at = action
        P_prop = ap * self.P_prop_max
        P_cpu = ac * self.P_cpu_max
        P_tx = at * self.P_tx_max
        P_load = P_prop + P_cpu + P_tx + self.P_base

        soc = float(soc0)
        alt = float(alt0)
        total_tput = 0.0
        min_soc_margin = float("inf")
        min_h_margin_km = float("inf")

        # 用从 env.orbit_sim 同步过来的 eclipse_fraction（每 episode β 角随机化），
        # 而不是硬编码 35min 阴影。这保证 robust MPC 在不同 β 角 episode 上的预测一致。
        from config import ORBITAL_CONFIG as _OC
        current_global_step = int(time_s / self.dt)
        orbital_period_s = float(_OC["orbital_period_min"]) * 60.0
        orbit_steps = max(1, int(orbital_period_s / max(self.dt, 1e-6)))
        sunlit_steps = max(0, int(orbit_steps * (1.0 - float(self.eclipse_fraction))))

        for step in range(self.horizon):
            min_soc_margin = min(min_soc_margin, soc - self.soc_min)
            min_h_margin_km = min(min_h_margin_km, (alt - self.h_min) / 1e3)

            if soc < self.soc_min + 0.02:
                return total_tput, False, min_soc_margin, min_h_margin_km
            if alt < self.h_min + 2e3:
                return total_tput, False, min_soc_margin, min_h_margin_km

            phase = (current_global_step + step) % orbit_steps
            if phase >= sunlit_steps:
                sunlit = False
                P_solar = 0.0
            else:
                sunlit = True
                P_solar = P_solar0 * solar_scale

            soc = self._predict_soc(soc, P_solar, P_load)
            alt = self._predict_altitude_with_density(alt, P_prop, density_scale)
            total_tput += self._value_score(state, P_cpu, P_tx)

        return total_tput, True, min_soc_margin, min_h_margin_km

    def _predict_altitude_with_density(self,
                                       altitude_m: float,
                                       P_prop: float,
                                       density_scale: float) -> float:
        r = self.R_e + altitude_m
        v_orbit = np.sqrt(self.mu / r)
        if self.enable_atmospheric_corotation:
            v_rel = max(v_orbit - self._omega_earth * r * np.cos(self.inclination_rad), 0.0)
        else:
            v_rel = v_orbit
        n = np.sqrt(self.mu / r**3)
        rho = self._density(altitude_m, density_scale=density_scale)
        F_d = 0.5 * self.drag_cd * self.drag_area_m2 * rho * v_rel * v_rel
        thrust = P_prop * 0.65 / (1000 * 9.80665)
        dh = (2 * (thrust - F_d) / (self.satellite_mass_kg * n)) * self.dt
        return float(np.clip(altitude_m + dh, self.h_crash, 450e3))

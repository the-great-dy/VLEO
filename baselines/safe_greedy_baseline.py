"""Safe Greedy 规则调度基线（强公平对照，不依赖 RL）。

设计目的：回应"传统 baseline 是否被不公平地评估"的质疑。Safe Greedy 是一个
精心设计的安全优先规则调度器，运行在与主方法**完全相同**的部署壳
（SAFE_BUDGET + credit gate）、任务序列、通信窗口、能源模型、seeds 下。
它把"显然正确"的工程规则都写进去，作为"规则能做到多好"的强上界对照。

规则（安全优先级硬序）：
  P0 轨道保命：高度 < warning → 全力推进；否则按 (nominal-h) 高度差悬停推进。
  P1 能量保护：SOC < hard_reserve 且高度安全 → 压低推进，把功率让给充电。
  P2 充电：SOC < soft_reserve → 对日(SUN) 充电，切 CPU/TX。
  P3 下传：在通信窗口且 onboard 有数据 → DOWNLINK，吃满 TX。
  P4 成像/处理：不在窗口且有高价值任务 → IMAGE（credit 不足时由 env credit gate 自动遮罩）。
  P5 否则：对日充电/低负载待命。
所有动作再经同一套 SAFE_BUDGET + credit gate（env 级）过滤。
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

import numpy as np
from config import ENERGY_CONFIG, ORBITAL_CONFIG
from environment.satellite_env import OBSERVATION_FEATURES
from utils.action_space import (
    default_grouped_action, pointing_unit_for_mode,
    POINTING_SUN, POINTING_IMAGE, POINTING_DOWNLINK,
    CPU_LOGITS_SLICE, TX_LOGITS_SLICE,
    IDX_CPU_VALUE_WEIGHT, IDX_CPU_URGENCY_WEIGHT,
    IDX_TX_VALUE_WEIGHT, IDX_TX_URGENCY_WEIGHT,
)

_FEATURE_INDEX = {name: idx for idx, name in enumerate(OBSERVATION_FEATURES)}


class SafeGreedyBaseline:
    """安全优先的贪心规则调度器（配合 SAFE_BUDGET + credit gate 部署壳）。"""

    def __init__(self, soft_reserve: float = 0.50, hard_reserve: float = 0.35):
        self.soft_reserve = float(soft_reserve)
        self.hard_reserve = float(hard_reserve)
        self.h_min = float(ORBITAL_CONFIG["altitude_min_km"])
        self.h_warn = float(ORBITAL_CONFIG.get("altitude_warning_km", self.h_min + 20.0))
        self.h_nom = float(ORBITAL_CONFIG["altitude_nominal_km"])
        self.h_max = float(ORBITAL_CONFIG["altitude_max_km"])

    def _f(self, state, key, default=0.0):
        i = _FEATURE_INDEX.get(key)
        return float(state[i]) if (i is not None and i < state.shape[0]) else float(default)

    def schedule(self, state: np.ndarray, env=None) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        soc = self._f(state, "soc", 1.0)
        h_km = self.h_min + self._f(state, "altitude_norm") * (self.h_max - self.h_min)
        in_window = self._f(state, "in_comm_window") > 0.5
        sunlit = self._f(state, "solar_input_norm") > 0.05

        # onboard 数据 / 高价值存在
        raw_u = max(self._f(state, "raw_high_queue_utilization"),
                    self._f(state, "raw_mid_queue_utilization"),
                    self._f(state, "raw_low_queue_utilization"))
        proc_u = max(self._f(state, "processed_high_queue_utilization"),
                     self._f(state, "processed_mid_queue_utilization"),
                     self._f(state, "processed_low_queue_utilization"))
        onboard = (raw_u + proc_u) > 0.01
        proc_ready = proc_u > 0.005
        high_value = (self._f(state, "expiring_high_value_norm") > 0.02
                      or self._f(state, "raw_high_queue_utilization") > 0.02
                      or self._f(state, "processed_high_queue_utilization") > 0.02)

        # ── P0 轨道保命 + 高度悬停推进 ──
        if h_km < self.h_warn:
            prop = 1.0
        elif h_km < self.h_nom:
            # warning~nominal 之间线性悬停推进（越低推越多）
            prop = float(np.clip((self.h_nom - h_km) / max(self.h_nom - self.h_warn, 1e-6), 0.15, 1.0))
        else:
            prop = 0.10  # 标称以上滑行，省燃料/能量
        # P1 能量保护：高度安全但 SOC 低 → 压推进，让充电主导（SAFE_BUDGET 会指向 SUN）
        if soc < self.hard_reserve and h_km >= self.h_warn:
            prop = min(prop, 0.15)

        # ── 指向 + 数据决策（安全优先级序）──
        if soc < self.soft_reserve:
            mode, cpu, tx = POINTING_SUN, 0.05, 0.0          # P2 充电
        elif in_window and onboard:
            mode, cpu, tx = POINTING_DOWNLINK, 0.30, 1.0     # P3 下传优先
        elif (not in_window) and high_value:
            mode, cpu, tx = POINTING_IMAGE, 0.85, 0.0        # P4 成像/处理高价值
        elif proc_ready and in_window:
            mode, cpu, tx = POINTING_DOWNLINK, 0.20, 1.0
        else:
            mode, cpu, tx = POINTING_SUN, 0.10, 0.05         # P5 待命/充电

        action = default_grouped_action(
            [prop, cpu, tx], pointing_unit=pointing_unit_for_mode(mode))
        # class 优先级 high>mid>low（与 value-aware heuristic 同口径）
        prio = np.array([1.0, 0.55, 0.15], dtype=np.float32)
        action[CPU_LOGITS_SLICE] = prio
        action[TX_LOGITS_SLICE] = prio
        action[IDX_CPU_VALUE_WEIGHT] = 1.0
        action[IDX_CPU_URGENCY_WEIGHT] = float(np.clip(self._f(state, "expiring_high_value_norm"), 0.0, 1.0))
        action[IDX_TX_VALUE_WEIGHT] = 1.0
        action[IDX_TX_URGENCY_WEIGHT] = float(np.clip(self._f(state, "expiring_high_value_norm"), 0.0, 1.0))
        return action

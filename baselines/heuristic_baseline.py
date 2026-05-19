"""
规则启发式与价值感知启发式基线。
规则启发式基线：基于日照预测的分阶段能量管理策略

策略逻辑（模拟航天工程师手工设计的规则）：
  ├─ 阶段1 日照充裕期：高通信 + 适量推进
  ├─ 阶段2 通信窗口尾声：提前降 Tx，降低无效空耗
  ├─ 阶段3 阴影区：低功耗，仅维持轨道
  ├─ 阶段4 轨道偏低紧急推进：全力推进
  └─ 阶段5 电量告急：切断所有非必要负载
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    # 降低导入劫持风险：不要把项目路径插到最高优先级
    sys.path.append(_PROJECT_ROOT)

import numpy as np
from config import ENERGY_CONFIG, ORBITAL_CONFIG
from environment.satellite_env import OBSERVATION_FEATURES
from utils.action_space import default_grouped_action


_FEATURE_INDEX = {name: idx for idx, name in enumerate(OBSERVATION_FEATURES)}


class HeuristicBaseline:
    """
    日照感知规则启发式基线
    代表传统航天器能量管理中"专家规则"方法
    """

    def __init__(self):
        self.soc_min      = ENERGY_CONFIG["battery_min_soc"]
        self.soc_crash    = ENERGY_CONFIG.get("battery_crash_soc", 0.05)
        self.h_min_km     = ORBITAL_CONFIG["altitude_min_km"]
        self.h_warning_km = ORBITAL_CONFIG.get("altitude_warning_km", self.h_min_km + 30.0)
        self.h_nominal_km = ORBITAL_CONFIG["altitude_nominal_km"]

    def schedule(self, state: np.ndarray) -> np.ndarray:
        """
        基于观测向量的规则决策。

        当前观测按 OBSERVATION_FEATURES 的 40 维顺序解释，避免状态扩展后
        硬编码下标继续读错语义。
        """
        state = np.asarray(state, dtype=np.float32)
        required_dim = len(OBSERVATION_FEATURES)
        if state.ndim != 1 or state.shape[0] < required_dim:
            raise ValueError(
                f"HeuristicBaseline 只支持当前 {required_dim} 维当前观测状态，"
                f"实际形状为 {state.shape}"
            )

        h_norm = float(state[_FEATURE_INDEX["altitude_norm"]])
        soc = float(state[_FEATURE_INDEX["soc"]])
        P_solar_norm = float(state[_FEATURE_INDEX["solar_input_norm"]])
        sunlit = P_solar_norm > 0.05
        in_comm_window = float(state[_FEATURE_INDEX["in_comm_window"]]) > 0.5
        window_remaining_norm = float(state[_FEATURE_INDEX["window_remaining_norm"]])
        window_ending_soon = in_comm_window and window_remaining_norm < 0.10

        # 恢复实际高度
        h_min  = ORBITAL_CONFIG["altitude_min_km"]
        h_max  = ORBITAL_CONFIG["altitude_max_km"]
        h_km   = h_min + h_norm * (h_max - h_min)

        # ── 阶段判断 ──────────────────────────────────────────────
        orbit_critical  = h_km < h_min           # 轨道不安全（<150km）
        orbit_warning   = h_km < self.h_warning_km # 轨道警告区（150~180km）
        energy_critical = soc <= (self.soc_crash + 0.02) # 接近能源终止线
        energy_warning  = soc < self.soc_min     # 电量警告区（5%~15%）

        # ── 规则1：轨道严重偏低 → 全力推进，切断通信 ──────────────
        # 规则基线按“保命优先”排序：轨道/能量危险优先级高于吞吐和通信窗口利用。
        if orbit_critical:
            return default_grouped_action([1.0, 0.0, 0.0])

        # ── 规则2：电量告急 → 仅维持最低功耗 ─────────────────────
        if energy_critical:
            prop = 0.3 if orbit_warning else 0.1
            return default_grouped_action([prop, 0.0, 0.0])

        # ── 规则3：通信窗口即将结束 → 降低 Tx，避免窗口后继续空耗 ───────
        if window_ending_soon:
            # 当前观测没有“距进入阴影”字段，因此这里只按通信窗口尾声保守降负载。
            prop = 0.4 if orbit_warning else 0.25
            return default_grouped_action([prop, 0.1, 0.1])

        # ── 规则4：阴影区 → 低功耗维持 ──────────────────────────
        if not sunlit:
            prop = 0.6 if orbit_warning else 0.35
            return default_grouped_action([prop, 0.05, 0.05])


        # ── 规则5：日照充裕 + 电量偏低 → 充电优先 ────────────────
        if energy_warning:
            prop = 0.35 if orbit_warning else 0.2
            return default_grouped_action([prop, 0.2, 0.2])

        # ── 规则6：正常日照区 → 均衡运行 ────────────────────────
        # 根据太阳能强度动态调整通信功率
        comm_ratio = min(0.85, 0.5 + P_solar_norm * 0.35)
        prop = 0.35 if orbit_warning else 0.25
        return default_grouped_action([prop, comm_ratio, comm_ratio * 0.9])


class ValueAwareHeuristicBaseline(HeuristicBaseline):
    """
    基于价值感知的启发式基线
    在传统规则基础上，增加对低价值任务的主动丢弃逻辑，
    当未来通信容量不足以支撑队列积压时触发丢弃。
    """
    def schedule(self, state: np.ndarray) -> np.ndarray:
        action = super().schedule(state)
        
        state = np.asarray(state, dtype=np.float32)
        future_capacity = float(state[_FEATURE_INDEX["future_contact_capacity_norm"]])
        raw_high = float(state[_FEATURE_INDEX["raw_high_queue_utilization"]])
        raw_mid = float(state[_FEATURE_INDEX["raw_mid_queue_utilization"]])
        proc_high = float(state[_FEATURE_INDEX["processed_high_queue_utilization"]])
        proc_mid = float(state[_FEATURE_INDEX["processed_mid_queue_utilization"]])
        
        # 简单估算队列压力（包含未处理和已处理的低价值任务）
        raw_low = float(state[_FEATURE_INDEX["raw_low_queue_utilization"]])
        proc_low = float(state[_FEATURE_INDEX["processed_low_queue_utilization"]])
        priority_logits = np.array([1.0, 0.55, 0.15], dtype=np.float32)
        expiring_high = float(state[_FEATURE_INDEX["expiring_high_value_norm"]])
        expiring_mid = float(state[_FEATURE_INDEX["expiring_mid_value_norm"]])
        expiring_low = float(state[_FEATURE_INDEX["expiring_low_value_norm"]])
        cpu_pressure = np.array([raw_high, raw_mid, raw_low], dtype=np.float32)
        tx_pressure = np.array([proc_high, proc_mid, proc_low], dtype=np.float32)
        expiring = np.array([expiring_high, expiring_mid, expiring_low], dtype=np.float32)
        cpu_value = float(np.clip(np.dot(priority_logits, cpu_pressure), 0.0, 1.0))
        cpu_urgency = float(np.clip(np.dot(priority_logits, expiring), 0.0, 1.0))
        tx_value = float(np.clip(np.dot(priority_logits, tx_pressure), 0.0, 1.0))
        tx_urgency = float(np.clip(np.dot(priority_logits, expiring), 0.0, 1.0))
        action[3] = cpu_value
        action[4] = cpu_urgency
        action[5] = tx_value
        action[6] = tx_urgency
        low_backlog = raw_low + proc_low
        
        raw_util = float(state[_FEATURE_INDEX["raw_queue_utilization"]])
        proc_util = float(state[_FEATURE_INDEX["processed_queue_utilization"]])
        queue_pressure = max(raw_util, proc_util)
        
        drop_strength = 0.0
        
        # 如果队列压力大，或者未来通信容量严重不足且有低优任务积压，触发丢弃
        if queue_pressure > 0.45:
            drop_strength = (queue_pressure - 0.45) / 0.55
        elif low_backlog > 0.1 and future_capacity < 0.5:
            drop_strength = min(1.0, (0.5 - future_capacity) * 2.0)
            
        action[7] = drop_strength
        return action

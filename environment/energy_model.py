"""
卫星能量系统、功耗分配和电池状态模型。
VLEO卫星能量系统模型
包含: 太阳能板模型、电池充放电模型、功耗管理
"""
import sys as _sys, os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in _sys.path:
    # 降低导入劫持风险：不要把项目路径插到最高优先级
    _sys.path.append(_PROJECT_ROOT)
# _PATH_INJECTED_


import numpy as np
from config import ENERGY_CONFIG, QUEUE_CONFIG
from safety.power_manager import (
    apply_propulsion_deadband_watts, priority_order, priority_label)
from utils.sanitizers import sanitize_action
from utils.action_space import PHYSICAL_ACTION_DIM


class SolarPanelModel:
    """太阳能板模型: 根据日照状态计算功率输出"""

    def __init__(self):
        self.P_max = ENERGY_CONFIG["solar_panel_power_w"]
        self.eta = ENERGY_CONFIG["solar_efficiency"]
        self.eta_nominal = max(float(ENERGY_CONFIG["solar_efficiency"]), 1e-9)

    def output_power(self, sunlit_fraction: float,
                     degradation_factor: float = 1.0) -> float:
        """
        太阳能板输出功率 (W)
        Args:
            sunlit_fraction: 日照强度归一化因子 [0, 1]
            degradation_factor: 太阳能板退化因子 (寿命衰减)
        """
        # solar_panel_power_w 表示标称效率下的阵列峰值发电输出。
        # 扰动实验会修改 eta；这里按 eta/eta_nominal 做相对退化，避免标称场景重复乘效率。
        eta_scale = max(float(self.eta), 0.0) / self.eta_nominal
        P_solar = self.P_max * eta_scale * sunlit_fraction * degradation_factor
        return max(P_solar, 0.0)


class BatteryModel:
    """
    锂离子电池模型
    状态: SOC (State of Charge), 单位 [0, 1]
    能量单位: Wh
    """

    def __init__(self):
        self.nominal_capacity_wh = float(ENERGY_CONFIG["battery_capacity_wh"])
        self.capacity_wh = self.nominal_capacity_wh
        self.soc_min = ENERGY_CONFIG["battery_min_soc"]
        self.soc_crash = ENERGY_CONFIG.get("battery_crash_soc", 0.05)
        self.soc_max = ENERGY_CONFIG["battery_max_soc"]
        self.degradation_enabled = bool(
            ENERGY_CONFIG.get("battery_cycle_degradation_enabled", True))
        self.capacity_loss_per_efc = float(
            ENERGY_CONFIG.get("battery_capacity_loss_per_efc", 0.0))
        self.max_degradation_fraction = float(np.clip(
            ENERGY_CONFIG.get("battery_degradation_max_fraction", 0.0),
            0.0, 0.95,
        ))
        self.equivalent_full_cycles = 0.0
        self.cycle_degradation = 0.0

        # 充放电效率
        self.eta_charge = 0.95
        self.eta_discharge = 0.95

        # 初始化
        self.soc = 0.80  # 初始SOC
        
        # 内部 RNG 默认固定种子，可通过 set_rng() 与环境级 RNG 对齐。
        self._rng = np.random.default_rng(42)

    def set_rng(self, rng: np.random.Generator):
        """
        设置外部可控随机数生成器，确保实验可复现。
        应在环境 __init__ 中调用，将环境级 RNG 传入。
        """
        self._rng = rng

    @property
    def energy_wh(self) -> float:
        """当前储存电量 (Wh)"""
        return self.soc * self.capacity_wh

    @property
    def usable_energy_wh(self) -> float:
        """可用电量 (高于最低安全线的部分)"""
        return max(0.0, self.energy_wh - self.soc_min * self.capacity_wh)

    @property
    def energy_margin_wh(self) -> float:
        """距最低安全线的裕度 (Wh) - 用于虚拟队列"""
        return self.energy_wh - self.soc_min * self.capacity_wh

    def _update_cycle_degradation(self, throughput_wh: float) -> float:
        """
        按等效完整循环(EFC)累计容量老化。

        EFC 使用充放电吞吐量估计：一次完整放空再充满约等于
        2 * nominal_capacity_wh 的能量吞吐。这里不做复杂 rainflow 计数，
        但能让频繁微循环在长仿真中逐步反映为可用容量下降。
        """
        if (not self.degradation_enabled
                or self.capacity_loss_per_efc <= 0.0
                or throughput_wh <= 0.0):
            return 0.0

        previous_capacity = float(self.capacity_wh)
        efc_increment = throughput_wh / max(2.0 * self.nominal_capacity_wh, 1e-9)
        self.equivalent_full_cycles += float(max(efc_increment, 0.0))
        target_degradation = min(
            self.equivalent_full_cycles * self.capacity_loss_per_efc,
            self.max_degradation_fraction,
        )
        if target_degradation <= self.cycle_degradation + 1e-12:
            return 0.0

        self.cycle_degradation = float(target_degradation)
        self.capacity_wh = self.nominal_capacity_wh * (1.0 - self.cycle_degradation)
        return max(0.0, previous_capacity - self.capacity_wh)

    def classify_soc(self, soc: float) -> tuple[str, int]:
        """Return energy risk stage: normal, warning, or failure."""
        if soc <= self.soc_crash:
            return "failure", 3
        if soc < self.soc_min:
            return "warning", 1
        return "normal", 0

    def step(self, P_solar_w: float, P_load_w: float, dt_s: float) -> dict:
        """
        单时间步电池状态更新
        Args:
            P_solar_w: 太阳能输入功率 (W)
            P_load_w: 负载总功率 (W) [推进+CPU+发射机+平台]
            dt_s: 时间步长 (s)
        Returns:
            dict: 状态信息
        """
        dt_h = dt_s / 3600.0   # 转换为小时

        P_net = P_solar_w - P_load_w   # 净功率 (W)
        previous_energy_wh = self.energy_wh

        if P_net >= 0:
            # 充电
            delta_wh = P_net * dt_h * self.eta_charge
        else:
            # 放电
            delta_wh = P_net * dt_h / self.eta_discharge

        candidate_energy_wh = np.clip(
            previous_energy_wh + delta_wh,
            0.0,
            self.soc_max * self.capacity_wh,
        )
        throughput_wh = abs(candidate_energy_wh - previous_energy_wh)
        capacity_loss_wh = self._update_cycle_degradation(throughput_wh)

        # 容量老化后，若当前存量超过新的最大可用容量，需要同步裁剪到物理上限。
        stored_energy_wh = min(candidate_energy_wh, self.soc_max * self.capacity_wh)
        new_soc = stored_energy_wh / max(self.capacity_wh, 1e-9)
        new_soc = float(np.clip(new_soc, 0.0, self.soc_max))

        energy_stage, energy_stage_code = self.classify_soc(float(new_soc))
        is_safe = new_soc >= self.soc_min
        is_crashed = new_soc <= self.soc_crash
        is_critical = energy_stage != "normal"

        self.soc = new_soc

        return {
            "soc": self.soc,
            "energy_wh": self.energy_wh,
            "energy_margin_wh": self.energy_margin_wh,
            "P_net_w": P_net,
            "capacity_wh": self.capacity_wh,
            "nominal_capacity_wh": self.nominal_capacity_wh,
            "equivalent_full_cycles": self.equivalent_full_cycles,
            "cycle_degradation": self.cycle_degradation,
            "capacity_loss_wh": capacity_loss_wh,
            "is_safe": is_safe,
            "is_crashed": is_crashed,
            "is_warning": energy_stage == "warning",
            "is_critical": is_critical,
            "safety_stage": energy_stage,
            "safety_stage_code": energy_stage_code,
        }

    def reset(self, initial_soc: float = None):
        """重置电池状态"""
        self.capacity_wh = self.nominal_capacity_wh
        self.equivalent_full_cycles = 0.0
        self.cycle_degradation = 0.0
        if initial_soc is None:
            # 使用内部可控 RNG，避免 reset() 依赖全局随机状态。
            initial_soc = self._rng.uniform(0.5, 0.9)
        self.soc = np.clip(initial_soc, self.soc_min, self.soc_max)


class PowerSubsystem:
    """
    卫星功率子系统
    管理三类可调负载:
      1. 推进系统 (Propulsion)
      2. 计算单元 (CPU/OBC)
      3. 发射机 (Transmitter/RF)
    + 平台基础功耗 (不可调)
    """

    def __init__(self):
        self.P_prop_max = ENERGY_CONFIG["power_propulsion_max_w"]
        self.P_cpu_max = ENERGY_CONFIG["power_cpu_max_w"]
        self.P_tx_max = ENERGY_CONFIG["power_tx_max_w"]
        self.P_baseline = ENERGY_CONFIG["power_baseline_w"]
        self.prop_ignition_threshold_w = float(
            ENERGY_CONFIG.get("propulsion_ignition_threshold_w", 0.0))

    def _apply_propulsion_deadband(self, P_prop_w: float) -> tuple[float, bool]:
        """推进功率低于点火门限时按执行器关断处理，避免低功率空耗。"""
        return apply_propulsion_deadband_watts(
            P_prop_w,
            threshold_w=self.prop_ignition_threshold_w,
        )

    def compute_total_load(self, action: np.ndarray,
                           *,
                           in_window: bool = False,
                           force_prop_priority: bool = False) -> dict:
        """
        根据动作向量计算各子系统功耗。

        Args:
            action: 前三维为 [alpha_prop, alpha_cpu, alpha_tx]，
                    后续任务选择维度不参与功率计算
            in_window: 当前是否处于通信窗口（决定 strict priority 顺序）。
                      env.step 主路径下，此函数 receiver 的 action 已经被
                      `_enforce_available_power` 用 strict_priority 裁剪过，
                      此分支不会触发；in_window 仅在 baseline / 外部脚本
                      直接调用本函数时（如 DPP 评分）影响裁剪顺序。
            force_prop_priority: 紧急情况（altitude < warning）强制把推进系统
                      置于最高优先级。

        Returns:
            dict: 各子系统功耗及总功耗

        语义：当 P_prop + P_cpu + P_tx + P_baseline > power_total_max_w 时，
        按 strict priority 顺序裁剪（与 env.step 主路径的
        `allocate_power_strict_priority` 一致），不再按比例等比例缩。
        这保证 baseline / DPP 评分时算出来的功率切片和 env 实际执行一致，
        避免 "score 时按比例分、execute 时按优先级分" 的语义错位。
        """
        # 动作只表示三个可调负载的比例，基础平台功耗始终存在且不可关闭。
        # 功率模型也做一次非有限值兜底，防止外部脚本绕过环境直接调用时污染计算。
        clean_action, _, _ = sanitize_action(action, dtype=np.float64)
        if clean_action.size < PHYSICAL_ACTION_DIM:
            clean_action = np.pad(
                clean_action, (0, PHYSICAL_ACTION_DIM - clean_action.size))
        alpha_prop, alpha_cpu, alpha_tx = clean_action[:PHYSICAL_ACTION_DIM]

        P_prop = alpha_prop * self.P_prop_max
        P_cpu = alpha_cpu * self.P_cpu_max
        P_tx = alpha_tx * self.P_tx_max
        P_prop, prop_deadband_applied = self._apply_propulsion_deadband(P_prop)

        # 总功率上限模拟太阳能板或电源管理器的供电极限。
        P_max_total = ENERGY_CONFIG.get("power_total_max_w", 120.0)
        adjustable_budget = max(float(P_max_total) - self.P_baseline, 0.0)

        # Strict-priority 裁剪（与 allocate_power_strict_priority 同一套语义）：
        #   in_window  + force_prop_priority → prop > tx > cpu
        #   out window + force_prop_priority → prop > cpu > tx
        #   in_window  + normal              → tx > prop > cpu
        #   out window + normal              → prop > cpu > tx
        requested = np.array([P_prop, P_cpu, P_tx], dtype=np.float64)
        adjustable_request = float(requested.sum())
        if adjustable_request > adjustable_budget + 1e-9:
            order = priority_order(bool(in_window), bool(force_prop_priority))
            allocated = np.zeros(3, dtype=np.float64)
            remaining = adjustable_budget
            for idx in order:
                take = min(float(requested[idx]), max(remaining, 0.0))
                allocated[idx] = take
                remaining -= take
            P_prop, P_cpu, P_tx = float(allocated[0]), float(allocated[1]), float(allocated[2])
            P_prop, scaled_deadband = self._apply_propulsion_deadband(P_prop)
            prop_deadband_applied = bool(prop_deadband_applied or scaled_deadband)

        P_total = P_prop + P_cpu + P_tx + self.P_baseline

        return {
            "P_propulsion_w": P_prop,
            "P_cpu_w": P_cpu,
            "P_tx_w": P_tx,
            "P_baseline_w": self.P_baseline,
            "P_total_w": P_total,
            "alpha_prop": alpha_prop,
            "alpha_cpu": alpha_cpu,
            "alpha_tx": alpha_tx,
            "propulsion_ignition_active": bool(
                P_prop >= self.prop_ignition_threshold_w > 0.0),
            "propulsion_deadband_applied": bool(prop_deadband_applied),
            "power_priority_order": priority_label(
                bool(in_window), bool(force_prop_priority)),
        }

    def throughput_rate(self, P_cpu_w: float, P_tx_w: float = None) -> float:
        """
        CPU 在轨数据处理速率 (MB/s)

          解耦 CPU 与发射机：
          - CPU 处理速率只取决于 CPU 功率，与发射机状态无关
          - 卫星在阴影区关闭发射机（P_tx=0）时，CPU 照常处理数据并入队缓存，
            等待下次通信窗口再传；这正是"先计算存入队列，遇到地面站再下传"机制的物理基础。
          - 对地传输速率由地面站信道容量单独决定（见 satellite_env.py step 6），
            与此函数无关。

        参数 P_tx_w 保留仅为向后兼容，不再参与计算。
        """
        cpu_capacity = P_cpu_w / max(self.P_cpu_max, 1e-9)   # 归一化计算能力 [0,1]
        # 这里返回的是“星上处理速率”，不是最终对地下传速率。
        max_rate = float(QUEUE_CONFIG.get(
            "data_service_rate_max_mbs",
            QUEUE_CONFIG.get("data_service_rate_max_mbps", 5.0),
        ))                                        # CPU 最大处理速率 (MB/s)
        return float(np.clip(cpu_capacity * max_rate, 0.0, max_rate))

    def tx_downlink_rate(self, P_tx_w: float) -> float:
        """
        发射机对地传输速率上限 (MB/s)
        实际下传量还受信道容量和通信窗口约束，此处为功率侧上限。
        """
        tx_capacity = P_tx_w / max(self.P_tx_max, 1e-9)
        max_rate = float(QUEUE_CONFIG.get("tx_downlink_rate_max_mbs", 5.0))
        return float(np.clip(tx_capacity * max_rate, 0.0, max_rate))

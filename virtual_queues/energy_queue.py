"""
能量、轨道和数据队列的 Lyapunov 状态模型。

论文中的三类队列状态:
  - Q_E: energy safety deficit queue
  - Q_H: orbit altitude safety deficit queue
  - Q_D: raw task backlog queue

共同的 value / prev_value / history / drift 逻辑放在 BaseVirtualQueue，
这里每个类只保留各自的物理更新方程。
"""
import sys as _sys, os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in _sys.path:
    _sys.path.append(_PROJECT_ROOT)

from config import QUEUE_CONFIG, ORBITAL_CONFIG
from utils.sanitizers import sanitize_scalar
from virtual_queues.base_queue import BaseVirtualQueue


class EnergyVirtualQueue(BaseVirtualQueue):
    """能量缺陷队列 Q_E = 电池能量 >= E_min 的累积违反。"""

    def __init__(self):
        super().__init__(
            QUEUE_CONFIG["energy_queue_max"],
            lyapunov_weight_scale=QUEUE_CONFIG["energy_weight_V"],
        )

    def reset(self, initial_energy_margin: float = 0.0):
        # initial_energy_margin > 0 表示电池已经安全。
        self._reset_value(max(0.0, -float(initial_energy_margin)))

    def update(self, energy_margin_wh: float) -> dict:
        margin = sanitize_scalar(
            energy_margin_wh,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self._begin_update()
        self._set_value(max(self.value - margin, 0.0))
        return self.state_dict()


class OrbitVirtualQueue(BaseVirtualQueue):
    """轨道缺陷队列 Q_H = 高度 h >= h_min 的累积违反。"""

    def __init__(self):
        self.h_min_m = ORBITAL_CONFIG["altitude_min_km"] * 1e3
        super().__init__(
            QUEUE_CONFIG["orbit_queue_max"],
            lyapunov_weight_scale=QUEUE_CONFIG["orbit_weight_V"],
        )

    def reset(self, initial_altitude_m: float):
        altitude_margin_km = (float(initial_altitude_m) - self.h_min_m) / 1e3
        self._reset_value(max(0.0, -altitude_margin_km))

    def update(self, altitude_m: float) -> dict:
        altitude = sanitize_scalar(
            altitude_m,
            nan=self.h_min_m,
            posinf=self.h_min_m,
            neginf=0.0,
        )
        altitude_margin_km = (altitude - self.h_min_m) / 1e3
        self._begin_update()
        self._set_value(max(self.value - altitude_margin_km, 0.0))
        return self.state_dict()


class DataTaskQueue(BaseVirtualQueue):
    """
    原始任务队列 Q_D。

    length / prev_length 作为兼容别名保留给旧代码，
    而共享队列机制存储规范的 value / prev_value。
    """

    def __init__(self):
        super().__init__(QUEUE_CONFIG["data_queue_max_mb"])
        self.total_arrived = 0.0
        self.total_serviced = 0.0

    @property
    def max_length(self) -> float:
        return self.max_value

    @max_length.setter
    def max_length(self, value: float) -> None:
        self.max_value = max(float(value), 1e-9)

    @property
    def length(self) -> float:
        return self.value

    @length.setter
    def length(self, value: float) -> None:
        self.value = self._coerce_value(value)

    @property
    def prev_length(self) -> float:
        return self.prev_value

    @prev_length.setter
    def prev_length(self, value: float) -> None:
        self.prev_value = self._coerce_value(value)

    def reset(self):
        self._reset_value(0.0)
        self.total_arrived = 0.0
        self.total_serviced = 0.0

    def step(self, arrival_mb: float, service_mb: float) -> dict:
        return self.update_with_removals(arrival_mb, service_mb, dropped_mb=0.0)

    def update_with_removals(
        self,
        arrival_mb: float,
        processed_mb: float,
        dropped_mb: float = 0.0,
    ) -> dict:
        """按本步到达、处理和主动丢弃量更新 raw queue。"""
        arrival = sanitize_scalar(arrival_mb, nan=0.0, posinf=0.0,
                                  neginf=0.0, min_value=0.0)
        processed = sanitize_scalar(processed_mb, nan=0.0, posinf=0.0,
                                    neginf=0.0, min_value=0.0)
        dropped = sanitize_scalar(dropped_mb, nan=0.0, posinf=0.0,
                                  neginf=0.0, min_value=0.0)
        self._begin_update()

        available = self.length + arrival
        actual_processed = min(processed, available)
        remaining_after_process = max(available - actual_processed, 0.0)
        actual_dropped = min(dropped, remaining_after_process)
        new_length_uncapped = max(remaining_after_process - actual_dropped, 0.0)
        overflow = max(new_length_uncapped - self.max_length, 0.0)
        self._set_value(new_length_uncapped)

        self.total_arrived += arrival
        self.total_serviced += actual_processed

        return {
            "queue_length": float(self.length),
            "arrived": float(arrival),
            "serviced": float(actual_processed),
            "dropped_mb": float(actual_dropped),
            "removed_mb": float(actual_processed + actual_dropped),
            "overflow_mb": float(overflow),
            "is_full": bool(self.length >= self.max_length * 0.95),
            "utilization": float(self.length / max(self.max_length, 1e-6)),
        }

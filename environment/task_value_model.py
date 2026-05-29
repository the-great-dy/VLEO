"""
任务价值跟踪器（时效衰减 · raw/processed 队列流转）。

物理队列只保存 MB 体积；本模块给这些数据体积补上任务语义：
priority、quality、AoI/VoI 时效性、折价交付价值、过期损失和丢弃损失。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from utils.action_space import VALUE_CLASS_NAMES


@dataclass
class TaskBatch:
    """一个控制步产生的任务批次，支持后续按 MB 部分处理/部分下传。"""

    mb: float
    value: float
    priority: float
    quality: float
    deadline_steps: int
    created_step: int
    scene_name: str = "generic"
    scene_class_code: float = 0.0
    cloud_cover: float = 0.0
    freshness_profile: str = "linear"
    freshness_power: float = 1.0
    freshness_peak_fraction: float = 0.35
    freshness_late_floor: float = 0.0

    @property
    def value_density(self) -> float:
        return self.value / max(self.mb, 1e-9)

    def age_steps(self, now_step: int) -> int:
        return max(0, int(now_step) - int(self.created_step))

    def aoi_steps(self, now_step: int) -> int:
        """Age of Information：该批数据从生成到当前时刻经历的控制步数。"""
        return self.age_steps(now_step)

    def deadline_urgency(self, now_step: int) -> float:
        return float(np.clip(self.age_steps(now_step) / max(self.deadline_steps, 1), 0.0, 1.0))

    def timeliness_weight(self, now_step: int, floor: float = 0.2,
                          power: float = 1.0,
                          overdue_grace_steps: int = 30,
                          overdue_decay_rate: float = 4.0) -> float:
        """时效性权重。"""
        age = self.age_steps(now_step)
        if age > self.deadline_steps:
            overdue = age - self.deadline_steps
            grace_steps = max(0, int(overdue_grace_steps))
            if grace_steps <= 0 or overdue > grace_steps:
                return 0.0
            decay_rate = max(float(overdue_decay_rate), 1e-6)
            return float(max(floor, 0.0) * np.exp(-decay_rate * overdue / max(grace_steps, 1)))

        x = float(np.clip(age / max(self.deadline_steps, 1), 0.0, 1.0))
        floor = float(np.clip(floor, 0.0, 1.0))
        profile = str(self.freshness_profile or "linear").lower()
        if profile == "hump":
            peak = float(np.clip(self.freshness_peak_fraction, 1e-3, 1.0 - 1e-3))
            late_floor = float(np.clip(self.freshness_late_floor, 0.0, 1.0))
            if x <= peak:
                y = x / peak
            else:
                y = late_floor + (1.0 - late_floor) * (1.0 - (x - peak) / max(1.0 - peak, 1e-6))
        elif profile == "late":
            y = x ** max(float(self.freshness_power or power), 1e-6)
        else:
            remaining_ratio = 1.0 - x
            y = remaining_ratio ** max(float(self.freshness_power or power), 1e-6)
        return float(floor + (1.0 - floor) * np.clip(y, 0.0, 1.0))

    def score(self, now_step: int, floor: float = 0.0,
              power: float = 1.0,
              overdue_grace_steps: int = 0,
              overdue_decay_rate: float = 4.0,
              value_weight: float = 1.0,
              urgency_weight: float = 0.0) -> float:
        timeliness_weight = self.timeliness_weight(
            now_step,
            floor=floor,
            power=power,
            overdue_grace_steps=overdue_grace_steps,
            overdue_decay_rate=overdue_decay_rate,
        )
        value_term = max(1e-9, max(0.0, self.value_density) * timeliness_weight)
        urgency = self.deadline_urgency(now_step)
        remaining_steps = max(1, int(self.deadline_steps) - int(self.age_steps(now_step)))
        value_exponent = float(np.exp(np.clip(value_weight, -1.0, 1.0)))
        urgency_exponent = float(np.exp(np.clip(urgency_weight, -1.0, 1.0)))
        deadline_term = max(1e-9, 1.0 / float(remaining_steps))
        return float((value_term ** value_exponent) * (deadline_term ** urgency_exponent) * np.exp(urgency))


class TaskValueTracker:
    """
    跟踪 raw_queue -> processed_queue -> 地面交付的任务价值。

    这里故意采用“批次”粒度，而不是逐包建模：每个控制步最多生成一个任务批次，
    部分处理/部分下传时按价值密度拆分，既保留任务语义，也避免仿真开销过大。
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.raw_batches: list[TaskBatch] = []
        self.processed_batches: list[TaskBatch] = []
        self.reset()

    def reset(self):
        self.raw_batches = []
        self.processed_batches = []
        self.total_generated_mb = 0.0
        self.total_generated_value = 0.0
        self.total_processed_mb = 0.0
        self.total_processed_value = 0.0
        self.total_delivered_mb = 0.0
        self.total_delivered_value = 0.0
        self.total_on_time_delivered_mb = 0.0
        self.total_on_time_delivered_value = 0.0
        self.total_expired_mb = 0.0
        self.total_expired_value = 0.0
        self.total_dropped_mb = 0.0
        self.total_dropped_value = 0.0
        self.total_delivery_delay_steps = 0.0
        self.total_value_weighted_delivery_delay_steps = 0.0
        self.delivery_events = 0

    @property
    def raw_mb(self) -> float:
        return float(sum(batch.mb for batch in self.raw_batches))

    @property
    def processed_mb(self) -> float:
        return float(sum(batch.mb for batch in self.processed_batches))

    @property
    def processed_value(self) -> float:
        return float(sum(batch.value for batch in self.processed_batches))

    def _decay_floor(self) -> float:
        return float(self.cfg.get("deadline_decay_floor", self.cfg.get("freshness_floor", 0.0)))

    def _decay_power(self) -> float:
        return float(self.cfg.get("deadline_decay_power", self.cfg.get("freshness_default_power", 1.0)))

    def _residual_density(self, batch: TaskBatch, now_step: int) -> float:
        timeliness_weight = batch.timeliness_weight(
            now_step,
            floor=self._decay_floor(),
            power=self._decay_power(),
            overdue_grace_steps=int(self.cfg.get("overdue_grace_steps", 0)),
            overdue_decay_rate=float(self.cfg.get("overdue_decay_rate", 4.0)),
        )
        return float(batch.value_density * timeliness_weight)

    @staticmethod
    def _class_id_from_density(density: float, high_density: float,
                               mid_density: float) -> int:
        if float(density) >= float(high_density):
            return 0
        if float(density) >= float(mid_density):
            return 1
        return 2

    def task_nominal_class_id(self, batch: TaskBatch) -> int:
        """按名义价值密度分类，用于交付/丢弃/过期等损失统计。"""
        high_density = float(self.cfg.get(
            "class_high_value_density",
            self.cfg.get("class_high_residual_value_density", 3.0),
        ))
        mid_density = float(self.cfg.get(
            "class_medium_value_density",
            self.cfg.get("class_medium_residual_value_density", 1.20),
        ))
        return self._class_id_from_density(
            batch.value_density, high_density, mid_density)

    def task_class_id(self, batch: TaskBatch, now_step: int) -> int:
        """按剩余价值密度分成 High/Medium/Low。"""
        residual_density = self._residual_density(batch, now_step)
        high_density = float(self.cfg.get("class_high_residual_value_density", 3.0))
        mid_density = float(self.cfg.get("class_medium_residual_value_density", 1.20))

        return self._class_id_from_density(
            residual_density, high_density, mid_density)

    def is_protected_batch(self, batch: TaskBatch, now_step: int) -> bool:
        return self.task_class_id(batch, now_step) == 0

    def is_droppable_batch(
        self,
        batch: TaskBatch,
        now_step: int,
        drop_context: dict | None = None,
    ) -> bool:
        """仅允许纯 Low 类任务被主动丢弃，防止 Medium 降级导致统计口径重叠。"""
        drop_context = drop_context or {}
        if self.is_protected_batch(batch, now_step):
            return False
        if self.task_class_id(batch, now_step) != 2:
            return False
        residual_density_threshold = float(
            self.cfg.get("low_residual_value_density_threshold", 1.20)
        )
        residual_density = self._residual_density(batch, now_step)
        if residual_density > residual_density_threshold:
            return False
        urgency_protection = float(
            self.cfg.get("low_drop_deadline_urgency_protection", 0.95)
        )
        if 0.0 <= urgency_protection <= 1.0:
            if batch.deadline_urgency(now_step) >= urgency_protection:
                return False
        resource_pressure = float(drop_context.get("resource_pressure", 0.0))
        return resource_pressure > float(self.cfg.get("low_drop_resource_pressure_threshold", 0.03))

    def _work_conserving_reallocation_order(self, donor_class_id: int) -> tuple[int, ...]:
        """为 donor 类定义剩余预算可回流的优先级顺序。"""
        if donor_class_id == 0:
            return (1, 2)
        if donor_class_id == 1:
            return (0, 2)
        return (0, 1)

    def droppable_backlog(self, now_step: int, drop_context: dict | None = None) -> dict:
        drop_context = drop_context or {}

        raw_mb = sum(
            batch.mb for batch in self.raw_batches
            if self.is_droppable_batch(batch, now_step, drop_context)
        )
        proc_mb = sum(
            batch.mb for batch in self.processed_batches
            if self.is_droppable_batch(batch, now_step, drop_context)
        )

        return {
            "droppable_raw_mb": float(raw_mb),
            "droppable_processed_mb": float(proc_mb),
            "droppable_backlog_mb": float(raw_mb + proc_mb),
        }

    def class_stats(self, now_step: int) -> dict:
        """返回 raw/processed 在 High/Mid/Low 上的压缩直方图。"""
        stats = {}
        for name in VALUE_CLASS_NAMES:
            stats[f"raw_{name}_mb"] = 0.0
            stats[f"raw_{name}_value"] = 0.0
            stats[f"processed_{name}_mb"] = 0.0
            stats[f"processed_{name}_value"] = 0.0
            stats[f"expiring_{name}_mb"] = 0.0
            stats[f"expiring_{name}_value"] = 0.0

        urgent_threshold = int(self.cfg.get("urgent_deadline_steps", 30))
        for queue_name, queue in (
            ("raw", self.raw_batches),
            ("processed", self.processed_batches),
        ):
            for batch in queue:
                name = VALUE_CLASS_NAMES[self.task_class_id(batch, now_step)]
                stats[f"{queue_name}_{name}_mb"] += float(batch.mb)
                stats[f"{queue_name}_{name}_value"] += float(batch.value)
                remaining = batch.deadline_steps - batch.age_steps(now_step)
                if remaining <= urgent_threshold:
                    stats[f"expiring_{name}_mb"] += float(batch.mb)
                    stats[f"expiring_{name}_value"] += float(batch.value)

        for key, value in list(stats.items()):
            stats[key] = float(value)
        return stats

    def _queue_value_before_deadline(self, queue: list[TaskBatch], now_step: int,
                                     deadline_step: int, class_id: int | None = None) -> float:
        total = 0.0
        for batch in queue:
            if class_id is not None and self.task_class_id(batch, now_step) != int(class_id):
                continue
            if int(batch.created_step) + int(batch.deadline_steps) <= int(deadline_step):
                total += float(batch.mb)
        return total

    def _queue_mb_by_class(self, queue: list[TaskBatch], now_step: int) -> np.ndarray:
        out = np.zeros(len(VALUE_CLASS_NAMES), dtype=np.float64)
        for batch in queue:
            out[self.task_class_id(batch, now_step)] += float(batch.mb)
        return out

    def processed_mb_by_class(self, now_step: int) -> np.ndarray:
        return self._queue_mb_by_class(self.processed_batches, now_step)

    def deliverability_features(self, now_step: int, future_capacity_bins: list[tuple[float, float]]) -> dict:
        bin_count = max(0, int(self.cfg.get("deliverability_bin_count", 8)))
        mb_norm = max(float(self.cfg.get("deliverability_bin_norm_mb", 512.0)), 1e-6)
        time_norm = max(float(self.cfg.get("deliverability_time_bin_norm_steps", 540.0)), 1e-6)
        features: dict[str, float] = {}
        bins = list(future_capacity_bins[:bin_count])
        while len(bins) < bin_count:
            bins.append((0.0, 0.0))
        for idx, (steps_to_bin, capacity_mb) in enumerate(bins):
            features[f"capacity_bin_{idx}_mb_norm"] = float(np.clip(capacity_mb / mb_norm, 0.0, 2.0))
            features[f"capacity_bin_{idx}_time_norm"] = float(np.clip(steps_to_bin / time_norm, 0.0, 2.0))
        raw_same = self._queue_mb_by_class(self.raw_batches, now_step)
        proc_same = self._queue_mb_by_class(self.processed_batches, now_step)
        for idx, name in enumerate(VALUE_CLASS_NAMES):
            features[f"concurrent_{name}_same_class_mb_norm"] = float(np.clip((raw_same[idx] + proc_same[idx]) / mb_norm, 0.0, 2.0))
        return features

    def deliverability_for_batch(
        self,
        batch: TaskBatch,
        now_step: int,
        future_capacity_until_deadline_mb: float,
        processed_backlog_before_deadline_mb: float,
        class_reservation_mb: float = 0.0,
        amount_mb: float | None = None,
    ) -> float:
        remaining_capacity = max(
            0.0,
            float(future_capacity_until_deadline_mb)
            - float(processed_backlog_before_deadline_mb)
            - max(0.0, float(class_reservation_mb)),
        )
        required_mb = float(batch.mb if amount_mb is None else amount_mb)
        return float(np.clip(remaining_capacity / max(required_mb, 1e-9), 0.0, 1.0))

    def deadline_contact_stats(
        self,
        now_step: int,
        steps_to_next_window: float,
    ) -> dict:
        """统计 high/mid 任务是否还能赶上下一个通信窗口。"""
        steps_to_window = max(0, int(np.ceil(float(steps_to_next_window))))

        def _queue_stats(queue: list[TaskBatch]) -> tuple[float, float, float, float]:
            total_value = deliverable_value = 0.0
            total_mb = deliverable_mb = 0.0
            for batch in queue:
                if self.task_class_id(batch, now_step) > 1:
                    continue
                value = max(0.0, float(batch.value))
                mb = max(0.0, float(batch.mb))
                total_value += value
                total_mb += mb
                remaining_steps = batch.deadline_steps - batch.age_steps(now_step)
                if remaining_steps >= steps_to_window:
                    deliverable_value += value
                    deliverable_mb += mb
            return total_value, deliverable_value, total_mb, deliverable_mb

        raw_value, raw_deliverable_value, raw_mb, raw_deliverable_mb = (
            _queue_stats(self.raw_batches)
        )
        proc_value, proc_deliverable_value, proc_mb, proc_deliverable_mb = (
            _queue_stats(self.processed_batches)
        )
        raw_ratio = raw_deliverable_value / max(raw_value, 1e-9)
        proc_ratio = proc_deliverable_value / max(proc_value, 1e-9)
        total_value = raw_value + proc_value
        deliverable_value = raw_deliverable_value + proc_deliverable_value
        combined_ratio = deliverable_value / max(total_value, 1e-9)
        mismatch = 0.0 if total_value <= 1e-9 else 1.0 - combined_ratio

        return {
            "raw_high_next_window_deliverable_ratio": float(np.clip(raw_ratio, 0.0, 1.0)),
            "processed_high_next_window_deliverable_ratio": float(np.clip(proc_ratio, 0.0, 1.0)),
            "high_value_deadline_contact_mismatch": float(np.clip(mismatch, 0.0, 1.0)),
            "raw_high_next_window_deliverable_mb": float(raw_deliverable_mb),
            "processed_high_next_window_deliverable_mb": float(proc_deliverable_mb),
            "high_value_backlog_mb": float(raw_mb + proc_mb),
            "high_value_backlog_value": float(total_value),
            "next_window_steps": float(steps_to_window),
        }

    def _make_batch(self, *, mb: float, value: float, priority: float, quality: float,
                    deadline_steps: int, created_step: int, scene_name: str = "generic",
                    scene_class_code: float = 0.0, cloud_cover: float = 0.0,
                    profile: dict | None = None) -> TaskBatch:
        profile = profile or {}
        return TaskBatch(
            mb=mb,
            value=value,
            priority=priority,
            quality=quality,
            deadline_steps=deadline_steps,
            created_step=int(created_step),
            scene_name=scene_name,
            scene_class_code=scene_class_code,
            cloud_cover=cloud_cover,
            freshness_profile=str(profile.get("freshness_profile", "linear")),
            freshness_power=float(profile.get("freshness_power", self.cfg.get("freshness_default_power", 1.0))),
            freshness_peak_fraction=float(profile.get("freshness_peak_fraction", 0.35)),
            freshness_late_floor=float(profile.get("freshness_late_floor", self.cfg.get("freshness_floor", 0.0))),
        )

    def add_arrival(self, mb: float, rng: np.random.Generator, now_step: int,
                    scene_context: dict | None = None) -> dict:
        """生成新的原始任务数据，并赋予 priority、quality 和 deadline。"""
        mb = float(max(mb, 0.0))
        if mb <= 0.0:
            return {"generated_mb": 0.0, "generated_value": 0.0}

        if scene_context:
            profile = dict(scene_context.get("profile", {}))
            scene_name = str(scene_context.get("scene_name", "generic"))
            scene_class_code = float(scene_context.get("scene_class_code", 0.0))
            priority_bounds = profile.get(
                "priority_range",
                (self.cfg.get("priority_min", 0.5), self.cfg.get("priority_max", 1.5)),
            )
            quality_bounds = profile.get(
                "quality_range",
                (self.cfg.get("quality_min", 0.6), self.cfg.get("quality_max", 1.2)),
            )
            deadline_bounds = profile.get(
                "deadline_range_steps",
                (self.cfg.get("deadline_min_steps", 60), self.cfg.get("deadline_max_steps", 360)),
            )
            cloud_bounds = profile.get("cloud_cover_range", (0.0, 0.0))

            priority = float(rng.uniform(float(priority_bounds[0]), float(priority_bounds[1])))
            raw_quality = float(rng.uniform(float(quality_bounds[0]), float(quality_bounds[1])))
            cloud_cover = float(rng.uniform(float(cloud_bounds[0]), float(cloud_bounds[1])))
            cloud_penalty = float(profile.get("cloud_penalty", 0.0))
            quality = raw_quality * max(0.05, 1.0 - cloud_penalty * cloud_cover)
            deadline_steps = int(rng.integers(int(deadline_bounds[0]), int(deadline_bounds[1]) + 1))
            base_value = float(self.cfg.get("base_value_per_mb", 1.0))
            base_multiplier = float(profile.get("base_value_multiplier", 1.0))
            value_density = base_value * base_multiplier * priority * quality
            value_density = float(np.clip(
                value_density,
                float(self.cfg.get("intrinsic_value_min", 0.01)),
                float(self.cfg.get("intrinsic_value_max", 100.0)),
            ))
            value = mb * value_density

            self.raw_batches.append(self._make_batch(
                mb=mb,
                value=value,
                priority=priority,
                quality=quality,
                deadline_steps=deadline_steps,
                created_step=now_step,
                scene_name=scene_name,
                scene_class_code=scene_class_code,
                cloud_cover=cloud_cover,
                profile=profile,
            ))
            self.total_generated_mb += mb
            self.total_generated_value += value
            return {
                "generated_mb": mb,
                "generated_value": value,
                "generated_value_density": float(value_density),
                "generated_priority": priority,
                "generated_quality": quality,
                "generated_raw_quality": raw_quality,
                "generated_deadline_steps": float(deadline_steps),
                "scene_name": scene_name,
                "scene_class_code": scene_class_code,
                "scene_cloud_cover": cloud_cover,
            }

        priority = float(rng.uniform(self.cfg.get("priority_min", 0.5),
                                     self.cfg.get("priority_max", 1.5)))
        quality = float(rng.uniform(self.cfg.get("quality_min", 0.6),
                                    self.cfg.get("quality_max", 1.2)))
        deadline_steps = int(rng.integers(
            int(self.cfg.get("deadline_min_steps", 60)),
            int(self.cfg.get("deadline_max_steps", 360)) + 1,
        ))
        base_value = float(self.cfg.get("base_value_per_mb", 1.0))
        value_density = float(np.clip(
            base_value * priority * quality,
            float(self.cfg.get("intrinsic_value_min", 0.01)),
            float(self.cfg.get("intrinsic_value_max", 100.0)),
        ))
        value = mb * value_density

        self.raw_batches.append(self._make_batch(
            mb=mb,
            value=value,
            priority=priority,
            quality=quality,
            deadline_steps=deadline_steps,
            created_step=now_step,
        ))
        self.total_generated_mb += mb
        self.total_generated_value += value
        return {"generated_mb": mb, "generated_value": value}

    def process(self, mb: float, now_step: int) -> dict:
        """按任务价值优先级从 raw_queue 移入 processed_queue。"""
        moved, value, deliverable_value, undeliverable_value = self._move_between_queues(
            self.raw_batches,
            self.processed_batches,
            float(max(mb, 0.0)),
            now_step,
        )
        self.total_processed_mb += moved
        self.total_processed_value += value
        return {
            "processed_mb": moved,
            "processed_value": value,
            "processed_deliverable_value": deliverable_value,
            "processed_undeliverable_value": undeliverable_value,
        }

    def process_by_priority(self, amount_mb: float, now_step: int, *,
                            value_weight: float = 1.0,
                            urgency_weight: float = 0.0,
                            future_capacity_fn=None) -> dict:
        """按 actor 给出的价值/紧迫度权重直接选择要处理的批次。

        Phase 1 硬规则：
          A. deliver_prob < min_deliver_prob 的批次直接跳过（CPU 留给可送达批次）。
          B. 类优先级 floor：按 class_id 顺序（0=high → 1=mid → 2=low）外层迭代，
             actor 的 value_weight/urgency_weight 只在同类内决定细排序。
        """
        # Phase 1 硬规则：延迟从 config 读取，避免 task_value_model ↔ config 循环依赖。
        try:
            from config import HARD_RULES_CONFIG  # noqa: WPS433
        except Exception:
            HARD_RULES_CONFIG = {}
        min_deliver_prob = float(HARD_RULES_CONFIG.get("min_deliver_prob_for_processing", 0.0))
        enable_gate = bool(HARD_RULES_CONFIG.get("enable_deliver_prob_gate", False))
        enable_class_floor = bool(HARD_RULES_CONFIG.get("enable_class_priority_floor", False))
        # Phase 2 硬规则 G: per-class deliverability gate 阈值。
        # 低价值任务要求"几乎确定能送达"才花 CPU；高价值放宽。
        # 不设或缺省时退化到统一 min_deliver_prob_for_processing。
        enable_class_aware_gate = bool(HARD_RULES_CONFIG.get("enable_class_aware_gate", False))
        gate_threshold_high = float(HARD_RULES_CONFIG.get("min_deliver_prob_high", min_deliver_prob))
        gate_threshold_mid = float(HARD_RULES_CONFIG.get("min_deliver_prob_medium", min_deliver_prob))
        gate_threshold_low = float(HARD_RULES_CONFIG.get("min_deliver_prob_low", min_deliver_prob))
        # class_id 0=high, 1=medium, 2=low → 对应阈值数组
        gate_thresholds_by_class = (gate_threshold_high, gate_threshold_mid, gate_threshold_low)

        amount = float(max(amount_mb, 0.0))
        result = {"processed_mb": 0.0, "processed_value": 0.0,
                  "processed_deliverable_value": 0.0, "processed_undeliverable_value": 0.0,
                  "cpu_unused_before_reallocation_mb": 0.0, "cpu_reallocated_mb": 0.0,
                  "skipped_undeliverable_mb": 0.0}
        for name in VALUE_CLASS_NAMES:
            result[f"processed_{name}_mb"] = 0.0
            result[f"processed_{name}_value"] = 0.0
            result[f"processed_{name}_deliverable_value"] = 0.0
            result[f"processed_{name}_undeliverable_value"] = 0.0
            result[f"cpu_reallocated_to_{name}_mb"] = 0.0

        # 规则 B: 单次 sorted 调用，class_id 作为主键 → 高类先处理；
        # 关闭时退化到原始 score-only 排序。这比外层 for class 的 3x 嵌套快 ~3x。
        indices = self._queue_order(
            self.raw_batches,
            now_step,
            prefer_low_value=False,
            value_weight=value_weight,
            urgency_weight=urgency_weight,
            class_priority_first=enable_class_floor,
        )

        # 规则 A 性能保护：连续 skip 超过 max_consec_skips 就停。
        # 防止"未来 capacity 全 0 时遍历整个队列做 O(N) deliverability 计算"
        # 这种最坏情况（实测之前从 30ms/step 涨到 343ms/step 的根因）。
        max_consec_skips = 8
        consec_skips = 0

        for idx in indices:
            if amount <= 1e-9:
                break
            batch = self.raw_batches[idx]
            class_idx = self.task_class_id(batch, now_step)
            class_name = VALUE_CLASS_NAMES[class_idx]
            take = min(batch.mb, amount)
            take_value = batch.value * take / max(batch.mb, 1e-9)
            deliver_prob = 1.0
            if future_capacity_fn is not None:
                deadline_abs = int(batch.created_step) + int(batch.deadline_steps)
                future_cap = float(future_capacity_fn(deadline_abs))
                backlog = self._queue_value_before_deadline(
                    self.processed_batches, now_step, deadline_abs, class_id=class_idx)
                same_raw = self._queue_value_before_deadline(
                    self.raw_batches, now_step, deadline_abs, class_id=class_idx)
                same_raw = max(0.0, same_raw - take)
                reservation = self._class_reservation_mb(class_idx) * same_raw
                deliver_prob = self.deliverability_for_batch(
                    batch,
                    now_step,
                    future_cap,
                    backlog,
                    reservation,
                    amount_mb=take,
                )
            # 规则 A (+ G class-aware): deliverability gate
            # 普通模式用统一 min_deliver_prob；G 启用时按 class 用不同阈值
            # （low 要求更高 deliver_prob 才处理，high 阈值最低）。
            if enable_gate:
                threshold = (
                    gate_thresholds_by_class[class_idx]
                    if enable_class_aware_gate
                    else min_deliver_prob
                )
                if deliver_prob < threshold:
                    result["skipped_undeliverable_mb"] += take
                    consec_skips += 1
                    if consec_skips >= max_consec_skips:
                        break
                    continue
            consec_skips = 0
            deliverable = take_value * deliver_prob
            undeliverable = max(0.0, take_value - deliverable)
            self.processed_batches.append(self._make_batch(
                mb=take,
                value=take_value,
                priority=batch.priority,
                quality=batch.quality,
                deadline_steps=batch.deadline_steps,
                created_step=batch.created_step,
                scene_name=batch.scene_name,
                scene_class_code=batch.scene_class_code,
                cloud_cover=batch.cloud_cover,
                profile={
                    "freshness_profile": batch.freshness_profile,
                    "freshness_power": batch.freshness_power,
                    "freshness_peak_fraction": batch.freshness_peak_fraction,
                    "freshness_late_floor": batch.freshness_late_floor,
                },
            ))
            batch.mb -= take
            batch.value -= take_value
            amount -= take
            result[f"processed_{class_name}_mb"] += take
            result[f"processed_{class_name}_value"] += take_value
            result[f"processed_{class_name}_deliverable_value"] += deliverable
            result[f"processed_{class_name}_undeliverable_value"] += undeliverable
            result["processed_mb"] += take
            result["processed_value"] += take_value
            result["processed_deliverable_value"] += deliverable
            result["processed_undeliverable_value"] += undeliverable

        result["cpu_unused_before_reallocation_mb"] = max(0.0, amount)
        self.total_processed_mb += float(result["processed_mb"])
        self.total_processed_value += float(result["processed_value"])
        self._compact(self.raw_batches)
        return {key: float(value) for key, value in result.items()}

    def deliver_by_priority(self, amount_mb: float, now_step: int, *,
                            value_weight: float = 1.0,
                            urgency_weight: float = 0.0) -> dict:
        """按 actor 给出的价值/紧迫度权重直接选择要下传的批次。

        Phase 1 硬规则 D：先 reserve tx_high_reserve_fraction 给 high 类，
        剩余预算（含 high 用不完的）再供 mid/low 自由竞争。
        """
        # Phase 1 硬规则：延迟从 config 读取，避免循环依赖。
        try:
            from config import HARD_RULES_CONFIG  # noqa: WPS433
        except Exception:
            HARD_RULES_CONFIG = {}
        high_reserve_fraction = float(HARD_RULES_CONFIG.get("tx_high_reserve_fraction", 0.0))
        enable_tx_reserve = bool(HARD_RULES_CONFIG.get("enable_tx_high_reserve", False))

        amount = float(max(amount_mb, 0.0))
        breakdown = self._empty_class_breakdown()
        delivered = value = on_time_mb = on_time_value = 0.0
        delay_sum = value_delay_sum = 0.0
        events = 0

        # 规则 D 第一阶段：reserved budget 只能给 high 类
        if enable_tx_reserve and high_reserve_fraction > 0.0 and amount > 1e-9:
            high_budget = amount * high_reserve_fraction
            (d, v, om, ov, ds, vds, ev) = self._remove_from_queue(
                self.processed_batches,
                high_budget,
                now_step,
                apply_timeliness_weight=True,
                class_id=0,  # 0 = high
                class_breakdown=breakdown,
                value_weight=value_weight,
                urgency_weight=urgency_weight,
            )
            delivered += d
            value += v
            on_time_mb += om
            on_time_value += ov
            delay_sum += ds
            value_delay_sum += vds
            events += ev

        # 第二阶段：剩余预算（含 high 配额未用完的）自由分配 high/mid/low。
        remaining_budget = max(0.0, amount - delivered)
        if remaining_budget > 1e-9:
            (d, v, om, ov, ds, vds, ev) = self._remove_from_queue(
                self.processed_batches,
                remaining_budget,
                now_step,
                apply_timeliness_weight=True,
                class_breakdown=breakdown,
                value_weight=value_weight,
                urgency_weight=urgency_weight,
            )
            delivered += d
            value += v
            on_time_mb += om
            on_time_value += ov
            delay_sum += ds
            value_delay_sum += vds
            events += ev
        # Apply per-class specificity discount; rebuild totals from per-class values.
        specificity_discounted_value = 0.0
        for class_id, name in enumerate(VALUE_CLASS_NAMES):
            class_val = float(breakdown.get(f"{name}_value", 0.0))
            discounted = class_val * self._specificity_discount(class_id, now_step)
            breakdown[f"{name}_value"] = discounted
            specificity_discounted_value += discounted
        if value > 1e-9:
            scale = specificity_discounted_value / value
        else:
            scale = 1.0
        value = specificity_discounted_value
        on_time_value *= scale
        value_delay_sum *= scale

        self.total_delivered_mb += delivered
        self.total_delivered_value += value
        self.total_on_time_delivered_mb += on_time_mb
        self.total_on_time_delivered_value += on_time_value
        self.total_delivery_delay_steps += delay_sum
        self.total_value_weighted_delivery_delay_steps += value_delay_sum
        self.delivery_events += events

        result = {
            "delivered_mb": delivered,
            "delivered_value": value,
            "timely_weighted_delivered_value": value,
            "voi_delivered_value": value,
            "on_time_delivered_mb": on_time_mb,
            "on_time_delivered_value": on_time_value,
            "avg_delivery_delay_steps": delay_sum / max(delivered, 1e-9),
            "aoi_steps": delay_sum / max(delivered, 1e-9),
            "average_aoi_steps": delay_sum / max(delivered, 1e-9),
            "value_weighted_aoi_steps": value_delay_sum / max(value, 1e-9),
            "tx_unused_before_reallocation_mb": max(0.0, amount - delivered),
            "tx_reallocated_mb": 0.0,
        }
        for name in VALUE_CLASS_NAMES:
            result[f"delivered_{name}_mb"] = float(breakdown.get(f"{name}_mb", 0.0))
            result[f"delivered_{name}_value"] = float(breakdown.get(f"{name}_value", 0.0))
            result[f"tx_reallocated_to_{name}_mb"] = 0.0
        return {key: float(value) for key, value in result.items()}

    def process_by_class(self, capacities_mb, now_step: int, *,
                         value_weight: float = 1.0,
                         urgency_weight: float = 0.0,
                         future_capacity_fn=None) -> dict:
        """按 High/Mid/Low 资源预算处理 raw queue。"""
        capacities = np.asarray(capacities_mb, dtype=np.float64).reshape(-1)
        if capacities.size < len(VALUE_CLASS_NAMES):
            capacities = np.pad(capacities, (0, len(VALUE_CLASS_NAMES) - capacities.size))

        result = {"processed_mb": 0.0, "processed_value": 0.0,
                  "processed_deliverable_value": 0.0, "processed_undeliverable_value": 0.0}
        consumed = np.zeros(len(VALUE_CLASS_NAMES), dtype=np.float64)
        for class_id, name in enumerate(VALUE_CLASS_NAMES):
            moved, value, deliverable_value, undeliverable_value = self._move_between_queues(
                self.raw_batches,
                self.processed_batches,
                float(max(capacities[class_id], 0.0)),
                now_step,
                class_id=class_id,
                value_weight=value_weight,
                urgency_weight=urgency_weight,
                future_capacity_fn=future_capacity_fn,
            )
            consumed[class_id] += moved
            self.total_processed_mb += moved
            self.total_processed_value += value
            result[f"processed_{name}_mb"] = moved
            result[f"processed_{name}_value"] = value
            result[f"processed_{name}_deliverable_value"] = deliverable_value
            result[f"processed_{name}_undeliverable_value"] = undeliverable_value
            result["processed_mb"] += moved
            result["processed_value"] += value
            result["processed_deliverable_value"] += deliverable_value
            result["processed_undeliverable_value"] += undeliverable_value

        result["cpu_unused_before_reallocation_mb"] = 0.0
        result["cpu_reallocated_mb"] = 0.0
        for name in VALUE_CLASS_NAMES:
            result[f"cpu_reallocated_to_{name}_mb"] = 0.0

        cpu_reallocation_enabled = bool(self.cfg.get(
            "cpu_work_conserving_reallocation",
            self.cfg.get("work_conserving_reallocation", True),
        ))
        if cpu_reallocation_enabled:
            for donor_class_id in range(len(VALUE_CLASS_NAMES)):
                remaining = float(max(capacities[donor_class_id], 0.0) - consumed[donor_class_id])
                if remaining > 0:
                    result["cpu_unused_before_reallocation_mb"] += remaining
                if remaining <= 1e-9:
                    continue
                for recv_class_id in self._work_conserving_reallocation_order(donor_class_id):
                    if remaining <= 1e-9:
                        break
                    moved, value, deliverable_value, undeliverable_value = self._move_between_queues(
                        self.raw_batches,
                        self.processed_batches,
                        remaining,
                        now_step,
                        class_id=recv_class_id,
                        value_weight=value_weight,
                        urgency_weight=urgency_weight,
                        future_capacity_fn=future_capacity_fn,
                    )
                    if moved <= 1e-9:
                        continue
                    recv_name = VALUE_CLASS_NAMES[recv_class_id]
                    consumed[recv_class_id] += moved
                    self.total_processed_mb += moved
                    self.total_processed_value += value
                    result[f"processed_{recv_name}_mb"] += moved
                    result[f"processed_{recv_name}_value"] += value
                    result[f"processed_{recv_name}_deliverable_value"] += deliverable_value
                    result[f"processed_{recv_name}_undeliverable_value"] += undeliverable_value
                    result["processed_mb"] += moved
                    result["processed_value"] += value
                    result["processed_deliverable_value"] += deliverable_value
                    result["processed_undeliverable_value"] += undeliverable_value
                    result["cpu_reallocated_mb"] += moved
                    result[f"cpu_reallocated_to_{recv_name}_mb"] += moved
                    remaining -= moved
        return {key: float(value) for key, value in result.items()}

    def deliver(self, mb: float, now_step: int) -> dict:
        """从 processed_queue 下传到地面端，交付价值按 VoI 时效性权重折减。"""
        (delivered, delivered_value, on_time_mb, on_time_value,
         delay_sum, value_delay_sum, events) = \
            self._remove_from_queue(
                self.processed_batches,
                float(max(mb, 0.0)),
                now_step,
                apply_timeliness_weight=True,
            )

        self.total_delivered_mb += delivered
        self.total_delivered_value += delivered_value
        self.total_on_time_delivered_mb += on_time_mb
        self.total_on_time_delivered_value += on_time_value
        self.total_delivery_delay_steps += delay_sum
        self.total_value_weighted_delivery_delay_steps += value_delay_sum
        self.delivery_events += events

        value_weighted_aoi = value_delay_sum / max(delivered_value, 1e-9)
        return {
            "delivered_mb": delivered,
            "delivered_value": delivered_value,
            "timely_weighted_delivered_value": delivered_value,
            "voi_delivered_value": delivered_value,
            "on_time_delivered_mb": on_time_mb,
            "on_time_delivered_value": on_time_value,
            "avg_delivery_delay_steps": delay_sum / max(delivered, 1e-9),
            "aoi_steps": delay_sum / max(delivered, 1e-9),
            "average_aoi_steps": delay_sum / max(delivered, 1e-9),
            "value_weighted_aoi_steps": value_weighted_aoi,
        }

    def deliver_by_class(self, capacities_mb, now_step: int, *,
                         value_weight: float = 1.0,
                         urgency_weight: float = 0.0) -> dict:
        """按 High/Mid/Low 下传预算交付 processed queue。"""
        capacities = np.asarray(capacities_mb, dtype=np.float64).reshape(-1)
        if capacities.size < len(VALUE_CLASS_NAMES):
            capacities = np.pad(capacities, (0, len(VALUE_CLASS_NAMES) - capacities.size))

        total_delivered = total_value = total_on_time_mb = total_on_time_value = 0.0
        total_delay = 0.0
        total_value_delay = 0.0
        total_events = 0
        result = {}
        consumed = np.zeros(len(VALUE_CLASS_NAMES), dtype=np.float64)
        for class_id, name in enumerate(VALUE_CLASS_NAMES):
            specificity = self._specificity_discount(class_id, now_step)
            (delivered, value, on_time_mb, on_time_value,
             delay_sum, value_delay_sum, events) = \
                self._remove_from_queue(
                    self.processed_batches,
                    float(max(capacities[class_id], 0.0)),
                    now_step,
                    apply_timeliness_weight=True,
                    class_id=class_id,
                    value_weight=value_weight,
                    urgency_weight=urgency_weight,
                )
            value *= specificity
            on_time_value *= specificity
            value_delay_sum *= specificity
            consumed[class_id] += delivered
            result[f"delivered_{name}_mb"] = delivered
            result[f"delivered_{name}_value"] = value
            total_delivered += delivered
            total_value += value
            total_on_time_mb += on_time_mb
            total_on_time_value += on_time_value
            total_delay += delay_sum
            total_value_delay += value_delay_sum
            total_events += events

        result["tx_unused_before_reallocation_mb"] = 0.0
        result["tx_reallocated_mb"] = 0.0
        for name in VALUE_CLASS_NAMES:
            result[f"tx_reallocated_to_{name}_mb"] = 0.0

        tx_reallocation_enabled = bool(self.cfg.get(
            "tx_work_conserving_reallocation",
            self.cfg.get("work_conserving_reallocation", True),
        ))
        if tx_reallocation_enabled:
            for donor_class_id in range(len(VALUE_CLASS_NAMES)):
                remaining = float(max(capacities[donor_class_id], 0.0) - consumed[donor_class_id])
                if remaining > 0:
                    result["tx_unused_before_reallocation_mb"] += remaining
                if remaining <= 1e-9:
                    continue
                for recv_class_id in self._work_conserving_reallocation_order(donor_class_id):
                    if remaining <= 1e-9:
                        break
                    (delivered, value, on_time_mb, on_time_value,
                     delay_sum, value_delay_sum, events) = \
                        self._remove_from_queue(
                            self.processed_batches,
                            remaining,
                            now_step,
                            apply_timeliness_weight=True,
                            class_id=recv_class_id,
                            value_weight=value_weight,
                            urgency_weight=urgency_weight,
                        )
                    if delivered <= 1e-9:
                        continue
                    recv_name = VALUE_CLASS_NAMES[recv_class_id]
                    recv_specificity = self._specificity_discount(recv_class_id, now_step)
                    value *= recv_specificity
                    on_time_value *= recv_specificity
                    value_delay_sum *= recv_specificity
                    consumed[recv_class_id] += delivered
                    result[f"delivered_{recv_name}_mb"] += delivered
                    result[f"delivered_{recv_name}_value"] += value
                    total_delivered += delivered
                    total_value += value
                    total_on_time_mb += on_time_mb
                    total_on_time_value += on_time_value
                    total_delay += delay_sum
                    total_value_delay += value_delay_sum
                    total_events += events
                    result["tx_reallocated_mb"] += delivered
                    result[f"tx_reallocated_to_{recv_name}_mb"] += delivered
                    remaining -= delivered

        self.total_delivered_mb += total_delivered
        self.total_delivered_value += total_value
        self.total_on_time_delivered_mb += total_on_time_mb
        self.total_on_time_delivered_value += total_on_time_value
        self.total_delivery_delay_steps += total_delay
        self.total_value_weighted_delivery_delay_steps += total_value_delay
        self.delivery_events += total_events

        result.update({
            "delivered_mb": total_delivered,
            "delivered_value": total_value,
            "timely_weighted_delivered_value": total_value,
            "voi_delivered_value": total_value,
            "on_time_delivered_mb": total_on_time_mb,
            "on_time_delivered_value": total_on_time_value,
            "avg_delivery_delay_steps": total_delay / max(total_delivered, 1e-9),
            "aoi_steps": total_delay / max(total_delivered, 1e-9),
            "average_aoi_steps": total_delay / max(total_delivered, 1e-9),
            "value_weighted_aoi_steps": total_value_delay / max(total_value, 1e-9),
        })
        return {key: float(value) for key, value in result.items()}

    def drop_raw(self, mb: float, now_step: int) -> dict:
        """raw_queue 溢出时优先丢弃低价值任务。"""
        breakdown = self._empty_class_breakdown()
        dropped, value, *_ = self._remove_from_queue(
            self.raw_batches, float(max(mb, 0.0)), now_step,
            prefer_low_value=True, class_breakdown=breakdown)
        self.total_dropped_mb += dropped
        self.total_dropped_value += value
        return {
            "dropped_raw_mb": dropped,
            "dropped_raw_value": value,
            **self._prefixed_class_breakdown("dropped_raw", breakdown),
        }

    def drop_processed(self, mb: float, now_step: int) -> dict:
        """processed_queue 溢出时优先丢弃低价值任务。"""
        breakdown = self._empty_class_breakdown()
        dropped, value, *_ = self._remove_from_queue(
            self.processed_batches, float(max(mb, 0.0)), now_step,
            prefer_low_value=True, class_breakdown=breakdown)
        self.total_dropped_mb += dropped
        self.total_dropped_value += value
        return {
            "dropped_processed_mb": dropped,
            "dropped_processed_value": value,
            **self._prefixed_class_breakdown("dropped_processed", breakdown),
        }

    def drop_low_value(self, mb: float, now_step: int, drop_context: dict | None = None) -> dict:
        """主动丢弃动态 Low 类 raw/processed 数据，保留被升类的紧急任务。"""
        drop_context = drop_context or {}
        amount = float(max(mb, 0.0))
        raw_low = sum(batch.mb for batch in self.raw_batches
                      if self.is_droppable_batch(batch, now_step, drop_context))
        proc_low = sum(batch.mb for batch in self.processed_batches
                       if self.is_droppable_batch(batch, now_step, drop_context))
        total_low = max(raw_low + proc_low, 1e-9)
        raw_budget = amount * raw_low / total_low if amount > 1e-9 else 0.0
        proc_budget = amount * proc_low / total_low if amount > 1e-9 else 0.0
        raw_breakdown = self._empty_class_breakdown()
        proc_breakdown = self._empty_class_breakdown()
        dropped_raw, raw_value, *_ = self._remove_from_queue(
            self.raw_batches, raw_budget, now_step,
            prefer_low_value=True, low_value_only=True,
            class_breakdown=raw_breakdown, drop_context=drop_context)
        dropped_proc, proc_value, *_ = self._remove_from_queue(
            self.processed_batches, proc_budget, now_step,
            prefer_low_value=True, low_value_only=True,
            class_breakdown=proc_breakdown, drop_context=drop_context)
        self.total_dropped_mb += dropped_raw + dropped_proc
        self.total_dropped_value += raw_value + proc_value
        total_value = float(raw_value + proc_value)
        breakdown_raw = self._prefixed_class_breakdown("active_dropped_raw", raw_breakdown)
        breakdown_proc = self._prefixed_class_breakdown("active_dropped_processed", proc_breakdown)
        return {
            "active_dropped_low_raw_mb": float(dropped_raw),
            "active_dropped_low_processed_mb": float(dropped_proc),
            "active_dropped_low_value": total_value,
            "active_dropped_total_value": total_value,
            **breakdown_raw,
            **breakdown_proc,
        }

    def expire(self, now_step: int) -> dict:
        """删除超过 deadline 宽限期的任务，并累计过期价值损失。"""
        raw_breakdown = self._empty_class_breakdown()
        proc_breakdown = self._empty_class_breakdown()
        raw_mb, raw_value = self._expire_queue(
            self.raw_batches, now_step, class_breakdown=raw_breakdown)
        proc_mb, proc_value = self._expire_queue(
            self.processed_batches, now_step, class_breakdown=proc_breakdown)
        expired_mb = raw_mb + proc_mb
        expired_value = raw_value + proc_value
        self.total_expired_mb += expired_mb
        self.total_expired_value += expired_value
        return {
            "expired_raw_mb": raw_mb,
            "expired_raw_value": raw_value,
            "expired_processed_mb": proc_mb,
            "expired_processed_value": proc_value,
            "expired_mb": expired_mb,
            "expired_value": expired_value,
            "expired_high_value": float(
                raw_breakdown["high_value"] + proc_breakdown["high_value"]),
            **self._prefixed_class_breakdown("expired_raw", raw_breakdown),
            **self._prefixed_class_breakdown("expired_processed", proc_breakdown),
        }

    def topk_stats(self, now_step: int) -> dict:
        """提取 top-k 高价值/高紧急度任务统计，作为状态向量的一部分。"""
        batches = list(self._active_batches())
        if not batches:
            return {
                "top_task_priority": 0.0,
                "top_task_quality": 0.0,
                "deadline_urgency": 0.0,
                "expiring_value": 0.0,
                "active_value": 0.0,
                "processed_backlog_value": 0.0,
            }

        k = max(1, int(self.cfg.get("top_k", 5)))
        top = sorted(batches, key=lambda b: b.score(
            now_step,
            floor=self._decay_floor(),
            power=self._decay_power(),
        ), reverse=True)[:k]
        urgent_threshold = int(self.cfg.get("urgent_deadline_steps", 30))
        expiring_value = 0.0
        for batch in batches:
            remaining = batch.deadline_steps - batch.age_steps(now_step)
            if remaining <= urgent_threshold:
                expiring_value += batch.value

        return {
            "top_task_priority": float(np.mean([b.priority for b in top])),
            "top_task_quality": float(np.mean([b.quality for b in top])),
            "deadline_urgency": float(np.mean([b.deadline_urgency(now_step) for b in top])),
            "expiring_value": float(expiring_value),
            "active_value": float(sum(b.value for b in batches)),
            "processed_backlog_value": self.processed_value,
        }

    def summary(self) -> dict:
        """返回 episode 级任务交付指标。"""
        value_per_mb = self.total_delivered_value / max(self.total_delivered_mb, 1e-9)
        generated_value = max(self.total_generated_value, 1e-9)
        expired_rate = float(self.total_expired_value / generated_value)
        dropped_rate = float(self.total_dropped_value / generated_value)
        avg_aoi = float(self.total_delivery_delay_steps / max(self.total_delivered_mb, 1e-9))
        proc_dl_ratio = float(self.total_processed_mb / max(self.total_delivered_mb, 1e-9))
        useful_processing_ratio = float(
            self.total_delivered_value / max(self.total_processed_value, 1e-9)
        )
        value_weighted_aoi = float(
            self.total_value_weighted_delivery_delay_steps
            / max(self.total_delivered_value, 1e-9)
        )
        value_weighted_deadline_success = float(
            self.total_on_time_delivered_value
            / max(self.total_delivered_value, 1e-9)
        )
        return {
            "generated_mb": float(self.total_generated_mb),
            "generated_value": float(self.total_generated_value),
            "processed_mb": float(self.total_processed_mb),
            "processed_value": float(self.total_processed_value),
            "delivered_mb": float(self.total_delivered_mb),
            "delivered_value": float(self.total_delivered_value),
            "proc_dl_ratio": proc_dl_ratio,
            "useful_processing_ratio": useful_processing_ratio,
            "deadline_success_rate": value_weighted_deadline_success,
            "value_weighted_deadline_success_rate": value_weighted_deadline_success,
            "expired_value_rate": expired_rate,
            "dropped_value_rate": dropped_rate,
            "avg_delivery_delay_steps": avg_aoi,
            "average_aoi_steps": avg_aoi,
            "value_weighted_aoi_steps": value_weighted_aoi,
            "voi_degradation_rate": expired_rate,
            "voi_loss_rate": float(np.clip(expired_rate + dropped_rate, 0.0, 1e9)),
            "value_per_mb": float(value_per_mb),
        }

    def _active_batches(self) -> Iterable[TaskBatch]:
        yield from self.raw_batches
        yield from self.processed_batches

    def _specificity_discount(self, class_id: int, now_step: int) -> float:
        """sigma = 1 / (1 + gamma * concurrent_same_class_mb / scale_mb)
        Discounts delivered value when many same-class tasks compete for limited capacity,
        matching Figure D.2: effective_value = intrinsic * freshness * specificity.
        """
        gamma = float(self.cfg.get("specificity_gamma", 1.0))
        if gamma <= 1e-9:
            return 1.0
        scale_mb = max(float(self.cfg.get("specificity_scale_mb", 256.0)), 1e-6)
        concurrent_mb = sum(
            b.mb for b in (*self.raw_batches, *self.processed_batches)
            if self.task_class_id(b, now_step) == class_id
        )
        return 1.0 / (1.0 + gamma * float(concurrent_mb) / scale_mb)

    def _class_reservation_mb(self, class_id: int) -> float:
        reservations = self.cfg.get("deliverability_reservation_by_class", (0.75, 0.35, 0.0))
        try:
            return float(reservations[int(class_id)])
        except (IndexError, TypeError, ValueError):
            return 0.0

    def _move_between_queues(self, source: list[TaskBatch], dest: list[TaskBatch],
                             amount_mb: float, now_step: int,
                             class_id: int | None = None,
                             value_weight: float = 1.0,
                             urgency_weight: float = 0.0,
                             future_capacity_fn=None) -> tuple[float, float, float, float]:
        moved = 0.0
        value = 0.0
        deliverable_value = 0.0
        undeliverable_value = 0.0
        for idx in self._queue_order(
                source, now_step, prefer_low_value=False, class_id=class_id,
                value_weight=value_weight, urgency_weight=urgency_weight):
            if amount_mb <= 1e-9:
                break
            batch = source[idx]
            take = min(batch.mb, amount_mb)
            moved += take
            take_value = batch.value * take / max(batch.mb, 1e-9)
            value += take_value
            deliver_prob = 1.0
            if future_capacity_fn is not None:
                deadline_abs = int(batch.created_step) + int(batch.deadline_steps)
                future_cap = float(future_capacity_fn(deadline_abs))
                backlog = self._queue_value_before_deadline(
                    self.processed_batches, now_step, deadline_abs, class_id=class_id)
                same_raw = self._queue_value_before_deadline(
                    source, now_step, deadline_abs, class_id=class_id)
                same_raw = max(0.0, same_raw - take)
                reservation = self._class_reservation_mb(int(class_id or self.task_class_id(batch, now_step))) * same_raw
                deliver_prob = self.deliverability_for_batch(
                    batch,
                    now_step,
                    future_cap,
                    backlog,
                    reservation,
                    amount_mb=take,
                )
            deliverable = take_value * deliver_prob
            deliverable_value += deliverable
            undeliverable_value += max(0.0, take_value - deliverable)
            dest.append(self._make_batch(
                mb=take,
                value=take_value,
                priority=batch.priority,
                quality=batch.quality,
                deadline_steps=batch.deadline_steps,
                created_step=batch.created_step,
                scene_name=batch.scene_name,
                scene_class_code=batch.scene_class_code,
                cloud_cover=batch.cloud_cover,
                profile={
                    "freshness_profile": batch.freshness_profile,
                    "freshness_power": batch.freshness_power,
                    "freshness_peak_fraction": batch.freshness_peak_fraction,
                    "freshness_late_floor": batch.freshness_late_floor,
                },
            ))
            batch.mb -= take
            batch.value -= take_value
            amount_mb -= take
        self._compact(source)
        return float(moved), float(value), float(deliverable_value), float(undeliverable_value)

    def _remove_from_queue(self, queue: list[TaskBatch], amount_mb: float,
                           now_step: int, prefer_low_value: bool = False,
                           apply_timeliness_weight: bool = False,
                           class_id: int | None = None,
                           low_value_only: bool = False,
                           class_breakdown: dict | None = None,
                           drop_context: dict | None = None,
                           value_weight: float = 1.0,
                           urgency_weight: float = 0.0) -> tuple:
        removed = value = on_time_mb = on_time_value = delay_sum = value_delay_sum = 0.0
        events = 0
        decay_floor = self._decay_floor()
        decay_power = self._decay_power()
        overdue_grace_steps = int(self.cfg.get("overdue_grace_steps", 0))
        overdue_decay_rate = float(self.cfg.get("overdue_decay_rate", 4.0))
        for idx in self._queue_order(
                queue, now_step, prefer_low_value=prefer_low_value,
                class_id=class_id, low_value_only=low_value_only,
                drop_context=drop_context,
                value_weight=value_weight, urgency_weight=urgency_weight):
            if amount_mb <= 1e-9:
                break
            batch = queue[idx]
            take = min(batch.mb, amount_mb)
            nominal_value = batch.value * take / max(batch.mb, 1e-9)
            if apply_timeliness_weight:
                weight = batch.timeliness_weight(
                    now_step, floor=decay_floor, power=decay_power,
                    overdue_grace_steps=overdue_grace_steps,
                    overdue_decay_rate=overdue_decay_rate)
                take_value = nominal_value * weight
            else:
                take_value = nominal_value
            age = batch.age_steps(now_step)
            removed_class_id = self.task_nominal_class_id(batch)
            removed += take
            value += take_value
            if class_breakdown is not None:
                class_name = VALUE_CLASS_NAMES[removed_class_id]
                class_breakdown[f"{class_name}_mb"] += float(take)
                class_breakdown[f"{class_name}_value"] += float(take_value)
            if age <= batch.deadline_steps:
                on_time_mb += take
                on_time_value += take_value
            delay_sum += age * take
            value_delay_sum += age * take_value
            events += 1
            batch.mb -= take
            batch.value -= nominal_value
            amount_mb -= take
        self._compact(queue)
        return (float(removed), float(value), float(on_time_mb),
                float(on_time_value), float(delay_sum),
                float(value_delay_sum), int(events))

    def _expire_queue(self, queue: list[TaskBatch], now_step: int,
                      class_breakdown: dict | None = None) -> tuple[float, float]:
        expired_mb = expired_value = 0.0
        grace_steps = max(0, int(self.cfg.get("overdue_grace_steps", 0)))
        keep = []
        for batch in queue:
            if batch.age_steps(now_step) > batch.deadline_steps + grace_steps:
                expired_mb += batch.mb
                expired_value += batch.value
                if class_breakdown is not None:
                    class_name = VALUE_CLASS_NAMES[self.task_nominal_class_id(batch)]
                    class_breakdown[f"{class_name}_mb"] += float(batch.mb)
                    class_breakdown[f"{class_name}_value"] += float(batch.value)
            else:
                keep.append(batch)
        queue[:] = keep
        return float(expired_mb), float(expired_value)

    def _queue_order(self, queue: list[TaskBatch], now_step: int,
                     prefer_low_value: bool,
                     class_id: int | None = None,
                     low_value_only: bool = False,
                     drop_context: dict | None = None,
                     value_weight: float = 1.0,
                     urgency_weight: float = 0.0,
                     class_priority_first: bool = False) -> list[int]:
        """返回按 score（或 (class_id, score) 元组）排好序的 queue 索引。

        class_priority_first=True 时把 class_id 作为主键（0=high 优先），同类内再
        按 score 细排——这等价于 "类优先级 floor"，但是单次 sorted 调用，避免
        外层迭代 3 次 _queue_order 的 O(3N log N) 浪费。
        """
        indices = []
        for idx, batch in enumerate(queue):
            if class_id is not None and self.task_class_id(batch, now_step) != int(class_id):
                continue
            if low_value_only and not self.is_droppable_batch(batch, now_step, drop_context):
                continue
            indices.append(idx)

        floor = self._decay_floor()
        power = self._decay_power()
        grace = int(self.cfg.get("overdue_grace_steps", 0))
        decay_rate = float(self.cfg.get("overdue_decay_rate", 4.0))

        if class_priority_first:
            # ── Phase 2 硬规则 C：分层 EDF ──
            # 主键：class_id 升序（0=high 优先于 mid 优先于 low）
            # 次键：deadline_tight bool（True=紧→0 在前；False=松→1 在后）
            # 末键：score 降序（同 tier 内按 value×urgency）
            # 这样保证：deadline 紧的 low **永远不会插队**普通 high
            # （tier=(2,0,*) ≺ tier=(0,1,*) 还是 tier=(0,1,*) 更小）。
            try:
                from config import HARD_RULES_CONFIG as _HR_CFG  # noqa: WPS433
            except Exception:
                _HR_CFG = {}
            edf_tight_steps = int(_HR_CFG.get("edf_tight_deadline_steps", 10))
            enable_edf = bool(_HR_CFG.get("enable_layered_edf", False))

            def _key(i):
                b = queue[i]
                cid = self.task_class_id(b, now_step)
                sc = b.score(
                    now_step, floor=floor, power=power,
                    overdue_grace_steps=grace, overdue_decay_rate=decay_rate,
                    value_weight=value_weight, urgency_weight=urgency_weight,
                )
                if enable_edf:
                    deadline_remaining = b.deadline_steps - b.age_steps(now_step)
                    tight = 0 if deadline_remaining <= edf_tight_steps else 1
                    return (cid, tight, -sc)
                return (cid, -sc)
            return sorted(indices, key=_key)

        return sorted(
            indices,
            key=lambda i: queue[i].score(
                now_step, floor=floor, power=power,
                overdue_grace_steps=grace, overdue_decay_rate=decay_rate,
                value_weight=value_weight, urgency_weight=urgency_weight,
            ),
            reverse=not prefer_low_value,
        )

    @staticmethod
    def _empty_class_breakdown() -> dict:
        out = {}
        for name in VALUE_CLASS_NAMES:
            out[f"{name}_mb"] = 0.0
            out[f"{name}_value"] = 0.0
        return out

    @staticmethod
    def _prefixed_class_breakdown(prefix: str, breakdown: dict) -> dict:
        out = {}
        for name in VALUE_CLASS_NAMES:
            out[f"{prefix}_{name}_mb"] = float(breakdown.get(f"{name}_mb", 0.0))
            out[f"{prefix}_{name}_value"] = float(breakdown.get(f"{name}_value", 0.0))
        return out

    @staticmethod
    def _compact(queue: list[TaskBatch]):
        queue[:] = [batch for batch in queue if batch.mb > 1e-9 and batch.value > 1e-9]

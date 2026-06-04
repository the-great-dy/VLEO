"""
面向时效任务价值交付的规则基线。

这些基线不使用神经网络，只读取环境中的任务价值统计和队列压力，
用固定规则选择连续资源分配动作。
"""

import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

import numpy as np
from config import ORBITAL_CONFIG, ENERGY_CONFIG, TASK_CONFIG
from utils.action_space import (
    default_grouped_action,
    CPU_LOGITS_SLICE,
    TX_LOGITS_SLICE,
    IDX_DROP_LOW,
    VALUE_CLASS_COUNT,
)


class StaticRuleBaseline:
    """静态规则基线：固定推进/计算/通信比例，只做必要安全降级。"""

    def schedule(self, state, env) -> np.ndarray:
        del state
        contact = getattr(env, "_contact", None) or {}
        in_window = bool(contact.get("in_window", False))
        soc = float(getattr(env.battery, "soc", 0.5))
        altitude_margin_km = (
            float(env.altitude_m) / 1e3
            - float(ORBITAL_CONFIG["altitude_min_km"])
        )

        # 固定比例是 baseline 的核心；下面只做保命级安全规则，避免明显非法动作。
        if soc < float(ENERGY_CONFIG.get("battery_min_soc", 0.15)):
            return default_grouped_action([0.20 if altitude_margin_km > 10.0 else 0.45, 0.0, 0.0])
        if altitude_margin_km < 20.0:
            return default_grouped_action([0.75, 0.10, 0.0 if not in_window else 0.10])
        if not in_window:
            return default_grouped_action([0.30, 0.25, 0.0])
        return default_grouped_action([0.30, 0.40, 0.60])


class _ValueRuleBase:
    def _common_action(
        self,
        env,
        urgency: float,
        value_pressure: float,
        *,
        cpu_logits: np.ndarray | None = None,
        tx_logits: np.ndarray | None = None,
        drop_low_strength: float = 0.0,
    ) -> np.ndarray:
        """根据共同的轨道/能源/队列安全逻辑生成动作。"""
        contact = getattr(env, "_contact", None) or {}
        in_window = bool(contact.get("in_window", False))
        raw_util = float(env.data_queue.length / max(env.data_queue.max_length, 1e-6))
        proc_util = float(env.comm_queue.value / max(env.comm_queue.max_value, 1e-6))
        soc = float(getattr(env.battery, "soc", 0.5))
        altitude_margin_km = (
            float(env.altitude_m) / 1e3
            - float(ORBITAL_CONFIG["altitude_min_km"])
        )

        alpha_prop = 0.25
        if altitude_margin_km < 20.0:
            alpha_prop = 0.75
        elif altitude_margin_km < 50.0:
            alpha_prop = 0.45

        if proc_util > 0.85 and not in_window:
            alpha_cpu = 0.05
        else:
            alpha_cpu = np.clip(0.15 + 0.65 * raw_util + 0.2 * value_pressure, 0.0, 1.0)

        alpha_tx = 0.0
        if in_window:
            alpha_tx = np.clip(0.2 + 0.7 * proc_util + 0.2 * urgency, 0.0, 1.0)

        if soc < float(ENERGY_CONFIG.get("battery_min_soc", 0.15)) + 0.10:
            alpha_cpu *= 0.35
            alpha_tx *= 0.45
            alpha_prop = max(alpha_prop, 0.35)

        action = default_grouped_action([alpha_prop, alpha_cpu, alpha_tx])
        # 调用方传完整 3-class logits(high/mid/low)，写入各自专用的 class logit 维度。
        if cpu_logits is not None:
            cpu_logits_arr = np.clip(np.asarray(cpu_logits, dtype=np.float32), 0.0, 1.0)
            cpu_logits_arr = np.resize(cpu_logits_arr, VALUE_CLASS_COUNT)
            action[CPU_LOGITS_SLICE] = cpu_logits_arr
        if tx_logits is not None:
            tx_logits_arr = np.clip(np.asarray(tx_logits, dtype=np.float32), 0.0, 1.0)
            tx_logits_arr = np.resize(tx_logits_arr, VALUE_CLASS_COUNT)
            action[TX_LOGITS_SLICE] = tx_logits_arr
        action[IDX_DROP_LOW] = float(np.clip(drop_low_strength, 0.0, 1.0))
        return action


class GreedyValueBaseline(_ValueRuleBase):
    """优先处理和下传当前价值密度最高的任务。"""

    def schedule(self, state, env) -> np.ndarray:
        del state
        stats = env.task_tracker.topk_stats(env.step_count)
        priority = float(stats.get("top_task_priority", 0.0))
        quality = float(stats.get("top_task_quality", 0.0))
        urgency = float(stats.get("deadline_urgency", 0.0))
        value_pressure = np.clip((priority * quality) / 1.8, 0.0, 1.0)
        logits = np.array([1.0, 0.45 + 0.35 * urgency, 0.15], dtype=np.float32)
        return self._common_action(
            env, urgency=urgency, value_pressure=value_pressure,
            cpu_logits=logits, tx_logits=logits)


class EDFBaseline(_ValueRuleBase):
    """EDF 基线：主要按 deadline 紧迫度分配计算和通信资源。"""

    def schedule(self, state, env) -> np.ndarray:
        del state
        stats = env.task_tracker.topk_stats(env.step_count)
        urgency = float(stats.get("deadline_urgency", 0.0))
        expiring = float(stats.get("expiring_value", 0.0))
        value_norm = max(float(TASK_CONFIG.get("value_norm", 500.0)), 1e-6)
        value_pressure = np.clip(expiring / value_norm, 0.0, 1.0)
        logits = np.array([0.85 + 0.15 * urgency, 0.55, 0.20], dtype=np.float32)
        return self._common_action(
            env, urgency=urgency, value_pressure=value_pressure,
            cpu_logits=logits, tx_logits=logits)


class LLFBaseline(_ValueRuleBase):
    """LLF baseline: approximate least-laxity-first scheduling."""

    def schedule(self, state, env) -> np.ndarray:
        del state
        stats = env.task_tracker.topk_stats(env.step_count)
        urgency = float(stats.get("deadline_urgency", 0.0))
        expiring = float(stats.get("expiring_value", 0.0))
        value_norm = max(float(TASK_CONFIG.get("value_norm", 500.0)), 1e-6)
        laxity_pressure = float(np.clip(0.25 + 0.75 * urgency, 0.0, 1.0))
        value_pressure = float(np.clip(
            0.5 * laxity_pressure + 0.5 * expiring / value_norm, 0.0, 1.0))
        logits = np.array([
            0.70 + 0.30 * laxity_pressure,
            0.65,
            0.15 * (1.0 - laxity_pressure),
        ], dtype=np.float32)
        return self._common_action(
            env,
            urgency=laxity_pressure,
            value_pressure=value_pressure,
            cpu_logits=logits,
            tx_logits=logits,
            drop_low_strength=0.20 * laxity_pressure,
        )

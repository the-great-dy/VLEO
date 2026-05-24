"""论文 CMDP 目标的任务奖励。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MissionRewardBreakdown:
    """r_t 的奖励成分；安全代价按设计被排除。"""

    total: float
    delivered_value: float
    on_time_delivered_value: float
    expired_value: float
    dropped_value: float
    energy_wh: float
    processed_deliverable_value: float
    processed_undeliverable_value: float
    components: dict


def compute_mission_reward(
    *,
    delivered_value: float,
    on_time_delivered_value: float,
    expired_value: float,
    dropped_value: float,
    dropped_mb: float = 0.0,
    transmitted_mb: float,
    processed_mb: float,
    total_power_w: float,
    dt_s: float,
    cfg: dict,
    deliverable_processing_credit_value: float = 0.0,
    processed_value: float = 0.0,
    processed_deliverable_value: float = 0.0,
    processed_undeliverable_value: float = 0.0,
    time_to_next_window_norm: float = 0.0,
    prospective_deliver_prob: float = 1.0,
    actuator_violation_mb: float = 0.0,
    # A2 类加权：让 critic 知道 high 比 low 重要不止是 value_density 那一项 ——
    # 这三个字段允许把 r_value 拆成 (w_h·v_h + w_m·v_m + w_l·v_l)·w_delivered_value，
    # 不传时退化到原来的 r_value = w_delivered_value · delivered_value（向后兼容）。
    delivered_high_value: float | None = None,
    delivered_mid_value: float | None = None,
    delivered_low_value: float | None = None,
) -> MissionRewardBreakdown:
    """计算干净的任务奖励目标 r_t。"""
    delivered_value = float(delivered_value)
    on_time_delivered_value = float(on_time_delivered_value)
    expired_value = float(expired_value)
    dropped_value = float(dropped_value)
    dropped_mb = max(0.0, float(dropped_mb))
    transmitted_mb = float(transmitted_mb)
    processed_mb = max(0.0, float(processed_mb))
    processed_value = max(0.0, float(processed_value))
    processed_deliverable_value = max(0.0, float(processed_deliverable_value))
    processed_undeliverable_value = max(0.0, float(processed_undeliverable_value))
    window_far = max(0.0, min(1.0, float(time_to_next_window_norm)))
    deliver_prob = max(0.0, min(1.0, float(prospective_deliver_prob)))
    actuator_violation_mb = max(0.0, float(actuator_violation_mb))
    deliverable_processing_credit_value = max(
        0.0, float(deliverable_processing_credit_value))
    energy_wh = max(0.0, float(total_power_w) * float(dt_s) / 3600.0)

    r_drop_penalty = 0.0
    r_drop_mb_penalty = 0.0
    r_expired_penalty = 0.0
    r_energy_penalty = 0.0
    r_processing_penalty = 0.0
    r_processing_deliverable = 0.0
    r_processing_opportunity_cost = 0.0
    r_proc_far_window = 0.0
    r_prospective_expiry = 0.0
    r_actuator_violation = 0.0
    processed_into_headroom_mb = 0.0
    processed_into_overflow_mb = 0.0
    excess_energy_wh = 0.0

    reward_mode = str(cfg.get("reward_mode", "value_aware")).lower()
    if reward_mode in {"throughput", "delivered_mb", "non_value"}:
        r_value = 0.0
        r_deadline = 0.0
        r_throughput = float(cfg.get("w_delivered_mb", 1.0)) * transmitted_mb
        r_processing_credit = 0.0
        total = r_throughput
        objective = "throughput"
    else:
        w_v = float(cfg.get("w_delivered_value", 1.0))
        # A2 class-aware reward：按 critic 是否看到分类 breakdown 决定走哪条。
        # value_density 已经把 priority×quality 算进去了 (~200x 类间差距)，但 critic
        # 拿到的是 reward 信号，replay buffer 里 high 类样本本身就稀少；显式 class 权重
        # 让 critic 对 high 的 gradient 强度再多 ~3x（不只是 sample 频次决定的）。
        use_class_split = (
            delivered_high_value is not None
            and delivered_mid_value is not None
            and delivered_low_value is not None
            and bool(cfg.get("enable_class_weighted_reward", True))
        )
        if use_class_split:
            w_h = float(cfg.get("class_high_reward_weight", 3.0))
            w_m = float(cfg.get("class_mid_reward_weight", 1.5))
            w_l = float(cfg.get("class_low_reward_weight", 0.5))
            class_weighted = (
                w_h * float(delivered_high_value)
                + w_m * float(delivered_mid_value)
                + w_l * float(delivered_low_value)
            )
            r_value = w_v * class_weighted
        else:
            r_value = w_v * delivered_value
        r_deadline = float(cfg.get("w_deadline_success", 0.0)) * on_time_delivered_value
        r_throughput = 0.0
        r_processing_credit = (
            float(cfg.get("w_deliverable_processing", 0.0))
            * deliverable_processing_credit_value
        )

        processed_q_mb = float(cfg.get("_processed_queue_mb", 0.0))
        future_cap_mb = float(cfg.get("_future_contact_capacity_mb", 0.0))
        margin = float(cfg.get("processing_capacity_margin", 0.70))
        headroom_mb = max(0.0, margin * future_cap_mb - processed_q_mb)
        processed_into_overflow_mb = max(0.0, processed_mb - headroom_mb)
        processed_into_headroom_mb = processed_mb - processed_into_overflow_mb

        r_processing_penalty = (
            float(cfg.get("w_processing_penalty_useful", 0.0)) * processed_into_headroom_mb
            + float(cfg.get("w_processing_penalty_overflow", 0.0)) * processed_into_overflow_mb
        )
        r_drop_penalty = float(cfg.get("w_drop_penalty", 0.0)) * dropped_value
        r_drop_mb_penalty = float(cfg.get("w_drop_mb_penalty", 0.0)) * dropped_mb
        r_expired_penalty = float(cfg.get("w_expired_penalty", 0.0)) * expired_value

        energy_budget_wh = max(0.0, float(cfg.get("energy_budget_wh_per_step", 0.0)))
        excess_energy_wh = max(0.0, energy_wh - energy_budget_wh)
        r_energy_penalty = float(cfg.get("w_energy_over_budget_penalty", 0.0)) * excess_energy_wh
        r_energy_penalty += float(cfg.get("w_energy_penalty", 0.0)) * energy_wh

        if (
            processed_deliverable_value <= 1e-12
            and processed_undeliverable_value <= 1e-12
            and processed_value > 0.0
        ):
            processed_deliverable_value = processed_value * deliver_prob
            processed_undeliverable_value = max(0.0, processed_value - processed_deliverable_value)
        r_processing_deliverable = (
            float(cfg.get("w_processing_deliverable_value", 0.0))
            * processed_deliverable_value
        )
        r_processing_opportunity_cost = -float(
            cfg.get("w_processing_opportunity_cost", 0.0)
        ) * processed_undeliverable_value

        w_prospective = float(cfg.get("w_prospective_expiry_shaping", 0.0))
        r_prospective_expiry = w_prospective * processed_value * (1.0 - deliver_prob)

        w_act_violation = float(cfg.get("w_actuator_violation_penalty", 0.0))
        r_actuator_violation = w_act_violation * actuator_violation_mb

        # ── 远窗口处理连续 shaping (替代死代码 r_proc_far_window=0) ──
        # 之前的设计有 gap：CPU gate 在 t>120s 收紧、far_cpu 指标 t>300s 才计数，
        # 中间这段既无 reward 反馈也无诊断。这里用连续 ramp 替代二值阈值：
        #   ≤ lead_s (默认 120s): 0 penalty (近窗口处理合理)
        #   lead_s ~ saturation_s: 线性 ramp 上升
        #   ≥ saturation_s (默认 600s): 满 penalty
        # 由此 agent 总能拿到平滑 gradient，知道"越远越不该处理"。
        w_proc_far = float(cfg.get("w_proc_far_window_penalty", 0.0))
        in_window_now = bool(cfg.get("_in_comm_window", False))
        if w_proc_far > 0.0 and processed_mb > 0.0 and not in_window_now:
            t_to_win = float(cfg.get("_time_to_next_window_s", 0.0))
            lead_s = float(cfg.get("proc_far_window_lead_s", 120.0))
            sat_s = float(cfg.get("proc_far_window_saturation_s", 600.0))
            ramp_span = max(sat_s - lead_s, 1.0)
            far_strength = float(np.clip((t_to_win - lead_s) / ramp_span, 0.0, 1.0))
            r_proc_far_window = -w_proc_far * float(processed_mb) * far_strength

        total = (r_value + r_deadline + r_processing_credit
                 + r_processing_deliverable + r_processing_opportunity_cost
                 + r_drop_penalty + r_drop_mb_penalty
                 + r_expired_penalty + r_energy_penalty + r_processing_penalty
                 + r_proc_far_window + r_prospective_expiry
                 + r_actuator_violation)
        objective = "value_aware_deliverability_gated"

    components = {
        "r_delivered_value": r_value,
        "r_deadline_success": r_deadline,
        "r_delivered_mb": r_throughput,
        "r_deliverable_processing": r_processing_credit,
        "r_processing_deliverable_value": r_processing_deliverable,
        "r_processing_opportunity_cost": r_processing_opportunity_cost,
        "r_drop_penalty": r_drop_penalty,
        "r_drop_mb_penalty": r_drop_mb_penalty,
        "r_expired_penalty": r_expired_penalty,
        "r_energy_penalty": r_energy_penalty,
        "r_processing_penalty": r_processing_penalty,
        "r_proc_far_window": r_proc_far_window,
        "r_prospective_expiry": r_prospective_expiry,
        "r_actuator_violation": r_actuator_violation,
        "window_far_norm": window_far,
        "prospective_deliver_prob": deliver_prob,
        "actuator_violation_mb": actuator_violation_mb,
        "processed_value_step": processed_value,
        "processed_deliverable_value_step": processed_deliverable_value,
        "processed_undeliverable_value_step": processed_undeliverable_value,
        "processed_into_headroom_mb": processed_into_headroom_mb,
        "processed_into_overflow_mb": processed_into_overflow_mb,
        "deliverable_processing_credit_value": deliverable_processing_credit_value,
        "energy_wh": energy_wh,
        "energy_budget_wh": energy_budget_wh if reward_mode not in {"throughput", "delivered_mb", "non_value"} else 0.0,
        "excess_energy_wh": excess_energy_wh,
        "processed_mb": processed_mb,
        "transmitted_mb": transmitted_mb,
        "reward_objective": objective,
    }
    return MissionRewardBreakdown(
        total=float(total),
        delivered_value=delivered_value,
        on_time_delivered_value=on_time_delivered_value,
        expired_value=expired_value,
        dropped_value=dropped_value,
        energy_wh=float(energy_wh),
        processed_deliverable_value=float(processed_deliverable_value),
        processed_undeliverable_value=float(processed_undeliverable_value),
        components=components,
    )

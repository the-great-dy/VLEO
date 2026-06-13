"""Paper CMDP safety-cost definition for the constraint critic.

重构后的 CMDP 安全代价 c_t 的核心项：

1.  ``state_safety_penalty``  : 阶段化的硬安全惩罚(轨道/能量/热的 warning/unsafe/failure)。
2.  ``queue_risk_penalties``  : 软+硬队列压力(原始队列利用率与 overflow)。
3.  ``over_processing_cost``  : 容量感知的累计处理惩罚——proc/dl 的主约束。
4.  ``task_loss_penalty``    : 高价值任务过期/丢弃的惩罚。
5.  ``low_value_waste_cost`` : 低价值数据占用 CPU 的额外惩罚（默认关闭，保留作消融）。
6.  ``unproductive_cpu_cost``: 窗口远且 processed_queue 饱和时继续处理的浪费惩罚（默认关闭，保留作消融）。

旧的 3 个冗余 cost 项(efficiency / processed_backlog / window_waste)已完全移除，
固定返回 0.0，仅保留字段以维持 logger/test 的向后兼容。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (
    DRL_CONFIG,
    ENERGY_CONFIG,
    ORBITAL_CONFIG,
    TASK_CONFIG,
    THERMAL_CONFIG,
)


@dataclass(frozen=True)
class SafetyCostBreakdown:
    """约束 Critic 学习的安全代价及其组成。

    重构后字段精简为论文需要的核心项;为保持向后兼容,
    保留了几个被遗留 logger 使用的字段并固定为 0.0。
    """

    # 核心约束项
    lyapunov_drift: float
    positive_lyapunov_drift: float
    queue_soft_penalty: float
    queue_hard_penalty: float
    over_processing_cost: float
    thermal_excess_penalty: float
    task_loss_penalty: float
    state_safety_penalty: float
    queue_cost: float
    energy_cost: float
    orbit_cost: float
    thermal_cost: float
    task_loss_cost: float
    # 训练损失与诊断
    total_cost: float
    training_cost: float
    normalized_cost: float
    training_cost_clip: float
    training_cost_clip_saturation: float
    dual_cost: float
    dual_cost_norm: float
    dual_violation_signal: float
    soft_constraint_cost: float
    hard_violation_cost: float
    raw_cost: float
    clipped_cost: float
    # over_processing 的详细诊断
    over_processing_raw_cost: float
    over_processing_normalized_cost: float
    over_processing_training_cost: float
    over_processing_clip_saturation: float
    backlog_excess_mb: float
    admission_excess_mb: float
    clearable_capacity_mb: float
    over_processing_ratio: float
    # 向后兼容字段：
    # - processed_backlog / window_waste / efficiency 已被移除，固定为 0.0
    # - low_value_waste / unproductive_cpu 默认系数为 0，仅保留消融入口
    processed_backlog_cost: float = 0.0
    window_waste_cost: float = 0.0
    low_value_waste_cost: float = 0.0
    unproductive_cpu_cost: float = 0.0
    efficiency_cost: float = 0.0
    # ── 顶刊 Issue#1: clean CMDP 约束分解（始终计算，便于日志；是否进 critic 取决于 flag）──
    constraint_total_clean: float = 0.0   # 物理安全+队列稳定（已剔除 QoS）
    qos_total: float = 0.0                # task_loss+over_processing+low_value_waste+unproductive_cpu
    constraint_state_safety_cost: float = 0.0  # 硬状态安全（折入 physical）
    constraint_drift_cost: float = 0.0         # Lyapunov 正向漂移（折入 queue 稳定）
    clean_constraint_cost_used: float = 0.0    # 1.0 表示本次 critic 用的是 clean cost


def _soft_cap_cost(value: float, cap: float) -> float:
    value = max(0.0, float(value))
    cap = max(0.0, float(cap))
    if cap <= 0.0 or value <= cap:
        return value
    return float(cap + cap * np.log1p((value - cap) / cap))


def compute_queue_risk_penalties(
    qd: float,
    qc: float,
    qd_max: float,
    qc_max: float,
    info: dict | None,
    cfg: dict | None = None,
    include_processed_backlog: bool = False,
) -> tuple[float, float]:
    """Return soft queue pressure and hard overflow costs.

    重构后 ``include_processed_backlog`` 默认关闭——backlog 的语义已被
    ``over_processing_cost`` 统一覆盖;保留参数只为兼容旧调用方。
    """
    cfg = cfg or DRL_CONFIG
    info = info or {}
    soft_start = float(cfg.get("lya_soft_util_threshold", 0.75))
    soft_coeff = float(cfg.get("lya_soft_util_penalty_coeff", 0.5))
    soft_clip = float(cfg.get("lya_soft_penalty_clip", 1.0))
    hard_coeff = float(cfg.get("lya_hard_overflow_penalty_coeff", 3.0))
    hard_clip = float(cfg.get("lya_hard_penalty_clip", 2.0))

    util_d = float(np.clip(qd / max(qd_max, 1e-6), 0.0, 1.0))
    util_c = float(np.clip(qc / max(qc_max, 1e-6), 0.0, 1.0))
    norm = max(1.0 - soft_start, 1e-6)
    soft_d = max(0.0, util_d - soft_start) / norm
    soft_c = max(0.0, util_c - soft_start) / norm
    soft_penalty = soft_coeff * (soft_d ** 2 + soft_c ** 2)
    soft_penalty = float(np.clip(soft_penalty, 0.0, soft_clip))

    overflow_d_mb = max(
        0.0,
        float(info.get("overflow_mb", 0.0)),
        float(info.get("raw_queue_overflow_mb", 0.0)),
        max(qd - qd_max, 0.0),
    )
    overflow_c_mb = max(
        0.0,
        float(info.get("comm_overflow_mb", 0.0)),
        float(info.get("processed_queue_overflow_mb", 0.0)),
        max(qc - qc_max, 0.0),
    )
    hard_penalty = hard_coeff * (
        overflow_d_mb / max(qd_max, 1e-6)
        + overflow_c_mb / max(qc_max, 1e-6)
    )
    hard_penalty = float(np.clip(hard_penalty, 0.0, hard_clip))
    return float(soft_penalty), float(hard_penalty)


def compute_state_safety_penalty(
    info: dict | None,
    cfg: dict | None = None,
) -> float:
    """将不安全的物理状态映射到c_t的硬违反部分。"""
    cfg = cfg or DRL_CONFIG
    info = info or {}

    stage_costs = dict(cfg.get("constraint_stage_costs", {}) or {})
    warning_cost = float(stage_costs.get(
        "warning", cfg.get("constraint_warning_cost", 0.08)))
    unsafe_cost = float(stage_costs.get(
        "unsafe", cfg.get("constraint_unsafe_cost", 0.8)))
    failure_cost = float(stage_costs.get(
        "failure", cfg.get("constraint_failure_cost", 3.0)))
    auxiliary_violation_cost = float(cfg.get(
        "constraint_auxiliary_violation_cost",
        cfg.get("constraint_power_violation_cost", 0.25),
    ))
    thermal_warning_cost = float(cfg.get(
        "constraint_thermal_warning_cost", warning_cost))

    cost = 0.0
    stage = str(info.get("risk_stage", "normal")).lower()
    primary_failure = bool(
        info.get("crashed", False)
        or info.get("orbit_crashed", False)
        or info.get("energy_crashed", False)
        or info.get("failure_state", False)
    )
    primary_failure = primary_failure or stage == "failure"
    if primary_failure:
        cost += failure_cost
    elif stage == "unsafe":
        cost += unsafe_cost
    elif stage == "warning":
        cost += warning_cost

    thermal_stage = str(info.get("thermal_stage", "normal")).lower()
    if thermal_stage in {"critical", "failure"} or bool(info.get("thermal_crashed", False)):
        if not primary_failure:
            cost += failure_cost
    elif thermal_stage in {"warning", "unsafe"} or not bool(info.get("thermal_safe", True)):
        cost += thermal_warning_cost

    if not bool(info.get("power_constraint_safe", True)):
        cost += auxiliary_violation_cost

    return float(max(0.0, cost))


def compute_task_loss_penalty(
    info: dict | None,
    cfg: dict | None = None,
) -> float:
    """高价值过期/丢弃任务损失的约束项。"""
    cfg = cfg or DRL_CONFIG
    info = info or {}
    high_value_loss = max(0.0, float(info.get("expired_high_value", 0.0)))
    high_value_loss += max(0.0, float(info.get("dropped_high_value", 0.0)))
    norm = max(1e-6, float(cfg.get(
        "constraint_task_loss_value_norm",
        TASK_CONFIG.get("value_norm", 5000.0),
    )))
    coeff = max(0.0, float(cfg.get("constraint_high_value_loss_coeff", 1.0)))
    clip = max(0.0, float(cfg.get("constraint_task_loss_clip", 2.0)))
    penalty = float(np.clip(coeff * high_value_loss / norm, 0.0, clip))

    # 训练早期对 high-value loss 约束做平滑启用,避免探索阶段导致 cost 爆发式抖动。
    global_step_raw = info.get("global_step", None)
    if global_step_raw is None:
        return penalty
    try:
        global_step = max(0, int(global_step_raw))
    except Exception:
        return penalty

    warmup_steps = max(0, int(cfg.get("constraint_task_loss_warmup_steps", 0)))
    anneal_steps = max(0, int(cfg.get("constraint_task_loss_anneal_steps", 0)))
    min_scale = float(np.clip(cfg.get("constraint_task_loss_min_scale", 0.0), 0.0, 1.0))

    if global_step <= warmup_steps:
        scale = min_scale
    elif anneal_steps <= 0:
        scale = 1.0
    else:
        progress = (global_step - warmup_steps) / max(anneal_steps, 1)
        scale = min_scale + (1.0 - min_scale) * float(np.clip(progress, 0.0, 1.0))
    return float(penalty * scale)


def compute_energy_margin_cost(
    info: dict | None,
    cfg: dict | None = None,
) -> float:
    """在电池警告边界以下运行的软CMDP成本。"""
    cfg = cfg or DRL_CONFIG
    info = info or {}
    if "soc" not in info and "energy_stage" not in info and "energy_crashed" not in info:
        return 0.0

    min_soc = float(ENERGY_CONFIG.get("battery_min_soc", 0.15))
    crash_soc = float(ENERGY_CONFIG.get("battery_crash_soc", 0.05))
    coeff = max(0.0, float(cfg.get("constraint_energy_margin_coeff", 0.25)))
    clip = max(0.0, float(cfg.get("constraint_energy_margin_clip", 1.0)))

    if "soc" in info:
        soc = float(info.get("soc", min_soc))
        norm = max(min_soc - crash_soc, 1e-6)
        pressure = max(0.0, min_soc - soc) / norm
    else:
        stage = str(info.get("energy_stage", "normal")).lower()
        pressure = 1.0 if stage in {"warning", "unsafe", "critical", "failure"} else 0.0
    if bool(info.get("energy_crashed", False)):
        pressure = max(pressure, 1.0)
    return float(np.clip(coeff * pressure, 0.0, clip))


def compute_orbit_margin_cost(
    info: dict | None,
    cfg: dict | None = None,
) -> float:
    """高度接近不安全VLEO边界的软CMDP成本。"""
    cfg = cfg or DRL_CONFIG
    info = info or {}
    if "altitude_km" not in info and "orbit_stage" not in info and "orbit_crashed" not in info:
        return 0.0

    warning_km = float(ORBITAL_CONFIG.get("altitude_warning_km", 180.0))
    min_km = float(ORBITAL_CONFIG.get("altitude_min_km", 150.0))
    crash_km = float(ORBITAL_CONFIG.get("altitude_crash_km", 122.0))
    coeff = max(0.0, float(cfg.get("constraint_orbit_margin_coeff", 0.25)))
    clip = max(0.0, float(cfg.get("constraint_orbit_margin_clip", 1.0)))

    if "altitude_km" in info:
        altitude_km = float(info.get("altitude_km", warning_km))
        warning_norm = max(warning_km - min_km, 1e-6)
        unsafe_norm = max(min_km - crash_km, 1e-6)
        warning_pressure = max(0.0, warning_km - altitude_km) / warning_norm
        unsafe_pressure = max(0.0, min_km - altitude_km) / unsafe_norm
        pressure = warning_pressure + unsafe_pressure
        # ── 主动轨道代价：在标称高度以下就开始发出信号，让 actor 在 250km 就感到
        # "高度偏低"，而不是等到 200km 才触发。彻底解决"标称高度=0代价→不推进"问题。
        proactive_scale = max(0.0, float(cfg.get("constraint_orbit_proactive_scale", 0.0)))
        if proactive_scale > 0.0:
            nominal_km = float(ORBITAL_CONFIG.get("altitude_nominal_km", 250.0))
            proactive_norm = max(nominal_km - warning_km, 1e-6)  # = 250-200=50km
            proactive_pressure = (
                max(0.0, nominal_km - altitude_km) / proactive_norm * proactive_scale
            )
            pressure += proactive_pressure
    else:
        stage = str(info.get("orbit_stage", "normal")).lower()
        pressure = 1.0 if stage in {"warning", "unsafe", "critical", "failure"} else 0.0
    if bool(info.get("orbit_crashed", False)):
        pressure = max(pressure, 1.0)
    return float(np.clip(coeff * pressure, 0.0, clip))


def compute_over_processing_details(
    info: dict | None,
    cfg: dict | None = None,
) -> dict[str, float]:
    """累积处理的容量感知准入成本详细信息。

    这是 proc/dl 的**唯一**主约束:惩罚处理超过近期可下传能力的数据。
    既包含 backlog (当前已处理但还没下传) 也包含 admission (整 episode 累计处理量
    相对于"已下传 + 未来可下传"的超额),所以历史的 backlog/window_waste/
    unproductive_cpu 三个旧 cost 在这里已经被等效覆盖。
    """
    cfg = cfg or DRL_CONFIG
    info = info or {}

    coeff = max(0.0, float(cfg.get("constraint_over_processing_coeff", 5.0)))
    clip = max(0.0, float(cfg.get("constraint_over_processing_clip", 4.0)))
    # 容量压力必须使用 RF product MB 口径：processed queue 与下传链路都承载压缩后的产品量。
    processed_queue_mb = max(0.0, float(
        info.get("processed_queue_product_mb", info.get("processed_queue_mb", 0.0))))
    processed_since_contact_mb = max(
        0.0,
        float(info.get(
            "processed_product_since_contact_mb",
            info.get("processed_since_contact_mb", 0.0),
        )),
    )
    delivered_since_contact_mb = max(
        0.0,
        float(info.get(
            "rf_downlinked_since_contact_mb",
            info.get("delivered_since_contact_mb", 0.0),
        )),
    )
    future_capacity_mb = max(
        0.0,
        float(info.get(
            "future_contact_capacity_product_mb",
            info.get("future_contact_capacity_mb", info.get("future_capacity_mb", 0.0)),
        )),
    )
    episode_processed_mb = max(
        0.0,
        float(info.get(
            "episode_processed_product_mb",
            info.get("episode_processed_mb", processed_since_contact_mb),
        )),
    )
    episode_delivered_mb = max(
        0.0,
        float(info.get(
            "episode_rf_downlinked_mb",
            info.get("episode_delivered_mb", delivered_since_contact_mb),
        )),
    )
    episode_raw_equiv_processed_mb = max(
        0.0,
        float(info.get("episode_raw_equivalent_processed_mb", 0.0)),
    )
    episode_raw_equiv_delivered_mb = max(
        0.0,
        float(info.get("episode_raw_equivalent_delivered_mb", 0.0)),
    )
    processed_voi_basis_value = max(
        0.0,
        float(info.get("episode_processed_voi_basis_value", info.get("processed_voi_basis_value", 0.0))),
    )
    delivered_value = max(
        0.0,
        float(info.get("episode_delivered_value", info.get("delivered_value", 0.0))),
    )
    norm_mb = max(
        1e-6,
        float(cfg.get(
            "constraint_capacity_norm_mb",
            cfg.get(
                "constraint_capacity_norm",
                info.get("constraint_capacity_norm_mb", info.get("constraint_capacity_norm", 500.0)),
            ),
        )),
    )
    margin = max(0.0, float(cfg.get("constraint_future_capacity_margin", 0.80)))
    clearable_capacity_mb = episode_delivered_mb + future_capacity_mb
    backlog_excess_mb = max(0.0, processed_queue_mb - margin * future_capacity_mb)
    admission_excess_mb = max(
        0.0,
        episode_processed_mb - margin * clearable_capacity_mb,
    )
    future_ratio_excess = max(
        0.0,
        processed_queue_mb / max(future_capacity_mb, 1e-6) - 1.0,
    )
    episode_clearable_ratio_excess = max(
        0.0,
        episode_processed_mb / max(clearable_capacity_mb, 1e-6) - 1.0,
    )
    backlog_ratio = (
        processed_queue_mb / max(margin * future_capacity_mb, 1e-6)
        if processed_queue_mb > 0.0
        else 0.0
    )
    admission_ratio = (
        episode_processed_mb / max(margin * clearable_capacity_mb, 1e-6)
        if episode_processed_mb > 0.0
        else 0.0
    )
    raw_equivalent_delivery_coverage = (
        episode_raw_equiv_delivered_mb / max(episode_raw_equiv_processed_mb, 1e-6)
        if episode_raw_equiv_processed_mb > 0.0
        else 0.0
    )
    raw_equivalent_proc_delivery_ratio = (
        episode_raw_equiv_processed_mb / max(episode_raw_equiv_delivered_mb, 1e-6)
        if episode_raw_equiv_processed_mb > 0.0
        else 0.0
    )
    value_realization_ratio = (
        delivered_value / max(processed_voi_basis_value, 1e-6)
        if processed_voi_basis_value > 0.0
        else 0.0
    )
    ratio_weight = max(0.0, float(
        cfg.get("constraint_over_processing_ratio_weight", 3.0)))
    excess_score = max(
        float(backlog_excess_mb / norm_mb),
        float(admission_excess_mb / norm_mb),
        ratio_weight * float(future_ratio_excess),
        ratio_weight * float(episode_clearable_ratio_excess),
        ratio_weight * float(max(0.0, backlog_ratio - 1.0)),
        ratio_weight * float(max(0.0, admission_ratio - 1.0)),
    )
    raw_cost = coeff * excess_score if coeff > 0.0 else 0.0
    training_cost = _soft_cap_cost(raw_cost, clip) if clip > 0.0 else raw_cost
    normalized_cost = raw_cost / max(clip, 1e-6) if clip > 0.0 else raw_cost
    clip_saturation = 1.0 if clip > 0.0 and raw_cost >= clip - 1e-9 else 0.0
    return {
        "cost": training_cost,
        "raw_cost": float(raw_cost),
        "normalized_cost": float(normalized_cost),
        "training_cost": training_cost,
        "clip_saturation": float(clip_saturation),
        "backlog_excess_mb": float(backlog_excess_mb),
        "admission_excess_mb": float(admission_excess_mb),
        "clearable_capacity_mb": float(clearable_capacity_mb),
        "over_processing_ratio": float(max(backlog_ratio, admission_ratio)),
        "rf_product_proc_downlink_ratio": float(max(backlog_ratio, admission_ratio)),
        "raw_equivalent_delivery_coverage": float(raw_equivalent_delivery_coverage),
        "raw_equivalent_proc_delivery_ratio": float(raw_equivalent_proc_delivery_ratio),
        "value_realization_ratio": float(value_realization_ratio),
    }


def compute_over_processing_cost(
    info: dict | None,
    cfg: dict | None = None,
) -> float:
    """Capacity-aware admission cost (唯一 proc/dl 主约束)."""
    return float(compute_over_processing_details(info, cfg)["cost"])


def compute_lyapunov_safety_cost(
    previous_queues: tuple[float, float, float, float],
    next_queues: tuple[float, float, float, float],
    queue_maxes: tuple[float, float, float, float],
    info: dict | None = None,
    cfg: dict | None = None,
) -> SafetyCostBreakdown:
    """计算约束评论家成本c_t。

    重构后只组合 4 类语义清晰的项,移除了历史 5 个互相打架的旧 cost。

    c_t = [positive_drift] + queue_soft + state_safety + thermal_excess
        + energy_margin + orbit_margin + over_processing + queue_hard + task_loss

    Queue order:
        (energy_queue, orbit_queue, raw_queue, processed_queue)
    """
    cfg = cfg or DRL_CONFIG
    qe, qh, qd, qc = [float(x) for x in previous_queues]
    qe2, qh2, qd2, qc2 = [float(x) for x in next_queues]
    qe_max, qh_max, qd_max, qc_max = [max(float(x), 1e-6) for x in queue_maxes]

    drift = 0.5 * ((qe2 / qe_max) ** 2 - (qe / qe_max) ** 2)
    drift += 0.5 * ((qh2 / qh_max) ** 2 - (qh / qh_max) ** 2)
    drift += 0.5 * ((qd2 / qd_max) ** 2 - (qd / qd_max) ** 2)
    drift += 0.5 * ((qc2 / qc_max) ** 2 - (qc / qc_max) ** 2)
    drift = float(drift)

    soft, hard = compute_queue_risk_penalties(
        qd2, qc2, qd_max, qc_max, info, cfg, include_processed_backlog=False)
    positive_drift = max(0.0, drift)
    state_penalty = compute_state_safety_penalty(info, cfg)
    info = info or {}
    thermal_excess_c = info.get("_thermal_excess_c", None)
    if thermal_excess_c is None:
        temp_c = float(info.get("thermal_temperature_c", info.get("temperature_c", 0.0)))
        warning_c = float(info.get(
            "thermal_warning_temp_c",
            THERMAL_CONFIG.get("warning_temp_c", 45.0),
        ))
        thermal_excess_c = max(0.0, temp_c - warning_c)
    thermal_cfg = dict(cfg.get("constraint_thermal_excess", {}) or {})
    thermal_norm = max(1e-6, float(thermal_cfg.get(
        "norm_c", cfg.get("constraint_thermal_excess_norm_c", 10.0))))
    thermal_coeff = max(0.0, float(thermal_cfg.get(
        "coeff", cfg.get("constraint_thermal_excess_coeff", 0.25))))
    thermal_excess_penalty = float(thermal_coeff * max(0.0, float(thermal_excess_c)) / thermal_norm)
    task_loss_penalty = compute_task_loss_penalty(info, cfg)
    energy_cost = compute_energy_margin_cost(info, cfg)
    orbit_cost = compute_orbit_margin_cost(info, cfg)
    over_processing_details = compute_over_processing_details(info, cfg)
    over_processing_cost = float(over_processing_details["training_cost"])

    queue_cost = float(soft + hard)
    thermal_cost = float(thermal_excess_penalty)
    task_loss_cost = float(task_loss_penalty)
    low_value_waste = compute_low_value_waste_cost(info, cfg)
    unproductive_cpu = compute_unproductive_cpu_cost(info, cfg)

    # 软约束组合：drift + 队列软压力 + 热超限 + 能量裕度 + 轨道裕度 + 容量超处理
    # + low_value_waste + unproductive_cpu（两项当前启用，系数由 config 控制）。
    soft_constraint_cost = float(
        positive_drift + soft + thermal_cost + energy_cost + orbit_cost
        + over_processing_cost + low_value_waste + unproductive_cpu
    )
    hard_violation_cost = float(hard + state_penalty + task_loss_cost)
    raw_cost = float(soft_constraint_cost + hard_violation_cost)

    # ── 顶刊 Issue#1: 分离 QoS 与物理安全/队列稳定 ─────────────────────
    # QoS（任务效用/效率塑形）：不应进入 safety critic。
    qos_cost = float(over_processing_cost + low_value_waste + unproductive_cpu + task_loss_cost)
    # clean 约束 = 物理安全(orbit/energy/thermal + 硬状态) + 队列稳定(soft+hard overflow + drift)。
    clean_constraint_cost = float(
        positive_drift + soft + hard + thermal_cost + energy_cost + orbit_cost + state_penalty
    )
    use_clean = bool(cfg.get("clean_constraint_cost_enabled", False))
    # critic / dual 的代价基准：flag 开 → clean；关 → 旧的混合 raw（默认，保持兼容）。
    basis_cost = clean_constraint_cost if use_clean else raw_cost

    training_clip = max(0.0, float(cfg.get(
        "constraint_training_cost_clip",
        cfg.get("lyapunov_drift_clip", 3.0),
    )))
    training_cost = _soft_cap_cost(basis_cost, training_clip) if training_clip > 0.0 else basis_cost
    normalized_cost = basis_cost / max(training_clip, 1e-6) if training_clip > 0.0 else basis_cost
    clip_saturation = 1.0 if training_clip > 0.0 and basis_cost >= training_clip - 1e-9 else 0.0
    clip = float(cfg.get("lyapunov_drift_clip", 3.0))
    clipped = float(np.clip(basis_cost, 0.0, clip))
    dual_norm = max(1e-6, float(cfg.get(
        "adaptive_lyapunov_constraint_norm", clip)))
    dual_signal_max = max(1.0, float(cfg.get(
        "adaptive_lyapunov_constraint_signal_max", 3.0)))
    dual_source = basis_cost if bool(cfg.get("adaptive_lyapunov_dual_uses_raw_cost", True)) else training_cost
    dual_signal = float(np.clip(dual_source / dual_norm, 0.0, dual_signal_max))

    return SafetyCostBreakdown(
        lyapunov_drift=drift,
        positive_lyapunov_drift=positive_drift,
        queue_soft_penalty=soft,
        queue_hard_penalty=hard,
        over_processing_cost=over_processing_cost,
        thermal_excess_penalty=thermal_excess_penalty,
        task_loss_penalty=task_loss_penalty,
        state_safety_penalty=state_penalty,
        queue_cost=queue_cost,
        energy_cost=energy_cost,
        orbit_cost=orbit_cost,
        thermal_cost=thermal_cost,
        task_loss_cost=task_loss_cost,
        total_cost=raw_cost,
        training_cost=training_cost,
        normalized_cost=normalized_cost,
        training_cost_clip=training_clip,
        training_cost_clip_saturation=clip_saturation,
        dual_cost=dual_source,
        dual_cost_norm=dual_norm,
        dual_violation_signal=dual_signal,
        soft_constraint_cost=soft_constraint_cost,
        hard_violation_cost=hard_violation_cost,
        raw_cost=raw_cost,
        clipped_cost=clipped,
        over_processing_raw_cost=float(over_processing_details["raw_cost"]),
        over_processing_normalized_cost=float(over_processing_details["normalized_cost"]),
        over_processing_training_cost=float(over_processing_details["training_cost"]),
        over_processing_clip_saturation=float(over_processing_details["clip_saturation"]),
        backlog_excess_mb=float(over_processing_details["backlog_excess_mb"]),
        admission_excess_mb=float(over_processing_details["admission_excess_mb"]),
        clearable_capacity_mb=float(over_processing_details["clearable_capacity_mb"]),
        over_processing_ratio=float(over_processing_details["over_processing_ratio"]),
        # 兼容旧字段
        processed_backlog_cost=0.0,
        window_waste_cost=0.0,
        low_value_waste_cost=float(low_value_waste),     # 已激活
        unproductive_cpu_cost=float(unproductive_cpu),   # 已激活：罚远窗口+高 queue 时 CPU 还在烧
        efficiency_cost=0.0,
        # ── 顶刊 Issue#1: clean 约束分解（始终计算）──
        constraint_total_clean=float(clean_constraint_cost),
        qos_total=float(qos_cost),
        constraint_state_safety_cost=float(state_penalty),
        constraint_drift_cost=float(positive_drift),
        clean_constraint_cost_used=1.0 if use_clean else 0.0,
    )


# ──────────────────────────────────────────────────────────────────────────
# 兼容性 stub: 历史 cost 函数被外部脚本/测试导入时返回 0.0,避免破坏 imports。
# 它们语义已被 over_processing_cost 覆盖,默认完全关闭。
# ──────────────────────────────────────────────────────────────────────────
def compute_efficiency_cost(info=None, cfg=None) -> float:
    """已废弃: 语义被 over_processing_cost 覆盖。默认返回 0.0。"""
    return 0.0


def compute_low_value_waste_cost(info=None, cfg=None) -> float:
    """惩罚处理低价值数据（鼓励用 drop_low_value 动作维度）。

    over_processing_cost 是 value-agnostic：总量超容才罚。但 agent 可能在总量
    合规的情况下，把 CPU 全花在处理低价值数据上 → 高 proc/dl 但都是垃圾下传。

    本 cost = coeff * (proc_low/norm) * low_share，仅在 raw_low>1 时激活
    （即 agent 有低价值原始数据可丢但没丢的情况下）。
    """
    from config import DRL_CONFIG
    cfg = cfg or DRL_CONFIG
    info = info or {}
    coeff = max(0.0, float(cfg.get("constraint_low_value_waste_coeff", 0.0)))
    if coeff <= 0.0:
        return 0.0
    clip = max(0.0, float(cfg.get("constraint_low_value_waste_clip", 0.0)))
    proc_low = max(0.0, float(info.get("processed_low_mb_step", 0.0)))
    proc_mid = max(0.0, float(info.get("processed_mid_mb_step", 0.0)))
    proc_high = max(0.0, float(info.get("processed_high_mb_step", 0.0)))
    proc_total = proc_low + proc_mid + proc_high
    if proc_total <= 1e-6:
        return 0.0
    raw_low = max(0.0, float(info.get("raw_low_mb", 0.0)))
    if raw_low <= 1.0:  # 没有低价值原始数据可丢，agent 无责任
        return 0.0
    low_share = proc_low / proc_total
    norm_mb = max(1e-6, float(cfg.get("constraint_low_value_waste_norm_mb", 5.0)))
    raw_cost = coeff * (proc_low / norm_mb) * low_share
    if clip > 0.0:
        raw_cost = min(raw_cost, clip)
    return float(raw_cost)


def compute_processed_backlog_cost(processed_queue_utilization=None, cfg=None) -> float:
    """已废弃: 语义被 over_processing_cost 覆盖。默认返回 0.0。"""
    return 0.0


def compute_unproductive_cpu_cost(info=None, cfg=None) -> float:
    """惩罚"窗口远但 CPU 还在拼命跑"的浪费功耗。

    关键观察：CPU 持续耗能（实测平均 8.5W），但只有最终下传的数据才有价值。
    当 1) 下个 TX 窗口很远 且 2) processed_queue 已经超过未来窗口能下传的量，
    继续处理只是在烧电池。

    cost = coeff * cpu_power_w/cpu_max_w * unproductive_indicator
    其中 unproductive_indicator = max(0, queue_future_contact_ratio - 1.0)
                              + max(0, time_to_window_far_penalty)
    """
    from config import DRL_CONFIG
    cfg = cfg or DRL_CONFIG
    info = info or {}
    coeff = max(0.0, float(cfg.get("constraint_unproductive_cpu_coeff", 0.0)))
    if coeff <= 0.0:
        return 0.0
    clip = max(0.0, float(cfg.get("constraint_unproductive_cpu_clip", 2.0)))

    # CPU 活跃强度（0~1）
    cpu_power_w = max(0.0, float(info.get("cpu_power_w", info.get("P_cpu_w", 0.0))))
    cpu_max_w = max(1e-6, float(info.get("cpu_max_w", 25.0)))
    cpu_activity = min(1.0, cpu_power_w / cpu_max_w)
    if cpu_activity <= 0.05:
        return 0.0  # CPU 已经几乎闲着，没有浪费

    # 队列对未来窗口的饱和度（>1.0 表示已经超过下一窗口能下的量）
    future_contact_ratio = float(info.get("processed_queue_future_contact_ratio", 0.0))
    queue_overflow_penalty = max(0.0, future_contact_ratio - 1.0)

    # 距离下个窗口的远度（远窗口 + CPU 高活跃 = 浪费）
    time_to_window_s = float(info.get("time_to_next_window_s", 0.0))
    far_threshold_s = float(cfg.get("constraint_unproductive_cpu_far_window_s", 300.0))
    far_penalty = max(0.0, min(1.0, (time_to_window_s - far_threshold_s) / max(far_threshold_s, 1.0)))

    # 队列已经爆 → 强浪费；窗口远 + queue 不空 → 弱浪费
    proc_queue_util = float(info.get("processed_queue_final_utilization",
                                     info.get("processed_queue_utilization", 0.0)))
    unproductive_indicator = queue_overflow_penalty + 0.5 * far_penalty * proc_queue_util
    if unproductive_indicator <= 1e-6:
        return 0.0
    raw_cost = coeff * cpu_activity * unproductive_indicator
    if clip > 0.0:
        raw_cost = min(raw_cost, clip)
    return float(raw_cost)


def compute_window_waste_cost(info=None, cfg=None) -> float:
    """已废弃: 学习窗口利用率由 reward (delivered_value) 直接驱动。默认返回 0.0。"""
    return 0.0


# ──────────────────────────────────────────────────────────────────────────
# 顶刊 Issue#4: 干净的 CMDP 约束语义。
#
# 审稿意见：单一 "safety cost" 把物理安全 + 队列稳定 + 任务 QoS 混在一起，
# CMDP 约束语义模糊。这里**不改动** compute_lyapunov_safety_cost（训练用的
# 标量信号保持兼容），而是新增一套加性 API，把代价显式拆成三类：
#
#   1. Physical safety   c_t = [c_orbit, c_energy, c_thermal]  → 进入 CMDP 约束
#   2. Queue stability   c_queue (soft+hard overflow)          → 可作单独约束
#   3. Mission QoS loss  (expired/dropped/low-value/over-proc) → reward/次级指标
#
# 论文中应写成向量约束 c_t = [c_orbit, c_energy, c_thermal, c_queue]，
# 并分别报告 violation rate；QoS 损失不再混入 safety。
# ──────────────────────────────────────────────────────────────────────────

# 约束类别 → 物理含义。CMDP 主约束只含 physical safety + queue stability。
CONSTRAINT_CATEGORIES = ("orbit", "energy", "thermal", "queue")
QOS_CATEGORIES = ("task_loss", "over_processing", "low_value_waste", "unproductive_cpu")


def compute_categorized_safety_cost(
    previous_queues: tuple[float, float, float, float],
    next_queues: tuple[float, float, float, float],
    queue_maxes: tuple[float, float, float, float],
    info: dict | None = None,
    cfg: dict | None = None,
) -> dict:
    """返回干净分类后的 CMDP 代价（顶刊 Issue#4）。

    输出:
        {
          "constraint_vector": {"orbit","energy","thermal","queue"},  # CMDP 约束分量
          "constraint_total":  float,                                  # = sum(constraint_vector)
          "qos_loss": {"task_loss","over_processing","low_value_waste","unproductive_cpu"},
          "qos_total": float,                                          # 不进入 CMDP 约束
          "violations": {category: bool},                              # 各分量是否越界（>tol）
        }

    设计原则：每个 constraint 分量都只反映**物理/稳定性安全**，与既有
    compute_*_margin_cost / queue penalty 复用同一函数，保证与训练信号一致，
    只是按语义重新归类、不再求和成一个笼统 scalar。
    """
    cfg = cfg or DRL_CONFIG
    info = info or {}
    _, _, qd, qc = [float(x) for x in next_queues]
    _, _, qd_max, qc_max = [max(float(x), 1e-6) for x in queue_maxes]

    # ── 1. Physical safety 分量 ──────────────────────────────
    c_orbit = float(compute_orbit_margin_cost(info, cfg))
    c_energy = float(compute_energy_margin_cost(info, cfg))
    # 热：state_safety 里的热阶段 + 连续热超限项，两者取和作为热约束分量。
    thermal_excess_c = info.get("_thermal_excess_c", None)
    if thermal_excess_c is None:
        temp_c = float(info.get("thermal_temperature_c", info.get("temperature_c", 0.0)))
        warning_c = float(info.get(
            "thermal_warning_temp_c", THERMAL_CONFIG.get("warning_temp_c", 45.0)))
        thermal_excess_c = max(0.0, temp_c - warning_c)
    thermal_cfg = dict(cfg.get("constraint_thermal_excess", {}) or {})
    thermal_norm = max(1e-6, float(thermal_cfg.get(
        "norm_c", cfg.get("constraint_thermal_excess_norm_c", 10.0))))
    thermal_coeff = max(0.0, float(thermal_cfg.get(
        "coeff", cfg.get("constraint_thermal_excess_coeff", 0.25))))
    c_thermal = float(thermal_coeff * max(0.0, float(thermal_excess_c)) / thermal_norm)

    # ── 2. Queue stability 分量（soft + hard overflow）────────
    soft, hard = compute_queue_risk_penalties(
        qd, qc, qd_max, qc_max, info, cfg, include_processed_backlog=False)
    c_queue = float(soft + hard)

    constraint_vector = {
        "orbit": c_orbit,
        "energy": c_energy,
        "thermal": c_thermal,
        "queue": c_queue,
    }
    constraint_total = float(sum(constraint_vector.values()))

    # ── 3. Mission QoS loss（不进入 CMDP 约束）────────────────
    qos_loss = {
        "task_loss": float(compute_task_loss_penalty(info, cfg)),
        "over_processing": float(compute_over_processing_cost(info, cfg)),
        "low_value_waste": float(compute_low_value_waste_cost(info, cfg)),
        "unproductive_cpu": float(compute_unproductive_cpu_cost(info, cfg)),
    }
    qos_total = float(sum(qos_loss.values()))

    # ── violation 判定：分量超过各自容差视为越界，用于报告 per-category rate ──
    tol = max(0.0, float(cfg.get("constraint_violation_tolerance", 1e-6)))
    violations = {cat: bool(val > tol) for cat, val in constraint_vector.items()}

    return {
        "constraint_vector": constraint_vector,
        "constraint_total": constraint_total,
        "qos_loss": qos_loss,
        "qos_total": qos_total,
        "violations": violations,
    }


def classify_violation_flags(info: dict | None) -> dict:
    """从 env info 直接读出 per-category 物理违规布尔（用于评估期 violation rate）。

    与 compute_categorized_safety_cost 互补：后者基于连续代价>容差，本函数基于
    环境已判定的安全布尔（*_safe / *_crashed / stage），更贴近"硬违规"统计口径。
    """
    info = info or {}
    stage = str(info.get("risk_stage", "normal")).lower()
    thermal_stage = str(info.get("thermal_stage", "normal")).lower()
    return {
        "orbit": bool(
            not info.get("orbit_safe", True)
            or info.get("orbit_crashed", False)
            or stage in {"unsafe", "failure"} and info.get("orbit_stage", "") in {"unsafe", "failure"}
        ),
        "energy": bool(
            not info.get("energy_safe", True)
            or info.get("energy_crashed", False)
        ),
        "thermal": bool(
            not info.get("thermal_safe", True)
            or info.get("thermal_crashed", False)
            or thermal_stage in {"critical", "failure"}
        ),
        "queue": bool(
            not info.get("raw_queue_safe", True)
            or not info.get("processed_queue_safe", True)
        ),
    }

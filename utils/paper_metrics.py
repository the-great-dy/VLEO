"""LS-PSF CMDP 实验的论文面向指标名称。"""

from __future__ import annotations

import math


PAPER_METRIC_ALIASES = {
    "Constraint Satisfaction Rate": (
        "overall_safe_rate",
        "step_safety_rate",
        "safety_rate",
    ),
    "Episode Safety Rate": ("episode_safety_rate", "safety_rate"),
    "Survival Rate": ("survival_rate",),
    "Delivered VoI": ("delivered_value_mean", "delivered_value_total"),
    "Safety-adjusted Delivered VoI": (
        "safety_adjusted_delivered_value",
        "checkpoint_value_score",
    ),
    "Clean Constraint Cost": (
        "constraint_total_clean_mean",
        "constraint_total_clean",
        "clean_constraint_cost_mean",
    ),
    "Mission QoS Cost": (
        "qos_total_mean",
        "qos_total",
        "mission_qos_cost_mean",
    ),
    "Deadline Success Rate": ("deadline_success_rate",),
    "Value-weighted Deadline Success": (
        "value_weighted_deadline_success_rate",
        "deadline_success_rate",
    ),
    "RF Downlinked MB": (
        "rf_downlinked_mb_mean",
        "rf_downlinked_mean",
        "rf_downlinked_mb",
        "downlink_mean",
        "downlink_mean_mb",
        "tx_mb_mean",
        "tx_mb_total",
    ),
    "Downlink MB": (
        "rf_downlinked_mb_mean",
        "rf_downlinked_mean",
        "rf_downlinked_mb",
        "downlink_mean",
        "downlink_mean_mb",
        "tx_mb_mean",
        "tx_mb_total",
    ),
    "Processed MB": (
        "processed_product_mb_mean",
        "processed_mean",
        "processed_mean_mb",
    ),
    "Raw-equivalent Processed MB": (
        "raw_equivalent_processed_mb_mean",
        "raw_equivalent_processed_mean",
        "raw_equivalent_processed_mb",
    ),
    "Raw-equivalent Delivered MB": (
        "raw_equivalent_delivered_mb_mean",
        "raw_equivalent_delivered_mean",
        "raw_equivalent_delivered_mb",
    ),
    "Raw-equivalent Delivery Coverage": (
        "raw_equivalent_delivery_coverage_mean",
        "episode_raw_equivalent_delivery_coverage",
        "raw_equivalent_delivery_coverage",
    ),
    "RF Product Proc/DL Ratio": (
        "rf_product_proc_downlink_ratio_mean",
        "episode_rf_product_proc_downlink_ratio",
        "rf_product_proc_downlink_ratio",
    ),
    "Value Realization Ratio": (
        "value_realization_ratio_mean",
        "episode_value_realization_ratio",
        "value_realization_ratio",
        "episode_useful_processing_ratio",
        "useful_processing_ratio",
    ),
    "Global Proc/DL Ratio": (
        "global_proc_downlink_ratio",
        "global_proc_dl_ratio",
    ),
    "Mean Episode Proc/DL Ratio": (
        "mean_episode_proc_downlink_ratio",
        "episode_proc_dl_ratio",
        "episode_proc_downlink_ratio",
    ),
    "Proc/DL Ratio": (
        "global_proc_downlink_ratio",
        "global_proc_dl_ratio",
        "proc_downlink_ratio",
        "proc_dl_ratio",
    ),
    "Useful Processing Ratio": (
        "episode_useful_processing_ratio",
        "useful_processing_ratio",
    ),
    "Window Utilization": ("comm_window_utilization", "window_utilization"),
    "Processed Queue Final Utilization": (
        "processed_queue_final_utilization",
        "processed_queue_final_util",
        "final_processed_queue_utilization",
    ),
    "TX Active in Contact Ratio": ("tx_active_in_contact_ratio",),
    "High-value Delivery Rate": ("high_value_delivery_rate", "high_value_downlink_rate"),
    "High-value Delivery Ratio": (
        "high_value_delivery_ratio",
        "high_value_delivery_rate",
        "high_value_downlink_rate",
    ),
    "High-value Process Rate (count)": ("high_value_process_rate_count",),
    "High-value Process Rate (value-weighted)": ("high_value_process_rate_value_weighted",),
    "High-value Delivery Rate (count)": ("high_value_delivery_rate_count",),
    "High-value Delivery Rate (value-weighted)": ("high_value_delivery_rate_value_weighted",),
    "High-value Expired Rate (count)": ("high_value_expired_rate_count",),
    "High-value Expired Rate (value-weighted)": ("high_value_expired_rate_value_weighted",),
    "Energy Violation Rate": (
        "energy_violation_rate",
        "energy_unsafe_rate",
        "constraint_energy_violation_rate",
    ),
    "Energy Efficiency": ("energy_efficiency",),
    "Energy per VoI": (
        "energy_per_value",
        "energy_per_delivered_value_episode",
    ),
    "Primary Goal Feasible": ("primary_goal_feasible",),
    "Primary Goal Violation": ("primary_goal_violation",),
    "Intervention Rate": (
        "safety_intervention_rate",
        "intervention_rate",
    ),
    "Total Action Modification Rate": (
        "chain_total_rate",
        "was_projected_rate",
    ),
    "Physical Projection Rate": (
        "boundary_clip_rate_eval",
        "power_clip_rate_eval",
        "physical_projection_rate",
    ),
    "Lyapunov Projection Rate": ("lyapunov_projected_rate_eval", "lyapunov_proj_rate", "lyapunov_projection_rate"),
    "PSF Intervention Rate": ("psf_modified_rate", "psf_filter_rate"),
    "VoI Degradation Rate": ("voi_degradation_rate", "expired_value_rate"),
    "VoI Loss Rate": ("voi_loss_rate",),
    "Average AoI": ("average_aoi_steps", "avg_delivery_delay_steps"),
    "Value-weighted AoI": ("value_weighted_aoi_steps", "average_aoi_steps"),
    "Mean Action Modification": (
        "mean_action_modification",
        "action_mod_l2_mean",
        "total_action_mod_l2_mean",
    ),
    "Raw/Executed Action L2": (
        "raw_executed_action_l2_mean",
        "action_mod_l2_mean",
        "mean_action_modification",
    ),
    "Shield Dependence Score": (
        "shield_dependence_score",
        "safety_layer_dependence_score",
    ),
    "Fuel Consumed (g)": ("fuel_consumed_g_mean", "fuel_consumed_g"),
    "Propellant Remaining Fraction": (
        "propellant_remaining_fraction_mean",
        "propellant_fraction_mean",
    ),
    "Delivered High VoI": ("delivered_high_value_mean", "high_value_downlink_value_mean"),
    "Delivered Mid VoI": ("delivered_mid_value_mean",),
    "Delivered Low VoI": ("delivered_low_value_mean",),
}


def add_paper_metrics(stats: dict) -> dict:
    """返回一个副本，添加了稳定的论文面向指标名称。"""
    out = dict(stats or {})
    for paper_name, keys in PAPER_METRIC_ALIASES.items():
        if paper_name in out:
            continue
        for key in keys:
            if key in out:
                out[paper_name] = out[key]
                break
    eps = 1e-9
    try:
        downlink = float(out.get("RF Downlinked MB", out.get("Downlink MB", 0.0)) or 0.0)
    except (TypeError, ValueError):
        downlink = 0.0
    if not math.isfinite(downlink):
        downlink = 0.0
    try:
        delivered = float(out.get("Delivered VoI", 0.0) or 0.0)
    except (TypeError, ValueError):
        delivered = 0.0
    if not math.isfinite(delivered):
        delivered = 0.0
    if downlink <= eps:
        for key in ("RF Product Proc/DL Ratio", "Global Proc/DL Ratio",
                    "Mean Episode Proc/DL Ratio", "Proc/DL Ratio"):
            out[key] = float("nan")
    if delivered <= eps:
        out["Energy per VoI"] = float("nan")
    return out


def compact_paper_table_row(stats: dict) -> dict:
    """适合 JSON/CSV/LaTeX 表格的小型有序行。"""
    enriched = add_paper_metrics(stats)
    return {
        "Constraint Satisfaction Rate": float(enriched.get("Constraint Satisfaction Rate", 0.0)),
        "Episode Safety Rate": float(enriched.get("Episode Safety Rate", enriched.get("safety_rate", 0.0))),
        "Survival Rate": float(enriched.get("Survival Rate", 0.0)),
        "Delivered VoI": float(enriched.get("Delivered VoI", 0.0)),
        "Safety-adjusted Delivered VoI": float(enriched.get("Safety-adjusted Delivered VoI", 0.0)),
        "Clean Constraint Cost": float(enriched.get("Clean Constraint Cost", 0.0)),
        "Mission QoS Cost": float(enriched.get("Mission QoS Cost", 0.0)),
        "Deadline Success Rate": float(enriched.get("Deadline Success Rate", 0.0)),
        "Value-weighted Deadline Success": float(enriched.get("Value-weighted Deadline Success", 0.0)),
        "RF Downlinked MB": float(enriched.get("RF Downlinked MB", 0.0)),
        "Downlink MB": float(enriched.get("Downlink MB", 0.0)),
        "Processed MB": float(enriched.get("Processed MB", 0.0)),
        "Raw-equivalent Processed MB": float(enriched.get("Raw-equivalent Processed MB", 0.0)),
        "Raw-equivalent Delivered MB": float(enriched.get("Raw-equivalent Delivered MB", 0.0)),
        "Raw-equivalent Delivery Coverage": float(enriched.get("Raw-equivalent Delivery Coverage", 0.0)),
        "RF Product Proc/DL Ratio": float(enriched.get("RF Product Proc/DL Ratio", 0.0)),
        "Value Realization Ratio": float(enriched.get("Value Realization Ratio", 0.0)),
        "Window Utilization": float(enriched.get("Window Utilization", 0.0)),
        "Processed Queue Final Utilization": float(enriched.get("Processed Queue Final Utilization", 0.0)),
        "TX Active in Contact Ratio": float(enriched.get("TX Active in Contact Ratio", 0.0)),
        "High-value Delivery Rate": float(enriched.get("High-value Delivery Rate", 0.0)),
        "High-value Delivery Ratio": float(enriched.get("High-value Delivery Ratio", 0.0)),
        "Energy Violation Rate": float(enriched.get("Energy Violation Rate", 0.0)),
        "Energy Efficiency": float(enriched.get("Energy Efficiency", 0.0)),
        "Energy per VoI": float(enriched.get("Energy per VoI", 0.0)),
        "Primary Goal Feasible": float(enriched.get("Primary Goal Feasible", 0.0)),
        "Primary Goal Violation": float(enriched.get("Primary Goal Violation", 0.0)),
        "Useful Processing Ratio": float(enriched.get("Useful Processing Ratio", 0.0)),
        "Value-weighted AoI": float(enriched.get("Value-weighted AoI", 0.0)),
        "VoI Loss Rate": float(enriched.get("VoI Loss Rate", 0.0)),
        "Intervention Rate": float(enriched.get("Intervention Rate", 0.0)),
        "Total Action Modification Rate": float(enriched.get("Total Action Modification Rate", 0.0)),
        "Physical Projection Rate": float(enriched.get("Physical Projection Rate", 0.0)),
        "Lyapunov Projection Rate": float(enriched.get("Lyapunov Projection Rate", 0.0)),
        "PSF Intervention Rate": float(enriched.get("PSF Intervention Rate", 0.0)),
        "Mean Action Modification": float(enriched.get("Mean Action Modification", 0.0)),
        "Raw/Executed Action L2": float(enriched.get("Raw/Executed Action L2", 0.0)),
        "Shield Dependence Score": float(enriched.get("Shield Dependence Score", 0.0)),
        "Fuel Consumed (g)": float(enriched.get("Fuel Consumed (g)", 0.0)),
        "Propellant Remaining Fraction": float(enriched.get("Propellant Remaining Fraction", 0.0)),
        "Delivered High VoI": float(enriched.get("Delivered High VoI", 0.0)),
        "Delivered Mid VoI": float(enriched.get("Delivered Mid VoI", 0.0)),
        "Delivered Low VoI": float(enriched.get("Delivered Low VoI", 0.0)),
    }

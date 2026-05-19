"""Paper-facing metric names for LS-PSF CMDP experiments."""

from __future__ import annotations


PAPER_METRIC_ALIASES = {
    "Constraint Satisfaction Rate": (
        "overall_safe_rate",
        "step_safety_rate",
        "safety_rate",
    ),
    "Episode Safety Rate": ("episode_safety_rate", "safety_rate"),
    "Survival Rate": ("survival_rate",),
    "Delivered VoI": ("delivered_value_mean", "delivered_value_total"),
    "Deadline Success Rate": ("deadline_success_rate",),
    "Value-weighted Deadline Success": (
        "value_weighted_deadline_success_rate",
        "deadline_success_rate",
    ),
    "Downlink MB": ("downlink_mean", "downlink_mean_mb", "tx_mb_mean", "tx_mb_total"),
    "Processed MB": (
        "processed_mean",
        "processed_mean_mb",
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
}


def add_paper_metrics(stats: dict) -> dict:
    """Return a copy with stable paper-facing metric names added."""
    out = dict(stats or {})
    for paper_name, keys in PAPER_METRIC_ALIASES.items():
        if paper_name in out:
            continue
        for key in keys:
            if key in out:
                out[paper_name] = out[key]
                break
    return out


def compact_paper_table_row(stats: dict) -> dict:
    """Small ordered row suitable for JSON/CSV/LaTeX tables."""
    enriched = add_paper_metrics(stats)
    return {
        "Constraint Satisfaction Rate": float(enriched.get("Constraint Satisfaction Rate", 0.0)),
        "Episode Safety Rate": float(enriched.get("Episode Safety Rate", enriched.get("safety_rate", 0.0))),
        "Survival Rate": float(enriched.get("Survival Rate", 0.0)),
        "Delivered VoI": float(enriched.get("Delivered VoI", 0.0)),
        "Deadline Success Rate": float(enriched.get("Deadline Success Rate", 0.0)),
        "Value-weighted Deadline Success": float(enriched.get("Value-weighted Deadline Success", 0.0)),
        "Downlink MB": float(enriched.get("Downlink MB", 0.0)),
        "Processed MB": float(enriched.get("Processed MB", 0.0)),
        "Global Proc/DL Ratio": float(enriched.get("Global Proc/DL Ratio", 0.0)),
        "Mean Episode Proc/DL Ratio": float(enriched.get("Mean Episode Proc/DL Ratio", 0.0)),
        "Proc/DL Ratio": float(enriched.get("Proc/DL Ratio", 0.0)),
        "Window Utilization": float(enriched.get("Window Utilization", 0.0)),
        "Processed Queue Final Utilization": float(enriched.get("Processed Queue Final Utilization", 0.0)),
        "TX Active in Contact Ratio": float(enriched.get("TX Active in Contact Ratio", 0.0)),
        "High-value Delivery Rate": float(enriched.get("High-value Delivery Rate", 0.0)),
        "High-value Delivery Ratio": float(enriched.get("High-value Delivery Ratio", 0.0)),
        "Useful Processing Ratio": float(enriched.get("Useful Processing Ratio", 0.0)),
        "Value-weighted AoI": float(enriched.get("Value-weighted AoI", 0.0)),
        "VoI Loss Rate": float(enriched.get("VoI Loss Rate", 0.0)),
        "Intervention Rate": float(enriched.get("Intervention Rate", 0.0)),
        "Total Action Modification Rate": float(enriched.get("Total Action Modification Rate", 0.0)),
        "Physical Projection Rate": float(enriched.get("Physical Projection Rate", 0.0)),
        "Lyapunov Projection Rate": float(enriched.get("Lyapunov Projection Rate", 0.0)),
        "PSF Intervention Rate": float(enriched.get("PSF Intervention Rate", 0.0)),
    }

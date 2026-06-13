"""吞吐瓶颈审计 —— 解释为什么 delivered value 仍低于某些基线。

本脚本固定策略，逐步收集通信窗口级别的数据流统计，输出每个通信窗口的：
  - 窗口开始时 processed queue 中可下传产品量
  - 窗口内 tx_alpha / tx_power
  - 窗口内 CPU 使用
  - 窗口内实际 downlinked MB
  - 窗口内 pointing fallback 次数
  - processed_but_not_downlinked（窗口结束后残留 processed 量）
  - 高价值任务全生命周期（generated → processed → downlinked → delivered / expired）

最终输出：
  1. 通信窗口利用率分解（窗口前有无 processed、窗口内 TX 利用率、pointing 干扰）
  2. 高价值任务全生命周期（高价值生成 → 处理 → 下传 → 交付 / 过期）
  3. useful_processing_ratio 分解（处理了多少，其中有多少下传，有多少 deadline 内交付）
  4. 瓶颈结论：当前瓶颈是哪个阶段（预处理不足 / 窗口内 TX 不足 / pointing 压制 /
     处理了低价值任务 / 处理后未能 deadline 前下传）

用法：
    python experiments/throughput_bottleneck_audit.py \\
        --model checkpoints_optimized/best_optimized.pt --device cuda
    python experiments/throughput_bottleneck_audit.py \\
        --model checkpoints_optimized/best_optimized.pt --device cuda \\
        --episodes 5 --seed 42
    # 对比多方法（需先运行 paper_compare.py 或手动多次调用）
    python experiments/throughput_bottleneck_audit.py --random  # 无训练策略
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from config import DRL_CONFIG, TRAIN_CONFIG
from environment.satellite_env import VLEOSatelliteEnv
from environment.wrappers import DilatedFrameStackWrapper
from scheduler.integrated_scheduler import IntegratedScheduler
from utils.action_space import IDX_POINTING, GROUPED_ACTION_DIM


def _finite(v, default=0.0) -> float:
    try:
        r = float(v)
        return r if math.isfinite(r) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _safe_div(num: float, den: float, eps: float = 1e-9) -> float:
    if abs(den) < eps:
        return float("nan")
    return float(num) / float(den)


# ── 窗口级记录结构 ─────────────────────────────────────────────────────────
WINDOW_RECORD_FIELDS = [
    "method", "seed", "episode", "window_idx",
    "window_start_step", "window_end_step", "window_duration_steps",
    "proc_queue_at_window_start_mb",  # 窗口开始时 processed queue（可下传数据量）
    "proc_queue_at_window_end_mb",    # 窗口结束后残留（processed_but_not_downlinked）
    "raw_queue_at_window_start_mb",   # 窗口开始时 raw queue（等待处理数据量）
    "tx_alpha_mean",                  # 窗口内平均 TX 功率比
    "cpu_alpha_mean",                 # 窗口内平均 CPU 功率比
    "prop_alpha_mean",                # 窗口内平均 prop 功率比
    "downlinked_mb",                  # 窗口内实际下传 MB
    "processed_mb",                   # 窗口内实际处理 MB
    "pointing_fallback_count",        # 窗口内 pointing fallback 次数
    "psf_intervene_count",            # 窗口内 PSF 介入次数
    "safe_budget_count",              # 窗口内 SAFE_BUDGET 触发次数
    "window_util",                    # 窗口内实际 TX 利用率（>0 步比例）
    "max_tx_capacity_mbps",           # 窗口内最大链路容量
    "soc_at_window_start",            # 窗口开始时电量
    "altitude_km_at_window_start",    # 窗口开始时高度
    "high_value_delivered_in_window", # 窗口内高价值任务交付
    "high_value_expired_in_window",   # 窗口内高价值任务过期
    "bottleneck_tag",                 # 本窗口瓶颈标签
]

# ── 高价值生命周期记录 ─────────────────────────────────────────────────────
HV_LIFECYCLE_FIELDS = [
    "method", "seed", "episode",
    "high_value_generated",           # 高价值任务总生成量（raw MB 等效）
    "high_value_processed",           # 高价值任务已处理量
    "high_value_downlinked",          # 高价值任务已下传量
    "high_value_delivered",           # 高价值任务已交付（deadline内）
    "high_value_expired_raw",         # 高价值任务在 raw queue 过期
    "high_value_expired_proc",        # 高价值任务处理后在 proc queue 过期（未及时下传）
    "high_value_dropped",             # 高价值任务主动丢弃
    "high_value_process_rate",        # processed/generated
    "high_value_downlink_rate",       # downlinked/processed
    "high_value_delivery_rate",       # delivered/generated
    "high_value_expired_rate",        # expired_raw/(generated+1e-9)
    "processed_but_not_downlinked_rate",  # (processed-downlinked)/processed
    "deadline_miss_after_processed",  # 处理后未能 deadline 内下传
    "useful_processing_ratio",        # delivered/(processed+1e-9)
]


def _tag_window_bottleneck(rec: dict) -> str:
    """根据窗口记录打一个最突出的瓶颈标签。"""
    proc_start = rec["proc_queue_at_window_start_mb"]
    dl         = rec["downlinked_mb"]
    tx_alpha   = rec["tx_alpha_mean"]
    fallback   = rec["pointing_fallback_count"]
    dur        = max(rec["window_duration_steps"], 1)
    util       = rec["window_util"]

    if proc_start < 0.1:
        return "NO_PROC_BEFORE_WINDOW"
    if fallback / dur > 0.30:
        return "POINTING_FALLBACK_DOMINANT"
    if tx_alpha < 0.10:
        return "LOW_TX_ALPHA"
    if dl < 0.1:
        return "ZERO_DOWNLINK_DESPITE_PROC"
    if util < 0.20:
        return "LOW_WINDOW_UTILIZATION"
    if rec["proc_queue_at_window_end_mb"] > proc_start * 0.5:
        return "PROCESSED_NOT_DOWNLINKED"
    return "OK"


def run(args) -> dict:
    from evaluate_optimized import _resolve_device
    device = _resolve_device(args.device)
    k = int(DRL_CONFIG.get("frame_stack", 8))

    scheduler = IntegratedScheduler(device=device, enable_lyapunov=True, use_psf=True)
    loaded = False
    if args.random:
        loaded = False
    elif args.model and os.path.exists(args.model):
        scheduler.load(args.model)
        loaded = True
    else:
        raise FileNotFoundError(
            f"未找到 checkpoint: {args.model}。无 checkpoint 想看覆盖结构请加 --random。")

    method_name = args.method_name or ("Ours" if loaded else "Random/untrained")

    # ── 全局聚合器 ─────────────────────────────────────────────────────────
    window_records = []
    hv_lifecycle_records = []

    # 全局计数
    total_steps = 0
    total_episodes = 0
    eps_high_delivered, eps_high_processed, eps_high_generated = [], [], []
    eps_high_expired_raw, eps_high_expired_proc, eps_high_dropped = [], [], []
    eps_high_downlinked = []
    eps_processed, eps_downlinked, eps_delivered_value = [], [], []
    eps_useful_proc_ratio = []
    eps_deadline_miss = []
    eps_processed_not_dl = []

    for ep in range(args.episodes):
        episode_seed = int(args.seed + ep)
        base_env = VLEOSatelliteEnv(seed=episode_seed)
        env = DilatedFrameStackWrapper(base_env, k=k)
        scheduler.reset_all_safety_stats()
        state = env.reset()
        done = False

        # 窗口级追踪
        in_window_prev   = False
        window_idx       = -1
        window_start_step = 0
        window_proc_start = 0.0
        window_proc_end   = 0.0
        window_raw_start  = 0.0
        window_soc_start  = 0.0
        window_alt_start  = 0.0
        window_tx_alphas  = []
        window_cpu_alphas = []
        window_prop_alphas = []
        window_dl_mb      = 0.0
        window_proc_mb    = 0.0
        window_fallback   = 0
        window_psf        = 0
        window_sb         = 0
        window_max_cap    = 0.0
        window_tx_active_steps = 0
        window_hv_delivered = 0.0
        window_hv_expired   = 0.0

        # episode 级高价值生命周期
        ep_hv_generated  = 0.0
        ep_hv_processed  = 0.0
        ep_hv_downlinked = 0.0
        ep_hv_delivered  = 0.0
        ep_hv_exp_raw    = 0.0
        ep_hv_exp_proc   = 0.0
        ep_hv_dropped    = 0.0
        ep_total_proc    = 0.0
        ep_total_dl      = 0.0
        ep_total_deliv   = 0.0
        ep_deadline_miss = 0.0

        step = 0
        while not done:
            in_window = (env._contact.get("in_window", False)
                         if env._contact is not None else False)
            prop_can_update = True
            if hasattr(env, "step_count") and hasattr(env, "N_PROP_SMOOTH"):
                prop_can_update = (env.step_count % env.N_PROP_SMOOTH == 0)

            action, was_projected, raw_action, psf_meta = scheduler.schedule(
                state, env.energy_queue.value, env.orbit_queue.value,
                env.data_queue.length, env.comm_queue.value,
                in_window=in_window, evaluate=True,
                h=env.altitude_m, soc=env.battery.soc, time_s=env.time_s,
                prop_can_update=prop_can_update,
                orbital_phase=env.orbit_sim.phase,
                tx_capacity_mbps=float((env._contact or {}).get("max_capacity_mbps", 0.0)),
                available_power_w=getattr(env, "available_power_w", None),
                env=env)

            state, reward, done, info = env.step(action, enforce_prop_smoothing=False)

            action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
            alpha_tx   = float(action_arr[2]) if action_arr.size > 2 else 0.0
            alpha_cpu  = float(action_arr[1]) if action_arr.size > 1 else 0.0
            alpha_prop = float(action_arr[0]) if action_arr.size > 0 else 0.0

            dl_step    = _finite(info.get("delivered_mb", info.get("actual_tx_mb", 0.0)))
            proc_step  = _finite(info.get("processed_mb", 0.0))
            deliv_step = _finite(info.get("delivered_value", 0.0))

            # 高价值生命周期统计
            ep_hv_generated  += _finite(info.get("high_value_generated_mb", 0.0))
            ep_hv_processed  += _finite(info.get("high_value_processed_mb",
                                                   info.get("cpu_requested_high", 0.0)))
            ep_hv_downlinked += _finite(info.get("high_value_downlinked_mb",
                                                   info.get("tx_requested_high", 0.0)))
            ep_hv_delivered  += _finite(info.get("delivered_high_value", 0.0))
            ep_hv_exp_raw    += _finite(info.get("high_value_expired_raw_mb",
                                                   info.get("expired_raw_high_mb", 0.0)))
            ep_hv_exp_proc   += _finite(info.get("high_value_expired_proc_mb",
                                                   info.get("expired_proc_high_mb", 0.0)))
            ep_hv_dropped    += _finite(info.get("high_value_dropped_mb", 0.0))
            ep_deadline_miss += _finite(info.get("deadline_missed_after_processed_mb", 0.0))
            ep_total_proc    += proc_step
            ep_total_dl      += dl_step
            ep_total_deliv   += deliv_step

            # 窗口进出检测
            if in_window and not in_window_prev:
                # 进入新窗口
                window_idx += 1
                window_start_step = step
                window_proc_start = _finite(
                    getattr(env, "comm_queue", None) and env.comm_queue.value or 0.0)
                window_raw_start  = _finite(
                    getattr(env, "data_queue", None) and env.data_queue.length or 0.0)
                window_soc_start  = _finite(getattr(env.battery, "soc", 0.0))
                window_alt_start  = _finite(env.altitude_m / 1e3)
                window_tx_alphas  = []
                window_cpu_alphas = []
                window_prop_alphas = []
                window_dl_mb      = 0.0
                window_proc_mb    = 0.0
                window_fallback   = 0
                window_psf        = 0
                window_sb         = 0
                window_max_cap    = 0.0
                window_tx_active_steps = 0
                window_hv_delivered = 0.0
                window_hv_expired   = 0.0

            if in_window:
                window_tx_alphas.append(alpha_tx)
                window_cpu_alphas.append(alpha_cpu)
                window_prop_alphas.append(alpha_prop)
                window_dl_mb  += dl_step
                window_proc_mb += proc_step
                cap = _finite((env._contact or {}).get("max_capacity_mbps", 0.0))
                window_max_cap = max(window_max_cap, cap)
                if bool(info.get("mission_pointing_fallback_applied", False)):
                    window_fallback += 1
                if bool(was_projected) or _finite(psf_meta.get("total_modification_l2", 0.0)) > 1e-6:
                    window_psf += 1
                if bool(info.get("safe_budget_fallback_applied", False)):
                    window_sb += 1
                if dl_step > 1e-4:
                    window_tx_active_steps += 1
                window_hv_delivered += _finite(info.get("delivered_high_value", 0.0))
                window_hv_expired   += _finite(info.get("high_value_expired_proc_mb",
                                                         info.get("expired_proc_high_mb", 0.0)))

            if (not in_window) and in_window_prev and window_idx >= 0:
                # 刚离开窗口：记录此窗口统计
                dur = step - window_start_step
                window_proc_end = _finite(
                    getattr(env, "comm_queue", None) and env.comm_queue.value or 0.0)
                n_w = max(len(window_tx_alphas), 1)
                rec = {
                    "method": method_name,
                    "seed": episode_seed,
                    "episode": ep,
                    "window_idx": window_idx,
                    "window_start_step": window_start_step,
                    "window_end_step": step,
                    "window_duration_steps": dur,
                    "proc_queue_at_window_start_mb": window_proc_start,
                    "proc_queue_at_window_end_mb": window_proc_end,
                    "raw_queue_at_window_start_mb": window_raw_start,
                    "tx_alpha_mean": float(np.mean(window_tx_alphas)) if window_tx_alphas else 0.0,
                    "cpu_alpha_mean": float(np.mean(window_cpu_alphas)) if window_cpu_alphas else 0.0,
                    "prop_alpha_mean": float(np.mean(window_prop_alphas)) if window_prop_alphas else 0.0,
                    "downlinked_mb": window_dl_mb,
                    "processed_mb": window_proc_mb,
                    "pointing_fallback_count": window_fallback,
                    "psf_intervene_count": window_psf,
                    "safe_budget_count": window_sb,
                    "window_util": window_tx_active_steps / max(dur, 1),
                    "max_tx_capacity_mbps": window_max_cap,
                    "soc_at_window_start": window_soc_start,
                    "altitude_km_at_window_start": window_alt_start,
                    "high_value_delivered_in_window": window_hv_delivered,
                    "high_value_expired_in_window": window_hv_expired,
                    "bottleneck_tag": "",
                }
                rec["bottleneck_tag"] = _tag_window_bottleneck(rec)
                window_records.append(rec)

            in_window_prev = in_window
            step += 1
            total_steps += 1

        # episode 结束：保存高价值生命周期
        ep_upr = _safe_div(ep_hv_delivered, ep_hv_processed)
        hv_lifecycle_records.append({
            "method": method_name,
            "seed": episode_seed,
            "episode": ep,
            "high_value_generated":   ep_hv_generated,
            "high_value_processed":   ep_hv_processed,
            "high_value_downlinked":  ep_hv_downlinked,
            "high_value_delivered":   ep_hv_delivered,
            "high_value_expired_raw": ep_hv_exp_raw,
            "high_value_expired_proc": ep_hv_exp_proc,
            "high_value_dropped":     ep_hv_dropped,
            "high_value_process_rate":    _safe_div(ep_hv_processed, ep_hv_generated),
            "high_value_downlink_rate":   _safe_div(ep_hv_downlinked, ep_hv_processed),
            "high_value_delivery_rate":   _safe_div(ep_hv_delivered, ep_hv_generated),
            "high_value_expired_rate":    _safe_div(ep_hv_exp_raw, ep_hv_generated),
            "processed_but_not_downlinked_rate": _safe_div(
                ep_total_proc - ep_total_dl, ep_total_proc),
            "deadline_miss_after_processed": ep_deadline_miss,
            "useful_processing_ratio": _safe_div(ep_hv_delivered, ep_hv_processed),
        })
        eps_hv_generated.append(ep_hv_generated) if False else None  # handled below

        eps_high_delivered.append(ep_hv_delivered)
        eps_high_processed.append(ep_hv_processed)
        eps_high_generated.append(ep_hv_generated)
        eps_high_expired_raw.append(ep_hv_exp_raw)
        eps_high_expired_proc.append(ep_hv_exp_proc)
        eps_high_dropped.append(ep_hv_dropped)
        eps_high_downlinked.append(ep_hv_downlinked)
        eps_processed.append(ep_total_proc)
        eps_downlinked.append(ep_total_dl)
        eps_delivered_value.append(ep_total_deliv)
        eps_useful_proc_ratio.append(ep_upr)
        eps_deadline_miss.append(ep_deadline_miss)
        eps_processed_not_dl.append(
            _safe_div(ep_total_proc - ep_total_dl, ep_total_proc))

        total_episodes += 1

    # ── 聚合统计 ────────────────────────────────────────────────────────────
    def _m(lst):
        arr = [x for x in lst if math.isfinite(x)]
        return float(np.mean(arr)) if arr else float("nan")

    def _tag_counts(records, tag_field="bottleneck_tag"):
        tags: dict[str, int] = {}
        for r in records:
            t = r.get(tag_field, "UNKNOWN")
            tags[t] = tags.get(t, 0) + 1
        return tags

    n_windows = len(window_records)
    tag_dist  = _tag_counts(window_records)
    dominant_tag = max(tag_dist, key=lambda t: tag_dist[t]) if tag_dist else "UNKNOWN"

    # 分组统计：有 processed vs 无 processed 的窗口
    has_proc = [r for r in window_records if r["proc_queue_at_window_start_mb"] > 0.1]
    no_proc  = [r for r in window_records if r["proc_queue_at_window_start_mb"] <= 0.1]

    def _mean_field(records, field):
        return _m([r[field] for r in records])

    summary = {
        "__meta__": {
            "timestamp": datetime.now().isoformat(),
            "model": args.model if loaded else "(untrained/random)",
            "method": method_name,
            "loaded_checkpoint": loaded,
            "device": device,
            "episodes": args.episodes,
            "total_steps": total_steps,
            "total_windows": n_windows,
        },
        "global_averages": {
            "delivered_value_mean": _m(eps_delivered_value),
            "processed_mb_mean": _m(eps_processed),
            "downlinked_mb_mean": _m(eps_downlinked),
            "proc_dl_ratio": _safe_div(sum(eps_processed), sum(eps_downlinked)),
            "useful_processing_ratio_mean": _m(eps_useful_proc_ratio),
            "processed_not_downlinked_rate": _m(eps_processed_not_dl),
            "deadline_miss_after_processed_mean": _m(eps_deadline_miss),
        },
        "high_value_lifecycle": {
            "generated_mean": _m(eps_high_generated),
            "processed_mean": _m(eps_high_processed),
            "downlinked_mean": _m(eps_high_downlinked),
            "delivered_mean": _m(eps_high_delivered),
            "expired_raw_mean": _m(eps_high_expired_raw),
            "expired_proc_mean": _m(eps_high_expired_proc),
            "dropped_mean": _m(eps_high_dropped),
            "process_rate": _safe_div(_m(eps_high_processed), _m(eps_high_generated)),
            "downlink_rate": _safe_div(_m(eps_high_downlinked), _m(eps_high_processed)),
            "delivery_rate": _safe_div(_m(eps_high_delivered), _m(eps_high_generated)),
            "expired_raw_rate": _safe_div(_m(eps_high_expired_raw), _m(eps_high_generated)),
            "high_value_expired_rate": _safe_div(
                _m(eps_high_expired_raw) + _m(eps_high_expired_proc),
                _m(eps_high_generated),
            ),
        },
        "window_breakdown": {
            "n_windows_total": n_windows,
            "n_windows_with_proc_at_start": len(has_proc),
            "n_windows_without_proc_at_start": len(no_proc),
            "windows_with_proc_fraction": _safe_div(len(has_proc), n_windows),
            "all_windows": {
                "proc_start_mean": _mean_field(window_records, "proc_queue_at_window_start_mb"),
                "proc_end_mean": _mean_field(window_records, "proc_queue_at_window_end_mb"),
                "downlinked_mean": _mean_field(window_records, "downlinked_mb"),
                "tx_alpha_mean": _mean_field(window_records, "tx_alpha_mean"),
                "cpu_alpha_mean": _mean_field(window_records, "cpu_alpha_mean"),
                "window_util_mean": _mean_field(window_records, "window_util"),
                "pointing_fallback_per_window": _mean_field(window_records, "pointing_fallback_count"),
                "psf_per_window": _mean_field(window_records, "psf_intervene_count"),
                "safe_budget_per_window": _mean_field(window_records, "safe_budget_count"),
                "hv_delivered_per_window": _mean_field(window_records, "high_value_delivered_in_window"),
                "hv_expired_per_window": _mean_field(window_records, "high_value_expired_in_window"),
            },
            "windows_with_proc_at_start": {
                "downlinked_mean": _mean_field(has_proc, "downlinked_mb") if has_proc else float("nan"),
                "tx_alpha_mean": _mean_field(has_proc, "tx_alpha_mean") if has_proc else float("nan"),
                "window_util_mean": _mean_field(has_proc, "window_util") if has_proc else float("nan"),
                "pointing_fallback_per_window": _mean_field(has_proc, "pointing_fallback_count") if has_proc else float("nan"),
            },
            "windows_without_proc_at_start": {
                "downlinked_mean": _mean_field(no_proc, "downlinked_mb") if no_proc else float("nan"),
                "tx_alpha_mean": _mean_field(no_proc, "tx_alpha_mean") if no_proc else float("nan"),
                "window_util_mean": _mean_field(no_proc, "window_util") if no_proc else float("nan"),
                "pointing_fallback_per_window": _mean_field(no_proc, "pointing_fallback_count") if no_proc else float("nan"),
            },
            "bottleneck_tag_distribution": tag_dist,
            "dominant_bottleneck_tag": dominant_tag,
        },
        "useful_processing_ratio_decomposition": {
            "total_processed": _m(eps_processed),
            "total_downlinked": _m(eps_downlinked),
            "total_delivered": _m(eps_delivered_value),
            "proc_to_dl_ratio": _safe_div(_m(eps_processed), _m(eps_downlinked)),
            "dl_to_delivered_ratio": _safe_div(_m(eps_downlinked), _m(eps_delivered_value)),
            "useful_proc_ratio_mean": _m(eps_useful_proc_ratio),
            "high_value_delivery_ratio_of_generated": _safe_div(
                _m(eps_high_delivered), _m(eps_high_generated)),
            "high_value_expired_of_generated": _safe_div(
                _m(eps_high_expired_raw) + _m(eps_high_expired_proc),
                _m(eps_high_generated),
            ),
        },
    }

    # ── 瓶颈结论生成 ─────────────────────────────────────────────────────────
    wb = summary["window_breakdown"]
    gl = summary["global_averages"]
    hv = summary["high_value_lifecycle"]

    bottleneck_conclusions = []

    no_proc_frac = _safe_div(wb["n_windows_without_proc_at_start"], max(n_windows, 1))
    if math.isfinite(no_proc_frac) and no_proc_frac > 0.40:
        bottleneck_conclusions.append(
            f"[瓶颈 A] 窗口前 processed queue 不足：{no_proc_frac:.1%} 的通信窗口开始时"
            f" processed=0，说明 contact-aware preprocessing 不足 → "
            f"建议启用 enable_contact_aware_preprocessing 或增大 contact_aware_lead_steps。"
        )

    all_w = wb["all_windows"]
    if all_w["tx_alpha_mean"] < 0.3 and all_w["window_util_mean"] < 0.4:
        bottleneck_conclusions.append(
            f"[瓶颈 B] 窗口内 TX 利用不足：tx_alpha_mean={all_w['tx_alpha_mean']:.2f}，"
            f"window_util={all_w['window_util_mean']:.1%} → "
            f"检查 SAFE_BUDGET TX floor / credit gate 是否过度限制窗口内发射。"
        )

    fallback_rate = all_w["pointing_fallback_per_window"] / max(
        wb["all_windows"].get("window_util_mean", 1.0) * 10 + 1, 1)
    if all_w["pointing_fallback_per_window"] > 5:
        bottleneck_conclusions.append(
            f"[瓶颈 C] Pointing fallback 压制下传："
            f"每窗口平均 {all_w['pointing_fallback_per_window']:.1f} 次 fallback → "
            f"PSF 或 SAFE_BUDGET 在窗口内频繁切换姿态，阻断 TX。"
        )

    pnd_rate = _finite(gl["processed_not_downlinked_rate"])
    if math.isfinite(pnd_rate) and pnd_rate > 0.40:
        bottleneck_conclusions.append(
            f"[瓶颈 D] Processed-but-not-downlinked 高：{pnd_rate:.1%} 的处理数据未下传 → "
            f"建议加入 processed-not-delivered penalty 或提高 in-contact TX floor。"
        )

    hv_exp_rate = _finite(hv["high_value_expired_rate"])
    if math.isfinite(hv_exp_rate) and hv_exp_rate > 0.30:
        bottleneck_conclusions.append(
            f"[瓶颈 E] 高价值任务大量过期：expired_rate={hv_exp_rate:.1%} → "
            f"raw 过期={hv['expired_raw_rate']:.1%}，proc 过期={hv['expired_proc_mean']:.1f}MB → "
            f"需加强高价值处理优先级 (enable_high_value_cpu_gate_escape) 和交付优先级。"
        )

    hv_proc_rate = _finite(hv["process_rate"])
    if math.isfinite(hv_proc_rate) and hv_proc_rate < 0.30:
        bottleneck_conclusions.append(
            f"[瓶颈 F] 高价值任务处理率低：process_rate={hv_proc_rate:.1%} → "
            f"CPU 资源未优先分配给高价值任务。检查 class_high_reward_weight 和 CPU logit 分配。"
        )

    dl_miss_mean = _finite(gl["deadline_miss_after_processed_mean"])
    if math.isfinite(dl_miss_mean) and dl_miss_mean > 0.5:
        bottleneck_conclusions.append(
            f"[瓶颈 G] 处理后未能 deadline 内下传：deadline_miss_after_processed={dl_miss_mean:.1f}MB/ep → "
            f"处理了但来不及在 deadline 前下传，需要更早开始处理或增大 TX 优先级。"
        )

    if not bottleneck_conclusions:
        bottleneck_conclusions.append("[OK] 未检出明显吞吐瓶颈；delivered value 低可能源于任务生成量本身或 reward 导向。")

    summary["bottleneck_conclusions"] = bottleneck_conclusions
    summary["dominant_bottleneck"] = dominant_tag

    # ── 打印 ────────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  吞吐瓶颈审计 ({method_name}, {total_episodes} eps, {n_windows} windows)")
    print(f"{'='*80}")
    print(f"\n  全局指标:")
    print(f"    delivered_value:         {_finite(gl['delivered_value_mean']):>10.1f}")
    print(f"    processed_mb:            {_finite(gl['processed_mb_mean']):>10.1f}")
    print(f"    downlinked_mb:           {_finite(gl['downlinked_mb_mean']):>10.1f}")
    print(f"    proc/dl:                 {_finite(gl['proc_dl_ratio']):>10.2f}")
    print(f"    useful_processing_ratio: {_finite(gl['useful_processing_ratio_mean']):>10.3f}")
    print(f"    processed_not_dl_rate:   {_finite(gl['processed_not_downlinked_rate']):>10.1%}")
    print(f"\n  高价值生命周期:")
    for k, v in hv.items():
        if isinstance(v, float):
            fmt = f"    {k:<38} {v:>8.3f}"
            print(fmt)
    print(f"\n  通信窗口分解 ({n_windows} windows):")
    print(f"    有 proc 的窗口: {wb['n_windows_with_proc_at_start']}/{n_windows}"
          f" ({_safe_div(wb['n_windows_with_proc_at_start'], max(n_windows,1)):.1%})")
    print(f"    主要瓶颈标签: {dominant_tag}")
    print(f"    标签分布: {tag_dist}")
    print(f"\n  瓶颈结论:")
    for c in bottleneck_conclusions:
        print(f"    {c}")

    # ── 保存结果 ─────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out     = args.output or f"results/throughput_bottleneck_audit_{ts}.json"
    csv_out = args.csv_output or os.path.splitext(out)[0] + "_windows.csv"
    hv_csv  = os.path.splitext(out)[0] + "_hv_lifecycle.csv"
    md_out  = os.path.splitext(out)[0] + "_summary.md"
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)

    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=WINDOW_RECORD_FIELDS)
        writer.writeheader()
        for rec in window_records:
            writer.writerow({k: rec.get(k, "") for k in WINDOW_RECORD_FIELDS})

    with open(hv_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HV_LIFECYCLE_FIELDS)
        writer.writeheader()
        for rec in hv_lifecycle_records:
            writer.writerow({k: rec.get(k, "") for k in HV_LIFECYCLE_FIELDS})

    summary["__meta__"]["window_csv"] = csv_out
    summary["__meta__"]["hv_lifecycle_csv"] = hv_csv
    summary["__meta__"]["summary_md"] = md_out
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Markdown summary ─────────────────────────────────────────────────────
    def _fv(v):
        if not isinstance(v, float) or not math.isfinite(v):
            return "n/a"
        return f"{v:.3f}"

    md_lines = [
        f"# Throughput Bottleneck Audit — {method_name}",
        f"",
        f"**Model**: `{args.model if loaded else '(untrained)'}`  ",
        f"**Episodes**: {total_episodes}  **Windows**: {n_windows}  ",
        f"",
        f"## Global Averages",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]
    for k, v in gl.items():
        md_lines.append(f"| {k} | {_fv(v)} |")

    md_lines += [
        f"",
        f"## High-Value Task Lifecycle",
        f"",
        f"| Stage | Value |",
        f"|-------|-------|",
    ]
    for k, v in hv.items():
        md_lines.append(f"| {k} | {_fv(v)} |")

    md_lines += [
        f"",
        f"## Communication Window Breakdown",
        f"",
        f"| Condition | Value |",
        f"|-----------|-------|",
        f"| Total windows | {n_windows} |",
        f"| Windows with proc>0 at start | {wb['n_windows_with_proc_at_start']} "
        f"({_fv(wb['windows_with_proc_fraction'])}) |",
        f"| Dominant bottleneck tag | **{dominant_tag}** |",
        f"",
        f"| Tag | Count |",
        f"|-----|-------|",
    ]
    for tag, cnt in sorted(tag_dist.items(), key=lambda x: -x[1]):
        md_lines.append(f"| {tag} | {cnt} |")

    md_lines += [
        f"",
        f"### Per-Window Averages (all)",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]
    for k, v in all_w.items():
        md_lines.append(f"| {k} | {_fv(v)} |")

    md_lines += [
        f"",
        f"## Bottleneck Conclusions",
        f"",
    ]
    for c in bottleneck_conclusions:
        md_lines.append(f"- {c}")
    md_lines.append("")

    with open(md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n  JSON:  {out}")
    print(f"  Window CSV: {csv_out}")
    print(f"  HV lifecycle CSV: {hv_csv}")
    print(f"  Markdown: {md_out}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="吞吐瓶颈审计")
    parser.add_argument("--model", default=os.path.join(
        TRAIN_CONFIG.get("optimized_checkpoint_dir", "checkpoints_optimized/"),
        "best_optimized.pt"))
    parser.add_argument("--random", action="store_true",
                        help="无 checkpoint，用未训练策略仅看结构")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=int(TRAIN_CONFIG.get("seed", 42)) + 8000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--method-name", default=None,
                        dest="method_name",
                        help="方法名（写入 CSV/JSON，默认为 Ours 或 Random/untrained）")
    args = parser.parse_args()
    run(args)

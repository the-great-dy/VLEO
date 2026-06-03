"""多时间尺度评估（顶刊 Issue #3）。

审稿意见：训练 episode 只有 ~6h（4 个轨道），无法支撑"VLEO 长期生存性 / 轨道
维持 / 长期任务调度"的结论。建议三层评估：

    层级           horizon              目的
    short          4 orbits / 6h        与训练一致，训练稳定性
    operational    24h / ~16 orbits     日周期能量/通信/任务积压
    lifetime       7 days (rolling)     轨道维持/燃料/长期队列稳定性

实现：固定策略权重，仅在评估时**覆盖** TRAIN_CONFIG["max_episode_steps"] 把
episode 拉长，对环境做 rollout（不反向传播），逐 horizon 跨多 seed 聚合关键
可持续性指标，并判断生存率/安全率是否在更长 horizon 上保持。

复用 evaluate_optimized.evaluate_model（与训练同口径观测链路 + 安全层），
并用 multi_seed.aggregate 做均值 ± 95% CI。

用法（按记忆要求长跑用 GPU）:
    python experiments/multi_horizon_eval.py --model checkpoints_optimized/best_optimized.pt --device cuda
    python experiments/multi_horizon_eval.py --model ckpt.pt --horizons short,operational --seeds 42,43,44
"""

from __future__ import annotations

import argparse
import json
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

from config import ENERGY_CONFIG, TRAIN_CONFIG
from evaluate_optimized import _resolve_device, evaluate_model
from experiments.multi_seed import aggregate, resolve_seeds

G0 = 9.80665
SECONDS_PER_DAY = 86400.0

# 标准 horizon 定义（以 dt=time_slot_s 为步长）。
_TIME_SLOT_S = float(TRAIN_CONFIG.get("time_slot_s", 10.0))


def horizon_specs() -> dict:
    base = int(TRAIN_CONFIG.get("max_episode_steps", 2160))
    steps_per_hour = int(round(3600.0 / max(_TIME_SLOT_S, 1e-6)))
    return {
        "short": {
            "max_episode_steps": base,
            "hours": base * _TIME_SLOT_S / 3600.0,
            "desc": "训练一致 (~6h / 4 orbits)",
        },
        "operational": {
            "max_episode_steps": 24 * steps_per_hour,
            "hours": 24.0,
            "desc": "24h 日周期",
        },
        "lifetime": {
            "max_episode_steps": 7 * 24 * steps_per_hour,
            "hours": 7 * 24.0,
            "desc": "7 天 rolling 生存性",
        },
    }


# 跨 horizon 追踪的可持续性指标（evaluate_model 返回键）。
SUSTAINABILITY_KEYS = [
    "survival_rate",
    "crash_count",
    "safety_rate",
    "orbit_safe_rate",
    "energy_safe_rate",
    "thermal_safe_rate",
    "processed_queue_peak_utilization",
    "downlink_mean_mb",
    "delivered_value_mean",
    "mean_prop_power",
    "reward_per_step_mean",
]


def _real_time_fuel_kg_per_day(mean_prop_power_w: float) -> float:
    """由平均推进功率反推**真实时间**燃料速率 (kg/day)，不含 episode 时间压缩。

    mdot = P·eff/(Isp·g0)²；用于长 horizon 下"长期燃料是否可接受"的判断。
    """
    isp_s = float(ENERGY_CONFIG.get("propulsion_isp_s", 1500.0))
    eff = float(ENERGY_CONFIG.get("propulsion_efficiency", 0.65))
    mdot_kg_s = max(0.0, float(mean_prop_power_w)) * eff / (isp_s * G0) ** 2
    return mdot_kg_s * SECONDS_PER_DAY


def run(args) -> dict:
    device = _resolve_device(args.device)
    seeds, seed_source = resolve_seeds(args.seeds)
    specs = horizon_specs()
    selected = [h.strip() for h in str(args.horizons).split(",") if h.strip()]
    for h in selected:
        if h not in specs:
            raise ValueError(f"未知 horizon={h}，可选 {list(specs.keys())}")

    original_max = int(TRAIN_CONFIG.get("max_episode_steps", 2160))
    report = {
        "__meta__": {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "device": device,
            "seeds": seeds,
            "seed_source": seed_source,
            "eval_episodes_per_seed": int(args.eval_episodes),
            "time_slot_s": _TIME_SLOT_S,
            "horizons": {h: specs[h] for h in selected},
        },
        "by_horizon": {},
    }

    try:
        for h in selected:
            spec = specs[h]
            TRAIN_CONFIG["max_episode_steps"] = int(spec["max_episode_steps"])
            print(f"\n{'=' * 78}\n  horizon={h}  ({spec['desc']}, "
                  f"max_steps={spec['max_episode_steps']}, {spec['hours']:.0f}h)\n{'=' * 78}")
            rows = []
            for seed in seeds:
                print(f"  [eval] horizon={h} seed={seed}")
                rows.append(evaluate_model(
                    args.model,
                    n_episodes=args.eval_episodes,
                    device=device,
                    eval_seed=seed,
                ))
            summary = aggregate(rows)
            fuel_rate = _real_time_fuel_kg_per_day(
                summary.get("mean_prop_power", {}).get("mean", 0.0))
            report["by_horizon"][h] = {
                "spec": spec,
                "summary": {k: summary[k] for k in SUSTAINABILITY_KEYS if k in summary},
                "real_time_fuel_kg_per_day": fuel_rate,
                "full_summary": summary,
                "by_seed": rows,
            }
            s = report["by_horizon"][h]["summary"]
            print(f"  survival={s.get('survival_rate', {}).get('mean', 0):.1%}  "
                  f"safety={s.get('safety_rate', {}).get('mean', 0):.1%}  "
                  f"q_peak={s.get('processed_queue_peak_utilization', {}).get('mean', 0):.2f}  "
                  f"fuel≈{fuel_rate:.4f} kg/day")
    finally:
        TRAIN_CONFIG["max_episode_steps"] = original_max

    # ── 跨 horizon 退化判断：生存/安全率是否随 horizon 拉长而崩塌 ──
    trend = {}
    for key in ("survival_rate", "safety_rate", "orbit_safe_rate", "energy_safe_rate"):
        trend[key] = {
            h: report["by_horizon"][h]["summary"].get(key, {}).get("mean", None)
            for h in selected
        }
    report["cross_horizon_trend"] = trend
    if "short" in selected and "lifetime" in selected:
        short_surv = trend["survival_rate"].get("short") or 0.0
        life_surv = trend["survival_rate"].get("lifetime") or 0.0
        report["__meta__"]["long_horizon_survival_holds"] = bool(life_surv >= 0.95 * short_surv)

    os.makedirs("results", exist_ok=True)
    out = args.output or f"results/multi_horizon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {out}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多时间尺度评估 (short/operational/lifetime)")
    parser.add_argument("--model", required=True, help="待评估 checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default=None,
                        help="逗号分隔种子；不填读取 TRAIN_CONFIG.eval_seeds")
    parser.add_argument("--horizons", default="short,operational,lifetime",
                        help="逗号分隔：short,operational,lifetime")
    parser.add_argument("--eval_episodes", type=int, default=3,
                        help="每 seed 评估 episode 数（长 horizon 建议小，否则太慢）")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(args)

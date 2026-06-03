"""任务价值场景敏感性实验（顶刊 Issue#7）。

审稿意见：任务价值模型是 synthetic（config 已诚实声明 scene_model_source=
"synthetic_scene_prior", scene_profiles_are_empirical=False）。因此论文不能说
"基于真实遥感任务价值分布验证"，只能说"在 synthetic but physically grounded
VLEO task-value benchmark 上验证"。为排除"方法只是在特定 synthetic scene rule
上调出来的"，需在多种任务价值分布下复测主结论。

本脚本固定策略权重，按场景覆盖 TASK_CONFIG（场景价值乘子 / 相位规则 / 应急事件
/ 相位随机化），跨多 seed 评估，比较关键 VoI 指标是否稳健：

    uniform_value        所有场景等价值 → 检查是否只是"高价值规则"在起作用
    mild_skew            价值谱压缩一半 → 贴近普通遥感任务
    heavy_tail_emergency 放大灾害/军事/应急价值 → 应急任务鲁棒性
    no_emergency         关闭突发灾害事件 → 主结论是否依赖突发灾害
    random_scene_prior   打乱相位场景顺序+相位随机平移 → 是否过拟合固定相位规则
    baseline             现有 config（对照）

用法（按记忆要求长跑用 GPU）:
    python experiments/scene_value_sensitivity.py --model checkpoints_optimized/best_optimized.pt --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from config import TASK_CONFIG
from evaluate_optimized import _resolve_device, evaluate_model
from experiments.multi_seed import aggregate, resolve_seeds

# 主结论相关的 VoI / 安全指标。
REPORT_KEYS = [
    "delivered_value_mean",
    "high_value_delivery_rate",
    "deadline_success_rate",
    "value_weighted_deadline_success_rate",
    "voi_loss_rate",
    "survival_rate",
    "safety_rate",
]

_TOP_SCENES = ("disaster", "military", "emergency_disaster")


def _scene_overrides() -> dict:
    """返回每个场景对 TASK_CONFIG 的覆盖字典（深拷贝后修改）。"""
    base_profiles = copy.deepcopy(TASK_CONFIG["scene_profiles"])

    def uniform_profiles():
        prof = copy.deepcopy(base_profiles)
        for s in prof.values():
            s["base_value_multiplier"] = 1.0
            s["arrival_multiplier"] = 1.0
            s["priority_range"] = (1.0, 1.0)
        return prof

    def mild_skew_profiles():
        # 价值谱向 1.0 压缩一半，保留 ordering 但削弱长尾。
        prof = copy.deepcopy(base_profiles)
        for s in prof.values():
            bvm = float(s.get("base_value_multiplier", 1.0))
            s["base_value_multiplier"] = 0.5 * bvm + 0.5 * 1.0
            lo, hi = s.get("priority_range", (1.0, 1.0))
            s["priority_range"] = (0.5 * lo + 0.5, 0.5 * hi + 0.5)
        return prof

    def heavy_tail_profiles():
        # 放大灾害/军事/应急的价值与优先级，制造更重的尾。
        prof = copy.deepcopy(base_profiles)
        for name in _TOP_SCENES:
            if name in prof:
                prof[name]["base_value_multiplier"] = float(prof[name].get("base_value_multiplier", 1.0)) * 2.0
                lo, hi = prof[name].get("priority_range", (1.0, 1.0))
                prof[name]["priority_range"] = (lo * 1.5, hi * 1.5)
        return prof

    return {
        "baseline": {},
        "uniform_value": {"scene_profiles": uniform_profiles()},
        "mild_skew": {"scene_profiles": mild_skew_profiles()},
        "heavy_tail_emergency": {"scene_profiles": heavy_tail_profiles()},
        "no_emergency": {"emergency_event_enable": False},
        "random_scene_prior": {
            "randomize_scene_rule_order": True,
            "randomize_scene_phase_offset": True,
            "scene_phase_offset_max_fraction": 1.0,
        },
    }


@contextmanager
def _temporary_task_config(overrides: dict):
    """深拷贝保存并覆盖 TASK_CONFIG 指定键，退出还原（env reset 时 live 读取）。"""
    saved = {key: copy.deepcopy(TASK_CONFIG.get(key)) for key in overrides}
    missing = {key for key in overrides if key not in TASK_CONFIG}
    try:
        TASK_CONFIG.update(copy.deepcopy(overrides))
        yield
    finally:
        for key in overrides:
            if key in missing:
                TASK_CONFIG.pop(key, None)
            else:
                TASK_CONFIG[key] = saved[key]


def run(args) -> dict:
    device = _resolve_device(args.device)
    seeds, seed_source = resolve_seeds(args.seeds)
    overrides = _scene_overrides()
    selected = [s.strip() for s in str(args.scenarios).split(",") if s.strip()] \
        if args.scenarios else list(overrides.keys())
    for s in selected:
        if s not in overrides:
            raise ValueError(f"未知 scenario={s}，可选 {list(overrides.keys())}")

    report = {
        "__meta__": {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "device": device,
            "seeds": seeds,
            "seed_source": seed_source,
            "eval_episodes_per_seed": int(args.eval_episodes),
            "scene_model_source": TASK_CONFIG.get("scene_model_source"),
            "scene_profiles_are_empirical": TASK_CONFIG.get("scene_profiles_are_empirical"),
            "scenarios": selected,
            "note": "synthetic but physically grounded VLEO task-value benchmark; 多分布鲁棒性复测",
        },
        "by_scenario": {},
    }

    for scen in selected:
        print(f"\n{'=' * 78}\n  scenario={scen}\n{'=' * 78}")
        rows = []
        with _temporary_task_config(overrides[scen]):
            for seed in seeds:
                print(f"  [eval] scenario={scen} seed={seed}")
                rows.append(evaluate_model(
                    args.model, n_episodes=args.eval_episodes,
                    device=device, eval_seed=seed))
        summary = aggregate(rows)
        report["by_scenario"][scen] = {
            "summary": {k: summary[k] for k in REPORT_KEYS if k in summary},
            "full_summary": summary,
            "by_seed": rows,
        }
        s = report["by_scenario"][scen]["summary"]
        print(f"  delivered_value={s.get('delivered_value_mean', {}).get('mean', 0):.1f}  "
              f"hi_del={s.get('high_value_delivery_rate', {}).get('mean', 0):.1%}  "
              f"survival={s.get('survival_rate', {}).get('mean', 0):.1%}")

    # ── 鲁棒性判断：uniform_value 下高价值交付优势是否仍显著高于"随机/均匀" ──
    # 若 uniform 下 hi_del 与 baseline 接近，说明优势来自结构性调度而非价值偏置规则。
    def _hi_del(scen):
        return report["by_scenario"].get(scen, {}).get(
            "summary", {}).get("high_value_delivery_rate", {}).get("mean", None)

    trend = {scen: _hi_del(scen) for scen in selected}
    report["high_value_delivery_rate_by_scenario"] = trend
    if "baseline" in selected and "uniform_value" in selected:
        base_hi = trend.get("baseline") or 0.0
        uni_hi = trend.get("uniform_value") or 0.0
        report["__meta__"]["conclusion_robust_to_uniform_value"] = bool(
            uni_hi >= 0.8 * base_hi)

    os.makedirs("results", exist_ok=True)
    out = args.output or f"results/scene_value_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {out}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="任务价值场景敏感性 (Issue#7)")
    parser.add_argument("--model", required=True, help="待评估 checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default=None,
                        help="逗号分隔种子；不填读取 TRAIN_CONFIG.eval_seeds")
    parser.add_argument("--scenarios", default=None,
                        help="逗号分隔场景子集；不填跑全部 6 个")
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(args)

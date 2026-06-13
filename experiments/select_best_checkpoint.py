"""权威 best-checkpoint 选择（canonical 20-seed eval 口径）。

[SAFE_BUDGET 2026-06-08] 训练内 periodic eval（单 seed / 少 episode）不可靠，禁止用于
最终 best 选择。本工具对每个候选 checkpoint 读取/运行 canonical 20-seed evaluation，
按 config.CHECKPOINT_SELECTION_CONFIG 的 safety floor + utility floor + anti-conservative
filter + safety-constrained utility score 选 best；没有候选满足全部条件时**保留上一版
交付基线**，绝不退化选择 window 极低的保守躺平模型。

用法：
  # 仅用已有 canonical eval JSON 比较并选 best（不重新跑 eval）
  python experiments/select_best_checkpoint.py \
      --candidate "delivery_baseline:OLD best_optimized+SAFE_BUDGET:results/multiseed_safebudget_soft060_20260607.json" \
      --candidate "retrain_best(200k):checkpoints_safebudget/best_optimized.pt:results/multiseed_safebudget_RETRAIN_best_20260608.json" \
      --candidate "retrain_latest(540k):checkpoints_safebudget/latest.pt:results/multiseed_safebudget_RETRAIN_latest_20260608.json"

  # 候选缺 JSON 时即时跑 canonical eval（需 --eval-missing，传 checkpoint 路径）
  python experiments/select_best_checkpoint.py --eval-missing \
      --candidate "retrain_stageF:checkpoints_safebudget/best_stage_Final.pt:"

候选格式：  label:checkpoint_path:eval_json_path
  - eval_json_path 为空且 --eval-missing → 用 multi_seed 跑 canonical eval 生成。
  - checkpoint_path 可为空（如纯 baseline 只有 JSON）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from config import CHECKPOINT_SELECTION_CONFIG, TRAIN_CONFIG  # noqa: E402

CFG = CHECKPOINT_SELECTION_CONFIG


# ── canonical 指标抽取（统一从 multi_seed 的 model_summary 读 mean/min）─────────
def _load_canonical_metrics(eval_json: str) -> dict:
    with open(eval_json, "r", encoding="utf-8") as f:
        blob = json.load(f)
    s = blob.get("model_summary") or {}

    def m(key, default=0.0):
        node = s.get(key)
        if isinstance(node, dict):
            return float(node.get("mean", default))
        return float(default)

    def m_any(keys, default=0.0):
        for key in keys:
            node = s.get(key)
            if isinstance(node, dict):
                return float(node.get("mean", default))
            if isinstance(node, (int, float)):
                return float(node)
        return float(default)

    def mn(key, default=0.0):
        node = s.get(key)
        if isinstance(node, dict):
            return float(node.get("min", default))
        return float(default)

    rf_downlinked_mb = m_any((
        "rf_downlinked_mb_mean",
        "downlink_mean_mb",
        "downlink_mean",
        "tx_mb_mean",
    ))
    processed_product_mb = m_any((
        "processed_product_mb_mean",
        "processed_mean_mb",
        "processed_mean",
    ))
    raw_equiv_delivered_mb = m_any((
        "raw_equivalent_delivered_mb_mean",
        "Raw-equivalent Delivered MB",
    ), default=rf_downlinked_mb)
    raw_equiv_processed_mb = m_any((
        "raw_equivalent_processed_mb_mean",
        "Raw-equivalent Processed MB",
    ), default=processed_product_mb)
    raw_equiv_coverage = m_any((
        "raw_equivalent_delivery_coverage_mean",
        "Raw-equivalent Delivery Coverage",
    ), default=(
        raw_equiv_delivered_mb / max(raw_equiv_processed_mb, 1e-9)
        if raw_equiv_processed_mb > 1e-9 else 0.0
    ))
    value_realization_ratio = m_any((
        "value_realization_ratio_mean",
        "Value Realization Ratio",
        "episode_useful_processing_ratio",
        "useful_processing_ratio",
    ))
    rf_product_proc_downlink_ratio = m_any((
        "rf_product_proc_downlink_ratio_mean",
        "RF Product Proc/DL Ratio",
        "proc_downlink_ratio",
        "proc_dl_ratio",
    ), default=(
        processed_product_mb / max(rf_downlinked_mb, 1e-9)
        if processed_product_mb > 1e-9 else 0.0
    ))

    return {
        "n_seeds": int(s.get("n_seeds", 0)) if not isinstance(s.get("n_seeds"), dict) else int(s["n_seeds"].get("mean", 0)),
        "episode_safety_rate": m("episode_safety_rate"),
        "worst_seed_episode_safety_rate": mn("episode_safety_rate"),
        "survival_rate": m("survival_rate"),
        "crash_count": m("crash_count"),
        "comm_window_utilization": m("comm_window_utilization"),
        "rf_downlinked_mb": rf_downlinked_mb,
        "downlink_mb": rf_downlinked_mb,
        "raw_equivalent_delivered_mb": raw_equiv_delivered_mb,
        "raw_equivalent_processed_mb": raw_equiv_processed_mb,
        "raw_equivalent_delivery_coverage": raw_equiv_coverage,
        "value_realization_ratio": value_realization_ratio,
        "delivered_value": m("delivered_value_mean"),
        "rf_product_proc_downlink_ratio": rf_product_proc_downlink_ratio,
        "proc_dl_ratio": rf_product_proc_downlink_ratio,
        "intervention_rate": m("intervention_rate"),
        "projection_rate": m("lyapunov_projection_rate") + m("psf_filter_rate"),
    }


def _run_canonical_eval(checkpoint_path: str, out_json: str, device: str, episodes: int) -> str:
    seeds = ",".join(str(x) for x in TRAIN_CONFIG.get("eval_seeds", [42]))
    cmd = [
        sys.executable, os.path.join(_ROOT, "experiments", "multi_seed.py"),
        "--mode", "eval", "--model", checkpoint_path, "--device", device,
        "--eval_episodes", str(episodes), "--seeds", seeds, "--output", out_json,
    ]
    print(f"  [canonical eval] {checkpoint_path} -> {out_json}")
    subprocess.run(cmd, check=True, cwd=_ROOT)
    return out_json


# ── 三段门控 ──────────────────────────────────────────────────────────────────
def _safety_floor(mx: dict) -> tuple[bool, str]:
    if mx["episode_safety_rate"] < CFG["min_episode_safety_rate"]:
        return False, f"ep_safe {mx['episode_safety_rate']:.2f}<{CFG['min_episode_safety_rate']}"
    if mx["survival_rate"] < CFG["min_survival_rate"]:
        return False, f"survival {mx['survival_rate']:.2f}<{CFG['min_survival_rate']}"
    if mx["crash_count"] > CFG["max_crash_count"] + 1e-9:
        return False, f"crash {mx['crash_count']:.2f}>0"
    return True, "ok"


def _utility_floor(mx: dict) -> tuple[bool, str]:
    if mx["comm_window_utilization"] < CFG["min_comm_window_utilization"]:
        return False, f"window {mx['comm_window_utilization']:.3f}<{CFG['min_comm_window_utilization']}"
    min_rf = float(CFG.get("min_rf_downlinked_mb", CFG.get("min_downlink_mb", 0.0)))
    if mx["rf_downlinked_mb"] < min_rf:
        return False, f"rf_downlinked {mx['rf_downlinked_mb']:.0f}<{min_rf:.0f}"
    min_raw_equiv = float(CFG.get("min_raw_equivalent_delivered_mb", 0.0))
    if mx["raw_equivalent_delivered_mb"] < min_raw_equiv:
        return False, f"raw_eq_delivered {mx['raw_equivalent_delivered_mb']:.0f}<{min_raw_equiv:.0f}"
    min_cov = float(CFG.get("min_raw_equivalent_delivery_coverage", 0.0))
    if mx["raw_equivalent_delivery_coverage"] < min_cov:
        return False, f"raw_eq_coverage {mx['raw_equivalent_delivery_coverage']:.3f}<{min_cov:.3f}"
    min_val_real = float(CFG.get("min_value_realization_ratio", 0.0))
    if mx["value_realization_ratio"] < min_val_real:
        return False, f"value_realization {mx['value_realization_ratio']:.3f}<{min_val_real:.3f}"
    if mx["delivered_value"] < CFG["min_delivered_value"]:
        return False, f"delivered {mx['delivered_value']:.0f}<{CFG['min_delivered_value']:.0f}"
    return True, "ok"


def _conservative_collapse(mx: dict) -> tuple[bool, str]:
    if mx["comm_window_utilization"] < CFG["conservative_collapse_window"]:
        return True, f"conservative_collapse(window {mx['comm_window_utilization']:.3f})"
    low_raw_equiv = float(CFG.get(
        "low_utility_raw_equivalent_delivered_mb",
        CFG.get("low_utility_downlink_mb", 0.0),
    ))
    if mx["raw_equivalent_delivered_mb"] < low_raw_equiv:
        return True, f"low_utility(raw_eq_delivered {mx['raw_equivalent_delivered_mb']:.0f})"
    if mx["delivered_value"] < CFG["low_delivery_value"]:
        return True, f"low_delivery(delivered {mx['delivered_value']:.0f})"
    return False, ""


# ── safety-constrained utility score（仅对通过全部门的候选计算）──────────────────
def _minmax_norm(vals: list[float]) -> list[float]:
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def _compute_scores(eligible: list[dict]) -> None:
    if not eligible:
        return
    dn = _minmax_norm([c["mx"]["delivered_value"] for c in eligible])
    rn = _minmax_norm([c["mx"]["raw_equivalent_delivered_mb"] for c in eligible])
    cn = _minmax_norm([c["mx"]["raw_equivalent_delivery_coverage"] for c in eligible])
    vn = _minmax_norm([c["mx"]["value_realization_ratio"] for c in eligible])
    win = _minmax_norm([c["mx"]["comm_window_utilization"] for c in eligible])
    pn = _minmax_norm([c["mx"]["rf_product_proc_downlink_ratio"] for c in eligible])
    for i, c in enumerate(eligible):
        mx = c["mx"]
        viol = max(0.0, 1.0 - mx["episode_safety_rate"])
        interv = mx["intervention_rate"]
        c["score"] = (
            CFG["score_w_delivered"] * dn[i]
            + float(CFG.get("score_w_raw_equivalent_delivered", 1.0)) * rn[i]
            + float(CFG.get("score_w_raw_equivalent_coverage", 0.5)) * cn[i]
            + float(CFG.get("score_w_value_realization", 0.5)) * vn[i]
            + CFG["score_w_window"] * win[i]
            - float(CFG.get(
                "score_w_rf_product_pressure",
                CFG.get("score_w_proc_dl", 0.0),
            )) * pn[i]
            - CFG["score_w_violation"] * viol
            - CFG["score_w_intervention"] * interv
        )


def _passes_replacement(mx: dict, baseline_mx: dict | None) -> tuple[bool, str]:
    """新模型替换锁定基线的条件。"""
    if mx["episode_safety_rate"] < CFG["min_episode_safety_rate"]:
        return False, f"ep_safe<{CFG['min_episode_safety_rate']}"
    if mx["comm_window_utilization"] < CFG["min_comm_window_utilization"]:
        return False, f"window<{CFG['min_comm_window_utilization']}"
    min_rf = float(CFG.get("min_rf_downlinked_mb", CFG.get("min_downlink_mb", 0.0)))
    if mx["rf_downlinked_mb"] < min_rf:
        return False, f"rf_downlinked<{min_rf:.0f}"
    min_raw_equiv = float(CFG.get("min_raw_equivalent_delivered_mb", 0.0))
    if mx["raw_equivalent_delivered_mb"] < min_raw_equiv:
        return False, f"raw_eq_delivered<{min_raw_equiv:.0f}"
    if mx["delivered_value"] < CFG["min_delivered_value"]:
        return False, f"delivered<{CFG['min_delivered_value']:.0f}"
    if baseline_mx and CFG.get("replace_requires_raw_equivalent_not_worse_than_baseline", True):
        baseline_raw_eq = float(baseline_mx.get("raw_equivalent_delivered_mb", 0.0))
        if mx["raw_equivalent_delivered_mb"] + 1e-9 < baseline_raw_eq:
            return False, (
                f"raw_eq_delivered {mx['raw_equivalent_delivered_mb']:.0f}"
                f"<baseline {baseline_raw_eq:.0f}"
            )
    if baseline_mx and CFG.get("replace_requires_value_realization_not_worse_than_baseline", False):
        baseline_val_real = float(baseline_mx.get("value_realization_ratio", 0.0))
        if mx["value_realization_ratio"] + 1e-9 < baseline_val_real:
            return False, (
                f"value_realization {mx['value_realization_ratio']:.3f}"
                f"<baseline {baseline_val_real:.3f}"
            )
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", action="append", default=[],
                    help="label:checkpoint_path:eval_json_path（可重复）")
    ap.add_argument("--eval-missing", action="store_true",
                    help="候选缺 eval_json 时即时跑 canonical 20-seed eval")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval_episodes", type=int, default=5)
    ap.add_argument("--baseline-label", default=CFG["locked_baseline_label"])
    ap.add_argument("--output", default="results/checkpoint_selection_report.json")
    args = ap.parse_args()

    specs = list(args.candidate)
    # 默认把锁定基线加入对比（若未显式传入）
    if not any(s.split(":", 2)[0] == "delivery_baseline" for s in specs):
        specs.insert(0, f"delivery_baseline:{CFG['locked_baseline_label']}:{CFG['locked_baseline_eval_json']}")

    cands = []
    for spec in specs:
        parts = spec.split(":", 2)
        label = parts[0]
        ckpt = parts[1] if len(parts) > 1 else ""
        ejson = parts[2] if len(parts) > 2 else ""
        if (not ejson or not os.path.exists(ejson)):
            if args.eval_missing and ckpt:
                ejson = ejson or f"results/_autoeval_{label.replace('/', '_')}.json"
                _run_canonical_eval(ckpt, ejson, args.device, args.eval_episodes)
            else:
                print(f"  [skip] {label}: 无 eval_json 且未开 --eval-missing")
                continue
        mx = _load_canonical_metrics(ejson)
        cands.append({"label": label, "ckpt": ckpt, "eval_json": ejson, "mx": mx})

    # 锁定基线指标（替换条件用）
    baseline = next((c for c in cands if c["label"] == "delivery_baseline"), None)
    baseline_mx = baseline["mx"] if baseline else None

    # 门控
    for c in cands:
        mx = c["mx"]
        c["safety_ok"], c["safety_reason"] = _safety_floor(mx)
        c["utility_ok"], c["utility_reason"] = _utility_floor(mx)
        c["collapse"], c["collapse_reason"] = _conservative_collapse(mx)
        c["eligible"] = bool(c["safety_ok"] and c["utility_ok"] and not c["collapse"])
        c["score"] = float("nan")

    eligible = [c for c in cands if c["eligible"]]
    _compute_scores(eligible)

    # 选 best：仅在 eligible 中按 score 取最高；空则保留基线
    best = None
    if eligible:
        best = max(eligible, key=lambda c: c["score"])
    for c in cands:
        c["is_best"] = bool(best is not None and c is best)

    # 替换裁决（相对锁定基线）
    replace_decision = "保留交付基线（无合格替换者）"
    if best is not None and best["label"] != "delivery_baseline":
        ok, why = _passes_replacement(best["mx"], baseline_mx)
        replace_decision = (f"替换为 {best['label']}" if ok
                            else f"保留交付基线（{best['label']} 未过替换门: {why}）")
    elif best is not None and best["label"] == "delivery_baseline":
        replace_decision = "保留交付基线（其本身即最高分）"

    # ── 输出对比表 ──
    cols = ["label", "ep_safe", "worst", "surv", "crash", "window",
            "rf_dl", "raw_eq_dl", "delivered", "raw_cov", "val_real", "rf_prod/dl",
            "interv", "proj",
            "safety_floor", "utility_floor", "collapse", "score", "best"]
    print("\n" + "=" * 150)
    hdr = "{:<26}{:>8}{:>7}{:>6}{:>6}{:>8}{:>10}{:>11}{:>11}{:>8}{:>9}{:>10}{:>8}{:>7}{:>8}{:>8}{:>11}{:>8}{:>6}".format(*cols)
    print(hdr)
    print("-" * 150)
    for c in cands:
        mx = c["mx"]
        row = "{:<26}{:>8.3f}{:>7.2f}{:>6.2f}{:>6.1f}{:>8.3f}{:>10.0f}{:>11.0f}{:>11.0f}{:>8.2f}{:>9.2f}{:>10.2f}{:>8.3f}{:>7.3f}{:>8}{:>8}{:>11}{:>8}{:>6}".format(
            c["label"][:26],
            mx["episode_safety_rate"], mx["worst_seed_episode_safety_rate"],
            mx["survival_rate"], mx["crash_count"], mx["comm_window_utilization"],
            mx["rf_downlinked_mb"], mx["raw_equivalent_delivered_mb"],
            mx["delivered_value"], mx["raw_equivalent_delivery_coverage"],
            mx["value_realization_ratio"], mx["rf_product_proc_downlink_ratio"],
            mx["intervention_rate"], mx["projection_rate"],
            "PASS" if c["safety_ok"] else "FAIL",
            "PASS" if c["utility_ok"] else "FAIL",
            "YES" if c["collapse"] else "no",
            ("%.3f" % c["score"]) if c["score"] == c["score"] else "-",
            "***" if c["is_best"] else "",
        )
        print(row)
    print("=" * 150)
    for c in cands:
        if not c["safety_ok"]:
            print(f"  [{c['label']}] safety FAIL: {c['safety_reason']}")
        if not c["utility_ok"]:
            print(f"  [{c['label']}] utility FAIL: {c['utility_reason']}")
        if c["collapse"]:
            print(f"  [{c['label']}] {c['collapse_reason']}")
    print(f"\n>>> 裁决: {replace_decision}")
    if baseline_mx:
        print(
            f"    锁定交付基线 = {args.baseline_label} "
            f"(raw_eq_dl={baseline_mx['raw_equivalent_delivered_mb']:.0f}, "
            f"value_real={baseline_mx['value_realization_ratio']:.2f})"
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "candidates": [{k: c[k] for k in ("label", "ckpt", "eval_json", "mx",
                                              "safety_ok", "utility_ok", "collapse",
                                              "eligible", "score", "is_best")} for c in cands],
            "replace_decision": replace_decision,
            "baseline_metrics": baseline_mx,
            "config": CFG,
        }, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved: {args.output}")


if __name__ == "__main__":
    main()

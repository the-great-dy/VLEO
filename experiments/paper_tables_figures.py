"""论文结果表格 + 图生成（读 paper_compare.py 的 JSON 产出）。

生成：
  - 主对比表（main comparison）   : tables/main_comparison.{csv,tex}
  - 消融表（ablation）            : tables/ablation.{csv,tex}
  - 场景泛化表（scenario）        : tables/scenario_generalization.{csv,tex}
  - safety-throughput trade-off 图: figures/fig_safety_throughput.png
  - proc/dl vs delivered 图        : figures/fig_procdl_delivered.png
  - same-shell baseline 对比表     : tables/same_shell_comparison.{csv,tex}

[Phase 1 eval 口径规则]
本脚本ONLY允许读取 paper_compare.py 产出的 final evaluation JSON（含 eval_config 字段）。
禁止混用训练过程中 periodic eval（单 seed/少 episode）数据。
如 JSON 中缺 eval_config 字段则视为旧格式，标注警告但仍可读。

[指标语义]
  delivered_value  = 地面实际接收到的 deadline-aware VoI（论文主指标）
  downlink         = 实际 RF 下传的压缩后 product MB
  processed        = 星上实际处理 raw data MB（含未下传部分）
  proc_dl          = processed/downlinked 聚合比率（>3.0 = 处理浪费，n/a 若 downlink 退化）
  comm_window_util = 通信窗口利用率
  episode_safety   = 每个 episode 满足全部安全约束的比率（=1.0 严格安全）
  survival_rate    = 无 crash 的 episode 占比
  crash_count      = 所有 seed×episode crash 次数总和（=0 强安全）

用法：
  python experiments/paper_tables_figures.py \
      --compare results/paper_compare_nominal.json \
      --ablation results/paper_ablation_nominal.json \
      --scenarios results/paper_compare_nominal.json,results/paper_compare_sparse_comm.json,...
"""
import sys, os, json, argparse
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _ROOT not in sys.path:
    sys.path.append(_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TABLES_DIR = os.path.join(_ROOT, "tables")
FIGS_DIR = os.path.join(_ROOT, "figures")

# 表列：(json_key, 表头, 小数位)
COLS = [
    ("episode_safety", "EpSafe", 3),
    ("worst_seed_safety", "WorstSeed", 3),
    ("survival_rate", "Surv", 3),
    ("crash_count", "Crash", 0),
    ("comm_window_util", "WinUtil", 3),
    ("downlink", "Downlink", 0),
    ("delivered_value", "Delivered", 0),
    ("proc_dl", "Proc/DL", 2),
    ("expired_value_rate", "Expired", 3),
    ("high_value_delivery", "HiDel", 3),
    ("intervention", "Interv", 3),
    ("runtime_ms_per_step", "RT(ms)", 2),
]


def _load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    return blob.get("rows", []), blob


_PROC_DL_DISPLAY_CAP = 20.0       # proc/dl 真实区间 2~7；超此值=退化(downlink≈0)，无意义
_DEGENERATE_DOWNLINK_MB = 200.0   # downlink < 此值视为退化，proc/dl 显示 n/a


def _fmt(v, dec):
    if isinstance(v, (int, float)):
        if not np.isfinite(float(v)):
            return "n/a"
        return f"{v:.{dec}f}" if dec > 0 else f"{int(round(v))}"
    return str(v)


def _render(r, k, d):
    """渲染单元格；proc/dl 在 downlink 退化或值爆炸时显示 n/a，避免天文数字污染表。"""
    v = r.get(k, 0.0)
    if k == "proc_dl":
        dl = float(r.get("downlink", 0.0))
        if dl < _DEGENERATE_DOWNLINK_MB or (isinstance(v, (int, float)) and v > _PROC_DL_DISPLAY_CAP):
            return "n/a"
    return _fmt(v, d)


def write_table(rows, title, out_base, row_label_key="config"):
    os.makedirs(TABLES_DIR, exist_ok=True)
    # CSV
    csv_path = os.path.join(TABLES_DIR, out_base + ".csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("method," + ",".join(h for _, h, _ in COLS) + "\n")
        for r in rows:
            f.write(r.get(row_label_key, "?") + "," +
                    ",".join(_render(r, k, d) for k, _, d in COLS) + "\n")
    # LaTeX
    tex_path = os.path.join(TABLES_DIR, out_base + ".tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% " + title + "\n")
        f.write("\\begin{tabular}{l" + "r" * len(COLS) + "}\n\\hline\n")
        f.write("Method & " + " & ".join(h for _, h, _ in COLS) + " \\\\\n\\hline\n")
        for r in rows:
            lbl = str(r.get(row_label_key, "?")).replace("_", "\\_")
            f.write(lbl + " & " + " & ".join(_render(r, k, d) for k, _, d in COLS) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    print(f"[table] {title} -> {csv_path} , {tex_path}")
    # 控制台预览
    print("  " + f"{'method':>32} " + " ".join(f"{h:>10}" for _, h, _ in COLS))
    for r in rows:
        print("  " + f"{str(r.get(row_label_key,'?'))[:32]:>32} " +
              " ".join(f"{_render(r, k, d):>10}" for k, _, d in COLS))


def fig_safety_throughput(rows, out_name="fig_safety_throughput.png"):
    os.makedirs(FIGS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in rows:
        x = r.get("episode_safety", 0.0)
        y = r.get("delivered_value", 0.0)
        sz = 60 + 240 * float(r.get("survival_rate", 0.0))
        ax.scatter(x, y, s=sz, alpha=0.75, edgecolors="k", linewidths=0.6)
        ax.annotate(r.get("config", "?"), (x, y), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.axvline(0.90, ls="--", c="r", lw=1, alpha=0.6, label="ep_safe=0.90 floor")
    ax.set_xlabel("Episode Safety Rate")
    ax.set_ylabel("Delivered VoI")
    ax.set_title("Safety–Throughput Trade-off (point size ∝ survival)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(FIGS_DIR, out_name)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[figure] {p}")


def fig_procdl_delivered(rows, out_name="fig_procdl_delivered.png"):
    os.makedirs(FIGS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in rows:
        x = r.get("proc_dl", 0.0)
        y = r.get("delivered_value", 0.0)
        # 退化点（downlink≈0 → proc/dl 爆炸）不画在该轴上，避免污染坐标范围
        if float(r.get("downlink", 0.0)) < _DEGENERATE_DOWNLINK_MB or x > _PROC_DL_DISPLAY_CAP:
            continue
        ep = float(r.get("episode_safety", 0.0))
        c = "tab:green" if ep >= 0.90 else "tab:red"
        ax.scatter(x, y, s=120, c=c, alpha=0.75, edgecolors="k", linewidths=0.6)
        ax.annotate(r.get("config", "?"), (x, y), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.axvline(2.0, ls="--", c="b", lw=1, alpha=0.6, label="proc/dl=2.0 target")
    ax.axvline(2.5, ls=":", c="b", lw=1, alpha=0.6, label="proc/dl=2.5")
    ax.set_xlabel("Proc/DL Ratio (lower = less wasteful processing)")
    ax.set_ylabel("Delivered VoI")
    ax.set_title("Proc/DL vs Delivered (green = ep_safe≥0.90)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(FIGS_DIR, out_name)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[figure] {p}")


def build_scenario_table(scenario_paths):
    """每个场景一行块：把各 scenario JSON 的关键方法摊平成 (scenario, method, metrics)。"""
    rows = []
    for path in scenario_paths:
        if not os.path.exists(path):
            print(f"  [skip scenario] {path} 不存在")
            continue
        rs, blob = _load_rows(path)
        sc = blob.get("scenario", "?")
        for r in rs:
            rr = dict(r)
            rr["config"] = f"{sc}:{r['config']}"
            rows.append(rr)
    return rows


def write_crash_audit_table(audit_path):
    """从 baseline_crash_audit.json 生成 tables/crash_audit.{csv,tex}。"""
    if not os.path.exists(audit_path):
        print(f"[warn] audit JSON 不存在: {audit_path}")
        return
    with open(audit_path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    rows = blob.get("rows", [])
    n_ep = rows[0].get("n_episodes", 0) if rows else 0  # 审计口径（如 20）
    os.makedirs(TABLES_DIR, exist_ok=True)
    hdr = ["Method", f"Crash(/{n_ep})", "DominantFailure", "SOC_pre", "Fuel_pre",
           "Prop_pre", "CPU_pre", "TX_pre", "Contact_pre", "SB_trig", "CG_trig"]

    def cells(r):
        ab = r.get("action_before_crash", {})
        return [r.get("method", "?"), str(r.get("crash_count", 0)),
                str(r.get("dominant_failure_type", "-")),
                f"{r.get('mean_soc_before_crash', 0):.3f}",
                f"{r.get('mean_fuel_frac_before_crash', 0):.3f}",
                f"{ab.get('prop', 0):.2f}", f"{ab.get('cpu', 0):.2f}", f"{ab.get('tx', 0):.2f}",
                f"{r.get('contact_state_before_crash', 0):.2f}",
                f"{r.get('safe_budget_triggered', 0):.2f}", f"{r.get('credit_gate_triggered', 0):.2f}"]
    csv_path = os.path.join(TABLES_DIR, "crash_audit.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(hdr) + "\n")
        for r in rows:
            f.write(",".join(cells(r)) + "\n")
    tex_path = os.path.join(TABLES_DIR, "crash_audit.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Table 1b: Baseline crash/violation fairness audit "
                f"({blob.get('episodes_per_seed','?')} ep/seed, no deployment shell)\n")
        f.write("\\begin{tabular}{l" + "r" * (len(hdr) - 1) + "}\n\\hline\n")
        f.write(" & ".join(h.replace("_", "\\_").replace("/", "/") for h in hdr) + " \\\\\n\\hline\n")
        for r in rows:
            f.write(" & ".join(c.replace("_", "\\_") for c in cells(r)) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    print(f"[table] Table 1b: Baseline crash fairness audit -> {csv_path} , {tex_path}")
    print("  " + " | ".join(hdr))
    for r in rows:
        print("  " + " | ".join(cells(r)))


# same-shell baseline config labels（与 paper_compare.py COMPARE_CONFIGS 一致）
_SAME_SHELL_LABELS = {
    "RL + SAFE_BUDGET + credit gate",
    "Heuristic + SB + CG",
    "DPP + SB + CG",
    "MPC + SB + CG",
    "Safe Greedy + SB + CG",
    "Rule-only Shell (no learned policy)",
    "Random + SB + CG",
    "Greedy Value + SB + CG",
    "EDF + SB + CG",
    "LLF + SB + CG",
    "Dec-MPC + SB + CG",
}

# Phase 1: 额外指标列（same-shell 对比专用）
SAME_SHELL_EXTRA_COLS = [
    ("expired_value_rate", "Expired", 3),
    ("high_value_delivery", "HiDel", 3),
]


def write_same_shell_table(rows, title="Same-shell baseline comparison", out_base="same_shell_comparison"):
    """筛出 same-shell 方法（sb_cg），输出对比表。"""
    same_rows = [r for r in rows if r.get("config", r.get("method", "")) in _SAME_SHELL_LABELS
                 or r.get("shell", "") in ("sb_cg",)]
    if not same_rows:
        # fallback: 包含 "SB + CG" 或 "Shell" 的行
        same_rows = [r for r in rows
                     if "SB + CG" in r.get("config", r.get("method", ""))
                     or "Shell" in r.get("config", r.get("method", ""))]
    if not same_rows:
        print("[warn] 未找到 same-shell 方法行，跳过 same_shell_comparison 表")
        return
    write_table(same_rows, title, out_base)


def _check_eval_config(blob: dict, path: str):
    """检查 JSON 是否含 eval_config 字段；旧格式打警告。"""
    if "eval_config" not in blob:
        print(f"[warn] {path} 缺少 eval_config 字段（旧格式/periodic eval），"
              f"请确认来源为 paper_compare.py --mode compare 的 final evaluation 输出。")
    else:
        ec = blob["eval_config"]
        etype = ec.get("eval_type", "unknown")
        if etype != "final_evaluation":
            print(f"[warn] {path} eval_type={etype}，应为 final_evaluation")
        else:
            print(f"[OK] {path} eval_type=final_evaluation, "
                  f"n_seeds={ec.get('n_seeds','?')}, "
                  f"eps_per_seed={ec.get('episodes_per_seed','?')}, "
                  f"checkpoint={ec.get('checkpoint','?')[:50]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", default="results/paper_compare_nominal.json")
    ap.add_argument("--ablation", default="results/paper_ablation_nominal.json")
    ap.add_argument("--scenarios", default=None, help="逗号分隔的多个 scenario JSON")
    ap.add_argument("--audit", default="results/baseline_crash_audit.json")
    args = ap.parse_args()

    if os.path.exists(args.compare):
        rows, blob = _load_rows(args.compare)
        _check_eval_config(blob, args.compare)
        write_table(rows, "Table 1: Main method comparison (canonical 20-seed)", "main_comparison")
        # Phase 3: same-shell baseline 对比表
        write_same_shell_table(rows)
        fig_safety_throughput(rows)
        fig_procdl_delivered(rows)
    else:
        print(f"[warn] compare JSON 不存在: {args.compare}")

    if os.path.exists(args.ablation):
        arows, ablob = _load_rows(args.ablation)
        _check_eval_config(ablob, args.ablation)
        write_table(arows, "Table 2: Ablation (deployment mechanisms)", "ablation")
    else:
        print(f"[warn] ablation JSON 不存在: {args.ablation}")

    if args.scenarios:
        paths = [p.strip() for p in args.scenarios.split(",") if p.strip()]
        srows = build_scenario_table(paths)
        if srows:
            write_table(srows, "Table 3: Scenario generalization", "scenario_generalization")

    write_crash_audit_table(args.audit)

    print("\n[done] tables -> tables/ , figures -> figures/")


if __name__ == "__main__":
    main()

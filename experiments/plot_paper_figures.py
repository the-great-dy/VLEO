"""
论文图表生成入口。
论文图表生成器：读取实验结果，生成发表质量的图表

生成以下图表（对应论文各 Section）：
  fig1_comparison_bar.png    — 多基线对比柱状图（Table I 可视化）
  fig2_ablation_bar.png      — 消融实验柱状图
  fig3_robustness_line.png   — 鲁棒性测试折线图（各扰动维度）

兼容性：
  1. 支持论文主消融标签 (A_Full ... H_No_BC)
  2. 读取当前 compare_all / ablation / robustness 输出结构

运行方式：
    python experiments/plot_paper_figures.py
    python experiments/plot_paper_figures.py --results_dir results/
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    # 降低导入劫持风险：不要把项目路径插到最高优先级
    sys.path.append(_PROJECT_ROOT)

import numpy as np
import json
import glob
import argparse
from datetime import datetime

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    def _setup_font():
        candidates = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        from matplotlib import font_manager
        available = {f.name for f in font_manager.fontManager.ttflist}
        for font in candidates:
            if font in available:
                matplotlib.rcParams["font.sans-serif"] = [font]
                matplotlib.rcParams["axes.unicode_minus"] = False
                break
    _setup_font()

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "font.size": 10,
    })
    MPL_OK = True
except ImportError:
    MPL_OK = False


# ── 颜色配置────────────────────────────────────────────────────
COLORS = {
    # compare_all 标签
    "LS-PSF CMDP (Ours)":        "#2196F3",
    "Ours":                       "#2196F3",
    "SAC w/o Safety":             "#D62728",
    "SAC-Lagrangian":             "#9467BD",
    "SAC + PSF":                  "#8BC34A",
    "SAC + Lyapunov":             "#00BCD4",
    "MPC":                        "#4CAF50",
    "Robust MPC":                 "#2E7D32",
    "DPP":                        "#607D8B",
    "Greedy Value":               "#00BCD4",
    "EDF":                        "#795548",
    "LLF":                        "#6D4C41",
    "启发式":                      "#FF9800",
    "Static Rule":                "#F44336",
    # 新消融标签
    "A_Full":          "#2196F3",
    "B_Throughput_Objective": "#8C564B",
    "C_No_CMDP":      "#F44336",
    "D_No_Adaptive_Dual": "#9C27B0",
    "E_No_PSF":       "#AB47BC",
    "F_No_Lyapunov":  "#4CAF50",
    "G_MLP_Backbone": "#607D8B",
    "H_No_BC":        "#FF9800",
}

DISPLAY_NAMES = {
    "LS-PSF CMDP (Ours)": "Ours",
    "Ours": "Ours",
    "MPC": "MPC",
    "Robust MPC": "Robust MPC",
    "DPP": "DPP",
    "Greedy Value": "Greedy",
    "EDF": "EDF",
    "LLF": "LLF",
    "启发式": "Heuristic",
    "Static Rule": "Static",
    "静态阈值": "Static",
    "SAC w/o Safety": "SAC",
    "SAC-Lagrangian": "SAC-Lag",
    "SAC + PSF": "SAC+PSF",
    "SAC + Lyapunov": "SAC+Lya",
}

DISPLAY_COLORS = {
    "Ours": "#1F77B4",
    "MPC": "#2CA02C",
    "Robust MPC": "#1B7F3A",
    "DPP": "#607D8B",
    "Greedy": "#17BECF",
    "EDF": "#8C564B",
    "LLF": "#6D4C41",
    "Heuristic": "#FF9900",
    "Static": "#D62728",
    "SAC": "#D62728",
    "SAC-Lag": "#9467BD",
    "SAC+PSF": "#8BC34A",
    "SAC+Lya": "#00BCD4",
}

DISPLAY_NAMES.update({
    "DECOUPLED-Heur": "Dec-Heur",
    "DECOUPLED-MPC": "Dec-MPC",
    "Value-aware Heuristic": "Value-Heur",
    "启发式": "Heuristic",
})

DISPLAY_COLORS.update({
    "Dec-Heur": "#7F7F7F",
    "Dec-MPC": "#595959",
    "Value-Heur": "#A6761D",
})

MAIN_COMPARISON_ORDER = [
    "LS-PSF CMDP (Ours)",
    "Ours",
    "MPC",
    "DPP",
    "DECOUPLED-Heur",
    "DECOUPLED-MPC",
    "Greedy Value",
    "EDF",
    "LLF",
    "启发式",
    "Static Rule",
]

# Value-aware Heuristic is kept in the JSON for diagnostics, but it nearly
# duplicates the generic heuristic in the current formal table and crowds Fig. 1.
MAIN_FIGURE_EXCLUDED_METHODS = {"Value-aware Heuristic"}
STRICT_SAFE_THRESHOLD = 1.0 - 1e-9

METHOD_ORDER = [
    "LS-PSF CMDP (Ours)", "Ours",
    "SAC w/o Safety", "SAC-Lagrangian", "SAC + PSF", "SAC + Lyapunov",
    "MPC", "Robust MPC", "DPP", "Greedy Value", "EDF", "LLF", "启发式",
    "Static Rule", "静态阈值",
]

# 四变体标签。
ABLATION_LABELS = {
    "A_Full":          "A. Ours\n(VoI-CMDP)",
    "B_Throughput_Objective": "B. w/o\nVoI Obj.",
    "C_No_CMDP":       "C. w/o\nCMDP",
    "D_No_Adaptive_Dual": "D. w/o\nAdaptive Dual",
    "E_No_PSF":        "E. w/o\nPSF",
    "F_No_Lyapunov":   "F. w/o\nLyapunov",
    "G_MLP_Backbone":  "G. MLP\nBackbone",
    "H_No_BC":         "H. w/o\nAP-BC",
}

# 消融实验中优先使用的 key 顺序。
ABLATION_KEY_CANDIDATES = [
    [
        "A_Full", "B_Throughput_Objective", "C_No_CMDP", "D_No_Adaptive_Dual",
        "E_No_PSF", "F_No_Lyapunov", "G_MLP_Backbone", "H_No_BC",
    ],
]


def _load_latest(pattern: str) -> dict:
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        return {}
    with open(files[-1], encoding="utf-8") as f:
        return json.load(f)


def _load_latest_robustness(results_dir: str) -> dict:
    """
    优先加载最新的非 smoke 鲁棒性结果。
    若全是 smoke 或旧格式（无 __meta__），回退到最新文件。
    """
    files = sorted(
        glob.glob(os.path.join(results_dir, "robustness_*.json")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not files:
        return {}

    fallback = {}
    fallback_name = None

    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, dict) or not data:
            continue

        if fallback == {}:
            fallback = data
            fallback_name = os.path.basename(fp)

        meta = data.get("__meta__", {}) if isinstance(data.get("__meta__", {}), dict) else {}
        is_smoke = bool(meta.get("is_smoke", False))
        if not is_smoke:
            print(f"  [robustness] 使用文件: {os.path.basename(fp)}")
            return data

    if fallback:
        if fallback_name:
            print(f"  [robustness] 未找到非 smoke 文件，使用: {fallback_name}")
        return fallback
    return {}


def _normalize_ablation_results(results: dict) -> dict:
    """
    兼容多种 ablation 输出结构：
      1) 顶层直接是变体 key
      2) {"independent_models": {...}}
      3) {"stress_test": {scenario: {variant: metrics}}}
    对 stress_test 结构，按场景做简单平均后返回四变体指标。
    """
    # 消融结果可能来自旧脚本、新脚本或压力测试结构，这里统一整理成“变体 -> 指标”。
    if not isinstance(results, dict) or not results:
        return {}

    # 情况1：顶层直接是变体
    known_keys = set().union(*(set(keys) for keys in ABLATION_KEY_CANDIDATES))
    if any(k in results for k in known_keys):
        return results

    # 情况2：独立模型结构
    if isinstance(results.get("independent_models"), dict):
        return results["independent_models"]

    # 情况3：压力测试结构，按场景平均
    stress = results.get("stress_test")
    if not isinstance(stress, dict) or not stress:
        return {}

    variants = [
        "A_Full", "B_Throughput_Objective", "C_No_CMDP", "D_No_Adaptive_Dual",
        "E_No_PSF", "F_No_Lyapunov", "G_MLP_Backbone", "H_No_BC",
    ]
    merged = {}

    for v in variants:
        metrics_list = []
        for _, scenario_result in stress.items():
            if isinstance(scenario_result, dict) and v in scenario_result:
                metrics_list.append(scenario_result[v])

        if not metrics_list:
            continue

        keys = set().union(*(m.keys() for m in metrics_list))
        merged[v] = {}
        for key in keys:
            vals = [m[key] for m in metrics_list if isinstance(m.get(key), (int, float))]
            if vals:
                merged[v][key] = float(np.mean(vals))

    return merged


def _pick_metric(item: dict, keys, default=0.0):
    for k in keys:
        v = item.get(k, None)
        if isinstance(v, (int, float)):
            return float(v)
    return float(default)


def _save_figure(fig, save_path: str):
    """Save both high-resolution PNG and editable PDF for paper use."""
    root, _ = os.path.splitext(save_path)
    fig.savefig(root + ".png", bbox_inches="tight", dpi=300)
    fig.savefig(root + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _ordered_methods(results: dict) -> list[str]:
    ordered = [m for m in METHOD_ORDER if m in results]
    ordered.extend([m for m in results if m not in ordered])
    return ordered


def _display_name(method: str) -> str:
    return DISPLAY_NAMES.get(method, method)


def _display_color(method: str) -> str:
    return DISPLAY_COLORS.get(_display_name(method), "#777777")


def _normalize_compare_results(results: dict) -> dict:
    """兼容 compare_all 的输出结构，只返回可直接横向比较的 结果。"""
    if not isinstance(results, dict) or not results:
        return {}
    comparable_results = results.get("comparable_results")
    if isinstance(comparable_results, dict) and comparable_results:
        return comparable_results
    return results


def _main_comparison_methods(results: dict) -> list[str]:
    """Return the compact, paper-facing method order for Fig. 1."""
    ordered = [
        method for method in MAIN_COMPARISON_ORDER
        if method in results and method not in MAIN_FIGURE_EXCLUDED_METHODS
    ]
    ordered.extend(
        method for method in _ordered_methods(results)
        if method not in ordered and method not in MAIN_FIGURE_EXCLUDED_METHODS
    )
    return ordered


def _strict_safe_mask(episode_safe, survival):
    """Strict mission safety: no failed episode and no survival loss."""
    return (
        np.asarray(episode_safe, dtype=float) >= STRICT_SAFE_THRESHOLD
    ) & (
        np.asarray(survival, dtype=float) >= STRICT_SAFE_THRESHOLD
    )


def _style_safety_bars(bars, strict_safe):
    """Fade and hatch bars whose methods do not satisfy strict episode safety."""
    for bar, is_safe in zip(bars, strict_safe):
        if not bool(is_safe):
            bar.set_alpha(0.42)
            bar.set_hatch("//")
            bar.set_edgecolor("#555555")
            bar.set_linewidth(0.7)


def _annotate_barh(ax, bars, values, *, is_rate: bool = False):
    """Add compact value labels to horizontal bars."""
    xmax = max([float(v) for v in values] + [1.0])
    pad = xmax * 0.015
    for bar, val in zip(bars, values):
        label = f"{float(val):.0%}" if is_rate else f"{float(val):.0f}"
        ax.text(
            bar.get_width() + pad,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=7,
        )


def fig1_comparison_bar(results: dict, save_path: str):
    """Main paper comparison with explicit safety constraints."""
    if not MPL_OK:
        return
    results = _normalize_compare_results(results)
    if not results:
        return

    methods = _main_comparison_methods(results)
    labels = [_display_name(m) for m in methods]
    colors = [_display_color(m) for m in methods]
    y = np.arange(len(methods))

    delivered = np.asarray(
        [_pick_metric(results[m], ["delivered_value_mean"], 0.0) for m in methods],
        dtype=float,
    )
    downlink = np.asarray(
        [_pick_metric(results[m], ["downlink_mean", "tx_mb_mean"], 0.0) for m in methods],
        dtype=float,
    )
    episode_safe = np.asarray(
        [_pick_metric(results[m], ["episode_safety_rate", "safety_rate"], 0.0) for m in methods],
        dtype=float,
    )
    survival = np.asarray(
        [_pick_metric(results[m], ["survival_rate"], 0.0) for m in methods],
        dtype=float,
    )
    prop = np.asarray(
        [_pick_metric(results[m], ["mean_prop_power"], 0.0) for m in methods],
        dtype=float,
    )
    cpu = np.asarray(
        [_pick_metric(results[m], ["mean_cpu_power"], 0.0) for m in methods],
        dtype=float,
    )
    tx = np.asarray(
        [_pick_metric(results[m], ["mean_tx_power"], 0.0) for m in methods],
        dtype=float,
    )
    strict_safe = _strict_safe_mask(episode_safe, survival)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.8))
    fig.suptitle("Safety-Constrained Performance Comparison", fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    bars = ax.barh(y, delivered, color=colors, height=0.62)
    _style_safety_bars(bars, strict_safe)
    _annotate_barh(ax, bars, delivered)
    ax.set_title("Delivered VoI")
    ax.set_xlabel("Value")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, max(delivered.max() * 1.18, 1.0))
    ax.invert_yaxis()

    ax = axes[0, 1]
    x_min = max(0.0, min(episode_safe.min() - 0.04, 0.86))
    ax.axvspan(STRICT_SAFE_THRESHOLD, 1.005, color="#E8F5E9", alpha=0.9, zorder=0)
    ax.axvline(STRICT_SAFE_THRESHOLD, ls="--", lw=1.0, color="#2E7D32", alpha=0.9)
    max_downlink = max(float(downlink.max()), 1.0)
    sizes = 70.0 + 230.0 * downlink / max_downlink
    label_offsets = {
        "MPC": (5, -8),
        "DPP": (5, 5),
        "Dec-Heur": (5, -9),
        "Dec-MPC": (5, 3),
        "Greedy": (5, 8),
        "EDF": (5, 4),
        "LLF": (5, 0),
    }
    for i, (label, color) in enumerate(zip(labels, colors)):
        marker = "*" if methods[i] in ("LS-PSF CMDP (Ours)", "Ours") else "o"
        edge = "#111111" if marker == "*" else "white"
        ax.scatter(
            episode_safe[i],
            delivered[i],
            s=sizes[i],
            color=color,
            marker=marker,
            edgecolors=edge,
            linewidths=0.7,
            alpha=0.92,
            zorder=3,
        )
        dx, dy = label_offsets.get(label, (4, 3))
        ax.annotate(label, (episode_safe[i], delivered[i]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=7)
    ax.set_title("Safety-Value Pareto View")
    ax.set_xlabel("Episode Safety Rate")
    ax.set_ylabel("Delivered VoI")
    ax.set_xlim(x_min, 1.005)
    ax.set_ylim(0.0, max(delivered.max() * 1.16, 1.0))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.0%}"))

    ax = axes[1, 0]
    height = 0.34
    ep_bars = ax.barh(y - height / 2, episode_safe, height=height,
                      color="#4C78A8", label="Episode safe")
    surv_bars = ax.barh(y + height / 2, survival, height=height,
                        color="#59A14F", label="Survival")
    ax.axvline(STRICT_SAFE_THRESHOLD, ls="--", lw=1.0, color="#2E7D32", alpha=0.9)
    ax.set_title("Safety and Survival")
    ax.set_xlabel("Rate")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 1.08)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.0%}"))
    ax.invert_yaxis()
    ax.legend(frameon=True, fontsize=7, loc="lower left",
              facecolor="white", framealpha=0.85, edgecolor="none")
    _annotate_barh(ax, ep_bars, episode_safe, is_rate=True)
    _annotate_barh(ax, surv_bars, survival, is_rate=True)

    ax = axes[1, 1]
    ax.barh(y, prop, color="#9467BD", height=0.62, label="Propulsion")
    ax.barh(y, cpu, left=prop, color="#2CA02C", height=0.62, label="CPU")
    ax.barh(y, tx, left=prop + cpu, color="#FF7F0E", height=0.62, label="Downlink")
    ax.set_title("Mean Adjustable Power")
    ax.set_xlabel("Power (W)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.legend(frameon=True, fontsize=7, loc="lower left",
              facecolor="white", framealpha=0.85, edgecolor="none")

    fig.text(
        0.5,
        0.012,
        "Hatched bars indicate methods with episode-safety or survival below 100%; "
        "point size in the Pareto view is proportional to downlink MB.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    _save_figure(fig, save_path)
    print(f"  [Fig.1] saved: {os.path.splitext(save_path)[0]}.png/.pdf")


def fig2_ablation_bar(results: dict, save_path: str):
    """Safety-module ablation figure."""
    if not results or not MPL_OK:
        return

    results = _normalize_ablation_results(results)
    if not results:
        print("  [跳过图2] 消融结果结构无法解析")
        return

    # 自动检测消融结果的 key 格式（新/旧兼容）。
    keys = None
    for candidate_keys in ABLATION_KEY_CANDIDATES:
        found = [k for k in candidate_keys if k in results]
        if len(found) >= 2:  # 至少有 2 个变体
            keys = found
            break

    if keys is None:
        print("  [跳过图2] 消融结果中未找到已知变体 key")
        return

    labels = [ABLATION_LABELS.get(key, key) for key in keys]
    colors = [COLORS.get(k, "#888888") for k in keys]
    x = np.arange(len(keys))

    metric_specs = [
        ("Constraint Satisfaction", ["Constraint Satisfaction Rate", "overall_safe_rate", "safety_rate"], True),
        ("Delivered VoI", ["Delivered VoI", "delivered_value_mean", "delivered_value_total"], False),
        ("RF Downlinked", ["RF Downlinked MB", "rf_downlinked_mb_mean", "Downlink MB", "downlink_mean", "tx_mb_total"], False),
        ("Raw-equiv Delivered", ["Raw-equivalent Delivered MB", "raw_equivalent_delivered_mb_mean", "raw_equivalent_delivered_mb"], False),
        ("Raw-equiv Coverage", ["Raw-equivalent Delivery Coverage", "raw_equivalent_delivery_coverage_mean", "episode_raw_equivalent_delivery_coverage"], True),
        ("Intervention", ["Intervention Rate", "safety_intervention_rate", "psf_filter_rate"], True),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(22.5, 4.5))
    fig.suptitle("Ablation Study of LS-PSF CMDP Components", fontsize=13, fontweight="bold")

    for ax, (title, value_keys, is_rate) in zip(axes, metric_specs):
        vals = [_pick_metric(results[k], value_keys, 0.0) for k in keys]
        bars = ax.bar(x, vals, color=colors, alpha=0.9, width=0.58)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        if is_rate:
            ax.set_ylim(0.0, 1.05)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        else:
            ax.set_ylim(0.0, max(vals + [1.0]) * 1.25)
        for bar, val in zip(bars, vals):
            label = f"{val:.0%}" if is_rate else f"{val:.0f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.025 if is_rate else max(vals + [1.0]) * 0.03),
                    label, ha="center", va="bottom", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save_figure(fig, save_path)
    print(f"  [Fig.2] saved: {os.path.splitext(save_path)[0]}.png/.pdf")


def fig3_robustness_line(results: dict, save_path: str):
    """Robustness curves using delivered value instead of high-variance reward."""
    if not results or not MPL_OK:
        return

    # 兼容带元信息的新格式
    meta = results.get("__meta__", {}) if isinstance(results.get("__meta__", {}), dict) else {}
    if "__meta__" in results:
        results = {k: v for k, v in results.items() if k != "__meta__"}
    if not results:
        return

    groups = {
        "Orbital Uncertainty: Altitude (km)":       {},
        "Orbital Uncertainty: Density Factor":      {},
        "Energy Uncertainty: Solar Efficiency (%)": {},
        "Workload Uncertainty: Arrival Factor":     {},
        "Energy Uncertainty: Battery Capacity (%)": {},
    }
    group_keys = list(groups.keys())
    prefix_map = {
        "初始高度":   0,
        "大气密度":   1,
        "太阳能效率": 2,
        "任务到达率": 3,
        "电池容量":   4,
    }

    for cond_name, cond_r in results.items():
        for prefix, gidx in prefix_map.items():
            if cond_name.startswith(prefix):
                gkey = group_keys[gidx]
                try:
                    val_str = cond_name.split()[-1]
                    val_str = val_str.replace("km", "").replace("%", "")
                    val_str = val_str.replace("×", "").replace("x", "").replace("X", "")
                    x_val = float(val_str)
                except Exception:
                    x_val = 0
                groups[gkey][x_val] = cond_r

    methods_to_plot = [
        "LS-PSF CMDP (Ours)", "MPC", "Robust MPC", "Static Rule",
    ]

    fig = plt.figure(figsize=(14, 8))
    title = "Robustness Analysis"
    if meta:
        n_ep = meta.get("n_episodes", "?")
        n_steps = meta.get("max_steps", "?")
        smoke = "smoke" if meta.get("is_smoke", False) else "full"
        title += f"\n(n_episodes={n_ep}, max_steps={n_steps}, mode={smoke})"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.28)

    axes_flat = [fig.add_subplot(gs[i // 3, i % 3])
                 for i in range(5)]

    for ax, (gname, gdata) in zip(axes_flat, groups.items()):
        if not gdata:
            ax.set_title(gname)
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", color="gray")
            continue

        x_vals = sorted(gdata.keys())
        plotted_names = set()
        for method in methods_to_plot:
            display = _display_name(method)
            if display in plotted_names:
                continue
            plotted_names.add(display)
            color = DISPLAY_COLORS.get(display, "#777777")
            y_vals = []
            for x in x_vals:
                source_name = method
                if source_name not in gdata[x]:
                    if method == "Static Rule":
                        source_name = "静态阈值"
                m = gdata[x].get(source_name, {})
                y_vals.append(_pick_metric(
                    m,
                    ["delivered_value_mean", "delivered_value_total", "downlink_mean_mb", "tx_mb_mean"],
                    np.nan,
                ))
            y_arr = np.array(y_vals, dtype=float)
            valid = ~np.isnan(y_arr)
            if valid.any():
                xv = np.array(x_vals, dtype=float)[valid]
                yv = y_arr[valid]
                ax.plot(xv, yv,
                        "o-", color=color, linewidth=2,
                        markersize=4, label=display)

        ax.set_xlabel(gname)
        ax.set_ylabel("Delivered Value")
        ax.set_title(gname)
        ax.legend(fontsize=7, frameon=False)

    if len(groups) < 6:
        fig.add_subplot(gs[1, 2]).set_visible(False)

    _save_figure(fig, save_path)
    print(f"  [Fig.3] saved: {os.path.splitext(save_path)[0]}.png/.pdf")


def run_plot_all(args):
    # 画图入口尽量自动读取最新结果，方便训练/评估结束后直接一键生成论文图。
    print("=" * 55)
    print("  论文图表生成器")
    print("=" * 55)

    os.makedirs(args.output_dir, exist_ok=True)

    if not MPL_OK:
        print("[错误] matplotlib 未安装")
        return

    compare = _load_latest(
        os.path.join(args.results_dir, "compare_all_*.json"))
    ablation = _load_latest(
        os.path.join(args.results_dir, "ablation_*.json"))
    robust = _load_latest_robustness(args.results_dir)

    if compare:
        fig1_comparison_bar(
            compare,
            os.path.join(args.output_dir, "fig1_comparison_bar.png"))
    else:
        print("  [跳过图1] 未找到 compare_all 结果，"
              "请先运行: python experiments/compare_all.py")

    if ablation:
        fig2_ablation_bar(
            ablation,
            os.path.join(args.output_dir, "fig2_ablation_bar.png"))
    else:
        print("  [跳过图2] 未找到 ablation 结果，"
              "请先运行: python experiments/ablation.py")

    if robust:
        fig3_robustness_line(
            robust,
            os.path.join(args.output_dir, "fig3_robustness_line.png"))
    else:
        print("  [跳过图3] 未找到 robustness 结果，"
              "请先运行: python experiments/robustness.py")

    print(f"\n  图表保存在: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成论文图表")
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--output_dir", default="figures/paper/")
    args = parser.parse_args()
    run_plot_all(args)

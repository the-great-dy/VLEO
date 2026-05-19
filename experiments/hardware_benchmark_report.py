"""
生成星载/嵌入式推理开销报告的入口。
星载/嵌入式硬件基准报告入口。

该脚本复用 evaluate_optimized.py 中的推理延迟、参数量和 FLOPs 估计，
额外记录硬件名称、控制周期和实时余量，方便论文中直接形成部署可行性表格。
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    # 仅脚本直跑时追加项目路径。
    sys.path.append(_PROJECT_ROOT)

import argparse
import json
import platform
from datetime import datetime

from evaluate_optimized import benchmark_onboard_feasibility
from config import TRAIN_CONFIG


def run_hardware_benchmark(args) -> dict:
    stats = benchmark_onboard_feasibility(
        args.model,
        n_calls=args.calls,
        warmup=args.warmup,
    )
    control_period_ms = float(args.control_period_s) * 1000.0
    p95 = float(stats.get("latency_ms_p95", 0.0))
    p99 = float(stats.get("latency_ms_p99", 0.0))

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "hardware_label": args.hardware_label,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "control_period_s": float(args.control_period_s),
        "control_period_ms": control_period_ms,
        "benchmark": stats,
        "realtime_margin": {
            "p95_margin_ms": float(control_period_ms - p95),
            "p99_margin_ms": float(control_period_ms - p99),
            "p95_period_fraction": float(p95 / max(control_period_ms, 1e-6)),
            "p99_period_fraction": float(p99 / max(control_period_ms, 1e-6)),
            "passes_p95_realtime": bool(p95 < control_period_ms),
            "passes_p99_realtime": bool(p99 < control_period_ms),
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[OK] hardware benchmark saved: {args.output}")
    print(f"p95={p95:.4f} ms, p99={p99:.4f} ms, period={control_period_ms:.1f} ms")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate onboard hardware benchmark report")
    parser.add_argument("--model", required=True, help="checkpoint path")
    parser.add_argument("--output", default="results/hardware_benchmark_report.json")
    parser.add_argument("--hardware_label", default="desktop_cpu",
                        help="例如 rk3588, jetson_orin_nano, zynq_ultrascale, flight_cpu")
    parser.add_argument("--calls", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=80)
    parser.add_argument("--control_period_s", type=float,
                        default=float(TRAIN_CONFIG.get("time_slot_s", 10.0)))
    run_hardware_benchmark(parser.parse_args())

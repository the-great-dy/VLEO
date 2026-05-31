"""
多随机种子训练与评估聚合入口。

Unified multi-seed experiment runner.

Modes:
  eval       - evaluate one checkpoint across multiple evaluation seeds.
  train-eval - train one independent model per seed, then evaluate each model.

This keeps paper statistics behind one entry point instead of splitting
evaluation-only and train-then-evaluate workflows into separate scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from config import TRAIN_CONFIG
from evaluate_optimized import evaluate_model, _resolve_device


METRIC_KEYS = [
    "reward_mean",
    "delivered_value_mean",
    "average_aoi_steps",
    "value_weighted_aoi_steps",
    "voi_degradation_rate",
    "voi_loss_rate",
    "deadline_success_rate",
    "value_weighted_deadline_success_rate",
    "expired_value_rate",
    "dropped_value_rate",
    "high_value_delivery_rate",
    "processed_mean_mb",
    "downlink_mean_mb",
    "global_proc_downlink_ratio",
    "mean_episode_proc_downlink_ratio",
    "proc_dl_ratio",
    "proc_downlink_ratio",
    "episode_proc_dl_ratio",
    "processed_queue_final_utilization",
    "tx_active_in_contact_ratio",
    "survival_rate",
    "crash_count",
    "episode_safety_rate",
    "step_safety_rate",
    "overall_safe_rate",
    "thermal_safe_rate",
    "normal_state_rate",
    "warning_state_rate",
    "unsafe_state_rate",
    "failure_state_rate",
    "safety_rate",
    "violation_percentage",
    "lyapunov_projection_rate",
    "psf_filter_rate",
    "intervention_rate",
    "mean_action_modification",
    "mean_prop_power",
    "mean_cpu_power",
    "mean_tx_power",
    "energy_efficiency",
    "comm_window_utilization",
]


def parse_seeds(seed_text: str) -> list[int]:
    seeds = []
    for part in str(seed_text).split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("seeds 不能为空，例如 --seeds 42,43,44,45,46")
    return seeds


def default_eval_seeds() -> list[int]:
    configured = TRAIN_CONFIG.get("eval_seeds", None)
    if configured is not None:
        if isinstance(configured, str):
            return parse_seeds(configured)
        seeds = [int(s) for s in configured]
        if seeds:
            return seeds

    base_seed = int(TRAIN_CONFIG.get("seed", 42))
    return [base_seed + i for i in range(5)]


def resolve_seeds(seed_text: str | None) -> tuple[list[int], str]:
    if seed_text is None:
        return default_eval_seeds(), "config.TRAIN_CONFIG.eval_seeds"
    return parse_seeds(seed_text), "command_line"


def metric_values(rows: list[dict], key: str) -> np.ndarray:
    # 只统计真正存在该指标的 seed；缺失的 seed 不再静默填 0.0 稀释均值/方差。
    return np.asarray(
        [float(row[key]) for row in rows if key in row and row[key] is not None],
        dtype=np.float64,
    )


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"n_seeds": int(n)}
    for key in METRIC_KEYS:
        vals = metric_values(rows, key)
        m = int(vals.size)  # 实际有该指标的 seed 数（可能 < n_seeds）
        std = float(np.std(vals, ddof=1)) if m > 1 else 0.0
        out[key] = {
            "mean": float(np.mean(vals)) if m else 0.0,
            "std": std,
            "ci95": float(1.96 * std / np.sqrt(m)) if m else 0.0,
            "min": float(np.min(vals)) if m else 0.0,
            "max": float(np.max(vals)) if m else 0.0,
            "n": m,
        }
    return out


def paired_stats(model_rows: list[dict], baseline_rows: list[dict]) -> dict:
    n = min(len(model_rows), len(baseline_rows))
    out = {"n_pairs": int(n)}
    if n == 0:
        return out

    for key in METRIC_KEYS:
        model_vals = metric_values(model_rows[:n], key)
        base_vals = metric_values(baseline_rows[:n], key)
        # 过滤缺失项后两侧长度可能不齐，按较短者对齐，避免广播错误。
        m = min(model_vals.size, base_vals.size)
        if m == 0:
            continue
        model_vals = model_vals[:m]
        base_vals = base_vals[:m]
        delta = model_vals - base_vals
        delta_std = float(np.std(delta, ddof=1)) if m > 1 else 0.0
        base_mean = float(np.mean(base_vals))
        out[key] = {
            "model_minus_baseline_mean": float(np.mean(delta)),
            "model_minus_baseline_std": delta_std,
            "model_minus_baseline_ci95": float(1.96 * delta_std / np.sqrt(max(n, 1))),
            "relative_change_pct": float(np.mean(delta) / (abs(base_mean) + 1e-6) * 100.0),
            "cohens_d_paired": float(np.mean(delta) / (delta_std + 1e-12)),
        }
    return out


def run_multi_seed_eval(args) -> dict:
    device = _resolve_device(args.device)
    seeds, seed_source = resolve_seeds(args.seeds)
    force_enable_lyapunov = False if args.no_lyapunov else None
    force_use_psf = False if args.no_psf else None

    model_rows = []
    baseline_rows = []
    for seed in seeds:
        print(f"[model] seed={seed}")
        model_rows.append(evaluate_model(
            args.model,
            n_episodes=args.eval_episodes,
            device=device,
            force_enable_lyapunov=force_enable_lyapunov,
            force_use_psf=force_use_psf,
            eval_seed=seed,
        ))

        if args.baseline:
            print(f"[baseline] seed={seed}")
            baseline_rows.append(evaluate_model(
                args.baseline,
                n_episodes=args.eval_episodes,
                device=device,
                force_enable_lyapunov=force_enable_lyapunov,
                force_use_psf=force_use_psf,
                eval_seed=seed,
            ))

    report = {
        "__meta__": {
            "timestamp": datetime.now().isoformat(),
            "mode": "eval",
            "model": args.model,
            "baseline": args.baseline,
            "device": device,
            "eval_episodes": int(args.eval_episodes),
            "seeds": seeds,
            "seed_source": seed_source,
        },
        "model_by_seed": model_rows,
        "model_summary": aggregate(model_rows),
    }
    if baseline_rows:
        report["baseline_by_seed"] = baseline_rows
        report["baseline_summary"] = aggregate(baseline_rows)
        report["paired_model_vs_baseline"] = paired_stats(model_rows, baseline_rows)

    _write_report(report, args.output)
    _print_eval_summary(report, args.output)
    return report


def run_subprocess(cmd: list[str], dry_run: bool = False) -> int:
    print(" ".join(cmd))
    if dry_run:
        return 0
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return int(subprocess.run(
        cmd,
        cwd=str(_PROJECT_ROOT),
        env=env,
        check=False,
    ).returncode)


def run_multi_seed_train_eval(args) -> dict:
    device = _resolve_device(args.device)
    seeds, seed_source = resolve_seeds(args.seeds)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    train_records = []
    for seed in seeds:
        seed_dir = output_root / f"seed_{seed}"
        ckpt_dir = seed_dir / "checkpoints"
        log_dir = seed_dir / "logs"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "train.py",
            "--device", device,
            "--seed", str(seed),
            "--total_steps", str(args.total_steps),
            "--eval_freq", str(args.eval_freq),
            "--eval_episodes", str(args.train_eval_episodes),
            "--save_freq", str(args.save_freq),
            "--n_envs", str(args.n_envs),
            "--env_backend", args.env_backend,
            "--checkpoint_dir", str(ckpt_dir),
            "--log_dir", str(log_dir),
        ]
        for name in ("warmup_steps", "update_freq", "update_actor_freq"):
            value = getattr(args, name)
            if value is not None:
                cmd.extend([f"--{name}", str(value)])

        rc = run_subprocess(cmd, dry_run=args.dry_run)
        best_ckpt = ckpt_dir / "best_optimized.pt"
        latest_ckpt = ckpt_dir / "latest.pt"
        chosen = best_ckpt if best_ckpt.exists() else latest_ckpt
        train_records.append({
            "seed": seed,
            "command": cmd,
            "returncode": rc,
            "checkpoint": str(chosen),
        })
        if rc != 0:
            if args.continue_on_error:
                continue
            raise RuntimeError(f"seed={seed} 训练失败，returncode={rc}")
        if args.dry_run:
            continue
        if not chosen.exists():
            raise FileNotFoundError(f"seed={seed} 未找到 checkpoint: {chosen}")

        eval_seed = seed + int(args.eval_seed_offset)
        stats = evaluate_model(
            str(chosen),
            n_episodes=args.eval_episodes,
            device=device,
            eval_seed=eval_seed,
        )
        stats["training_seed"] = int(seed)
        stats["eval_seed"] = int(eval_seed)
        stats["checkpoint"] = str(chosen)
        rows.append(stats)

    report = {
        "__meta__": {
            "timestamp": datetime.now().isoformat(),
            "mode": "train-eval",
            "seed_source": seed_source,
            "train_seeds": seeds,
            "total_steps_per_seed": int(args.total_steps),
            "eval_episodes_per_model": int(args.eval_episodes),
            "device": device,
            "dry_run": bool(args.dry_run),
        },
        "train_records": train_records,
        "model_by_training_seed": rows,
        "model_summary": aggregate(rows) if rows else {},
    }
    _write_report(report, args.output)
    print(f"[OK] saved: {args.output}")
    return report


def _write_report(report: dict, output: str) -> None:
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def _print_eval_summary(report: dict, output: str) -> None:
    downlink = report["model_summary"]["downlink_mean_mb"]
    safety = report["model_summary"]["safety_rate"]
    print(f"\n[OK] saved: {output}")
    print("model downlink_mean_mb: "
          f"{downlink['mean']:.2f} ± {downlink['ci95']:.2f} (95% CI)")
    print("model safety_rate: "
          f"{safety['mean']:.3f} ± {safety['ci95']:.3f} (95% CI)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified multi-seed evaluation and train-evaluation")
    parser.add_argument("--mode", choices=["eval", "train-eval"], default="eval")
    parser.add_argument("--model", default=None, help="eval mode 待评估 checkpoint")
    parser.add_argument("--baseline", default=None, help="eval mode 可选 baseline checkpoint")
    parser.add_argument("--seeds", default=None,
                        help="逗号分隔随机种子；不填则读取 TRAIN_CONFIG.eval_seeds")
    parser.add_argument("--eval_episodes", type=int,
                        default=int(TRAIN_CONFIG.get("eval_episodes", 30)))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no_lyapunov", action="store_true",
                        help="eval mode 关闭 Lyapunov 层")
    parser.add_argument("--no_psf", action="store_true",
                        help="eval mode 关闭 PSF")
    parser.add_argument("--output", default=None)

    parser.add_argument("--total_steps", type=int,
                        default=int(TRAIN_CONFIG.get("total_steps", 1500000)))
    parser.add_argument("--eval_freq", type=int,
                        default=int(TRAIN_CONFIG.get("eval_freq", 20000)))
    parser.add_argument("--train_eval_episodes", type=int,
                        default=int(TRAIN_CONFIG.get("eval_episodes", 30)))
    parser.add_argument("--save_freq", type=int,
                        default=int(TRAIN_CONFIG.get("save_freq", 50000)))
    parser.add_argument("--n_envs", type=int,
                        default=int(TRAIN_CONFIG.get("n_envs", 1)))
    parser.add_argument("--env_backend", choices=["auto", "serial", "subproc"],
                        default=TRAIN_CONFIG.get("env_backend", "auto"))
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--update_freq", type=int, default=None)
    parser.add_argument("--update_actor_freq", type=int, default=None)
    parser.add_argument("--eval_seed_offset", type=int, default=10000)
    parser.add_argument("--output_dir", default="checkpoints_multiseed/")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    return parser


def main() -> dict:
    parser = build_parser()
    args = parser.parse_args()
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "eval" if args.mode == "eval" else "train_eval"
        args.output = f"results/multi_seed_{suffix}_{ts}.json"
    if args.mode == "eval":
        if not args.model:
            parser.error("--mode eval 需要 --model")
        return run_multi_seed_eval(args)
    return run_multi_seed_train_eval(args)


if __name__ == "__main__":
    main()

"""Objective-sensitivity experiments for the paper CMDP.

The sweep only changes the two mission reward weights used in the paper:
delivered value and on-time delivered value. Safety trade-offs are evaluated
from CMDP constraint metrics, not by adding safety penalties to reward.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from config import REWARD_CONFIG, TRAIN_CONFIG
from evaluate_optimized import _resolve_device, evaluate_model
from train import train as train_main


OBJECTIVE_PROFILES = {
    "balanced": {
        "w_delivered_value": 2.0,
        "w_deadline_success": 0.3,
    },
    "value_dominant": {
        "w_delivered_value": 3.0,
        "w_deadline_success": 0.1,
    },
    "deadline_sensitive": {
        "w_delivered_value": 1.5,
        "w_deadline_success": 0.7,
    },
}


def _apply_profile(profile: dict) -> dict:
    old = copy.deepcopy(REWARD_CONFIG)
    # 只覆盖被扫描的两个权重，保留完整的论文 reward 配置（过期罚分、class 权重等）。
    # 旧实现在这里 REWARD_CONFIG.clear() 会把其余 ~50 个调好的权重全部清零，使每个
    # Pareto 点都用残缺 reward（如 w_expired_penalty 退回 0）训练 —— 既不是论文方法，
    # 安全/价值 trade-off 轴也几乎不动，头号 Pareto 图因此失真。
    REWARD_CONFIG.update(profile)
    return old


def _restore_reward_config(old: dict) -> None:
    REWARD_CONFIG.clear()
    REWARD_CONFIG.update(old)


def _axis_value(stats: dict, key: str):
    return float(stats.get(key, 0.0)) if stats else None


def _aggregate_axis(vals: list) -> dict:
    """跨 seed 聚合单个 Pareto 轴：mean ± std ± 95%CI。"""
    arr = np.asarray([v for v in vals if v is not None], dtype=np.float64)
    m = int(arr.size)
    if m == 0:
        return {"mean": None, "std": None, "ci95": None, "min": None, "max": None, "n": 0}
    std = float(np.std(arr, ddof=1)) if m > 1 else 0.0
    return {
        "mean": float(arr.mean()),
        "std": std,
        "ci95": float(1.96 * std / np.sqrt(m)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": m,
    }


_PARETO_AXIS_KEYS = {
    "x_safety_violation_pct": "violation_percentage",
    "y_downlink_mb": "downlink_mean_mb",
    "y_delivered_value": "delivered_value_mean",
    "energy_efficiency": "energy_efficiency",
}


def run_pareto_frontier(args) -> dict:
    device = _resolve_device(args.device)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    n_seeds = max(1, int(getattr(args, "seeds_per_profile", 1)))
    results = {}

    for idx, (name, weights) in enumerate(OBJECTIVE_PROFILES.items()):
        seed_runs = []
        for s in range(n_seeds):
            run_seed = int(args.seed) + idx * 100 + s
            seed_root = out_root / name / f"seed_{run_seed}"
            ckpt_dir = seed_root / "checkpoints"
            log_dir = seed_root / "logs"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            old_config = _apply_profile(weights)
            try:
                if args.train:
                    train_args = SimpleNamespace(
                        device=device,
                        seed=run_seed,
                        total_steps=int(args.total_steps),
                        checkpoint_dir=str(ckpt_dir),
                        log_dir=str(log_dir),
                        eval_freq=int(args.eval_freq),
                        eval_episodes=int(args.train_eval_episodes),
                        save_freq=int(args.save_freq),
                        keep_step_checkpoints=False,
                        n_envs=int(args.n_envs),
                        env_backend=args.env_backend,
                        warmup_steps=args.warmup_steps,
                        update_freq=args.update_freq,
                        update_actor_freq=args.update_actor_freq,
                        resume_path=None,
                        no_lyapunov=False,
                        no_psf=False,
                        constraint_variant="ours",
                        tensorboard=False,
                        wandb_project=None,
                        wandb_run_name=None,
                    )
                    if args.dry_run:
                        print(
                            f"[DRY-RUN] train profile {name} seed={run_seed}, "
                            f"steps={train_args.total_steps}, checkpoint_dir={ckpt_dir}"
                        )
                    else:
                        train_main(train_args)
            finally:
                _restore_reward_config(old_config)

            ckpt = ckpt_dir / "best_optimized.pt"
            if not ckpt.exists():
                ckpt = ckpt_dir / "latest.pt"
            stats = {}
            if ckpt.exists() and not args.dry_run:
                stats = evaluate_model(
                    str(ckpt),
                    n_episodes=int(args.eval_episodes),
                    device=device,
                    eval_seed=int(args.seed) + 1000 + idx * 100 + s,
                )
            seed_runs.append({
                "seed": run_seed,
                "checkpoint": str(ckpt),
                "axes": {ax: _axis_value(stats, src) for ax, src in _PARETO_AXIS_KEYS.items()},
                "stats": stats,
            })

        # 跨 seed 聚合：头号 Pareto 图按 mean ± 95%CI 画带误差棒的点，
        # 不再用单 seed 点估计支撑“占优”结论（n_seeds=1 时 ci95=0，仍兼容旧用法）。
        results[name] = {
            "objective_weights": weights,
            "n_seeds": n_seeds,
            "seed_runs": seed_runs,
            "pareto_axes": {
                ax: _aggregate_axis([r["axes"][ax] for r in seed_runs])
                for ax in _PARETO_AXIS_KEYS
            },
        }

    report = {
        "__meta__": {
            "timestamp": datetime.now().isoformat(),
            "train": bool(args.train),
            "total_steps_per_profile": int(args.total_steps),
            "eval_episodes": int(args.eval_episodes),
            "seeds_per_profile": max(1, int(getattr(args, "seeds_per_profile", 1))),
            "profiles": list(OBJECTIVE_PROFILES.keys()),
            "sweep_type": "mission_objective_weights_only",
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OK] Pareto results saved: {args.output}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMDP objective-weight sensitivity")
    parser.add_argument(
        "--train",
        action="store_true",
        help="train each objective profile; without it only evaluates existing checkpoints",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--total_steps", type=int, default=300000)
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=int(TRAIN_CONFIG.get("eval_freq", 20000)),
    )
    parser.add_argument(
        "--train_eval_episodes",
        type=int,
        default=int(TRAIN_CONFIG.get("eval_episodes", 30)),
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=int(TRAIN_CONFIG.get("eval_episodes", 30)),
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=int(TRAIN_CONFIG.get("save_freq", 50000)),
    )
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument(
        "--env_backend",
        choices=["serial", "auto", "subproc"],
        default="serial",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--seed",
        type=int,
        default=int(TRAIN_CONFIG.get("seed", 42)) + 30000,
    )
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--update_freq", type=int, default=None)
    parser.add_argument("--update_actor_freq", type=int, default=None)
    parser.add_argument("--seeds_per_profile", type=int, default=1,
                        help="每个 objective profile 独立训练的 seed 数；论文头号 Pareto 图建议 >=3 以出误差棒")
    parser.add_argument("--output_dir", default="checkpoints_pareto/")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/pareto_frontier_{ts}.json"
    run_pareto_frontier(args)

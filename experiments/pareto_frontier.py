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
    REWARD_CONFIG.clear()
    REWARD_CONFIG.update(profile)
    return old


def _restore_reward_config(old: dict) -> None:
    REWARD_CONFIG.clear()
    REWARD_CONFIG.update(old)


def run_pareto_frontier(args) -> dict:
    device = _resolve_device(args.device)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    results = {}

    for idx, (name, weights) in enumerate(OBJECTIVE_PROFILES.items()):
        ckpt_dir = out_root / name / "checkpoints"
        log_dir = out_root / name / "logs"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        old_config = _apply_profile(weights)
        try:
            if args.train:
                train_args = SimpleNamespace(
                    device=device,
                    seed=int(args.seed) + idx,
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
                        "[DRY-RUN] train profile "
                        f"{name}: seed={train_args.seed}, steps={train_args.total_steps}, "
                        f"checkpoint_dir={train_args.checkpoint_dir}"
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
                eval_seed=int(args.seed) + 1000 + idx,
            )
        results[name] = {
            "objective_weights": weights,
            "checkpoint": str(ckpt),
            "stats": stats,
            "pareto_axes": {
                "x_safety_violation_pct": (
                    float(stats.get("violation_percentage", 0.0)) if stats else None
                ),
                "y_downlink_mb": (
                    float(stats.get("downlink_mean_mb", 0.0)) if stats else None
                ),
                "y_delivered_value": (
                    float(stats.get("delivered_value_mean", 0.0)) if stats else None
                ),
                "energy_efficiency": (
                    float(stats.get("energy_efficiency", 0.0)) if stats else None
                ),
            },
        }

    report = {
        "__meta__": {
            "timestamp": datetime.now().isoformat(),
            "train": bool(args.train),
            "total_steps_per_profile": int(args.total_steps),
            "eval_episodes": int(args.eval_episodes),
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
    parser.add_argument("--output_dir", default="checkpoints_pareto/")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/pareto_frontier_{ts}.json"
    run_pareto_frontier(args)

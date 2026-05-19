#!/usr/bin/env python
"""
论文实验流水线的一键编排入口。

一键复现实验流水线：
  1. 主模型训练
  2. 主模型评估与板载推理 benchmark
  3. 全 baseline 对比
  4. 标准鲁棒性测试
  5. 真实 trace CSV 鲁棒性测试
  6. 多 seed 统计
  7. 消融实验与 PSF 消融
  8. 论文图表生成

说明：
  - 默认按 config.py 的正式配置运行，耗时较长。
  - 使用 --smoke 可快速验证整条链路是否能跑通。
  - 使用 --dry_run 只打印命令，不实际执行。
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from config import TRAIN_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _as_rel(path: Path) -> str:
    """尽量输出相对路径，方便复制命令。"""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _split_csv_paths(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _default_trace_csvs() -> list[str]:
    """自动发现项目内已经生成好的真实 trace CSV。"""
    candidates = [
        RESULTS_DIR / "goce_real_trace_link.csv",
        RESULTS_DIR / "slats_real_trace_link.csv",
    ]
    return [_as_rel(path) for path in candidates if path.exists()]


def _command_to_text(cmd: list[str]) -> str:
    return " ".join(cmd)


def _run_step(name: str, cmd: list[str], args, records: list[dict]) -> bool:
    """执行单个步骤，并把命令、耗时和状态写入记录。"""
    started = datetime.now()
    print("\n" + "=" * 80)
    print(f"[{name}]")
    print("-" * 80)
    print(_command_to_text(cmd))

    record = {
        "name": name,
        "command": cmd,
        "started_at": started.isoformat(),
        "status": "dry_run" if args.dry_run else "running",
    }

    if args.dry_run:
        record["finished_at"] = datetime.now().isoformat()
        record["returncode"] = None
        records.append(record)
        return True

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            check=False,
        )
        ok = result.returncode == 0
        record["returncode"] = int(result.returncode)
        record["status"] = "ok" if ok else "failed"
        return ok
    finally:
        finished = datetime.now()
        record["finished_at"] = finished.isoformat()
        record["elapsed_seconds"] = (finished - started).total_seconds()
        records.append(record)


def _add_if(cmd: list[str], condition: bool, *items: str) -> list[str]:
    if condition:
        cmd.extend(items)
    return cmd


def _train_cmd(args) -> list[str]:
    cmd = [
        sys.executable, "train.py",
        "--device", args.device,
        "--total_steps", str(args.total_steps),
        "--eval_freq", str(args.eval_freq),
        "--eval_episodes", str(args.eval_episodes),
        "--save_freq", str(args.save_freq),
        "--n_envs", str(args.n_envs),
        "--env_backend", args.env_backend,
        "--seed", str(args.seed),
        "--checkpoint_dir", args.checkpoint_dir,
        "--log_dir", args.log_dir,
    ]
    if args.warmup_steps is not None:
        cmd.extend(["--warmup_steps", str(args.warmup_steps)])
    if args.update_freq is not None:
        cmd.extend(["--update_freq", str(args.update_freq)])
    if args.update_actor_freq is not None:
        cmd.extend(["--update_actor_freq", str(args.update_actor_freq)])
    return cmd


def _best_checkpoint(args) -> str:
    # 烟雾模式只跑极少步，未必能完成正式 best 选择；后续环节用 latest 验证链路即可。
    name = "latest.pt" if getattr(args, "smoke", False) else "best_optimized.pt"
    return str(Path(args.checkpoint_dir) / name)


def _build_steps(args) -> list[tuple[str, list[str]]]:
    """根据参数构造完整命令序列。"""
    python = sys.executable
    best_ckpt = _best_checkpoint(args)
    steps: list[tuple[str, list[str]]] = []

    if not args.skip_train:
        steps.append(("主训练 train.py", _train_cmd(args)))

    if not args.skip_eval:
        eval_cmd = [
            python, "evaluate_optimized.py",
            "--model", best_ckpt,
            "--eval_episodes", str(args.eval_episodes),
            "--device", args.device,
            "--output", args.eval_output,
        ]
        _add_if(eval_cmd, not args.skip_benchmark,
                "--benchmark_onboard",
                "--benchmark_calls", str(args.benchmark_calls),
                "--benchmark_warmup", str(args.benchmark_warmup))
        steps.append(("主模型评估 evaluate_optimized.py", eval_cmd))

    if not args.skip_compare:
        steps.append((
            "全方法对比 compare_all.py",
            [
                python, "experiments/compare_all.py",
                "--checkpoint", best_ckpt,
                "--n_episodes", str(args.eval_episodes),
                "--device", args.device,
                "--max_steps", str(args.max_steps),
            ] if args.max_steps is not None else [
                python, "experiments/compare_all.py",
                "--checkpoint", best_ckpt,
                "--n_episodes", str(args.eval_episodes),
                "--device", args.device,
            ],
        ))
        if args.max_steps is not None:
            # 短步数 smoke 可能尚未遇到通信窗口；结果会由 compare_all 标记为非论文主表。
            steps[-1][1].append("--allow_zero_delivery")
        learned_ckpts = {
            "--sac_checkpoint": Path(args.ablation_dir) / "variant_F" / "best.pt",
            "--sac_lagrangian_checkpoint": Path(args.ablation_dir) / "variant_G" / "best.pt",
            "--sac_psf_checkpoint": Path(args.learning_baseline_dir) / "sac_psf" / "best.pt",
            "--sac_lya_checkpoint": Path(args.learning_baseline_dir) / "sac_lyapunov" / "best.pt",
        }
        for flag, path in learned_ckpts.items():
            if path.exists():
                steps[-1][1].extend([flag, str(path)])

    if not args.skip_robustness:
        robust_cmd = [
            python, "experiments/robustness.py",
            "--checkpoint", best_ckpt,
            "--n_episodes", str(args.eval_episodes),
            "--device", args.device,
            "--profile", args.robustness_profile,
        ]
        if args.max_steps is not None:
            robust_cmd.extend(["--max_steps", str(args.max_steps)])
        steps.append(("标准鲁棒性 robustness.py", robust_cmd))

    if not args.skip_trace_robustness:
        for trace_csv in args.trace_csvs:
            trace_name = Path(trace_csv).stem
            trace_cmd = [
                python, "experiments/robustness.py",
                "--checkpoint", best_ckpt,
                "--n_episodes", str(args.eval_episodes),
                "--device", args.device,
                "--profile", args.robustness_profile,
                "--trace_csv", trace_csv,
            ]
            if args.trace_cycle:
                trace_cmd.append("--trace_cycle")
            if args.trace_altitude_mode:
                trace_cmd.extend(["--trace_altitude_mode", args.trace_altitude_mode])
            if args.max_steps is not None:
                trace_cmd.extend(["--max_steps", str(args.max_steps)])
            steps.append((f"真实 trace 鲁棒性 {trace_name}", trace_cmd))

    if args.multi_seed_eval_only and not args.skip_multi_seed:
        multi_seed_cmd = [
            python, "experiments/multi_seed.py",
            "--mode", "eval",
            "--model", best_ckpt,
            "--eval_episodes", str(args.eval_episodes),
            "--device", args.device,
            "--output", args.multi_seed_output,
        ]
        if args.seeds:
            multi_seed_cmd.extend(["--seeds", args.seeds])
        steps.append(("多 seed 评估 multi_seed.py --mode eval", multi_seed_cmd))

    if (not args.multi_seed_eval_only) and (not args.skip_multi_seed):
        multi_seed_cmd = [
            python, "experiments/multi_seed.py",
            "--mode", "train-eval",
            "--total_steps", str(args.multi_seed_total_steps),
            "--eval_freq", str(args.eval_freq),
            "--train_eval_episodes", str(args.eval_episodes),
            "--eval_episodes", str(args.eval_episodes),
            "--save_freq", str(args.save_freq),
            "--n_envs", str(args.n_envs),
            "--env_backend", args.env_backend,
            "--device", args.device,
            "--output", args.multi_seed_output,
        ]
        if args.seeds:
            multi_seed_cmd.extend(["--seeds", args.seeds])
        if args.warmup_steps is not None:
            multi_seed_cmd.extend(["--warmup_steps", str(args.warmup_steps)])
        if args.update_freq is not None:
            multi_seed_cmd.extend(["--update_freq", str(args.update_freq)])
        if args.update_actor_freq is not None:
            multi_seed_cmd.extend(["--update_actor_freq", str(args.update_actor_freq)])
        steps.append(("严格多训练 seed multi_seed.py --mode train-eval", multi_seed_cmd))

    if args.train_ablation_models and not args.skip_ablation:
        ablation_train_cmd = [
            python, "experiments/ablation.py",
            "--train_independent_models",
            "--train_only",
            "--device", args.device,
            "--total_steps", str(args.ablation_total_steps),
            "--ablation_dir", args.ablation_dir,
            "--full_model_source", best_ckpt,
            "--eval_freq", str(args.eval_freq),
            "--train_eval_episodes", str(args.eval_episodes),
            "--summary_eval_episodes", str(args.eval_episodes),
            "--save_freq", str(args.save_freq),
            "--n_envs", str(args.n_envs),
            "--env_backend", args.env_backend,
            "--seed", str(args.seed),
        ]
        if args.warmup_steps is not None:
            ablation_train_cmd.extend(["--warmup_steps", str(args.warmup_steps)])
        if args.update_freq is not None:
            ablation_train_cmd.extend(["--update_freq", str(args.update_freq)])
        if args.update_actor_freq is not None:
            ablation_train_cmd.extend(["--update_actor_freq", str(args.update_actor_freq)])
        steps.append(("独立消融模型训练 ablation.py --train_independent_models", ablation_train_cmd))

    if args.train_learning_baselines and not args.skip_compare:
        learning_baseline_cmd = [
            python, "experiments/ablation.py",
            "--train_learning_baselines",
            "--train_only",
            "--device", args.device,
            "--total_steps", str(args.ablation_total_steps),
            "--learning_baseline_dir", args.learning_baseline_dir,
            "--eval_freq", str(args.eval_freq),
            "--train_eval_episodes", str(args.eval_episodes),
            "--summary_eval_episodes", str(args.eval_episodes),
            "--save_freq", str(args.save_freq),
            "--n_envs", str(args.n_envs),
            "--env_backend", args.env_backend,
            "--seed", str(args.seed),
        ]
        if args.warmup_steps is not None:
            learning_baseline_cmd.extend(["--warmup_steps", str(args.warmup_steps)])
        if args.update_freq is not None:
            learning_baseline_cmd.extend(["--update_freq", str(args.update_freq)])
        if args.update_actor_freq is not None:
            learning_baseline_cmd.extend(["--update_actor_freq", str(args.update_actor_freq)])
        steps.append(("独立学习型 baseline 训练 ablation.py --train_learning_baselines", learning_baseline_cmd))

    if not args.skip_ablation:
        ablation_cmd = [
            python, "experiments/ablation.py",
            "--checkpoint", best_ckpt,
            "--ablation_dir", args.ablation_dir,
            "--n_episodes", str(args.eval_episodes),
            "--device", args.device,
            "--use_independent_models",
        ]
        if args.max_steps is not None:
            ablation_cmd.extend(["--max_steps", str(args.max_steps)])
        if args.diagnostic_ablation:
            ablation_cmd.append("--stress_test")
        steps.append(("独立模型消融 ablation.py", ablation_cmd))

    if not args.skip_psf_ablation:
        psf_cmd = [
            python, "experiments/ablation.py",
            "--psf_sweep",
            "--checkpoint", best_ckpt,
            "--n_episodes", str(args.eval_episodes),
            "--device", args.device,
        ]
        if args.max_steps is not None:
            psf_cmd.extend(["--max_steps", str(args.max_steps)])
        steps.append(("PSF 消融 ablation.py --psf_sweep", psf_cmd))

    if not args.skip_pareto:
        pareto_cmd = [
            python, "experiments/pareto_frontier.py",
            "--train",
            "--total_steps", str(args.pareto_total_steps),
            "--eval_freq", str(args.eval_freq),
            "--train_eval_episodes", str(args.eval_episodes),
            "--eval_episodes", str(args.eval_episodes),
            "--save_freq", str(args.save_freq),
            "--n_envs", str(args.pareto_n_envs),
            "--env_backend", args.pareto_env_backend,
            "--device", args.device,
            "--seed", str(args.seed + 30000),
            "--output", args.pareto_output,
        ]
        if args.warmup_steps is not None:
            pareto_cmd.extend(["--warmup_steps", str(args.warmup_steps)])
        if args.update_freq is not None:
            pareto_cmd.extend(["--update_freq", str(args.update_freq)])
        if args.update_actor_freq is not None:
            pareto_cmd.extend(["--update_actor_freq", str(args.update_actor_freq)])
        steps.append(("Pareto 权重敏感性 pareto_frontier.py", pareto_cmd))

    if not args.skip_eval_traces:
        trace_export_cmd = [
            python, "experiments/export_eval_traces.py",
            "--checkpoint", best_ckpt,
            "--episodes", str(args.trace_episodes),
            "--seed", str(args.seed + 20000),
            "--device", args.device,
            "--output", args.eval_trace_output,
        ]
        if args.max_steps is not None:
            trace_export_cmd.extend(["--max_steps", str(args.max_steps)])
        steps.append(("逐步统计帧导出 export_eval_traces.py", trace_export_cmd))

    if not args.skip_figures:
        steps.append((
            "论文图表 plot_paper_figures.py",
            [
                python, "experiments/plot_paper_figures.py",
                "--results_dir", args.results_dir,
                "--output_dir", args.figures_dir,
            ],
        ))

    return steps


def _apply_smoke_defaults(args) -> None:
    """烟雾模式只验证链路，不产出论文级结果。"""
    if not args.smoke:
        return
    args.total_steps = min(args.total_steps, 2)
    args.ablation_total_steps = min(args.ablation_total_steps, 2)
    args.multi_seed_total_steps = min(args.multi_seed_total_steps, 2)
    args.pareto_total_steps = min(args.pareto_total_steps, 2)
    args.eval_freq = 1
    args.eval_episodes = 1
    args.save_freq = 1000
    args.n_envs = 1
    args.max_steps = 2
    args.benchmark_calls = 3
    args.benchmark_warmup = 1
    args.warmup_steps = 0
    args.update_freq = 1
    args.update_actor_freq = 1
    args.trace_episodes = 1
    args.pareto_n_envs = 1
    args.pareto_env_backend = "serial"


def _write_summary(args, records: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": "smoke" if args.smoke else "full",
        "dry_run": bool(args.dry_run),
        "continue_on_error": bool(args.continue_on_error),
        "checkpoint": _best_checkpoint(args),
        "trace_csvs": args.trace_csvs,
        "records": records,
        "all_ok": all(item["status"] in ("ok", "dry_run", "skipped") for item in records),
    }
    out_path = RESULTS_DIR / f"run_all_experiments_{_timestamp()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="一键运行主训练、评估、鲁棒性、多 seed、消融和论文图表。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--smoke", action="store_true",
                        help="快速烟雾模式：每项只跑极小步数，用于验证链路")
    parser.add_argument("--dry_run", action="store_true",
                        help="只打印将要执行的命令，不实际运行")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="某一步失败后继续执行后续步骤")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--total_steps", type=int,
                        default=int(TRAIN_CONFIG.get("total_steps", 1500000)))
    parser.add_argument("--ablation_total_steps", type=int,
                        default=int(TRAIN_CONFIG.get("total_steps", 1500000)))
    parser.add_argument("--eval_episodes", type=int,
                        default=int(TRAIN_CONFIG.get("eval_episodes", 30)))
    parser.add_argument("--eval_freq", type=int,
                        default=int(TRAIN_CONFIG.get("eval_freq", 5000)))
    parser.add_argument("--save_freq", type=int,
                        default=int(TRAIN_CONFIG.get("save_freq", 50000)))
    parser.add_argument("--n_envs", type=int,
                        default=int(TRAIN_CONFIG.get("n_envs", 1)))
    parser.add_argument("--env_backend", choices=["auto", "serial", "subproc"],
                        default=TRAIN_CONFIG.get("env_backend", "auto"))
    parser.add_argument("--seed", type=int, default=int(TRAIN_CONFIG.get("seed", 42)))
    parser.add_argument("--multi_seed_total_steps", type=int,
                        default=int(TRAIN_CONFIG.get("total_steps", 1500000)))
    parser.add_argument("--pareto_total_steps", type=int, default=300000)
    parser.add_argument("--pareto_n_envs", type=int, default=1)
    parser.add_argument("--pareto_env_backend",
                        choices=["serial", "auto", "subproc"], default="serial")
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--update_freq", type=int, default=None)
    parser.add_argument("--update_actor_freq", type=int, default=None)
    parser.add_argument("--benchmark_calls", type=int, default=300,
                        help="板载推理 benchmark 的调度调用次数")
    parser.add_argument("--benchmark_warmup", type=int, default=50,
                        help="板载推理 benchmark 的预热调用次数")
    parser.add_argument("--checkpoint_dir",
                        default=TRAIN_CONFIG.get("optimized_checkpoint_dir", "checkpoints_optimized/"))
    parser.add_argument("--log_dir",
                        default=TRAIN_CONFIG.get("optimized_log_dir", "logs_optimized/"))
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--figures_dir", default="figures/paper/")
    parser.add_argument("--eval_output", default=f"results/evaluation_report_{_timestamp()}.json")
    parser.add_argument("--multi_seed_output", default=f"results/multi_seed_{_timestamp()}.json")
    parser.add_argument("--pareto_output", default=f"results/pareto_frontier_{_timestamp()}.json")
    parser.add_argument("--eval_trace_output", default=f"results/eval_traces_{_timestamp()}.csv")
    parser.add_argument("--seeds", default=None,
                        help="多 seed 评估种子，逗号分隔；不填则读取 config.py")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="每个 episode 最大评估步数；正式跑建议保持空")
    parser.add_argument("--robustness_profile", choices=["standard", "extreme"], default="standard")
    parser.add_argument("--trace_csv", default=None,
                        help="真实 trace CSV，多个文件用逗号分隔；不填则自动找 results 下 GOCE/SLATS")
    parser.add_argument("--trace_cycle", action="store_true",
                        help="真实 trace 长度不足时循环复用")
    parser.add_argument("--trace_altitude_mode", choices=["ignore", "force"], default="ignore")
    parser.add_argument("--ablation_dir", default="checkpoints_ablation/")
    parser.add_argument("--learning_baseline_dir", default="checkpoints_learning_baselines/")
    parser.add_argument("--train_ablation_models", action="store_true", default=False,
                        help="额外训练 A-H 独立消融模型；非常耗时，默认关闭")
    parser.add_argument("--train_learning_baselines", action="store_true", default=False,
                        help="额外独立训练 SAC+PSF / SAC+Lyapunov 学习型对比 baseline")
    parser.add_argument("--skip_train_ablation_models", action="store_true",
                        help="跳过独立消融模型训练")
    parser.add_argument("--diagnostic_ablation", action="store_true",
                        help="附加运行共享 checkpoint 的诊断压力测试；论文主消融不依赖该结果")
    parser.add_argument("--multi_seed_eval_only", action="store_true",
                        help="只对同一个 checkpoint 做多 eval_seed 统计；正式论文不建议使用")
    parser.add_argument("--trace_episodes", type=int, default=3)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_benchmark", action="store_true")
    parser.add_argument("--skip_compare", action="store_true")
    parser.add_argument("--skip_robustness", action="store_true")
    parser.add_argument("--skip_trace_robustness", action="store_true")
    parser.add_argument("--skip_multi_seed", action="store_true")
    parser.add_argument("--skip_ablation", action="store_true")
    parser.add_argument("--skip_psf_ablation", action="store_true")
    parser.add_argument("--skip_pareto", action="store_true")
    parser.add_argument("--skip_eval_traces", action="store_true")
    parser.add_argument("--skip_figures", action="store_true")
    args = parser.parse_args()
    if args.skip_train_ablation_models:
        args.train_ablation_models = False

    _apply_smoke_defaults(args)
    trace_csvs = _split_csv_paths(args.trace_csv)
    if not trace_csvs:
        trace_csvs = _default_trace_csvs()
    args.trace_csvs = trace_csvs

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    steps = _build_steps(args)

    print("\n" + "=" * 80)
    print("  VLEO 论文实验一键流水线")
    print("=" * 80)
    print(f"  模式: {'SMOKE' if args.smoke else 'FULL'}")
    print(f"  dry_run: {args.dry_run}")
    print(f"  checkpoint: {_best_checkpoint(args)}")
    print(f"  trace_csvs: {args.trace_csvs if args.trace_csvs else '无'}")
    print(f"  步骤数: {len(steps)}")

    records: list[dict] = []
    all_ok = True
    for name, cmd in steps:
        ok = _run_step(name, cmd, args, records)
        if not ok:
            all_ok = False
            print(f"\n[失败] {name}")
            if not args.continue_on_error:
                break

    summary_path = _write_summary(args, records)
    print("\n" + "=" * 80)
    print("  流水线完成" if all_ok else "  流水线结束：存在失败步骤")
    print("=" * 80)
    print(f"  摘要: {_as_rel(summary_path)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

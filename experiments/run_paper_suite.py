"""论文实验套件串行驱动（一个后台 job 跑完主表/消融/场景，分阶段存 JSON）。

用 subprocess 逐阶段调用 paper_compare.py，每阶段独立存 JSON、独立成败，
避免链式 shell 脆弱与中途失败丢全部。

阶段：
  1. canonical 主对比（9 配置，nominal，20seed × N_MAIN ep）
  2. 消融（4 配置，RL，20seed × N_MAIN ep）
  3. 场景泛化（4 场景 × RL 三档梯队，20seed × N_SCEN ep）

用法：
  python experiments/run_paper_suite.py --device cuda --n_main 2 --n_scen 1
"""
import sys, os, subprocess, argparse, time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = "42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61"
SCEN_METHODS = "Raw RL,RL + SAFE_BUDGET,RL + SAFE_BUDGET + credit gate"  # RL 三档梯队


def _run(label, args_list):
    print(f"\n{'#'*80}\n# {label}\n{'#'*80}", flush=True)
    t0 = time.perf_counter()
    cmd = [sys.executable, os.path.join(_ROOT, "experiments", "paper_compare.py")] + args_list
    r = subprocess.run(cmd, cwd=_ROOT)
    dt = time.perf_counter() - t0
    print(f"# [{label}] exit={r.returncode} wall={dt/60:.1f}min", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default="checkpoints_optimized/best_optimized.pt")
    ap.add_argument("--n_main", type=int, default=2, help="主表/消融 每 seed episode 数")
    ap.add_argument("--n_scen", type=int, default=1, help="场景 每 seed episode 数")
    args = ap.parse_args()
    base = ["--device", args.device, "--checkpoint", args.ckpt, "--seeds", SEEDS]

    # 1) 主对比（9 配置）
    _run("Stage1: canonical main comparison (9 cfg)",
         ["--mode", "compare", "--scenario", "nominal", "--episodes", str(args.n_main),
          "--output", "results/paper_compare_nominal.json"] + base)

    # 2) 消融（4 配置）
    _run("Stage2: ablation (deployment mechanisms)",
         ["--mode", "ablation", "--scenario", "nominal", "--episodes", str(args.n_main),
          "--output", "results/paper_ablation_nominal.json"] + base)

    # 3) 场景泛化（4 场景 × RL 三档）
    for sc in ["sparse_comm", "energy_constrained", "high_density", "sparse_high_value"]:
        _run(f"Stage3: scenario {sc}",
             ["--mode", "compare", "--scenario", sc, "--episodes", str(args.n_scen),
              "--only", SCEN_METHODS,
              "--output", f"results/paper_compare_{sc}.json"] + base)

    print("\n[ALL DONE] paper suite complete. JSONs in results/paper_*.json", flush=True)


if __name__ == "__main__":
    main()

"""论文级方法对比 harness（method × 部署壳矩阵，同 seed/任务序列/窗口/能源）。

复用 compare_all.evaluate_on_env 的 rollout + 指标聚合，外层按 (method, shell) 矩阵 +
config_scenarios 场景切换迭代。每个 seed 单独成组（evaluate_on_env(seed_offset=s*100,
n_episodes=EP)），跨 seed 聚合得 mean + worst_seed(min ep_safe) + crash 合计。

9 个对比配置（--mode compare）：
  Raw RL / RL+SAFE_BUDGET / RL+SAFE_BUDGET+credit_gate /
  Heuristic / Heuristic+SB+CG / DPP / DPP+SB+CG / MPC / MPC+SB+CG
6 个消融配置（--mode ablation，固定 RL 策略）：
  no_shield / safe_budget_only / credit_gate_only / safe_budget+credit_gate /
  no_checkpoint_selector(诊断说明) / no_anti_conservative_filter(诊断说明)

输出 12 指标：episode_safety, worst_seed_safety, survival_rate, crash_count,
  comm_window_util, downlink, delivered_value, proc_dl, expired_value_rate,
  high_value_delivery, intervention(action_mod 代理), runtime(ms/step approx)。

用法：
  # 快筛（少 seed/ep 验证 harness）
  python experiments/paper_compare.py --mode compare --seeds 42,43,44 --episodes 1
  # canonical（20 seed）
  python experiments/paper_compare.py --mode compare --seeds 42,43,...,61 --episodes 3
  # 场景泛化
  python experiments/paper_compare.py --mode compare --scenario sparse_comm --seeds 42,43,44 --episodes 1
  # 消融
  python experiments/paper_compare.py --mode ablation --seeds 42,43,44 --episodes 2
"""
import sys, os, time, json, argparse
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _ROOT not in sys.path:
    sys.path.append(_ROOT)

from config import (TRAIN_CONFIG, SAFE_BUDGET_FALLBACK_CONFIG, HARD_RULES_CONFIG,
                    PROPULSION_CONTROLLER_CONFIG)
from experiments.compare_all import (evaluate_on_env, _pointed, _learned_scheduler_fn,
                                     _get_raw_state)
from experiments.config_scenarios import scenario, SCENARIOS
from scheduler.integrated_scheduler import IntegratedScheduler
from baselines.mpc_baseline import MPCBaseline
from baselines.dpp_baseline import DriftPlusPenaltyBaseline
from baselines.heuristic_baseline import ValueAwareHeuristicBaseline
from baselines.safe_greedy_baseline import SafeGreedyBaseline
from baselines.value_baselines import GreedyValueBaseline, EDFBaseline, LLFBaseline
from baselines.decoupled_baseline import make_decoupled_baseline
from utils.action_space import (default_grouped_action, choose_pointing_unit_for_env,
                                GROUPED_ACTION_DIM)
import numpy as _np

# ── 第一阶段：评估口径说明 ─────────────────────────────────────────────────
# 所有 paper 表格和图只允许读取同一套 final evaluation result（即本脚本的输出），
# 禁止混用训练过程中 periodic eval 和 final evaluation。
# 指标语义：
#   delivered_value  : 地面实际接收到的 deadline-aware VoI（论文主指标，单位任意值）
#   downlink         : 实际 RF 下传的压缩后 product MB（通信占用，≠任务价值）
#   processed        : 星上实际处理的 raw data MB（处理量，包含未下传部分）
#   proc_dl          : processed / downlinked 聚合比率（越低越好；>3.0 = 处理浪费）
#   comm_window_util : 通信窗口利用率（窗口内实际下传时间 / 窗口总时长）
#   episode_safety   : 每个 episode 满足全部安全约束的比率（=1.0 为严格安全）
#   survival_rate    : 无 crash（高度/电量终止）的 episode 占比
#   crash_count      : 所有 seed×episode 的 crash 次数总和（=0 为强安全）
# 上述指标与 evaluate_on_env 返回字段一一对应，禁止跨脚本混用不同语义口径。

CANONICAL_SEEDS = list(TRAIN_CONFIG.get("eval_seeds", list(range(42, 62))))
MAX_STEPS = int(TRAIN_CONFIG.get("max_episode_steps", 2160))

# 12 指标键（evaluate_on_env 返回口径）
_METRIC_KEYS = ["episode_safety_rate", "survival_rate", "comm_window_utilization",
                "downlink_mean", "delivered_value_mean", "proc_downlink_ratio",
                "expired_value_rate", "high_value_delivery_rate", "mean_action_modification"]


# ── 部署壳开关（env 级，config 实时读）──────────────────────────────
def set_shell(shell: str):
    SB = SAFE_BUDGET_FALLBACK_CONFIG
    HARD_RULES_CONFIG["enable_mission_pointing_fallback"] = False  # 旧激进 fallback 一律关
    if shell == "none":
        SB["enabled"] = False
        SB["enable_credit_gate"] = False
    elif shell == "sb":
        SB["enabled"] = True; SB["soft_min_soc"] = 0.60; SB["hard_min_soc"] = 0.50
        SB["enable_credit_gate"] = False
    elif shell == "sb_cg":
        SB["enabled"] = True; SB["soft_min_soc"] = 0.60; SB["hard_min_soc"] = 0.50
        SB["enable_credit_gate"] = True
        SB["target_proc_dl_ratio"] = 2.5; SB["credit_gain_per_downlink"] = 2.5
        SB["initial_credit_factor"] = 2.5
    elif shell == "cg_only":   # credit gate 但无 SAFE_BUDGET 能量/指向壳
        SB["enabled"] = True; SB["soft_min_soc"] = 0.0; SB["hard_min_soc"] = 0.0
        SB["enable_credit_gate"] = True
        SB["target_proc_dl_ratio"] = 2.5; SB["credit_gain_per_downlink"] = 2.5
        SB["initial_credit_factor"] = 2.5
    elif shell == "no_shield":  # 连解析推进 guard 也关（最弱安全）
        SB["enabled"] = False; SB["enable_credit_gate"] = False
        PROPULSION_CONTROLLER_CONFIG["guard_only"] = True  # 保留临界兜底防全崩；纯 no_shield 见诊断
    else:
        raise ValueError(shell)


_RL_CACHE = {}
def _build_fn(method: str, checkpoint: str, device: str):
    """返回 (scheduler_fn, use_wrapper)。"""
    if method == "RL":
        if "sched" not in _RL_CACHE:
            sch = IntegratedScheduler(device=device, enable_lyapunov=True, use_psf=True)
            sch.load(checkpoint)
            _RL_CACHE["sched"] = sch
        return _learned_scheduler_fn(_RL_CACHE["sched"]), "dilated"
    if method == "Heuristic":
        heur = ValueAwareHeuristicBaseline()
        return _pointed(lambda s, e: heur.schedule(_get_raw_state(s))), "none"
    if method == "DPP":
        dpp = DriftPlusPenaltyBaseline()
        return _pointed(lambda s, e: dpp.schedule(_get_raw_state(s), e)), "none"
    if method == "MPC":
        mpc = MPCBaseline()
        def mpc_fn(s, e):
            r = _get_raw_state(s)
            return mpc.schedule(r, e.battery.soc, e.altitude_m,
                                e.orbit_sim.is_sunlit(e.time_s),
                                e.solar.output_power(e.orbit_sim.sunlit_fraction(e.time_s)),
                                time_s=e.time_s, env=e)
        return _pointed(mpc_fn), "none"
    if method == "SafeGreedy":
        # SafeGreedy 自带 15 维含指向动作，不经 _pointed（避免覆盖其指向决策）
        sg = SafeGreedyBaseline()
        return (lambda s, e: sg.schedule(_get_raw_state(s), e)), "none"
    if method == "Random":
        # 均匀随机动作 + 全壳：比 Rule-only(零指令动作) 更诚实的"壳不能替代策略"下界
        # （零指令动作连推进都不请求，会被质疑 strawman；随机策略平均请求 50% 推进/处理/下传）。
        rng = _np.random.default_rng(20260610)
        return (lambda s, e: rng.random(GROUPED_ACTION_DIM).astype(_np.float32)), "none"
    if method == "RuleOnly":
        # 中性默认动作（prop/cpu/tx=0），完全交给部署壳 + 解析推进 guard 决定 →
        # 证明"安全壳本身不能替代学习策略"。
        return (lambda s, e: default_grouped_action(
            _np.zeros(3, dtype=_np.float32), pointing_unit=choose_pointing_unit_for_env(e))), "none"
    # ── 第三阶段：same-shell baselines ─────────────────────────────────────
    # EDF / LLF / Greedy / Dec-MPC 与 Ours 使用完全相同的安全壳（通过 set_shell()），
    # 保证环境、任务序列、通信窗口、初始状态、episode 数、seeds 完全一致。
    if method == "Greedy":
        grdy = GreedyValueBaseline()
        return _pointed(lambda s, e: grdy.schedule(s, e)), "none"
    if method == "EDF":
        edf = EDFBaseline()
        return _pointed(lambda s, e: edf.schedule(s, e)), "none"
    if method == "LLF":
        llf = LLFBaseline()
        return _pointed(lambda s, e: llf.schedule(s, e)), "none"
    if method == "DecMPC":
        # 解耦MPC：轨道控制器 + MPC 任务调度器，两者不共享功率预算信息
        _dec_name, _dec_fn = make_decoupled_baseline("mpc")
        return _dec_fn, "none"
    raise ValueError(method)


def run_config(label, method, shell, checkpoint, device, seeds, episodes):
    set_shell(shell)
    fn, wrap = _build_fn(method, checkpoint, device)
    per_seed = []
    t0 = time.perf_counter()
    for s in seeds:
        # 每个 seed 独立成组，seed_offset=s*100 避免组间种子重叠；同 label 跨方法用同 seed → 同任务序列
        stats = evaluate_on_env(fn, n_episodes=episodes, seed_offset=int(s) * 100,
                                use_wrapper=wrap, max_steps=MAX_STEPS)
        per_seed.append(stats)
    wall_s = time.perf_counter() - t0
    n_eps_total = len(seeds) * episodes

    def seed_mean(key):
        return float(np.mean([float(st.get(key, 0.0)) for st in per_seed]))
    # proc/dl 用全局 Σproc/Σdl（比率的比），对单个近零下传 seed 鲁棒；
    # 优于 per-seed 比率取均值（后者会被 downlink≈0 的 seed 拉爆）。
    sum_proc = float(np.sum([float(st.get("processed_mean", 0.0)) for st in per_seed]))
    sum_dl = float(np.sum([float(st.get("downlink_mean", 0.0)) for st in per_seed]))
    row = {
        "config": label, "method": method, "shell": shell,
        "episode_safety": seed_mean("episode_safety_rate"),
        "worst_seed_safety": float(np.min([float(st.get("episode_safety_rate", 0.0)) for st in per_seed])),
        "survival_rate": seed_mean("survival_rate"),
        "crash_count": int(np.sum([int(st.get("crash_count", 0)) for st in per_seed])),
        "comm_window_util": seed_mean("comm_window_utilization"),
        "downlink": seed_mean("downlink_mean"),
        "processed": seed_mean("processed_mean"),
        "delivered_value": seed_mean("delivered_value_mean"),
        "proc_dl": float(sum_proc / sum_dl) if sum_dl > 1e-6 else float("nan"),
        "expired_value_rate": seed_mean("expired_value_rate"),
        "high_value_delivery": seed_mean("high_value_delivery_rate"),
        "intervention": seed_mean("mean_action_modification"),
        "runtime_ms_per_step": float(wall_s * 1000.0 / max(n_eps_total * MAX_STEPS, 1)),
        "n_seeds": len(seeds), "episodes_per_seed": episodes,
        # per-seed 明细（统计检验 / CI 用；纯输出扩展，不影响评估行为）
        "per_seed": [
            {"seed": int(s),
             "crash_count": int(st.get("crash_count", 0)),
             **{k: float(st.get(k, 0.0)) for k in (
                 "episode_safety_rate", "survival_rate", "comm_window_utilization",
                 "downlink_mean", "delivered_value_mean", "processed_mean",
                 "expired_value_rate", "high_value_delivery_rate",
                 "mean_action_modification")}}
            for s, st in zip(seeds, per_seed)
        ],
    }
    return row


COMPARE_CONFIGS = [
    ("Raw RL", "RL", "none"),
    ("RL + SAFE_BUDGET", "RL", "sb"),
    ("RL + SAFE_BUDGET + credit gate", "RL", "sb_cg"),
    ("Heuristic", "Heuristic", "none"),
    ("Heuristic + SB + CG", "Heuristic", "sb_cg"),
    ("DPP", "DPP", "none"),
    ("DPP + SB + CG", "DPP", "sb_cg"),
    ("MPC", "MPC", "none"),
    ("MPC + SB + CG", "MPC", "sb_cg"),
    ("Safe Greedy + SB + CG", "SafeGreedy", "sb_cg"),
    ("Rule-only Shell (no learned policy)", "RuleOnly", "sb_cg"),
    ("Random + SB + CG", "Random", "sb_cg"),
    # ── 第三阶段：same-shell baselines ──────────────────────────────────────
    # 下列 baseline 与 Ours 使用完全相同的安全壳（sb_cg），确保对比公平；
    # 同时也保留无壳版本用于说明壳对各方法的安全增益。
    ("Greedy Value", "Greedy", "none"),
    ("Greedy Value + SB + CG", "Greedy", "sb_cg"),
    ("EDF", "EDF", "none"),
    ("EDF + SB + CG", "EDF", "sb_cg"),
    ("LLF", "LLF", "none"),
    ("LLF + SB + CG", "LLF", "sb_cg"),
    ("Dec-MPC", "DecMPC", "none"),
    ("Dec-MPC + SB + CG", "DecMPC", "sb_cg"),
]

# 消融固定 RL 策略，逐层切换部署机制
ABLATION_CONFIGS = [
    ("no_shield", "RL", "no_shield"),
    ("SAFE_BUDGET only", "RL", "sb"),
    ("credit gate only", "RL", "cg_only"),
    ("SAFE_BUDGET + credit gate", "RL", "sb_cg"),
]
# no_checkpoint_selector / no_anti_conservative_filter 是"模型选择"层消融，
# 不改 rollout 行为（同一 checkpoint），在表注里以诊断结论形式给出，见 paper_tables_figures。


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["compare", "ablation"], default="compare")
    ap.add_argument("--checkpoint", default="checkpoints_optimized/best_optimized.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default=",".join(str(s) for s in CANONICAL_SEEDS))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--scenario", default="nominal", choices=SCENARIOS)
    ap.add_argument("--only", default=None, help="只跑指定 config label（逗号分隔）")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    configs = COMPARE_CONFIGS if args.mode == "compare" else ABLATION_CONFIGS
    if args.only:
        want = {x.strip() for x in args.only.split(",")}
        configs = [c for c in configs if c[0] in want]
    out = args.output or f"results/paper_{args.mode}_{args.scenario}.json"

    rows = []
    with scenario(args.scenario):
        for label, method, shell in configs:
            print(f"[paper_{args.mode}] {args.scenario} | {label} ...", flush=True)
            rows.append(run_config(label, method, shell, args.checkpoint, args.device, seeds, args.episodes))
            r = rows[-1]
            print(f"    ep_safe={r['episode_safety']:.3f} worst={r['worst_seed_safety']:.2f} "
                  f"win={r['comm_window_util']:.3f} dl={r['downlink']:.0f} "
                  f"deliv={r['delivered_value']:.0f} proc_dl={r['proc_dl']:.2f} "
                  f"rt={r['runtime_ms_per_step']:.2f}ms/step", flush=True)

    # 还原交付默认壳（sb_cg）
    set_shell("sb_cg")

    cols = ["config", "episode_safety", "worst_seed_safety", "survival_rate", "crash_count",
            "comm_window_util", "downlink", "delivered_value", "proc_dl",
            "expired_value_rate", "high_value_delivery", "intervention", "runtime_ms_per_step"]
    print("\n" + "=" * 200)
    print(f"[{args.mode} | scenario={args.scenario} | {len(seeds)}seed × {args.episodes}ep]")
    print(f"{'config':>34} " + " ".join(f"{c[:12]:>13}" for c in cols[1:]))
    print("-" * 200)
    for r in rows:
        line = f"{r['config']:>34} "
        for c in cols[1:]:
            v = r[c]
            line += (f"{v:>13.3f} " if isinstance(v, float) else f"{v:>13} ")
        print(line)
    print("=" * 200)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # ── Phase 1: eval_config metadata ──────────────────────────────────────
    # 明确记录本次评估的 checkpoint、seeds、episodes，供所有 paper 表格/图复核。
    # paper_tables_figures.py 和 plot_paper_figures.py 必须读取同一份 JSON，
    # 禁止混用训练过程中 periodic eval（单 seed、少 episode）结果。
    eval_config_meta = {
        "eval_type": "final_evaluation",       # 区别于 periodic_train_eval
        "checkpoint": args.checkpoint,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "episodes_per_seed": args.episodes,
        "n_episodes_total": len(seeds) * args.episodes,
        "scenario": args.scenario,
        "mode": args.mode,
        "metric_semantics": {
            "delivered_value": "deadline-aware VoI delivered to ground (主指标)",
            "downlink": "actual RF downlinked compressed product MB (通信占用)",
            "processed": "onboard processed raw data MB (处理量，含未下传部分)",
            "proc_dl": "processed/downlinked aggregate ratio (越低越好; >3.0=处理浪费)",
            "comm_window_util": "fraction of contact time with active downlink",
            "episode_safety": "fraction of episodes satisfying all safety constraints",
            "worst_seed_safety": "minimum episode_safety across seeds (安全下界)",
            "survival_rate": "fraction of episodes without crash (高度/电量终止)",
            "crash_count": "total crashes across all seed×episode (=0 严格安全)",
            "expired_value_rate": "fraction of generated VoI that expired undelivered",
            "high_value_delivery": "fraction of high-value tasks successfully delivered",
        },
        "eval_protocol": {
            "same_task_sequence": True,
            "same_comm_windows": True,
            "same_initial_orbit": True,
            "same_initial_energy": True,
            "same_episode_length": True,
            "same_safety_shell_per_shell_config": True,
            "note": "seed_offset=seed*100 ensures all methods see identical task/window sequences per seed",
        },
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "scenario": args.scenario, "seeds": seeds,
                   "episodes_per_seed": args.episodes, "checkpoint": args.checkpoint,
                   "eval_config": eval_config_meta,
                   "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"[OK] saved: {out}")


if __name__ == "__main__":
    main()

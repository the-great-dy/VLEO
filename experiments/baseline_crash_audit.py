"""Baseline 崩溃/违规公平性审计。

回应审稿质疑："传统 baseline 崩溃是否因评估不公平"。本审计逐 episode 追踪传统
baseline（DPP/MPC/Heuristic，无部署壳）的终止失败类型与崩溃前状态，证明其崩溃源于
**无法处理 VLEO 能源-轨道-处理耦合**，而非评估设置不公（所有方法同 env/seed/窗口/能源）。

失败类型由 env 的分量崩溃标志判定：
  orbit_decay  : info.orbit_crashed（高度 ≤ crash 线；常因欠推或燃料耗尽）
  energy_depletion : info.energy_crashed（SOC ≤ crash 线）
  thermal      : info.thermal_crashed
  fuel 注记    : 崩溃前 propellant 余量≈0 → 标 (fuel_exhausted)
注：comm-window / attitude / queue overflow 在本 env 是 QoS 软违规，不是终止失败原因；
   单列其崩溃前发生率，但不计入 dominant terminal failure。

输出每方法：crash_count / dominant_failure_type / 各类型计数 /
  mean_soc_before_crash / mean_action_before_crash(prop,cpu,tx) /
  contact_state_before_crash / SAFE_BUDGET_triggered / credit_gate_triggered。

用法：
  python experiments/baseline_crash_audit.py --methods DPP,MPC,Heuristic --seeds 42,43,44,45,46 --episodes 2
"""
import sys, os, json, argparse
from collections import deque, Counter
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _ROOT not in sys.path:
    sys.path.append(_ROOT)

from config import TRAIN_CONFIG
from environment.satellite_env import VLEOSatelliteEnv
from experiments.paper_compare import _build_fn, set_shell

MAX_STEPS = int(TRAIN_CONFIG.get("max_episode_steps", 2160))
K = 15  # 崩溃前回溯窗口步数


def _failure_type(info, fuel_frac):
    types = []
    if float(info.get("orbit_crashed", 0.0)) > 0.5:
        types.append("orbit_decay" + ("(fuel_exhausted)" if fuel_frac <= 0.02 else "(under_thrust)"))
    if float(info.get("energy_crashed", 0.0)) > 0.5:
        types.append("energy_depletion")
    if float(info.get("thermal_crashed", 0.0)) > 0.5:
        types.append("thermal")
    return types or ["other"]


def audit_method(method, ckpt, device, seeds, episodes):
    fn, wrap = _build_fn(method, ckpt, device)  # shell 已由外层 set_shell('none') 关闭
    crashes = []
    n_ep = 0
    for s in seeds:
        for _ in range(episodes):
            env = VLEOSatelliteEnv(seed=int(s))
            state = env.reset()
            recent = deque(maxlen=K)  # (soc, prop, cpu, tx, in_window, sb_applied, cg_applied)
            done = False
            crashed = False
            last_info = {}
            while not done:
                a = fn(state, env)
                av = np.asarray(a, dtype=np.float32).reshape(-1)
                state, _, done, info = env.step(a, enforce_prop_smoothing=False)
                last_info = info
                in_win = bool(info.get("in_window", info.get("in_comm_window", False)))
                recent.append((
                    float(env.battery.soc),
                    float(av[0]) if av.size > 0 else 0.0,
                    float(av[1]) if av.size > 1 else 0.0,
                    float(av[2]) if av.size > 2 else 0.0,
                    1.0 if in_win else 0.0,
                    1.0 if info.get("safe_budget_fallback_applied", False) else 0.0,
                    1.0 if info.get("safe_budget_alpha_cpu_capped", False) else 0.0,
                ))
                if float(info.get("crashed", info.get("failure_state", 0.0))) > 0.5:
                    crashed = True
            n_ep += 1
            if crashed and recent:
                arr = np.asarray(recent, dtype=np.float32)
                fuel_frac = float(getattr(env, "propellant_kg", 0.0)) / max(
                    float(getattr(env, "_initial_propellant_kg", getattr(env, "propellant_kg", 1.0))), 1e-6)
                crashes.append({
                    "types": _failure_type(last_info, fuel_frac),
                    "soc": float(arr[:, 0].mean()),
                    "prop": float(arr[:, 1].mean()),
                    "cpu": float(arr[:, 2].mean()),
                    "tx": float(arr[:, 3].mean()),
                    "contact": float(arr[:, 4].mean()),
                    "sb": float(arr[:, 5].mean()),
                    "cg": float(arr[:, 6].mean()),
                    "fuel_frac": fuel_frac,
                })
    # 聚合
    type_counter = Counter(t for c in crashes for t in c["types"])
    dom = type_counter.most_common(1)[0][0] if type_counter else "—(no crash)"

    def cm(key):
        return float(np.mean([c[key] for c in crashes])) if crashes else float("nan")
    return {
        "method": method, "n_episodes": n_ep, "crash_count": len(crashes),
        "dominant_failure_type": dom,
        "failure_breakdown": dict(type_counter),
        "mean_soc_before_crash": cm("soc"),
        "mean_fuel_frac_before_crash": cm("fuel_frac"),
        "action_before_crash": {"prop": cm("prop"), "cpu": cm("cpu"), "tx": cm("tx")},
        "contact_state_before_crash": cm("contact"),
        "safe_budget_triggered": cm("sb"),
        "credit_gate_triggered": cm("cg"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="DPP,MPC,Heuristic")
    ap.add_argument("--checkpoint", default="checkpoints_optimized/best_optimized.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--output", default="results/baseline_crash_audit.json")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    set_shell("none")  # 审计传统 baseline 在无部署壳下的失败

    rows = []
    for m in [x.strip() for x in args.methods.split(",") if x.strip()]:
        print(f"[audit] {m} ...", flush=True)
        rows.append(audit_method(m, args.checkpoint, args.device, seeds, args.episodes))
        r = rows[-1]
        print(f"    crash {r['crash_count']}/{r['n_episodes']} | dom={r['dominant_failure_type']} | "
              f"soc={r['mean_soc_before_crash']:.3f} fuel={r['mean_fuel_frac_before_crash']:.3f} "
              f"prop={r['action_before_crash']['prop']:.2f} contact={r['contact_state_before_crash']:.2f}", flush=True)
    set_shell("sb_cg")  # 还原交付默认壳

    print("\n" + "=" * 150)
    hdr = ["method", "crash", "dominant_failure", "soc_pre", "fuel_pre", "prop_pre", "cpu_pre", "tx_pre", "contact_pre", "SB_trig", "CG_trig"]
    print(f"{hdr[0]:>12}{hdr[1]:>9}{hdr[2]:>26}" + "".join(f"{h:>10}" for h in hdr[3:]))
    print("-" * 150)
    for r in rows:
        ab = r["action_before_crash"]
        print(f"{r['method']:>12}{str(r['crash_count'])+'/'+str(r['n_episodes']):>9}"
              f"{r['dominant_failure_type'][:26]:>26}"
              f"{r['mean_soc_before_crash']:>10.3f}{r['mean_fuel_frac_before_crash']:>10.3f}"
              f"{ab['prop']:>10.2f}{ab['cpu']:>10.2f}{ab['tx']:>10.2f}"
              f"{r['contact_state_before_crash']:>10.2f}{r['safe_budget_triggered']:>10.2f}{r['credit_gate_triggered']:>10.2f}")
    print("=" * 150)
    for r in rows:
        print(f"  {r['method']}: failure_breakdown = {r['failure_breakdown']}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"seeds": seeds, "episodes_per_seed": args.episodes, "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] saved: {args.output}")


if __name__ == "__main__":
    main()

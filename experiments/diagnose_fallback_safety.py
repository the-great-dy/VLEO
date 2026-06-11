"""诊断 fallback ON 为什么打穿 episode_safety。

复现"激进 fallback ON"配置（enable_mission_pointing_fallback=True, 当前 SOC 门控），
逐步记录 SOC / 指向 / fallback 动作 / 能量变化 / 窗口 / 日照阴影，输出：
  - fallback_trigger_rate / to_downlink / to_image / to_charge(safety_guard_sun)
  - fallback 前后指向变化统计
  - crash/warning episode 的崩前 N 步回溯（SOC、action、fallback_action、energy_delta、contact、eclipse）
  - 判断：是否是 fallback 强行 IMAGE/DOWNLINK 抢走对日充电、打穿安全裕度

本脚本只读取 env.step() 已暴露的 info 字段，不修改环境，可快速运行。

用法:
  python experiments/diagnose_fallback_safety.py --model checkpoints_optimized/best_optimized.pt \
      --device cuda --seeds 42,43,44 --episodes 2 --min_soc 0.55 --retro_steps 20
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

import argparse
import numpy as np

from environment.satellite_env import VLEOSatelliteEnv
from environment.wrappers import DilatedFrameStackWrapper
from scheduler.integrated_scheduler import IntegratedScheduler
from config import TRAIN_CONFIG, DRL_CONFIG, HARD_RULES_CONFIG, ENERGY_CONFIG
from utils.action_space import POINTING_IMAGE, POINTING_DOWNLINK, POINTING_SUN

_MODE_NAME = {POINTING_IMAGE: "IMAGE", POINTING_DOWNLINK: "DOWNLINK", POINTING_SUN: "SUN"}


def _mission_stage(info: dict) -> str:
    if float(info.get("failure_state", 0.0)) > 0.5:
        return "failure"
    if float(info.get("unsafe_state", 0.0)) > 0.5:
        return "unsafe"
    if float(info.get("warning_state", 0.0)) > 0.5:
        return "warning"
    return "normal"


def _available_power_w(env) -> float:
    base = env.env if hasattr(env, "env") else env
    return float(getattr(base, "_last_available_power_w", 0.0))


def run_seed(model_path, device, seed, episodes, retro_steps):
    env = DilatedFrameStackWrapper(
        VLEOSatelliteEnv(seed=seed), k=DRL_CONFIG.get("frame_stack", 8))
    base_env = env.env if hasattr(env, "env") else env
    scheduler = IntegratedScheduler(device=device, enable_lyapunov=True, use_psf=True)
    scheduler.load(model_path)
    scheduler.reset_all_safety_stats()
    capacity_wh = float(ENERGY_CONFIG.get("battery_capacity_wh", 500.0))

    seed_steps = 0
    seed_trigger = 0
    seed_to_dl = seed_to_img = seed_to_charge = 0
    ep_summaries = []
    retros = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        prev_soc = float(base_env.battery.soc)
        trace = []
        ep_safe = True
        while not done:
            in_window = (base_env._contact.get("in_window", False)
                         if base_env._contact is not None else False)
            prop_can_update = True
            if hasattr(base_env, "step_count") and hasattr(base_env, "N_PROP_SMOOTH"):
                prop_can_update = (base_env.step_count % base_env.N_PROP_SMOOTH == 0)
            action, _, _, _ = scheduler.schedule(
                state, base_env.energy_queue.value, base_env.orbit_queue.value,
                base_env.data_queue.length, base_env.comm_queue.value,
                in_window, evaluate=True,
                h=base_env.altitude_m, soc=base_env.battery.soc,
                time_s=base_env.time_s, prop_can_update=prop_can_update,
                orbital_phase=base_env.orbit_sim.phase,
                tx_capacity_mbps=float((base_env._contact or {}).get("max_capacity_mbps", 0.0)),
                available_power_w=_available_power_w(env), env=base_env)
            state, reward, done, info = env.step(action, enforce_prop_smoothing=False)

            soc = float(info.get("soc", 0.0))
            applied = bool(info.get("mission_pointing_fallback_applied", False))
            reason = str(info.get("mission_pointing_fallback_reason", "disabled"))
            mode_before = int(info.get("mission_pointing_mode_before", -1))
            mode_after = int(info.get("mission_pointing_mode_after", -1))
            stage = _mission_stage(info)
            energy_delta_wh = (soc - prev_soc) * capacity_wh

            seed_steps += 1
            if applied:
                seed_trigger += 1
                if mode_after == POINTING_DOWNLINK:
                    seed_to_dl += 1
                elif mode_after == POINTING_IMAGE:
                    seed_to_img += 1
                elif mode_after == POINTING_SUN:
                    seed_to_charge += 1
            if stage != "normal":
                ep_safe = False

            trace.append({
                "step": int(getattr(base_env, "step_count", len(trace))),
                "soc": soc, "alt_km": float(info.get("altitude_km", 0.0)),
                "stage": stage, "overall_safe": float(info.get("overall_safe", 1.0)),
                "energy_stage": str(info.get("energy_stage", "normal")),
                "orbit_stage": str(info.get("orbit_stage", "normal")),
                "thermal_stage": str(info.get("thermal_stage", "normal")),
                "mode_before": mode_before, "mode_after": mode_after,
                "fb_applied": applied, "fb_reason": reason,
                "in_window": bool(info.get("in_comm_window", in_window)),
                "sunlit": bool(float(info.get("sunlit", 1.0)) > 0.5),
                "energy_delta_wh": energy_delta_wh,
                "alpha_tx": float(action[2]) if len(action) > 2 else 0.0,
                "alpha_prop": float(action[0]) if len(action) > 0 else 0.0,
            })
            prev_soc = soc

        # 找首个进入非 normal 的步，回溯前 retro_steps 步
        first_bad = next((i for i, r in enumerate(trace) if r["stage"] != "normal"), None)
        min_soc = min(r["soc"] for r in trace)
        ep_summaries.append({
            "seed": seed, "ep": ep, "ep_safe": ep_safe,
            "min_soc": min_soc, "min_alt_km": min(r["alt_km"] for r in trace),
            "steps": len(trace), "first_bad_step": (trace[first_bad]["step"] if first_bad is not None else None),
        })
        if first_bad is not None:
            lo = max(0, first_bad - retro_steps)
            retros.append({
                "seed": seed, "ep": ep,
                "bad_stage": trace[first_bad]["stage"],
                "rows": trace[lo:first_bad + 1],
            })

    return {
        "steps": seed_steps, "trigger": seed_trigger,
        "to_dl": seed_to_dl, "to_img": seed_to_img, "to_charge": seed_to_charge,
        "ep_summaries": ep_summaries, "retros": retros,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints_optimized/best_optimized.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--min_soc", type=float, default=0.55,
                    help="复现失败配置用的 mission_pointing_min_soc（默认 0.55=激进档）")
    ap.add_argument("--retro_steps", type=int, default=20)
    args = ap.parse_args()

    # 复现"激进 fallback ON"配置
    HARD_RULES_CONFIG["enable_mission_pointing_fallback"] = True
    HARD_RULES_CONFIG["mission_pointing_min_soc"] = float(args.min_soc)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print(f"[diag] model={args.model} fallback=ON min_soc={args.min_soc} "
          f"seeds={seeds} episodes={args.episodes}")
    tot = {"steps": 0, "trigger": 0, "to_dl": 0, "to_img": 0, "to_charge": 0}
    all_eps, all_retros = [], []
    for sd in seeds:
        r = run_seed(args.model, args.device, sd, args.episodes, args.retro_steps)
        for k in tot:
            tot[k] += r[k]
        all_eps.extend(r["ep_summaries"])
        all_retros.extend(r["retros"])

    n = max(1, tot["steps"])
    n_eps = max(1, len(all_eps))
    safe_eps = sum(1 for e in all_eps if e["ep_safe"])
    print("\n================ FALLBACK 触发统计 ================")
    print(f"total_steps                : {tot['steps']}")
    print(f"fallback_trigger_rate      : {tot['trigger']/n:.3f}")
    print(f"fallback_to_downlink_rate  : {tot['to_dl']/n:.3f}  ({tot['to_dl']} steps)")
    print(f"fallback_to_image_rate     : {tot['to_img']/n:.3f}  ({tot['to_img']} steps)")
    print(f"fallback_to_charge_rate    : {tot['to_charge']/n:.3f}  ({tot['to_charge']} steps, =safety_guard_sun)")
    print(f"episode_safety_rate        : {safe_eps/n_eps:.3f}  ({safe_eps}/{n_eps})")

    print("\n================ EPISODE 概览 ================")
    print(f"{'seed':>5} {'ep':>3} {'safe':>5} {'min_soc':>8} {'min_alt':>8} {'1st_bad':>8}")
    for e in all_eps:
        print(f"{e['seed']:>5} {e['ep']:>3} {str(e['ep_safe']):>5} "
              f"{e['min_soc']:>8.3f} {e['min_alt_km']:>8.1f} {str(e['first_bad_step']):>8}")

    print("\n================ 崩前回溯 (前 N 步) ================")
    if not all_retros:
        print("无 warning/unsafe/failure episode（该配置下全程安全）。")
    for rt in all_retros[:6]:  # 最多打印 6 个不安全 episode 的回溯
        print(f"\n--- seed={rt['seed']} ep={rt['ep']} 首次进入 [{rt['bad_stage']}] ---")
        print(f"{'step':>5} {'soc':>6} {'dEwh':>7} {'alt':>6} {'win':>4} {'sun':>4} "
              f"{'mode_b':>8} {'mode_a':>8} {'fb':>4} {'reason':>16} {'a_tx':>5} {'stage':>8} {'e_stg':>8}")
        for r in rt["rows"]:
            print(f"{r['step']:>5} {r['soc']:>6.3f} {r['energy_delta_wh']:>7.2f} "
                  f"{r['alt_km']:>6.1f} {str(r['in_window'])[:1]:>4} {str(r['sunlit'])[:1]:>4} "
                  f"{_MODE_NAME.get(r['mode_before'],'?'):>8} {_MODE_NAME.get(r['mode_after'],'?'):>8} "
                  f"{str(r['fb_applied'])[:1]:>4} {r['fb_reason']:>16} {r['alpha_tx']:>5.2f} "
                  f"{r['stage']:>8} {r['energy_stage']:>8}")

    # 自动判断
    print("\n================ 自动判断 ================")
    forced_task = tot["to_dl"] + tot["to_img"]
    if all_retros:
        # 统计崩前窗口里 fallback 强制 task 指向 + 放电的比例
        bad_rows = [r for rt in all_retros for r in rt["rows"]]
        forced_in_retro = sum(1 for r in bad_rows
                              if r["fb_applied"] and r["mode_after"] in (POINTING_IMAGE, POINTING_DOWNLINK))
        discharge_in_retro = sum(1 for r in bad_rows if r["energy_delta_wh"] < 0)
        print(f"崩前窗口步数            : {len(bad_rows)}")
        print(f"  其中 fallback 强制 IMAGE/DOWNLINK: {forced_in_retro} "
              f"({forced_in_retro/max(1,len(bad_rows)):.2f})")
        print(f"  其中净放电步            : {discharge_in_retro} "
              f"({discharge_in_retro/max(1,len(bad_rows)):.2f})")
        if forced_in_retro / max(1, len(bad_rows)) > 0.3:
            print(">>> 判定：fallback 在崩前持续强制 IMAGE/DOWNLINK，抢走对日充电 → 打穿能量裕度。"
                  "\n>>> 需要前瞻式能量预算（post_action_soc + 充电机会 + reserve）来抑制。")
        else:
            print(">>> 判定：崩溃与 fallback 强制 task 指向相关性不高，需进一步看其它驱动。")
    else:
        print("该配置未触发不安全 episode。")


if __name__ == "__main__":
    main()

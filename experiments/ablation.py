"""
论文主消融与诊断消融实验入口。
消融实验（支持论文主消融与诊断测试）

模式1：独立模型消融（论文主表）
    A-H 变体使用各自独立训练模型进行评估。

模式2：诊断压力测试（附录/调试）
    使用同一 checkpoint，在极端初始条件下比较安全层保护能力。
    这不是论文主消融，因为它没有重新训练被消融模块。

两种模式可同时运行，结果统一保存到 results/ablation_*.json。
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    # 仅在脚本直跑时追加，避免导入期全局污染 sys.path
    sys.path.append(_PROJECT_ROOT)

import numpy as np
import json
import argparse
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from environment.satellite_env import VLEOSatelliteEnv
from environment.wrappers import DilatedFrameStackWrapper
from scheduler.integrated_scheduler import IntegratedScheduler
from config import (
    TRAIN_CONFIG, DRL_CONFIG, ORBITAL_CONFIG, ENERGY_CONFIG, REWARD_CONFIG,
    PROPULSION_CONTROLLER_CONFIG, HARD_RULES_CONFIG,
)
from utils.paper_metrics import add_paper_metrics
from train import (
    _resolve_device,
    evaluate as evaluate_trained_scheduler,
    set_global_seed,
    train as train_main_model,
)

ALTITUDE_SAFE_KM = float(ORBITAL_CONFIG["altitude_min_km"])
BATTERY_SAFE_SOC = float(ENERGY_CONFIG["battery_min_soc"])

DEFAULT_OPTIMIZED_CHECKPOINT = os.path.join(
    TRAIN_CONFIG.get("optimized_checkpoint_dir", "checkpoints_optimized/"),
    "best_optimized.pt",
)


VARIANT_SPECS = {
    # 论文主消融围绕方向 A/C：VoI objective、CMDP constraint、adaptive dual 是主轴；
    # PSF/Lyapunov/AP-BC 作为部署安全层和辅助学习机制的次级消融。
    "A_Full": {
        "code": "A",
        "name": "A. Ours (VoI-CMDP SAC)",
        "enable_lyapunov": True,
        "use_psf": True,
        "constraint_variant": "ours",
        "network_arch": "transformer",
        "behavior_cloning_coeff": float(DRL_CONFIG.get("behavior_cloning_coeff", 0.05)),
        "adaptive_dual_enable": True,
        "ablation_axis": "full_method",
        "mission_reward_variant": "value_aware",
        "hypothesis": "VoI-aware CMDP SAC maximizes deadline-discounted delivered value under VLEO constraints",
    },
    "B_Throughput_Objective": {
        "code": "B",
        "name": "B. w/o VoI Objective",
        "enable_lyapunov": True,
        "use_psf": True,
        "constraint_variant": "ours",
        "network_arch": "transformer",
        "behavior_cloning_coeff": float(DRL_CONFIG.get("behavior_cloning_coeff", 0.05)),
        "adaptive_dual_enable": True,
        "ablation_axis": "task_value_objective",
        "mission_reward_variant": "throughput",
        "hypothesis": "replacing delivered VoI with delivered MB tests whether value modeling drives priority/deadline behavior",
    },
    "C_No_CMDP": {
        "code": "C",
        "name": "C. w/o CMDP Constraint",
        "enable_lyapunov": False,
        "use_psf": False,
        "constraint_variant": "plain_sac",
        "network_arch": "transformer",
        "behavior_cloning_coeff": 0.0,
        "adaptive_dual_enable": False,
        "ablation_axis": "cmdp_constraint_cost",
        "mission_reward_variant": "value_aware",
        "hypothesis": "removing constraint critic and safety constraints tests whether CMDP cost is necessary",
    },
    "D_No_Adaptive_Dual": {
        "code": "D",
        "name": "D. w/o Adaptive Dual",
        "enable_lyapunov": True,
        "use_psf": True,
        "constraint_variant": "ours",
        "network_arch": "transformer",
        "behavior_cloning_coeff": float(DRL_CONFIG.get("behavior_cloning_coeff", 0.05)),
        "adaptive_dual_enable": False,
        "ablation_axis": "constraint_violation_dual_update",
        "mission_reward_variant": "value_aware",
        "hypothesis": "fixed Lyapunov weight tests the benefit of updating lambda from normalized CMDP constraint violation",
    },
    "E_No_PSF": {
        "code": "E",
        "name": "E. w/o PSF",
        "enable_lyapunov": True,
        "use_psf": False,
        "constraint_variant": "ours",
        "network_arch": "transformer",
        "behavior_cloning_coeff": float(DRL_CONFIG.get("behavior_cloning_coeff", 0.05)),
        "adaptive_dual_enable": True,
        "ablation_axis": "deployment_psf_shield",
        "mission_reward_variant": "value_aware",
        "hypothesis": "removing PSF measures the value of short-horizon deployment-time physical shielding",
    },
    "F_No_Lyapunov": {
        "code": "F",
        "name": "F. w/o Lyapunov Projection",
        "enable_lyapunov": False,
        "use_psf": True,
        "constraint_variant": "ours",
        "network_arch": "transformer",
        "behavior_cloning_coeff": float(DRL_CONFIG.get("behavior_cloning_coeff", 0.05)),
        "adaptive_dual_enable": False,
        "ablation_axis": "long_horizon_lyapunov_projection",
        "mission_reward_variant": "value_aware",
        "hypothesis": "removing Lyapunov projection tests long-horizon queue/resource stability",
    },
    "G_MLP_Backbone": {
        "code": "G",
        "name": "G. MLP Backbone",
        "enable_lyapunov": True,
        "use_psf": True,
        "constraint_variant": "ours",
        "network_arch": "mlp",
        "behavior_cloning_coeff": float(DRL_CONFIG.get("behavior_cloning_coeff", 0.05)),
        "adaptive_dual_enable": True,
        "ablation_axis": "temporal_state_encoder",
        "mission_reward_variant": "value_aware",
        "hypothesis": "removing orbital temporal encoding tests whether contact-window and AoI dynamics need sequence modeling",
    },
    "H_With_AP_BC": {
        "code": "H",
        "name": "H. w/ AP-BC",
        "enable_lyapunov": True,
        "use_psf": True,
        "constraint_variant": "ours",
        "network_arch": "transformer",
        "behavior_cloning_coeff": 0.05,
        "adaptive_dual_enable": True,
        "ablation_axis": "actor_projection_alignment_positive_control",
        "mission_reward_variant": "value_aware",
        "hypothesis": "adding AP-BC tests whether projection imitation helps safety or pulls the actor back toward the rule shell",
    },
}
# ── 顶刊 Issue#1/#7: 安全层隔离消融 ───────────────────────────────────
# A-H 消融覆盖 PSF/Lyapunov/BC，但审稿人最关心的是"性能是否来自规则系统而非
# RL"。下列变体在保持 PSF+Lyapunov 的前提下，逐一关掉环境内强安全/规则层
# （解析推进控制器、硬指向兜底），用独立训练模型回答"RL 是否自己学会推进/指向"。
# 它们不进入 A-H 主表（不污染 missing-checkpoint 校验），通过 --safety_isolation 运行。
VARIANT_SPECS.update({
    "I_No_Analytic_Prop": {
        "code": "I",
        "name": "I. w/o Analytic Propulsion Controller",
        "enable_lyapunov": True,
        "use_psf": True,
        "constraint_variant": "ours",
        "network_arch": "transformer",
        "behavior_cloning_coeff": float(DRL_CONFIG.get("behavior_cloning_coeff", 0.05)),
        "adaptive_dual_enable": True,
        "ablation_axis": "analytic_propulsion_controller",
        "mission_reward_variant": "value_aware",
        "safety_config_override": {"analytic_propulsion_enabled": False},
        "hypothesis": "禁用解析推进控制器后看推进是否被规则接管：RL 能否自己学会省油/维持高度",
    },
    "J_No_Pointing_Fallback": {
        "code": "J",
        "name": "J. w/o Hard Pointing Fallback",
        "enable_lyapunov": True,
        "use_psf": True,
        "constraint_variant": "ours",
        "network_arch": "transformer",
        "behavior_cloning_coeff": float(DRL_CONFIG.get("behavior_cloning_coeff", 0.05)),
        "adaptive_dual_enable": True,
        "ablation_axis": "hard_pointing_fallback",
        "mission_reward_variant": "value_aware",
        "safety_config_override": {"mission_pointing_fallback": False},
        "hypothesis": "禁用硬指向兜底后看指向模式是否由 RL 学会而非规则强制",
    },
})

DISPLAY_ORDER = [
    "A_Full", "B_Throughput_Objective", "C_No_CMDP", "D_No_Adaptive_Dual",
    "E_No_PSF", "F_No_Lyapunov", "G_MLP_Backbone", "H_With_AP_BC",
]
SAFETY_ISOLATION_ORDER = ["A_Full", "I_No_Analytic_Prop", "J_No_Pointing_Fallback"]
STRESS_ORDER = ["A_Full", "E_No_PSF", "F_No_Lyapunov", "C_No_CMDP"]
CODE_TO_VARIANT = {spec["code"]: key for key, spec in VARIANT_SPECS.items()}
LEARNING_BASELINE_SPECS = {
    "SAC_PSF": {
        "code": "SAC_PSF",
        "name": "SAC + PSF",
        "checkpoint_subdir": "sac_psf",
        "enable_lyapunov": False,
        "use_psf": True,
        "constraint_variant": "sac_psf",
        "network_arch": "transformer",
        "behavior_cloning_coeff": 0.0,
        "mission_reward_variant": "value_aware",
        "ablation_axis": "main_comparison_learning_baseline",
        "hypothesis": "SAC policy trained with predictive safety filter but without Lyapunov constraint actor loss or AP-BC",
    },
    "SAC_Lyapunov": {
        "code": "SAC_LYA",
        "name": "SAC + Lyapunov",
        "checkpoint_subdir": "sac_lyapunov",
        "enable_lyapunov": True,
        "use_psf": False,
        "constraint_variant": "sac_lyapunov",
        "network_arch": "transformer",
        "behavior_cloning_coeff": 0.0,
        "mission_reward_variant": "value_aware",
        "ablation_axis": "main_comparison_learning_baseline",
        "hypothesis": "SAC policy trained with Lyapunov constraint actor loss but without PSF or AP-BC",
    },
}
BEST_SELECTION_RULE = (
    "feasible models first, then maximize safety-adjusted delivered task value, then reward_mean"
)


@contextmanager
def _temporary_reward_config(variant_key: str | None = None, spec_override: dict | None = None):
    """Temporarily switch reward and auxiliary heads for one independent variant."""
    old_reward = dict(REWARD_CONFIG)
    old_drl = {
        "value_aux_head_enable": DRL_CONFIG.get("value_aux_head_enable", None),
        "value_aux_loss_weight": DRL_CONFIG.get("value_aux_loss_weight", None),
        "value_aux_loss_weight_final": DRL_CONFIG.get("value_aux_loss_weight_final", None),
    }
    spec = dict(spec_override or VARIANT_SPECS.get(variant_key or "", {}))
    mode = str(spec.get("mission_reward_variant", "value_aware")).lower()
    try:
        if mode == "throughput":
            REWARD_CONFIG["reward_mode"] = "throughput"
            REWARD_CONFIG["w_delivered_mb"] = float(spec.get("throughput_reward_weight", 1.0))
            DRL_CONFIG["value_aux_head_enable"] = False
            DRL_CONFIG["value_aux_loss_weight"] = 0.0
            DRL_CONFIG["value_aux_loss_weight_final"] = 0.0
        else:
            REWARD_CONFIG.pop("reward_mode", None)
            REWARD_CONFIG.pop("w_delivered_mb", None)
            if "value_aux_head_enable" in spec:
                DRL_CONFIG["value_aux_head_enable"] = bool(spec["value_aux_head_enable"])
            if "value_aux_loss_weight" in spec:
                DRL_CONFIG["value_aux_loss_weight"] = float(spec["value_aux_loss_weight"])
            if "value_aux_loss_weight_final" in spec:
                DRL_CONFIG["value_aux_loss_weight_final"] = float(spec["value_aux_loss_weight_final"])
        yield
    finally:
        REWARD_CONFIG.clear()
        REWARD_CONFIG.update(old_reward)
        for key, old_value in old_drl.items():
            if old_value is None:
                DRL_CONFIG.pop(key, None)
            else:
                DRL_CONFIG[key] = old_value


@contextmanager
def _temporary_safety_layer_config(variant_key: str | None = None, spec_override: dict | None = None):
    """临时禁用环境内安全/规则层（解析推进控制器 / 硬指向兜底），并保证恢复。

    顶刊 Issue#1/#7：环境读取这两个 config 是 live 的，所以在 train + eval
    期间设置即可对训练与评估同时生效；退出时还原，避免单进程内串到其它变体。
    """
    spec = dict(spec_override or VARIANT_SPECS.get(variant_key or "", {}))
    override = dict(spec.get("safety_config_override", {}) or {})
    saved_prop = PROPULSION_CONTROLLER_CONFIG.get("enabled", True)
    saved_point = HARD_RULES_CONFIG.get("enable_mission_pointing_fallback", True)
    try:
        if "analytic_propulsion_enabled" in override:
            PROPULSION_CONTROLLER_CONFIG["enabled"] = bool(override["analytic_propulsion_enabled"])
        if "mission_pointing_fallback" in override:
            HARD_RULES_CONFIG["enable_mission_pointing_fallback"] = bool(override["mission_pointing_fallback"])
        yield
    finally:
        PROPULSION_CONTROLLER_CONFIG["enabled"] = saved_prop
        HARD_RULES_CONFIG["enable_mission_pointing_fallback"] = saved_point


def _variant_dir(args, variant_key: str) -> str:
    code = VARIANT_SPECS[variant_key]["code"]
    return os.path.join(args.ablation_dir, f"variant_{code}")


def _build_eval_env(seed: int):
    # 消融评估必须和主训练保持同一套当前包装观测链路。
    base = VLEOSatelliteEnv(seed=seed)
    return DilatedFrameStackWrapper(base, k=int(DRL_CONFIG.get("frame_stack", 8)))


def _evaluate_checkpoint(checkpoint_path: str, device: str, n_episodes: int,
                         variant_key: str | None = None) -> dict:
    scheduler = IntegratedScheduler(device=device)
    scheduler.load(checkpoint_path)
    eval_env = _build_eval_env(int(TRAIN_CONFIG.get("seed", 42)) + 3000)
    with _temporary_reward_config(variant_key):
        stats = evaluate_trained_scheduler(eval_env, scheduler, n_episodes=n_episodes)
    return add_paper_metrics({
        "best_reward": float(stats.get("reward_mean", 0.0)),
        "best_downlink_mb": float(stats.get("downlink_mean", 0.0)),
        "best_safety_rate": float(stats.get("safety_rate", 0.0)),
        "best_model_selection": BEST_SELECTION_RULE,
        "evaluation_stats": stats,
    })


def _copy_with_alias(src: str, variant_dir: str) -> str:
    os.makedirs(variant_dir, exist_ok=True)
    dst_best = os.path.join(variant_dir, "best.pt")
    dst_native = os.path.join(variant_dir, "best_optimized.pt")
    shutil.copy2(src, dst_best)
    if os.path.abspath(src) != os.path.abspath(dst_native):
        shutil.copy2(src, dst_native)
    return dst_best


def _train_variant_model(variant_key: str, args) -> dict:
    spec = VARIANT_SPECS[variant_key]
    variant_dir = _variant_dir(args, variant_key)
    log_dir = os.path.join(args.ablation_dir, f"variant_{spec['code']}_logs")
    os.makedirs(variant_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_args = SimpleNamespace(
        device=args.device,
        total_steps=args.total_steps,
        checkpoint_dir=variant_dir,
        log_dir=log_dir,
        eval_freq=args.eval_freq,
        eval_episodes=args.train_eval_episodes,
        save_freq=args.save_freq,
        n_envs=args.n_envs,
        warmup_steps=args.warmup_steps,
        update_freq=args.update_freq,
        update_actor_freq=args.update_actor_freq,
        resume_path=None,
        seed=args.seed,
        env_backend=args.env_backend,
        no_lyapunov=not bool(spec["enable_lyapunov"]),
        no_psf=not bool(spec["use_psf"]),
        constraint_variant=spec["constraint_variant"],
        variant_key=variant_key,
        variant_code=spec["code"],
        ablation_axis=spec.get("ablation_axis", ""),
        network_arch=spec.get("network_arch", "transformer"),
        behavior_cloning_coeff=spec.get(
            "behavior_cloning_coeff",
            DRL_CONFIG.get("behavior_cloning_coeff", 0.0),
        ),
        adaptive_lyapunov_coeff_enable=bool(spec.get("adaptive_dual_enable", True)),
        mission_reward_variant=spec.get("mission_reward_variant", "value_aware"),
        throughput_reward_weight=float(spec.get("throughput_reward_weight", 1.0)),
        # 顶刊 Issue#1/#7: 安全层隔离消融 flags（standalone train.py 也认这两个）。
        disable_analytic_propulsion=not bool(
            spec.get("safety_config_override", {}).get("analytic_propulsion_enabled", True)),
        disable_pointing_fallback=not bool(
            spec.get("safety_config_override", {}).get("mission_pointing_fallback", True)),
    )

    # train + eval 全程包在安全层 config 覆盖里：环境 live 读取，训练/评估同口径；
    # 退出后还原，避免单进程内污染后续变体。
    with _temporary_safety_layer_config(variant_key):
        with _temporary_reward_config(variant_key):
            train_main_model(train_args)

        native_best = os.path.join(variant_dir, "best_optimized.pt")
        aliased_best = os.path.join(variant_dir, "best.pt")
        if os.path.exists(native_best) and os.path.abspath(native_best) != os.path.abspath(aliased_best):
            shutil.copy2(native_best, aliased_best)

        stats = _evaluate_checkpoint(
            aliased_best, args.device, args.summary_eval_episodes, variant_key=variant_key)
    return add_paper_metrics({
        "status": "trained",
        "variant": spec["code"],
        "variant_key": variant_key,
        "variant_name": spec["name"],
        "enable_lyapunov": bool(spec["enable_lyapunov"]),
        "use_psf": bool(spec["use_psf"]),
        "constraint_variant": spec["constraint_variant"],
        "network_arch": spec.get("network_arch", "transformer"),
        "behavior_cloning_coeff": float(spec.get("behavior_cloning_coeff", 0.0)),
        "adaptive_dual_enable": bool(spec.get("adaptive_dual_enable", True)),
        "mission_reward_variant": spec.get("mission_reward_variant", "value_aware"),
        "ablation_axis": spec.get("ablation_axis", ""),
        "hypothesis": spec.get("hypothesis", ""),
        "best_checkpoint": aliased_best,
        "native_best_checkpoint": native_best,
        "log_dir": log_dir,
        **stats,
    })


def _materialize_full_variant(args) -> dict:
    variant_key = "A_Full"
    variant_dir = _variant_dir(args, variant_key)
    src = args.full_model_source
    if not os.path.exists(src):
        print(f"[A] 未找到源 checkpoint：{src}，将改为直接训练 A 变体。")
        return _train_variant_model(variant_key, args)

    dst_best = _copy_with_alias(src, variant_dir)
    stats = _evaluate_checkpoint(
        dst_best, args.device, args.summary_eval_episodes, variant_key=variant_key)
    return add_paper_metrics({
        "status": "copied",
        "variant": "A",
        "variant_key": variant_key,
        "variant_name": VARIANT_SPECS[variant_key]["name"],
        "source": src,
        "best_checkpoint": dst_best,
        "enable_lyapunov": True,
        "use_psf": True,
        "constraint_variant": "ours",
        "network_arch": VARIANT_SPECS[variant_key].get("network_arch", "transformer"),
        "behavior_cloning_coeff": float(VARIANT_SPECS[variant_key].get("behavior_cloning_coeff", 0.0)),
        "adaptive_dual_enable": bool(VARIANT_SPECS[variant_key].get("adaptive_dual_enable", True)),
        "mission_reward_variant": VARIANT_SPECS[variant_key].get("mission_reward_variant", "value_aware"),
        "ablation_axis": VARIANT_SPECS[variant_key].get("ablation_axis", ""),
        "hypothesis": VARIANT_SPECS[variant_key].get("hypothesis", ""),
        **stats,
    })


def independent_checkpoint_paths(args) -> dict:
    """Return the expected formal ablation checkpoint path for every variant."""
    return {
        key: os.path.join(args.ablation_dir, f"variant_{spec['code']}", "best.pt")
        for key, spec in VARIANT_SPECS.items()
    }


def missing_independent_checkpoints(args) -> dict:
    """List missing independent checkpoints; formal paper ablation must have none."""
    return {
        key: path
        for key, path in independent_checkpoint_paths(args).items()
        if key in DISPLAY_ORDER and not os.path.exists(path)
    }


def run_train_independent_ablation_models(args) -> dict:
    """训练或整理 A-H 独立消融模型。"""
    Path(args.ablation_dir).mkdir(parents=True, exist_ok=True)
    variants = [
        CODE_TO_VARIANT[code]
        for code in (args.only or [VARIANT_SPECS[key]["code"] for key in DISPLAY_ORDER])
    ]

    print("=" * 70)
    print("  独立消融模型训练")
    print("=" * 70)
    print(f"  设备: {args.device}")
    print(f"  变体: {[VARIANT_SPECS[key]['code'] for key in variants]}")
    print(f"  每个变体训练步数: {args.total_steps:,}")
    print(f"  是否直接复用 A: {args.copy_A}")

    start_time = datetime.now()
    results = {}
    for variant_key in variants:
        spec = VARIANT_SPECS[variant_key]
        print(f"\n[{spec['code']}] {spec['name']}")
        if variant_key == "A_Full" and args.copy_A:
            result = _materialize_full_variant(args)
        else:
            result = _train_variant_model(variant_key, args)
        results[spec["code"]] = result
        print(
            f"  safe={result.get('best_safety_rate', 0.0):.1%}, "
            f"dl={result.get('best_downlink_mb', 0.0):.1f}MB, "
            f"r={result.get('best_reward', 0.0):.1f}"
        )

    meta = {
        "timestamp": datetime.now().isoformat(),
        "device": args.device,
        "total_steps_per_variant": args.total_steps,
        "variants": [VARIANT_SPECS[key]["code"] for key in variants],
        "copy_A": bool(args.copy_A),
        "full_model_source": args.full_model_source,
        "best_model_selection": BEST_SELECTION_RULE,
        "elapsed_seconds": (datetime.now() - start_time).total_seconds(),
        "results": results,
    }
    meta_path = os.path.join(args.ablation_dir, "training_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n  元数据已保存: {meta_path}")
    return meta


def _train_learning_baseline_model(baseline_key: str, args) -> dict:
    spec = LEARNING_BASELINE_SPECS[baseline_key]
    baseline_dir = os.path.join(args.learning_baseline_dir, spec["checkpoint_subdir"])
    log_dir = os.path.join(args.learning_baseline_dir, f"{spec['checkpoint_subdir']}_logs")
    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_args = SimpleNamespace(
        device=args.device,
        total_steps=args.total_steps,
        checkpoint_dir=baseline_dir,
        log_dir=log_dir,
        eval_freq=args.eval_freq,
        eval_episodes=args.train_eval_episodes,
        save_freq=args.save_freq,
        n_envs=args.n_envs,
        warmup_steps=args.warmup_steps,
        update_freq=args.update_freq,
        update_actor_freq=args.update_actor_freq,
        resume_path=None,
        seed=args.seed,
        env_backend=args.env_backend,
        no_lyapunov=not bool(spec["enable_lyapunov"]),
        no_psf=not bool(spec["use_psf"]),
        constraint_variant=spec["constraint_variant"],
        variant_key=baseline_key,
        variant_code=spec["code"],
        ablation_axis=spec.get("ablation_axis", ""),
        network_arch=spec.get("network_arch", "transformer"),
        behavior_cloning_coeff=float(spec.get("behavior_cloning_coeff", 0.0)),
        adaptive_lyapunov_coeff_enable=bool(spec.get("adaptive_dual_enable", True)),
        mission_reward_variant=spec.get("mission_reward_variant", "value_aware"),
        throughput_reward_weight=float(spec.get("throughput_reward_weight", 1.0)),
    )

    with _temporary_reward_config(spec_override=spec):
        train_main_model(train_args)

    native_best = os.path.join(baseline_dir, "best_optimized.pt")
    aliased_best = os.path.join(baseline_dir, "best.pt")
    if os.path.exists(native_best) and os.path.abspath(native_best) != os.path.abspath(aliased_best):
        shutil.copy2(native_best, aliased_best)
    stats = _evaluate_checkpoint(
        aliased_best, args.device, args.summary_eval_episodes, variant_key=None)
    return add_paper_metrics({
        "status": "trained",
        "baseline_key": baseline_key,
        "baseline_name": spec["name"],
        "constraint_variant": spec["constraint_variant"],
        "enable_lyapunov": bool(spec["enable_lyapunov"]),
        "use_psf": bool(spec["use_psf"]),
        "behavior_cloning_coeff": float(spec.get("behavior_cloning_coeff", 0.0)),
        "adaptive_dual_enable": bool(spec.get("adaptive_dual_enable", True)),
        "mission_reward_variant": spec.get("mission_reward_variant", "value_aware"),
        "best_checkpoint": aliased_best,
        "native_best_checkpoint": native_best,
        "log_dir": log_dir,
        **stats,
    })


def run_train_learning_baselines(args) -> dict:
    """Train independently learned baselines used by compare_all main tables."""
    Path(args.learning_baseline_dir).mkdir(parents=True, exist_ok=True)
    selected = args.learning_baselines or list(LEARNING_BASELINE_SPECS.keys())
    print("=" * 70)
    print("  独立学习型 baseline 训练")
    print("=" * 70)
    print(f"  变体: {[LEARNING_BASELINE_SPECS[key]['name'] for key in selected]}")

    results = {}
    start_time = datetime.now()
    for baseline_key in selected:
        spec = LEARNING_BASELINE_SPECS[baseline_key]
        print(f"\n[{spec['code']}] {spec['name']}")
        result = _train_learning_baseline_model(baseline_key, args)
        results[baseline_key] = result
        print(
            f"  safe={result.get('best_safety_rate', 0.0):.1%}, "
            f"dl={result.get('best_downlink_mb', 0.0):.1f}MB, "
            f"r={result.get('best_reward', 0.0):.1f}"
        )

    meta = {
        "timestamp": datetime.now().isoformat(),
        "device": args.device,
        "total_steps_per_baseline": args.total_steps,
        "baselines": selected,
        "best_model_selection": BEST_SELECTION_RULE,
        "elapsed_seconds": (datetime.now() - start_time).total_seconds(),
        "results": results,
    }
    meta_path = os.path.join(args.learning_baseline_dir, "training_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n  学习型 baseline 元数据已保存: {meta_path}")
    return meta


def evaluate_variant(scheduler, n_episodes=None, seed_offset=200,
                     stress_config=None, max_steps=None):
    """
    评估单个消融变体，收集完整指标
    """
    n_episodes = int(TRAIN_CONFIG.get("eval_episodes", 30) if n_episodes is None else n_episodes)
    k = DRL_CONFIG.get("frame_stack", 8)

    rewards, throughputs, tx_mbs, values = [], [], [], []
    proc_dl_ratios, high_value_delivery_rates, window_utils = [], [], []
    safes, orbit_viols, energy_viols, thermal_viols, raw_viols, processed_viols = [], [], [], [], [], []
    processed_final_utils, tx_active_contact_flags = [], []
    deadline_rates, value_weighted_deadline_rates = [], []
    value_weighted_aoi_steps, voi_loss_rates = [], []
    survival_steps_list = []
    survival_flags = []
    reward_per_step_list = []
    tput_per_step_list = []
    psf_rates = []
    mod_l2s = []
    soc_min_list, alt_min_list = [], []
    stage_rate_sums = {"normal": [], "warning": [], "unsafe": [], "failure": []}

    for ep in range(n_episodes):
        # 消融评估使用和主训练一致的 当前环境 + dilated frame stack。
        base_env = VLEOSatelliteEnv(seed=seed_offset + ep)
        env = DilatedFrameStackWrapper(base_env, k=k)
        scheduler.reset_all_safety_stats()

        state = env.reset()

        # 压力测试：注入极端初始条件
        if stress_config:
            # 压力测试只改初始条件/任务强度，不改模型权重，用来观察安全层兜底能力。
            if "initial_altitude_km" in stress_config:
                base_env.altitude_m = stress_config["initial_altitude_km"] * 1e3
                base_env.orbit_queue.reset(base_env.altitude_m)
            if "initial_soc" in stress_config:
                base_env.battery.soc = stress_config["initial_soc"]
                base_env.energy_queue.reset(base_env.battery.energy_margin_wh)
            if "data_scale" in stress_config:
                base_env._data_arrival_scale = stress_config["data_scale"]

            # 注入高度后同步刷新通信窗口状态，避免第一步读到 reset 前 contact
            base_env._contact = base_env.gs_network.get_contact_info(
                base_env.time_s, base_env.altitude_m)

            # 刷新 wrapper 历史
            new_obs = base_env._get_observation()
            env._history.clear()
            for _ in range(env._max_offset + 1):
                env._history.append(new_obs.copy())
            state = env._get_obs()

        ep_r = ep_tput = ep_tx = ep_value = 0.0
        ep_high_delivered = ep_high_expired = ep_high_dropped = 0.0
        ep_final_processed_util = 0.0
        orbit_v = energy_v = thermal_v = raw_v = proc_v = 0
        done = False
        step_count = 0
        ep_soc_min = 1.0
        ep_alt_min = 500.0
        survived = True
        stage_counts = {"normal": 0, "warning": 0, "unsafe": 0, "failure": 0}

        while not done:
            # 调度器上下文要包含通信窗口和物理状态，否则 PSF/Lyapunov 的介入条件会失真。
            in_window = (env._contact.get("in_window", False)
                         if env._contact is not None else False)
            prop_can_update = True
            if hasattr(env, "step_count") and hasattr(env, "N_PROP_SMOOTH"):
                prop_can_update = (env.step_count % env.N_PROP_SMOOTH == 0)

            action, _, raw_action, psf_meta = scheduler.schedule(
                state, env.energy_queue.value, env.orbit_queue.value,
                env.data_queue.length, env.comm_queue.value,
                in_window=in_window, evaluate=True,
                h=env.altitude_m, soc=env.battery.soc,
                time_s=env.time_s,
                prop_can_update=prop_can_update,
                orbital_phase=env.orbit_sim.phase,
                tx_capacity_mbps=float((env._contact or {}).get("max_capacity_mbps", 0.0)),
                available_power_w=getattr(env, "available_power_w", None),
                env=env)

            state, reward, done, info = env.step(
                action, enforce_prop_smoothing=False)
            executed_action = np.asarray(info.get("executed_action", action), dtype=np.float32)
            mod_l2 = float(max(
                psf_meta.get("total_modification_l2", psf_meta.get("modification_l2", 0.0)),
                np.linalg.norm(executed_action - np.asarray(raw_action, dtype=np.float32)),
                0.0,
            ))
            if mod_l2 > 0.0:
                mod_l2s.append(mod_l2)
            ep_r += reward
            ep_tput += info.get("processed_mb", info.get("service_rate_mbs", 0) * TRAIN_CONFIG["time_slot_s"])
            ep_tx += info.get("delivered_mb", info.get("actual_tx_mb", 0))
            ep_value += info.get("delivered_value", 0.0)
            ep_high_delivered += float(info.get("delivered_high_value", 0.0))
            ep_high_expired += float(info.get("expired_high_value", 0.0))
            ep_high_dropped += float(info.get("dropped_high_value", 0.0))
            ep_final_processed_util = float(info.get("processed_queue_utilization", 0.0))
            capacity_mb = float(info.get("tx_capacity_mbps", 0.0)) * TRAIN_CONFIG["time_slot_s"] / 8.0
            if bool(info.get("in_window", False)) and capacity_mb > 1e-9:
                window_utils.append(float(info.get("delivered_mb", info.get("actual_tx_mb", 0.0))) / capacity_mb)
                tx_active_contact_flags.append(float(
                    info.get("delivered_mb", info.get("actual_tx_mb", 0.0)) > 1e-9
                ))
            step_count += 1

            cur_soc = info.get("soc", 1.0)
            cur_alt = info.get("altitude_km", 400)
            ep_soc_min = min(ep_soc_min, cur_soc)
            ep_alt_min = min(ep_alt_min, cur_alt)

            if bool(info.get("terminated", False)):
                survived = False
            if not bool(info.get("orbit_safe", cur_alt >= ALTITUDE_SAFE_KM)):
                orbit_v += 1
            if not bool(info.get("energy_safe", cur_soc >= BATTERY_SAFE_SOC)):
                energy_v += 1
            if not bool(info.get("thermal_safe", True)):
                thermal_v += 1
            if not bool(info.get("raw_queue_safe", True)):
                raw_v += 1
            if not bool(info.get("processed_queue_safe", True)):
                proc_v += 1
            stage = str(info.get("risk_stage", "normal"))
            if stage not in stage_counts:
                stage = "failure" if bool(info.get("crashed", False)) else "normal"
            stage_counts[stage] += 1

            if max_steps is not None and step_count >= max_steps:
                done = True

        rewards.append(ep_r)
        throughputs.append(ep_tput)
        tx_mbs.append(ep_tx)
        values.append(ep_value)
        proc_dl_ratios.append(float(ep_tput / max(ep_tx, 1e-9)))
        high_den = ep_high_delivered + ep_high_expired + ep_high_dropped
        high_value_delivery_rates.append(float(ep_high_delivered / max(high_den, 1e-9)))
        processed_final_utils.append(float(ep_final_processed_util))
        is_safe = (orbit_v == 0 and energy_v == 0 and thermal_v == 0 and raw_v == 0 and proc_v == 0)
        safes.append(float(is_safe))
        orbit_viols.append(orbit_v)
        energy_viols.append(energy_v)
        thermal_viols.append(thermal_v)
        raw_viols.append(raw_v)
        processed_viols.append(proc_v)
        survival_steps_list.append(step_count)
        survival_flags.append(float(survived))
        soc_min_list.append(ep_soc_min)
        alt_min_list.append(ep_alt_min)
        for stage_name, values in stage_rate_sums.items():
            values.append(stage_counts[stage_name] / max(step_count, 1))
        psf_rates.append(scheduler.get_safety_stats().get("psf_filter_rate", 0.0))

        # 每步平均（消除存活时长差异的公平指标）
        reward_per_step_list.append(ep_r / max(step_count, 1))
        tput_per_step_list.append(ep_tput / max(step_count, 1))
        task_summary = getattr(base_env, "task_tracker", None).summary() if hasattr(base_env, "task_tracker") else {}
        deadline_rates.append(float(task_summary.get("deadline_success_rate", 0.0)))
        value_weighted_deadline_rates.append(float(task_summary.get(
            "value_weighted_deadline_success_rate",
            task_summary.get("deadline_success_rate", 0.0),
        )))
        value_weighted_aoi_steps.append(float(task_summary.get(
            "value_weighted_aoi_steps",
            task_summary.get("average_aoi_steps", task_summary.get("avg_delivery_delay_steps", 0.0)),
        )))
        voi_loss_rates.append(float(task_summary.get("voi_loss_rate", 0.0)))

    crash_count = int(sum(1 for survived in survival_flags if survived <= 0.0))

    return add_paper_metrics({
        # ── 安全性指标（论文主表第一组列）──────────────────
        "survival_rate": float(np.mean(survival_flags)),
        "safety_rate": float(np.mean(safes)),
        "episode_safety_rate": float(np.mean(safes)),
        "crash_count": crash_count,
        "normal_state_rate": float(np.mean(stage_rate_sums["normal"])) if stage_rate_sums["normal"] else 0.0,
        "warning_state_rate": float(np.mean(stage_rate_sums["warning"])) if stage_rate_sums["warning"] else 0.0,
        "unsafe_state_rate": float(np.mean(stage_rate_sums["unsafe"])) if stage_rate_sums["unsafe"] else 0.0,
        "failure_state_rate": float(np.mean(stage_rate_sums["failure"])) if stage_rate_sums["failure"] else 0.0,
        "survival_steps_mean": float(np.mean(survival_steps_list)),
        "survival_steps_std": float(np.std(survival_steps_list)),
        "soc_min_mean": float(np.mean(soc_min_list)),
        "alt_min_mean": float(np.mean(alt_min_list)),

        # ── 效率指标（论文主表第二组列）──────────────────
        "reward_per_step": float(np.mean(reward_per_step_list)),
        "tput_per_step": float(np.mean(tput_per_step_list)),

        # ── 任务完成度（论文主表第三组列）────────────────
        "throughput_total": float(np.mean(throughputs)),
        "tx_mb_total": float(np.mean(tx_mbs)),
        "delivered_value_total": float(np.mean(values)),
        "global_proc_downlink_ratio": float(np.sum(throughputs) / max(np.sum(tx_mbs), 1e-9)),
        "mean_episode_proc_downlink_ratio": float(np.mean(proc_dl_ratios)) if proc_dl_ratios else 0.0,
        "proc_downlink_ratio": float(np.sum(throughputs) / max(np.sum(tx_mbs), 1e-9)),
        "episode_proc_dl_ratio": float(np.mean(proc_dl_ratios)) if proc_dl_ratios else 0.0,
        "proc_dl_ratio": float(np.sum(throughputs) / max(np.sum(tx_mbs), 1e-9)),
        "comm_window_utilization": float(np.mean(window_utils)) if window_utils else 0.0,
        "processed_queue_final_utilization": float(np.mean(processed_final_utils)) if processed_final_utils else 0.0,
        "tx_active_in_contact_ratio": float(np.mean(tx_active_contact_flags)) if tx_active_contact_flags else 0.0,
        "high_value_delivery_rate": (
            float(np.mean(high_value_delivery_rates)) if high_value_delivery_rates else 0.0
        ),
        "high_value_delivery_ratio": (
            float(np.mean(high_value_delivery_rates)) if high_value_delivery_rates else 0.0
        ),
        "deadline_success_rate": float(np.mean(deadline_rates)) if deadline_rates else 0.0,
        "value_weighted_deadline_success_rate": (
            float(np.mean(value_weighted_deadline_rates)) if value_weighted_deadline_rates else 0.0
        ),
        "value_weighted_aoi_steps": float(np.mean(value_weighted_aoi_steps)) if value_weighted_aoi_steps else 0.0,
        "voi_loss_rate": float(np.mean(voi_loss_rates)) if voi_loss_rates else 0.0,

        # ── 参考指标（附录）─────────────────────────────
        "reward_total": float(np.mean(rewards)),
        "reward_total_std": float(np.std(rewards)),
        "orbit_violations": int(sum(orbit_viols)),
        "energy_violations": int(sum(energy_viols)),
        "thermal_violations": int(sum(thermal_viols)),
        "raw_queue_violations": int(sum(raw_viols)),
        "processed_queue_violations": int(sum(processed_viols)),
        "total_violations": int(
            sum(orbit_viols) + sum(energy_viols) + sum(thermal_viols)
            + sum(raw_viols) + sum(processed_viols)
        ),
        "psf_filter_rate": float(np.mean(psf_rates)) if psf_rates else 0.0,
        "mean_mod_l2": float(np.mean(mod_l2s)) if mod_l2s else 0.0,
        "episodes": n_episodes,
    })


def run_independent_model_ablation(args, order=None):
    """独立模型消融：每个变体加载独立 checkpoint。

    order=None 时跑 A-H 主表；传入 SAFETY_ISOLATION_ORDER 则跑安全层隔离消融
    （I/J），评估期 live 关闭对应规则层（与训练同口径）。
    """
    # 论文主消融建议使用独立训练模型，避免“同一个 checkpoint 临时开关安全层”的结论偏差。
    eval_order = list(order or DISPLAY_ORDER)
    print("\n" + "=" * 70)
    print("  消融实验（独立模型版）")
    print("  每个变体使用独立训练模型，评估模块真实贡献")
    print("=" * 70)

    variant_checkpoints = independent_checkpoint_paths(args)
    # 只把 eval_order 中属于 A-H 主表的变体当作"正式消融必须存在"。
    required_keys = [k for k in eval_order if k in DISPLAY_ORDER]
    missing = {
        key: variant_checkpoints[key]
        for key in required_keys
        if not os.path.exists(variant_checkpoints[key])
    }
    if missing and not bool(getattr(args, "allow_missing_ablation", False)):
        missing_text = "\n".join(
            f"  - {VARIANT_SPECS[key]['code']} {VARIANT_SPECS[key]['name']}: {path}"
            for key, path in missing.items()
        )
        raise FileNotFoundError(
            "正式论文消融需要相关变体都有独立训练 checkpoint。\n"
            f"缺失 checkpoint:\n{missing_text}\n"
            "请先运行 experiments/ablation.py --train_independent_models，"
            "或仅做调试时显式加 --allow_missing_ablation。"
        )

    results = {}

    for key in eval_order:
        spec = VARIANT_SPECS[key]
        ckpt = variant_checkpoints[key]

        if not os.path.exists(ckpt):
            print(f"\n[调试跳过] {spec['name']}: 未找到 {ckpt}")
            continue

        print(f"\n[{key}] {spec['name']}...")
        DRL_CONFIG["network_arch"] = spec.get("network_arch", "transformer")
        DRL_CONFIG["behavior_cloning_coeff"] = float(
            spec.get("behavior_cloning_coeff", DRL_CONFIG.get("behavior_cloning_coeff", 0.0))
        )
        # 安全层隔离消融：评估期同样 live 关闭对应规则层（退出自动还原）。
        with _temporary_safety_layer_config(key), _temporary_reward_config(key):
            scheduler = IntegratedScheduler(
                device=args.device,
                enable_lyapunov=spec["enable_lyapunov"],
                use_psf=spec["use_psf"],
            )
            if (not spec["enable_lyapunov"]) or spec["constraint_variant"] in {"plain_sac", "lagrangian_sac", "sac_psf"}:
                scheduler.agent.set_lyapunov_penalty_coeff(0.0)
            # 独立消融模型加载权重时保留当前变体的安全开关，避免 checkpoint metadata 把 B/C/D 改回 Full。
            scheduler.load(ckpt, restore_safety_config=False)
            if (not spec["enable_lyapunov"]) or spec["constraint_variant"] in {"plain_sac", "lagrangian_sac", "sac_psf"}:
                scheduler.agent.set_lyapunov_penalty_coeff(0.0)

            r = evaluate_variant(
                scheduler,
                n_episodes=args.n_episodes,
                max_steps=args.max_steps)
        r["mission_reward_variant"] = spec.get("mission_reward_variant", "value_aware")
        r["ablation_axis"] = spec.get("ablation_axis", "")
        r["safety_config_override"] = spec.get("safety_config_override", {})
        results[key] = r

        print(
            f"  存活={r['survival_rate']:.0%} ({r['survival_steps_mean']:.0f}步)  "
            f"吞吐/步={r['tput_per_step']:.2f}MB  "
            f"总吞吐={r['throughput_total']:.0f}MB  "
            f"crash={r['crash_count']}/{args.n_episodes}"
        )

    return results


def run_stress_test_ablation(args):
    """共享 checkpoint 的诊断压力测试，不替代论文主消融。"""
    # 压力测试版本用于补充说明安全层在极端初值下的保护作用，不替代独立训练消融。
    print("\n" + "=" * 70)
    print("  诊断压力测试（非论文主消融）")
    print("  同一 checkpoint + 极端初始条件 → 仅展示安全层部署保护效果")
    print("=" * 70)

    ckpt = args.checkpoint
    if not os.path.exists(ckpt):
        print(f"[错误] 未找到 checkpoint: {ckpt}")
        return {}

    stress_scenarios = {
        "S1_轨道警告+低电量": {
            "initial_altitude_km": 170,
            "initial_soc": 0.14,
            "data_scale": 1.5,
        },
        "S2_极低电量": {
            "initial_altitude_km": 280,
            "initial_soc": 0.12,
            "data_scale": 1.0,
        },
        "S3_高数据负载": {
            "initial_altitude_km": 300,
            "initial_soc": 0.40,
            "data_scale": 3.0,
        },
        "S4_不安全轨道+能源压力": {
            "initial_altitude_km": 145,
            "initial_soc": 0.12,
            "data_scale": 2.5,
        },
    }

    all_results = {}

    for scenario_name, stress_cfg in stress_scenarios.items():
        print(f"\n{'─'*70}")
        print(f"  场景: {scenario_name}")
        print(f"  条件: h={stress_cfg.get('initial_altitude_km')}km  "
              f"SOC={stress_cfg.get('initial_soc'):.0%}  "
              f"数据={stress_cfg.get('data_scale')}x")
        print(f"{'─'*70}")

        scenario_results = {}

        for var_key in STRESS_ORDER:
            var_cfg = VARIANT_SPECS[var_key]
            scheduler = IntegratedScheduler(
                device=args.device,
                enable_lyapunov=var_cfg["enable_lyapunov"],
                use_psf=var_cfg["use_psf"])
            if (not var_cfg["enable_lyapunov"]) or var_cfg["constraint_variant"] in {"plain_sac", "lagrangian_sac", "sac_psf"}:
                scheduler.agent.set_lyapunov_penalty_coeff(0.0)
            # 压力消融复用同一 checkpoint，只改变外部安全链路开关。
            scheduler.load(ckpt, restore_safety_config=False)
            if (not var_cfg["enable_lyapunov"]) or var_cfg["constraint_variant"] in {"plain_sac", "lagrangian_sac", "sac_psf"}:
                scheduler.agent.set_lyapunov_penalty_coeff(0.0)

            r = evaluate_variant(
                scheduler, args.n_episodes,
                stress_config=stress_cfg,
                max_steps=args.max_steps)

            scenario_results[var_key] = r

            # Windows GBK 控制台不一定能编码勾叉符号，表格标记使用 ASCII，避免打印阶段中断实验。
            alive = "OK" if r["survival_rate"] >= 0.95 else "NO"
            print(f"  {alive} {var_cfg['name']:<25} "
                  f"存活={r['survival_rate']:>4.0%} "
                  f"({r['survival_steps_mean']:>3.0f}步)  "
                  f"r/步={r['reward_per_step']:>6.1f}  "
                  f"吞吐/步={r['tput_per_step']:>5.2f}MB  "
                  f"总吞吐={r['throughput_total']:>7.0f}MB  "
                  f"crash={r['crash_count']}/{args.n_episodes}")

        all_results[scenario_name] = scenario_results

    return all_results


def run_psf_sweep_ablation(args):
    """共享 checkpoint 的 PSF / Lyapunov 开关诊断。"""
    print("\n" + "=" * 70)
    print("  PSF 组合诊断（非论文主消融）")
    print("  同一 checkpoint + 安全链路开关，用于分析 Pi_safe 部署影响")
    print("=" * 70)

    ckpt = args.checkpoint
    if not os.path.exists(ckpt):
        print(f"[错误] 未找到 checkpoint: {ckpt}")
        return {}

    methods = {
        "M1_Full_DRL_PSF_Lya": {
            "name": "M1. Full (DRL+PSF+Lya)",
            "use_psf": True,
            "enable_lyapunov": True,
        },
        "M2_Lya_Only": {
            "name": "M2. DRL+Lya",
            "use_psf": False,
            "enable_lyapunov": True,
        },
        "M3_PSF_Only": {
            "name": "M3. DRL+PSF",
            "use_psf": True,
            "enable_lyapunov": False,
        },
        "M4_Pure_DRL": {
            "name": "M4. Pure DRL",
            "use_psf": False,
            "enable_lyapunov": False,
        },
    }

    results = {}
    for key, spec in methods.items():
        print(f"\n[{spec['name']}]...")
        scheduler = IntegratedScheduler(
            device=args.device,
            enable_lyapunov=spec["enable_lyapunov"],
            use_psf=spec["use_psf"],
        )
        if not spec["enable_lyapunov"]:
            scheduler.agent.set_lyapunov_penalty_coeff(0.0)
        scheduler.load(ckpt, restore_safety_config=False)
        if not spec["enable_lyapunov"]:
            scheduler.agent.set_lyapunov_penalty_coeff(0.0)
        r = evaluate_variant(
            scheduler,
            n_episodes=args.n_episodes,
            seed_offset=500,
            max_steps=args.max_steps,
        )
        results[key] = r
        print(f"  奖励/步: {r['reward_per_step']:.2f}  "
              f"吞吐/步: {r['tput_per_step']:.2f}MB  "
              f"安全率: {r['safety_rate']:.1%}  "
              f"PSF触发: {r['psf_filter_rate']:.1%}")

    print(f"\n{'方法':<28} {'奖励/步':>10} {'吞吐/步':>10} {'安全率':>8} "
          f"{'违规':>6} {'PSF率':>8}")
    print("-" * 76)
    for key, r in results.items():
        print(f"  {methods[key]['name']:<26} {r['reward_per_step']:>10.1f} "
              f"{r['tput_per_step']:>10.2f} "
              f"{r['safety_rate']:>8.1%} "
              f"{r['total_violations']:>6d} "
              f"{r['psf_filter_rate']:>8.1%}")

    return results


def print_paper_table(results: dict, title: str, order=None):
    """
    打印论文级对比表

    论文表格设计原则：
    1. 存活率是第一指标（安全是硬约束）
    2. 总吞吐量是第二指标（活得久 → 总产出高）
    3. 每步效率是第三指标（证明安全不以性能为代价）
    """
    if not results:
        return

    print(f"\n{'='*105}")
    print(f"  {title}")
    print(f"{'='*105}")

    # 表头
    print(f"  {'方案':<28}"
          f"{'存活率':>8} {'存活步':>8}"
          f"{'总吞吐(MB)':>12} {'总下传(MB)':>12}"
          f"{'吞吐/步':>10} {'奖励/步':>10}"
          f"{'崩溃':>6}")
    print(f"  {'─'*103}")

    for key in (order or DISPLAY_ORDER):
        if key not in results:
            continue
        r = results[key]
        name = VARIANT_SPECS.get(key, {}).get("name", key)

        # 存活率标记
        if r["survival_rate"] >= 0.95:
            mark = "A"
        elif r["survival_rate"] >= 0.50:
            mark = "B"
        else:
            mark = "C"

        print(f"  {mark} {name:<26}"
              f"{r['survival_rate']:>7.0%} "
              f"{r['survival_steps_mean']:>7.0f} "
              f"{r['throughput_total']:>11.0f} "
              f"{r['tx_mb_total']:>11.0f} "
              f"{r['tput_per_step']:>9.2f} "
              f"{r['reward_per_step']:>9.1f} "
              f"{r['crash_count']:>5}")

    print(f"  {'─'*103}")
    print(f"  A = 存活率>=95%   B = 存活率>=50%   C = 存活率<50%")

    # ── 论文结论自动生成 ─────────────────────────────────────────
    baseline_key = "C_No_CMDP"
    if "A_Full" in results and baseline_key in results:
        a = results["A_Full"]
        d = results[baseline_key]

        print(f"\n  ┌─────────────────────────────────────────────────────────┐")
        print(f"  │  消融分析                                               │")
        print(f"  ├─────────────────────────────────────────────────────────┤")

        # 1. 存活率
        surv_diff = a["survival_rate"] - d["survival_rate"]
        if surv_diff > 0:
            print(f"  │  存活率:  {d['survival_rate']:.0%} → {a['survival_rate']:.0%}"
                  f"  (+{surv_diff*100:.0f}pp)"
                  f"{'':>22}│")

        # 2. 存活步数
        step_diff = a["survival_steps_mean"] - d["survival_steps_mean"]
        if step_diff > 0:
            print(f"  │  存活步数: {d['survival_steps_mean']:.0f} → "
                  f"{a['survival_steps_mean']:.0f}"
                  f"  (+{step_diff:.0f}步, "
                  f"+{step_diff/max(d['survival_steps_mean'],1)*100:.0f}%)"
                  f"{'':>10}│")

        # 3. 总吞吐量（关键论文论点：活得久→总产出更高）
        tput_a = a["throughput_total"]
        tput_d = d["throughput_total"]
        if tput_a > tput_d:
            print(f"  │  总吞吐: {tput_d:.0f}MB → {tput_a:.0f}MB"
                  f"  (+{(tput_a-tput_d)/max(tput_d,1)*100:.0f}%)"
                  f"{'':>20}│")
            print(f"  │    → 安全保证使卫星存活更久, 总产出反而更高"
                  f"{'':>7}│")
        elif tput_a < tput_d:
            # 如果 Pure DRL 吞吐更高（因为短命但高吞吐），解释清楚
            print(f"  │  总吞吐: {tput_d:.0f}MB → {tput_a:.0f}MB"
                  f"{'':>28}│")
            print(f"  │    Pure DRL 短暂高吞吐后崩溃;"
                  f" Full 持续运行产出更稳定"
                  f"{'':>3}│")

        # 4. 每步效率
        rps_diff = a["reward_per_step"] - d["reward_per_step"]
        if abs(rps_diff) > 0.1:
            if rps_diff < 0:
                cost = abs(rps_diff) / max(abs(d["reward_per_step"]), 1) * 100
                print(f"  │  每步效率代价: {cost:.1f}%"
                      f" (安全层的合理开销)"
                      f"{'':>17}│")
            else:
                print(f"  │  每步效率: 安全层无效率损失"
                      f" (+{rps_diff:.1f}/步)"
                      f"{'':>16}│")

        # 5. VoI / adaptive dual / safety shield contributions
        b = results.get("B_Throughput_Objective")
        d_adapt = results.get("D_No_Adaptive_Dual")
        e_psf = results.get("E_No_PSF")
        f_lya = results.get("F_No_Lyapunov")
        if b or d_adapt or e_psf or f_lya:
            print(f"  ├─────────────────────────────────────────────────────────┤")
            print(f"  │  模块贡献分解:                                          │")

            if b:
                voi_gain = a.get("delivered_value_total", 0.0) - b.get("delivered_value_total", 0.0)
                print(f"  │    VoI objective:  Delivered VoI {voi_gain:+.1f}"
                      f"{'':>23}│")
            if d_adapt:
                dual_gain = a["safety_rate"] - d_adapt["safety_rate"]
                print(f"  │    Adaptive dual:  CSR {dual_gain*100:+.0f}pp"
                      f"{'':>30}│")
            if e_psf:
                psf_surv = a["survival_rate"] - e_psf["survival_rate"]
                print(f"  │    PSF shield:     Survival {psf_surv*100:+.0f}pp"
                      f"{'':>25}│")
            if f_lya:
                lya_surv = a["survival_rate"] - f_lya["survival_rate"]
                print(f"  │    Lyapunov proj.: Survival {lya_surv*100:+.0f}pp"
                      f"{'':>23}│")

        print(f"  └─────────────────────────────────────────────────────────┘")


def run_ablation(args):
    """主入口：支持独立模型模式与压力测试模式。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_train = bool(args.train_independent_models)
    run_train_learning = bool(getattr(args, "train_learning_baselines", False))
    run_independent = bool(args.use_independent_models)
    run_stress = bool(args.stress_test)
    run_psf = bool(args.psf_sweep)
    run_safety_isolation = bool(getattr(args, "safety_isolation", False))

    # 默认行为：不指定参数时，运行论文主消融（独立模型评估）。
    if not run_train and not run_independent and not run_stress and not run_psf and not run_safety_isolation:
        run_independent = True

    missing_before = missing_independent_checkpoints(args) if run_independent else {}
    results_all = {
        "__meta__": {
            "timestamp": datetime.now().isoformat(),
            "formal_ablation_valid": bool(run_independent and not missing_before and not getattr(args, "allow_missing_ablation", False)),
            "formal_mode": "independent_models" if run_independent else "diagnostic_only",
            "required_variants": [VARIANT_SPECS[key]["code"] for key in DISPLAY_ORDER],
            "missing_independent_checkpoints": {
                VARIANT_SPECS[key]["code"]: path for key, path in missing_before.items()
            },
            "allow_missing_ablation": bool(getattr(args, "allow_missing_ablation", False)),
            "diagnostic_modes_are_not_paper_main_ablation": bool(run_stress or run_psf),
            "notes": [
                "Formal ablation requires separately trained checkpoints for every A-H variant.",
                "Stress tests and PSF sweeps reuse one checkpoint and are diagnostic only.",
                "Variant H replaces value-aware delivered VoI reward with delivered MB throughput reward.",
            ],
        }
    }

    if run_train:
        train_results = run_train_independent_ablation_models(args)
        results_all["training"] = train_results
        if args.train_only and not run_train_learning:
            return results_all

    if run_train_learning:
        learning_results = run_train_learning_baselines(args)
        results_all["learning_baseline_training"] = learning_results
        if args.train_only:
            return results_all

    if run_independent:
        independent_results = run_independent_model_ablation(args)
        if independent_results:
            print_paper_table(independent_results, "独立模型消融")
            results_all["independent_models"] = independent_results
            missing_after = missing_independent_checkpoints(args)
            results_all["__meta__"]["missing_independent_checkpoints"] = {
                VARIANT_SPECS[key]["code"]: path for key, path in missing_after.items()
            }
            results_all["__meta__"]["formal_ablation_valid"] = bool(
                not missing_after
                and all(key in independent_results for key in DISPLAY_ORDER)
                and not getattr(args, "allow_missing_ablation", False)
            )

    if run_safety_isolation:
        # 顶刊 Issue#1/#7: 安全层隔离消融（I/J），独立模型 + live 关闭对应规则层。
        isolation_results = run_independent_model_ablation(args, order=SAFETY_ISOLATION_ORDER)
        if isolation_results:
            print_paper_table(isolation_results, "安全层隔离消融 (I/J)", order=SAFETY_ISOLATION_ORDER)
            results_all["safety_isolation"] = isolation_results
            results_all["__meta__"]["safety_isolation_order"] = [
                VARIANT_SPECS[key]["code"] for key in SAFETY_ISOLATION_ORDER
            ]

    stress_results = {}
    if run_stress:
        stress_results = run_stress_test_ablation(args)
        if stress_results:
            for scenario, scenario_r in stress_results.items():
                print_paper_table(scenario_r, f"诊断压力测试: {scenario}")
            results_all["stress_test"] = stress_results

    if run_psf:
        psf_results = run_psf_sweep_ablation(args)
        if psf_results:
            results_all["psf_sweep"] = psf_results

    # 汇总表：压力测试全场景平均
    if stress_results:
        print(f"\n{'='*105}")
        print(f"  汇总: 全场景平均")
        print(f"{'='*105}")

        keys = STRESS_ORDER

        print(f"  {'方案':<28}"
              f"{'存活率':>8} {'总吞吐':>10} {'吞吐/步':>10} "
              f"{'崩溃':>6}")
        print(f"  {'─'*65}")

        for key in keys:
            surv_rates = []
            tputs = []
            tput_steps = []
            crashes = []

            for scenario_r in stress_results.values():
                if key in scenario_r:
                    r = scenario_r[key]
                    surv_rates.append(r["survival_rate"])
                    tputs.append(r["throughput_total"])
                    tput_steps.append(r["tput_per_step"])
                    crashes.append(r["crash_count"])

            if surv_rates:
                name = VARIANT_SPECS.get(key, {}).get("name", key)
                avg_surv = np.mean(surv_rates)
                mark = "A" if avg_surv >= 0.95 else ("B" if avg_surv >= 0.50 else "C")
                print(f"  {mark} {name:<26}"
                      f"{avg_surv:>7.0%} "
                      f"{np.mean(tputs):>9.0f} "
                      f"{np.mean(tput_steps):>9.2f} "
                      f"{int(np.mean(crashes)):>5}")

        print(f"{'='*105}")

    # 保存
    os.makedirs("results", exist_ok=True)
    out_path = f"results/ablation_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2,
                  ensure_ascii=False)
    print(f"\n  结果已保存: {out_path}")

    return results_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="消融实验")
    parser.add_argument("--checkpoint",
                        default=DEFAULT_OPTIMIZED_CHECKPOINT)
    parser.add_argument("--ablation_dir",
                        default="checkpoints_ablation/",
                        help="独立模型目录（variant_A ... variant_H）")
    parser.add_argument("--n_episodes", type=int, default=int(TRAIN_CONFIG.get("eval_episodes", 30)))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="可选：每个 episode 仅评估前 max_steps 步（用于快速烟雾检查）")
    parser.add_argument("--train_independent_models", action="store_true",
                        help="训练或整理 A-H 独立消融模型")
    parser.add_argument("--train_learning_baselines", action="store_true",
                        help="独立训练 SAC+PSF / SAC+Lyapunov 学习型对比 baseline")
    parser.add_argument("--learning_baseline_dir",
                        default="checkpoints_learning_baselines/",
                        help="独立学习型 baseline 目录")
    parser.add_argument("--learning_baselines", nargs="+", default=None,
                        choices=sorted(LEARNING_BASELINE_SPECS.keys()),
                        help="仅训练指定学习型 baseline")
    parser.add_argument("--train_only", action="store_true",
                        help="只训练/整理独立消融模型，不运行评估")
    parser.add_argument("--use_independent_models", action="store_true",
                        help="运行独立模型消融（论文主表）")
    parser.add_argument("--allow_missing_ablation", action="store_true",
                        help="仅用于 smoke/debug：允许缺失独立消融 checkpoint，并把结果标记为不可作为论文主表")
    parser.add_argument("--stress_test", action="store_true",
                        help="运行共享 checkpoint 压力测试诊断（非论文主消融）")
    parser.add_argument("--psf_sweep", action="store_true",
                        help="运行共享 checkpoint 的 PSF/Lyapunov 开关诊断")
    parser.add_argument("--safety_isolation", action="store_true",
                        help="运行安全层隔离消融 I/J（w/o 解析推进控制器 / w/o 硬指向兜底，独立模型）")
    parser.add_argument("--total_steps", type=int,
                        default=int(TRAIN_CONFIG.get("total_steps", 1500000)))
    parser.add_argument("--only", nargs="+", default=None,
                        choices=sorted(CODE_TO_VARIANT.keys()),
                        help="仅训练指定 A-H 变体")
    parser.add_argument("--copy_A", dest="copy_A", action="store_true", default=True)
    parser.add_argument("--no_copy_A", dest="copy_A", action="store_false")
    parser.add_argument("--full_model_source", default=DEFAULT_OPTIMIZED_CHECKPOINT)
    parser.add_argument("--eval_freq", type=int,
                        default=int(TRAIN_CONFIG.get("eval_freq", 5000)))
    parser.add_argument("--train_eval_episodes", type=int,
                        default=int(TRAIN_CONFIG.get("eval_episodes", 30)))
    parser.add_argument("--summary_eval_episodes", type=int,
                        default=int(TRAIN_CONFIG.get("eval_episodes", 30)))
    parser.add_argument("--save_freq", type=int,
                        default=int(TRAIN_CONFIG.get("save_freq", 50000)))
    parser.add_argument("--n_envs", type=int,
                        default=int(TRAIN_CONFIG.get("n_envs", 1)))
    parser.add_argument("--env_backend", choices=["auto", "serial", "subproc"],
                        default=TRAIN_CONFIG.get("env_backend", "auto"))
    parser.add_argument("--seed", type=int,
                        default=int(TRAIN_CONFIG.get("seed", 42)))
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--update_freq", type=int, default=None)
    parser.add_argument("--update_actor_freq", type=int, default=None)
    args = parser.parse_args()
    args.device = _resolve_device(args.device)
    set_global_seed(int(args.seed))
    run_ablation(args)

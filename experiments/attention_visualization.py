"""
可视化 Transformer 注意力与 当前状态演化。
"""

import argparse
import os
import sys

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    def _setup_font():
        candidates = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        from matplotlib import font_manager

        available = {font.name for font in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in available:
                matplotlib.rcParams["font.sans-serif"] = [name]
                matplotlib.rcParams["axes.unicode_minus"] = False
                break

    _setup_font()
    plt.rcParams.update({"figure.dpi": 150, "axes.grid": False})
    MPL_OK = True
except ImportError:
    MPL_OK = False

from config import DRL_CONFIG, TRAIN_CONFIG
from environment.satellite_env import OBSERVATION_FEATURES, VLEOSatelliteEnv
from environment.wrappers import DilatedFrameStackWrapper
from scheduler.integrated_scheduler import IntegratedScheduler

DEFAULT_OPTIMIZED_CHECKPOINT = os.path.join(
    TRAIN_CONFIG.get("optimized_checkpoint_dir", "checkpoints_optimized/"),
    "best_optimized.pt",
)


STATE_LABELS = list(OBSERVATION_FEATURES)
STATE_DIM = len(STATE_LABELS)
_FEATURE_INDEX = {name: idx for idx, name in enumerate(OBSERVATION_FEATURES)}


def _idx(name: str) -> int:
    return _FEATURE_INDEX[name]


FEATURE_GROUPS = {
    "Orbit": ([
        _idx("altitude_norm"), _idx("drag_strength_norm"),
        _idx("altitude_safety_margin_norm"), _idx("orbit_queue_pressure"),
        _idx("prop_update_phase"),
    ], "#E6F1FB"),
    "Energy": ([
        _idx("soc"), _idx("solar_input_norm"), _idx("last_total_power_norm"),
        _idx("energy_queue_pressure"), _idx("thermal_margin_norm"),
    ], "#EAF3DE"),
    "Actions": ([
        _idx("prev_alpha_prop"), _idx("prev_alpha_cpu"), _idx("prev_alpha_tx"),
    ], "#FAF0E6"),
    "Comm": ([
        _idx("in_comm_window"), _idx("time_to_next_window_norm"),
        _idx("window_remaining_norm"), _idx("tx_capacity_norm"),
        _idx("processed_queue_utilization"), _idx("processed_queue_pressure"),
        _idx("future_contact_capacity_norm"), _idx("next_window_in_range"),
    ], "#F5E6FA"),
    "Task": ([
        _idx("raw_queue_utilization"), _idx("raw_high_queue_utilization"),
        _idx("raw_mid_queue_utilization"), _idx("raw_low_queue_utilization"),
        _idx("processed_high_queue_utilization"), _idx("processed_mid_queue_utilization"),
        _idx("processed_low_queue_utilization"), _idx("expiring_high_value_norm"),
        _idx("expiring_mid_value_norm"), _idx("expiring_low_value_norm"),
        _idx("total_processed_value_norm"), _idx("topk_priority_norm"),
        _idx("topk_quality_norm"), _idx("deadline_urgency"),
        _idx("current_scene_class_norm"), _idx("upcoming_task_intensity_norm"),
        _idx("cpu_backpressure_ratio"),
    ], "#FDECEC"),
}


def extract_attention_weights(encoder, state_seq: torch.Tensor) -> np.ndarray | None:
    """
    提取最后一层编码器的注意力图，输出形状为 (T, T)，
    并在多头之间做平均，便于直接画热力图。
    """

    attention_maps = []

    def hook_fn(module, inputs, output):
        if not hasattr(module, "self_attn"):
            return
        with torch.no_grad():
            q = k = v = inputs[0]
            _, attn_weights = module.self_attn(
                q, k, v, need_weights=True, average_attn_weights=False
            )
            attention_maps.append(attn_weights.detach().cpu().numpy())

    hooks = [layer.register_forward_hook(hook_fn) for layer in encoder.transformer.layers]
    try:
        with torch.no_grad():
            encoder(state_seq)
    finally:
        for hook in hooks:
            hook.remove()

    if not attention_maps:
        return None

    attn = attention_maps[-1][0]  # (heads, T, T)
    return attn.mean(axis=0)


def _safe_in_window(env) -> bool:
    contact = getattr(env, "_contact", None)
    if contact is None:
        return False
    return bool(contact.get("in_window", False))


def collect_episode_data(checkpoint_path: str, n_steps: int = 540, seed: int = 0):
    base_env = VLEOSatelliteEnv(seed=seed)
    env = DilatedFrameStackWrapper(base_env, k=int(DRL_CONFIG.get("frame_stack", 8)))

    scheduler = IntegratedScheduler(device="cpu")
    if os.path.exists(checkpoint_path):
        scheduler.load(checkpoint_path)
        print(f"  已加载 checkpoint: {checkpoint_path}")
    else:
        print(f"  未找到 checkpoint，改用随机初始化权重: {checkpoint_path}")

    encoder = scheduler.agent.actor.encoder
    encoder.eval()

    states = []
    attentions = []
    sunlit_seq = []
    window_seq = []
    altitude_seq = []
    soc_seq = []

    state = env.reset()
    for step in range(n_steps):
        # Dilated wrapper 的帧顺序是“当前 -> 更早历史”，所以索引 0
        # 才是这个时刻真正要拿来展示的当前状态。
        current_state = np.asarray(state[0], dtype=np.float32)
        states.append(current_state)
        sunlit_seq.append(float(current_state[_idx("solar_input_norm")]))
        window_seq.append(float(current_state[_idx("in_comm_window")]))
        altitude_seq.append(float(current_state[_idx("altitude_norm")]))
        soc_seq.append(float(current_state[_idx("soc")]))

        state_tensor = torch.tensor(
            state[None, ...], dtype=torch.float32, device="cpu"
        )
        # 注意力必须从 Actor 真正看到的时序堆叠输入里提取，
        # 不能只给单个 40 维状态，否则就不是 Transformer 的真实关注模式。
        attn = extract_attention_weights(encoder, state_tensor)
        if attn is not None:
            attentions.append(attn)

        in_window = _safe_in_window(env)
        prop_can_update = True
        if hasattr(env, "step_count") and hasattr(env, "N_PROP_SMOOTH"):
            prop_can_update = (env.step_count % env.N_PROP_SMOOTH == 0)

        action, _, _, _ = scheduler.schedule(
            state,
            env.energy_queue.value,
            env.orbit_queue.value,
            env.data_queue.length,
            env.comm_queue.value,
            in_window=in_window,
            evaluate=True,
            h=env.altitude_m,
            soc=env.battery.soc,
            time_s=env.time_s,
            prop_can_update=prop_can_update,
            orbital_phase=env.orbit_sim.phase,
            tx_capacity_mbps=float((env._contact or {}).get("max_capacity_mbps", 0.0)),
            available_power_w=getattr(env, "available_power_w", None),
            env=env,
        )
        state, _, done, _ = env.step(action, enforce_prop_smoothing=False)
        if done:
            break

        if (step + 1) % 50 == 0:
            print(f"  已采集 {step + 1} 步")

    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(attentions, dtype=np.float32) if attentions else None,
        np.asarray(sunlit_seq, dtype=np.float32),
        np.asarray(window_seq, dtype=np.float32),
        np.asarray(altitude_seq, dtype=np.float32),
        np.asarray(soc_seq, dtype=np.float32),
    )


def _feature_variability(states: np.ndarray) -> np.ndarray:
    if len(states) < 2:
        return np.zeros_like(states)
    # 某些模型/检查点如果拿不到注意力图，就退化成“特征变化强度”代理图，
    # 这样脚本仍然能输出可解释的结果，而不是直接报废。
    diffs = np.abs(np.diff(states, axis=0, prepend=states[[0]]))
    max_per_feature = np.maximum(diffs.max(axis=0, keepdims=True), 1e-6)
    return diffs / max_per_feature


def plot_attention_analysis(
    states: np.ndarray,
    attentions: np.ndarray | None,
    sunlit: np.ndarray,
    window: np.ndarray,
    altitude: np.ndarray,
    soc: np.ndarray,
    save_path: str,
):
    if not MPL_OK:
        raise RuntimeError("注意力可视化需要 matplotlib")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    t = np.arange(len(states)) * TRAIN_CONFIG.get("time_slot_s", 10) / 60.0
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1.1, 1.6, 1.6, 1.2], hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, altitude, color="tab:blue", linewidth=1.3, label="Altitude")
    ax1.plot(t, soc, color="tab:green", linewidth=1.1, alpha=0.85, label="SOC")
    ax1.fill_between(t, 0, sunlit, color="#F8E27A", alpha=0.18, label="Sunlit")
    ax1.fill_between(t, 0, window, color="#B9A4F3", alpha=0.14, label="Contact window")
    ax1.set_xlim(0, t[-1] if len(t) else 1.0)
    ax1.set_title("(a) Episode overview")
    ax1.set_xlabel("Time (min)")
    ax1.legend(loc="upper right", ncol=4, fontsize=8)

    ax2 = fig.add_subplot(gs[1, 0])
    state_heat = states.T
    im2 = ax2.imshow(
        state_heat,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        extent=[0, t[-1] if len(t) else 1.0, STATE_DIM - 0.5, -0.5],
    )
    ax2.set_yticks(range(STATE_DIM))
    ax2.set_yticklabels(STATE_LABELS, fontsize=8)
    ax2.set_xlabel("Time (min)")
    ax2.set_title(f"(b) State evolution ({STATE_DIM}-D state)")
    plt.colorbar(im2, ax=ax2, shrink=0.7, label="Normalized state value")

    ax3 = fig.add_subplot(gs[2, 0])
    if attentions is not None and len(attentions) > 0:
        attn_map = attentions[-1]
        im3 = ax3.imshow(attn_map, aspect="auto", cmap="magma", interpolation="nearest")
        ax3.set_title("(c) Last-layer self-attention (head-averaged)")
        ax3.set_xlabel("Key token")
        ax3.set_ylabel("Query token")
        plt.colorbar(im3, ax=ax3, shrink=0.7, label="Attention weight")
    else:
        variability = _feature_variability(states)
        im3 = ax3.imshow(
            variability.T,
            aspect="auto",
            cmap="hot",
            interpolation="nearest",
            extent=[0, t[-1] if len(t) else 1.0, STATE_DIM - 0.5, -0.5],
        )
        ax3.set_yticks(range(STATE_DIM))
        ax3.set_yticklabels(STATE_LABELS, fontsize=8)
        ax3.set_title("(c) Feature variability proxy")
        ax3.set_xlabel("Time (min)")
        plt.colorbar(im3, ax=ax3, shrink=0.7, label="Relative variability")

    ax4 = fig.add_subplot(gs[3, 0])
    comm_feature_names = [
        "in_comm_window",
        "time_to_next_window_norm",
        "window_remaining_norm",
        "processed_queue_utilization",
        "processed_queue_pressure",
        "prop_update_phase",
        "future_contact_capacity_norm",
        "cpu_backpressure_ratio",
        "next_window_in_range",
    ]
    comm_feature_idx = [_idx(name) for name in comm_feature_names]
    comm_features = states[:, comm_feature_idx]
    labels4 = [
        "In_window", "T_to_window", "Window_remain", "Processed_Q",
        "Processed_pressure", "Prop_phase", "Future_contact",
        "CPU_backpressure", "Next_window_known",
    ]
    colors4 = ["gold", "darkorange", "green", "purple", "teal", "steelblue", "crimson", "black", "slategray"]
    for idx, (label, color) in enumerate(zip(labels4, colors4)):
        values = comm_features[:, idx]
        value_range = values.max() - values.min()
        if value_range > 1e-6:
            values = (values - values.min()) / value_range
        ax4.plot(t, values, color=color, linewidth=1.2, alpha=0.85, label=label)
    ax4.set_xlim(0, t[-1] if len(t) else 1.0)
    ax4.set_xlabel("Time (min)")
    ax4.set_ylabel("Normalized value")
    ax4.set_title("(d) Communication and action-phase features")
    ax4.legend(fontsize=8, ncol=3, loc="upper right")

    fig.suptitle("Transformer 注意力 / 状态演化可视化", fontsize=14)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  图像已保存: {save_path}")


def run_attention_viz(args):
    if not MPL_OK:
        raise RuntimeError("当前环境未安装 matplotlib")

    print("=" * 55)
    print("  Transformer 注意力 / 状态分析")
    print("=" * 55)

    states, attentions, sunlit, window, altitude, soc = collect_episode_data(
        args.checkpoint, n_steps=args.n_steps, seed=args.seed
    )

    save_path = os.path.join(args.output_dir, "attention_analysis.png")
    plot_attention_analysis(
        states=states,
        attentions=attentions,
        sunlit=sunlit,
        window=window,
        altitude=altitude,
        soc=soc,
        save_path=save_path,
    )

    print("\n  已完成。")
    print(f"  - 图 (b) 使用统一后的 {STATE_DIM} 维当前状态标签。")
    print("  - 图 (d) 已显式包含 Prop_phase。")
    if attentions is not None and len(attentions) > 0:
        print("  - 图 (c) 使用 Transformer 注意力图。")
    else:
        print("  - 图 (c) 未取到注意力图，已退化为特征变化强度代理图。")


def build_parser():
    parser = argparse.ArgumentParser(
        description="可视化 Transformer 注意力与 当前状态演化。"
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_OPTIMIZED_CHECKPOINT,
        help="要加载的 checkpoint 路径。",
    )
    parser.add_argument(
        "--output_dir",
        default="figures/paper/",
        help="输出图像目录。",
    )
    parser.add_argument("--n_steps", type=int, default=540)
    parser.add_argument("--seed", type=int, default=int(TRAIN_CONFIG.get("seed", 42)))
    return parser


if __name__ == "__main__":
    run_attention_viz(build_parser().parse_args())

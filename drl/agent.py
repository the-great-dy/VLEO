"""
SAC 训练基础设施与 checkpoint 管理。
SAC 训练基础设施。

这里保留通用训练机制：网络、经验回放、状态归一化、AMP、NaN guard、
学习率调度和 checkpoint。论文算法目标函数放在
algorithms/decoupled_constraint_sac.py，避免方法贡献和工程保护混在一起。
"""

import sys as _sys, os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in _sys.path:
    # 降低导入劫持风险：不要把项目路径插到最高优先级
    _sys.path.append(_PROJECT_ROOT)
import os
import numpy as np
import torch
import torch.nn.functional as F
from config import DRL_CONFIG, TRAIN_CONFIG
from drl.networks import Actor, Critic
from drl.replay_buffer import ReplayBuffer, TransitionBoundary
from utils.sanitizers import sanitize_scalar


def _safe_torch_load(path: str, map_location):
    """优先使用 weights_only，默认禁用不安全回退。"""
    # 默认走 weights_only，防止从不可信 checkpoint 反序列化任意 Python 对象。
    allow_unsafe_fallback = str(os.environ.get("ALLOW_UNSAFE_TORCH_LOAD", "")).strip().lower() in {
        "1", "true", "yes", "y", "on"
    }
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        if allow_unsafe_fallback:
            print("[WARN] torch.load(weights_only=True) 不可用，已按 ALLOW_UNSAFE_TORCH_LOAD 回退为不安全加载。")
            return torch.load(path, map_location=map_location)
        raise RuntimeError(
            "当前 PyTorch 版本不支持 weights_only 安全加载。"
            "如需兼容旧检查点，可设置环境变量 ALLOW_UNSAFE_TORCH_LOAD=1 后重试。"
        ) from exc
    except Exception as exc:
        if allow_unsafe_fallback:
            print(f"[WARN] 安全加载失败({exc})，已按 ALLOW_UNSAFE_TORCH_LOAD 回退为不安全加载。")
            return torch.load(path, map_location=map_location)
        raise RuntimeError(
            "检查点安全加载失败，默认已阻止不安全反序列化。"
            "如确认来源可信，可设置 ALLOW_UNSAFE_TORCH_LOAD=1 后重试。"
        ) from exc


# ── GPU 检测 ──────────────────────────────────────────────────────────
def _cuda_available() -> bool:
    """安全检测 CUDA 是否真正可用（包含编译支持检查）"""
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def auto_device() -> str:
    """自动选择最优计算设备，CUDA 不可用时优雅降级"""
    if _cuda_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and \
       torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class RunningMeanStd:
    """OpenAI Gym 风格的 RunningMeanStd，用于在线估计观测均值和方差。"""

    def __init__(self, shape, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        x = x.reshape(-1, *self.mean.shape)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count: int) -> None:
        if batch_count <= 0:
            return
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = np.maximum(m2 / total_count, 1e-12)
        self.count = float(total_count)

    def normalize_np(self, x: np.ndarray, clip: float = 5.0,
                     epsilon: float = 1e-8) -> np.ndarray:
        y = (np.asarray(x, dtype=np.float32) - self.mean.astype(np.float32)) / np.sqrt(
            self.var.astype(np.float32) + float(epsilon)
        )
        return np.clip(y, -float(clip), float(clip)).astype(np.float32)

    def normalize_torch(self, x: torch.Tensor, clip: float = 5.0,
                        epsilon: float = 1e-8) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        var = torch.as_tensor(self.var, dtype=x.dtype, device=x.device)
        y = (x - mean) / torch.sqrt(var + float(epsilon))
        return torch.clamp(y, -float(clip), float(clip))

    def state_dict(self) -> dict:
        return {
            "mean": torch.as_tensor(self.mean, dtype=torch.float32),
            "var": torch.as_tensor(self.var, dtype=torch.float32),
            "count": float(self.count),
        }

    def load_state_dict(self, state: dict) -> None:
        def _to_numpy(v):
            if torch.is_tensor(v):
                return v.detach().cpu().numpy()
            return np.asarray(v)

        self.mean = _to_numpy(state.get("mean", self.mean)).astype(np.float64)
        self.var = np.maximum(_to_numpy(state.get("var", self.var)).astype(np.float64), 1e-12)
        self.count = float(state.get("count", self.count))


# ══════════════════════════════════════════════════════════════════════
# SAC 智能体
# ══════════════════════════════════════════════════════════════════════
class SACAgent:
    """
    SAC training harness.

    该类不再作为论文算法名出现；调度器实际实例化的是
    DecoupledConstraintSAC。保留 SACAgent 是为了复用训练基础设施和兼容旧测试。
    """
    def __init__(self, state_dim: int = 40,
                 action_dim: int | None = None,
                 device: str = "auto"):

        self.state_dim  = state_dim
        self.action_dim = int(action_dim or DRL_CONFIG.get("action_dim", 10))
        self.device     = torch.device(
            auto_device() if device == "auto" else device)
        print(f"[SACAgent] 使用设备: {self.device}")

        # ── 超参数 ──────────────────────────────────────────────────
        cfg = DRL_CONFIG
        self.gamma      = cfg["gamma"]
        self.tau        = cfg["tau"]
        self.batch_size = cfg["batch_size"]
        self.warmup     = cfg["warmup_steps"]
        self.lya_coeff  = max(0.0, float(cfg["lyapunov_penalty_coeff"]))
        self.grad_clip  = float(cfg.get("gradient_clip", 10.0))
        self.update_actor_freq = max(1, int(cfg.get("update_actor_freq", 1)))
        self.nan_guard_enable = bool(cfg.get("nan_guard_enable", True))
        self.alpha_log_clip = float(cfg.get("alpha_log_clip", 10.0))
        self.alpha_min = max(0.0, float(cfg.get("alpha_min", 0.0)))
        self.behavior_cloning_coeff = max(0.0, float(cfg.get("behavior_cloning_coeff", 0.0)))
        self.behavior_cloning_max_weight = max(0.0, float(cfg.get("behavior_cloning_max_weight", 1.0)))
        self.value_aux_head_enable = bool(cfg.get("value_aux_head_enable", False))
        self.value_aux_loss_weight = max(0.0, float(cfg.get("value_aux_loss_weight", 0.0)))
        self.value_aux_loss_weight_final = max(
            0.0, float(cfg.get("value_aux_loss_weight_final", self.value_aux_loss_weight))
        )
        self.value_action_aux_loss_weight = max(
            0.0, float(cfg.get("value_action_aux_loss_weight", 0.0)))
        self.value_action_aux_loss_weight_final = max(
            0.0,
            float(cfg.get(
                "value_action_aux_loss_weight_final",
                self.value_action_aux_loss_weight,
            )),
        )
        self.value_aux_weight_decay_steps = max(
            1, int(cfg.get("value_aux_weight_decay_steps", 1))
        )
        self.value_aux_expiring_high_weight = max(
            0.0, float(cfg.get("value_aux_expiring_high_weight", 1.25)))
        self.value_aux_high_pressure_margin = max(
            0.0, float(cfg.get("value_aux_high_pressure_margin", 1.10)))
        self.value_aux_expiring_high_threshold = max(
            0.0, float(cfg.get("value_aux_expiring_high_threshold", 0.10)))
        self.value_aux_low_pressure_margin = max(
            0.0, float(cfg.get("value_aux_low_pressure_margin", 1.25)))
        self.value_aux_future_contact_tight_threshold = float(
            cfg.get("value_aux_future_contact_tight_threshold", 0.25))
        self.value_aux_future_contact_relaxed_threshold = float(
            cfg.get("value_aux_future_contact_relaxed_threshold", 0.55))
        self.value_aux_processed_future_contact_threshold = float(cfg.get(
            "value_aux_processed_future_contact_threshold",
            cfg.get("value_aux_processed_pressure_threshold", 0.75),
        ))
        # 兼容旧 checkpoints/scripts。
        self.value_aux_processed_pressure_threshold = (
            self.value_aux_processed_future_contact_threshold
        )
        self.use_state_normalization = bool(cfg.get("state_normalization", True))
        self.state_norm_clip = max(1.0, float(cfg.get("state_norm_clip", 5.0)))
        self.state_norm_epsilon = max(1e-12, float(cfg.get("state_norm_epsilon", 1e-4)))
        self.frame_stack = int(cfg.get("frame_stack", 1))
        self.nan_guard_hits = 0
        self.update_steps = 0
        self.total_steps = 0
        self.state_rms = RunningMeanStd(
            shape=(state_dim,),
            epsilon=self.state_norm_epsilon,
        )

        # ── 网络初始化（backbone 由 DRL_CONFIG["network_arch"] 指定） ─────────
        hidden = cfg["hidden_dim"]
        self.actor   = Actor(state_dim, self.action_dim, hidden).to(self.device)
        self.critic  = Critic(state_dim, self.action_dim, hidden).to(self.device)
        self.critic_target = Critic(state_dim, self.action_dim, hidden).to(self.device)
        # Reward Q、Deliverable Q 和约束/漂移 Q 分开估计，避免不同时间尺度信号互相污染。
        self.deliverable_critic = Critic(state_dim, self.action_dim, hidden).to(self.device)
        self.deliverable_critic_target = Critic(state_dim, self.action_dim, hidden).to(self.device)
        self.constraint_critic = Critic(state_dim, self.action_dim, hidden).to(self.device)
        self.constraint_critic_target = Critic(state_dim, self.action_dim, hidden).to(self.device)
        
        # 同步目标网络权重
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.deliverable_critic_target.load_state_dict(self.deliverable_critic.state_dict())
        self.constraint_critic_target.load_state_dict(
            self.constraint_critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False
        for p in self.deliverable_critic_target.parameters():
            p.requires_grad = False
        for p in self.constraint_critic_target.parameters():
            p.requires_grad = False
        # target critic 提供稳定 TD 目标，必须关闭 Dropout。
        self.critic_target.eval()
        self.deliverable_critic_target.eval()
        self.constraint_critic_target.eval()

        # ── 优化器 ──────────────────────────────────────────────────
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=cfg["lr_actor"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg["lr_critic"])
        self.deliverable_critic_opt = torch.optim.Adam(
            self.deliverable_critic.parameters(), lr=cfg["lr_critic"])
        self.constraint_critic_opt = torch.optim.Adam(
            self.constraint_critic.parameters(), lr=cfg["lr_critic"])

        # 自动熵系数 (Temperature alpha) 调节
        self.target_entropy = -self.action_dim * cfg["target_entropy_scale"]
        self.log_alpha      = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha          = self.log_alpha.exp().item()
        self.alpha_opt      = torch.optim.Adam([self.log_alpha], lr=cfg["lr_alpha"])

        # ── 学习率调度器 ──────────────────────────────────────────
        # 支持三种调度方式: constant | cosine | exponential
        self.lr_schedule_type = cfg.get("lr_schedule_type", "constant")
        self.lr_actor_init = cfg["lr_actor"]
        self.lr_critic_init = cfg["lr_critic"]
        self.lr_alpha_init = cfg["lr_alpha"]
        self.total_train_steps = TRAIN_CONFIG.get("total_steps", int(1e6))
        
        if self.lr_schedule_type == "cosine":
            # CosineAnnealingLR 的周期使用总更新次数，而不是环境交互步数。
            # 计算公式：total_updates = total_steps / update_freq
            update_freq = cfg.get("update_freq", 1)
            total_updates = max(1, self.total_train_steps // update_freq)
            
            # 余弦退火调度
            self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.actor_opt, 
                T_max=total_updates,
                eta_min=self.lr_actor_init * cfg.get("lr_min_scale", 0.01)
            )
            self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.critic_opt,
                T_max=total_updates,
                eta_min=self.lr_critic_init * cfg.get("lr_min_scale", 0.01)
            )
            self.deliverable_critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.deliverable_critic_opt,
                T_max=total_updates,
                eta_min=self.lr_critic_init * cfg.get("lr_min_scale", 0.01)
            )
            self.constraint_critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.constraint_critic_opt,
                T_max=total_updates,
                eta_min=self.lr_critic_init * cfg.get("lr_min_scale", 0.01)
            )
        elif self.lr_schedule_type == "exponential":
            # 指数衰减
            self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.actor_opt,
                gamma=0.9999
            )
            self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.critic_opt,
                gamma=0.9999
            )
            self.deliverable_critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.deliverable_critic_opt,
                gamma=0.9999
            )
            self.constraint_critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.constraint_critic_opt,
                gamma=0.9999
            )
        else:
            self.actor_scheduler = None
            self.critic_scheduler = None
            self.deliverable_critic_scheduler = None
            self.constraint_critic_scheduler = None
            
        # 学习率预热（前warmup_steps缓慢增加学习率）
        self.use_lr_warmup = TRAIN_CONFIG.get("use_lr_warmup", False)
        self.lr_warmup_steps = TRAIN_CONFIG.get("lr_warmup_steps", 0)
        self.lr_warmup_init_scale = TRAIN_CONFIG.get("lr_warmup_init_scale", 0.1)

        # ── 经验回放与混合精度 ──────────────────────────────────────
        self.buffer = ReplayBuffer(
            cfg["buffer_size"],
            state_dim,
            self.action_dim,
            frame_stack=self.frame_stack,
        )

        # n-step aggregator（默认开启；n=1 时与单步等价）。
        try:
            from config import N_STEP_CONFIG
            _n_step_cfg = N_STEP_CONFIG
        except ImportError:
            _n_step_cfg = {"enabled": False, "n": 1}
        self._n_step_enabled = bool(_n_step_cfg.get("enabled", False))
        self._n_step_n = max(1, int(_n_step_cfg.get("n", 1)))
        if self._n_step_enabled and self._n_step_n > 1:
            from drl.n_step_aggregator import NStepAggregator
            self._NStepAggClass = NStepAggregator
        else:
            self._NStepAggClass = None
        self._n_step_aggs: dict = {}  # env_id -> NStepAggregator（每个并行 env 独立）

        requested_amp = bool(cfg.get("use_amp", True))
        self.use_amp = bool(requested_amp and self.device.type == "cuda" and _cuda_available())
        if requested_amp and not self.use_amp and self.device.type == "cuda":
            print("[SACAgent] AMP 请求已启用，但当前CUDA环境不可用，自动回退为 FP32")
        if not requested_amp:
            print("[SACAgent] AMP 已按配置关闭（数值稳定优先）")
        # 使用 torch.amp API，兼容当前 PyTorch 的 AMP 调用方式。
        self.scaler  = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _trigger_nan_guard(self, stage_code: int, detail: str = "") -> dict:
        """发现非有限值时统一记录并返回可序列化统计。"""
        self.nan_guard_hits += 1
        extra = f" | {detail}" if detail else ""
        print(
            f"[SACAgent][NaNGuard] step={self.total_steps:,} "
            f"stage={stage_code} hits={self.nan_guard_hits}{extra}"
        )
        safe_alpha = float(self.alpha) if np.isfinite(self.alpha) else 1.0
        return {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "deliverable_critic_loss": 0.0,
            "constraint_critic_loss": 0.0,
            "constraint_actor_loss": 0.0,
            "lyapunov_penalty_coeff": float(max(self.lya_coeff, 0.0)),
            "alpha": safe_alpha,
            "actor_lr": self.actor_opt.param_groups[0]["lr"],
            "critic_lr": self.critic_opt.param_groups[0]["lr"],
            "nan_guard_triggered": 1.0,
            "nan_guard_stage": float(stage_code),
            "nan_guard_hits": float(self.nan_guard_hits),
        }

    @staticmethod
    def _all_finite(*tensors) -> bool:
        return all(torch.isfinite(t).all().item() for t in tensors)

    def _compute_td_targets(
        self,
        r: torch.Tensor,
        d: torch.Tensor,
        lya: torch.Tensor,
        deliverable_r: torch.Tensor,
        reward_next_q: torch.Tensor,
        deliverable_next_q: torch.Tensor,
        constraint_next_q: torch.Tensor,
        gamma_pow: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """兼容旧调用；实际公式由算法类 compute_td_targets 定义。"""
        return self.compute_td_targets(
            r, d, lya, deliverable_r,
            reward_next_q, deliverable_next_q, constraint_next_q,
            gamma_pow=gamma_pow,
        )

    def compute_td_targets(
        self,
        r: torch.Tensor,
        d: torch.Tensor,
        lya: torch.Tensor,
        deliverable_r: torch.Tensor,
        reward_next_q: torch.Tensor,
        deliverable_next_q: torch.Tensor,
        constraint_next_q: torch.Tensor,
        gamma_pow: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """默认 TD 目标；DecoupledConstraintSAC 会显式覆盖该公式。

        gamma_pow=None → 单步：target = r + (1-d)*γ*Q'
        gamma_pow tensor → n-step：target = R_n + (1-d)*γ^n*Q'   (R_n 已经在 r 中累计)
        """
        g = gamma_pow if gamma_pow is not None else float(self.gamma)
        reward_target = r + (1 - d) * g * reward_next_q
        deliverable_target = deliverable_r + (1 - d) * g * deliverable_next_q
        constraint_target = lya + (1 - d) * g * constraint_next_q
        return reward_target, deliverable_target, constraint_target

    @property
    def _deliverable_critic_enabled(self) -> bool:
        return bool(DRL_CONFIG.get("deliverable_critic_enable", False))

    @property
    def _deliverable_critic_actor_coeff(self) -> float:
        return float(DRL_CONFIG.get("deliverable_critic_actor_coeff", 0.0))

    def compute_actor_objective(
        self,
        normalized_states: torch.Tensor,
        behavior_actions: torch.Tensor,
        behavior_weights: torch.Tensor,
        raw_states: torch.Tensor | None = None,
    ) -> dict:
        """默认 actor 目标；论文算法类覆盖该方法。"""
        a_new, log_pi, mean_action = self.actor.sample(normalized_states)
        q1_new, q2_new = self.critic(normalized_states, a_new)
        reward_q_new = torch.min(q1_new, q2_new)
        sac_reward = reward_q_new
        if self._deliverable_critic_enabled:
            d1_new, d2_new = self.deliverable_critic(normalized_states, a_new)
            sac_reward = sac_reward + self._deliverable_critic_actor_coeff * torch.min(d1_new, d2_new)
        c1_new, c2_new = self.constraint_critic(normalized_states, a_new)
        constraint_q_new = torch.max(c1_new, c2_new)
        sac_actor_loss = (self.alpha * log_pi - sac_reward).mean()
        if self.lya_coeff > 0.0:
            constraint_actor_loss = self.lya_coeff * constraint_q_new.mean()
        else:
            constraint_actor_loss = torch.zeros(
                (), device=self.device, dtype=sac_actor_loss.dtype)
        behavior_cloning_loss = torch.zeros(
            (), device=self.device, dtype=sac_actor_loss.dtype)
        behavior_w_clamped = torch.clamp(
            behavior_weights, 0.0, self.behavior_cloning_max_weight)
        if self.behavior_cloning_coeff > 0.0 and self.behavior_cloning_max_weight > 0.0:
            weight_sum = behavior_w_clamped.sum()
            if weight_sum.item() > 1e-8:
                per_sample_bc = F.mse_loss(
                    mean_action, behavior_actions, reduction="none"
                ).mean(dim=-1, keepdim=True)
                behavior_cloning_loss = (
                    per_sample_bc * behavior_w_clamped
                ).sum() / torch.clamp(weight_sum, min=1e-8)
        actor_loss = (
            sac_actor_loss
            + constraint_actor_loss
            + self.behavior_cloning_coeff * behavior_cloning_loss
        )
        return {
            "actor_loss": actor_loss,
            "sac_actor_loss": sac_actor_loss,
            "constraint_actor_loss": constraint_actor_loss,
            "behavior_cloning_loss": behavior_cloning_loss,
            "behavior_w_clamped": behavior_w_clamped,
            "log_pi": log_pi,
            "finite_tensors": (
                a_new, log_pi, q1_new, q2_new,
                (d1_new if self._deliverable_critic_enabled else q1_new),
                (d2_new if self._deliverable_critic_enabled else q2_new),
                c1_new, c2_new,
                actor_loss, constraint_actor_loss, behavior_cloning_loss,
            ),
            "value_aux_loss": torch.zeros(
                (), device=self.device, dtype=sac_actor_loss.dtype),
            "value_aux_weight": 0.0,
            "value_aux_accuracy": 0.0,
            "value_action_aux_loss": torch.zeros(
                (), device=self.device, dtype=sac_actor_loss.dtype),
            "value_action_aux_weight": 0.0,
            "actor_reward_q_mean": reward_q_new.detach().mean(),
            "actor_augmented_q_mean": sac_reward.detach().mean(),
            "actor_constraint_q_mean": constraint_q_new.detach().mean(),
        }

    def _current_value_aux_weight(self) -> float:
        """线性退火辅助损失权重，前期引导判别，后期交还给 RL 目标。"""
        if not self.value_aux_head_enable:
            return 0.0
        start = float(self.value_aux_loss_weight)
        end = float(self.value_aux_loss_weight_final)
        if self.value_aux_weight_decay_steps <= 1:
            return end
        progress = min(float(self.total_steps) / float(self.value_aux_weight_decay_steps), 1.0)
        return (1.0 - progress) * start + progress * end

    def _current_value_action_aux_weight(self) -> float:
        """Linearly anneal the auxiliary action-shaping loss weight."""
        start = float(self.value_action_aux_loss_weight)
        end = float(self.value_action_aux_loss_weight_final)
        if self.value_aux_weight_decay_steps <= 1:
            return end
        progress = min(float(self.total_steps) / float(self.value_aux_weight_decay_steps), 1.0)
        return (1.0 - progress) * start + progress * end

    def set_lyapunov_penalty_coeff(self, coeff: float) -> float:
        """设置 Actor 约束 Q 的全局权重，训练器可用它做自适应调节。"""
        self.lya_coeff = sanitize_scalar(
            coeff,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
            min_value=0.0,
        )
        return float(self.lya_coeff)

    def get_lyapunov_penalty_coeff(self) -> float:
        """读取当前 Actor 约束 Q 全局权重。"""
        return float(self.lya_coeff)

    def _clamp_log_alpha(self):
        """裁剪熵系数，避免 alpha 过低导致策略后期过早确定化。"""
        lower = -self.alpha_log_clip
        if self.alpha_min > 0.0:
            lower = max(lower, float(np.log(max(self.alpha_min, 1e-8))))
        self.log_alpha.data.clamp_(lower, self.alpha_log_clip)

    # ── 动作采样 ──────────────────────────────────────────────────
    def select_action(self, state: np.ndarray,
                      evaluate: bool = False) -> np.ndarray:
        """
        根据当前时序状态选择动作
        Args:
            state: (T, state_dim) 维度的时序堆叠数组
        """
        return self.select_actions(
            np.asarray(state, dtype=np.float32)[None, ...],
            evaluate=evaluate,
        )[0]

    def select_actions(self, states: np.ndarray,
                       evaluate: bool = False) -> np.ndarray:
        """
        批量选择动作。

        多环境训练时一次把 (B, T, state_dim) 状态送进 Actor，避免每个环境单独触发一次
        小 batch GPU 推理，显著降低调度开销并提高 GPU 利用率。
        """
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 3:
            raise ValueError(f"states 期望形状为 (B,T,D)，实际为 {states.shape}")

        # 预热期使用随机探索
        if self.total_steps < self.warmup and not evaluate:
            return np.random.uniform(
                0, 1, size=(states.shape[0], self.action_dim)
            ).astype(np.float32)

        # 此时传入的是 (B, T, state_dim) 序列，可直接批量推理。
        states_for_net = self._normalize_states_np(states)
        s = torch.FloatTensor(states_for_net).to(self.device)
        
        was_training = self.actor.training
        if evaluate:
            # 评估/选模必须关闭 Dropout，否则同一 checkpoint、同一状态也可能输出不同动作。
            self.actor.eval()

        try:
            with torch.no_grad():
                if evaluate:
                    # 评估模式直接取均值 (无噪声)
                    _, _, action = self.actor.sample(s)
                else:
                    # 训练模式采样
                    action, _, _ = self.actor.sample(s)
        finally:
            if evaluate and was_training:
                self.actor.train()

        return action.cpu().numpy().astype(np.float32)

    # ── 存储经验 ──────────────────────────────────────────────────
    def store(self, s, a, r, s2, d, lya=0.0, terminated=None,
              deliverable_reward: float = 0.0,
              raw_action=None,
              behavior_action=None, behavior_weight: float = 0.0,
              env_id: int = 0):
        """将交互数据存入 ReplayBuffer。

        这里记录 terminated 标志，区别于 done（done=terminated|truncated），
        这样 Critic 在计算 target_q 时可以使用正确的 bootstrap mask。
        """
        # Replay 的 done 字段只允许表达物理终止，时间截断必须继续 bootstrap。
        if terminated is None:
            boundary = TransitionBoundary.from_step(bool(d))
        else:
            boundary = TransitionBoundary(
                terminated=bool(terminated),
                truncated=bool(d) and not bool(terminated),
            )

        self._update_state_normalizer(s, s2)

        if self._NStepAggClass is not None:
            # n-step：每个 env 独立维护自己的 aggregator，避免多环境轨迹交叉污染。
            if env_id not in self._n_step_aggs:
                self._n_step_aggs[env_id] = self._NStepAggClass(n=self._n_step_n, gamma=float(self.gamma))
            agg = self._n_step_aggs[env_id]
            ready = agg.ingest(
                s=s, a=a, r=float(r), s2=s2, d=boundary.terminated,
                lya=float(lya), deliverable_reward=float(deliverable_reward),
                raw_action=raw_action,
                behavior_action=behavior_action, behavior_weight=float(behavior_weight),
            )
            for tr in ready:
                self.buffer.push(
                    tr.s, tr.a, tr.r, tr.s2, tr.d, tr.lya,
                    deliverable_reward=tr.deliverable_r,
                    raw_action=tr.raw_action,
                    behavior_action=tr.behavior_action,
                    behavior_weight=tr.behavior_weight,
                    n_step_gamma_pow=tr.n_step_gamma_pow,
                )
        else:
            self.buffer.push(
                s, a, r, s2, boundary.terminated, lya,
                deliverable_reward=deliverable_reward,
                raw_action=raw_action,
                behavior_action=behavior_action,
                behavior_weight=behavior_weight,
            )  # 存储 terminated 而非 episode_done。
        self.total_steps += 1

    def reset_env_aggregator(self, env_id: int) -> None:
        """Episode 结束时重置指定 env 的 n-step aggregator（避免跨 episode 轨迹拼接）。"""
        if env_id in self._n_step_aggs:
            self._n_step_aggs[env_id].reset()

    # ── 网络更新 ──────────────────────────────────────────────────
    def update(self) -> dict:
        """从经验池采样并更新网络权重"""
        if len(self.buffer) < self.batch_size:
            return {}

        self.update_steps += 1
        update_actor_now = (self.update_steps % self.update_actor_freq == 0)

        # 从回放池取出的 done 实际表示 terminated，时间截断不会错误切断 bootstrap。
        (
            s, a, r, s2, d, lya, deliverable_r,
            behavior_a, behavior_w, n_step_gamma_pow, raw_a,
        ) = self.buffer.sample(self.batch_size)

        def to_t(x):
            return torch.FloatTensor(x).to(self.device, non_blocking=True)

        s, a, r, s2, d, lya, deliverable_r, behavior_a, behavior_w, n_step_gamma_pow, raw_a = map(
            to_t, [s, a, r, s2, d, lya, deliverable_r, behavior_a, behavior_w, n_step_gamma_pow, raw_a]
        )
        s_raw = s
        # 0.0 哨兵值：用 self.gamma；>0：用存储的 γ^n_eff（n-step 路径）。
        gamma_pow_for_target = torch.where(
            n_step_gamma_pow > 0.0,
            n_step_gamma_pow,
            torch.full_like(n_step_gamma_pow, float(self.gamma)),
        )
        if self.nan_guard_enable and (not self._all_finite(s, a, r, s2, d, lya, deliverable_r, behavior_a, behavior_w, raw_a)):
            return self._trigger_nan_guard(1, "non-finite replay batch")
        s = self._normalize_states_tensor(s)
        s2 = self._normalize_states_tensor(s2)
        if self.nan_guard_enable and (not self._all_finite(s, s2)):
            return self._trigger_nan_guard(1, "non-finite normalized replay batch")

        # 使用 torch.amp.autocast，保持 AMP 路径和当前 PyTorch API 一致。
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            # ── Critic 更新 ─────────────────────────────────────
            with torch.no_grad():
                a2, log_pi2, _ = self.actor.sample(s2)
                self.critic_target.eval()
                self.deliverable_critic_target.eval()
                self.constraint_critic_target.eval()
                q1_t, q2_t = self.critic_target(s2, a2)
                reward_next_q = torch.min(q1_t, q2_t) - self.alpha * log_pi2
                d1_t, d2_t = self.deliverable_critic_target(s2, a2)
                deliverable_next_q = torch.min(d1_t, d2_t) - self.alpha * log_pi2
                c1_t, c2_t = self.constraint_critic_target(s2, a2)
                # 约束 Q 使用较保守的高估分支，避免 actor 利用低估的安全风险。
                constraint_next_q = torch.max(c1_t, c2_t)
                
                # Reward Critic 只学习环境/训练奖励；Deliverable Critic 学近端处理投资回报；Lyapunov 漂移进入独立约束 Critic。
                # gamma_pow_for_target：>0 → 使用 n-step γ^n；=γ → 单步（哨兵 0.0 已被替换）。
                target_q, target_d, target_c = self._compute_td_targets(
                    r, d, lya, deliverable_r, reward_next_q, deliverable_next_q, constraint_next_q,
                    gamma_pow=gamma_pow_for_target)
                replay_reward_mean = float(r.mean().item())
                replay_reward_std = float(r.std(unbiased=False).item())
                replay_deliverable_reward_mean = float(deliverable_r.mean().item())
                replay_deliverable_reward_std = float(
                    deliverable_r.std(unbiased=False).item())
                replay_constraint_cost_mean = float(lya.mean().item())
                replay_constraint_cost_std = float(lya.std(unbiased=False).item())
                gamma_pow_mean = float(gamma_pow_for_target.mean().item())
                target_q_mean = float(target_q.mean().item())
                target_q_std = float(target_q.std(unbiased=False).item())
                target_d_mean = float(target_d.mean().item())
                target_d_std = float(target_d.std(unbiased=False).item())
                target_c_mean = float(target_c.mean().item())
                target_c_std = float(target_c.std(unbiased=False).item())
                if self.nan_guard_enable and (not self._all_finite(
                    a2, log_pi2, q1_t, q2_t, reward_next_q, d1_t, d2_t,
                    deliverable_next_q, c1_t, c2_t,
                    constraint_next_q, target_q, target_d, target_c
                )):
                    return self._trigger_nan_guard(2, "critic target path")

            q1, q2 = self.critic(s, a)
            with torch.no_grad():
                q1_raw_diag, q2_raw_diag = self.critic(s, raw_a)
                q_exec_diag = torch.min(q1.detach(), q2.detach())
                q_raw_diag = torch.min(q1_raw_diag, q2_raw_diag)
                critic_q_exec_mean = float(q_exec_diag.mean().item())
                critic_q_raw_mean = float(q_raw_diag.mean().item())
                critic_q_raw_minus_exec_mean = float(
                    (q_raw_diag - q_exec_diag).mean().item())
                critic_q_raw_exec_abs_gap_mean = float(
                    (q_raw_diag - q_exec_diag).abs().mean().item())
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            d1_pred, d2_pred = self.deliverable_critic(s, a)
            deliverable_critic_loss = (
                F.mse_loss(d1_pred, target_d) + F.mse_loss(d2_pred, target_d)
            )
            c1, c2 = self.constraint_critic(s, a)
            constraint_critic_loss = (
                F.mse_loss(c1, target_c) + F.mse_loss(c2, target_c)
            )
            if self.nan_guard_enable and (not self._all_finite(
                q1, q2, critic_loss, d1_pred, d2_pred,
                deliverable_critic_loss, c1, c2, constraint_critic_loss
            )):
                return self._trigger_nan_guard(3, "critic loss")

        self.critic_opt.zero_grad()
        self.deliverable_critic_opt.zero_grad()
        self.constraint_critic_opt.zero_grad()
        total_critic_loss = critic_loss + deliverable_critic_loss + constraint_critic_loss
        self.scaler.scale(total_critic_loss).backward()
        self.scaler.unscale_(self.critic_opt)
        self.scaler.unscale_(self.deliverable_critic_opt)
        self.scaler.unscale_(self.constraint_critic_opt)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_clip)
        torch.nn.utils.clip_grad_norm_(
            self.deliverable_critic.parameters(), max_norm=self.grad_clip)
        torch.nn.utils.clip_grad_norm_(
            self.constraint_critic.parameters(), max_norm=self.grad_clip)
        self.scaler.step(self.critic_opt)
        self.scaler.step(self.deliverable_critic_opt)
        self.scaler.step(self.constraint_critic_opt)

        actor_loss_value = 0.0
        constraint_actor_loss_value = 0.0
        behavior_cloning_loss_value = 0.0
        value_aux_loss_value = 0.0
        value_aux_weight_value = float(self._current_value_aux_weight())
        value_aux_accuracy_value = 0.0
        value_action_aux_loss_value = 0.0
        value_action_aux_weight_value = float(self._current_value_action_aux_weight())
        behavior_weight_mean = 0.0
        actor_reward_q_mean_value = 0.0
        actor_augmented_q_mean_value = 0.0
        actor_constraint_q_mean_value = 0.0
        if update_actor_now:
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                # Actor 目标函数由具体算法类定义；训练基础设施只负责反传和保护。
                actor_terms = self.compute_actor_objective(
                    s, behavior_a, behavior_w, raw_states=s_raw
                )
                actor_loss = actor_terms["actor_loss"]
                log_pi = actor_terms["log_pi"]
                constraint_actor_loss = actor_terms["constraint_actor_loss"]
                behavior_cloning_loss = actor_terms["behavior_cloning_loss"]
                behavior_w_clamped = actor_terms["behavior_w_clamped"]
                if self.nan_guard_enable and (not self._all_finite(
                    *actor_terms["finite_tensors"]
                )):
                    self.scaler.update()
                    return self._trigger_nan_guard(4, "actor loss")

            self.actor_opt.zero_grad()
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_opt)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip)
            self.scaler.step(self.actor_opt)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                # ── Alpha 熵系数更新 ────────────────────────────────
                alpha_loss = -(self.log_alpha *
                               (log_pi + self.target_entropy).detach()).mean()
                if self.nan_guard_enable and (not self._all_finite(alpha_loss, self.log_alpha)):
                    self.scaler.update()
                    return self._trigger_nan_guard(5, "alpha loss")

            self.alpha_opt.zero_grad()
            self.scaler.scale(alpha_loss).backward()
            self.scaler.unscale_(self.alpha_opt)
            self.scaler.step(self.alpha_opt)
            self._clamp_log_alpha()
            actor_loss_value = float(actor_loss.item())
            constraint_actor_loss_value = float(constraint_actor_loss.item())
            behavior_cloning_loss_value = float(behavior_cloning_loss.item())
            value_aux_loss_value = float(actor_terms.get("value_aux_loss", torch.zeros((), device=self.device)).item())
            value_aux_weight_value = float(actor_terms.get("value_aux_weight", value_aux_weight_value))
            value_aux_accuracy_value = float(actor_terms.get("value_aux_accuracy", 0.0))
            value_action_aux_loss_value = float(actor_terms.get(
                "value_action_aux_loss",
                torch.zeros((), device=self.device),
            ).item())
            value_action_aux_weight_value = float(actor_terms.get(
                "value_action_aux_weight",
                value_action_aux_weight_value,
            ))
            behavior_weight_mean = float(behavior_w_clamped.mean().item())
            actor_reward_q_mean_value = float(actor_terms.get(
                "actor_reward_q_mean",
                torch.zeros((), device=self.device),
            ).item())
            actor_augmented_q_mean_value = float(actor_terms.get(
                "actor_augmented_q_mean",
                torch.zeros((), device=self.device),
            ).item())
            actor_constraint_q_mean_value = float(actor_terms.get(
                "actor_constraint_q_mean",
                torch.zeros((), device=self.device),
            ).item())

        # scaler.update() 在所有 scaler.step() 调用完成后统一调用一次。
        # 这是 PyTorch 官方推荐的用法，确保 scale factor 在一轮更新中保持一致
        self.scaler.update()

        if self.nan_guard_enable:
            if not torch.isfinite(self.log_alpha).all().item():
                self.log_alpha.data.zero_()
                return self._trigger_nan_guard(6, "log_alpha reset to 0.0")
            self._clamp_log_alpha()

        # ── 软更新目标网络 ──────────────────────────────────────
        for p, pt in zip(self.critic.parameters(),
                         self.critic_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
        for p, pt in zip(self.deliverable_critic.parameters(),
                         self.deliverable_critic_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
        for p, pt in zip(self.constraint_critic.parameters(),
                         self.constraint_critic_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
        self.critic_target.eval()
        self.deliverable_critic_target.eval()
        self.constraint_critic_target.eval()

        self.alpha = float(self.log_alpha.exp().item())
        if self.nan_guard_enable and (not np.isfinite(self.alpha)):
            self.log_alpha.data.zero_()
            self.alpha = 1.0
            return self._trigger_nan_guard(7, "alpha became non-finite")
        
        # ── 学习率调度更新 ──────────────────────────────────────
        # 【预热阶段】: 前 warmup_steps 线性增加学习率
        if self.use_lr_warmup and self.total_steps < self.lr_warmup_steps:
            warmup_factor = self.lr_warmup_init_scale + \
                           (1.0 - self.lr_warmup_init_scale) * self.total_steps / self.lr_warmup_steps
            for param_group in self.actor_opt.param_groups:
                param_group['lr'] = self.lr_actor_init * warmup_factor
            for param_group in self.critic_opt.param_groups:
                param_group['lr'] = self.lr_critic_init * warmup_factor
            for param_group in self.deliverable_critic_opt.param_groups:
                param_group['lr'] = self.lr_critic_init * warmup_factor
            for param_group in self.constraint_critic_opt.param_groups:
                param_group['lr'] = self.lr_critic_init * warmup_factor
            for param_group in self.alpha_opt.param_groups:
                param_group['lr'] = self.lr_alpha_init * warmup_factor
        # 【正常调度】: warmup 之后应用学习率调度
        elif self.actor_scheduler is not None:
            self.actor_scheduler.step()
            self.critic_scheduler.step()
            self.deliverable_critic_scheduler.step()
            self.constraint_critic_scheduler.step()

        return {
            "actor_loss":  actor_loss_value,
            "critic_loss": critic_loss.item(),
            "deliverable_critic_loss": deliverable_critic_loss.item(),
            "constraint_critic_loss": constraint_critic_loss.item(),
            "constraint_actor_loss": constraint_actor_loss_value,
            "behavior_cloning_loss": behavior_cloning_loss_value,
            "value_aux_loss": value_aux_loss_value,
            "value_aux_weight": value_aux_weight_value,
            "value_aux_accuracy": value_aux_accuracy_value,
            "value_action_aux_loss": value_action_aux_loss_value,
            "value_action_aux_weight": value_action_aux_weight_value,
            "behavior_weight_mean": behavior_weight_mean,
            "actor_update_applied": 1.0 if update_actor_now else 0.0,
            "actor_reward_q_mean": actor_reward_q_mean_value,
            "actor_augmented_q_mean": actor_augmented_q_mean_value,
            "actor_constraint_q_mean": actor_constraint_q_mean_value,
            "critic_q_exec_mean": critic_q_exec_mean,
            "critic_q_raw_mean": critic_q_raw_mean,
            "critic_q_raw_minus_exec_mean": critic_q_raw_minus_exec_mean,
            "critic_q_raw_exec_abs_gap_mean": critic_q_raw_exec_abs_gap_mean,
            "replay_reward_mean": replay_reward_mean,
            "replay_reward_std": replay_reward_std,
            "replay_deliverable_reward_mean": replay_deliverable_reward_mean,
            "replay_deliverable_reward_std": replay_deliverable_reward_std,
            "replay_constraint_cost_mean": replay_constraint_cost_mean,
            "replay_constraint_cost_std": replay_constraint_cost_std,
            "n_step_gamma_pow_mean": gamma_pow_mean,
            "target_q_mean": target_q_mean,
            "target_q_std": target_q_std,
            "target_deliverable_q_mean": target_d_mean,
            "target_deliverable_q_std": target_d_std,
            "target_constraint_q_mean": target_c_mean,
            "target_constraint_q_std": target_c_std,
            "lyapunov_penalty_coeff": float(self.lya_coeff),
            "alpha":       self.alpha,
            "actor_lr":    self.actor_opt.param_groups[0]['lr'],
            "critic_lr":   self.critic_opt.param_groups[0]['lr'],
            "nan_guard_triggered": 0.0,
            "nan_guard_stage": 0.0,
            "nan_guard_hits": float(self.nan_guard_hits),
        }

    # ── 保存 / 加载 ───────────────────────────────────────────────
    def save(self, path: str, metadata: dict | None = None):
        # 保存完整训练状态，保证中断续训时不仅恢复权重，也恢复优化器动量和学习率调度进度。
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        metadata = dict(metadata or {})
        metadata.setdefault("state_dim", int(self.state_dim))
        metadata.setdefault("action_dim", int(self.action_dim))
        metadata.setdefault("frame_stack", int(DRL_CONFIG.get("frame_stack", 1)))
        metadata.setdefault("network_arch", str(DRL_CONFIG.get("network_arch", "transformer")))
        metadata.setdefault("behavior_cloning_coeff", float(self.behavior_cloning_coeff))
        metadata.setdefault("state_normalization", bool(self.use_state_normalization))
        metadata.setdefault("lyapunov_penalty_coeff", float(self.lya_coeff))
        ckpt = {
            "actor":           self.actor.state_dict(),
            "critic":          self.critic.state_dict(),
            "critic_target":   self.critic_target.state_dict(),
            "deliverable_critic": self.deliverable_critic.state_dict(),
            "deliverable_critic_target": self.deliverable_critic_target.state_dict(),
            "constraint_critic": self.constraint_critic.state_dict(),
            "constraint_critic_target": self.constraint_critic_target.state_dict(),
            "log_alpha":       self.log_alpha.data,
            "lyapunov_penalty_coeff": float(self.lya_coeff),
            "total_steps":     self.total_steps,
            "metadata":        metadata,
            "state_rms":       self.state_rms.state_dict(),
            # 保存优化器状态（Adam 动量 m/v），保证续训轨迹连续。
            "actor_opt":       self.actor_opt.state_dict(),
            "critic_opt":      self.critic_opt.state_dict(),
            "deliverable_critic_opt": self.deliverable_critic_opt.state_dict(),
            "constraint_critic_opt": self.constraint_critic_opt.state_dict(),
            "alpha_opt":       self.alpha_opt.state_dict(),
        }
        # 保存学习率调度器状态（余弦退火进度）。
        if self.actor_scheduler is not None:
            ckpt["actor_scheduler"] = self.actor_scheduler.state_dict()
        if self.critic_scheduler is not None:
            ckpt["critic_scheduler"] = self.critic_scheduler.state_dict()
        if self.constraint_critic_scheduler is not None:
            ckpt["constraint_critic_scheduler"] = (
                self.constraint_critic_scheduler.state_dict()
            )
        if self.deliverable_critic_scheduler is not None:
            ckpt["deliverable_critic_scheduler"] = (
                self.deliverable_critic_scheduler.state_dict()
            )
        torch.save(ckpt, path)

    def load(self, path: str) -> dict:
        ckpt = _safe_torch_load(path, map_location=self.device)
        metadata = ckpt.get("metadata", {}) if isinstance(ckpt, dict) else {}
        saved_state_dim = metadata.get("state_dim") if isinstance(metadata, dict) else None
        if saved_state_dim is not None and int(saved_state_dim) != int(self.state_dim):
            raise RuntimeError(
                f"检查点状态维度不兼容: checkpoint state_dim={saved_state_dim}, "
                f"当前 state_dim={self.state_dim}。请使用同一观测 schema 的检查点，或从头训练。"
            )
        saved_action_dim = metadata.get("action_dim") if isinstance(metadata, dict) else None
        if saved_action_dim is not None and int(saved_action_dim) != int(self.action_dim):
            raise RuntimeError(
                f"检查点动作维度不兼容: checkpoint action_dim={saved_action_dim}, "
                f"当前 action_dim={self.action_dim}。分组动作空间变更后请从头训练。"
            )
        saved_arch = metadata.get("network_arch") if isinstance(metadata, dict) else None
        current_arch = str(DRL_CONFIG.get("network_arch", "transformer"))
        if saved_arch is not None and str(saved_arch) != current_arch:
            raise RuntimeError(
                f"检查点网络结构不兼容: checkpoint network_arch={saved_arch}, "
                f"当前 network_arch={current_arch}。请使用同一 backbone 的检查点，或从头训练。"
            )
        try:
            actor_state = self.actor.load_state_dict(ckpt["actor"], strict=False)
            critic_state = self.critic.load_state_dict(ckpt["critic"], strict=False)
            critic_t_state = self.critic_target.load_state_dict(ckpt["critic_target"], strict=False)
        except RuntimeError:
            raise RuntimeError(
                f"检查点与当前网络结构不兼容: {path}。"
                "请使用同版本模型结构对应的检查点，或从头开始训练。"
            ) from None
        if "deliverable_critic" in ckpt and "deliverable_critic_target" in ckpt:
            try:
                deliverable_state = self.deliverable_critic.load_state_dict(
                    ckpt["deliverable_critic"], strict=False)
                deliverable_t_state = self.deliverable_critic_target.load_state_dict(
                    ckpt["deliverable_critic_target"], strict=False)
            except RuntimeError:
                print("[SACAgent] Deliverable Critic与当前结构不兼容，已重新初始化。")
                deliverable_state = None
                deliverable_t_state = None
        else:
            print("[SACAgent] 检查点缺少Deliverable Critic，已用当前初始化参数启动该分支。")
            deliverable_state = None
            deliverable_t_state = None
        if "constraint_critic" in ckpt and "constraint_critic_target" in ckpt:
            try:
                constraint_state = self.constraint_critic.load_state_dict(
                    ckpt["constraint_critic"], strict=False)
                constraint_t_state = self.constraint_critic_target.load_state_dict(
                    ckpt["constraint_critic_target"], strict=False)
            except RuntimeError:
                print("[SACAgent] 约束Critic与当前结构不兼容，已重新初始化。")
                constraint_state = None
                constraint_t_state = None
        else:
            print("[SACAgent] 检查点缺少约束Critic，已用当前初始化参数启动该分支。")
            constraint_state = None
            constraint_t_state = None
        self.log_alpha.data = ckpt["log_alpha"]
        if not torch.isfinite(self.log_alpha).all().item():
            print("[SACAgent] 检查点中的 log_alpha 非有限，已重置为0")
            self.log_alpha.data.zero_()
        self._clamp_log_alpha()
        self.alpha = float(self.log_alpha.exp().item())
        if not np.isfinite(self.alpha):
            print("[SACAgent] 检查点中的 alpha 非有限，已重置为1.0")
            self.log_alpha.data.zero_()
            self.alpha = 1.0
        if "lyapunov_penalty_coeff" in ckpt:
            self.set_lyapunov_penalty_coeff(ckpt["lyapunov_penalty_coeff"])
        self.total_steps    = ckpt.get("total_steps", 0)
        if "state_rms" in ckpt:
            self.state_rms.load_state_dict(ckpt["state_rms"])

        # 兼容结构迭代（例如位置编码缓冲区变更）造成的 key 差异
        state_items = [("actor", actor_state),
                       ("critic", critic_state),
                       ("critic_target", critic_t_state)]
        if deliverable_state is not None:
            state_items.append(("deliverable_critic", deliverable_state))
        if deliverable_t_state is not None:
            state_items.append(("deliverable_critic_target", deliverable_t_state))
        if constraint_state is not None:
            state_items.append(("constraint_critic", constraint_state))
        if constraint_t_state is not None:
            state_items.append(("constraint_critic_target", constraint_t_state))
        for name, state in state_items:
            if state.missing_keys or state.unexpected_keys:
                print(
                    f"[SACAgent] 加载{name}时存在兼容差异: "
                    f"missing={state.missing_keys}, unexpected={state.unexpected_keys}"
                )

        # 恢复优化器状态。
        if "actor_opt" in ckpt:
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
        if "critic_opt" in ckpt:
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
        if "deliverable_critic_opt" in ckpt:
            self.deliverable_critic_opt.load_state_dict(ckpt["deliverable_critic_opt"])
        if "constraint_critic_opt" in ckpt:
            self.constraint_critic_opt.load_state_dict(ckpt["constraint_critic_opt"])
        if "alpha_opt" in ckpt:
            self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
        # 恢复学习率调度器状态。
        if "actor_scheduler" in ckpt and self.actor_scheduler is not None:
            self.actor_scheduler.load_state_dict(ckpt["actor_scheduler"])
        if "critic_scheduler" in ckpt and self.critic_scheduler is not None:
            self.critic_scheduler.load_state_dict(ckpt["critic_scheduler"])
        if ("constraint_critic_scheduler" in ckpt
                and self.constraint_critic_scheduler is not None):
            self.constraint_critic_scheduler.load_state_dict(
                ckpt["constraint_critic_scheduler"])
        if ("deliverable_critic_scheduler" in ckpt
                and self.deliverable_critic_scheduler is not None):
            self.deliverable_critic_scheduler.load_state_dict(
                ckpt["deliverable_critic_scheduler"])
        self.critic_target.eval()
        self.constraint_critic_target.eval()
        print(f"[SACAgent] 模型已加载: {path}  步数={self.total_steps:,}")
        return metadata or {}

    def _normalize_states_np(self, states: np.ndarray) -> np.ndarray:
        if not self.use_state_normalization:
            return np.asarray(states, dtype=np.float32)
        return self.state_rms.normalize_np(
            states,
            clip=self.state_norm_clip,
            epsilon=self.state_norm_epsilon,
        )

    def _normalize_states_tensor(self, states: torch.Tensor) -> torch.Tensor:
        if not self.use_state_normalization:
            return states
        return self.state_rms.normalize_torch(
            states,
            clip=self.state_norm_clip,
            epsilon=self.state_norm_epsilon,
        )

    def _update_state_normalizer(self, *states) -> None:
        if not self.use_state_normalization:
            return
        observations = []
        for state in states:
            if torch.is_tensor(state):
                state = state.detach().cpu().numpy()
            arr = np.asarray(state, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[-1] == self.state_dim:
                observations.append(arr[:, 0, :])
            elif arr.ndim == 2 and arr.shape[-1] == self.state_dim:
                observations.append(arr[0:1, :])
            elif arr.ndim == 1 and arr.shape[0] == self.state_dim:
                observations.append(arr[None, :])
        if observations:
            self.state_rms.update(np.concatenate(observations, axis=0))

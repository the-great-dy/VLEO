"""
SAC actor/critic 网络与状态编码器定义。
SAC 网络结构定义。

默认使用面向 LS-PSF CMDP 的 Transformer 状态编码器，也保留 MLP
backbone 作为消融入口。Actor 与 Critic 共享编码器构造逻辑，但各自
维护独立参数。
"""

import sys as _sys, os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in _sys.path:
    # 降低导入劫持风险：不要把项目路径插到最高优先级。
    _sys.path.append(_PROJECT_ROOT)

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from config import DRL_CONFIG
from environment.satellite_env import OBSERVATION_FEATURES

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPSILON     = 1e-6


def weights_init(m):
    # Linear 层统一使用 Xavier 初始化，保持 actor/critic 初始尺度稳定。
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class OrbitalPositionalEncoding(nn.Module):
    """
    轨道时间位置编码。

    DilatedFrameStackWrapper 提供的是不等间隔历史帧，这里用真实 offset
    构造正余弦位置编码，让 Transformer 能区分近邻帧和长跨度历史帧。
    """

    def __init__(self, d_model: int):
        super().__init__()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        self.register_buffer("div", div)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            positions: (T,) 每个 token 对应的时间 offset。
        """
        T = x.size(1)
        if positions is None:
            # 未提供 offset 时退化为普通等间隔 token 编码。
            pos = torch.arange(0, T, device=x.device, dtype=x.dtype)
        else:
            if positions.numel() != T:
                raise ValueError(
                    f"positions length {positions.numel()} does not match sequence length {T}"
                )
            pos = positions.to(device=x.device, dtype=x.dtype)

        pe = torch.zeros(T, self.d_model, device=x.device, dtype=x.dtype)
        div = self.div.to(dtype=x.dtype)
        pos = pos.unsqueeze(1)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:self.d_model // 2])
        return x + pe.unsqueeze(0)


class OrbitalTransformerEncoder(nn.Module):
    """
    面向轨道任务的 Transformer 状态编码器。

    Frame stack 的第 0 帧是当前状态，后续帧是更早历史。编码器把通信窗口、
    任务时效、热状态等时序特征交给 Transformer，把物理状态交给 MLP，
    最后融合成 Actor/Critic 的共享状态表示。

    设计要点：
    1. 时序分支关注窗口、容量、deadline、expiring value 和场景等动态信号。
    2. 物理分支保留高度、SOC、队列压力、历史动作等当前帧信号。
    3. 通信门控使用 in_window 和 processed_queue_future_contact_ratio 调整时序特征权重。
    """

    def __init__(self, state_dim: int = 40,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.d_model   = d_model

        # 按观测名称取索引，避免 OBSERVATION_FEATURES 顺序变化后网络读错字段。
        # 只有真正有时间动态含义的字段进入 Transformer 分支。
        feature_index = {name: i for i, name in enumerate(OBSERVATION_FEATURES)}
        temporal_features = list(DRL_CONFIG.get("transformer_temporal_features", ()))
        if not temporal_features:
            raise KeyError("DRL_CONFIG['transformer_temporal_features'] 未配置")
        missing_features = [name for name in temporal_features if name not in feature_index]
        if missing_features:
            raise KeyError(
                "transformer_temporal_features 包含未定义观测字段: "
                + ", ".join(missing_features)
            )
        self.temporal_idx = [feature_index[name] for name in temporal_features]
        self.idx_in_window = feature_index["in_comm_window"]
        self.idx_time_to_next_window = feature_index["time_to_next_window_norm"]
        self.idx_window_remaining = feature_index["window_remaining_norm"]
        self.idx_processed_queue_future_contact_ratio = feature_index[
            "processed_queue_future_contact_ratio"
        ]
        self.idx_processed_high_next_window_deliverable_ratio = feature_index[
            "processed_high_next_window_deliverable_ratio"
        ]
        self.idx_raw_high_next_window_deliverable_ratio = feature_index[
            "raw_high_next_window_deliverable_ratio"
        ]
        self.idx_high_value_deadline_contact_mismatch = feature_index[
            "high_value_deadline_contact_mismatch"
        ]
        # 剩余字段只读取当前帧，由 MLP 处理。
        self.physical_idx = [i for i in range(state_dim)
                              if i not in self.temporal_idx]

        # 时序分支：特征嵌入 + 轨道相位特征 + 位置编码 + Transformer。
        self.temporal_embed = nn.Linear(len(self.temporal_idx), d_model)
        self.pos_enc        = OrbitalPositionalEncoding(d_model)
        # 与 DilatedFrameStackWrapper 的历史帧 offset 保持一致。
        self.token_offsets  = list(DRL_CONFIG.get(
            "dilated_offsets", [0, 1, 3, 9, 27, 90, 270, 540]))
        self.orbit_phase_proj = nn.Linear(1, d_model)  # 窗口内外的轨道相位提示。
        encoder_layer       = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
            norm_first=True)          # Pre-LN 通常更稳定。
        self.transformer    = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        # 当前帧物理分支。
        self.physical_mlp = nn.Sequential(
            nn.Linear(len(self.physical_idx), d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # 通信门控：用是否在窗口内和 processed queue 压力调制时序表示。
        self.comm_gate = nn.Sequential(
            nn.Linear(5, d_model),
            nn.Sigmoid(),
        )

        # 融合时序特征和当前物理特征。
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        self.output_dim = d_model
        self.apply(weights_init)

    def _build_positions(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build token time offsets, preferring configured dilated offsets."""
        # offset 不足时向后补齐；过长时截断到当前序列长度。
        offsets = self.token_offsets
        if len(offsets) < T:
            last = float(offsets[-1]) if offsets else 0.0
            offsets = offsets + [last + float(i + 1) for i in range(T - len(offsets))]
        else:
            offsets = offsets[:T]
        return torch.tensor(offsets, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, state_dim) ??(B, T, state_dim)
        Returns:
            (B, d_model)
        """
        assert x.dim() == 3, f"Fatal Error: Expected 3D tensor (B, T, D) for Frame Stacking, but got {x.shape}"

        B, T, D = x.shape

        # 时序分支：窗口、容量、任务紧急度、场景和热状态。
        temporal = x[:, :, self.temporal_idx]          # (B, T, temporal_dim)
        t_embed  = self.temporal_embed(temporal)        # (B, T, d_model)
        # 窗口内使用剩余窗口比例，窗口外使用距下一窗口的反向比例。
        orbit_phase = torch.where(
            x[:, :, self.idx_in_window] > 0.5,
            x[:, :, self.idx_window_remaining],
            1.0 - x[:, :, self.idx_time_to_next_window],
        ).clamp(0.0, 1.0).unsqueeze(-1)                    # (B, T, 1)
        phase_feat = self.orbit_phase_proj(orbit_phase)     # (B, T, d_model)
        t_embed = t_embed + phase_feat
        positions = self._build_positions(T, t_embed.device, t_embed.dtype)
        t_embed = self.pos_enc(t_embed, positions=positions)
        t_feat   = self.transformer(t_embed)            # (B, T, d_model)
        # Dilated wrapper 顺序为 [当前, 更早, ...]，token 0 对应当前决策时刻。
        t_feat   = t_feat[:, 0, :]                     # 当前帧聚合表示 (B, d_model)

        # 当前物理分支。
        physical = x[:, 0, self.physical_idx]          # (B, physical_dim)
        p_feat   = self.physical_mlp(physical)         # (B, d_model)

        # 通信门控：processed backlog 已接近未来可下传容量时，增强相关时序信号。
        comm_signal = x[:, 0, [
            self.idx_in_window,
            self.idx_processed_queue_future_contact_ratio,
            self.idx_processed_high_next_window_deliverable_ratio,
            self.idx_raw_high_next_window_deliverable_ratio,
            self.idx_high_value_deadline_contact_mismatch,
        ]]
        gate        = self.comm_gate(comm_signal)      # (B, d_model)
        t_feat      = t_feat * gate

        # 融合输出。
        combined = torch.cat([t_feat, p_feat], dim=-1)  # (B, 2*d_model)
        out      = self.fusion(combined)                 # (B, d_model)
        return out


class MLPStateEncoder(nn.Module):
    """Backbone ablation: encode only the current observation frame with an MLP."""

    def __init__(self, state_dim: int = 40, d_model: int = 128):
        super().__init__()
        self.output_dim = int(d_model)
        self.net = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"Expected 3D tensor (B, T, D), got {x.shape}"
        return self.net(x[:, 0, :])


def _make_state_encoder(state_dim: int, hidden_dim: int) -> nn.Module:
    arch = str(DRL_CONFIG.get("network_arch", "transformer")).lower()
    d_model = max(128, hidden_dim // 2)
    if arch == "mlp":
        return MLPStateEncoder(state_dim=state_dim, d_model=d_model)
    if arch != "transformer":
        raise ValueError(f"Unknown DRL_CONFIG['network_arch']={arch!r}")
    return OrbitalTransformerEncoder(
        state_dim=state_dim,
        d_model=d_model,
        n_heads=8,
        n_layers=4,
        dropout=0.1,
    )


class Actor(nn.Module):
    """
    SAC actor. The state encoder is selected by DRL_CONFIG["network_arch"].

    输出未压缩的高斯均值和 log_std；sample() 中再经过 sigmoid 映射到
    [0, 1] 动作空间。value_aux_layer 是高/均衡/低价值策略伪标签的辅助头。
    """

    def __init__(self, state_dim: int = 40,
                 action_dim: int | None = None,
                 hidden_dim: int = 256):
        super().__init__()

        self.encoder = _make_state_encoder(state_dim, hidden_dim)
        enc_dim = self.encoder.output_dim
        action_dim = int(action_dim or DRL_CONFIG.get("action_dim", 10))

        # Actor head：在编码器输出上继续建模策略分布。
        self.mlp = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.mu_layer     = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.value_aux_head_enable = bool(DRL_CONFIG.get("value_aux_head_enable", False))
        self.value_aux_num_classes = max(2, int(DRL_CONFIG.get("value_aux_num_classes", 3)))
        self.value_aux_layer = nn.Linear(hidden_dim, self.value_aux_num_classes) \
            if self.value_aux_head_enable else None

        self.apply(weights_init)

    def forward(self, state: torch.Tensor, return_aux: bool = False):
        # 这里返回的是高斯分布参数；动作边界映射在 sample() 中完成。
        feat = self.encoder(state)
        h    = self.mlp(feat)
        mu      = self.mu_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        # 防御非有限值，避免构造 Normal 分布时崩溃。
        mu      = torch.nan_to_num(mu,      nan=0.0, posinf=1.0, neginf=-1.0)
        log_std = torch.nan_to_num(log_std, nan=LOG_STD_MIN)
        if not return_aux:
            return mu, log_std
        aux_logits = None
        if self.value_aux_layer is not None:
            aux_logits = self.value_aux_layer(h)
            aux_logits = torch.nan_to_num(aux_logits, nan=0.0, posinf=1.0, neginf=-1.0)
        return mu, log_std, aux_logits

    def predict_value_priority_logits(self, state: torch.Tensor):
        """Return value-priority auxiliary logits when the auxiliary head exists."""
        _, _, aux_logits = self.forward(state, return_aux=True)
        return aux_logits

    def sample(self, state: torch.Tensor):
        mu, log_std = self.forward(state)
        std  = torch.clamp(log_std.exp(), 1e-6, 10.0)
        dist = Normal(mu, std)
        x    = dist.rsample()
        # 不直接 clamp 动作，避免 dead gradients；用 sigmoid squashing 并修正 log_prob。
        y    = torch.sigmoid(x)
        log_prob = dist.log_prob(x)
        # Sigmoid 变量变换的雅可比修正。
        log_prob -= (F.logsigmoid(x) + F.logsigmoid(-x))
        log_prob  = log_prob.sum(dim=-1, keepdim=True)
        return y, log_prob, torch.sigmoid(mu)


class Critic(nn.Module):
    """
    双 Q 网络：输入 [state, action]，输出两个独立 Q 估计。

    reward Critic 和 constraint Critic 使用同一结构，但参数独立；外层算法决定
    当前实例学习 Q_r 还是 Q_c。
    """

    def __init__(self, state_dim: int = 40,
                 action_dim: int | None = None,
                 hidden_dim: int = 256):
        super().__init__()
        action_dim = int(action_dim or DRL_CONFIG.get("action_dim", 10))

        # Q1
        self.encoder1 = _make_state_encoder(state_dim, hidden_dim)
        enc_dim = self.encoder1.output_dim
        self.q1 = nn.Sequential(
            nn.Linear(enc_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, 1),
        )

        # Q2
        self.encoder2 = _make_state_encoder(state_dim, hidden_dim)
        self.q2 = nn.Sequential(
            nn.Linear(enc_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, 1),
        )

        self.apply(weights_init)

    def forward(self, state: torch.Tensor,
                action: torch.Tensor):
        # 双编码器避免 Q1/Q2 共享状态特征导致过强相关。
        f1 = self.encoder1(state)
        f2 = self.encoder2(state)
        sa1 = torch.cat([f1, action], dim=-1)
        sa2 = torch.cat([f2, action], dim=-1)
        return self.q1(sa1), self.q2(sa2)

    def q1_only(self, state: torch.Tensor,
                action: torch.Tensor):
        f1  = self.encoder1(state)
        sa1 = torch.cat([f1, action], dim=-1)
        return self.q1(sa1)



"""Actor 目标使用的自适应全局 Lyapunov 对偶更新。"""

from __future__ import annotations

from dataclasses import dataclass

from config import DRL_CONFIG
from utils.sanitizers import sanitize_scalar


@dataclass(frozen=True)
class AdaptiveDualUpdate:
    """全局约束权重 lambda_t 的一次更新。"""

    coeff: float
    constraint_value: float
    constraint_ema: float
    constraint_threshold: float
    constraint_violation: float

    @property
    def pressure_raw(self) -> float:
        """旧日志/测试的向后兼容别名。"""
        return self.constraint_value

    @property
    def pressure_ema(self) -> float:
        """旧日志/测试的向后兼容别名。"""
        return self.constraint_ema


def adaptive_lyapunov_coeff_step(
    current_coeff: float,
    constraint_ema: float,
    constraint_value: float,
    enabled: bool = True,
    cfg: dict | None = None,
) -> AdaptiveDualUpdate:
    """
    更新 actor 的全局安全权重。

    论文形式：
        lambda_{t+1} = clip(lambda_t + eta * (EMA(c_t_norm) - d))

    输入是归一化的 CMDP 约束代价 c_t。投影率、PSF 干预率和
    动作修改距离仅用于诊断；不应驱动此对偶更新。
    """
    cfg = cfg or DRL_CONFIG
    min_coeff = max(0.0, float(cfg.get("adaptive_lyapunov_coeff_min", 0.0)))
    max_coeff = max(min_coeff, float(cfg.get("adaptive_lyapunov_coeff_max", 1.0)))
    actor_cap = sanitize_scalar(
        cfg.get("adaptive_lyapunov_actor_cap", max_coeff),
        nan=max_coeff,
        posinf=max_coeff,
        neginf=min_coeff,
        min_value=min_coeff,
        max_value=max_coeff,
    )
    coeff = sanitize_scalar(
        current_coeff,
        nan=min_coeff,
        posinf=max_coeff,
        neginf=min_coeff,
        min_value=min_coeff,
        max_value=max_coeff,
    )

    raw = sanitize_scalar(
        constraint_value,
        nan=0.0,
        posinf=float(cfg.get("adaptive_lyapunov_constraint_signal_max", 3.0)),
        neginf=0.0,
        min_value=0.0,
        max_value=float(cfg.get("adaptive_lyapunov_constraint_signal_max", 3.0)),
    )
    beta = sanitize_scalar(
        cfg.get("adaptive_lyapunov_coeff_ema_beta", 0.995),
        nan=0.995,
        posinf=0.9999,
        neginf=0.0,
        min_value=0.0,
        max_value=0.9999,
    )
    prev_ema = sanitize_scalar(
        constraint_ema,
        nan=0.0,
        posinf=float(cfg.get("adaptive_lyapunov_constraint_signal_max", 3.0)),
        neginf=0.0,
        min_value=0.0,
        max_value=float(cfg.get("adaptive_lyapunov_constraint_signal_max", 3.0)),
    )
    next_ema = float(beta * prev_ema + (1.0 - beta) * raw)
    threshold = float(cfg.get(
        "adaptive_lyapunov_constraint_threshold",
        cfg.get("adaptive_lyapunov_coeff_target_pressure", 0.02),
    ))
    # ── PID-Lagrangian (Stooke et al. 2020) ────────────────────────────
    # 纯积分控制器（coeff += lr*(raw-thr)）被瞬时噪声 raw 驱动会产生极限环震荡
    # （safe↔hi 来回摆）。改用：
    #   I 项（积分）：由平滑 EMA 驱动 → 稳定基线，去掉高频抖动
    #   D 项（微分）：raw 快速上升时加阻尼 → 抑制过冲
    # constraint_violation 报告值仍用 EMA-threshold（诊断口径一致）。
    constraint_violation = float(next_ema - threshold)

    if not enabled:
        return AdaptiveDualUpdate(
            coeff=coeff,
            constraint_value=raw,
            constraint_ema=next_ema,
            constraint_threshold=threshold,
            constraint_violation=constraint_violation,
        )

    lr = max(0.0, float(cfg.get("adaptive_lyapunov_coeff_lr", 0.02)))
    # 微分阻尼系数：raw 相对上一步 EMA 的瞬时跳变越大，越往回压
    kd = max(0.0, float(cfg.get("adaptive_lyapunov_coeff_kd", 0.0)))
    derivative = float(raw - prev_ema)  # >0 表示约束在快速恶化
    # 积分项用平滑 EMA 误差；微分项对快速变化做反向阻尼
    next_coeff = coeff + lr * constraint_violation - kd * derivative
    next_coeff = sanitize_scalar(
        next_coeff,
        nan=coeff,
        posinf=max_coeff,
        neginf=min_coeff,
        min_value=min_coeff,
        max_value=max_coeff,
    )
    next_coeff = min(next_coeff, actor_cap)
    return AdaptiveDualUpdate(
        coeff=next_coeff,
        constraint_value=raw,
        constraint_ema=next_ema,
        constraint_threshold=threshold,
        constraint_violation=constraint_violation,
    )

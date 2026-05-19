"""Adaptive global Lyapunov dual update used by the actor objective."""

from __future__ import annotations

from dataclasses import dataclass

from config import DRL_CONFIG
from utils.sanitizers import sanitize_scalar


@dataclass(frozen=True)
class AdaptiveDualUpdate:
    """One update of the global constraint weight lambda_t."""

    coeff: float
    constraint_value: float
    constraint_ema: float
    constraint_threshold: float
    constraint_violation: float

    @property
    def pressure_raw(self) -> float:
        """Backward-compatible alias for old logs/tests."""
        return self.constraint_value

    @property
    def pressure_ema(self) -> float:
        """Backward-compatible alias for old logs/tests."""
        return self.constraint_ema


def adaptive_lyapunov_coeff_step(
    current_coeff: float,
    constraint_ema: float,
    constraint_value: float,
    enabled: bool = True,
    cfg: dict | None = None,
) -> AdaptiveDualUpdate:
    """
    Update the actor's global safety weight.

    The paper form is:
        lambda_{t+1} = clip(lambda_t + eta * (EMA(c_t_norm) - d))

    The input is the normalized CMDP constraint cost c_t. Projection rate,
    PSF intervention rate and action modification distance are diagnostics
    only; they must not drive this dual update.
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
    # 用当前原始压力更新 actor 侧权重，这样当后续阶段变轻时，系数能更快回落。
    # EMA 仍然保留给日志和诊断。
    constraint_violation = float(raw - threshold)

    if not enabled:
        return AdaptiveDualUpdate(
            coeff=coeff,
            constraint_value=raw,
            constraint_ema=next_ema,
            constraint_threshold=threshold,
            constraint_violation=constraint_violation,
        )

    lr = max(0.0, float(cfg.get("adaptive_lyapunov_coeff_lr", 0.02)))
    next_coeff = coeff + lr * constraint_violation
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

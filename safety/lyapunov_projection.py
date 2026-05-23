"""Lyapunov projection 算子 (Chow et al. 2018, NeurIPS)。

对每个 (s, a_raw)，求解：
    a* = argmin_a ||a - a_raw||²
          s.t.   L(f(s,a)) ≤ L(s) + ε(s)
                a ∈ [0,1]^d

f 是 SafetyDynamicsPredictor 给出的一步前向预测。我们对 L(f(s,·)) 在 a_raw
处做一阶展开（有限差分对物理三维 alpha_prop / alpha_cpu / alpha_tx 求梯度），
得到线性约束：
    g · (a - a_raw)  ≤  ε(s) + L(s) - L(f(s, a_raw))
其闭式解（半空间到点的投影）：
    a* = a_raw  -  max(0, g·a_raw - b) · g / ||g||²
最后再 clip 到 [0,1]。剩余约束违反时，做最多 max_iter 次轮迭代直至满足
或退到 backup 动作。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import DRL_CONFIG, LYAPUNOV_CONFIG
from safety.dynamics_predictor import SafetyDynamicsPredictor
from safety.lyapunov_function import LyapunovFunction
from utils.action_space import PHYSICAL_ACTION_DIM


@dataclass(frozen=True)
class LyapunovProjectionResult:
    action: np.ndarray
    intervened: bool
    iterations: int
    l_now: float
    l_next_raw: float
    l_next_projected: float
    slack: float
    violation: float
    grad_norm: float


class LyapunovProjector:
    """对动作做 Lyapunov-feasible 投影。

    只投影前 PHYSICAL_ACTION_DIM 维（alpha_prop / cpu / tx）。剩余的优先级
    权重与 drop_low 不影响硬安全集，原样保留。
    """

    def __init__(
        self,
        lyapunov: LyapunovFunction | None = None,
        predictor: SafetyDynamicsPredictor | None = None,
        *,
        finite_diff_eps: float | None = None,
        max_iter: int | None = None,
        feasibility_tol: float | None = None,
    ):
        self.lyapunov = lyapunov or LyapunovFunction()
        self.predictor = predictor or SafetyDynamicsPredictor()
        self.finite_diff_eps = float(
            finite_diff_eps if finite_diff_eps is not None
            else LYAPUNOV_CONFIG.get("projection_finite_diff_eps", 1e-2))
        self.max_iter = int(
            max_iter if max_iter is not None
            else LYAPUNOV_CONFIG.get("projection_max_iter", 3))
        self.feasibility_tol = float(
            feasibility_tol if feasibility_tol is not None
            else LYAPUNOV_CONFIG.get("projection_feasibility_tol", 1e-3))

    def project(
        self,
        action: np.ndarray,
        state_phys: dict,
        backup_action: np.ndarray | None = None,
    ) -> LyapunovProjectionResult:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        out = action.copy()
        n_phys = min(PHYSICAL_ACTION_DIM, out.size)

        l_now = self.lyapunov.value(state_phys)
        slack = self.lyapunov.slack(l_now)
        next_state_raw = self.predictor.step(state_phys, out).to_dict()
        l_next_raw = self.lyapunov.value(next_state_raw)
        violation = float(l_next_raw - l_now - slack)

        if violation <= self.feasibility_tol:
            return LyapunovProjectionResult(
                action=out.astype(np.float32),
                intervened=False,
                iterations=0,
                l_now=float(l_now),
                l_next_raw=float(l_next_raw),
                l_next_projected=float(l_next_raw),
                slack=float(slack),
                violation=float(max(0.0, violation)),
                grad_norm=0.0,
            )

        l_next = l_next_raw
        grad_norm_last = 0.0
        iterations = 0
        for it in range(1, self.max_iter + 1):
            iterations = it
            grad = self._finite_diff_gradient(state_phys, out, l_baseline=l_next, n_phys=n_phys)
            grad_norm_sq = float(np.dot(grad, grad))
            grad_norm_last = float(np.sqrt(max(grad_norm_sq, 0.0)))
            if grad_norm_sq <= 1e-12:
                # 梯度为 0，无法用半空间投影，直接退到 backup。
                break
            # 半空间投影：g · (a - a0) ≤ b
            # 这里 a0 = out, b = l_now + slack - l_next + g·out
            # → 我们想要 g·a_new ≤ l_now + slack - l_next + g·out  ⟺  Δ = max(0, l_next - l_now - slack)
            excess = max(0.0, float(l_next - l_now - slack))
            step = excess / grad_norm_sq
            out_phys = out[:n_phys] - step * grad
            out_phys = np.clip(out_phys, 0.0, 1.0)
            out[:n_phys] = out_phys
            next_state_proj = self.predictor.step(state_phys, out).to_dict()
            l_next = self.lyapunov.value(next_state_proj)
            if (l_next - l_now - slack) <= self.feasibility_tol:
                break

        # 仍不可行 → 退到 backup（默认 = 全零物理动作；仅在 backup 也可行时采用）。
        if (l_next - l_now - slack) > self.feasibility_tol and backup_action is not None:
            backup = np.asarray(backup_action, dtype=np.float64).reshape(-1)
            if backup.size < out.size:
                backup = np.pad(backup, (0, out.size - backup.size))
            cand = out.copy()
            cand[:n_phys] = np.clip(backup[:n_phys], 0.0, 1.0)
            next_backup = self.predictor.step(state_phys, cand).to_dict()
            l_backup = self.lyapunov.value(next_backup)
            if l_backup < l_next:
                out = cand
                l_next = l_backup

        return LyapunovProjectionResult(
            action=out.astype(np.float32),
            intervened=True,
            iterations=iterations,
            l_now=float(l_now),
            l_next_raw=float(l_next_raw),
            l_next_projected=float(l_next),
            slack=float(slack),
            violation=float(max(0.0, l_next - l_now - slack)),
            grad_norm=float(grad_norm_last),
        )

    def _finite_diff_gradient(
        self,
        state_phys: dict,
        action: np.ndarray,
        *,
        l_baseline: float,
        n_phys: int,
    ) -> np.ndarray:
        eps = max(1e-6, float(self.finite_diff_eps))
        grad = np.zeros(n_phys, dtype=np.float64)
        for i in range(n_phys):
            a_plus = action.copy()
            a_plus[i] = float(np.clip(a_plus[i] + eps, 0.0, 1.0))
            # 中心差分会跨越 [0,1] 边界，所以用单边差分。
            l_plus = self.lyapunov.value(self.predictor.step(state_phys, a_plus).to_dict())
            grad[i] = (l_plus - l_baseline) / max(a_plus[i] - action[i], eps)
        return grad


# 历史导出名（experiments/compare_all.py 等历史代码仍按旧名 import）。
LyapunovActionProjection = LyapunovProjector


__all__ = ["LyapunovProjector", "LyapunovProjectionResult", "LyapunovActionProjection"]

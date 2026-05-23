"""推理时 short-horizon MPC planner（不改训练，仅在 schedule() 时介入）。

动机：actor (transformer-SAC) 的训练信号被 γ=0.995 限制在 ~150 步等效视野，但
卫星调度的下传决策需要 540 步前瞻（contact 窗口规律）。在推理时显式 rollout
H 步并按 reward + deliverable critic 给出的终端价值打分，让 planner 显式
利用周期性而不是依赖 γ^540 的弱信号。

算法（random shooting，CEM 的最简形式）：
  1. 从 actor 采样 1 个 mean action a_actor（评估模式）
  2. 在 a_actor 周围加高斯噪声生成 N-1 个候选 a_1..a_{N-1}（含 a_actor）
  3. 对每个候选：
       - 用 SafetyDynamicsPredictor rollout H 步（候选动作只用于第 1 步，
         后续 H-1 步用 a_actor，因为我们只关心"现在选哪个 first action"）
       - 累计内部 reward（用预测器算出的 delivered_mb 作为代理 reward）
       - 终端用 reward + deliverable critic 评估完整 obs
  4. 选 score 最高的候选的**第一个动作**

这跟 TD-MPC2 的本质区别：
  * 我们用真实物理 predictor，不是 latent world model
  * 用现有 critic 当终端 Q，不重训
  * planning horizon 短（H=10），不期望覆盖完整 540 步窗口；只是让 first
    action 比 reactive policy 更 forward-looking
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from config import DRL_CONFIG, INFERENCE_MPC_CONFIG, QUEUE_CONFIG, TRAIN_CONFIG
from safety.dynamics_predictor import SafetyDynamicsPredictor
from utils.action_space import PHYSICAL_ACTION_DIM


@dataclass(frozen=True)
class MPCPlanResult:
    action: np.ndarray
    used: bool
    best_score: float
    actor_score: float
    n_candidates_evaluated: int
    horizon: int


class InferenceMPCPlanner:
    """Short-horizon shooting planner，包裹 SAC actor 用于推理。"""

    def __init__(
        self,
        predictor: SafetyDynamicsPredictor | None = None,
        *,
        cfg: dict | None = None,
    ):
        cfg = cfg or INFERENCE_MPC_CONFIG
        self.cfg = dict(cfg)
        self.predictor = predictor or SafetyDynamicsPredictor()
        self.num_candidates = int(cfg.get("num_candidates", 32))
        self.horizon = int(cfg.get("horizon_steps", 10))
        self.noise_std_physical = float(cfg.get("noise_std_physical", 0.20))
        self.noise_std_priority = float(cfg.get("noise_std_priority", 0.10))
        self.delivered_weight = float(cfg.get("delivered_weight", 1.0))
        self.constraint_weight = float(cfg.get("constraint_weight", 1.0))
        self.terminal_weight = float(cfg.get("terminal_weight", 1.0))
        self.gamma = float(cfg.get("gamma", DRL_CONFIG.get("gamma", 0.99)))
        self.processed_queue_max_mb = float(QUEUE_CONFIG.get("comm_queue_max", 4096.0))
        self.dt_s = float(self.predictor.dt_s)

    # ── 主接口 ─────────────────────────────────────────────────────────
    def plan(
        self,
        *,
        observation: np.ndarray,
        physical_state: dict,
        actor_mean_action: np.ndarray,
        critic_value_fn,
        action_dim: int,
        rng: np.random.Generator | None = None,
    ) -> MPCPlanResult:
        """返回 first action 与诊断信息。

        参数：
          observation       : 当前 (T, D) obs（仅用于 critic 终端打分，rollout 不更新它）
          physical_state    : 反归一化后的物理量（给 predictor 用）
          actor_mean_action : actor 在 evaluate 模式下输出的 mean（用作基准 + 后续 rollout 默认动作）
          critic_value_fn   : callable(obs, action) -> scalar Q value（终端价值评估）
          action_dim        : 完整动作维度
          rng               : 可选 numpy Generator（确定性测试用）
        """
        rng = rng if rng is not None else np.random.default_rng()
        actor_action = np.asarray(actor_mean_action, dtype=np.float64).reshape(-1)
        if actor_action.size < action_dim:
            actor_action = np.pad(actor_action, (0, action_dim - actor_action.size))
        actor_action = actor_action[:action_dim]

        candidates = self._make_candidates(actor_action, rng)
        scores = np.zeros(len(candidates), dtype=np.float64)
        for i, cand in enumerate(candidates):
            scores[i] = self._score_candidate(
                cand,
                tail_action=actor_action,
                physical_state=physical_state,
                observation=observation,
                critic_value_fn=critic_value_fn,
            )

        best_idx = int(np.argmax(scores))
        best_action = candidates[best_idx].astype(np.float32)
        actor_score = float(scores[0])
        best_score = float(scores[best_idx])
        return MPCPlanResult(
            action=best_action,
            used=bool(best_idx != 0 or self.cfg.get("force_use_mpc_output", False)),
            best_score=best_score,
            actor_score=actor_score,
            n_candidates_evaluated=int(len(candidates)),
            horizon=self.horizon,
        )

    # ── candidate 生成 ─────────────────────────────────────────────────
    def _make_candidates(
        self,
        actor_action: np.ndarray,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        action_dim = actor_action.size
        cands: list[np.ndarray] = [np.clip(actor_action, 0.0, 1.0)]
        if self.num_candidates <= 1:
            return cands
        # 物理维度噪声较大（控制效果敏感），优先级维度噪声较小。
        std = np.full(action_dim, self.noise_std_priority, dtype=np.float64)
        std[:min(PHYSICAL_ACTION_DIM, action_dim)] = self.noise_std_physical
        for _ in range(self.num_candidates - 1):
            noise = rng.normal(0.0, std)
            cands.append(np.clip(actor_action + noise, 0.0, 1.0))
        # 额外加两个极端候选（探索边界）：
        if action_dim >= 3 and self.cfg.get("include_safe_anchor", True):
            cands.append(np.clip(np.concatenate([
                np.array([1.0, 0.0, 0.0]),  # 全推力 / 不耗 CPU / 不耗 TX → 高度优先
                actor_action[3:],
            ]), 0.0, 1.0))
            cands.append(np.clip(np.concatenate([
                np.array([0.0, 0.5, 1.0]),  # 零推力 + 中等 CPU + 全 TX → 下传优先
                actor_action[3:],
            ]), 0.0, 1.0))
        return cands

    # ── candidate 打分 ────────────────────────────────────────────────
    def _score_candidate(
        self,
        candidate: np.ndarray,
        *,
        tail_action: np.ndarray,
        physical_state: dict,
        observation: np.ndarray,
        critic_value_fn,
    ) -> float:
        # rollout 用 (cand, tail, tail, ..., tail)，长度 H。
        actions = [candidate] + [tail_action] * max(0, self.horizon - 1)
        trajectory = self.predictor.rollout(physical_state, actions)

        score = 0.0
        gamma_pow = 1.0
        current_qc = float(physical_state.get("processed_queue_mb", 0.0))
        for step in trajectory:
            # 内部 reward = 本步净下传 MB - 约束违反惩罚。
            delivered_mb = max(0.0, current_qc - step.processed_queue_mb)
            constraint_penalty = 0.0
            if step.processed_queue_mb >= 0.99 * self.processed_queue_max_mb:
                constraint_penalty += 1.0
            if step.soc <= 0.10:
                constraint_penalty += 1.0
            if step.altitude_m <= 130e3:
                constraint_penalty += 1.0
            score += gamma_pow * (
                self.delivered_weight * delivered_mb
                - self.constraint_weight * constraint_penalty
            )
            gamma_pow *= self.gamma
            current_qc = step.processed_queue_mb

        # 终端：用 critic 评估"延续 tail_action 的预期价值"。
        if self.terminal_weight > 0.0 and critic_value_fn is not None:
            terminal_q = float(critic_value_fn(observation, tail_action))
            score += gamma_pow * self.terminal_weight * terminal_q
        return float(score)


__all__ = ["InferenceMPCPlanner", "MPCPlanResult"]

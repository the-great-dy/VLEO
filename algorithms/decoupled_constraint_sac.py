"""解耦的 reward/constraint Critic SAC。

这是论文面向的算法类。外层 SACAgent 提供训练基础设施；
本类定义 CMDP 学习方程。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from drl.agent import SACAgent
from environment.satellite_env import OBSERVATION_FEATURES


class DecoupledConstraintSAC(SACAgent):
    """
    解耦约束-Critic SAC。

    Reward Q_r 只学任务收益。Constraint Q_c 学 CMDP 安全代价 c_t。
    Actor 目标：

        alpha log pi(a|s) - Q_r(s,a)
        + lambda_t Q_c(s,a)
        + eta * ||mu(s) - a_safe||^2

    最后一项是动作投影行为克隆项 AP-BC；仅在部署安全层改变动作的样本上使用。
    """

    # 按观测名称解析索引，避免 OBSERVATION_FEATURES 调整顺序后辅助头读错状态。
    _FEATURE_INDEX = {name: idx for idx, name in enumerate(OBSERVATION_FEATURES)}

    @classmethod
    def _feature_column(cls, current: torch.Tensor, name: str) -> torch.Tensor:
        if name not in cls._FEATURE_INDEX:
            raise KeyError(f"OBSERVATION_FEATURES 缺少 value auxiliary 特征: {name}")
        return current[:, cls._FEATURE_INDEX[name]]

    def _build_value_aux_targets(self, raw_states: torch.Tensor) -> torch.Tensor:
        """基于当前状态构造 3 类伪标签: 0=high_first, 1=balanced, 2=low_drop。"""
        current = raw_states[:, 0, :]
        # 辅助头只读 OBSERVATION_FEATURES 中的命名列，避免观测顺序变化后
        # value auxiliary 继续按旧索引解释状态。
        in_window = self._feature_column(current, "in_comm_window")
        raw_high = self._feature_column(current, "raw_high_queue_utilization")
        raw_low = self._feature_column(current, "raw_low_queue_utilization")
        proc_high = self._feature_column(current, "processed_high_queue_utilization")
        proc_low = self._feature_column(current, "processed_low_queue_utilization")
        exp_high = self._feature_column(current, "expiring_high_value_norm")
        exp_low = self._feature_column(current, "expiring_low_value_norm")
        processed_future_contact = self._feature_column(
            current, "processed_queue_future_contact_ratio")
        future_contact = self._feature_column(current, "future_contact_capacity_norm")
        raw_high_deliverable = self._feature_column(
            current, "raw_high_next_window_deliverable_ratio")
        processed_high_deliverable = self._feature_column(
            current, "processed_high_next_window_deliverable_ratio")
        deadline_mismatch = self._feature_column(
            current, "high_value_deadline_contact_mismatch")

        exp_high_weight = float(getattr(self, "value_aux_expiring_high_weight", 1.25))
        high_margin = float(getattr(self, "value_aux_high_pressure_margin", 1.10))
        exp_high_threshold = float(getattr(self, "value_aux_expiring_high_threshold", 0.10))
        low_margin = float(getattr(self, "value_aux_low_pressure_margin", 1.25))
        future_tight_threshold = float(getattr(
            self, "value_aux_future_contact_tight_threshold", 0.25))
        future_relaxed_threshold = float(getattr(
            self, "value_aux_future_contact_relaxed_threshold", 0.55))
        processed_pressure_threshold = float(getattr(
            self, "value_aux_processed_future_contact_threshold",
            getattr(self, "value_aux_processed_pressure_threshold", 0.75)))

        # high/low pressure 是行为提示，不进入 reward。reward 仍只看最终交付价值。
        high_pressure = raw_high + proc_high + exp_high_weight * exp_high
        low_pressure = raw_low + proc_low + exp_low
        tx_tight = (future_contact < future_tight_threshold) & (in_window < 0.5)
        tx_relaxed = (future_contact > future_relaxed_threshold) | (in_window > 0.5)

        high_first = (
            (high_pressure > high_margin * low_pressure)
            & (
                (exp_high > exp_high_threshold)
                | tx_tight
                | ((raw_high_deliverable > 0.35) & (processed_high_deliverable > 0.35))
            )
        )
        low_drop = (
            (low_pressure > low_margin * high_pressure)
            & (
                tx_tight
                | ((processed_future_contact > processed_pressure_threshold) & (~tx_relaxed))
                | ((deadline_mismatch > 0.65) & (processed_future_contact > 0.80))
            )
        )

        targets = torch.ones_like(high_pressure, dtype=torch.long)
        targets = torch.where(high_first, torch.zeros_like(targets), targets)
        targets = torch.where(low_drop & (~high_first), torch.full_like(targets, 2), targets)
        return targets

    def _decode_action_ratios_for_loss(
        self,
        action: torch.Tensor,
        scale: float = 4.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """以可微方式解码紧凑优先级动作，用于辅助损失。"""
        cpu_value = torch.clamp(action[:, 3], 0.0, 1.0)
        cpu_urgency = torch.clamp(action[:, 4], 0.0, 1.0)
        tx_value = torch.clamp(action[:, 5], 0.0, 1.0)
        tx_urgency = torch.clamp(action[:, 6], 0.0, 1.0)
        cpu_logits = torch.stack([cpu_value, cpu_urgency, 0.5 * torch.ones_like(cpu_value)], dim=-1)
        tx_logits = torch.stack([tx_value, tx_urgency, 0.5 * torch.ones_like(tx_value)], dim=-1)
        cpu_ratios = torch.softmax(cpu_logits * scale, dim=-1)
        tx_ratios = torch.softmax(tx_logits * scale, dim=-1)
        drop_low = torch.clamp(action[:, 7], 0.0, 1.0)
        return cpu_ratios, tx_ratios, drop_low

    def compute_td_targets(
        self,
        r: torch.Tensor,
        d: torch.Tensor,
        c: torch.Tensor,
        deliverable_r: torch.Tensor,
        reward_next_q: torch.Tensor,
        deliverable_next_q: torch.Tensor,
        constraint_next_q: torch.Tensor,
        gamma_pow: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """为 reward、deliverable 和 constraint critics 分别计算 Bellman 目标。

        gamma_pow=None → 单步：γ。
        gamma_pow tensor → n-step：γ^n_eff（per-sample，buffer 存储）。
        n-step 下 r/c/deliverable_r 应该已经是 R_n / C_n / D_n（NStepAggregator 累计后）。
        """
        g = gamma_pow if gamma_pow is not None else float(self.gamma)
        target_reward_q = r + (1.0 - d) * g * reward_next_q
        target_deliverable_q = deliverable_r + (1.0 - d) * g * deliverable_next_q
        target_constraint_q = c + (1.0 - d) * g * constraint_next_q
        return target_reward_q, target_deliverable_q, target_constraint_q

    def compute_actor_objective(
        self,
        normalized_states: torch.Tensor,
        behavior_actions: torch.Tensor,
        behavior_weights: torch.Tensor,
        raw_states: torch.Tensor | None = None,
    ) -> dict:
        """计算 LS-PSF CMDP actor 目标。"""
        action, log_pi, mean_action = self.actor.sample(normalized_states)
        q1_reward, q2_reward = self.critic(normalized_states, action)
        if self._deliverable_critic_enabled:
            q1_deliverable, q2_deliverable = self.deliverable_critic(
                normalized_states, action)
        q1_constraint, q2_constraint = self.constraint_critic(
            normalized_states, action)

        # Reward critic 和 deliverable critic 联合指导 actor；constraint critic 仅在 lyapunov 开启时惩罚。
        sac_reward = torch.min(q1_reward, q2_reward)
        if self._deliverable_critic_enabled:
            sac_reward = sac_reward + self._deliverable_critic_actor_coeff * torch.min(q1_deliverable, q2_deliverable)
        sac_actor_loss = (
            self.alpha * log_pi - sac_reward
        ).mean()
        if self.lya_coeff > 0.0:
            constraint_actor_loss = (
                self.lya_coeff * torch.max(q1_constraint, q2_constraint).mean()
            )
        else:
            constraint_actor_loss = torch.zeros(
                (), device=self.device, dtype=sac_actor_loss.dtype)

        behavior_weights = torch.clamp(
            behavior_weights, 0.0, self.behavior_cloning_max_weight)
        behavior_cloning_loss = torch.zeros(
            (), device=self.device, dtype=sac_actor_loss.dtype)
        if self.behavior_cloning_coeff > 0.0 and self.behavior_cloning_max_weight > 0.0:
            weight_sum = behavior_weights.sum()
            if weight_sum.item() > 1e-8:
                # AP-BC only imitates safety-layer corrections stored with
                # nonzero behavior_weights; unmodified samples do not pull the policy.
                per_sample_loss = F.mse_loss(
                    mean_action, behavior_actions, reduction="none"
                ).mean(dim=-1, keepdim=True)
                behavior_cloning_loss = (
                    per_sample_loss * behavior_weights
                ).sum() / torch.clamp(weight_sum, min=1e-8)

        value_aux_weight = float(self._current_value_aux_weight())
        value_aux_loss = torch.zeros(
            (), device=self.device, dtype=sac_actor_loss.dtype)
        value_aux_accuracy = 0.0
        targets = None
        if self.value_aux_head_enable and raw_states is not None and value_aux_weight > 0.0:
            aux_logits = self.actor.predict_value_priority_logits(normalized_states)
            if aux_logits is not None:
                targets = self._build_value_aux_targets(raw_states)
                value_aux_loss = F.cross_entropy(aux_logits, targets)
                preds = torch.argmax(aux_logits, dim=-1)
                value_aux_accuracy = float((preds == targets).float().mean().item())

        value_action_aux_loss = torch.zeros(
            (), device=self.device, dtype=sac_actor_loss.dtype)
        value_action_aux_weight = float(self._current_value_action_aux_weight())
        if raw_states is not None and value_action_aux_weight > 0.0:
            if targets is None:
                targets = self._build_value_aux_targets(raw_states)
            cpu_ratios, tx_ratios, drop_low = self._decode_action_ratios_for_loss(mean_action)
            low_drop_mask = targets == 2
            high_first_mask = targets == 0
            if low_drop_mask.any():
                value_action_aux_loss = value_action_aux_loss + (
                    (1.0 - drop_low[low_drop_mask]).pow(2).mean()
                    + cpu_ratios[low_drop_mask, 2].pow(2).mean()
                    + tx_ratios[low_drop_mask, 2].pow(2).mean()
                )
            if high_first_mask.any():
                # high-first 只轻推 TX high 优先级；不再把 CPU high 写成硬规则，
                # 避免单站/窄窗口场景下过早处理造成 processed backlog 偏高。
                value_action_aux_loss = value_action_aux_loss + (
                    0.25 * (1.0 - tx_ratios[high_first_mask, 0]).pow(2).mean()
                    + 0.10 * drop_low[high_first_mask].pow(2).mean()
                )

        actor_loss = (
            sac_actor_loss
            + constraint_actor_loss
            + self.behavior_cloning_coeff * behavior_cloning_loss
            + value_aux_weight * value_aux_loss
            + value_action_aux_weight * value_action_aux_loss
        )
        return {
            "actor_loss": actor_loss,
            "sac_actor_loss": sac_actor_loss,
            "constraint_actor_loss": constraint_actor_loss,
            "behavior_cloning_loss": behavior_cloning_loss,
            "value_aux_loss": value_aux_loss,
            "value_aux_weight": value_aux_weight,
            "value_aux_accuracy": value_aux_accuracy,
            "value_action_aux_loss": value_action_aux_loss,
            "value_action_aux_weight": value_action_aux_weight,
            "behavior_w_clamped": behavior_weights,
            "log_pi": log_pi,
            "finite_tensors": (
                action, log_pi, q1_reward, q2_reward,
                q1_deliverable if self._deliverable_critic_enabled else q1_reward,
                q2_deliverable if self._deliverable_critic_enabled else q2_reward,
                q1_constraint, q2_constraint, actor_loss,
                constraint_actor_loss, behavior_cloning_loss, value_aux_loss,
                value_action_aux_loss,
            ),
        }

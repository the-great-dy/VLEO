"""
上帝视角 rollout MPC 上界基线。

上帝视角 rollout MPC 基线。

该基线会复制当前环境，在复制体中滚动展开候选动作树，因此能看到当前
episode 的未来随机流、通信窗口、场景相位和队列演化。它不是可部署策略，
而是用于论文对照的 oracle upper-bound proxy，用来回答：
“RL 的优势是否只是因为传统 MPC 被过短视野限制住？”
"""

from __future__ import annotations

import copy
import sys
import os

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from config import ENERGY_CONFIG, TRAIN_CONFIG


class OracleMPCBaseline:
    """
    复制环境做未来 rollout 的 MPC。

    参数:
        horizon: 未来展开步数。比普通 MPC 的短视野更长。
        beam_width: 每一层保留的高分候选序列数量，控制计算开销。
        discount: 未来分数折扣。
    """

    def __init__(self, horizon: int = 12, beam_width: int = 8,
                 discount: float = 0.995):
        self.horizon = max(1, int(horizon))
        self.beam_width = max(1, int(beam_width))
        self.discount = float(np.clip(discount, 0.0, 1.0))
        self.action_candidates = self._build_action_candidates()

    @staticmethod
    def _build_action_candidates() -> list[np.ndarray]:
        prop_floor = (
            float(ENERGY_CONFIG.get("propulsion_ignition_threshold_w", 0.0))
            / max(float(ENERGY_CONFIG.get("power_propulsion_max_w", 1.0)), 1e-9)
        )
        prop_floor = float(np.clip(prop_floor, 0.0, 1.0))
        templates = [
            [0.0, 0.0, 0.0],
            [prop_floor, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [prop_floor, 1.0, 0.0],
            [prop_floor, 0.0, 1.0],
            [prop_floor, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [1.0, 0.3, 0.0],
            [0.0, 0.5, 1.0],
        ]
        unique = []
        seen = set()
        for item in templates:
            arr = np.asarray(item, dtype=np.float32)
            arr = np.clip(arr, 0.0, 1.0)
            key = tuple(np.round(arr, 6).tolist())
            if key not in seen:
                seen.add(key)
                unique.append(arr)
        return unique

    def schedule(self, state: np.ndarray, env) -> np.ndarray:
        del state
        if env is None:
            return np.array([0.0, 0.5, 0.5], dtype=np.float32)

        beam = [(0.0, copy.deepcopy(env), None, False)]
        for depth in range(self.horizon):
            expanded = []
            for prefix_score, env_copy, first_action, is_done in beam:
                if is_done:
                    expanded.append((prefix_score, env_copy, first_action, True))
                    continue
                for action in self.action_candidates:
                    trial_env = copy.deepcopy(env_copy)
                    _, reward, done, info = trial_env.step(action.copy())
                    step_score = self._score_step(float(reward), info, depth)
                    first = action.copy() if first_action is None else first_action
                    expanded.append((prefix_score + step_score, trial_env, first, bool(done)))
            if not expanded:
                break
            expanded.sort(key=lambda item: item[0], reverse=True)
            beam = expanded[:self.beam_width]

        if not beam:
            return np.array([0.0, 0.5, 0.5], dtype=np.float32)
        return np.asarray(beam[0][2], dtype=np.float32)

    def _score_step(self, reward: float, info: dict, depth: int) -> float:
        weight = self.discount ** depth
        delivered_value = float(info.get("delivered_value", 0.0))
        deadline_value = float(info.get("deadline_success_value", 0.0))
        downlink_mb = float(info.get("delivered_mb", info.get("actual_tx_mb", 0.0)))
        processed_mb = float(info.get("processed_mb", 0.0))
        safety_penalty = 0.0
        if not bool(info.get("overall_safe", True)):
            safety_penalty -= 250.0
        if bool(info.get("terminated", False)):
            safety_penalty -= 1000.0
        queue_penalty = (
            float(info.get("raw_queue_overflow_mb", 0.0))
            + float(info.get("processed_queue_overflow_mb", 0.0))
        )
        # reward 保留环境目标，额外强调交付价值与安全，避免 oracle 为局部 shaping 牺牲主目标。
        score = (
            reward
            + 2.0 * delivered_value
            + 0.5 * deadline_value
            + 0.05 * downlink_mb
            + 0.01 * processed_mb
            - 2.0 * queue_penalty
            + safety_penalty
        )
        return float(weight * score)

    @property
    def metadata(self) -> dict:
        return {
            "baseline_type": "oracle_rollout_mpc_upper_bound_proxy",
            "horizon": self.horizon,
            "beam_width": self.beam_width,
            "time_slot_s": float(TRAIN_CONFIG.get("time_slot_s", 10.0)),
            "uses_future_environment_rollout": True,
            "deployable_online_policy": False,
        }

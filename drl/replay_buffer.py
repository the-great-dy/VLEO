"""Transformer SAC 的经验回放和时序序列缓冲区。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from config import DRL_CONFIG


@dataclass(frozen=True)
class TransitionBoundary:
    """明确的转移边界语义。

    episode_done 用于环境重置。terminated 是唯一存储在回放中用于
    TD bootstrap masking 的值；truncated/time-limit 转移必须继续 bootstrap。
    """

    terminated: bool
    truncated: bool

    @property
    def episode_done(self) -> bool:
        return bool(self.terminated or self.truncated)

    @property
    def bootstrap_mask(self) -> float:
        return 0.0 if self.terminated else 1.0

    @classmethod
    def from_step(cls, done: bool, info: dict | None = None) -> "TransitionBoundary":
        info = info or {}
        if "terminated" in info or "truncated" in info:
            terminated = bool(info.get("terminated", False))
            truncated = bool(info.get("truncated", False))
        else:
            terminated = bool(done)
            truncated = False
        return cls(terminated=terminated, truncated=truncated)


@dataclass(frozen=True)
class TemporalStackSpec:
    """堆叠观测序列的形状和采样 offset。"""

    frame_stack: int
    state_dim: int
    offsets: tuple[int, ...]

    @classmethod
    def from_config(
        cls,
        *,
        state_dim: int,
        frame_stack: int | None = None,
        offsets: list[int] | tuple[int, ...] | None = None,
    ) -> "TemporalStackSpec":
        k = int(frame_stack if frame_stack is not None else DRL_CONFIG.get("frame_stack", 1))
        if offsets is None:
            offsets = tuple(range(k))
        else:
            offsets = tuple(int(x) for x in offsets[:k])
        if len(offsets) != k:
            raise ValueError(f"offsets length {len(offsets)} != frame_stack {k}")
        return cls(frame_stack=k, state_dim=int(state_dim), offsets=tuple(offsets))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.frame_stack, self.state_dim)


class TemporalHistoryBuffer:
    """可复用的时序采样器，当前帧在 token 0。"""

    def __init__(self, spec: TemporalStackSpec):
        self.spec = spec
        self.max_offset = max(spec.offsets) if spec.offsets else 0
        self.frames = deque(maxlen=self.max_offset + 1)

    def reset(self, obs) -> np.ndarray:
        frame = np.asarray(obs, dtype=np.float32).copy()
        self.frames.clear()
        for _ in range(self.max_offset + 1):
            self.frames.append(frame.copy())
        return self.stack()

    def append(self, obs) -> np.ndarray:
        self.frames.append(np.asarray(obs, dtype=np.float32).copy())
        return self.stack()

    def replace_latest(self, obs) -> None:
        if not self.frames:
            self.reset(obs)
            return
        self.frames[-1] = np.asarray(obs, dtype=np.float32).copy()

    def stack(self) -> np.ndarray:
        if not self.frames:
            return np.zeros(self.spec.shape, dtype=np.float32)
        n = len(self.frames)
        sampled = []
        for offset in self.spec.offsets:
            idx = max(0, n - 1 - int(offset))
            sampled.append(self.frames[idx])
        return np.asarray(sampled, dtype=np.float32)


class ReplayBuffer:
    """Numpy replay storage for (time, feature) Transformer observations."""

    def __init__(
        self,
        capacity: int,
        state_dim: int = 40,
        action_dim: int | None = None,
        frame_stack: int | None = None,
    ):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim or DRL_CONFIG.get("action_dim", 10))
        self.ptr = 0
        self.size = 0
        self.temporal_spec = TemporalStackSpec.from_config(
            state_dim=self.state_dim,
            frame_stack=frame_stack,
        )
        self.T = self.temporal_spec.frame_stack

        self.states = np.zeros((self.capacity, self.T, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.T, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.lya_drifts = np.zeros((self.capacity, 1), dtype=np.float32)
        self.deliverable_rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.behavior_actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.behavior_weights = np.zeros((self.capacity, 1), dtype=np.float32)
        # n-step TD：存储这条转移对应的 γ^n_eff（默认 1.0 表示"单步，由调用方乘 γ"）。
        # NStepAggregator 在 push 时会传入真实 γ^n；旧调用方留默认 → critic 维持原行为。
        self.n_step_gamma_pow = np.zeros((self.capacity, 1), dtype=np.float32)

    def _coerce_state(self, state) -> np.ndarray:
        arr = np.asarray(state, dtype=np.float32)
        if arr.shape != self.temporal_spec.shape:
            raise ValueError(
                f"state shape {arr.shape} != expected {self.temporal_spec.shape}"
            )
        return arr

    def _coerce_action(self, action) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size < self.action_dim:
            arr = np.pad(arr, (0, self.action_dim - arr.size), mode="constant")
        elif arr.size > self.action_dim:
            arr = arr[:self.action_dim]
        return arr.astype(np.float32, copy=False)

    def push(
        self,
        s,
        a,
        r,
        s2,
        d,
        lya=0.0,
        deliverable_reward: float = 0.0,
        behavior_action=None,
        behavior_weight: float = 0.0,
        n_step_gamma_pow: float = 0.0,
    ):
        i = self.ptr
        self.states[i] = self._coerce_state(s)
        self.actions[i] = self._coerce_action(a)
        self.rewards[i] = float(r)
        self.next_states[i] = self._coerce_state(s2)
        self.dones[i] = float(bool(d))
        self.lya_drifts[i] = float(lya)
        self.deliverable_rewards[i] = float(deliverable_reward)
        if behavior_action is None:
            self.behavior_actions[i] = self.actions[i]
            self.behavior_weights[i] = 0.0
        else:
            self.behavior_actions[i] = self._coerce_action(behavior_action)
            self.behavior_weights[i] = max(float(behavior_weight), 0.0)
        # 0.0 (sentinel) → critic 走老路用 self.gamma；>0 → 走 n-step。
        self.n_step_gamma_pow[i] = float(n_step_gamma_pow)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, int(batch_size))
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
            self.lya_drifts[idx],
            self.deliverable_rewards[idx],
            self.behavior_actions[idx],
            self.behavior_weights[idx],
            self.n_step_gamma_pow[idx],
        )

    def __len__(self):
        return self.size

"""Transformer 时序观测的环境包装器。

包装器现在是 drl.replay_buffer.TemporalHistoryBuffer 的薄适配层。
这让帧堆叠成为标准的时序采样机制，而不是在每个包装器内手写队列逻辑。
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

import numpy as np

from config import DRL_CONFIG
from drl.replay_buffer import TemporalHistoryBuffer, TemporalStackSpec


class FrameStackWrapper:
    """保留连续帧堆叠以对比稀释采样的消融。"""

    def __init__(self, env, k: int):
        self.env = env
        self.k = int(k)
        state_dim = int(getattr(env, "state_dim", DRL_CONFIG.get("state_dim", 40)))
        self.temporal_spec = TemporalStackSpec.from_config(
            state_dim=state_dim,
            frame_stack=self.k,
            offsets=tuple(range(self.k)),
        )
        self._temporal_history = TemporalHistoryBuffer(self.temporal_spec)
        # Compatibility for older scripts that inspect the raw deque.
        self.frames = self._temporal_history.frames

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        return self._temporal_history.reset(obs)

    def step(self, action: np.ndarray, **kwargs) -> tuple:
        obs, reward, done, info = self.env.step(action, **kwargs)
        return self._temporal_history.append(obs), reward, done, info

    def _get_obs(self) -> np.ndarray:
        return self._temporal_history.stack()

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.env, name)


class DilatedFrameStackWrapper:
    """
    稀释时序采样器。

    Offset 是离当前步的距离，token 0 总是最新观测。输出形状为 (k, state_dim)。
    """

    DEFAULT_OFFSETS = [0, 1, 3, 9, 27, 90, 270, 540]

    def __init__(self, env, k: int = 8, offsets: list = None):
        self.env = env
        self.k = int(k)
        if offsets is not None:
            self.offsets = sorted(int(x) for x in offsets)
        else:
            cfg_offsets = DRL_CONFIG.get("dilated_offsets", self.DEFAULT_OFFSETS)
            self.offsets = sorted(int(x) for x in list(cfg_offsets)[:self.k])
        if len(self.offsets) != self.k:
            raise ValueError(
                f"offsets length ({len(self.offsets)}) must equal k ({self.k})")

        state_dim = int(getattr(env, "state_dim", DRL_CONFIG.get("state_dim", 40)))
        self.temporal_spec = TemporalStackSpec.from_config(
            state_dim=state_dim,
            frame_stack=self.k,
            offsets=tuple(self.offsets),
        )
        self._temporal_history = TemporalHistoryBuffer(self.temporal_spec)
        self._max_offset = self._temporal_history.max_offset
        # Compatibility: trace robustness scripts force raw observations here.
        self._history = self._temporal_history.frames

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        return self._temporal_history.reset(obs)

    def step(self, action: np.ndarray, **kwargs) -> tuple:
        obs, reward, done, info = self.env.step(action, **kwargs)
        return self._temporal_history.append(obs), reward, done, info

    def _get_obs(self) -> np.ndarray:
        return self._temporal_history.stack()

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.env, name)

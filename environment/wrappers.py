"""Environment wrappers for Transformer temporal observations.

The wrapper is now a thin adapter around drl.replay_buffer.TemporalHistoryBuffer.
That keeps frame stacking as a standard temporal sampling mechanism instead of
hand-written queue logic inside each wrapper.
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

import numpy as np

from config import DRL_CONFIG
from drl.replay_buffer import TemporalHistoryBuffer, TemporalStackSpec


class FrameStackWrapper:
    """Continuous frame stacking kept for ablation against dilated sampling."""

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
    Dilated temporal sampler.

    Offsets are distances from the current step, and token 0 is always the
    latest observation. The output shape is (k, state_dim).
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

"""N-step return aggregator（SAC 标准扩展）。

把单步转移 (s_t, a_t, r_t, s_{t+1}, d_t) 累计成 n-step：
    s_t, a_t, R_n_t, s_{t+n}, d_t..t+n, γ^n_eff

其中 R_n_t = r_t + γ·r_{t+1} + ... + γ^(n-1)·r_{t+n-1}（直到 done 截断）。
γ^n_eff 表示这条转移最终走了多少步，用于 critic target 里替换原来的 γ。

episode 终止处理：
  * 收到 terminated=True → flush 队列中所有 partial-n 转移（每个用各自累计长度）。
  * truncated（time limit）→ 走 bootstrap 路径，等同非终止；调用方 done=False 即可。

`deliverable_reward` 和 `lya_drift` 也按同样的折扣累计（保持三个 critic 一致）。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class _Transition:
    s: Any
    a: Any
    r: float
    s2: Any
    d: bool
    lya: float
    deliverable_r: float
    raw_action: Any
    behavior_action: Any
    behavior_weight: float


@dataclass
class NStepTransition:
    """要 push 到 replay buffer 的一条 n-step 累计转移。"""

    s: Any
    a: Any
    r: float
    s2: Any
    d: bool
    lya: float
    deliverable_r: float
    raw_action: Any
    behavior_action: Any
    behavior_weight: float
    n_step_gamma_pow: float


class NStepAggregator:
    """无状态队列：每次 ingest 一条新转移，返回 0 或多条已就绪的 n-step 转移。"""

    def __init__(self, n: int, gamma: float):
        self.n = max(1, int(n))
        self.gamma = float(gamma)
        self._queue: deque[_Transition] = deque()

    def ingest(
        self,
        s,
        a,
        r: float,
        s2,
        d: bool,
        lya: float = 0.0,
        deliverable_reward: float = 0.0,
        raw_action=None,
        behavior_action=None,
        behavior_weight: float = 0.0,
    ) -> list[NStepTransition]:
        """加入一条新转移，返回新 ready 的 n-step 转移列表（可能为空）。"""
        self._queue.append(_Transition(
            s=s, a=a, r=float(r), s2=s2, d=bool(d), lya=float(lya),
            deliverable_r=float(deliverable_reward),
            raw_action=raw_action,
            behavior_action=behavior_action,
            behavior_weight=float(behavior_weight),
        ))
        out: list[NStepTransition] = []

        if bool(d):
            # 终止：flush 所有 partial 转移，每条用各自从 head 到 tail 的累积长度。
            while self._queue:
                head_idx_from_tail = len(self._queue)  # n_eff
                out.append(self._build_nstep(head_idx_from_tail))
                self._queue.popleft()
            return out

        # 非终止：只在队列长度达到 n 时 flush 一条（head 走完整 n 步）。
        if len(self._queue) >= self.n:
            out.append(self._build_nstep(self.n))
            self._queue.popleft()
        return out

    def reset(self, *, flush: bool = False) -> list[NStepTransition]:
        """在 episode 边界重置队列。

        flush=True 时先输出剩余的 partial n-step transition，再清空队列。
        这适用于 time-limit truncation：样本仍可从最后一个 next_state bootstrap，
        但绝不能与下一条 episode 的初始状态拼接。
        """
        if not flush:
            self._queue.clear()
            return []

        out: list[NStepTransition] = []
        while self._queue:
            out.append(self._build_nstep(len(self._queue)))
            self._queue.popleft()
        return out

    # ── 内部：从 queue head 开始累计 n_steps 步 ──────────────────────────
    def _build_nstep(self, n_steps: int) -> NStepTransition:
        n_steps = max(1, int(min(n_steps, len(self._queue))))
        head = self._queue[0]
        R = 0.0
        D_acc = 0.0
        L_acc = 0.0
        gamma_pow = 1.0
        end_t = head
        n_eff = 0
        for k in range(n_steps):
            t = self._queue[k]
            R += gamma_pow * t.r
            D_acc += gamma_pow * t.deliverable_r
            L_acc += gamma_pow * t.lya
            gamma_pow *= self.gamma
            end_t = t
            n_eff += 1
            if bool(t.d):
                break
        # gamma_pow 现在 = γ^n_eff（已经乘过 n_eff 次）。
        return NStepTransition(
            s=head.s, a=head.a,
            r=float(R), s2=end_t.s2, d=bool(end_t.d),
            lya=float(L_acc),
            deliverable_r=float(D_acc),
            raw_action=head.raw_action,
            behavior_action=head.behavior_action,
            behavior_weight=head.behavior_weight,
            n_step_gamma_pow=float(gamma_pow),
        )


__all__ = ["NStepAggregator", "NStepTransition"]

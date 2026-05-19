"""Shared CMDP queue state and Lyapunov helpers."""

from __future__ import annotations


class BaseVirtualQueue:
    """Common base for queue value, history, drift and Lyapunov weight."""

    def __init__(self, max_value: float, lyapunov_weight_scale: float = 0.0):
        self.max_value = float(max_value)
        self.lyapunov_weight_scale = float(lyapunov_weight_scale)
        self.value = 0.0
        self.prev_value = 0.0
        self.history = []

    def _coerce_value(self, value: float) -> float:
        """Keep queue state finite and inside its physical buffer range."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        if numeric != numeric:
            numeric = 0.0
        return float(max(0.0, min(numeric, self.max_value)))

    def _reset_value(self, value: float = 0.0) -> None:
        self.prev_value = 0.0
        self.value = self._coerce_value(value)
        self.history = []

    def _begin_update(self) -> None:
        self.prev_value = float(self.value)

    def _set_value(self, value: float) -> None:
        self.value = self._coerce_value(value)
        self.history.append(self.value)

    @property
    def drift(self) -> float:
        return 0.5 * (self.value ** 2 - self.prev_value ** 2)

    @property
    def lyapunov_weight(self) -> float:
        return self.value * self.lyapunov_weight_scale

    @property
    def urgency(self) -> float:
        return self.value / max(self.max_value, 1e-6)

    @property
    def is_stable(self) -> bool:
        return self.value < self.max_value * 0.8

    def state_dict(self) -> dict:
        return {
            "queue_value": float(self.value),
            "drift": float(self.drift),
            "is_stable": bool(self.is_stable),
            "urgency": float(self.urgency),
        }

"""Backward-compatible logger imports.

New code should import MetricLogger from utils.metric_logger.  The old
TrainingLogger name is kept so existing scripts keep working.
"""

from utils.metric_logger import (
    MetricJSONEncoder,
    MetricLogger,
    TrainingLogger,
    configure_console_logging,
)

__all__ = [
    "MetricJSONEncoder",
    "MetricLogger",
    "TrainingLogger",
    "configure_console_logging",
]

"""LS-PSF scheduler entrypoint.

The paper-facing name is LS-PSF:

    pi_theta(s) -> Pi_feas -> Pi_LS_PSF -> deployment safeguards

`IntegratedScheduler` is kept as the implementation class for backward
compatibility with existing experiment scripts.
"""

from scheduler.integrated_scheduler import IntegratedScheduler


class LSPSFScheduler(IntegratedScheduler):
    """Named alias for the Lyapunov-Shielded Predictive Safety Filter scheduler."""


__all__ = ["LSPSFScheduler", "IntegratedScheduler"]

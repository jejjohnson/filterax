"""Ensemble smoothers and the exact Kalman smoother.

Backward-pass refiners that turn a sequential-filter history into the
smoothing distribution ``p(x_t | y_{1:T})``. The ensemble smoothers
operate on the stacked ``forecast_history`` / ``analysis_history``
produced by an :class:`filterax.AssimilationResult`; the linear-Gaussian
:class:`KalmanSmoother` runs its own exact filter over the observations.
"""

from filterax._filters._parametric import (
    KalmanSmoother as KalmanSmoother,
    KalmanSmoothingResult as KalmanSmoothingResult,
)
from filterax._smoothers import (
    IES as IES,
    EnKS as EnKS,
    EnsembleRTS as EnsembleRTS,
    EnsembleSqrtSmoother as EnsembleSqrtSmoother,
    FixedLagSmoother as FixedLagSmoother,
)

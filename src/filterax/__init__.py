"""filterax — differentiable ensemble Kalman filters and processes for JAX."""

from filterax._src._protocols import (
    AbstractDynamics as AbstractDynamics,
    AbstractInflator as AbstractInflator,
    AbstractLocalizer as AbstractLocalizer,
    AbstractNoise as AbstractNoise,
    AbstractObsOperator as AbstractObsOperator,
    AbstractProcess as AbstractProcess,
    AbstractScheduler as AbstractScheduler,
    AbstractSequentialFilter as AbstractSequentialFilter,
)
from filterax._src._types import (
    AnalysisResult as AnalysisResult,
    FilterConfig as FilterConfig,
    FilterState as FilterState,
    ProcessConfig as ProcessConfig,
    ProcessState as ProcessState,
    UKIState as UKIState,
)
from filterax._src.gain import kalman_gain as kalman_gain
from filterax._src.likelihood import (
    InnovationStatistics as InnovationStatistics,
    innovation_statistics as innovation_statistics,
    log_likelihood as log_likelihood,
)
from filterax._src.statistics import (
    cross_covariance as cross_covariance,
    ensemble_anomalies as ensemble_anomalies,
    ensemble_covariance as ensemble_covariance,
    ensemble_mean as ensemble_mean,
)


__version__ = "0.1.0"

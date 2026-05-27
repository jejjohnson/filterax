"""filterax — differentiable ensemble Kalman filters and processes for JAX."""

from filterax import filters as filters, optax as optax, processes as processes
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
    AssimilationResult as AssimilationResult,
    FilterConfig as FilterConfig,
    FilterState as FilterState,
    ProcessConfig as ProcessConfig,
    ProcessState as ProcessState,
    UKIState as UKIState,
)
from filterax._src.gain import kalman_gain as kalman_gain
from filterax._src.inflation import (
    inflate_multiplicative as inflate_multiplicative,
    inflate_rtpp as inflate_rtpp,
    inflate_rtps as inflate_rtps,
)
from filterax._src.inflators import (
    RTPP as RTPP,
    RTPS as RTPS,
    MultiplicativeInflator as MultiplicativeInflator,
)
from filterax._src.likelihood import (
    InnovationStatistics as InnovationStatistics,
    innovation_covariance as innovation_covariance,
    innovation_statistics as innovation_statistics,
    log_likelihood as log_likelihood,
)
from filterax._src.localization import (
    gaspari_cohn as gaspari_cohn,
    gaussian_taper as gaussian_taper,
    hard_cutoff as hard_cutoff,
    localize as localize,
)
from filterax._src.models import (
    ETKF as ETKF,
    LETKF as LETKF,
    EnSRF as EnSRF,
    StochasticEnKF as StochasticEnKF,
)
from filterax._src.perturbations import (
    perturbed_observations as perturbed_observations,
)
from filterax._src.process_models import (
    EKI as EKI,
    EKS as EKS,
    UKI as UKI,
    ProcessResult as ProcessResult,
)
from filterax._src.schedulers import (
    DataMisfitController as DataMisfitController,
    EKSStableScheduler as EKSStableScheduler,
    FixedScheduler as FixedScheduler,
)
from filterax._src.statistics import (
    cross_covariance as cross_covariance,
    ensemble_anomalies as ensemble_anomalies,
    ensemble_covariance as ensemble_covariance,
    ensemble_mean as ensemble_mean,
)


__version__ = "0.0.0"

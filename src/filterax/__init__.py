"""filterax — differentiable ensemble Kalman filters and processes for JAX."""

from filterax import (
    differentiable as differentiable,
    filters as filters,
    optax as optax,
    pipekit as pipekit,
    processes as processes,
    smoothers as smoothers,
    utils as utils,
)
from filterax._filters._latent import (
    EncodedDynamics as EncodedDynamics,
    IdentityLatentMap as IdentityLatentMap,
    LatentDynamics as LatentDynamics,
    LiftedObs as LiftedObs,
    decode_ensemble as decode_ensemble,
    identity_latent_map as identity_latent_map,
    latent_ensemble as latent_ensemble,
)
from filterax._filters._models import (
    ETKF as ETKF,
    LETKF as LETKF,
    EnSRF as EnSRF,
    LatentETKF as LatentETKF,
    LatentLETKF as LatentLETKF,
    StochasticEnKF as StochasticEnKF,
)
from filterax._primitives._gain import (
    kalman_gain as kalman_gain,
    localized_kalman_gain as localized_kalman_gain,
)
from filterax._primitives._inflation import (
    inflate_adaptive as inflate_adaptive,
    inflate_additive as inflate_additive,
    inflate_multiplicative as inflate_multiplicative,
    inflate_rtpp as inflate_rtpp,
    inflate_rtps as inflate_rtps,
    ledoit_wolf_shrinkage as ledoit_wolf_shrinkage,
)
from filterax._primitives._inflators import (
    RTPP as RTPP,
    RTPS as RTPS,
    AdditiveInflator as AdditiveInflator,
    MultiplicativeInflator as MultiplicativeInflator,
)
from filterax._primitives._likelihood import (
    InnovationStatistics as InnovationStatistics,
    innovation_covariance as innovation_covariance,
    innovation_statistics as innovation_statistics,
    log_likelihood as log_likelihood,
)
from filterax._primitives._localization import (
    adaptive_localization as adaptive_localization,
    euclidean_distance as euclidean_distance,
    gaspari_cohn as gaspari_cohn,
    gaussian_taper as gaussian_taper,
    hard_cutoff as hard_cutoff,
    haversine_distance as haversine_distance,
    localization_matrix as localization_matrix,
    localize as localize,
    soar_taper as soar_taper,
)
from filterax._primitives._perturbations import (
    perturbed_observations as perturbed_observations,
)
from filterax._primitives._statistics import (
    cross_covariance as cross_covariance,
    ensemble_anomalies as ensemble_anomalies,
    ensemble_covariance as ensemble_covariance,
    ensemble_mean as ensemble_mean,
)
from filterax._processes._process_models import (
    EKI as EKI,
    EKS as EKS,
    UKI as UKI,
    ProcessResult as ProcessResult,
)
from filterax._processes._schedulers import (
    DataMisfitController as DataMisfitController,
    EKSStableScheduler as EKSStableScheduler,
    FixedScheduler as FixedScheduler,
)
from filterax._protocols import (
    AbstractDynamics as AbstractDynamics,
    AbstractInflator as AbstractInflator,
    AbstractLocalizer as AbstractLocalizer,
    AbstractNoise as AbstractNoise,
    AbstractObsOperator as AbstractObsOperator,
    AbstractProcess as AbstractProcess,
    AbstractScheduler as AbstractScheduler,
    AbstractSequentialFilter as AbstractSequentialFilter,
)
from filterax._smoothers import (
    IES as IES,
    EnKS as EnKS,
    EnsembleRTS as EnsembleRTS,
    EnsembleSqrtSmoother as EnsembleSqrtSmoother,
    FixedLagSmoother as FixedLagSmoother,
)
from filterax._train._differentiable import (
    differentiable_assimilate as differentiable_assimilate,
)
from filterax._types import (
    AnalysisResult as AnalysisResult,
    AssimilationResult as AssimilationResult,
    FilterConfig as FilterConfig,
    FilterState as FilterState,
    LatentAssimilationResult as LatentAssimilationResult,
    ProcessConfig as ProcessConfig,
    ProcessState as ProcessState,
    SmoothingResult as SmoothingResult,
    UKIState as UKIState,
)


__version__ = "0.0.4"

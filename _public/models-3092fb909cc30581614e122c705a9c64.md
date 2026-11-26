---
status: draft
version: 0.1.0
---

# ekalmX — Layer 2: Models

High-level, ready-to-use ensemble methods. Minimal boilerplate — configure, call, get results.

Layer 2 models compose L0 primitives and L1 components into complete workflows: forecast-analysis-inflate loops for filters, init-update-converge loops for processes.

---

## Sequential Filter Models

### `LETKF`

```python
class LETKF(eqx.Module):
    """Local Ensemble Transform Kalman Filter — full assimilation loop.

    Composes: AbstractDynamics (forecast) → L1 LETKF (analysis)
    → AbstractInflator (post-processing), repeated over observation windows.

    Parameters
    ----------
    dynamics : AbstractDynamics
        Forward model. Applied via eqx.filter_vmap over ensemble.
    obs_op : AbstractObsOperator
        Maps state → observation space.
    localizer : AbstractLocalizer
        Covariance localization (required for LETKF).
    inflator : optional AbstractInflator
        Ensemble inflation. Default: no inflation.
    config : FilterConfig
        Static settings (n_ensemble, etc.).
    """
    dynamics: AbstractDynamics
    obs_op: AbstractObsOperator
    localizer: AbstractLocalizer
    inflator: Optional[AbstractInflator] = None
    config: FilterConfig = FilterConfig(n_ensemble=50)

    def assimilate(
        self,
        init_ensemble: Float[Array, "N_e N_x"],
        observations: list[tuple[Float[Array, "N_y"], Scalar]],
        obs_noise: AbstractLinearOperator,
        t0: Scalar = 0.0,
        dt: Scalar = 1.0,
        state_coords: Optional[Float[Array, "N_x D"]] = None,
        obs_coords: Optional[Float[Array, "N_y D"]] = None,
    ) -> AssimilationResult:
        """Run full assimilation over observation sequence.

        Parameters
        ----------
        init_ensemble : initial ensemble (N_e, N_x)
        observations : list of (obs_values, obs_time) pairs
        obs_noise : observation error covariance R
        t0 : initial time
        dt : time step between observation windows
        state_coords : grid point coordinates (for localization)
        obs_coords : observation coordinates (for localization)

        Returns
        -------
        AssimilationResult with final ensemble and per-window history
        """
        ...

    def assimilate_and_smooth(
        self,
        init_ensemble: Float[Array, "N_e N_x"],
        observations: list[tuple[Float[Array, "N_y"], Scalar]],
        obs_noise: AbstractLinearOperator,
        smoother: Optional[EnKS] = None,
        **kwargs,
    ) -> AssimilationResult:
        """Filter forward, then smooth backward.

        Convenience method combining filter + smoother in one call.
        """
        ...
```

### `ETKF` (L2)

```python
class ETKF(eqx.Module):
    """Ensemble Transform Kalman Filter — full assimilation loop.

    Same interface as LETKF but without mandatory localization.
    Suitable for low-dimensional problems or when localization
    is not needed.
    """
    dynamics: AbstractDynamics
    obs_op: AbstractObsOperator
    localizer: Optional[AbstractLocalizer] = None
    inflator: Optional[AbstractInflator] = None
    config: FilterConfig = FilterConfig(n_ensemble=50)

    def assimilate(self, init_ensemble, observations, obs_noise, **kwargs):
        ...
```

### `EnSRF` (L2)

```python
class EnSRF(eqx.Module):
    """Ensemble Square Root Filter — full assimilation loop.

    Wraps L1 EnSRF with forecast-analysis-inflate loop.
    """
    dynamics: AbstractDynamics
    obs_op: AbstractObsOperator
    localizer: Optional[AbstractLocalizer] = None
    inflator: Optional[AbstractInflator] = None
    whitaker: bool = eqx.field(static=True, default=True)
    serial: bool = eqx.field(static=True, default=False)
    config: FilterConfig = FilterConfig(n_ensemble=50)

    def assimilate(self, init_ensemble, observations, obs_noise, **kwargs):
        ...
```

### `AssimilationResult`

```python
class AssimilationResult(eqx.Module):
    """Output of a full assimilation run."""
    particles: Float[Array, "N_e N_x"]           # Final analysis ensemble
    history: list[AnalysisResult]                  # Per-window results
    forecast_history: list[Float[Array, "N_e N_x"]]  # Per-window forecasts
    log_likelihood: Optional[Scalar] = None        # Total log-likelihood
```

---

## Process Models (EKP)

### `EKI` (L2)

```python
class EKI(eqx.Module):
    """Ensemble Kalman Inversion — full process loop.

    Composes: L1 EKI process + scheduler + convergence check.

    Parameters
    ----------
    forward_fn : callable
        Forward model: params → predictions.
        Applied via eqx.filter_vmap over ensemble.
    obs : observation vector
    noise_cov : observation noise covariance
    scheduler : step size strategy (default: DataMisfitController)
    config : ProcessConfig (n_iterations, etc.)
    """
    forward_fn: Callable
    obs: Float[Array, "N_d"]
    noise_cov: AbstractLinearOperator
    scheduler: AbstractScheduler = DataMisfitController()
    config: ProcessConfig = ProcessConfig(scheduler=DataMisfitController(), n_iterations=50)

    def run(
        self,
        init_particles: Float[Array, "J N_p"],
    ) -> ProcessResult:
        """Run EKI to convergence.

        Iterates: evaluate forward model → update ensemble → check convergence.
        Terminates when n_iterations reached or scheduler signals completion.

        Returns
        -------
        ProcessResult with calibrated ensemble, mean, and covariance
        """
        ...

    def run_with_callback(
        self,
        init_particles: Float[Array, "J N_p"],
        callback: Optional[Callable[[ProcessState, int], None]] = None,
    ) -> ProcessResult:
        """Run with per-iteration callback for logging/diagnostics."""
        ...
```

### `EKS` (L2)

```python
class EKS(eqx.Module):
    """Ensemble Kalman Sampler — full process loop.

    Produces approximate posterior samples (ensemble does NOT collapse).
    Runs longer than EKI — typically hundreds of iterations.
    """
    forward_fn: Callable
    obs: Float[Array, "N_d"]
    noise_cov: AbstractLinearOperator
    scheduler: AbstractScheduler = EKSStableScheduler()
    config: ProcessConfig = ProcessConfig(scheduler=EKSStableScheduler(), n_iterations=500)

    def run(self, init_particles) -> ProcessResult:
        ...
```

### `UKI` (L2)

```python
class UKI(eqx.Module):
    """Unscented Kalman Inversion — full process loop.

    Parametric: maintains mean + covariance, not particles.
    Uses 2p+1 sigma points for quadrature.
    """
    forward_fn: Callable
    obs: Float[Array, "N_d"]
    noise_cov: AbstractLinearOperator
    scheduler: AbstractScheduler = DataMisfitController()
    alpha: float = eqx.field(static=True, default=1.0)
    beta: float = eqx.field(static=True, default=2.0)
    kappa: float = eqx.field(static=True, default=0.0)
    config: ProcessConfig = ProcessConfig(scheduler=DataMisfitController(), n_iterations=50)

    def run(
        self,
        init_mean: Float[Array, "N_p"],
        init_cov: AbstractLinearOperator,
    ) -> ProcessResult:
        ...
```

### `ProcessResult`

```python
class ProcessResult(eqx.Module):
    """Output of a full process run."""
    particles: Optional[Float[Array, "J N_p"]]    # Final ensemble (None for UKI)
    mean: Float[Array, "N_p"]                       # Posterior mean
    covariance: AbstractLinearOperator              # Posterior covariance
    history: list[ProcessState]                      # Per-iteration states
    converged: bool                                  # Whether scheduler signaled completion
    n_iterations: int                                # Actual iterations run
```

---

## optax Transforms

EKP processes wrapped as `optax.GradientTransformation` for composability.

### `ekalmx.optax.eki`

```python
def eki(
    forward_fn: Callable,
    obs: Float[Array, "N_d"],
    noise_cov: AbstractLinearOperator,
    scheduler: AbstractScheduler = DataMisfitController(),
    n_ensemble: int = 100,
    seed: int = 0,
) -> optax.GradientTransformation:
    """Wrap EKI as an optax GradientTransformation.

    The caller provides forward evaluations (not gradients).
    The `grads` argument to update() is unused — the ensemble
    provides search directions via cross-covariance.

    Parallel to optax_bayes: both are structured update rules
    fitting the optax init → update → state pattern.

    Compatible with optax.chain, inject_hyperparams, etc.

    Returns
    -------
    optax.GradientTransformation with init() and update() methods
    """
    ...
```

### `ekalmx.optax.eks`

```python
def eks(
    forward_fn: Callable,
    obs: Float[Array, "N_d"],
    noise_cov: AbstractLinearOperator,
    scheduler: AbstractScheduler = EKSStableScheduler(),
    n_ensemble: int = 100,
    seed: int = 0,
) -> optax.GradientTransformation:
    """Wrap EKS as an optax GradientTransformation.

    Produces posterior samples — ensemble does not collapse.
    """
    ...
```

### `ekalmx.optax.uki`

```python
def uki(
    forward_fn: Callable,
    obs: Float[Array, "N_d"],
    noise_cov: AbstractLinearOperator,
    scheduler: AbstractScheduler = DataMisfitController(),
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> optax.GradientTransformation:
    """Wrap UKI as an optax GradientTransformation.

    Parametric: state carries mean + covariance (not particles).
    """
    ...
```

---

## Utilities

### `diagnostics`

```python
def spread(
    particles: Float[Array, "N_e N_x"],
) -> Float[Array, "N_x"]:
    """Per-variable ensemble standard deviation."""

def rank_histogram(
    particles: Float[Array, "T N_e N_x"],
    truth: Float[Array, "T N_x"],
) -> Float[Array, "N_e+1"]:
    """Rank histogram (Talagrand diagram) for ensemble reliability.

    Flat histogram = well-calibrated ensemble.
    U-shaped = underdispersive. Dome = overdispersive.
    """

def innovation_check(
    innovations: Float[Array, "T N_y"],
    innovation_covs: list[AbstractLinearOperator],
) -> dict:
    """Desroziers et al. (2005) innovation consistency diagnostics.

    Returns
    -------
    dict with:
        mean_innovation : should be ≈ 0
        normalized_variance : should be ≈ 1
        chi_squared : chi² test statistic
    """

def effective_ensemble_size(
    particles: Float[Array, "N_e N_x"],
) -> Scalar:
    """Effective ensemble size based on weight degeneracy.

    For equally-weighted ensemble (standard EnKF): ESS = N_e.
    Useful for particle filter diagnostics.
    """
```

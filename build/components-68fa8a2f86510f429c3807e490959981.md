---
status: draft
version: 0.1.0
---

# ekalmX — Layer 1: Components

Protocols, building blocks, and configurable filter/process/smoother steps. All are `eqx.Module` subclasses — pytree-compatible, JIT-friendly, serializable.

---

## Protocols

### `AbstractSequentialFilter`

```python
class AbstractSequentialFilter(eqx.Module):
    """Protocol for sequential ensemble filters.

    One analysis update per assimilation window.
    Forecast is typically handled externally (vmap dynamics over ensemble).
    """

    @abc.abstractmethod
    def analysis(
        self,
        particles: Float[Array, "N_e N_x"],
        obs: Float[Array, "N_y"],
        obs_op: AbstractObsOperator,
        obs_noise: AbstractLinearOperator,
    ) -> AnalysisResult:
        """Assimilate observations into ensemble.

        Parameters
        ----------
        particles : prior (forecast) ensemble
        obs : observation vector
        obs_op : maps state → obs space
        obs_noise : observation error covariance R

        Returns
        -------
        AnalysisResult with updated particles (and optional log_likelihood)
        """
        ...
```

**Implementations:** `StochasticEnKF`, `ETKF`, `ETKF_Livings`, `EnSRF`, `EnSRF_Whitaker`, `EnSRF_Serial`, `ESTKF`, `LETKF`

### `AbstractProcess`

```python
class AbstractProcess(eqx.Module):
    """Protocol for iterative ensemble Kalman processes.

    Iterate until convergence. Caller provides forward evaluations
    at each step (forward model may be expensive / non-JAX).
    """

    @abc.abstractmethod
    def init(
        self,
        particles: Float[Array, "J N_p"],
        obs: Float[Array, "N_d"],
        noise_cov: AbstractLinearOperator,
    ) -> ProcessState:
        """Initialize process state from prior ensemble."""
        ...

    @abc.abstractmethod
    def update(
        self,
        state: ProcessState,
        forward_evals: Float[Array, "J N_d"],
    ) -> ProcessState:
        """Single EKP update step.

        Parameters
        ----------
        state : current process state
        forward_evals : G(θⱼ) for each ensemble member

        Returns
        -------
        Updated ProcessState
        """
        ...
```

**Implementations:** `EKI`, `ETKI`, `EKS_Process`, `UKI`, `UTKI`, `GNKI`, `SparseInversion`

### `AbstractDynamics`

```python
class AbstractDynamics(eqx.Module):
    """Forward model: propagate a single state from t0 to t1.

    Designed for use with eqx.filter_vmap to broadcast over ensemble:
        forecast = eqx.filter_vmap(dynamics)(particles, t0, t1)
    """

    @abc.abstractmethod
    def __call__(
        self,
        state: Float[Array, "N_x"],
        t0: Scalar,
        t1: Scalar,
    ) -> Float[Array, "N_x"]:
        ...
```

**Example implementations (user-supplied):**
- somax model wrapper (ocean dynamics)
- diffrax ODE wrapper (Lorenz, custom ODEs)
- Neural ODE (eqx.nn.MLP as learned RHS)
- Identity (no dynamics — for pure analysis testing)

### `AbstractObsOperator`

```python
class AbstractObsOperator(eqx.Module):
    """Map state space → observation space.

    Designed for use with eqx.filter_vmap:
        obs_particles = eqx.filter_vmap(obs_op)(particles)
    """

    @abc.abstractmethod
    def __call__(
        self,
        state: Float[Array, "N_x"],
    ) -> Float[Array, "N_y"]:
        ...
```

**Example implementations (user-supplied):**
- `LinearObs(H)` — matrix multiplication H @ x
- `SparseObs(indices)` — extract state at observation locations
- `InterpolatingObs(coords)` — interpolate state to observation coordinates
- Neural decoder (ROAD-EnKF pattern)

### `AbstractLocalizer`

```python
class AbstractLocalizer(eqx.Module):
    """Covariance localization strategy.

    Applied to the Kalman gain or covariance matrices
    to suppress spurious long-range correlations.
    """

    @abc.abstractmethod
    def __call__(
        self,
        cov: Float[Array, "N_x N_y"],
        coords: Any,
    ) -> Float[Array, "N_x N_y"]:
        ...
```

**Implementations:**
- `GaspariCohn(radius)` — standard 5th-order taper
- `GaussianLocalizer(radius)` — Gaussian decay
- `CutoffLocalizer(radius)` — hard binary cutoff

### `AbstractInflator`

```python
class AbstractInflator(eqx.Module):
    """Ensemble inflation strategy.

    Applied after analysis to maintain ensemble spread.
    """

    @abc.abstractmethod
    def __call__(
        self,
        particles: Float[Array, "N_e N_x"],
        **kwargs,
    ) -> Float[Array, "N_e N_x"]:
        ...
```

**Implementations:**
- `MultiplicativeInflator(factor)` — constant factor
- `RTPS(alpha)` — relaxation to prior spread
- `RTPP(alpha)` — relaxation to prior perturbations
- `AdaptiveInflator(min_factor, max_factor)` — Anderson (2007)

### `AbstractNoise`

```python
class AbstractNoise(eqx.Module):
    """Noise model backed by gaussx operators.

    Provides both the covariance operator (for Kalman gain)
    and sampling (for stochastic methods).
    """

    @abc.abstractmethod
    def covariance(self) -> AbstractLinearOperator:
        """Return covariance as a gaussx/lineax operator."""
        ...

    @abc.abstractmethod
    def sample(self, key: PRNGKey, shape: tuple) -> Array:
        """Draw noise samples."""
        ...
```

**Implementations:**
- `DiagonalNoise(variance)` — diagonal covariance
- `FullNoise(cov_matrix)` — dense covariance
- `LowRankNoise(basis, diagonal)` — low-rank + diagonal via gaussx

### `AbstractScheduler`

```python
class AbstractScheduler(eqx.Module):
    """Learning rate / step size for EKP processes."""

    @abc.abstractmethod
    def get_dt(self, state: ProcessState) -> Scalar:
        """Return step size for current iteration."""
        ...
```

**Implementations:**
- `FixedScheduler(dt)` — constant step size
- `DataMisfitController(target_misfit)` — adaptive, terminates when algo_time reaches 1.0
- `EKSStableScheduler()` — stability-aware for EKS sampling

---

## Sequential Filters

All implement `AbstractSequentialFilter`. Each provides a single `analysis` step — the L2 model layer handles the forecast-analysis-inflate loop.

### `StochasticEnKF`

```python
class StochasticEnKF(AbstractSequentialFilter):
    """Stochastic Ensemble Kalman Filter (Evensen 1994).

    Perturbed observations: y_pert[j] = y + ε[j], ε ~ N(0, R)
    Update: X_a = X_f + K (y_pert - H X_f)
    Kalman gain: K = C^{xH} (C^{HH} + R)⁻¹

    Stochastic — introduces sampling noise. Simple and robust.
    """
    seed: int = eqx.field(static=True, default=0)
```

### `ETKF`

```python
class ETKF(AbstractSequentialFilter):
    """Ensemble Transform Kalman Filter (Bishop et al. 2001).

    Deterministic — no perturbed observations.
    Transform matrix: W = (I + X'ᵀ Hᵀ R⁻¹ H X')⁻¹
    Update: X_a = X_f W

    Preferred for small-to-medium ensemble sizes.
    """
```

### `EnSRF`

```python
class EnSRF(AbstractSequentialFilter):
    """Ensemble Square Root Filter (Whitaker & Compo 2002).

    Deterministic. Separate mean and perturbation updates:
    - Mean: x̄_a = x̄_f + K (y - H x̄_f)
    - Perturbations: X'_a = X'_f (I - K̃ H)^{1/2}

    Variants controlled by config:
    - whitaker: bool — use Whitaker & Compo formulation
    - serial: bool — process observations one at a time
    """
    whitaker: bool = eqx.field(static=True, default=True)
    serial: bool = eqx.field(static=True, default=False)
```

### `ESTKF`

```python
class ESTKF(AbstractSequentialFilter):
    """Ensemble Square Root Transform Kalman Filter.

    Hybrid of ETKF and EnSRF. Transform via eigendecomposition.
    """
```

### `LETKF`

```python
class LETKF(AbstractSequentialFilter):
    """Local Ensemble Transform Kalman Filter (Hunt et al. 2007).

    Performs independent ETKF analysis at each grid point
    using only local observations (within localization radius).

    Standard for operational NWP. Embarrassingly parallel over grid points.
    """
    localizer: AbstractLocalizer

    def analysis(
        self,
        particles: Float[Array, "N_e N_x"],
        obs: Float[Array, "N_y"],
        obs_op: AbstractObsOperator,
        obs_noise: AbstractLinearOperator,
        state_coords: Optional[Float[Array, "N_x D"]] = None,
        obs_coords: Optional[Float[Array, "N_y D"]] = None,
    ) -> AnalysisResult:
        """Localized analysis.

        Requires state_coords and obs_coords for distance computation.
        Internally vmaps ETKF over grid points.
        """
        ...
```

---

## Smoothers

Backward-pass methods that refine filter estimates using future observations.

### `EnKS`

```python
class EnKS(eqx.Module):
    """Ensemble Kalman Smoother.

    Backward pass over filter results. At each time step t:
    G_t = C^{a,f}_{t,t+1} (C^{f,f}_{t+1})⁻¹
    X^s_t = X^a_t + G_t (X^s_{t+1} - X^f_{t+1})

    where C^{a,f} is the cross-covariance between analysis at t
    and forecast at t+1.
    """

    def smooth(
        self,
        filter_results: list[AnalysisResult],
        forecast_particles: list[Float[Array, "N_e N_x"]],
    ) -> list[AnalysisResult]:
        """Run backward smoother pass.

        Parameters
        ----------
        filter_results : forward filter output at each time step
        forecast_particles : forecast ensemble at each time step

        Returns
        -------
        Smoothed AnalysisResult at each time step
        """
        ...
```

### `EnsembleRTS`

```python
class EnsembleRTS(eqx.Module):
    """Ensemble Rauch-Tung-Striebel Smoother.

    Ensemble analog of the classical RTS smoother.
    Uses ensemble covariance instead of parametric covariance.
    """

    def smooth(
        self,
        filter_results: list[AnalysisResult],
        forecast_particles: list[Float[Array, "N_e N_x"]],
        dynamics: AbstractDynamics,
    ) -> list[AnalysisResult]:
        ...
```

### `FixedLagSmoother`

```python
class FixedLagSmoother(eqx.Module):
    """Fixed-lag ensemble smoother.

    Only looks back a fixed number of time steps (lag).
    Bounded memory — suitable for online applications.
    """
    lag: int = eqx.field(static=True)

    def update(
        self,
        new_filter_result: AnalysisResult,
        new_forecast: Float[Array, "N_e N_x"],
        history: list[AnalysisResult],
    ) -> list[AnalysisResult]:
        """Smooth the last `lag` time steps given new observation."""
        ...
```

---

## Processes (EKP)

All implement `AbstractProcess`. The caller provides forward evaluations — the process provides the ensemble update rule.

### `EKI`

```python
class EKI(AbstractProcess):
    """Ensemble Kalman Inversion (Iglesias, Law & Stuart 2013).

    Standard optimization process:
    θ_{n+1}⁽ʲ⁾ = θ_n⁽ʲ⁾ + Δt · C^{θG}_n (C^{GG}_n + Δt⁻¹ Γ_y)⁻¹ (y - G(θ_n⁽ʲ⁾))

    Converges to a point estimate (ensemble collapses).
    """
    scheduler: AbstractScheduler
    prior_mean: Optional[Float[Array, "N_p"]] = None
    prior_cov: Optional[AbstractLinearOperator] = None
```

### `ETKI`

```python
class ETKI(AbstractProcess):
    """Ensemble Transform Kalman Inversion.

    Output-scalable variant of EKI: O(d) instead of O(d³)
    where d is the observation dimension.
    """
    scheduler: AbstractScheduler
```

### `EKS_Process`

```python
class EKS_Process(AbstractProcess):
    """Ensemble Kalman Sampler / ALDI (Garbuno-Inigo et al. 2020).

    Produces posterior samples (ensemble does NOT collapse).
    Adds Brownian motion term for ergodicity:
    dθ⁽ʲ⁾ = C^{θG} Γ_y⁻¹ (y - G(θ⁽ʲ⁾)) dt + (diffusion terms) dW

    Use EKSStableScheduler for numerical stability.
    """
    scheduler: AbstractScheduler
```

### `UKI`

```python
class UKI(AbstractProcess):
    """Unscented Kalman Inversion (Huang, Schneider & Stuart 2022).

    Parametric: maintains explicit mean and covariance (not particles).
    Uses sigma points for quadrature — no ensemble collapse.
    State is UKIState(mean, covariance) instead of ProcessState.

    Deterministic: 2p+1 sigma points for p parameters.
    """
    scheduler: AbstractScheduler
    alpha: float = eqx.field(static=True, default=1.0)
    beta: float = eqx.field(static=True, default=2.0)
    kappa: float = eqx.field(static=True, default=0.0)

    def init(
        self,
        mean: Float[Array, "N_p"],
        covariance: AbstractLinearOperator,
        obs: Float[Array, "N_d"],
        noise_cov: AbstractLinearOperator,
    ) -> UKIState:
        ...

    def update(
        self,
        state: UKIState,
        forward_evals: Float[Array, "2p+1 N_d"],
    ) -> UKIState:
        ...
```

### `GNKI`

```python
class GNKI(AbstractProcess):
    """Gauss-Newton Kalman Inversion.

    Estimates the Jacobian explicitly from ensemble perturbations.
    Uses Gauss-Newton-style update for faster convergence
    on well-conditioned problems.
    """
    scheduler: AbstractScheduler
```

### `SparseInversion`

```python
class SparseInversion(AbstractProcess):
    """Sparse Ensemble Kalman Inversion.

    Adds L0/L1 sparsity penalties to the EKI update.
    For parameter estimation where most parameters are expected to be zero
    or near-zero (variable selection).
    """
    scheduler: AbstractScheduler
    penalty: str = eqx.field(static=True, default="l1")  # "l0" | "l1"
    penalty_weight: float = eqx.field(static=True, default=0.1)
```

---

## Schedulers

Step size strategies for EKP processes.

### `FixedScheduler`

```python
class FixedScheduler(AbstractScheduler):
    """Constant step size."""
    dt: float = eqx.field(static=True)

    def get_dt(self, state):
        return self.dt
```

### `DataMisfitController`

```python
class DataMisfitController(AbstractScheduler):
    """Adaptive step size based on data misfit.

    Chooses Δt so that the ensemble makes steady progress
    toward fitting the data. Terminates when algo_time reaches 1.0.

    Based on Iglesias (2021).
    """
    target_misfit: float = eqx.field(static=True, default=1.0)

    def get_dt(self, state):
        ...
```

### `EKSStableScheduler`

```python
class EKSStableScheduler(AbstractScheduler):
    """Stability-aware scheduler for EKS.

    Constrains Δt to maintain ensemble spread and prevent
    divergence in the Langevin dynamics.
    """
    max_dt: float = eqx.field(static=True, default=1.0)

    def get_dt(self, state):
        ...
```

---

## Data Types

### `FilterState`

```python
class FilterState(eqx.Module):
    """Running state for sequential filters."""
    particles: Float[Array, "N_e N_x"]
    step: int = eqx.field(static=True)
```

### `ProcessState`

```python
class ProcessState(eqx.Module):
    """Running state for EKP processes."""
    particles: Float[Array, "J N_p"]
    forward_evals: Float[Array, "J N_d"]
    obs: Float[Array, "N_d"]
    noise_cov: AbstractLinearOperator
    step: int = eqx.field(static=True)
    algo_time: float
```

### `UKIState`

```python
class UKIState(eqx.Module):
    """Parametric state for Unscented Kalman Inversion."""
    mean: Float[Array, "N_p"]
    covariance: AbstractLinearOperator
    step: int = eqx.field(static=True)
```

### `AnalysisResult`

```python
class AnalysisResult(eqx.Module):
    """Output of any analysis/update step."""
    particles: Float[Array, "N_e N_x"]
    log_likelihood: Optional[Scalar] = None
    diagnostics: Optional[dict] = None
```

### `FilterConfig`

```python
class FilterConfig(eqx.Module):
    """Static configuration for sequential filters."""
    localizer: Optional[AbstractLocalizer] = None
    inflator: Optional[AbstractInflator] = None
    n_ensemble: int = eqx.field(static=True)
```

### `ProcessConfig`

```python
class ProcessConfig(eqx.Module):
    """Static configuration for EKP processes."""
    scheduler: AbstractScheduler
    n_iterations: int = eqx.field(static=True)
```

### `PatchMetadata`

```python
class PatchMetadata(eqx.Module):
    """Metadata for patch reconstruction."""
    origins: Int[Array, "N_patches D"]
    patch_size: tuple[int, ...] = eqx.field(static=True)
    stride: tuple[int, ...] = eqx.field(static=True)
    global_shape: tuple[int, ...] = eqx.field(static=True)
```

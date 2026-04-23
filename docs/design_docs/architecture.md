---
status: draft
version: 0.1.0
---

# ekalmX — Architecture

## Overview

ekalmX follows a three-layer progressive disclosure architecture. Each layer is self-contained and useful alone. Lower layers are pure and composable; higher layers provide convenience.

---

## Three-Layer Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 2 — Models                                                       │
│  Ready-to-use ensemble methods. Minimal config, maximal convenience.    │
│  Sequential: LETKF, ETKF, EnSRF, ESTKF, StochasticEnKF               │
│  Processes:  EKI, EKS, UKI, ETKI, GNKI                                │
│  optax:      EKI/EKS/UKI as optax.GradientTransformation              │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 1 — Components                                                   │
│  Protocols, building blocks, configurable filter/process steps.         │
│  AbstractSequentialFilter, AbstractProcess                              │
│  AbstractDynamics, AbstractObsOperator, AbstractLocalizer               │
│  AbstractInflator, AbstractNoise, AbstractScheduler                     │
│  FilterState, ProcessState, AnalysisResult                              │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 0 — Primitives                                                   │
│  Pure functions. Stateless, differentiable, composable.                 │
│  ensemble_mean, ensemble_anomalies, ensemble_covariance                 │
│  kalman_gain, cross_covariance, log_likelihood                          │
│  gaspari_cohn, localize, inflate, perturbed_observations                │
│  create_patches, assign_obs, blend_patches                              │
└─────────────────────────────────────────────────────────────────────────┘

Not part of the library API:
┌─────────────────────────────────────────────────────────────────────────┐
│  Zoo — Reference implementations (educational, not maintained to core   │
│  standard). Continuous-time filters, dynamical systems for testing.     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Protocols

ekalmX defines two protocol families for its two modes of operation, plus shared extension points for user-supplied components.

### Filter protocols (sequential state estimation)

```python
class AbstractSequentialFilter(eqx.Module):
    """One analysis update per assimilation window."""

    @abc.abstractmethod
    def forecast(
        self,
        particles: Float[Array, "N_e N_x"],
        dynamics: AbstractDynamics,
        t0: Scalar,
        t1: Scalar,
    ) -> Float[Array, "N_e N_x"]:
        """Propagate ensemble forward in time."""
        ...

    @abc.abstractmethod
    def analysis(
        self,
        particles: Float[Array, "N_e N_x"],
        obs: Float[Array, "N_y"],
        obs_op: AbstractObsOperator,
        obs_noise: AbstractLinearOperator,
    ) -> AnalysisResult:
        """Assimilate observations into ensemble."""
        ...
```

### Process protocols (iterative parameter estimation)

```python
class AbstractProcess(eqx.Module):
    """Iterate until convergence (EKI, EKS, UKI, etc.)."""

    @abc.abstractmethod
    def init(
        self,
        particles: Float[Array, "J N_p"],
        obs: Float[Array, "N_d"],
        noise_cov: AbstractLinearOperator,
    ) -> ProcessState:
        ...

    @abc.abstractmethod
    def update(
        self,
        state: ProcessState,
        forward_evals: Float[Array, "J N_d"],
    ) -> ProcessState:
        """Single EKP update step. Caller provides forward evaluations."""
        ...
```

### optax interface (processes as gradient transformations)

```python
def eki_as_optax(
    forward_fn: Callable,
    obs: Float[Array, "N_d"],
    noise_cov: AbstractLinearOperator,
    scheduler: AbstractScheduler,
    **kwargs,
) -> optax.GradientTransformation:
    """Wrap EKI as an optax GradientTransformation.

    The caller provides forward evaluations (not gradients).
    Compatible with optax.chain, schedules, clipping, logging.

    Parallel to optax_bayes: both are structured update rules
    that fit the optax init → update → state pattern.
    """
    ...
```

### Shared extension points

```python
class AbstractDynamics(eqx.Module):
    """Forward model: propagate state from t0 to t1."""
    @abc.abstractmethod
    def __call__(
        self, state: Float[Array, "N_x"], t0: Scalar, t1: Scalar
    ) -> Float[Array, "N_x"]:
        ...

class AbstractObsOperator(eqx.Module):
    """Map state space → observation space."""
    @abc.abstractmethod
    def __call__(self, state: Float[Array, "N_x"]) -> Float[Array, "N_y"]:
        ...

class AbstractLocalizer(eqx.Module):
    """Covariance localization via tapering."""
    @abc.abstractmethod
    def __call__(
        self, cov: Float[Array, "N_x N_y"], coords: Any
    ) -> Float[Array, "N_x N_y"]:
        ...

class AbstractInflator(eqx.Module):
    """Ensemble inflation to maintain spread."""
    @abc.abstractmethod
    def __call__(
        self, particles: Float[Array, "N_e N_x"], **kwargs
    ) -> Float[Array, "N_e N_x"]:
        ...

class AbstractNoise(eqx.Module):
    """Noise model backed by gaussx operators."""
    @abc.abstractmethod
    def covariance(self) -> AbstractLinearOperator:
        """Return covariance as a gaussx/lineax operator."""
        ...

    @abc.abstractmethod
    def sample(self, key: PRNGKey, shape: tuple) -> Array:
        ...

class AbstractScheduler(eqx.Module):
    """Learning rate / step size for EKP processes."""
    @abc.abstractmethod
    def get_dt(self, state: ProcessState) -> Scalar:
        ...
```

All protocols are `eqx.Module` subclasses — pytree-compatible, JIT-friendly, serializable.

---

## Key Data Types

### Core state containers

```python
class FilterState(eqx.Module):
    """Running state for sequential filters."""
    particles: Float[Array, "N_e N_x"]
    step: int = eqx.field(static=True)

class ProcessState(eqx.Module):
    """Running state for EKP processes."""
    particles: Float[Array, "J N_p"]
    forward_evals: Float[Array, "J N_d"]
    obs: Float[Array, "N_d"]
    noise_cov: AbstractLinearOperator
    step: int = eqx.field(static=True)
    algo_time: float

class UKIState(eqx.Module):
    """Parametric state for Unscented Kalman Inversion."""
    mean: Float[Array, "N_p"]
    covariance: AbstractLinearOperator
    step: int = eqx.field(static=True)
```

### Analysis output

```python
class AnalysisResult(eqx.Module):
    """Output of any analysis/update step."""
    particles: Float[Array, "N_e N_x"]
    log_likelihood: Optional[Scalar] = None
    diagnostics: Optional[dict] = None
```

### Configuration

```python
class FilterConfig(eqx.Module):
    """Static configuration for sequential filters."""
    localizer: Optional[AbstractLocalizer] = None
    inflator: Optional[AbstractInflator] = None
    n_ensemble: int = eqx.field(static=True)

class ProcessConfig(eqx.Module):
    """Static configuration for EKP processes."""
    scheduler: AbstractScheduler
    n_iterations: int = eqx.field(static=True)
```

Design notes:
- Store `particles` only — compute mean, anomalies, covariance on demand via L0 primitives.
- Ensemble dimension is the leading axis (`N_e, N_x`) for natural `eqx.filter_vmap` over members.
- `log_likelihood` is optional — only computed when needed for differentiable training.
- Configuration is `eqx.Module` — serializable, pytree-compatible, JIT-friendly.

---

## Package Layout

```
ekalmx/
├── __init__.py                    # Public API re-exports
├── _src/
│   ├── _protocols.py              # AbstractSequentialFilter, AbstractProcess,
│   │                              # AbstractDynamics, AbstractObsOperator,
│   │                              # AbstractLocalizer, AbstractInflator,
│   │                              # AbstractNoise, AbstractScheduler
│   ├── _types.py                  # FilterState, ProcessState, UKIState,
│   │                              # AnalysisResult, FilterConfig, ProcessConfig
│   │
│   ├── # --- Layer 0: Primitives ---
│   ├── statistics.py              # ensemble_mean, ensemble_anomalies,
│   │                              # ensemble_covariance, cross_covariance
│   ├── gain.py                    # kalman_gain (gaussx-backed)
│   ├── likelihood.py              # log_likelihood, innovation_statistics
│   ├── localization.py            # gaspari_cohn, gaussian_taper, hard_cutoff
│   ├── inflation.py               # multiplicative, rtps, rtpp, adaptive
│   ├── perturbations.py           # perturbed_observations
│   ├── patches.py                 # create_patches, assign_obs, blend_patches
│   │
│   ├── # --- Layer 1: Components ---
│   ├── filters/                   # Sequential filter analysis steps
│   │   ├── stochastic_enkf.py     # StochasticEnKF
│   │   ├── etkf.py                # ETKF, ETKF_Livings
│   │   ├── ensrf.py               # EnSRF (Whitaker, serial variants via config)
│   │   ├── estkf.py               # ESTKF
│   │   └── letkf.py               # LETKF
│   ├── processes/                 # Iterative EKP update steps
│   │   ├── eki.py                 # EKI (Ensemble Kalman Inversion)
│   │   ├── eks.py                 # EKS / ALDI (Ensemble Kalman Sampler)
│   │   ├── uki.py                 # UKI (Unscented Kalman Inversion)
│   │   ├── etki.py                # ETKI (Ensemble Transform KI)
│   │   └── gnki.py                # GNKI (Gauss-Newton KI)
│   │
│   ├── # --- Layer 2: Models ---
│   ├── models/
│   │   ├── sequential.py          # Full filter loop: forecast → analysis → inflate
│   │   └── inversion.py           # Full process loop: init → update → convergence
│   ├── optax/
│   │   └── transforms.py          # EKI/EKS/UKI as optax.GradientTransformation
│   │
│   └── utils/
│       └── diagnostics.py         # spread, rank_histogram, innovation_check,
│                                  # desroziers_consistency
│
├── zoo/                           # Reference implementations (not core API)
│   ├── continuous/                # Kalman-Bucy, EKF-Bucy, UKF-Bucy, etc.
│   └── dynamical_systems/         # L63, L96, KS (for testing/demos)
│
├── docs/
├── notebooks/
└── tests/
```

---

## Dependency Stack

### Required

| Package | Version | Role |
|---------|---------|------|
| `jax` | >= 0.5 | Core array framework |
| `jaxlib` | >= 0.5 | Backend |
| `equinox` | >= 0.11 | Module system, pytrees |
| `gaussx` | >= 0.1 | Structured covariance operators, ensemble_covariance, kalman_gain |
| `optax` | >= 0.2 | EKP as GradientTransformation, learning rate schedules |
| `jaxtyping` | >= 0.2.28 | Shape annotations |
| `beartype` | >= 0.18 | Runtime type checking |
| `einops` | >= 0.8 | Readable tensor operations |

### Optional

| Package | Version | Role | Used by |
|---------|---------|------|---------|
| `diffrax` | >= 0.5 | ODE/SDE integration | Zoo continuous-time filters, ODE-based dynamics |
| `optimistix` | >= 0.1 | Solver backends | EKI fixed-point, advanced inner minimization |
| `lineax` | >= 0.1 | Linear solvers | Transitive via gaussx, advanced covariance ops |

### Zoo / examples only (not library dependencies)

| Package | Role |
|---------|------|
| `matplotlib` | Visualization |
| `xarray` | Structured output |
| `geo_toolz` | Pre/post-processing |
| `somax` | Ocean model examples |

### Dependency diagram

```
                ┌───────────────┐
                │ jax, equinox  │
                │ einops        │
                └───────┬───────┘
                        │ required
                ┌───────▼───────┐
                │  gaussx       │ structured covariance ops
                └───────┬───────┘
                        │ required
                ┌───────▼───────┐
                │   ekalmX      │
                └──┬────┬────┬──┘
                   │    │    │
         ┌─────────┘    │    └─────────┐
         │ required     │ optional     │ optional
   ┌─────▼─────┐  ┌────▼─────┐  ┌─────▼──────┐
   │  optax    │  │ diffrax  │  │ optimistix │
   │ (EKP as   │  │ (ODE     │  │ (solver    │
   │  optax    │  │  dynamics)│  │  backends) │
   │  transform│  │          │  │            │
   └───────────┘  └──────────┘  └────────────┘

   zoo/examples only (not library deps):
   matplotlib, xarray, geo_toolz, somax
```

---

## gaussx Integration Points

ekalmX delegates structured covariance operations to gaussx rather than reimplementing them.

| ekalmX operation | gaussx backend | Benefit |
|-----------------|---------------|---------|
| Ensemble covariance | `gaussx.recipes.ensemble_covariance` → `LowRankUpdate` | Rank-aware, never N×N dense |
| Cross-covariance | `gaussx.recipes.ensemble_cross_covariance` | Structured matrix |
| Kalman gain K | `gaussx.recipes.kalman_gain` | Woodbury solve when J << N |
| Obs noise R | `lineax.DiagonalLinearOperator` | Trivial inversion |
| Localization taper | Hadamard product on operator | Structure-preserving |
| Log-likelihood | `gaussx.ops.logdet` on innovation covariance | Structure-exploiting logdet |
| Process noise Q | `gaussx.operators.LowRankUpdate` or `Diagonal` | Flexible structure |

---

## optax Integration

EKP processes wrap as `optax.GradientTransformation` for composability with the optax ecosystem:

```python
import ekalmx
import optax

# EKI as optax transform — parallel to optax_bayes
eki_transform = ekalmx.optax.eki(
    forward_fn=simulator,
    obs=observations,
    noise_cov=R,
    scheduler=ekalmx.DataMisfitController(),
    n_ensemble=100,
)

# Compose with optax utilities
optimizer = optax.chain(
    eki_transform,
    optax.clip_by_global_norm(1.0),
)

# Standard optax loop
state = optimizer.init(params)
for step in range(n_steps):
    updates, state = optimizer.update(grads=None, state=state, params=params)
    params = optax.apply_updates(params, updates)
```

The caller provides forward evaluations (not gradients). The `grads` argument is unused — the ensemble provides the search directions. This mirrors optax_bayes where both are structured update rules fitting the `init → update → state` pattern.

---

## CI / Quality Gates

| Check | Command | Scope |
|-------|---------|-------|
| Tests | `uv run pytest tests -x` | Full suite |
| Lint | `uv run ruff check .` | Entire repo |
| Format | `uv run ruff format --check .` | Entire repo |
| Typecheck | `uv run ty check ekalmx` | Package only |

All four must pass before merge. GitHub Actions on push/PR.
Conventional commits required (`feat:`, `fix:`, `docs:`, `test:`, etc.).

**Build system:** hatchling (PEP 621)
**Python:** >= 3.12, < 3.14
**License:** MIT

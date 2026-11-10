---
status: draft
version: 0.1.0
---

# filterX — Ensemble Kalman Filter Research Catalogue

> Inventory of existing EnSKF implementations in `kalman_filter/` and their mapping to filterX design goals.

---

## 1. Existing Codebase Summary

The `kalman_filter/` directory contains ~5,000 lines of working JAX/NumPy code across three areas: a continuous-time filter library, a discrete-time EnKF algorithm zoo, and patch-based spatial decomposition implementations. This is the raw material that filterX will formalize.

### Directory map

```
kalman_filter/
├── kf_continuous.py                       # 392 lines — continuous-time filters (Equinox)
└── ensemble_kalman_filter/
    ├── enskf_zoo.py                       # 2,263 lines — 9 discrete EnKF variants (JAX)
    ├── enskf_patch_jax.py                 # 785 lines — patch-based EnKF (JAX)
    ├── enskf_patch_dask.py                # 594 lines — patch-based EnKF (NumPy/Dask)
    ├── enskf_patch_guide.md               # 380 lines — deployment guide
    ├── enskf_patch_jax.md                 # 483 lines — JAX concepts tutorial
    ├── enskf_patch_dask.md                # 379 lines — patch blending guide
    └── param_estimation/
        ├── filters.py                     # 181 lines — KF/EnSRF/particle filter
        ├── models.py                      # 214 lines — L63/L96/KS/pyQG test models
        ├── loss.py                        # 176 lines — KL, log-likelihood, var cost
        └── torch_enskf.py                 # ~500 lines — Woodbury EnKF, power iteration
```

---

## 2. Algorithm Inventory

### 2.1 Discrete-time ensemble filters (`enskf_zoo.py`)

All implementations follow Vetra-Carvalho et al. (2018). Pure JAX with einops and jaxtyping.

| Algorithm | Function | Category | Localization | Serial | Notes |
|-----------|----------|----------|:---:|:---:|-------|
| **Stochastic EnKF** | `analysis_step_senkf` | Stochastic | -- | -- | Perturbed observations, explicit Kalman gain |
| **Stochastic EnKF (localized)** | `analysis_step_senkf_localized` | Stochastic | Yes | -- | Hadamard product localization on P |
| **ETKF** | `analysis_step_etkf` | Deterministic | -- | -- | Ensemble transform, no perturbation needed |
| **ETKF (Livings)** | `analysis_step_etkf_livings` | Deterministic | -- | -- | Livings et al. (2005) variant |
| **EnSRF** | `analysis_step_ensrf` | Deterministic | -- | -- | Square-root filter, mean + perturbation split |
| **EnSRF (Whitaker)** | `analysis_step_ensrf_whitaker` | Deterministic | -- | -- | Whitaker & Compo (2002) formulation |
| **EnSRF (Whitaker, localized)** | `analysis_step_ensrf_whitaker_localized` | Deterministic | Yes | -- | Localized Whitaker variant |
| **EnSRF (serial)** | `analysis_step_ensrf_serial` | Deterministic | -- | Yes | One observation at a time |
| **ESTKF** | `analysis_step_estkf` | Deterministic | -- | -- | Hybrid square-root transform |

**Shared infrastructure:**

| Component | Function | Purpose |
|-----------|----------|---------|
| State statistics | `filter_step_full`, `filter_step`, `filter_step_with_sqrt`, `filter_step_with_cov`, `filter_step_with_sqrt_inv` | 5 variants of ensemble → (mean, anomalies, covariance) |
| Perturbation | `perturbation_step_stochastic` | Perturbed observations for stochastic methods |
| Localization | `LocalizationMatrices` (NamedTuple) | State-obs (`PH_loc`) and obs-obs (`HPH_loc`) taper matrices |

**Data structures:**

```python
class EnsembleState(NamedTuple):
    Xf: Float[Array, "N_x N_e"]     # Prior ensemble (state × members)
    HXf: Float[Array, "N_y N_e"]    # Obs-space projection
    Y: Float[Array, "N_y"]          # Observations
    R: Float[Array, "N_y"]          # Obs error variance (diagonal)

class AnalysisOutput(NamedTuple):
    Xa: Float[Array, "N_x N_e"]     # Posterior ensemble
    mX: Float[Array, "N_x"]         # State mean
    mY: Float[Array, "N_y"]         # Obs mean
    W: Float[Array, "N_e N_e"]      # Transform matrix (if applicable)
```

### 2.2 Continuous-time filters (`kf_continuous.py`)

All filters are `eqx.Module` subclasses, integrated via Diffrax (Dopri5). Uses einops throughout.

| Filter | Dynamics | Covariance | Key property |
|--------|----------|------------|-------------|
| **LinearKalmanBucy** | Linear (F matrix) | Full | Optimal for LTI systems |
| **ExtendedKalmanBucy** | Nonlinear (f + Jacobian) | Full | 1st-order linearization |
| **UnscentedKalmanBucy** | Nonlinear (sigma points) | Full | 3rd-order, no Jacobians |
| **CubatureKalmanBucy** | Nonlinear (cubature pts) | Full | Parameter-free sigma variant |
| **EnsembleKalmanBucy** | Nonlinear (ensemble) | Sample | Gaspari-Cohn localization |
| **DiagonalKalmanBucy** | Linear | Diagonal | O(n) complexity |
| **LowRankKalmanBucy** | Linear | P = SS^T, rank r | O(nr) memory |
| **HInfinityFilter** | Linear | Full + gamma | Worst-case robust |
| **ParticleFilter** | Nonlinear (ensemble) | Sample | Sampling-based |

**Utilities:**
- `generate_sigma_points()` — unscented transform (alpha, beta, kappa)
- `generate_cubature_points()` — 2n cubature points via Cholesky
- `gaspari_cohn_localization()` — scalar taper function
- `solve_continuous_kalman_filter()` — Diffrax ODE wrapper

### 2.3 Patch-based spatial decomposition

Two parallel implementations (JAX + NumPy/Dask) for large-domain EnKF.

| Component | JAX (`enskf_patch_jax.py`) | NumPy/Dask (`enskf_patch_dask.py`) |
|-----------|---|---|
| Patch creation | `create_overlapping_patches()` | Same API, numpy arrays |
| Obs assignment | `assign_observations_to_patches()` | Same API, with buffer zone |
| Analysis | `enkf_analysis_single_patch()` | Same, with Tikhonov fallback |
| Blending | `blend_overlapping_patches_jax()` | `blend_overlapping_patches()` |
| Taper | `gaspari_cohn_1d_jax()` | `gaspari_cohn_2d_broadcast()` |
| Orchestration | -- | `global_enkf_with_dask_patches()` (xarray I/O) |
| Parallelism | vmap over patches | dask.delayed |

**Metadata types:** `PatchMetadata`, `GlobalMetadata` (NamedTuples for grid/coordinate tracking).

### 2.4 Parameter estimation (`param_estimation/`)

| Component | File | Description |
|-----------|------|-------------|
| Linear KF step | `filters.py` | `filter_step_linear`, `apply_filtering_fixed_linear` |
| Nonlinear EKF step | `filters.py` | `filter_step_nonlinear`, `apply_filtering_fixed_nonlinear` |
| EnSRF step | `filters.py` | `ensrf_step`, `ensrf_steps` (with Ledoit-Wolf shrinkage) |
| Particle filter | `filters.py` | `particle_filter`, `resample_particles`, `update_weights` |
| Woodbury EnKF | `torch_enskf.py` | `woodbury_inversion_logdet` (O(dr^2) when N_e < N_y) |
| Power iteration | `torch_enskf.py` | Largest singular value estimation |
| Gaspari-Cohn | `torch_enskf.py` | Full taper matrix via nested vmap |
| KL divergence | `loss.py` | `KL_gaussian`, `KL_sum` |
| Log-likelihood | `loss.py` | `log_likelihood` (handles missing obs) |
| Variational cost | `loss.py` | `var_cost` (KL + expected log-likelihood) |
| Matrix sqrt | `filters.py` | `sqrtm` via eigendecomposition |
| Covariance shrinkage | `filters.py` | `ledoit_wolf` regularization |

### 2.5 Test models (`param_estimation/models.py`)

| Model | State dim | Type | Integration |
|-------|-----------|------|-------------|
| Lorenz 63 | 3 | Chaotic ODE | RK4 |
| Lorenz 96 | N (configurable) | Multi-scale ODE | RK4 |
| Kuramoto-Sivashinsky | N (spectral) | Spatiotemporal PDE | Spectral ETD |
| pyQG (2-layer QG) | 2 × nx × ny | Ocean model | Adams-Bashforth 3 |

---

## 3. Design Patterns Already Established

### 3.1 JAX idioms (consistent across codebase)

| Pattern | Usage | filterX implication |
|---------|-------|---------------------|
| `eqx.Module` | Continuous-time filters | Adopt for all filter types |
| `NamedTuple` | Discrete-time state/output | Migrate to `eqx.Module` for consistency |
| `jaxtyping` annotations | All JAX files | Keep — first-class shape documentation |
| `einops.einsum` | All matrix ops | Keep — prevents transpose bugs |
| `jax.lax.scan` | Time loops | Keep — scan-compatible filter steps |
| `jax.random.PRNGKey` | Stochastic methods | Keep — deterministic randomness |
| `@jit` / `@partial(jit, ...)` | All pure functions | Migrate to `eqx.filter_jit` |

### 3.2 Mathematical structure

The codebase consistently separates:

1. **Forecast step** — propagate ensemble forward (user-supplied dynamics)
2. **Statistics step** — compute mean, anomalies, covariances from ensemble
3. **Analysis step** — apply Kalman update (algorithm-specific)
4. **Post-processing** — inflation, localization, diagnostics

This four-phase structure maps cleanly to a protocol-based design.

### 3.3 Covariance handling strategies

| Strategy | Where used | Complexity |
|----------|-----------|-----------|
| Full explicit covariance | `filter_step_with_cov` | O(N_x^2) |
| Square-root factors | `filter_step_with_sqrt` | O(N_x × N_e) |
| Inverse square-root | `filter_step_with_sqrt_inv` | O(N_y × N_e) |
| Woodbury identity | `woodbury_inversion_logdet` | O(d × r^2) |
| Diagonal | `DiagonalKalmanBucy` | O(N_x) |
| Low-rank P = SS^T | `LowRankKalmanBucy` | O(N_x × r) |
| Ledoit-Wolf shrinkage | `ensrf_step` | Regularization |
| Localization (Gaspari-Cohn) | Multiple | Tapering |

This variety directly motivates **gaussx integration** — structured operators (diagonal, low-rank, Kronecker) can unify these representations behind a single `lineax.AbstractLinearOperator` interface.

---

## 4. Mapping to filterX Architecture

### 4.1 What maps to filterX core

| Existing code | filterX target | Notes |
|---------------|---------------|-------|
| `EnsembleState` | `ekalmx.EnsembleState` (eqx.Module) | Add ensemble_size as static field |
| `AnalysisOutput` | `ekalmx.AnalysisResult` (eqx.Module) | Add diagnostics (innovation, spread) |
| `filter_step_*` variants | `ekalmx.ensemble_statistics()` | Single function, covariance mode as config |
| `analysis_step_senkf` | `ekalmx.StochasticEnKF.analysis()` | Filter protocol |
| `analysis_step_etkf` | `ekalmx.ETKF.analysis()` | Filter protocol |
| `analysis_step_ensrf*` | `ekalmx.EnSRF.analysis()` | Variants via config, not separate classes |
| `analysis_step_estkf` | `ekalmx.ESTKF.analysis()` | Filter protocol |
| Localization matrices | `ekalmx.Localizer` protocol | Gaspari-Cohn as default impl |
| Patch decomposition | `ekalmx.LocalEnKF` or user-level | Spatial decomposition as higher-level recipe |
| `var_cost`, `KL_gaussian` | `ekalmx.diagnostics` | Filter evaluation metrics |

### 4.2 What maps to filterX protocols

Based on the existing code, the following abstract protocols emerge:

```python
class AbstractFilter(eqx.Module):
    """Base protocol: analysis step."""
    @abc.abstractmethod
    def analysis(self, forecast, obs, obs_op, obs_noise) -> AnalysisResult: ...

class AbstractLocalizer(eqx.Module):
    """Covariance localization."""
    @abc.abstractmethod
    def taper(self, P: Array, coords: Array) -> Array: ...

class AbstractInflator(eqx.Module):
    """Ensemble inflation."""
    @abc.abstractmethod
    def inflate(self, ensemble: Array, factor: float) -> Array: ...

class AbstractObsOperator(eqx.Module):
    """H: state space → observation space."""
    @abc.abstractmethod
    def __call__(self, x: Array) -> Array: ...
```

### 4.3 What maps to EKP (ens_kalman_process.md)

The existing `param_estimation/` code is the bridge between filterX (sequential DA) and the EKP design (inverse problems). Key overlaps:

| Concept | filterX (sequential DA) | EKP (inverse problems) |
|---------|------------------------|----------------------|
| Ensemble | State particles over time | Parameter particles, no time |
| Kalman gain | K = C^{xH} (C^{HH} + R)^{-1} | Same formula |
| Update | Sequential (per assimilation window) | Iterative (until convergence) |
| Forward model | Dynamics M(x) | Simulator G(theta) |
| Observation | H(x) at obs locations | G(theta) output |
| Covariance handling | Localization, inflation | Inflation, regularization |

**Shared infrastructure candidates:**
- Ensemble statistics (mean, anomalies, cross-covariance)
- Woodbury / structured solves (gaussx)
- Gaspari-Cohn and other taper functions
- Kalman gain computation

### 4.4 What stays outside filterX

| Code | Disposition |
|------|------------|
| `models.py` (L63, L96, KS, pyQG) | Example/test utilities, not library |
| `loss.py` (KL, log-likelihood) | Diagnostics module or examples |
| Patch decomposition | Higher-level recipe or separate module |
| Dask implementation | Deployment concern, not core library |
| Continuous-time filters | Separate module or later phase (Diffrax integration) |

---

## 5. Gap Analysis

### 5.1 Algorithms present but needing formalization

| Algorithm | Status | Work needed |
|-----------|--------|-------------|
| Stochastic EnKF | Working | Protocol conformance, eqx.Module migration |
| ETKF | Working | Same |
| ETKF (Livings) | Working | Evaluate if distinct enough to keep |
| EnSRF (3 variants) | Working | Unify into single EnSRF with config |
| ESTKF | Working | Protocol conformance |
| Woodbury solve | Working | Migrate to gaussx backend |
| Gaspari-Cohn localization | Working (3 implementations) | Single canonical implementation |

### 5.2 Algorithms in EKP design but not yet implemented

| Algorithm | EKP doc section | Notes |
|-----------|----------------|-------|
| EKI (Ensemble Kalman Inversion) | Process Zoo — optimization | Core EKP algorithm |
| ETKI (Ensemble Transform KI) | Process Zoo — optimization | Output-scalable O(d) |
| EKS / ALDI (Ensemble Kalman Sampler) | Process Zoo — sampling | Langevin diffusion for posterior |
| UKI (Unscented Kalman Inversion) | Process Zoo — sampling | Parametric posterior (mean + cov) |
| UTKI (Transform Unscented) | Process Zoo — sampling | Output-scalable UKI |
| GNKI (Gauss-Newton KI) | Process Zoo — optimization | Explicit Jacobian estimation |
| SparseInversion | Process Zoo — optimization | L0/L1 sparsity |

### 5.3 Infrastructure gaps

| Gap | Existing workaround | filterX solution |
|-----|---------------------|-----------------|
| No unified state type | 2 NamedTuples + ad-hoc | `eqx.Module` hierarchy |
| No protocol system | Bare functions | Abstract base classes |
| Localization scattered | 3 separate implementations | `AbstractLocalizer` protocol |
| No inflation protocol | Inline multiplication | `AbstractInflator` protocol |
| No adaptive scheduling | Fixed parameters | `Scheduler` protocol (from EKP design) |
| No diagnostics API | Ad-hoc loss functions | `filterx.diagnostics` module |
| 5 filter_step variants | User picks manually | Single function + config enum |
| No gaussx integration | Explicit matrix ops | `lineax.AbstractLinearOperator` for covariances |

### 5.4 Missing features for production DA

| Feature | Priority | Notes |
|---------|----------|-------|
| Adaptive multiplicative inflation | High | Anderson (2007) |
| Relaxation-to-prior (RTPS/RTPP) | High | Whitaker & Hamill (2012) |
| Observation batching / super-obbing | Medium | For large obs networks |
| Asynchronous observations | Medium | Obs at different times within window |
| Hybrid EnVar (ensemble + static B) | Medium | Combines flow-dependent + climatological |
| LETKF (Local ETKF) | High | Standard for operational NWP |
| Vertical localization | Medium | Separate horizontal/vertical scales |
| Adaptive localization (GC radius tuning) | Low | Anderson (2012) |
| Innovation diagnostics | High | Desroziers et al. (2005) consistency checks |
| Rank histograms | Medium | Ensemble reliability |

---

## 6. Proposed filterX Module Structure

Based on the inventory above and the EKP design:

```
filterx/
├── __init__.py
├── _src/
│   ├── _protocols.py              # AbstractFilter, AbstractLocalizer, AbstractInflator,
│   │                              # AbstractObsOperator, AbstractScheduler
│   ├── _types.py                  # EnsembleState, AnalysisResult, FilterConfig
│   ├── statistics.py              # ensemble_mean, anomalies, cross_cov, sample_cov
│   │                              # (unifies the 5 filter_step_* variants)
│   ├── gain.py                    # kalman_gain (explicit, woodbury, sqrt, gaussx-backed)
│   ├── filters/
│   │   ├── stochastic_enkf.py     # StochasticEnKF
│   │   ├── etkf.py                # ETKF, ETKF_Livings
│   │   ├── ensrf.py               # EnSRF (Whitaker, serial variants via config)
│   │   ├── estkf.py               # ESTKF
│   │   └── letkf.py               # LETKF (new — high priority)
│   ├── localization/
│   │   ├── gaspari_cohn.py        # Canonical GC taper (unifies 3 existing impls)
│   │   └── cutoff.py              # Hard cutoff, Gaussian taper
│   ├── inflation/
│   │   ├── multiplicative.py      # Fixed + adaptive (Anderson 2007)
│   │   └── rtps.py                # Relaxation to prior spread
│   ├── diagnostics/
│   │   ├── innovation.py          # Innovation statistics, Desroziers checks
│   │   ├── spread.py              # Ensemble spread, rank histograms
│   │   └── costs.py               # KL, log-likelihood (from loss.py)
│   └── utils/
│       ├── linalg.py              # sqrtm, ledoit_wolf, woodbury
│       └── random.py              # Perturbed obs generation
├── docs/
└── tests/
```

### Relationship to EKP

```
filterx/                           # Sequential DA (this document)
├── filters/                       # EnKF variants for state estimation
├── localization/                  # Spatial covariance tapering
└── diagnostics/                   # DA-specific metrics

filterx/                           # Inverse problems (ens_kalman_process.md)
├── processes/                     # EKI, EKS, UKI, GNKI, etc.
├── schedulers/                    # DataMisfitController, FixedScheduler
└── constraints/                   # Bijective transforms for bounded params

# Shared:
├── statistics.py                  # Ensemble mean, cov, cross-cov
├── gain.py                        # Kalman gain computation
└── utils/linalg.py                # Woodbury, sqrtm, shrinkage
```

---

## 7. References

### Implemented algorithms
- Evensen, G. (1994). Sequential data assimilation with a nonlinear quasi-geostrophic model using Monte Carlo methods to forecast error statistics. *JGR*.
- Whitaker, J. S. & Hamill, T. M. (2002). Ensemble data assimilation without perturbed observations. *MWR*.
- Livings, D. M. et al. (2005). Unbiased ensemble square root filters. *Physica D*.
- Vetra-Carvalho, S. et al. (2018). State-of-the-art stochastic data assimilation methods for high-dimensional non-Gaussian problems. *Tellus A*.

### Gap-filling targets
- Anderson, J. L. (2007). An adaptive covariance inflation error correction algorithm for ensemble filters. *Tellus A*.
- Desroziers, G. et al. (2005). Diagnosis of observation, background and analysis-error statistics in observation space. *QJRMS*.
- Hunt, B. R. et al. (2007). Efficient data assimilation for spatiotemporal chaos: A local ensemble transform Kalman filter. *Physica D*.
- Whitaker, J. S. & Hamill, T. M. (2012). Evaluating methods to account for system errors in ensemble data assimilation. *MWR*.

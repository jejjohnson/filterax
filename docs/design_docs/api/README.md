---
status: draft
version: 0.1.0
---

# ekalmX — API Overview

## Surface Inventory

The public API is organized by layer. Each layer is self-contained and useful alone.

---

### Layer 0 — Primitives

Pure functions. Stateless, differentiable, composable. Import directly for maximum control.

| Submodule | Function | Signature (simplified) | Status |
|-----------|----------|----------------------|--------|
| `statistics` | `ensemble_mean` | `(N_e, N_x) → (N_x,)` | Planned |
| `statistics` | `ensemble_anomalies` | `(N_e, N_x) → (N_e, N_x)` | Planned |
| `statistics` | `ensemble_covariance` | `(N_e, N_x) → LowRankUpdate` | Planned |
| `statistics` | `cross_covariance` | `(N_e, N_x), (N_e, N_y) → (N_x, N_y)` | Planned |
| `gain` | `kalman_gain` | `particles, obs_particles, R → (N_x, N_y)` | Planned |
| `likelihood` | `log_likelihood` | `innovation, innovation_cov → Scalar` | Planned |
| `likelihood` | `innovation_statistics` | `particles, obs, obs_op, R → dict` | Planned |
| `localization` | `gaspari_cohn` | `distances, radius → taper` | Planned |
| `localization` | `gaussian_taper` | `distances, radius → taper` | Planned |
| `localization` | `hard_cutoff` | `distances, radius → taper` | Planned |
| `localization` | `localize` | `cov, taper → localized_cov` | Planned |
| `inflation` | `inflate_multiplicative` | `particles, factor → particles` | Planned |
| `inflation` | `inflate_rtps` | `analysis, forecast, alpha → particles` | Planned |
| `inflation` | `inflate_rtpp` | `analysis, forecast, alpha → particles` | Planned |
| `inflation` | `inflate_adaptive` | `particles, obs, obs_op, R → particles` | Planned |
| `perturbations` | `perturbed_observations` | `key, obs, R, N_e → (N_e, N_y)` | Planned |
| `patches` | `create_patches` | `field, patch_size, stride → patches, metadata` | Planned |
| `patches` | `assign_obs_to_patches` | `obs_coords, obs_values, metadata, buffer → dict` | Planned |
| `patches` | `blend_patches` | `patches, metadata, taper_fn → field` | Planned |

### Layer 1 — Components

Protocols, building blocks, configurable filter/process/smoother steps.

**Protocols:**

| Protocol | Methods | Detail |
|----------|---------|--------|
| `AbstractSequentialFilter` | `forecast`, `analysis` | [components.md](components.md) |
| `AbstractProcess` | `init`, `update` | [components.md](components.md) |
| `AbstractDynamics` | `__call__(state, t0, t1) → state` | [components.md](components.md) |
| `AbstractObsOperator` | `__call__(state) → obs` | [components.md](components.md) |
| `AbstractLocalizer` | `__call__(cov, coords) → tapered_cov` | [components.md](components.md) |
| `AbstractInflator` | `__call__(particles, ...) → particles` | [components.md](components.md) |
| `AbstractNoise` | `covariance() → operator`, `sample(key, shape) → Array` | [components.md](components.md) |
| `AbstractScheduler` | `get_dt(state) → Scalar` | [components.md](components.md) |

**Sequential filters:**

| Class | Algorithm | Key property | Status |
|-------|-----------|-------------|--------|
| `StochasticEnKF` | Perturbed observations EnKF | Stochastic, explicit Kalman gain | Planned |
| `ETKF` | Ensemble Transform KF | Deterministic, transform matrix | Planned |
| `ETKF_Livings` | Livings et al. variant | Modified transform | Planned |
| `EnSRF` | Ensemble Square Root Filter | Deterministic, mean + perturbation split | Planned |
| `EnSRF_Whitaker` | Whitaker & Compo variant | Robust square root | Planned |
| `EnSRF_Serial` | Serial obs processing | One obs at a time | Planned |
| `ESTKF` | Ensemble Square Root Transform | Hybrid transform | Planned |
| `LETKF` | Local Ensemble Transform KF | Localized ETKF | Planned |

**Smoothers:**

| Class | Algorithm | Key property | Status |
|-------|-----------|-------------|--------|
| `EnKS` | Ensemble Kalman Smoother | Backward pass over filter results | Planned |
| `EnsembleRTS` | Ensemble RTS Smoother | Rauch-Tung-Striebel for ensembles | Planned |
| `FixedLagSmoother` | Fixed-lag ensemble smoother | Bounded memory backward window | Planned |

**Processes (EKP):**

| Class | Algorithm | Key property | Status |
|-------|-----------|-------------|--------|
| `EKI` | Ensemble Kalman Inversion | Standard optimization | Planned |
| `ETKI` | Ensemble Transform KI | Output-scalable O(d) | Planned |
| `EKS_Process` | Ensemble Kalman Sampler / ALDI | Posterior sampling via Langevin | Planned |
| `UKI` | Unscented Kalman Inversion | Parametric posterior (mean + cov) | Planned |
| `UTKI` | Transform Unscented KI | Output-scalable UKI | Planned |
| `GNKI` | Gauss-Newton KI | Explicit Jacobian estimation | Planned |
| `SparseInversion` | Sparse EKI | L0/L1 sparsity penalties | Planned |

**Schedulers:**

| Class | Algorithm | Status |
|-------|-----------|--------|
| `FixedScheduler` | Constant Δt | Planned |
| `DataMisfitController` | Adaptive, terminates at T=1 | Planned |
| `EKSStableScheduler` | Stability-aware for sampling | Planned |

### Layer 2 — Models

Ready-to-use ensemble methods. Minimal boilerplate.

| Class | Mode | Wraps | Status |
|-------|------|-------|--------|
| `LETKF` | State estimation | L1 LETKF + forecast loop + inflate | Planned |
| `ETKF` | State estimation | L1 ETKF + forecast loop + inflate | Planned |
| `EnSRF` | State estimation | L1 EnSRF + forecast loop + inflate | Planned |
| `EKI` | Parameter estimation | L1 EKI + scheduler + convergence | Planned |
| `EKS` | Posterior sampling | L1 EKS_Process + scheduler + convergence | Planned |
| `UKI` | Parameter estimation | L1 UKI + scheduler + convergence | Planned |

**optax transforms:**

| Function | Wraps | Status |
|----------|-------|--------|
| `ekalmx.optax.eki` | EKI as `optax.GradientTransformation` | Planned |
| `ekalmx.optax.eks` | EKS as `optax.GradientTransformation` | Planned |
| `ekalmx.optax.uki` | UKI as `optax.GradientTransformation` | Planned |

### Utilities

| Function | Purpose | Status |
|----------|---------|--------|
| `diagnostics.spread` | Ensemble spread over time | Planned |
| `diagnostics.rank_histogram` | Ensemble reliability | Planned |
| `diagnostics.innovation_check` | Innovation consistency (Desroziers) | Planned |
| `diagnostics.effective_ensemble_size` | Effective sample size | Planned |

### Zoo (not part of public API)

| Module | Contents |
|--------|----------|
| `zoo.continuous` | Kalman-Bucy, EKF-Bucy, UKF-Bucy, CKF-Bucy, Ensemble-Bucy, Diagonal-Bucy, LowRank-Bucy, H-infinity, Particle |
| `zoo.dynamical_systems` | Lorenz 63, Lorenz 96, Kuramoto-Sivashinsky |

---

## Notation and Conventions

| Symbol | Meaning | Shape |
|--------|---------|-------|
| `N_e` / `J` | Ensemble size (N_e for filters, J for processes) | — |
| `N_x` | State dimension | — |
| `N_y` / `N_d` | Observation dimension | — |
| `N_p` | Parameter dimension (processes) | — |
| `particles` | Ensemble members | `(N_e, N_x)` |
| `obs` | Observation vector | `(N_y,)` |
| `R` | Observation noise covariance | `AbstractLinearOperator` |
| `Q` | Process noise covariance | `AbstractLinearOperator` |
| `K` | Kalman gain | `(N_x, N_y)` |
| `key` | JAX PRNG key | `PRNGKey` |

**Axis convention:** Ensemble dimension is always the leading axis (Decision D10).

**Covariance convention:** Covariances are `lineax.AbstractLinearOperator` (via gaussx), not dense arrays. Diagonal noise is `lineax.DiagonalLinearOperator`. Ensemble covariance is `gaussx.operators.LowRankUpdate`.

**Differentiability:** All L0/L1/L2 components are differentiable via `jax.grad`. No special mode or flag needed (Decision D9).

---

## Import Conventions

```python
import ekalmx                              # Full public API
import ekalmx.filters                      # L1 sequential filters
import ekalmx.processes                    # L1 EKP processes
import ekalmx.smoothers                    # L1 smoothers
import ekalmx.optax                        # optax transforms
import ekalmx.utils.diagnostics as diag    # Diagnostics utilities
```

---

## Detail Files

| File | Covers |
|---|---|
| [primitives.md](primitives.md) | Layer 0 — ensemble statistics, gain, localization, inflation, patches |
| [components.md](components.md) | Layer 1 — protocols, filters, processes, smoothers, schedulers, data types |
| [models.md](models.md) | Layer 2 — LETKF/ETKF/EnSRF, EKI/EKS/UKI, optax transforms, diagnostics |

---

## Performance Notes

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| `ensemble_mean` | O(N_e × N_x) | Single reduction |
| `ensemble_anomalies` | O(N_e × N_x) | Subtract mean |
| `ensemble_covariance` | O(N_e × N_x) | Returns low-rank operator, never forms N_x × N_x |
| `kalman_gain` | O(N_e² × N_y + N_e³) | Woodbury when N_e << N_x; gaussx dispatches |
| `log_likelihood` | O(N_y × N_e²) | Via gaussx logdet on innovation covariance |
| `gaspari_cohn` | O(N_x × N_y) | Piecewise polynomial evaluation |
| `localize` | O(N_x × N_y) | Hadamard product |
| ETKF analysis | O(N_e² × N_y + N_e³) | Transform matrix via eigendecomposition |
| LETKF analysis | O(N_local × N_e² × N_y_local) | Per-gridpoint local analysis |
| EKI update | O(J² × N_d + J × N_p) | Ensemble Kalman gain |

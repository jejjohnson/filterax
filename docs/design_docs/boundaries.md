---
status: draft
version: 0.1.0
---

# ekalmX — Boundaries

## Overview

This document defines what ekalmX owns, what it delegates, and how it interacts with the ecosystem.

---

## Ownership Map

| Concern | Owner | Notes |
|---------|-------|-------|
| Ensemble Kalman filter algorithms (EnKF, ETKF, EnSRF, LETKF, ESTKF) | **ekalmX** | Core competency |
| Ensemble Kalman processes (EKI, EKS, UKI, ETKI, GNKI) | **ekalmX** | Core competency |
| Ensemble smoothers (EnKS, ensemble RTS, fixed-lag) | **ekalmX** | Core competency |
| EKP as optax transforms | **ekalmX** | Thin wrapper over processes |
| Ensemble statistics (mean, anomalies, covariance) | **ekalmX** L0 | Delegates to gaussx for structured ops |
| Kalman gain computation | **ekalmX** L0 | Via `gaussx.recipes.kalman_gain` |
| Localization (Gaspari-Cohn, etc.) | **ekalmX** L0 | Owns taper functions and localization strategies |
| Inflation (multiplicative, RTPS, RTPP, adaptive) | **ekalmX** L0 | Owns inflation strategies |
| Patch-based decomposition (array-level) | **ekalmX** L0 | JAX array ops for spatial decomposition |
| Patch-based decomposition (xarray-level) | **geo_toolz** / **xrpatcher** | User composes with ekalmX patches |
| Log-likelihood computation | **ekalmX** L0 | For differentiable training |
| Innovation diagnostics, rank histograms | **ekalmX** utils | DA-specific evaluation metrics |
| Structured covariance operators | **gaussx** | ekalmX consumes via dependency |
| Optimization algorithms | **optimistix** | Pluggable backend |
| Learning rate schedules | **optax** | ekalmX wraps for EKP schedulers |
| Forward models (ocean, atmosphere, ODEs) | **somax** / **diffrax** / user | Pluggable via `AbstractDynamics` |
| Observation operators | **user** | Pluggable via `AbstractObsOperator` |
| Variational DA (4DVar, 4DVarNet) | **vardax** | Sister library, different DA paradigm |
| Natural-gradient optimization (BLR) | **optax_bayes** | Sister library |
| xarray-level DA orchestration | **xr_assimilate** | Consumes ekalmX as compute backend |
| xarray pre/post-processing | **geo_toolz** | Upstream (preprocess) and downstream (evaluate) |
| Training / assimilation loops | **user** | Library, not framework |
| Continuous-time filters (Kalman-Bucy, etc.) | **ekalmX** zoo/ | Reference only, not core API |
| Test dynamical systems (L63, L96, KS) | **ekalmX** zoo/ | For demos and testing |

---

## Decision Table

| Scenario | Recommendation |
|----------|---------------|
| Estimate ocean state from satellite observations | `LETKF(dynamics=somax_model, obs_op=my_H, ...).assimilate(ensemble, obs)` |
| Calibrate simulator parameters without gradients | `EKI(forward_fn, obs, noise_cov, ...).run(init_particles)` |
| Learn neural ODE dynamics end-to-end through filter | Build EnKF at L1, wrap in `jax.grad`, user-owned training loop |
| Use EKI as drop-in optimizer (optax style) | `ekalmx.optax.eki(forward_fn, obs, ...)` composed with `optax.chain` |
| Smooth state estimates (backward pass) | `EnKS(filter_result).smooth()` or L0 ensemble RTS primitives |
| Localized filtering on large spatial domain | LETKF with `GaspariCohn(radius)` localizer, or patch decomposition via L0 |
| Evaluate filter performance | `ekalmx.utils.diagnostics` — spread, rank histogram, innovation checks |
| xarray-level DA orchestration | xr_assimilate (consumes ekalmX as compute backend) |
| Variational DA (4DVar, adjoint methods) | vardax (sister library) |
| Work with structured covariance operators directly | gaussx (dependency) |
| Preprocess satellite data before assimilation | geo_toolz (upstream) |
| Evaluate results with RMSE, spectral scores | geo_toolz (downstream) |

---

## Ecosystem Interactions

### How somax models become dynamics

```python
import somax
import ekalmx

# somax owns the forward model
swm = somax.ShallowWaterModel(grid=grid, params=params)

# Wrap as ekalmX dynamics (user code or thin adapter)
class SWMDynamics(ekalmx.AbstractDynamics):
    model: somax.ShallowWaterModel
    dt: float

    def __call__(self, state, t0, t1):
        return self.model.integrate(state, t0=t0, t1=t1, dt=self.dt).ys[-1]

# Plug into filter
letkf = ekalmx.LETKF(
    dynamics=SWMDynamics(model=swm, dt=60.0),
    obs_op=my_obs_op,
    localizer=ekalmx.GaspariCohn(radius=500e3),
    config=ekalmx.FilterConfig(n_ensemble=50),
)
result = letkf.assimilate(ensemble, observations)
```

### How EKP becomes an optax transform

```python
import ekalmx
import optax

# EKI as optax — parallel to optax_bayes
optimizer = ekalmx.optax.eki(
    forward_fn=expensive_simulator,
    obs=observations,
    noise_cov=R,
    scheduler=ekalmx.DataMisfitController(),
    n_ensemble=100,
)

# Compose with optax ecosystem
optimizer = optax.chain(optimizer, optax.clip_by_global_norm(1.0))

# Standard loop
state = optimizer.init(params)
for step in range(n_steps):
    updates, state = optimizer.update(grads=None, state=state, params=params)
    params = optax.apply_updates(params, updates)
```

### How xr_assimilate consumes ekalmX

```python
import xr_assimilate as xra
import ekalmx

# ekalmX provides the array-level compute
filter_backend = ekalmx.LETKF(dynamics=my_dynamics, obs_op=my_H, ...)

# xr_assimilate orchestrates xarray I/O, time stepping, diagnostics
assimilator = xra.Assimilator(
    backend=filter_backend,
    observations=obs_dataset,      # xr.Dataset
    background=background_dataset, # xr.Dataset
)
result = assimilator.run()  # returns xr.Dataset
```

### How differentiable training works

```python
import ekalmx
import optax

# All ekalmX filters are differentiable by construction (pure JAX)
def loss_fn(dynamics_params, ensemble, observations):
    dynamics = MyNeuralODE(params=dynamics_params)
    filter = ekalmx.ETKF(obs_op=my_H, config=config)

    # Forward pass through filter — fully differentiable
    state = ekalmx.FilterState(particles=ensemble, step=0)
    for obs in observations:
        particles = eqx.filter_vmap(dynamics)(state.particles, t0, t1)
        result = filter.analysis(particles, obs.values, obs.operator, obs.noise)
        state = ekalmx.FilterState(particles=result.particles, step=state.step + 1)

    return -result.log_likelihood  # backprop through entire filter

# Standard JAX training loop
grads = jax.grad(loss_fn)(dynamics_params, ensemble, observations)
```

### Dependency graph

```
geo_toolz (preprocess)
    ↓
somax / diffrax / user code ──→ ekalmX ←── gaussx (covariance ops)
    (AbstractDynamics)             ↑  ↑        ↑
                              optax  optimistix lineax
                              (EKP     (solver   (transitive)
                               transforms) backends)
    ↓
xr_assimilate (xarray orchestration, consumes ekalmX)
    ↓
geo_toolz (evaluate)

Sister libraries (same level, different paradigm):
    vardax      — variational DA
    optax_bayes — natural-gradient Bayesian optimization
```

---

## Scope

### In Scope

- Discrete-time ensemble Kalman filters (EnKF, ETKF, EnSRF, ESTKF, LETKF)
- Ensemble Kalman processes (EKI, EKS, UKI, ETKI, GNKI, Sparse)
- Ensemble smoothers (EnKS, ensemble RTS, fixed-lag)
- EKP as optax transforms
- Ensemble statistics, Kalman gain, localization, inflation primitives
- Patch-based spatial decomposition (array-level)
- Log-likelihood and innovation diagnostics
- Reference implementations in zoo/ (continuous-time filters, toy dynamical systems)

### Out of Scope

- Variational data assimilation (4DVar, 4DVarNet) — vardax
- Forward models (ocean, atmosphere, ODEs) — somax, diffrax, user code
- Structured linear algebra (operators, solve, logdet) — gaussx
- Natural-gradient optimization (BLR) — optax_bayes
- xarray-level pre/post-processing — geo_toolz
- xarray-level DA orchestration — xr_assimilate
- Optimization algorithms (L-BFGS, Newton) — optimistix
- Training loops or experiment infrastructure — user-owned
- Gaussian process inference — pyrox_gp

---

## Testing Strategy

### Test organization

| Category | What's tested | Module |
|----------|---------------|--------|
| Primitives | ensemble_mean, anomalies, covariance, cross_cov correctness; Kalman gain against analytic solutions | `test_statistics.py`, `test_gain.py` |
| Localization | Gaspari-Cohn taper properties (compact support, smoothness, symmetry); localized vs unlocalized covariance | `test_localization.py` |
| Inflation | Multiplicative, RTPS, RTPP preserve ensemble mean; spread increases | `test_inflation.py` |
| Likelihood | Log-likelihood against analytic Gaussian; innovation statistics | `test_likelihood.py` |
| Patches | Create/assign/blend roundtrip; overlap blending weights sum to 1 | `test_patches.py` |
| Filters | Each filter reduces RMSE vs prior on toy problem; analysis mean/spread plausible | `test_filters/` |
| Processes | EKI/EKS/UKI converge on known inverse problem (e.g., linear Gaussian) | `test_processes/` |
| Smoothers | EnKS/ensemble RTS improve on filter estimate; match analytic smoother on linear case | `test_smoothers/` |
| optax | EKI as GradientTransformation matches standalone EKI; composable with optax.chain | `test_optax.py` |
| JAX transforms | All components work under `jax.jit`, `jax.grad`, `eqx.filter_vmap` | Cross-cutting |
| Protocol conformance | Every filter/process/localizer/inflator satisfies its abstract protocol | Cross-cutting |
| Zoo | Smoke tests only — L63/L96 run, continuous-time filters don't crash | `test_zoo/` |

### Test priorities

1. **JAX transform compatibility** — everything JITs, grads, vmaps
2. **Protocol conformance** — all implementations satisfy their abstract interface
3. **Correctness on linear Gaussian** — where analytic answers exist, match them
4. **Convergence on toy problems** — filters reduce error, processes find parameters

---

## Open Questions

1. **pyrox_gp overlap** — Both ekalmX and pyrox_gp need Kalman filter/smoother primitives (ekalmX for ensemble DA, pyrox_gp for state-space GPs). Plan: push shared math to gaussx so both consume without circular dependencies. Needs further design.

2. **Hybrid EnVar** — Ensemble-variational hybrid methods combine flow-dependent ensemble covariance (ekalmX) with static background covariance (vardax). Crosses the boundary between the two libraries. Deferred to a future phase when both libraries are mature.

3. **xr_assimilate integration depth** — ekalmX is the compute backend, but the API contract between them (how xr_assimilate calls ekalmX, what state/result types cross the boundary) isn't defined yet.

4. **coordax integration** — Should ensemble states use coordax for coordinate-aware arrays? Depends on coordax maturity. Same open question as vardax.

5. **3D / volumetric support** — How much native 3D support vs treating 3D as multilayer 2D via `eqx.filter_vmap`?

6. **Particle filters** — The zoo has a continuous-time particle filter. Should ekalmX core include discrete particle filters (bootstrap PF, optimal PF), or is that a separate library?

7. **Ensemble size adaptivity** — Dynamic ensemble resizing during assimilation. Research topic, not immediate.

8. **Package rename** — Directory is currently `filterX`, library will be `ekalmx`. Rename when standalone repo is created.

---

## Items Deferred from gaussx

The following items were originally scoped for gaussx but belong here per the ownership map. Full API sketches from the gaussx implementation plan are preserved below for reference.

### Ensemble DA primitives (originally gaussx Phase 15.6, 20.2, 20.3)

These map directly to ekalmX L0 primitives (localization + inflation):

- `gaspari_cohn(distances, radius)` — 5th-order piecewise polynomial taper (C2 smooth, compact support)
- `localize_covariance(cov, taper)` — Hadamard (element-wise) product with taper matrix
- `inflate_multiplicative(particles, factor)` — ensemble anomaly inflation
- `inflate_rtps(analysis, forecast, alpha)` — relaxation to prior spread
- `inflate_rtpp(analysis, forecast, alpha)` — relaxation to prior perturbations
- `ledoit_wolf_shrinkage(X_anomalies)` — optimal shrinkage for rank-deficient ensemble covariance (originally Phase 19.4)

### EnKF analysis step variants (originally gaussx Phase 19.3)

Core competency — four standard ensemble analysis formulations:

- `enkf_analysis_stochastic` — perturbed observations
- `enkf_analysis_etkf` — deterministic symmetric square-root (Ensemble Transform KF)
- `enkf_analysis_ensrf` — serial square-root (Whitaker & Hamill)
- `enkf_analysis_estkf` — mean-preserving square-root transform

Source: `enskf_zoo.py` (SEnKF, ETKF, EnSRF, ESTKF variants)

### Specialized filtering variants (originally gaussx Phases 18.6–18.8, 19.1, 19.2, 19.5)

| Item | Description | Source |
|------|-------------|--------|
| DARE / infinite horizon filter (18.6) | Steady-state Kalman with constant gain via Discrete Algebraic Riccati Equation | BayesNewton `ops.py` |
| Pairs filter (18.7) | Kalman filter on consecutive state pairs `[u_{n-1}, u_n]` for sparse Markov models | BayesNewton `ops.py` |
| Mean-field Kalman filter (18.8) | Block-diagonal factorization across latent dimensions for multi-output SSGPs | BayesNewton `ops.py` |
| Square-root Kalman filter (19.1) | Propagates Cholesky factor directly (P = SS^T), guarantees PSD | kalman_filter snippets |
| Low-rank covariance filter (19.2) | P ≈ SS^T with S ∈ R^{n×r}, SVD truncation per step | kalman_filter snippets |
| Continuous-time Lyapunov (19.5) | `lyapunov_ode`, `solve_lyapunov_steady_state` for Kalman-Bucy filters | `kf_continuous.py` |

These can land in ekalmX core or zoo depending on maturity and demand.

---

## Future Work

| Phase | Focus | Depends on |
|-------|-------|------------|
| Phase 1 | Core primitives + ETKF/EnSRF/LETKF + EKI/EKS | gaussx stable |
| Phase 2 | Smoothers (EnKS, ensemble RTS) + remaining processes (UKI, ETKI, GNKI) | Phase 1 |
| Phase 3 | optax integration, differentiable training examples | Phase 1 |
| Phase 4 | xr_assimilate integration, geo_toolz pipeline examples | Phase 1 + xr_assimilate |
| Phase 5 | Advanced: LETKF at scale, adaptive inflation/localization, particle filters | Phase 1-2 |
| Future | Hybrid EnVar, pyrox_gp shared primitives, coordax | Ecosystem maturity |

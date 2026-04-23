---
status: draft
version: 0.1.0
---

# ekalmX — Vision

## One-liner

> Differentiable ensemble Kalman methods for filtering, inversion, and joint estimation in JAX/Equinox.

---

## Motivation

Ensemble Kalman methods are the workhorse of large-scale data assimilation and derivative-free inference. The same core machinery — ensemble statistics, Kalman gain, covariance localization — underpins three distinct use cases:

1. **State estimation** — sequential EnKF/ETKF/LETKF for tracking dynamical systems from noisy observations.
2. **Parameter estimation** — Ensemble Kalman Inversion (EKI) and sampling (EKS) for calibrating expensive black-box simulators without gradients.
3. **Joint state + parameter estimation** — backpropagating through the entire filter to learn neural dynamics, observation operators, and noise parameters end-to-end.

Today, researchers reach for different tools depending on which mode they need: FilterPy or DART for classical state estimation, EnsembleKalmanProcesses.jl for inversion, torchEnKF for differentiable filtering. None of these share infrastructure, and switching between modes means switching languages or frameworks.

ekalmX provides all three modes in a single JAX/Equinox library, with shared ensemble primitives, structured covariance operations via gaussx, and full differentiability by default.

---

## User Stories

**DA researcher** — "I have a somax ocean model and satellite observations. I want to run a localized ETKF to estimate ocean state, then swap in EKI to calibrate model parameters — without rewriting my observation operator or ensemble infrastructure."

**ML+DA researcher** — "I want to learn a neural ODE dynamics model by backpropagating through an EnKF. The filter should be a differentiable layer in my training loop, not a black box."

**Inverse problem researcher** — "I have an expensive climate simulator I can't differentiate. I want to calibrate 50 parameters from observations using Ensemble Kalman Inversion with 100 ensemble members."

**Student / newcomer** — "I want to call `LETKF(model, obs_op).assimilate(ensemble, obs)` and get an analysis ensemble back. I don't need to understand Woodbury identities."

---

## Design Principles

1. **Equinox-native** — All components are `eqx.Module` pytrees. No mutable state, no side effects. Fully compatible with `jax.jit`, `jax.grad`, `eqx.filter_vmap`, and the equinox ecosystem (optimistix, diffrax, lineax, gaussx).

2. **Protocol-driven** — Core abstractions (`AbstractFilter`, `AbstractObsOperator`, `AbstractDynamics`, `AbstractLocalizer`) are defined as protocols. Users bring their own forward models (somax, neural ODEs, custom) and observation operators.

3. **Differentiable by default** — The entire filter is a differentiable computation graph. Backpropagating through EnKF for joint estimation is not an afterthought — it's the design constraint that shaped every implementation choice.

4. **Progressive disclosure** — Three layers of API complexity:
   - **Layer 0 (Primitives)**: Pure functions — ensemble statistics, Kalman gain, localization tapers, covariance operations. Power users compose these freely.
   - **Layer 1 (Components)**: Protocols and building blocks — filter steps, localizers, inflators, schedulers. Researchers snap these together into custom pipelines.
   - **Layer 2 (Models)**: Ready-to-use ensemble methods — `ETKF(...)`, `LETKF(...)`, `EKI(...)`. All filters are differentiable by construction (see Decision D9). Minimal boilerplate for standard workflows.

5. **Library, not framework** — Ships building blocks, not an opinionated pipeline. Assimilation loops, training loops, and experiment orchestration are user-owned. ekalmX provides `filter_step` and `process_step`, not `fit()`.

6. **gaussx-backed linear algebra** — Covariance operations (ensemble covariance as low-rank operator, Kalman gain via Woodbury, localization tapers) delegate to gaussx structured operators where possible. ekalmX doesn't reinvent linear algebra.

---

## Identity

### ekalmX IS

- A zoo of ensemble Kalman algorithms: EnKF, ETKF, EnSRF, ESTKF, LETKF (state estimation); EKI, EKS, UKI, ETKI, GNKI (inversion/sampling); differentiable EnKF (joint estimation)
- Differentiable — the filter is a JAX computation graph, backprop flows through it
- Protocol-driven — bring your own forward model, observation operator, localizer
- JAX/Equinox-native — pytrees, JIT, vmap, grad, scan
- Three-layer progressive API (primitives → components → models)
- gaussx-backed for structured covariance operations

### ekalmX IS NOT

| Not this | Use instead |
|----------|-------------|
| Variational data assimilation (4DVar, 4DVarNet) | vardax |
| Forward models (ocean, atmosphere, ODEs) | somax, diffrax, or user's own |
| xarray pre/post-processing (regrid, detrend, evaluate) | geo_toolz |
| Optimization algorithms (L-BFGS, Newton, solvers) | optimistix (pluggable backend) |
| Structured linear algebra (operators, solve, logdet) | gaussx (dependency) |
| Natural-gradient / Bayesian Learning Rule optimization | optax_bayes |
| Training loops or experiment infrastructure | user-owned |

---

## Migration Context

### Internal

The `kalman_filter/` directory contains ~5,000 lines of working but informal JAX/NumPy code: 9 discrete EnKF variants, 9 continuous-time filters, patch-based spatial decomposition, and parameter estimation utilities. ekalmX formalizes this into a proper library with protocols, types, and a layered API. See [research/research_enskf.md](research/research_enskf.md) for the full inventory.

### External

ekalmX provides a JAX/Equinox alternative to:

| Tool | Language | Limitation ekalmX addresses |
|------|----------|----------------------------|
| [torchEnKF](https://github.com/ymchen0/torchEnKF) | PyTorch | Not JAX-native; no structured covariance; limited algorithm zoo |
| [ROAD-EnKF](https://github.com/ymchen0/ROAD-EnKF) | PyTorch | Same; adds reduced-order but still PyTorch-only |
| [EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl) | Julia | Inversion-only; no sequential filtering; no Python ecosystem |
| [FilterPy](https://github.com/rlabbe/filterpy) | NumPy/SciPy | Not differentiable; no GPU; no ensemble methods at scale |
| [DART](https://github.com/NCAR/DART) | Fortran | Production-grade but monolithic; not composable with ML |

### Key references

- Evensen, G. (1994). Sequential data assimilation with a nonlinear quasi-geostrophic model. *JGR*.
- Iglesias, Law & Stuart (2013). Ensemble Kalman methods for inverse problems. *Inverse Problems*.
- Huang, Schneider & Stuart (2022). Unscented Kalman Inversion. *SIAM/ASA JUQ*.
- Chen et al. (2022). Auto-differentiable Ensemble Kalman Filters. *SIMODS*.
- Chen et al. (2023). Reduced-Order Autodifferentiable Ensemble Kalman Filters. *arXiv:2301.11961*.
- Vetra-Carvalho et al. (2018). State-of-the-art stochastic data assimilation methods. *Tellus A*.

---

## Connection to Ecosystem

```
                    ┌──────────────┐
                    │   somax      │  Forward models (ocean, atmos)
                    │   diffrax    │  ODE/SDE integration
                    │   user code  │  Neural ODEs, custom dynamics
                    └──────┬───────┘
                           │ pluggable via AbstractDynamics
                    ┌──────▼───────┐
   geo_toolz ──→   │   ekalmX     │   ←── gaussx (covariance ops)
   (preprocess)    │              │   ←── optimistix (optional solvers)
                    └──────┬───────┘
                           │ analysis ensemble
                    ┌──────▼───────┐
                    │   geo_toolz  │  Evaluate (RMSE, spectral scores)
                    │   user code  │  Visualization, diagnostics
                    └──────────────┘

Sister libraries (same level, different DA paradigm):
    vardax      — variational DA (4DVar, 4DVarNet)
    optax_bayes — natural-gradient Bayesian optimization (BLR)
```

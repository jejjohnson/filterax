# Ensemble Kalman Processes: Design Document

**Derivative-Free Bayesian Optimisation and Inference via Ensemble Filtering**

Based on the CliMA EnsembleKalmanProcesses.jl library and the theoretical
foundations of Iglesias, Law & Stuart (2013), Huang, Schneider & Stuart (2022),
Garbuno-Inigo et al. (2020), and Calvello, Reich & Stuart (2022).

---

## Table of Contents

1. [General Mathematical Formulation](#1-general-mathematical-formulation)
2. [Numerical Requirements](#2-numerical-requirements)
3. [Process Zoo](#3-process-zoo)
4. [JAX API Design](#4-jax-api-design)
5. [Example Applications](#5-example-applications)
6. [Connection to the Bayesian Learning Rule](#6-connection-to-the-bayesian-learning-rule)
7. [GaussX Integration](#7-gaussx-integration)
8. [References](#8-references)

---

## 1. General Mathematical Formulation

### 1.1 The Inverse Problem

We seek parameters θ ∈ ℝᵖ given the data-model relation:

```
y = G(θ) + η,      η ∼ 𝒩(0, Γ_y)
```

where G: ℝᵖ → ℝᵈ is the (possibly black-box) forward map, y ∈ ℝᵈ are
observations, and Γ_y is the observation noise covariance.

The MLE (maximum likelihood) minimises the data misfit:

```
ℒ_MLE(θ) = ½ (y − G(θ))ᵀ Γ_y⁻¹ (y − G(θ))
```

The MAP (maximum a posteriori) adds a Gaussian prior 𝒩(m₀, C₀):

```
ℒ_MAP(θ) = ½ (y − G(θ))ᵀ Γ_y⁻¹ (y − G(θ)) + ½ (θ − m₀)ᵀ C₀⁻¹ (θ − m₀)
```

### 1.2 The Ensemble Representation

Instead of maintaining a parametric distribution q(θ|λ) (as in the BLR),
ensemble Kalman processes maintain a particle ensemble:

```
Θ_n = {θ_n⁽¹⁾, θ_n⁽²⁾, …, θ_n⁽ᴶ⁾}     J particles in ℝᵖ
```

All distributional quantities are estimated empirically from the ensemble:

```
Ensemble mean:           θ̄_n = (1/J) Σⱼ θ_n⁽ʲ⁾
Parameter covariance:    C^{θθ}_n = (1/J) Σⱼ (θ_n⁽ʲ⁾ − θ̄_n)(θ_n⁽ʲ⁾ − θ̄_n)ᵀ
Output mean:             Ḡ_n = (1/J) Σⱼ G(θ_n⁽ʲ⁾)
Cross-covariance:        C^{θG}_n = (1/J) Σⱼ (θ_n⁽ʲ⁾ − θ̄_n)(G(θ_n⁽ʲ⁾) − Ḡ_n)ᵀ
Output covariance:       C^{GG}_n = (1/J) Σⱼ (G(θ_n⁽ʲ⁾) − Ḡ_n)(G(θ_n⁽ʲ⁾) − Ḡ_n)ᵀ
```

The cross-covariance C^{θG} is the key object. In the linear case G(θ) = Hθ,
it reduces to C^{θθ}Hᵀ, and the ensemble update becomes the exact Kalman
update. For nonlinear G, it provides a stochastic linearisation — an implicit,
derivative-free approximation to the Jacobian.

### 1.3 The Core EKI Update

The ensemble Kalman inversion (EKI) update for particle j is:

```
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│  θ_{n+1}⁽ʲ⁾ = θ_n⁽ʲ⁾ + Δt · K_n (y − G(θ_n⁽ʲ⁾))             │
│                                                                   │
│  where  K_n = C^{θG}_n (Γ_y + Δt · C^{GG}_n)⁻¹    Kalman gain  │
│         Δt = learning rate (adaptive)                             │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

Each particle is shifted toward observations by an amount proportional to the
ensemble-estimated correlation between parameters and outputs. The Δt·C^{GG}
term in the denominator acts as regularisation — the Kalman gain is damped
when the ensemble spread in output space is large relative to the observation
noise.

### 1.4 The Augmented State (Prior Enforcement)

To enforce the prior at every iteration (the "infinite-time" or Tikhonov
variant), the forward map, data, and noise covariance are augmented:

```
G̃(θ) = [G(θ), θ]ᵀ       ỹ = [y, m₀]ᵀ       Γ̃_y = diag(Γ_y, C₀)
```

This converts the MAP problem into an MLE problem in the augmented space.
The ensemble update is identical in form, but now the cross-covariance
C^{θG̃} includes the identity block C^{θθ}, which pulls particles back
toward the prior.

### 1.5 The Ensemble Kalman Sampler (EKS / ALDI)

EKS adds a Langevin diffusion to produce posterior *samples* rather than
just a point estimate. The ALDI (Affine-invariant Langevin Dynamics) variant:

```
θ*_{n+1}⁽ʲ⁾ = θ_n⁽ʲ⁾ − (Δt/J) Σ_k ⟨G(θ_n⁽ᵏ⁾)−Ḡ_n, Γ_y⁻¹(G(θ_n⁽ʲ⁾)−y)⟩ θ_n⁽ᵏ⁾
              + ((d+1)/J)(θ_n⁽ʲ⁾ − θ̄_n)
              − Δt · C^{θθ}_n Γ_θ⁻¹ θ*_{n+1}⁽ʲ⁾          ← prior pull (implicit)

θ_{n+1}⁽ʲ⁾ = θ*_{n+1}⁽ʲ⁾ + √(2Δt · C^{θθ}_n) ξ_n⁽ʲ⁾    ← Langevin noise
```

where ξ ∼ 𝒩(0, I). The (d+1)/J term is a finite-sample correction
(ALDI), and the implicit prior pull stabilises the dynamics. The ensemble
does not collapse — it samples the posterior at stationarity.

### 1.6 Unscented Kalman Inversion (UKI)

UKI replaces the random ensemble with a deterministic sigma-point
quadrature. It explicitly tracks mean m_n and covariance C_n:

```
Predict:
    m̂_{n+1} = r + α(m_n − r)
    Ĉ_{n+1} = α² C_n + Σ_ω

Generate sigma points (symmetric, 2p+1 points):
    θ̂⁰ = m̂_{n+1}
    θ̂ʲ = m̂_{n+1} ± c_j [√Ĉ_{n+1}]_j     j = 1, …, p

Evaluate forward model:
    ŷʲ = G(θ̂ʲ)

Analysis (Kalman update):
    C^{θy} = Σ_j W_j (θ̂ʲ − m̂)(ŷʲ − ŷ⁰)ᵀ
    C^{yy} = Σ_j W_j (ŷʲ − ŷ⁰)(ŷʲ − ŷ⁰)ᵀ + Σ_ν
    m_{n+1} = m̂_{n+1} + C^{θy} (C^{yy})⁻¹ (y − ŷ⁰)
    C_{n+1} = Ĉ_{n+1} − C^{θy} (C^{yy})⁻¹ (C^{θy})ᵀ
```

where r = prior mean, α controls mean-reversion, and Σ_ω, Σ_ν are
tuned to ensure convergence to the posterior (see Huang et al. 2022).

### 1.7 Gauss-Newton Kalman Inversion (GNKI)

GNKI estimates the Jacobian from the ensemble and performs a
Gauss-Newton step. The ensemble approximation to the Jacobian is:

```
G̃_n ≈ C^{θG}_n (C^{θθ}_n)⁻¹     (statistical linearisation)
```

Then the Kalman gain becomes:

```
K_n = C^{θθ}_n G̃_nᵀ (G̃_n C^{θθ}_n G̃_nᵀ + Γ_y)⁻¹
```

The update includes both a data-fit term and a prior-pull term:

```
θ_{n+1}⁽ʲ⁾ = θ_n⁽ʲ⁾ + α { K_n(y − G(θ_n⁽ʲ⁾)) + (I − K_n G̃_n)(m₀ − θ_n⁽ʲ⁾) }
```

In the linear-Gaussian case, GNKI recovers the exact posterior mean
and covariance. It provides uncertainty through ensemble spread.

---

## 2. Numerical Requirements

### 2.1 Ensemble Size

```
┌─────────────────────┬──────────────┬───────────────────────────────────┐
│ Process             │ Ensemble J   │ Notes                             │
├─────────────────────┼──────────────┼───────────────────────────────────┤
│ EKI / ETKI          │ J ≥ p+1      │ Ideally J ≈ 2p−10p for p ≤ 100. │
│                     │ (minimum)    │ Localization helps when J ≪ p.   │
│                     │              │                                   │
│ UKI (symmetric)     │ J = 2p+1     │ Fixed. Deterministic quadrature. │
│ UKI (simplex)       │ J = p+2      │ Fixed. More efficient.           │
│                     │              │                                   │
│ EKS / ALDI          │ J ≥ p+1      │ Larger J = better posterior       │
│                     │              │ approximation (no collapse).      │
│                     │              │                                   │
│ GNKI                │ J ≥ p+1      │ Needs invertible C^{θθ}, so J>p. │
└─────────────────────┴──────────────┴───────────────────────────────────┘
```

Rule of thumb from EKP.jl: for p ≤ 100, use J ≈ max(p+1, 10p). For p > 100,
localization or ETKI is necessary.

### 2.2 Covariance Estimation and Rank Deficiency

The empirical covariance C^{θθ} has rank at most J−1. When J ≪ p (the
typical regime for expensive simulators), this is severely rank-deficient.

**Safeguards:**

```
Inflation:       C^{θθ} ← (1+δ) C^{θθ}             δ ≈ 0.01−0.1
Additive:        C^{θθ} ← C^{θθ} + εI               ε ≈ 1e-6
Localization:    C^{θθ} ← L ⊙ C^{θθ}               L = tapering matrix
SEC:             Sampling error correction (Raanes et al., 2019)
```

Localization tapers long-range spurious correlations. It is available for
EKI and EKS but not ETKI or UKI.

### 2.3 Learning Rate (Timestep) Scheduling

The learning rate Δt controls convergence speed and stability:

```
┌──────────────────────────┬──────────────────────────────────────────────┐
│ Scheduler                │ Behaviour                                    │
├──────────────────────────┼──────────────────────────────────────────────┤
│ DefaultScheduler(Δt)     │ Fixed step. Simple but may diverge.          │
│                          │                                              │
│ DataMisfitController     │ Adaptive: Δt chosen so that the data misfit │
│   (terminate_at=T)       │ decreases by a controlled factor each step. │
│                          │ For finite-time: Σ Δt_n → T=1 triggers      │
│                          │ termination (MAP at T=1).                    │
│                          │                                              │
│ EKSStableScheduler       │ Adaptive: Δt ∝ 1/‖C^{GG}‖, ensuring the   │
│                          │ EKS particle system remains stable.          │
│                          │                                              │
│ MutableScheduler(inner)  │ Wraps another scheduler, allows runtime      │
│                          │ modification of Δt.                          │
└──────────────────────────┴──────────────────────────────────────────────┘
```

**The T=1 principle (finite-time EKI):** The algorithm time Σ Δt_n = T acts
as an inverse temperature. At T=1, the ensemble mean approximates the MAP.
At T → ∞, the ensemble collapses to the MLE (prior information is lost).
The DataMisfitController adaptively selects Δt_n to reach T=1 efficiently.

### 2.4 Constraint Handling

EKP operates in an unconstrained space via bijective transforms T: ϕ → θ.

```
θ = T(ϕ)    ← unconstrained parameters (EKP works here)
ϕ = T⁻¹(θ)  ← constrained physical parameters (model works here)

Common transforms:
  Bounded (a,b):    T = logit-type, ϕ ∈ (a,b) ↔ θ ∈ ℝ
  Positive:         T = log, ϕ > 0 ↔ θ ∈ ℝ
  Unbounded:        T = identity
```

The forward map seen by EKP is G̃ = H ∘ Ψ ∘ T⁻¹, where Ψ is the physical
simulator and H is the observation operator.

### 2.5 Failure Handling

For expensive simulators, some ensemble members may fail (e.g., a climate
model crashes for certain parameter values). EKP supports:

```
FailedParticles:     Flag failed ensemble members per iteration.
                     Failed particles are excluded from covariance
                     estimates and replaced by the ensemble mean update.
```

### 2.6 JAX-Specific Considerations

```
Requirement               Implementation Note
─────────────────────────────────────────────────────────────────
vmap over ensemble         jax.vmap(forward_model)(ensemble) is the
                           natural way to evaluate G on all J particles.
                           Requires the model to be JAX-traceable.

jit-compatible state       All state is arrays/NamedTuples. No Python
                           mutation — use functional updates.

scan for iteration loop    The update loop is scan-compatible if the
                           forward model is pure.

Parallelism for expensive  For non-JAX simulators (e.g., Fortran climate
models                     codes), use multiprocessing/MPI to evaluate
                           G in parallel, then feed results back to JAX.

float64                    Covariance estimation benefits from float64,
                           especially when J is small relative to p.

Random keys                EKI perturbed observations and EKS diffusion
                           require PRNG keys threaded through state.
```

---

## 3. Process Zoo

### 3.1 Optimisation Processes

```
┌───────────────────────┬───────────┬──────────┬────────────┬─────────────────┐
│ Process               │ J (min)   │ Scalable │ Prior      │ Solution        │
│                       │           │ in d_obs │ enforced   │                 │
├───────────────────────┼───────────┼──────────┼────────────┼─────────────────┤
│ EKI (finite-time)     │ ~10p      │ No       │ Init only  │ MAP at T=1      │
│ Inversion()           │           │          │            │ MLE at T→∞     │
│                       │           │          │            │                 │
│ EKI (infinite-time)   │ ~10p      │ No       │ Every step │ MAP at T→∞    │
│ Inversion(prior)      │           │          │ (augmented)│                 │
│                       │           │          │            │                 │
│ ETKI (finite-time)    │ ~p        │ Yes      │ Init only  │ MAP at T=1      │
│ TransformInversion()  │           │ O(d)     │            │                 │
│                       │           │          │            │                 │
│ ETKI (infinite-time)  │ ~p        │ Yes      │ Every step │ MAP at T→∞    │
│ TransformInversion    │           │ O(d)     │            │                 │
│   (prior)             │           │          │            │                 │
│                       │           │          │            │                 │
│ Sparse EKI            │ ~10p      │ No       │ Every step │ MAP + sparsity  │
│ SparseInversion       │           │          │ + L0/L1    │                 │
│   (prior)             │           │          │            │                 │
└───────────────────────┴───────────┴──────────┴────────────┴─────────────────┘
```

### 3.2 Inference / Sampling Processes

```
┌───────────────────────┬───────────┬──────────┬────────────┬─────────────────┐
│ Process               │ J (min)   │ Unc.     │ Mechanism  │ Output          │
│                       │           │ Quant.   │            │                 │
├───────────────────────┼───────────┼──────────┼────────────┼─────────────────┤
│ EKS / ALDI            │ ~10p      │ Yes      │ Langevin   │ Posterior       │
│ Sampler(prior)        │           │          │ diffusion  │ samples         │
│                       │           │          │            │ (Gaussian-like) │
│                       │           │          │            │                 │
│ UKI                   │ 2p+1      │ Yes      │ Sigma-pt   │ Posterior       │
│ Unscented(prior)      │ (fixed)   │          │ quadrature │ (m_n, C_n)      │
│                       │           │          │            │                 │
│ UTKI                  │ 2p+1      │ Yes      │ Sigma-pt   │ Posterior       │
│ TransformUnscented    │ (fixed)   │ (output  │ + sqrt     │ (m_n, C_n)      │
│   (prior)             │           │ scalable)│ filter     │                 │
│                       │           │          │            │                 │
│ GNKI                  │ ~p+1      │ Yes      │ Gauss-     │ Posterior       │
│ GaussNewtonInversion  │           │          │ Newton +   │ (ensemble       │
│   (prior)             │           │          │ stat.lin.  │  spread)        │
└───────────────────────┴───────────┴──────────┴────────────┴─────────────────┘
```

### 3.3 Pseudocode

**Algorithm: EKI (Ensemble Kalman Inversion)**

```
Input:  G(·), y, Γ_y, prior 𝒩(m₀, C₀), J, N_iter, scheduler
Init:   Draw θ⁽ʲ⁾₀ ~ 𝒩(m₀, C₀)  for j = 1, …, J

for n = 0, …, N_iter−1:
    Evaluate:  g⁽ʲ⁾ = G(θ_n⁽ʲ⁾)       for all j     (parallel)
    Compute:   θ̄, Ḡ, C^{θG}, C^{GG}   from ensemble
    Schedule:  Δt_n = scheduler(n, C^{GG}, Γ_y, …)
    Update:    θ_{n+1}⁽ʲ⁾ = θ_n⁽ʲ⁾ + Δt_n · C^{θG}(Γ_y + Δt_n · C^{GG})⁻¹ (y − g⁽ʲ⁾)

return θ̄_{N_iter}, C^{θθ}_{N_iter}
```

**Algorithm: EKS / ALDI (Ensemble Kalman Sampler)**

```
Input:  G(·), y, Γ_y, prior 𝒩(m₀, C₀), J, N_iter
Init:   Draw θ⁽ʲ⁾₀ ~ 𝒩(m₀, C₀)  for j = 1, …, J

for n = 0, …, N_iter−1:
    Evaluate:  g⁽ʲ⁾ = G(θ_n⁽ʲ⁾)
    Compute:   C^{θθ}_n from ensemble
    Schedule:  Δt_n (adaptive, e.g., EKSStableScheduler)

    for j = 1, …, J:
        drift  = −(Δt_n/J) Σ_k ⟨g⁽ᵏ⁾−Ḡ, Γ_y⁻¹(g⁽ʲ⁾−y)⟩ θ⁽ᵏ⁾
                 + ((d+1)/J)(θ⁽ʲ⁾ − θ̄)
        prior  = Δt_n · C^{θθ} Γ_θ⁻¹                        (implicit)
        noise  = √(2Δt_n · C^{θθ}) ξ⁽ʲ⁾,     ξ ~ 𝒩(0, I)
        θ*     = solve((I + prior), θ⁽ʲ⁾ + drift)
        θ_{n+1}⁽ʲ⁾ = θ* + noise

return ensemble {θ_{N_iter}⁽ʲ⁾}     (samples from approximate posterior)
```

**Algorithm: UKI (Unscented Kalman Inversion)**

```
Input:  G(·), y, Γ_y, prior 𝒩(m₀, C₀), α, Σ_ω, Σ_ν, N_iter
Init:   m₀ = prior mean, C₀ = prior covariance

for n = 0, …, N_iter−1:
    Predict:   m̂ = r + α(m_n − r),   Ĉ = α² C_n + Σ_ω
    Sigma pts: θ̂ʲ = m̂ ± c [√Ĉ]_j       j = 1, …, p
    Evaluate:  ŷʲ = G(θ̂ʲ)               (2p+1 evaluations)
    Analysis:  C^{θy}, C^{yy} from sigma-point statistics
               m_{n+1} = m̂ + C^{θy} (C^{yy})⁻¹ (y − ŷ⁰)
               C_{n+1} = Ĉ − C^{θy} (C^{yy})⁻¹ (C^{θy})ᵀ

return m_{N_iter}, C_{N_iter}    (posterior mean and covariance)
```

---

## 4. JAX API Design

### 4.1 Design Principles

1. **Not Optax.** The update protocol is fundamentally different: the caller
   provides forward-model evaluations G(Θ), not gradients. The API is honest
   about this rather than shoehorning into `update(grads, state)`.

2. **Functional and immutable.** All state is a frozen NamedTuple of arrays.
   `update` returns a new state, never mutates.

3. **vmap-native.** Forward model evaluation is expressed as
   `jax.vmap(forward_fn)(ensemble)`, leveraging JAX's natural batching.

4. **Process-polymorphic.** The `Process` object (EKI, EKS, UKI, GNKI, ...)
   configures the update rule. The outer loop is process-agnostic.

5. **Scheduler-composable.** Learning rate schedulers are separate objects
   that can be swapped independently of the process.

6. **scan-compatible.** The full iteration loop can be expressed as
   `jax.lax.scan` when the forward model is JAX-traceable.

### 4.2 Core Types

```python
from typing import NamedTuple, Protocol, Callable, Optional
import jax.numpy as jnp

# ── State ──────────────────────────────────────────────────────

class EKPState(NamedTuple):
    """Immutable state for all ensemble Kalman processes."""
    ensemble: jnp.ndarray       # (J, p) — particle positions
    g_ensemble: jnp.ndarray     # (J, d) — last forward-model evaluations
    observations: jnp.ndarray   # (d,) or (d_aug,)
    obs_noise_cov: jnp.ndarray  # (d, d) or (d_aug, d_aug)
    step: jnp.ndarray           # scalar, iteration counter
    algo_time: jnp.ndarray      # scalar, cumulative Σ Δt
    key: jnp.ndarray            # PRNG key (for EKS noise, EKI perturbations)

class UKIState(NamedTuple):
    """State for Unscented Kalman Inversion (parametric, not ensemble)."""
    mean: jnp.ndarray           # (p,)
    covariance: jnp.ndarray     # (p, p)
    observations: jnp.ndarray
    obs_noise_cov: jnp.ndarray
    step: jnp.ndarray
    key: jnp.ndarray

# ── Process protocol ──────────────────────────────────────────

class Process(Protocol):
    def init(
        self,
        key: jnp.ndarray,
        prior_mean: jnp.ndarray,
        prior_cov: jnp.ndarray,
        observations: jnp.ndarray,
        obs_noise_cov: jnp.ndarray,
    ) -> EKPState | UKIState:
        ...

    def update(
        self,
        state: EKPState | UKIState,
        g_ensemble: jnp.ndarray,
        scheduler: Scheduler,
    ) -> EKPState | UKIState:
        ...

# ── Scheduler protocol ────────────────────────────────────────

class Scheduler(Protocol):
    def get_dt(
        self,
        state: EKPState | UKIState,
        g_ensemble: jnp.ndarray,
    ) -> jnp.ndarray:
        ...
```

### 4.3 Process Constructors

```python
# ── Optimisation processes ─────────────────────────────────────

class Inversion:
    """Ensemble Kalman Inversion (EKI).

    Args:
        J:              ensemble size
        prior:          if provided, infinite-time (augmented state)
        localization:   optional tapering matrix or Localizer
    """
    def __init__(self, J=50, prior=None, localization=None): ...

class TransformInversion:
    """Ensemble Transform Kalman Inversion (ETKI).

    Output-scalable (O(d) not O(d³)). No localization support.
    """
    def __init__(self, J=50, prior=None): ...

class SparseInversion:
    """Sparsity-inducing EKI.  Adds L0/L1 penalisation."""
    def __init__(self, J=50, prior=None, penalty="L1", gamma=1.0): ...

# ── Inference / sampling processes ─────────────────────────────

class Sampler:
    """Ensemble Kalman Sampler (EKS / ALDI).

    Args:
        J:              ensemble size
        prior:          required (Gaussian prior for regularisation)
        variant:        "aldi" (default, finite-sample corrected) | "eks"
    """
    def __init__(self, J=50, prior=..., variant="aldi"): ...

class Unscented:
    """Unscented Kalman Inversion (UKI).

    Args:
        prior:          required
        quadrature:     "symmetric" (2p+1 points) | "simplex" (p+2)
        alpha:          mean-reversion parameter (default from theory)
    """
    def __init__(self, prior=..., quadrature="symmetric", alpha=None): ...

class TransformUnscented:
    """Output-scalable UKI via square-root filter."""
    def __init__(self, prior=..., quadrature="symmetric"): ...

class GaussNewtonInversion:
    """Gauss-Newton Kalman Inversion (GNKI)."""
    def __init__(self, J=50, prior=..., alpha=1.0): ...
```

### 4.4 Scheduler Constructors

```python
class FixedScheduler:
    """Constant Δt."""
    def __init__(self, dt=0.1): ...

class DataMisfitController:
    """Adaptive Δt controlling data-misfit decrease.

    For finite-time algorithms: terminates when algo_time ≥ terminate_at.
    """
    def __init__(self, terminate_at=1.0, on_terminate="stop"): ...

class EKSStableScheduler:
    """Adaptive Δt ∝ 1/‖C^{GG}‖ for EKS stability."""
    def __init__(self): ...
```

### 4.5 Public API Functions

```python
# ── Core loop functions ────────────────────────────────────────

def init(
    process: Process,
    key: jnp.ndarray,
    prior_mean: jnp.ndarray,      # (p,)
    prior_cov: jnp.ndarray,       # (p, p)
    observations: jnp.ndarray,    # (d,)
    obs_noise_cov: jnp.ndarray,   # (d, d)
) -> EKPState | UKIState:
    """Initialise the process state."""

def update(
    process: Process,
    state: EKPState | UKIState,
    g_ensemble: jnp.ndarray,       # (J, d) forward-model evaluations
    scheduler: Scheduler = FixedScheduler(0.1),
) -> EKPState | UKIState:
    """Perform one update step given forward-model evaluations."""

# ── Extraction functions ───────────────────────────────────────

def get_ensemble(state) -> jnp.ndarray:            # (J, p)
def get_mean(state) -> jnp.ndarray:                # (p,)
def get_covariance(state) -> jnp.ndarray:           # (p, p)
def get_g_ensemble(state) -> jnp.ndarray:           # (J, d)
def get_algo_time(state) -> jnp.ndarray:            # scalar
def is_terminated(state, scheduler) -> bool:         # early stopping check

# ── Constraint transforms ─────────────────────────────────────

def transform_constrained_to_unconstrained(ϕ, bounds) -> θ:
def transform_unconstrained_to_constrained(θ, bounds) -> ϕ:
```

### 4.6 Usage Example

```python
import jax
import jax.numpy as jnp
import jax.random as jr
import ekp   # our library

# ── Problem setup ──────────────────────────────────────────────
key = jr.PRNGKey(42)

prior_mean = jnp.zeros(5)
prior_cov = jnp.eye(5)
observations = jnp.array([1.0, 2.0, 3.0])
obs_noise_cov = 0.1 * jnp.eye(3)

def forward_model(theta):
    """Some expensive black-box simulator."""
    return theta[:3] ** 2 + 0.5 * theta[3:]  # toy example

# ── EKI (optimisation) ────────────────────────────────────────
process = ekp.Inversion(J=50)
scheduler = ekp.DataMisfitController(terminate_at=1.0)
state = ekp.init(process, key, prior_mean, prior_cov,
                 observations, obs_noise_cov)

for step in range(30):
    ensemble = ekp.get_ensemble(state)                  # (50, 5)
    g_ens = jax.vmap(forward_model)(ensemble)            # (50, 3)
    state = ekp.update(process, state, g_ens, scheduler)
    if ekp.is_terminated(state, scheduler):
        break

theta_map = ekp.get_mean(state)

# ── EKS (posterior sampling) ──────────────────────────────────
sampler = ekp.Sampler(J=100, prior=(prior_mean, prior_cov))
scheduler_eks = ekp.EKSStableScheduler()
state_eks = ekp.init(sampler, key, prior_mean, prior_cov,
                     observations, obs_noise_cov)

for step in range(200):
    ensemble = ekp.get_ensemble(state_eks)
    g_ens = jax.vmap(forward_model)(ensemble)
    state_eks = ekp.update(sampler, state_eks, g_ens, scheduler_eks)

posterior_mean = ekp.get_mean(state_eks)
posterior_cov = ekp.get_covariance(state_eks)

# ── UKI (parametric posterior) ────────────────────────────────
uki = ekp.Unscented(prior=(prior_mean, prior_cov))
state_uki = ekp.init(uki, key, prior_mean, prior_cov,
                     observations, obs_noise_cov)

for step in range(20):
    # UKI generates its own sigma points internally
    sigma_pts = ekp.get_ensemble(state_uki)              # (2p+1, p)
    g_sigma = jax.vmap(forward_model)(sigma_pts)          # (2p+1, d)
    state_uki = ekp.update(uki, state_uki, g_sigma)

m_post, C_post = ekp.get_mean(state_uki), ekp.get_covariance(state_uki)
```

### 4.7 scan-Compatible Loop

```python
def ekp_scan_step(carry, _):
    state, key = carry
    key, subkey = jr.split(key)
    ensemble = ekp.get_ensemble(state)
    g_ens = jax.vmap(forward_model)(ensemble)
    state = ekp.update(process, state, g_ens, scheduler)
    return (state, key), ekp.get_mean(state)

(final_state, _), trajectory = jax.lax.scan(
    ekp_scan_step, (state, key), None, length=30
)
```

---

## 5. Example Applications

### 5.1 Parameter Estimation (Black-Box Calibration)

**Setting:** Calibrate parameters ϕ of a climate model Ψ(ϕ) against
observational data y, where Ψ is a Fortran/C code with no adjoint.

**EKP framing:** EKI is the natural tool. The forward map G = H ∘ Ψ ∘ T⁻¹
maps unconstrained parameters θ to observables. The ensemble is evaluated
in parallel (one MPI rank per particle), and the Kalman update is a cheap
matrix operation.

```python
# G(θ) runs the climate model — NOT a JAX function
# Evaluated externally, results collected as numpy arrays

process = ekp.Inversion(J=50, prior=(m0, C0))
scheduler = ekp.DataMisfitController(terminate_at=1.0)

for step in range(20):
    ensemble = np.array(ekp.get_ensemble(state))  # to numpy for MPI
    g_ens = run_climate_model_parallel(ensemble)    # MPI scatter/gather
    state = ekp.update(process, state, jnp.array(g_ens), scheduler)
```

**What you gain:** 20 iterations × 50 ensemble members = 1000 model
evaluations total, compared to O(10⁵) for finite-difference gradients
or MCMC. No adjoint needed.

### 5.2 PDE-Constrained Inverse Problems

**Setting:** Estimate spatially-varying coefficients κ(x) in a PDE
(e.g., Darcy flow: −∇·(κ∇u) = f) from sparse observations of u.

**EKP framing:** The parameter is κ discretised on a grid (p ~ 10²−10⁴).
G(κ) solves the PDE and extracts observations. UKI or ETKI is appropriate
for moderate p; EKI with localization for large p.

```python
# Forward model: solve Darcy PDE, extract pressure at sensor locations
def forward_model(kappa_unconstrained):
    kappa = jnp.exp(kappa_unconstrained)  # positivity constraint
    u = solve_darcy(kappa, mesh, boundary_conditions)
    return u[sensor_indices]

process = ekp.TransformInversion(J=100, prior=(m0, C0))
# ETKI scales linearly in observation dimension d
```

**What you gain over BLR:** No need to differentiate through the PDE solver.
Works with legacy PDE codes.

### 5.3 Bayesian Inference with Uncertainty Quantification

**Setting:** Quantify posterior uncertainty on parameters of a nonlinear
model, not just a point estimate.

**EKP framing:** Use EKS/ALDI for posterior samples, or UKI for a
parametric (m, C) posterior.

```python
sampler = ekp.Sampler(J=200, prior=(m0, C0), variant="aldi")

for step in range(500):
    ensemble = ekp.get_ensemble(state)
    g_ens = jax.vmap(forward_model)(ensemble)
    state = ekp.update(sampler, state, g_ens, ekp.EKSStableScheduler())

# Posterior samples (Gaussian-like, non-collapsing)
posterior_samples = ekp.get_ensemble(state)       # (200, p)
posterior_mean = ekp.get_mean(state)
posterior_cov = ekp.get_covariance(state)

# 95% credible intervals
stds = jnp.sqrt(jnp.diag(posterior_cov))
ci_lower = posterior_mean - 1.96 * stds
ci_upper = posterior_mean + 1.96 * stds
```

### 5.4 Satellite Retrieval / Remote Sensing

**Setting:** Retrieve atmospheric state (temperature profile, gas
concentrations) from satellite radiance measurements using a radiative
transfer model (RTM) as the forward model.

**EKP framing:** The RTM is often a Fortran code with no adjoint. The
observation vector y is the measured spectrum (d ~ 10²−10⁴ channels),
and the state vector θ is the atmospheric profile (p ~ 10¹−10²).

```python
# RTM: atmospheric_state → simulated_radiances
process = ekp.TransformInversion(J=50, prior=(climatology_mean, climatology_cov))
scheduler = ekp.DataMisfitController(terminate_at=1.0)

# Observation noise: instrument noise + forward model error
obs_noise_cov = instrument_noise + representativity_error
```

**What you gain:** EKP naturally handles the high-dimensional observation
space (ETKI scales as O(d)), and the prior encodes climatological knowledge.

### 5.5 Neural Network Hyperparameter Tuning

**Setting:** Tune hyperparameters (learning rate, weight decay, dropout
rate, etc.) of a neural network where each evaluation = one full training
run.

**EKP framing:** Each "forward model evaluation" trains the network and
returns validation metrics. EKI with J ~ 20−50 and ~10 iterations =
200−500 training runs, competitive with Bayesian optimisation.

```python
def train_and_evaluate(hyperparams):
    lr, wd, dropout = transform(hyperparams)
    model = train_network(lr=lr, wd=wd, dropout=dropout, epochs=50)
    return jnp.array([val_loss(model), val_accuracy(model)])

process = ekp.Inversion(J=30, prior=(hp_mean, hp_cov))
```

### 5.6 Geophysical Data Assimilation

**Setting:** Sequential state estimation for weather/ocean models. At each
timestep, assimilate new observations into the model state.

**EKP framing:** This is the classical Ensemble Kalman Filter (EnKF) use
case. EKP's `Inversion()` with Δt=1 and one update per observation window
recovers the EnKF. The ensemble propagates through the dynamical model
between updates.

```python
# Sequential assimilation loop
state = ekp.init(ekp.Inversion(J=100), key, x0_mean, x0_cov, y0, R)

for t, y_t in enumerate(observations_over_time):
    # Propagate ensemble through dynamics
    ensemble = ekp.get_ensemble(state)
    ensemble_propagated = jax.vmap(dynamics_model)(ensemble)

    # Re-initialize with propagated ensemble and new observations
    state = state._replace(
        ensemble=ensemble_propagated,
        observations=y_t,
    )

    # Assimilate
    g_ens = jax.vmap(observation_operator)(ensemble_propagated)
    state = ekp.update(ekp.Inversion(J=100), state, g_ens,
                       ekp.FixedScheduler(dt=1.0))
```

### 5.7 Application Summary

```
┌──────────────────────────┬──────────────┬───────────────┬──────────────────┐
│ Application              │ Best Process │ Key Advantage │ Typical J × iter │
├──────────────────────────┼──────────────┼───────────────┼──────────────────┤
│ Black-box calibration    │ EKI / ETKI   │ No adjoint    │ 50 × 20 = 1000  │
│ PDE inverse problems     │ ETKI / UKI   │ Output scale  │ 100 × 15        │
│ Posterior UQ             │ EKS / UKI    │ Uncertainty   │ 200 × 500       │
│ Satellite retrieval      │ ETKI         │ High d_obs    │ 50 × 10         │
│ Hyperparameter tuning    │ EKI          │ Black-box     │ 30 × 10         │
│ Data assimilation        │ EKI (Δt=1)   │ Sequential    │ 100 × 1/window  │
│ Methane source inversion │ ETKI / EKS   │ Spatial prior │ 50 × 20         │
│ Extreme value parameters │ UKI          │ Parametric UQ │ (2p+1) × 20    │
└──────────────────────────┴──────────────┴───────────────┴──────────────────┘
```

---

## 6. Connection to the Bayesian Learning Rule

### 6.1 Shared Mathematical Foundations

Both EKP and BLR target the same posterior: p(θ|y) ∝ p(y|θ)p(θ). Both
maintain a Gaussian-ish approximation that evolves iteratively. And both
can be understood as natural-gradient descent on a variational objective
in different disguises.

### 6.2 Precise Correspondences

```
┌──────────────────────────┬──────────────────────────────────────────────┐
│ BLR Concept              │ EKP Equivalent                               │
├──────────────────────────┼──────────────────────────────────────────────┤
│ Variational family       │ Implicit ensemble distribution               │
│ q(θ) = 𝒩(m, Σ)         │ {θ⁽¹⁾, …, θ⁽ᴶ⁾} with empirical (θ̄, C^{θθ})│
│                          │                                              │
│ Natural params (η, Λ)   │ No explicit parametric state (except UKI)    │
│                          │                                              │
│ Learning rate ρ          │ Timestep Δt                                  │
│                          │                                              │
│ Prior λ₀                │ Augmented state (infinite-time) or init      │
│                          │                                              │
│ Gradient g = ∇_θ ℓ      │ Implicit via cross-covariance C^{θG}        │
│                          │ (derivative-free)                            │
│                          │                                              │
│ Hessian H = ∇²_θ ℓ     │ Implicit via output covariance C^{GG}       │
│                          │ (derivative-free)                            │
│                          │                                              │
│ Precision update         │ Ensemble collapse / spread                   │
│ s ← (1−ρ)s + ρ(s₀−H)  │ (implicit through Kalman update)             │
│                          │                                              │
│ Mean update              │ θ̄ ← θ̄ + K(y − Ḡ)                          │
│ m ← η/s                 │                                              │
│                          │                                              │
│ Posterior extraction     │ get_mean(state), get_covariance(state)       │
│ get_posterior(state)     │                                              │
│                          │                                              │
│ BLR-FullRank with ρ=1   │ Kalman filter (one observation)              │
│ and exact Hessian        │                                              │
│                          │                                              │
│ BLR-Diagonal GGN         │ ≈ EKI with diagonal localization            │
│ (Adam-like)              │                                              │
└──────────────────────────┴──────────────────────────────────────────────┘
```

### 6.3 Key Differences

```
┌─────────────────────┬─────────────────────────┬──────────────────────────┐
│ Dimension           │ BLR                     │ EKP                      │
├─────────────────────┼─────────────────────────┼──────────────────────────┤
│ Gradients required  │ Yes (∇ℓ, optionally ∇²ℓ)│ No (derivative-free)    │
│ Forward model evals │ 1 per step              │ J per step               │
│ Scaling in p        │ O(p) diagonal, O(p²)    │ O(Jp) + O(J²d) solve    │
│                     │ full-rank               │                          │
│ Scaling in d_obs    │ N/A (operates on loss)  │ O(d³) EKI, O(d) ETKI    │
│ Uncertainty repr.   │ Parametric (s or Λ)     │ Nonparametric (ensemble) │
│ Rank of covariance  │ Full (or diagonal)      │ rank-(J−1)              │
│ Best regime         │ Cheap differentiable    │ Expensive black-box      │
│                     │ models (neural nets)    │ simulators               │
│ Multimodality       │ Cannot capture          │ EKS can partially explore│
│ jit/scan friendly   │ Fully                   │ If forward model is JAX  │
└─────────────────────┴─────────────────────────┴──────────────────────────┘
```

### 6.4 When to Use Which

```
Can you differentiate through G(θ)?
├── YES
│   ├── p > 10⁵ (large neural net)?
│   │   └── BLR-Diagonal (= Adam with uncertainty)
│   ├── p < 10³ and want exact posterior covariance?
│   │   └── BLR-FullRank
│   └── p moderate, want to compare?
│       └── Both. BLR for speed, EKI/UKI for validation.
│
└── NO (black-box simulator)
    ├── p < 10² and want uncertainty?
    │   └── UKI (deterministic, parametric posterior)
    ├── p < 10² and want posterior samples?
    │   └── EKS / ALDI
    ├── p ~ 10²−10³, just want MAP?
    │   └── ETKI (output-scalable)
    └── p > 10³?
        └── EKI with localization, or rethink parameterisation
```

---

## 7. GaussX Integration

### 7.1 Overview

[GaussX](../gaussx/README.md) provides structured linear operators, operations, and recipes that filterX can use for ensemble covariance computation, Kalman gain, and matrix square roots. This replaces hand-rolled linear algebra with structure-exploiting dispatch.

### 7.2 Dependencies

| Package | Role |
|---|---|
| `gaussx` (>=0.1) | Structured operators, `solve`, `logdet`, `cholesky`, `sqrt`; ensemble recipes |
| `lineax` (>=0.1) | Base operator abstraction (transitive via gaussx) |

### 7.3 Integration Points

| filterX Concept | GaussX Backend | Notes |
|---|---|---|
| C^{θθ} (ensemble parameter covariance) | `gaussx.recipes.ensemble_covariance(ensemble)` | Low-rank: rank ≤ J−1 |
| C^{θG} (cross-covariance) | `gaussx.recipes.ensemble_cross_covariance(ensemble, g_ensemble)` | |
| Kalman gain K = C^{θG}(Γ_y + Δt·C^{GG})⁻¹ | `gaussx.recipes.kalman_gain(C_theta_g, obs_cov_op)` | Exploits LowRankUpdate structure |
| UKI covariance C_n (§3.3) | `gaussx.operators.LowRankUpdate` or dense operator | Depending on dimension |
| UKI sigma points via sqrt(C) | `gaussx.ops.sqrt(cov_op)` or `gaussx.ops.cholesky(cov_op)` | Structure-exploiting |
| (Γ_y + Δt·C^{GG})⁻¹ solve | `gaussx.ops.solve(obs_cov_op, rhs)` | Low-rank + diagonal structure |

### 7.4 Architecture

```
┌───────────────────────────────────────────┐
│  filterX — Ensemble Kalman Processes       │
│  Process protocol, schedulers, loops       │
├───────────────────────────────────────────┤
│  gaussx.recipes — Kalman filter/smoother   │
│  ensemble_covariance, kalman_gain          │
├───────────────────────────────────────────┤
│  gaussx.ops — solve, logdet, sqrt, chol   │
│  gaussx.operators — LowRankUpdate, etc.    │
├───────────────────────────────────────────┤
│  lineax — solvers    │  matfree — SLQ     │
└───────────────────────────────────────────┘
```

### 7.5 Examples

**EKI update with GaussX operators:**

```python
import gaussx.ops as gops
from gaussx.operators import low_rank_plus_diag
from gaussx.recipes import ensemble_covariance, ensemble_cross_covariance, kalman_gain

def eki_update(ensemble, g_ensemble, y, obs_noise_cov, dt):
    J, p = ensemble.shape

    # Ensemble covariances — low-rank (rank J-1) by construction
    C_theta_g = ensemble_cross_covariance(ensemble, g_ensemble)
    C_gg = ensemble_covariance(g_ensemble)

    # Observation covariance: Γ_y + Δt · C^{GG} as structured operator
    # (diagonal noise + low-rank ensemble contribution)
    obs_cov_op = low_rank_plus_diag(
        W=C_gg.factors,            # low-rank part
        d=jnp.diag(obs_noise_cov)  # diagonal noise
    )

    # Kalman gain via Woodbury solve — O(d r²) instead of O(d³)
    K = kalman_gain(C_theta_g, obs_cov_op)

    # Update: θ_new = θ + K (y - G(θ))
    innovation = y - g_ensemble  # (J, d)
    ensemble_new = ensemble + jax.vmap(lambda innov: K @ innov)(innovation)
    return ensemble_new
```

**UKI sigma points with GaussX:**

```python
import gaussx.ops as gops
import lineax as lx

def uki_sigma_points(mean, covariance, alpha=1.0):
    p = mean.shape[0]
    cov_op = lx.MatrixLinearOperator(covariance)

    # Cholesky via GaussX — dispatches to structure-optimal algorithm
    L = gops.cholesky(cov_op)
    L_dense = L.as_matrix()

    # 2p+1 sigma points
    sigma_pts = jnp.zeros((2 * p + 1, p))
    sigma_pts = sigma_pts.at[0].set(mean)
    for i in range(p):
        sigma_pts = sigma_pts.at[1 + i].set(mean + alpha * L_dense[:, i])
        sigma_pts = sigma_pts.at[1 + p + i].set(mean - alpha * L_dense[:, i])
    return sigma_pts
```

---

## 8. References

1. Iglesias, M., Law, K. & Stuart, A. (2013). "Ensemble Kalman methods for inverse problems." *Inverse Problems* 29(4).
2. Huang, D., Schneider, T. & Stuart, A. (2022). "Unscented Kalman Inversion." *J. Comput. Phys.* 464.
3. Garbuno-Inigo, A. et al. (2020). "Interacting Langevin Diffusions: Gradient Structure and Ensemble Kalman Sampler." *SIAM J. Appl. Dyn. Syst.* 19(1).
4. Garbuno-Inigo, A. et al. (2020). "Affine Invariant Interacting Langevin Dynamics for Bayesian Inference." *SIAM J. Appl. Math.* 80(6).
5. Calvello, E., Reich, S. & Stuart, A. (2022). "Ensemble Kalman Methods: A Mean Field Perspective." arXiv:2209.11371.
6. Chada, N., Stuart, A. & Tong, X. (2020). "Tikhonov Regularization within Ensemble Kalman Inversion." *SIAM J. Numer. Anal.*
7. Huang, B. et al. (2022). "Ensemble Transform Kalman Inversion." *Inverse Problems* 38(10).
8. Chada, N. et al. (2021). "Iterative Ensemble Kalman Methods: A Unified Perspective with Some New Variants." *Found. Data Sci.*
9. Bishop, C., Etherton, B. & Majumdar, S. (2001). "Adaptive Sampling with the Ensemble Transform Kalman Filter." *Mon. Wea. Rev.* 129.
10. Schneider, T., Stuart, A. & Wu, J.-L. (2020). "Learning Stochastic Closures Using Ensemble Kalman Inversion." *Trans. Math. Appl.*
11. Iglesias, M. & Yang, Y. (2021). "Adaptive Regularisation for Ensemble Kalman Inversion." *Inverse Problems* 37(2).
12. Khan, M.E. & Rue, H. (2023). "The Bayesian Learning Rule." *JMLR* 24(281).
13. CliMA, "EnsembleKalmanProcesses.jl." https://github.com/CliMA/EnsembleKalmanProcesses.jl

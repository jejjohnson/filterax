---
status: draft
version: 0.1.0
---

# ekalmX x Ensemble Kalman Processes as optax GradientTransformations

**Subject:** Wrapping EKI, EKS, and UKI as `optax.GradientTransformation` objects,
enabling composability with the optax ecosystem (schedules, clipping, logging)
while preserving the gradient-free ensemble semantics.

**Date:** 2026-04-03

---

## 1  Motivation

Ensemble Kalman Processes (EKP) — EKI, EKS, UKI — are iterative update rules
for derivative-free parameter estimation. Each method follows the same rhythm:
initialize state, evaluate the forward model, compute an update, repeat. This is
structurally identical to the `init -> update -> state` loop that optax codifies.

Wrapping EKP as `optax.GradientTransformation` gives us:

- **Composability with `optax.chain`.** Step-size schedules, gradient clipping,
  weight decay, and logging combinators all work out of the box.
- **Parallel to `optax_bayes`.** Bayesian linear regression (BLR) already fits
  the optax pattern — EKP is the gradient-free counterpart. Users get a uniform
  API for both gradient-based and ensemble-based updates.
- **Familiar API for JAX practitioners.** Anyone who has used optax can adopt EKP
  without learning a new loop structure.
- **Hybrid optimization.** Gradient-based and gradient-free steps can coexist in
  the same training loop, opening the door to warm-starting, ensemble refinement,
  and multi-fidelity strategies.

---

## 2  The optax Protocol

An `optax.GradientTransformation` is a `NamedTuple` of two functions:

```python
class GradientTransformation(NamedTuple):
    init:   Callable[[Params], OptState]
    update: Callable[[Updates, OptState, Optional[Params]], tuple[Updates, OptState]]
```

The contract:

| Argument | Standard meaning | EKP meaning |
|----------|-----------------|-------------|
| `params` | Current model parameters | Current parameter estimate (ensemble mean) |
| `grads` / `updates` (input) | Gradient of loss w.r.t. params | **Unused** — ensemble provides search directions |
| `updates` (output) | Additive change to apply to params | Delta to the ensemble mean: $\bar{\theta}^{(n+1)} - \bar{\theta}^{(n)}$ |
| `state` | Optimizer internal state | Ensemble members, algorithmic time, PRNG key, etc. |

The key insight: optax does not require `grads` to be actual gradients. Any
`PyTree` with the same structure as `params` is valid. For EKP, we pass `None`
(or a zero tree) and the update function ignores it entirely.

---

## 3  EKI as optax

Ensemble Kalman Inversion (Iglesias, Law & Stuart 2013) is a deterministic
ensemble method that drives the ensemble toward the MAP estimate.

### State

```python
class EKIOptaxState(eqx.Module):
    particles: Float[Array, "J N_p"]       # ensemble members
    algo_time: Scalar                       # accumulated algorithmic time
    step: int                               # iteration counter
    forward_fn: Callable                    # θ → G(θ), stored for internal eval
    obs: Float[Array, "N_d"]               # target observations
    noise_cov: AbstractLinearOperator       # Γ (observation noise covariance)
    scheduler: AbstractScheduler            # adaptive Δt controller
```

### init

```python
def init(params: PyTree) -> EKIOptaxState:
    # 1. Flatten params to a vector θ_0 ∈ R^{N_p}
    # 2. Generate J ensemble members: θ_j = θ_0 + ε_j,  ε_j ~ N(0, σ_init^2 I)
    # 3. Initialize algo_time = 0.0, step = 0
    ...
```

### update

```python
def update(
    grads: PyTree,              # ignored (gradient-free)
    state: EKIOptaxState,
    params: Optional[PyTree],   # current mean estimate
) -> tuple[PyTree, EKIOptaxState]:
    θ = state.particles                          # (J, N_p)
    Δt = state.scheduler.get_dt(state)

    # 1. Forward model evaluation
    G = jax.vmap(state.forward_fn)(θ)            # (J, N_d)

    # 2. Ensemble statistics
    θ_mean = ensemble_mean(θ)                    # (N_p,)
    G_mean = ensemble_mean(G)                    # (N_d,)
    C_θG = cross_covariance(θ, G)                # (N_p, N_d)
    C_GG = ensemble_covariance(G)                # (N_d, N_d)

    # 3. EKI update for each member
    #    δθ_j = Δt · C^{θG} (C^{GG} + Δt^{-1} Γ)^{-1} (y - G_j)
    S = C_GG + (1 / Δt) * state.noise_cov
    innovation = state.obs - G                    # (J, N_d)
    δθ = Δt * innovation @ solve(S, C_θG.T).T    # (J, N_p)

    # 4. Update ensemble
    θ_new = θ + δθ
    new_mean = ensemble_mean(θ_new)

    # 5. Return mean update (for optax.apply_updates) and new state
    mean_update = new_mean - θ_mean               # unflatten to params tree
    new_state = EKIOptaxState(
        particles=θ_new,
        algo_time=state.algo_time + Δt,
        step=state.step + 1,
        ...
    )
    return unflatten(mean_update), new_state
```

The "updates" returned are the change to the ensemble mean. When the caller runs
`params = optax.apply_updates(params, updates)`, the `params` tree tracks the
mean parameter — the full ensemble lives inside `state`.

---

## 4  EKS as optax

Ensemble Kalman Sampler (Garbuno-Inigo et al. 2020) adds a Brownian noise term
so the ensemble samples the posterior rather than collapsing to a point.

### Differences from EKI

| Aspect | EKI | EKS |
|--------|-----|-----|
| Noise term | None | $\sqrt{2\Delta t}\,C^{1/2}_{\theta\theta}\,\xi_j$, $\xi_j \sim \mathcal{N}(0,I)$ |
| Ensemble collapse | Yes (converges to MAP) | No (samples the posterior) |
| State extras | — | PRNG key for noise generation |
| Convergence | $t_{\text{algo}} \to 1$ | Ergodic average |

### State

```python
class EKSOptaxState(eqx.Module):
    particles: Float[Array, "J N_p"]
    algo_time: Scalar
    step: int
    key: PRNGKeyArray                       # for Brownian noise
    forward_fn: Callable
    obs: Float[Array, "N_d"]
    noise_cov: AbstractLinearOperator
    scheduler: AbstractScheduler
```

### update

The EKS update follows the same structure as EKI but adds:

```python
# After computing EKI-style δθ:
C_θθ = ensemble_covariance(θ)                     # (N_p, N_p)
key, subkey = jax.random.split(state.key)
ξ = jax.random.normal(subkey, shape=(J, N_p))     # (J, N_p)
noise = jnp.sqrt(2 * Δt) * ξ @ cholesky(C_θθ).T  # (J, N_p)
θ_new = θ + δθ + noise
```

### The "single update" problem

Because the ensemble does not collapse, there is no single best parameter
estimate. The solution: return the change in the posterior mean as the `updates`
output (same as EKI), but the full ensemble for sampling is available in `state`.
The caller can either:

1. Use `params` as a point estimate (the evolving mean), or
2. Ignore `params` and extract `state.particles` for posterior samples.

---

## 5  UKI as optax

Unscented Kalman Inversion (Huang, Schneider & Stuart 2022) is parametric: it
maintains a mean and covariance rather than an explicit ensemble. Sigma points
are generated deterministically at each step.

### State

```python
class UKIOptaxState(eqx.Module):
    mean: Float[Array, "N_p"]               # current parameter mean
    covariance: AbstractLinearOperator       # current parameter covariance
    algo_time: Scalar
    step: int
    forward_fn: Callable
    obs: Float[Array, "N_d"]
    noise_cov: AbstractLinearOperator
    scheduler: AbstractScheduler
```

### update

```python
def update(
    grads: PyTree,
    state: UKIOptaxState,
    params: Optional[PyTree],
) -> tuple[PyTree, UKIOptaxState]:
    Δt = state.scheduler.get_dt(state)

    # 1. Generate 2N_p + 1 sigma points from (mean, covariance)
    sigma_pts = unscented_sigma_points(state.mean, state.covariance)  # (2N_p+1, N_p)

    # 2. Evaluate forward model on sigma points
    G = jax.vmap(state.forward_fn)(sigma_pts)                         # (2N_p+1, N_d)

    # 3. Compute predicted statistics (unscented transform)
    G_mean = weighted_mean(G, weights)                                # (N_d,)
    C_θG = weighted_cross_covariance(sigma_pts, G, weights)           # (N_p, N_d)
    C_GG = weighted_covariance(G, weights)                            # (N_d, N_d)

    # 4. Kalman update
    S = C_GG + (1 / Δt) * state.noise_cov
    K = C_θG @ solve(S, eye(N_d))                                    # (N_p, N_d)
    new_mean = state.mean + Δt * K @ (state.obs - G_mean)
    new_cov = state.covariance - Δt * K @ C_GG @ K.T

    # 5. Return delta_mean as update
    delta_mean = new_mean - state.mean
    new_state = UKIOptaxState(
        mean=new_mean,
        covariance=new_cov,
        algo_time=state.algo_time + Δt,
        step=state.step + 1,
        ...
    )
    return unflatten(delta_mean), new_state
```

UKI requires $2N_p + 1$ forward evaluations per step (vs. $J$ for EKI/EKS),
making it more expensive per step but potentially faster to converge for
moderate-dimensional problems.

---

## 6  Composability Patterns

### Basic: EKI + schedule + clipping

```python
optimizer = optax.chain(
    ekalmx.optax.eki(
        forward_fn=simulator,
        obs=observations,
        noise_cov=R,
        n_ensemble=100,
    ),
    optax.clip_by_global_norm(1.0),
)

state = optimizer.init(params)
for step in range(n_steps):
    updates, state = optimizer.update(grads=None, state=state, params=params)
    params = optax.apply_updates(params, updates)
```

### EKS for posterior sampling

```python
eks_transform = ekalmx.optax.eks(
    forward_fn=simulator,
    obs=observations,
    noise_cov=R,
    n_ensemble=200,
    key=jax.random.PRNGKey(0),
)

state = eks_transform.init(params)
samples = []
for step in range(n_steps):
    updates, state = eks_transform.update(grads=None, state=state, params=params)
    params = optax.apply_updates(params, updates)
    if step > burnin:
        samples.append(state.particles)
```

### UKI for moderate dimensions

```python
optimizer = ekalmx.optax.uki(
    forward_fn=simulator,
    obs=observations,
    noise_cov=R,
    init_cov=1.0,   # scalar → σ^2 I
)
```

### Hybrid: gradient-based warm-start, then EKI refinement

```python
# Phase 1: Adam for fast initial convergence (requires gradients)
phase1 = optax.adam(1e-3)

# Phase 2: EKI for gradient-free refinement
phase2 = ekalmx.optax.eki(forward_fn=simulator, obs=obs, noise_cov=R)

# Run phase 1
state1 = phase1.init(params)
for step in range(1000):
    grads = jax.grad(loss_fn)(params)
    updates, state1 = phase1.update(grads, state1, params)
    params = optax.apply_updates(params, updates)

# Switch to phase 2
state2 = phase2.init(params)
for step in range(100):
    updates, state2 = phase2.update(grads=None, state=state2, params=params)
    params = optax.apply_updates(params, updates)
```

---

## 7  Design Challenges

### 7.1  Forward model evaluation: who calls it?

optax's `update` contract passes `(grads, state, params)`. There is no slot for
a forward model or its evaluations. Three options:

| Option | Mechanism | Pros | Cons |
|--------|-----------|------|------|
| **(a) Store `forward_fn` in state** | `state.forward_fn` evaluated inside `update` | Clean API, self-contained | Impure state, `forward_fn` must be pytree-compatible |
| **(b) Caller passes via `extra_args`** | `optax.inject_hyperparams` or custom wrapper | Pure, explicit | Breaks `optax.chain` composability |
| **(c) Closure capture** | `forward_fn` captured by the `update` closure at construction time | Clean, no state pollution | Standard optax pattern |

**Recommendation: (c).** The `eki(forward_fn=...)` constructor captures
`forward_fn` in the closure that becomes the `update` function. The function
is never stored in state and never needs to be a pytree leaf. This is how
`optax.inject_hyperparams` works internally.

### 7.2  Ensemble as state vs. params

The ensemble lives in `OptState`, not in the `params` tree. This means:

- `optax.apply_updates` operates on the ensemble **mean**, which is a
  "representative" parameter with the same tree structure as the original params.
- The full ensemble is only accessible via `state.particles` (or `state.mean` /
  `state.covariance` for UKI).
- Downstream optax transforms (clipping, weight decay) act on the mean update,
  not on individual ensemble members.

This is a deliberate choice: the ensemble is an internal implementation detail.
The optax interface presents EKP as producing a single parameter update per step.

### 7.3  Convergence detection

EKI naturally converges as the ensemble collapses ($t_{\text{algo}} \to 1$).
But optax assumes an infinite loop — there is no built-in convergence signal.

Solutions:

- **Check `state.algo_time >= 1.0`** in the outer loop and break.
- **Adaptive scheduler** (e.g., `DataMisfitController`) shrinks $\Delta t$ as
  the data misfit decreases, naturally approaching $t = 1$.
- **EKS does not converge** — it is a sampler, so the infinite loop is correct.

### 7.4  PyTree flattening

optax operates on arbitrary PyTrees (nested dicts, NamedTuples, etc.). The
ensemble is stored as a flat `(J, N_p)` array. The `init` function must flatten
the params tree, and `update` must unflatten the mean update back to the
original tree structure. Use `jax.flatten_util.ravel_pytree` for this.

### 7.5  JIT compatibility

The `forward_fn` captured in the closure must be JIT-compatible. If it contains
non-JAX operations (e.g., calling an external simulator), the user must wrap it
in `jax.pure_callback` or run the loop outside JIT.

---

## 8  Complexity

| Method | Forward evals / step | State size | Per-step cost | Converges to |
|--------|---------------------|------------|---------------|--------------|
| EKI | $J$ | $O(J \cdot N_p)$ | $O(J \cdot N_d^2 + J \cdot N_p \cdot N_d)$ | MAP estimate |
| EKS | $J$ | $O(J \cdot N_p)$ + PRNG key | $O(J \cdot N_d^2 + J \cdot N_p^2)$ | Posterior samples |
| UKI | $2N_p + 1$ | $O(N_p^2)$ (mean + cov) | $O(N_p^2 \cdot N_d + N_d^3)$ | MAP + covariance |

| Regime | Recommended method |
|--------|-------------------|
| $N_p$ small ($< 50$), cheap forward model | UKI — deterministic, fast convergence |
| $N_p$ moderate, expensive forward model | EKI — $J \ll 2N_p+1$ members suffice |
| Posterior uncertainty needed | EKS — ensemble samples the posterior |
| $N_p$ large ($> 10^3$) | EKI with subspace projection or localization |

---

## 9  API

All three constructors live in `ekalmx.optax` and return `optax.GradientTransformation`.

### `ekalmx.optax.eki`

```python
def eki(
    forward_fn: Callable[[Float[Array, "N_p"]], Float[Array, "N_d"]],
    obs: Float[Array, "N_d"],
    noise_cov: AbstractLinearOperator,
    n_ensemble: int = 50,
    init_spread: float = 1.0,
    scheduler: AbstractScheduler = ConstantScheduler(dt=1.0),
    key: PRNGKeyArray = jax.random.PRNGKey(0),
) -> optax.GradientTransformation:
    """Ensemble Kalman Inversion as an optax GradientTransformation.

    Args:
        forward_fn: Forward model mapping parameters to observables.
            Captured in closure, not stored in state.
        obs: Target observations.
        noise_cov: Observation noise covariance Gamma.
        n_ensemble: Number of ensemble members J.
        init_spread: Standard deviation for initial ensemble perturbation.
        scheduler: Adaptive step-size controller.
        key: PRNG key for initial ensemble generation.

    Returns:
        optax.GradientTransformation with EKI update semantics.
        The `grads` argument to `update` is ignored.
    """
    ...
```

### `ekalmx.optax.eks`

```python
def eks(
    forward_fn: Callable[[Float[Array, "N_p"]], Float[Array, "N_d"]],
    obs: Float[Array, "N_d"],
    noise_cov: AbstractLinearOperator,
    n_ensemble: int = 100,
    init_spread: float = 1.0,
    scheduler: AbstractScheduler = ConstantScheduler(dt=0.1),
    key: PRNGKeyArray = jax.random.PRNGKey(0),
) -> optax.GradientTransformation:
    """Ensemble Kalman Sampler as an optax GradientTransformation.

    Adds Brownian noise so the ensemble samples the posterior rather than
    collapsing. State contains a PRNG key that advances each step.

    Args:
        forward_fn: Forward model mapping parameters to observables.
        obs: Target observations.
        noise_cov: Observation noise covariance Gamma.
        n_ensemble: Number of ensemble members J.
        init_spread: Standard deviation for initial ensemble perturbation.
        scheduler: Step-size controller (smaller Δt for stability).
        key: PRNG key for ensemble initialization and Brownian noise.

    Returns:
        optax.GradientTransformation with EKS update semantics.
        Access `state.particles` for posterior samples after burn-in.
    """
    ...
```

### `ekalmx.optax.uki`

```python
def uki(
    forward_fn: Callable[[Float[Array, "N_p"]], Float[Array, "N_d"]],
    obs: Float[Array, "N_d"],
    noise_cov: AbstractLinearOperator,
    init_cov: float | Float[Array, "N_p N_p"] = 1.0,
    scheduler: AbstractScheduler = ConstantScheduler(dt=1.0),
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> optax.GradientTransformation:
    """Unscented Kalman Inversion as an optax GradientTransformation.

    Parametric method: maintains mean + covariance instead of an ensemble.
    Generates 2*N_p + 1 sigma points at each step.

    Args:
        forward_fn: Forward model mapping parameters to observables.
        obs: Target observations.
        noise_cov: Observation noise covariance Gamma.
        init_cov: Initial parameter covariance. Scalar → σ^2 I.
        scheduler: Step-size controller.
        alpha: Sigma point spread parameter (UKF tuning).
        beta: Prior distribution parameter (2.0 for Gaussian).
        kappa: Secondary scaling parameter.

    Returns:
        optax.GradientTransformation with UKI update semantics.
        Access `state.covariance` for the posterior covariance estimate.
    """
    ...
```

---

## 10  References

1. Iglesias, M. A., Law, K. J. H. & Stuart, A. M. (2013). *Ensemble Kalman Methods for Inverse Problems.* Inverse Problems, 29(4), 045001.

2. Schillings, C. & Stuart, A. M. (2017). *Analysis of the Ensemble Kalman Filter for Inverse Problems.* SIAM J. Numer. Anal., 55(3), 1264--1290.

3. Garbuno-Inigo, A., Hoffmann, F., Li, W. & Stuart, A. M. (2020). *Interacting Langevin Diffusions: Gradient Structure and Ensemble Kalman Sampler.* SIAM J. Appl. Dyn. Syst., 19(1), 412--441.

4. Huang, D. Z., Schneider, T. & Stuart, A. M. (2022). *Unscented Kalman Inversion: Efficient Gaussian Approximation to the Posterior Distribution.* arXiv:2102.01580.

5. Kovachki, N. B. & Stuart, A. M. (2019). *Ensemble Kalman Inversion: A Derivative-Free Technique for Machine Learning Tasks.* Inverse Problems, 35(9), 095005.

6. DeepMind (2020). *optax: Composable Gradient Transformations.* https://github.com/google-deepmind/optax.

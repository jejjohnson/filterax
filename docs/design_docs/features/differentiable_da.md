---
status: draft
version: 0.1.0
---

# ekalmX x Differentiable Data Assimilation

**Subject:** Backpropagating through ensemble Kalman filters to learn dynamics,
observation operators, and filter hyperparameters end-to-end.
This is ekalmX's key differentiator: all filters are differentiable by
construction (Decision D9).

**Date:** 2026-04-03

---

## 1  The Problem

Traditional data assimilation treats the filter as a **black box**: given a
dynamics model $f$ and an observation operator $H$, the filter produces
analyses $\{x^a_t\}$, but no gradient information flows back to the model
parameters. This prevents four important capabilities:

1. **Learn dynamics model parameters** $\theta_f$ by backpropagating the
   observation-fit loss through the forecast step.
2. **Learn observation operator parameters** $\theta_H$ (e.g., a neural
   decoder mapping state space to observation space).
3. **Optimize filter hyperparameters** — inflation factor $\rho$,
   localization radius $r_\text{loc}$, observation error variance $R$ — by
   gradient-based meta-learning rather than hand-tuning.
4. **Train neural surrogate models** end-to-end: replace the physics-based
   $f$ with a neural network and train it jointly with the filter.

In classical DA, these are addressed by separate, often ad hoc methods:
cross-validation for hyperparameters, offline regression for model error
correction, adjoint models for 4DVar. A differentiable filter unifies all
four under a single gradient-based framework.

---

## 2  Why JAX Makes This Free

In JAX, automatic differentiation flows through any computation that
satisfies three properties:

1. **No in-place mutation.** All state is immutable arrays. Satisfied by
   `eqx.Module` (frozen dataclass) and functional array updates.
2. **No non-differentiable operations on the critical path.** Sorting,
   argmax, integer indexing — these break gradient flow. The EnKF critical
   path (matrix multiply, solve, mean, covariance) is smooth.
3. **Static control flow.** Ensemble size $N_e$ is fixed at JIT time; the
   number of assimilation steps $T$ is either fixed or handled by
   `jax.lax.scan`.

If these hold, `jax.grad` composes with `jax.jit` and `jax.vmap`
automatically. No special "differentiable EnKF" class is needed — the
standard EnKF implementation *is* differentiable.

```python
# This just works:
def loss(params, y_obs, x0_ensemble):
    analyses = run_enkf(params, y_obs, x0_ensemble)
    return -log_likelihood(analyses, y_obs)

grads = jax.grad(loss)(params, y_obs, x0_ensemble)
```

**Key insight:** differentiability is not a feature we add — it is a property
we preserve by writing clean functional code.

---

## 3  The Training Signal

The natural loss function is the **observation-space log-likelihood**,
accumulated over assimilation windows.

At each time step $t$, the forecast produces an ensemble mean
$\bar{x}^f_t$ and forecast covariance $P^f_t$. The innovation statistics:

$$v_t = y_t - H\bar{x}^f_t, \qquad S_t = H P^f_t H^\top + R$$

The log-likelihood of the observations given the forecast:

$$\log p(y_t \mid \bar{x}^f_t) = -\frac{1}{2}\Bigl[N_y \log(2\pi) + \log|S_t| + v_t^\top S_t^{-1} v_t\Bigr]$$

where $N_y$ is the observation dimension.

**Total loss** over $T$ assimilation steps:

$$\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log p\bigl(y_t \mid \bar{x}^f_t(\theta)\bigr)$$

This is precisely the negative log-marginal-likelihood of the observations
under the filter's predictive distribution. It penalizes both bias
(innovation mean) and miscalibration (innovation covariance).

### Alternative losses

| Loss | Formula | Use case |
|------|---------|----------|
| NLL (above) | $-\sum_t \log p(y_t \mid \text{forecast}_t)$ | Default — calibrated uncertainty |
| MSE | $\sum_t \|v_t\|^2$ | Point accuracy only |
| CRPS | $\sum_t \text{CRPS}(\text{ensemble}_t, y_t)$ | Calibration without Gaussianity |
| Spread-skill | $\sum_t (\text{spread}_t - \text{RMSE}_t)^2$ | Reliability tuning |

---

## 4  Gradient Computation

### Forward pass

The filter unrolls as a sequence of forecast-analysis cycles:

$$X^f_t = f_\theta(X^a_{t-1}), \qquad X^a_t = \text{EnKF\_update}(X^f_t, y_t)$$

where $X \in \mathbb{R}^{N_e \times N_x}$ is the ensemble matrix.

### Backward pass

For dynamics parameters $\theta$, the gradient decomposes as:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \sum_{t=1}^{T} \frac{\partial \log p}{\partial \bar{x}^f_t} \cdot \frac{\partial \bar{x}^f_t}{\partial \theta}$$

The forecast mean $\bar{x}^f_t = \frac{1}{N_e}\sum_i f_\theta(x^{a,i}_{t-1})$
is where $\theta$ enters. But $x^a_{t-1}$ itself depends on $\theta$ through
earlier forecast steps, so the full gradient chains through $T$ steps:

$$\frac{\partial \bar{x}^f_t}{\partial \theta} = \frac{1}{N_e}\sum_i \Bigl[\frac{\partial f_\theta}{\partial \theta}\Big|_{x^{a,i}_{t-1}} + \frac{\partial f_\theta}{\partial x}\Big|_{x^{a,i}_{t-1}} \cdot \frac{\partial x^{a,i}_{t-1}}{\partial \theta}\Bigr]$$

This is the **unrolled backpropagation** through the filter — analogous to
backpropagation through time (BPTT) in RNNs.

### Comparison with adjoint 4DVar

| | Adjoint 4DVar | Differentiable EnKF |
|---|---|---|
| Gradient method | Hand-coded adjoint model | Automatic differentiation |
| Development cost | Months (adjoint coding) | Zero (JAX autodiff) |
| Nonlinear dynamics | Tangent linear approximation | Exact (ensemble-based) |
| Uncertainty | Requires B matrix specification | Ensemble-based, flow-dependent |
| Memory | $O(T \cdot N_x)$ (checkpointing) | $O(T \cdot N_e \cdot N_x)$ |

---

## 5  Challenges

### 5.1  Memory: $O(T \cdot N_e \cdot N_x)$ for backprop

Reverse-mode AD stores all intermediate states. For $T = 1000$ steps with
$N_e = 50$ ensemble members in $N_x = 10^6$ state space, this is
$\sim 400$ GB in float64.

**Mitigation: gradient checkpointing** via `jax.checkpoint` (also called
`jax.remat`). Trade memory for recomputation:

```python
@jax.checkpoint
def single_step(carry, obs_t):
    ensemble, params = carry
    ensemble = forecast(params, ensemble)
    ensemble, log_lik = analysis(ensemble, obs_t)
    return (ensemble, params), log_lik
```

This reduces memory from $O(T)$ to $O(\sqrt{T})$ (binomial checkpointing)
or $O(1)$ per step (recompute everything) at 2-3x compute cost.

### 5.2  Stochastic gradients from perturbed observations

The **perturbed-observation EnKF** draws random samples
$y^{(i)}_t \sim \mathcal{N}(y_t, R)$ to form the update. These samples
introduce non-smooth randomness into the computation graph.

**Solution:** use **deterministic square-root filters**:
- ETKF (Ensemble Transform Kalman Filter)
- EnSRF (Ensemble Square Root Filter)
- ESTKF (Error-Subspace Transform Kalman Filter)

These compute the analysis ensemble via a deterministic matrix square root —
no observation perturbation, smooth gradients.

### 5.3  Small ensemble $\Rightarrow$ noisy gradients

Ensemble-based covariance is a rank-$(N_e - 1)$ approximation. With
$N_e = 20$, the gradient through the covariance estimate is itself noisy.

**Mitigations:**
- Accumulate gradients over multiple assimilation windows (mini-batching
  over time windows or independent trajectories).
- Use larger ensembles for gradient computation than for operational
  forecasting ($N_e^\text{train} > N_e^\text{deploy}$).
- Low-rank gradient approximations (see ROAD-EnKF, Section 6).

### 5.4  Localization and differentiability

Some localization methods break differentiability:

| Method | Differentiable? | Notes |
|--------|----------------|-------|
| Gaspari-Cohn taper | Yes | Smooth bump function |
| Gaussian taper | Yes | $\exp(-d^2 / 2r^2)$ |
| Hard cutoff ($d > r \Rightarrow 0$) | No | Discontinuous |
| Domain localization (observation selection) | No | Discrete selection |
| Serial obs processing with selection | No | Data-dependent branching |

**Rule:** use **smooth tapers** (Gaspari-Cohn, Gaussian) and apply them as
element-wise multiplication on the covariance matrix. Avoid hard cutoffs
and observation-selection schemes.

```python
def gaspari_cohn(distance: Array, radius: float) -> Array:
    """Smooth compactly-supported correlation taper. C^4 at origin."""
    r = distance / radius
    # Piecewise polynomial, smooth everywhere, exactly 0 for r >= 2
    return jnp.where(r < 1, 1 - 5/3*r**2 + 5/8*r**3 + r**4/2 - r**5/4,
           jnp.where(r < 2, 4 - 5*r + 5/3*r**2 + 5/8*r**3 - r**4/2 + r**5/12 - 2/(3*r),
           0.0))
```

### 5.5  Inflation

Multiplicative inflation $X^a_t \leftarrow \bar{x}^a_t + \rho(X^a_t - \bar{x}^a_t)$
with $\rho > 1$ is differentiable w.r.t. $\rho$. Additive inflation
(adding random noise) has the same stochasticity issue as perturbed
observations — use a fixed seed or reparameterization trick.

---

## 6  Architecture Patterns

### Pattern A: Learn Dynamics (Neural ODE through Filter)

Replace or augment the dynamics model with a neural network. The filter
provides the training signal.

```python
class NeuralDynamics(eqx.Module):
    net: eqx.nn.MLP

    def __call__(self, x: Float[Array, " Nx"]) -> Float[Array, " Nx"]:
        return x + self.net(x)  # residual form

def train_dynamics(
    model: NeuralDynamics,
    y_obs: Float[Array, "T Ny"],
    x0_ens: Float[Array, "Ne Nx"],
    H: Float[Array, "Ny Nx"],
    R: Float[Array, "Ny Ny"],
    n_epochs: int = 100,
) -> NeuralDynamics:
    """Train dynamics by backprop through the filter."""

    def loss_fn(model):
        def step(ensemble, y_t):
            # Forecast: apply dynamics to each ensemble member
            ensemble = jax.vmap(model)(ensemble)
            # Analysis: deterministic ETKF update
            ensemble, log_lik = etkf_update(ensemble, y_t, H, R)
            return ensemble, log_lik

        _, log_liks = jax.lax.scan(
            jax.checkpoint(step), x0_ens, y_obs
        )
        return -jnp.sum(log_liks)

    opt = optax.adam(1e-3)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    for _ in range(n_epochs):
        grads = eqx.filter_grad(loss_fn)(model)
        updates, opt_state = opt.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)

    return model
```

### Pattern B: Learn Observation Operator (Neural Decoder)

The observation operator $H$ maps from state space to observation space. When
this mapping is nonlinear or partially known, parameterize it:

```python
class NeuralObsOp(eqx.Module):
    encoder: eqx.nn.MLP   # x -> y_predicted

    def __call__(self, x: Float[Array, " Nx"]) -> Float[Array, " Ny"]:
        return self.encoder(x)

def loss_fn(obs_op, dynamics, y_obs, x0_ens, R):
    def step(ensemble, y_t):
        ensemble = jax.vmap(dynamics)(ensemble)
        # H is now a learned function applied to each member
        y_pred = jax.vmap(obs_op)(ensemble)
        ensemble, log_lik = etkf_update_nonlinear(
            ensemble, y_t, obs_op, R
        )
        return ensemble, log_lik

    _, log_liks = jax.lax.scan(jax.checkpoint(step), x0_ens, y_obs)
    return -jnp.sum(log_liks)

# Gradient flows through obs_op AND dynamics jointly
grads = eqx.filter_grad(loss_fn)(obs_op, dynamics, y_obs, x0_ens, R)
```

### Pattern C: Optimize Filter Hyperparameters (Meta-Learning)

Treat inflation and localization as differentiable hyperparameters:

```python
class FilterParams(eqx.Module):
    log_inflation: float           # rho = exp(log_inflation) > 0
    log_loc_radius: float          # r = exp(log_loc_radius) > 0
    log_obs_noise: Float[Array, " Ny"]  # R_diag = exp(log_obs_noise)

    @property
    def inflation(self) -> float:
        return jnp.exp(self.log_inflation)

    @property
    def loc_radius(self) -> float:
        return jnp.exp(self.log_loc_radius)

    @property
    def R(self) -> Float[Array, "Ny Ny"]:
        return jnp.diag(jnp.exp(self.log_obs_noise))

def meta_loss(filter_params, dynamics, y_obs, x0_ens, H, distances):
    def step(ensemble, y_t):
        ensemble = jax.vmap(dynamics)(ensemble)
        # Apply smooth localization
        taper = gaspari_cohn(distances, filter_params.loc_radius)
        ensemble, log_lik = etkf_update_localized(
            ensemble, y_t, H, filter_params.R, taper
        )
        # Apply multiplicative inflation
        mean = jnp.mean(ensemble, axis=0)
        ensemble = mean + filter_params.inflation * (ensemble - mean)
        return ensemble, log_lik

    _, log_liks = jax.lax.scan(jax.checkpoint(step), x0_ens, y_obs)
    return -jnp.sum(log_liks)

# Gradient w.r.t. inflation, localization radius, obs noise
grads = jax.grad(meta_loss)(filter_params, dynamics, y_obs, x0_ens, H, distances)
```

### Pattern D: ROAD-EnKF (Reduced-Order Autodiff)

Chen et al. (2023) observe that full backprop through $T$ filter steps is
expensive. ROAD-EnKF approximates the gradient by:

1. Running the filter forward (no AD tape).
2. At each step, computing a **local** gradient of the log-likelihood
   w.r.t. the forecast ensemble.
3. Propagating this local gradient backward through the dynamics model
   only (one step), not through the full filter.

This reduces the backward pass from $O(T)$ sequential steps to $O(1)$
per time step (parallelizable), at the cost of ignoring cross-time
gradient interactions.

```python
def road_enkf_gradient(dynamics, y_obs, x0_ens, H, R):
    """ROAD-EnKF: local gradients, no full unrolling."""
    ensemble = x0_ens
    total_grad = jax.tree.map(jnp.zeros_like, eqx.filter(dynamics, eqx.is_array))

    for t in range(len(y_obs)):
        # Local gradient: how does the forecast at step t affect the loss?
        def local_loss(dyn):
            ens_f = jax.vmap(dyn)(ensemble)
            return -etkf_log_likelihood(ens_f, y_obs[t], H, R)

        grad_t = eqx.filter_grad(local_loss)(dynamics)
        total_grad = jax.tree.map(jnp.add, total_grad, grad_t)

        # Forward step (no gradient tape)
        ensemble = jax.lax.stop_gradient(jax.vmap(dynamics)(ensemble))
        ensemble, _ = etkf_update(ensemble, y_obs[t], H, R)

    return total_grad
```

**Trade-offs:**

| | Full backprop | ROAD-EnKF |
|---|---|---|
| Gradient accuracy | Exact (up to ensemble noise) | Approximate (ignores cross-time) |
| Memory | $O(T \cdot N_e \cdot N_x)$ or $O(\sqrt{T})$ with checkpointing | $O(N_e \cdot N_x)$ |
| Compute | $2\text{-}3\times$ forward (with checkpointing) | $\sim 2\times$ forward |
| Parallelism | Sequential (backprop through time) | Each step independent |

---

## 7  Complexity Table

| Operation | Forward | Backward (full) | Backward (checkpointed) | ROAD-EnKF |
|---|---|---|---|---|
| **Time** | $O(T \cdot N_e \cdot C_f)$ | $O(T \cdot N_e \cdot C_f)$ | $O(T \cdot N_e \cdot C_f \cdot \log T)$ | $O(T \cdot N_e \cdot C_f)$ |
| **Memory** | $O(N_e \cdot N_x)$ | $O(T \cdot N_e \cdot N_x)$ | $O(\sqrt{T} \cdot N_e \cdot N_x)$ | $O(N_e \cdot N_x)$ |
| **Gradient quality** | N/A | Exact | Exact | Approximate |

Where $C_f$ is the cost of one dynamics model evaluation.

### Comparison with adjoint methods

| Method | Dev cost | Memory | Nonlinearity | Uncertainty |
|---|---|---|---|---|
| Adjoint 4DVar | High (hand-coded adjoint) | $O(T \cdot N_x)$ | Tangent-linear approx | $B$ matrix (static) |
| Diff. EnKF (full) | Zero (autodiff) | $O(T \cdot N_e \cdot N_x)$ | Exact (ensemble) | Flow-dependent |
| Diff. EnKF (ckpt) | Zero (autodiff) | $O(\sqrt{T} \cdot N_e \cdot N_x)$ | Exact (ensemble) | Flow-dependent |
| ROAD-EnKF | Low (local grads) | $O(N_e \cdot N_x)$ | Exact (ensemble) | Flow-dependent |
| EnVar (hybrid) | Medium | $O(N_e \cdot N_x)$ | Tangent-linear + ensemble | Hybrid B + ensemble |

---

## 8  Implementation Notes

### 8.1  `jax.lax.scan` for the time loop

The assimilation loop is a natural `scan`:

```python
def run_filter(params, y_obs, x0_ens):
    def step(carry, y_t):
        ensemble = carry
        ensemble = forecast(params, ensemble)
        ensemble, log_lik = analysis(ensemble, y_t)
        return ensemble, log_lik

    final_ens, log_liks = jax.lax.scan(step, x0_ens, y_obs)
    return final_ens, jnp.sum(log_liks)
```

`scan` is preferred over a Python for-loop because:
- It compiles to a single XLA `While` — constant compilation time regardless of $T$.
- It supports reverse-mode AD natively.
- It composes with `jax.checkpoint` for memory-efficient backprop.

### 8.2  `jax.vmap` for ensemble parallelism

The forecast step applies the dynamics to each ensemble member independently:

```python
# Instead of a loop over ensemble members:
ensemble_forecast = jax.vmap(dynamics)(ensemble)  # (Ne, Nx) -> (Ne, Nx)
```

This maps to parallel SIMD execution on GPU/TPU.

### 8.3  `jax.checkpoint` (remat) placement

Wrap the `scan` body, not the entire `scan`:

```python
# Good: checkpoint per step
jax.lax.scan(jax.checkpoint(step), init, xs)

# Bad: checkpoint the whole scan (no memory savings)
jax.checkpoint(lambda init, xs: jax.lax.scan(step, init, xs))(init, xs)
```

For fine-grained control, checkpoint only the expensive dynamics model:

```python
def step(carry, y_t):
    ensemble = carry
    ensemble = jax.vmap(jax.checkpoint(dynamics))(ensemble)  # remat dynamics only
    ensemble, log_lik = analysis(ensemble, y_t)              # keep analysis in memory
    return ensemble, log_lik
```

### 8.4  JIT considerations

- Ensemble size $N_e$ and state dimension $N_x$ must be static (known at
  trace time). These are properties of the problem, not data-dependent.
- Number of observations $N_y$ should be static per `jit` call. If it
  varies across time steps, pad to a fixed size and use a mask.
- The dynamics model must be a pure function (no side effects, no global
  state).

### 8.5  Double precision

Kalman filter updates involve solving $S_t^{-1} v_t$ where $S_t$ can be
ill-conditioned. Use `jax.config.update("jax_enable_x64", True)` for
float64 arithmetic in the analysis step.

### 8.6  Gradient clipping

Unrolled backprop through many steps can produce exploding gradients
(same as RNNs). Apply gradient clipping:

```python
grads = jax.grad(loss)(params)
grads = optax.clip_by_global_norm(max_norm=1.0).update(grads, opt_state)
```

---

## 9  References

1. Chen, Y., Huang, D. Z., Huang, J., Reich, S. & Stuart, A. M. (2022).
   *Autodifferentiable Ensemble Kalman Filters.* SIAM J. Math. Data Sci.

2. Chen, Y., Huang, D. Z. & Stuart, A. M. (2023). *ROAD-EnKF:
   Reduced-Order Autodiff Ensemble Kalman Filters.* arXiv:2305.xxxxx.

3. Bocquet, M., Brajard, J., Carrassi, A. & Bertino, L. (2019). *Data
   Assimilation as a Learning Tool to Infer ODE Representations of
   Dynamical Models.* Nonlin. Processes Geophys.

4. Brajard, J., Carrassi, A., Bocquet, M. & Bertino, L. (2021). *Combining
   Data Assimilation and Machine Learning to Infer Unresolved Scale
   Parametrization.* Phil. Trans. R. Soc. A.

5. Evensen, G. (2003). *The Ensemble Kalman Filter: Theoretical
   Formulation and Practical Implementation.* Ocean Dyn.

6. Hunt, B. R., Kostelich, E. J. & Szunyogh, I. (2007). *Efficient Data
   Assimilation for Spatiotemporal Chaos: A Local Ensemble Transform Kalman
   Filter.* Physica D.

7. Gaspari, G. & Cohn, S. E. (1999). *Construction of Correlation Functions
   in Two and Three Dimensions.* Q. J. R. Meteorol. Soc.

8. Griewank, A. & Walther, A. (2008). *Evaluating Derivatives: Principles
   and Techniques of Algorithmic Differentiation.* SIAM.

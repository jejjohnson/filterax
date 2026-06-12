# Differentiable Assimilation

Every filterax analysis step is a pure JAX computation, so the entire
forecast–analysis cycle is differentiable by construction: `jax.grad`
of any scalar function of the assimilation output — with respect to
dynamics parameters, observation-operator parameters, inflation
factors, even the initial ensemble — just works. This chapter covers
*why* you would want that gradient, and the two strategies filterax
ships for computing it over long windows without exhausting memory.

## Why differentiate through a filter?

The filter consumes a model and produces, per window, the marginal
data likelihood $\log p(y_t \mid y_{1:t-1})$ — the Gaussian
innovation likelihood of chapter 13. Summed over a window, this is a
principled training signal:

$$
\mathcal{L}(\theta)
  = -\sum_{t=1}^{T} \log p\big(y_t \mid X^f_t(\theta)\big),
$$

where $\theta$ parameterises whatever you want to learn:

- **Parameter estimation** — physical constants of the forecast
  model, observation-operator gains, error-covariance amplitudes.
  Gradient descent on $\mathcal{L}$ replaces hand-tuning or
  state-augmentation tricks.
- **Learned forecast models** — train a neural surrogate (an
  `equinox` module satisfying `AbstractDynamics`) *through* the
  assimilation, so the model is optimised for exactly the metric the
  DA system cares about: predicting the next observation given the
  filtered state, rather than one-step state error in isolation.
- **Learned codecs** — latent DA (chapter 15) backpropagates the
  same loss into autoencoder weights.

## `lax.scan` and the checkpointing trade-off

The L2 wrappers (`filterax.ETKF.assimilate`, …) loop over windows in
Python. That is differentiable for short $T$, but it unrolls into one
giant traced graph — compile time grows with $T$ and it does not
compose with `jax.checkpoint`. `differentiable_assimilate` is the
drop-in replacement: observations and timestamps are stacked along a
leading time axis and the cycle becomes a fixed-shape `jax.lax.scan`,
giving a single fused XLA `While` whose trace is $O(1)$ in $T$.

Reverse-mode AD must store the forward residuals of every step, so
the backward tape costs $O(T \cdot N_e N_x)$ memory. Passing
`checkpoint=True` wraps the scan body in `jax.checkpoint`
(rematerialisation): forward intermediates are discarded and
recomputed during the backward pass, cutting memory to
$O(\sqrt{T} \cdot N_e N_x)$ at roughly 2–3× compute. It is a pure
memory/compute trade — the values and gradients are unchanged.

## Adjoint strategies

The gradient policy of the scan is selected with the ``adjoint=``
argument, using the vocabulary shared across pipekit, filterax, and
vardax (names mirror diffrax's adjoint classes):

| Strategy | Gradient | Backward memory | When |
|---|---|---|---|
| `DirectAdjoint` (default) | exact | $O(T)$ | short rollouts |
| `RecursiveCheckpointAdjoint` | exact | $O(\sqrt{T})$-ish (recompute) | long rollouts, exact gradients required |
| `TruncatedAdjoint(k)` | biased: no cross-cycle flow beyond $k$ (per-cycle outputs keep local gradients) | $O(1)$ in $T$ for final-state losses; bounded gradient depth otherwise | long *chaotic* rollouts; learned-forecast training |

`TruncatedAdjoint` runs every carry before the trailing-$k$ window
under `stop_gradient`, while per-cycle outputs keep their *local*
gradients — so a loss summed over the returned history reproduces the
ROAD-EnKF local-gradient estimator (each window contributes its own
term; no cross-window adjoint products form). This is not merely an
approximation: for
chaotic dynamics the exact adjoint norm grows like
$e^{\lambda_{\max} T \Delta t}$ (Lea et al. 2000), so truncation acts
as gradient *regularisation* — the same insight behind ROAD-EnKF
(below), of which `TruncatedAdjoint(k=1)` is the cycle-level
generalisation to any deterministic filter. Forward values are
identical under every strategy; only gradients differ. The
structurally identical `pipekit_cycle.adjoints` specs are accepted
interchangeably, and the same vocabulary selects diffrax adjoints for
the dynamics layer via `pipekit_jax.DiffraxForwardModel` (chapter 16).

## What is refused: stochastic components

Two ingredients break smooth gradients and are rejected with a
`ValueError` at call time rather than silently producing noisy
gradients:

- **Stochastic filters** — `StochasticEnKF` (perturbed observations,
  chapter 5) and `ETKF_Livings` (random mean-preserving rotations)
  inject per-step PRNG draws. Under `jax.grad` the draws are treated
  as constants, so the "gradient" is the gradient of one arbitrary
  noise realisation. Use the deterministic square-root family
  instead: `ETKF`, `EnSRF`, `ESTKF`, `EnSRF_Serial`, `LETKF`.
- **`AdditiveInflator`** — the one stochastic inflator (chapter 12),
  for the same reason. `MultiplicativeInflator` / `RTPS` / `RTPP`
  pass through cleanly.

## The ETKF `eigh` trap

A subtlety worth knowing because it shaped the implementation: the
textbook ETKF takes an eigendecomposition of the $(N_e, N_e)$
ensemble-space matrix $\tilde{C} = (N_e-1) I + Y' R^{-1} Y'^\top$.
When $N_y < N_e$ — the usual case — $\tilde{C}$ has $N_e - N_y$
eigenvalues *structurally repeated* at $N_e - 1$, and the
reverse-mode derivative of `jnp.linalg.eigh` divides by eigenvalue
gaps: repeated eigenvalues give **NaN gradients**. filterax's ETKF
instead extracts the rank-$N_y$ spectrum of $Y' R^{-1} Y'^\top$ via a
thin QR plus a small $(N_y, N_y)$ `eigh`, and applies all matrix
functions of $\tilde{C}$ in identity-plus-low-rank form — no
degenerate eigenvectors are ever differentiated. See
[chapter 6](06_etkf.md) for the full derivation; the practical
consequence here is simply that `jax.grad` through the ETKF is
NaN-free whenever $Y'$ has full column rank.

## ROAD-EnKF: local gradients

For very long horizons even $O(\sqrt{T})$ memory bites, and the
full-tape gradient may not be worth its cost. ROAD-EnKF (Chen,
Sanz-Alonso & Willett 2023) takes the **local-gradient** approach:

1. At step $t$, compute the per-step loss
   $L_t = -\log p(y_t \mid X^f_t)$ *with* gradient enabled, getting
   the local gradient $\nabla_\theta L_t$.
2. Sum the local gradients across steps:

   $$
   \nabla_\theta L \approx \sum_t \nabla_\theta L_t .
   $$

3. Advance the ensemble between steps through
   `jax.lax.stop_gradient`, so no autodiff tape spans more than one
   filter cycle.

Backward-pass memory stays $O(N_e N_x)$ — **independent of $T$**.
The price is bias: the cross-time terms
$\partial L_t / \partial X^a_{t-1} \cdot \partial X^a_{t-1} /
\partial \theta$ are dropped. In practice this is the accepted
ROAD-EnKF trade-off — a stable filter forgets its initial conditions,
so the dropped chains decay, and the memory savings dominate for long
$T$. At $T = 1$ there are no cross-time terms and ROAD's gradient
equals the full-tape gradient exactly (filterax pins this in
`tests/test_road.py`).

`road_enkf_loss_and_grad` returns `(loss, dynamics_grad)` where the
gradient is a PyTree matching the dynamics module (only inexact array
leaves carry gradients); `road_enkf_grad_step` composes it with any
`optax` optimizer for a one-line training step.

## Implementation in filterax

Mirroring `tests/test_differentiable.py`'s Pattern A — learn a scalar
dynamics parameter by descending the observation-space NLL:

```python
import jax
import jax.numpy as jnp
import lineax as lx

import filterax as flx
from filterax.differentiable import road_enkf_loss_and_grad


class LinearDynamics(flx.AbstractDynamics):
    M: jnp.ndarray

    def __call__(self, state, t0, t1):
        return self.M @ state


# Truth evolves with M* = 0.95 I; the model starts at M = I.
N_e, N_x, N_y, T = 20, 2, 2, 6
key = jax.random.key(0)
particles = jax.random.normal(key, (N_e, N_x)) + jnp.array([1.0, -0.5])

state = jnp.array([1.0, -0.5])
truth = []
for _ in range(T):
    state = 0.95 * state
    truth.append(state)
obs = jnp.stack(truth) + 0.05 * jax.random.normal(jax.random.key(1), (T, N_y))
times = jnp.arange(1.0, T + 1.0)
R = lx.DiagonalLinearOperator(jnp.full((N_y,), 0.05**2))


def nll(scale):
    dyn = LinearDynamics(M=scale * jnp.eye(N_x))
    result = flx.differentiable_assimilate(
        flx.filters.ETKF(), dyn, lambda x: x, particles, obs, times, R,
        checkpoint=True,
    )
    return -jnp.sum(result.log_likelihoods)


grad = jax.grad(nll)(1.0)
print("dNLL/dscale at M = I:", float(grad))   # positive: descend toward 0.95
print("NLL(1.00):", float(nll(1.0)))
print("NLL(0.95):", float(nll(0.95)))

# Same window through the ROAD-EnKF local-gradient path.
loss, grads = road_enkf_loss_and_grad(
    LinearDynamics(M=jnp.eye(N_x)), particles, obs, times, R, lambda x: x,
)
print("ROAD loss:", float(loss), " dM[0,0]:", float(grads.M[0, 0]))
```

```
dNLL/dscale at M = I: 350.47052001953125
NLL(1.00): -4.433971405029297
NLL(0.95): -13.869117736816406
ROAD loss: -4.433971405029297  dM[0,0]: 110.14859008789062
```

Both paths agree on the loss and on the *sign* of the gradient — the
NLL is lower at the true $M = 0.95\,I$, and a descent step from
$M = I$ moves toward it. The magnitudes differ because ROAD drops
the cross-time terms; for $T = 6$ the local gradient is already a
serviceable descent direction at a fraction of the memory.

## Where next

- [Chapter 6 — ETKF](06_etkf.md): the rank-$N_y$ QR + small-`eigh`
  spectrum that makes the transform gradient-safe.
- [Chapter 13 — Diagnostics & likelihood](13_diagnostics.md): the
  innovation log-likelihood used as the loss, and its structured
  $O(N_e^2 N_y + N_e^3)$ evaluation.
- [Chapter 15 — Latent-space ensemble DA](15_latent_da.md): the same
  gradients flowing into autoencoder weights.
- [Chapter 5 — Stochastic EnKF](05_stochastic_enkf.md): why
  perturbed-observation filters are excluded from training loops.
- [API: Differentiable training](../api/differentiable.md) —
  `differentiable_assimilate`, `road_enkf_loss_and_grad`,
  `road_enkf_grad_step`.

## References

- Chen, Y., Sanz-Alonso, D., & Willett, R. (2023). *Reduced-Order
  Autodifferentiable Ensemble Kalman Filters.* Inverse Problems,
  39(12), 124001. (ROAD-EnKF.)
- Griewank, A. & Walther, A. (2000). *Algorithm 799: revolve — an
  implementation of checkpointing for the reverse or adjoint mode of
  computational differentiation.* ACM TOMS, 26(1), 19–45. (The
  checkpointing memory/compute trade.)
- Bishop, C. H., Etherton, B. J., & Majumdar, S. J. (2001). *Adaptive
  sampling with the ensemble transform Kalman filter. Part I:
  Theoretical aspects.* Mon. Wea. Rev., 129, 420–436.

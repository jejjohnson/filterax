# Ensemble Kalman Processes — EKI, EKS, UKI

The filters of chapters 5-8 track a *state* through time. The same
ensemble-Kalman algebra also solves static **inverse problems**: given
one batch of data

$$
y = G(\theta) + \eta, \qquad \eta \sim \mathcal{N}(0, \Gamma),
$$

estimate the parameters $\theta \in \mathbb{R}^{N_p}$ of a possibly
expensive, possibly non-differentiable forward model
$G: \mathbb{R}^{N_p} \to \mathbb{R}^{N_d}$. Iterate an ensemble of
candidate parameters $\Theta \in \mathbb{R}^{J \times N_p}$ (rows are
members, exactly as the filter chapters use $X$; the ensemble size $J$
plays the role of $N_e$) and let the ensemble's own cross-covariances
play the role of derivatives. No gradients of $G$ are ever required —
which is the whole point.

## EKI — a derivative-free Gauss-Newton flow

Ensemble Kalman Inversion (Iglesias, Law & Stuart 2013) repeatedly
applies a tempered Kalman update to the parameter ensemble. With
forward evaluations $G^{(j)} = G(\theta^{(j)}_n)$, sample
cross-covariance $C^{\theta G}_n$ and obs-space covariance $C^{GG}_n$:

$$
\theta^{(j)}_{n+1} = \theta^{(j)}_n + \Delta t_n\, C^{\theta G}_n
    \big(C^{G G}_n + \Delta t_n^{-1} \Gamma\big)^{-1}
    \big(y - G(\theta^{(j)}_n)\big).
$$

Why "Gauss-Newton-like"? Linearise $G$ and the increment becomes
$C^{\theta\theta} \nabla G^\top (\cdots)^{-1} r$ — a least-squares
Gauss-Newton step in which the ensemble statistics stand in for the
Jacobian, preconditioned by the parameter covariance. The tempering
$\Delta t^{-1} \Gamma$ makes $n$ steps of size $\Delta t$ with
$\sum \Delta t_n = 1$ equivalent (in the linear-Gaussian limit) to
*one* full Kalman update from the prior ensemble — the artificial time
$t_n = \sum_{m \le n} \Delta t_m$ interpolates prior ($t = 0$) to
posterior ($t = 1$). Two properties to internalise:

- **Subspace property**: every iterate stays in the affine span of the
  initial ensemble; EKI works even when $J < N_p$ (underdetermined).
- **Ensemble collapse**: spread $\to 0$ as $t \to 1$. EKI is a point
  estimator. The final spread is *not* posterior uncertainty.

## EKS — Langevin sampling instead of collapse

The Ensemble Kalman Sampler in its ALDI form (Garbuno-Inigo et al.
2020) turns the flow into an interacting Langevin SDE whose stationary
distribution is the posterior $p(\theta \mid y)$:

$$
\mathrm{d}\theta^{(j)} =
    C^{\theta G} \Gamma^{-1} \big(y - G^{(j)}\big)\, \mathrm{d}t
    \;-\; \frac{N_p + 1}{J} \big(\theta^{(j)} - \bar{\theta}\big)\, \mathrm{d}t
    \;+\; \sqrt{2\, C^{\theta\theta}}\;\, \mathrm{d}W^{(j)} .
$$

The first term is the EKI drift; the $(N_p+1)/J$ correction removes
finite-sample bias; the ensemble-preconditioned noise
$\sqrt{2 C^{\theta\theta}}\,\mathrm{d}W$ keeps the spread alive — at
stationarity the members are approximate posterior samples. filterax
draws the noise in *ensemble space* using the factor
$\sqrt{C^{\theta\theta}} = \Theta'^\top / \sqrt{J-1}$, so no dense
$N_p \times N_p$ Cholesky is formed.

## UKI — sigma points instead of an ensemble

Unscented Kalman Inversion (Huang, Schneider & Stuart 2022) replaces
the Monte Carlo ensemble with a parametric Gaussian belief
$\theta \sim \mathcal{N}(\mu_n, \Sigma_n)$, probed by $2 N_p + 1$
deterministic sigma points (Wan & van der Merwe 2000):

$$
\chi^0 = \mu, \qquad
\chi^{\pm}_j = \mu \pm \sqrt{N_p + \lambda}\; [\sqrt{\Sigma}]_{:,j},
\qquad \lambda = \alpha^2 (N_p + \kappa) - N_p,
$$

with quadrature weights $W_m^i, W_c^i$. The forward model is evaluated
at every sigma point and the unscented statistics feed a tempered
Kalman update on $(\mu, \Sigma)$:

$$
\mu_{n+1} = \mu_n + \Delta t\, C^{\theta y} S_n^{-1} (y - \hat{y}_0),
\qquad
\Sigma_{n+1} = \Sigma_n - \Delta t\, C^{\theta y} S_n^{-1} C^{\theta y \top}.
$$

Deterministic, reproducible, no collapse, and $\Sigma_n$ is calibrated
in the linear-Gaussian case — at the price of $2 N_p + 1$ forward
evaluations per step, which caps it at moderate $N_p \lesssim 100$.

## Step-size schedulers

Every process consumes a $\Delta t_n$ from an `AbstractScheduler`:

- **`FixedScheduler(dt=...)`** — constant; you tune it. With
  $\Delta t = 1$, one EKI/UKI step *is* the full Kalman update.
- **`DataMisfitController`** (Iglesias 2016) — the standard adaptive
  rule. With the ensemble-mean Mahalanobis misfit
  $\Phi_n = J^{-1} \sum_j \lVert y - G(\theta^{(j)})\rVert^2_{\Gamma^{-1}}$,

$$
\Delta t_n = \min\big(\text{target\_misfit} / \Phi_n,\;
    1 - t_n\big),
$$

  small careful steps while far from the data, ramping up as the misfit
  falls, and clamped so the artificial time lands exactly on $t = 1$ —
  the standard stopping rule. Past convergence it returns
  $\Delta t = 0$ and further updates are no-ops, not NaNs.
- **`EKSStableScheduler`** — bounds $\Delta t \le
  \text{target}/\lVert C^{\theta\theta}\rVert$ (via a cheap Frobenius
  bound) so the EKS Langevin discretisation stays in its stability
  region.

## The variant zoo, briefly

| Process | One-line description |
|---|---|
| `processes.GNKI` | Explicit ensemble Jacobian $\tilde{J} = (C^{\theta\theta})^{-1} C^{\theta G}$ + Gauss-Newton step with prior pull-back; fast, needs $J > N_p$; recovers the exact linear-Gaussian posterior |
| `processes.TEKI` | Tikhonov EKI: augments $\tilde{G}(\theta) = (G(\theta), \theta)$, $\tilde{y} = (y, m_0)$, $\tilde{\Gamma} = \mathrm{blockdiag}(\Gamma, \Sigma_0)$ — vanilla EKI on the augmented system converges to the MAP instead of drifting on ill-posed problems |
| `processes.SparseInversion` | EKI step + L¹ soft-threshold prox $\mathrm{sign}(z)\max(|z| - \lambda, 0)$ — drives inactive parameters exactly to zero (variable selection) |
| `processes.ETKI` | Transform-flavoured EKI; currently shares EKI's Woodbury-routed solve |

## Three API layers

filterax exposes the processes at three altitudes, all built on the
same Layer-1 update rules:

1. **Layer 1** (`filterax.processes.*`): `init(particles, obs,
   noise_cov)` / `update(state, forward_evals)`. *You* evaluate
   $G$ each step — full control over batching, checkpointing, or HPC
   dispatch of the forward model.
2. **Layer 2** (`filterax.EKI`, `filterax.EKS`, `filterax.UKI`): owns
   the `init → vmap(G) → update` loop. `run(init_particles)` iterates
   until `n_iterations` or `algo_time ≥ 1` and returns a
   `ProcessResult` with mean, particles, and per-step history.
3. **optax** (`filterax.optax.eki/eks/uki`): each process as an
   `optax.GradientTransformation`. The ensemble lives in the opt
   state, `params` carries the running mean, and the `grads` argument
   is ignored — so derivative-free EKP steps compose with optax
   schedules, clipping, and gradient-based warm-starts.

## Implementation in filterax

```python
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

import filterax as flx

# Linear inverse problem: y = G theta + eta, theta_true = [1, -1].
G = jnp.array([[1.0, 0.5], [-0.3, 1.2], [0.7, 0.1]])
y = G @ jnp.array([1.0, -1.0])
Gamma = lx.DiagonalLinearOperator(0.1 * jnp.ones(3))
forward = lambda theta: G @ theta

init = 2.0 * jr.normal(jr.key(0), (100, 2))      # J = 100 prior draws

# Layer 2: forward model + process + scheduler in one .run().
# FixedScheduler(dt=1.0) makes the first step a full Kalman update,
# so algo_time hits 1 immediately and the loop stops.
eki = flx.EKI(
    forward_fn=forward, obs=y, noise_cov=Gamma,
    scheduler=flx.FixedScheduler(dt=1.0),
)
result = eki.run(init)
result.mean                       # [0.985, -0.985] — at the posterior mean
bool(result.converged)            # True (algo_time reached 1)
result.particles.std(axis=0)      # ~0.03 — collapsed: EKI is a point estimator

# Layer 1: explicit loop — you own the forward evaluations.
process = flx.processes.EKI(scheduler=flx.FixedScheduler(dt=1.0))
state = process.init(init, y, Gamma)
for _ in range(3):
    evals = jax.vmap(forward)(state.particles)
    state = process.update(state, evals)
state.particles.mean(axis=0)      # [0.998, -0.998]
```

And the optax flavour — the same EKI driven through a standard optax
update loop (with `grads=None`):

```python
import optax

transform = flx.optax.eki(
    forward_fn=forward, obs=y, noise_cov=Gamma,
    n_ensemble=200, init_spread=5.0,
    scheduler=flx.FixedScheduler(dt=1.0),
)
params = jnp.zeros(2)
state = transform.init(params)
for _ in range(3):
    updates, state = transform.update(None, state, params)
    params = optax.apply_updates(params, updates)
params                            # [0.996, -0.998]
```

For real problems with expensive nonlinear $G$, swap in
`flx.DataMisfitController()` and a larger iteration budget
(`flx.ProcessConfig(n_iterations=...)`); the controller takes many
small steps while the misfit is large and stops at the noise level.

## Choosing a process

Need posterior samples → **EKS**. Need a calibrated parametric
covariance and $N_p \lesssim 100$ → **UKI**. Sparse parameters →
**SparseInversion**. Ill-posed, drifting EKI → **TEKI**.
Well-conditioned with $J > N_p$ and a hurry → **GNKI**. Otherwise →
**EKI**, the simple robust default.

## Where next

- [Chapter 4 — Kalman update & ensemble gain](04_kalman_update.md):
  the cross-covariance algebra reused here.
- [Chapter 10 — Ensemble smoothers](10_smoothers.md): IES, the
  closely-related iterative smoother for time-series inverse problems.
- [API: Processes](../api/processes.md) ·
  [API: Schedulers](../api/schedulers.md) ·
  [API: optax integration](../api/optax.md)

## References

- Iglesias, M. A., Law, K. J. H., & Stuart, A. M. (2013). *Ensemble
  Kalman methods for inverse problems.* Inverse Problems 29(4).
- Iglesias, M. A. (2016). *A regularizing iterative ensemble Kalman
  method for PDE-constrained inverse problems.* Inverse Problems 32(2).
- Schillings, C., & Stuart, A. M. (2017). *Analysis of the ensemble
  Kalman filter for inverse problems.* SIAM J. Numer. Anal. 55(3).
- Garbuno-Inigo, A., Nüsken, N., & Reich, S. (2020). *Affine invariant
  interacting Langevin dynamics for Bayesian inference.* SIAM J. Appl.
  Dyn. Syst. 19(3).
- Huang, D. Z., Schneider, T., & Stuart, A. M. (2022). *Iterated Kalman
  methodology for inverse problems.* J. Comput. Phys. 463.
- Chada, N. K., Stuart, A. M., & Tong, X. T. (2019). *Tikhonov
  regularization within ensemble Kalman inversion.* SIAM J. Numer.
  Anal. 58(2).
- Schneider, T., Stuart, A. M., & Wu, J.-L. (2022). *Ensemble Kalman
  inversion for sparse learning of dynamical systems from
  time-averaged data.* J. Comput. Phys. 470.

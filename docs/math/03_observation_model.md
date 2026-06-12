# Observation Model

The observation model is the bridge between the state the filter
estimates and the data it actually sees. This chapter fixes the
measurement equation, explains how ensembles handle nonlinear
observation operators without ever linearising them explicitly, and
introduces the two observation-space objects every analysis step
consumes: the perturbed observations of the stochastic filters and the
innovation statistics used for the update and for diagnostics.

## The measurement equation

Observations are modelled as a deterministic map of the true state plus
additive Gaussian noise:

$$
y = \mathcal{H}(x) + \varepsilon, \qquad
\varepsilon \sim \mathcal{N}(0, R),
$$

with $y \in \mathbb{R}^{N_y}$, the observation operator
$\mathcal{H} : \mathbb{R}^{N_x} \to \mathbb{R}^{N_y}$, and the
observation-error covariance $R \in \mathbb{R}^{N_y \times N_y}$.
Equivalently, the likelihood is
$p(y \mid x) = \mathcal{N}\!\big(y;\, \mathcal{H}(x),\, R\big)$.

$\mathcal{H}$ can be as simple as a selection of grid points (point
gauges), an interpolation stencil (satellite footprints), or a full
radiative-transfer model (radiances). We write $\mathcal{H}$ for the
general nonlinear operator and $H$ for a linear (or linearised) one,
$y = Hx + \varepsilon$.

## Linear versus nonlinear $\mathcal{H}$

The classical Kalman update (chapter 4) needs the matrices $P H^\top$
and $H P H^\top$ — it is built for linear $H$. The extended Kalman
filter handles nonlinearity by differentiating $\mathcal{H}$
analytically; ensembles do something simpler. Map every member through
the full nonlinear operator, row-wise:

$$
Y = \mathcal{H}(X) \in \mathbb{R}^{N_e \times N_y},
\qquad
Y_{j\cdot} = \mathcal{H}\big(x^{(j)}\big)^\top,
$$

and form observation-space mean and anomalies exactly as in chapter 2:

$$
\bar{y} = \frac{1}{N_e} \sum_{j=1}^{N_e} \mathcal{H}\big(x^{(j)}\big),
\qquad
Y' = Y - \mathbf{1}\bar{y}^\top.
$$

The substitutions $X' H^\top \to Y'$ then turn every appearance of $H$
in the Kalman algebra into a sample statistic:

$$
P H^\top \;\to\; C^{xH} = \frac{1}{N_e - 1} X'^\top Y',
\qquad
H P H^\top \;\to\; C^{HH} = \frac{1}{N_e - 1} Y'^\top Y'.
$$

For linear $H$ these are exact (then $Y' = X' H^\top$ identically).
For nonlinear $\mathcal{H}$, the ensemble provides an implicit,
derivative-free **statistical linearisation**: the regression of
observation-space anomalies on state-space anomalies, evaluated where
the ensemble actually is. No tangent-linear code, no Jacobian — the
caller supplies $\mathcal{H}$ as a plain callable on a single state
vector and filterax `vmap`s it over the rows.

## Structured $R$

filterax never accepts $R$ as a raw dense array: it is always a
`lineax.AbstractLinearOperator`, so its structure is visible to the
linear algebra downstream. The overwhelmingly common case is
uncorrelated instrument noise,

$$
R = \operatorname{diag}\big(\sigma_1^2, \dots, \sigma_{N_y}^2\big),
$$

passed as `lineax.DiagonalLinearOperator(variances)` — solves are
pointwise divisions and sampling needs only the square roots of the
diagonal. Correlated errors (e.g. spatially correlated satellite
retrievals) can be passed as any structured operator — low-rank,
Toeplitz, Kronecker — and gaussx's structural dispatch keeps solves and
square roots cheap. Only a genuinely dense $R$ falls back to a dense
Cholesky, and even then the $O(N_y^3)$ factorisation is amortised
across the ensemble.

## Perturbed observations

Stochastic filters (chapter 5) require each member to be updated
against its own noisy copy of the data,

$$
y^{(j)}_{\text{pert}} = y + \varepsilon^{(j)}, \qquad
\varepsilon^{(j)} \sim \mathcal{N}(0, R), \qquad
j = 1, \dots, N_e,
$$

so the observation error enters the analysis ensemble with the correct
covariance (the *why* is chapter 5's subject). The draws are zero-mean,
so the perturbed set preserves the observation in expectation,
$\mathbb{E}\big[y^{(j)}_{\text{pert}}\big] = y$; the realised sample
mean of $N_e$ draws still deviates from $y$ by $O(1/\sqrt{N_e})$,
which is part of the stochastic filter's Monte Carlo noise budget.

In filterax the draw is structure-aware: for diagonal $R$ the
perturbations are $\varepsilon^{(j)} = z^{(j)} \odot \sqrt{\operatorname{diag}(R)}$
with standard-normal $z^{(j)}$ at $O(N_e N_y)$ cost and no matrix ever
formed; for general $R$ a structured square root $L$ with
$LL^\top = R$ comes from `gaussx.root_decomposition` and
$\varepsilon^{(j)} = L z^{(j)}$.

## Innovation statistics

The **innovation** is the observation-space misfit of the forecast,

$$
d = y - \mathcal{H}(\bar{x}),
$$

where in ensemble practice $\mathcal{H}(\bar{x})$ is computed as the
ensemble mean $\bar{y}$ of $Y = \mathcal{H}(X)$ (the two coincide for
linear $H$). Under the Gaussian model the innovation has zero mean and
covariance

$$
S = C^{HH} + R,
$$

the **innovation covariance**: forecast uncertainty as seen through
$\mathcal{H}$, plus observation noise. $S$ is the matrix the Kalman
gain inverts (chapter 4), and it is also the filter's self-diagnosis
instrument:

- the **log-likelihood** of the data under the forecast,
  $\log p(y \mid \text{forecast}) = -\tfrac{1}{2}\big[N_y \log 2\pi +
  \log\lvert S\rvert + d^\top S^{-1} d\big]$, is the training signal
  for differentiable DA;
- the **whitened innovation** $S^{-1/2} d$ should be standard normal
  component-wise when $P$, $R$, and $\mathcal{H}$ are all correctly
  specified — the basis of $\chi^2$ and Desroziers-style consistency
  checks.

Because $C^{HH}$ has rank $\le N_e - 1$, filterax assembles $S$ as a
`gaussx.LowRankUpdate` over the structured $R$ base, and both
$\log\lvert S\rvert$ and $S^{-1}d$ go through the matrix-determinant
lemma and the Woodbury identity at $O(N_e^2 N_y + N_e^3)$.

## Implementation in filterax

```python
import jax
import jax.numpy as jnp
import lineax as lx
from filterax import innovation_statistics, perturbed_observations

N_e, N_x, N_y = 30, 8, 3
X = jax.random.normal(jax.random.key(0), (N_e, N_x))   # forecast ensemble

def obs_op(x):                       # nonlinear H on a single state vector
    return jnp.tanh(x[:3])

y = jnp.array([0.3, -0.1, 0.2])
R = lx.DiagonalLinearOperator(0.05 * jnp.ones(N_y))

# Innovation d, S = C^HH + R (as a LowRankUpdate), whitened residual,
# and log p(y | forecast) — obs_op is vmapped over the rows internally.
stats = innovation_statistics(X, y, obs_op, R)
stats["innovation"]                  # (3,)
stats["normalized_innovation"]       # S^{-1/2} d — unit-variance if consistent
stats["log_likelihood"]              # scalar training/diagnostic signal

# Perturbed observations for the stochastic EnKF (chapter 5).
y_pert = perturbed_observations(jax.random.key(1), y, R, N_e)   # (30, 3)
```

The `obs_op` argument is any callable from a single state vector to a
single observation vector; linear operators, interpolators, and learned
networks all qualify. `perturbed_observations` consumes an explicit
PRNG key — pass a fresh key per assimilation window, or successive
windows will see identical perturbations.

## Where next

- [Chapter 4 — Kalman update](04_kalman_update.md): the gain
  $K = C^{xH} S^{-1}$ built from this chapter's $Y'$, $R$, and $S$.
- [Chapter 5 — Stochastic EnKF](05_stochastic_enkf.md): where
  `perturbed_observations` earns its keep.
- [Chapter 2 — Ensemble representation](02_ensemble_representation.md):
  the mean/anomaly algebra reused here in observation space.
- [Primitives API](../api/primitives.md) — `innovation_covariance`,
  `innovation_statistics`, `log_likelihood`, `perturbed_observations`.
- [Diagnostics API](../api/diagnostics.md) — rank histograms, spread
  and consistency checks built on the innovation statistics.

## References

- Burgers, G., van Leeuwen, P. J., & Evensen, G. (1998). *Analysis
  scheme in the ensemble Kalman filter.* MWR 126(6).
- Desroziers, G., Berre, L., Chapnik, B., & Poli, P. (2005).
  *Diagnosis of observation, background and analysis-error statistics
  in observation space.* QJRMS 131(613).
- Janjić, T., et al. (2018). *On the representation error in data
  assimilation.* QJRMS 144(713).
- Evensen, G. (2003). *The Ensemble Kalman Filter: theoretical
  formulation and practical implementation.* Ocean Dynamics 53. §5.

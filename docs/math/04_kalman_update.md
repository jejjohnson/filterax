# Kalman Update and the Ensemble Gain

The analysis step of every filter in this library is built around one
matrix: the Kalman gain. This chapter derives it as the best linear
unbiased estimator, translates it into ensemble statistics, and then
turns to the computational question that shapes filterax's design —
how to apply $S^{-1}$ when $S$ is large but structured.

## The best linear unbiased update

Let $x^f$ be the forecast state with error covariance $P$, and let
$y = Hx + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, R)$ be an
observation (linear $H$ for the derivation; the ensemble handles
nonlinear $\mathcal{H}$ by the implicit linearisation of chapter 3).
Seek an estimator that is linear in the innovation $d = y - Hx^f$:

$$
x^a = x^f + K d.
$$

Unbiasedness is automatic ($\mathbb{E}[d] = 0$ when forecast and
observation errors are zero-mean). Choosing $K$ to minimise the total
posterior variance $\operatorname{tr}(P^a)$, where
$P^a = (I - KH) P (I - KH)^\top + K R K^\top$, gives by differentiating
in $K$ and setting the result to zero:

$$
K = P H^\top \big(H P H^\top + R\big)^{-1}.
$$

This is the BLUE — the *best linear unbiased estimator* — and, in the
linear-Gaussian case, also the exact Bayesian posterior mean, with
posterior covariance $P^a = (I - KH)P$. The gain has a clean reading:
it is the regression coefficient of state error on innovation,
"uncertainty in the state as seen by the observations, divided by the
total uncertainty of the observations themselves."

## The ensemble gain

Ensembles never form $P$. Substituting the sample statistics of
chapters 2–3 — $P H^\top \to C^{xH}$, $H P H^\top \to C^{HH}$ — the
gain becomes

$$
K = C^{xH} S^{-1}, \qquad S = C^{HH} + R,
$$

with

$$
C^{xH} = \frac{1}{N_e - 1} X'^\top Y'
\in \mathbb{R}^{N_x \times N_y},
\qquad
C^{HH} = \frac{1}{N_e - 1} Y'^\top Y'
\in \mathbb{R}^{N_y \times N_y}.
$$

Everything is built from the two anomaly matrices. $C^{xH}$ is computed
densely — it is $(N_x, N_y)$ and $N_y$ is typically small — and the
interesting question is the solve against $S$.

## Assembling $S$ as a low-rank update

$C^{HH}$ has rank $\le N_e - 1$ (chapter 2), so $S$ is a structured
$R$ plus a low-rank term:

$$
S = R + U U^\top, \qquad U = \frac{Y'^\top}{\sqrt{N_e - 1}}
\in \mathbb{R}^{N_y \times N_e}.
$$

filterax assembles exactly this object — a `gaussx.LowRankUpdate` with
base $R$ and factor $U$ — and never the dense $(N_y, N_y)$ matrix.
Structural dispatch in gaussx then routes the solve through the
**Woodbury identity** (quoting the implementation's own formula, with
$(HX)'$ the observation-space anomalies $Y'$):

$$
S^{-1} = R^{-1} - R^{-1} U
    \left( (N_{e} - 1) I + U^{\top} R^{-1} U \right)^{-1}
    U^{\top} R^{-1},
\qquad
U = \frac{(HX)^{\prime\top}}{\sqrt{N_{e} - 1}}.
$$

Only the small $(N_e, N_e)$ inner matrix is ever factorised. When
$R^{-1}$ is cheap (diagonal, low-rank, Toeplitz, …), applying $S^{-1}$
costs

$$
O\big(N_e^2 N_y + N_e^3\big)
\qquad \text{instead of} \qquad
O\big(N_y^3\big)
$$

for the dense factorisation. The distinction is decisive whenever
$N_y \gg N_e$ — a satellite swath with $N_y \sim 10^5$ observations and
$N_e = 50$ members costs millions of flops through Woodbury versus
$\sim 10^{15}$ dense. The same structure gives $\log\lvert S \rvert$
through the matrix-determinant lemma, which is how the innovation
log-likelihood of chapter 3 stays cheap.

The gain itself, $K = C^{xH} S^{-1}$, is materialised as a dense
$(N_x, N_y)$ array deliberately: it is consumed once by the analysis
update, and $N_y$ is the per-window observation count, not the state
dimension.

## Localized gains

The raw ensemble gain inherits the sampling noise of $C^{xH}$
(chapter 2): a distant, physically unrelated observation gets a
non-zero gain row of typical size $1/\sqrt{N_e - 1}$. The practical
remedy is Schur-product localization,

$$
K = (\rho^{xy} \circ C^{xH})\,(\rho^{yy} \circ C^{HH} + R)^{-1},
$$

with distance-based tapers $\rho$ (Gaspari–Cohn and friends). The
Hadamard product destroys the low-rank structure of $C^{HH}$, so the
localized gain pays the dense $O(N_y^3)$ price — one of several
trade-offs covered properly in chapter 11. In filterax this is
`localized_kalman_gain`, which reduces exactly to the plain gain when
$\rho \equiv 1$.

## Implementation in filterax

```python
import jax
import jax.numpy as jnp
import lineax as lx
from filterax import kalman_gain

N_e, N_x, N_y = 25, 40, 10
X = jax.random.normal(jax.random.key(0), (N_e, N_x))   # forecast ensemble
Y = X[:, :N_y]                     # linear H: observe the first 10 components
R = lx.DiagonalLinearOperator(0.2 * jnp.ones(N_y))

K = kalman_gain(X, Y, R)           # (40, 10) — Woodbury solve inside

# Sanity check against the dense textbook formula.
Xp = X - X.mean(axis=0)
Yp = Y - Y.mean(axis=0)
C_xH = Xp.T @ Yp / (N_e - 1)
S = Yp.T @ Yp / (N_e - 1) + 0.2 * jnp.eye(N_y)
K_dense = C_xH @ jnp.linalg.inv(S)
assert jnp.allclose(K, K_dense, atol=1e-4)
```

`kalman_gain(particles, obs_particles, obs_noise)` takes the state
ensemble $X$, the observation-space ensemble $Y = \mathcal{H}(X)$ (for
nonlinear $\mathcal{H}$, apply your callable via `jax.vmap` first or
let the filter classes do it), and $R$ as a `lineax` operator. The
optional `solver=` keyword overrides gaussx's structural dispatch; the
default picks Woodbury for low-rank-over-structured-$R$ and a dense
Cholesky otherwise. Like all primitives, the function is pure,
`jit`-compatible, and differentiable end-to-end.

## Where next

- [Chapter 5 — Stochastic EnKF](05_stochastic_enkf.md): the first
  complete filter built on this gain — one $K$, applied to every
  member's own perturbed innovation.
- [Chapter 2 — Ensemble representation](02_ensemble_representation.md):
  why $C^{HH}$ is rank-deficient, and why that is good news here.
- [Chapter 3 — Observation model](03_observation_model.md): where $Y$,
  $S$, and the innovation come from.
- [Primitives API](../api/primitives.md) — `kalman_gain`,
  `cross_covariance`, `innovation_covariance`.
- [Localization API](../api/localization.md) —
  `localized_kalman_gain`, `gaspari_cohn`, `localization_matrix`.

## References

- Kalman, R. E. (1960). *A new approach to linear filtering and
  prediction problems.* J. Basic Eng. 82(1).
- Evensen, G. (2003). *The Ensemble Kalman Filter: theoretical
  formulation and practical implementation.* Ocean Dynamics 53.
- Hager, W. W. (1989). *Updating the inverse of a matrix.* SIAM Review
  31(2). (The Woodbury identity and its history.)
- Tippett, M. K., Anderson, J. L., Bishop, C. H., Hamill, T. M., &
  Whitaker, J. S. (2003). *Ensemble square root filters.* MWR 131(7).
- Gaspari, G., & Cohn, S. E. (1999). *Construction of correlation
  functions in two and three dimensions.* QJRMS 125(554).

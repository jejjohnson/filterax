# Ensemble Representation

The ensemble Kalman filter's central object is not a probability
density but a finite sample from one. This chapter fixes the
conventions — how the sample is laid out, how its moments are defined —
and develops the one structural fact that everything downstream
exploits: the sample covariance of $N_e$ members is a rank-deficient,
low-rank matrix that never needs to be formed explicitly.

## The ensemble matrix

An ensemble of $N_e$ state vectors in $\mathbb{R}^{N_x}$ is stored as a
matrix

$$
X = \begin{pmatrix} x^{(1)\top} \\ \vdots \\ x^{(N_e)\top} \end{pmatrix}
\in \mathbb{R}^{N_e \times N_x},
$$

with **members as rows**: axis 0 is the ensemble axis, axis 1 the state
axis. This matches the code throughout filterax — `particles[j]` is
member $j$, and mapping a per-state function over the ensemble is
`jax.vmap(fn)(particles)`. (Much of the DA literature writes the
transpose, with members as columns; translate accordingly when
comparing formulas.)

The ensemble is a Monte Carlo representation of the filtering density:
member $j$ is one plausible state, and probability statements become
sample statistics over the rows.

## Mean and anomalies

The ensemble mean is the row average,

$$
\bar{x} = \frac{1}{N_e} \sum_{j=1}^{N_e} x^{(j)} \in \mathbb{R}^{N_x},
$$

and the **anomaly matrix** (centred perturbations) is

$$
X' = X - \mathbf{1}\bar{x}^\top \in \mathbb{R}^{N_e \times N_x},
$$

where $\mathbf{1} \in \mathbb{R}^{N_e}$ is the vector of ones. By
construction the rows of $X'$ sum to zero — the anomalies live in the
$(N_e - 1)$-dimensional subspace orthogonal to $\mathbf{1}$. The pair
$(\bar{x}, X')$ carries exactly the same information as $X$, and most
of the analysis algebra in chapters 4–5 is phrased on it: every
ensemble Kalman filter is, in one form or another, a linear transform
applied to these anomalies.

## Sample covariance and the Bessel correction

The (Bessel-corrected) sample covariance is

$$
P = \frac{1}{N_e - 1}\, X'^\top X' \in \mathbb{R}^{N_x \times N_x}.
$$

The divisor $N_e - 1$ rather than $N_e$ is the standard unbiasedness
correction: centring on the *sample* mean $\bar{x}$ removes one degree
of freedom, and dividing by $N_e$ would bias $P$ low by a factor
$(N_e - 1)/N_e$. For $N_e = 20$ that is a 5% systematic
underestimation of every variance — directly feeding the
filter-divergence spiral of chapter 1. The EnKF literature and
filterax both use $N_e - 1$ everywhere; this is why the package
requires $N_e \ge 2$.

## Rank and its consequences

Because the rows of $X'$ sum to zero,

$$
\operatorname{rank}(P) \le N_e - 1,
$$

regardless of $N_x$. With $N_e = 50$ members and $N_x = 10^6$ state
variables, the ensemble asserts that *all* forecast uncertainty lies in
a 49-dimensional subspace. Two consequences matter in practice:

- **The null space.** Directions outside the anomaly span have zero
  sample variance: the filter is blind to errors there and will make no
  correction along them. Analysis increments are always linear
  combinations of the rows of $X'$ — the update cannot leave the
  ensemble subspace.
- **Spurious correlations.** Each off-diagonal entry of $P$ is an
  average of $N_e$ products and carries sampling noise of order
  $1/\sqrt{N_e}$. Physically uncorrelated state pairs therefore show
  spurious sample correlations of typical size $1/\sqrt{N_e - 1}$
  ($\approx 0.14$ for $N_e = 50$), which the Kalman update happily acts
  on. Localization (chapter 11) suppresses them by tapering $P$ with
  distance; inflation (chapter 12) compensates for the variance the
  noise and the rank truncation lose.

What looks like a defect is also the key computational fact: a rank-
$(N_e - 1)$ matrix is fully described by its factors.

## Covariance as a low-rank operator

Define the scaled anomaly factor

$$
U = \frac{X'^\top}{\sqrt{N_e - 1}} \in \mathbb{R}^{N_x \times N_e},
\qquad
P = U U^\top.
$$

Every operation the filter needs from $P$ is available from $U$
directly: a matrix–vector product $Pv = U(U^\top v)$ costs
$O(N_e N_x)$; solves and log-determinants of $P$-plus-something go
through the Woodbury identity and the matrix-determinant lemma at
$O(N_e^2 N_x + N_e^3)$ instead of $O(N_x^3)$ (chapter 4 uses exactly
this for the innovation covariance). The dense
$(N_x, N_x)$ matrix is never needed — and at geophysical scale it
would not fit in memory anyway.

filterax enforces this structurally: `ensemble_covariance` returns a
`gaussx.LowRankUpdate` linear operator built from $U$, **never a dense
array**. The operator plugs into the `lineax`/`gaussx` ecosystem, where
structural dispatch picks low-rank-aware algorithms for `solve`,
`logdet`, and square roots automatically.

## Implementation in filterax

```python
import gaussx
import jax
import jax.numpy as jnp
from filterax import ensemble_anomalies, ensemble_covariance, ensemble_mean

X = jax.random.normal(jax.random.key(0), (20, 500))   # N_e=20, N_x=500

x_bar = ensemble_mean(X)          # x̄, shape (500,)
X_prime = ensemble_anomalies(X)   # X′, shape (20, 500), rows sum to zero
P = ensemble_covariance(X)        # P as a low-rank operator, rank ≤ 19

assert isinstance(P, gaussx.LowRankUpdate)
v = jnp.ones(500)
Pv = P.mv(v)                      # P v without ever forming (500, 500)
```

All three functions are pure, `jit`-compatible, and differentiable.
`ensemble_covariance` delegates to `gaussx.ensemble_covariance` with
`bessel=True`, so the $1/(N_e - 1)$ convention is baked in; it raises
`ValueError` if $N_e < 2$. The returned operator supports the full
`lineax.AbstractLinearOperator` interface — `mv`, composition,
`as_matrix()` (for small debugging cases only) — plus gaussx's
structure-aware `solve` and `logdet`.

The cross-covariance counterpart, `filterax.cross_covariance`, computes
$C^{xH} = \frac{1}{N_e-1} X'^\top Y'$ between a state-space and an
observation-space ensemble; it returns a dense $(N_x, N_y)$ array
because $N_y$ is typically small. It is the workhorse of the Kalman
gain in chapter 4.

## Where next

- [Chapter 3 — Observation model](03_observation_model.md): mapping the
  ensemble into observation space, where the same mean/anomaly algebra
  applies to $Y = \mathcal{H}(X)$.
- [Chapter 4 — Kalman update](04_kalman_update.md): the gain
  $K = C^{xH} S^{-1}$, where the low-rank structure of this chapter
  pays off via Woodbury.
- [Chapter 5 — Stochastic EnKF](05_stochastic_enkf.md): the first
  complete analysis step built on these statistics.
- [Primitives API](../api/primitives.md) — `ensemble_mean`,
  `ensemble_anomalies`, `ensemble_covariance`, `cross_covariance`.
- [Inflation API](../api/inflation.md) and
  [Localization API](../api/localization.md) — the countermeasures to
  sampling noise and rank deficiency.

## References

- Evensen, G. (2003). *The Ensemble Kalman Filter: theoretical
  formulation and practical implementation.* Ocean Dynamics 53.
- Houtekamer, P. L., & Mitchell, H. L. (1998). *Data assimilation using
  an ensemble Kalman filter technique.* MWR 126(3).
- Hamill, T. M., Whitaker, J. S., & Snyder, C. (2001).
  *Distance-dependent filtering of background error covariance
  estimates in an ensemble Kalman filter.* MWR 129(11).
- Vetra-Carvalho, S., et al. (2018). *State-of-the-art stochastic data
  assimilation methods for high-dimensional non-Gaussian problems.*
  Tellus A 70(1).

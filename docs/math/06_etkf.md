# ETKF — Ensemble Transform Kalman Filter

The stochastic EnKF (chapter 5) gets the analysis covariance right only
*in expectation* — every member sees its own perturbed observation, and
those draws add $O(1/\sqrt{N_e})$ Monte Carlo noise to the update. The
ETKF (Bishop, Etherton & Majumdar 2001) removes the sampling step
entirely: it computes a single deterministic *transform* of the
forecast ensemble whose sample mean and sample covariance match the
Kalman analysis exactly.

## The ensemble-space ansatz

Recall the conventions from chapter 2: the ensemble matrix
$X \in \mathbb{R}^{N_e \times N_x}$ holds members as rows,
$\bar{x}$ is the ensemble mean, $X' = X - \mathbf{1}\bar{x}^\top$ the
anomalies, and $P = \frac{1}{N_e-1} X'^\top X'$ the sample covariance.
The observed ensemble is $Y = \mathcal{H}(X)$ with anomalies $Y'$ and
innovation $d = y - \bar{y}$.

The key observation: the Kalman update only ever moves the state within
the span of the forecast anomalies. So instead of working with
$N_x \times N_x$ covariances, write the analysis as a *weighted
recombination of the forecast members* and solve for the weights. Define
the ensemble-space transform precision

$$
\tilde{C} = (N_e - 1)\, I + Y' R^{-1} Y'^{\top}
    \in \mathbb{R}^{N_e \times N_e}, \qquad
T = \tilde{C}^{-1}.
$$

(With members as rows, $Y' R^{-1} Y'^\top$ is the $N_e \times N_e$
inner-product matrix; texts that store members as columns write the
transposes the other way around.) The mean weights and the transform:

$$
\bar{w} = T\, Y' R^{-1} d, \qquad
W = \sqrt{(N_e - 1)\, T},
$$

and the analysis ensemble is assembled entirely from forecast anomalies:

$$
\bar{x}_a = \bar{x} + \bar{w}^{\top} X', \qquad
X'_a = W X', \qquad
X_a = \mathbf{1}\bar{x}_a^{\top} + X'_a.
$$

A short calculation (substitute $C^{HH} = \frac{1}{N_e-1}Y'^\top Y'$
and apply the Sherman-Morrison-Woodbury identity to
$S = C^{HH} + R$) shows that $\bar{x}_a$ equals the Kalman mean
$\bar{x} + K d$ and that
$\frac{1}{N_e-1} X_a'^\top X_a' = (I - KH)\,P$ — the analysis
covariance is exact with respect to the sample statistics, with no
randomness anywhere. The conformance test
`test_etkf_matches_kalman_in_linear_gaussian` pins this to
`atol = 1e-9`.

## Why the symmetric square root

$T$ is symmetric positive definite, so it has infinitely many square
roots: any $W = T^{1/2} U$ with $U$ orthogonal reproduces the same
analysis covariance. They do **not** all produce a valid ensemble. The
anomalies must stay mean-zero — $\mathbf{1}^\top X'_a = 0$ — otherwise
the sample mean of $X_a$ silently drifts away from the Kalman mean
$\bar{x}_a$ and the filter is biased.

Because anomaly rows sum to zero ($Y'^\top \mathbf{1} = 0$), the
constant vector is an eigenvector of $\tilde{C}$:

$$
\tilde{C}\, \mathbf{1} = (N_e - 1)\, \mathbf{1}
\quad\Longrightarrow\quad
W \mathbf{1} = \sqrt{(N_e - 1) \cdot \tfrac{1}{N_e - 1}}\; \mathbf{1}
             = \mathbf{1}
$$

for the *symmetric* square root
$W = U_C \sqrt{(N_e-1)\Lambda^{-1}}\, U_C^\top$ (where
$\tilde{C} = U_C \Lambda U_C^\top$). Then
$\mathbf{1}^\top W X' = (W\mathbf{1})^\top X' = \mathbf{1}^\top X' = 0$:
the transform maps mean-zero anomalies to mean-zero anomalies. Tippett
et al. (2003) and Livings et al. (2008) show the symmetric choice is the
unique PSD square root with this property — any other rotation either
breaks the mean or loses positive semi-definiteness. filterax's `EnSRF`
(chapter 7) uses exactly the same symmetric transform for its
perturbation half.

## Differentiability — the QR + small-eigh spectrum

This subsection is the reason filterax carries its own ETKF core rather
than delegating to `gaussx.etkf_transform`.

The textbook recipe eigendecomposes the full $N_e \times N_e$ matrix
$\tilde{C}$. But $Y' R^{-1} Y'^\top$ has rank at most $N_y$, so
$\tilde{C}$ has $N_e - N_y$ eigenvalues *structurally equal* to
$N_e - 1$. Forward-mode that is harmless; in reverse mode, the
JVP/VJP rule for `eigh` divides by pairwise eigenvalue gaps
$\lambda_i - \lambda_j$, and repeated eigenvalues produce `NaN`
gradients. Any training loop that backpropagates through the analysis
(chapter 14) would die on the very first step.

filterax sidesteps the degenerate subspace instead of regularising it:

1. Thin QR: $Y' = Q\, R_{qr}$ with $Q \in \mathbb{R}^{N_e \times N_y}$.
2. Form the small symmetric PSD matrix
   $M_y = R_{qr} R^{-1} R_{qr}^\top \in \mathbb{R}^{N_y \times N_y}$
   and eigendecompose it: $M_y = V \operatorname{diag}(\lambda_i) V^\top$.
3. Lift: $U_y = Q V$ gives
   $Y' R^{-1} Y'^\top = U_y \operatorname{diag}(\lambda_i) U_y^\top$.

Every matrix function of $\tilde{C}$ then takes the rank-$N_y$
correction form

$$
g(\tilde{C}) = g(N_e - 1)\, I
    + U_y\, \operatorname{diag}\!\big(g(N_e - 1 + \lambda_i) - g(N_e - 1)\big)\, U_y^{\top},
$$

used twice per analysis: $g(\lambda) = 1/\lambda$ for $T$ and
$g(\lambda) = \sqrt{(N_e-1)/\lambda}$ for $W$. The degenerate
eigenvalues never meet an `eigh`, the small $(N_y, N_y)$ problem has
generically distinct eigenvalues, and `jax.grad` through the whole
analysis is finite. The solve $R^{-1} Y'$ routes through gaussx
structural dispatch, so a diagonal $R$ never densifies.

## ETKF_Livings — breaking the deterministic symmetry

The symmetric transform is deterministic, and over hundreds of cycles
the same $W$-shaped contraction can develop preferred directions: the
ensemble drifts toward a small invariant subspace and the higher
moments become non-Gaussian. Livings et al. (2008) compose the
transform with a *random mean-preserving rotation*

$$
W^{\mathrm{rot}} = W\, \Theta, \qquad
\Theta \in O(N_e), \quad \Theta\, \mathbf{1} = \mathbf{1},
$$

built by drawing a uniform rotation in the $(N_e - 1)$-dimensional
complement of $\mathbf{1}$ (QR of a Gaussian matrix) and lifting it back
with a Householder basis. The constraint $\Theta\mathbf{1} = \mathbf{1}$
keeps the analysis mean and the mean-zero anomaly property exactly; the
randomness averages out preferred-direction artefacts over time. Use a
fresh `key` per cycle — repeating the same rotation every window
defeats the purpose.

## Implementation in filterax

```python
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

import filterax as flx

# Forecast ensemble: N_e = 30 members of a 4-dimensional state.
particles = 1.0 + 0.5 * jr.normal(jr.key(0), (30, 4))

# Observe components 0 and 2 with independent noise.
H = jnp.zeros((2, 4)).at[0, 0].set(1.0).at[1, 2].set(1.0)
obs = jnp.array([0.4, -0.1])
R = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))

result = flx.filters.ETKF().analysis(particles, obs, lambda x: H @ x, R)
result.particles.shape         # (30, 4)
result.log_likelihood          # Gaussian log p(d | S), for diagnostics

# Observed components contract; unobserved ones barely move.
particles.std(axis=0, ddof=1)         # [0.401, 0.497, 0.473, 0.497]
result.particles.std(axis=0, ddof=1)  # [0.248, 0.491, 0.263, 0.484]

# Livings variant: same mean, randomly rotated perturbations.
rotated = flx.filters.ETKF_Livings(key=jr.key(1)).analysis(
    particles, obs, lambda x: H @ x, R, key=jr.key(2)
)
jnp.allclose(
    rotated.particles.mean(axis=0), result.particles.mean(axis=0), atol=1e-5
)  # True

# The analysis is differentiable end-to-end (the QR + small-eigh core).
import jax

def posterior_spread(scale):
    res = flx.filters.ETKF().analysis(scale * particles, obs, lambda x: H @ x, R)
    return res.particles.std(axis=0, ddof=1).sum()

jax.grad(posterior_spread)(1.0)  # finite — no repeated-eigenvalue NaN
```

`filterax.filters.ETKF` is the Layer-1 analysis step: one posterior
ensemble from one observation vector. The Layer-2 `filterax.ETKF`
wraps it with dynamics, inflation, and the multi-window `assimilate()`
loop (chapter 16).

## Cost

Per analysis: $O(N_e N_y^2)$ for the thin QR and the small eigh,
$O(N_e^2 N_y)$ for assembling the weights, $O(N_e^2 N_x)$ for applying
them — all independent of $N_x^2$. The transform never touches an
$N_x \times N_x$ or (for structured $R$) an $N_y \times N_y$ dense
matrix. For $N_y > N_e$ the QR-based spectrum no longer applies
directly; batch the observations or use domain localization
(chapter 8), which makes every local $N_y$ small.

## When ETKF is the right answer

- Deterministic, reproducible analyses — no perturbation noise.
- Small-to-moderate ensembles where the stochastic EnKF's
  $O(1/\sqrt{N_e})$ sampling error is the dominant error term.
- Differentiable pipelines — gradient-safe by construction.
- As the inner solver for LETKF (chapter 8), which runs one ETKF per
  grid point.

Reach for something else when observations vastly outnumber ensemble
members without localization (cost), when $R$ is non-diagonal *and*
huge (the solve dominates), or when you specifically want serial
observation processing — `EnSRF_Serial`, chapter 7.

## Where next

- [Chapter 5 — Stochastic EnKF](05_stochastic_enkf.md): the
  perturbed-observations alternative.
- [Chapter 7 — EnSRF & ESTKF](07_ensrf_estkf.md): sibling square-root
  filters; same transform, different phrasing and subspace.
- [Chapter 8 — LETKF](08_letkf.md): one ETKF per grid point with
  tapered $R^{-1}$.
- [Chapter 14 — Differentiable assimilation](14_differentiable.md):
  why gradient-safety of the analysis matters.
- [API: Filters](../api/filters.md) ·
  [API: Advanced filters](../api/filters_advanced.md) ·
  [API: Differentiable](../api/differentiable.md)

## References

- Bishop, C. H., Etherton, B. J., & Majumdar, S. J. (2001). *Adaptive
  sampling with the ensemble transform Kalman filter. Part I:
  Theoretical aspects.* MWR 129(3).
- Tippett, M. K., Anderson, J. L., Bishop, C. H., Hamill, T. M., &
  Whitaker, J. S. (2003). *Ensemble square root filters.* MWR 131(7).
- Livings, D. M., Dance, S. L., & Nichols, N. K. (2008). *Unbiased
  ensemble square root filters.* Physica D 237(8).
- Hunt, B. R., Kostelich, E. J., & Szunyogh, I. (2007). *Efficient data
  assimilation for spatiotemporal chaos: A local ensemble transform
  Kalman filter.* Physica D 230(1-2).

# EnSRF & ESTKF — Square-Root Variants

The ETKF (chapter 6) is one member of a family: every *deterministic
square-root filter* produces the exact Kalman analysis covariance
$P_a = (I - KH)\,P$ on the sample statistics, without perturbed
observations. The family members differ in **where** the square root is
taken — observation space, ensemble space, or the error subspace — and
in whether observations are processed in one batch or one at a time.
This chapter covers the two variants filterax ships alongside ETKF:
the EnSRF (batch and serial forms) and the ESTKF.

## EnSRF — mean from the gain, perturbations from the transform

Whitaker & Hamill (2002) phrase the square-root update in the most
Kalman-familiar way possible. Split the ensemble into mean and
anomalies and update them separately:

$$
\bar{x}_a = \bar{x} + K\, d, \qquad
K = C^{xH} S^{-1}, \quad S = C^{HH} + R,
$$

with the perturbations transformed so that their sample covariance
equals $(I - KH)\,P$ exactly. In filterax's batch `EnSRF` the
perturbation half is the *symmetric ETKF transform* applied to the
anomalies,

$$
X'_a = W X', \qquad W = \sqrt{(N_e - 1)\, \tilde{C}^{-1}},
$$

so the batch EnSRF and the ETKF are numerically the same filter — the
symmetric square root is the unique PSD- and mean-preserving choice
(chapter 6), and both classes share the differentiable QR + small-eigh
core. The two classes coexist so you can use whichever phrasing is
idiomatic in your community: "mean from $K$, perturbations from a
square root" (EnSRF) versus "weights in ensemble space" (ETKF).

## Serial processing — Whitaker & Hamill's scalar recursion

The classical EnSRF distinction only appears when observations are
assimilated **one at a time**. For a single scalar observation $y_k$
with variance $R_{kk}$, the gain and innovation variance are scalars
and no matrix inversion is needed at all. `EnSRF_Serial` sweeps over
the observations with the recursion

$$
\begin{aligned}
K_k &= C^{x H_k} \big/ \big(C^{H_k H_k} + R_{kk}\big) \\
\bar{x}_a^{(k)} &= \bar{x}_a^{(k-1)}
    + K_k \big(y_k - H_k \bar{x}_a^{(k-1)}\big) \\
\alpha_k &= 1 \Big/ \Big(1 + \sqrt{R_{kk} \big/ \big(C^{H_k H_k} + R_{kk}\big)}\Big) \\
X'^{(k)}_a &= X'^{(k-1)}_a - \alpha_k K_k \big(H_k X'^{(k-1)}_a\big)
\end{aligned}
$$

where every quantity is recomputed from the *current* (partially
updated) ensemble. The reduced gain $\alpha_k K_k$ is the scalar
square-root trick: a full gain $K_k$ on the perturbations would shrink
them twice over (once too much), and $\alpha_k \in (\tfrac12, 1)$ is
exactly the factor that lands the anomaly variance on
$(1 - K_k H_k)\,C^{xx}$. Each scalar update is an inner product, so the
whole sweep costs $O(N_e N_x N_y)$ — no eigendecomposition anywhere.

Requirements and trade-offs:

- **Diagonal $R$ only.** Serial processing assumes uncorrelated
  observation errors; assimilating one component of a correlated $y$
  at a time is simply wrong. filterax raises `NotImplementedError` for
  non-diagonal `obs_noise` — pre-decorrelate ($R^{-1/2} y$) or use the
  batch filters.
- **When serial beats batch.** The batch transform costs
  $O(N_e^2 N_y + N_e^3)$; the serial sweep $O(N_e N_x N_y)$. With
  large $N_e$ and cheap $\mathcal{H}$ the serial form wins; it also
  pairs naturally with per-observation localization (each scalar update
  can carry its own taper) and with streaming pipelines where
  observations arrive incrementally.
- **Order sensitivity.** In exact arithmetic with linear $H$ the result
  is order-independent; with nonlinear $\mathcal{H}$ or per-observation
  localization the assimilation order matters slightly. filterax
  re-applies `obs_op` to the running ensemble inside every scalar step,
  which keeps the sweep consistent for nonlinear $\mathcal{H}$ at the
  cost of $N_y$ extra operator evaluations.

## ESTKF — the error-subspace transform

Nerger et al. (2012) observed that the ETKF's $N_e$-dimensional
ensemble space is one dimension too big: anomalies always satisfy
$X'^\top \mathbf{1} = 0$, so they span at most $N_e - 1$ directions.
The ESTKF works directly in that error subspace. Choose a
mean-preserving orthonormal map $L \in \mathbb{R}^{N_e \times (N_e-1)}$
with $L^\top L = I$ and $L^\top \mathbf{1} = 0$ (filterax builds it
from a Householder reflector), and project:

$$
\tilde{X} = L^{\top} X' \in \mathbb{R}^{(N_e-1) \times N_x}, \qquad
\tilde{Y} = L^{\top} Y' \in \mathbb{R}^{(N_e-1) \times N_y}.
$$

Because $L$ spans exactly the anomaly subspace, nothing is lost. The
transform precision is the $(N_e-1)$-dimensional analogue of
$\tilde{C}$:

$$
A = (N_e - 1)\, I + \tilde{Y} R^{-1} \tilde{Y}^{\top}
    \in \mathbb{R}^{(N_e-1) \times (N_e-1)},
$$

with weights and symmetric square root

$$
\tilde{w} = A^{-1} \tilde{Y} R^{-1} d, \qquad
\tilde{W} = \sqrt{(N_e - 1)\, A^{-1}},
$$

lifted back to the full ensemble:

$$
\bar{x}_a = \bar{x} + \tilde{w}^{\top} \tilde{X}, \qquad
X_a = \mathbf{1}\bar{x}_a^{\top} + L\, \tilde{W} \tilde{X}.
$$

The relation to ETKF: identical analysis mean and covariance, with the
eigen-problem shrunk from $N_e^3$ to $(N_e-1)^3$ and the degenerate
$\mathbf{1}$ direction removed *by construction* rather than handled as
a known eigenvector. filterax's implementation reuses the same
rank-$N_y$ QR + small-eigh spectrum as ETKF (the reduced anomalies
$\tilde{Y}$ have the same rank-deficiency story), so gradients stay
finite here too. ESTKF is the analysis core of the PDAF ecosystem, and
matching it matters when cross-validating against PDAF-based
experiments.

## Implementation in filterax

All filters share the `analysis(particles, obs, obs_op, obs_noise)`
signature, so comparing them is a one-liner each:

```python
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

import filterax as flx

particles = jr.normal(jr.key(0), (25, 3))   # N_e = 25, N_x = 3
obs = jnp.array([0.3, -0.2, 0.5])
R = lx.DiagonalLinearOperator(0.2 * jnp.ones(3))
obs_op = lambda x: x                        # identity observations

etkf   = flx.filters.ETKF().analysis(particles, obs, obs_op, R)
ensrf  = flx.filters.EnSRF().analysis(particles, obs, obs_op, R)
serial = flx.filters.EnSRF_Serial().analysis(particles, obs, obs_op, R)
estkf  = flx.filters.ESTKF().analysis(particles, obs, obs_op, R)

mean = lambda r: r.particles.mean(axis=0)
cov = lambda r: jnp.cov(r.particles.T, ddof=1)

# All four produce the same analysis mean and covariance here
# (linear H, diagonal R) — they differ only in how they get there.
jnp.allclose(mean(ensrf),  mean(etkf), atol=1e-5)   # True
jnp.allclose(mean(serial), mean(etkf), atol=1e-5)   # True
jnp.allclose(mean(estkf),  mean(etkf), atol=1e-5)   # True
jnp.allclose(cov(ensrf),   cov(etkf),  atol=1e-5)   # True
jnp.allclose(cov(serial),  cov(etkf),  atol=1e-4)   # True
jnp.allclose(cov(estkf),   cov(etkf),  atol=1e-5)   # True
```

The *individual members* are not identical across variants — each
square root is a different rotation of the same analysis-covariance
ellipsoid — but the first two sample moments agree, which is all the
Kalman update specifies.

## Choosing within the square-root family

| Filter | Square root taken in | Cost | Use when |
|---|---|---|---|
| `ETKF` | ensemble space ($N_e$) | $O(N_e^2 N_y + N_e N_y^2)$ | Default deterministic filter |
| `EnSRF` | ensemble space ($N_e$) | same as ETKF | You prefer the mean-from-$K$ phrasing |
| `EnSRF_Serial` | observation space, scalar | $O(N_e N_x N_y)$ | Diagonal $R$, streaming obs, no eigh wanted |
| `ESTKF` | error subspace ($N_e - 1$) | same as ETKF, smaller eigh | PDAF compatibility, exact subspace bookkeeping |
| `ETKF_Livings` | ensemble space + rotation | ETKF + $O(N_e^3)$ | Long reanalyses where determinism breeds artefacts |

All deterministic variants still need inflation (chapter 12) in cycled
operation — removing sampling noise does not remove the systematic
variance underestimation of a finite ensemble.

## Where next

- [Chapter 6 — ETKF](06_etkf.md): the transform mathematics and the
  differentiability story shared by all of these filters.
- [Chapter 8 — LETKF](08_letkf.md): the localized transform filter.
- [Chapter 12 — Inflation](12_inflation.md): why square-root filters
  still under-disperse in cycles.
- [API: Filters](../api/filters.md) ·
  [API: Advanced filters](../api/filters_advanced.md)

## References

- Whitaker, J. S., & Hamill, T. M. (2002). *Ensemble data assimilation
  without perturbed observations.* MWR 130(7).
- Tippett, M. K., Anderson, J. L., Bishop, C. H., Hamill, T. M., &
  Whitaker, J. S. (2003). *Ensemble square root filters.* MWR 131(7).
- Nerger, L., Janjić, T., Schröter, J., & Hiller, W. (2012). *A unification
  of ensemble square root Kalman filters.* MWR 140(7).
- Andrews, A. (1968). *A square root formulation of the Kalman
  covariance equations.* AIAA Journal 6(6).

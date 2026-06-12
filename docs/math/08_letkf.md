# LETKF — Local Ensemble Transform Kalman Filter

A finite ensemble can only represent $N_e - 1$ directions of
uncertainty, and its sample covariance $P = \frac{1}{N_e-1}X'^\top X'$
is polluted by spurious long-range correlations: with $N_e = 50$
members, the sample correlation between two physically unrelated grid
points has standard deviation $\approx 1/\sqrt{N_e} \approx 0.14$ even
when the true correlation is zero. A global analysis dutifully uses
those phantom correlations to "correct" Antarctica with a thermometer
in Berlin. Localization suppresses them by construction; the LETKF
(Hunt, Kostelich & Szunyogh 2007) does it by *domain decomposition*:
every grid point solves its own small ETKF using only nearby
observations.

## Domain localization with a tapered $R^{-1}$

Attach coordinates to the state components ($N_x$ points
$s_i \in \mathbb{R}^D$) and to the observations ($N_y$ points
$o_k \in \mathbb{R}^D$). For grid point $i$:

1. Compute distances $d_{ik} = \lVert s_i - o_k \rVert_2$ and taper
   weights $\rho_{ik} = \rho(d_{ik}/r)$, by default the Gaspari-Cohn
   fifth-order compactly supported function.
2. Apply a **hard cutoff at $r$**: observations with $d_{ik} > r$ are
   dropped entirely. (Gaspari-Cohn has support out to $2r$, so a purely
   taper-based rule would let weakly tapered far observations leak in;
   filterax enforces the documented selection radius exactly.)
3. Inflate the surviving observation variances by $1/\rho_{ik}$ —
   equivalently, taper the precision:

$$
\big[R^{-1}_{\mathrm{loc},i}\big]_{kk} = \frac{\rho_{ik}}{R_{kk}},
\qquad \rho_{ik} = 0 \text{ for } d_{ik} > r .
$$

4. Solve the ETKF transform (chapter 6) with $R^{-1}$ replaced by
   $R^{-1}_{\mathrm{loc},i}$ and write back only component $i$ of the
   analysis.

Down-weighting an observation by inflating its error variance is what
"R-localization" means: distant data is treated as *less certain*
rather than *less correlated*. As $\rho_{ik} \to 0$ the observation's
precision contribution vanishes continuously, so the analysis varies
smoothly from one grid point to the next — essential to avoid
discontinuities in the analyzed fields. R-localization assumes a
**diagonal $R$** (Hunt et al. 2007 §2); filterax accepts a
`lineax.DiagonalLinearOperator` and rejects anything else.

## The per-point weights formulation

The decisive trick in Hunt et al. (2007) is that each local problem is
solved *in ensemble space*, so its output is a pair of weight objects
rather than a state:

$$
\tilde{C}_i = (N_e - 1)\, I + Y' R^{-1}_{\mathrm{loc},i} Y'^{\top},
\qquad
\bar{w}_i = \tilde{C}_i^{-1}\, Y' R^{-1}_{\mathrm{loc},i}\, d,
\qquad
W_i = \sqrt{(N_e - 1)\, \tilde{C}_i^{-1}} .
$$

The global anomaly matrices $X'$, $Y'$ and the innovation
$d = y - \bar{y}$ are computed once; only the tapered precision changes
per point. The analysis at grid point $i$ recombines the *global*
anomalies with the *local* weights:

$$
\bar{x}_a[i] = \bar{x}[i] + \bar{w}_i^{\top} X'_{[:,i]}, \qquad
X_a[j, i] = \bar{x}_a[i] + \big(W_i\, X'\big)_{[j, i]} .
$$

Each local solve is a complete, independent ETKF — $N_x$ of them, with
no data dependencies. filterax `jax.vmap`s the per-point weight
computation over `state_coords` and contracts all the weight matrices
against the anomalies in two einsum calls. Each local transform uses
the same differentiable QR + small-eigh spectrum as the global ETKF,
so the localized analysis remains gradient-safe.

## B-localization versus R-localization

There are two places to cut off spurious correlations, and they are
*not* the same operation:

- **Covariance (B-)localization** (chapter 11) takes the Schur product
  of the sample covariance with a taper matrix:
  $P_{\mathrm{loc}} = \rho_B \circ P$. It acts in *state space*,
  raises the rank of the effective covariance (a Schur product of PSD
  matrices is PSD, and tapering restores degrees of freedom the
  ensemble lacks), and slots naturally into gain-based filters
  (`localized_kalman_gain`, the stochastic EnKF).
- **Domain / R-localization** (this chapter) leaves $P$ untouched and
  instead shrinks each grid point's *observation window*, tapering
  $R^{-1}$ with distance. It acts in *observation space*, keeps every
  local problem a dense low-dimensional ETKF, and is the natural fit
  for transform filters, where the ensemble-space algebra never forms
  $P$ at all.

For broad tapers the two give similar analyses (Greybush et al. 2011
compare them systematically; B-localization tends to produce slightly
smoother, more balanced fields, R-localization is cheaper per
observation in transform filters). The practical rule: gain-based
filter → B-loc; transform filter → R-loc/LETKF.

## Cost

The global ETKF costs $O(N_e^2 N_y + N_e^3)$ once. The LETKF solves
$N_x$ local problems of size $N_{y,\mathrm{loc}}$ (the typical number
of observations within radius $r$):

$$
O\big(N_x \cdot (N_e^2 N_{y,\mathrm{loc}} + N_e^3)\big),
$$

linear in the state dimension with a small constant — and the $N_x$
solves are embarrassingly parallel, which is exactly what `vmap` on an
accelerator wants. This is why LETKF (and its ESTKF-core sibling in
PDAF) is the workhorse of operational ensemble NWP: cost scales with
the grid, not with the cube of the global observation count, and the
localization simultaneously fixes the rank problem — each local
analysis only needs the ensemble to resolve *local* error directions.

The price: a localization radius to tune (too small starves the
analysis of data and fragments balance; too large readmits sampling
noise), and dynamical balances spanning scales longer than $r$ can be
disrupted — diagnose with the tools of chapter 13.

## Implementation in filterax

`LETKF.analysis` needs two extra keyword arguments relative to the
global filters: `state_coords` $(N_x, D)$ and `obs_coords` $(N_y, D)$.

```python
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

import filterax as flx

# 1-D grid of 8 state points, ensemble of 20 members.
particles = jr.normal(jr.key(0), (20, 8))
state_coords = jnp.arange(8.0)[:, None]          # (N_x, 1)

# Observe grid points 1, 4 and 6.
obs_idx = jnp.array([1, 4, 6])
obs_coords = state_coords[obs_idx]               # (N_y, 1)
obs = jnp.array([0.5, -0.3, 0.2])
R = lx.DiagonalLinearOperator(0.25 * jnp.ones(3))

letkf = flx.filters.LETKF(radius=2.0)            # Gaspari-Cohn taper by default
result = letkf.analysis(
    particles, obs, lambda x: x[obs_idx], R,
    state_coords=state_coords, obs_coords=obs_coords,
)
result.particles.shape                           # (20, 8)

# Compare the per-component update size against the global ETKF:
global_result = flx.filters.ETKF().analysis(
    particles, obs, lambda x: x[obs_idx], R
)
jnp.abs(result.particles - particles).mean(axis=0)
# [0.028, 0.422, 0.014, 0.068, 0.404, 0.098, 0.536, 0.104]
jnp.abs(global_result.particles - particles).mean(axis=0)
# [0.180, 0.432, 0.029, 0.096, 0.404, 0.112, 0.541, 0.186]
```

Observed points (1, 4, 6) receive nearly the same correction in both;
the far end-points (0, 7) are corrected by the global ETKF purely
through sampling-noise correlations — the LETKF correction there is an
order of magnitude smaller. A custom `taper_fn(distances, radius)` can
replace Gaspari-Cohn (e.g. `flx.gaussian_taper`). The Layer-2
`filterax.LETKF` model threads the coordinates through the
`assimilate()` cycle for you.

## When LETKF is the right answer

- $N_x \gg N_e$ (any realistic geophysical grid) — the rank argument
  alone forces localization.
- Dense observation networks where a global transform would cost
  $O(N_y^2)$ or worse.
- States with meaningful spatial coordinates and roughly local error
  correlations.

It is the wrong tool for non-spatial state vectors (global parameters
have no distance to observations — see chapter 9's inverse-problem
methods), for strongly non-local observation operators (a satellite
radiance integrating a whole column needs a definition of "location"
per observation, usually the weighting-function peak), and for
correlated $R$ (decorrelate first, or use the batch filters of
chapters 6-7).

## Where next

- [Chapter 6 — ETKF](06_etkf.md): the transform solved at every grid
  point.
- [Chapter 11 — Localization](11_localization.md): B-localization,
  taper functions, and adaptive radii.
- [Chapter 12 — Inflation](12_inflation.md): LETKF in cycles still
  needs spread maintenance.
- [Chapter 13 — Diagnostics](13_diagnostics.md): innovation statistics
  for tuning the radius.
- [API: Filters](../api/filters.md) ·
  [API: Localization](../api/localization.md)

## References

- Hunt, B. R., Kostelich, E. J., & Szunyogh, I. (2007). *Efficient data
  assimilation for spatiotemporal chaos: A local ensemble transform
  Kalman filter.* Physica D 230(1-2).
- Ott, E., et al. (2004). *A local ensemble Kalman filter for
  atmospheric data assimilation.* Tellus A 56(5).
- Gaspari, G., & Cohn, S. E. (1999). *Construction of correlation
  functions in two and three dimensions.* QJRMS 125(554).
- Greybush, S. J., Kalnay, E., Miyoshi, T., Ide, K., & Hunt, B. R.
  (2011). *Balance and ensemble Kalman filter localization techniques.*
  MWR 139(2).

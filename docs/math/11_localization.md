# Localization

A finite ensemble pays for its cheapness with sampling noise. The
sample covariance $P = \frac{1}{N_e-1} X'^\top X'$ built from $N_e$
members has rank at most $N_e - 1$, and — more damaging in practice —
its entries between physically *unrelated* variables are not zero but
random, with magnitude of order $1/\sqrt{N_e}$. A 20-member ensemble
over a global grid will report correlations of $\sim 0.2$ between a
grid point in the North Atlantic and one over Australia. The Kalman
gain takes those correlations at face value and spreads every
observation's increment across the entire domain. Over repeated
cycles the accumulated noise overwhelms the signal and the filter
diverges.

Localization is the standard cure: damp covariance entries as a
function of the physical distance $d_{ij}$ between variables $i$ and
$j$, on the prior knowledge that physical correlations decay with
distance while sampling noise does not.

## The Schur product

The localized covariance is the Schur (Hadamard, element-wise)
product of the sample covariance with a taper matrix $\rho$:

$$
P_{\text{loc}} = \rho \circ P,
\qquad
\rho_{ij} = \rho(d_{ij} / r),
$$

where $r$ is the localization **half-width** (filterax's convention:
compactly supported tapers vanish at distance $2r$). The crucial
structural fact is the **Schur product theorem**: the element-wise
product of two positive semi-definite matrices is positive
semi-definite. So as long as the taper matrix $\rho$ is itself a
valid correlation matrix — i.e. the taper function is positive
definite — localization can *never* break the PSD-ness of the
covariance, no matter how aggressively it cuts.

That condition is what separates the taper functions below.

## Taper functions

### Gaspari-Cohn

The operational gold standard ([Gaspari & Cohn 1999][gc99]). With
$z = d / r$, the 5th-order piecewise polynomial is

$$
\rho(z) = \begin{cases}
  -\tfrac14 z^5 + \tfrac12 z^4 + \tfrac58 z^3 - \tfrac53 z^2 + 1
    & 0 \le z \le 1, \\[4pt]
  \tfrac1{12} z^5 - \tfrac12 z^4 + \tfrac58 z^3 + \tfrac53 z^2
    - 5 z + 4 - \dfrac{2}{3 z}
    & 1 < z \le 2, \\[4pt]
  0 & z > 2.
\end{cases}
$$

[gc99]: references.md

It has **compact support** (exactly zero beyond $d = 2r$, so distant
blocks drop out of the linear algebra entirely), is **$C^2$ smooth**
(value, first, and second derivatives match at $z = 1$ and $z = 2$),
and is **positive definite** — it was constructed by self-convolving
a compactly supported kernel precisely so the Schur product theorem
applies. It closely mimics a Gaussian over its support.

### Gaussian

$$
\rho(d) = \exp\!\left( -\frac{d^2}{2 r^2} \right).
$$

Infinitely smooth and positive definite, but **not** compactly
supported — it decays exponentially yet never reaches zero, so no
sparsity is gained. Useful when compact support is not needed (e.g.
adjoint sensitivity studies where smoothness of $\partial \rho /
\partial r$ matters more than cost).

### SOAR

The Second-Order Auto-Regressive taper (Thiebaux & Pedder 1987):

$$
\rho(d) = \left( 1 + \frac{d}{r} \right) e^{-d / r}.
$$

Positive definite and $C^1$ (one derivative fewer than Gaspari-Cohn),
with *approximate* compact support — it falls below $0.01$ by
$d \approx 5r$. A common correlation model for background errors in
operational variational systems; as a localization taper it is a
simpler alternative when strict compact support is not required.

### Hard cutoff

$$
\rho(d) = \mathbb{1}\{ d \le r \}.
$$

Discontinuous, and — the important part — **not positive definite**
in general. The Schur product theorem does not apply, the localized
matrix can acquire negative eigenvalues, and the filter can go
unstable in ways that are hard to diagnose. filterax ships it as a
debugging baseline only; use Gaspari-Cohn or SOAR in production.

## Distances and the taper matrix

Building $\rho$ means evaluating the taper on a pairwise distance
matrix. `filterax.localization_matrix(coords_a, coords_b, radius)`
fuses the two steps, delegating to `gaussx.localization_matrix` with
the compact-support parameter $c = 2r$ (gaussx parameterises by the
support length, filterax by the half-width — the entries agree
exactly with `gaspari_cohn(metric(a, b), r)`). The default metric is
`euclidean_distance`; pass `haversine_distance` for (lat, lon)
coordinates in radians on the sphere. Both are re-exported from
gaussx unchanged.

## The localized Kalman gain

Tapering $P$ propagates into the gain. With $\rho^{xy}$ the
state–observation taper and $\rho^{yy}$ the observation–observation
taper,

$$
K_{\text{loc}}
  = \big(\rho^{xy} \circ C^{xH}\big)
    \big(\rho^{yy} \circ C^{HH} + R\big)^{-1},
$$

which is **B-localization**: the taper acts on the background
covariance, evaluated between physical coordinates. By the Schur
product theorem the tapered innovation covariance stays PSD whenever
$\rho^{yy}$ is. With $\rho \equiv 1$ this reduces exactly to the
unlocalized gain of chapter 4.

There is a real cost. The unlocalized gain keeps $S = C^{HH} + R$ as
a rank-$(N_e-1)$ update of $R$ and solves it through the Woodbury
identity at $O(N_e^2 N_y + N_e^3)$. The Hadamard product **destroys
that low-rank structure** — $\rho^{yy} \circ C^{HH}$ is generically
full-rank — so `localized_kalman_gain` materialises the dense
$(N_y, N_y)$ innovation covariance and pays $O(N_e N_x N_y + N_y^3)$.
This is the algebraic reason the LETKF (chapter 8) prefers
**R-localization**: instead of tapering covariances, it solves a
small local analysis at each grid point with distant observations'
errors inflated by $1/\rho$, keeping every local solve cheap. B-loc
(this chapter) is the natural fit for gain-based filters
(`StochasticEnKF`, serial EnSRF); R-loc for transform filters.

## Adaptive localization

Distance-based tapers need a radius, and the right radius depends on
the (unknown) true correlation structure. Anderson's hierarchical
argument ([Anderson 2007][a07]) suggests a *data-driven* alternative:
keep a gain entry only when its sample correlation exceeds its own
sampling uncertainty. For each (state $i$, observation $j$) pair with
sample correlation $r_{ij} = C^{xH}_{ij} / (\sigma_{x_i}
\sigma_{y_j})$, the noise floor is

$$
\operatorname{se}(r) \approx \frac{1 - r^2}{\sqrt{N_e - 2}},
$$

[a07]: references.md

and `adaptive_localization` returns the hard mask
$\rho_{ij} = \mathbb{1}\{ |r_{ij}| > \text{significance} \cdot
\operatorname{se}(r_{ij}) \}$, multiplied into the gain as a Schur
product. No radius to tune — but it needs enough members for the
noise floor to be informative (typically $N_e \ge 20$, and $N_e \ge 3$
is enforced) and costs an extra $O(N_e N_x N_y)$ per cycle.

## Implementation in filterax

```python
import jax
import jax.numpy as jnp
import lineax as lx

import filterax as flx

# A 40-point 1-D grid observed at every 4th point.
N_x, N_e = 40, 10
grid = jnp.arange(N_x, dtype=jnp.float32)[:, None]
obs_idx = jnp.arange(0, N_x, 4)
obs_coords = grid[obs_idx]

# Small ensemble -> noisy sample covariance with spurious long-range entries.
key = jax.random.key(0)
ensemble = jax.random.normal(key, (N_e, N_x))
anom = ensemble - ensemble.mean(axis=0)
P = anom.T @ anom / (N_e - 1)

# Taper and localize: P_loc = rho ∘ P.
rho = flx.localization_matrix(grid, grid, radius=3.0)
P_loc = flx.localize(P, rho)

far = jnp.abs(grid[:, 0, None] - grid[None, :, 0]) > 6.0
print("max |P|     at d > 2r:", float(jnp.abs(P)[far].max()))
print("max |P_loc| at d > 2r:", float(jnp.abs(P_loc)[far].max()))

# Localized gain: taper both the cross- and obs-space covariances.
obs_particles = ensemble[:, obs_idx]
R = lx.DiagonalLinearOperator(0.25 * jnp.ones(obs_idx.shape[0]))
rho_xy = flx.localization_matrix(grid, obs_coords, radius=3.0)
rho_yy = flx.localization_matrix(obs_coords, obs_coords, radius=3.0)
K_loc = flx.localized_kalman_gain(ensemble, obs_particles, R, rho_xy, rho_yy)
print("K_loc shape:", K_loc.shape)
```

```
max |P|     at d > 2r: 1.0718183517456055
max |P_loc| at d > 2r: 0.0
K_loc shape: (40, 10)
```

The raw sample covariance carries $O(1)$ spurious entries between
points more than $2r$ apart; after the Gaspari-Cohn Schur product
they are identically zero, and the gain pulls each state point only
toward nearby observations.

## Where next

- [Chapter 4 — Kalman update & ensemble gain](04_kalman_update.md):
  the unlocalized gain this chapter tapers.
- [Chapter 8 — LETKF](08_letkf.md): R-localization — the
  transform-filter alternative to the B-localization shown here.
- [Chapter 12 — Inflation](12_inflation.md): localization's companion
  fix; localization itself contributes to spread deficits that
  inflation must repair.
- [API: Localization](../api/localization.md) — `gaspari_cohn`,
  `gaussian_taper`, `soar_taper`, `hard_cutoff`,
  `localization_matrix`, `localize`, `localized_kalman_gain`,
  `adaptive_localization`.

## References

- Gaspari, G. & Cohn, S. E. (1999). *Construction of correlation
  functions in two and three dimensions.* Q. J. R. Meteorol. Soc.,
  125, 723–757.
- Houtekamer, P. L. & Mitchell, H. L. (2001). *A sequential ensemble
  Kalman filter for atmospheric data assimilation.* Mon. Wea. Rev.,
  129, 123–137.
- Anderson, J. L. (2007). *Exploring the need for localization in
  ensemble data assimilation using a hierarchical ensemble filter.*
  Physica D, 230, 99–111.
- Thiebaux, H. J. & Pedder, M. A. (1987). *Spatial Objective
  Analysis.* Academic Press.
- Hunt, B. R., Kostelich, E. J., & Szunyogh, I. (2007). *Efficient
  data assimilation for spatiotemporal chaos: A local ensemble
  transform Kalman filter.* Physica D, 230, 112–126.

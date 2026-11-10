---
numbering:
  enumerator: "8.%s"
---

# LETKF: the local ensemble transform Kalman filter

A global ETKF on a large grid has two problems, and [chapter 6](06_etkf.md) ends on both.
Its update is confined to the $N_\mathcal{E} - 1$ directions of the forecast anomalies, and its sample covariance contains spurious long-range correlations that those directions then act on.
The LETKF {cite:p}`hunt2007efficient`, which follows the local filter of {cite:t}`ott2004local`, answers both by solving a separate small ETKF at every grid point, using only the observations nearby.

Three things organise this chapter.
The local problem is an ordinary ETKF with one change, a weighted observation precision.
The weights are set by a taper and a hard cutoff whose conventions decide how the analysis behaves in space.
And the weights have numerical consequences for gradients and float32, which are specific to how `gaussx` implements them.

## Why localize

Two independent variables have a true correlation of zero, but their sample correlation over $N_\mathcal{E}$ members has a standard deviation of about $1 / \sqrt{N_\mathcal{E} - 1}$.
A global update acts on all such correlations as if they were real.

```{code-block} python
:caption: The spread of the sample correlation between two independent variables.
:label: code-letkf-spurious
import einx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

jax.config.update("jax_enable_x64", True)

# Shapes: K = 2,000 draws; N_E = 10 members; two variables

# x, x′ ~ 𝒩(0, 1), independent
pairs = jr.normal(jr.key(0), (2000, 10, 2))  # → (K, N_E, 2)
# anomalies of each pair
pair_anomalies = einx.subtract(  # → (K, N_E, 2)
    "draw member pair, draw pair -> draw member pair",
    pairs,
    einx.mean("draw [member] pair", pairs),
)
# ∑ₑ x̃⁽ᵉ⁾ x̃′⁽ᵉ⁾
product_sum = einx.sum(  # → (K,)
    "draw [member]", pair_anomalies[:, :, 0] * pair_anomalies[:, :, 1]
)
# ∑ₑ x̃⁽ᵉ⁾² for each variable
square_sums = einx.sum("draw [member] pair", pair_anomalies**2)  # → (K, 2)
# the sample correlation
correlations = product_sum / jnp.sqrt(square_sums[:, 0] * square_sums[:, 1])  # → (K,)
print(f"{jnp.std(correlations):.3f} {1.0 / jnp.sqrt(9.0):.3f}")
```

It prints `0.335 0.333`.
With ten members, a correlation of one third between unrelated sites is an ordinary draw.

## The local problem

Attach a coordinate $\vect{s}_i \in \mathbb{R}^{D_s}$ to each state component $i \in \{1, \dots, D_x\}$ and a coordinate $\vect{s}_k \in \mathbb{R}^{D_s}$ to each observation $k \in \{1, \dots, D_y\}$.
Their Euclidean distance is $d_{ik} = \Vert \vect{s}_i - \vect{s}_k \Vert_2$.
Grid point $i$ gets one weight for every observation, collected in $\gvect{\rho}_i \in [0, 1]^{D_y}$ and in the diagonal matrix $\mat{D}_{\rho,i} = \op{diag}(\gvect{\rho}_i)$:

$$[\gvect{\rho}_i]_k = \begin{cases} \rho(d_{ik}; \ell) & d_{ik} \le \ell, \\ 0 & d_{ik} > \ell, \end{cases}$$ (eq-letkf-weights)

where $\rho$ is the taper and $\ell$ is the `radius` argument.
Localization then replaces the observation precision by a weighted one:

$$\gmat{\Sigma}_\varepsilon^{-1} \to \mat{D}_{\rho,i}^{1/2} \gmat{\Sigma}_\varepsilon^{-1} \mat{D}_{\rho,i}^{1/2}.$$ (eq-letkf-weighted-precision)

This is called R-localization.
For a diagonal $\gmat{\Sigma}_\varepsilon$ it divides each variance by its weight, so the entry becomes $[\gvect{\rho}_i]_k / [\gmat{\Sigma}_\varepsilon]_{kk}$.
A distant observation is treated as less certain, and at weight zero it is gone.

The rest is [chapter 6](06_etkf.md) with the same anomalies $\tilde{\mat{X}} \in \mathbb{R}^{N_\mathcal{E} \times D_x}$ and $\tilde{\mat{H}} \in \mathbb{R}^{N_\mathcal{E} \times D_y}$, mean residual $\bar{\vect{r}}$ and $c = N_\mathcal{E} - 1$.
Each point has its own weight precision, mean weights and transform, all in ensemble space:

$$\gmat{\Lambda}_{w,i} = c \mat{I} + \tilde{\mat{H}} \mat{D}_{\rho,i}^{1/2} \gmat{\Sigma}_\varepsilon^{-1} \mat{D}_{\rho,i}^{1/2} \tilde{\mat{H}}^\top, \qquad \gmat{\Sigma}_{w,i} = \gmat{\Lambda}_{w,i}^{-1},$$ (eq-letkf-precision)

$$\bar{\vect{w}}_i = \gmat{\Sigma}_{w,i} \tilde{\mat{H}} \mat{D}_{\rho,i}^{1/2} \gmat{\Sigma}_\varepsilon^{-1} \mat{D}_{\rho,i}^{1/2} \bar{\vect{r}}, \qquad \mat{W}_i = (c \gmat{\Sigma}_{w,i})^{1/2},$$ (eq-letkf-local-weights)

with $\bar{\vect{w}}_i \in \mathbb{R}^{N_\mathcal{E}}$ and $\mat{W}_i \in \mathbb{R}^{N_\mathcal{E} \times N_\mathcal{E}}$.
The weighted problem is the unweighted filter run on $\tilde{\mat{H}} \mat{D}_{\rho,i}^{1/2}$ and $\mat{D}_{\rho,i}^{1/2} \bar{\vect{r}}$, and this is how gaussx computes it.
So everything chapter 6 proves about one transform holds at each point, and in particular $\mat{W}_i \vect{1} = \vect{1}$.

With $\tilde{\vect{x}}_{[i]} \in \mathbb{R}^{N_\mathcal{E}}$ for column $i$ of $\tilde{\mat{X}}$, component $i$ of the analysis combines the global anomalies with the local weights:

$$[\bar{\vect{x}}_{t \mid t}]_i = [\bar{\vect{x}}]_i + \tilde{\vect{x}}_{[i]}^\top \bar{\vect{w}}_i, \qquad [\tilde{\mat{X}}_{t \mid t}]_{[i]} = \mat{W}_i \tilde{\vect{x}}_{[i]}.$$ (eq-letkf-analysis)

### What locality buys

In the global ETKF the increment $\tilde{\mat{X}}^\top \bar{\vect{w}}$ is a single combination of the forecast anomalies, so it lies in their span.
In [](#eq-letkf-analysis) every component takes its own combination, so the increment as a field is no longer in that span.
This is why localization fixes the rank problem, and not just the spurious correlations: each point needs the ensemble to resolve only its local errors.
[](#code-letkf-analysis) measures that fraction.

## The taper, the cutoff and the half-width

Three conventions meet in [](#eq-letkf-weights), and they do not line up the way the literature might suggest.

**`radius` is the half-width of the taper, and also the hard cutoff.**
filterax defines its tapers by a half-width: `gaspari_cohn(d, radius)` {cite:p}`gaspari1999construction` is piecewise rational in $z = d / \ell$, with a fifth-order polynomial on $z \le 1$ and a branch with a $-2 / (3 z)$ term on $1 < z < 2$, and it is zero for every $d \ge 2 \ell$.
`LETKF` then drops every observation beyond $d = \ell$ by the explicit cutoff.
So the second branch of the polynomial, on $1 < z \le 2$, is never reached, and the default weight is only the first branch:

$$\rho(d; \ell) = -\frac{1}{4} z^5 + \frac{1}{2} z^4 + \frac{5}{8} z^3 - \frac{5}{3} z^2 + 1, \qquad z = \frac{d}{\ell} \le 1.$$ (eq-letkf-gaspari-cohn)

**The default weight jumps at the cutoff.**
At $z = 1$ this polynomial equals $5 / 24$, not zero.
So the weight falls from about 0.21 to exactly 0 as an observation crosses $d = \ell$, and with a strong observation that is a visible step in the analysis.
A smooth transition from one grid point to the next is the usual argument for tapering in the LETKF, and it does not hold for the default.

**The other uses of the same taper reach further.**
`localization_matrix(a, b, r)` of [chapter 11](11_localization.md) has no cutoff, so at one value of the radius its taper reaches $2 r$ where `LETKF` reaches $r$.

To make the weight continuous, pass a taper whose support ends at the cutoff.
With half of `radius` as the half-width inside the taper, the weight reaches zero exactly at `radius`, and the cutoff then changes nothing.

```{code-block} python
:caption: The tapers at the cutoff, and the mean increment along a line from one observation, with and without a step.
:label: code-letkf-cutoff
import filterax as flx
import gaussx
import lineax as lx
import numpy as np

# Shapes: N_E = 20 members; D_x = 41 grid points; D_y = 1 observation; D_s = 1


# user
# ρ(d; r / 2), so that the taper's support ends at the cutoff r
def full_support(distances: Float[Array, "..."], radius: float) -> Float[Array, "..."]:
    # today (filterax)
    return flx.gaspari_cohn(distances, 0.5 * radius)


one = jnp.asarray(1.0)
for name, taper in (
    ("gaspari_cohn", flx.gaspari_cohn),
    ("gaussian_taper", flx.gaussian_taper),
    ("soar_taper", flx.soar_taper),
    ("hard_cutoff", flx.hard_cutoff),
    ("full_support", full_support),
):
    print(name, f"{taper(one, 1.0):.4f}")
# today (filterax)
# ρ_xy at d = 1.5 r, which LETKF would cut
past_cutoff = flx.localization_matrix(jnp.zeros((1, 1)), jnp.full((1, 1), 1.5), 1.0)
print(f"{past_cutoff[0, 0]:.4f}")

# s₁, …, s_Dx on [0, 4]
grid = jnp.linspace(0.0, 4.0, 41)  # → (D_x,)
grid_coords = einx.id("point -> point 1", grid)  # → (D_x, D_s)
# s₁ = 0
line_obs_coords = jnp.zeros((1, 1))  # → (D_y, D_s)
# four smooth modes on the grid
modes = jnp.stack(  # → (4, D_x)
    [jnp.ones(41), jnp.cos(0.7 * grid), jnp.sin(0.7 * grid), jnp.cos(0.3 * grid)]
)
# x⁽ᵉ⁾, smooth along the line
smooth_members = einx.dot(  # → (N_E, D_x)
    "member mode, mode point -> member point", jr.normal(jr.key(0), (20, 4)), modes
)
# y = x̄₁ + 1
line_target = einx.mean("[member] point", smooth_members)[:1] + 1.0  # → (D_y,)
# today (lineax)
line_noise = lx.DiagonalLinearOperator(jnp.array([0.1]))  # (D_y, D_y) operator


# user
# h(x) = x₁
def observe_first(state: Float[Array, " point"]) -> Float[Array, " obs"]:
    return state[:1]


for name, taper in (("gaspari_cohn", flx.gaspari_cohn), ("full_support", full_support)):
    # today (filterax)
    line_filter = flx.filters.LETKF(radius=2.0, taper_fn=taper)
    # today (jax)
    local = jax.jit(
        lambda ensemble: line_filter.analysis(
            ensemble,
            line_target,
            observe_first,
            line_noise,
            state_coords=grid_coords,
            obs_coords=line_obs_coords,
        ).particles
    )(smooth_members)
    # x̄_t|t − x̄
    increment = einx.mean("[member] point", local - smooth_members)
    steps = jnp.abs(jnp.diff(increment))  # → (D_x − 1,)
    print(
        name,
        f"{jnp.abs(increment).max():.3f} {steps.max():.3f}",
        f"{grid[int(steps.argmax())]:.2f}",
    )
```

The first five lines print `0.2083`, `0.6065`, `0.7358`, `1.0000` and `0.0000`, the weight at the cutoff for each taper in turn.
The next prints `0.0165`, which is what `localization_matrix` still gives at $1.5 r$.
Along the line, it prints `0.970 0.430 2.00` for the default and `0.970 0.131 1.40` for the full-support taper.
Those are the largest increment, the largest change between neighbouring points 0.1 apart, and where that change occurs.
With the default, 0.430 of an increment of at most 0.970 disappears between two neighbouring points, exactly at $d = \ell$.
With the full-support taper, the largest step is the smooth slope of the taper, well inside the cutoff.

### Which observations count

The `taper_fn` receives the distances and `radius` unchanged.
So a Gaussian or SOAR taper is cut at $d = \ell$ in the same way, at weights of about 0.61 and 0.74.
Only the Layer 1 filter has this argument: the Layer 2 `filterax.LETKF` takes `radius` and always uses the default.

The distance is Euclidean in the coordinates given, and `LETKF` has no metric argument.
A periodic or spherical domain needs coordinates in which the Euclidean distance means what you want.
Those are points on a circle for a ring, and points on a sphere for the globe.
[](#code-letkf-twin) does this for a ring and compares it with plain coordinates.

## Why the noise must be diagonal

`LETKF` raises `NotImplementedError` for anything but a bare `lx.DiagonalLinearOperator`, which also excludes a diagonal covariance wrapped in a tag or written as `lx.IdentityLinearOperator`.
The reason for the limit is in [](#eq-letkf-weighted-precision).

Write $S$ for the observations with non-zero weight.
Removing the others properly leaves the precision $([\gmat{\Sigma}_\varepsilon]_{SS})^{-1}$, the inverse of the block.
A zero weight instead leaves $[\gmat{\Sigma}_\varepsilon^{-1}]_{SS}$, the block of the inverse, which is a different matrix unless $\gmat{\Sigma}_\varepsilon$ is diagonal.
So with correlated noise a weight of zero removes the observation's residual and anomalies, and still changes how the others are read.
The off-diagonal terms of the precision also scale with $\sqrt{\rho}$ rather than $\rho$, as [gaussx#787](https://github.com/jejjohnson/gaussx/issues/787) records.

`gaussx.etkf_transform` itself accepts any noise operator with `obs_weights`.
[](#code-letkf-correlated) compares a zero weight against dropping the observation, for both kinds of noise.

```{code-block} python
:caption: A zero weight against dropping the observation, for diagonal and correlated noise.
:label: code-letkf-correlated
# Shapes: N_E = 12 members; D_y = 3 observations; |S| = 2 kept

# h⁽ᵉ⁾
three_mapped = jr.normal(jr.key(1), (12, 3))  # → (N_E, D_y)
# y
three_target = jnp.array([0.4, -0.1, 0.7])  # → (D_y,)
# ρ
keep_two = jnp.array([1.0, 1.0, 0.0])  # → (D_y,)
# Σ_ε, correlated
correlated = jnp.array([[0.3, 0.1, 0.0], [0.1, 0.4, 0.1], [0.0, 0.1, 0.5]])
# Σ_ε, with the same variances
diagonal = jnp.diagonal(correlated)  # → (D_y,)
for full_noise, kept_noise in (
    # today (lineax)
    (lx.DiagonalLinearOperator(diagonal), lx.DiagonalLinearOperator(diagonal[:2])),
    # today (lineax)
    (
        lx.MatrixLinearOperator(correlated, lx.positive_semidefinite_tag),
        lx.MatrixLinearOperator(correlated[:2, :2], lx.positive_semidefinite_tag),
    ),
):
    # today (gaussx)
    weighted = gaussx.etkf_transform(
        three_mapped, three_target, full_noise, obs_weights=keep_two
    )
    # today (gaussx)
    dropped = gaussx.etkf_transform(three_mapped[:, :2], three_target[:2], kept_noise)
    print(
        f"{jnp.abs(weighted[0] - dropped[0]).max():.0e}",
        f"{jnp.abs(weighted[1] - dropped[1]).max():.0e}",
    )
```

It prints `1e-17 2e-16` for diagonal noise and `3e-04 4e-03` for correlated noise, in the mean weights and then the transform.
Decorrelating the observations first, as in [chapter 7](07_ensrf_estkf.md), makes the noise diagonal, although the whitened observations are no longer local to one site.

## Cost

The implementation vmaps `gaussx.etkf_transform` over the grid points, and a vmapped function has one static shape.
So the observations with zero weight are not removed: they are multiplied by zero and stay in the arrays.
Per grid point the work therefore follows the global $D_y$, and the local count of observations does not reduce it.

| Quantity | Size or cost |
| --- | --- |
| One local transform, $D_y \ge N_\mathcal{E}$ | $O(N_\mathcal{E}^2 D_y + N_\mathcal{E}^3)$ |
| One local transform, $D_y < N_\mathcal{E}$ | $O(N_\mathcal{E} D_y^2 + N_\mathcal{E}^2 D_y)$ |
| All the transforms held at once | $D_x N_\mathcal{E}^2$ numbers |
| Applying them to the anomalies | $O(D_x N_\mathcal{E}^2)$ |
| Grid points | $D_x$ independent solves, in one `jax.vmap` |

The whole analysis costs $D_x$ local transforms against one for the global filter, and the array of transforms is the memory limit on a large grid.

## B-localization and R-localization

There are two places to suppress spurious correlations, and they are not the same operation.

Covariance, or B-localization ([chapter 11](11_localization.md)), takes the Schur product of the sample covariances with a taper matrix.
It acts in state space, it suits gain-based filters, and it comes through `localized_kalman_gain`.

R-localization, as here, leaves the sample covariance untouched and reweights the observation precision at each point.
It acts in observation space, it keeps every local problem an ETKF in $N_\mathcal{E}$ dimensions, and it suits transform filters, whose algebra never forms $\hat{\gmat{\Sigma}}_x$.

{cite:t}`greybush2011balance` compare the two, and study how tighter localization damages dynamical balance.
The practical rule is that a gain-based filter takes B-localization and a transform filter takes R-localization.

## Algorithm

[](#code-letkf-algorithm) writes one analysis out in dense numpy, dropping the observations beyond the cutoff instead of multiplying them by zero.
That makes it an independent check on both the weights and the claim that a zero weight removes an observation.

```{code-block} python
:caption: The LETKF as a dense numpy reference, one grid point at a time.
:label: code-letkf-algorithm
# user
def letkf_by_hand(
    ensemble, mapped, target, variances, state_points, obs_points, radius
):
    num_members, num_points = ensemble.shape
    # c = N_E − 1
    prior_term = num_members - 1.0
    # x̄ and X̃
    state_mean = ensemble.mean(0)
    state_anomalies = ensemble - state_mean
    # H̃ and r̄ = y − h̄
    mapped_anomalies = mapped - mapped.mean(0)
    residual = target - mapped.mean(0)
    analysis = np.empty_like(ensemble)
    for point in range(num_points):
        # dᵢₖ = ‖sᵢ − sₖ‖₂
        distances = np.linalg.norm(obs_points - state_points[point], axis=1)
        # the observations within the cutoff
        kept = distances <= radius
        # ρ(dᵢₖ; ℓ) / σ²_ε,k
        # today (filterax)
        precision = np.asarray(flx.gaspari_cohn(jnp.asarray(distances[kept]), radius))
        precision = precision / variances[kept]
        local_anomalies = mapped_anomalies[:, kept]
        # Λ_w,i = c I + H̃ D_ρ^½ Σ_ε⁻¹ D_ρ^½ H̃ᵀ
        weighted_anomalies = local_anomalies * precision
        weight_precision = prior_term * np.eye(num_members)
        weight_precision = weight_precision + weighted_anomalies @ local_anomalies.T
        # Λ_w,i = Φ diag(γ) Φᵀ
        eigenvalues, eigenvectors = np.linalg.eigh(weight_precision)
        # w̄ᵢ = Σ_w,i H̃ D_ρ^½ Σ_ε⁻¹ D_ρ^½ r̄
        projected = weighted_anomalies @ residual[kept]
        mean_weights = eigenvectors @ ((eigenvectors.T @ projected) / eigenvalues)
        # Wᵢ = Φ diag(√(c / γ)) Φᵀ
        transform = (eigenvectors * np.sqrt(prior_term / eigenvalues)) @ eigenvectors.T
        # x̄ᵢ + x̃ᵀ w̄ᵢ
        point_mean = state_mean[point] + state_anomalies[:, point] @ mean_weights
        # x̄_t|t,i + Wᵢ x̃
        analysis[:, point] = point_mean + transform @ state_anomalies[:, point]
    return analysis
```

## Implementation in filterax

There are three layers.

- [`gaussx.etkf_transform`](https://jejjohnson.github.io/gaussx/) with `obs_weights` is one local solve, returning $\bar{\vect{w}}_i$ and $\mat{W}_i$. Applying it with one row of weights per grid point is the local ETKF.
- [`filterax.filters.LETKF`](https://jejjohnson.github.io/filterax/reference/filters/) is the Layer 1 analysis step. It takes `radius` and `taper_fn`, and it needs `state_coords` and `obs_coords` as keywords.
- `filterax.LETKF` is the Layer 2 cycle of [chapter 16](16_assimilation_cycle.md), which takes `radius` and an `inflator`, and receives the coordinates at `assimilate`.

The example has $N_\mathcal{E} = 12$ members on a line of $D_x = 16$ points, observing points 2, 5 and 11 with `radius` $= 2.5$.
So points 8, 14 and 15 have no observation within reach.

```{code-block} python
:caption: One analysis, against the global ETKF, and the part of each increment outside the span of the forecast anomalies.
:label: code-letkf-analysis
# Shapes: N_E = 12 members; D_x = 16 points; D_y = 3 observations; D_s = 1

# x⁽ᵉ⁾ ~ 𝒩(𝟏, 0.25 I)
members = 1.0 + 0.5 * jr.normal(jr.key(1), (12, 16))  # → (N_E, D_x)
# sᵢ
state_coords = einx.id("point -> point 1", jnp.arange(16.0))  # → (D_x, D_s)
# the observed points
indices = jnp.array([2, 5, 11])  # → (D_y,)
# sₖ
obs_coords = state_coords[indices]  # → (D_y, D_s)
# y
observation = jnp.array([0.4, -0.1, 0.7])  # → (D_y,)
variances = jnp.array([0.1, 0.2, 0.5])  # → (D_y,)
# today (lineax)
# Σ_ε = diag(σ²_ε)
noise = lx.DiagonalLinearOperator(variances)  # (D_y, D_y) operator


# user
# h(x) = x at the observed points
def observe(state: Float[Array, " point"]) -> Float[Array, " obs"]:
    return state[indices]


# today (filterax)
local = flx.filters.LETKF(radius=2.5).analysis(
    members,
    observation,
    observe,
    noise,
    state_coords=state_coords,
    obs_coords=obs_coords,
)
# today (filterax)
global_etkf = flx.filters.ETKF().analysis(members, observation, observe, noise)


# user
# the mean |x_t|t − x| at each point
def update_size(analysis: Float[Array, "member point"]) -> Float[Array, " point"]:
    return einx.mean("[member] point", jnp.abs(analysis - members))  # → (D_x,)


# user
# the fraction of the mean increment outside the span of X̃
def outside_span(analysis: Float[Array, "member point"]) -> Array:
    # X̃ᵀ
    basis = einx.id("member point -> point member", members - members.mean(0))
    # x̄_t|t − x̄
    increment = einx.mean("[member] point", analysis - members)  # → (D_x,)
    # today (jax)
    weights = jnp.linalg.lstsq(basis, increment)[0]  # → (N_E,)
    remainder = increment - basis @ weights  # → (D_x,)
    return jnp.linalg.norm(remainder) / jnp.linalg.norm(increment)  # → ()


print(update_size(local.particles).round(3))
print(update_size(global_etkf.particles).round(3))
print(f"{outside_span(global_etkf.particles):.0e} {outside_span(local.particles):.3f}")
print(jnp.abs(local.particles - members)[:, jnp.array([8, 14, 15])].max())
print(f"{local.log_likelihood:.6f} {global_etkf.log_likelihood:.6f}")
```

The update sizes print as:

| Point | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LETKF | 0.05 | 0.006 | 0.541 | 0.13 | 0.208 | 0.669 | 0.209 | 0.195 | 0 | 0.034 | 0.014 | 0.083 | 0.012 | 0.003 | 0 | 0 |
| ETKF | 0.053 | 0.033 | 0.507 | 0.139 | 0.281 | 0.582 | 0.161 | 0.178 | 0.218 | 0.085 | 0.236 | 0.049 | 0.159 | 0.173 | 0.627 | 0.293 |

The observed points 2, 5 and 11 get a similar correction from both, and the far points are moved by the global ETKF purely through sample correlations.
Point 14 moves by 0.627 under the global filter and not at all under the local one.

It then prints `6e-16 0.598`: the global increment lies in the span of the anomalies to round-off, and a fraction 0.598 of the local increment lies outside it.
The three unreached points print `0.0`, so they keep their forecast exactly.

The last line, `-4.006959 -4.006959`, shows that `result.log_likelihood` is the same global evidence as the ETKF's.
It uses all the observations with no weights, so it does not describe any local problem.

## Agreement with the references

Three references apply.
The dense loop of [](#code-letkf-algorithm) must match, which also shows that a weight of zero removes an observation for diagonal noise.
An explicit `jax.vmap` over `gaussx.etkf_transform` must match, since that is what the filter does.
And a radius large enough to reach everything, with a taper of 1, must give back the global ETKF.

```{code-block} python
:caption: The LETKF against a dense loop, an explicit vmap, and the global ETKF in the limit of no localization.
:label: code-letkf-agreement
# h⁽ᵉ⁾ = h(x⁽ᵉ⁾)
mapped = jax.vmap(observe)(members)  # → (N_E, D_y)
dense = letkf_by_hand(
    np.asarray(members),
    np.asarray(mapped),
    np.asarray(observation),
    np.asarray(variances),
    np.asarray(state_coords),
    np.asarray(obs_coords),
    2.5,
)
print(f"{np.abs(np.asarray(local.particles) - dense).max():.0e}")


# user
# (w̄ᵢ, Wᵢ) at one grid point
def local_transform(point: Float[Array, " dim"]) -> tuple[Array, Array]:
    # sₖ − sᵢ
    offsets = einx.subtract(  # → (D_y, D_s)
        "obs dim, dim -> obs dim", obs_coords, point
    )
    # dᵢₖ = ‖sᵢ − sₖ‖₂
    distances = jnp.sqrt(einx.sum("obs [dim]", offsets**2))  # → (D_y,)
    # ρ, cut at ℓ
    # today (filterax)
    weights = jnp.where(distances <= 2.5, flx.gaspari_cohn(distances, 2.5), 0.0)
    # today (gaussx)
    return gaussx.etkf_transform(mapped, observation, noise, obs_weights=weights)


# today (jax)
all_means, all_transforms = jax.vmap(local_transform)(state_coords)
# today (filterax)
anomalies = flx.ensemble_anomalies(members)  # → (N_E, D_x)
# [x̄_t|t]ᵢ = [x̄]ᵢ + x̃ᵀ w̄ᵢ
explicit_mean = einx.mean("[member] point", members) + einx.dot(  # → (D_x,)
    "point member, member point -> point", all_means, anomalies
)
# [X̃_t|t]ᵢ = Wᵢ x̃
explicit_anomalies = einx.dot(  # → (N_E, D_x)
    "point member other, other point -> member point", all_transforms, anomalies
)
explicit = einx.add(  # → (N_E, D_x)
    "point, member point -> member point", explicit_mean, explicit_anomalies
)
print(f"{jnp.abs(explicit - local.particles).max():.0e}")
# Wᵢ 𝟏 = 𝟏 at every point
fixed = einx.dot(  # → (D_x, N_E)
    "point member other, other -> point member", all_transforms, jnp.ones(12)
)
print(f"{jnp.abs(fixed - 1.0).max():.0e}")

# today (filterax)
unlocalized = flx.filters.LETKF(radius=1e6, taper_fn=flx.hard_cutoff).analysis(
    members,
    observation,
    observe,
    noise,
    state_coords=state_coords,
    obs_coords=obs_coords,
)
print(f"{jnp.abs(unlocalized.particles - global_etkf.particles).max():.0e}")
```

It prints `1e-15`, `0e+00`, `9e-16` and `6e-16`, all of them round-off in float64.
The same `jax.vmap` is the production code, so its second line is exactly zero.

## A Lorenz-96 twin

[Chapter 6](06_etkf.md) showed the global ETKF diverging on Lorenz-96 at 20 members {cite:p}`lorenz2006predictability`.
[](#code-letkf-twin) repeats that twin, with $D_x = 40$ sites on a ring, every second site observed with unit variance, and no inflation unless stated.

The ring needs a coordinate of its own, since site 0 and site 39 are neighbours.
Placing the sites on a circle of circumference 40 in $D_s = 2$ makes the Euclidean chord a good stand-in for the distance along the ring.

```{code-block} python
:caption: The LETKF against the global ETKF on Lorenz-96, for one realisation of each run.
:label: code-letkf-twin
# Shapes: T = 200 windows; D_x = 40 sites; D_y = 20 observed sites; D_s = 2; N_E members


# user
# dxₖ/dt = (xₖ₊₁ − xₖ₋₂) xₖ₋₁ − xₖ + 8
def tendency(state: Float[Array, " state"]) -> Float[Array, " state"]:
    return (jnp.roll(state, -1) - jnp.roll(state, 2)) * jnp.roll(state, 1) - state + 8.0


# user
# one RK4 step of Δt = 0.05
def rk4(state: Float[Array, " state"], _=None, __=None) -> Float[Array, " state"]:
    k1 = tendency(state)
    k2 = tendency(state + 0.025 * k1)
    k3 = tendency(state + 0.025 * k2)
    k4 = tendency(state + 0.05 * k3)
    return state + 0.05 / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# user
# xₜ₊₁ = f(xₜ), stacked
def advance(state: Float[Array, " state"], _) -> tuple[Array, Array]:
    next_state = rk4(state)
    return next_state, next_state


# x₀ on the attractor, after 1,000 steps from a perturbed rest state
spun_up = jax.lax.fori_loop(  # → (D_x,)
    0, 1000, lambda _, state: rk4(state), 8.0 + jr.normal(jr.key(0), (40,))
)
_, truth = jax.lax.scan(advance, spun_up, length=200)  # → (T, D_x)
# H selects every second site
sites = jnp.arange(0, 40, 2)  # → (D_y,)
# yₜ = H xₜ + εₜ, εₜ ~ 𝒩(0, I)
observed = truth[:, sites] + jr.normal(jr.key(1), (200, 20))  # → (T, D_y)
unit_noise = lx.DiagonalLinearOperator(jnp.ones(20))  # (D_y, D_y) operator
# the sites on a circle of circumference 40
angles = 2.0 * jnp.pi * jnp.arange(40.0) / 40.0  # → (D_x,)
ring = jnp.stack([jnp.cos(angles), jnp.sin(angles)], -1) * 40.0 / (2.0 * jnp.pi)
# the sites on a line, where sites 0 and 39 are far apart
line = einx.id("site -> site 1", jnp.arange(40.0))  # → (D_x, 1)
# ‖s₁ − s₀‖₂ and ‖s₂₀ − s₀‖₂
neighbour_chord = jnp.linalg.norm(ring[1] - ring[0])  # → ()
farthest_chord = jnp.linalg.norm(ring[20] - ring[0])  # → ()
print(f"{neighbour_chord:.3f} {farthest_chord:.2f}")


# user
# h(x) = H x
def observe_sites(state: Float[Array, " state"]) -> Float[Array, " obs"]:
    return state[sites]


# user
# forecast and analysis over all windows, compiled once by the scan
def run_twin(chosen, num_members, coords=None, inflator=None, dtype=jnp.float64):
    extra = {} if coords is None else {
        "state_coords": coords.astype(dtype),
        "obs_coords": coords[sites].astype(dtype),
    }
    # today (lineax)
    window_noise = lx.DiagonalLinearOperator(jnp.ones(20, dtype))  # (D_y, D_y) operator
    # x⁽ᵉ⁾ ~ 𝒩(x₀, 4 I)
    start = spun_up + 2.0 * jr.normal(jr.key(2), (num_members, 40))  # → (N_E, D_x)

    def window(particles, target):
        # X_t|t−1 = f(X_t−1|t−1)
        forecast = jax.vmap(rk4)(particles)  # → (N_E, D_x)
        # today (filterax)
        analysis = chosen.analysis(
            forecast, target, observe_sites, window_noise, **extra
        ).particles
        if inflator is not None:
            analysis = inflator(analysis, forecast)  # → (N_E, D_x)
        return analysis, analysis

    _, history = jax.lax.scan(window, start.astype(dtype), observed.astype(dtype))
    settled = history[100:]  # → (T / 2, N_E, D_x)
    # x̄_t|t
    settled_mean = einx.mean("t [member] state", settled)  # → (T / 2, D_x)
    # RMSE of x̄_t|t against x_true
    rmse = jnp.sqrt(jnp.mean((settled_mean - truth[100:]) ** 2))  # → ()
    # x̃⁽ᵉ⁾_t|t
    anomalies = einx.subtract(  # → (T / 2, N_E, D_x)
        "t member state, t state -> t member state", settled, settled_mean
    )
    # √(mean diag Σ̂_x|y)
    mean_spread = jnp.sqrt(jnp.sum(anomalies**2) / (100 * (num_members - 1) * 40))
    return f"{rmse:.2f} {mean_spread:.2f}"


# today (filterax)
global_filter = flx.filters.ETKF()
for num_members in (5, 10, 20):
    print(num_members, "ETKF", run_twin(global_filter, num_members))
    # today (filterax)
    local_filter = flx.filters.LETKF(radius=3.0)
    print(num_members, "LETKF", run_twin(local_filter, num_members, ring))

for radius in (0.5, 5.0, 12.0):
    # today (filterax)
    local_filter = flx.filters.LETKF(radius=radius)
    print("radius", radius, run_twin(local_filter, 10, ring))

# today (filterax)
print("full support", run_twin(flx.filters.LETKF(6.0, full_support), 10, ring))
# today (filterax)
print("line", run_twin(flx.filters.LETKF(radius=5.0), 10, line))
# today (filterax)
print("RTPS", run_twin(flx.filters.LETKF(radius=3.0), 10, ring, flx.RTPS(alpha=0.25)))
# today (filterax)
print("float32", run_twin(flx.filters.LETKF(radius=3.0), 10, ring, None, jnp.float32))

# today (filterax)
cycle = flx.LETKF(dynamics=rk4, obs_op=observe_sites, radius=3.0)
start = spun_up + 2.0 * jr.normal(jr.key(2), (10, 40))  # → (N_E, D_x)
windows = [(observed[t], 0.05 * (t + 1)) for t in range(2)]
# today (filterax)
layer_two = cycle.assimilate(start, windows, unit_noise, ring, ring[sites])
print(layer_two.analysis_history.shape)
```

It prints:

| $N_\mathcal{E}$ | ETKF RMSE | ETKF spread | LETKF RMSE, `radius` 3 | LETKF spread |
| --- | --- | --- | --- | --- |
| 5 | `4.37` | `0.16` | `0.50` | `0.31` |
| 10 | `4.89` | `0.23` | `0.45` | `0.33` |
| 20 | `2.37` | `0.26` | `0.40` | `0.34` |

The global ETKF has diverged at all three sizes, with an error above the unit observation noise.
The LETKF stays near 0.5 even with 5 members, and its spread is still below its error, so it is under-dispersive without inflation.

The first line of output, `0.999 12.73`, gives the chord between neighbours and the largest chord on the ring.
The remaining runs fix $N_\mathcal{E} = 10$:

| Run | RMSE | Spread |
| --- | --- | --- |
| `radius` 0.5 | `0.90` | `0.93` |
| `radius` 3 | `0.45` | `0.33` |
| `radius` 5 | `0.46` | `0.28` |
| `radius` 12 | `3.99` | `0.26` |
| `full_support`, `radius` 6 | `0.43` | `0.32` |
| `radius` 5, on a line | `0.43` | `0.30` |
| `radius` 3, with `RTPS(alpha=0.25)` | `0.39` | `0.41` |
| `radius` 3, in float32 | `0.45` | `0.33` |

The radius has failures at both ends.
At 0.5 only the observed sites see an observation, so the others get no analysis update and the error is near 0.9.
At 12 the radius nearly spans the ring, whose largest chord is 12.73, so the filter is almost global and diverges as the global one does.

The taper with half-width 3 and no step scores about the same as the default with `radius` 3, so on this domain-wide score the step costs little.
Its harm is to the local structure of the field, which [](#code-letkf-cutoff) measures and an RMSE does not.

Cutting the ring at a seam shows no cost in this twin, at `0.43` against `0.46` with `radius` 5.
That gap is too small to read much into from one realisation, and the circle remains the correct distance.

`RTPS` raises the spread to match the error and also lowers the error, which is the subject of [chapter 12](12_inflation.md).
The float32 run scores the same as float64, since the unit variances here are the easy case.
The last print, `(2, 10, 40)`, is the Layer 2 history over two windows.

## Differentiating through the weights

A local analysis is an ETKF, so it inherits the closed-form derivative of [chapter 6](06_etkf.md), at repeated eigenvalues as well.
The weights add three points of their own.

**A weight of exactly zero has a derivative of zero, by convention.**
The factor $\mat{D}_\rho^{1/2}$ has an infinite slope at $\rho = 0$, so gaussx guards it and returns 0 ([gaussx#787](https://github.com/jejjohnson/gaussx/issues/787)).
For diagonal noise that is a convention and not the true answer, because the precision $\rho / \sigma^2$ is linear in $\rho$ and a one-sided derivative exists.
A compactly supported taper has no slope at its end, so differentiating through a taper that reaches zero is unaffected.
Differentiating directly with respect to a weight sitting at zero is the case to avoid.

**The derivative does not see the cutoff.**
With the default taper the weight jumps at $d = \ell$, and no gradient describes a jump.
`radius` is also a static field, and passing it a tracer draws a warning from Equinox, so treat it as a hyperparameter and tune it by running.

**A coordinate that coincides with an observation gives `NaN`.**
The distance is computed as a square root of a sum of squares, whose derivative is infinite at zero.
So `jax.grad` with respect to `obs_coords` or `state_coords` returns `NaN` when a grid point sits on an observation, as in the usual observing network on a grid.
Gradients with respect to anything else are unaffected.

```{code-block} python
:caption: Gradients of the local analysis, and the three points above.
:label: code-letkf-gradient
# Shapes: N_E = 12; D_x = 16; D_y = 3


# user
# 𝓛(s) = ∑ sin X_t|t, with Σ_ε = s diag(σ²_ε)
def noise_loss(scale: Float[Array, ""]) -> Float[Array, ""]:
    # today (lineax)
    scaled = lx.DiagonalLinearOperator(scale * variances)  # (D_y, D_y) operator
    # today (filterax)
    analysis = flx.filters.LETKF(radius=2.5).analysis(
        members,
        observation,
        observe,
        scaled,
        state_coords=state_coords,
        obs_coords=obs_coords,
    )
    return jnp.sum(jnp.sin(analysis.particles))  # → ()


# today (jax)
compiled_noise_loss = jax.jit(noise_loss)
at = jnp.asarray(1.0)
step = 1e-6
central = compiled_noise_loss(at + step) - compiled_noise_loss(at - step)  # → ()
print(f"{jax.jit(jax.grad(noise_loss))(at):.6f} {central / (2.0 * step):.6f}")


# user
# 𝓛(ρ) = ∑ sin W, for one local transform
def weight_loss(weights: Float[Array, " obs"]) -> Float[Array, ""]:
    # today (gaussx)
    _, transform = gaussx.etkf_transform(
        mapped, observation, noise, obs_weights=weights
    )
    return jnp.sum(jnp.sin(transform))  # → ()


# today (jax)
compiled_weight_loss = jax.jit(weight_loss)
# today (jax)
weight_gradient = jax.jit(jax.grad(weight_loss))
at_zero = jnp.array([1.0, 0.5, 0.0])  # → (D_y,)
nudge = jnp.array([0.0, 0.0, 1.0])  # → (D_y,)
forward = compiled_weight_loss(at_zero + 1e-6 * nudge) - compiled_weight_loss(at_zero)
forward = forward / 1e-6  # → ()
gradient_at_zero = weight_gradient(at_zero)[2]  # → ()
gradient_near_zero = weight_gradient(at_zero + 1e-12 * nudge)[2]  # → ()
print(f"{gradient_at_zero:.6f} {gradient_near_zero:.6f} {forward:.6f}")


# user
# 𝓛(δ) = ∑ sin X_t|t, with every observation moved along the line by δ
def coordinate_loss(shift: Float[Array, ""]) -> Float[Array, ""]:
    # today (filterax)
    analysis = flx.filters.LETKF(radius=2.5).analysis(
        members,
        observation,
        observe,
        noise,
        state_coords=state_coords,
        obs_coords=obs_coords + shift,
    )
    return jnp.sum(jnp.sin(analysis.particles))  # → ()


# today (jax)
coordinate_gradient = jax.jit(jax.grad(coordinate_loss))
print(coordinate_gradient(jnp.asarray(0.0)))
print(f"{coordinate_gradient(jnp.asarray(0.3)):.6f}")
```

It prints `1.667363 1.667363` for the noise gradient against its central difference, at the repeated eigenvalues of every local $\gmat{\Lambda}_{w,i}$.
It then prints `0.000000 0.068758 0.068758`: the gradient at a weight of zero, the gradient at $10^{-12}$ and the forward difference.
Finally it prints `nan` with the observations on the grid, and a finite `0.636013` when they are moved off it by 0.3.

## Float32 with more observations than members

The route inside `etkf_transform` is chosen from the array shape, so it counts every observation, whatever its weight.
With $D_y \ge N_\mathcal{E}$, which is the normal situation for a global network, every grid point takes the ensemble-space route, even when only a few observations are within its reach.
That route loses accuracy in float32 when the noise variances span many decades ([gaussx#785](https://github.com/jejjohnson/gaussx/issues/785), and [chapter 6](06_etkf.md)).

[](#code-letkf-float32) builds one local problem with $N_\mathcal{E} = 5$, $D_y = 8$ and only two observations with non-zero weight, and solves it twice.
The first solve carries the zero-weight observations and takes the ensemble-space route.
The second drops them, which takes the rank-$D_y$ route.

```{code-block} python
:caption: One local problem in float32, with the zero-weight observations kept and dropped, against a float64 dense solve.
:label: code-letkf-float32
# Shapes: N_E = 5 members; D_y = 8 observations; 2 with non-zero weight

rng = np.random.default_rng(1)
wide_mapped = rng.normal(size=(5, 8))
wide_target = rng.normal(size=8)
wide_variances = np.logspace(-6, 6, 8)
reach = np.array([1, 6])
reach_weights = np.zeros(8)
reach_weights[reach] = 1.0

# w̄ = Λ_w⁻¹ H̃ D_ρ^½ Σ_ε⁻¹ D_ρ^½ r̄, by a dense float64 solve
wide_anomalies = wide_mapped - wide_mapped.mean(0)
weighted_precision = reach_weights / wide_variances
expected = np.linalg.solve(
    4.0 * np.eye(5) + (wide_anomalies * weighted_precision) @ wide_anomalies.T,
    (wide_anomalies * weighted_precision) @ (wide_target - wide_mapped.mean(0)),
)

# today (gaussx)
kept_weights, _ = gaussx.etkf_transform(
    jnp.asarray(wide_mapped, jnp.float32),
    jnp.asarray(wide_target, jnp.float32),
    # today (lineax)
    lx.DiagonalLinearOperator(jnp.asarray(wide_variances, jnp.float32)),
    obs_weights=jnp.asarray(reach_weights, jnp.float32),
)
# today (gaussx)
dropped_weights, _ = gaussx.etkf_transform(
    jnp.asarray(wide_mapped[:, reach], jnp.float32),
    jnp.asarray(wide_target[reach], jnp.float32),
    # today (lineax)
    lx.DiagonalLinearOperator(jnp.asarray(wide_variances[reach], jnp.float32)),
)
for computed in (kept_weights, dropped_weights):
    error = np.abs(np.asarray(computed) - expected).max() / np.abs(expected).max()
    print(f"{error:.0e}")
```

It prints `5e-03` with the zero-weight observations kept and `5e-08` with them dropped.
So the accuracy limit comes from the global count, and `LETKF` has no way to shrink the arrays under `jax.vmap`.
With ordinary variances there is no such problem, as the float32 run in [](#code-letkf-twin) shows.
With variances spanning many decades, use float64.

## When it breaks

The LETKF removes the rank problem, and the local version of each ETKF limit remains.

- **A radius that is too small or too large.** [](#code-letkf-twin) measures both ends, and the right value depends on the ensemble size.
- **A step in the analysis.** The default weight jumps at `radius`, as [](#code-letkf-cutoff) measures. Use a taper whose support ends at the cutoff if smoothness matters.
- **A wrong distance.** On a periodic or spherical domain, plain coordinates cut the domain at a seam. There is no metric argument, so embed the coordinates, although this twin shows little cost from the seam.
- **Lost spread.** Local analyses still underestimate error, so a cycled run usually needs [chapter 12](12_inflation.md).
- **Broken balance.** Each point solves its own problem, so dynamical balances over scales longer than the radius can be disrupted {cite:p}`greybush2011balance`. Use [chapter 13](13_diagnostics.md) to look for this.
- **Non-local observations.** An observation that integrates over a whole column or the whole domain has no single location, and `obs_coords` requires one.
- **Correlated noise.** It is refused, as above. Decorrelate first, or use the batch filters of chapters [6](06_etkf.md) and [7](07_ensrf_estkf.md).

## When LETKF is the right answer

Use `LETKF` when $D_x \gg N_\mathcal{E}$ on a state with meaningful coordinates and roughly local error correlations, and the noise is diagonal.
That covers most geophysical grids, and the twin above is a small example of it.
The solves are independent, so they suit an accelerator.

It is the wrong tool for a state with no spatial meaning, such as the global parameters of an inverse problem in [chapter 9](09_ensemble_kalman_processes.md).
For a small state with a large enough ensemble, the global filters of chapters [6](06_etkf.md) and [7](07_ensrf_estkf.md) are simpler and cheaper.

## See also

- [Chapter 6](06_etkf.md): the transform solved at every grid point
- [Chapter 7](07_ensrf_estkf.md): the serial EnSRF, which does not form the matrix behind the float32 limit
- [Chapter 11](11_localization.md): B-localization and the tapers
- [Chapter 12](12_inflation.md): keeping spread over cycles
- [Chapter 13](13_diagnostics.md): innovation statistics for tuning the radius
- [Chapter 15](15_latent_da.md): the latent-space filters, where `LatentLETKF` applies no localization today
- [API: filters](https://jejjohnson.github.io/filterax/reference/filters/) and [localization](https://jejjohnson.github.io/filterax/reference/localization/)

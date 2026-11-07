---
numbering:
  enumerator: "11.%s"
---

# Localization

A finite ensemble estimates a $D_x \times D_x$ covariance from $N_\mathcal{E}$ samples, and the entries between unrelated variables come out random instead of zero.
The gain takes those entries at face value, so every observation moves the whole state a little, in directions that are pure sampling noise.
Localization damps covariance entries as a function of the physical distance between variables.
The prior knowledge behind it is that real correlations decay with distance and sampling noise does not.

Three questions organise this chapter.
Why does an element-wise product of a covariance with a taper remain a covariance?
What does a `radius` mean, given that the tapers in filterax read it differently?
And where in the library is a taper applied, given that the covariance form and the observation-weighting form live in different layers?

## Sampling noise in the covariance

Take the state ensemble $\mat{X} \in \mathbb{R}^{N_\mathcal{E} \times D_x}$ with anomalies $\tilde{\mat{X}}$, and the sample covariance $\hat{\gmat{\Sigma}}_x = \tilde{\mat{X}}^\top \tilde{\mat{X}} / (N_\mathcal{E} - 1)$.
Its rank is at most $N_\mathcal{E} - 1$, so it has no information outside that many directions.
Its sample correlations, for components $i$ and $j$ with spreads $\sigma_i$ and $\sigma_j$, are

$$\hat{c}_{ij} = \frac{[\hat{\gmat{\Sigma}}_x]_{ij}}{\sigma_i \sigma_j},$$ (eq-loc-sample-correlation)

and when the true correlation is zero their standard deviation is about $1 / \sqrt{N_\mathcal{E} - 1}$.
That size depends on neither the distance between the variables nor the state dimension.

```{code-block} python
:caption: Sample correlations of independent components, which are all zero in truth.
:label: code-loc-spurious
import einx
import filterax as flx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
from jaxtyping import Array, Float

jax.config.update("jax_enable_x64", True)

# Shapes: N_E = 20 members; D_x = 500 components

# x⁽ᵉ⁾ ~ 𝒩(𝟎, I)
white = jr.normal(jr.key(2), (20, 500))  # → (N_E, D_x)
# today (filterax)
# Σ̂_x
white_covariance = flx.ensemble_covariance(white).as_matrix()  # → (D_x, D_x)
# σᵢ = √[Σ̂_x]ᵢᵢ
white_spread = jnp.sqrt(jnp.diagonal(white_covariance))  # → (D_x,)
# σᵢ σⱼ
spread_product = einx.multiply("i, j -> i j", white_spread, white_spread)
# ĉᵢⱼ = [Σ̂_x]ᵢⱼ / (σᵢ σⱼ)
correlation = white_covariance / spread_product  # → (D_x, D_x)
# ĉᵢⱼ with i < j
off_diagonal = correlation[jnp.triu_indices(500, k=1)]
print(f"{jnp.std(off_diagonal):.3f} {1.0 / jnp.sqrt(19.0):.3f}")
print(f"{jnp.abs(off_diagonal).max():.2f}")
```

It prints `0.229 0.229` and `0.81`.
The measured scatter matches $1 / \sqrt{19}$, and the largest of these false correlations is 0.81.

## The Schur product

The localized covariance is the Schur, or element-wise, product of the sample covariance with a taper matrix $\rho_{xx} \in \mathbb{R}^{D_x \times D_x}$:

$$\hat{\gmat{\Sigma}}_x^\text{loc} = \rho_{xx} \circ \hat{\gmat{\Sigma}}_x, \qquad [\rho_{xx}]_{ij} = \rho(d_{ij}; \ell),$$ (eq-loc-schur)

where $d_{ij}$ is the distance between variables $i$ and $j$ and $\ell$ is the taper's half-width, the `radius` argument.

### The Schur product theorem

If two square matrices are positive semi-definite, so is their Schur product.
The proof fits on a line.
Write each matrix as a sum of outer products, $\rho_{xx} = \sum_k \vect{a}_k \vect{a}_k^\top$ and $\hat{\gmat{\Sigma}}_x = \sum_l \vect{b}_l \vect{b}_l^\top$, and multiply element by element:

$$\rho_{xx} \circ \hat{\gmat{\Sigma}}_x = \sum_{k, l} (\vect{a}_k \circ \vect{b}_l)(\vect{a}_k \circ \vect{b}_l)^\top,$$ (eq-loc-schur-theorem)

which is a sum of outer products, hence positive semi-definite.
So a taper that is itself a valid correlation function can never break the covariance, however hard it cuts.

[](#eq-loc-schur-theorem) also shows two more things.
The rank can grow to the product of the two ranks, so localization adds directions the ensemble did not span.
And with $\rho(0; \ell) = 1$ the diagonal is untouched, so localization keeps every variance and changes only the correlations.

## The tapers

Each taper below takes distances and a `radius` $\ell$, and returns weights with the shape of the distances.
With $z = d / \ell$, the Gaspari-Cohn taper {cite:p}`gaspari1999construction` is a fifth-order piecewise rational function,

$$\rho_\text{GC}(z) = \begin{cases} -\frac{1}{4} z^5 + \frac{1}{2} z^4 + \frac{5}{8} z^3 - \frac{5}{3} z^2 + 1 & 0 \le z \le 1, \\ \frac{1}{12} z^5 - \frac{1}{2} z^4 + \frac{5}{8} z^3 + \frac{5}{3} z^2 - 5 z + 4 - \frac{2}{3 z} & 1 < z \le 2, \\ 0 & z > 2. \end{cases}$$ (eq-loc-gaspari-cohn)

It vanishes at $d = 2 \ell$, which is why filterax calls $\ell$ a half-width.
Its construction by convolution makes it a correlation function in up to three dimensions, which is what the theorem needs.
The other three are

$$\rho_\text{Gauss}(d) = \exp\left(-\frac{d^2}{2 \ell^2}\right), \qquad \rho_\text{SOAR}(d) = \left(1 + \frac{d}{\ell}\right) \exp\left(-\frac{d}{\ell}\right), \qquad \rho_\text{cut}(d) = \mathbb{1}\{d \le \ell\}.$$ (eq-loc-other-tapers)

The Gaussian has no compact support, and neither does the SOAR taper {cite:p}`thiebaux1987spatial`, which is a Matérn-$3/2$ correlation with length scale $\sqrt{3} \ell$.
Both are correlation functions in any dimension.

The hard cutoff is not.
In one dimension its Fourier transform is $2 \sin(\omega \ell) / \omega$, which changes sign.
So by Bochner's theorem it is not positive definite, and the Schur product theorem does not apply to it.

### The same `radius` is not the same width

[](#code-loc-tapers) evaluates all four at the same $\ell$.

```{code-block} python
:caption: The four tapers with $\ell = 1$, at the same distances.
:label: code-loc-tapers
# Shapes: 5 distances

# d, in units of ℓ
distances = jnp.array([0.0, 1.0, 1.5, 2.0, 3.0])  # → (5,)
# today (filterax)
for taper in (flx.gaspari_cohn, flx.gaussian_taper, flx.soar_taper, flx.hard_cutoff):
    # ρ(d; ℓ = 1)
    print(taper.__name__, taper(distances, 1.0).round(4))

# today (filterax)
# ρ_SOAR at d = 5 ℓ and d = 6.64 ℓ
print(flx.soar_taper(jnp.array([5.0, 6.64]), 1.0).round(4))
```

It prints:

| Taper | $d = 0$ | $d = \ell$ | $d = 1.5 \ell$ | $d = 2 \ell$ | $d = 3 \ell$ |
| --- | --- | --- | --- | --- | --- |
| `gaspari_cohn` | 1 | 0.2083 | 0.0165 | 0 | 0 |
| `gaussian_taper` | 1 | 0.6065 | 0.3247 | 0.1353 | 0.0111 |
| `soar_taper` | 1 | 0.7358 | 0.5578 | 0.406 | 0.1991 |
| `hard_cutoff` | 1 | 1 | 0 | 0 | 0 |

At $d = \ell$ the Gaspari-Cohn weight is 0.2083, the Gaussian's is 0.6065 and the SOAR weight is 0.7358.
The same number therefore gives a much narrower taper with `gaspari_cohn` than with the other two.
The SOAR taper has fallen only to 0.0404 at $d = 5 \ell$, and it reaches $0.01$ near $d = 6.64 \ell$.

The Gaussian that matches Gaspari-Cohn follows from their curvature at the origin.
For small $z$, $\rho_\text{GC} \approx 1 - \frac{5}{3} z^2$ and $\rho_\text{Gauss} \approx 1 - d^2 / (2 L^2)$, so a Gaussian with length scale $L$ matches a Gaspari-Cohn taper of half-width $\ell$ when

$$L = \sqrt{\frac{3}{10}} \, \ell.$$ (eq-loc-matched-gaussian)

### Smoothness and positive definiteness

[](#code-loc-smoothness) measures the first claim and the matrices that follow from it.

```{code-block} python
:caption: Derivative jumps of Gaspari-Cohn, its match to a Gaussian, and the smallest eigenvalue of each taper matrix.
:label: code-loc-smoothness
# Shapes: D_x = 40 sites on a line; N_E = 10 members


# user
# ρ_GC(z), with ℓ = 1
def taper_of_z(z: Float[Array, ""]) -> Float[Array, ""]:
    # today (filterax)
    return flx.gaspari_cohn(z, 1.0)


# dᵏρ_GC / dzᵏ for k = 3, 4, 5
third = jax.jit(jax.jacfwd(jax.jacfwd(jax.jacfwd(taper_of_z))))
fourth = jax.jit(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(taper_of_z)))))
fifth = jax.jit(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(taper_of_z))))))
left, right = 1.0 - 1e-9, 1.0 + 1e-9
print(f"z=1 fourth {fourth(left):.3f} {fourth(right):.3f}")
print(f"z=1 fifth {fifth(left):.3f} {fifth(right):.3f}")
print(f"z=2 third {third(2.0 - 1e-9):.3f} {third(2.0 + 1e-9):.3f}")
print(f"z=2 fourth {fourth(2.0 - 1e-9):.3f} {fourth(2.0 + 1e-9):.3f}")
print(f"z=0 third {third(1e-9):.3f}")
# today (filterax)
# d²ρ_SOAR / dd² just either side of d = 0, with ℓ = 1
soar_curvature = jax.jit(jax.jacfwd(jax.jacfwd(lambda d: flx.soar_taper(d, 1.0))))
print(f"soar {soar_curvature(-1e-9):.3f} {soar_curvature(1e-9):.3f}")

# d ∈ [0, 2 ℓ]
span = jnp.linspace(0.0, 2.0, 401)  # → (401,)
# today (filterax)
# |ρ_GC(d; 1) − ρ_Gauss(d; √(3 / 10))|
mismatch = jnp.abs(
    flx.gaspari_cohn(span, 1.0) - flx.gaussian_taper(span, jnp.sqrt(0.3))
)  # → (401,)
print(f"{mismatch.max():.3f}")

# the sites, as a coordinate column
grid = einx.id("site -> site 1", jnp.arange(40.0))  # → (D_x, 1)
# today (filterax)
# d_ij = ‖sᵢ − sⱼ‖₂
separation = flx.euclidean_distance(grid, grid)  # → (D_x, D_x)
# x⁽ᵉ⁾ ~ 𝒩(𝟎, I)
small = jr.normal(jr.key(1), (10, 40))  # → (N_E, D_x)
# today (filterax)
# Σ̂_x
small_covariance = flx.ensemble_covariance(small).as_matrix()  # → (D_x, D_x)
print("sample rank", jnp.linalg.matrix_rank(small_covariance))
# today (filterax)
for taper in (flx.gaspari_cohn, flx.gaussian_taper, flx.soar_taper, flx.hard_cutoff):
    # ρ_xx
    taper_matrix = taper(separation, 3.0)  # → (D_x, D_x)
    # today (filterax)
    # ρ_xx ∘ Σ̂_x
    localized = flx.localize(small_covariance, taper_matrix)  # → (D_x, D_x)
    print(
        taper.__name__,
        f"{jnp.linalg.eigvalsh(taper_matrix).min():.3f}",
        f"{jnp.linalg.eigvalsh(localized).min():.3f}",
        jnp.linalg.matrix_rank(localized),
    )
```

The derivatives of Gaspari-Cohn print as follows.
At $z = 1$ the fourth derivatives agree, at -18.000 and -18.000, and the fifth jumps from -30.000 to 90.000.
At $z = 2$ the third derivative is continuous at 0.000 and the fourth jumps from 7.500 to 0.000.
The term $\frac{5}{8} z^3$, read with $z = |d| / \ell$, makes the third derivative 3.750 just right of the origin and its negative just left of it.
So the taper is $C^2$ as a function of position, and the origin is the point that limits it.
The SOAR taper is also $C^2$, since its second derivative at the origin is -1.000 from either side at $\ell = 1$, and only the third derivative jumps.

The largest gap between Gaspari-Cohn and the matched Gaussian [](#eq-loc-matched-gaussian) on $[0, 2 \ell]$ is 0.034.

The sample covariance has rank 9, and at $\ell = 3$ the matrices are:

| Taper | Smallest eigenvalue of $\rho_{xx}$ | Smallest eigenvalue of $\rho_{xx} \circ \hat{\gmat{\Sigma}}_x$ | Rank of the product |
| --- | --- | --- | --- |
| `gaspari_cohn` | 0.007 | 0.187 | 40 |
| `gaussian_taper` | 0.000 | 0.075 | 40 |
| `soar_taper` | 0.003 | 0.055 | 40 |
| `hard_cutoff` | -1.581 | -0.169 | 40 |

The three correlation functions give a full-rank, positive definite covariance from a rank-9 one.
The hard cutoff gives an indefinite one.

## Distances and the taper matrix

`filterax.localization_matrix(coords_a, coords_b, radius, metric=...)` builds $\rho(d(\vect{s}_i, \vect{s}_j); \ell)$ for two sets of coordinates, and always with the Gaspari-Cohn taper.
It calls `gaussx.localization_matrix` with compact-support argument `c = 2 * radius`, since gaussx parametrises by support and filterax by half-width.
The default metric is `euclidean_distance`.
`haversine_distance` takes $(\text{latitude}, \text{longitude})$ in radians and returns metres on a sphere of radius $6.371 \times 10^6$ m by default, so $\ell$ must then be in metres too.
A metric is any function `(coords_a, coords_b) -> distances`.

A periodic domain needs its own metric, and with it the theorem has a limit.
The arc distance on a circle is not a Euclidean distance, so $\rho_\text{GC}$ of it need not be positive definite.
[](#code-loc-metrics) measures this.

```{code-block} python
:caption: The great-circle metric, a periodic metric, and the point where a periodic taper stops being positive definite.
:label: code-loc-metrics
# Shapes: D_x = 40 sites on a circle; 200 points on the sphere


# user
# d_ij = min(|sᵢ − sⱼ|, 40 − |sᵢ − sⱼ|)
def periodic_distance(
    coords_a: Float[Array, "a 1"], coords_b: Float[Array, "b 1"]
) -> Float[Array, "a b"]:
    # today (filterax)
    straight = flx.euclidean_distance(coords_a, coords_b)  # → (a, b)
    return jnp.minimum(straight, 40.0 - straight)  # → (a, b)


for half_width in (2.0, 8.0, 12.0):
    # today (filterax)
    # [ρ_xx]ᵢⱼ = ρ_GC(d_ij; ℓ)
    wrapped = flx.localization_matrix(  # → (D_x, D_x)
        grid, grid, half_width, metric=periodic_distance
    )
    print(half_width, f"{jnp.linalg.eigvalsh(wrapped).min():.4f}")

# today (filterax)
# one degree of longitude on the equator, in metres
one_degree = flx.haversine_distance(jnp.zeros((1, 2)), jnp.array([[0.0, jnp.pi / 180]]))
print(f"{one_degree[0, 0]:.0f}")

# (u₁, u₂) uniform on the unit square
uniform = jr.uniform(jr.key(0), (200, 2))  # → (200, 2)
# φ = arcsin(2 u₁ − 1), ϑ = 2 π u₂
sphere = jnp.column_stack(  # → (200, 2)
    [jnp.arcsin(2.0 * uniform[:, 0] - 1.0), 2.0 * jnp.pi * uniform[:, 1]]
)
# today (filterax)
# [ρ]ᵢⱼ = ρ_GC(d_ij; ℓ = 1), with d_ij on the unit sphere
on_sphere = flx.localization_matrix(  # → (200, 200)
    sphere,
    sphere,
    1.0,
    metric=lambda a, b: flx.haversine_distance(a, b, radius=1.0),
)
print(f"{jnp.linalg.eigvalsh(on_sphere).min():.1e}")
```

The smallest eigenvalue on the circle of circumference 40 is 0.0139 for $\ell = 2$, 0.0003 for $\ell = 8$ and -0.0023 for $\ell = 12$.
So the periodic taper is safe while the support $2 \ell$ stays within half the domain, and it turned indefinite at $2 \ell = 24$.
One degree of longitude prints as 111195 metres.
On 200 uniform points on the sphere, with great-circle distance and $\ell = 1$, the smallest eigenvalue is 1.4e-05.
That is positive on this draw and proves nothing in general.

## The localized gain

Using [](#eq-loc-schur) as the forecast covariance in the Kalman update of [chapter 4](04_kalman_update.md), with a linear $h$ that selects observed components, gives

$$\mat{G}_\text{loc} = (\rho_{xx} \circ \hat{\gmat{\Sigma}}_x) \mat{H}^\top \left(\mat{H} (\rho_{xx} \circ \hat{\gmat{\Sigma}}_x) \mat{H}^\top + \gmat{\Sigma}_\varepsilon\right)^{-1}.$$ (eq-loc-gain-background)

Selecting rows and columns of an element-wise product is the same as multiplying the selected rows and columns.
So with the state-observation taper $\rho_{xy} \in \mathbb{R}^{D_x \times D_y}$ and the observation-observation taper $\rho_{yy} \in \mathbb{R}^{D_y \times D_y}$ this is

$$\mat{G}_\text{loc} = (\rho_{xy} \circ \hat{\gmat{\Sigma}}_{xy}) (\rho_{yy} \circ \hat{\gmat{\Sigma}}_h + \gmat{\Sigma}_\varepsilon)^{-1},$$ (eq-loc-gain)

with $\mat{G}_\text{loc} \in \mathbb{R}^{D_x \times D_y}$.
This is covariance localization, also called B-localization {cite:p}`houtekamer2001sequential,hamill2001distance`.
It is `localized_kalman_gain`, and [](#eq-loc-gain) is also what it computes for a nonlinear $h$ or one that averages over a footprint.
For those, [](#eq-loc-gain) needs positions for the observations and is no longer the same as [](#eq-loc-gain-background) {cite:p}`campbell2010vertical`.

There is a real cost.
Without tapers, $\hat{\gmat{\Sigma}}_y$ is a rank-$(N_\mathcal{E} - 1)$ update of $\gmat{\Sigma}_\varepsilon$ with structured solves.
The Schur product destroys that structure, so `localized_kalman_gain` forms the dense $D_y \times D_y$ matrix and pays $O(N_\mathcal{E} D_x D_y + N_\mathcal{E} D_y^2 + D_y^3 + D_x D_y^2)$.
That is the algebraic reason LETKF ([chapter 8](08_letkf.md)) works with observation weights instead, as the implementation section sets out.

### Agreement with a dense form

[](#code-loc-agreement) checks the production gain against [](#eq-loc-gain-background) in dense numpy, and checks three further facts: the unit taper, gain tapering and a footprint operator.

```{code-block} python
:caption: The localized gain against the dense background-covariance form, in float64.
:label: code-loc-agreement
# Shapes: N_E = 10 members; D_x = 40 sites; D_y = 20 observed sites

# the observed sites
sites = jnp.arange(0, 40, 2)  # → (D_y,)
# s at the observed sites
obs_coords = grid[sites]  # → (D_y, 1)
# Σ_ε = I
noise = lx.DiagonalLinearOperator(jnp.ones(20))  # (D_y, D_y) operator
# h⁽ᵉ⁾ = H x⁽ᵉ⁾
mapped = small[:, sites]  # → (N_E, D_y)
# today (filterax)
# ρ_xx
rho_xx = flx.localization_matrix(grid, grid, 3.0)  # → (D_x, D_x)
# today (filterax)
# ρ_xy
rho_xy = flx.localization_matrix(grid, obs_coords, 3.0)  # → (D_x, D_y)
# today (filterax)
# ρ_yy
rho_yy = flx.localization_matrix(obs_coords, obs_coords, 3.0)  # → (D_y, D_y)
# today (filterax)
# G_loc = (ρ_xy ∘ Σ̂_xy) (ρ_yy ∘ Σ̂_h + Σ_ε)⁻¹
gain = flx.localized_kalman_gain(small, mapped, noise, rho_xy, rho_yy)  # → (D_x, D_y)


# user
# G_loc = B Hᵀ (H B Hᵀ + Σ_ε)⁻¹, with B = ρ_xx ∘ Σ̂_x, in plain numpy
def dense_gain(ensemble, taper, obs_matrix, noise_matrix):
    anomalies = ensemble - ensemble.mean(0)
    background = taper * (anomalies.T @ anomalies) / (len(ensemble) - 1)
    cross = background @ obs_matrix.T
    innovation = obs_matrix @ cross + noise_matrix
    return np.linalg.solve(innovation, cross.T).T


selection = np.eye(40)[np.asarray(sites)]  # (D_y, D_x)
reference = dense_gain(np.asarray(small), np.asarray(rho_xx), selection, np.eye(20))
print(f"{np.abs(np.asarray(gain) - reference).max():.0e}")

# today (filterax)
# G with ρ_xy = 𝟏, ρ_yy = 𝟏
unit = flx.localized_kalman_gain(
    small, mapped, noise, jnp.ones((40, 20)), jnp.ones((20, 20))
)  # → (D_x, D_y)
# today (filterax)
# G = Σ̂_xy (Σ̂_h + Σ_ε)⁻¹
plain = flx.kalman_gain(small, mapped, noise)  # → (D_x, D_y)
print(f"{jnp.abs(unit - plain).max():.0e}")

# today (filterax)
# ρ_xy ∘ G, tapering the finished gain
tapered_gain = flx.localize(plain, rho_xy)  # → (D_x, D_y)
print(f"{jnp.abs(tapered_gain - gain).max():.2f} {jnp.abs(gain).max():.2f}")

# h averages 9 neighbouring sites around each observed site, on a line
footprint = np.zeros((20, 40))
for row, site in enumerate(np.asarray(sites)):
    footprint[row, max(site - 4, 0) : site + 5] = 1.0
footprint = footprint / footprint.sum(1, keepdims=True)  # (D_y, D_x)
# h⁽ᵉ⁾ = H x⁽ᵉ⁾, for the footprint H
averaged = einx.dot(  # → (N_E, D_y)
    "obs state, member state -> member obs", jnp.asarray(footprint), small
)
# today (filterax)
# G_loc from the observations' positions
by_position = flx.localized_kalman_gain(small, averaged, noise, rho_xy, rho_yy)
exact = dense_gain(np.asarray(small), np.asarray(rho_xx), footprint, np.eye(20))
print(f"{np.abs(np.asarray(by_position) - exact).max():.2f} {np.abs(exact).max():.2f}")
```

It prints `3e-15`, `4e-16`, `0.33 0.72` and `0.26 0.27`.
The production gain matches the dense form and the unit taper recovers `kalman_gain`, both to float64 round-off.
Tapering the finished gain with `localize` is a different estimator, and differs from [](#eq-loc-gain) by up to 0.33 against a largest entry of 0.72.
For a 9-site footprint at $\ell = 3$, the position-based [](#eq-loc-gain) differs from [](#eq-loc-gain-background) by up to 0.26, against a largest entry of 0.27.
So a wide footprint needs the taper applied in state space.

## Adaptive localization

A distance-based taper needs a radius, and it needs distances.
`adaptive_localization` needs neither: it builds a $D_x \times D_y$ mask from the ensemble, keeping an entry only when its sample correlation exceeds its own approximate standard error.
With $\hat{c}_{ij}$ now the correlation between $x_i$ and $h_j$, and $\tau$ the `significance` argument,

$$[\rho_{xy}]_{ij} = \mathbb{1}\left\{|\hat{c}_{ij}| > \tau \frac{1 - \hat{c}_{ij}^2}{\sqrt{N_\mathcal{E} - 2}}\right\}.$$ (eq-loc-adaptive)

The motivation comes from the hierarchical filter of {cite:t}`anderson2007exploring`, which asks when a regression coefficient is too noisy to trust.
The code does not run a hierarchy of ensembles: [](#eq-loc-adaptive) is one hard threshold on one ensemble.

Solving the boundary of [](#eq-loc-adaptive) for $|\hat{c}|$ gives the threshold in closed form,

$$c^\star = \frac{\sqrt{N_\mathcal{E} - 2} \left(\sqrt{1 + 4 \tau^2 / (N_\mathcal{E} - 2)} - 1\right)}{2 \tau},$$ (eq-loc-adaptive-threshold)

so $\tau$ is roughly a z-score and the keep rate of a null correlation is set by $\tau$, not by $N_\mathcal{E}$.

```{code-block} python
:caption: How many truly uncorrelated pairs the mask keeps.
:label: code-loc-adaptive
# Shapes: N_E members; D_x = 400; D_y = 50

for num_members, significance in ((20, 1.0), (80, 1.0), (20, 2.0), (20, 3.0)):
    # x⁽ᵉ⁾ ~ 𝒩(𝟎, I)
    states = jr.normal(jr.key(3), (num_members, 400))  # → (N_E, D_x)
    # h⁽ᵉ⁾ ~ 𝒩(𝟎, I), independent of x⁽ᵉ⁾
    outputs = jr.normal(jr.key(4), (num_members, 50))  # → (N_E, D_y)
    # today (filterax)
    # [ρ_xy]ᵢⱼ ∈ {0, 1}
    mask = flx.adaptive_localization(  # → (D_x, D_y)
        states, outputs, significance=significance
    )
    # c⋆
    root = np.sqrt(num_members - 2.0)
    bound = root * (np.sqrt(1.0 + 4.0 * significance**2 / root**2) - 1.0)
    print(num_members, significance, f"{bound / (2.0 * significance):.3f}", mask.mean())
```

| $N_\mathcal{E}$ | $\tau$ | $c^\star$ | Fraction of null pairs kept |
| --- | --- | --- | --- |
| 20 | 1 | 0.224 | 0.3406 |
| 80 | 1 | 0.112 | 0.32245 |
| 20 | 2 | 0.397 | 0.081 |
| 20 | 3 | 0.518 | 0.01915 |

The default $\tau = 1$ keeps about a third of the pure-noise entries at any ensemble size, and $\tau = 3$ keeps about 2% of them.
So the mask thins the noise, with a threshold that you still have to choose.
It is not positive definite, and it raises `ValueError` below $N_\mathcal{E} = 3$.
Its docstring applies it to the finished gain, with `localize`, and the cycle below shows why that matters.

## Implementation in filterax

The primitives are at [`filterax`](https://jejjohnson.github.io/filterax/reference/localization/), and only LETKF applies a taper for you.

| Layer | Offers | Form |
| --- | --- | --- |
| Primitive | `gaspari_cohn`, `gaussian_taper`, `soar_taper`, `hard_cutoff` | $\rho(d; \ell)$ on any array of distances |
| Primitive | `euclidean_distance`, `haversine_distance`, `localization_matrix` | a dense taper matrix, always Gaspari-Cohn in `localization_matrix` |
| Primitive | `localize`, `localized_kalman_gain` | [](#eq-loc-schur), [](#eq-loc-gain) on dense arrays |
| Primitive | `adaptive_localization` | the mask [](#eq-loc-adaptive) |
| Layer 1 | `filters.LETKF(radius, taper_fn)` | observation weights at each grid point |
| Layer 2 | `LETKF` | the same, in the assimilation loop |

The other filters, `StochasticEnKF`, `EnSRF`, `EnSRF_Serial`, `ETKF` and `ESTKF`, do not accept a taper.
To use [](#eq-loc-gain) you write the update yourself, as the next section does.
`AbstractLocalizer` has no concrete subclass in the library, and no filter reads `FilterConfig.localizer`.

### LETKF has its own cut

LETKF weights each observation's inverse variance by $\rho$ at each grid point, which is R-localization {cite:p}`hunt2007efficient` ([chapter 8](08_letkf.md)), and that keeps the local solves small.
It computes its own Euclidean distance, so it takes no `metric`, and on a sphere the coordinates must be Cartesian.
It also drops every observation beyond $d = \ell$ before tapering, so the second branch of [](#eq-loc-gaspari-cohn) is never used.
The result is a support of $\ell$, half that of `localization_matrix` at the same `radius`, with a jump at the cut.

```{code-block} python
:caption: One observation at the origin and state sites at distances 0, 1 and 3, with $\ell = 2$ in both forms.
:label: code-loc-letkf-cut
# Shapes: N_E = 20 members; D_x = 3 sites; D_y = 1 observation

# s of the state sites
site_coords = jnp.array([[0.0], [1.0], [3.0]])  # → (D_x, 1)
# x⁽ᵉ⁾ ~ 𝒩(𝟎, I)
three_sites = jr.normal(jr.key(5), (20, 3))  # → (N_E, D_x)
# today (lineax)
# Σ_ε = 0.1
tight = lx.DiagonalLinearOperator(0.1 * jnp.ones(1))  # (D_y, D_y) operator
# today (filterax)
# R-localized analysis at each site
local = flx.filters.LETKF(radius=2.0).analysis(
    three_sites,
    jnp.array([5.0]),
    lambda state: state[:1],
    tight,
    state_coords=site_coords,
    obs_coords=jnp.zeros((1, 1)),
)
# x̄_t|t − x̄
increment = einx.mean("[member] state", local.particles - three_sites)  # → (D_x,)
# today (filterax)
# ρ_GC(d; ℓ = 2)
weights = flx.localization_matrix(site_coords, jnp.zeros((1, 1)), 2.0)  # → (D_x, 1)
print(increment.round(3), einx.id("site 1 -> site", weights).round(4))
```

It prints `[4.65  0.365 0.   ] [1.     0.6849 0.0165]`.
The site at $d = 3$ has a Gaspari-Cohn weight of 0.0165 from `localization_matrix`, and it receives no increment at all from LETKF.
The jump at the cut is 0.2083, the value of [](#eq-loc-gaspari-cohn) at $z = 1$.

## Algorithm

[](#code-loc-cycle) writes a localized stochastic EnKF on Lorenz-96, the setting of [chapter 6](06_etkf.md), where the unlocalized filter fails.
Each window forecasts every member and forms the gain [](#eq-loc-gain).
It then moves each member towards its own perturbed observation ([chapter 5](05_stochastic_enkf.md)) and applies a small multiplicative inflation ([chapter 12](12_inflation.md)).
The sites sit on a circle, so the tapers use `periodic_distance`.

```{code-block} python
:caption: The localized stochastic EnKF with a choice of tapers, jitted through one scan per configuration.
:label: code-loc-cycle
# Shapes: T = 300 windows; D_x = 40 sites; D_y = 20 observed sites; N_E members


# user
# dxₖ/dt = (xₖ₊₁ − xₖ₋₂) xₖ₋₁ − xₖ + 8
def tendency(state: Float[Array, " state"]) -> Float[Array, " state"]:
    return (jnp.roll(state, -1) - jnp.roll(state, 2)) * jnp.roll(state, 1) - state + 8.0


# user
# one RK4 step of Δt = 0.05
def rk4(state: Float[Array, " state"]) -> Float[Array, " state"]:
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
_, truth = jax.lax.scan(advance, spun_up, length=300)  # → (T, D_x)
# yₜ = H xₜ + εₜ, εₜ ~ 𝒩(𝟎, I)
observed = truth[:, sites] + jr.normal(jr.key(1), (300, 20))  # → (T, D_y)


# user
# one localized stochastic EnKF run: RMSE and RMS spread of the second half
def run_cycle(gain_fn, num_members: int, factor: float) -> tuple[float, float]:
    # x⁽ᵉ⁾ ~ 𝒩(x₀, 4 I)
    start = spun_up + 2.0 * jr.normal(jr.key(2), (num_members, 40))  # → (N_E, D_x)

    # user
    def window(ensemble, inputs):
        target, key = inputs
        # x⁽ᵉ⁾_t|t−1 = f(x⁽ᵉ⁾)
        forecast = jax.vmap(rk4)(ensemble)  # → (N_E, D_x)
        # h⁽ᵉ⁾ = H x⁽ᵉ⁾_t|t−1
        members_at_sites = forecast[:, sites]  # → (N_E, D_y)
        # G_loc
        window_gain = gain_fn(forecast, members_at_sites)  # → (D_x, D_y)
        # today (filterax)
        # y⁽ᵉ⁾ = y + ε⁽ᵉ⁾, ε⁽ᵉ⁾ ~ 𝒩(𝟎, Σ_ε)
        perturbed = flx.perturbed_observations(key, target, noise, num_members)
        # x⁽ᵉ⁾_t|t = x⁽ᵉ⁾_t|t−1 + G_loc (y⁽ᵉ⁾ − h⁽ᵉ⁾)
        analysis = forecast + einx.dot(  # → (N_E, D_x)
            "state obs, member obs -> member state",
            window_gain,
            perturbed - members_at_sites,
        )
        # today (filterax)
        # x̃⁽ᵉ⁾ ← ζ x̃⁽ᵉ⁾
        inflated = flx.inflate_multiplicative(analysis, factor)  # → (N_E, D_x)
        return inflated, inflated

    keys = jr.split(jr.key(3), 300)
    _, history = jax.lax.scan(window, start, (observed, keys))  # → (T, N_E, D_x)
    # x̄_t|t
    settled_mean = einx.mean("t [member] state", history[100:])  # → (T − 100, D_x)
    # today (filterax)
    errors = jax.vmap(flx.utils.rmse_vs_truth)(settled_mean, truth[100:])
    # today (filterax)
    spreads = jax.vmap(flx.utils.rms_spread)(history[100:])
    rmse = jnp.sqrt(jnp.mean(errors**2))  # → ()
    spread = jnp.sqrt(jnp.mean(spreads**2))  # → ()
    return float(rmse), float(spread)


# d_xy, d_yy on the circle
circle_xy = periodic_distance(grid, obs_coords)  # → (D_x, D_y)
circle_yy = periodic_distance(obs_coords, obs_coords)  # → (D_y, D_y)


# user
# G_loc for a taper function and half-width
def tapered(taper, half_width):
    # ρ_xy
    weights_xy = taper(circle_xy, half_width)  # → (D_x, D_y)
    # ρ_yy
    weights_yy = taper(circle_yy, half_width)  # → (D_y, D_y)

    # today (filterax)
    return lambda forecast, mapped: flx.localized_kalman_gain(
        forecast, mapped, noise, weights_xy, weights_yy
    )


# user
# ρ_xy ∘ G, with the adaptive mask on the finished gain
def masked_gain(significance: float):
    # user
    def gain_fn(forecast, mapped):
        # today (filterax)
        mask = flx.adaptive_localization(  # → (D_x, D_y)
            forecast, mapped, significance=significance
        )
        # today (filterax)
        # ρ_xy ∘ G
        return flx.localize(flx.kalman_gain(forecast, mapped, noise), mask)

    return gain_fn


# user
# (ρ_xy ∘ Σ̂_xy) (Σ̂_h + Σ_ε)⁻¹, with the adaptive mask on the cross-covariance alone
def masked_cross(forecast, mapped):
    # today (filterax)
    mask = flx.adaptive_localization(forecast, mapped, significance=2.0)
    # today (filterax)
    return flx.localized_kalman_gain(forecast, mapped, noise, mask, jnp.ones((20, 20)))


# today (filterax)
configurations = {
    "no taper": tapered(lambda d, _: jnp.ones_like(d), 1.0),
    "gaspari_cohn, l = 1": tapered(flx.gaspari_cohn, 1.0),
    "gaspari_cohn, l = 2": tapered(flx.gaspari_cohn, 2.0),
    "gaspari_cohn, l = 4": tapered(flx.gaspari_cohn, 4.0),
    "gaspari_cohn, l = 8": tapered(flx.gaspari_cohn, 8.0),
    "hard_cutoff, l = 4": tapered(flx.hard_cutoff, 4.0),
    "hard_cutoff, l = 8": tapered(flx.hard_cutoff, 8.0),
    "mask on gain, tau = 1": masked_gain(1.0),
    "mask on gain, tau = 2": masked_gain(2.0),
    "mask on cross, tau = 2": masked_cross,
}
for num_members in (10, 20):
    for name, gain_fn in configurations.items():
        rmse, spread = run_cycle(gain_fn, num_members, 1.05)
        print(f"N_E = {num_members}, {name}: {rmse:.2f} {spread:.2f}")
rmse, spread = run_cycle(tapered(flx.gaspari_cohn, 4.0), 10, 1.0)
print(f"N_E = 10, gaspari_cohn, l = 4, zeta = 1: {rmse:.2f} {spread:.2f}")
```

It prints the RMSE and the RMS spread of the last 200 windows, both with $\zeta = 1.05$:

| Gain | $N_\mathcal{E} = 10$: RMSE | spread | $N_\mathcal{E} = 20$: RMSE | spread |
| --- | --- | --- | --- | --- |
| No taper | 5.21 | 0.22 | 4.65 | 0.30 |
| `gaspari_cohn`, $\ell = 1$ | 0.69 | 1.17 | 0.74 | 1.40 |
| `gaspari_cohn`, $\ell = 2$ | 0.54 | 0.57 | 0.52 | 0.65 |
| `gaspari_cohn`, $\ell = 4$ | 0.50 | 0.43 | 0.45 | 0.50 |
| `gaspari_cohn`, $\ell = 8$ | 0.71 | 0.34 | 0.38 | 0.41 |
| `hard_cutoff`, $\ell = 4$ | 0.51 | 0.42 | 0.44 | 0.48 |
| `hard_cutoff`, $\ell = 8$ | nan | nan | 0.43 | 0.39 |
| Mask on $\mat{G}$, $\tau = 1$ | 4.88 | 0.26 | 3.73 | 0.34 |
| Mask on $\mat{G}$, $\tau = 2$ | 4.52 | 0.32 | 0.47 | 0.41 |
| Mask on $\hat{\gmat{\Sigma}}_{xy}$, $\tau = 2$ | nan | nan | nan | nan |

Without a taper both ensembles fail, with an error far above the unit observation noise and a small spread.
With a Gaspari-Cohn taper the same filter settles well below the noise.
The best half-width among these grows with the ensemble: $\ell = 4$ at $N_\mathcal{E} = 10$ and $\ell = 8$ at $N_\mathcal{E} = 20$, since more members resolve correlations further out before noise wins.
At $\ell = 1$ the spread sits far above the error.
Observations reach the sites between them with a weight of only 0.2083, so the inflation meets little correction there.

Localization does not make inflation unnecessary.
With $N_\mathcal{E} = 10$, $\ell = 4$ and $\zeta = 1$, the same run gives an RMSE of 0.57 with a spread of 0.29, so the error is about twice the spread.

## Gradients and float32

The tapers are differentiable in $\ell$ through `localization_matrix`, so a half-width can be trained ([chapter 14](14_differentiable.md)).
Gaussx guards the $1 / z$ term, so the gradient at $d = 0$ is finite.
`hard_cutoff` and `adaptive_localization` are comparisons, so their gradients are zero everywhere and no error warns you.

```{code-block} python
:caption: The gradient of a loss through the gain with respect to $\ell$, against a central difference, then the two tapers with no gradient.
:label: code-loc-gradient
# Shapes: N_E = 10; D_x = 40; D_y = 20


# user
# 𝓛(ℓ) = ‖G_loc r̄‖², with r̄ a fixed residual
def half_width_loss(half_width: Float[Array, ""]) -> Float[Array, ""]:
    # today (filterax)
    weights_xy = flx.localization_matrix(grid, obs_coords, half_width)
    # today (filterax)
    weights_yy = flx.localization_matrix(obs_coords, obs_coords, half_width)
    # today (filterax)
    window_gain = flx.localized_kalman_gain(
        small, mapped, noise, weights_xy, weights_yy
    )  # → (D_x, D_y)
    return jnp.sum((window_gain @ jnp.ones(20)) ** 2)  # → ()


at = jnp.asarray(3.0)
step = 1e-6
central = (half_width_loss(at + step) - half_width_loss(at - step)) / (2.0 * step)
print(f"{jax.grad(half_width_loss)(at):.6f} {central:.6f}")
# today (filterax)
# d ρ_GC / d d at d = 0
print(jax.grad(lambda d: flx.gaspari_cohn(d, 1.0))(jnp.asarray(0.0)))
# today (filterax)
# d/dℓ ∑ ρ_cut(d_xy; ℓ)
print(jax.grad(lambda r: jnp.sum(flx.hard_cutoff(circle_xy, r)))(at))
# today (filterax)
# d/dτ ∑ [ρ_xy]ᵢⱼ
print(
    jax.grad(
        lambda s: jnp.sum(flx.adaptive_localization(small, mapped, significance=s))
    )(jnp.asarray(1.0))
)

# z ∈ [1.5, 2] in float32
near_support = jnp.linspace(1.5, 2.0, 20_001, dtype=jnp.float32)  # → (20001,)
# today (filterax)
# ρ_GC(z), in float32
in_float32 = flx.gaspari_cohn(near_support, jnp.float32(1.0))  # → (20001,)
print(in_float32.dtype, f"{in_float32.min():.1e}", jnp.sum(in_float32 < 0.0))
```

It prints `0.147164 0.147164`, `0.0`, `0.0` and `0.0`, and then `float32 -1.2e-06 643`.
So the gradient matches the central difference, and the gradients of the two comparisons are exactly zero.
The last line is the one place float32 shows.
The rational branch of [](#eq-loc-gaspari-cohn) cancels large terms near $z = 2$, so 643 of those 20,001 float32 values come out slightly negative, as low as $-1.2 \times 10^{-6}$.
That is round-off in a weight meant to be in $[0, 1]$, and it is far smaller than any entry that matters.

## When it breaks

The table above has two `nan` rows, and [](#code-loc-hard-cut) shows why the hard cutoff gives one.

```{code-block} python
:caption: The innovation covariance of the first forecast at $N_\mathcal{E} = 10$, with a hard cutoff at $\ell = 8$ and with Gaspari-Cohn at $\ell = 4$.
:label: code-loc-hard-cut
# Shapes: N_E = 10 members; D_x = 40 sites; D_y = 20 observed sites

# x⁽ᵉ⁾ ~ 𝒩(x₀, 4 I)
first = spun_up + 2.0 * jr.normal(jr.key(2), (10, 40))  # → (N_E, D_x)
# x⁽ᵉ⁾_t|t−1
first_forecast = jax.vmap(rk4)(first)  # → (N_E, D_x)
# today (filterax)
# Σ̂_h
first_cov = flx.ensemble_covariance(first_forecast[:, sites]).as_matrix()
for taper, half_width in ((flx.hard_cutoff, 8.0), (flx.gaspari_cohn, 4.0)):
    # today (filterax)
    # ρ_yy ∘ Σ̂_h + Σ_ε
    innovation = flx.localize(first_cov, taper(circle_yy, half_width)) + jnp.eye(20)
    print(taper.__name__, f"{jnp.linalg.eigvalsh(innovation).min():.2f}")
```

It prints `hard_cutoff -0.27` and `gaspari_cohn 2.10`.
With the hard cutoff, the matrix that `localized_kalman_gain` tags as positive semi-definite has a negative eigenvalue in the first window.
The run then returns `nan`.
The Gaspari-Cohn taper with the same support stays above the noise variance of 1.

The limits, in the order you are likely to meet them:

- A hard cutoff can kill the filter, as above, though at other settings in the table it survives, so keep it as a debugging baseline only.
- Too small a half-width cuts real signal and leaves the state underconstrained, as the $\ell = 1$ row shows.
- Too large a half-width lets noise back in, which is the $\ell = 8$ row at $N_\mathcal{E} = 10$.
- A data-driven mask needs its own tuning: in the table it works only at $N_\mathcal{E} = 20$ with $\tau = 2$ on the finished gain.
- The same mask on $\hat{\gmat{\Sigma}}_{xy}$ alone, with $\hat{\gmat{\Sigma}}_y$ left untapered, returns `nan`.
- The half-width has no universal value, since it depends on $N_\mathcal{E}$, on the observation density and on the system.
- Tapers damp correlations by distance whatever their physical origin, so relations such as balance between wind and mass fields are weakened {cite:p}`kepert2009covariance`.
- A satellite radiance or a column average has no clear position, and [](#eq-loc-gain) then differs from [](#eq-loc-gain-background), as [](#code-loc-agreement) measures.
- A periodic metric can leave the theorem behind, so keep the support below half the domain, as [](#code-loc-metrics) measures.

## When localization is the right answer

Use a taper when $N_\mathcal{E} \ll D_x$ and correlations decay with a distance you can compute, which covers most geophysical grids.
Use `LETKF` for a large state with diagonal observation noise, where the local solves are small and parallel.
Use `localized_kalman_gain` when you need a global gain, correlated observations through $\rho_{yy}$, or a taper in $\ell$ you want to differentiate.
Use `adaptive_localization` when no meaningful distance exists, and tune $\tau$ rather than taking the default, as the table above shows.
For a parameter vector with no spatial order, `ledoit_wolf_shrinkage` in [chapter 12](12_inflation.md) is the alternative.

## See also

- [Chapter 4](04_kalman_update.md): the unlocalized gain this chapter tapers
- [Chapter 5](05_stochastic_enkf.md): the perturbed-observation update used in the cycle above
- [Chapter 8](08_letkf.md): R-localization, one transform per grid point
- [Chapter 12](12_inflation.md): inflation, the companion to localization
- [Chapter 14](14_differentiable.md): training a half-width through the filter
- [API: localization](https://jejjohnson.github.io/filterax/reference/localization/) and [filters](https://jejjohnson.github.io/filterax/reference/filters/)

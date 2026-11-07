---
numbering:
  enumerator: "4.%s"
---

# The Kalman update and the ensemble gain

The analysis step of every filter in this library is built around one matrix, the Kalman gain.
It says how far to move the state for a given misfit in observation space.
This chapter derives it, replaces its covariances with sample statistics, and then turns to the question that shapes the implementation: how to solve against the innovation covariance.

Three things organise it.
The gain follows from asking for the best linear update, which a reader can verify by hand.
The solve has two routes, a dense one and a Woodbury one, and the cheaper route depends only on the shapes.
And the same innovation covariance gives the diagnostics, through `innovation_statistics`, so its accuracy decides what those numbers can tell you.

## The best linear unbiased update

Let the state have the forecast mean $\gvect{\mu}_x \in \mathbb{R}^{D_x}$ and covariance $\gmat{\Sigma}_x \in \mathbb{R}^{D_x \times D_x}$, and observe it with a linear $\mat{H} \in \mathbb{R}^{D_y \times D_x}$, $\vect{y} = \mat{H} \vect{x} + \gvect{\varepsilon}$.
Seek an estimate that is linear in the residual $\vect{r} = \vect{y} - \mat{H} \gvect{\mu}_x \in \mathbb{R}^{D_y}$:

$$\hat{\vect{x}} = \gvect{\mu}_x + \mat{G} \vect{r}, \qquad \mat{G} \in \mathbb{R}^{D_x \times D_y}.$$ (eq-gain-estimate)

It is unbiased for any $\mat{G}$, because $\mathbb{E}[\vect{r}] = \vect{0}$.
Its error is $\vect{x} - \hat{\vect{x}} = (\mat{I} - \mat{G} \mat{H})(\vect{x} - \gvect{\mu}_x) - \mat{G} \gvect{\varepsilon}$, and the two terms are independent, so its covariance is

$$\gmat{\Sigma}_{x \mid y} = (\mat{I} - \mat{G} \mat{H}) \gmat{\Sigma}_x (\mat{I} - \mat{G} \mat{H})^\top + \mat{G} \gmat{\Sigma}_\varepsilon \mat{G}^\top.$$ (eq-gain-joseph)

Choosing $\mat{G}$ to minimise the total variance $\op{tr}(\gmat{\Sigma}_{x \mid y})$, and setting the derivative with respect to $\mat{G}$ to zero, gives $-2 (\mat{I} - \mat{G} \mat{H}) \gmat{\Sigma}_x \mat{H}^\top + 2 \mat{G} \gmat{\Sigma}_\varepsilon = \mat{0}$.
Collecting the $\mat{G}$ terms leaves the gain {cite:p}`kalman1960new`:

$$\mat{G} = \gmat{\Sigma}_{xy} \gmat{\Sigma}_y^{-1}, \qquad \gmat{\Sigma}_{xy} = \gmat{\Sigma}_x \mat{H}^\top, \qquad \gmat{\Sigma}_y = \mat{H} \gmat{\Sigma}_x \mat{H}^\top + \gmat{\Sigma}_\varepsilon.$$ (eq-gain-optimal)

Now substitute back into [](#eq-gain-joseph).
Expanding gives $(\mat{I} - \mat{G} \mat{H}) \gmat{\Sigma}_x - (\gmat{\Sigma}_{xy} - \mat{G} \gmat{\Sigma}_y) \mat{G}^\top$, and the bracket vanishes at the optimum, so

$$\gmat{\Sigma}_{x \mid y} = (\mat{I} - \mat{G} \mat{H}) \gmat{\Sigma}_x = \gmat{\Sigma}_x - \mat{G} \gmat{\Sigma}_{xy}^\top.$$ (eq-gain-posterior)

The short form holds only at the optimal gain, whereas [](#eq-gain-joseph) holds for any $\mat{G}$.

Nothing above assumed Gaussian errors, only their first two moments.
If they are Gaussian, the same formulas are the exact posterior mean and covariance, because $\vect{x}$ and $\vect{y}$ are then jointly Gaussian.
The gain also reads as a regression coefficient: it is the covariance of the state with the observation, divided by the variance of the observation.

## The ensemble gain

An ensemble has no $\gmat{\Sigma}_x$, but it has the sample statistics of [chapter 3](03_observation_model.md), with $c = N_\mathcal{E} - 1$:

$$\hat{\mat{G}} = \hat{\gmat{\Sigma}}_{xy} \hat{\gmat{\Sigma}}_y^{-1}, \qquad \hat{\gmat{\Sigma}}_{xy} = \frac{1}{c} \tilde{\mat{X}}^\top \tilde{\mat{H}}, \qquad \hat{\gmat{\Sigma}}_y = \frac{1}{c} \tilde{\mat{H}}^\top \tilde{\mat{H}} + \gmat{\Sigma}_\varepsilon.$$ (eq-gain-ensemble)

The mean update and the covariance then have the same form as above:

$$\bar{\vect{x}}_{t \mid t} = \bar{\vect{x}} + \hat{\mat{G}} \bar{\vect{r}}, \qquad \hat{\gmat{\Sigma}}_{x \mid y} = \hat{\gmat{\Sigma}}_x - \hat{\mat{G}} \hat{\gmat{\Sigma}}_{xy}^\top.$$ (eq-gain-analysis)

For a linear $h$ this is the Kalman update on the sample covariance.
For a nonlinear $h$ it is the statistical linearisation of [chapter 3](03_observation_model.md).
Filters differ in how they turn this into an ensemble: [chapter 5](05_stochastic_enkf.md) applies $\hat{\mat{G}}$ to each member's own perturbed residual, and [chapter 6](06_etkf.md) transforms the anomalies without forming it.

## Two routes to the solve

$\hat{\gmat{\Sigma}}_{xy}$ is a dense $D_x \times D_y$ array, which costs $O(N_\mathcal{E} D_x D_y)$.
The interesting part is the solve against $\hat{\gmat{\Sigma}}_y \in \mathbb{R}^{D_y \times D_y}$, which is a structured matrix plus a term of rank at most $c$.

The dense route forms it and factorises it, at $O(N_\mathcal{E} D_y^2 + D_y^3)$.

The Woodbury route keeps the two parts separate.
Write the anomaly term with the factor $\mat{U}_h = \tilde{\mat{H}}^\top / \sqrt{c} \in \mathbb{R}^{D_y \times N_\mathcal{E}}$, so that $\hat{\gmat{\Sigma}}_y = \gmat{\Sigma}_\varepsilon + \mat{U}_h \mat{U}_h^\top$.
The Woodbury identity {cite:p}`hager1989updating` then gives

$$\hat{\gmat{\Sigma}}_y^{-1} = \gmat{\Sigma}_\varepsilon^{-1} - \gmat{\Sigma}_\varepsilon^{-1} \tilde{\mat{H}}^\top \big(c \mat{I} + \tilde{\mat{H}} \gmat{\Sigma}_\varepsilon^{-1} \tilde{\mat{H}}^\top\big)^{-1} \tilde{\mat{H}} \gmat{\Sigma}_\varepsilon^{-1},$$ (eq-gain-woodbury)

which can be checked by multiplying by $\hat{\gmat{\Sigma}}_y$.
The bracket is $\gmat{\Lambda}_w$ of [chapter 6](06_etkf.md), so this route and the ETKF invert the same $N_\mathcal{E} \times N_\mathcal{E}$ matrix.
In terms of $\mat{U}_h$ it is $c$ times the capacitance $\mat{I} + \mat{U}_h^\top \gmat{\Sigma}_\varepsilon^{-1} \mat{U}_h$, whose determinant gives the log-determinant:

$$\log \vert \hat{\gmat{\Sigma}}_y \vert = \log \vert \gmat{\Sigma}_\varepsilon \vert + \log \big\vert \mat{I} + \mat{U}_h^\top \gmat{\Sigma}_\varepsilon^{-1} \mat{U}_h \big\vert.$$ (eq-gain-determinant)

With $\gmat{\Sigma}_\varepsilon^{-1}$ cheap to apply, this route costs $O(N_\mathcal{E}^2 D_y + N_\mathcal{E}^3)$ and never forms $\gmat{\Sigma}_\varepsilon$.
It needs $\gmat{\Sigma}_\varepsilon$ to be positive definite.

The cheaper route depends on which matrix is smaller, and so filterax chooses by shape:

| Route | Default when | Cost of the solve | Factorises | Needs |
| --- | --- | --- | --- | --- |
| Dense | $N_\mathcal{E} \ge D_y$ | $O(N_\mathcal{E} D_y^2 + D_y^3)$ | $D_y \times D_y$ | $\gmat{\Sigma}_\varepsilon$ positive semi-definite |
| Woodbury | $N_\mathcal{E} < D_y$ | $O(N_\mathcal{E}^2 D_y + N_\mathcal{E}^3)$ | $N_\mathcal{E} \times N_\mathcal{E}$ | $\gmat{\Sigma}_\varepsilon$ positive definite |

A satellite swath with $D_y = 10^5$ and $N_\mathcal{E} = 50$ shows the difference: about $2.5 \times 10^8$ operations through Woodbury, against $10^{15}$ for the dense factorisation.
In the opposite regime the capacitance is the larger matrix, and forming $\hat{\gmat{\Sigma}}_y$ is what saves the work.

## Implementation in filterax

The Layer 0 functions below build the update, and each takes the mapped ensemble or applies $h$ itself.

| Function | Returns | Formula |
| --- | --- | --- |
| `kalman_gain(particles, obs_particles, obs_noise)` | dense $D_x \times D_y$ array | [](#eq-gain-ensemble) |
| `cross_covariance(particles, obs_particles)` | dense $D_x \times D_y$ array | $\hat{\gmat{\Sigma}}_{xy}$ |
| `innovation_covariance(obs_particles, obs_noise)` | `gaussx.LowRankUpdate` | $\hat{\gmat{\Sigma}}_y$ |
| `log_likelihood(innovation, innovation_cov)` | scalar | $\log \mathcal{N}(\bar{\vect{r}}; \vect{0}, \hat{\gmat{\Sigma}}_y)$ |
| `innovation_statistics(particles, obs, obs_op, obs_noise)` | a dict of four results | [](#eq-gain-whitened) below |
| `localized_kalman_gain(..., rho_xy, rho_yy)` | dense $D_x \times D_y$ array | [](#eq-gain-localized) below |

`kalman_gain` takes `solver=` to override the structural dispatch of gaussx, and `dense_innovation=` to force a route.
The gain itself is formed as a dense array on purpose, since the analysis uses it once and $D_y$ is the per-window observation count.

## Algorithm

[](#code-gain-by-hand) writes the two routes out in numpy, as references for what follows.

```{code-block} python
:caption: The gain [](#eq-gain-ensemble) on both routes, as dense numpy references with a diagonal noise covariance.
:label: code-gain-by-hand
import einx
import filterax as flx
import gaussx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
from jaxtyping import Array, Float

jax.config.update("jax_enable_x64", True)

# Shapes: N_E members; D_x state components; D_y observations


# user
# Ĝ = Σ̂_xy Σ̂_y⁻¹, forming Σ̂_y, in numpy
def gain_formed(members, mapped, variances):
    c = len(members) - 1.0
    state_anomalies = members - members.mean(axis=0)
    mapped_anomalies = mapped - mapped.mean(axis=0)
    cross = state_anomalies.T @ mapped_anomalies / c
    obs_covariance = mapped_anomalies.T @ mapped_anomalies / c + np.diag(variances)
    return np.linalg.solve(obs_covariance, cross.T).T


# user
# Ĝ through the Woodbury identity, in numpy
def gain_woodbury(members, mapped, variances):
    c = len(members) - 1.0
    state_anomalies = members - members.mean(axis=0)
    mapped_anomalies = mapped - mapped.mean(axis=0)
    cross = state_anomalies.T @ mapped_anomalies / c
    # Σ_ε⁻¹ H̃ᵀ
    whitened = mapped_anomalies.T / variances[:, np.newaxis]
    # Λ_w = c I + H̃ Σ_ε⁻¹ H̃ᵀ
    precision = c * np.eye(len(members)) + mapped_anomalies @ whitened
    # Σ̂_xy Σ_ε⁻¹
    first = cross / variances
    # Σ̂_xy Σ_ε⁻¹ H̃ᵀ Λ_w⁻¹ H̃ Σ_ε⁻¹
    second = cross @ whitened @ np.linalg.solve(precision, whitened.T)
    return first - second
```

## Agreement with a dense reference

[](#code-gain-agreement) compares `kalman_gain` with these references, on the default route and on both forced routes.
It covers one shape on each side of the rule, plus a very small ensemble.

```{code-block} python
:caption: The gain on every route against the dense and Woodbury references, in float64.
:label: code-gain-agreement
# Shapes: N_E members; D_x = 40; D_y observations

for num_members, obs_size in ((25, 10), (25, 40), (8, 20)):
    # X, with h(x) selecting the first D_y components
    members = jr.normal(jr.key(0), (num_members, 40))  # → (N_E, D_x)
    mapped = members[:, :obs_size]  # → (N_E, D_y)
    variances = 0.2 * np.ones(obs_size)
    # today (lineax)
    # Σ_ε = 0.2 I
    noise = lx.DiagonalLinearOperator(jnp.asarray(variances))  # (D_y, D_y) operator
    formed = gain_formed(np.asarray(members), np.asarray(mapped), variances)
    through = gain_woodbury(np.asarray(members), np.asarray(mapped), variances)
    # today (filterax)
    default = flx.kalman_gain(members, mapped, noise)  # → (D_x, D_y)
    # today (filterax)
    dense = flx.kalman_gain(members, mapped, noise, dense_innovation=True)
    # today (filterax)
    woodbury = flx.kalman_gain(members, mapped, noise, dense_innovation=False)
    errors = [
        np.abs(formed - through).max(),
        jnp.abs(default - formed).max(),
        jnp.abs(dense - formed).max(),
        jnp.abs(woodbury - formed).max(),
    ]
    print(num_members, obs_size, [f"{error:.0e}" for error in errors])
```

It prints:

| $N_\mathcal{E}$ | $D_y$ | Default route | Woodbury by hand | Default | Forced dense | Forced Woodbury |
| --- | --- | --- | --- | --- | --- | --- |
| 25 | 10 | dense | `2e-15` | `4e-16` | `4e-16` | `2e-15` |
| 25 | 40 | Woodbury | `4e-15` | `3e-15` | `1e-15` | `3e-15` |
| 8 | 20 | Woodbury | `3e-15` | `3e-15` | `1e-15` | `3e-15` |

The errors are the largest absolute difference from `gain_formed`.
Every route matches the dense reference to float64 round-off, and so does the hand-written Woodbury form [](#eq-gain-woodbury).

The derivation can also be checked directly.
[](#code-gain-optimal) evaluates the gradient of the total variance at the sample gain, where it should vanish.
It compares the two forms of the covariance there and away from the optimum.
It then applies [](#eq-gain-analysis) to a real ensemble and compares it with the ETKF, which must have the same mean and covariance.

```{code-block} python
:caption: The optimality of the gain, the two covariance forms, and the mean and covariance of the update against the ETKF.
:label: code-gain-optimal
# Shapes: N_E = 30 members; D_x = 6; D_y = 3

members = jr.normal(jr.key(1), (30, 6))  # → (N_E, D_x)
# H selects the first three components
obs_map = jnp.eye(3, 6)  # → (D_y, D_x)
# Σ_ε
noise_matrix = 0.5 * jnp.eye(3)  # → (D_y, D_y)
# today (lineax)
noise = lx.DiagonalLinearOperator(0.5 * jnp.ones(3))  # (D_y, D_y) operator
mapped = members[:, :3]  # → (N_E, D_y)
# today (filterax)
# Σ̂_x
state_covariance = flx.cross_covariance(members, members)  # → (D_x, D_x)
# today (filterax)
# Ĝ
gain = flx.kalman_gain(members, mapped, noise)  # → (D_x, D_y)


# user
# (I − G H) Σ̂_x (I − G H)ᵀ + G Σ_ε Gᵀ
def joseph(trial: Float[Array, "state obs"]) -> Float[Array, "state state"]:
    # I − G H
    remainder = jnp.eye(6) - trial @ obs_map  # → (D_x, D_x)
    # (I − G H) Σ̂_x (I − G H)ᵀ
    first = remainder @ state_covariance @ einx.id("a b -> b a", remainder)
    # G Σ_ε Gᵀ
    second = trial @ noise_matrix @ einx.id("a b -> b a", trial)
    return first + second  # → (D_x, D_x)


# user
# tr Σ_x|y(G)
def total_variance(trial: Float[Array, "state obs"]) -> Float[Array, ""]:
    return jnp.trace(joseph(trial))  # → ()


# user
# Σ̂_x − G H Σ̂_x
def short_form(trial: Float[Array, "state obs"]) -> Float[Array, "state state"]:
    return state_covariance - trial @ obs_map @ state_covariance  # → (D_x, D_x)


print(f"{jnp.abs(jax.grad(total_variance)(gain)).max():.0e}")
print(f"{jnp.abs(joseph(gain) - short_form(gain)).max():.0e}")
print(f"{jnp.abs(joseph(1.1 * gain) - short_form(1.1 * gain)).max():.3f}")

# y
target = jnp.array([1.0, -0.5, 0.2])  # → (D_y,)
# today (filterax)
# r̄ = y − h̄
residual = target - flx.ensemble_mean(mapped)  # → (D_y,)
# today (filterax)
# x̄_t|t = x̄ + Ĝ r̄
gain_mean = flx.ensemble_mean(members) + einx.dot(  # → (D_x,)
    "state obs, obs -> state", gain, residual
)
# today (filterax)
analysed = flx.filters.ETKF().analysis(
    members, target, lambda state: state[:3], noise
).particles
# today (filterax)
etkf_mean = flx.ensemble_mean(analysed)  # → (D_x,)
# today (filterax)
etkf_covariance = flx.cross_covariance(analysed, analysed)  # → (D_x, D_x)
mean_gap = jnp.abs(gain_mean - etkf_mean).max()
covariance_gap = jnp.abs(short_form(gain) - etkf_covariance).max()
print(f"{mean_gap:.0e} {covariance_gap:.0e}")
```

It prints `7e-16`, `3e-16`, `0.115` and `6e-16 6e-16`.
The gradient vanishes at $\hat{\mat{G}}$, the two forms agree there, and they differ once the gain is scaled by 1.1.
The last line says that [](#eq-gain-analysis) is what the ETKF computes.

## Differentiating the gain

Both routes are differentiable, so the gain can sit inside a training loop ([chapter 14](14_differentiable.md)).
[](#code-gain-gradient) differentiates with respect to the noise variance on each route, and compares with a central difference.

```{code-block} python
:caption: The gradient of a scalar of the gain with respect to the noise variance, on both routes.
:label: code-gain-gradient
# Shapes: N_E = 20 members; D_x = 100; D_y = 3 or 30


# user
# 𝓛(σ²) = ∑ sin Ĝ, with Σ_ε = σ² I
def gain_loss(
    variance: Float[Array, ""], mapped: Float[Array, "member obs"]
) -> Float[Array, ""]:
    # today (lineax)
    scaled = lx.DiagonalLinearOperator(variance * jnp.ones(mapped.shape[1]))
    # today (filterax)
    return jnp.sum(jnp.sin(flx.kalman_gain(wide, mapped, scaled)))  # → ()


wide = jr.normal(jr.key(6), (20, 100))  # → (N_E, D_x)
at = jnp.asarray(0.2)
step = 1e-6
for obs_size in (3, 30):
    wide_mapped = wide[:, :obs_size]  # → (N_E, D_y)
    gradient = jax.grad(gain_loss)(at, wide_mapped)  # → ()
    above = gain_loss(at + step, wide_mapped)  # → ()
    below = gain_loss(at - step, wide_mapped)  # → ()
    print(f"{obs_size}: {gradient:.6f} {(above - below) / (2.0 * step):.6f}")
```

It prints `3: 0.445424 0.445424` and `30: -9.795411 -9.795411`, the first on the dense route and the second through Woodbury.

## Innovation statistics

The residual and its covariance tell you whether the filter's picture of itself is consistent with the data, and `innovation_statistics` returns them.
Define the whitened residual $\bar{\vect{r}}_\text{w} \in \mathbb{R}^{D_y}$ and the log-likelihood of the forecast:

$$\bar{\vect{r}}_\text{w} = \hat{\gmat{\Sigma}}_y^{-1/2} \bar{\vect{r}}, \qquad \log p(\vect{y}) = -\frac{1}{2} \big(D_y \log 2 \pi + \log \vert \hat{\gmat{\Sigma}}_y \vert + \bar{\vect{r}}^\top \hat{\gmat{\Sigma}}_y^{-1} \bar{\vect{r}}\big).$$ (eq-gain-whitened)

The function returns four things:

| Key | Value | How it is computed |
| --- | --- | --- |
| `innovation` | $\bar{\vect{r}} = \vect{y} - \bar{\vect{h}}$ | direct |
| `innovation_cov` | $\hat{\gmat{\Sigma}}_y$ | a `gaussx.LowRankUpdate` of `obs_noise` |
| `normalized_innovation` | $\bar{\vect{r}}_\text{w}$ | the principal root, by a contour quadrature of 30 shifted solves |
| `log_likelihood` | $\log p(\vect{y})$ | by QR when $N_\mathcal{E} > D_y$, else through [](#eq-gain-woodbury) and [](#eq-gain-determinant) |

The principal root does not depend on the order of the observations, whereas a Cholesky factor would.
With $N_\mathcal{E} > D_y$ the shifted solves use a formed $\hat{\gmat{\Sigma}}_y$, and otherwise they go through Woodbury.

The QR route exists because $N_\mathcal{E} > D_y$ makes the capacitance the larger system, and forming $\hat{\gmat{\Sigma}}_y$ would square its condition number.
Stack the factors, $\mat{S} = \begin{bmatrix} \mat{U}_h & \gmat{\Sigma}_\varepsilon^{1/2} \end{bmatrix}^\top$, and take the triangular factor $\mat{T} \in \mathbb{R}^{D_y \times D_y}$ of its thin QR.
Then $\mat{T}^\top \mat{T} = \mat{S}^\top \mat{S} = \hat{\gmat{\Sigma}}_y$, and

$$\log \vert \hat{\gmat{\Sigma}}_y \vert = 2 \sum_i \log \vert [\mat{T}]_{ii} \vert, \qquad \bar{\vect{r}}^\top \hat{\gmat{\Sigma}}_y^{-1} \bar{\vect{r}} = \Vert \mat{T}^{-\top} \bar{\vect{r}} \Vert_2^2,$$ (eq-gain-qr)

at $O(N_\mathcal{E} D_y^2)$.
Filter analyses report the same quantity, as `AnalysisResult.log_likelihood`.

[](#code-gain-innovation) checks all four against a dense reference with a nonlinear $h$, on each side of the rule.

```{code-block} python
:caption: The four results of `innovation_statistics` against a dense eigendecomposition, in float64.
:label: code-gain-innovation
# Shapes: N_E = 30 or 4 members; D_x = 8; D_y = 3


# user
# h(x) = tanh(x₀:₃)
def tanh_obs(state: Float[Array, " state"]) -> Float[Array, " obs"]:
    return jnp.tanh(state[:3])


# user
# Dense reference, in numpy
def statistics_by_hand(mapped, target, variance):
    centred = mapped - mapped.mean(axis=0)
    residual = target - mapped.mean(axis=0)
    size = len(target)
    covariance = centred.T @ centred / (len(mapped) - 1.0) + variance * np.eye(size)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    root_inverse = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    whitened = root_inverse @ residual
    logdet = np.log(eigenvalues).sum()
    log_prob = -0.5 * (size * np.log(2.0 * np.pi) + logdet + whitened @ whitened)
    return residual, covariance, whitened, log_prob


# y
target = jnp.array([0.3, -0.1, 0.2])  # → (D_y,)
# today (lineax)
# Σ_ε = 0.05 I
noise = lx.DiagonalLinearOperator(0.05 * jnp.ones(3))  # (D_y, D_y) operator
draws = jr.normal(jr.key(0), (30, 8))  # → (30, D_x)
for num_members in (30, 4):
    members = draws[:num_members]  # → (N_E, D_x)
    # today (filterax)
    stats = flx.innovation_statistics(members, target, tanh_obs, noise)
    mapped = jax.vmap(tanh_obs)(members)  # → (N_E, D_y)
    expected = statistics_by_hand(np.asarray(mapped), np.asarray(target), 0.05)
    errors = [
        jnp.abs(stats["innovation"] - expected[0]).max(),
        jnp.abs(stats["innovation_cov"].as_matrix() - expected[1]).max(),
        jnp.abs(stats["normalized_innovation"] - expected[2]).max(),
        jnp.abs(stats["log_likelihood"] - expected[3]),
    ]
    print(num_members, [f"{error:.0e}" for error in errors])

print(type(stats["innovation_cov"]).__name__, stats["innovation_cov"].U.shape)
```

It prints `30 ['0e+00', '2e-16', '2e-16', '2e-16']`, `4 ['0e+00', '6e-17', '2e-16', '9e-16']` and `LowRankUpdate (3, 4)`, so every result matches the reference to round-off on both routes.

### Consistency

If the model is right, the residual should look like a draw with the covariance $\gmat{\Sigma}_y$, so that $\chi^2 = \bar{\vect{r}}^\top \hat{\gmat{\Sigma}}_y^{-1} \bar{\vect{r}}$ has mean $D_y$ and $\Vert \bar{\vect{r}}_\text{w} \Vert^2 = \chi^2$.
That is the basis of the observation-space checks of {cite:t}`desroziers2005diagnosis`.
A finite ensemble shifts it, because $\bar{\vect{h}}$ has its own sampling error and because $\hat{\gmat{\Sigma}}_y^{-1}$ is noisy:

$$\mathbb{E}\big[\bar{\vect{r}} \bar{\vect{r}}^\top\big] = \gmat{\Sigma}_y + \frac{1}{N_\mathcal{E}} \mat{H} \gmat{\Sigma}_x \mat{H}^\top.$$ (eq-gain-consistency)

[](#code-gain-consistency) measures the mean of $\chi^2$ on a twin experiment in which everything is correctly specified, including a truth drawn from the same density as the members.

```{code-block} python
:caption: The mean of $\chi^2$ with a correctly specified model, and $D_y = 3$.
:label: code-gain-consistency
# Shapes: R = 20,000 repeats; N_E members; D_y = 3


# user
# χ² = r̄ᵀ Σ̂_y⁻¹ r̄ on one twin experiment
def chi_square(key: Array, num_members: int) -> Float[Array, ""]:
    member_key, truth_key, noise_key = jr.split(key, 3)
    # h⁽ᵉ⁾ ~ 𝒩(𝟎, I), the members already mapped by H
    twin_mapped = jr.normal(member_key, (num_members, 3))  # → (N_E, D_y)
    # H x_true ~ 𝒩(𝟎, I)
    truth = jr.normal(truth_key, (3,))  # → (D_y,)
    # y = H x_true + ε
    twin_target = truth + jnp.sqrt(0.05) * jr.normal(noise_key, (3,))  # → (D_y,)
    # today (filterax)
    # r̄ = y − h̄
    residual = twin_target - flx.ensemble_mean(twin_mapped)  # → (D_y,)
    # today (filterax)
    # Σ̂_h
    sampled = flx.cross_covariance(twin_mapped, twin_mapped)  # → (D_y, D_y)
    # Σ̂_y⁻¹ r̄
    solved = jnp.linalg.solve(sampled + 0.05 * jnp.eye(3), residual)  # → (D_y,)
    return einx.dot("obs, obs ->", residual, solved)  # → ()


keys = jr.split(jr.key(9), 20000)  # → (R,)
for num_members in (10, 30, 1000):
    values = jax.lax.map(  # → (R,)
        lambda key: chi_square(key, num_members), keys, batch_size=2000
    )
    error_bar = values.std() / np.sqrt(20000.0)
    print(f"{num_members}: {values.mean():.2f} ± {error_bar:.2f}")
```

It prints:

| $N_\mathcal{E}$ | Mean of $\chi^2$ |
| --- | --- |
| 10 | `5.28 ± 0.04` |
| 30 | `3.51 ± 0.02` |
| 1000 | `3.00 ± 0.02` |

A value above $D_y = 3$ at small $N_\mathcal{E}$ is therefore not evidence of a bad model.
Read the statistic against what the same ensemble size gives on a correct model, and see [chapter 13](13_diagnostics.md) for how to use it in practice.

## Float32 and small observation noise

Both routes lose accuracy when the noise is small, and they lose it differently.
The dense route has only the condition number of $\hat{\gmat{\Sigma}}_y$ to contend with.
The Woodbury route inverts $\gmat{\Sigma}_\varepsilon$ and then subtracts a nearly equal quantity, so with noise small against the ensemble term it cancels badly.
[](#code-gain-float32) measures the relative error of the gain against a float64 reference, and the absolute error of the log-likelihood.

```{code-block} python
:caption: Float32 gains on both routes, and float32 log-likelihoods, against float64 references.
:label: code-gain-float32
# Shapes: N_E members; D_x = 6 or 20; D_y observations

rng = np.random.default_rng(0)
for num_members, state_size, obs_size, variances in (
    (30, 6, 3, [1.0, 1.0, 1.0]),
    (30, 6, 3, [1e-4, 1e-4, 1e-4]),
    (30, 6, 3, [1e-6, 1.0, 1e6]),
    (5, 20, 12, [1.0] * 12),
    (5, 20, 12, [1e-4] * 12),
):
    double = rng.normal(size=(num_members, state_size))
    image = double[:, :obs_size]
    spread = np.array(variances)
    reference = gain_formed(double, image, spread)
    single = jnp.asarray(double, jnp.float32)  # → (N_E, D_x)
    single_image = jnp.asarray(image, jnp.float32)  # → (N_E, D_y)
    # today (lineax)
    single_noise = lx.DiagonalLinearOperator(jnp.asarray(spread, jnp.float32))
    relative = []
    for route in (True, False):
        # today (filterax)
        gain = flx.kalman_gain(
            single, single_image, single_noise, dense_innovation=route
        )
        error = np.abs(np.asarray(gain, np.float64) - reference).max()
        relative.append(f"{error / np.abs(reference).max():.0e}")
    print(num_members, obs_size, f"{spread[0]:.0e}", relative)
```

It prints:

| $N_\mathcal{E}$ | $D_y$ | Variances | Dense | Woodbury |
| --- | --- | --- | --- | --- |
| 30 | 3 | all 1 | `2e-07` | `1e-07` |
| 30 | 3 | all $10^{-4}$ | `1e-07` | `1e-03` |
| 30 | 3 | $10^{-6}$ to $10^{6}$ | `6e-08` | `1e-02` |
| 5 | 12 | all 1 | `4e-07` | `3e-07` |
| 5 | 12 | all $10^{-4}$ | `1e-03` | `3e-03` |

With $N_\mathcal{E} \ge D_y$ the default dense route holds float32 accuracy even when the variances span twelve decades, and forcing Woodbury there costs several digits.
With $N_\mathcal{E} < D_y$ Woodbury is the default, and a small noise variance costs digits on either route, so use float64 there.
[Chapter 18](18_numerics.md) collects these limits across the library.

The QR route of the log-likelihood exists for the same reason.
[](#code-gain-loglik-float32) compares it with the structured route, `log_likelihood` on `innovation_covariance`, at a small noise variance.

```{code-block} python
:caption: The log-likelihood in float32 by QR and through the structured operator, against float64.
:label: code-gain-loglik-float32
# Shapes: N_E = 30 members; D_x = 6; D_y = 3

double = rng.normal(size=(30, 6))
double_mapped = double[:, :3]
goal = np.array([0.5, -0.5, 0.2])
for variance in (1.0, 1e-4, 1e-6):
    *_, expected = statistics_by_hand(double_mapped, goal, variance)
    single_mapped = jnp.asarray(double_mapped, jnp.float32)  # → (N_E, D_y)
    single_goal = jnp.asarray(goal, jnp.float32)  # → (D_y,)
    # today (lineax)
    single_noise = lx.DiagonalLinearOperator(
        variance * jnp.ones(3, jnp.float32)
    )
    # today (filterax)
    single_stats = flx.innovation_statistics(
        jnp.asarray(double, jnp.float32),
        single_goal,
        lambda state: state[:3],
        single_noise,
    )
    # today (filterax)
    structured = flx.log_likelihood(
        single_stats["innovation"],
        flx.innovation_covariance(single_mapped, single_noise),
    )
    by_qr = np.abs(np.float64(single_stats["log_likelihood"]) - expected)
    by_operator = np.abs(np.float64(structured) - expected)
    print(f"{variance:.0e}: {by_qr:.0e} {by_operator:.0e}")
```

It prints:

| Noise variance | QR | Structured operator |
| --- | --- | --- |
| 1 | `2e-07` | `2e-07` |
| $10^{-4}$ | `3e-07` | `5e-06` |
| $10^{-6}$ | `3e-07` | `2e-02` |

The QR route holds float32 accuracy throughout, while the structured route loses it as the noise shrinks.
Filters and `innovation_statistics` take the QR route when $N_\mathcal{E} > D_y$, so this cost falls only on a direct call to `log_likelihood`.

## When it breaks

**The gain carries the sampling noise of its covariances.**
A true correlation of zero appears in $\hat{\gmat{\Sigma}}_{xy}$ at the size $1 / \sqrt{c}$ ([chapter 2](02_ensemble_representation.md)), so an observation moves state components it knows nothing about.
Localization tapers those entries with a function of distance, $\rho$ {cite:p}`gaspari1999construction`:

$$\hat{\mat{G}}_\rho = (\rho_{xy} \circ \hat{\gmat{\Sigma}}_{xy}) (\rho_{yy} \circ \hat{\gmat{\Sigma}}_h + \gmat{\Sigma}_\varepsilon)^{-1},$$ (eq-gain-localized)

with the element-wise product $\circ$.
The product destroys the low rank of $\hat{\gmat{\Sigma}}_h$, so `localized_kalman_gain` always forms $\hat{\gmat{\Sigma}}_y$ densely, at $O(N_\mathcal{E} D_x D_y + D_x D_y^2 + D_y^3)$, where the middle term is the solve against $D_x$ right-hand sides and dominates once $D_y > N_\mathcal{E}$.
[Chapter 11](11_localization.md) covers the trade-offs.
[](#code-gain-spurious) observes one component of a state whose components are independent, so the true gain is zero everywhere else.

```{code-block} python
:caption: Spurious gain on 99 unrelated components, its localized form, and the reduction at $\rho \equiv 1$.
:label: code-gain-spurious
# Shapes: N_E = 20 members; D_x = 100; D_y = 1

independent = jr.normal(jr.key(6), (20, 100))  # → (N_E, D_x)
observed = independent[:, :1]  # → (N_E, D_y)
# today (lineax)
# Σ_ε = 1
unit_noise = lx.DiagonalLinearOperator(jnp.ones(1))  # (D_y, D_y) operator
# today (filterax)
# Ĝ
plain = flx.kalman_gain(independent, observed, unit_noise)  # → (D_x, D_y)
state_sites = einx.id("site -> site 1", jnp.arange(100.0))  # → (D_x, 1)
obs_sites = jnp.zeros((1, 1))  # → (D_y, 1)
# today (filterax)
# ρ_xy, Gaspari-Cohn with half-width 5
taper_xy = flx.localization_matrix(state_sites, obs_sites, 5.0)  # → (D_x, D_y)
# today (filterax)
# ρ_yy
taper_yy = flx.localization_matrix(obs_sites, obs_sites, 5.0)  # → (D_y, D_y)
# today (filterax)
# Ĝ_ρ
tapered = flx.localized_kalman_gain(  # → (D_x, D_y)
    independent, observed, unit_noise, taper_xy, taper_yy
)
# today (filterax)
# Ĝ_ρ with ρ ≡ 1
untapered = flx.localized_kalman_gain(  # → (D_x, D_y)
    independent, observed, unit_noise, jnp.ones((100, 1)), jnp.ones((1, 1))
)
print(f"{plain[0, 0]:.3f} {tapered[0, 0]:.3f}")
print(f"{jnp.sqrt(jnp.mean(plain[1:] ** 2)):.3f} {jnp.abs(plain[1:]).max():.3f}")
print(f"{jnp.abs(tapered[10:]).max():.3f}")
print(f"{jnp.abs(untapered - plain).max():.0e}")
```

It prints `0.498 0.498`, `0.110 0.298`, `0.000` and `0e+00`.
Without localization, components with no relation to the observation receive a gain of typical size 0.11, compared with the true value of 0.5 on the observed one.
With the taper, nothing beyond its support of twice the half-width moves at all.
With $\rho \equiv 1$ the function reduces to the plain gain.

**The analysis variance is biased low.**
The gain is a nonlinear function of the sample covariance, and the analysis variance is concave in it.
For one component with variance $v$ and noise variance $s$, [](#eq-gain-posterior) gives the analysis variance $v s / (v + s)$, which is concave in $v$.
Jensen's inequality then says that an unbiased $\hat{v}$ gives $\mathbb{E}[\hat{v} s / (\hat{v} + s)] \le v s / (v + s)$.
So even with an unbiased forecast covariance, the ensemble reports too little spread after the update {cite:p}`vanleeuwen1999comment,furrer2007estimation,sacher2008sampling`.
The root of this is using the same members to estimate the gain and to be updated by it.
{cite:t}`houtekamer1998data` split the ensemble in two to avoid it.

```{code-block} python
:caption: The mean sample analysis variance with $v = s = 1$, where the exact value is $1 / 2$.
:label: code-gain-bias
# Shapes: R = 4,000 repeats; N_E members; D_x = 1

for num_members in (5, 20, 100):
    repeats = jr.normal(jr.key(4), (4000, num_members, 1))  # → (R, N_E, 1)
    # today (filterax)
    # v̂
    sampled = jax.vmap(flx.cross_covariance)(repeats, repeats)  # → (R, 1, 1)
    # v̂ s / (v̂ + s), with s = 1
    analysis_variance = sampled / (sampled + 1.0)  # → (R, 1, 1)
    mean_value = analysis_variance.mean()  # → ()
    print(f"{num_members}: {sampled.mean():.3f} {mean_value:.3f}")
```

It prints:

| $N_\mathcal{E}$ | Mean of $\hat{v}$ | Mean analysis variance |
| --- | --- | --- |
| 5 | `1.007` | `0.447` |
| 20 | `0.992` | `0.485` |
| 100 | `0.999` | `0.497` |

The forecast variance is unbiased and the analysis variance is low, with the bias shrinking as the ensemble grows.
It compounds over cycles, and it is one reason inflation ([chapter 12](12_inflation.md)) is needed.

**A singular noise covariance fails through Woodbury.**
With $N_\mathcal{E} < D_y$ and a zero variance anywhere in $\gmat{\Sigma}_\varepsilon$, the default route returns `NaN`, and `dense_innovation=True` avoids it ([chapter 3](03_observation_model.md)).

**`innovation_statistics` does not scale to very many members.**
With $N_\mathcal{E} > D_y$ it forms $\hat{\gmat{\Sigma}}_y$ through `as_matrix()`, which builds an $N_\mathcal{E} \times N_\mathcal{E}$ matrix of weights on the way ([chapter 2](02_ensemble_representation.md)).
At $N_\mathcal{E} = 200000$ and $D_y = 3$ that is a request for 160 GB in float32, which failed when we ran it.
`kalman_gain` and `filters.ETKF().analysis` both completed on the same ensemble.

## When to call the gain directly

Most users never call `kalman_gain`, because `StochasticEnKF` does it for them and the square-root filters do not form it at all.
Call it when you are writing your own update, inspecting which observations move which components, or building a diagnostic such as the degrees of freedom for signal.
Pass `dense_innovation=False` if a matrix-free `solver` needs the structured operator, since the default forms $\hat{\gmat{\Sigma}}_y$ whenever $N_\mathcal{E} \ge D_y$.

Use `innovation_statistics` when you want everything at once, for monitoring.
For a log-likelihood alone, a filter's `AnalysisResult.log_likelihood` already has it, without the 30 shifted solves of the whitened residual.

## See also

- [Chapter 3](03_observation_model.md): where $\tilde{\mat{H}}$, $\gmat{\Sigma}_\varepsilon$ and the statistical linearisation come from
- [Chapter 5](05_stochastic_enkf.md): one gain, applied to each member's perturbed residual
- [Chapter 6](06_etkf.md): the same capacitance as a transform of the anomalies
- [Chapter 11](11_localization.md): the tapers in [](#eq-gain-localized)
- [Chapter 13](13_diagnostics.md): consistency checks built on the innovation statistics
- [API: primitives](https://jejjohnson.github.io/filterax/reference/primitives/), [localization](https://jejjohnson.github.io/filterax/reference/localization/) and [diagnostics](https://jejjohnson.github.io/filterax/reference/diagnostics/)

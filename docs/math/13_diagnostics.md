---
numbering:
  enumerator: "13.%s"
---

# Diagnostics and likelihood

A filter that runs is not a filter that works.
The functions in this chapter ask whether an assimilation is statistically healthy.
They detect a wrong covariance, a wrong amount of inflation or too small an ensemble, and they do not fix any of them.
All of them are pure functions of arrays you already have, so they cost little and own no state.

One statement organises the chapter.
If the forecast and the noise are correctly specified, the innovation is a draw from $\mathcal{N}(\vect{0}, \gmat{\Sigma}_y)$ and the truth is one more member of the ensemble.
Each diagnostic is a different projection of that statement.
For each one this chapter gives the value it should take, the finite-ensemble correction to that value, and how it fails.
Those corrections matter: in the toy problem below, a calibrated ensemble of 10 members with 3 observations has a $\chi^2 / D_y$ of about 1.3.

## The innovation and its covariance

Everything starts from the part of the observation that the forecast did not predict.
With the forecast members mapped to $\vect{h}^{(e)} = h(\vect{x}^{(e)})$, the innovation $\bar{\vect{r}} \in \mathbb{R}^{D_y}$ and the covariance $\hat{\gmat{\Sigma}}_y \in \mathbb{R}^{D_y \times D_y}$ are

$$\bar{\vect{r}} = \vect{y} - \bar{\vect{h}}, \qquad \hat{\gmat{\Sigma}}_y = \hat{\gmat{\Sigma}}_h + \gmat{\Sigma}_\varepsilon.$$ (eq-diag-innovation)

`filterax.utils.innovation` computes the first, and `innovation_covariance` the second as a `gaussx.LowRankUpdate`.
Two normalisations follow.
`normalized_innovation` divides each component by its own standard deviation, so it uses only the diagonal.
`innovation_statistics` whitens with the principal inverse root, which uses all of $\hat{\gmat{\Sigma}}_y$:

$$\frac{\bar{r}_i}{\sqrt{[\hat{\gmat{\Sigma}}_y]_{ii}}}, \qquad \bar{\vect{r}}_\text{wh} = \hat{\gmat{\Sigma}}_y^{-1/2} \bar{\vect{r}}.$$ (eq-diag-normalised)

The first has components near $\mathcal{N}(0, 1)$ that can still be correlated, and the second should be close to $\mathcal{N}(\vect{0}, \mat{I})$.
A Cholesky factor would also whiten, but its result depends on the order of the observations.
The principal root does not, so a permuted observation vector gives the same whitened residual, permuted.

### The likelihood

The innovation likelihood is both a diagnostic and the training signal of [chapter 14](14_differentiable.md):

$$\log \mathcal{N}(\bar{\vect{r}}; \vect{0}, \hat{\gmat{\Sigma}}_y) = -\frac{1}{2} \left(D_y \log 2 \pi + \log \vert \hat{\gmat{\Sigma}}_y \vert + \bar{\vect{r}}^\top \hat{\gmat{\Sigma}}_y^{-1} \bar{\vect{r}}\right).$$ (eq-diag-log-likelihood)

`log_likelihood` evaluates it for any operator through gaussx dispatch.
On a `LowRankUpdate` that means the matrix-determinant lemma and Woodbury, at $O(N_\mathcal{E}^2 D_y + N_\mathcal{E}^3)$.
With $N_\mathcal{E} > D_y$ the filters and `innovation_statistics` take a thin QR of the stacked factors $[\mat{U}, \gmat{\Sigma}_\varepsilon^{1/2}]$ instead, where $\mat{U} = \tilde{\mat{H}}^\top / \sqrt{N_\mathcal{E} - 1}$, at $O(N_\mathcal{E} D_y^2)$.
That route gives the triangular $\mat{T}$ with $\mat{T}^\top \mat{T} = \hat{\gmat{\Sigma}}_y$ without squaring a condition number.
The log-determinant then comes off the diagonal of $\mat{T}$.

[](#code-diag-innovation) computes all of this on one forecast and checks it against a dense reference.

```{code-block} python
:caption: One forecast through every innovation function, against a numpy reference, in float64.
:label: code-diag-innovation
import einx
import filterax as flx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
from jaxtyping import Array, Float

jax.config.update("jax_enable_x64", True)

# Shapes: N_E = 40 members; D_x = 6; D_y = 3

# a centre shared by the truth and the members
centre = jr.normal(jr.key(1), (6,))  # → (D_x,)
# x_true ~ 𝒩(centre, I)
x_true = centre + jr.normal(jr.key(2), (6,))  # → (D_x,)
# x⁽ᵉ⁾ ~ 𝒩(centre, I)
members = centre + jr.normal(jr.key(3), (40, 6))  # → (N_E, D_x)
# today (lineax)
# Σ_ε = 0.5 I
noise = lx.DiagonalLinearOperator(0.5 * jnp.ones(3))  # (D_y, D_y) operator
# y = H x_true + ε
observation = x_true[:3] + jnp.sqrt(0.5) * jr.normal(jr.key(4), (3,))  # → (D_y,)


# user
# h(x) selects the first three components
def observe(state: Float[Array, " state"]) -> Float[Array, " obs"]:
    return state[:3]


# today (filterax)
statistics = flx.innovation_statistics(members, observation, observe, noise)
# r̄
residual = statistics["innovation"]  # → (D_y,)
# Σ̂_y
innovation_op = statistics["innovation_cov"]  # (D_y, D_y) operator
innovation_dense = innovation_op.as_matrix()  # → (D_y, D_y)
# Σ̂_y⁻¹ r̄
solved = jnp.linalg.solve(innovation_dense, residual)  # → (D_y,)
# r̄_wh
whitened = statistics["normalized_innovation"]  # → (D_y,)
# h̄
mapped = jax.vmap(observe)(members)  # → (N_E, D_y)
mapped_mean = einx.mean("[member] obs", mapped)  # → (D_y,)

# today (filterax)
print(f"{jnp.abs(flx.utils.innovation(observation, mapped_mean) - residual).max():.0e}")
# today (filterax)
# rᵢ / √[Σ̂_y]ᵢᵢ
print(
    flx.utils.normalized_innovation(
        observation, mapped_mean, jnp.diagonal(innovation_dense)
    ).round(3)
)
print(whitened.round(3))
# today (filterax)
# χ² = r̄ᵀ Σ̂_y⁻¹ r̄
chi2 = flx.utils.chi2_consistency(residual, solved)  # → ()
# today (filterax)
# χ² / D_y
chi2_ratio = flx.utils.chi2_normalized(residual, solved, 3)  # → ()
# ‖r̄_wh‖²
print(f"{chi2:.8f} {jnp.sum(whitened**2):.8f} {chi2_ratio:.5f}")


# user
# log 𝒩(r̄; 𝟎, Σ̂_y), dense, in plain numpy
def dense_log_likelihood(ensemble, target, variance):
    size = ensemble.shape[1]
    covariance = np.cov(ensemble, rowvar=False) + variance * np.eye(size)
    gap = target - ensemble.mean(0)
    _, logdet = np.linalg.slogdet(covariance)
    quadratic = gap @ np.linalg.solve(covariance, gap)
    return -0.5 * (size * np.log(2.0 * np.pi) + logdet + quadratic)


# today (filterax)
print(
    f"{statistics['log_likelihood']:.8f}",
    f"{flx.log_likelihood(residual, innovation_op):.8f}",
    f"{dense_log_likelihood(np.asarray(mapped), np.asarray(observation), 0.5):.8f}",
)
# today (filterax)
analysis = flx.filters.ETKF().analysis(members, observation, observe, noise)
print(f"{analysis.log_likelihood:.8f}")

# the observations in another order
order = jnp.array([2, 0, 1])  # → (D_y,)
# today (filterax)
reordered = flx.innovation_statistics(
    members, observation[order], lambda state: observe(state)[order], noise
)["normalized_innovation"]  # → (D_y,)
print(f"{jnp.abs(reordered - whitened[order]).max():.0e}")
# L⁻¹ r̄, with L L ᵀ = Σ̂_y
by_cholesky = np.linalg.solve(np.linalg.cholesky(innovation_dense), residual)
permuted = np.asarray(innovation_dense)[np.ix_(order, order)]
reordered_cholesky = np.linalg.solve(np.linalg.cholesky(permuted), residual[order])
print(f"{np.abs(reordered_cholesky - by_cholesky[np.asarray(order)]).max():.2f}")
```

The first line prints `0e+00`, so `utils.innovation` and the `innovation` key agree.
The next two print `[-0.878 -0.897  0.52 ]` for the per-component form and `[-0.807 -0.954  0.704]` for the whitened one.
They differ because $\hat{\gmat{\Sigma}}_y$ has off-diagonal entries.
The line `2.05616259 2.05616259 0.68539` holds $\chi^2$ from a solve, the squared norm of the whitened residual and $\chi^2 / D_y$.
All three log-likelihoods and the numpy reference print `-4.50287410`, and so does `analysis.log_likelihood`.
Reordering the observations changes the principal-root residual only by a permutation, to `1e-16`, while the Cholesky residual changes by up to `0.33`.

## Chi-squared consistency

Contracting $\bar{\vect{r}}$ against $\hat{\gmat{\Sigma}}_y^{-1}$ gives the classic consistency statistic {cite:p}`mehra1970identification`:

$$\chi^2 = \bar{\vect{r}}^\top \hat{\gmat{\Sigma}}_y^{-1} \bar{\vect{r}}, \qquad \mathbb{E}[\chi^2] = D_y, \qquad \mathbb{V}[\chi^2] = 2 D_y.$$ (eq-diag-chi2)

Its two forms are `chi2_consistency` and `chi2_normalized`, which returns $\chi^2 / D_y$ with a target of 1 and a null standard deviation of $\sqrt{2 / D_y}$.
Both take $\hat{\gmat{\Sigma}}_y^{-1} \bar{\vect{r}}$ already solved, so they use whichever solver produced the analysis.
A value that stays above 1 means an overconfident, underdispersive filter, which needs inflation ([chapter 12](12_inflation.md)), and a value below 1 means an overdispersive one.

### The finite-ensemble correction

[](#eq-diag-chi2) holds when $\hat{\gmat{\Sigma}}_y$ is the true covariance of $\bar{\vect{r}}$, and an ensemble breaks that in two ways.
The mean $\bar{\vect{h}}$ has its own sampling error, which adds $\gmat{\Sigma}_h / N_\mathcal{E}$ to the covariance of $\bar{\vect{r}}$.
And the inverse is a convex function, so the inverse of a noisy $\hat{\gmat{\Sigma}}_y$ is too large on average.
Both push $\chi^2 / D_y$ above 1 for a perfectly calibrated filter.

```{code-block} python
:caption: The mean of $\chi^2 / D_y$ when the truth is one more member, so that the filter is calibrated by construction.
:label: code-diag-chi2-null
# Shapes: 10,000 draws; N_E members; D_y = 3


# user
# χ² / D_y for one draw, with truth and members exchangeable, and Σ_ε = 0.5 I
def null_ratio(key: jax.Array, num_members: int) -> Float[Array, ""]:
    truth_key, member_key, noise_key = jr.split(key, 3)
    # h_true ~ 𝒩(𝟎, I)
    true_mapped = jr.normal(truth_key, (3,))  # → (D_y,)
    # h⁽ᵉ⁾ ~ 𝒩(𝟎, I)
    draws = jr.normal(member_key, (num_members, 3))  # → (N_E, D_y)
    # y = h_true + ε
    target = true_mapped + jnp.sqrt(0.5) * jr.normal(noise_key, (3,))  # → (D_y,)
    # r̄
    gap = target - einx.mean("[member] obs", draws)  # → (D_y,)
    # today (filterax)
    # Σ̂_y
    covariance = flx.innovation_covariance(draws, noise).as_matrix()  # → (D_y, D_y)
    # today (filterax)
    # χ² / D_y
    return flx.utils.chi2_normalized(gap, jnp.linalg.solve(covariance, gap), 3)


for num_members in (10, 40, 200):
    keys = jr.split(jr.key(0), 10_000)
    ratios = jax.vmap(null_ratio, in_axes=(0, None))(keys, num_members)
    print(num_members, f"{ratios.mean():.3f}")
print(f"{jnp.sqrt(2.0 / 3.0):.3f}")
```

It prints `1.273`, `1.054` and `1.005` for $N_\mathcal{E} = 10$, 40 and 200, and `0.816` for the null standard deviation $\sqrt{2 / 3}$ of one window.
So a calibrated 10-member ensemble reads about 1.27 here.
Tuning inflation until the statistic reaches exactly 1 would ask for more spread than a calibrated filter has.
The correction fades as $N_\mathcal{E}$ grows, and $N_\mathcal{E} \gg D_y$ is the case where [](#eq-diag-chi2) can be read directly.

## Spread and skill

The per-variable spread $\sigma_i = \sqrt{[\hat{\gmat{\Sigma}}_x]_{ii}}$ is `ensemble_spread`, and `rms_spread` reduces it to one number.
A collapsing spread warns of filter divergence, and it needs no truth at all.

In a twin experiment the truth $\vect{x}_\text{true}$ is known, so `rmse_vs_truth` gives the error of the mean, and `spread_skill_ratio` compares the two {cite:p}`whitaker1998relationship`:

$$\mathrm{RMSE} = \sqrt{\frac{1}{D_x} \sum_i (\bar{x}_i - x_{\text{true}, i})^2}, \qquad \mathrm{SSR} = \frac{\sqrt{\frac{1}{D_x} \sum_i \sigma_i^2}}{\mathrm{RMSE}}.$$ (eq-diag-spread-skill)

$\mathrm{SSR} < 1$ means underdispersive, and $\mathrm{SSR} > 1$ overdispersive.

The target is not exactly 1 {cite:p}`fortin2014why`.
If the truth and the members are exchangeable with variance $\sigma^2$, the mean $\bar{x}$ has variance $\sigma^2 / N_\mathcal{E}$ and is independent of the truth.
So $\mathbb{E}[(\bar{x} - x_\text{true})^2] = \sigma^2 (1 + 1 / N_\mathcal{E})$, and the calibrated value is

$$\mathrm{SSR}_\text{calibrated} = \sqrt{\frac{N_\mathcal{E}}{N_\mathcal{E} + 1}}.$$ (eq-diag-ssr-target)

The ratio also needs care across time.
`spread_skill_ratio` works on one window, and the mean of those ratios is biased upwards, so pool the squares of the spread and the error first.

```{code-block} python
:caption: The spread-skill ratio of a calibrated ensemble, pooled and averaged per window, against the finite-ensemble target.
:label: code-diag-spread-skill
# Shapes: 5,000 windows; N_E members; D_x = 40


# user
# spread, error and ratio in one window, with truth and members exchangeable
def one_window(key: jax.Array, num_members: int) -> tuple[Array, Array, Array]:
    truth_key, member_key = jr.split(key)
    # x_true ~ 𝒩(𝟎, I)
    truth_draw = jr.normal(truth_key, (40,))  # → (D_x,)
    # x⁽ᵉ⁾ ~ 𝒩(𝟎, I)
    ensemble = jr.normal(member_key, (num_members, 40))  # → (N_E, D_x)
    ensemble_mean = einx.mean("[member] state", ensemble)  # → (D_x,)
    # today (filterax)
    return (
        flx.utils.rms_spread(ensemble),
        flx.utils.rmse_vs_truth(ensemble_mean, truth_draw),
        flx.utils.spread_skill_ratio(ensemble, truth_draw),
    )


for num_members in (5, 10, 40):
    keys = jr.split(jr.key(1), 5000)
    spreads, errors, ratios = jax.vmap(one_window, in_axes=(0, None))(keys, num_members)
    # √(mean spread²) / √(mean error²)
    pooled = jnp.sqrt(jnp.mean(spreads**2) / jnp.mean(errors**2))  # → ()
    target = np.sqrt(num_members / (num_members + 1.0))
    print(num_members, f"{pooled:.3f} {ratios.mean():.3f} {target:.3f}")
# today (filterax)
# σᵢ of a small ensemble
print(flx.utils.ensemble_spread(jnp.array([[0.0, 0.0], [2.0, 4.0]])).round(4))
```

It prints:

| $N_\mathcal{E}$ | Pooled over windows | Mean of the per-window ratios | [](#eq-diag-ssr-target) |
| --- | --- | --- | --- |
| 5 | 0.912 | 0.928 | 0.913 |
| 10 | 0.953 | 0.971 | 0.953 |
| 40 | 0.986 | 1.005 | 0.988 |

The pooled ratio follows the target, and the mean of per-window ratios sits about 0.02 above it.
So $\mathrm{SSR} = 0.95$ is a calibrated reading at $N_\mathcal{E} = 10$.
The last line prints `[1.4142 2.8284]`, the two spreads of a small example.

## Rank histograms

A rank histogram tests the whole distribution and not just its second moment {cite:p}`hamill2001interpretation`.
For each variable and each window, `rank_histogram` counts how many of the $N_\mathcal{E}$ members lie strictly below the truth, which gives a rank in $\{0, \dots, N_\mathcal{E}\}$.
If the truth is exchangeable with the members, every one of the $N_\mathcal{E} + 1$ ranks is equally likely.
A U shape means the truth often falls outside the ensemble, so it is underdispersive.
A dome means overdispersive, and a slope means a bias.

`rank_histogram_chi2` turns flatness into one number, with $O_k$ the count in bin $k$:

$$\chi^2_\text{rank} = \sum_k \frac{(O_k - E_k)^2}{E_k}, \qquad E_k = \frac{N_\text{total}}{N_\text{bins}}.$$ (eq-diag-rank-chi2)

Under a uniform null with independent counts this is $\chi^2$ with $N_\text{bins} - 1$ degrees of freedom, so its expectation is $N_\text{bins} - 1 = N_\mathcal{E}$.

```{code-block} python
:caption: Rank histograms of calibrated, underdispersive, overdispersive and biased ensembles, then two ways to get a misleading result.
:label: code-diag-rank
# Shapes: T = 2,000 windows; N_E = 9 members; D_x = 2


# user
# draws of (ensemble, truth), with members 𝒩(bias, scale²) and truth 𝒩(0, 1)
def rank_case(scale: float, bias: float) -> tuple[Array, Array]:
    # x_true ~ 𝒩(𝟎, I)
    truth_draws = jr.normal(jr.key(0), (2000, 2))  # → (T, D_x)
    # x⁽ᵉ⁾
    unit_draws = jr.normal(jr.key(1), (2000, 9, 2))  # → (T, N_E, D_x)
    ensemble_draws = bias + scale * unit_draws  # → (T, N_E, D_x)
    return ensemble_draws, truth_draws


for name, scale, bias in (
    ("calibrated", 1.0, 0.0),
    ("underdispersive", 0.5, 0.0),
    ("overdispersive", 2.0, 0.0),
    ("biased", 1.0, 0.5),
):
    ensemble_draws, truth_draws = rank_case(scale, bias)
    # today (filterax)
    counts = flx.utils.rank_histogram(ensemble_draws, truth_draws)  # → (N_E + 1,)
    # today (filterax)
    print(name, counts, f"{flx.utils.rank_histogram_chi2(counts):.1f}")

# the same calibrated forecast, with both components a copy of one draw
ensemble_draws, truth_draws = rank_case(1.0, 0.0)
copied_truth = einx.id("time -> time state", truth_draws[:, 0], state=2)  # → (T, D_x)
copied_ensemble = einx.id(  # → (T, N_E, D_x)
    "time member -> time member state", ensemble_draws[:, :, 0], state=2
)
# today (filterax)
counts = flx.utils.rank_histogram(copied_ensemble, copied_truth)  # → (N_E + 1,)
# today (filterax)
print("copied", f"{flx.utils.rank_histogram_chi2(counts):.1f}")

# yₜ = xₜ + εₜ, εₜ ~ 𝒩(𝟎, I), verifying against observations
noisy_truth = truth_draws + jr.normal(jr.key(2), (2000, 2))  # → (T, D_x)
# today (filterax)
counts = flx.utils.rank_histogram(ensemble_draws, noisy_truth)  # → (N_E + 1,)
# today (filterax)
print("against y", counts, f"{flx.utils.rank_histogram_chi2(counts):.1f}")
# h⁽ᵉ⁾ + ε⁽ᵉ⁾, the members with observation noise added
noisy_members = ensemble_draws + jr.normal(jr.key(3), (2000, 9, 2))  # → (T, N_E, D_x)
# today (filterax)
counts = flx.utils.rank_histogram(noisy_members, noisy_truth)  # → (N_E + 1,)
# today (filterax)
print("noise in both", counts, f"{flx.utils.rank_histogram_chi2(counts):.1f}")
```

With $N_\mathcal{E} = 9$ there are 10 bins, and the null expectation of $\chi^2_\text{rank}$ is 9.

| Case | Counts | $\chi^2_\text{rank}$ |
| --- | --- | --- |
| calibrated | `[420 392 410 415 393 382 390 416 366 416]` | 7.3 |
| underdispersive, members with half the spread | `[985 325 277 241 213 208 220 239 348 944]` | 2042.6 |
| overdispersive, members with twice the spread | `[114 242 426 583 647 669 527 440 256  96]` | 1012.9 |
| biased, members shifted by 0.5 | `[812 595 522 462 365 334 266 248 211 185]` | 887.7 |
| truth against $\vect{y}$, members without noise | `[660 399 321 326 298 303 301 331 359 702]` | 516.4 |
| truth against $\vect{y}$, members with noise added | `[395 369 411 431 410 385 395 406 374 424]` | 9.3 |

The shapes are as described above.
Two entries warn about use.
Verifying against noisy observations gives a false U from a calibrated ensemble until the members carry the same noise.
And the counts must be independent.
The same calibrated forecast with two copied components prints as `copied 32.2`, far above the null of 9, with no calibration problem at all.
Neighbouring sites and successive windows of a real run are correlated in the same way, if less extremely.

## CRPS

The continuous ranked probability score is a strictly proper scoring rule for a forecast distribution against one scalar {cite:p}`gneiting2007strictly`.
It rewards both calibration and sharpness.
It compares the forecast distribution function $F$ with a step at the observation $y$:

$$\mathrm{CRPS} = \int \left(F(x) - \mathbb{1}\{x \ge y\}\right)^2 \mathrm{d} x = \mathbb{E} \vert x - y \vert - \frac{1}{2} \mathbb{E} \vert x - x' \vert,$$ (eq-diag-crps)

for independent draws $x$ and $x'$ from $F$, and it has the units of $y$.
Lower is better.
{cite:t}`hersbach2000decomposition` studies it for ensembles, with $F$ the empirical distribution of the members.

The double sum in the second term has a one-line reduction.
Take the sorted members $x_{(1)} \le \dots \le x_{(N_\mathcal{E})}$.
The value $x_{(e)}$ appears with a plus sign against the $e - 1$ smaller members, and with a minus sign against the $N_\mathcal{E} - e$ larger ones, so

$$\frac{1}{2} \mathbb{E} \vert x - x' \vert = \frac{1}{2 N_\mathcal{E}^2} \sum_{e, e'} \vert x^{(e)} - x^{(e')} \vert = \frac{1}{N_\mathcal{E}^2} \sum_e (2 e - 1 - N_\mathcal{E}) \, x_{(e)},$$ (eq-diag-crps-sorted)

which `crps_ensemble` evaluates at $O(N_\mathcal{E} \log N_\mathcal{E})$, and `crps_ensemble_batch` averages it over an observation vector.

The sum over $e, e'$ includes the pairs $e = e'$, so the ensemble score is larger than that of the distribution it samples.
For a calibrated Gaussian forecast with unit variance, $\mathbb{E} \vert x - y \vert = 2 / \sqrt{\pi}$ and the second term is $(1 - 1 / N_\mathcal{E}) / \sqrt{\pi}$, which gives

$$\mathbb{E}[\mathrm{CRPS}] = \frac{1}{\sqrt{\pi}} \left(1 + \frac{1}{N_\mathcal{E}}\right).$$ (eq-diag-crps-mean)

So scores from ensembles of different sizes are not directly comparable.

```{code-block} python
:caption: CRPS against a brute-force double sum and against its integral definition, then its finite-ensemble mean.
:label: code-diag-crps
# Shapes: N_E = 10 members; 200,001 grid points; 50,000 draws

# x⁽ᵉ⁾
scalar_members = jr.normal(jr.key(3), (10,))  # → (N_E,)
scalar_obs = jnp.asarray(0.3)  # → ()
# today (filterax)
score = flx.utils.crps_ensemble(scalar_members, scalar_obs)  # → ()
# 𝔼|x − y| − ½ 𝔼|x − x′|, by a double sum
as_numpy = np.asarray(scalar_members)
direct = np.abs(as_numpy - 0.3).mean() - np.abs(
    np.subtract.outer(as_numpy, as_numpy)
).sum() / (2.0 * 10**2)
# ∫ (F(x) − 𝟙{x ≥ y})² dx
grid_points = np.linspace(-10.0, 10.0, 200_001)
empirical = np.searchsorted(np.sort(as_numpy), grid_points, side="right") / 10.0
integral = np.trapezoid((empirical - (grid_points >= 0.3)) ** 2, grid_points)
print(f"{score:.6f} {direct:.6f} {integral:.6f}")


# user
# CRPS for one calibrated draw
def one_score(key: jax.Array, num_members: int) -> Float[Array, ""]:
    member_key, obs_key = jr.split(key)
    # today (filterax)
    return flx.utils.crps_ensemble(
        jr.normal(member_key, (num_members,)), jr.normal(obs_key, ())
    )


for num_members in (10, 40):
    keys = jr.split(jr.key(2), 50_000)
    scores = jax.vmap(one_score, in_axes=(0, None))(keys, num_members)
    print(
        num_members,
        f"{scores.mean():.3f}",
        f"{(1.0 + 1.0 / num_members) / np.sqrt(np.pi):.3f}",
    )

# today (filterax)
# the mean CRPS over D_y = 2 observations
print(
    flx.utils.crps_ensemble_batch(
        jnp.array([[0.0, 1.0], [1.0, 3.0]]), jnp.array([0.5, 2.0])
    )
)
```

The first line prints `0.233546 0.233546 0.233531`, so the sorted identity matches the double sum exactly and the integral to the resolution of its grid.
The means print `0.618 0.621` at $N_\mathcal{E} = 10$ and `0.575 0.578` at $N_\mathcal{E} = 40$, close to [](#eq-diag-crps-mean).
The batch form prints `0.375`, the mean of 0.25 and 0.5.

## Desroziers diagnostics

A long record of departures lets you read the error covariances off in observation space {cite:p}`desroziers2005diagnosis`.
Write $\bar{\vect{r}}_{t \mid t-1} = \vect{y} - \mat{H} \bar{\vect{x}}_{t \mid t-1}$ for the forecast departure, the innovation, and $\bar{\vect{r}}_{t \mid t} = \vect{y} - \mat{H} \bar{\vect{x}}_{t \mid t}$ for the analysis departure.
Both are in $\mathbb{R}^{D_y}$.

The derivation takes three lines.
The analysis mean is $\bar{\vect{x}}_{t \mid t} = \bar{\vect{x}}_{t \mid t-1} + \mat{G} \bar{\vect{r}}_{t \mid t-1}$, and $\mat{I} - \mat{H} \mat{G} = \gmat{\Sigma}_\varepsilon \gmat{\Sigma}_y^{-1}$, so

$$\bar{\vect{r}}_{t \mid t} = \gmat{\Sigma}_\varepsilon \gmat{\Sigma}_y^{-1} \bar{\vect{r}}_{t \mid t-1}.$$ (eq-diag-departures)

If the filter's $\gmat{\Sigma}_y$ is the true covariance of $\bar{\vect{r}}_{t \mid t-1}$, multiplying out gives the three statistics filterax accumulates:

$$\mathbb{E}\left[\bar{\vect{r}}_{t \mid t-1} \bar{\vect{r}}_{t \mid t-1}^\top\right] = \gmat{\Sigma}_y, \qquad \mathbb{E}\left[\bar{\vect{r}}_{t \mid t} \bar{\vect{r}}_{t \mid t-1}^\top\right] = \gmat{\Sigma}_\varepsilon, \qquad \mathbb{E}\left[\bar{\vect{r}}_{t \mid t} \bar{\vect{r}}_{t \mid t}^\top\right] = \gmat{\Sigma}_\varepsilon \gmat{\Sigma}_y^{-1} \gmat{\Sigma}_\varepsilon = \gmat{\Sigma}_\varepsilon - \mat{H} \gmat{\Sigma}_{x \mid y} \mat{H}^\top.$$ (eq-diag-desroziers)

These are `desroziers_innovation_cov`, `desroziers_R_estimate` and `desroziers_analysis_residual_cov`, each an average over the $T$ windows of an array with shape $(T, D_y)$.
The middle one is the most useful.
Subtracting it from the first estimates $\mat{H} \gmat{\Sigma}_x \mat{H}^\top$, so one record separates forecast error from observation error.

If the filter assumed $\gmat{\Sigma}_\varepsilon^\text{used}$ and $\gmat{\Sigma}_y^\text{used}$, then the same algebra gives

$$\mathbb{E}\left[\bar{\vect{r}}_{t \mid t} \bar{\vect{r}}_{t \mid t-1}^\top\right] = \gmat{\Sigma}_\varepsilon^\text{used} \left(\gmat{\Sigma}_y^\text{used}\right)^{-1} \gmat{\Sigma}_y,$$ (eq-diag-desroziers-misspecified)

which is neither the true nor the assumed noise, and in practice it moves the assumed value towards the true one.

```{code-block} python
:caption: The three statistics on a record generated from a known linear-Gaussian system, with correct and wrong noise covariances, a short record and a biased one.
:label: code-diag-desroziers
# Shapes: T = 50,000 windows; D_x = 3; D_y = 2

# Σ_x, H, Σ_ε
true_forecast = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 1.0]])
obs_matrix = np.eye(3)[:2]
true_noise = np.diag([0.5, 0.8])


# user
# forecast and analysis departures from a filter that assumes `used_noise`
def departures(used_noise, num_windows=50_000, bias=0.0):
    rng = np.random.default_rng(0)
    # G = Σ_x Hᵀ (H Σ_x Hᵀ + Σ_ε^used)⁻¹
    used_innovation = obs_matrix @ true_forecast @ obs_matrix.T + used_noise
    used_gain = true_forecast @ obs_matrix.T @ np.linalg.inv(used_innovation)
    # x_true − x̄_t|t−1 ~ 𝒩(𝟎, Σ_x)
    forecast_factor = np.linalg.cholesky(true_forecast)
    forecast_error = rng.normal(size=(num_windows, 3)) @ forecast_factor.T
    # ε ~ 𝒩(𝟎, Σ_ε)
    obs_error = rng.normal(size=(num_windows, 2)) @ np.linalg.cholesky(true_noise).T
    # r̄_t|t−1 = H (x_true − x̄_t|t−1) + ε + bias
    forecast_departure = forecast_error @ obs_matrix.T + obs_error + bias
    # r̄_t|t = (I − H G) r̄_t|t−1
    influence = obs_matrix @ used_gain
    analysis_departure = forecast_departure - forecast_departure @ influence.T
    return jnp.asarray(forecast_departure), jnp.asarray(analysis_departure)


forecast_dep, analysis_dep = departures(true_noise)
# today (filterax)
# Σ̂_y
print(flx.utils.desroziers_innovation_cov(forecast_dep).round(3))
# today (filterax)
# Σ̂_ε, not symmetric at finite T
estimate = flx.utils.desroziers_R_estimate(forecast_dep, analysis_dep)  # → (D_y, D_y)
print(estimate.round(3))
print(f"{jnp.abs(estimate - einx.id('i j -> j i', estimate)).max():.1e}")
# today (filterax)
# Σ_ε Σ_y⁻¹ Σ_ε
print(flx.utils.desroziers_analysis_residual_cov(analysis_dep).round(3))
true_innovation = obs_matrix @ true_forecast @ obs_matrix.T + true_noise
print((true_noise @ np.linalg.inv(true_innovation) @ true_noise).round(3))
# H Σ_x Hᵀ ≈ 𝔼[r̄_f r̄_fᵀ] − Σ̂_ε
print((flx.utils.desroziers_innovation_cov(forecast_dep) - estimate).round(3))

# a filter that assumes half the true noise
half_noise = 0.5 * true_noise
forecast_half, analysis_half = departures(half_noise)
# today (filterax)
print(flx.utils.desroziers_R_estimate(forecast_half, analysis_half).round(3))
half_innovation = obs_matrix @ true_forecast @ obs_matrix.T + half_noise
print((half_noise @ np.linalg.inv(half_innovation) @ true_innovation).round(3))

# today (filterax)
# only the first 100 windows
short = flx.utils.desroziers_R_estimate(forecast_dep[:100], analysis_dep[:100])
print(f"{np.abs(np.asarray(short) - true_noise).max():.3f}")

# a constant bias of 1 in the first channel
forecast_biased, analysis_biased = departures(true_noise, bias=np.array([1.0, 0.0]))
# today (filterax)
print(flx.utils.desroziers_R_estimate(forecast_biased, analysis_biased).round(3))
```

With 50,000 windows, the statistics print close to their targets:

| Quantity | Estimate | Target |
| --- | --- | --- |
| $\gmat{\Sigma}_y$ | $\begin{bmatrix} 1.506 & 0.499 \\ 0.499 & 1.776 \end{bmatrix}$ | $\begin{bmatrix} 1.5 & 0.5 \\ 0.5 & 1.8 \end{bmatrix}$ |
| $\gmat{\Sigma}_\varepsilon$ | $\begin{bmatrix} 0.502 & 0.002 \\ -0.002 & 0.789 \end{bmatrix}$ | $\begin{bmatrix} 0.5 & 0 \\ 0 & 0.8 \end{bmatrix}$ |
| $\gmat{\Sigma}_\varepsilon \gmat{\Sigma}_y^{-1} \gmat{\Sigma}_\varepsilon$ | $\begin{bmatrix} 0.184 & -0.081 \\ -0.081 & 0.387 \end{bmatrix}$ | $\begin{bmatrix} 0.184 & -0.082 \\ -0.082 & 0.392 \end{bmatrix}$ |
| $\mat{H} \gmat{\Sigma}_x \mat{H}^\top$, the first minus the second | $\begin{bmatrix} 1.004 & 0.497 \\ 0.5 & 0.988 \end{bmatrix}$ | $\begin{bmatrix} 1 & 0.5 \\ 0.5 & 1 \end{bmatrix}$ |

The estimate of $\gmat{\Sigma}_\varepsilon$ is not symmetric at finite $T$, by up to `3.5e-03` here, so symmetrise it before use.

For a filter that assumes half the true noise, the estimate prints as $\begin{bmatrix} 0.31 & -0.032 \\ -0.035 & 0.526 \end{bmatrix}$, against $\begin{bmatrix} 0.308 & -0.033 \\ -0.033 & 0.533 \end{bmatrix}$ from [](#eq-diag-desroziers-misspecified).
That lies between the assumed diagonal, 0.25 and 0.4, and the true one, 0.5 and 0.8.

The docstring asks for 100 or more windows, and the largest error at $T = 100$ prints as `0.119`.
Finally, the functions average raw products with no mean removed.
A constant bias of 1 in the first channel moves the first diagonal entry to `0.872`, against a true 0.5, and it adds `-0.165` off the diagonal.

## Degrees of freedom for signal

How much did the observations constrain the analysis?
The degrees of freedom for signal is the trace of the influence of the observations on the analysis in observation space {cite:p}`cardinali2004influence`:

$$\mathrm{DFS} = \op{tr}(\mat{G} \mat{H}) = \op{tr}\left(\gmat{\Sigma}_\varepsilon^{-1} \mat{H} \gmat{\Sigma}_{x \mid y} \mat{H}^\top\right).$$ (eq-diag-dfs)

The second form uses $\mat{G} = \gmat{\Sigma}_{x \mid y} \mat{H}^\top \gmat{\Sigma}_\varepsilon^{-1}$, and it lets a transform filter read the same number off its analysis ensemble.
`dfs_from_gain(gain, obs_matrix)` computes the first form.
It lies in $[0, D_y]$, near 0 when the prior dominates and near $D_y$ when the observations do.
With an ensemble, the rank of $\hat{\gmat{\Sigma}}_x$ caps it at $\min(D_y, N_\mathcal{E} - 1)$.

`dfs_from_ensemble` computes something different, despite its name.
With $\sigma_{i, t \mid t-1}^2$ the forecast variance of component $i$, it returns

$$\sum_{i = 1}^{D_x} \frac{[\tilde{\mat{X}}_{t \mid t}^\top \tilde{\mat{X}}_{t \mid t-1}]_{ii}}{(N_\mathcal{E} - 1) \, \sigma_{i, t \mid t-1}^2},$$ (eq-diag-dfs-ensemble)

the share of the forecast anomalies that survives in the analysis, summed over the state.
It is $D_x$ when the observations do nothing, and it falls as they gain influence, which is the opposite direction to [](#eq-diag-dfs).
For a scalar under the ETKF, the transform is $\sqrt{1 - g}$ with gain $g$, so a component contributes $\sqrt{1 - g}$ here and $g$ in the DFS.
Read it as a measure of retained spread, not as the DFS.

```{code-block} python
:caption: The two functions, and the analysis-ensemble form, from weak to strong observations, then the rank cap.
:label: code-diag-dfs
# Shapes: N_E = 50 members; D_x = 8; D_y = 4

# x⁽ᵉ⁾ ~ 𝒩(𝟎, I)
prior_members = jr.normal(jr.key(4), (50, 8))  # → (N_E, D_x)
# H selects the first four components
selection = jnp.eye(8)[:4]  # → (D_y, D_x)


# user
# DFS from the gain, from the analysis ensemble, and the retained-spread sum
@jax.jit
def three_measures(variance: Float[Array, ""]) -> tuple[Array, Array, Array]:
    # today (lineax)
    # Σ_ε = σ² I
    scaled_noise = lx.DiagonalLinearOperator(variance * jnp.ones(4))
    # today (filterax)
    # G
    gain = flx.kalman_gain(prior_members, prior_members[:, :4], scaled_noise)
    # today (filterax)
    # X_t|t
    updated = flx.filters.ETKF().analysis(
        prior_members, jnp.zeros(4), lambda state: state[:4], scaled_noise
    ).particles  # → (N_E, D_x)
    # today (filterax)
    # Σ̂_x|y on the observed components
    observed_cov = flx.ensemble_covariance(updated[:, :4]).as_matrix()  # → (D_y, D_y)
    # today (filterax)
    return (
        # tr(G H)
        flx.utils.dfs_from_gain(gain, selection),
        # tr(Σ_ε⁻¹ H Σ̂_x|y Hᵀ)
        jnp.trace(observed_cov) / variance,
        # the retained spread
        flx.utils.dfs_from_ensemble(prior_members, updated),
    )


for variance in (1e-6, 1.0, 1e6):
    by_gain, by_analysis, retained = three_measures(jnp.asarray(variance))
    print(variance, f"{by_gain:.4f} {by_analysis:.4f} {retained:.4f}")

# N_E = 4 members and D_y = 6 almost perfect observations
few_members = jr.normal(jr.key(5), (4, 8))  # → (N_E, D_x)
# today (lineax)
tiny_noise = lx.DiagonalLinearOperator(1e-6 * jnp.ones(6))  # (D_y, D_y) operator
# today (filterax)
few_gain = flx.kalman_gain(few_members, few_members[:, :6], tiny_noise)
# today (filterax)
print(f"{flx.utils.dfs_from_gain(few_gain, jnp.eye(8)[:6]):.4f}")
```

It prints:

| $\sigma^2$ in $\gmat{\Sigma}_\varepsilon = \sigma^2 \mat{I}$ | `dfs_from_gain` | Analysis-ensemble form | `dfs_from_ensemble` |
| --- | --- | --- | --- |
| $10^{-6}$ | 4.0000 | 4.0000 | 3.7342 |
| 1 | 1.8123 | 1.8123 | 6.7496 |
| $10^{6}$ | 0.0000 | 0.0000 | 8.0000 |

The two forms of [](#eq-diag-dfs) agree, and they rise from 0 to $D_y$ as the observations sharpen.
`dfs_from_ensemble` falls from $D_x$ instead, and its complement $D_x - 6.7496 = 1.25$ does not equal the DFS of 1.81 either.
The last line prints `3.0000` for $D_y = 6$ almost perfect observations and $N_\mathcal{E} = 4$, which is the cap $N_\mathcal{E} - 1$.

## Weight degeneracy

Importance-weighted ensembles have two degeneracy monitors {cite:p}`liu1998sequential`.
For normalised weights $w_e$, with $\sum_e w_e = 1$,

$$N_\text{eff} = \frac{1}{\sum_e w_e^2}, \qquad \mathbb{H}[\vect{w}] = -\sum_e w_e \log w_e.$$ (eq-diag-weights)

These are `effective_ensemble_size` and `weight_entropy`.
Equal weights give $N_\text{eff} = N_\mathcal{E}$ and $\mathbb{H} = \log N_\mathcal{E}$, and one dominant member gives 1 and 0.
Since both are Rényi entropies of different orders, $N_\text{eff} \le \exp \mathbb{H}[\vect{w}] \le N_\mathcal{E}$.

Neither function normalises its input, and filterax has no particle filter of its own.
The mean weights $\bar{\vect{w}}$ of [chapter 6](06_etkf.md) are transform coefficients that can be negative, so they are not importance weights either.

```{code-block} python
:caption: The two monitors on equal, mixed and dominated weights, and on weights that do not sum to one.
:label: code-diag-weights
# Shapes: N_E = 10 members

for name, weights in (
    ("equal", jnp.ones(10) / 10.0),
    ("mixed", jax.nn.softmax(jr.normal(jr.key(0), (10,)))),
    ("dominated", jnp.zeros(10).at[0].set(1.0)),
    ("not normalised", jnp.ones(10)),
):
    # today (filterax)
    effective = flx.utils.effective_ensemble_size(weights)  # → ()
    # today (filterax)
    entropy = flx.utils.weight_entropy(weights)  # → ()
    print(name, f"{effective:.3f} {entropy:.3f} {jnp.exp(entropy):.3f}")
```

It prints:

| Weights | $N_\text{eff}$ | $\mathbb{H}[\vect{w}]$ | $\exp \mathbb{H}[\vect{w}]$ |
| --- | --- | --- | --- |
| equal | 10.000 | 2.303 | 10.000 |
| mixed, a softmax of Gaussian draws | 4.926 | 1.917 | 6.797 |
| dominated by one member | 1.000 | `-0.000` | 1.000 |
| `jnp.ones(10)`, not normalised | 0.100 | `-0.000` | 1.000 |

The last row is the trap, since unnormalised weights give an impossible $N_\text{eff}$ below 1, and an entropy that looks like total degeneracy.

## Implementation in filterax

The first two rows live at the top level of filterax, and the others in [`filterax.utils`](https://jejjohnson.github.io/filterax/reference/diagnostics/).
Every `utils` function takes pre-computed arrays, so none of them runs an analysis.

| Function | Takes | Healthy value | Needs |
| --- | --- | --- | --- |
| `log_likelihood`, `innovation_covariance` | $\bar{\vect{r}}$ and an operator; mapped members and $\gmat{\Sigma}_\varepsilon$ | compare between runs | one window |
| `innovation_statistics` | members, $\vect{y}$, $h$, $\gmat{\Sigma}_\varepsilon$ | as its parts | one window |
| `innovation`, `normalized_innovation` | $\vect{y}$, $\bar{\vect{h}}$, and the diagonal of $\hat{\gmat{\Sigma}}_y$ | mean 0, variance 1 | one window, read over many |
| `chi2_consistency`, `chi2_normalized` | $\bar{\vect{r}}$, $\hat{\gmat{\Sigma}}_y^{-1} \bar{\vect{r}}$ | $D_y$, and 1 with the finite-ensemble correction | one window, read over many |
| `ensemble_spread`, `rms_spread` | one ensemble | stable, not collapsing | one window |
| `rmse_vs_truth`, `spread_skill_ratio` | ensemble and truth | [](#eq-diag-ssr-target) | a twin experiment |
| `rank_histogram`, `rank_histogram_chi2` | ensembles and truths over time | flat, near $N_\mathcal{E}$ when independent | a twin experiment and many windows |
| `crps_ensemble`, `crps_ensemble_batch` | ensemble, observation | lower is better | one window, read over many |
| `desroziers_innovation_cov`, `desroziers_R_estimate`, `desroziers_analysis_residual_cov` | departures $(T, D_y)$ | [](#eq-diag-desroziers) | a long record |
| `dfs_from_gain`, `dfs_from_ensemble` | $\mat{G}, \mat{H}$; forecast and analysis ensembles | system dependent | one window |
| `effective_ensemble_size`, `weight_entropy` | normalised weights | near $N_\mathcal{E}$, near $\log N_\mathcal{E}$ | one window |

`filters.*.analysis` returns `log_likelihood` on its result, and the Layer 2 `assimilate` stacks those as `log_likelihoods`, along with `forecast_history` and `analysis_history`.
So the departures for the Desroziers functions come straight from those histories, as the next section does.
`analysis_history` holds the ensemble after inflation.
That leaves the mean and hence the departures unchanged, and it does change `dfs_from_ensemble` and any spread read from it.

## Algorithm: a health check for a run

[](#code-diag-pipeline) runs the checks above together on the Lorenz-96 twin of [chapter 6](06_etkf.md): 20 members without inflation, 40 members without inflation, and 20 members with RTPS.
Each row of the table below is the second half of the run.

```{code-block} python
:caption: The diagnostics together on three Lorenz-96 runs, with the cycle jitted through one scan per run.
:label: code-diag-pipeline
# Shapes: T = 200 windows; D_x = 40 sites; D_y = 20 observed sites; N_E members


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
selection = jnp.eye(40)[sites]  # → (D_y, D_x)
# yₜ = H xₜ + εₜ, εₜ ~ 𝒩(𝟎, I)
observed = truth[:, sites] + jr.normal(jr.key(1), (200, 20))  # → (T, D_y)
windows = [(observed[t], 0.05 * (t + 1)) for t in range(200)]
# today (lineax)
unit_noise = lx.DiagonalLinearOperator(jnp.ones(20))  # (D_y, D_y) operator


# user
# one window's χ² / D_y, normalised-innovation variance, CRPS, DFS and N_eff
def per_window(
    predicted: Float[Array, "member state"],
    target: Float[Array, " obs"],
    true_state: Float[Array, " state"],
) -> dict[str, Array]:
    # h⁽ᵉ⁾
    predicted_mapped = predicted[:, sites]  # → (N_E, D_y)
    # h̄
    predicted_mean = einx.mean("[member] obs", predicted_mapped)  # → (D_y,)
    # today (filterax)
    # r̄
    gap = flx.utils.innovation(target, predicted_mean)  # → (D_y,)
    # today (filterax)
    # Σ̂_y
    covariance = flx.innovation_covariance(predicted_mapped, unit_noise).as_matrix()
    # today (filterax)
    # rᵢ / √[Σ̂_y]ᵢᵢ
    standardised = flx.utils.normalized_innovation(
        target, predicted_mean, jnp.diagonal(covariance)
    )  # → (D_y,)
    # today (filterax)
    # G
    gain = flx.kalman_gain(predicted, predicted_mapped, unit_noise)  # → (D_x, D_y)
    # log wₑ = −½ ‖y − h⁽ᵉ⁾‖²
    log_weights = -0.5 * einx.sum("member [obs]", (target - predicted_mapped) ** 2)
    weights = jax.nn.softmax(log_weights)  # → (N_E,)
    # today (filterax)
    return {
        "chi2": flx.utils.chi2_normalized(gap, jnp.linalg.solve(covariance, gap), 20),
        "variance": jnp.mean(standardised**2),
        "crps": flx.utils.crps_ensemble_batch(predicted, true_state),
        "dfs": flx.utils.dfs_from_gain(gain, selection),
        "effective": flx.utils.effective_ensemble_size(weights),
    }


# user
# forecast → analysis → inflate, jitted, the same steps as `assimilate`
def run_twin(num_members: int, inflator) -> tuple[Array, Array, Array]:
    # x⁽ᵉ⁾ ~ 𝒩(x₀, 4 I)
    start = spun_up + 2.0 * jr.normal(jr.key(2), (num_members, 40))  # → (N_E, D_x)
    # today (filterax)
    etkf = flx.filters.ETKF()

    # user
    def window(ensemble, inputs):
        target, step = inputs
        # x⁽ᵉ⁾_t|t−1
        predicted = jax.vmap(rk4)(ensemble)  # → (N_E, D_x)
        # today (filterax)
        # X_t|t and log 𝒩(r̄; 𝟎, Σ̂_y)
        step_result = etkf.analysis(
            predicted, target, lambda state: state[sites], unit_noise
        )
        updated = step_result.particles  # → (N_E, D_x)
        if inflator is not None:
            # today (filterax)
            updated = inflator(updated, predicted, step=step)  # → (N_E, D_x)
        return updated, (predicted, updated, step_result.log_likelihood)

    _, histories = jax.lax.scan(window, start, (observed, jnp.arange(200)))
    return histories


# today (filterax)
runs = {
    "N_E = 20": run_twin(20, None),
    "N_E = 40": run_twin(40, None),
    "N_E = 20, RTPS": run_twin(20, flx.RTPS(0.3)),
}
# today (filterax)
# the Layer 2 model over the first 5 windows, against the scan
short_result = flx.ETKF(dynamics=rk4, obs_op=lambda state: state[sites]).assimilate(
    spun_up + 2.0 * jr.normal(jr.key(2), (20, 40)), windows[:5], unit_noise
)
_, scan_analysis, scan_likelihood = runs["N_E = 20"]
print(f"{jnp.abs(short_result.analysis_history - scan_analysis[:5]).max():.0e}")
print(f"{jnp.abs(short_result.log_likelihoods - scan_likelihood[:5]).max():.0e}")

jitted_per_window = jax.jit(jax.vmap(per_window))
for name, (forecast_all, analysis_all, likelihood_all) in runs.items():
    # the second half of the run, once the filter has settled
    forecasts = forecast_all[100:]  # → (T / 2, N_E, D_x)
    analyses = analysis_all[100:]  # → (T / 2, N_E, D_x)
    settled_truth = truth[100:]  # → (T / 2, D_x)
    settled_obs = observed[100:]  # → (T / 2, D_y)
    # x̄_t|t−1, x̄_t|t
    forecast_mean = einx.mean("t [member] state", forecasts)  # → (T / 2, D_x)
    analysis_mean = einx.mean("t [member] state", analyses)  # → (T / 2, D_x)
    # today (filterax)
    errors = jax.vmap(flx.utils.rmse_vs_truth)(analysis_mean, settled_truth)
    # today (filterax)
    spreads = jax.vmap(flx.utils.rms_spread)(analyses)
    # r̄_t|t−1, r̄_t|t
    forecast_dep = settled_obs - forecast_mean[:, sites]  # → (T / 2, D_y)
    analysis_dep = settled_obs - analysis_mean[:, sites]  # → (T / 2, D_y)
    # today (filterax)
    estimate = flx.utils.desroziers_R_estimate(forecast_dep, analysis_dep)
    # today (filterax)
    outer = flx.utils.desroziers_innovation_cov(forecast_dep)
    # today (filterax)
    residual_cov = flx.utils.desroziers_analysis_residual_cov(analysis_dep)
    # today (filterax)
    counts = flx.utils.rank_histogram(forecasts, settled_truth)  # → (N_E + 1,)
    summary = jitted_per_window(forecasts, settled_obs, settled_truth)
    # today (filterax)
    retained = jax.vmap(flx.utils.dfs_from_ensemble)(forecasts, analyses)
    print(name)
    print(f"  RMSE {jnp.sqrt(jnp.mean(errors**2)):.2f}")
    print(f"  spread {jnp.sqrt(jnp.mean(spreads**2)):.2f}")
    print(f"  SSR {jnp.sqrt(jnp.mean(spreads**2) / jnp.mean(errors**2)):.2f}")
    print(f"  chi2 {summary['chi2'].mean():.2f}")
    print(f"  variance {summary['variance'].mean():.2f}")
    print(f"  log-likelihood {jnp.mean(likelihood_all[100:]):.1f}")
    print(f"  rank chi2 {flx.utils.rank_histogram_chi2(counts):.0f}")
    print(f"  crps {summary['crps'].mean():.2f}")
    print(f"  dfs {summary['dfs'].mean():.2f}")
    print(f"  retained {retained.mean():.2f}")
    print(f"  noise {jnp.mean(jnp.diagonal(estimate)):.2f}")
    print(f"  innovation {jnp.mean(jnp.diagonal(outer)):.2f}")
    print(f"  residual {jnp.mean(jnp.diagonal(residual_cov)):.2f}")
    print(f"  effective {jnp.median(summary['effective']):.2f}")
```

The first two lines print `1e-14` and `4e-14`: the scan reproduces `assimilate` on the first five windows, both its analysis and its log-likelihood.
The rest of the block prints:

| Diagnostic | $N_\mathcal{E} = 20$ | $N_\mathcal{E} = 40$ | $N_\mathcal{E} = 20$, `RTPS(0.3)` |
| --- | --- | --- | --- |
| RMSE of the analysis mean | 2.37 | 0.31 | 0.33 |
| RMS spread | 0.26 | 0.25 | 0.34 |
| $\mathrm{SSR}$, pooled | 0.11 | 0.81 | 1.03 |
| $\chi^2 / D_y$, mean | 6.63 | 1.00 | 0.97 |
| Variance of the normalised innovation | 6.56 | 1.00 | 0.97 |
| Log-likelihood, mean | -85.4 | -29.1 | -29.2 |
| $\chi^2_\text{rank}$ of the forecast, with null 20, 40 and 20 | 22398 | 805 | 60 |
| CRPS of the forecast against the truth, mean | 1.63 | 0.18 | 0.19 |
| Diagonal of $\hat{\gmat{\Sigma}}_\varepsilon$, mean, with truth 1 | 6.63 | 1.00 | 0.97 |
| Diagonal of the innovation covariance, mean | 7.12 | 1.08 | 1.10 |
| Diagonal of the analysis-residual covariance, mean | 6.24 | 0.94 | 0.87 |
| DFS, mean | 1.22 | 1.23 | 2.00 |
| `dfs_from_ensemble`, mean | 37.15 | 37.47 | 37.30 |
| $N_\text{eff}$ of likelihood weights on the forecast, median | 2.94 | 17.15 | 5.29 |

Almost every row separates the failed run from the two healthy ones, and three of them need no truth: $\chi^2 / D_y$, the log-likelihood and the Desroziers estimate.
The spread alone does not: at 0.26 it looks fine, and only its ratio to the error or to the innovations reveals the failure.
The healthy runs also differ.
Forty members without inflation are underdispersive at $\mathrm{SSR} = 0.81$, while `RTPS` brings 20 members to 1.03 and a much flatter rank histogram.
The DFS and `dfs_from_ensemble` hardly notice the failure, because they describe the update and not its correctness.

## Numerics

The diagnostics are simple arithmetic with few numerical traps, and the likelihood is where the care goes.
[](#code-diag-numerics) measures three cases in float32.
With $\gmat{\Sigma}_\varepsilon$ small against $\hat{\gmat{\Sigma}}_h$, the Woodbury route loses accuracy, and so the QR route is the one the filters use.
The block also checks a likelihood gradient, and the floors that keep ratios finite.

```{code-block} python
:caption: The likelihood routes in float32 against a float64 reference, a gradient of the likelihood, and the behaviour at zero spread.
:label: code-diag-numerics
# Shapes: N_E = 30 members; D_y = 3

# h⁽ᵉ⁾
draws = np.random.default_rng(1).normal(size=(30, 3))
# r̄
gap = np.array([0.1, -0.2, 0.3])
mapped_single = jnp.asarray(draws, jnp.float32)  # → (N_E, D_y)
gap_single = jnp.asarray(gap, jnp.float32)  # → (D_y,)


# user
# the Woodbury route, and the QR route through innovation_statistics, for Σ_ε = σ² I
@jax.jit
def both_routes(variance: Float[Array, ""]) -> tuple[Array, Array]:
    # today (lineax)
    single_noise = lx.DiagonalLinearOperator(variance * jnp.ones(3, jnp.float32))
    # today (filterax)
    woodbury = flx.log_likelihood(
        gap_single, flx.innovation_covariance(mapped_single, single_noise)
    )  # → ()
    # today (filterax)
    by_qr = flx.innovation_statistics(
        mapped_single,
        gap_single + einx.mean("[member] obs", mapped_single),
        lambda state: state,
        single_noise,
    )["log_likelihood"]  # → ()
    return woodbury, by_qr


for variance in (1.0, 1e-4, 1e-6):
    # log 𝒩(r̄; 𝟎, Σ̂_y), dense, in float64 numpy
    covariance = np.cov(draws, rowvar=False) + variance * np.eye(3)
    _, logdet = np.linalg.slogdet(covariance)
    quadratic = gap @ np.linalg.solve(covariance, gap)
    reference = -0.5 * (3.0 * np.log(2.0 * np.pi) + logdet + quadratic)
    woodbury, by_qr = both_routes(jnp.float32(variance))
    print(
        variance,
        f"{abs(float(woodbury) - reference):.1e}",
        f"{abs(float(by_qr) - reference):.1e}",
    )

# X
base_members = jr.normal(jr.key(0), (30, 3))  # → (N_E, D_y)


# user
# log 𝒩(r̄; 𝟎, H̃ᵀ H̃ / c + σ² I), as a function of σ²
def in_variance(variance: Float[Array, ""]) -> Float[Array, ""]:
    # today (lineax)
    scaled = lx.DiagonalLinearOperator(variance * jnp.ones(3))  # (D_y, D_y) operator
    # today (filterax)
    return flx.log_likelihood(
        jnp.array([0.1, -0.2, 0.3]), flx.innovation_covariance(base_members, scaled)
    )  # → ()


at = jnp.asarray(0.1)
step = 1e-6
central = (in_variance(at + step) - in_variance(at - step)) / (2.0 * step)
print(f"{jax.grad(in_variance)(at):.6f} {central:.6f}")

# x⁽ᵉ⁾ with component 1 collapsed
stuck = jnp.array([[-1.0, 0.0], [1.0, 0.0]])  # → (N_E, D_x)
# today (filterax)
print(flx.utils.ensemble_spread(stuck).round(3))
# today (filterax)
print(jnp.isnan(jax.grad(flx.utils.rms_spread)(stuck)).any())
# today (filterax)
# the error floor when the mean equals the truth
print(f"{flx.utils.spread_skill_ratio(stuck, jnp.zeros(2)):.0e}")
```

The errors in the float32 log-likelihood, against the float64 reference, print as:

| $\sigma^2$ | Woodbury route | QR route |
| --- | --- | --- |
| 1 | `6.0e-07` | `1.3e-07` |
| $10^{-4}$ | `2.2e-04` | `1.2e-07` |
| $10^{-6}$ | `9.8e-03` | `3.2e-07` |

The QR route holds its accuracy across six decades, and that is why the filters report it.
The next line prints `-1.152792 -1.152792`, so the gradient of the likelihood in $\sigma^2$ matches a central difference.
A collapsed component reads as zero in `ensemble_spread`, printed as `[1.414 0.   ]`.
But `jnp.std` has an infinite slope there, so the gradient of `rms_spread` holds `NaN`, printed as `True`, and a spread-based training loss needs a floor.
The floors inside the diagnostics keep the ratios finite.
When the mean equals the truth, `spread_skill_ratio` divides by $10^{-30}$ and prints `1e+30`, which is a large number and not a meaningful one.

## When it breaks

The common ways to misread these numbers:

- One window is not enough: $\chi^2 / D_y$ has a null standard deviation of $\sqrt{2 / D_y}$, so use a running mean.
- The target is not exactly 1: $\chi^2 / D_y$, the SSR and the CRPS all have finite-ensemble corrections, measured above.
- `rank_histogram_chi2` assumes independent counts, so on a real run compare it between configurations, as in [](#code-diag-pipeline).
- Ranking the truth against an observation without noise in the members produces a false U shape, as [](#code-diag-rank) measures.
- The Desroziers functions are not centred, so remove the mean of each departure first if bias is possible, as [](#code-diag-desroziers) shows.
- `desroziers_R_estimate` obeys [](#eq-diag-desroziers-misspecified) on a misspecified run, so one pass only moves towards the true noise.
- `effective_ensemble_size` returns a wrong number for unnormalised weights, with no warning, as [](#code-diag-weights) prints.
- `dfs_from_ensemble` runs the other way from the DFS, as [](#code-diag-dfs) measures, and it also sees any inflation in `analysis_history`.

## When each one is the right answer

Use `rms_spread` and `chi2_normalized` in every run, since they need no truth and they catch divergence early.
Use `log_likelihood` or `result.log_likelihood` to compare configurations on the same observations, and as a training loss.
Use `rmse_vs_truth`, `spread_skill_ratio`, `rank_histogram` and `crps_ensemble_batch` in a twin experiment, where you can verify against the truth.
Use the Desroziers functions on an operational record, to check the observation-error covariance you assumed.
Use `dfs_from_gain` to ask how much an observing system contributes, and the form in [](#eq-diag-dfs) when the filter never forms a gain.
Use `effective_ensemble_size` and `weight_entropy` only with real importance weights.

## See also

- [Chapter 3](03_observation_model.md): where $h$ and $\gmat{\Sigma}_\varepsilon$ come from
- [Chapter 6](06_etkf.md): the Lorenz-96 twin used above, and the divergence it shows
- [Chapter 11](11_localization.md): the treatment when spread is fine and the error comes from sampling noise
- [Chapter 12](12_inflation.md): what to do when $\chi^2 / D_y$ stays above 1, and the adaptive scheme that reads it
- [Chapter 14](14_differentiable.md): the log-likelihood as a training loss
- [API: diagnostics](https://jejjohnson.github.io/filterax/reference/diagnostics/)

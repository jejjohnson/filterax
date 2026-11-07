---
numbering:
  enumerator: "12.%s"
---

# Inflation

A finite ensemble Kalman filter underestimates its own uncertainty, and each analysis makes that worse.
Once the spread is too small the gain shrinks, the observations stop mattering, and the error grows while the ensemble reports that all is well.
Inflation puts spread back deliberately.

Three questions organise this chapter.
Why does the spread come out too small, even with a perfect model?
What does each inflator do to the covariance, and in particular to the correlations and the rank?
And what can go wrong, given that an adaptive belief can freeze and an inflator can create spread that nothing removes?
Ledoit-Wolf shrinkage appears at the end because it treats the same disease from a different side.
It regularises a covariance estimate, and it does not touch the ensemble.

## Why the spread comes out too small

Take one observed scalar, with forecast variance $\sigma^2$ and observation variance $\sigma_\varepsilon^2$.
The optimal analysis variance is $v(\sigma^2)$, where

$$v(s) = \frac{s \, \sigma_\varepsilon^2}{s + \sigma_\varepsilon^2}.$$ (eq-infl-analysis-variance)

An ensemble only has the sample variance $\hat{s}$, with $\mathbb{E}[\hat{s}] = \sigma^2$, and sampling error acts in two directions.
The variance the ensemble reports is $v(\hat{s})$, and $v$ is concave in $s$, so by Jensen's inequality the report is too low on average:

$$\mathbb{E}[v(\hat{s})] \le v(\sigma^2).$$ (eq-infl-reported)

The error the analysis actually has comes from the gain $\hat{g} = \hat{s} / (\hat{s} + \sigma_\varepsilon^2)$.
The optimal gain minimises the following error, so a noisy gain can only add to it:

$$\mathbb{E}\left[(1 - \hat{g})^2 \sigma^2 + \hat{g}^2 \sigma_\varepsilon^2\right] \ge v(\sigma^2).$$ (eq-infl-actual)

So [](#eq-infl-reported) and [](#eq-infl-actual) pull apart even with a linear model and Gaussian noise.
The ensemble reports less than the optimum, and it achieves more error than the optimum.

```{code-block} python
:caption: The two sides of the gap for $N_\mathcal{E} = 10$ and $\sigma^2 = \sigma_\varepsilon^2 = 1$, using the sampling distribution of the variance.
:label: code-infl-underestimate
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

# Shapes: 200,000 draws of the sample variance; N_E = 10

# ŝ ~ σ² χ²(N_E − 1) / (N_E − 1), with σ² = 1
sampled = jr.chisquare(jr.key(0), 9, shape=(200_000,)) / 9.0  # → (200000,)
# v(ŝ) = ŝ σ_ε² / (ŝ + σ_ε²), with σ_ε² = 1
reported = sampled / (sampled + 1.0)  # → (200000,)
# ĝ = ŝ / (ŝ + σ_ε²)
noisy_gain = sampled / (sampled + 1.0)  # → (200000,)
# (1 − ĝ)² σ² + ĝ² σ_ε²
achieved = (1.0 - noisy_gain) ** 2 + noisy_gain**2  # → (200000,)
print(f"{reported.mean():.3f} {achieved.mean():.3f}")
```

It prints `0.474 0.529`, against an optimum of $v(1) = 0.5$.
The ratio of the two is the spread deficit that an analysis starts from, before any model error or nonlinearity adds to it.
Model error adds a second cause, since members that run through an imperfect model do not carry that model's error in their spread.

Localization is sometimes listed as a third cause, but on a single analysis it acts the other way.
In [](#code-infl-localized-spread) the localized analysis keeps more spread than the unlocalized one, because the localized gain contracts each site less.

```{code-block} python
:caption: One perturbed-observation analysis from the same forecast and draws, with and without a taper.
:label: code-infl-localized-spread
# Shapes: N_E = 20 members; D_x = 40 sites; D_y = 20 observed sites

# the sites, as a coordinate column
grid = einx.id("site -> site 1", jnp.arange(40.0))  # → (D_x, 1)
# the observed sites
sites = jnp.arange(0, 40, 2)  # → (D_y,)
obs_coords = grid[sites]  # → (D_y, 1)
# Σ_ε = I
noise = lx.DiagonalLinearOperator(jnp.ones(20))  # (D_y, D_y) operator


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

# x⁽ᵉ⁾ ~ 𝒩(x₀, 0.25 I)
forecast = spun_up + 0.5 * jr.normal(jr.key(2), (20, 40))  # → (N_E, D_x)
# h⁽ᵉ⁾ = H x⁽ᵉ⁾
mapped = forecast[:, sites]  # → (N_E, D_y)
# today (filterax)
# y⁽ᵉ⁾ = y + ε⁽ᵉ⁾
perturbed = flx.perturbed_observations(jr.key(3), observed[0], noise, 20)
# today (filterax)
# ρ_xy
rho_xy = flx.localization_matrix(grid, obs_coords, 4.0)  # → (D_x, D_y)
# today (filterax)
# ρ_yy
rho_yy = flx.localization_matrix(obs_coords, obs_coords, 4.0)  # → (D_y, D_y)
# today (filterax)
for gain in (
    flx.kalman_gain(forecast, mapped, noise),
    flx.localized_kalman_gain(forecast, mapped, noise, rho_xy, rho_yy),
):
    # x⁽ᵉ⁾_t|t = x⁽ᵉ⁾ + G (y⁽ᵉ⁾ − h⁽ᵉ⁾)
    analysis = forecast + einx.dot(  # → (N_E, D_x)
        "state obs, member obs -> member state", gain, perturbed - mapped
    )
    # today (filterax)
    # √(mean diag Σ̂_x|y)
    print(f"{flx.utils.rms_spread(analysis):.3f}")
# today (filterax)
print(f"{flx.utils.rms_spread(forecast):.3f}")
```

It prints `0.423`, `0.459` and `0.492`: the unlocalized analysis, the localized one, and the forecast they both start from.

## Multiplicative inflation

The simplest fix scales the anomalies about the mean {cite:p}`anderson1999monte`.
With factor $\zeta$,

$$\vect{x}^{(e)} \leftarrow \bar{\vect{x}} + \zeta \left(\vect{x}^{(e)} - \bar{\vect{x}}\right), \qquad \hat{\gmat{\Sigma}}_x \leftarrow \zeta^2 \hat{\gmat{\Sigma}}_x.$$ (eq-infl-multiplicative)

It keeps the mean, every correlation and the span of the anomalies.
The docstring gives $\zeta \in [1.01, 1.10]$ as typical for operational numerical weather prediction.
Applied uniformly, it boosts densely observed regions and data-void regions alike.
A direction with no spread stays with no spread, since $\zeta \cdot 0 = 0$.

## Relaxation: RTPS and RTPP

Relaxation methods blend the analysis with the forecast, so the correction is largest where the analysis contracted most.
Write $\sigma_{i, t \mid t}$ and $\sigma_{i, t \mid t-1}$ for the analysis and forecast spread of component $i$, and take $\alpha \in [0, 1]$.

### RTPS

Relaxation to prior spread {cite:p}`whitaker2012evaluating` relaxes the spread of each component and rescales its anomalies to match:

$$\sigma_i^\text{target} = (1 - \alpha) \sigma_{i, t \mid t} + \alpha \sigma_{i, t \mid t-1}, \qquad \tilde{x}^{(e)}_i \leftarrow \kappa_i \tilde{x}^{(e)}_{i}, \qquad \kappa_i = (1 - \alpha) + \alpha \frac{\sigma_{i, t \mid t-1}}{\sigma_{i, t \mid t} + \epsilon},$$ (eq-infl-rtps)

where $\tilde{x}^{(e)}_i$ is the analysis anomaly and $\epsilon = 10^{-12}$ is an absolute floor set by gaussx.
In matrix form the covariance becomes $\mat{D} \hat{\gmat{\Sigma}}_{x \mid y} \mat{D}$ with $\mat{D} = \op{diag}(\gvect{\kappa})$, and a two-sided diagonal scaling leaves every correlation unchanged.
So RTPS adjusts the marginal spreads and keeps the correlation structure the analysis produced.

### RTPP

Relaxation to prior perturbations {cite:p}`zhang2004impacts` blends the anomaly matrices themselves:

$$\tilde{\mat{X}}_{t \mid t} \leftarrow (1 - \alpha) \tilde{\mat{X}}_{t \mid t} + \alpha \tilde{\mat{X}}_{t \mid t-1}.$$ (eq-infl-rtpp)

Its covariance contains the cross term between the two anomaly matrices,

$$(1 - \alpha)^2 \hat{\gmat{\Sigma}}_{x \mid y} + \alpha^2 \hat{\gmat{\Sigma}}_x + \alpha (1 - \alpha) \frac{\tilde{\mat{X}}_{t \mid t}^\top \tilde{\mat{X}}_{t \mid t-1} + \tilde{\mat{X}}_{t \mid t-1}^\top \tilde{\mat{X}}_{t \mid t}}{N_\mathcal{E} - 1},$$ (eq-infl-rtpp-covariance)

so it mixes the correlations of forecast and analysis, and partly undoes the correlation information the observations provided.
Members keep their identity through [](#eq-infl-rtpp), and $\alpha = 1$ gives back the forecast anomalies exactly.
At $\alpha = 1$ RTPS gives back the forecast spread but not the forecast members.

The two coincide when the analysis is an exact rescaling $\tilde{\mat{X}}_{t \mid t} = a \tilde{\mat{X}}_{t \mid t-1}$, since both then give $((1 - \alpha) a + \alpha) \tilde{\mat{X}}_{t \mid t-1}$.
Any real analysis reshapes the anomalies, and then they differ.

## Additive inflation

When a direction carries no spread at all, nothing that multiplies anomalies can help.
Additive inflation {cite:p}`hamill2005accounting` adds noise with its own covariance $\gmat{\Sigma}_\eta \in \mathbb{R}^{D_x \times D_x}$, the covariance of the unrepresented model error:

$$\vect{x}^{(e)} \leftarrow \vect{x}^{(e)} + \gvect{\eta}^{(e)} - \bar{\gvect{\eta}}, \qquad \gvect{\eta}^{(e)} \sim \mathcal{N}(\vect{0}, \gmat{\Sigma}_\eta).$$ (eq-infl-additive)

`inflate_additive` subtracts the sample mean of the draws, so the ensemble mean is exactly preserved.
The Bessel divisor makes that centring unbiased: $\mathbb{E}[(\gvect{\eta}^{(e)} - \bar{\gvect{\eta}})(\gvect{\eta}^{(e)} - \bar{\gvect{\eta}})^\top] = (1 - 1 / N_\mathcal{E}) \gmat{\Sigma}_\eta$, and summing over members and dividing by $N_\mathcal{E} - 1$ leaves $\gmat{\Sigma}_\eta$.
So $\hat{\gmat{\Sigma}}_x$ grows by $\gmat{\Sigma}_\eta$ in expectation.
It samples through `perturbed_observations`, so a diagonal $\gmat{\Sigma}_\eta$ is never formed as a matrix.

## Agreement with closed forms

[](#code-infl-closed-forms) applies all four to a real ETKF analysis and compares each to a plain numpy form, then checks the properties above.

```{code-block} python
:caption: Each inflator against a numpy reference on one ETKF analysis, in float64.
:label: code-infl-closed-forms
# Shapes: N_E = 20 members; D_x = 8; D_y = 3

# x⁽ᵉ⁾ ~ 𝒩(𝟎, I)
prior = jr.normal(jr.key(0), (20, 8))  # → (N_E, D_x)
# today (lineax)
# Σ_ε = 0.25 I
weak_noise = lx.DiagonalLinearOperator(0.25 * jnp.ones(3))  # (D_y, D_y) operator
# today (filterax)
# X_t|t, observing components 0, 2 and 4
posterior = flx.filters.ETKF().analysis(
    prior,
    jnp.array([0.5, -0.5, 1.0]),
    lambda state: state[jnp.array([0, 2, 4])],
    weak_noise,
).particles  # → (N_E, D_x)

# today (filterax)
results = {
    "multiplicative": flx.inflate_multiplicative(posterior, 1.3),
    "rtps": flx.inflate_rtps(posterior, prior, 0.5),
    "rtpp": flx.inflate_rtpp(posterior, prior, 0.5),
    "additive": flx.inflate_additive(
        jr.key(1), posterior, lx.DiagonalLinearOperator(0.1 * jnp.ones(8))
    ),
}


# user
# the numpy forms of the multiplicative, RTPS and RTPP updates
def numpy_forms(after, before):
    after_mean = after.mean(0)
    after_anomalies = after - after_mean
    before_anomalies = before - before.mean(0)
    scale = 0.5 + 0.5 * before_anomalies.std(0) / after_anomalies.std(0)
    return {
        "multiplicative": after_mean + 1.3 * after_anomalies,
        "rtps": after_mean + scale * after_anomalies,
        "rtpp": after_mean + 0.5 * after_anomalies + 0.5 * before_anomalies,
    }


references = numpy_forms(np.asarray(posterior), np.asarray(prior))


# user
# the correlation matrix of an ensemble
def correlations(ensemble: Float[Array, "member state"]) -> Float[Array, "state state"]:
    # today (filterax)
    covariance = flx.ensemble_covariance(ensemble).as_matrix()  # → (D_x, D_x)
    # σᵢ = √[Σ̂_x]ᵢᵢ
    spread = jnp.sqrt(jnp.diagonal(covariance))  # → (D_x,)
    return covariance / einx.multiply("i, j -> i j", spread, spread)  # → (D_x, D_x)


# x̄_t|t
posterior_mean = einx.mean("[member] state", posterior)  # → (D_x,)
base_correlation = correlations(posterior)  # → (D_x, D_x)
# today (filterax)
print(f"analysis {flx.utils.rms_spread(posterior):.3f}")
print(f"forecast {flx.utils.rms_spread(prior):.3f}")
for name, inflated in results.items():
    mean_shift = jnp.abs(einx.mean("[member] state", inflated) - posterior_mean).max()
    # |Δ correlation|
    correlation_shift = jnp.abs(correlations(inflated) - base_correlation).max()
    reference_error = (
        jnp.abs(inflated - references[name]).max() if name in references else jnp.nan
    )
    print(
        name,
        # today (filterax)
        f"{flx.utils.rms_spread(inflated):.3f}",
        f"{mean_shift:.0e} {correlation_shift:.3f} {reference_error:.0e}",
    )

# X̃_t|t = a X̃_t|t−1, with a = 0.4
scaled_analysis = 0.4 * prior  # → (N_E, D_x)
# today (filterax)
gap = flx.inflate_rtps(scaled_analysis, prior, 0.5) - flx.inflate_rtpp(
    scaled_analysis, prior, 0.5
)  # → (N_E, D_x)
print(f"{jnp.abs(gap).max():.0e}")
# today (filterax)
# X̃ from RTPP with α = 1
back = flx.ensemble_anomalies(flx.inflate_rtpp(posterior, prior, 1.0))  # → (N_E, D_x)
# today (filterax)
print(f"{jnp.abs(back - flx.ensemble_anomalies(prior)).max():.0e}")
```

It prints the analysis spread `0.832` and the forecast spread `0.984`, and then the columns of this table:

| Inflator | RMS spread | Largest change in the mean | Largest change in a correlation | Error against numpy |
| --- | --- | --- | --- | --- |
| multiplicative, $\zeta = 1.3$ | 1.081 | `2e-16` | 0.000 | `4e-16` |
| RTPS, $\alpha = 0.5$ | 0.899 | `2e-16` | 0.000 | `3e-12` |
| RTPP, $\alpha = 0.5$ | 0.897 | `2e-16` | 0.163 | `4e-16` |
| additive, $\gmat{\Sigma}_\eta = 0.1 \mat{I}$ | 0.894 | `1e-16` | 0.375 | not applicable |

Every inflator keeps the mean to round-off.
Multiplicative inflation and RTPS keep every correlation, as the algebra says, and RTPP and additive noise do not.
The RTPS error of `3e-12` is the floor $\epsilon$, which the numpy form leaves out.
The block then prints `3e-12`, the gap between RTPS and RTPP on an exact rescaling, which is the same floor.
Its last line prints `4e-16`, which confirms that RTPP at $\alpha = 1$ gives back the forecast anomalies.

[](#code-infl-collapse) shows the difference that matters most in practice, which is what each inflator does for a component whose spread has gone to zero.
It also measures what additive noise contributes on average.

```{code-block} python
:caption: Spread restored on a collapsed component, and the mean covariance added by the centred draws.
:label: code-infl-collapse
# Shapes: N_E = 20 members; D_x = 2; 2,000 draws of the additive noise

# x⁽ᵉ⁾ with spread 1 in component 0 and none in component 1
collapsed = jnp.column_stack(  # → (N_E, D_x)
    [jr.normal(jr.key(0), (20,)), jnp.zeros(20)]
)
# x⁽ᵉ⁾_t|t−1 with spread 2 in both components
wider = 2.0 * jr.normal(jr.key(1), (20, 2))  # → (N_E, D_x)
# today (lineax)
# Σ_η = 0.1 I
small_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))  # (D_x, D_x) operator
# today (filterax)
for name, inflated in {
    "multiplicative": flx.inflate_multiplicative(collapsed, 1.3),
    "rtps": flx.inflate_rtps(collapsed, wider, 0.5),
    "rtpp": flx.inflate_rtpp(collapsed, wider, 0.5),
    "additive": flx.inflate_additive(jr.key(2), collapsed, small_noise),
}.items():
    # today (filterax)
    print(name, flx.utils.ensemble_spread(inflated).round(3))


# user
# the sample variance of the centred draws, added to an ensemble with no spread
def added_variance(key: jax.Array) -> Float[Array, " state"]:
    # today (filterax)
    draws = flx.inflate_additive(key, jnp.zeros((20, 2)), small_noise)  # → (N_E, D_x)
    # today (filterax)
    return flx.utils.ensemble_spread(draws) ** 2  # → (D_x,)


# 𝔼 Σ̂ of the draws
mean_added = jax.vmap(added_variance)(jr.split(jr.key(3), 2000))  # → (2000, D_x)
print(einx.mean("[draw] state", mean_added).round(4))

# today (filterax)
inflator = flx.AdditiveInflator(small_noise, jr.key(4))
base = jnp.zeros((5, 2))  # → (N_E, D_x)
# the same draw without a step, and different draws with one
print(
    bool(jnp.all(inflator(base) == inflator(base))),
    bool(jnp.all(inflator(base, step=0) == inflator(base, step=1))),
)
```

The spreads print as follows, with component 0 starting near 1 and component 1 at zero:

| Inflator | Component 0 | Collapsed component 1 |
| --- | --- | --- |
| multiplicative, $\zeta = 1.3$ | 1.268 | 0 |
| RTPS, $\alpha = 0.5$ | 1.578 | 0 |
| RTPP, $\alpha = 0.5$ | 1.176 | 1.085 |
| additive, $\gmat{\Sigma}_\eta = 0.1 \mat{I}$ | 1.02 | 0.295 |

Multiplicative inflation and RTPS cannot revive a dead component, since both multiply zero.
RTPP brings it back from the forecast, and additive noise brings it back from nothing.
The mean variance added by the centred draws prints as `[0.1004 0.0992]` against $\gmat{\Sigma}_\eta = 0.1 \mat{I}$, so the centring is unbiased.
The last line prints `True False`: with no `step` the same draw repeats, and with a `step` it differs.

## Adaptive inflation

Hand-tuning $\zeta$ is awkward when the innovations themselves say whether the filter is over- or underdispersive.
`inflate_adaptive` holds a Gaussian belief $\zeta \sim \mathcal{N}(\mu_\zeta, \sigma_\zeta^2)$ and updates it each window.
It is a simplified form of the scheme of {cite:t}`anderson2009spatially`, with one global scalar and a fixed likelihood variance.

With the innovation $\bar{\vect{r}} \in \mathbb{R}^{D_y}$ and its covariance $\hat{\gmat{\Sigma}}_y \in \mathbb{R}^{D_y \times D_y}$ ([chapter 13](13_diagnostics.md)), the data-implied value is the clipped consistency statistic,

$$\hat{\zeta}_\text{obs} = \op{clip}\left(\frac{\bar{\vect{r}}^\top \hat{\gmat{\Sigma}}_y^{-1} \bar{\vect{r}}}{D_y}, \zeta_\text{min}, \zeta_\text{max}\right),$$ (eq-infl-adaptive-obs)

with defaults $\zeta_\text{min} = 1$ and $\zeta_\text{max} = 1.2$.
Under correct specification $\mathbb{E}[\chi^2 / D_y] = 1$, with variance $\sigma_\text{obs}^2 = 2 / D_y$, which the code uses as the likelihood variance.
The posterior is the Gaussian product,

$$\mu_\zeta \leftarrow \frac{\sigma_\zeta^2 \hat{\zeta}_\text{obs} + \sigma_\text{obs}^2 \mu_\zeta}{\sigma_\zeta^2 + \sigma_\text{obs}^2}, \qquad \sigma_\zeta^2 \leftarrow \frac{\sigma_\zeta^2 \sigma_\text{obs}^2}{\sigma_\zeta^2 + \sigma_\text{obs}^2}.$$ (eq-infl-adaptive-posterior)

Two consequences follow by hand.
The precision only ever grows, by $D_y / 2$ per window:

$$\frac{1}{\sigma_{\zeta, T}^2} = \frac{1}{\sigma_{\zeta, 0}^2} + \frac{T D_y}{2}.$$ (eq-infl-adaptive-precision)

So $\mu_\zeta$ becomes a running average of $\hat{\zeta}_\text{obs}$ over the whole history, and there is no forgetting.

The statistic also measures a different thing from the factor it is used as.
If the true forecast variance is $\zeta^2 \sigma^2$ and the ensemble reports $\sigma^2$, then for one component $\mathbb{E}[\chi^2 / D_y] = (\zeta^2 \sigma^2 + \sigma_\varepsilon^2) / (\sigma^2 + \sigma_\varepsilon^2)$.
That approaches $\zeta^2$ for precise observations and $1$ for poor ones, and neither limit is $\zeta$.

```{code-block} python
:caption: One update against the Gaussian product, then a healthy run, an underdispersive run and a change of regime.
:label: code-infl-adaptive
# Shapes: T = 2,000 windows; D_y = 10 observations

# r̄
one_innovation = jnp.array([1.5, -1.2, 0.9])  # → (3,)
# Σ̂_y = 0.5 I
one_covariance = 0.5 * jnp.eye(3)  # → (3, 3)
# today (filterax)
# (μ_ζ, σ_ζ²) after one window, with ζ_max raised to 1.5
mean_after, var_after = flx.inflate_adaptive(
    1.0, 0.05, one_innovation, one_covariance, max_factor=1.5
)
# ζ̂_obs = clip(r̄ᵀ Σ̂_y⁻¹ r̄ / D_y, 1, 1.5)
by_hand = np.clip(np.sum(np.asarray(one_innovation) ** 2) / 0.5 / 3.0, 1.0, 1.5)
# σ_obs² = 2 / D_y
obs_variance = 2.0 / 3.0
hand_mean = (0.05 * by_hand + obs_variance * 1.0) / (0.05 + obs_variance)
hand_var = 0.05 * obs_variance / (0.05 + obs_variance)
print(f"{mean_after:.6f} {hand_mean:.6f} {var_after:.6f} {hand_var:.6f}")


# user
# (μ_ζ, σ_ζ²) over a run, with r̄ₜ ~ 𝒩(𝟎, truth_variances[t] I) and Σ̂_y = I
def adaptive_run(
    truth_variances: Float[Array, " time"],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    # r̄ₜ
    draws = jr.normal(jr.key(5), (truth_variances.shape[0], 10))  # → (T, D_y)
    innovations = einx.multiply(  # → (T, D_y)
        "time, time obs -> time obs", jnp.sqrt(truth_variances), draws
    )

    # user
    def window(belief, innovation):
        # today (filterax)
        return flx.inflate_adaptive(*belief, innovation, jnp.eye(10)), None

    start = (jnp.asarray(1.0), jnp.asarray(0.05))
    (mean, var), _ = jax.lax.scan(window, start, innovations)
    return mean, var


healthy = jnp.ones(2000)  # → (T,)
underdispersive = 4.0 * jnp.ones(2000)  # → (T,)
regime_change = jnp.concatenate([jnp.ones(1000), 4.0 * jnp.ones(1000)])  # → (T,)
for name, variances in (
    ("healthy", healthy),
    ("underdispersive", underdispersive),
    ("regime change", regime_change),
):
    mean, var = adaptive_run(variances)
    print(name, f"{mean:.3f} {var:.1e}")
# 1 / σ_ζ² = 1 / σ_ζ,₀² + T D_y / 2
print(f"{1.0 / (1.0 / 0.05 + 2000 * 10 / 2):.1e}")
# 𝔼[χ² / D_y] for a true factor ζ = 1.5, with σ² = 1 and three values of σ_ε²
print([round((1.5**2 + r) / (1.0 + r), 3) for r in (0.01, 1.0, 10.0)])
```

The first line prints `1.034884 1.034884 0.046512 0.046512`, so one update matches the Gaussian product by hand.
The three runs of 2,000 windows, starting from $\mu_\zeta = 1$ and $\sigma_\zeta^2 = 0.05$, end as follows:

| Run | True innovation variance | $\mu_\zeta$ | $\sigma_\zeta^2$ |
| --- | --- | --- | --- |
| healthy | 1 throughout | 1.066 | `1.0e-04` |
| underdispersive | 4 throughout | 1.197 | `1.0e-04` |
| regime change | 1, then 4 | 1.129 | `1.0e-04` |

The closed form [](#eq-infl-adaptive-precision) prints `1.0e-04`, matching all three.
A healthy filter ends above 1 because the clip keeps every excursion above 1 and removes those below it, so the estimate is biased upwards.
The underdispersive run sits at the clip, however large the true deficit.
The regime change ends between the two, since by then a new window has a weight of only about $5 \times 10^{-4}$ in [](#eq-infl-adaptive-posterior).
The last line prints `[2.238, 1.625, 1.114]`, the expected statistic for a true factor $\zeta = 1.5$ at $\sigma_\varepsilon^2 = 0.01, 1$ and $10$, none of which is 1.5.

## Ledoit-Wolf shrinkage

Rather than inflate the ensemble, one can regularise the covariance estimate {cite:p}`ledoit2004well`.
Shrinkage replaces $\hat{\gmat{\Sigma}}_x$ by a convex combination with a scaled identity,

$$\hat{\gmat{\Sigma}}_\lambda = (1 - \lambda) \hat{\gmat{\Sigma}}_x + \lambda \tau \mat{I}, \qquad \tau = \frac{\op{tr}(\hat{\gmat{\Sigma}}_x)}{D_x},$$ (eq-infl-shrinkage)

with $\hat{\gmat{\Sigma}}_\lambda \in \mathbb{R}^{D_x \times D_x}$, $\tau$ the average sample eigenvalue, and the intensity chosen to minimise the expected squared Frobenius distance to the truth:

$$\lambda = \min\left(1, \frac{b^2}{d^2}\right), \qquad d^2 = \Vert \hat{\gmat{\Sigma}}_x - \tau \mat{I} \Vert_F^2, \qquad b^2 = \frac{1}{N_\mathcal{E}^2} \sum_e \Vert \tilde{\vect{x}}^{(e)} \tilde{\vect{x}}^{(e) \top} - \hat{\gmat{\Sigma}}_x \Vert_F^2.$$ (eq-infl-shrinkage-intensity)

Here $d^2$ measures how far the sample covariance is from the target, and $b^2$ how far it is from itself under sampling.
The sample covariance uses the Bessel divisor, whereas {cite:t}`ledoit2004well` use $1 / N_\mathcal{E}$, which is a small difference.

Three properties follow directly.
The trace is unchanged, $\op{tr}(\hat{\gmat{\Sigma}}_\lambda) = \op{tr}(\hat{\gmat{\Sigma}}_x)$, so shrinkage moves variance between directions and adds none, which is why it is not inflation.
The smallest eigenvalue is at least $\lambda \tau$, so the result is positive definite whenever $\lambda \tau > 0$, however few members there are.
And $\lambda$ is scale-invariant: multiplying the ensemble by $a$ multiplies both $b^2$ and $d^2$ by $a^4$.

### Computing it without squaring the units

Everything comes from the $N_\mathcal{E} \times N_\mathcal{E}$ Gram matrix $\mat{K}_{\tilde{X} \tilde{X}} = \tilde{\mat{X}} \tilde{\mat{X}}^\top$, with no $D_x \times D_x$ array, at $O(N_\mathcal{E}^2 D_x)$.
With $c = 1 / (N_\mathcal{E} - 1)$,

$$\op{tr}(\hat{\gmat{\Sigma}}_x) = c \op{tr}(\mat{K}_{\tilde{X} \tilde{X}}), \qquad \Vert \hat{\gmat{\Sigma}}_x \Vert_F^2 = c^2 \Vert \mat{K}_{\tilde{X} \tilde{X}} \Vert_F^2, \qquad b^2 = \frac{1}{N_\mathcal{E}^2} \left(\sum_e [\mat{K}_{\tilde{X} \tilde{X}}]_{ee}^2 + (N_\mathcal{E} c^2 - 2 c) \Vert \mat{K}_{\tilde{X} \tilde{X}} \Vert_F^2\right).$$ (eq-infl-shrinkage-gram)

The formula for $b^2$ comes from expanding the norm and using $\op{tr}(\tilde{\mat{X}} \hat{\gmat{\Sigma}}_x \tilde{\mat{X}}^\top) = c \Vert \mat{K}_{\tilde{X} \tilde{X}} \Vert_F^2$.
But both $b^2$ and $d^2$ hold the fourth power of the anomalies, which leaves the float32 range for quite ordinary units.
So gaussx first divides the factor by its largest entry, computes $\lambda$ from the unit-scale factor, and restores the scale only in $\tau$.
That makes the computed $\lambda$ scale-invariant too, as the mathematics says it should be.
[](#code-infl-shrinkage) measures this, along with the properties above.

```{code-block} python
:caption: Shrinkage against a numpy reference, its three properties, its benefit over the sample covariance, and scale invariance in float32.
:label: code-infl-shrinkage
# Shapes: N_E = 10 members; D_x = 30


# user
# λ by its definition, in plain numpy
def dense_shrinkage(ensemble):
    count, size = ensemble.shape
    anomalies = ensemble - ensemble.mean(0)
    sample = anomalies.T @ anomalies / (count - 1)
    target = np.trace(sample) / size
    distance = np.sum((sample - target * np.eye(size)) ** 2)
    deviation = sum(
        np.sum((np.outer(row, row) - sample) ** 2) for row in anomalies
    ) / count**2
    return min(1.0, deviation / distance), sample, target


# [Σ_x]ᵢⱼ = 0.9^|i − j|, the true covariance
true_covariance = 0.9 ** jnp.abs(  # → (D_x, D_x)
    einx.subtract("i, j -> i j", jnp.arange(30.0), jnp.arange(30.0))
)
# x⁽ᵉ⁾ ~ 𝒩(𝟎, Σ_x)
correlated = einx.dot(  # → (N_E, D_x)
    "member state, other state -> member other",
    jr.normal(jr.key(0), (10, 30)),
    jnp.linalg.cholesky(true_covariance),
)
# today (filterax)
# Σ̂_λ and λ
shrunk, intensity = flx.ledoit_wolf_shrinkage(correlated)  # (D_x, D_x), ()
reference_intensity, sample, target = dense_shrinkage(np.asarray(correlated))
print(f"{intensity:.6f} {reference_intensity:.6f}")
# tr(Σ̂_λ), tr(Σ̂_x)
print(f"{jnp.trace(shrunk):.6f} {np.trace(sample):.6f}")
# smallest eigenvalue against λ τ
print(f"{jnp.linalg.eigvalsh(shrunk).min():.6f} {intensity * target:.6f}")
# ‖Σ̂_λ − Σ_x‖_F, ‖Σ̂_x − Σ_x‖_F
print(
    f"{jnp.linalg.norm(shrunk - true_covariance):.2f}",
    f"{jnp.linalg.norm(jnp.asarray(sample) - true_covariance):.2f}",
)
# today (gaussx)
# the same Σ̂_λ as a structured operator
operator, _ = gaussx.ledoit_wolf_shrinkage(correlated)
print(type(operator).__name__, f"{jnp.abs(operator.as_matrix() - shrunk).max():.0e}")


# user
# λ by its definition, in float32 with no rescaling
@jax.jit
def naive_shrinkage(ensemble: Float[Array, "member state"]) -> Float[Array, ""]:
    count, size = ensemble.shape
    anomalies = ensemble - einx.mean("[member] state", ensemble)
    sample = einx.dot("member i, member j -> i j", anomalies, anomalies) / (count - 1.0)
    target = jnp.trace(sample) / size
    distance = jnp.sum((sample - target * jnp.eye(size, dtype=sample.dtype)) ** 2)
    outer = einx.multiply("member i, member j -> member i j", anomalies, anomalies)
    deviation = jnp.sum((outer - sample) ** 2) / count**2
    return jnp.minimum(1.0, deviation / distance)


single = jnp.asarray(correlated, dtype=jnp.float32)  # → (N_E, D_x)
for scale in (1e-12, 1.0, 1e12):
    scaled = jnp.float32(scale) * single  # → (N_E, D_x)
    # today (filterax)
    _, stable = flx.ledoit_wolf_shrinkage(scaled)
    print(scale, stable.dtype, f"{stable:.5f} {naive_shrinkage(scaled):.5f}")
```

The first lines print `0.140867 0.140867`, `40.589968 40.589968` and `0.190593 0.190593`.
So $\lambda$ matches its definition, the trace is preserved, and the smallest eigenvalue equals $\lambda \tau$ exactly, because with $N_\mathcal{E} < D_x$ the sample covariance is singular.
The next line prints `14.92 18.80`, so shrinkage lowers the Frobenius error against the true covariance on this draw.
The line after prints `LowRankUpdate 0e+00`, since gaussx returns the same matrix as an operator that solves through Woodbury, and `filterax.ledoit_wolf_shrinkage` calls `as_matrix` on it.

In float32 the block prints:

| Ensemble scale | `ledoit_wolf_shrinkage` | Definition without rescaling |
| --- | --- | --- |
| $10^{-12}$ | 0.14087 | `nan` |
| 1 | 0.14087 | 0.14087 |
| $10^{12}$ | 0.14087 | `nan` |

Without rescaling, the fourth powers underflow at the small scale and overflow at the large one, and $\lambda$ comes out `nan`.
The small scale is not exotic: a trace gas at parts per trillion has mole fractions of that order.

`inflate_*` act on the ensemble, and the next forecast sees the result.
`ledoit_wolf_shrinkage` returns a matrix, and no filter consumes it for you.
So it suits a parameter-space ensemble ([chapter 9](09_ensemble_kalman_processes.md)) or any computation that needs a well-conditioned covariance.

## Implementation in filterax

| Layer | Offers | Notes |
| --- | --- | --- |
| Primitive | `inflate_multiplicative`, `inflate_rtps`, `inflate_rtpp` | call gaussx, and are differentiable in the coefficient |
| Primitive | `inflate_additive(key, particles, noise_cov)` | draws through `perturbed_observations`, then centres |
| Primitive | `inflate_adaptive` | returns the belief $(\mu_\zeta, \sigma_\zeta^2)$, and inflates nothing itself |
| Primitive | `ledoit_wolf_shrinkage` | dense $(\hat{\gmat{\Sigma}}_\lambda, \lambda)$; `gaussx.ledoit_wolf_shrinkage` returns the operator |
| Module | `MultiplicativeInflator`, `RTPS`, `RTPP`, `AdditiveInflator` | `AbstractInflator` wrappers for the Layer 2 `inflator=` slot |

The Layer 2 loop of [chapter 16](16_assimilation_cycle.md) runs forecast, analysis and then inflate.
The inflator receives the analysis and the forecast, so this is posterior inflation, and `analysis_history` holds the ensemble after inflation.
`RTPS` and `RTPP` raise `ValueError` without a forecast ensemble.

`AdditiveInflator` receives `step=` from the loop and folds it into its `base_key`, so windows draw independently.
Called by hand with neither `step` nor `key`, it repeats one draw, as [](#code-infl-collapse) prints.
It is the one stochastic inflator, and `differentiable_assimilate` rejects it ([chapter 14](14_differentiable.md)).

The coefficient of each module is a static field, so `eqx.filter_grad` cannot train it.
To learn a factor, call the pure function with an array, as [](#code-infl-numerics) does.
`inflate_adaptive` has no module and the loop has no slot for it, so a run that uses it writes its own cycle.

## Algorithm

[](#code-infl-cycle) writes the forecast, analysis and inflate cycle by hand, in the setting of [chapter 6](06_etkf.md), where the ETKF with 20 members fails without inflation.

```{code-block} python
:caption: ETKF on Lorenz-96 with each inflator, using the same observations and the same initial ensemble.
:label: code-infl-cycle
# Shapes: T = 300 windows; D_x = 40 sites; D_y = 20 observed sites; N_E = 20 members


# user
# forecast → analysis → inflate: RMSE and RMS spread over the last 200 windows
def run_cycle(inflator, num_members: int = 20) -> tuple[float, float]:
    # x⁽ᵉ⁾ ~ 𝒩(x₀, 4 I)
    start = spun_up + 2.0 * jr.normal(jr.key(2), (num_members, 40))  # → (N_E, D_x)
    # today (filterax)
    etkf = flx.filters.ETKF()

    # user
    def window(ensemble, inputs):
        target, step = inputs
        # x⁽ᵉ⁾_t|t−1 = f(x⁽ᵉ⁾)
        predicted = jax.vmap(rk4)(ensemble)  # → (N_E, D_x)
        # today (filterax)
        # X_t|t
        updated = etkf.analysis(
            predicted, target, lambda state: state[sites], noise
        ).particles  # → (N_E, D_x)
        if inflator is not None:
            # today (filterax)
            updated = inflator(updated, predicted, step=step)  # → (N_E, D_x)
        return updated, updated

    _, history = jax.lax.scan(  # → (T, N_E, D_x)
        window, start, (observed, jnp.arange(300))
    )
    # x̄_t|t
    settled_mean = einx.mean("t [member] state", history[100:])  # → (200, D_x)
    # today (filterax)
    errors = jax.vmap(flx.utils.rmse_vs_truth)(settled_mean, truth[100:])
    # today (filterax)
    spreads = jax.vmap(flx.utils.rms_spread)(history[100:])
    rmse = jnp.sqrt(jnp.mean(errors**2))  # → ()
    spread = jnp.sqrt(jnp.mean(spreads**2))  # → ()
    return float(rmse), float(spread)


# today (filterax)
inflators = {
    "none": None,
    "multiplicative 1.02": flx.MultiplicativeInflator(1.02),
    "multiplicative 1.05": flx.MultiplicativeInflator(1.05),
    "multiplicative 1.10": flx.MultiplicativeInflator(1.10),
    "rtps 0.3": flx.RTPS(0.3),
    "rtps 0.5": flx.RTPS(0.5),
    "rtps 0.8": flx.RTPS(0.8),
    "rtpp 0.3": flx.RTPP(0.3),
    "rtpp 0.5": flx.RTPP(0.5),
    "rtpp 0.8": flx.RTPP(0.8),
    "additive 0.05": flx.AdditiveInflator(
        lx.DiagonalLinearOperator(0.05 * jnp.ones(40)), jr.key(9)
    ),
}
for name, inflator in inflators.items():
    rmse, spread = run_cycle(inflator)
    print(f"{name}: {rmse:.2f} {spread:.2f}")
# today (filterax)
rmse, spread = run_cycle(flx.RTPP(0.5), num_members=25)
print(f"rtpp 0.5, N_E = 25: {rmse:.2f} {spread:.2f}")
```

| Inflator | RMSE | Spread |
| --- | --- | --- |
| None | 3.74 | 0.26 |
| `MultiplicativeInflator(1.02)` | 0.35 | 0.31 |
| `MultiplicativeInflator(1.05)` | 0.37 | 0.41 |
| `MultiplicativeInflator(1.10)` | 0.45 | 0.54 |
| `RTPS(0.3)` | 0.35 | 0.36 |
| `RTPS(0.5)` | 0.41 | 0.50 |
| `RTPS(0.8)` | nan | nan |
| `RTPP(0.3)` | 1.24 | 0.32 |
| `RTPP(0.5)` | 2.81 | 0.46 |
| `RTPP(0.8)` | 0.98 | 1.01 |
| `AdditiveInflator`, $\gmat{\Sigma}_\eta = 0.05 \mat{I}$ | 0.82 | 0.52 |

Without inflation the filter fails.
A factor of only $\zeta = 1.02$ brings the error down by a factor of about ten, with the spread matching it, and larger factors overshoot.
RTPS at $\alpha = 0.3$ does as well, and at $\alpha = 0.8$ the run returns `nan`.
RTPP at $\alpha = 0.3$ and $0.5$ does far worse than RTPS at the same $\alpha$.
This ensemble size is marginal for it: the extra line, RTPP at $\alpha = 0.5$ with $N_\mathcal{E} = 25$, prints an RMSE of 0.35 and a spread of 0.42.
Additive noise with variance 0.05 keeps the filter on track at a higher error.

## Numerics

Mean preservation and the closed forms are exact to round-off, as [](#code-infl-closed-forms) prints, and ordinary spreads present no float32 difficulty.
Two details do matter, both in RTPS.

The floor $\epsilon$ in [](#eq-infl-rtps) is absolute, so RTPS is not scale-invariant at spreads near $10^{-12}$, unlike the shrinkage above.
And `jnp.std` has an infinite slope at zero spread, so differentiating through RTPS with respect to the members gives `NaN` on a collapsed component.

```{code-block} python
:caption: The RTPS floor against a very small spread, a gradient through a collapsed component, and a learned factor.
:label: code-infl-numerics
# Shapes: N_E = 2 members; D_x = 1 or 2

# today (filterax)
# RTPS with α = 1, at spreads of about 1 and about 10⁻¹²
for tiny in (1.0, 1e-12):
    restored = flx.inflate_rtps(  # → (N_E, D_x)
        tiny * jnp.array([[-1.0], [1.0]]), tiny * jnp.array([[-2.0], [2.0]]), 1.0
    )
    print(f"{restored[1, 0] / tiny:.3f}")

# x⁽ᵉ⁾ with component 1 collapsed
stuck = jnp.array([[-1.0, 0.0], [1.0, 0.0]])  # → (N_E, D_x)
wide = jnp.array([[-2.0, -2.0], [2.0, 2.0]])  # → (N_E, D_x)


# user
# 𝓛(X) = ∑ RTPS(X, X_t|t−1; 0.5)²
def through_rtps(ensemble: Float[Array, "member state"]) -> Float[Array, ""]:
    # today (filterax)
    return jnp.sum(flx.inflate_rtps(ensemble, wide, 0.5) ** 2)  # → ()


print(jnp.isnan(jax.grad(through_rtps)(stuck)).any())


# user
# 𝓛(ζ) = ∑ (ζ X̃)²
def through_factor(factor: Float[Array, ""]) -> Float[Array, ""]:
    # today (filterax)
    return jnp.sum(flx.inflate_multiplicative(wide, factor) ** 2)  # → ()


print(f"{jax.grad(through_factor)(jnp.asarray(1.5)):.3f}")
# today (filterax)
# the array leaves of a module, which leave out its static coefficient
print(len(jax.tree_util.tree_leaves(flx.MultiplicativeInflator(1.5))))
```

It prints `2.000` and `1.000`: the RTPS output spread in units of the small scale, where the forecast spread is 2.
At unit scale the output is right.
At a spread of $10^{-12}$ the floor equals the analysis spread, so $\kappa = 1$ and nothing is restored.
The next line prints `True`, since the gradient holds `NaN`, so a differentiable run with RTPS needs no component collapsed exactly to zero.
The line after prints `48.000`, which is $2 \zeta \sum_e \Vert \tilde{\vect{x}}^{(e)} \Vert^2$ at $\zeta = 1.5$, so the pure function is differentiable in its factor.
The last prints `0`, since the module carries its coefficient as static data with no array leaf for `eqx.filter_grad` to find.

## When it breaks

```{code-block} python
:caption: Half the state observed, with neutral dynamics, under multiplicative inflation and RTPS.
:label: code-infl-unobserved
# Shapes: N_E = 20 members; D_x = 4; D_y = 2; T = 100 windows

# x⁽ᵉ⁾ ~ 𝒩(𝟎, I)
neutral_start = jr.normal(jr.key(0), (20, 4))  # → (N_E, D_x)
# today (lineax)
# Σ_ε = I
neutral_noise = lx.DiagonalLinearOperator(jnp.ones(2))  # (D_y, D_y) operator


# user
# the spread after T windows, with f(x) = x and component 0 and 1 observed
def neutral_spread(inflator) -> Float[Array, " state"]:
    # today (filterax)
    etkf = flx.filters.ETKF()

    # user
    def window(ensemble, _):
        # today (filterax)
        updated = etkf.analysis(
            ensemble, jnp.zeros(2), lambda state: state[:2], neutral_noise
        ).particles  # → (N_E, D_x)
        # today (filterax)
        return inflator(updated, ensemble), None

    final, _ = jax.lax.scan(window, neutral_start, None, length=100)
    # today (filterax)
    return flx.utils.ensemble_spread(final)  # → (D_x,)


# today (filterax)
print(neutral_spread(flx.MultiplicativeInflator(1.05)).round(3))
# today (filterax)
print(neutral_spread(flx.RTPS(0.5)).round(3))
```

[](#code-infl-unobserved) shows the classic failure of uniform inflation, with no dynamics to bound the spread.
It prints `[  0.32    0.32  145.123 146.375]` for multiplicative inflation and `[0.14  0.143 1.129 1.121]` for RTPS.
Under $\zeta = 1.05$ the two unobserved components grow by that factor in every window, and reach about 145 after 100 windows.
Under RTPS they stay near their starting spread, since the analysis hardly contracted them and there is nothing to restore.
In a chaotic system the attractor bounds this growth, but spread in a data-void region still grows until it does.

The other limits, each measured above:

- Too much relaxation is as bad as too little: RTPS at $\alpha = 0.8$ returns `nan` in [](#code-infl-cycle).
- RTPP at $\alpha = 0.5$ gives an RMSE of 2.81 at $N_\mathcal{E} = 20$, where $\zeta = 1.02$ gives 0.35.
- Multiplicative inflation and RTPS cannot revive a collapsed direction, as [](#code-infl-collapse) shows.
- RTPP needs a forecast with spread in that direction, and only additive noise creates spread from nothing.
- Additive noise adds error along with spread: its draws ignore the dynamics, and in [](#code-infl-cycle) it gives 0.82 where $\zeta = 1.02$ gives 0.35.
- An adaptive belief freezes, and with the default clip it cannot see an overdispersive filter at all, by [](#eq-infl-adaptive-obs) and [](#code-infl-adaptive).
- Shrinkage lowers the Frobenius error of one covariance, and it adds neither members nor spread.

## When inflation is the right answer

Almost always, in cycling, since every filter in this library loses spread for the reasons at the head of this chapter.
A small constant inflation is the first thing to try.
Use `MultiplicativeInflator` when coverage is roughly even and you want a single, cheap, differentiable knob.
Use `RTPS` when coverage is uneven, since it inflates only what the analysis contracted and it keeps the correlations.
Use `RTPP` when the analysis is suspected of damaging cross-variable relations and you can afford to give back part of the observations' correlation information.
Use `AdditiveInflator` when you have an estimate of model error and spread is missing in directions the ensemble does not span.
Use `inflate_adaptive` for a slowly varying system, where a long running average is the wanted behaviour.
Use `ledoit_wolf_shrinkage` when you need one well-conditioned covariance matrix, not a better ensemble.

## See also

- [Chapter 6](06_etkf.md): the divergence that these inflators treat
- [Chapter 11](11_localization.md): localization, the companion treatment for sampling noise
- [Chapter 13](13_diagnostics.md): the $\chi^2$ and spread-skill statistics that tell you whether to inflate
- [Chapter 14](14_differentiable.md): why `AdditiveInflator` is rejected under `jax.grad`
- [Chapter 16](16_assimilation_cycle.md): where inflation sits in the assimilation loop
- [API: inflation](https://jejjohnson.github.io/filterax/reference/inflation/) and [primitives](https://jejjohnson.github.io/filterax/reference/primitives/)

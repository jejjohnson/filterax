---
numbering:
  enumerator: "5.%s"
---

# The stochastic EnKF

The stochastic ensemble Kalman filter {cite:p}`evensen1994sequential,burgers1998analysis,houtekamer1998data` is the original and simplest member of the family.
It computes one Kalman gain from the forecast ensemble and updates every member against its own randomly perturbed copy of the observation.

Three questions organise this chapter.
Why do the perturbations have to be there at all?
What do they cost at a finite ensemble size?
And which of the two routes to the gain does the update take, since that decides what float32 and a singular noise covariance can do?

## The perturbed-observation update

Take the forecast ensemble $\mat{X} \in \mathbb{R}^{N_\mathcal{E} \times D_x}$ with members as rows, the mapped members $\vect{h}^{(e)} = h(\vect{x}^{(e)}) \in \mathbb{R}^{D_y}$, the observation $\vect{y} \in \mathbb{R}^{D_y}$ and the noise covariance $\gmat{\Sigma}_\varepsilon \in \mathbb{R}^{D_y \times D_y}$.
Their anomalies are $\tilde{\mat{X}}$ and $\tilde{\mat{H}}$, with $c = N_\mathcal{E} - 1$, so that $\hat{\gmat{\Sigma}}_{xy} = \tilde{\mat{X}}^\top \tilde{\mat{H}} / c$ and $\hat{\gmat{\Sigma}}_h = \tilde{\mat{H}}^\top \tilde{\mat{H}} / c$.
The gain $\hat{\mat{G}} \in \mathbb{R}^{D_x \times D_y}$ of [chapter 4](04_kalman_update.md) is computed once:

$$\hat{\mat{G}} = \hat{\gmat{\Sigma}}_{xy} \hat{\gmat{\Sigma}}_y^{-1}, \qquad \hat{\gmat{\Sigma}}_y = \hat{\gmat{\Sigma}}_h + \gmat{\Sigma}_\varepsilon.$$ (eq-enkf-gain)

Each member then has its own residual, built from its own draw $\gvect{\varepsilon}^{(e)} \in \mathbb{R}^{D_y}$:

$$\vect{x}_{t \mid t}^{(e)} = \vect{x}^{(e)} + \hat{\mat{G}} \left(\vect{y} + \gvect{\varepsilon}^{(e)} - \vect{h}^{(e)}\right), \qquad \gvect{\varepsilon}^{(e)} \sim \mathcal{N}(\vect{0}, \gmat{\Sigma}_\varepsilon).$$ (eq-enkf-update)

The observation operator enters only through $\vect{h}^{(e)}$, so a nonlinear $h$ needs no special handling.
The draws are independent across members and across windows.
Reusing one is a real error, which [](#code-enkf-repeated-key) measures.

{cite:t}`vanleeuwen2020consistent` argues that the draw belongs to the modelled observation $\vect{h}^{(e)}$ rather than to $\vect{y}$.
Since $-\gvect{\varepsilon}^{(e)}$ has the same distribution, the two readings give the same mean and covariance for a Gaussian prior and likelihood, and the name "perturbed observations" has stuck.

## Why the perturbations are needed

Stack the draws as rows of $\mat{E} \in \mathbb{R}^{N_\mathcal{E} \times D_y}$, with sample mean $\bar{\gvect{\varepsilon}} \in \mathbb{R}^{D_y}$ and anomalies $\tilde{\mat{E}}$.
Averaging [](#eq-enkf-update) over the members, and then subtracting that average, separates the mean from the anomalies:

$$\bar{\vect{x}}_{t \mid t} = \bar{\vect{x}} + \hat{\mat{G}} \left(\bar{\vect{r}} + \bar{\gvect{\varepsilon}}\right), \qquad \tilde{\mat{X}}_{t \mid t} = \tilde{\mat{X}} + \left(\tilde{\mat{E}} - \tilde{\mat{H}}\right) \hat{\mat{G}}^\top.$$ (eq-enkf-split)

Write $\hat{\gmat{\Sigma}}_{\varepsilon} = \tilde{\mat{E}}^\top \tilde{\mat{E}} / c$ for the sample covariance of the draws, and $\hat{\gmat{\Sigma}}_{x\varepsilon} = \tilde{\mat{X}}^\top \tilde{\mat{E}} / c$ and $\hat{\gmat{\Sigma}}_{h\varepsilon} = \tilde{\mat{H}}^\top \tilde{\mat{E}} / c$ for their sample cross-covariances.
The sample covariance of the anomalies in [](#eq-enkf-split) then expands, for every finite ensemble, into

$$\hat{\gmat{\Sigma}}_{t \mid t} = \hat{\gmat{\Sigma}}_x - \hat{\gmat{\Sigma}}_{xy} \hat{\mat{G}}^\top - \hat{\mat{G}} \hat{\gmat{\Sigma}}_{xy}^\top + \hat{\mat{G}} \hat{\gmat{\Sigma}}_h \hat{\mat{G}}^\top + \hat{\mat{G}} \hat{\gmat{\Sigma}}_{\varepsilon} \hat{\mat{G}}^\top + \mat{N} + \mat{N}^\top,$$ (eq-enkf-expansion)

where the term $\mat{N} = (\hat{\gmat{\Sigma}}_{x\varepsilon} - \hat{\mat{G}} \hat{\gmat{\Sigma}}_{h\varepsilon}) \hat{\mat{G}}^\top \in \mathbb{R}^{D_x \times D_x}$ collects the sample correlation between the draws and the forecast.

Take the expectation over the draws with the forecast held fixed.
Then $\mathbb{E}[\hat{\gmat{\Sigma}}_{\varepsilon}] = \gmat{\Sigma}_\varepsilon$ and $\mathbb{E}[\mat{N}] = \vect{0}$.
The gain satisfies $\hat{\mat{G}} \hat{\gmat{\Sigma}}_y = \hat{\gmat{\Sigma}}_{xy}$, so the last two surviving terms combine into $\hat{\gmat{\Sigma}}_{xy} \hat{\mat{G}}^\top$, and what remains is the Kalman covariance:

$$\mathbb{E}\left[\hat{\gmat{\Sigma}}_{t \mid t}\right] = \hat{\gmat{\Sigma}}_x - \hat{\mat{G}} \hat{\gmat{\Sigma}}_{xy}^\top = \hat{\gmat{\Sigma}}_{x \mid y}.$$ (eq-enkf-expected-covariance)

This holds with the sample covariances, so it needs no linear $h$ and no large ensemble.
For a linear $\mat{H}$ the same statement is the familiar identity $(\mat{I} - \hat{\mat{G}} \mat{H}) \hat{\gmat{\Sigma}}_x (\mat{I} - \hat{\mat{G}} \mat{H})^\top + \hat{\mat{G}} \gmat{\Sigma}_\varepsilon \hat{\mat{G}}^\top = (\mat{I} - \hat{\mat{G}} \mat{H}) \hat{\gmat{\Sigma}}_x$.

Now remove the draws.
Setting $\mat{E} = \mat{0}$ in [](#eq-enkf-expansion) leaves the same terms without $\gmat{\Sigma}_\varepsilon$, so

$$\hat{\gmat{\Sigma}}_{t \mid t}^\text{unperturbed} = \hat{\gmat{\Sigma}}_{x \mid y} - \hat{\mat{G}} \gmat{\Sigma}_\varepsilon \hat{\mat{G}}^\top.$$ (eq-enkf-unperturbed)

The unperturbed update is therefore too confident at every analysis, by exactly the term the observation noise should contribute, and the shortfall compounds over cycles.
That is why gaussx offers no `perturb=False` flag: the deterministic alternative is a different filter ([chapter 6](06_etkf.md)), and not an option on this one.

## The price: sampling noise

[](#eq-enkf-expected-covariance) holds only in expectation, and one analysis takes one draw.
The mean in [](#eq-enkf-split) is off the Kalman mean by exactly $\hat{\mat{G}} \bar{\gvect{\varepsilon}}$.
Since $\bar{\gvect{\varepsilon}} \sim \mathcal{N}(\vect{0}, \gmat{\Sigma}_\varepsilon / N_\mathcal{E})$, its size has a closed form:

$$\mathbb{E}\left[\left\Vert \hat{\mat{G}} \bar{\gvect{\varepsilon}} \right\Vert_2^2\right] = \frac{1}{N_\mathcal{E}} \op{tr}\left(\hat{\mat{G}} \gmat{\Sigma}_\varepsilon \hat{\mat{G}}^\top\right).$$ (eq-enkf-mean-noise)

The covariance is off through $\hat{\gmat{\Sigma}}_{\varepsilon} - \gmat{\Sigma}_\varepsilon$ and through $\mat{N}$ in [](#eq-enkf-expansion), and both shrink as $1 / \sqrt{N_\mathcal{E}}$.
This noise comes on top of the sampling error the forecast ensemble already carries.
The deterministic square-root filters of chapters [6](06_etkf.md) and [7](07_ensrf_estkf.md) have neither term.

## Algorithm

[](#code-enkf-algorithm) writes [](#eq-enkf-gain) and [](#eq-enkf-update) out in full, for a dense $\gmat{\Sigma}_\varepsilon$ and a given stack of perturbed observations.

```{code-block} python
:caption: The perturbed-observation update written out in full.
:label: code-enkf-algorithm
import einx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

jax.config.update("jax_enable_x64", True)


# user
def enkf_by_hand(
    members: Float[Array, "member state"],
    mapped: Float[Array, "member obs"],
    perturbed: Float[Array, "member obs"],
    noise_covariance: Float[Array, "obs obs"],
) -> Float[Array, "member state"]:
    # Shapes: N_E members; D_x state components; D_y observations
    # c = N_E − 1
    prior_term = members.shape[0] - 1.0
    # X̃ = X − 𝟏 x̄ᵀ
    state_anomalies = members - einx.mean("[member] state", members)  # → (N_E, D_x)
    # H̃
    mapped_anomalies = mapped - einx.mean("[member] obs", mapped)  # → (N_E, D_y)
    # Σ̂_xyᵀ = H̃ᵀ X̃ / c
    cross = einx.dot(  # → (D_y, D_x)
        "member obs, member state -> obs state", mapped_anomalies, state_anomalies
    )
    cross = cross / prior_term  # → (D_y, D_x)
    # Σ̂_h = H̃ᵀ H̃ / c
    mapped_covariance = einx.dot(  # → (D_y, D_y)
        "member obs, member other -> obs other", mapped_anomalies, mapped_anomalies
    )
    mapped_covariance = mapped_covariance / prior_term  # → (D_y, D_y)
    # Ĝᵀ = Σ̂_y⁻¹ Σ̂_xyᵀ
    gain_transpose = jnp.linalg.solve(  # → (D_y, D_x)
        mapped_covariance + noise_covariance, cross
    )
    # y + ε⁽ᵉ⁾ − h⁽ᵉ⁾
    residuals = perturbed - mapped  # → (N_E, D_y)
    # x⁽ᵉ⁾ + Ĝ (y + ε⁽ᵉ⁾ − h⁽ᵉ⁾)
    return members + einx.dot(  # → (N_E, D_x)
        "member obs, obs state -> member state", residuals, gain_transpose
    )
```

| Route to $\hat{\mat{G}}$ | Default when | Cost of the gain | Needs |
| --- | --- | --- | --- |
| Dense $\hat{\gmat{\Sigma}}_y$ | $N_\mathcal{E} \ge D_y$ | $O(N_\mathcal{E} D_y^2 + D_y^3)$ | a positive semi-definite $\gmat{\Sigma}_\varepsilon$ |
| Woodbury capacitance | $N_\mathcal{E} < D_y$ | $O(N_\mathcal{E}^3 + N_\mathcal{E}^2 D_y)$ | a positive definite $\gmat{\Sigma}_\varepsilon$ |

Applying the gain to the residuals adds $O(N_\mathcal{E} D_y D_x)$, and the observation operator is evaluated once per member.

## Implementation in filterax

There are three layers, and each is useful on its own.

- [`gaussx.enkf_analysis`](https://jejjohnson.github.io/gaussx/) is the update itself. It takes either a `key` or a prebuilt `perturbed_obs`, and it exposes `localization`, `obs_localization`, `dense_innovation` and `solver`.
- [`filterax.filters.StochasticEnKF`](https://jejjohnson.github.io/filterax/reference/filters/) is the Layer 1 analysis step. It maps `(particles, obs, obs_op, obs_noise)` to an `AnalysisResult`, and of those gaussx options it passes on only `key`.
- `filterax.StochasticEnKF` is the Layer 2 cycle of [chapter 16](16_assimilation_cycle.md), which derives the key of window $t$ as `jr.fold_in(base_key, t)`.

[`filterax.perturbed_observations`](https://jejjohnson.github.io/filterax/reference/primitives/) draws the stack $\vect{y} + \gvect{\varepsilon}^{(e)}$ through `gaussx.sample_mvn`, while `gaussx.enkf_analysis` draws its own with a factor $\mat{L} \mat{L}^\top = \gmat{\Sigma}_\varepsilon$. Both have the right distribution, but they produce the same draws for a given key only when $\gmat{\Sigma}_\varepsilon$ is diagonal: with Toeplitz or low-rank noise the two routes give different members.

The example has $N_\mathcal{E} = 30$ members of a state with $D_x = 4$, drawn around 1 with standard deviation 0.5, and observes components 0 and 2 with noise variance 0.1.

```{code-block} python
:caption: One analysis step. The key on the filter is only a default, so a cycle passes a fresh one.
:label: code-enkf-analysis
import filterax as flx
import gaussx
import jax.random as jr
import lineax as lx

# Shapes: N_E = 30 members; D_x = 4 state components; D_y = 2 observations

# x⁽ᵉ⁾ ~ 𝒩(𝟏, 0.25 I)
members = 1.0 + 0.5 * jr.normal(jr.key(0), (30, 4))  # → (N_E, D_x)
# H selects components 0 and 2
obs_map = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])  # → (D_y, D_x)
# y
observation = jnp.array([0.4, -0.1])  # → (D_y,)
# today (lineax)
# Σ_ε = 0.1 I
noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))  # (D_y, D_y) operator


# user
# h(x) = H x
def observe(state: Float[Array, " state"]) -> Float[Array, " obs"]:
    return obs_map @ state


# user
# √(diag Σ̂_x)
def spread(ensemble: Float[Array, "member state"]) -> Float[Array, " state"]:
    # today (filterax)
    anomalies = flx.ensemble_anomalies(ensemble)  # → (N_E, D_x)
    return jnp.sqrt(einx.sum("[member] state", anomalies**2) / 29.0)  # → (D_x,)


# today (filterax)
model = flx.filters.StochasticEnKF(key=jr.key(1))
# today (filterax)
# X_t|t and log 𝒩(r̄; 𝟎, Σ̂_y)
result = model.analysis(members, observation, observe, noise)
# today (filterax)
fresh = model.analysis(members, observation, observe, noise, key=jr.key(2))
print(spread(members).round(3), spread(result.particles).round(3))
print(spread(fresh.particles).round(3), f"{result.log_likelihood:.3f}")
```

It prints `[0.43  0.528 0.531 0.522] [0.256 0.518 0.294 0.523]` and `[0.289 0.491 0.272 0.492] -3.401`.
The observed components contract and the others move only through their sample correlation with them.
A second key gives a different spread each time, where the ETKF of [chapter 6](06_etkf.md) gets the single Kalman value `[0.255 0.516 0.272 0.511]`.

`result.log_likelihood` is the Gaussian evidence of the mean residual under $\hat{\gmat{\Sigma}}_y$, the same number the ETKF reports on these inputs, and it does not depend on the draw.

## Agreement with the Kalman update

Three checks follow from the algebra above.
Given the same draws, the production route must match [](#code-enkf-algorithm).
With no draws, the covariance must match [](#eq-enkf-unperturbed) exactly.
And averaged over many draws, the mean and covariance must approach the Kalman values, by [](#eq-enkf-expected-covariance).

```{code-block} python
:caption: The production update against the written-out one, the exact shortfall without perturbations, and the average over 4,000 draws.
:label: code-enkf-agreement
# Shapes: N_E = 30; D_x = 4; D_y = 2; K = 4,000 draws

# h⁽ᵉ⁾ = H x⁽ᵉ⁾
mapped = jax.vmap(observe)(members)  # → (N_E, D_y)
# Σ_ε
noise_matrix = 0.1 * jnp.eye(2)  # → (D_y, D_y)
# today (filterax)
# y + ε⁽ᵉ⁾
perturbed = flx.perturbed_observations(  # → (N_E, D_y)
    jr.key(1), observation, noise, 30
)
# today (gaussx)
from_stack = gaussx.enkf_analysis(  # → (N_E, D_x)
    members, mapped, observation, noise, perturbed_obs=perturbed
)
by_hand = enkf_by_hand(members, mapped, perturbed, noise_matrix)  # → (N_E, D_x)
hand_error = jnp.abs(by_hand - from_stack).max()  # → ()
key_error = jnp.abs(result.particles - from_stack).max()  # → ()

# today (gaussx)
# Ĝ
gain = gaussx.ensemble_kalman_gain(members, mapped, noise)  # → (D_x, D_y)
# today (filterax)
# Σ̂_x
prior_covariance = flx.ensemble_covariance(members).as_matrix()  # → (D_x, D_x)
# H Σ̂_x
mapped_cross = obs_map @ prior_covariance  # → (D_y, D_x)
# Σ̂_x|y = Σ̂_x − Ĝ H Σ̂_x
kalman_covariance = prior_covariance - einx.dot(  # → (D_x, D_x)
    "state obs, obs other -> state other", gain, mapped_cross
)
# r̄ = y − h̄
residual = observation - einx.mean("[member] obs", mapped)  # → (D_y,)
# x̄ + Ĝ r̄
kalman_mean = einx.mean("[member] state", members) + gain @ residual  # → (D_x,)

# y, repeated for every member
repeated = jnp.broadcast_to(observation, mapped.shape)  # → (N_E, D_y)
# today (gaussx)
unperturbed = gaussx.enkf_analysis(  # → (N_E, D_x)
    members, mapped, observation, noise, perturbed_obs=repeated
)
# Ĝ Σ_ε Ĝᵀ
noise_term = 0.1 * einx.dot("state obs, other obs -> state other", gain, gain)
# today (filterax)
unperturbed_covariance = flx.ensemble_covariance(unperturbed).as_matrix()
shortfall_error = jnp.abs(  # → ()
    unperturbed_covariance - (kalman_covariance - noise_term)
).max()
print(f"{hand_error:.0e} {key_error:.0e} {shortfall_error:.0e}")
print(jnp.diagonal(kalman_covariance).round(4))
print(jnp.diagonal(unperturbed_covariance).round(4))


# user
# X_t|t for one draw
def one_draw(key: PRNGKeyArray) -> Float[Array, "member state"]:
    # today (filterax)
    return model.analysis(members, observation, observe, noise, key=key).particles


# today (jax)
draws = jax.vmap(one_draw)(jr.split(jr.key(3), 4000))  # → (K, N_E, D_x)
# (1 / K) ∑ x̄_t|t
average_mean = einx.mean("[draw member] state", draws)  # → (D_x,)
# (1 / K) ∑ Σ̂_t|t
average_covariance = einx.mean(  # → (D_x, D_x)
    "[draw] state other",
    jax.vmap(lambda ensemble: flx.ensemble_covariance(ensemble).as_matrix())(draws),
)
mean_gap = jnp.abs(average_mean - kalman_mean).max()  # → ()
covariance_gap = jnp.abs(average_covariance - kalman_covariance).max()  # → ()
print(f"{mean_gap:.1e} {covariance_gap:.1e}")
```

It prints `2e-16 0e+00 2e-16`, then the two diagonals `[0.0648 0.2662 0.0737 0.2608]` and `[0.0228 0.2617 0.0193 0.2569]`, then `1.0e-03 2.7e-04`.
The first line is round-off, and it shows that `key` and `perturbed_obs` use the same draws for this diagonal noise.
The diagonals show what the unperturbed ensemble loses on the observed components.
The last line is Monte Carlo error over 4,000 draws, not a bias.

## Sampling noise, measured

[](#code-enkf-sampling-noise) repeats the analysis over 500 draws at each ensemble size and compares the error in the mean with the closed form [](#eq-enkf-mean-noise).

```{code-block} python
:caption: The noise one draw adds to the mean and the covariance, against ensemble size.
:label: code-enkf-sampling-noise
# Shapes: K = 500 draws; D_x = 4; D_y = 2; N_E varies


# user
# X_t|t for one stack of draws, optionally with its sample mean removed
def update_with(
    ensemble: Float[Array, "member state"],
    mapped_ensemble: Float[Array, "member obs"],
    key: PRNGKeyArray,
    centre: bool,
) -> Float[Array, "member state"]:
    # today (filterax)
    stack = flx.perturbed_observations(  # → (N_E, D_y)
        key, observation, noise, ensemble.shape[0]
    )
    if centre:
        # y + ε⁽ᵉ⁾ − ε̄
        stack = stack - einx.mean("[member] obs", stack) + observation  # → (N_E, D_y)
    # today (gaussx)
    return gaussx.enkf_analysis(  # → (N_E, D_x)
        ensemble, mapped_ensemble, observation, noise, perturbed_obs=stack
    )


# user
# closed-form mean noise, then measured mean and covariance noise
def noise_levels(size: int, centre: bool) -> tuple[Array, Array, Array]:
    # x⁽ᵉ⁾ ~ 𝒩(𝟏, 0.25 I)
    ensemble = 1.0 + 0.5 * jr.normal(jr.key(size), (size, 4))  # → (N_E, D_x)
    mapped_ensemble = jax.vmap(observe)(ensemble)  # → (N_E, D_y)
    # today (gaussx)
    gain = gaussx.ensemble_kalman_gain(ensemble, mapped_ensemble, noise)  # → (D_x, D_y)
    # today (filterax)
    prior = flx.ensemble_covariance(ensemble).as_matrix()  # → (D_x, D_x)
    # Σ̂_x|y = Σ̂_x − Ĝ H Σ̂_x
    target_covariance = prior - gain @ (obs_map @ prior)  # → (D_x, D_x)
    # r̄ = y − h̄
    residual = observation - einx.mean("[member] obs", mapped_ensemble)  # → (D_y,)
    # x̄ + Ĝ r̄
    target_mean = einx.mean("[member] state", ensemble) + gain @ residual  # → (D_x,)
    # √(tr(Ĝ Σ_ε Ĝᵀ) / N_E)
    predicted = jnp.sqrt(0.1 * jnp.sum(gain**2) / size)  # → ()
    # today (jax)
    draws = jax.vmap(  # → (K, N_E, D_x)
        lambda key: update_with(ensemble, mapped_ensemble, key, centre)
    )(jr.split(jr.key(size + 1), 500))
    # x̄_t|t − (x̄ + Ĝ r̄), one row per draw
    mean_errors = einx.mean("draw [member] state", draws) - target_mean  # → (K, D_x)
    # √(mean ‖·‖²)
    mean_rms = jnp.sqrt(jnp.mean(einx.sum("draw [state]", mean_errors**2)))  # → ()
    # today (filterax)
    covariances = jax.vmap(  # → (K, D_x, D_x)
        lambda draw: flx.ensemble_covariance(draw).as_matrix()
    )(draws)
    # ‖Σ̂_t|t − Σ̂_x|y‖_F / ‖Σ̂_x|y‖_F, as an RMS over draws
    gaps = einx.sum("draw [state other]", (covariances - target_covariance) ** 2)
    return predicted, mean_rms, jnp.sqrt(jnp.mean(gaps) / jnp.sum(target_covariance**2))


for size in (10, 40, 160, 640):
    print(size, [f"{level:.2e}" for level in noise_levels(size, False)])
print([f"{level:.2e}" for level in noise_levels(40, True)])
```

It prints:

| $N_\mathcal{E}$ | Closed-form mean noise | Measured mean noise | Relative covariance noise |
| --- | --- | --- | --- |
| 10 | `1.05e-01` | `1.09e-01` | `3.82e-01` |
| 40 | `5.47e-02` | `5.26e-02` | `1.60e-01` |
| 160 | `2.62e-02` | `2.59e-02` | `8.39e-02` |
| 640 | `1.29e-02` | `1.27e-02` | `3.89e-02` |

The measured mean noise matches [](#eq-enkf-mean-noise), and both the mean and the covariance noise halve with each fourfold increase in members.

Centring the stack removes $\bar{\gvect{\varepsilon}}$ by construction, and `perturbed_obs` makes that available.
The last print repeats $N_\mathcal{E} = 40$ with centred draws and gives `5.47e-02`, `2.51e-16` and `1.60e-01`.
The mean noise falls to round-off and the covariance noise is unchanged, because $\hat{\gmat{\Sigma}}_{\varepsilon}$ and $\mat{N}$ depend only on the anomalies.

## Which route, and float32

The two routes of the table solve the same system, and in float64 they agree to round-off.
They do not agree in float32 when observations are strong or their variances span many decades.
The Woodbury form subtracts two nearly equal large terms when $\gmat{\Sigma}_\varepsilon^{-1}$ is large, and float32 loses the difference.

[](#code-enkf-float32) measures the gain in float32 against a float64 dense solve.

```{code-block} python
:caption: The float32 gain on both routes, against a float64 dense solve.
:label: code-enkf-float32
import numpy as np


# user
# Ĝ = Σ̂_xy (Σ̂_h + Σ_ε)⁻¹, by a dense float64 solve
def dense_gain(obs_members, variances):
    anomalies = obs_members - obs_members.mean(0)
    divisor = len(obs_members) - 1
    innovation = anomalies.T @ anomalies / divisor + np.diag(variances)
    return np.linalg.solve(innovation, anomalies.T @ anomalies / divisor).T


rng = np.random.default_rng(1)
for num_members, num_obs, variances in [
    (30, 3, np.array([1.0, 1.0, 1.0])),
    (30, 3, np.array([1e-6, 1e-6, 1e-6])),
    (30, 3, np.array([1e-6, 1.0, 1e6])),
    (5, 8, np.ones(8)),
    (5, 8, np.logspace(-6, 6, 8)),
]:
    obs_members = rng.normal(size=(num_members, num_obs))
    expected = dense_gain(obs_members, variances)
    errors = []
    for dense in (True, False):
        as_float32 = jnp.asarray(obs_members, jnp.float32)  # → (N_E, D_y)
        # today (gaussx)
        computed = gaussx.ensemble_kalman_gain(  # → (D_y, D_y)
            as_float32,
            as_float32,
            lx.DiagonalLinearOperator(jnp.asarray(variances, jnp.float32)),
            dense_innovation=dense,
        )
        error = np.abs(np.asarray(computed) - expected).max() / np.abs(expected).max()
        errors.append(f"{error:.0e}")
    print(num_members, num_obs, errors)
```

It prints:

| $N_\mathcal{E}$ | $D_y$ | Variances | Dense route | Woodbury route | Default |
| --- | --- | --- | --- | --- | --- |
| 30 | 3 | all 1 | `2e-07` | `1e-07` | dense |
| 30 | 3 | all $10^{-6}$ | `1e-07` | `1e-01` | dense |
| 30 | 3 | $10^{-6}$ to $10^{6}$ | `4e-08` | `6e-02` | dense |
| 5 | 8 | all 1 | `3e-07` | `2e-07` | Woodbury |
| 5 | 8 | $10^{-6}$ to $10^{6}$ | `9e-08` | `5e-02` | Woodbury |

The dense route holds float32 accuracy throughout.
So the exposed case is the last row, with fewer members than observations and a wide range of variances, where the default takes the Woodbury route.
`filters.StochasticEnKF` has no argument for this.
Call `gaussx.enkf_analysis(..., dense_innovation=True)` directly, or move to float64.

A singular $\gmat{\Sigma}_\varepsilon$ fails the same way, because the Woodbury route solves against it.

```{code-block} python
:caption: A zero noise variance with fewer members than observations.
:label: code-enkf-singular
# Shapes: N_E = 5 members; D_x = D_y = 8

# x⁽ᵉ⁾ ~ 𝒩(𝟎, I)
small = jr.normal(jr.key(0), (5, 8))  # → (N_E, D_x)
# today (lineax)
# Σ_ε = diag(1, …, 1, 0)
singular = lx.DiagonalLinearOperator(jnp.ones(8).at[7].set(0.0))  # (D_y, D_y) operator
# today (filterax)
default_route = flx.filters.StochasticEnKF(key=jr.key(1)).analysis(
    small, jnp.zeros(8), lambda state: state, singular
)
# today (gaussx)
dense_route = gaussx.enkf_analysis(  # → (N_E, D_x)
    small, small, jnp.zeros(8), singular, key=jr.key(1), dense_innovation=True
)
print(jnp.isfinite(default_route.particles).all(), jnp.isfinite(dense_route).all())
```

It prints `False` and `True`: the filter returns non-finite members with no error raised, and the forced dense route is finite.

## Differentiating the update

The update has no eigendecomposition, so the trouble of [chapter 6](06_etkf.md) does not arise.
With the key held fixed, the draw is a differentiable function of the noise through $\gvect{\varepsilon}^{(e)} = \mat{L} \vect{n}^{(e)}$ with fixed $\vect{n}^{(e)}$, so [](#eq-enkf-update) is differentiable as written.

```{code-block} python
:caption: The gradient with respect to the observation noise, with the draw held fixed, against a central difference.
:label: code-enkf-gradient
# user
# 𝓛(σ²) = ∑ sin X_t|t, with Σ_ε = σ² I
def loss(variance: Float[Array, ""]) -> Float[Array, ""]:
    # today (lineax)
    scaled = lx.DiagonalLinearOperator(variance * jnp.ones(2))  # (D_y, D_y) operator
    # today (filterax)
    analysis = model.analysis(members, observation, observe, scaled)
    return jnp.sum(jnp.sin(analysis.particles))  # → ()


at = jnp.asarray(0.1)
step = 1e-6
central = (loss(at + step) - loss(at - step)) / (2.0 * step)  # → ()
print(f"{jax.grad(loss)(at):.6f} {central:.6f}")
```

It prints `76.584646 76.584646`.
The gradient is that of one realisation, so a training loss built on it inherits the sampling noise above.
Fixing the key across training steps fits to that one draw.
[Chapter 14](14_differentiable.md) discusses the alternatives.

## When it breaks

### Small ensembles

The sampling noise adds to the ordinary weakness of a small ensemble, which is that it underestimates its own error.
[](#code-enkf-twin) runs the stochastic EnKF and the ETKF on the Lorenz-96 twin of [chapter 6](06_etkf.md), with $D_x = 40$ and every second site observed with unit variance {cite:p}`lorenz2006predictability`.
Neither filter has inflation or localization.

```{code-block} python
:caption: The stochastic EnKF against the ETKF on Lorenz-96, for one realisation at each ensemble size.
:label: code-enkf-twin
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
# yₜ = H xₜ + εₜ, εₜ ~ 𝒩(0, I)
observed = truth[:, sites] + jr.normal(jr.key(1), (200, 20))  # → (T, D_y)
unit_noise = lx.DiagonalLinearOperator(jnp.ones(20))  # (D_y, D_y) operator


# user
# h(x) = H x
def observe_sites(state: Float[Array, " state"]) -> Float[Array, " obs"]:
    return state[sites]


# user
# one forecast and one analysis per window, compiled once by the scan
def run_twin(analyse, num_members: int) -> Float[Array, "T member state"]:
    # x⁽ᵉ⁾ ~ 𝒩(x₀, 4 I)
    start = spun_up + 2.0 * jr.normal(jr.key(2), (num_members, 40))  # → (N_E, D_x)

    def window(particles, inputs):
        target, index = inputs
        # X_t|t−1 = f(X_t−1|t−1)
        forecast = jax.vmap(rk4)(particles)  # → (N_E, D_x)
        analysis = analyse(forecast, target, index)  # → (N_E, D_x)
        return analysis, analysis

    _, history = jax.lax.scan(window, start, (observed, jnp.arange(200)))
    return history  # → (T, N_E, D_x)


# user
# RMSE and spread over the second half of the run
def scores(history: Float[Array, "T member state"]) -> tuple[Array, Array]:
    num_members = history.shape[1]
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
    return rmse, jnp.sqrt(jnp.sum(anomalies**2) / (100 * (num_members - 1) * 40))


# today (filterax)
stochastic = flx.filters.StochasticEnKF()
# today (filterax)
deterministic = flx.filters.ETKF()


# user
# a fresh key per window, as the Layer 2 model derives it
def analyse_stochastic(forecast, target, index):
    key = jr.fold_in(jr.PRNGKey(3), index)
    return stochastic.analysis(
        forecast, target, observe_sites, unit_noise, key=key
    ).particles


# user
def analyse_deterministic(forecast, target, index):
    return deterministic.analysis(forecast, target, observe_sites, unit_noise).particles


for num_members in (40, 60, 100):
    stochastic_scores = scores(run_twin(analyse_stochastic, num_members))
    deterministic_scores = scores(run_twin(analyse_deterministic, num_members))
    print(
        num_members,
        [f"{value:.2f}" for value in stochastic_scores],
        [f"{value:.2f}" for value in deterministic_scores],
    )

# today (filterax)
cycle = flx.StochasticEnKF(dynamics=rk4, obs_op=observe_sites, seed=3)
start = spun_up + 2.0 * jr.normal(jr.key(2), (40, 40))  # → (N_E, D_x)
windows = [(observed[t], 0.05 * (t + 1)) for t in range(5)]
# today (filterax)
layer_two = cycle.assimilate(start, windows, unit_noise).analysis_history
layer_one = run_twin(analyse_stochastic, 40)[:5]  # → (5, N_E, D_x)
print(f"{jnp.abs(layer_two - layer_one).max():.0e}")
```

It prints:

| $N_\mathcal{E}$ | Stochastic RMSE | Stochastic spread | ETKF RMSE | ETKF spread |
| --- | --- | --- | --- | --- |
| 40 | `0.66` | `0.21` | `0.31` | `0.25` |
| 60 | `3.52` | `0.23` | `0.30` | `0.26` |
| 100 | `0.50` | `0.24` | `0.35` | `0.26` |

The ETKF is healthy at all three sizes, and the stochastic EnKF is worse at each.
At 60 members it has diverged, with the error fifteen times the spread, although 40 members did not.
So these runs sit close to its divergence threshold, and they depend on the draws.
Inflation and localization ([chapter 12](12_inflation.md), [chapter 11](11_localization.md)) treat this for both filters.

The last print, `5e-15`, shows that the Layer 2 model and the compiled loop above agree over five windows.

### A repeated key

`analysis(..., key=None)` falls back to the key on the filter, so a hand-written loop that never passes a key reuses one draw every window.
Each member then sees the same perturbed observation again and again, and it converges to that value instead of to the posterior.
[](#code-enkf-repeated-key) makes this exact: 100 members of a static state with prior variance 1 assimilate 50 windows of unit-variance observations, so the true posterior variance is $1 / 51$.

```{code-block} python
:caption: Fifty windows of a static state, with a fresh key each window and with one repeated key.
:label: code-enkf-repeated-key
# Shapes: N_E = 100 members; D_x = D_y = 3; 50 windows

# x⁽ᵉ⁾ ~ 𝒩(𝟎, I)
prior_members = jr.normal(jr.key(1), (100, 3))  # → (N_E, D_x)
target_zero = jnp.zeros(3)  # → (D_y,)
# today (lineax)
# Σ_ε = I
identity_noise = lx.DiagonalLinearOperator(jnp.ones(3))  # (D_y, D_y) operator


# user
# one analysis with h(x) = x
@jax.jit
def assimilate_once(ensemble, key):
    # today (filterax)
    return flx.filters.StochasticEnKF().analysis(
        ensemble, target_zero, lambda state: state, identity_noise, key=key
    ).particles


fresh_members = prior_members
repeated_members = prior_members
base_key = jr.key(7)
for index in range(50):
    fresh_members = assimilate_once(fresh_members, jr.fold_in(base_key, index))
    repeated_members = assimilate_once(repeated_members, base_key)
for ensemble in (fresh_members, repeated_members):
    # today (filterax)
    variances = jnp.diagonal(flx.ensemble_covariance(ensemble).as_matrix())  # → (D_x,)
    print(variances.round(3))
print(f"{1.0 / 51.0:.3f}")
```

It prints `[0.019 0.017 0.027]` with fresh keys, `[0.712 0.966 1.156]` with one key, and `0.020` as the true value.
With a repeated key, the spread stays near the noise variance however many observations arrive.
Layer 2 is not exposed to this, because it folds the window index into its key.

### Non-Gaussian priors

[](#eq-enkf-update) is a Gaussian update in whatever coordinates the state is given.
For a non-Gaussian prior it is biased, and the bias does not shrink with ensemble size.
The transform of [chapter 6](06_etkf.md) has the same bias, though the two filters shape the ensemble differently {cite:p}`lawson2004implications`.

For one observation of one variable, the ETKF multiplies the anomalies by a constant, so it keeps the shape of the prior exactly.
The stochastic update convolves the prior with Gaussian draws, which moves its shape towards a Gaussian.
[](#code-enkf-skewness) measures this on a lognormal prior, with importance weights on the forecast members as the reference.

```{code-block} python
:caption: The skewness of the analysis from a lognormal prior.
:label: code-enkf-skewness
# Shapes: N_E = 4,000 members; D_x = D_y = 1

# x⁽ᵉ⁾ = exp(0.5 n⁽ᵉ⁾), n⁽ᵉ⁾ ~ 𝒩(0, 1)
lognormal = jnp.exp(0.5 * jr.normal(jr.key(9), (4000, 1)))  # → (N_E, D_x)
target_one = jnp.array([1.0])  # → (D_y,)
# today (lineax)
# Σ_ε = 0.25
quarter = lx.DiagonalLinearOperator(jnp.array([0.25]))  # (D_y, D_y) operator


# user
# the third standardised moment, with weights that sum to one
def skewness(values: Float[Array, " member"], weights: Float[Array, " member"]):
    centre = jnp.sum(weights * values)  # → ()
    variance = jnp.sum(weights * (values - centre) ** 2)  # → ()
    return jnp.sum(weights * (values - centre) ** 3) / variance**1.5  # → ()


uniform = jnp.full(4000, 1.0 / 4000.0)  # → (N_E,)
# today (filterax)
transform_members = flx.filters.ETKF().analysis(
    lognormal, target_one, lambda state: state, quarter
).particles
# today (filterax)
perturbed_members = flx.filters.StochasticEnKF(key=jr.key(10)).analysis(
    lognormal, target_one, lambda state: state, quarter
).particles
# wₑ ∝ 𝒩(y; x⁽ᵉ⁾, 0.25)
importance = jnp.exp(-0.5 * (target_one[0] - lognormal[:, 0]) ** 2 / 0.25)  # → (N_E,)
importance = importance / jnp.sum(importance)  # → (N_E,)
print(
    f"{skewness(lognormal[:, 0], uniform):.3f}",
    f"{skewness(transform_members[:, 0], uniform):.3f}",
    f"{skewness(perturbed_members[:, 0], uniform):.3f}",
    f"{skewness(lognormal[:, 0], importance):.3f}",
)
print(
    f"{jnp.mean(transform_members):.3f}",
    f"{jnp.mean(perturbed_members):.3f}",
    f"{jnp.sum(importance * lognormal[:, 0]):.3f}",
)
```

It prints the skewnesses `1.842 1.842 0.506 0.528`, for the prior, the ETKF, the stochastic EnKF and the importance-weighted reference.
The second line gives the means `1.057 1.055 0.979`, in the same order without the prior.
Both Kalman updates share nearly the same mean error, which is the bias of the Gaussian assumption.
The ETKF keeps all of the prior's skewness, the stochastic EnKF ends close to the reference here, and neither has the right mean.
The remedy for the bias is a change of coordinates, which the `gaussx.enkf_analysis` docstring discusses, and not a choice between these two filters.

## When the stochastic EnKF is the right answer

Use `filters.StochasticEnKF` when you want the simplest correct reference, with one gain and one line of update.
It also suits an ensemble large enough that $1 / \sqrt{N_\mathcal{E}}$ noise does not matter.
It is also natural when many members meet few observations, since the dense route costs only $O(N_\mathcal{E} D_y^2 + D_y^3)$ for the gain and has no ensemble-space matrix.
Its randomness is what the ensemble Kalman processes of [chapter 9](09_ensemble_kalman_processes.md) build on, so it is the direct bridge to them.

Reach for a square-root filter when the ensemble is small, since [](#code-enkf-twin) shows the price, or when you want a reproducible analysis with no key to manage.
Reach for the tapered gain of [chapter 11](11_localization.md) when you need the stochastic update with localization, through `gaussx.enkf_analysis(..., localization=...)`, because `filters.StochasticEnKF` does not pass tapers on.

## See also

- [Chapter 3](03_observation_model.md): perturbed observations as part of the observation model
- [Chapter 4](04_kalman_update.md): the gain and the Woodbury solve behind it
- [Chapter 6](06_etkf.md): the ETKF, which removes the draws altogether
- [Chapter 11](11_localization.md) and [chapter 12](12_inflation.md): the treatments for a small ensemble
- [Chapter 16](16_assimilation_cycle.md): the cycle that threads a fresh key through each window
- [API: filters](https://jejjohnson.github.io/filterax/reference/filters/), [primitives](https://jejjohnson.github.io/filterax/reference/primitives/) and [differentiable training](https://jejjohnson.github.io/filterax/reference/differentiable/)

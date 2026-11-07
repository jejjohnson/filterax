---
numbering:
  enumerator: "16.%s"
---

# The assimilation cycle

The analysis chapters take one forecast ensemble to one analysis ensemble.
Real assimilation is a cycle of forecast, analysis and inflation over a stream of observations {cite:p}`evensen2009data,carrassi2018data,asch2016data`.
Filterax keeps the mathematics apart from the loop, so one analysis step can serve a Python loop, a compiled scan or an external orchestrator.

Three things organise this chapter.
The filtering recursion says what the loop must compute, and for a linear model it has an exact answer to test against.
The protocols and result types are the contract between the layers.
And the same step can be driven by several loops, which differ in what they retain and what they drop.

## The filtering recursion

Filtering alternates a prediction, which carries the posterior through the dynamics, with an update, which applies Bayes' rule to one window:

$$p(\vect{x}_t \mid \vect{y}_{1:t-1}) = \int p(\vect{x}_t \mid \vect{x}_{t-1}) p(\vect{x}_{t-1} \mid \vect{y}_{1:t-1}) \mathrm{d} \vect{x}_{t-1},$$ (eq-cycle-predict)

$$p(\vect{x}_t \mid \vect{y}_{1:t}) = \frac{p(\vect{y}_t \mid \vect{x}_t) p(\vect{x}_t \mid \vect{y}_{1:t-1})}{p(\vect{y}_t \mid \vect{y}_{1:t-1})}.$$ (eq-cycle-update)

The denominators accumulate into the evidence, which is the quantity diagnostics ([chapter 13](13_diagnostics.md)) and training ([chapter 14](14_differentiable.md)) rely on:

$$\log p(\vect{y}_{1:T}) = \sum_{t=1}^{T} \log p(\vect{y}_t \mid \vect{y}_{1:t-1}).$$ (eq-cycle-evidence)

The ensemble form runs each window with a time stamp $\tau_t$, and $\tau_0$ is the `t0` argument.
Each member moves from $\tau_{t-1}$ to $\tau_t$ through $f$, the analysis produces $\mat{X}^\star_{t \mid t}$, and the optional inflator acts on it with the forecast available:

$$\mat{X}_{t \mid t-1} = f(\mat{X}_{t-1 \mid t-1}; \tau_{t-1}, \tau_t), \qquad \mat{X}^\star_{t \mid t} = \op{analysis}(\mat{X}_{t \mid t-1}, \vect{y}_t), \qquad \mat{X}_{t \mid t} = \op{inflate}(\mat{X}^\star_{t \mid t}, \mat{X}_{t \mid t-1}).$$ (eq-cycle-ensemble)

All three ensembles are $N_\mathcal{E} \times D_x$, $f$ applies member by member, and $\tau_t$ is a scalar.
The factor in [](#eq-cycle-evidence) is approximated at window $t$ by $\log \mathcal{N}(\bar{\vect{r}}_t; \vect{0}, \hat{\gmat{\Sigma}}_{y,t})$, computed from the forecast and so before inflation of that window.
Two restrictions come with the loop.
The forecast has no $\gvect{\eta}_t$ of its own, so model error enters only if $f$ samples it or an inflator stands in for it.
And $\gmat{\Sigma}_\varepsilon$ is fixed for the whole call, which also fixes $D_y$.

### The linear case

Take $f(\vect{x}) = \mat{F} \vect{x}$ and $h(\vect{x}) = \mat{H} \vect{x}$, with no inflation.
A square-root filter ([chapter 6](06_etkf.md)) then keeps the sample mean and covariance exactly on the Kalman recursion, started from the initial sample moments:

$$\gvect{\mu}_{x,t \mid t-1} = \mat{F} \gvect{\mu}_{x,t-1 \mid t-1}, \qquad \gmat{\Sigma}_{x,t \mid t-1} = \mat{F} \gmat{\Sigma}_{x,t-1 \mid t-1} \mat{F}^\top,$$ (eq-cycle-kalman-predict)

$$\gvect{\mu}_{x,t \mid t} = \gvect{\mu}_{x,t \mid t-1} + \mat{G}_t \vect{r}_t, \qquad \gmat{\Sigma}_{x,t \mid t} = \gmat{\Sigma}_{x,t \mid t-1} - \mat{G}_t \mat{H} \gmat{\Sigma}_{x,t \mid t-1},$$ (eq-cycle-kalman-update)

with $\vect{r}_t = \vect{y}_t - \mat{H} \gvect{\mu}_{x,t \mid t-1}$, $\gmat{\Sigma}_{y,t} = \mat{H} \gmat{\Sigma}_{x,t \mid t-1} \mat{H}^\top + \gmat{\Sigma}_\varepsilon$ and $\mat{G}_t = \gmat{\Sigma}_{x,t \mid t-1} \mat{H}^\top \gmat{\Sigma}_{y,t}^{-1}$.
Linear prediction moves the sample moments exactly, because a linear map of an ensemble has the mapped mean and covariance.
Chapter 6 shows the analysis does the same.
So no sampling error appears at any window once the initial ensemble is given, and this is an exact test of a whole cycle.

## Implementation in filterax

### Layer 1: the analysis

A Layer 1 filter is a stateless [`AbstractSequentialFilter`](https://jejjohnson.github.io/filterax/reference/protocols/), with one method taking a forecast ensemble, an observation, an observation operator and a noise covariance.
It owns no time, no dynamics and no state, and it returns an `AnalysisResult`:

| Field | Shape | Meaning |
| --- | --- | --- |
| `particles` | $N_\mathcal{E} \times D_x$ | the analysis ensemble |
| `log_likelihood` | scalar, or `None` | $\log \mathcal{N}(\bar{\vect{r}}; \vect{0}, \hat{\gmat{\Sigma}}_y)$ |
| `diagnostics` | `dict`, or `None` | optional extras from the filter |

Layer 1 lives in `filterax.filters`: `ETKF`, `ETKF_Livings`, `EnSRF`, `EnSRF_Serial`, `ESTKF`, `LETKF` and `StochasticEnKF`.

### Layer 2: the loop

The Layer 2 models in the top-level namespace are `flx.ETKF`, `flx.EnSRF`, `flx.StochasticEnKF`, `flx.LETKF`, and the latent wrappers of [chapter 15](15_latent_da.md).
Sharing a name does not make them Layer 1 classes.
`flx.ETKF` has `dynamics`, `obs_op` and `inflator` fields and an `assimilate` method, and it is not an `AbstractSequentialFilter`.
`assimilate` takes a sequence of `(y, t)` pairs and runs [](#eq-cycle-ensemble) in a plain Python `for` loop, which is easy to replace.
It returns an `AssimilationResult`:

| Field | Shape | Meaning |
| --- | --- | --- |
| `particles` | $N_\mathcal{E} \times D_x$ | the terminal ensemble, ready for a follow-up forecast |
| `forecast_history` | $T \times N_\mathcal{E} \times D_x$ | $\mat{X}_{t \mid t-1}$ at each window |
| `analysis_history` | $T \times N_\mathcal{E} \times D_x$ | $\mat{X}_{t \mid t}$ at each window, after inflation |
| `log_likelihoods` | $T$, or `None` | the factors of [](#eq-cycle-evidence), one per window |

A missing likelihood at one window is padded with `NaN` so the time axes stay aligned.
Every shipped filter reports one, so no shipped model reaches that path.
With no windows, the histories have a leading axis of length 0 and `log_likelihoods` is `None`.

The example is a linear model of two damped rotations, with $N_\mathcal{E} = 8$ members and $D_x = 4$.
Components 0 and 2 are observed at noise variance 0.1 over $T = 10$ windows.
It implements three protocols as small modules.

```{code-block} python
:caption: Three protocols implemented as modules, and one Layer 2 run.
:label: code-cycle-setup
import einx
import equinox as eqx
import filterax as flx
import jax
import jax.numpy as jnp
import jax.random as jr
import jaxtyping as jt
import lineax as lx
import numpy as np
from pipekit import Operator
from pipekit_cycle import DAState, EnsembleDACycle, protocols

jax.config.update("jax_enable_x64", True)

# Shapes: N_E = 8 members; D_x = 4; D_y = 2; T = 10 windows


# user
class LinearModel(flx.AbstractDynamics):
    matrix: jt.Float[jt.Array, "state state"]

    # f(x) = F x
    def __call__(self, state, t0, t1):
        return einx.dot("state other, other -> state", self.matrix, state)


# user
class LinearObservation(flx.AbstractObsOperator):
    matrix: jt.Float[jt.Array, "obs state"]

    # h(x) = H x
    def __call__(self, state):
        return einx.dot("obs state, state -> obs", self.matrix, state)


# user
class DiagonalNoise(flx.AbstractNoise):
    variance: jt.Float[jt.Array, " obs"]

    # Σ_ε
    def covariance(self) -> lx.AbstractLinearOperator:
        return lx.DiagonalLinearOperator(self.variance)

    # εₜ ~ 𝒩(𝟎, Σ_ε), with the batch shape in front
    def sample(self, key: jt.PRNGKeyArray, shape: tuple[int, ...]) -> jt.Array:
        draws = jr.normal(key, (*shape, self.variance.shape[0]))  # → (*shape, D_y)
        return jnp.sqrt(self.variance) * draws


# F: two damped rotations
model_matrix = jnp.array(  # → (D_x, D_x)
    [
        [0.9, 0.1, 0.0, 0.0],
        [-0.1, 0.9, 0.0, 0.0],
        [0.0, 0.0, 0.95, 0.2],
        [0.0, 0.0, -0.2, 0.95],
    ]
)
# H selects components 0 and 2
obs_matrix = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])  # → (D_y, D_x)
dynamics = LinearModel(model_matrix)
obs_op = LinearObservation(obs_matrix)
noise_model = DiagonalNoise(0.1 * jnp.ones(2))
# Σ_ε = 0.1 I
noise = noise_model.covariance()  # (D_y, D_y) operator


# user
# xₜ₊₁ = F xₜ, stacked
def advance(state, _):
    next_state = dynamics(state, 0.0, 1.0)
    return next_state, next_state


_, truth = jax.lax.scan(advance, jnp.ones(4), length=10)  # → (T, D_x)
# εₜ ~ 𝒩(𝟎, Σ_ε)
errors = noise_model.sample(jr.key(1), (10,))  # → (T, D_y)
# yₜ = H x_true,t + εₜ
observed = jax.vmap(obs_op)(truth) + errors  # → (T, D_y)
# τₜ = t
times = jnp.arange(1.0, 11.0)  # → (T,)
windows = list(zip(observed, times))  # T of ((D_y,), ())
# x⁽ᵉ⁾ ~ 𝒩(𝟏, I)
members = 1.0 + jr.normal(jr.key(0), (8, 4))  # → (N_E, D_x)

# today (filterax)
model = flx.ETKF(dynamics=dynamics, obs_op=obs_op)
# today (filterax)
result = model.assimilate(members, windows, noise)
print(result.particles.shape, result.analysis_history.shape)
print(result.forecast_history.shape, result.log_likelihoods.shape)
print(f"{jnp.sum(result.log_likelihoods):.3f}")
# today (filterax)
no_windows = model.assimilate(members, [], noise)
print(no_windows.analysis_history.shape, no_windows.log_likelihoods)
```

It prints `(8, 4) (10, 8, 4)`, then `(10, 8, 4) (10,)`, then `-5.497`, then `(0, 8, 4) None`.

### Agreement with the Kalman filter

[](#code-cycle-kalman) runs [](#eq-cycle-kalman-predict) and [](#eq-cycle-kalman-update) from the initial sample moments, as a dense numpy reference, and compares every window.

```{code-block} python
:caption: Ten ETKF windows against a dense Kalman filter started from the same sample moments.
:label: code-cycle-kalman
# Shapes: N_E = 8 members; D_x = 4; D_y = 2; T = 10 windows

F = np.asarray(model_matrix)
H = np.asarray(obs_matrix)
R = 0.1 * np.eye(2)
mean = np.asarray(members).mean(0)
covariance = np.cov(np.asarray(members).T)
kalman_means, kalman_covariances, kalman_logs = [], [], []
for target in np.asarray(observed):
    mean = F @ mean
    covariance = F @ covariance @ F.T
    innovation = H @ covariance @ H.T + R
    gain = covariance @ H.T @ np.linalg.inv(innovation)
    residual = target - H @ mean
    quadratic = residual @ np.linalg.solve(innovation, residual)
    log_det = np.linalg.slogdet(innovation)[1]
    kalman_logs.append(-0.5 * (quadratic + log_det + 2.0 * np.log(2.0 * np.pi)))
    mean = mean + gain @ residual
    covariance = covariance - gain @ H @ covariance
    kalman_means.append(mean)
    kalman_covariances.append(covariance)

history = np.asarray(result.analysis_history)
mean_error = np.abs(history.mean(1) - np.array(kalman_means)).max()
covariance_error = max(
    np.abs(np.cov(history[t].T) - kalman_covariances[t]).max() for t in range(10)
)
log_error = np.abs(np.asarray(result.log_likelihoods) - np.array(kalman_logs)).max()
print(f"{mean_error:.0e} {covariance_error:.0e} {log_error:.0e}")
```

It prints `3e-16 2e-16 1e-15`, which is float64 round-off in the mean, the covariance and the likelihood.

## The protocols

Every extension point is an abstract `equinox.Module`, so an implementation is a PyTree that survives `jit`, `grad` and `vmap`.
They do not all have a consumer yet, which is worth knowing before implementing one.

| Protocol | Method | Consumed by | Ships with |
| --- | --- | --- | --- |
| `AbstractDynamics` | `__call__(state, t0, t1)` | every filterax loop | `LatentDynamics`, `EncodedDynamics` |
| `AbstractObsOperator` | `__call__(state)` | every analysis | `LiftedObs` |
| `AbstractSequentialFilter` | `analysis(particles, obs, obs_op, obs_noise, **kwargs)` | `differentiable_assimilate`, `FilterAnalysisStep` | the Layer 1 filters |
| `AbstractInflator` | `__call__(particles, forecast_particles, **kwargs)` | every filterax loop | `MultiplicativeInflator`, `RTPS`, `RTPP`, `AdditiveInflator` |
| `AbstractProcess` | `init(particles, obs, noise_cov)`, `update(state, forward_evals, **kwargs)` | the Layer 2 process loop | `EKI`, `EKS_Process`, `UKI`, `ETKI`, `GNKI`, `SparseInversion`, `TEKI` |
| `AbstractScheduler` | `get_dt(state)` | the processes | `FixedScheduler`, `DataMisfitController`, `EKSStableScheduler` |
| `AbstractNoise` | `covariance()`, `sample(key, shape)` | nothing yet | no subclass |
| `AbstractLocalizer` | `__call__(cov, coords)` | nothing yet | no subclass |

The dynamics are called on one member, so the loops wrap them in a `vmap` over members with the two times held fixed.
A plain function with the right signature also works in Layer 2, as [chapter 6](06_etkf.md) shows, since the fields are not checked at run time.

`AbstractNoise` bundles a covariance and a sampler, and the filters take the operator alone, so pass `noise_model.covariance()` as in [](#code-cycle-setup).
`AbstractLocalizer` has no consumer because `LETKF` takes a `radius` and the other localized routes take taper arrays ([chapter 11](11_localization.md)).
A subclass is useful in your own loop, built on the primitives as in [](#code-cycle-types).

### State and configuration

| Type | Fields | Role |
| --- | --- | --- |
| `FilterState` | `particles`, `step` | a scan carry for your own filter loop; no shipped loop uses it |
| `FilterConfig` | `n_ensemble`, `localizer`, `inflator` | accepted by every Layer 2 model and reserved; its fields have no effect today |
| `ProcessState` | `particles`, `forward_evals`, `obs`, `noise_cov`, `step`, `algo_time`, `prior_cov` | the state threaded through `AbstractProcess` |
| `UKIState` | `mean`, `covariance`, `step` | the parametric belief of `UKI`, from `init_parametric` and `update_parametric` |
| `ProcessConfig` | `scheduler`, `n_iterations` | scheduler and iteration count for the Layer 2 processes; unlike `FilterConfig`, it takes effect |

Iteration counters are scalar array leaves, so they pass through `jax.lax.scan` without triggering recompilation.

[](#code-cycle-types) uses each of these.
It writes the cycle by hand as a scan over `FilterState`, implements a localizer, tests `FilterConfig`, and runs one process of each state type.
The inversion problem $\vect{y} = \mat{A} \gvect{\theta} + \gvect{\varepsilon}$ has $D_\theta = 2$ and a known Gaussian posterior.
One `UKI` step of unit size from the prior $\mathcal{N}(\vect{0}, \mat{I})$ should reach it exactly ([chapter 9](09_ensemble_kalman_processes.md)).

```{code-block} python
:caption: A hand-written cycle over `FilterState`, a localizer, `FilterConfig`, and the two process states.
:label: code-cycle-types
# Shapes: N_E = 8 members; D_x = 4; D_y = 2; T = 10 windows; D_θ = 2

# today (filterax)
analysis_filter = flx.filters.ETKF()


# user
# one window of the cycle: forecast, then analysis
def cycle_step(state: flx.FilterState, inputs):
    target, previous, now = inputs

    # user
    # X_t|t−1 = f(X_t−1|t−1; τₜ₋₁, τₜ)
    def forecast_member(member):
        return dynamics(member, previous, now)

    forecast = jax.vmap(forecast_member)(state.particles)  # → (N_E, D_x)
    # today (filterax)
    analysis = analysis_filter.analysis(forecast, target, obs_op, noise)
    # today (filterax)
    next_state = flx.FilterState(particles=analysis.particles, step=state.step + 1)
    return next_state, (analysis.particles, analysis.log_likelihood)


# τₜ₋₁
previous_times = jnp.concatenate((jnp.zeros(1), times[:-1]))  # → (T,)
# today (filterax)
first_state = flx.FilterState(particles=members, step=jnp.asarray(0))
final_state, (scan_history, scan_logs) = jax.lax.scan(
    cycle_step, first_state, (observed, previous_times, times)
)
print(final_state.step, f"{jnp.abs(scan_history - result.analysis_history).max():.0e}")
print(f"{jnp.abs(scan_logs - result.log_likelihoods).max():.0e}")


# user
class GaspariCohnLocalizer(flx.AbstractLocalizer):
    radius: jt.Float[jt.Array, ""]

    # ρ ∘ Σ̂_xy, with coords = (state coordinates, observation coordinates)
    def __call__(self, cov, coords):
        state_coords, obs_coords = coords
        # today (filterax)
        taper = flx.localization_matrix(state_coords, obs_coords, self.radius)
        # today (filterax)
        return flx.localize(cov, taper)


state_sites = einx.id("site -> site 1", jnp.arange(4.0))  # → (D_x, 1)
obs_sites = jnp.array([[0.0], [2.0]])  # → (D_y, 1)
mapped = jax.vmap(obs_op)(members)  # → (N_E, D_y)
# today (filterax)
# Σ̂_xy
cross = flx.cross_covariance(members, mapped)  # → (D_x, D_y)
localizer = GaspariCohnLocalizer(radius=jnp.asarray(1.0))
tapered = localizer(cross, (state_sites, obs_sites))
print(jnp.count_nonzero(cross), jnp.count_nonzero(tapered))

# today (filterax)
configured = flx.ETKF(
    dynamics=dynamics,
    obs_op=obs_op,
    config=flx.FilterConfig(
        n_ensemble=99, inflator=flx.MultiplicativeInflator(factor=5.0)
    ),
).assimilate(members, windows, noise)
print(f"{jnp.abs(configured.particles - result.particles).max():.0e}")

# A
forward_matrix = jnp.array([[1.0, 0.5], [-0.3, 1.2]])  # → (D_y, D_θ)
# y
target = jnp.array([0.5, 0.9])  # → (D_y,)


# user
# 𝒢(θ) = A θ
def forward(theta):
    return einx.dot("obs param, param -> obs", forward_matrix, theta)


# today (filterax)
eki = flx.processes.EKI(scheduler=flx.FixedScheduler(dt=0.5))
# today (filterax)
process_state = eki.init(jr.normal(jr.key(0), (8, 2)), target, noise)
for _ in range(4):
    evaluations = jax.vmap(forward)(process_state.particles)  # → (N_E, D_y)
    process_state = eki.update(process_state, evaluations)
print(type(process_state).__name__, process_state.step, process_state.algo_time)

# today (filterax)
uki = flx.processes.UKI(scheduler=flx.FixedScheduler(dt=1.0))
# today (filterax)
belief = uki.init_parametric(
    jnp.zeros(2), lx.DiagonalLinearOperator(jnp.ones(2)), target, noise
)
# today (filterax)
points, _, _ = flx.processes.sigma_points(
    belief.mean, belief.covariance, alpha=uki.alpha, beta=uki.beta, kappa=uki.kappa
)
# today (filterax)
belief = uki.update_parametric(
    belief,
    jax.vmap(forward)(points),
    obs=target,
    noise_cov=noise,
    dt=jnp.asarray(1.0),
)
# Σ_θ|y = (I + Aᵀ Σ_ε⁻¹ A)⁻¹, and μ_θ|y = Σ_θ|y Aᵀ Σ_ε⁻¹ y
A = np.asarray(forward_matrix)
posterior_covariance = np.linalg.inv(np.eye(2) + A.T @ A / 0.1)
posterior_mean = posterior_covariance @ A.T @ np.asarray(target) / 0.1
print(type(belief).__name__, belief.step, belief.mean.round(4))
print(f"{np.abs(np.asarray(belief.mean) - posterior_mean).max():.0e}")
print(f"{np.abs(belief.covariance.as_matrix() - posterior_covariance).max():.0e}")
print(isinstance(analysis_filter, flx.AbstractSequentialFilter))
print(isinstance(model, flx.AbstractSequentialFilter))
print(isinstance(eki, flx.AbstractProcess))
```

It prints:

| Output | Meaning |
| --- | --- |
| `10 2e-15` | the scan's step counter, and its difference from `assimilate` |
| `2e-15` | the difference in the likelihoods |
| `8 5` | the nonzero entries of $\hat{\gmat{\Sigma}}_{xy}$ before and after the taper |
| `0e+00` | a `FilterConfig` holding a factor-5 inflator changes nothing |
| `ProcessState 4 2.0` | four `EKI` steps of size 0.5, in `step` and `algo_time` |
| `UKIState 1 [0.1068 0.7347]` | the belief after one `UKI` step |
| `6e-17`, `3e-16` | its mean and covariance against the closed-form posterior |
| `True`, `False`, `True` | Layer 1 is an `AbstractSequentialFilter`, Layer 2 is not, and `EKI` is an `AbstractProcess` |

## Inflators in the loop

An inflator runs after the analysis and before the next forecast, and it receives both ensembles so that relaxation methods can reach the forecast.
The derivations are in [chapter 12](12_inflation.md), and all of these preserve the analysis mean.

| Inflator | Action on the anomalies | Parameter |
| --- | --- | --- |
| `MultiplicativeInflator` {cite:p}`anderson1999monte` | $\tilde{\mat{X}} \leftarrow \zeta \tilde{\mat{X}}$ | `factor`, $\zeta$ |
| `RTPS` {cite:p}`whitaker2012evaluating` | scales each component towards the forecast spread | `alpha`, $\alpha$ |
| `RTPP` {cite:p}`zhang2004impacts` | $\tilde{\mat{X}} \leftarrow (1 - \alpha) \tilde{\mat{X}} + \alpha \tilde{\mat{X}}_{t \mid t-1}$ | `alpha`, $\alpha$ |
| `AdditiveInflator` {cite:p}`hamill2005accounting` | adds a mean-free draw from a given covariance | `noise_cov`, `base_key` |

Three details of the loop matter.
The stored `analysis_history` holds the inflated ensemble.
The likelihood at window $t$ comes from the forecast, so inflation at window $t$ first affects the likelihood at window $t + 1$.
And the Layer 2 loop passes `step=` to the inflator, which `AdditiveInflator` folds into its key for a fresh draw at each window.
Called with no `step`, it repeats one draw.

```{code-block} python
:caption: Where inflation acts in the loop, and how `AdditiveInflator` uses `step`.
:label: code-cycle-inflator-order
# Shapes: N_E = 8 members; D_x = 4; D_y = 2; T = 10 windows

# today (filterax)
inflated = flx.ETKF(
    dynamics=dynamics, obs_op=obs_op, inflator=flx.MultiplicativeInflator(factor=1.1)
).assimilate(members, windows, noise)
# the likelihood differences at the first two windows
print(f"{inflated.log_likelihoods[0] - result.log_likelihoods[0]:.4f}", end=" ")
print(f"{inflated.log_likelihoods[1] - result.log_likelihoods[1]:.4f}")
# today (filterax)
first_analysis = analysis_filter.analysis(
    inflated.forecast_history[0], observed[0], obs_op, noise
).particles
# today (filterax)
expected = flx.inflate_multiplicative(first_analysis, 1.1)  # → (N_E, D_x)
print(f"{jnp.abs(inflated.analysis_history[0] - expected).max():.0e}")

# today (filterax)
additive = flx.AdditiveInflator(
    noise_cov=lx.DiagonalLinearOperator(0.01 * jnp.ones(4)), base_key=jr.key(4)
)
print(f"{jnp.abs(additive(members, step=0) - additive(members, step=1)).max():.3f}")
print(f"{jnp.abs(additive(members) - additive(members)).max():.3f}")
```

It prints `0.0000 -0.0445`, then `0e+00`, then `0.218`, then `0.000`.
So inflation leaves the first likelihood alone and changes the second, and the first stored analysis is exactly the inflated one.
Only a changing `step` gives a new draw.

[](#code-cycle-inflation) returns to the diverging run of [chapter 6](06_etkf.md).
That is Lorenz-96 with $D_x = 40$, half the sites observed with unit variance, and only $N_\mathcal{E} = 20$ members, over 200 windows.
It uses `differentiable_assimilate` for speed, since [](#code-cycle-pipekit) shows it matches the Python loop.
That function refuses `AdditiveInflator`, so the last case is a scan over `FilterState` that passes `step=` as the Layer 2 loop does.

```{code-block} python
:caption: Inflators on a small Lorenz-96 ensemble that diverges without them. RMSE and spread over the last 100 windows.
:label: code-cycle-inflation
# Shapes: T = 200 windows; D_x = 40 sites; D_y = 20 observed sites; N_E = 20 members


# user
# dxₖ/dt = (xₖ₊₁ − xₖ₋₂) xₖ₋₁ − xₖ + 8
def tendency(state):
    return (jnp.roll(state, -1) - jnp.roll(state, 2)) * jnp.roll(state, 1) - state + 8.0


# user
# one RK4 step of Δt = 0.05
def rk4(state, _=None, __=None):
    k1 = tendency(state)
    k2 = tendency(state + 0.025 * k1)
    k3 = tendency(state + 0.025 * k2)
    k4 = tendency(state + 0.05 * k3)
    return state + 0.05 / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# user
# xₜ₊₁ = f(xₜ), stacked
def advance_chaotic(state, _):
    next_state = rk4(state)
    return next_state, next_state


# user
# h(x) selects every second site
def observe_sites(state):
    return state[::2]


# x₀ on the attractor, after 1,000 steps from a perturbed rest state
spun_up = jax.lax.fori_loop(  # → (D_x,)
    0, 1000, lambda _, state: rk4(state), 8.0 + jr.normal(jr.key(0), (40,))
)
_, chaotic_truth = jax.lax.scan(advance_chaotic, spun_up, length=200)  # → (T, D_x)
# yₜ = H xₜ + εₜ, εₜ ~ 𝒩(0, I)
chaotic_observed = chaotic_truth[:, ::2] + jr.normal(jr.key(1), (200, 20))  # → (T, D_y)
chaotic_times = 0.05 * jnp.arange(1.0, 201.0)  # → (T,)
unit_noise = lx.DiagonalLinearOperator(jnp.ones(20))  # (D_y, D_y) operator
# x⁽ᵉ⁾ ~ 𝒩(x₀, 4 I)
chaotic_start = spun_up + 2.0 * jr.normal(jr.key(2), (20, 40))  # → (N_E, D_x)


# user
# RMSE of x̄_t|t and the mean spread, over the second half of the run
def skill(chaotic_history):
    settled = chaotic_history[100:]  # → (T / 2, N_E, D_x)
    settled_mean = einx.mean("t [member] state", settled)  # → (T / 2, D_x)
    rmse = jnp.sqrt(jnp.mean((settled_mean - chaotic_truth[100:]) ** 2))  # → ()
    # x̃⁽ᵉ⁾_t|t
    anomalies = einx.subtract(  # → (T / 2, N_E, D_x)
        "t member state, t state -> t member state", settled, settled_mean
    )
    # √(mean diag Σ̂_x|y)
    spread = jnp.sqrt(jnp.sum(anomalies**2) / (100 * 19 * 40))  # → ()
    return rmse, spread


# today (filterax)
for name, candidate in [
    ("none", None),
    ("multiplicative 1.05", flx.MultiplicativeInflator(factor=1.05)),
    ("multiplicative 1.2", flx.MultiplicativeInflator(factor=1.2)),
    ("RTPS 0.5", flx.RTPS(alpha=0.5)),
    ("RTPS 0.9", flx.RTPS(alpha=0.9)),
    ("RTPP 0.5", flx.RTPP(alpha=0.5)),
    ("RTPP 0.8", flx.RTPP(alpha=0.8)),
]:
    # today (filterax)
    run = flx.differentiable_assimilate(
        flx.filters.ETKF(),
        rk4,
        observe_sites,
        chaotic_start,
        chaotic_observed,
        chaotic_times,
        unit_noise,
        inflator=candidate,
    )
    rmse, spread = skill(run.analysis_history)
    print(name, f"{rmse:.2f} {spread:.2f}")

# today (filterax)
additive_inflator = flx.AdditiveInflator(
    noise_cov=lx.DiagonalLinearOperator(0.05 * jnp.ones(40)), base_key=jr.key(4)
)


# user
# one window with additive inflation, using the step counter for a fresh draw
def additive_step(state: flx.FilterState, target):
    forecast = jax.vmap(rk4)(state.particles)  # → (N_E, D_x)
    # today (filterax)
    analysis = analysis_filter.analysis(forecast, target, observe_sites, unit_noise)
    # today (filterax)
    spread_out = additive_inflator(analysis.particles, forecast, step=state.step)
    # today (filterax)
    return flx.FilterState(particles=spread_out, step=state.step + 1), spread_out


# today (filterax)
first_additive = flx.FilterState(particles=chaotic_start, step=jnp.asarray(0))
_, additive_history = jax.lax.scan(additive_step, first_additive, chaotic_observed)
rmse, spread = skill(additive_history)
print("additive 0.05", f"{rmse:.2f} {spread:.2f}")
```

It prints:

| Inflator | RMSE | Spread |
| --- | --- | --- |
| None | `2.37` | `0.26` |
| `MultiplicativeInflator(1.05)` | `0.36` | `0.39` |
| `MultiplicativeInflator(1.2)` | `0.55` | `0.79` |
| `RTPS(0.5)` | `0.38` | `0.47` |
| `RTPS(0.9)` | `nan` | `nan` |
| `RTPP(0.5)` | `1.60` | `0.42` |
| `RTPP(0.8)` | `1.02` | `1.01` |
| `AdditiveInflator` with variance 0.05 | `1.00` | `0.53` |

These are single settings on one toy, so they are not a ranking of the methods.
They do show the shape of the trade-off.
Without inflation the error is nine times the spread, and modest multiplicative inflation or `RTPS` brings it well below the observation noise with a matching spread.
More is not better.
A factor of 1.2 costs accuracy, and `RTPS(0.9)` makes the ensemble blow up with no exception raised, so the run ends in `nan`.
Check that a history is finite before reading its statistics.

## Cycling through pipekit

[pipekit-cycle](https://github.com/jejjohnson/pipekit) orchestrates assimilation cycles for any algorithm library through three runtime-checkable protocols.
Conformance is structural: filterax never imports pipekit, and the adapters in [`filterax.pipekit`](https://jejjohnson.github.io/filterax/reference/pipekit/) lift filterax components to the right shapes.

| pipekit protocol | Contract | filterax adapter | What the adapter does |
| --- | --- | --- | --- |
| `AnalysisStep` | `__call__(forecast, obs, *, obs_op, obs_err_cov)` | `FilterAnalysisStep(filter, analysis_kwargs)` | stacks a list of members and returns the same kind of container, coerces a dense `obs_err_cov` to a tagged operator, and raises `ValueError` if it is missing |
| `ForwardModel` | `step(state, dt)`, `dt`, `state_signature` | `DynamicsForwardModel(dynamics, dt)` | integrates over $[0, \mathrm{d}t]$ |
| `ObservationOperator` | `__call__(state)`, `linearize(state)` | `LinearizableObsOperator(obs_op)` | derives the Jacobian with `jax.jacfwd` |

`EnsembleDACycle` holds the ensemble as a Python list of member states.
It draws each observation from an `obs_source`, a pipekit `Operator` called with the cycle index.
$\gmat{\Sigma}_\varepsilon$ travels on the `DAState`, and is a dense array or a lineax operator.
[](#code-cycle-pipekit) runs the toy model of this chapter through it, for real.

```{code-block} python
:caption: The same run through `EnsembleDACycle`, the Python loop and the scan, and the inflated variant.
:label: code-cycle-pipekit
from filterax.pipekit import (
    DynamicsForwardModel,
    FilterAnalysisStep,
    LinearizableObsOperator,
)

# Shapes: N_E = 8 members; D_x = 4; D_y = 2; T = 10 windows


# user
class ObservationAt(Operator):
    def __init__(self, values):
        self.values = values

    # yₜ for the cycle index t
    def _apply(self, index):
        return self.values[index]


# today (filterax)
forward_model = DynamicsForwardModel(dynamics, dt=1.0)
# today (filterax)
linearizable = LinearizableObsOperator(obs_op)
# today (filterax)
analysis_step = FilterAnalysisStep(analysis_filter)
print(
    isinstance(forward_model, protocols.ForwardModel),
    isinstance(linearizable, protocols.ObservationOperator),
    isinstance(analysis_step, protocols.AnalysisStep),
)
# today (pipekit_cycle)
cycle = EnsembleDACycle(
    forward_model=forward_model,
    obs_op=linearizable,
    analysis_step=analysis_step,
    obs_source=ObservationAt(observed),
    n_steps=10,
    n_members=8,
)
# today (pipekit_cycle)
start_state = DAState(obs_err_cov=0.1 * jnp.eye(2))
analysed, end_state = cycle(list(members), start_state)
print(type(analysed).__name__, len(analysed), end_state.cycle_count, end_state.t)
stacked = jnp.stack(analysed)  # → (N_E, D_x)
print(f"{jnp.abs(stacked - result.particles).max():.0e}")

# today (filterax)
scanned = flx.differentiable_assimilate(
    analysis_filter, dynamics, obs_op, members, observed, times, noise
)
print(f"{jnp.abs(scanned.analysis_history - result.analysis_history).max():.0e}")
# ∂h/∂x from jax.jacfwd, against H
print(f"{jnp.abs(linearizable.linearize(members[0]) - obs_matrix).max():.0e}")


# user
class InflatedStep(eqx.Module):
    step: flx.pipekit.FilterAnalysisStep
    inflator: flx.AbstractInflator

    # X_t|t = inflate(analysis(X_t|t−1, yₜ), X_t|t−1)
    def __call__(self, forecast, obs, *, obs_op, obs_err_cov):
        stacked_forecast = jnp.stack(forecast)  # → (N_E, D_x)
        analysis = self.step(
            stacked_forecast, obs, obs_op=obs_op, obs_err_cov=obs_err_cov
        )
        return list(self.inflator(analysis, stacked_forecast))


inflating_step = InflatedStep(analysis_step, flx.RTPP(alpha=0.5))
inflating_cycle = EnsembleDACycle(
    forward_model=forward_model,
    obs_op=linearizable,
    analysis_step=inflating_step,
    obs_source=ObservationAt(observed),
    n_steps=10,
    n_members=8,
)
# today (pipekit_cycle)
relaxed, _ = inflating_cycle(list(members), DAState(obs_err_cov=0.1 * jnp.eye(2)))
# today (filterax)
relaxed_direct = flx.ETKF(
    dynamics=dynamics, obs_op=obs_op, inflator=flx.RTPP(alpha=0.5)
).assimilate(members, windows, noise)
print(f"{jnp.abs(jnp.stack(relaxed) - relaxed_direct.particles).max():.0e}")
# the other shapes that line up with pipekit-cycle's protocols
print(
    isinstance(flx.RTPP(alpha=0.5), protocols.EnsembleInflator),
    isinstance(noise_model, protocols.ObservationNoise),
    isinstance(localizer, protocols.CovarianceLocalizer),
    isinstance(flx.FixedScheduler(dt=1.0), protocols.StepScheduler),
    isinstance(eki, protocols.IterativeProcess),
    isinstance(flx.identity_latent_map(3), protocols.LatentMap),
)
```

It prints `True True True`, then `list 8 10 10.0`, then `6e-16`, `2e-15`, `0e+00` and `1e-15`, then six times `True`.
The container comes back as the list it went in as, and `DAState` has counted 10 cycles and advanced its clock.
All three loops agree to round-off, and the derived Jacobian equals $\mat{H}$.
pipekit owns the loop, the scheduling and the state container, and filterax owns the ensemble mathematics inside `FilterAnalysisStep`.
Swapping `ETKF` for another filter changes nothing else, and `LETKF` passes its coordinates through `analysis_kwargs`.

The other filterax types also fit pipekit-cycle's wider protocols structurally, though `EnsembleDACycle` has no slot for them.
That is why inflation needs the small `InflatedStep` wrapper above, whose result matches the Layer 2 run with the same `RTPP`.
The loop also drops things that Layer 2 keeps, as follows.

- `FilterAnalysisStep` returns only the ensemble, so `log_likelihood` and `diagnostics` are discarded.
- `obs_source` receives an index that restarts at 0 on every call of the cycle, so resuming a run needs an offset of your own.
- `DynamicsForwardModel` integrates over $[0, \mathrm{d}t]$ on every cycle, because pipekit does not pass absolute time, so the dynamics must be autonomous.
- pipekit passes no PRNG key to the analysis, so a stochastic filter reuses its stored key and repeats its perturbations, where Layer 2 `flx.StochasticEnKF` folds in a new key per window.

[](#code-cycle-limits) measures the last two.

```{code-block} python
:caption: Two limits of the adapters: autonomous forecasts and repeated noise draws.
:label: code-cycle-limits
# Shapes: N_E = 8 members; D_x = 4; D_y = 2


# user
class TimeDependent(flx.AbstractDynamics):
    # x(τₜ) = x(τₜ₋₁) + τₜ² − τₜ₋₁²
    def __call__(self, state, t0, t1):
        return state + t1**2 - t0**2


# user
# h(x) selects component 0
def observe_first(state):
    return state[:1]


flat_noise = lx.DiagonalLinearOperator(1e12 * jnp.ones(1))  # (1, 1) operator
start_points = jnp.array([[-1.0], [0.0], [1.0]])  # → (3, 1)
stamped = [(jnp.array([0.0]), t) for t in (1.0, 2.0, 3.0)]
# today (filterax)
direct_run = flx.ETKF(dynamics=TimeDependent(), obs_op=observe_first).assimilate(
    start_points, stamped, flat_noise
)
print(einx.mean("t [member] state", direct_run.forecast_history)[:, 0].round(3))
# today (filterax)
adapter = DynamicsForwardModel(TimeDependent(), dt=1.0)
value = jnp.zeros(1)  # → (1,)
stepped = []
for _ in range(3):
    value = adapter.step(value, adapter.dt)
    stepped.append(float(value[0]))
print(stepped)

# today (filterax)
stochastic_step = FilterAnalysisStep(flx.filters.StochasticEnKF(key=jr.key(3)))
first = stochastic_step(members, observed[0], obs_op=obs_op, obs_err_cov=noise)
second = stochastic_step(members, observed[0], obs_op=obs_op, obs_err_cov=noise)
print(f"{jnp.abs(first - second).max():.1f}")
```

It prints `[1. 4. 9.]`, then `[1.0, 2.0, 3.0]`, then `0.0`.
The Layer 2 forecast follows the time-dependent forcing from $\tau_{t-1}$ to $\tau_t$, while the adapter applies the first interval three times.
Two calls of the stochastic step give identical perturbations.

## When it breaks

The loop adds few failure modes of its own, and most of what can go wrong here fails silently.

- With no model noise and no inflation, the spread collapses and the observations stop mattering, as the first row after [](#code-cycle-inflation) shows.
- Too much inflation gives `nan` and no exception, as its `RTPS(0.9)` row shows.
- Setting `inflator` or `localizer` on a `FilterConfig` has no effect, as [](#code-cycle-types) shows, so set `inflator` on the model.
- An observation network that changes between windows does not fit `assimilate` or `differentiable_assimilate`, which each take one $\gmat{\Sigma}_\varepsilon$ per call, so write your own loop around `filter.analysis`.
- Time-dependent dynamics and stochastic filters both lose something through the pipekit adapters, as [](#code-cycle-limits) shows.

## When each loop is the right answer

All of these call the same Layer 1 analysis, so they agree where they overlap.

| Need | Use | What you give up |
| --- | --- | --- |
| A quick experiment or a short $T$ | `model.assimilate(...)` | speed, since each window dispatches operation by operation |
| Training through the filter, or a long $T$ | `differentiable_assimilate` | fixed-shape stacked inputs, and no stochastic filter or additive inflation |
| Gradient memory independent of $T$ | `road_enkf_loss_and_grad` | a biased gradient ([chapter 14](14_differentiable.md)) |
| Orchestration across libraries | `EnsembleDACycle` with `filterax.pipekit` | the likelihoods, inflation unless wrapped, and absolute time |
| A filter with no Layer 2 model, such as `ESTKF` | `differentiable_assimilate` or `FilterAnalysisStep` | as above |
| Custom cycling logic | your own loop around `filter.analysis(...)` | whatever you do not write back in |

## See also

- [Chapter 1](01_problem_setting.md): the filtering problem this cycle solves
- [Chapter 6](06_etkf.md): the analysis step used throughout this chapter
- [Chapter 9](09_ensemble_kalman_processes.md): the processes behind `AbstractProcess`, `ProcessState` and `UKIState`
- [Chapter 11](11_localization.md): the taper primitives behind `AbstractLocalizer`
- [Chapter 12](12_inflation.md): the inflate step of the loop
- [Chapter 14](14_differentiable.md): the scan-based replacement for the Python loop
- [Chapter 15](15_latent_da.md): the latent wrappers that reuse this loop in $\vect{z}$
- [API: protocols and types](https://jejjohnson.github.io/filterax/reference/protocols/) and [pipekit integration](https://jejjohnson.github.io/filterax/reference/pipekit/)

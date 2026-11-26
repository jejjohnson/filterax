---
numbering:
  enumerator: "1.%s"
---

# The Bayesian filtering problem

Data assimilation estimates a system from a model of how it evolves and partial, noisy observations of it.
The exact answer is a probability density over the state, and no large model can store or propagate that density.
filterax replaces the density with an ensemble of samples, and updates those samples with a Kalman gain built from their own statistics.

Three steps organise this chapter.
The Bayes filter states the exact recursion, and shows why it cannot be computed.
Linear maps and Gaussian errors close the recursion on two moments, which is the Kalman filter, and ensembles estimate those moments by sampling.
Finally, the same update serves two families of method, sequential filters for a state that moves and iterative processes for an unknown that stays still.

## The state-space model

Let $\vect{x}_t \in \mathbb{R}^{D_x}$ be the state at window $t$ and $\vect{y}_t \in \mathbb{R}^{D_y}$ the observations at that window.
Two equations define the problem:

$$\vect{x}_t = f(\vect{x}_{t-1}) + \gvect{\eta}_t, \qquad \gvect{\eta}_t \sim \mathcal{N}(\vect{0}, \gmat{\Sigma}_\eta),$$ (eq-problem-dynamics)

$$\vect{y}_t = h(\vect{x}_t) + \gvect{\varepsilon}_t, \qquad \gvect{\varepsilon}_t \sim \mathcal{N}(\vect{0}, \gmat{\Sigma}_\varepsilon).$$ (eq-problem-observation)

The dynamics $f : \mathbb{R}^{D_x} \to \mathbb{R}^{D_x}$ may be a discretised PDE, an ODE integrator or a learned emulator.
The observation operator $h : \mathbb{R}^{D_x} \to \mathbb{R}^{D_y}$ maps the state to what an instrument measures ([chapter 3](03_observation_model.md)).
The covariances $\gmat{\Sigma}_\eta \in \mathbb{R}^{D_x \times D_x}$ and $\gmat{\Sigma}_\varepsilon \in \mathbb{R}^{D_y \times D_y}$ describe the model error and the observation error.
Both errors are independent of each other and across time.

These assumptions give the two conditional independences the rest of the chapter uses.
The state is Markov, so $\vect{x}_t$ depends on the past only through $\vect{x}_{t-1}$.
And $\vect{y}_t$ depends on everything else only through $\vect{x}_t$.

## The Bayes filter

The exact solution alternates two steps {cite:p}`sarkka2013bayesian`.
Given the analysis density at $t-1$, the forecast step pushes it through the dynamics:

$$p(\vect{x}_t \mid \vect{y}_{1:t-1}) = \int p(\vect{x}_t \mid \vect{x}_{t-1}) \, p(\vect{x}_{t-1} \mid \vect{y}_{1:t-1}) \, \mathrm{d}\vect{x}_{t-1}.$$ (eq-problem-predict)

The analysis step conditions on the new observation by Bayes' theorem:

$$p(\vect{x}_t \mid \vect{y}_{1:t}) = \frac{p(\vect{y}_t \mid \vect{x}_t) \, p(\vect{x}_t \mid \vect{y}_{1:t-1})}{p(\vect{y}_t \mid \vect{y}_{1:t-1})}.$$ (eq-problem-update)

The denominator is the evidence of this window, and it is worth keeping.
Multiplying it across windows gives the marginal likelihood of the whole record:

$$\log p(\vect{y}_{1:T}) = \sum_{t=1}^{T} \log p(\vect{y}_t \mid \vect{y}_{1:t-1}).$$ (eq-problem-evidence)

That sum is what a filter can report about how well the model explains the data, and [chapter 14](14_differentiable.md) trains on it.

The recursion is exact and also out of reach.
A density on a grid with 100 points per axis needs $100^{D_x}$ values, which is already $10^6$ at $D_x = 3$.
Every practical filter approximates this recursion, and filters differ in what they keep.

## Closing the recursion: the Kalman filter

Suppose $f$ and $h$ are linear, $f(\vect{x}) = \mat{F} \vect{x}$ and $h(\vect{x}) = \mat{H} \vect{x}$, and the initial density is Gaussian.
Then every density in [](#eq-problem-predict) and [](#eq-problem-update) is Gaussian, and the recursion closes on a mean and a covariance {cite:p}`kalman1960new`.
The forecast step is

$$\gvect{\mu}_{x,t \mid t-1} = \mat{F} \gvect{\mu}_{x,t-1 \mid t-1}, \qquad \gmat{\Sigma}_{x,t \mid t-1} = \mat{F} \gmat{\Sigma}_{x,t-1 \mid t-1} \mat{F}^\top + \gmat{\Sigma}_\eta,$$ (eq-problem-kalman-forecast)

and the analysis step is

$$\gvect{\mu}_{x,t \mid t} = \gvect{\mu}_{x,t \mid t-1} + \mat{G}_t \vect{r}_t, \qquad \gmat{\Sigma}_{x,t \mid t} = (\mat{I} - \mat{G}_t \mat{H}) \gmat{\Sigma}_{x,t \mid t-1},$$ (eq-problem-kalman-analysis)

with the residual $\vect{r}_t = \vect{y}_t - \mat{H} \gvect{\mu}_{x,t \mid t-1} \in \mathbb{R}^{D_y}$, its covariance $\gmat{\Sigma}_{y,t} = \mat{H} \gmat{\Sigma}_{x,t \mid t-1} \mat{H}^\top + \gmat{\Sigma}_\varepsilon$ and the gain $\mat{G}_t = \gmat{\Sigma}_{x,t \mid t-1} \mat{H}^\top \gmat{\Sigma}_{y,t}^{-1} \in \mathbb{R}^{D_x \times D_y}$.
[Chapter 4](04_kalman_update.md) derives the gain.
The evidence of the window is $\mathcal{N}(\vect{r}_t; \vect{0}, \gmat{\Sigma}_{y,t})$.

Two things stop this working at scale.
The covariance has $D_x^2$ entries, which is $10^{12}$ at $D_x = 10^6$.
And the forecast needs $\mat{F}$, a tangent-linear model and its adjoint, which a complex nonlinear $f$ may not have in code at all.

## Ensembles: sampling the moments

The ensemble Kalman filter {cite:p}`evensen1994sequential,evensen2003ensemble` replaces the pair $(\gvect{\mu}_x, \gmat{\Sigma}_x)$ with $N_\mathcal{E}$ samples, the rows of $\mat{X} \in \mathbb{R}^{N_\mathcal{E} \times D_x}$.
The forecast sends each member through the full nonlinear model:

$$\vect{x}^{(e)}_{t \mid t-1} = f\big(\vect{x}^{(e)}_{t-1 \mid t-1}\big) + \gvect{\eta}^{(e)}_t, \qquad \gvect{\eta}^{(e)}_t \sim \mathcal{N}(\vect{0}, \gmat{\Sigma}_\eta).$$ (eq-problem-ensemble-forecast)

The sample mean and sample covariance of the result stand in for [](#eq-problem-kalman-forecast), and [chapter 2](02_ensemble_representation.md) defines them.
The analysis then applies a Kalman update built from those sample statistics, so that no $\mat{F}$, $\mat{H}$ or adjoint is needed.

| Representation | Stores | Forecast needs | Assumes |
| --- | --- | --- | --- |
| Bayes filter on a grid | $n^{D_x}$ values for $n$ points per axis | the transition density | only the state-space model |
| Kalman filter | $D_x^2$ values | $\mat{F}$ and its adjoint | linear maps, Gaussian errors |
| Ensemble Kalman filter | $N_\mathcal{E} D_x$ values | $N_\mathcal{E}$ runs of $f$ | a Gaussian update |

The price is sampling error, of order $1 / \sqrt{N_\mathcal{E}}$, and a covariance of rank at most $N_\mathcal{E} - 1$.
The review of {cite:t}`carrassi2018data` sets this approach among the other methods of the field, and {cite:t}`vetracarvalho2018state` give the stochastic variants in one notation.

## Two families

filterax covers two kinds of problem, with one protocol for each.

A sequential filter tracks a state that moves.
Each window brings new observations, the filter assimilates them once, and the dynamics carry the result to the next window.

An iterative process estimates an unknown that stays still, $\gvect{\theta} \in \mathbb{R}^{D_\theta}$, from one fixed data vector, $\vect{y} = h(\gvect{\theta}) + \gvect{\varepsilon}$ {cite:p}`iglesias2013ensemble`.
There is no time here, so the question is how to apply the data more than once without counting it twice.
The answer is to split the likelihood into tempered factors,

$$p(\gvect{\theta} \mid \vect{y}) \propto p(\gvect{\theta}) \prod_{i=1}^{N_\text{iter}} p(\vect{y} \mid \gvect{\theta})^{\Delta t_i}, \qquad \sum_{i=1}^{N_\text{iter}} \Delta t_i = 1,$$ (eq-problem-tempering)

where each factor is a Gaussian likelihood with the covariance $\gmat{\Sigma}_\varepsilon / \Delta t_i$.
Each iteration is therefore a Kalman update with inflated noise, with $h$ evaluated afresh at the current members:

$$\gvect{\theta}^{(e)} \leftarrow \gvect{\theta}^{(e)} + \hat{\gmat{\Sigma}}_{\theta y} \big(\hat{\gmat{\Sigma}}_h + \Delta t_i^{-1} \gmat{\Sigma}_\varepsilon\big)^{-1} \big(\vect{y} - h(\gvect{\theta}^{(e)})\big).$$ (eq-problem-eki-update)

Here $\hat{\gmat{\Sigma}}_{\theta y} \in \mathbb{R}^{D_\theta \times D_y}$ is the sample cross-covariance between the members and their mapped values, and [chapter 9](09_ensemble_kalman_processes.md) develops the family.
Re-evaluating $h$ is the point of iterating, because it replaces the single statistical linearisation of a filter with one at each step.

| | Sequential filter | Iterative process |
| --- | --- | --- |
| Unknown | a state $\vect{x}_t$ that moves | an unknown $\gvect{\theta}$ that stays still |
| Data | new $\vect{y}_t$ at each window | one $\vect{y}$, tempered over iterations |
| Between updates | the dynamics $f$ | a fresh evaluation of $h$ |
| Protocol | `AbstractSequentialFilter` | `AbstractProcess` |
| One step | `analysis(particles, obs, obs_op, obs_noise)` | `update(state, forward_evals)` |
| Returns | `AnalysisResult` | `ProcessState` |

## Implementation in filterax

The library has three layers, and a problem can enter at any of them.

| Layer | What it holds | Examples |
| --- | --- | --- |
| 0, primitives | pure functions on arrays and operators | `ensemble_covariance`, `kalman_gain`, `perturbed_observations` |
| 1, components | one update step, plus the protocols to extend | `filters.ETKF().analysis`, `processes.EKI().update`, `AbstractObsOperator` |
| 2, models | the whole loop, with histories | `ETKF.assimilate`, `EKI.run` |

The example is a linear system with $D_x = 3$ and $T = 5$ windows, observing components 0 and 2 with noise variance 0.1.
Because it is linear, the Kalman filter is available as a reference.

```{code-block} python
:caption: A linear system, with dynamics and an observation operator written against the protocols.
:label: code-problem-system
import einx
import filterax as flx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

jax.config.update("jax_enable_x64", True)

# Shapes: T = 5 windows; N_E = 20 members; D_x = 3; D_y = 2


# user
class LinearDynamics(flx.AbstractDynamics):
    matrix: jax.Array  # (D_x, D_x)

    # f(x) = F x
    def __call__(
        self, state: Float[Array, " state"], t0: Array, t1: Array
    ) -> Float[Array, " state"]:
        return self.matrix @ state


# user
class LinearObs(flx.AbstractObsOperator):
    matrix: jax.Array  # (D_y, D_x)

    # h(x) = H x
    def __call__(self, state: Float[Array, " state"]) -> Float[Array, " obs"]:
        return self.matrix @ state


# F
transition = jnp.array(  # → (D_x, D_x)
    [[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.1, 0.0, 0.9]]
)
# H selects components 0 and 2
obs_map = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # → (D_y, D_x)
dynamics = LinearDynamics(transition)
obs_op = LinearObs(obs_map)
# today (lineax)
# Σ_ε = 0.1 I
noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))  # (D_y, D_y) operator
# x⁽ᵉ⁾ ~ 𝒩(𝟏, I)
initial = 1.0 + jr.normal(jr.key(0), (20, 3))  # → (N_E, D_x)

# x_t = F x_t−1 and y_t = H x_t + ε_t, from a known start
state = jnp.array([1.0, 2.0, 0.0])  # → (D_x,)
observations = []
for key in jr.split(jr.key(1), 5):
    state = transition @ state  # → (D_x,)
    draw = jnp.sqrt(0.1) * jr.normal(key, (2,))  # → (D_y,)
    observations.append(obs_map @ state + draw)
```

Layer 1 owns one analysis, so the forecast in between is the caller's.
Layer 2 runs the same loop and keeps the histories.
It calls `dynamics(state, t0, t1)` on each member and adds no model noise of its own.

```{code-block} python
:caption: The forecast-analysis cycle at Layer 1 and at Layer 2, which agree exactly.
:label: code-problem-cycle
# today (filterax)
analysis_step = flx.filters.ETKF()


# user
# x⁽ᵉ⁾_t|t−1 = f(x⁽ᵉ⁾_t−1|t−1)
def forecast(members: Float[Array, "member state"]) -> Float[Array, "member state"]:
    return jax.vmap(lambda member: dynamics(member, 0.0, 1.0))(members)


particles = initial  # → (N_E, D_x)
log_evidence = []
for observation in observations:
    predicted = forecast(particles)  # → (N_E, D_x)
    # today (filterax)
    # X_t|t and log 𝒩(r̄_t; 𝟎, Σ̂_y,t)
    result = analysis_step.analysis(predicted, observation, obs_op, noise)
    particles = result.particles  # → (N_E, D_x)
    log_evidence.append(result.log_likelihood)

# today (filterax)
model = flx.ETKF(dynamics=dynamics, obs_op=obs_op)
windows = [(y, float(t + 1)) for t, y in enumerate(observations)]
# today (filterax)
run = model.assimilate(initial, windows, noise)
print(run.analysis_history.shape, run.log_likelihoods.shape)
print(jnp.abs(run.particles - particles).max())
```

It prints `(5, 20, 3) (5,)` and `0.0`.
So `analysis_history` stacks the windows, and the two layers compute the same numbers.

## Agreement with the Kalman filter

With no model noise and a linear $f$, the ensemble moments follow [](#eq-problem-kalman-forecast) exactly, because $\mat{F} \tilde{\mat{X}}^\top$ has the sample covariance $\mat{F} \hat{\gmat{\Sigma}}_x \mat{F}^\top$.
The ETKF analysis matches the Kalman update on the sample covariance ([chapter 6](06_etkf.md)).
By induction, the ensemble mean and sample covariance must then equal a Kalman filter started from the sample moments of `initial`, at every window.

```{code-block} python
:caption: The ETKF cycle against a dense Kalman filter started from the same sample moments, in float64.
:label: code-problem-kalman
# user
# The dense Kalman filter, in numpy, with Σ_η = q I
def kalman_by_hand(mean, covariance, transition, obs_map, noise_cov, observations, q):
    log_evidence = []
    for y in observations:
        mean = transition @ mean
        covariance = transition @ covariance @ transition.T + q * np.eye(len(mean))
        residual = y - obs_map @ mean
        obs_covariance = obs_map @ covariance @ obs_map.T + noise_cov
        cross = covariance @ obs_map.T
        gain = np.linalg.solve(obs_covariance, cross.T).T
        mean = mean + gain @ residual
        covariance = covariance - gain @ cross.T
        whitened = np.linalg.solve(obs_covariance, residual)
        _, logdet = np.linalg.slogdet(obs_covariance)
        constant = len(y) * np.log(2.0 * np.pi)
        log_evidence.append(-0.5 * (constant + logdet + residual @ whitened))
    return mean, covariance, np.array(log_evidence)


# today (filterax)
# Σ̂_x at the start
start_covariance = flx.ensemble_covariance(initial).as_matrix()  # → (D_x, D_x)
# today (filterax)
# x̄ at the start
start_mean = flx.ensemble_mean(initial)  # → (D_x,)
kalman_mean, kalman_covariance, kalman_evidence = kalman_by_hand(
    np.asarray(start_mean),
    np.asarray(start_covariance),
    np.asarray(transition),
    np.asarray(obs_map),
    0.1 * np.eye(2),
    [np.asarray(y) for y in observations],
    0.0,
)
# today (filterax)
final_mean = flx.ensemble_mean(run.particles)  # → (D_x,)
# today (filterax)
final_covariance = flx.ensemble_covariance(run.particles).as_matrix()  # → (D_x, D_x)
mean_error = jnp.abs(final_mean - kalman_mean).max()
covariance_error = jnp.abs(final_covariance - kalman_covariance).max()
evidence_error = jnp.abs(run.log_likelihoods - kalman_evidence).max()
print(f"{mean_error:.0e} {covariance_error:.0e} {evidence_error:.0e}")
```

It prints `1e-16 8e-17 7e-16`, which is float64 round-off in the mean, the covariance and the sum [](#eq-problem-evidence).

Model noise changes this, because then [](#eq-problem-ensemble-forecast) only matches [](#eq-problem-kalman-forecast) in expectation.
[](#code-problem-sampling) adds $\gmat{\Sigma}_\eta = 0.05 \mat{I}$ to the same system, and measures the analysis mean against the Kalman mean over 20 independent runs.

```{code-block} python
:caption: With model noise the ensemble mean approaches the Kalman mean at the Monte Carlo rate.
:label: code-problem-sampling
# Shapes: R = 20 runs; T = 5 windows; N_E members; D_x = 3


# user
# One run of the noisy ensemble forecast with the ETKF analysis
def noisy_run(key: PRNGKeyArray, num_members: int) -> Float[Array, " state"]:
    start_key, *noise_keys = jr.split(key, 6)
    # x⁽ᵉ⁾ ~ 𝒩(𝟏, I)
    members = 1.0 + jr.normal(start_key, (num_members, 3))  # → (N_E, D_x)
    for observation, noise_key in zip(observations, noise_keys):
        # ηₜ⁽ᵉ⁾ ~ 𝒩(𝟎, 0.05 I)
        model_noise = jnp.sqrt(0.05) * jr.normal(noise_key, members.shape)
        predicted = forecast(members) + model_noise  # → (N_E, D_x)
        # today (filterax)
        # X_t|t
        members = analysis_step.analysis(
            predicted, observation, obs_op, noise
        ).particles
    # today (filterax)
    # x̄_T|T
    return flx.ensemble_mean(members)  # → (D_x,)


# μ_T|T with Σ_η = 0.05 I, from 𝒩(𝟏, I)
kalman_noisy, _, _ = kalman_by_hand(
    np.ones(3),
    np.eye(3),
    np.asarray(transition),
    np.asarray(obs_map),
    0.1 * np.eye(2),
    [np.asarray(y) for y in observations],
    0.05,
)

keys = jr.split(jr.key(2), 20)  # → (R,)
for num_members in (20, 200, 2000):
    means = jax.vmap(lambda key: noisy_run(key, num_members))(keys)  # → (R, D_x)
    # RMS of x̄_T|T − μ_T|T
    rmse = jnp.sqrt(jnp.mean((means - kalman_noisy) ** 2))  # → ()
    print(f"N_E = {num_members}: RMSE {rmse:.4f}, x √N_E {rmse * num_members**0.5:.3f}")
```

It prints `N_E = 20: RMSE 0.1345, x √N_E 0.601`, `N_E = 200: RMSE 0.0386, x √N_E 0.545` and `N_E = 2000: RMSE 0.0146, x √N_E 0.652`.
The scaled column is nearly constant, which is the $1 / \sqrt{N_\mathcal{E}}$ rate.

## The iterative family in filterax

`processes.EKI` follows `AbstractProcess`, whose `init` builds a state and whose `update` takes forward evaluations made by the caller.
With a single step at $\Delta t = 1$, [](#eq-problem-eki-update) averages to the Kalman mean update, so the mean must match a filter on the same ensemble.
The spread need not match.
Every member sees the same $\vect{y}$, so the update misses the term $\mat{G} \gmat{\Sigma}_\varepsilon \mat{G}^\top$ that [chapter 4](04_kalman_update.md) derives, and the spread collapses faster than the posterior.

```{code-block} python
:caption: One EKI step against one ETKF analysis, and then the Layer 2 run loop, on a linear inverse problem.
:label: code-problem-eki
# Shapes: N_E = 40 members; D_θ = 2 parameters; D_y = 3 observations

# A in h(θ) = A θ
forward_matrix = jnp.array([[1.0, 0.5], [-0.3, 1.2], [0.4, -0.7]])  # → (D_y, D_θ)
# y = A θ_true + ε
data = forward_matrix @ jnp.array([0.7, -0.4]) + jnp.array([0.02, -0.01, 0.015])
# today (lineax)
# Σ_ε = 0.01 I
data_noise = lx.DiagonalLinearOperator(0.01 * jnp.ones(3))  # (D_y, D_y) operator
# θ⁽ᵉ⁾ ~ 𝒩(𝟎, I)
prior = jr.normal(jr.key(3), (40, 2))  # → (N_E, D_θ)


# user
# h(θ) = A θ
def forward(theta: Float[Array, " param"]) -> Float[Array, " obs"]:
    return forward_matrix @ theta


# today (filterax)
process = flx.processes.EKI(scheduler=flx.FixedScheduler(dt=1.0))
# today (filterax)
process_state = process.init(prior, data, data_noise)
# h(θ⁽ᵉ⁾)
evaluations = jax.vmap(forward)(process_state.particles)  # → (N_E, D_y)
# today (filterax)
# one EKI update with Δt = 1
stepped = process.update(process_state, evaluations).particles  # → (N_E, D_θ)
# today (filterax)
filtered = analysis_step.analysis(prior, data, forward, data_noise).particles
gap = jnp.abs(flx.ensemble_mean(stepped) - flx.ensemble_mean(filtered)).max()
# today (filterax)
# √(diag Σ̂_θ) after each
stepped_spread = jnp.sqrt(jnp.diag(flx.cross_covariance(stepped, stepped)))
# today (filterax)
filtered_spread = jnp.sqrt(jnp.diag(flx.cross_covariance(filtered, filtered)))
print(f"{gap:.0e}", stepped_spread.round(4), filtered_spread.round(4))
# today (filterax)
print(flx.ensemble_mean(filtered).round(3))

# today (filterax)
# The full loop, with the default adaptive step size
inversion = flx.EKI(forward_fn=forward, obs=data, noise_cov=data_noise).run(prior)
print(inversion.n_iterations, inversion.converged, inversion.mean.round(3))
```

It prints `6e-17 [0.0072 0.0046] [0.0895 0.0678]`, then `[ 0.719 -0.403]`, then `4 True [ 0.694 -0.392]`.
The means agree to round-off, while the EKI spread is under a tenth of the filter's.
The Layer 2 loop stops when the accumulated $\Delta t_i$ reaches 1, as [](#eq-problem-tempering) requires.
Its mean, `[ 0.694 -0.392]`, is close to the Kalman mean `[ 0.719 -0.403]` but not equal to it, so EKI is an optimiser here and not a sampler.
`EKS` is the process to use for posterior samples ([chapter 9](09_ensemble_kalman_processes.md)).

## When it breaks

Two approximations separate an ensemble Kalman filter from the Bayes filter, and they fail differently.

The first is sampling.
A small ensemble in a large state has spurious correlations and too little spread, and the filter can stop listening to the data.
[Chapter 2](02_ensemble_representation.md) measures the sampling error, and [chapter 6](06_etkf.md) shows the resulting divergence on Lorenz-96.
Localization ([chapter 11](11_localization.md)) and inflation ([chapter 12](12_inflation.md)) exist to treat it.

The second is the Gaussian update, and more members do not fix it.
[](#code-problem-closure) takes a two-component prior, $p(x) = \frac{1}{2} \mathcal{N}(x; -2, 0.25) + \frac{1}{2} \mathcal{N}(x; 2, 0.25)$, with $y = x + \varepsilon$, $y = 0.5$ and unit noise variance.
The exact posterior comes from quadrature on a grid, which is affordable in one dimension.

```{code-block} python
:caption: A bimodal prior. The ensemble update converges to the wrong mean as the ensemble grows.
:label: code-problem-closure
# Shapes: N_q = 20,001 grid points; N_E members; D_x = D_y = 1

# Bayes' theorem by quadrature, in numpy
grid = np.linspace(-6.0, 6.0, 20001)
left = np.exp(-0.5 * ((grid + 2.0) / 0.5) ** 2)
right = np.exp(-0.5 * ((grid - 2.0) / 0.5) ** 2)
weights = (left + right) * np.exp(-0.5 * (0.5 - grid) ** 2)
weights = weights / weights.sum()
exact_mean = (weights * grid).sum()
exact_variance = (weights * (grid - exact_mean) ** 2).sum()
print(f"exact: mean {exact_mean:.3f}, variance {exact_variance:.3f}")

# today (lineax)
# Σ_ε = 1
unit_noise = lx.DiagonalLinearOperator(jnp.ones(1))  # (D_y, D_y) operator
for num_members in (100, 1000, 10000, 100000):
    component_key, draw_key = jr.split(jr.key(5))
    # the mode of each member, ±2
    modes = 4.0 * jr.bernoulli(component_key, 0.5, (num_members, 1)) - 2.0
    # x⁽ᵉ⁾ ~ p(x)
    members = modes + 0.5 * jr.normal(draw_key, (num_members, 1))  # → (N_E, 1)
    # today (filterax)
    # X_t|t
    analysed = analysis_step.analysis(
        members, jnp.array([0.5]), lambda x: x, unit_noise
    ).particles
    # today (filterax)
    mean = flx.ensemble_mean(analysed)[0]  # → ()
    # today (filterax)
    anomalies = flx.ensemble_anomalies(analysed)  # → (N_E, 1)
    variance = einx.sum("[member] state", anomalies**2)[0] / (num_members - 1.0)
    print(f"N_E = {num_members}: mean {mean:.3f}, variance {variance:.3f}")
```

It prints `exact: mean 1.162, variance 1.631`, then:

| $N_\mathcal{E}$ | Analysis mean | Analysis variance |
| --- | --- | --- |
| 100 | `0.426` | `0.808` |
| 1000 | `0.418` | `0.811` |
| 10000 | `0.403` | `0.810` |
| 100000 | `0.404` | `0.810` |

The answer settles quickly and stays wrong, in both the mean and the variance.
The update sees only the first two moments of the prior, so it cannot know that the probability sits in two lobes.
This is an error of the closure and not of sampling.
Nonlinear observation operators fail in a related way, which [chapter 3](03_observation_model.md) measures.

## When an ensemble Kalman method is the right answer

Use a sequential filter when the state is large, $f$ is expensive and has no adjoint, and the forecast density is roughly unimodal.
Use an iterative process when the unknown is static and $h$ is cheap enough to run many times, as in calibration.

Reach for something else in three cases.
A small linear-Gaussian problem has an exact answer, in `SquareRootKF` and `KalmanSmoother` ([chapter 17](17_exact_references.md)).
A strongly multimodal density needs a method that does not close on two moments, and filterax does not offer one.
And when $f$ does have an adjoint, a variational method is a real competitor.

## See also

- [Chapter 2](02_ensemble_representation.md): the ensemble as a low-rank covariance, and its sampling error
- [Chapter 3](03_observation_model.md): observation operators and structured noise
- [Chapter 4](04_kalman_update.md): the gain, its derivation and its two solve routes
- [Chapter 9](09_ensemble_kalman_processes.md): the iterative process family
- [Chapter 16](16_assimilation_cycle.md): the assimilation cycle in full
- [API: protocols](https://jejjohnson.github.io/filterax/reference/protocols/), [filters](https://jejjohnson.github.io/filterax/reference/filters/) and [processes](https://jejjohnson.github.io/filterax/reference/processes/)

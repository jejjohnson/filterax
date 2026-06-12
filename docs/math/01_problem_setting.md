# Problem Setting

Sequential data assimilation is the problem of tracking the state of a
dynamical system through time, given a stream of partial, noisy
observations and a model of the dynamics. Unlike the batch (variational)
formulation, the sequential problem never looks at all the data at once:
each new observation updates the current belief about the state, and that
updated belief is propagated forward to meet the next observation. The
ensemble Kalman filter family — the subject of these chapters — is a
Monte Carlo solution to this problem that scales to state dimensions
where exact Bayesian filtering is hopeless.

## The state-space model

Let $x_t \in \mathbb{R}^{N_x}$ denote the state at time $t$ and
$y_t \in \mathbb{R}^{N_y}$ the observations. Two equations define the
problem:

$$
x_t = \mathcal{M}(x_{t-1}) + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, Q),
$$

$$
y_t = \mathcal{H}(x_t) + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, R).
$$

The dynamics $\mathcal{M}$ may be a discretised PDE, an ODE integrator,
or a learned emulator; the observation operator $\mathcal{H}$ maps the
state into observation space (chapter 3). Model error $\eta_t$ and
observation error $\varepsilon_t$ are assumed independent of each other
and across time.

## The Bayes filter

The exact solution alternates two steps. Given the posterior (the
*analysis*) at time $t-1$, the **forecast step** pushes it through the
dynamics via the Chapman–Kolmogorov equation:

$$
p(x_t \mid y_{1:t-1}) = \int p(x_t \mid x_{t-1})\, p(x_{t-1} \mid y_{1:t-1})\, \mathrm{d}x_{t-1}.
$$

The **analysis step** conditions on the new observation via Bayes'
theorem:

$$
p(x_t \mid y_{1:t}) \propto p(y_t \mid x_t)\, p(x_t \mid y_{1:t-1}).
$$

This recursion is exact but intractable in general: the densities live
in $\mathbb{R}^{N_x}$, and for a weather or ocean model $N_x$ is
$10^6$–$10^9$. Every practical filter is an approximation to this
recursion; the approximations differ in what they keep.

## The Gaussian approximation

If the dynamics and observation operator are linear and all errors are
Gaussian, both steps preserve Gaussianity and the recursion closes on a
mean and covariance — the classical Kalman filter (Kalman 1960). Writing
the forecast (prior) moments as $x^f, P^f$ and the analysis (posterior)
moments as $x^a, P^a$:

$$
\begin{aligned}
\text{forecast:} &\quad x^f_t = M\, x^a_{t-1}, \qquad P^f_t = M P^a_{t-1} M^\top + Q, \\
\text{analysis:} &\quad x^a_t = x^f_t + K_t\, d_t, \qquad P^a_t = (I - K_t H)\, P^f_t,
\end{aligned}
$$

with innovation $d_t = y_t - H x^f_t$ and Kalman gain
$K_t = P^f_t H^\top (H P^f_t H^\top + R)^{-1}$ (derived in chapter 4).

Two things break at scale. Storing and propagating $P \in
\mathbb{R}^{N_x \times N_x}$ costs $O(N_x^2)$ memory and $O(N_x^3)$
work — out of the question for large $N_x$. And the covariance
propagation $M P M^\top$ requires the tangent-linear model $M$ and its
adjoint, which for a complex nonlinear $\mathcal{M}$ may not exist in
code at all.

## Why ensembles

The ensemble Kalman filter (Evensen 1994) replaces the explicit
$(x, P)$ pair with $N_e$ samples — an ensemble matrix
$X \in \mathbb{R}^{N_e \times N_x}$ whose rows are state vectors.
The forecast step becomes embarrassingly simple: propagate each member
through the *full nonlinear* model,

$$
x^{f,(j)}_t = \mathcal{M}\big(x^{a,(j)}_{t-1}\big) + \eta^{(j)}_t,
\qquad j = 1, \dots, N_e,
$$

and the forecast mean and covariance are estimated as sample moments of
the propagated ensemble (chapter 2). This sidesteps both failures at
once:

- **Second moments by Monte Carlo.** The covariance is never stored;
  it is implicit in $N_e$ state vectors, at $O(N_e N_x)$ memory. With
  $N_e \sim 20$–$100$, this is a few model states, not a matrix.
- **No tangent-linear model.** The nonlinear $\mathcal{M}$ is used
  as-is, $N_e$ times. No adjoint, no linearisation, no extra code. The
  same trick handles nonlinear $\mathcal{H}$ in the analysis step
  (chapter 3).

The price is sampling error: all second moments are estimated from
$N_e$ samples, with errors of order $1/\sqrt{N_e}$, and the sample
covariance has rank at most $N_e - 1$. Chapter 2 makes this precise.

## The forecast–analysis cycle

Operationally, the filter is a loop:

```
X ← initial ensemble                      # (N_e, N_x)
for each observation window t:
    X ← M applied to each row of X        # forecast (caller's model)
    X ← analysis(X, y_t, H, R)            # Bayes update (filterax)
```

The analysis step is where the ensemble Kalman filters differ:
the stochastic EnKF perturbs the observations (chapter 5), the
deterministic square-root filters (ETKF, EnSRF — chapters 6–7) transform
the anomalies exactly. All of them consume the same inputs — forecast
ensemble, observation, $\mathcal{H}$, $R$ — and produce a posterior
ensemble.

## How filters fail

A finite ensemble in a chaotic system fails in characteristic ways, and
the failure modes drive the design of every practical EnKF system:

- **Spurious correlations.** With $N_e \ll N_x$, the sample covariance
  contains $O(1/\sqrt{N_e})$ noise in every entry, including between
  state components that are physically uncorrelated. A distant
  observation then produces a confident, wrong increment. The cure is
  *localization* — tapering covariances by distance — covered in
  chapter 11.
- **Variance underestimation and filter divergence.** Sampling error,
  rank deficiency, and unmodelled error sources all bias the ensemble
  spread low. An overconfident filter weights observations too little,
  drifts from the truth, and eventually ignores the data entirely —
  *filter divergence*. The cure is *inflation* — deliberately boosting
  the spread — covered in chapter 12.

Neither pathology appears in the equations of this chapter; both are
consequences of $N_e$ being finite. Keep them in mind as the chapters
proceed: the clean theory of chapters 2–5 is exact only as
$N_e \to \infty$.

## Where filterax sits

filterax implements **analysis steps**: pure functions and lightweight
`equinox` modules that take a forecast ensemble, an observation vector,
an observation operator, and an observation-noise operator, and return
the posterior ensemble. The forecast step — running $\mathcal{M}$ — is
deliberately the caller's job: filterax never owns your dynamics. You
can `jax.vmap` your own model over the ensemble, plug in a `diffrax`
integrator, or use a neural emulator; the analysis step does not care.

For multi-window experiments, the cycling itself (forecast → analyse →
inflate, with history bookkeeping) is composed via the Layer-2 models
(`filterax.StochasticEnKF`, `filterax.ETKF`, …) or, in larger pipelines,
via `pipekit` cycles — chapter 16. Everything is JAX-native: every
analysis step is `jit`-able, `vmap`-able, and differentiable.

## Implementation in filterax

A minimal forecast–analysis cycle with caller-owned dynamics:

```python
import jax
import jax.numpy as jnp
import lineax as lx
from filterax.filters import StochasticEnKF

key = jax.random.key(0)
N_e, N_x = 20, 3

def forecast(x):                     # M — the caller's model, used as-is
    return 0.95 * x + 0.05 * jnp.roll(x, 1)

def obs_op(x):                       # H — observe the first two components
    return x[:2]

R = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
filt = StochasticEnKF(key=jax.random.key(1))

X = jax.random.normal(key, (N_e, N_x))          # initial ensemble
observations = [jnp.array([0.5, -0.2]), jnp.array([0.4, -0.1])]
for t, y in enumerate(observations):
    X = jax.vmap(forecast)(X)                   # forecast — caller's job
    result = filt.analysis(X, y, obs_op, R, key=jax.random.fold_in(key, t))
    X = result.particles                        # analysis — filterax's job
```

The `analysis` call is the entire interface. Each chapter that follows
unpacks one piece of what happens inside it.

## Where next

- [Chapter 2 — Ensemble representation](02_ensemble_representation.md):
  the $(N_e, N_x)$ matrix, anomalies, and the low-rank covariance.
- [Chapter 3 — Observation model](03_observation_model.md): $\mathcal{H}$,
  structured $R$, and perturbed observations.
- [Chapter 4 — Kalman update](04_kalman_update.md): the BLUE gain and
  its ensemble estimator.
- [Chapter 5 — Stochastic EnKF](05_stochastic_enkf.md): the first
  complete filter.
- [Filters API](../api/filters.md) — the Layer-1 analysis steps and
  Layer-2 cycling models.
- [pipekit integration](../api/pipekit.md) — cycling in larger pipelines.

## References

- Kalman, R. E. (1960). *A new approach to linear filtering and
  prediction problems.* J. Basic Eng. 82(1).
- Evensen, G. (1994). *Sequential data assimilation with a nonlinear
  quasi-geostrophic model using Monte Carlo methods to forecast error
  statistics.* JGR Oceans 99(C5).
- Evensen, G. (2003). *The Ensemble Kalman Filter: theoretical
  formulation and practical implementation.* Ocean Dynamics 53.
- Carrassi, A., Bocquet, M., Bertino, L., & Evensen, G. (2018). *Data
  assimilation in the geosciences: An overview of methods, issues, and
  perspectives.* WIREs Climate Change 9(5).
- Vetra-Carvalho, S., et al. (2018). *State-of-the-art stochastic data
  assimilation methods for high-dimensional non-Gaussian problems.*
  Tellus A 70(1).

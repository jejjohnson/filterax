---
numbering:
  enumerator: "10.%s"
---

# Ensemble smoothers: EnKS, EnsembleRTS, fixed-lag, square-root and IES

A filter is causal: its analysis at time $t$ uses the observations up to $t$ and no later ones.
For a reanalysis, or any retrospective study, the object you want also uses the observations after $t$:

$$p(\vect{x}_t \mid \vect{y}_{1:T}), \qquad t < T.$$ (eq-smoothers-density)

An ensemble smoother gets there cheaply.
It runs the filter once, stores the ensembles, and sweeps backward, recombining them with no adjoint and no extra runs of the model.

One regression organises the chapter.
The backward step regresses the present on the future, using the members themselves as samples of the joint distribution.
EnKS and EnsembleRTS are that step, FixedLagSmoother truncates it, and EnsembleSqrtSmoother keeps its mean and replaces its anomalies.
The same two things make the regression fail in each case: too few members to estimate it, or a forecast that treats a wrong model as exact.
IES closes the chapter, and it is a different method for a different problem, an iteration over one window with no backward pass.

## The backward step

### The linear-Gaussian recursion

Take $\vect{x}_{t+1} = \mat{F} \vect{x}_t + \gvect{\eta}_t$ with $\vect{x}_t \in \mathbb{R}^{D_x}$, $\mat{F} \in \mathbb{R}^{D_x \times D_x}$ and $\gvect{\eta}_t \sim \mathcal{N}(\vect{0}, \gmat{\Sigma}_\eta)$.
Given $\vect{y}_{1:t}$, the pair $(\vect{x}_t, \vect{x}_{t+1})$ is jointly Gaussian, with cross-covariance $\gmat{\Sigma}_{x,t \mid t} \mat{F}^\top$.
Conditioning on $\vect{x}_{t+1}$ gives the mean $\gvect{\mu}_{x,t \mid t} + \mat{G}_{\text{smooth},t} (\vect{x}_{t+1} - \gvect{\mu}_{x,t+1 \mid t})$, with the smoother gain

$$\mat{G}_{\text{smooth},t} = \gmat{\Sigma}_{x,t \mid t} \mat{F}^\top \gmat{\Sigma}_{x,t+1 \mid t}^{-1} \in \mathbb{R}^{D_x \times D_x}.$$ (eq-smoothers-gain)

Given $\vect{x}_{t+1}$, the later observations say nothing more about $\vect{x}_t$, by the Markov property.
Averaging over $\vect{x}_{t+1} \mid \vect{y}_{1:T}$ then gives the Rauch-Tung-Striebel recursion {cite:p}`rauch1965maximum`, which starts from the filter at $t = T$:

$$\gvect{\mu}_{x,t \mid T} = \gvect{\mu}_{x,t \mid t} + \mat{G}_{\text{smooth},t} \left(\gvect{\mu}_{x,t+1 \mid T} - \gvect{\mu}_{x,t+1 \mid t}\right),$$ (eq-smoothers-rts-mean)

$$\gmat{\Sigma}_{x,t \mid T} = \gmat{\Sigma}_{x,t \mid t} + \mat{G}_{\text{smooth},t} \left(\gmat{\Sigma}_{x,t+1 \mid T} - \gmat{\Sigma}_{x,t+1 \mid t}\right) \mat{G}_{\text{smooth},t}^\top.$$ (eq-smoothers-rts-covariance)

The smoothed covariance never exceeds the filtered one, since $\gmat{\Sigma}_{x,T \mid T} \le \gmat{\Sigma}_{x,T \mid T-1}$ starts the recursion and the difference stays negative semi-definite.
{cite:t}`cosme2012smoothing` set out this family of linear-Gaussian solutions.

### The same step on members

The filter stores two ensembles for each window, the analysis $\mat{X}_{t \mid t}$ and the next forecast $\mat{X}_{t+1 \mid t}$, each in $\mathbb{R}^{N_\mathcal{E} \times D_x}$.
Member $e$ of the forecast was propagated from member $e$ of the analysis, so the two rows together are one sample of $(\vect{x}_t, \vect{x}_{t+1})$.
That correspondence is all the method relies on.
Their sample cross-covariance estimates $\gmat{\Sigma}_{x,t \mid t} \mat{F}^\top$ with no $\mat{F}$ in sight, so the dynamics can be nonlinear and need no tangent.
Write the anomalies as $\tilde{\mat{X}}_{t \mid t}$ and $\tilde{\mat{X}}_{t+1 \mid t}$, and take the pseudo-inverse, because the forecast covariance has rank at most $N_\mathcal{E} - 1$:

$$\hat{\mat{G}}_{\text{smooth},t} = \tilde{\mat{X}}_{t \mid t}^\top \tilde{\mat{X}}_{t+1 \mid t} \left(\tilde{\mat{X}}_{t+1 \mid t}^\top \tilde{\mat{X}}_{t+1 \mid t}\right)^{+}.$$ (eq-smoothers-sample-gain)

Applying this to each member gives $\vect{x}_{t \mid T}^{(e)} = \vect{x}_{t \mid t}^{(e)} + \hat{\mat{G}}_{\text{smooth},t} (\vect{x}_{t+1 \mid T}^{(e)} - \vect{x}_{t+1 \mid t}^{(e)})$ {cite:p}`evensen2000ensemble`.
Stack the rows, and write $\mat{D}_{t+1} = \mat{X}_{t+1 \mid T} - \mat{X}_{t+1 \mid t} \in \mathbb{R}^{N_\mathcal{E} \times D_x}$ for what smoothing changed about the future.
The identity $(\mat{A}^\top \mat{A})^{+} \mat{A}^\top = \mat{A}^{+}$ then removes the square matrix altogether:

$$\mat{X}_{t \mid T} = \mat{X}_{t \mid t} + \mat{D}_{t+1} \tilde{\mat{X}}_{t+1 \mid t}^{+} \tilde{\mat{X}}_{t \mid t}.$$ (eq-smoothers-step)

This is `gaussx.ensemble_rts_step`, whose docstring writes it as $X_t^s = X_t^a + D F^{+} A$, with $F$ the forecast anomalies and $A$ the analysis anomalies.
The recursion starts at the final window, where $\mat{X}_{T \mid T}$ is the analysis, and no later data exist to change it.

The product is contracted through the smaller of $N_\mathcal{E}$ and $D_x$, so no larger square matrix appears, and $\tilde{\mat{X}}_{t+1 \mid t}^{+}$ comes from a singular value decomposition.
Singular values at or below a cutoff are dropped, by default $N_\mathcal{E} + \sqrt{D_x}$ times the machine epsilon relative to the largest.
When $N_\mathcal{E} \le D_x$, rounding always leaves one such value, because $\tilde{\mat{X}}_{t+1 \mid t}$ annihilates the constant vector.
A whole backward pass costs $T$ of these decompositions, each $O(N_\mathcal{E} D_x \min(N_\mathcal{E}, D_x))$.
Going through singular values rather than $\tilde{\mat{X}} \tilde{\mat{X}}^\top$ matters in float32, which [](#code-smoothers-float32) measures.

### EnKS and EnsembleRTS are one algorithm

Read [](#eq-smoothers-step) as a Kalman update whose observation is the future, and it is the Ensemble Kalman Smoother.
Read [](#eq-smoothers-sample-gain) as the RTS gain with its $\gmat{\Sigma}_{x,t \mid t} \mat{F}^\top$ estimated from members, and it is an ensemble RTS smoother {cite:p}`evensen2003ensemble`.
filterax ships both names over one function, so that code can say which reading it means.
Both run a reversed `jax.lax.scan`.

### No process noise: the step inverts the dynamics

Suppose the forecast carries no noise, so that $\mat{X}_{t+1 \mid t} = \mat{X}_{t \mid t} \mat{F}^\top$ with an invertible $\mat{F}$, and suppose $N_\mathcal{E} > D_x$ with a full-rank $\tilde{\mat{X}}_{t \mid t}$.
Then $\tilde{\mat{X}}_{t+1 \mid t} = \tilde{\mat{X}}_{t \mid t} \mat{F}^\top$, so $\tilde{\mat{X}}_{t+1 \mid t}^{+} = \mat{F}^{-\top} \tilde{\mat{X}}_{t \mid t}^{+}$, and $\tilde{\mat{X}}_{t \mid t}^{+} \tilde{\mat{X}}_{t \mid t} = \mat{I}$.
Putting this in [](#eq-smoothers-step), the analysis cancels:

$$\mat{X}_{t \mid T} = \mat{X}_{t \mid t} + \left(\mat{X}_{t+1 \mid T} - \mat{X}_{t \mid t} \mat{F}^\top\right) \mat{F}^{-\top} = \mat{X}_{t+1 \mid T} \mat{F}^{-\top}.$$ (eq-smoothers-inverse)

So the whole smoothed trajectory is the final analysis run backward through $\mat{F}^{-1}$, and nothing else of the filter's history survives.
If the model is exact, that is correct: every state is a known function of the final one, and $\mat{X}_{T \mid T}$ has all the data.
If the truth has process noise, the backward run is wrong in two ways.
It has no way to absorb what the model missed, and a damped $\mat{F}$ has an $\mat{F}^{-1}$ that amplifies the final error at every step.
[](#code-smoothers-no-noise) shows the identity and both outcomes.
The exact smoother with $\gmat{\Sigma}_\eta = \mat{0}$ reduces to the same thing through [](#eq-smoothers-gain), and [chapter 17](17_exact_references.md) covers `KalmanSmoother`.

## The square-root smoother

`EnsembleSqrtSmoother` is the backward analogue of the deterministic filters of chapters 6 and 7 {cite:p}`tippett2003ensemble`.
It takes the mean from [](#eq-smoothers-rts-mean) with the sample gain, and it replaces the member-wise anomalies with a symmetric transform of the analysis anomalies.
Take $\mat{V} = (\tilde{\mat{X}}_{t+1 \mid t} \tilde{\mat{X}}_{t+1 \mid t}^\top)^{+} \tilde{\mat{X}}_{t+1 \mid t} \in \mathbb{R}^{N_\mathcal{E} \times D_x}$, so that $\hat{\mat{G}}_{\text{smooth},t} = \tilde{\mat{X}}_{t \mid t}^\top \mat{V}$, and define

$$\gmat{\Psi} = \mat{I} + \mat{V} \left(\tilde{\mat{X}}_{t+1 \mid T}^\top \tilde{\mat{X}}_{t+1 \mid T} - \tilde{\mat{X}}_{t+1 \mid t}^\top \tilde{\mat{X}}_{t+1 \mid t}\right) \mat{V}^\top, \qquad \tilde{\mat{X}}_{t \mid T} = \gmat{\Psi}^{1/2} \tilde{\mat{X}}_{t \mid t},$$ (eq-smoothers-sqrt)

with $\gmat{\Psi} \in \mathbb{R}^{N_\mathcal{E} \times N_\mathcal{E}}$ and negative eigenvalues clamped to zero.
The sample covariance of the result is $\tilde{\mat{X}}_{t \mid t}^\top \gmat{\Psi} \tilde{\mat{X}}_{t \mid t} / (N_\mathcal{E} - 1)$, and substituting gives exactly [](#eq-smoothers-rts-covariance) on sample covariances.
The constant vector is in the null space of $\mat{V}^\top$, so $\gmat{\Psi}^{1/2} \vect{1} = \vect{1}$, and the anomalies stay centred, by the argument of [](#eq-etkf-mean-preserving).

### What differs from EnKS

The means agree, since both move by $\hat{\mat{G}}_{\text{smooth},t}$ times the mean of $\mat{D}_{t+1}$.
The anomalies need more care than "a rotation".
Split the analysis anomalies into their regression on the forecast and a residual, $\tilde{\mat{X}}_{t \mid t} = \tilde{\mat{X}}_{t+1 \mid t} \hat{\mat{G}}_{\text{smooth},t}^\top + \mat{R}_t$, with $\tilde{\mat{X}}_{t+1 \mid t}^\top \mat{R}_t = \mat{0}$.
[](#eq-smoothers-step) then gives the EnKS anomalies as

$$\tilde{\mat{X}}_{t \mid T} = \tilde{\mat{X}}_{t+1 \mid T} \hat{\mat{G}}_{\text{smooth},t}^\top + \mat{R}_t.$$ (eq-smoothers-enks-anomalies)

Their sample covariance is [](#eq-smoothers-rts-covariance) plus the cross terms between $\tilde{\mat{X}}_{t+1 \mid T}$ and $\mat{R}_t$.
With $N_\mathcal{E} - 1 \le D_x$ and a forecast of full rank, $\mat{R}_t = \mat{0}$, the cross terms vanish, and the two smoothers differ only by a rotation, as the docstring says.
With $N_\mathcal{E} - 1 > D_x$ they do not vanish in general, and the covariances differ.
[](#code-smoothers-sqrt) measures both regimes.

## Fixed-lag smoothing

The full pass lets arbitrarily old states feel arbitrarily new data.
`FixedLagSmoother(lag=L)` restarts [](#eq-smoothers-step) for each anchor $s$, from the analysis at $\min(T, s + L)$ back to $s$, and keeps only the result at $s$.
With $L = 0$ it is the filter, and with $L \ge T - 1$ it is EnKS.
The batch sweep costs $T L$ backward steps.
The class offers that batch form only: in a streaming setting you hold the last $L + 1$ ensembles yourself and call `smooth` on that window.

## IES: Gauss-Newton on one window

The iterative ensemble smoother {cite:p}`chen2013levenberg,evensen2019efficient` has no backward pass and no time index.
It treats one window as an inverse problem, $\vect{y} = h(\gvect{\theta}) + \gvect{\varepsilon}$, where $h$ maps parameters, or an initial state, to the whole observation record.
The notation is that of [chapter 9](09_ensemble_kalman_processes.md): $\gmat{\Theta}^{(i)} \in \mathbb{R}^{N_\mathcal{E} \times D_\theta}$ is the iterate, and $\tilde{\mat{H}}^{(i)}$ holds the anomalies of its image.

Each member draws one perturbed observation, $\vect{y}^{(e)} = \vect{y} + \gvect{\varepsilon}^{(e)}$, and keeps it throughout, so that it minimises one fixed objective,

$$\mathcal{J}^{(e)}(\gvect{\theta}) = \frac{1}{2} \Vert \vect{y}^{(e)} - h(\gvect{\theta}) \Vert^2_{\gmat{\Sigma}_\varepsilon^{-1}} + \frac{1}{2} \Vert \gvect{\theta} - \gvect{\theta}^{(e,0)} \Vert^2_{(\hat{\gmat{\Sigma}}_\theta^{(0)})^{-1}},$$ (eq-smoothers-ies-cost)

with $\hat{\gmat{\Sigma}}_\theta^{(0)}$ the covariance of the initial ensemble.
Linearising $h$ about the current iterate, with a Jacobian $\hat{\mat{J}}^{(i)} \in \mathbb{R}^{D_y \times D_\theta}$, makes this a linear least-squares problem whose Kalman-form solution is the Gauss-Newton target:

$$\gvect{\theta}^{(e,i+1)} = (1 - \alpha) \gvect{\theta}^{(e,i)} + \alpha \left[\gvect{\theta}^{(e,0)} + \hat{\mat{G}}^{(i)} \left(\vect{y}^{(e)} - h(\gvect{\theta}^{(e,i)}) + \hat{\mat{J}}^{(i)} (\gvect{\theta}^{(e,i)} - \gvect{\theta}^{(e,0)})\right)\right],$$ (eq-smoothers-ies)

$$\hat{\mat{G}}^{(i)} = \hat{\gmat{\Sigma}}_\theta^{(0)} \hat{\mat{J}}^{(i)\top} \left(\hat{\mat{J}}^{(i)} \hat{\gmat{\Sigma}}_\theta^{(0)} \hat{\mat{J}}^{(i)\top} + \gmat{\Sigma}_\varepsilon\right)^{-1}.$$ (eq-smoothers-ies-gain)

$\hat{\mat{J}}^{(i)}$ is the ensemble-average sensitivity, the least-squares fit $\tilde{\mat{H}}^{(i)} = \tilde{\gmat{\Theta}}^{(i)} \hat{\mat{J}}^{(i)\top}$, so $\hat{\mat{J}}^{(i)\top} = (\tilde{\gmat{\Theta}}^{(i)})^{+} \tilde{\mat{H}}^{(i)}$.
Only its products with rows of parameters are needed, so everything goes through the $N_\mathcal{E} \times N_\mathcal{E}$ matrix $\tilde{\gmat{\Theta}}^{(0)} (\tilde{\gmat{\Theta}}^{(i)})^{+}$, and no $D_\theta \times D_\theta$ matrix is formed.
The damping $\alpha \in (0, 1]$ is the `step_size` argument, in the Levenberg-Marquardt style.

The anchor to $\gvect{\theta}^{(e,0)}$ separates this from EKI, which drifts from its current iterate and has no memory of where it began.
For a linear $h$ the term in $\hat{\mat{J}}^{(i)}$ cancels the change in $h(\gvect{\theta}^{(e,i)})$, so the bracket is the same at every iteration.
With $\alpha = 1$ the first iterate is then the stochastic ensemble smoother, and later iterates stay there.
With $\alpha < 1$ the distance to it falls by exactly $1 - \alpha$ per iteration.

## Implementation in filterax

The backward smoothers take stacked histories of shape $(T, N_\mathcal{E}, D_x)$, from the Layer 2 `assimilate` of any filter ([chapter 16](16_assimilation_cycle.md)) or from a loop of your own.
They return a `SmoothingResult`, whose `smoothed_history` has the same shape.
Its `particles` is the final window, so a smoother chains into a forecast as a filter does.

| Class | Method | Takes | Note |
| --- | --- | --- | --- |
| `EnKS`, `EnsembleRTS` | `smooth` | the two histories | [](#eq-smoothers-step), one backward pass |
| `FixedLagSmoother(lag)` | `smooth` | the two histories | one backward pass per anchor |
| `EnsembleSqrtSmoother` | `smooth` | the two histories | [](#eq-smoothers-sqrt), through an `eigh` of $\tilde{\mat{X}} \tilde{\mat{X}}^\top$ |
| `IES(n_iterations, step_size)` | `solve` | an initial ensemble, $\vect{y}$, $\gmat{\Sigma}_\varepsilon$ and $h$ | returns an `AnalysisResult` |

A Layer 2 filter runs a deterministic `dynamics`, so its forecasts have no process noise, and [](#eq-smoothers-inverse) applies when $\mat{F}$ is linear and invertible.
A post-analysis inflator does not change that, because the next forecast still propagates the stored analysis, which [](#code-smoothers-no-noise) checks with `RTPS`.
To carry process noise into the forecast, write the loop yourself around a Layer 1 filter, as [](#code-smoothers-toy) does.

The example uses the model of [](#code-smoothers-toy) with $D_x = 4$ and two observed components, over four windows with 30 members.

```{code-block} python
:caption: A Layer 2 filter, then EnKS and EnsembleRTS on its stored history.
:label: code-smoothers-history
import einx
import equinox as eqx
import filterax as flx
import gaussx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
from jaxtyping import Array, Float

jax.config.update("jax_enable_x64", True)

# Shapes: T = 4 windows; N_E = 30 members; D_x = 4 components; D_y = 2 observations

# F, two damped rotations
transition = jnp.array(  # → (D_x, D_x)
    [
        [0.95, 0.1, 0.0, 0.0],
        [-0.1, 0.95, 0.0, 0.0],
        [0.0, 0.0, 0.9, 0.2],
        [0.0, 0.0, -0.2, 0.9],
    ]
)
# H selects components 0 and 2
obs_map = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])  # → (D_y, D_x)
# today (lineax)
# Σ_ε = 0.25 I
noise = lx.DiagonalLinearOperator(0.25 * jnp.ones(2))  # (D_y, D_y) operator


# user
# f(x) = F x, with no process noise
def advance(state: Float[Array, " state"], _, __) -> Float[Array, " state"]:
    return transition @ state


# user
# h(x) = H x
def observe(state: Float[Array, " state"]) -> Float[Array, " obs"]:
    return obs_map @ state


# user
# √(mean diag Σ̂_x) in each window
def spread(history: Float[Array, "t member state"]) -> Float[Array, " t"]:
    mean = einx.mean("t [member] state", history, keepdims=True)
    variance = einx.sum("t [member state]", (history - mean) ** 2)
    return jnp.sqrt(variance / (history.shape[1] - 1.0) / history.shape[2])


start = jr.normal(jr.key(0), (30, 4))  # → (N_E, D_x)
targets = jnp.array([[0.5, -0.5], [0.6, -0.4], [0.4, -0.6], [0.5, -0.5]])  # → (T, D_y)
windows = [(targets[t], t + 1.0) for t in range(4)]
# today (filterax)
filtered = flx.ETKF(dynamics=advance, obs_op=observe).assimilate(start, windows, noise)
forecast_history = filtered.forecast_history  # → (T, N_E, D_x)
analysis_history = filtered.analysis_history  # → (T, N_E, D_x)

# today (filterax)
enks = flx.EnKS().smooth(forecast_history, analysis_history)
print(enks.smoothed_history.shape)
print(jnp.array_equal(enks.particles, analysis_history[-1]))
print(spread(analysis_history).round(3), spread(enks.smoothed_history).round(3))

# today (filterax)
rts = flx.EnsembleRTS().smooth(forecast_history, analysis_history)
print(jnp.array_equal(rts.smoothed_history, enks.smoothed_history))
```

It prints `(4, 30, 4)` and `True`, so the final window is untouched.
It then prints `[0.768 0.69  0.612 0.532] [0.637 0.599 0.565 0.532]`, the spread of the filter and of EnKS in each window.
The last line is `True`, so `EnsembleRTS` gives the same arrays as `EnKS`.
This short run has no process noise, so the smoothed spread is the final analysis spread grown backward through $\mat{F}^{-1}$, by [](#eq-smoothers-inverse).

## Algorithm

[](#code-smoothers-algorithm) writes out the step in both forms, as a dense numpy reference with a square gain and with the pseudo-inverse of [](#eq-smoothers-step).

```{code-block} python
:caption: One backward step by hand, in the form of [](#eq-smoothers-sample-gain) and the form of [](#eq-smoothers-step), against gaussx.
:label: code-smoothers-algorithm
# Shapes: N_E = 30 members; D_x = 4 components


# user
# X_t|T = X_t|t + D_t+1 Ĝᵀ, with the D_x × D_x sample gain
def step_through_gain(smoothed_next, analysis, forecast_next):
    analysis_anomalies = analysis - analysis.mean(0)
    forecast_anomalies = forecast_next - forecast_next.mean(0)
    cross = analysis_anomalies.T @ forecast_anomalies
    gain = cross @ np.linalg.pinv(forecast_anomalies.T @ forecast_anomalies)
    return analysis + (smoothed_next - forecast_next) @ gain.T


# user
# X_t|T = X_t|t + D_t+1 X̃⁺_t+1|t X̃_t|t
def step_through_pinv(smoothed_next, analysis, forecast_next):
    analysis_anomalies = analysis - analysis.mean(0)
    forecast_anomalies = forecast_next - forecast_next.mean(0)
    weights = (smoothed_next - forecast_next) @ np.linalg.pinv(forecast_anomalies)
    return analysis + weights @ analysis_anomalies


arrays = (
    np.asarray(enks.smoothed_history[2]),
    np.asarray(analysis_history[1]),
    np.asarray(forecast_history[2]),
)
# today (gaussx)
stepped = gaussx.ensemble_rts_step(*(jnp.asarray(array) for array in arrays))
print(np.abs(step_through_gain(*arrays) - stepped).max() < 1e-12)
print(np.abs(step_through_pinv(*arrays) - stepped).max() < 1e-12)
print(jnp.abs(enks.smoothed_history[1] - stepped).max() < 1e-12)
```

It prints `True` three times: both forms match gaussx, and gaussx matches the second window of the backward pass above.

## Agreement with the exact smoother

The toy model is the one in `tests/test_smoother_workflows.py`.
It runs the $\mat{F}$ above over $T = 40$ windows, with $\gmat{\Sigma}_\eta = 0.1 \mat{I}$ in the truth and in each forecast, and with 100 members.
It is linear and Gaussian, so `gaussx.kalman_filter` and `gaussx.rts_smoother` give the exact answer.
Squared errors are taken on the mean, and they leave out the final window, where every smoother is the analysis.

```{code-block} python
:caption: The ensemble smoothers and the exact smoother on the toy model, over three independent draws.
:label: code-smoothers-toy
# Shapes: T = 40 windows; N_E = 100 members; D_x = 4; D_y = 2

NUM_TIMES = 40


# user
# xₜ₊₁ = F xₜ + ηₜ and yₜ = H xₜ + εₜ, with the seeds of the test file
@eqx.filter_jit
def simulate(seed: int, process_std: float):
    keys = jr.split(jr.key(seed), 2 * NUM_TIMES + 1)

    def step(state, time):
        kick = process_std * jr.normal(keys[2 * time + 1], (4,))  # → (D_x,)
        state = transition @ state + kick  # → (D_x,)
        wobble = 0.5 * jr.normal(keys[2 * time + 2], (2,))  # → (D_y,)
        return state, (state, obs_map @ state + wobble)

    start_state = jr.normal(keys[0], (4,))  # → (D_x,)
    _, outputs = jax.lax.scan(step, start_state, jnp.arange(NUM_TIMES))
    return outputs  # (T, D_x) and (T, D_y)


# user
# X_t+1|t = X_t|t Fᵀ + noise, then an ETKF analysis, with the seeds of the test file
@eqx.filter_jit
def filter_histories(seed: int, observations, process_std: float, num_members: int):
    keys = jr.split(jr.key(1000 + seed), NUM_TIMES + 1)

    def step(ensemble, inputs):
        key, target = inputs
        kick = process_std * jr.normal(key, (num_members, 4))  # → (N_E, D_x)
        moved = einx.dot("member state, out state -> member out", ensemble, transition)
        forecast = moved + kick  # → (N_E, D_x)
        # today (filterax)
        result = flx.filters.ETKF().analysis(forecast, target, observe, noise)
        return result.particles, (forecast, result.particles)

    first = jr.normal(keys[0], (num_members, 4))  # → (N_E, D_x)
    _, outputs = jax.lax.scan(step, first, (keys[1:], observations))
    return outputs  # both (T, N_E, D_x)


# user
# the mean squared error of x̄, leaving out the final window
def squared_error(history, truth):
    mean = einx.mean("t [member] state", history)  # → (T, D_x)
    return jnp.mean((mean[:-1] - truth[:-1]) ** 2)  # → ()


process_std = float(np.sqrt(0.1))
runs = []
for seed in range(3):
    truth, observations = simulate(seed, process_std)
    histories = filter_histories(seed, observations, process_std, 100)
    runs.append((truth, observations, *histories))


# user
# a backward pass, compiled once per ensemble shape
@eqx.filter_jit
def smooth_with(smoother, forecasts, analyses):
    # today (filterax)
    return smoother.smooth(forecasts, analyses).smoothed_history


# user
# the exact filter and smoother, on one draw
@eqx.filter_jit
def exact(observations):
    # today (gaussx)
    state = gaussx.kalman_filter(
        transition,
        obs_map,
        0.1 * jnp.eye(4),
        0.25 * jnp.eye(2),
        observations,
        jnp.zeros(4),
        jnp.eye(4),
    )
    # today (gaussx)
    means, covariances = gaussx.rts_smoother(state, transition)
    return state.filtered_means, state.filtered_covs, means, covariances


# user
# the squared errors of the filter, EnKS, the square-root smoother and the exact pair
def errors_of(truth, observations, forecasts, analyses):
    enks_history = smooth_with(flx.EnKS(), forecasts, analyses)
    root_history = smooth_with(flx.EnsembleSqrtSmoother(), forecasts, analyses)
    filtered_means, _, smoothed_means, _ = exact(observations)
    return np.array(
        [
            squared_error(analyses, truth),
            squared_error(enks_history, truth),
            squared_error(root_history, truth),
            jnp.mean((filtered_means[:-1] - truth[:-1]) ** 2),
            jnp.mean((smoothed_means[:-1] - truth[:-1]) ** 2),
        ]
    )


errors = np.stack([errors_of(*run) for run in runs])  # → (3, 5)
total = errors.sum(0)  # → (5,)
# EnKS, the square-root smoother and the exact smoother, each over its filter
print((total[1] / total[0]).round(3), (total[2] / total[0]).round(3))
print((total[4] / total[3]).round(3))
# draw by draw
print((errors[:, 1] / errors[:, 0]).round(3), (errors[:, 4] / errors[:, 3]).round(3))
# the variance ratio of the exact smoother, at the middle window
_, filtered_covs, _, smoothed_covs = exact(runs[0][1])
print((jnp.trace(smoothed_covs[20]) / jnp.trace(filtered_covs[20])).round(3))
```

It prints:

| Quantity | Value |
| --- | --- |
| EnKS over the filter, squared error, summed over the three draws | `0.776` |
| `EnsembleSqrtSmoother` over the filter, the same | `0.776` |
| Exact RTS over the exact filter, the same | `0.792` |
| EnKS over the filter, draw by draw | `[0.668 0.852 0.849]` |
| Exact RTS over the exact filter, draw by draw | `[0.677 0.821 0.915]` |
| Exact RTS over the exact filter, $\op{tr}(\gmat{\Sigma}_{x,t \mid T}) / \op{tr}(\gmat{\Sigma}_{x,t \mid t})$ at the middle window | `0.68` |

So EnKS with 100 members brings the squared error to about 0.78 of the filter's.
The exact smoother's expected improvement is the variance ratio, 0.68, and the test file compares against that number.
The two are not the same kind of quantity, and it would be wrong to read the gap of 0.1 as the cost of using an ensemble.
On these same three draws the exact smoother's own squared-error ratio is 0.79, and the draw-by-draw spread of either is wider than the gap.
What three draws can say is that EnKS matches the exact smoother about as closely as they can resolve.
The square-root smoother prints the same number because its mean is the same.

### The square-root smoother against EnKS

[](#code-smoothers-sqrt) checks the mean, then [](#eq-smoothers-rts-covariance) at every window of the first draw, then repeats with fewer members than components.

```{code-block} python
:caption: The means agree, and only the square-root smoother satisfies the sample RTS covariance recursion at every window.
:label: code-smoothers-sqrt
# Shapes: T = 40 windows; N_E = 100; D_x = 4, and T = 5; N_E = 10; D_x = 20


# user
# Σ̂_x
def sample_cov(ensemble: Float[Array, "member state"]) -> Float[Array, "state state"]:
    # today (filterax)
    return flx.ensemble_covariance(ensemble).as_matrix()


# user
# the largest gap from the recursion for the covariance, a dense numpy reference
def recursion_gap(smoothed, forecasts, analyses) -> float:
    smoothed, forecasts, analyses = map(np.asarray, (smoothed, forecasts, analyses))
    gaps = []
    for t in range(smoothed.shape[0] - 1):
        analysis_anomalies = analyses[t] - analyses[t].mean(0)
        forecast_anomalies = forecasts[t + 1] - forecasts[t + 1].mean(0)
        # Ĝ = X̃ᵀ_t|t (X̃⁺_t+1|t)ᵀ
        gain = analysis_anomalies.T @ np.linalg.pinv(forecast_anomalies).T
        change = np.cov(smoothed[t + 1].T) - np.cov(forecasts[t + 1].T)
        # Σ̂_x,t|t + Ĝ (Σ̂_x,t+1|T − Σ̂_x,t+1|t) Ĝᵀ
        expected = np.cov(analyses[t].T) + gain @ change @ gain.T
        gaps.append(np.abs(np.cov(smoothed[t].T) - expected).max())
    return max(gaps)


# user
# the gaps between the two smoothers, and from each to the recursion
def compare(forecasts, analyses) -> None:
    plain = smooth_with(flx.EnKS(), forecasts, analyses)  # → (T, N_E, D_x)
    rooted = smooth_with(flx.EnsembleSqrtSmoother(), forecasts, analyses)
    plain_means = einx.mean("t [member] state", plain)  # → (T, D_x)
    rooted_means = einx.mean("t [member] state", rooted)  # → (T, D_x)
    mean_gap = jnp.abs(plain_means - rooted_means).max()
    first_gap = jnp.abs(sample_cov(plain[0]) - sample_cov(rooted[0])).max()
    member_gap = jnp.abs(plain[0] - rooted[0]).max()
    plain_recursion = recursion_gap(plain, forecasts, analyses)
    rooted_recursion = recursion_gap(rooted, forecasts, analyses)
    print(f"{mean_gap:.0e} {first_gap:.0e} {member_gap:.2f}")
    print(f"{plain_recursion:.0e} {rooted_recursion:.0e}")


_, _, forecasts, analyses = runs[0]
compare(forecasts, analyses)

# N_E < D_x, on a random stable model
wide_map = 0.95 * jnp.eye(20) + 0.1 * jr.normal(jr.key(2), (20, 20))  # → (D_x, D_x)
wide_obs = jnp.eye(20)[:6]  # → (6, D_x)
# today (lineax)
wide_noise = lx.DiagonalLinearOperator(0.25 * jnp.ones(6))  # (6, 6) operator
wide_forecasts, wide_analyses = [], []
ensemble = jr.normal(jr.key(3), (10, 20))  # → (N_E, D_x)
for time in range(5):
    moved = einx.dot("member state, out state -> member out", ensemble, wide_map)
    forecast = moved + 0.3 * jr.normal(jr.key(100 + time), (10, 20))  # → (N_E, D_x)
    target = jr.normal(jr.key(200 + time), (6,))  # → (6,)
    # today (filterax)
    result = flx.filters.ETKF().analysis(
        forecast, target, lambda state: wide_obs @ state, wide_noise
    )
    ensemble = result.particles  # → (N_E, D_x)
    wide_forecasts.append(forecast)
    wide_analyses.append(ensemble)
wide_forecasts = jnp.stack(wide_forecasts)  # → (T, N_E, D_x)
wide_analyses = jnp.stack(wide_analyses)  # → (T, N_E, D_x)
compare(wide_forecasts, wide_analyses)
```

It prints:

| Case | Mean gap | Covariance gap, first window | Member gap, first window | EnKS from the recursion | Square root from the recursion |
| --- | --- | --- | --- | --- | --- |
| $N_\mathcal{E} = 100$, $D_x = 4$ | `2e-15` | `5e-02` | `0.85` | `4e-02` | `2e-15` |
| $N_\mathcal{E} = 10$, $D_x = 20$ | `1e-14` | `3e-15` | `0.23` | `3e-15` | `3e-15` |

The means agree in both cases.
With $N_\mathcal{E} < D_x$ the covariances agree too, and the members differ, so the two really do differ by a rotation.
With $N_\mathcal{E} > D_x$ the covariances differ by up to 0.05, and only the square-root smoother satisfies [](#eq-smoothers-rts-covariance), through the cross terms of [](#eq-smoothers-enks-anomalies).
The `EnsembleSqrtSmoother` docstring claims a rotation for all finite ensembles, which this case contradicts.

### The fixed-lag smoother

```{code-block} python
:caption: The error ratio of `FixedLagSmoother` as the lag grows, and its two limits.
:label: code-smoothers-lag
# Shapes: T = 40 windows; N_E = 100; D_x = 4

for lag in (1, 3, 10, 39):
    above, below = 0.0, 0.0
    for truth, _, forecasts, analyses in runs:
        # today (filterax)
        lagged_history = smooth_with(flx.FixedLagSmoother(lag=lag), forecasts, analyses)
        above += squared_error(lagged_history, truth)
        below += squared_error(analyses, truth)
    print(lag, f"{above / below:.3f}")

_, _, forecasts, analyses = runs[0]
full_pass = smooth_with(flx.EnKS(), forecasts, analyses)  # → (T, N_E, D_x)
# today (filterax)
at_zero = flx.FixedLagSmoother(lag=0).smooth(forecasts, analyses).smoothed_history
at_most = smooth_with(flx.FixedLagSmoother(lag=39), forecasts, analyses)
print(jnp.array_equal(at_zero, analyses), jnp.abs(at_most - full_pass).max() < 1e-12)
```

It prints `1 0.897`, `3 0.802`, `10 0.754` and `39 0.776`, then `True True`: a lag of 0 returns the filter, and a lag of 39 returns EnKS.
Most of the gain arrives in the first few lags.
Lag 10 comes out below the full pass, but the draw-by-draw spread above is wider than that difference, so this run does not show that long lags hurt.

### IES

[](#code-smoothers-ies) first takes the linear problem of [chapter 9](09_ensemble_kalman_processes.md), whose reference is a stochastic update with the same perturbed observations.
It then takes that chapter's nonlinear model [](#eq-processes-curved), with the mean of each member's objective [](#eq-smoothers-ies-cost) as the measure of progress.

```{code-block} python
:caption: IES against the stochastic ensemble smoother on a linear model, and its progress on a nonlinear one.
:label: code-smoothers-ies
# Shapes: N_E = 50 or 200 members; D_θ = 2 parameters; D_y = 3 observations

line_map = jnp.array([[1.0, 0.5], [-0.3, 1.2], [0.7, 0.1]])  # → (D_y, D_θ)
line_target = line_map @ jnp.array([1.0, -1.0]) + jnp.array([0.2, -0.3, 0.1])
# today (lineax)
line_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(3))  # (D_y, D_y) operator
line_start = 2.0 * jr.normal(jr.key(1), (50, 2))  # → (N_E, D_θ)


# user
def line_forward(theta: Float[Array, " param"]) -> Float[Array, " obs"]:
    return line_map @ theta


# today (filterax)
# y⁽ᵉ⁾ = y + ε⁽ᵉ⁾, the draw that IES makes at its default seed
line_data = flx.perturbed_observations(jr.PRNGKey(0), line_target, line_noise, 50)
# a dense numpy reference: θ⁽ᵉ⁾ + Ĝ (y⁽ᵉ⁾ − H θ⁽ᵉ⁾)
matrix = np.asarray(line_map)
prior_cov = np.cov(np.asarray(line_start).T)
innovation = matrix @ prior_cov @ matrix.T + 0.1 * np.eye(3)
gain = np.linalg.solve(innovation, matrix @ prior_cov).T
residuals = np.asarray(line_data) - np.asarray(line_start) @ matrix.T
reference = np.asarray(line_start) + residuals @ gain.T

for damping, iterations in ((1.0, 1), (1.0, 5), (0.5, 1), (0.5, 2), (0.5, 8)):
    # today (filterax)
    ies = flx.IES(n_iterations=iterations, step_size=damping)
    end = ies.solve(line_start, line_target, line_noise, line_forward).particles
    print(damping, iterations, f"{np.abs(np.asarray(end) - reference).max():.1e}")


# user
# h(θ) of the nonlinear model in chapter 9
def curved(theta: Float[Array, " param"]) -> Float[Array, " obs"]:
    return jnp.stack(
        [
            theta[0] + 0.3 * theta[1] ** 2,
            theta[1] + 0.3 * theta[0] ** 2,
            theta[0] * theta[1],
        ]
    )


curved_target = curved(jnp.array([1.0, -0.5])) + jnp.array([0.15, -0.2, 0.1])
draws = jr.normal(jr.key(0), (200, 2))  # → (N_E, D_θ)
# today (filterax)
curved_data = flx.perturbed_observations(jr.PRNGKey(0), curved_target, line_noise, 200)
# Σ̂_θ⁽⁰⁾⁻¹, a dense numpy reference
prior_precision = np.linalg.inv(np.cov(np.asarray(draws).T))


# user
# the mean over the members of 𝒥⁽ᵉ⁾(θ⁽ᵉ⁾)
def mean_cost(ensemble) -> float:
    misfit = np.asarray(curved_data - jax.vmap(curved)(ensemble))
    moved = np.asarray(ensemble - draws)
    data_term = 0.5 * (misfit**2).sum(1) / 0.1
    prior_term = 0.5 * ((moved @ prior_precision) * moved).sum(1)
    return float((data_term + prior_term).mean())


print(f"{mean_cost(draws):.3f}")
for iterations in (1, 2, 3, 5, 10):
    # today (filterax)
    ies = flx.IES(n_iterations=iterations)
    end = ies.solve(draws, curved_target, line_noise, curved).particles
    print(
        iterations,
        f"{mean_cost(end):.3f}",
        np.asarray(einx.mean("[member] param", end)).round(4),
    )
```

The linear part prints `1.0 1 8.9e-15` and `1.0 5 5.8e-15`, so the first iterate is the stochastic update and later ones stay there.
With $\alpha = 0.5$ it prints `3.3e+00`, `1.6e+00` and `2.5e-02` after 1, 2 and 8 iterations, which halves at each iteration as the formula says.

The nonlinear part starts from a mean cost of `26.212`, and then prints:

| Iterations | Mean cost | Ensemble mean |
| --- | --- | --- |
| 1 | `5.896` | `[ 0.7008 -0.6762]` |
| 2 | `2.590` | `[ 0.8343 -0.6145]` |
| 3 | `2.256` | `[ 0.8506 -0.5931]` |
| 5 | `2.238` | `[ 0.8603 -0.5817]` |
| 10 | `2.239` | `[ 0.8603 -0.5813]` |

One iteration, the plain ensemble smoother, leaves the mean far from the posterior mean of `[ 0.8699 -0.5688]` found on the grid in [](#code-processes-nonlinear).
Five iterations take it within 0.013, at a cost of 1000 evaluations of $h$.
The cost then settles rather than falling further, because the ensemble-average sensitivity is not each member's own Jacobian.

## Numerics: float32 and gradients

The singular values of the forecast anomalies are the point of the step.
A direction $10^{-4}$ of the largest has a squared singular value of $10^{-8}$ relative, which is below float32's epsilon, so $\tilde{\mat{X}} \tilde{\mat{X}}^\top$ cannot hold it.
The anomalies themselves can.
`gaussx.ensemble_rts_step` works from them, while `EnsembleSqrtSmoother` still goes through `eigh` of the product, with a cutoff at $N_\mathcal{E}$ times the epsilon relative to the largest eigenvalue.
[](#code-smoothers-float32) builds a forecast with exactly this spread of scales, and moves the smoothed ensemble a little along a weak direction.
It then compares each smoother in float32 with its own float64 result.

```{code-block} python
:caption: One backward step with singular values spanning four decades, in float32 against float64.
:label: code-smoothers-float32
# Shapes: T = 2 windows; N_E = 30 members; D_x = 4

base = jr.normal(jr.key(4), (30, 4))  # → (N_E, D_x)
scales = jnp.array([1.0, 1e-2, 1e-4, 1e-4])  # → (D_x,)
# X_t+1|t, with its directions scaled by four decades
scaled = base * scales  # → (N_E, D_x)
upcoming = scaled + jnp.array([1.0, 2.0, 3.0, 4.0])  # → (N_E, D_x)
earlier = 0.9 * scaled  # → (N_E, D_x)
later = upcoming + jnp.array([0.5, 0.0, 1e-4, 0.0])  # → (N_E, D_x)

results = {}
for precision in (jnp.float64, jnp.float32):
    fore = jnp.stack([earlier, upcoming]).astype(precision)  # → (T, N_E, D_x)
    ana = jnp.stack([earlier, later]).astype(precision)  # → (T, N_E, D_x)
    for name, smoother in (("EnKS", flx.EnKS()), ("sqrt", flx.EnsembleSqrtSmoother())):
        # today (filterax)
        ensemble = smoother.smooth(fore, ana).smoothed_history[0]  # → (N_E, D_x)
        # x̄_t|T
        results[name, precision] = einx.mean("[member] state", ensemble)  # → (D_x,)
for name in ("EnKS", "sqrt"):
    wide = results[name, jnp.float64]  # → (D_x,)
    narrow = results[name, jnp.float32].astype(jnp.float64)  # → (D_x,)
    gap = jnp.abs(narrow - wide).max()  # → ()
    print(name, f"{gap:.1e} {wide[2]:.8f} {narrow[2]:.8f}")
```

It prints `EnKS 5.0e-08 0.00010794 0.00010789` and `sqrt 7.4e-05 0.00010794 0.00003438`: the largest gap in the mean, then the mean's weak component in float64 and in float32.
The EnKS step keeps that component to four digits in float32.
The square-root smoother returns about a third of it, because its cutoff has dropped the direction.
In float64 the two routes agree, so this limit only shows in float32 with a wide spread of scales.

The derivative of the pseudo-inverse has the closed form of {cite:t}`golub1973differentiation`, which divides by no difference of singular values.
So EnKS keeps finite gradients where singular values repeat or vanish.
It holds while the rank is fixed, and a singular value crossing the cutoff changes $\mat{X}_{t \mid T}$ discontinuously.
`EnsembleSqrtSmoother` differentiates `eigh` instead, whose rule divides by the gaps ([chapter 6](06_etkf.md)).
With $N_\mathcal{E} - 1 > D_x$, the matrix $\tilde{\mat{X}} \tilde{\mat{X}}^\top$ has a repeated zero eigenvalue.
With $N_\mathcal{E} \le D_x$ that matrix is fine, but after an ETKF with few observations $\gmat{\Psi}$ has a repeated eigenvalue of 1.

```{code-block} python
:caption: The gradient with respect to a scale on the forecast, against a central difference.
:label: code-smoothers-gradient
# Shapes: T = 4 and N_E = 100, D_x = 4; or T = 5 and N_E = 10, D_x = 20


# user
# 𝓛(a) = ‖X_0|T‖²_F, when the forecasts are multiplied by a
def forecast_loss(smoother, forecasts, analyses, scale):
    # today (filterax)
    history = smoother.smooth(scale * forecasts, analyses).smoothed_history
    return jnp.sum(history[0] ** 2)  # → ()


at = jnp.asarray(1.0)
step = 1e-6
_, _, forecasts, analyses = runs[0]
for label, smoother, fore, ana in (
    ("EnKS, N_E > D_x", flx.EnKS(), forecasts[:4], analyses[:4]),
    ("sqrt, N_E > D_x", flx.EnsembleSqrtSmoother(), forecasts[:4], analyses[:4]),
    ("sqrt, N_E < D_x", flx.EnsembleSqrtSmoother(), wide_forecasts, wide_analyses),
):

    # user
    def loss(value):
        return forecast_loss(smoother, fore, ana, value)

    gradient = jax.grad(loss)(at)  # → ()
    central = (loss(at + step) - loss(at - step)) / (2.0 * step)  # → ()
    print(label, f"{gradient:.3f} {central:.3f}")
```

It prints `EnKS, N_E > D_x -2120.743 -2120.743`, `sqrt, N_E > D_x nan -2122.005` and `sqrt, N_E < D_x -1092.115 -1098.199`.
So EnKS agrees with the central difference, and the square-root smoother gives `NaN` in the first case and a silently wrong value, off by 0.6%, in the second.
Do not train through `EnsembleSqrtSmoother`.

## When it breaks

**A forecast with no process noise against a truth that has some.**
[](#code-smoothers-no-noise) filters the toy model with no noise in the forecast, on the same observation model, for a truth with and without process noise.
It checks [](#eq-smoothers-inverse), both per window and across the whole run, and then prints the errors.

```{code-block} python
:caption: Smoothing a deterministic forecast. The smoothed run is the final analysis pulled back through $\mat{F}^{-1}$.
:label: code-smoothers-no-noise
# Shapes: T = 40 windows; N_E = 100; D_x = 4

# F⁻ᵀ, a dense numpy reference
inverse_step = jnp.asarray(np.linalg.inv(np.asarray(transition)).T)  # → (D_x, D_x)
# F⁻ᵀ ³⁹
inverse_run = jnp.linalg.matrix_power(inverse_step, 39)  # → (D_x, D_x)
growth = np.abs(np.linalg.eigvals(np.asarray(inverse_step))).max()
print(f"{growth:.4f} {growth**39:.1f}")

for truth_std in (process_std, 0.0):
    per_draw = []
    for seed in range(3):
        truth, observations = simulate(seed, truth_std)
        forecasts, analyses = filter_histories(seed, observations, 0.0, 100)
        smoothed = smooth_with(flx.EnKS(), forecasts, analyses)  # → (T, N_E, D_x)
        # x̄ of the filter and of the smoother, in the first window
        filter_first = einx.mean("[member] state", analyses[0])  # → (D_x,)
        smoother_first = einx.mean("[member] state", smoothed[0])  # → (D_x,)
        per_draw.append(
            [
                squared_error(analyses, truth),
                squared_error(smoothed, truth),
                jnp.mean((filter_first - truth[0]) ** 2),
                jnp.mean((smoother_first - truth[0]) ** 2),
            ]
        )
    # X_t|T = X_t+1|T F⁻ᵀ
    one_gap = jnp.abs(smoothed[1] @ inverse_step - smoothed[0]).max()
    # X_1|T = X_T|T (F⁻ᵀ)³⁹
    run_gap = jnp.abs(analyses[-1] @ inverse_run - smoothed[0]).max()
    print(f"{one_gap:.0e} {run_gap:.0e}")
    filter_all, smoother_all, filter_early, smoother_early = np.array(per_draw).T
    print(
        filter_all.mean().round(3),
        smoother_all.mean().round(3),
        (smoother_all.sum() / filter_all.sum()).round(2),
    )
    print(filter_early.round(3), smoother_early.round(3))

# A post-analysis inflator leaves the identity in place
# today (filterax)
inflated = flx.ETKF(dynamics=advance, obs_op=observe, inflator=flx.RTPS(alpha=0.5))
stored = inflated.assimilate(start, windows, noise)
# today (filterax)
result = flx.EnKS().smooth(stored.forecast_history, stored.analysis_history)
kept = result.smoothed_history  # → (4, 30, D_x)
print(f"{jnp.abs(kept[1] @ inverse_step - kept[0]).max():.0e}")
```

It prints `1.0847 23.8`, the largest growth of $\mat{F}^{-1}$ per step and over the 39 steps of the run, and then:

| Truth | Gap per window, over the run | Filter error | Smoother error | Ratio | First window, filter by draw | First window, smoother by draw |
| --- | --- | --- | --- | --- | --- | --- |
| $\gmat{\Sigma}_\eta = 0.1 \mat{I}$ | `4e-15`, `1e-14` | `0.303` | `0.379` | `1.25` | `[1.352 0.549 0.741]` | `[0.069 0.99  3.498]` |
| $\gmat{\Sigma}_\eta = \mat{0}$ | `3e-15`, `7e-15` | `0.073` | `0.009` | `0.13` | `[1.422 0.463 0.306]` | `[0.046 0.085 0.018]` |

and a last line of `4e-16` for the same identity under `RTPS`.

The gaps show that [](#eq-smoothers-inverse) holds to rounding, so the explanation of the first row is a measured one and not an interpretation.
The smoothed run is the final analysis pulled back through $\mat{F}^{-1}$, which grows the error of that analysis, and it cannot express any process noise the truth had.
So the smoother is 25% worse than the filter overall.
Early in the run it can be far worse, by a factor of 4.7 in the first window of the third draw, though it is better in the first draw.
The second row uses the same smoother and the same algebra on a truth that really has no noise, and the error falls to 0.13 of the filter's.
So the failure belongs to the mismatch between the forecast and the truth, not to the smoother.
With process noise back in the forecast, $\tilde{\mat{X}}_{t+1 \mid t}$ no longer equals $\tilde{\mat{X}}_{t \mid t} \mat{F}^\top$, the identity no longer holds, and the ratio returns to the 0.78 of [](#code-smoothers-toy).

**Too few members.**
With a small ensemble the smoother can do worse than the filter.
[](#code-smoothers-small) also breaks the member correspondence on purpose.
The reason I would give is sampling error in the gain, applied again at every window, but that is an interpretation, and the block only measures the outcome.

```{code-block} python
:caption: The error ratio as the ensemble shrinks, and with the forecast members shuffled.
:label: code-smoothers-small
# Shapes: T = 40 windows; N_E varies; D_x = 4

for num_members in (5, 10, 20):
    above, below = 0.0, 0.0
    for seed, (truth, observations, _, _) in enumerate(runs):
        forecasts, analyses = filter_histories(
            seed, observations, process_std, num_members
        )
        smoothed = smooth_with(flx.EnKS(), forecasts, analyses)  # → (T, N_E, D_x)
        above += squared_error(smoothed, truth)
        below += squared_error(analyses, truth)
    print(num_members, f"{above / below:.3f}")

above, below = 0.0, 0.0
order = jr.permutation(jr.key(9), 100)  # → (N_E,)
for truth, _, forecasts, analyses in runs:
    shuffled = smooth_with(flx.EnKS(), forecasts[:, order], analyses)
    above += squared_error(shuffled, truth)
    below += squared_error(analyses, truth)
print(f"{above / below:.3f}")
```

It prints `5 1.389`, `10 1.041` and `20 0.868`, then `1.008` for the shuffled case.
With $D_x = 4$, 5 members make the smoother 39% worse than the filter, 10 members leave it about even, and 20 start to pay.
Shuffling the forecast against the analysis destroys the regression, and the improvement of [](#code-smoothers-toy) disappears.

Two other limits are worth knowing:

- The final window is never improved, so evaluate a smoother away from the end of the run.
- The backward step assumes the member correspondence through every stored window. Anything that reorders or resamples members between the stored analysis and the stored forecast breaks it silently, as the shuffled case shows.

## When a smoother is the right answer

Use `EnKS` for a reanalysis: one backward pass over a stored history.
It needs a forecast with process noise, or a model you trust as exact, and an ensemble large enough to estimate the gain.
Use `EnsembleRTS` for the same run when the RTS reading is the one your code means.

Use `EnsembleSqrtSmoother` if you need the sample covariance to obey the RTS recursion exactly, with $N_\mathcal{E} > D_x$, and you are not differentiating through it.
Use `FixedLagSmoother` when memory or latency bounds how far back you can go.
Use `IES` when one window maps through a strongly nonlinear $h$ and you want each member to solve its own objective.
History matching is the usual case.

If the forecast is deterministic and the model is exact, the smoother is doing strong-constraint estimation, and the ratio of 0.13 above shows it does that well.
If the forecast is deterministic and the model is not exact, add process noise to the forecast before you smooth.

## See also

- [Chapter 6](06_etkf.md): the forward transform that the square-root smoother mirrors, and the `eigh` gradient problem
- [Chapter 9](09_ensemble_kalman_processes.md): EKI and its relatives, the iterative methods without an anchor
- [Chapter 14](14_differentiable.md): differentiating through filter and smoother chains
- [Chapter 16](16_assimilation_cycle.md): where `forecast_history` and `analysis_history` come from
- [Chapter 17](17_exact_references.md): `KalmanSmoother`, the exact linear-Gaussian reference
- [API: smoothers](https://jejjohnson.github.io/filterax/reference/smoothers/) and [filters](https://jejjohnson.github.io/filterax/reference/filters/)

---
numbering:
  enumerator: "17.%s"
---

# Exact references: the square-root Kalman filter and the RTS smoother

An ensemble filter is an approximation, so it needs something exact to be measured against.
For a linear-Gaussian state-space model that thing exists: the posterior is Gaussian, and its mean and covariance have a closed recursion {cite:p}`kalman1960new,rauch1965maximum`.
filterax ships two exact solvers for this case, [`filters.SquareRootKF`](https://jejjohnson.github.io/filterax/reference/filters/) and [`KalmanSmoother`](https://jejjohnson.github.io/filterax/reference/smoothers/), and neither is an ensemble method.

Three ideas organise this chapter.
On a linear-Gaussian model these two classes compute the posterior itself, which we check against a dense solve of the joint Gaussian.
The square-root form carries a factor instead of a covariance, and it parallelises in time because the filter's steps combine associatively.
And an ensemble converges to these references at the rate $1 / \sqrt{N_\mathcal{E}}$, which we measure.

## What they compute

The model has a state $\vect{x}_t \in \mathbb{R}^{D_x}$ and observations $\vect{y}_t \in \mathbb{R}^{D_y}$ for $t = 1, \dots, T$:

$$\vect{x}_t = \mat{F}_t \vect{x}_{t-1} + \gvect{\eta}_t, \qquad \vect{y}_t = \mat{H}_t \vect{x}_t + \gvect{\varepsilon}_t,$$ (eq-exact-model)

with $\mat{F}_t \in \mathbb{R}^{D_x \times D_x}$, $\mat{H}_t \in \mathbb{R}^{D_y \times D_x}$, $\gvect{\eta}_t \sim \mathcal{N}(\vect{0}, \gmat{\Sigma}_\eta)$ and $\gvect{\varepsilon}_t \sim \mathcal{N}(\vect{0}, \gmat{\Sigma}_\varepsilon)$.
Write $\gvect{\mu}_{x,t \mid s} \in \mathbb{R}^{D_x}$ and $\gmat{\Sigma}_{x,t \mid s} \in \mathbb{R}^{D_x \times D_x}$ for the mean and covariance of $\vect{x}_t$ given $\vect{y}_{1:s}$.
The filter alternates a prediction and an update.
With the residual $\vect{r}_t \in \mathbb{R}^{D_y}$, the innovation covariance $\gmat{\Sigma}_{y,t}$ and the gain $\mat{G}_t \in \mathbb{R}^{D_x \times D_y}$,

$$\gvect{\mu}_{x,t \mid t-1} = \mat{F}_t \gvect{\mu}_{x,t-1 \mid t-1}, \qquad \gmat{\Sigma}_{x,t \mid t-1} = \mat{F}_t \gmat{\Sigma}_{x,t-1 \mid t-1} \mat{F}_t^\top + \gmat{\Sigma}_\eta,$$ (eq-exact-predict)

$$\vect{r}_t = \vect{y}_t - \mat{H}_t \gvect{\mu}_{x,t \mid t-1}, \qquad \gmat{\Sigma}_{y,t} = \mat{H}_t \gmat{\Sigma}_{x,t \mid t-1} \mat{H}_t^\top + \gmat{\Sigma}_\varepsilon, \qquad \mat{G}_t = \gmat{\Sigma}_{x,t \mid t-1} \mat{H}_t^\top \gmat{\Sigma}_{y,t}^{-1},$$ (eq-exact-innovation)

$$\gvect{\mu}_{x,t \mid t} = \gvect{\mu}_{x,t \mid t-1} + \mat{G}_t \vect{r}_t, \qquad \gmat{\Sigma}_{x,t \mid t} = \gmat{\Sigma}_{x,t \mid t-1} - \mat{G}_t \gmat{\Sigma}_{y,t} \mat{G}_t^\top.$$ (eq-exact-update)

The evidence is a by-product, since each residual is Gaussian given the past:

$$\log p(\vect{y}_{1:T}) = \sum_{t=1}^{T} \log \mathcal{N}(\vect{r}_t; \vect{0}, \gmat{\Sigma}_{y,t}).$$ (eq-exact-evidence)

The Rauch-Tung-Striebel pass {cite:p}`rauch1965maximum` then runs backward from $\gvect{\mu}_{x,T \mid T}$, with the smoother gain $\mat{G}_{\text{smooth},t} \in \mathbb{R}^{D_x \times D_x}$:

$$\mat{G}_{\text{smooth},t} = \gmat{\Sigma}_{x,t \mid t} \mat{F}_{t+1}^\top \gmat{\Sigma}_{x,t+1 \mid t}^{-1},$$ (eq-exact-smoother-gain)

$$\gvect{\mu}_{x,t \mid T} = \gvect{\mu}_{x,t \mid t} + \mat{G}_{\text{smooth},t} (\gvect{\mu}_{x,t+1 \mid T} - \gvect{\mu}_{x,t+1 \mid t}),$$ (eq-exact-smoother-mean)

$$\gmat{\Sigma}_{x,t \mid T} = \gmat{\Sigma}_{x,t \mid t} + \mat{G}_{\text{smooth},t} (\gmat{\Sigma}_{x,t+1 \mid T} - \gmat{\Sigma}_{x,t+1 \mid t}) \mat{G}_{\text{smooth},t}^\top.$$ (eq-exact-smoother-covariance)

Nothing here is estimated, and no sampling enters.
So on a linear-Gaussian model the error of an ensemble method against these numbers is the error of the ensemble method, which is what makes them a reference.
Outside that model they are not the posterior, and [chapter 20](20_inversion_example.md) needs a grid instead.

## The two classes

The classes differ in more than their names.

| | `filters.SquareRootKF` | `KalmanSmoother` |
| --- | --- | --- |
| Returns | filtered and predicted marginals, $\log p(\vect{y}_{1:T})$ | filtered and smoothed marginals, $\log p(\vect{y}_{1:T})$ |
| Solver | `gaussx.parallel_kalman_filter(square_root=True)` | `gaussx.kalman_filter`, then `gaussx.rts_smoother` |
| Form | factors, in an associative scan over time | covariances, sequential in time |
| Prior | one transition before $\vect{y}_1$ | at the time of $\vect{y}_1$, so $T - 1$ transitions |
| Model | one $\mat{F}$, $\mat{H}$, $\gmat{\Sigma}_\eta$, $\gmat{\Sigma}_\varepsilon$ for all times | a stack over time, or one for all times |
| Missing data | none in the wrapper | a `(T, D_y)` mask, with `NaN` allowed at masked entries |
| Irregular times | no | `KalmanSmoother.from_sde` |

`SquareRootFilterResult` holds `filtered_means`, `filtered_covs`, `predicted_means`, `predicted_covs` and `log_likelihood`.
`KalmanSmoothingResult` holds `filtered_means`, `filtered_covs`, `smoothed_means`, `smoothed_covs` and `log_likelihood`, and it keeps no cross-time covariances.

The prior convention is the difference most likely to catch you out.
To make the two agree, start `KalmanSmoother` from the one-step prediction of the `SquareRootKF` prior, $\mathcal{N}(\mat{F} \gvect{\mu}_{x,0}, \mat{F} \gmat{\Sigma}_{x,0} \mat{F}^\top + \gmat{\Sigma}_\eta)$.

[](#code-exact-agreement) runs both on the same model, with $D_x = 3$, $D_y = 2$ and $T = 40$, on data drawn from the model.
It compares them with two references that use no filter code at all.
One is a dense Joseph-form recursion for [](#eq-exact-update).
The other is one conditioning of the joint Gaussian of $(\vect{x}_{1:T}, \vect{y}_{1:T})$, whose state block has side $T D_x$.

```{code-block} python
:caption: Both classes against a dense Joseph-form filter and a dense solve of the joint Gaussian.
:label: code-exact-agreement
import einx
import filterax as flx
import gaussx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
from jaxtyping import Array, Float, Key

jax.config.update("jax_enable_x64", True)

# Shapes: T = 40 times; D_x = 3 state components; D_y = 2 observations

num_steps = 40
# F
transition = jnp.array(  # → (D_x, D_x)
    [[1.0, 0.1, 0.0], [0.0, 0.9, 0.1], [0.05, 0.0, 0.95]]
)
# H
obs_model = jnp.array([[1.0, 0.0, 0.5], [0.0, 1.0, -0.3]])  # → (D_y, D_x)
# Σ_η
process_noise = 0.1 * jnp.eye(3) + 0.02  # → (D_x, D_x)
# Σ_ε
obs_noise = 0.2 * jnp.eye(2)  # → (D_y, D_y)
# μ_x,0
init_mean = jnp.zeros(3)  # → (D_x,)
# Σ_x,0
init_cov = jnp.eye(3)  # → (D_x, D_x)
# L_η with L_η L_ηᵀ = Σ_η
process_root = jnp.linalg.cholesky(process_noise)  # → (D_x, D_x)
# L_ε with L_ε L_εᵀ = Σ_ε
obs_root = jnp.linalg.cholesky(obs_noise)  # → (D_y, D_y)


# user
# xₜ = F xₜ₋₁ + ηₜ and yₜ = H xₜ + εₜ
def draw_step(
    state: Float[Array, " state"], key: Key[Array, ""]
) -> tuple[Float[Array, " state"], tuple[Array, Array]]:
    process_key, obs_key = jr.split(key)
    next_state = transition @ state + process_root @ jr.normal(process_key, (3,))
    observed = obs_model @ next_state + obs_root @ jr.normal(obs_key, (2,))
    return next_state, (next_state, observed)


# x₀ ~ 𝒩(μ_x,0, Σ_x,0)
first_state = jr.normal(jr.key(5), (3,))  # → (D_x,)
_, (truth, observations) = jax.lax.scan(  # → (T, D_x), (T, D_y)
    draw_step, first_state, jr.split(jr.key(6), num_steps)
)

# today (filterax)
# the factor-propagating parallel filter
square_root_filter = flx.filters.SquareRootKF(
    transition, obs_model, process_noise, obs_noise
)
kalman = jax.jit(lambda y, m, P: square_root_filter.filter(y, m, P))(
    observations, init_mean, init_cov
)


# user
# F Σ Fᵀ + Σ_η
def predicted_cov(F, cov, noise):
    moved = einx.dot("row state, state col -> row col", F, cov)
    return einx.dot("row state, col state -> row col", moved, F) + noise


# today (filterax)
# the same model, from the prior of x₁: 𝒩(F μ_x,0, F Σ_x,0 Fᵀ + Σ_η)
smoother = flx.KalmanSmoother(
    transition,
    process_noise,
    obs_model,
    obs_noise,
    transition @ init_mean,
    predicted_cov(transition, init_cov, process_noise),
)
smoothed = jax.jit(lambda y: smoother.smooth(y))(observations)


# user
# the recursion above in dense numpy, with the Joseph form of the update
def joseph_filter(F, H, Q, R, y, mean, cov):
    means, covs, log_evidence = [], [], 0.0
    for observed in np.asarray(y):
        mean = F @ mean
        cov = F @ cov @ F.T + Q
        innovation = H @ cov @ H.T + R
        residual = observed - H @ mean
        gain = np.linalg.solve(innovation, H @ cov).T
        mean = mean + gain @ residual
        contraction = np.eye(len(mean)) - gain @ H
        cov = contraction @ cov @ contraction.T + gain @ R @ gain.T
        log_evidence -= 0.5 * residual @ np.linalg.solve(innovation, residual)
        log_evidence -= 0.5 * np.linalg.slogdet(2 * np.pi * innovation)[1]
        means.append(mean)
        covs.append(cov)
    return np.stack(means), np.stack(covs), log_evidence


# user
# condition the joint Gaussian of (x_1:T, y_1:T) once, with no recursion
def joint_posterior(F, H, Q, R, y, mean, cov):
    size, count = len(mean), len(y)
    prior_means, marginals = [], []
    for _ in range(count):
        prior_means.append(mean)
        marginals.append(cov)
        mean = F @ mean
        cov = F @ cov @ F.T + Q
    joint = np.zeros((count * size, count * size))
    for start in range(count):
        block = marginals[start]
        for later in range(start, count):
            lower = later * size, (later + 1) * size
            upper = start * size, (start + 1) * size
            joint[lower[0] : lower[1], upper[0] : upper[1]] = block
            joint[upper[0] : upper[1], lower[0] : lower[1]] = block.T
            block = F @ block
    stacked_obs = np.kron(np.eye(count), H)
    innovation = stacked_obs @ joint @ stacked_obs.T + np.kron(np.eye(count), R)
    residual = np.asarray(y).ravel() - stacked_obs @ np.concatenate(prior_means)
    gain = np.linalg.solve(innovation, stacked_obs @ joint).T
    posterior_mean = np.concatenate(prior_means) + gain @ residual
    posterior_cov = joint - gain @ stacked_obs @ joint
    log_evidence = -0.5 * residual @ np.linalg.solve(innovation, residual)
    log_evidence -= 0.5 * np.linalg.slogdet(2 * np.pi * innovation)[1]
    diagonal = [
        posterior_cov[t * size : (t + 1) * size, t * size : (t + 1) * size]
        for t in range(count)
    ]
    return posterior_mean.reshape(count, size), np.stack(diagonal), log_evidence


model = [np.asarray(a) for a in (transition, obs_model, process_noise, obs_noise)]
dense = joseph_filter(*model, observations, np.asarray(init_mean), np.asarray(init_cov))
prior_of_first = np.asarray(transition @ init_mean)
cov_of_first = np.asarray(predicted_cov(transition, init_cov, process_noise))
joint = joint_posterior(*model, observations, prior_of_first, cov_of_first)


# user
# largest absolute entry of a difference
def gap(first, second) -> float:
    return float(np.abs(np.asarray(first) - np.asarray(second)).max())


print(f"{gap(kalman.filtered_means, dense[0]):.0e}")
print(f"{gap(kalman.filtered_covs, dense[1]):.0e}")
print(f"{gap(kalman.log_likelihood, dense[2]):.0e}")
print(f"{gap(smoothed.filtered_means, kalman.filtered_means):.0e}")
print(f"{gap(smoothed.smoothed_means, joint[0]):.0e}")
print(f"{gap(smoothed.smoothed_covs, joint[1]):.0e}")
print(f"{gap(smoothed.log_likelihood, joint[2]):.0e}")
```

It prints `6e-15`, `2e-15`, `1e-14`, `6e-15`, `3e-13`, `2e-14` and `7e-13`.
The first three compare `SquareRootKF` with the Joseph-form filter, on the means, the covariances and the log-likelihood.
The fourth compares the filtered means of the two classes.
The last three compare `KalmanSmoother` with the dense joint solve, on the smoothed means, the smoothed covariances and the log-likelihood.
All seven are float64 round-off, so the classes compute the posterior and not an approximation to it.

## Irregular times and missing data

`KalmanSmoother` does not need a time grid.
`KalmanSmoother.from_sde` takes a gaussx SDE kernel and discretises it over the actual gaps $\Delta t_t = t_t - t_{t-1}$:

$$\mat{F}_t = \exp(\mat{A} \Delta t_t), \qquad \gmat{\Sigma}_{\eta,t} = \gmat{\Sigma}_{x,\infty} - \mat{F}_t \gmat{\Sigma}_{x,\infty} \mat{F}_t^\top,$$ (eq-exact-discretise)

where $\mat{A}$ is the drift of the SDE and $\gmat{\Sigma}_{x,\infty}$ its stationary covariance, which is also the prior of $\vect{x}_1$.
A Matérn kernel has such a form, and then the smoothed mean is Gaussian process regression.
So [](#code-exact-sde) has a reference with no state space in it at all: a dense GP solve with the Matérn-3/2 kernel, at 30 irregular times, one of which is masked.
The masked entry is `NaN`, which the class never reads.

A mask makes an entry inert.
Its row of $\mat{H}_t$ is zeroed, its row and column of $\gmat{\Sigma}_{\varepsilon,t}$ become those of the identity, and its datum becomes 0.
Then $\gmat{\Sigma}_{y,t}$ splits into a block for the observed entries and an identity block for the rest.
So that column of the gain vanishes, and the update equals the one with the entry deleted.

```{code-block} python
:caption: The state-space smoother over irregular times, against a dense Gaussian process solve.
:label: code-exact-sde
# Shapes: T = 30 times; D_x = 2 for Matérn-3/2 (f and f′); D_y = 1


# times t_1 < ⋯ < t_T, irregularly spaced
times = jnp.sort(jr.uniform(jr.key(7), (30,), maxval=10.0))  # → (T,)
# y = sin t + noise, of variance 0.05
values = jnp.sin(times) + jnp.sqrt(0.05) * jr.normal(jr.key(8), (30,))  # → (T,)
# one masked time
mask = jnp.ones((30, 1), dtype=bool).at[12, 0].set(False)  # → (T, D_y)
# today (gaussx)
# Matérn-3/2 with σ_k² = 1.5 and ℓ = 2
kernel = gaussx.MaternSDE(jnp.asarray(1.5), jnp.asarray(2.0), order=1)
# today (filterax)
# the discretisation above, with H reading f
irregular = flx.KalmanSmoother.from_sde(
    kernel, times, jnp.array([[1.0, 0.0]]), jnp.array([[0.05]])
)
# y, with NaN where masked
masked_values = jnp.where(mask, einx.id("t -> t 1", values), jnp.nan)  # → (T, D_y)
# today (filterax)
state_space = jax.jit(lambda y, seen: irregular.smooth(y, seen))(masked_values, mask)

# k(τ) = σ_k² (1 + √3 |τ| / ℓ) exp(−√3 |τ| / ℓ), in dense numpy
grid, kept = np.asarray(times), np.flatnonzero(np.asarray(mask[:, 0]))
scaled_lag = np.sqrt(3.0) * np.abs(grid[:, None] - grid[None, :]) / 2.0
covariance = 1.5 * (1.0 + scaled_lag) * np.exp(-scaled_lag)
# K_XX + σ² I, on the observed times only
gram = covariance[np.ix_(kept, kept)] + 0.05 * np.eye(len(kept))
residual = np.linalg.solve(gram, np.asarray(values)[kept])
# μ = K_*X (K_XX + σ² I)⁻¹ y, at all 30 times
gp_mean = covariance[:, kept] @ residual
# σ² = diag(K_** − K_*X (K_XX + σ² I)⁻¹ K_X*)
reduction = np.linalg.solve(gram, covariance[kept, :])
gp_variance = np.diag(covariance - covariance[:, kept] @ reduction)
# log 𝒩(y; 0, K_XX + σ² I)
gp_evidence = -0.5 * np.asarray(values)[kept] @ residual
gp_evidence -= 0.5 * np.linalg.slogdet(2 * np.pi * gram)[1]

print(f"{gap(state_space.smoothed_means[:, 0], gp_mean):.0e}")
print(f"{gap(state_space.smoothed_covs[:, 0, 0], gp_variance):.0e}")
print(f"{gap(state_space.log_likelihood, gp_evidence):.0e}")
```

It prints `3e-15`, `8e-16` and `9e-15`, for the smoothed mean, the smoothed variance and the log-likelihood.
A filter over 30 irregular times and a dense GP solve therefore give the same answer, with the masked time contributing nothing to the evidence.

## The square-root form

The covariance form of [](#eq-exact-update) subtracts, so rounding can push $\gmat{\Sigma}_{x,t \mid t}$ out of the positive semi-definite cone.
A square-root filter carries a factor $\mat{L}_x$ with $\gmat{\Sigma}_x = \mat{L}_x \mat{L}_x^\top$ instead, so the covariance it reports is a Gram matrix.

Call $\op{tria}(\mat{A})$ the lower-triangular $\mat{L}$ with $\mat{L} \mat{L}^\top = \mat{A} \mat{A}^\top$, which is the transpose of the $\mat{R}$ in a QR decomposition of $\mat{A}^\top$.
Prediction stacks the two factors of [](#eq-exact-predict) side by side:

$$\mat{L}_{x,t \mid t-1} = \op{tria}\big(\begin{bmatrix} \mat{F}_t \mat{L}_{x,t-1 \mid t-1} & \mat{L}_\eta \end{bmatrix}\big).$$ (eq-exact-tria-predict)

The update gets everything from one triangularisation.
With $\mat{L}_x = \mat{L}_{x,t \mid t-1}$, and $\gmat{\Psi} \in \mathbb{R}^{(D_y + D_x) \times (D_y + D_x)}$ partitioned into blocks,

$$\gmat{\Psi} = \op{tria} \begin{bmatrix} \mat{H} \mat{L}_x & \mat{L}_\varepsilon \\ \mat{L}_x & \mat{0} \end{bmatrix} = \begin{bmatrix} \gmat{\Psi}_{11} & \mat{0} \\ \gmat{\Psi}_{21} & \gmat{\Psi}_{22} \end{bmatrix}.$$ (eq-exact-psi)

Multiplying the stacked matrix by its transpose gives

$$\gmat{\Psi} \gmat{\Psi}^\top = \begin{bmatrix} \gmat{\Sigma}_y & \mat{H} \gmat{\Sigma}_x \\ \gmat{\Sigma}_x \mat{H}^\top & \gmat{\Sigma}_x \end{bmatrix}.$$ (eq-exact-psi-gram)

Reading [](#eq-exact-psi-gram) block by block gives each quantity by hand.
The top left says $\gmat{\Psi}_{11} \gmat{\Psi}_{11}^\top = \gmat{\Sigma}_y$, so $\gmat{\Psi}_{11} = \mat{L}_y$.
The bottom left says $\gmat{\Psi}_{21} \mat{L}_y^\top = \gmat{\Sigma}_x \mat{H}^\top$, so the gain is $\mat{G} = \gmat{\Psi}_{21} \mat{L}_y^{-1}$.
The bottom right says $\gmat{\Psi}_{21} \gmat{\Psi}_{21}^\top + \gmat{\Psi}_{22} \gmat{\Psi}_{22}^\top = \gmat{\Sigma}_x$, and $\gmat{\Psi}_{21} \gmat{\Psi}_{21}^\top = \mat{G} \gmat{\Sigma}_y \mat{G}^\top$, so

$$\mat{L}_{x,t \mid t} = \gmat{\Psi}_{22}, \qquad \gvect{\mu}_{x,t \mid t} = \gvect{\mu}_{x,t \mid t-1} + \gmat{\Psi}_{21} \mat{L}_y^{-1} \vect{r}_t.$$ (eq-exact-tria-update)

No covariance is formed and nothing subtracts.

### Combining steps in parallel

The recursion looks sequential, but the steps combine associatively {cite:p}`sarkka2021temporal`, and so they can run as a parallel scan.
Take a block of steps from $s$ to $u$.
Describe it by what it says about the state at its end given the state at its start, and about the data in between:

$$p(\vect{x}_u \mid \vect{x}_s, \vect{y}_{s+1:u}) = \mathcal{N}(\vect{x}_u; \mat{F}_\text{cond} \vect{x}_s + \gvect{\mu}_\text{cond}, \gmat{\Sigma}_\text{cond}),$$ (eq-exact-element-conditional)

$$p(\vect{y}_{s+1:u} \mid \vect{x}_s) \propto \exp\big(-\tfrac{1}{2} \vect{x}_s^\top \gmat{\Lambda}_\text{lik} \vect{x}_s + \vect{b}_\text{lik}^\top \vect{x}_s\big).$$ (eq-exact-element-likelihood)

These five quantities, with $\mat{F}_\text{cond}, \gmat{\Sigma}_\text{cond}, \gmat{\Lambda}_\text{lik} \in \mathbb{R}^{D_x \times D_x}$ and $\gvect{\mu}_\text{cond}, \vect{b}_\text{lik} \in \mathbb{R}^{D_x}$, are an element.
For one step $t \ge 2$ they are the update of [](#eq-exact-update) with $\gmat{\Sigma}_\eta$ as the predicted covariance.
With $\gmat{\Sigma}_{y \mid x} = \mat{H}_t \gmat{\Sigma}_\eta \mat{H}_t^\top + \gmat{\Sigma}_\varepsilon$ and $\mat{G}_\text{cond} = \gmat{\Sigma}_\eta \mat{H}_t^\top \gmat{\Sigma}_{y \mid x}^{-1}$,

$$\mat{F}_\text{cond} = (\mat{I} - \mat{G}_\text{cond} \mat{H}_t) \mat{F}_t, \qquad \gvect{\mu}_\text{cond} = \mat{G}_\text{cond} \vect{y}_t, \qquad \gmat{\Sigma}_\text{cond} = (\mat{I} - \mat{G}_\text{cond} \mat{H}_t) \gmat{\Sigma}_\eta,$$ (eq-exact-element-conditional-step)

$$\vect{b}_\text{lik} = \mat{F}_t^\top \mat{H}_t^\top \gmat{\Sigma}_{y \mid x}^{-1} \vect{y}_t, \qquad \gmat{\Lambda}_\text{lik} = \mat{F}_t^\top \mat{H}_t^\top \gmat{\Sigma}_{y \mid x}^{-1} \mat{H}_t \mat{F}_t.$$ (eq-exact-element-likelihood-step)

The first step absorbs the prior instead: its $\mat{F}_\text{cond}$, $\vect{b}_\text{lik}$ and $\gmat{\Lambda}_\text{lik}$ are zero, and its $\gvect{\mu}_\text{cond}$ and $\gmat{\Sigma}_\text{cond}$ are the filtered moments at $t = 1$.

To join a block $i$ from $s$ to $u$ with a block $j$ from $u$ to $v$, integrate out $\vect{x}_u$.
That Gaussian integral gives

$$\mat{F}_{ij} = \mat{F}_j (\mat{I} + \gmat{\Sigma}_i \gmat{\Lambda}_j)^{-1} \mat{F}_i, \qquad \gvect{\mu}_{ij} = \mat{F}_j (\mat{I} + \gmat{\Sigma}_i \gmat{\Lambda}_j)^{-1} (\gvect{\mu}_i + \gmat{\Sigma}_i \vect{b}_j) + \gvect{\mu}_j,$$ (eq-exact-combine-conditional)

$$\gmat{\Sigma}_{ij} = \mat{F}_j (\mat{I} + \gmat{\Sigma}_i \gmat{\Lambda}_j)^{-1} \gmat{\Sigma}_i \mat{F}_j^\top + \gmat{\Sigma}_j,$$ (eq-exact-combine-covariance)

$$\vect{b}_{ij} = \mat{F}_i^\top (\mat{I} + \gmat{\Lambda}_j \gmat{\Sigma}_i)^{-1} (\vect{b}_j - \gmat{\Lambda}_j \gvect{\mu}_i) + \vect{b}_i, \qquad \gmat{\Lambda}_{ij} = \mat{F}_i^\top (\mat{I} + \gmat{\Lambda}_j \gmat{\Sigma}_i)^{-1} \gmat{\Lambda}_j \mat{F}_i + \gmat{\Lambda}_i,$$ (eq-exact-combine-likelihood)

with the subscript cond or lik dropped.
The first element has $\mat{F}_\text{cond} = \mat{0}$, so every prefix that starts at it has $\mat{F}_{ij} = \mat{0}$.
The $\gvect{\mu}$ and $\gmat{\Sigma}$ of such a prefix are the filtered moments at its end.
So an associative scan returns all $T$ filtered marginals, with $O(\log T)$ span given enough processors.

The square-root form of {cite:t}`yaghoobi2025parallel` carries $\mat{L}_\text{cond}$ and $\mat{L}_\text{lik}$ with $\gmat{\Sigma}_\text{cond} = \mat{L}_\text{cond} \mat{L}_\text{cond}^\top$ and $\gmat{\Lambda}_\text{lik} = \mat{L}_\text{lik} \mat{L}_\text{lik}^\top$.
Each combination triangularises a stack holding an identity,

$$\gmat{\Xi} = \op{tria} \begin{bmatrix} \mat{L}_i^\top \mat{L}_{\text{lik},j} & \mat{I} \end{bmatrix}, \qquad \mat{M} = \mat{L}_i \gmat{\Xi}^{-\top}, \qquad \mat{M} \mat{M}^\top = (\mat{I} + \gmat{\Sigma}_i \gmat{\Lambda}_j)^{-1} \gmat{\Sigma}_i,$$ (eq-exact-xi)

where $\gmat{\Xi}, \mat{M} \in \mathbb{R}^{D_x \times D_x}$.
The factor of [](#eq-exact-combine-covariance) is then $\mat{L}_{ij} = \op{tria}([\mat{F}_j \mat{M}, \mat{L}_j])$.
`gaussx.parallel_kalman_filter(square_root=True)` does exactly this, and `SquareRootKF` is a thin wrapper around it that reports the covariances $\mat{L} \mat{L}^\top$.
The model covariances are factored once, outside the scan, by a Cholesky of their correlation matrix.
That follows a diagonal shift of $4 D \epsilon$, with $D$ the matrix's side and $\epsilon$ the machine epsilon, so a covariance that is singular only by rounding still factors.

## Algorithm

[](#code-exact-algorithm) writes this out in dense numpy for the model above.
It builds the $T$ elements, reduces them from the left and in a balanced tree, and takes one update through [](#eq-exact-psi).
The two reductions must agree with each other, and with the last filtered moments of `SquareRootKF`, because the operation is associative and the elements describe the same posterior.

```{code-block} python
:caption: Elements, their associative combination, and one update by triangularisation, in dense numpy.
:label: code-exact-algorithm
# user
# one element, in the covariance form of the two equations above
def step_element(F, H, Q, R, y):
    innovation = H @ Q @ H.T + R
    gain = np.linalg.solve(innovation, H @ Q).T
    contraction = np.eye(len(F)) - gain @ H
    likelihood_map = H @ F
    weighted = np.linalg.solve(innovation, likelihood_map)
    return (
        contraction @ F,
        gain @ y,
        contraction @ Q,
        likelihood_map.T @ np.linalg.solve(innovation, y),
        likelihood_map.T @ weighted,
    )


# user
# join block i and block j, integrating out the state between them
def combine(first, second):
    F_i, mean_i, cov_i, info_i, prec_i = first
    F_j, mean_j, cov_j, info_j, prec_j = second
    identity = np.eye(len(F_i))
    forward = np.linalg.inv(identity + cov_i @ prec_j)
    backward = np.linalg.inv(identity + prec_j @ cov_i)
    return (
        F_j @ forward @ F_i,
        F_j @ forward @ (mean_i + cov_i @ info_j) + mean_j,
        F_j @ forward @ cov_i @ F_j.T + cov_j,
        F_i.T @ backward @ (info_j - prec_j @ mean_i) + info_i,
        F_i.T @ backward @ prec_j @ F_i + prec_i,
    )


# user
# reduce pairwise, in a balanced tree
def reduce_balanced(elements):
    while len(elements) > 1:
        pairs = zip(elements[0::2], elements[1::2], strict=False)
        joined = [combine(first, second) for first, second in pairs]
        if len(elements) % 2:
            joined.append(elements[-1])
        elements = joined
    return elements[0]


F, H, Q, R = model
data = np.asarray(observations)
# μ_x,1|1 and Σ_x,1|1 from the Joseph filter, for the first element
first_element = (0 * F, dense[0][0], dense[1][0], np.zeros(3), 0 * F)
elements = [first_element] + [step_element(F, H, Q, R, y) for y in data[1:]]

from_left = elements[0]
for element in elements[1:]:
    from_left = combine(from_left, element)
balanced = reduce_balanced(elements)

print(f"{gap(from_left[1], balanced[1]):.0e} {gap(from_left[2], balanced[2]):.0e}")
print(f"{gap(balanced[1], kalman.filtered_means[-1]):.0e}")
print(f"{gap(balanced[2], kalman.filtered_covs[-1]):.0e}")


# user
# L with L Lᵀ = A Aᵀ, from a QR of Aᵀ
def tria(stack):
    upper = np.linalg.qr(stack.T, mode="r")
    signs = np.sign(np.diag(upper))
    return (signs[:, None] * upper).T


# L_x, L_ε
prior_root = np.linalg.cholesky(cov_of_first)
noise_root = np.linalg.cholesky(R)
# Ψ, partitioned at D_y
psi = tria(
    np.block([[H @ prior_root, noise_root], [prior_root, np.zeros((3, 2))]])
)
gain_block, updated_root = psi[2:, :2], psi[2:, 2:]
# G = Ψ₂₁ L_y⁻¹
gain = np.linalg.solve(psi[:2, :2].T, gain_block.T).T
# μ_x,1|1 = μ_x,1|0 + G r₁
residual = data[0] - H @ prior_of_first
print(f"{gap(prior_of_first + gain @ residual, dense[0][0]):.0e}")
print(f"{gap(updated_root @ updated_root.T, dense[1][0]):.0e}")
```

It prints `2e-15 2e-16`, `6e-15`, `2e-15`, `3e-16` and `2e-16`.
The first pair says the two reductions agree on the last mean and covariance.
The next two say they match `SquareRootKF`, and the last two say the update by triangularisation matches the Joseph form.

## Float32 and strong observations

The point of a factor is what happens in float32, and the test of it is the mean.
A covariance can be positive definite while the mean beside it is wrong, and this is not a hypothetical.
Before gaussx v0.6.9 the means of `square_root=True` were up to billions of standard deviations off on the models below, while the covariances still looked healthy, as `tests/test_filters_advanced.py` records.

The cause was the mean shift of [](#eq-exact-combine-conditional).
The push-through identity gives $(\mat{I} + \gmat{\Sigma}_i \gmat{\Lambda}_j)^{-1} \vect{v} = \vect{v} - \mat{M} \mat{M}^\top \gmat{\Lambda}_j \vect{v}$, which subtracts nearly equal vectors when $\gmat{\Sigma}_i \gmat{\Lambda}_j$ is near $1 / \epsilon$.
Applied to $\vect{v} = \gmat{\Sigma}_i \vect{b}_j$, a large vector, the rounding swamps the answer.
By [](#eq-exact-xi) the same quantity has a form with nothing to cancel:

$$(\mat{I} + \gmat{\Sigma}_i \gmat{\Lambda}_j)^{-1} \gmat{\Sigma}_i \vect{b}_j = \mat{M} \mat{M}^\top \vect{b}_j.$$ (eq-exact-shift)

gaussx uses this for the shift and keeps the push-through only for $\gvect{\mu}_i$.
[Chapter 18](18_numerics.md) measures the two forms side by side.

[](#code-exact-float32) runs the constant-velocity model of those tests, with $T = 64$ and data drawn from the model, against float64 `SquareRootKF` as the reference.
It reports the largest mean error in reference standard deviations, the log-likelihood error, and the smallest eigenvalue of the filtered covariances.

```{code-block} python
:caption: Float32 against a float64 reference on three strongly observed models, for the square-root and covariance forms.
:label: code-exact-float32
# Shapes: T = 64 times; D_x = 2; D_y = 1

# F, constant velocity
velocity_model = jnp.array([[1.0, 0.1], [0.0, 1.0]])  # → (D_x, D_x)
# H reads the position
position = jnp.array([[1.0, 0.0]])  # → (D_y, D_x)


# user
# xₜ = F xₜ₋₁ + ηₜ and yₜ = H xₜ + εₜ, with Σ_η = q I, Σ_ε = r I and Σ_x,0 = p I
def strong_data(q: float, r: float, p: float) -> Float[Array, "time obs"]:
    # user
    def advance(state, key):
        process_key, obs_key = jr.split(key)
        state = velocity_model @ state + jnp.sqrt(q) * jr.normal(process_key, (2,))
        return state, position @ state + jnp.sqrt(r) * jr.normal(obs_key, (1,))

    first = jnp.sqrt(p) * jr.normal(jr.key(0), (2,))
    return jax.lax.scan(advance, first, jr.split(jr.key(1), 64))[1]


# user
# the arguments of every solver, in one precision
def strong_inputs(q, r, p, data, dtype):
    return (
        velocity_model.astype(dtype),
        position.astype(dtype),
        (q * jnp.eye(2)).astype(dtype),
        (r * jnp.eye(1)).astype(dtype),
        data.astype(dtype),
        jnp.zeros(2, dtype),
        (p * jnp.eye(2)).astype(dtype),
    )


# today (filterax)
# jitted, so the three models share one compilation per precision
square_root = jax.jit(
    lambda F, H, Q, R, y, m, P: flx.filters.SquareRootKF(F, H, Q, R).filter(y, m, P)
)
# today (gaussx)
covariance_form = jax.jit(gaussx.parallel_kalman_filter)


# today (filterax)
# the sequential smoother, started from the one-step prediction of the prior
@jax.jit
def sequential(F, H, Q, R, y, m, P):
    return flx.KalmanSmoother(F, Q, H, R, F @ m, predicted_cov(F, P, Q)).smooth(y)


for q, r, p in [(1e-8, 1e-6, 1.0), (1e-10, 1e-8, 1.0), (1e-12, 1e-10, 100.0)]:
    data = strong_data(q, r, p)
    reference = square_root(*strong_inputs(q, r, p, data, jnp.float64))
    # √(diag Σ_x,t|t), from the reference
    deviations = jnp.sqrt(jax.vmap(jnp.diag)(reference.filtered_covs))  # → (T, D_x)
    print(f"q={q:g} r={r:g} p={p:g}, reference {reference.log_likelihood:.3f}")
    for name, solver in [
        ("sqrt", square_root),
        ("cov", covariance_form),
        ("seq", sequential),
    ]:
        single = solver(*strong_inputs(q, r, p, data, jnp.float32))
        error = jnp.abs(single.filtered_means - reference.filtered_means) / deviations
        smallest = jnp.linalg.eigvalsh(single.filtered_covs).min()
        print(
            f"  {name}: {jnp.nanmax(error):.2g} sd, "
            f"{single.log_likelihood - reference.log_likelihood:.3g}, {smallest:.1e}"
        )
    double = sequential(*strong_inputs(q, r, p, data, jnp.float64))
    error = jnp.abs(double.filtered_means - reference.filtered_means) / deviations
    print(f"  float64 seq against sqrt: {error.max():.0e} sd")
```

It prints the following, where `sqrt` is `SquareRootKF`, `cov` is the covariance form of `gaussx.parallel_kalman_filter` and `seq` is `KalmanSmoother`, which is also in covariance form:

| $\gmat{\Sigma}_\eta$, $\gmat{\Sigma}_\varepsilon$, $\gmat{\Sigma}_{x,0}$ | Form | Largest mean error (sd) | Log-likelihood error | Smallest eigenvalue |
| --- | --- | --- | --- | --- |
| $10^{-8}$, $10^{-6}$, $1$ | `sqrt` | `0.0034` | `-0.00188` | `7.4e-08` |
| | `cov` | `0.041` | `-0.0304` | `7.4e-08` |
| | `seq` | `0.018` | `0.0154` | `7.4e-08` |
| $10^{-10}$, $10^{-8}$, $1$ | `sqrt` | `0.055` | `-0.016` | `7.4e-10` |
| | `cov` | `1.9` | `-1.17` | `7.4e-10` |
| | `seq` | `1` | `0.818` | `-1.9e-10` |
| $10^{-12}$, $10^{-10}$, $100$ | `sqrt` | `4.3` | `-0.769` | `7.4e-12` |
| | `cov` | `2.2e+03` | `nan` | `-1.2e-03` |
| | `seq` | `2.9` | `nan` | `-1.9e-06` |

The variances are multiples of the identity, and the reference log-likelihoods are `329.432`, `472.193` and `610.348`.
In float64 the two families agree to `6e-11`, `4e-09` and `2e-06` standard deviations, so the reference does not depend on the form.

The factor pays for itself on the first two rows, with mean errors 5 to 35 times smaller than the covariance forms.
It is also the only form that stays positive definite throughout.
It is not exact, though.
On the last row its means are 4.3 standard deviations off, and `tests/test_filters_advanced.py` only bounds that error by 10.
There the sequential filter's means are closer, even though its covariance is indefinite and its log-likelihood is `NaN`.
So at that conditioning use float64 for a reference, whichever form you choose.

## The ensemble against the reference

An ensemble filter on this model has no bias from nonlinearity, so what remains is sampling error, which should fall as $1 / \sqrt{N_\mathcal{E}}$.
[](#code-exact-convergence) measures it.
Each member takes its own draw of $\gvect{\eta}_t$ in the forecast, then `filters.ETKF` or `filters.StochasticEnKF` assimilates, and `EnKS` runs the backward pass.
The numbers are averages over 6 independent runs of the RMS difference from the reference over all components and all $T = 40$ times.

```{code-block} python
:caption: Ensemble filter and smoother errors against the exact reference, as $N_\mathcal{E}$ grows.
:label: code-exact-convergence
# Shapes: T = 40 times; D_x = 3; D_y = 2; N_E members; 6 independent runs

# today (lineax)
# Σ_ε = 0.2 I
diagonal_noise = lx.DiagonalLinearOperator(0.2 * jnp.ones(2))  # (D_y, D_y) operator


# user
# h(x) = H x
def observe(state: Float[Array, " state"]) -> Float[Array, " obs"]:
    return obs_model @ state


# user
# one twin run: the forecast and analysis ensembles, and the summed log-likelihood
def ensemble_run(key, num_members, make_filter):
    start_key, run_key = jr.split(key)
    # x⁽ᵉ⁾ ~ 𝒩(μ_x,0, Σ_x,0)
    start = init_mean + jr.normal(start_key, (num_members, 3))  # → (N_E, D_x)

    # user
    def cycle(members, inputs):
        window_key, observed = inputs
        draw_key, filter_key = jr.split(window_key)
        # x⁽ᵉ⁾ ← F x⁽ᵉ⁾ + η⁽ᵉ⁾
        moved = einx.dot(  # → (N_E, D_x)
            "member state, other state -> member other", members, transition
        )
        draws = jr.normal(draw_key, (num_members, 3))  # → (N_E, D_x)
        forecast = moved + einx.dot(  # → (N_E, D_x)
            "other state, member state -> member other", process_root, draws
        )
        # today (filterax)
        analysis = make_filter(filter_key).analysis(
            forecast, observed, observe, diagonal_noise
        )
        outputs = forecast, analysis.particles, analysis.log_likelihood
        return analysis.particles, outputs

    window_keys = jr.split(run_key, num_steps)  # → (T,)
    return jax.lax.scan(cycle, start, (window_keys, observations))[1]


# user
# RMS difference of the ensemble mean and sample covariance from a reference
def errors_against(history, mean, cov):
    # x̄ₜ
    sample_mean = einx.mean("t [member] state", history)  # → (T, D_x)
    # X̃ₜ
    anomalies = einx.subtract(  # → (T, N_E, D_x)
        "t member state, t state -> t member state", history, sample_mean
    )
    # Σ̂ₓ,ₜ
    sample_cov = einx.dot(  # → (T, D_x, D_x)
        "t member state, t member other -> t state other", anomalies, anomalies
    ) / (history.shape[1] - 1)
    mean_error = jnp.sqrt(jnp.mean((sample_mean - mean) ** 2))  # → ()
    cov_error = jnp.sqrt(jnp.mean((sample_cov - cov) ** 2))  # → ()
    return jnp.stack([mean_error, cov_error])  # → (2,)


run_keys = jr.split(jr.key(10), 6)  # → (6,)
rows = {}
for name, make_filter in [
    ("ETKF", lambda _: flx.filters.ETKF()),
    ("EnKF", lambda key: flx.filters.StochasticEnKF(key=key)),
]:
    for num_members in (10, 40, 160, 640):
        # user
        def score(key):
            history = ensemble_run(key, num_members, make_filter)
            forecast, analysis, log_likelihoods = history
            # today (filterax)
            backward = flx.EnKS().smooth(forecast, analysis).smoothed_history
            smooth_moments = smoothed.smoothed_means, smoothed.smoothed_covs
            return (
                errors_against(analysis, kalman.filtered_means, kalman.filtered_covs),
                errors_against(backward, *smooth_moments),
                jnp.abs(jnp.sum(log_likelihoods) - kalman.log_likelihood),
            )

        filtered, backward, evidence = jax.jit(jax.vmap(score))(run_keys)
        rows[name, num_members] = float(jnp.mean(filtered[:, 0]))
        print(
            f"{name} {num_members:3d}: filter {jnp.mean(filtered[:, 0]):.3f} "
            f"{jnp.mean(filtered[:, 1]):.3f}, smoother {jnp.mean(backward[:, 0]):.3f} "
            f"{jnp.mean(backward[:, 1]):.3f}, evidence {jnp.mean(evidence):.2f}"
        )
# the slope of log error against log N_E, from N_E = 10 to N_E = 640
print(f"{np.log(rows['ETKF', 640] / rows['ETKF', 10]) / np.log(64):.2f}")
```

It prints the mean error and the covariance error for each filter and its smoother, with the absolute error in the summed log-likelihood:

| Filter | $N_\mathcal{E}$ | Filter mean | Filter covariance | Smoother mean | Smoother covariance | $\log p(\vect{y}_{1:T})$ |
| --- | --- | --- | --- | --- | --- | --- |
| `ETKF` | 10 | `0.452` | `0.133` | `0.412` | `0.124` | `7.58` |
| | 40 | `0.204` | `0.064` | `0.214` | `0.048` | `2.22` |
| | 160 | `0.099` | `0.031` | `0.100` | `0.026` | `0.88` |
| | 640 | `0.046` | `0.016` | `0.041` | `0.012` | `0.42` |
| `StochasticEnKF` | 10 | `0.453` | `0.142` | `0.426` | `0.128` | `8.76` |
| | 40 | `0.210` | `0.071` | `0.233` | `0.056` | `2.48` |
| | 160 | `0.105` | `0.035` | `0.103` | `0.029` | `1.21` |
| | 640 | `0.050` | `0.018` | `0.059` | `0.014` | `0.34` |

The last line it prints is `-0.55`, the slope of log error against log $N_\mathcal{E}$ for the `ETKF` filter mean, close to the $-1/2$ that sampling error predicts.
Each fourfold increase in members about halves the error columns, and the evidence error falls at least as fast, so no bias shows at these sizes.
The stochastic filter adds observation perturbations on top of the sampled model noise.
On this model that costs little: `0.210` against `0.204` at $N_\mathcal{E} = 40$.

Read the other way, this sets a tolerance for a test: at $N_\mathcal{E} = 640$ on this model a correct filter sits about `0.05` from the reference.

## When it breaks

The references are exact for one model class, and they can fail you in the ways below.

- **Nonlinear or non-Gaussian.** `SquareRootKF` and `KalmanSmoother` take matrices, so a nonlinear model has no entry point, and the ensemble methods then have no exact reference. Use a grid, as in [chapter 20](20_inversion_example.md).
- **Scale.** The cost is $O(T (D_x^3 + D_y^3))$ with dense $D_x \times D_x$ matrices, which is why they are references for small problems and not solvers for large ones.
- **Float32 at extreme conditioning.** The table above shows every form losing the mean by the last row.
- **What the wrapper leaves out.** `SquareRootKF` has one $\mat{F}$, $\mat{H}$, $\gmat{\Sigma}_\eta$ and $\gmat{\Sigma}_\varepsilon$ for all times and no mask. For either, call `gaussx.parallel_kalman_filter` directly.
- **The prior convention.** Giving both classes the same prior is the mistake to expect, and [](#code-exact-prior) measures it.

```{code-block} python
:caption: Handing `KalmanSmoother` the prior of `SquareRootKF` without predicting it first.
:label: code-exact-prior
# Shapes: T = 40 times; D_x = 3; D_y = 2

# today (filterax)
# the prior of x₀ used as the prior of x₁, which skips one transition
misread = flx.KalmanSmoother(
    transition, process_noise, obs_model, obs_noise, init_mean, init_cov
)
misread_result = jax.jit(lambda y: misread.smooth(y))(observations)
# |μ_x,t|t − μ_x,t|t|, at the first and last times
wrong_filter = jnp.abs(misread_result.filtered_means - kalman.filtered_means)
print(f"{wrong_filter[0].max():.3f} {wrong_filter[-1].max():.0e}")
print(f"{misread_result.log_likelihood - kalman.log_likelihood:.3f}")
```

It prints `0.037 7e-04` and `0.171`.
The filtered mean is off by `0.037` at the first time and the model has nearly forgotten it by the last, while the log-likelihood stays off by `0.171`.
Nothing raises, since both priors are valid.

## When the exact references are the right answer

Use them when the model is linear and Gaussian and you want the answer rather than an estimate of it.
That covers the baseline of a twin experiment, the ground truth of a test for a new filter, and a Gaussian process in time at $O(T)$ through `from_sde`.
Use `SquareRootKF` when the covariance must stay positive definite in float32, or when `gaussx.parallel_kalman_filter`'s scan suits your hardware.
Use `KalmanSmoother` when you need smoothed marginals, missing data or irregular times.

Reach for an ensemble method when the model is nonlinear or $D_x$ is too large for a dense matrix.

## See also

- [Chapter 4](04_kalman_update.md): the Kalman update these classes compute
- [Chapter 6](06_etkf.md): the ETKF, measured here against the reference
- [Chapter 10](10_smoothers.md): the ensemble smoothers, which converge to `KalmanSmoother`
- [Chapter 18](18_numerics.md): float32, cancellation and gradients across the library
- [API: filters](https://jejjohnson.github.io/filterax/reference/filters/) and [smoothers](https://jejjohnson.github.io/filterax/reference/smoothers/)

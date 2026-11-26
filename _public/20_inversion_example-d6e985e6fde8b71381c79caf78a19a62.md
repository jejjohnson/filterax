---
numbering:
  enumerator: "20.%s"
---

# Worked example: a small nonlinear inverse problem

The ensemble Kalman processes of [chapter 9](09_ensemble_kalman_processes.md) solve inverse problems without gradients of the forward model.
They are built on a Gaussian update, so on a nonlinear problem it matters what each one actually returns.
This chapter runs them on a problem small enough to have a grid reference, and measures what each gets right and wrong.

Three questions organise it.
What does each method return next to the true posterior on one nonlinear problem?
Which of its errors are there even when the problem is linear, and can be worked out by hand?
And what happens when the posterior has two modes, which no Gaussian update can represent?

## The problem and its reference

The unknown $\gvect{\theta} \in \mathbb{R}^{D_\theta}$ with $D_\theta = 2$ sets the amplitude and the decay rate of a curve, on a log scale so both stay positive.
It is observed at $D_y = 5$ times $\tau_j$:

$$g_j(\gvect{\theta}) = \exp(\theta_1) \exp\big(-\exp(\theta_2) \tau_j\big), \qquad \vect{y} = g(\gvect{\theta}) + \gvect{\varepsilon}, \qquad \gvect{\varepsilon} \sim \mathcal{N}(\vect{0}, \sigma^2 \mat{I}),$$ (eq-inversion-forward)

with $\sigma = 0.4$ and $\gvect{\theta}_\text{true} = (0.5, -0.5)$.
That noise is large against the curve, so the posterior is visibly non-Gaussian.
The prior is $\gvect{\theta} \sim \mathcal{N}(\vect{0}, \mat{I})$, which gives

$$\log p(\gvect{\theta} \mid \vect{y}) = -\frac{1}{2 \sigma^2} \Vert \vect{y} - g(\gvect{\theta}) \Vert^2 - \frac{1}{2} \Vert \gvect{\theta} \Vert^2 + \text{const}.$$ (eq-inversion-posterior)

In two dimensions this needs no sampling, since [](#code-inversion-reference) evaluates it on a grid of $801 \times 801$ points.
The same block defines the scores used below: the mean, the standard deviations, the correlation and the skewness of each marginal.
It also scores the distance of the mean from the grid's, in posterior standard deviations.
The last score is $\mathrm{KL}(p \Vert q)$, from the grid to a Gaussian $q$ with the method's mean and covariance.

```{code-block} python
:caption: A decay curve with two unknowns, and its posterior on a dense grid.
:label: code-inversion-reference
import einx
import filterax as flx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import optax
from filterax.processes import EKS_Process
from jaxtyping import Array, Float

jax.config.update("jax_enable_x64", True)

# Shapes: D_θ = 2 parameters; D_y = 5 observations; N_E members

# τ, the observation times
times = jnp.array([0.25, 0.5, 1.0, 2.0, 4.0])  # → (D_y,)
noise_std = 0.4


# user
# g(θ) = exp(θ₁) exp(−exp(θ₂) τ)
@jax.jit
def forward(theta: Float[Array, " param"]) -> Float[Array, " obs"]:
    return jnp.exp(theta[0] - jnp.exp(theta[1]) * times)  # → (D_y,)


# y = g(θ_true) + ε
theta_true = jnp.array([0.5, -0.5])  # → (D_θ,)
data = forward(theta_true) + noise_std * jr.normal(jr.key(0), (5,))  # → (D_y,)
# today (lineax)
# Σ_ε = σ² I
noise = lx.DiagonalLinearOperator(noise_std**2 * jnp.ones(5))  # (D_y, D_y) operator
# Σ_θ = I
prior_cov = lx.DiagonalLinearOperator(jnp.ones(2))  # (D_θ, D_θ) operator
prior_mean = jnp.zeros(2)  # → (D_θ,)

# the unnormalised log posterior on a grid of 801 × 801 points, in dense numpy
axis = np.linspace(-4.0, 4.0, 801)
first, second = np.meshgrid(axis, axis, indexing="ij")
curves = np.exp(first[..., None] - np.exp(second[..., None]) * np.asarray(times))
misfit = ((np.asarray(data) - curves) ** 2).sum(-1) / noise_std**2
log_density = -0.5 * misfit - 0.5 * (first**2 + second**2)
density = np.exp(log_density - log_density.max())
density = density / density.sum()
grid = np.stack([first.ravel(), second.ravel()], 1)
weights = density.ravel()


# user
# mean, covariance and marginal skewness of weighted points, in dense numpy
def moments(points, point_weights=None):
    count = len(points)
    if point_weights is None:
        point_weights = np.full(count, 1.0 / count)
    mean = point_weights @ points
    centred = points - mean
    cov = (centred * point_weights[:, None]).T @ centred
    skew = (point_weights @ centred**3) / np.diag(cov) ** 1.5
    return mean, cov, skew


# user
# KL(grid ‖ 𝒩(m, C)), by summation over the grid
def kl_from_grid(mean, cov):
    centred = grid - mean
    quadratic = (centred * np.linalg.solve(cov, centred.T).T).sum(1)
    log_gaussian = -0.5 * (quadratic + np.linalg.slogdet(2.0 * np.pi * cov)[1])
    cell = (axis[1] - axis[0]) ** 2
    positive = weights > 0
    log_ratio = np.log(weights[positive] / cell) - log_gaussian[positive]
    return float(weights[positive] @ log_ratio)


grid_mean, grid_cov, grid_skew = moments(grid, weights)
grid_std = np.sqrt(np.diag(grid_cov))
grid_map = grid[np.argmax(weights)]


# user
# one row of the table: mean, standard deviations, correlation, skewness and KL
def row(label, mean, cov, skew=None):
    std = np.sqrt(np.diag(cov))
    shift = np.abs(mean - grid_mean) / grid_std
    text = f"{label}: mean {np.round(mean, 3)}, sd {np.round(std, 3)}"
    text += f", corr {cov[0, 1] / (std[0] * std[1]):.3f}, off {np.round(shift, 2)} sd"
    if skew is not None:
        text += f", skew {np.round(skew, 2)}"
    print(text + f", KL {kl_from_grid(mean, cov):.3f}")


row("grid", grid_mean, grid_cov, grid_skew)
print(f"MAP {np.round(grid_map, 2)}")
```

It prints:

- `grid: mean [ 0.386 -0.827], sd [0.244 0.529], corr 0.567, off [0. 0.] sd, skew [-0.31 -0.32], KL 0.045`
- `MAP [ 0.45 -0.77]`

The posterior is skewed, so its mean and its mode differ, by about a quarter of a standard deviation in $\theta_1$.
The `KL 0.045` is a floor: it is the distance to the Gaussian with the posterior's own mean and covariance, the best any Gaussian method can do.

## EKI, UKI and EKS on it

All three methods are Layer 2 models with a `run` method, and all start from the prior.

- [`EKI`](https://jejjohnson.github.io/filterax/reference/processes/) {cite:p}`iglesias2013ensemble` moves an ensemble with a tempered Kalman update until the artificial time reaches 1, with steps from `DataMisfitController` {cite:p}`iglesias2021adaptive`. It has no prior term, so the prior enters only through the first ensemble.
- [`UKI`](https://jejjohnson.github.io/filterax/reference/processes/) {cite:p}`huang2022iterated` keeps a mean and a covariance, and evaluates $g$ at $2 D_\theta + 1$ sigma points in each step, with the same scheduler.
- [`EKS`](https://jejjohnson.github.io/filterax/reference/processes/) {cite:p}`garbunoinigo2020interacting` adds a prior drift and ensemble-scaled noise to the EKI update, so that its ensemble samples rather than collapses. We discard the first 100 of its 200 iterations and pool the rest.

`TEKI` {cite:p}`chada2020tikhonov` is here too, as a check on the mode, because it treats the prior as data and targets the MAP.
EKS is the only stochastic method, so it also runs from a second seed, and once with twice the members and `target_stiffness = 0.03` for 1,500 iterations.
Those two use the Layer 1 `EKS_Process` in a `jax.lax.scan`, because the Layer 2 loop is too slow for that many steps.

```{code-block} python
:caption: The Layer 2 process models against the grid posterior.
:label: code-inversion-methods
# Shapes: N_E = 100 members; D_θ = 2; D_y = 5; N_iter iterations

# θ⁽ᵉ⁾ ~ 𝒩(0, I)
start = jr.normal(jr.key(1), (100, 2))  # → (N_E, D_θ)

# today (filterax)
# EKI, with the data misfit controller, stopping at t = 1
eki = flx.EKI(forward_fn=forward, obs=data, noise_cov=noise).run(start)
row("EKI", np.asarray(eki.mean), np.asarray(eki.covariance.as_matrix()))
print(f"  {eki.n_iterations} iterations")

# today (filterax)
# UKI, with 2 D_θ + 1 sigma points in each iteration
uki = flx.UKI(forward_fn=forward, obs=data, noise_cov=noise).run(prior_mean, prior_cov)
row("UKI", np.asarray(uki.mean), np.asarray(uki.covariance.as_matrix()))
print(f"  {uki.n_iterations} iterations")

# today (filterax)
# TEKI, the prior as data, which targets the MAP
teki_config = flx.ProcessConfig(
    scheduler=flx.FixedScheduler(dt=1.0), n_iterations=60
)
teki = flx.TEKI(
    forward_fn=forward,
    obs=data,
    noise_cov=noise,
    prior_mean=prior_mean,
    prior_cov=prior_cov,
    config=teki_config,
).run(start)
print(f"TEKI: mean {np.round(np.asarray(teki.mean), 3)}")

# today (filterax)
# EKS with a prior, 200 iterations, sampling after the first 100
eks_config = flx.ProcessConfig(scheduler=flx.EKSStableScheduler(), n_iterations=200)
# today (filterax)
eks = flx.EKS(
    forward_fn=forward,
    obs=data,
    noise_cov=noise,
    prior_mean=prior_mean,
    prior_cov=prior_cov,
    seed=0,
    config=eks_config,
).run(start)
tail = np.asarray(eks.history_particles[100:]).reshape(-1, 2)
row("EKS", *moments(tail))
print(f"  t = {eks.history_algo_time[-1]:.1f}")


# user
# the same sampler at Layer 1, as one scan, keeping the last half of its iterations
def eks_tail(num_members, target_stiffness, iterations, seed):
    scheduler = flx.EKSStableScheduler(target_stiffness=target_stiffness)
    # today (filterax)
    process = EKS_Process(
        scheduler=scheduler, key=seed, prior_mean=prior_mean, prior_cov=prior_cov
    )
    first_members = jr.normal(jr.key(1 + seed), (num_members, 2))  # → (N_E, D_θ)

    # user
    # one EKS update
    def advance(state, _):
        evaluated = jax.vmap(forward)(state.particles)  # → (N_E, D_y)
        state = process.update(state, evaluated)
        return state, state.particles

    initial = process.init(first_members, data, noise)
    _, history = jax.lax.scan(advance, initial, length=iterations)
    return history[iterations // 2 :]  # → (N_iter / 2, N_E, D_θ)


for label, settings in [
    ("EKS, another seed", (100, 0.1, 200, 1)),
    ("EKS, 200 members, smaller steps", (200, 0.03, 1500, 0)),
]:
    samples = jax.jit(lambda: eks_tail(*settings))()
    row(label, *moments(np.asarray(samples).reshape(-1, 2)))
```

It prints, with each line's iteration count, the rows of this table:

| Method | Mean | Standard deviations | Correlation | Mean off by (sd) | KL | Forward runs |
| --- | --- | --- | --- | --- | --- | --- |
| Grid | `0.386`, `-0.827` | `0.244`, `0.529` | `0.567` | | `0.045` | 641,601 |
| EKI | `0.37`, `-0.832` | `0.353`, `0.578` | `0.593` | `0.07`, `0.01` | `0.168` | 400 |
| UKI | `0.35`, `-0.803` | `0.36`, `0.741` | `0.621` | `0.15`, `0.04` | `0.244` | 15 |
| EKS | `0.459`, `-0.751` | `0.265`, `0.568` | `0.688` | `0.3`, `0.14` | `0.106` | 20,000 |
| EKS, another seed | `0.443`, `-0.773` | `0.259`, `0.595` | `0.694` | `0.23`, `0.1` | `0.097` | 20,000 |
| EKS, 200 members, smaller steps | `0.458`, `-0.786` | `0.254`, `0.586` | `0.696` | `0.29`, `0.08` | `0.127` | 300,000 |

EKI stopped after 4 iterations, UKI after 3, and EKS ran to an artificial time of about `11.5`.
`TEKI` prints a mean of `[ 0.446 -0.744]`, next to the grid's MAP of `[ 0.45 -0.77]`.
EKS's skewness is `[-0.19 -0.19]`, `[0.03 0.06]` and `[ 0.01 -0.1 ]` in the three runs, against the grid's `[-0.31 -0.32]`.

Each method is wrong in its own way.

- **EKI** gets the mean closest, at very little cost, and its spread is not a posterior: the standard deviation of $\theta_1$ is `0.353` against `0.244`.
- **UKI** is the cheapest by far, with the widest covariance, 40% to 50% too wide on both parameters, and it cannot represent skew.
- **EKS** gets the standard deviations closest, within about 13%, and has the lowest KL of the three. Its mean of $\theta_1$ sits at the mode rather than the posterior mean, a quarter to a third of a standard deviation off. Its skewness is too small, and it varies between runs.

The shift in $\theta_1$ is not sampling noise or step size.
The values `0.459`, `0.443` and `0.458` all sit near the MAP of `0.45`, including the run with more members and smaller steps.
The update uses ensemble statistics in place of $\nabla g$, which is exact only for a linear $g$ {cite:p}`garbunoinigo2020interacting`.
That is the likely source, though we have not isolated it.

## Why EKI's spread is not the posterior

A linear $g(\gvect{\theta}) = \mat{H} \gvect{\theta}$ separates errors of nonlinearity from errors of the method, because the posterior is then known in closed form.
With $\mat{A} = \mat{H}^\top \gmat{\Sigma}_\varepsilon^{-1} \mat{H} \in \mathbb{R}^{D_\theta \times D_\theta}$ and a prior of covariance $\gmat{\Sigma}_0$, it has precision $\gmat{\Sigma}_0^{-1} + \mat{A}$.

`filterax.EKI` is deterministic: every member sees the same $\vect{y}$, with no perturbed observations.
One step of size $\Delta t$ updates member $e$ by

$$\gvect{\theta}^{(e)} \leftarrow \gvect{\theta}^{(e)} + \hat{\gmat{\Sigma}}_{\theta h} (\hat{\gmat{\Sigma}}_h + \Delta t^{-1} \gmat{\Sigma}_\varepsilon)^{-1} (\vect{y} - g(\gvect{\theta}^{(e)})).$$ (eq-inversion-eki)

That gives a mean and a covariance with different gains, which is the difference from a Kalman update.
With a single step $\Delta t = 1$ the mean takes the exact Kalman update.
The anomalies are multiplied by $\mat{I} - \hat{\mat{G}} \mat{H}$, so the covariance contracts twice as far as it should.
In the limit of small steps the flow of [](#eq-inversion-eki) is, for a linear $g$,

$$\frac{\mathrm{d} \gmat{\Sigma}}{\mathrm{d} t} = -2 \gmat{\Sigma} \mat{A} \gmat{\Sigma}, \qquad \frac{\mathrm{d} \vect{m}}{\mathrm{d} t} = -\gmat{\Sigma} \mat{A} (\vect{m} - \gvect{\theta}_\text{ls}),$$ (eq-inversion-flow)

with $\vect{m} \in \mathbb{R}^{D_\theta}$ the ensemble mean and $\gvect{\theta}_\text{ls}$ the least-squares solution, and it integrates to

$$\gmat{\Sigma}(1) = (\gmat{\Sigma}_0^{-1} + 2 \mat{A})^{-1}, \qquad \vect{m}(1) = \gvect{\theta}_\text{ls} + (\mat{I} + 2 \gmat{\Sigma}_0 \mat{A})^{-1/2} (\vect{m}_0 - \gvect{\theta}_\text{ls}).$$ (eq-inversion-limit)

You can check this by differentiating: $\gmat{\Sigma}(t) \mat{A} = (\mat{I} + 2 t \gmat{\Sigma}_0 \mat{A})^{-1} \gmat{\Sigma}_0 \mat{A}$ commutes with $\mat{I} + 2 t \gmat{\Sigma}_0 \mat{A}$.
So at $t = 1$ the covariance has counted the data twice, and the mean is not the posterior mean either.
Neither is a matter of ensemble size.

UKI differs in exactly this respect, since it updates the covariance with the Kalman formula.
Each step adds $\Delta t \mat{A}$ to its precision, and the steps sum to 1, so on a linear $g$ it lands on the posterior.

```{code-block} python
:caption: EKI and UKI against the closed-form posterior of a linear problem, with EKI's limits worked out.
:label: code-inversion-linear
# Shapes: N_E = 500 members; D_θ = 2; D_y = 3

# H
linear_map = jnp.array([[1.0, 0.5], [-0.3, 1.2], [0.7, 0.1]])  # → (D_y, D_θ)
linear_data = jnp.array([0.9, -1.4, 0.6])  # → (D_y,)
linear_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(3))  # (D_y, D_y) operator


# user
# g(θ) = H θ
@jax.jit
def linear_forward(theta: Float[Array, " param"]) -> Float[Array, " obs"]:
    return linear_map @ theta


H, y = np.asarray(linear_map), np.asarray(linear_data)
# A = Hᵀ Σ_ε⁻¹ H, and the closed-form posterior under Σ_θ = I
data_precision = H.T @ H / 0.1
exact_cov = np.linalg.inv(np.eye(2) + data_precision)
exact_mean = exact_cov @ H.T @ y / 0.1
linear_start = jr.normal(jr.key(1), (500, 2))  # → (N_E, D_θ)
sample_mean, sample_cov, _ = moments(np.asarray(linear_start))
# θ_ls, the least-squares solution
least_squares = np.linalg.solve(data_precision, H.T @ y / 0.1)
# I + 2 Σ₀ A
stretch = np.eye(2) + 2.0 * sample_cov @ data_precision
values, vectors = np.linalg.eig(stretch)
# (I + 2 Σ₀ A)^(−½)
inverse_root = (vectors / np.sqrt(values)) @ np.linalg.inv(vectors)
# m(1) = θ_ls + (I + 2 Σ₀ A)^(−½) (m₀ − θ_ls)
flow_mean = least_squares + inverse_root @ (sample_mean - least_squares)
# Σ(1) = (Σ₀⁻¹ + 2 A)⁻¹
flow_cov = np.linalg.inv(np.linalg.inv(sample_cov) + 2.0 * data_precision)


# user
# a mean and two standard deviations, rounded
def line(label, mean, cov):
    std = np.sqrt(np.diag(np.asarray(cov)))
    print(f"{label}: mean {np.round(np.asarray(mean), 3)}, sd {np.round(std, 3)}")


line("exact", exact_mean, exact_cov)
line("flow", flow_mean, flow_cov)
for label, scheduler, steps in [
    ("misfit", flx.DataMisfitController(), 50),
    ("Δt = 1", flx.FixedScheduler(dt=1.0), 1),
    ("Δt = 0.01", flx.FixedScheduler(dt=0.01), 100),
]:
    config = flx.ProcessConfig(scheduler=scheduler, n_iterations=steps)
    # today (filterax)
    result = flx.EKI(
        forward_fn=linear_forward,
        obs=linear_data,
        noise_cov=linear_noise,
        config=config,
    ).run(linear_start)
    title = f"EKI {label} ({result.n_iterations})"
    line(title, result.mean, result.covariance.as_matrix())
# today (filterax)
result = flx.UKI(
    forward_fn=linear_forward, obs=linear_data, noise_cov=linear_noise
).run(jnp.zeros(2), prior_cov)
line(f"UKI ({result.n_iterations})", result.mean, result.covariance.as_matrix())
```

It prints:

| | Mean | Standard deviations |
| --- | --- | --- |
| Exact posterior | `1.133`, `-0.782` | `0.246`, `0.237` |
| [](#eq-inversion-limit) | `0.977`, `-0.683` | `0.176`, `0.17` |
| EKI, $\Delta t = 0.01$, 100 steps | `0.978`, `-0.684` | `0.176`, `0.17` |
| EKI, $\Delta t = 1$, 1 step | `1.13`, `-0.786` | `0.062`, `0.057` |
| EKI, data misfit controller, 4 steps | `1.001`, `-0.699` | `0.158`, `0.152` |
| UKI, 4 steps | `1.133`, `-0.782` | `0.246`, `0.237` |

UKI matches the posterior to the digits shown, and small-step EKI matches [](#eq-inversion-limit).
A single EKI step gets the mean right, up to the sampling of its first ensemble, with spread a quarter of what it should be.
The adaptive steps land in between.

So read an EKI ensemble as a point estimate with a convergence diagnostic, and never as a posterior, even on a linear model.
Its spread was too wide on the nonlinear problem above and too narrow here, so it can err either way.

## The same inversion through optax

The same processes are available as `optax` transforms from [`filterax.optax`](https://jejjohnson.github.io/filterax/reference/optax/), which lets an inversion sit in an `optax.chain` or a training loop.
The ensemble lives in the optimiser state, and `params` holds the running mean.
Its `grads` argument is ignored.
The transforms have fewer options than the Layer 2 models: `flx.optax.eks` takes no prior, so it targets the likelihood alone.

```{code-block} python
:caption: EKI as an `optax.GradientTransformation`.
:label: code-inversion-optax
# Shapes: N_E = 100 members; D_θ = 2; D_y = 5

# today (filterax)
# the ensemble lives in the optimiser state, seeded around the params
transform = flx.optax.eki(
    forward_fn=forward,
    obs=data,
    noise_cov=noise,
    n_ensemble=100,
    init_spread=1.0,
    key=1,
)
params = prior_mean  # → (D_θ,)
state = transform.init(params)
updates_done = 0
while state.algo_time < 1.0 - 1e-9:
    # today (optax)
    # grads is ignored: the ensemble supplies the search direction
    updates, state = transform.update(None, state, params)
    params = optax.apply_updates(params, updates)
    updates_done += 1
# today (filterax)
ensemble_cov = flx.ensemble_covariance(state.particles).as_matrix()  # → (D_θ, D_θ)
row("optax EKI", np.asarray(params), np.asarray(ensemble_cov))
print(f"  {updates_done} updates")
```

It prints `optax EKI: mean [ 0.376 -0.812], sd [0.35  0.568], corr 0.599, off [0.04 0.03] sd, KL 0.161` after `4 updates`.
That is the Layer 2 answer up to the first ensemble, which differs because the transform seeds its own draw and recentres it on `params`.

## When it breaks: a posterior with two modes

Take $g(\theta) = \theta^2$ with scalar $\theta$, $y = 1$, noise variance 0.1 and a standard normal prior.
The posterior has two modes near $\pm 1$, and the sign of $\theta$ does not affect the data.
Since $g$ is even, its covariance with $\theta$ under any symmetric belief is zero, so the Kalman update has nothing to work with.
[](#code-inversion-bimodal) measures the mean, the standard deviation and the mass within 0.3 of the origin, where the true posterior has almost none.

```{code-block} python
:caption: A bimodal posterior, against which every method here fails in its own way.
:label: code-inversion-bimodal
# Shapes: N_E = 100 members; D_θ = 1; D_y = 1

# user
# g(θ) = θ²
@jax.jit
def squared(theta: Float[Array, " param"]) -> Float[Array, " obs"]:
    return theta**2


square_data = jnp.array([1.0])  # → (D_y,)
square_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(1))  # (D_y, D_y) operator
line_prior = lx.DiagonalLinearOperator(jnp.ones(1))  # (D_θ, D_θ) operator
# the posterior on a line, in dense numpy
line_axis = np.linspace(-4.0, 4.0, 4001)
line_log = -0.5 * (1.0 - line_axis**2) ** 2 / 0.1 - 0.5 * line_axis**2
line_density = np.exp(line_log - line_log.max())
line_density = line_density / line_density.sum()


# user
# the mean, standard deviation and mass near the origin of weighted points
def summary(label, points, point_weights=None):
    mean, cov, _ = moments(points, point_weights)
    count = len(points)
    if point_weights is None:
        point_weights = np.full(count, 1.0 / count)
    near = point_weights @ (np.abs(points[:, 0]) < 0.3)
    deviation = np.sqrt(cov[0, 0])
    print(f"{label}: mean {mean[0]:.3f}, sd {deviation:.3f}, near 0 {near:.3f}")


summary("grid", line_axis[:, None], line_density)
line_start = jr.normal(jr.key(1), (100, 1))  # → (N_E, D_θ)
# today (filterax)
result = flx.EKI(forward_fn=squared, obs=square_data, noise_cov=square_noise).run(
    line_start
)
summary(f"EKI ({result.n_iterations})", np.asarray(result.particles))
# today (filterax)
line_uki = flx.UKI(forward_fn=squared, obs=square_data, noise_cov=square_noise)
for centre in (0.0, 0.1):
    # today (filterax)
    run = line_uki.run(jnp.array([centre]), line_prior)
    deviation = jnp.sqrt(run.covariance.as_matrix()[0, 0])
    print(f"UKI from {centre}: mean {run.mean[0]:.3f}, sd {deviation:.3f}")
# today (filterax)
sampled = flx.EKS(
    forward_fn=squared,
    obs=square_data,
    noise_cov=square_noise,
    prior_mean=jnp.zeros(1),
    prior_cov=line_prior,
    config=eks_config,
).run(line_start)
summary("EKS", np.asarray(sampled.history_particles[100:]).reshape(-1, 1))
```

It prints:

| Method | Mean | Standard deviation | Mass near 0 |
| --- | --- | --- | --- |
| Grid | `0.000` | `0.939` | `0.010` |
| EKI, 27 iterations | `0.017` | `1.107` | `0.210` |
| EKS | `0.035` | `0.995` | `0.250` |
| UKI from 0 | `0.000` | `1.000` | |
| UKI from 0.1 | `0.105` | `0.967` | |

Every method reports a plausible mean and standard deviation, and that is what makes this case dangerous.
UKI from the prior mean returns the prior exactly, since its sigma points are symmetric, and starting from 0.1 hardly moves it.
EKI and EKS keep members near the origin, 21 and 25 times the true mass, because they fill in the valley between the modes.
Nothing here reports failure, so if the forward model allows several modes, check for them separately.

## When each is the right answer

Use `EKI` for a point estimate when the forward model is expensive and has no gradients, since four iterations of 100 members got closest to the mean here.
Use `TEKI` when you want the MAP under a prior.
Use `UKI` for a Gaussian summary at very low cost on a near-linear problem.
It is exact when linear, and it needs only $2 D_\theta + 1$ runs in each step, which grows with $D_\theta$.
Use `EKS` when you want samples and can afford thousands of forward runs, accepting that its mean leaned toward the mode on the skewed posterior here.
Use a grid or an MCMC sampler when the answer must be right, if the problem is small enough to allow it.

## See also

- [Chapter 9](09_ensemble_kalman_processes.md): the processes and schedulers
- [Chapter 14](14_differentiable.md): differentiating through a filter, which needs gradients of the model
- [Chapter 17](17_exact_references.md): the exact reference for the linear-Gaussian state-space case
- [API: processes](https://jejjohnson.github.io/filterax/reference/processes/), [schedulers](https://jejjohnson.github.io/filterax/reference/schedulers/) and [optax](https://jejjohnson.github.io/filterax/reference/optax/)

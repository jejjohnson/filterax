---
numbering:
  enumerator: "3.%s"
---

# The observation model

The observation model connects the state a filter estimates to the data it actually sees.
An ensemble handles this connection without a Jacobian, by mapping every member through the operator and reading off sample statistics.
This chapter works out what those statistics mean, and how filterax represents the operator and the noise.

Three ideas organise it.
A sample cross-covariance is a statistical linearisation of $h$, which averages the Jacobian over the ensemble.
The observation-error covariance is always an operator, so its structure stays available to the linear algebra.
And stochastic filters need perturbed observations, whose draws follow that structure and whose sampling noise can be stated exactly.

## The measurement equation

Observations are a deterministic map of the state plus additive Gaussian noise:

$$\vect{y} = h(\vect{x}) + \gvect{\varepsilon}, \qquad \gvect{\varepsilon} \sim \mathcal{N}(\vect{0}, \gmat{\Sigma}_\varepsilon),$$ (eq-observation-model)

with $\vect{y} \in \mathbb{R}^{D_y}$, $h : \mathbb{R}^{D_x} \to \mathbb{R}^{D_y}$ and $\gmat{\Sigma}_\varepsilon \in \mathbb{R}^{D_y \times D_y}$.
The likelihood is therefore $p(\vect{y} \mid \vect{x}) = \mathcal{N}(\vect{y}; h(\vect{x}), \gmat{\Sigma}_\varepsilon)$.
The operator can be a selection of grid points, an interpolation stencil or a full radiative-transfer model, and $\mat{H}$ denotes a linear one.
$\gmat{\Sigma}_\varepsilon$ covers more than instrument noise, since it also has to carry the error of representing the measured quantity with the model's state {cite:p}`janjic2018representation`.

## The ensemble in observation space

Mapping each member gives $\vect{h}^{(e)} = h(\vect{x}^{(e)}) \in \mathbb{R}^{D_y}$, and these stack as rows just as the state members do.
Their mean $\bar{\vect{h}} \in \mathbb{R}^{D_y}$ and anomalies $\tilde{\mat{H}} \in \mathbb{R}^{N_\mathcal{E} \times D_y}$ follow [chapter 2](02_ensemble_representation.md).
For a nonlinear $h$ the mean differs from the map of the mean, $\bar{\vect{h}} \ne h(\bar{\vect{x}})$, and filterax always uses $\bar{\vect{h}}$.

The Kalman update needs two products with $\mat{H}$, and the ensemble replaces both, with $c = N_\mathcal{E} - 1$:

$$\hat{\gmat{\Sigma}}_{xy} = \frac{1}{c} \tilde{\mat{X}}^\top \tilde{\mat{H}}, \qquad \hat{\gmat{\Sigma}}_h = \frac{1}{c} \tilde{\mat{H}}^\top \tilde{\mat{H}}.$$ (eq-observation-sample)

For a linear operator, $\tilde{\mat{H}} = \tilde{\mat{X}} \mat{H}^\top$ holds identically, and so nothing is approximated:

$$\hat{\gmat{\Sigma}}_{xy} = \hat{\gmat{\Sigma}}_x \mat{H}^\top, \qquad \hat{\gmat{\Sigma}}_h = \mat{H} \hat{\gmat{\Sigma}}_x \mat{H}^\top.$$ (eq-observation-linear)

```{code-block} python
:caption: The identities [](#eq-observation-linear) on a linear operator.
:label: code-observation-linear
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

# Shapes: N_E = 30 members; D_x = 8; D_y = 3

# X
members = jr.normal(jr.key(0), (30, 8))  # → (N_E, D_x)
# H selects the first three components
obs_map = jnp.eye(3, 8)  # → (D_y, D_x)


# user
# h(x) = H x
def linear_obs(state: Float[Array, " state"]) -> Float[Array, " obs"]:
    return obs_map @ state


# h⁽ᵉ⁾
mapped = jax.vmap(linear_obs)(members)  # → (N_E, D_y)
# today (filterax)
# Σ̂_x, dense because the state is small
state_covariance = flx.cross_covariance(members, members)  # → (D_x, D_x)
# today (filterax)
# Σ̂_xy = X̃ᵀ H̃ / c
cross = flx.cross_covariance(members, mapped)  # → (D_x, D_y)
# today (filterax)
# Σ̂_h = H̃ᵀ H̃ / c
mapped_covariance = flx.cross_covariance(mapped, mapped)  # → (D_y, D_y)
# Σ̂_x Hᵀ
expected_cross = einx.dot(  # → (D_x, D_y)
    "state other, obs other -> state obs", state_covariance, obs_map
)
# H Σ̂_x Hᵀ
expected_mapped = einx.dot(  # → (D_y, D_y)
    "other state, state obs -> other obs", obs_map, expected_cross
)
cross_error = jnp.abs(cross - expected_cross).max()
mapped_error = jnp.abs(mapped_covariance - expected_mapped).max()
print(f"{cross_error:.0e} {mapped_error:.0e}")
```

It prints `0e+00 0e+00`, because this $\mat{H}$ only selects components and so the two sides are the same numbers.

## Statistical linearisation

A nonlinear $h$ has no $\mat{H}$, but the anomalies still define one.
Regress the mapped anomalies on the state anomalies, $\tilde{\vect{h}}^{(e)} = \hat{\mat{H}} \tilde{\vect{x}}^{(e)} + \tilde{\vect{e}}^{(e)}$, with $\hat{\mat{H}} \in \mathbb{R}^{D_y \times D_x}$ chosen by least squares and $\tilde{\vect{e}}^{(e)} \in \mathbb{R}^{D_y}$ the residual.
The normal equations give

$$\hat{\mat{H}} = \hat{\gmat{\Sigma}}_{xy}^\top \hat{\gmat{\Sigma}}_x^{+}, \qquad \hat{\gmat{\Sigma}}_{xy} = \hat{\gmat{\Sigma}}_x \hat{\mat{H}}^\top,$$ (eq-observation-regression)

where $^{+}$ is the pseudo-inverse.
So $\hat{\gmat{\Sigma}}_{xy}$ already plays the role of $\hat{\gmat{\Sigma}}_x \mat{H}^\top$ in [](#eq-observation-linear), and the filter never has to compute $\hat{\mat{H}}$.

What this linearisation means follows from Stein's lemma {cite:p}`stein1981estimation`.
For Gaussian $\vect{x}$ with covariance $\gmat{\Sigma}_x$, and in the limit of many members,

$$\gmat{\Sigma}_{xy} = \gmat{\Sigma}_x \, \mathbb{E}[\mat{J}_h(\vect{x})]^\top,$$ (eq-observation-stein)

with $\mat{J}_h = \partial h / \partial \vect{x} \in \mathbb{R}^{D_y \times D_x}$.
The regression coefficient is therefore the Jacobian averaged over the forecast density.
An extended Kalman filter would use the Jacobian at the mean, which is a different matrix whenever $h$ curves across the ensemble.

The residual is uncorrelated with the state anomalies by construction, so the mapped covariance splits in two:

$$\hat{\gmat{\Sigma}}_h = \hat{\mat{H}} \hat{\gmat{\Sigma}}_x \hat{\mat{H}}^\top + \hat{\gmat{\Sigma}}_e.$$ (eq-observation-residual)

The part of $h$ that no line explains, $\hat{\gmat{\Sigma}}_e$, enters $\hat{\gmat{\Sigma}}_y = \hat{\gmat{\Sigma}}_h + \gmat{\Sigma}_\varepsilon$ exactly like extra observation noise.
Nonlinearity therefore reduces the gain, which is the right instinct as long as the line is a fair description of $h$.

[](#code-observation-stein) checks all of this with $h(\vect{x}) = \tanh(\mat{H} \vect{x})$, on 4000 members with mean $(0.5, -0.2, 1)$ and standard deviation 0.6.

```{code-block} python
:caption: The regression [](#eq-observation-regression) against the averaged Jacobian and the Jacobian at the mean.
:label: code-observation-stein
# Shapes: N_E = 4,000 members; D_x = 3; D_y = 2

# x⁽ᵉ⁾ ~ 𝒩(μ, 0.36 I)
curved_members = jnp.array([0.5, -0.2, 1.0]) + 0.6 * jr.normal(  # → (N_E, D_x)
    jr.key(0), (4000, 3)
)
# H selects components 0 and 2
curved_map = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # → (D_y, D_x)


# user
# h(x) = tanh(H x)
def curved_obs(state: Float[Array, " state"]) -> Float[Array, " obs"]:
    return jnp.tanh(curved_map @ state)


# h⁽ᵉ⁾
curved_mapped = jax.vmap(curved_obs)(curved_members)  # → (N_E, D_y)
# today (filterax)
# Σ̂_x
sample_x = flx.cross_covariance(curved_members, curved_members)  # → (D_x, D_x)
# today (filterax)
# Σ̂_xy
sample_xy = flx.cross_covariance(curved_members, curved_mapped)  # → (D_x, D_y)
# today (filterax)
# Σ̂_h
sample_h = flx.cross_covariance(curved_mapped, curved_mapped)  # → (D_y, D_y)
# Ĥᵀ = Σ̂_x⁻¹ Σ̂_xy
regression_t = jnp.linalg.solve(sample_x, sample_xy)  # → (D_x, D_y)
# Ĥ
regression = einx.id("state obs -> obs state", regression_t)  # → (D_y, D_x)
# J_h(x⁽ᵉ⁾)
jacobians = jax.vmap(jax.jacfwd(curved_obs))(curved_members)  # → (N_E, D_y, D_x)
# 𝔼[J_h]
averaged = einx.mean("[member] obs state", jacobians)  # → (D_y, D_x)
# today (filterax)
# x̄
centre = flx.ensemble_mean(curved_members)  # → (D_x,)
# J_h(x̄)
at_mean = jax.jacfwd(curved_obs)(centre)  # → (D_y, D_x)
print(f"{regression[0, 0]:.3f} {averaged[0, 0]:.3f} {at_mean[0, 0]:.3f}")
print(f"{regression[1, 2]:.3f} {averaged[1, 2]:.3f} {at_mean[1, 2]:.3f}")
# today (filterax)
# h̄ against h(x̄)
print(flx.ensemble_mean(curved_mapped).round(3), curved_obs(centre).round(3))
# Ĥ Σ̂_x
projected = einx.dot(  # → (D_y, D_x)
    "obs state, state other -> obs other", regression, sample_x
)
# Ĥ Σ̂_x Ĥᵀ
explained = einx.dot(  # → (D_y, D_y)
    "obs state, other state -> obs other", projected, regression
)
print(jnp.diag(sample_h).round(4), jnp.diag(explained).round(4))
```

It prints `0.689 0.689 0.802`, `0.462 0.470 0.420`, `[0.357 0.66 ] [0.445 0.761]` and `[0.1831 0.0946] [0.1723 0.0794]`.
The regression sits next to the averaged Jacobian and away from the Jacobian at the mean, as [](#eq-observation-stein) says.
The third line shows that $\bar{\vect{h}}$ and $h(\bar{\vect{x}})$ differ.
The last line is [](#eq-observation-residual): the diagonal of $\hat{\gmat{\Sigma}}_h$ exceeds the explained part, and the difference is $\hat{\gmat{\Sigma}}_e$.

## Observation operators in filterax

An operator in filterax maps one state vector to one observation vector.
The analysis applies it across the members with `jax.vmap(obs_op)(particles)`, so it never sees the ensemble axis.

Two forms are accepted, and they compute the same thing:

| Form | Use it when | Example |
| --- | --- | --- |
| a plain callable | $h$ has no state of its own | `lambda state: state[:3]` |
| a subclass of `flx.AbstractObsOperator` | $h$ carries arrays or trainable parameters | a linear map, a learned network, `flx.LiftedObs` |

`AbstractObsOperator` is an `equinox.Module`, and it requires one method, `__call__(state)`.
Being a pytree is what makes it useful.
Its arrays are leaves, so `eqx.filter_grad` differentiates an analysis with respect to them, which is the mechanism [chapter 14](14_differentiable.md) trains with.
`flx.LiftedObs` composes a decoder with an inner operator, so that a latent filter can observe through it ([chapter 15](15_latent_da.md)).

```{code-block} python
:caption: A parameterised operator, used by a filter and differentiated through the analysis.
:label: code-observation-protocol
# Shapes: N_E = 30 members; D_x = 8; D_y = 3


# user
class ScaledTanhObs(flx.AbstractObsOperator):
    scale: jax.Array  # (D_y,)

    # h(x) = tanh(a ⊙ x₀:₃)
    def __call__(self, state: Float[Array, " state"]) -> Float[Array, " obs"]:
        return jnp.tanh(self.scale * state[:3])


scaled_obs = ScaledTanhObs(jnp.array([1.0, 0.5, 2.0]))
# y
target = jnp.array([0.3, -0.1, 0.2])  # → (D_y,)
# today (lineax)
# Σ_ε = 0.05 I
noise = lx.DiagonalLinearOperator(0.05 * jnp.ones(3))  # (D_y, D_y) operator
# today (filterax)
analysis_step = flx.filters.ETKF()
# today (filterax)
# X_t|t with the module
with_module = analysis_step.analysis(members, target, scaled_obs, noise).particles
# today (filterax)
# X_t|t with an equal plain function
with_function = analysis_step.analysis(
    members, target, lambda state: scaled_obs(state), noise
).particles
print(jax.vmap(scaled_obs)(members).shape, jnp.abs(with_module - with_function).max())


# user
# 𝓛(a) = ∑ sin X_t|t
def loss(operator: ScaledTanhObs) -> Float[Array, ""]:
    # today (filterax)
    analysed = analysis_step.analysis(members, target, operator, noise).particles
    return jnp.sum(jnp.sin(analysed))  # → ()


# ∂𝓛/∂a₀
gradient = eqx.filter_grad(loss)(scaled_obs).scale[0]  # → ()
step = 1e-6
shift = jnp.array([step, 0.0, 0.0])  # → (D_y,)
forward = loss(ScaledTanhObs(scaled_obs.scale + shift))  # → ()
backward = loss(ScaledTanhObs(scaled_obs.scale - shift))  # → ()
print(f"{gradient:.6f} {(forward - backward) / (2.0 * step):.6f}")
```

It prints `(30, 3) 0.0` and `-3.999721 -3.999721`, so the two forms agree exactly and the gradient matches a central difference.

A Layer 1 `analysis` takes `obs_op` per call, so each window can have its own operator.
A Layer 2 model such as `flx.ETKF` holds one `obs_op` for the whole run.
A moving observing network therefore needs a Layer 1 loop, as in [chapter 1](01_problem_setting.md).

## Structured observation noise

filterax takes $\gmat{\Sigma}_\varepsilon$ as a `lineax.AbstractLinearOperator`, so its structure is visible to whatever needs it.
Most often the instrument noise is uncorrelated, and a `lineax.DiagonalLinearOperator` holds it.
Correlated errors, such as spatially correlated satellite retrievals, can use a structured operator.

Two places consume that structure.
One is the draw of perturbed observations below, where gaussx picks a factor $\mat{L}_\varepsilon$ with $\mat{L}_\varepsilon \mat{L}_\varepsilon^\top = \gmat{\Sigma}_\varepsilon$ from the class of the operator.
The other is the Woodbury route of the gain, which solves against $\gmat{\Sigma}_\varepsilon$ with its own structured solve ([chapter 4](04_kalman_update.md)).
The dense route of the gain does not use it: it calls `as_matrix()` and pays for a $D_y \times D_y$ array.

| $\gmat{\Sigma}_\varepsilon$ | Operator | Factor used to sample | Cost |
| --- | --- | --- | --- |
| diagonal | `lineax.DiagonalLinearOperator` | element-wise square root | $O(D_y)$ per draw |
| blocks | `gaussx.BlockDiag` | each block by its own rule | per block |
| Kronecker | `gaussx.Kronecker` | per-factor eigendecomposition | cubic in each factor |
| Toeplitz | `gaussx.Toeplitz` | FFT circulant embedding | $O(D_y \log D_y)$ |
| diagonal plus low rank | `gaussx.LowRankUpdate` | one factor for each part | base cost plus $O(D_y k)$, for rank $k$ |
| anything else | any operator | dense symmetric square root | $O(D_y^3)$ once |

## Perturbed observations

A stochastic filter updates each member against its own noisy copy of the data {cite:p}`burgers1998analysis`:

$$\vect{y}^{(e)} = \vect{y} + \gvect{\varepsilon}^{(e)}, \qquad \gvect{\varepsilon}^{(e)} = \mat{L}_\varepsilon \gvect{\xi}^{(e)}, \qquad \gvect{\xi}^{(e)} \sim \mathcal{N}(\vect{0}, \mat{I}),$$ (eq-observation-perturbed)

which gives the analysis $\vect{x}^{(e)}_{t \mid t} = \vect{x}^{(e)} + \hat{\mat{G}} (\vect{y}^{(e)} - \vect{h}^{(e)})$ with the gain $\hat{\mat{G}} \in \mathbb{R}^{D_x \times D_y}$ of [chapter 4](04_kalman_update.md).

The perturbation is there for the spread.
For a linear $h$, the analysis anomalies are $(\mat{I} - \hat{\mat{G}} \mat{H}) \tilde{\vect{x}}^{(e)} + \hat{\mat{G}} \tilde{\gvect{\varepsilon}}^{(e)}$.
With noise independent of the members, this gives in expectation

$$\hat{\gmat{\Sigma}}_{x \mid y} = (\mat{I} - \hat{\mat{G}} \mat{H}) \hat{\gmat{\Sigma}}_x (\mat{I} - \hat{\mat{G}} \mat{H})^\top + \hat{\mat{G}} \gmat{\Sigma}_\varepsilon \hat{\mat{G}}^\top,$$ (eq-observation-burgers)

which is the correct Kalman covariance ([chapter 4](04_kalman_update.md)).
Without the perturbations the last term is missing, and the ensemble ends up too narrow.

The draws have sampling noise of their own, and filterax does not recentre them.
Their sample mean $\bar{\gvect{\varepsilon}}$ has variance $\gmat{\Sigma}_\varepsilon / N_\mathcal{E}$, and it moves the analysis mean by exactly $\hat{\mat{G}} \bar{\gvect{\varepsilon}}$:

$$\bar{\vect{x}}_{t \mid t} = \bar{\vect{x}} + \hat{\mat{G}} (\bar{\vect{r}} + \bar{\gvect{\varepsilon}}).$$ (eq-observation-mean-shift)

### Algorithm

[](#code-observation-perturbed) writes [](#eq-observation-perturbed) by hand for a diagonal $\gmat{\Sigma}_\varepsilon$, and rebuilds a `StochasticEnKF` analysis from primitives.
It then compares the spread with and without perturbations on one component, with both variances equal to 1.
There [](#eq-observation-burgers) gives $1 / 2$, and its first term alone gives $1 / 4$.

```{code-block} python
:caption: Perturbed observations by hand, a stochastic analysis from primitives, and the spread the perturbations restore.
:label: code-observation-perturbed
# Shapes: N_E = 30 members; D_x = 8; D_y = 3

# σ²
variances = jnp.array([0.05, 0.1, 0.2])  # → (D_y,)
diagonal_noise = lx.DiagonalLinearOperator(variances)  # (D_y, D_y) operator
draw_key = jr.key(7)
# today (filterax)
# y + ε⁽ᵉ⁾
perturbed = flx.perturbed_observations(draw_key, target, diagonal_noise, 30)
# ξ⁽ᵉ⁾
standard = jr.normal(draw_key, (30, 3))  # → (N_E, D_y)
# y + √σ² ⊙ ξ⁽ᵉ⁾
by_hand = target + jnp.sqrt(variances) * standard  # → (N_E, D_y)
print(perturbed.shape, jnp.abs(perturbed - by_hand).max())

# h⁽ᵉ⁾
image = jax.vmap(scaled_obs)(members)  # → (N_E, D_y)
# today (filterax)
# Ĝ
gain = flx.kalman_gain(members, image, diagonal_noise)  # → (D_x, D_y)
# x⁽ᵉ⁾ + Ĝ (y⁽ᵉ⁾ − h⁽ᵉ⁾)
rebuilt = members + einx.dot(  # → (N_E, D_x)
    "state obs, member obs -> member state", gain, perturbed - image
)
# today (filterax)
stochastic = flx.filters.StochasticEnKF(key=draw_key).analysis(
    members, target, scaled_obs, diagonal_noise
)
print(f"{jnp.abs(stochastic.particles - rebuilt).max():.0e}")

# Shapes: N_E = 20,000 members; D_x = D_y = 1
scalar_members = jr.normal(jr.key(8), (20000, 1))  # → (N_E, 1)
# today (lineax)
# Σ_ε = 1
scalar_noise = lx.DiagonalLinearOperator(jnp.ones(1))  # (1, 1) operator
# today (filterax)
# Ĝ
scalar_gain = flx.kalman_gain(  # → (1, 1)
    scalar_members, scalar_members, scalar_noise
)
# today (filterax)
# y + ε⁽ᵉ⁾
scalar_perturbed = flx.perturbed_observations(
    jr.key(9), jnp.zeros(1), scalar_noise, 20000
)
# x⁽ᵉ⁾ + Ĝ (y − x⁽ᵉ⁾), with y = 0
same_y = scalar_members - scalar_gain[0, 0] * scalar_members  # → (N_E, 1)
# x⁽ᵉ⁾ + Ĝ (y⁽ᵉ⁾ − x⁽ᵉ⁾)
own_y = same_y + scalar_gain[0, 0] * scalar_perturbed  # → (N_E, 1)
# today (filterax)
same_variance = flx.cross_covariance(same_y, same_y)[0, 0]  # → ()
# today (filterax)
own_variance = flx.cross_covariance(own_y, own_y)[0, 0]  # → ()
print(f"{same_variance:.3f} {own_variance:.3f}")
```

It prints `(30, 3) 0.0`, `4e-16` and `0.250 0.512`.
The hand-written draw is the same array, and the rebuilt analysis matches `StochasticEnKF` to round-off.
The last line compares a shared $\vect{y}$ with one per member.
The first variance is the too-narrow $1 / 4$, and the second is close to $1 / 2$, with the sampling noise of the draws on top.

### Agreement with the structure

[](#code-observation-structured) draws 20,000 perturbations from each structure, and compares their sample covariance with `as_matrix()`.
It also checks that the two routes of the gain agree on each one.

```{code-block} python
:caption: Sampling from each structured noise operator, and the gain on both routes.
:label: code-observation-structured
# Shapes: S = 20,000 draws; D_y = 4; N_E = 30 members; D_x = 6

# today (lineax)
pair = lx.MatrixLinearOperator(  # (2, 2) operator
    jnp.array([[1.0, 0.5], [0.5, 1.0]]), lx.positive_semidefinite_tag
)
# today (lineax)
scaled_pair = lx.MatrixLinearOperator(  # (2, 2) operator
    jnp.array([[0.2, 0.06], [0.06, 0.2]]), lx.positive_semidefinite_tag
)
# today (lineax)
diagonal = lx.DiagonalLinearOperator(jnp.array([0.1, 0.2, 0.3, 0.4]))
# today (lineax)
base_part = lx.DiagonalLinearOperator(0.1 * jnp.ones(4))
# today (lineax)
dense = lx.MatrixLinearOperator(
    0.2 * jnp.eye(4) + 0.05, lx.positive_semidefinite_tag
)
structures = {
    "diagonal": diagonal,
    # today (gaussx)
    "blocks": gaussx.BlockDiag(pair, lx.DiagonalLinearOperator(jnp.array([0.3, 0.1]))),
    # today (gaussx)
    "Kronecker": gaussx.Kronecker(pair, scaled_pair),
    # today (gaussx)
    "low rank": gaussx.LowRankUpdate(
        base_part, 0.3 * jnp.array([[1.0], [1.0], [0.0], [0.0]])
    ),
    # today (gaussx)
    "Toeplitz": gaussx.Toeplitz(jnp.array([0.4, 0.2, 0.1, 0.05])),
    "dense": dense,
}
state_members = jr.normal(jr.key(2), (30, 6))  # → (N_E, D_x)
observed = state_members[:, :4]  # → (N_E, D_y)
centre_y = jnp.array([1.0, -1.0, 0.5, 0.0])  # → (D_y,)
for name, operator in structures.items():
    # today (filterax)
    draws = flx.perturbed_observations(jr.key(1), centre_y, operator, 20000)
    # today (filterax)
    # Σ̂_ε, from the draws
    empirical = flx.cross_covariance(draws, draws)  # → (D_y, D_y)
    error = jnp.abs(empirical - operator.as_matrix()).max()
    # today (filterax)
    # Ĝ with the matrix formed
    formed = flx.kalman_gain(state_members, observed, operator, dense_innovation=True)
    # today (filterax)
    # Ĝ through Woodbury
    woodbury = flx.kalman_gain(
        state_members, observed, operator, dense_innovation=False
    )
    print(f"{name}: {error:.3f} {jnp.abs(formed - woodbury).max():.0e}")
```

It prints:

| Structure | Largest covariance error | Gain, formed against Woodbury |
| --- | --- | --- |
| diagonal | `0.004` | `9e-16` |
| blocks | `0.020` | `3e-15` |
| Kronecker | `0.002` | `1e-15` |
| low rank | `0.002` | `1e-15` |
| Toeplitz | `0.011` | `9e-16` |
| dense | `0.003` | `8e-16` |

A sample variance $v$ from 20,000 draws has the standard deviation $v \sqrt{2 / 20000} = 0.01 v$, so these errors are of the expected size.
The largest variance is 1, in the blocks case, which also has the largest error.
The errors shrink with more draws: at 400,000 the blocks case gives `0.002` and the Toeplitz case `0.003`.
The two routes of the gain agree to round-off on every structure.

## When it breaks

**A linearisation can be blind.**
With $x \sim \mathcal{N}(0, 1)$ and $h(x) = x^2$, the population covariance $\gmat{\Sigma}_{xy}$ is zero, since $x$ and $x^2$ are uncorrelated.
The averaged Jacobian $\mathbb{E}[2 x]$ is zero as well, so a value of $y$ far from the prior should teach a lot and the update learns nothing.
[](#code-observation-blind) sets $y = 4$ with noise variance 0.01, where the exact posterior has one mode near each of $\pm 2$, and it computes that posterior by quadrature.

```{code-block} python
:caption: A symmetric prior under a quadratic operator gives almost no gain, and the analysis ignores the data.
:label: code-observation-blind
# Shapes: N_E = 1,000 members; D_x = D_y = 1; N_q = 48,001 grid points

prior_members = jr.normal(jr.key(3), (1000, 1))  # → (N_E, 1)
# today (lineax)
# Σ_ε = 0.01
small_noise = lx.DiagonalLinearOperator(jnp.array([0.01]))  # (1, 1) operator
measured = jnp.array([4.0])  # → (1,)


# user
# h(x) = x²
def square(state: Float[Array, " state"]) -> Float[Array, " state"]:
    return state**2


squared = jax.vmap(square)(prior_members)  # → (N_E, 1)
# today (filterax)
# Ĝ
blind_gain = flx.kalman_gain(prior_members, squared, small_noise)  # → (1, 1)
# today (filterax)
# X_t|t
blind = analysis_step.analysis(prior_members, measured, square, small_noise).particles
# today (filterax)
blind_mean = flx.ensemble_mean(blind)[0]  # → ()
# today (filterax)
blind_variance = flx.cross_covariance(blind, blind)[0, 0]  # → ()
# today (filterax)
# h̄ after the analysis
blind_image = flx.ensemble_mean(jax.vmap(square)(blind))[0]  # → ()
blind_summary = f"{blind_mean:.3f} {blind_variance:.3f} {blind_image:.3f}"
print(f"{blind_gain[0, 0]:.3f} {blind_summary}")

# The exact posterior by quadrature, in numpy
grid = np.linspace(-6.0, 6.0, 48001)
log_density = -0.5 * grid**2 - 0.5 * (grid**2 - 4.0) ** 2 / 0.01
weights = np.exp(log_density)
weights = weights / weights.sum()
exact_mean = (weights * grid).sum()
exact_variance = (weights * (grid - exact_mean) ** 2).sum()
exact_image = (weights * grid**2).sum()
print(f"{exact_mean:.3f} {exact_variance:.3f} {exact_image:.3f}")
```

It prints `-0.015 -0.067 1.027 1.031` and `0.000 3.994 3.994`.
The columns are the gain (first line only), then the mean, the variance and the mean of $h$.
The exact posterior has variance near 4 and predicts the measurement, while the analysis keeps the prior's unit variance and still predicts about 1.
The small gain here comes from sampling alone.

Iterating does not rescue a blind linearisation, though it does help a poor one.
[](#code-observation-iterated) repeats the problem with EKI, and then with a one-sided prior $\mathcal{N}(1.5, 0.04)$, where the exact posterior has a single mode.

```{code-block} python
:caption: The ETKF and EKI on the same quadratic operator, with a symmetric and a one-sided prior.
:label: code-observation-iterated
# Shapes: N_E = 1,000 members; D_x = D_y = 1

for name, centre_x, width in (("symmetric", 0.0, 1.0), ("one-sided", 1.5, 0.2)):
    start = centre_x + width * jr.normal(jr.key(3), (1000, 1))  # → (N_E, 1)
    # today (filterax)
    # X_t|t, in one step
    one_step = analysis_step.analysis(start, measured, square, small_noise).particles
    # today (filterax)
    # The EKI loop, with the default adaptive step size
    iterated = flx.EKI(
        forward_fn=square, obs=measured, noise_cov=small_noise
    ).run(start)
    # p(x) ∝ 𝒩(x; m, w²) 𝒩(4; x², 0.01), by quadrature in numpy
    log_density = -0.5 * ((grid - centre_x) / width) ** 2
    log_density = log_density - 0.5 * (grid**2 - 4.0) ** 2 / 0.01
    weights = np.exp(log_density)
    weights = weights / weights.sum()
    # today (filterax)
    step_image = flx.ensemble_mean(jax.vmap(square)(one_step))[0]  # → ()
    # today (filterax)
    eki_image = flx.ensemble_mean(jax.vmap(square)(iterated.particles))[0]  # → ()
    exact_image = (weights * grid**2).sum()
    steps = f"{iterated.n_iterations} steps, {iterated.converged}"
    print(f"{name}: {step_image:.3f} {eki_image:.3f} ({steps}) {exact_image:.3f}")
```

It prints `symmetric: 1.031 1.114 (50 steps, False) 3.994` and `one-sided: 4.207 3.853 (11 steps, True) 3.968`.
The columns are the mean of $h$ after the ETKF, after EKI and under the exact posterior.
The symmetric prior stays near 1 either way, and EKI uses all its iterations without reaching the end of its schedule.
With the one-sided prior, EKI halves the error of the single update.

**A singular noise covariance fails on one route only.**
`perturbed_observations` handles a singular $\gmat{\Sigma}_\varepsilon$, since a zero variance just gives a zero perturbation.
The Woodbury route of the gain solves against $\gmat{\Sigma}_\varepsilon$, so it returns `NaN` without raising, even when $\hat{\gmat{\Sigma}}_y$ is invertible.
That is the default whenever $N_\mathcal{E} < D_y$, and `dense_innovation=True` is the way out.

```{code-block} python
:caption: A noise-free channel with fewer members than observations.
:label: code-observation-singular
# Shapes: N_E = 4 members; D_x = D_y = 6

few = jr.normal(jr.key(0), (4, 6))  # → (N_E, D_x)
# today (lineax)
# Σ_ε with one zero variance
exact_channel = lx.DiagonalLinearOperator(jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]))
# today (filterax)
# the last channel of y + ε⁽ᵉ⁾
print(flx.perturbed_observations(jr.key(1), jnp.zeros(6), exact_channel, 4)[:, 5])
# today (filterax)
print(flx.kalman_gain(few, few, exact_channel)[0, 0])
# today (filterax)
formed_gain = flx.kalman_gain(few, few, exact_channel, dense_innovation=True)
print(f"{formed_gain[0, 0]:.4f}")
```

It prints `[0. 0. 0. 0.]`, `nan` and `0.0919`.

**A repeated key repeats the noise.**
`perturbed_observations` consumes the key it is given, and the same key gives the same draws.
`StochasticEnKF.analysis` falls back to its stored key when none is passed, so a hand-written loop has to pass a fresh key each window.
The Layer 2 `flx.StochasticEnKF` folds a new key per window on its own.

```{code-block} python
:caption: The sample mean of the perturbations, on many channels so its size is stable, and the effect of a repeated key.
:label: code-observation-keys
# Shapes: N_E members; D_y = 1,000 channels

# today (lineax)
# Σ_ε = I
unit_noise = lx.DiagonalLinearOperator(jnp.ones(1000))  # (D_y, D_y) operator
for num_members in (10, 100, 1000):
    # today (filterax)
    # ε⁽ᵉ⁾
    epsilon = flx.perturbed_observations(
        jr.key(1), jnp.zeros(1000), unit_noise, num_members
    )
    # today (filterax)
    # ε̄
    epsilon_mean = flx.ensemble_mean(epsilon)  # → (D_y,)
    rms = jnp.sqrt(jnp.mean(epsilon_mean**2))  # → ()
    print(f"{num_members}: {rms:.4f} {1.0 / np.sqrt(num_members):.4f}")

# today (filterax)
first = flx.perturbed_observations(jr.key(1), target, noise, 5)
# today (filterax)
second = flx.perturbed_observations(jr.key(1), target, noise, 5)
print(jnp.array_equal(first, second))
```

It prints `10: 0.3088 0.3162`, `100: 0.0988 0.1000`, `1000: 0.0306 0.0316` and `True`, so the mean offset follows $1 / \sqrt{N_\mathcal{E}}$ and the repeated key repeats every draw.

## When each form is the right answer

Use a plain callable for a fixed $h$ with nothing to train.
Write an `AbstractObsOperator` when $h$ has arrays you want to differentiate, or to compose it with other modules.
Use a diagonal noise operator unless the errors really are correlated, and then keep the structure rather than forming a dense matrix.
Use `perturbed_observations` directly only when you are writing a stochastic update by hand, since `StochasticEnKF` already draws them.

Think again about a Kalman update at all when $h$ is strongly nonlinear across the ensemble, as the quadratic case showed.
The iterated methods of [chapter 9](09_ensemble_kalman_processes.md) help when the linearisation is merely poor, though not when it is blind.

## See also

- [Chapter 2](02_ensemble_representation.md): the mean and anomaly algebra reused here in observation space
- [Chapter 4](04_kalman_update.md): the gain, built from this chapter's $\tilde{\mat{H}}$ and $\gmat{\Sigma}_\varepsilon$
- [Chapter 5](05_stochastic_enkf.md): the filter that consumes `perturbed_observations`
- [Chapter 15](15_latent_da.md): observing a latent state through `LiftedObs`
- [API: protocols](https://jejjohnson.github.io/filterax/reference/protocols/) and [primitives](https://jejjohnson.github.io/filterax/reference/primitives/)

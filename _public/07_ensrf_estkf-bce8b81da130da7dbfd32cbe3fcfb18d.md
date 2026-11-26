---
numbering:
  enumerator: "7.%s"
---

# EnSRF and ESTKF: the square-root family

The ETKF of [chapter 6](06_etkf.md) is one member of a family of deterministic square-root filters {cite:p}`tippett2003ensemble`.
Each of them reproduces the Kalman analysis covariance on the sample statistics without perturbed observations, and the square-root idea itself goes back to {cite:t}`andrews1968square`.
The members differ in where the square root is taken and in whether the observations arrive as one batch or one at a time.

This chapter covers the three filters filterax ships alongside `ETKF`, and one relation organises it.
The batch `EnSRF` and the `ESTKF` are the ETKF under other names, and in filterax they run the same code.
`EnSRF_Serial` is different code and takes a product of one-observation ETKFs, so it has the same mean and covariance with different members.

## The batch EnSRF: mean from the gain

{cite:t}`whitaker2002ensemble` phrase the update in the most Kalman-familiar way, by treating the mean and the anomalies separately.
With $\tilde{\mat{X}}$ and $\tilde{\mat{H}}$ the anomalies in $\mathbb{R}^{N_\mathcal{E} \times D_x}$ and $\mathbb{R}^{N_\mathcal{E} \times D_y}$, $\bar{\vect{r}} = \vect{y} - \bar{\vect{h}}$ the mean residual and $c = N_\mathcal{E} - 1$, the mean takes the gain and the anomalies take a square root:

$$\bar{\vect{x}}_{t \mid t} = \bar{\vect{x}} + \hat{\mat{G}} \bar{\vect{r}}, \qquad \tilde{\mat{X}}_{t \mid t} = \mat{W} \tilde{\mat{X}}, \qquad \mat{W} = (c \gmat{\Sigma}_w)^{1/2}.$$ (eq-ensrf-batch)

Here $\gmat{\Sigma}_w$ is the weight covariance of [chapter 6](06_etkf.md), and $\mat{W}$ is its symmetric transform.
The push-through identity of that chapter gives $\hat{\mat{G}} = \tilde{\mat{X}}^\top \gmat{\Sigma}_w \tilde{\mat{H}} \gmat{\Sigma}_\varepsilon^{-1}$, so the gain form of the mean and the ETKF's weight form are one expression.
The batch EnSRF and the ETKF are therefore the same filter, and only the phrasing differs.
`filters.EnSRF` calls the same internal function as `filters.ETKF`, and it keeps its name for people who think of the update this way.

## The serial EnSRF: one observation at a time

The classical distinction appears when observations are assimilated one by one.
Let $k \in \{1, \dots, D_y\}$ index the observations, each with a variance $\sigma_{\varepsilon,k}^2 = [\gmat{\Sigma}_\varepsilon]_{kk}$.
At step $k$, let $\tilde{\vect{h}}_k \in \mathbb{R}^{N_\mathcal{E}}$ be column $k$ of the mapped anomalies of the current ensemble, and $\bar{r}_k$ the scalar residual.
The sample variance of the mapped members, the innovation variance and the gain are

$$\hat{\sigma}_{h,k}^2 = \frac{\tilde{\vect{h}}_k^\top \tilde{\vect{h}}_k}{c}, \qquad \hat{\sigma}_{y,k}^2 = \hat{\sigma}_{h,k}^2 + \sigma_{\varepsilon,k}^2, \qquad \hat{\vect{g}}_k = \frac{\tilde{\mat{X}}^\top \tilde{\vect{h}}_k}{c \, \hat{\sigma}_{y,k}^2},$$ (eq-ensrf-scalar-gain)

with $\hat{\vect{g}}_k \in \mathbb{R}^{D_x}$, and no matrix inverse anywhere.
The mean takes the full gain, and the anomalies take a reduced one, scaled by $\beta_k \in (1/2, 1)$:

$$\bar{\vect{x}} \leftarrow \bar{\vect{x}} + \hat{\vect{g}}_k \bar{r}_k, \qquad \tilde{\mat{X}} \leftarrow \tilde{\mat{X}} - \beta_k \tilde{\vect{h}}_k \hat{\vect{g}}_k^\top.$$ (eq-ensrf-scalar-update)

### Why the gain is reduced

The factor $\beta_k$ follows from asking for the Kalman covariance.
Dropping the index and writing $\hat{\gmat{\Sigma}}_{xh} = \tilde{\mat{X}}^\top \tilde{\vect{h}} / c = \hat{\sigma}_y^2 \hat{\vect{g}}$, the anomalies in [](#eq-ensrf-scalar-update) have sample covariance

$$\hat{\gmat{\Sigma}}_x - \left(2 \beta \hat{\sigma}_y^2 - \beta^2 \hat{\sigma}_h^2\right) \hat{\vect{g}} \hat{\vect{g}}^\top.$$ (eq-ensrf-reduced-covariance)

The Kalman covariance is $\hat{\gmat{\Sigma}}_x - \hat{\vect{g}} \hat{\gmat{\Sigma}}_{xh}^\top = \hat{\gmat{\Sigma}}_x - \hat{\sigma}_y^2 \hat{\vect{g}} \hat{\vect{g}}^\top$.
So $\beta$ must solve $\hat{\sigma}_h^2 \beta^2 - 2 \hat{\sigma}_y^2 \beta + \hat{\sigma}_y^2 = 0$.
With $\hat{\sigma}_h^2 = \hat{\sigma}_y^2 - \sigma_\varepsilon^2$, the root below one is

$$\beta = \frac{\hat{\sigma}_y}{\hat{\sigma}_y + \sigma_\varepsilon} = \frac{1}{1 + \sqrt{\sigma_\varepsilon^2 / \hat{\sigma}_y^2}}.$$ (eq-ensrf-beta)

The full gain $\beta = 1$ is the unperturbed update of [chapter 5](05_stochastic_enkf.md), which is short by $\sigma_\varepsilon^2 \hat{\vect{g}} \hat{\vect{g}}^\top$.
Reducing the gain is the deterministic way to supply that missing spread.

### Each step is a one-observation ETKF

Substituting $\hat{\vect{g}}_k$ into [](#eq-ensrf-scalar-update) shows that the anomalies are multiplied on the left by a symmetric matrix:

$$\tilde{\mat{X}} \leftarrow \mat{W}_k \tilde{\mat{X}}, \qquad \mat{W}_k = \mat{I} - \frac{\beta_k}{c \, \hat{\sigma}_{y,k}^2} \tilde{\vect{h}}_k \tilde{\vect{h}}_k^\top.$$ (eq-ensrf-step-transform)

Its eigenvalues are 1 on the complement of $\tilde{\vect{h}}_k$, and on $\tilde{\vect{h}}_k$ itself they are $1 - \beta_k \hat{\sigma}_{h,k}^2 / \hat{\sigma}_{y,k}^2 = \sigma_{\varepsilon,k} / \hat{\sigma}_{y,k}$.

The ETKF with this one observation has $\gmat{\Lambda}_w = c \mat{I} + \tilde{\vect{h}}_k \tilde{\vect{h}}_k^\top / \sigma_{\varepsilon,k}^2$.
Its transform $\sqrt{c} \gmat{\Lambda}_w^{-1/2}$ has eigenvalue 1 on the complement, and on $\tilde{\vect{h}}_k$ the precision eigenvalue is $c \hat{\sigma}_{y,k}^2 / \sigma_{\varepsilon,k}^2$, which gives $\sigma_{\varepsilon,k} / \hat{\sigma}_{y,k}$ again.
So $\mat{W}_k$ is exactly the symmetric ETKF transform for one observation, and $\hat{\vect{g}}_k \bar{r}_k$ is its mean update.

A full sweep is then a product,

$$\tilde{\mat{X}}_{t \mid t} = \mat{W}_{D_y} \cdots \mat{W}_2 \mat{W}_1 \tilde{\mat{X}}.$$ (eq-ensrf-product)

Each factor fixes $\vect{1}$, since $\tilde{\vect{h}}_k^\top \vect{1} = 0$, so the product is mean-preserving as [chapter 6](06_etkf.md) requires.
But a product of symmetric matrices is not symmetric in general, so this transform is not the batch $\mat{W}$.
By the sequential form of Bayes' rule with diagonal noise, its mean and covariance still equal the batch values for a linear $h$.
Only the members differ, and they depend on the order of the sweep.

## ESTKF: the error subspace

{cite:t}`nerger2012unification` observe that the ETKF's $N_\mathcal{E}$-dimensional weight space is one dimension too big, because $\vect{1}^\top \tilde{\mat{X}} = \vect{0}$.
The ESTKF works in the complement of $\vect{1}$.
Choose $\mat{L} \in \mathbb{R}^{N_\mathcal{E} \times (N_\mathcal{E} - 1)}$ with $\mat{L}^\top \mat{L} = \mat{I}$ and $\mat{L}^\top \vect{1} = \vect{0}$, which implies $\mat{L} \mat{L}^\top = \mat{I} - \vect{1} \vect{1}^\top / N_\mathcal{E}$.
The reduced anomalies and reduced precision are

$$\mat{X}_L = \mat{L}^\top \tilde{\mat{X}}, \qquad \mat{H}_L = \mat{L}^\top \tilde{\mat{H}}, \qquad \gmat{\Lambda}_L = c \mat{I} + \mat{H}_L \gmat{\Sigma}_\varepsilon^{-1} \mat{H}_L^\top,$$ (eq-ensrf-reduced-precision)

with $\mat{X}_L \in \mathbb{R}^{c \times D_x}$, $\mat{H}_L \in \mathbb{R}^{c \times D_y}$ and $\gmat{\Lambda}_L \in \mathbb{R}^{c \times c}$.
The analysis uses the mean weights $\bar{\vect{w}}_L \in \mathbb{R}^c$ and the symmetric root $\mat{W}_L \in \mathbb{R}^{c \times c}$:

$$\bar{\vect{w}}_L = \gmat{\Lambda}_L^{-1} \mat{H}_L \gmat{\Sigma}_\varepsilon^{-1} \bar{\vect{r}}, \qquad \mat{W}_L = (c \gmat{\Lambda}_L^{-1})^{1/2},$$ (eq-ensrf-reduced-weights)

$$\bar{\vect{x}}_{t \mid t} = \bar{\vect{x}} + \mat{X}_L^\top \bar{\vect{w}}_L, \qquad \tilde{\mat{X}}_{t \mid t} = \mat{L} \mat{W}_L \mat{X}_L.$$ (eq-ensrf-reduced-analysis)

### Why it equals the ETKF

Since $\tilde{\mat{H}}^\top \vect{1} = \vect{0}$, the vector $\vect{1}$ is an eigenvector of $\gmat{\Lambda}_w$ with eigenvalue $c$, and $\mat{L}$ spans the remaining ones.
So the precision splits as

$$\gmat{\Lambda}_w = \frac{c}{N_\mathcal{E}} \vect{1} \vect{1}^\top + \mat{L} \gmat{\Lambda}_L \mat{L}^\top,$$ (eq-ensrf-split-precision)

and any spectral function splits in the same way, $\phi(\gmat{\Lambda}_w) = \phi(c) \vect{1} \vect{1}^\top / N_\mathcal{E} + \mat{L} \phi(\gmat{\Lambda}_L) \mat{L}^\top$.
Applied to $\tilde{\mat{X}}$ the first term vanishes, because $\vect{1}^\top \tilde{\mat{X}} = \vect{0}$, which gives $\mat{W} \tilde{\mat{X}} = \mat{L} \mat{W}_L \mat{L}^\top \tilde{\mat{X}}$ and the same result for the mean.
This is not an approximation, and it is the sense in which the two filters are the same analysis.
So `filters.ESTKF` builds no $\mat{L}$ and decomposes no $c \times c$ matrix.
It runs the ETKF route of [chapter 6](06_etkf.md), including its rank-$D_y$ shortcut, and keeps the name for people coming from the error-subspace literature.

## Algorithm

[](#code-ensrf-algorithm) writes out two forms the production code does not contain as such.
They are the reduced ESTKF, with $\mat{L}$ from a Householder reflection, and the serial sweep with its explicit step transforms.

```{code-block} python
:caption: The reduced ESTKF and the serial sweep written out in full, for a linear map and a diagonal noise covariance.
:label: code-ensrf-algorithm
import einx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

jax.config.update("jax_enable_x64", True)


# user
def estkf_by_hand(
    members: Float[Array, "member state"],
    mapped: Float[Array, "member obs"],
    observation: Float[Array, " obs"],
    variances: Float[Array, " obs"],
) -> Float[Array, "member state"]:
    # Shapes: N_E members; c = N_E − 1 modes; D_x state; D_y observations
    num_members = members.shape[0]
    # c = N_E − 1
    prior_term = num_members - 1.0
    # v = 𝟏 / √N_E − e₀, the axis of the reflection that sends 𝟏 / √N_E to e₀
    axis = jnp.ones(num_members) / jnp.sqrt(num_members)  # → (N_E,)
    axis = axis.at[0].add(-1.0)  # → (N_E,)
    axis = axis / jnp.linalg.norm(axis)  # → (N_E,)
    # I − 2 v vᵀ
    reflection = jnp.eye(num_members) - 2.0 * einx.dot(  # → (N_E, N_E)
        "member, other -> member other", axis, axis
    )
    # L, its last c columns
    basis = reflection[:, 1:]  # → (N_E, c)
    # x̄
    state_mean = einx.mean("[member] state", members)  # → (D_x,)
    # X̃
    state_anomalies = members - state_mean  # → (N_E, D_x)
    # h̄
    mapped_mean = einx.mean("[member] obs", mapped)  # → (D_y,)
    # r̄ = y − h̄
    residual = observation - mapped_mean  # → (D_y,)
    # X_L = Lᵀ X̃
    reduced_state = einx.dot(  # → (c, D_x)
        "member mode, member state -> mode state", basis, state_anomalies
    )
    # H_L = Lᵀ H̃
    reduced_mapped = einx.dot(  # → (c, D_y)
        "member mode, member obs -> mode obs", basis, mapped - mapped_mean
    )
    # H_L Σ_ε⁻¹
    whitened = reduced_mapped / variances  # → (c, D_y)
    # Λ_L = c I + H_L Σ_ε⁻¹ H_Lᵀ
    precision = prior_term * jnp.eye(num_members - 1) + einx.dot(  # → (c, c)
        "mode obs, other obs -> mode other", whitened, reduced_mapped
    )
    # Λ_L = Φ diag(γ) Φᵀ
    eigenvalues, eigenvectors = jnp.linalg.eigh(precision)
    # W_L = Φ diag(√(c / γ)) Φᵀ
    root = einx.dot(  # → (c, c)
        "mode axis, other axis -> mode other",
        eigenvectors * jnp.sqrt(prior_term / eigenvalues),
        eigenvectors,
    )
    # w̄_L = Λ_L⁻¹ H_L Σ_ε⁻¹ r̄
    mean_weights = jnp.linalg.solve(  # → (c,)
        precision, einx.dot("mode obs, obs -> mode", whitened, residual)
    )
    # x̄_t|t = x̄ + X_Lᵀ w̄_L
    analysis_mean = state_mean + einx.dot(  # → (D_x,)
        "mode, mode state -> state", mean_weights, reduced_state
    )
    # X̃_t|t = L W_L X_L
    analysis_anomalies = einx.dot(  # → (N_E, D_x)
        "member mode, mode other, other state -> member state",
        basis,
        root,
        reduced_state,
    )
    return analysis_mean + analysis_anomalies


# user
def serial_by_hand(
    members: Float[Array, "member state"],
    mapped: Float[Array, "member obs"],
    observation: Float[Array, " obs"],
    variances: Float[Array, " obs"],
) -> tuple[Array, Array]:
    # Shapes: N_E members; D_x state; D_y observations
    num_members = members.shape[0]
    # c = N_E − 1
    prior_term = num_members - 1.0
    # x̄
    state_mean = einx.mean("[member] state", members)  # → (D_x,)
    # X̃
    state_anomalies = members - state_mean  # → (N_E, D_x)
    # h̄
    mapped_mean = einx.mean("[member] obs", mapped)  # → (D_y,)
    # H̃
    mapped_anomalies = mapped - mapped_mean  # → (N_E, D_y)
    # the running product of the step transforms
    total = jnp.eye(num_members)  # → (N_E, N_E)
    for index in range(mapped.shape[1]):
        # h̃ₖ
        column = mapped_anomalies[:, index]  # → (N_E,)
        # σ̂²_y,k = h̃ₖᵀ h̃ₖ / c + σ²_ε,k
        innovation = column @ column / prior_term + variances[index]  # → ()
        # ĝₖ = X̃ᵀ h̃ₖ / (c σ̂²_y,k)
        gain = einx.dot("member state, member -> state", state_anomalies, column)
        gain = gain / (prior_term * innovation)  # → (D_x,)
        # the same gain for the mapped mean and anomalies
        mapped_gain = einx.dot("member obs, member -> obs", mapped_anomalies, column)
        mapped_gain = mapped_gain / (prior_term * innovation)  # → (D_y,)
        # r̄ₖ
        step_residual = observation[index] - mapped_mean[index]  # → ()
        # x̄ ← x̄ + ĝₖ r̄ₖ
        state_mean = state_mean + gain * step_residual  # → (D_x,)
        mapped_mean = mapped_mean + mapped_gain * step_residual  # → (D_y,)
        # βₖ = 1 / (1 + √(σ²_ε,k / σ̂²_y,k))
        beta = 1.0 / (1.0 + jnp.sqrt(variances[index] / innovation))  # → ()
        # Wₖ = I − βₖ h̃ₖ h̃ₖᵀ / (c σ̂²_y,k)
        step = jnp.eye(num_members) - beta / (prior_term * innovation) * einx.dot(
            "member, other -> member other", column, column
        )  # → (N_E, N_E)
        # X̃ ← Wₖ X̃, and likewise H̃
        state_anomalies = step @ state_anomalies  # → (N_E, D_x)
        mapped_anomalies = step @ mapped_anomalies  # → (N_E, D_y)
        total = step @ total  # → (N_E, N_E)
    return state_mean + state_anomalies, total
```

Applying $\mat{W}_k$ to $\tilde{\mat{H}}$ as well is what the linear book-keeping looks like; the production code gets the same quantity by re-evaluating $h$.

| Filter | Square root taken in | Observations | Noise covariance | Cost of one analysis |
| --- | --- | --- | --- | --- |
| `ETKF`, `EnSRF`, `ESTKF` | rank-$D_y$ route, $D_y < N_\mathcal{E}$ | batch | any lineax operator | $O(N_\mathcal{E} D_y^2 + N_\mathcal{E} D_y D_x)$ |
| `ETKF`, `EnSRF`, `ESTKF` | ensemble space, otherwise | batch | any lineax operator | $O(N_\mathcal{E}^2 D_y + N_\mathcal{E}^3 + N_\mathcal{E}^2 D_x)$ |
| `EnSRF_Serial` | one scalar per observation | one at a time | `lx.DiagonalLinearOperator` only | $O(N_\mathcal{E} D_x D_y)$, plus $D_y$ evaluations of $h$ on all members |

Each serial step evaluates the whole of $h$ on every member, reads one component and discards the rest.
So when $h$ itself costs about $D_y$ per member, that term is $O(N_\mathcal{E} D_y^2)$.

## Implementation in filterax

All of these share the `analysis(particles, obs, obs_op, obs_noise)` signature, so comparing them is one line each.
[`filters.EnSRF`, `filters.ESTKF`](https://jejjohnson.github.io/filterax/reference/filters/) and [`filters.EnSRF_Serial`](https://jejjohnson.github.io/filterax/reference/filters_advanced/) are Layer 1 analysis steps.
Layer 2 has `filterax.EnSRF` for the cycle of [chapter 16](16_assimilation_cycle.md), and no cycle class for `ESTKF` or `EnSRF_Serial`, so those run in a hand-written loop such as [](#code-ensrf-twin).

The example has $N_\mathcal{E} = 30$ members with $D_x = 4$, and three linear observations with different noise variances.

```{code-block} python
:caption: The four filters on one linear problem. Three of them return the same array.
:label: code-ensrf-analysis
import filterax as flx
import jax.random as jr
import lineax as lx

# Shapes: N_E = 30 members; D_x = 4 state components; D_y = 3 observations

# x⁽ᵉ⁾ ~ 𝒩(𝟏, 0.25 I)
members = 1.0 + 0.5 * jr.normal(jr.key(0), (30, 4))  # → (N_E, D_x)
# H
obs_map = jnp.array(  # → (D_y, D_x)
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]
)
# y
observation = jnp.array([0.4, -0.1, 0.7])  # → (D_y,)
variances = jnp.array([0.1, 0.2, 0.5])  # → (D_y,)
# today (lineax)
# Σ_ε = diag(σ²_ε)
noise = lx.DiagonalLinearOperator(variances)  # (D_y, D_y) operator


# user
# h(x) = H x
def observe(state: Float[Array, " state"]) -> Float[Array, " obs"]:
    return obs_map @ state


# today (filterax)
etkf = flx.filters.ETKF().analysis(members, observation, observe, noise)
# today (filterax)
ensrf = flx.filters.EnSRF().analysis(members, observation, observe, noise)
# today (filterax)
estkf = flx.filters.ESTKF().analysis(members, observation, observe, noise)
# today (filterax)
serial = flx.filters.EnSRF_Serial().analysis(members, observation, observe, noise)


# user
# the largest gaps in the mean, the covariance and the members
def gaps(first: Float[Array, "member state"], second: Float[Array, "member state"]):
    mean_gap = jnp.abs(  # → ()
        einx.mean("[member] state", first) - einx.mean("[member] state", second)
    ).max()
    # today (filterax)
    first_covariance = flx.ensemble_covariance(first).as_matrix()  # → (D_x, D_x)
    # today (filterax)
    second_covariance = flx.ensemble_covariance(second).as_matrix()  # → (D_x, D_x)
    covariance_gap = jnp.abs(first_covariance - second_covariance).max()  # → ()
    member_gap = jnp.abs(first - second).max()  # → ()
    return f"{mean_gap:.0e} {covariance_gap:.0e} {member_gap:.3f}"


print(jnp.array_equal(etkf.particles, ensrf.particles))
print(jnp.array_equal(etkf.particles, estkf.particles))
print(gaps(serial.particles, etkf.particles))
print(f"{etkf.log_likelihood:.6f} {serial.log_likelihood:.6f}")
```

It prints `True`, `True`, `4e-16 1e-16 0.028` and `-4.358415 -4.358415`.
`EnSRF` and `ESTKF` return the `ETKF` array bit for bit.
`EnSRF_Serial` matches the mean and the covariance to round-off, and its members differ.
All four report the same evidence, computed from the forecast ensemble before the sweep.

## Agreement with the references

Three identities above can be measured: [](#eq-ensrf-reduced-analysis) against the production ESTKF, [](#eq-ensrf-product) against the production serial sweep, and the claim that one serial step is a one-observation ETKF.
The last check compares the anomalies through the explicit transform, and reverses the sweep to see the order dependence.

```{code-block} python
:caption: The written-out forms against the production filters, in float64.
:label: code-ensrf-agreement
# h⁽ᵉ⁾ = H x⁽ᵉ⁾
mapped = jax.vmap(observe)(members)  # → (N_E, D_y)
reduced = estkf_by_hand(members, mapped, observation, variances)  # → (N_E, D_x)
swept, product = serial_by_hand(members, mapped, observation, variances)
print(f"{jnp.abs(reduced - etkf.particles).max():.0e}")
print(f"{jnp.abs(swept - serial.particles).max():.0e}")
# ‖W − Wᵀ‖ for the product, and W 𝟏 − 𝟏
transposed = einx.id("member other -> other member", product)  # → (N_E, N_E)
print(f"{jnp.abs(product - transposed).max():.3f}")
print(f"{jnp.abs(product @ jnp.ones(30) - 1.0).max():.0e}")

# today (lineax)
# the first observation alone
first_noise = lx.DiagonalLinearOperator(variances[:1])  # (1, 1) operator
# today (filterax)
one_batch = flx.filters.ETKF().analysis(
    members, observation[:1], lambda state: observe(state)[:1], first_noise
)
# today (filterax)
one_serial = flx.filters.EnSRF_Serial().analysis(
    members, observation[:1], lambda state: observe(state)[:1], first_noise
)
print(f"{jnp.abs(one_batch.particles - one_serial.particles).max():.0e}")

# the sweep in the reverse order
order = jnp.array([2, 1, 0])  # → (D_y,)
# today (filterax)
backwards = flx.filters.EnSRF_Serial().analysis(
    members,
    observation[order],
    lambda state: observe(state)[order],
    lx.DiagonalLinearOperator(variances[order]),
)
print(gaps(backwards.particles, serial.particles))
```

It prints `2e-15`, `2e-15`, `0.009`, `7e-16`, `9e-16` and `2e-16 3e-17 0.059`.
The reduced form and the written-out sweep match the production filters to round-off.
The product transform is not symmetric, by 0.009, and it still fixes $\vect{1}$.
One serial step is the one-observation ETKF, member for member.
Reversing the order keeps the mean and the covariance and moves the members.

## Numerics

### Float32

The serial sweep never forms a matrix with the whole range of $\gmat{\Sigma}_\varepsilon^{-1}$ in it, because each step meets one variance alone.
The ensemble-space route of the batch filters does form one, and that is its accuracy limit in float32 when $D_y \ge N_\mathcal{E}$ ([gaussx#785](https://github.com/jejjohnson/gaussx/issues/785)).
[](#code-ensrf-float32) measures the mean increment and the analysis covariance against a float64 dense solve, with observations processed in the given order.

```{code-block} python
:caption: Float32 analyses against a float64 dense reference, with noise variances spanning twelve decades.
:label: code-ensrf-float32
import numpy as np


# user
# x̄ + Ĝ r̄ and Σ̂_x − Ĝ Σ̂_xyᵀ, by a dense float64 solve, for h(x) = x
def dense_reference(ensemble, target, noise_variances):
    mean = ensemble.mean(0)
    anomalies = ensemble - mean
    prior = anomalies.T @ anomalies / (len(ensemble) - 1)
    gain = np.linalg.solve(prior + np.diag(noise_variances), prior).T
    return gain @ (target - mean), prior - gain @ prior


# user
# the relative errors of the mean increment and the covariance
def float32_errors(chosen, ensemble, target, noise_variances):
    increment, covariance = dense_reference(ensemble, target, noise_variances)
    # today (filterax)
    analysis = chosen.analysis(
        jnp.asarray(ensemble, jnp.float32),
        jnp.asarray(target, jnp.float32),
        lambda state: state,
        lx.DiagonalLinearOperator(jnp.asarray(noise_variances, jnp.float32)),
    ).particles
    analysis = np.asarray(analysis, float)
    mean_error = analysis.mean(0) - ensemble.mean(0) - increment
    covariance_error = np.cov(analysis, rowvar=False) - covariance
    return (
        f"{np.abs(mean_error).max() / np.abs(increment).max():.0e}",
        f"{np.abs(covariance_error).max() / np.abs(covariance).max():.0e}",
    )


rng = np.random.default_rng(1)
for num_members, num_obs, noise_variances in [
    (10, 3, np.array([1e-6, 1.0, 1e6])),
    (3, 3, np.array([1e-6, 1.0, 1e6])),
    (5, 8, np.logspace(-6, 6, 8)),
    (10, 3, np.ones(3)),
]:
    ensemble = rng.normal(size=(num_members, num_obs))
    target = rng.normal(size=num_obs)
    # today (filterax)
    batch = float32_errors(flx.filters.ETKF(), ensemble, target, noise_variances)
    # today (filterax)
    sweep = float32_errors(
        flx.filters.EnSRF_Serial(), ensemble, target, noise_variances
    )
    print(num_members, num_obs, batch, sweep)
```

It prints:

| $N_\mathcal{E}$ | $D_y$ | Variances | `ETKF` mean | `ETKF` covariance | `EnSRF_Serial` mean | `EnSRF_Serial` covariance |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 3 | $10^{-6}$ to $10^{6}$ | `1e-07` | `2e-07` | `1e-07` | `1e-07` |
| 3 | 3 | $10^{-6}$ to $10^{6}$ | `6e-02` | `2e-02` | `8e-08` | `1e-07` |
| 5 | 8 | $10^{-6}$ to $10^{6}$ | `1e-02` | `4e-03` | `1e-07` | `4e-07` |
| 10 | 3 | all 1 | `2e-07` | `3e-07` | `1e-07` | `4e-08` |

With $D_y < N_\mathcal{E}$ both are accurate, since the batch filters then take the rank-$D_y$ route.
With $D_y \ge N_\mathcal{E}$ and a wide range of variances, `EnSRF_Serial` holds float32 accuracy where the ETKF route does not.
For a diagonal noise covariance, that makes it the practical answer to that limit when float64 is not an option.

### Gradients

The serial sweep has no eigendecomposition, so it has no repeated eigenvalue to worry about.
Its derivative is that of a chain of rational functions and scalar square roots.
Differentiating with respect to a zero noise variance is the one point to avoid, where $\sqrt{\sigma_\varepsilon^2}$ in [](#eq-ensrf-beta) has an infinite slope.
The batch filters use the closed-form derivative of [chapter 6](06_etkf.md).

```{code-block} python
:caption: Gradients with respect to the noise scale, against central differences.
:label: code-ensrf-gradient
# user
# 𝓛(s) = ∑ sin X_t|t, with Σ_ε = s diag(σ²_ε)
def loss_for(chosen):
    def loss(scale: Float[Array, ""]) -> Float[Array, ""]:
        # today (lineax)
        scaled = lx.DiagonalLinearOperator(scale * variances)  # (D_y, D_y) operator
        # today (filterax)
        analysis = chosen.analysis(members, observation, observe, scaled)
        return jnp.sum(jnp.sin(analysis.particles))  # → ()

    return loss


at = jnp.asarray(1.0)
step = 1e-6
for chosen in (flx.filters.EnSRF_Serial(), flx.filters.ESTKF()):
    loss = loss_for(chosen)
    central = (loss(at + step) - loss(at - step)) / (2.0 * step)  # → ()
    print(f"{jax.grad(loss)(at):.6f} {central:.6f}")
```

It prints `9.223319 9.223319` and `9.219315 9.219315`.
The two filters give different values because their members differ, and each loss is a function of the members.

## When it breaks

These filters share the limits of [chapter 6](06_etkf.md): the update stays in the span of the forecast anomalies, and a small ensemble underestimates its own error.
Removing the sampling noise does not cure that.
[](#code-ensrf-twin) runs the Lorenz-96 twin of that chapter {cite:p}`lorenz2006predictability`, with $D_x = 40$, every second site observed with unit variance, and no inflation or localization.

```{code-block} python
:caption: The batch and serial filters on Lorenz-96, which share their failure at small ensemble size.
:label: code-ensrf-twin
# Shapes: T = 200 windows; D_x = 40 sites; D_y = 20 observed sites; N_E members


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
_, truth = jax.lax.scan(advance, spun_up, length=200)  # → (T, D_x)
# H selects every second site
sites = jnp.arange(0, 40, 2)  # → (D_y,)
# yₜ = H xₜ + εₜ, εₜ ~ 𝒩(0, I)
observed = truth[:, sites] + jr.normal(jr.key(1), (200, 20))  # → (T, D_y)
unit_noise = lx.DiagonalLinearOperator(jnp.ones(20))  # (D_y, D_y) operator


# user
# RMSE and spread over the second half of the run, with no cycle class needed
def run_twin(chosen, num_members: int) -> tuple[Array, Array]:
    # x⁽ᵉ⁾ ~ 𝒩(x₀, 4 I)
    start = spun_up + 2.0 * jr.normal(jr.key(2), (num_members, 40))  # → (N_E, D_x)

    def window(particles, target):
        # X_t|t−1 = f(X_t−1|t−1)
        forecast = jax.vmap(rk4)(particles)  # → (N_E, D_x)
        # today (filterax)
        analysis = chosen.analysis(
            forecast, target, lambda state: state[sites], unit_noise
        ).particles
        return analysis, analysis

    _, history = jax.lax.scan(window, start, observed)  # → (T, N_E, D_x)
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


for num_members in (20, 40, 60):
    for chosen in (flx.filters.ETKF(), flx.filters.EnSRF_Serial()):
        rmse, mean_spread = run_twin(chosen, num_members)
        print(num_members, type(chosen).__name__, f"{rmse:.2f} {mean_spread:.2f}")
```

It prints:

| $N_\mathcal{E}$ | `ETKF` RMSE | `ETKF` spread | `EnSRF_Serial` RMSE | `EnSRF_Serial` spread |
| --- | --- | --- | --- | --- |
| 20 | `2.37` | `0.26` | `3.17` | `0.27` |
| 40 | `0.31` | `0.25` | `0.31` | `0.25` |
| 60 | `0.30` | `0.26` | `0.30` | `0.26` |

Once the ensemble is big enough, the two scores nearly coincide.
At 20 members both diverge, with error far above spread, and the different members lead them to different chaotic trajectories.

Two more limits belong to the serial form.

**Correlated noise is refused, and whitening is the route around that.**
Processing one component of a correlated $\vect{y}$ at a time would be wrong, so `EnSRF_Serial` raises `NotImplementedError` for anything but a bare `lx.DiagonalLinearOperator`.
With $\gmat{\Sigma}_\varepsilon = \mat{L}_\varepsilon \mat{L}_\varepsilon^\top$, assimilating $\mat{L}_\varepsilon^{-1} \vect{y}$ through $\mat{L}_\varepsilon^{-1} h$ with unit variances is the same problem.

**A nonlinear $h$ makes the order matter.**
The production code re-evaluates $h$ on the running ensemble, so each step takes a fresh statistical linearisation.
That is a different approximation from the batch one, and it depends on the order of the sweep.

```{code-block} python
:caption: Whitening a correlated noise covariance, and the order dependence under a nonlinear map.
:label: code-ensrf-limits
# Shapes: N_E = 30; D_x = 4; D_y = 3

# Σ_ε, correlated
correlated = jnp.array([[0.3, 0.1, 0.0], [0.1, 0.4, 0.1], [0.0, 0.1, 0.5]])
# today (lineax)
dense_noise = lx.MatrixLinearOperator(correlated, lx.positive_semidefinite_tag)
try:
    # today (filterax)
    flx.filters.EnSRF_Serial().analysis(members, observation, observe, dense_noise)
except NotImplementedError as error:
    print(type(error).__name__)
# Σ_ε = L_ε L_εᵀ
factor = jnp.linalg.cholesky(correlated)  # → (D_y, D_y)


# user
# L_ε⁻¹ h(x)
def whitened_observe(state: Float[Array, " state"]) -> Float[Array, " obs"]:
    # today (jax)
    return jax.scipy.linalg.solve_triangular(factor, observe(state), lower=True)


# today (jax)
# L_ε⁻¹ y
whitened_target = jax.scipy.linalg.solve_triangular(factor, observation, lower=True)
# today (filterax)
whitened_sweep = flx.filters.EnSRF_Serial().analysis(
    members,
    whitened_target,
    whitened_observe,
    lx.DiagonalLinearOperator(jnp.ones(3)),
)
# today (filterax)
correlated_batch = flx.filters.ETKF().analysis(
    members, observation, observe, dense_noise
)
print(gaps(whitened_sweep.particles, correlated_batch.particles))


# user
# a nonlinear h
def curved(state: Float[Array, " state"]) -> Float[Array, " obs"]:
    return jnp.stack([state[0] ** 2, jnp.sin(state[2]), state[1] * state[2]])


# today (filterax)
curved_batch = flx.filters.ETKF().analysis(members, observation, curved, noise)
# today (filterax)
curved_sweep = flx.filters.EnSRF_Serial().analysis(members, observation, curved, noise)
# today (filterax)
curved_backwards = flx.filters.EnSRF_Serial().analysis(
    members,
    observation[order],
    lambda state: curved(state)[order],
    lx.DiagonalLinearOperator(variances[order]),
)
print(gaps(curved_sweep.particles, curved_batch.particles))
print(gaps(curved_backwards.particles, curved_sweep.particles))
```

It prints `NotImplementedError`, `3e-16 6e-17 0.019`, `2e-02 3e-02 0.103` and `1e-01 2e-02 0.134`.
The whitened sweep matches the batch filter with the correlated covariance, to round-off in the mean and the covariance.
Under the nonlinear map, the serial mean differs from the batch mean by `2e-02` and from its own reversed sweep by `1e-01`.
So here the order matters more than the choice of serial over batch.

## When each is the right answer

Use `ETKF`, `EnSRF` or `ESTKF` as your default deterministic batch filter, and pick the name your community uses, since the analysis is the same.
They take any structured noise covariance, and `ESTKF` is the natural name when cross-checking against error-subspace experiments such as those run with PDAF.

Use `EnSRF_Serial` in three cases:

- The noise is diagonal, $D_y \ge N_\mathcal{E}$ and float32 is required, by [](#code-ensrf-float32).
- Observations arrive incrementally, so that each one can be assimilated as it comes.
- You want no eigendecomposition anywhere in the derivative.

Use `ETKF_Livings` of [chapter 6](06_etkf.md) for long runs where the deterministic transform leaves outlying members.
None of these replaces the treatments for a small ensemble: use [chapter 8](08_letkf.md) and [chapter 12](12_inflation.md) for those.

## See also

- [Chapter 5](05_stochastic_enkf.md): the stochastic EnKF, and the term the reduced gain replaces
- [Chapter 6](06_etkf.md): the transform, its derivative and its float32 behaviour, shared by the batch filters here
- [Chapter 8](08_letkf.md): the localized transform, which runs one ETKF per grid point
- [Chapter 12](12_inflation.md): why square-root filters still lose spread over cycles
- [API: filters](https://jejjohnson.github.io/filterax/reference/filters/) and [advanced filters](https://jejjohnson.github.io/filterax/reference/filters_advanced/)

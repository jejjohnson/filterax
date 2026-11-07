---
numbering:
  enumerator: "6.%s"
---

# ETKF: the ensemble transform Kalman filter

The stochastic EnKF ([chapter 5](05_stochastic_enkf.md)) gets the analysis covariance right only in expectation, because every member sees its own perturbed observation and those draws add sampling noise to the update.
The ETKF {cite:p}`bishop2001adaptive` removes the sampling step: it computes one deterministic transform of the forecast ensemble, whose sample mean and sample covariance match the Kalman analysis exactly.

Three ideas make it work, and they organise this chapter.
The update can be written entirely in the $N_\mathcal{E}$-dimensional ensemble space.
Only the symmetric square root of the weight covariance keeps the ensemble centred on the Kalman mean.
And the matrix being decomposed has a repeated eigenvalue and a huge dynamic range, so the way it is computed decides whether gradients and float32 survive.

## The update in ensemble space

Take the forecast ensemble $\mat{X} \in \mathbb{R}^{N_\mathcal{E} \times D_x}$ with members as rows, its mean $\bar{\vect{x}} \in \mathbb{R}^{D_x}$ and its anomalies $\tilde{\mat{X}} = \mat{X} - \vect{1} \bar{\vect{x}}^\top$.
Map each member through the observation operator, $\vect{h}^{(e)} = h(\vect{x}^{(e)})$, to get the mean $\bar{\vect{h}} \in \mathbb{R}^{D_y}$, the anomalies $\tilde{\mat{H}} \in \mathbb{R}^{N_\mathcal{E} \times D_y}$ and the mean residual $\bar{\vect{r}} = \vect{y} - \bar{\vect{h}}$.
The sample covariances are then $\hat{\gmat{\Sigma}}_{xy} = \tilde{\mat{X}}^\top \tilde{\mat{H}} / c$ and $\hat{\gmat{\Sigma}}_h = \tilde{\mat{H}}^\top \tilde{\mat{H}} / c$, where $c = N_\mathcal{E} - 1$.

The Kalman update moves the state only within the span of the forecast anomalies.
So rather than work with $D_x \times D_x$ covariances, write the analysis as a recombination of the forecast members and solve for the weights.
Define the precision of the weights in ensemble space,

$$\gmat{\Lambda}_w = c \mat{I} + \tilde{\mat{H}} \gmat{\Sigma}_\varepsilon^{-1} \tilde{\mat{H}}^\top, \qquad \gmat{\Sigma}_w = \gmat{\Lambda}_w^{-1},$$ (eq-etkf-precision)

both $N_\mathcal{E} \times N_\mathcal{E}$.
The mean weights $\bar{\vect{w}} \in \mathbb{R}^{N_\mathcal{E}}$ and the transform $\mat{W} \in \mathbb{R}^{N_\mathcal{E} \times N_\mathcal{E}}$ are

$$\bar{\vect{w}} = \gmat{\Sigma}_w \tilde{\mat{H}} \gmat{\Sigma}_\varepsilon^{-1} \bar{\vect{r}}, \qquad \mat{W} = (c \gmat{\Sigma}_w)^{1/2},$$ (eq-etkf-weights)

and the analysis ensemble is assembled from the forecast anomalies alone:

$$\bar{\vect{x}}_{t \mid t} = \bar{\vect{x}} + \tilde{\mat{X}}^\top \bar{\vect{w}}, \qquad \tilde{\mat{X}}_{t \mid t} = \mat{W} \tilde{\mat{X}}.$$ (eq-etkf-analysis)

### Why this is the Kalman update

The gain follows from a push-through identity.
For $\hat{\gmat{\Sigma}}_y = \hat{\gmat{\Sigma}}_h + \gmat{\Sigma}_\varepsilon$,

$$\hat{\mat{G}} = \hat{\gmat{\Sigma}}_{xy} \hat{\gmat{\Sigma}}_y^{-1} = \tilde{\mat{X}}^\top \gmat{\Sigma}_w \tilde{\mat{H}} \gmat{\Sigma}_\varepsilon^{-1},$$ (eq-etkf-gain)

which can be checked by multiplying $\tilde{\mat{H}} \hat{\gmat{\Sigma}}_y^{-1} / c = \gmat{\Sigma}_w \tilde{\mat{H}} \gmat{\Sigma}_\varepsilon^{-1}$ by $\gmat{\Lambda}_w$ on the left and $\hat{\gmat{\Sigma}}_y$ on the right, where both sides become $\tilde{\mat{H}} \gmat{\Sigma}_\varepsilon^{-1} \hat{\gmat{\Sigma}}_y$.
So $\bar{\vect{x}} + \hat{\mat{G}} \bar{\vect{r}}$ is the mean in [](#eq-etkf-analysis).

The covariance follows in the same way.
The Kalman analysis covariance is $\hat{\gmat{\Sigma}}_x - \hat{\mat{G}} \hat{\gmat{\Sigma}}_{xy}^\top$, and substituting [](#eq-etkf-gain) gives

$$\hat{\gmat{\Sigma}}_{x \mid y} = \frac{1}{c} \tilde{\mat{X}}^\top \left(\mat{I} - \gmat{\Sigma}_w \tilde{\mat{H}} \gmat{\Sigma}_\varepsilon^{-1} \tilde{\mat{H}}^\top\right) \tilde{\mat{X}} = \tilde{\mat{X}}^\top \gmat{\Sigma}_w \tilde{\mat{X}},$$ (eq-etkf-covariance)

since $\tilde{\mat{H}} \gmat{\Sigma}_\varepsilon^{-1} \tilde{\mat{H}}^\top = \gmat{\Lambda}_w - c \mat{I}$.
The anomalies $\mat{W} \tilde{\mat{X}}$ have sample covariance $\tilde{\mat{X}}^\top \mat{W}^\top \mat{W} \tilde{\mat{X}} / c$, so any $\mat{W}$ with $\mat{W}^\top \mat{W} = c \gmat{\Sigma}_w$ reproduces [](#eq-etkf-covariance) exactly, with no randomness anywhere.
For a nonlinear $h$ the same algebra holds with the sample covariances, which is a statistical linearisation of $h$ over the ensemble and not the exact posterior.

## Why the symmetric square root

The condition $\mat{W}^\top \mat{W} = c \gmat{\Sigma}_w$ leaves a rotation free, and not every choice gives a valid ensemble.
The analysis anomalies must stay centred, $\vect{1}^\top \tilde{\mat{X}}_{t \mid t} = \vect{0}$, otherwise the sample mean of the analysis drifts away from the Kalman mean and the filter is biased {cite:p}`livings2008unbiased`.

Anomalies sum to zero, $\tilde{\mat{H}}^\top \vect{1} = \vect{0}$, so the constant vector is an eigenvector of [](#eq-etkf-precision), with $\gmat{\Lambda}_w \vect{1} = c \vect{1}$.
The symmetric square root keeps that eigenvector, and its eigenvalue is one:

$$\mat{W} \vect{1} = (c \gmat{\Sigma}_w)^{1/2} \vect{1} = \vect{1}.$$ (eq-etkf-mean-preserving)

Then $\vect{1}^\top \mat{W} \tilde{\mat{X}} = (\mat{W} \vect{1})^\top \tilde{\mat{X}} = \vect{1}^\top \tilde{\mat{X}} = \vect{0}$, so the symmetric root maps centred anomalies to centred anomalies.
A Cholesky factor has no such property, and neither does a general $\mat{W}$ {cite:p}`tippett2003ensemble,wang2004which`.

### Breaking the symmetry: ETKF_Livings

The symmetric transform is deterministic, and over many cycles it can leave the ensemble with a few outlying members and non-Gaussian higher moments.
The rotation that remains free can be spent on this.
Compose the transform with a random rotation $\mat{Q} \in \mathbb{R}^{N_\mathcal{E} \times N_\mathcal{E}}$ that fixes the constant vector,

$$\tilde{\mat{X}}_{t \mid t} = \mat{Q} \mat{W} \tilde{\mat{X}}, \qquad \mat{Q}^\top \mat{Q} = \mat{I}, \qquad \mat{Q} \vect{1} = \vect{1}.$$ (eq-etkf-rotation)

Applied on the left, $\mat{Q}$ leaves the sample covariance alone, because $(\mat{Q} \mat{W})^\top \mat{Q} \mat{W} = \mat{W}^\top \mat{W}$, and $\mat{Q}^\top \vect{1} = \vect{1}$ keeps the anomalies centred.
[`ETKF_Livings`](https://jejjohnson.github.io/filterax/reference/filters_advanced/) draws it uniformly in the $(N_\mathcal{E} - 1)$-dimensional complement of $\vect{1}$, from the QR factor of a Gaussian matrix lifted back with a Householder basis.
Use a fresh `key` every cycle, since repeating one rotation defeats the purpose.

## Few observations: the rank-$D_y$ route

The excess of $\gmat{\Lambda}_w$ over $c \mat{I}$ has rank at most $D_y$.
So when $D_y < N_\mathcal{E}$ there is no need for an $N_\mathcal{E} \times N_\mathcal{E}$ decomposition: write

$$\tilde{\mat{H}} \gmat{\Sigma}_\varepsilon^{-1} \tilde{\mat{H}}^\top = \mat{U} \op{diag}(\gvect{\lambda}) \mat{U}^\top,$$ (eq-etkf-low-rank)

with $\mat{U} \in \mathbb{R}^{N_\mathcal{E} \times R}$ orthonormal, $\gvect{\lambda} \in \mathbb{R}^R$ non-negative and $R \le D_y$.
A thin QR of $\tilde{\mat{H}}$ gives a basis, and a $D_y \times D_y$ eigendecomposition finishes the job.
Every eigenvalue on the complement of $\mat{U}$ equals $c$, so any function of $\gmat{\Lambda}_w$ needs only these factors:

$$\phi(\gmat{\Lambda}_w) = \phi(c) \mat{I} + \mat{U} \op{diag}\big(\phi(c + \gvect{\lambda}) - \phi(c)\big) \mat{U}^\top.$$ (eq-etkf-spectral-function)

With $\phi(\gamma) = 1 / \gamma$ this applies $\gmat{\Sigma}_w$, and with $\phi(\gamma) = \gamma^{-1/2}$ it applies the root in $\tilde{\mat{X}}_{t \mid t} = \sqrt{c} \gmat{\Lambda}_w^{-1/2} \tilde{\mat{X}}$.
Both are matrix-vector products, so no $N_\mathcal{E} \times N_\mathcal{E}$ array exists on this route.
[`gaussx.etkf_analysis`](https://jejjohnson.github.io/gaussx/) takes it by default, and `filters.ETKF` calls that function.

| Route | When | Cost of one analysis | Forms |
| --- | --- | --- | --- |
| Rank-$D_y$ | $D_y < N_\mathcal{E}$ | $O(N_\mathcal{E} D_y^2 + N_\mathcal{E} D_y D_x)$ | $D_y \times D_y$ |
| Ensemble-space | otherwise | $O(N_\mathcal{E}^2 D_y + N_\mathcal{E}^3 + N_\mathcal{E}^2 D_x)$ | $N_\mathcal{E} \times N_\mathcal{E}$ |

Both routes apply $\gmat{\Sigma}_\varepsilon^{-1}$ through one structured solve, so a diagonal, block-diagonal or Kronecker noise covariance never becomes dense.

## Algorithm

[](#code-etkf-algorithm) writes out the ensemble-space route on the problem of the next section, using the dense quantities the production route avoids.

```{code-block} python
:caption: The update [](#eq-etkf-precision)--[](#eq-etkf-analysis) written out in full, for a dense $\gmat{\Sigma}_\varepsilon$.
:label: code-etkf-algorithm
import einx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

jax.config.update("jax_enable_x64", True)


# user
def etkf_by_hand(
    members: Float[Array, "member state"],
    mapped: Float[Array, "member obs"],
    observation: Float[Array, " obs"],
    noise_covariance: Float[Array, "obs obs"],
) -> Float[Array, "member state"]:
    # Shapes: N_E members; D_x state components; D_y observations
    num_members = members.shape[0]
    # c = N_E − 1
    prior_term = num_members - 1.0
    # x̄
    state_mean = einx.mean("[member] state", members)  # → (D_x,)
    # X̃ = X − 𝟏 x̄ᵀ
    state_anomalies = members - state_mean  # → (N_E, D_x)
    # h̄
    mapped_mean = einx.mean("[member] obs", mapped)  # → (D_y,)
    # H̃
    mapped_anomalies = mapped - mapped_mean  # → (N_E, D_y)
    # r̄ = y − h̄
    residual = observation - mapped_mean  # → (D_y,)
    # Σ_ε⁻¹ H̃ᵀ
    whitened = jnp.linalg.solve(  # → (D_y, N_E)
        noise_covariance, einx.id("member obs -> obs member", mapped_anomalies)
    )
    # Λ_w = c I + H̃ Σ_ε⁻¹ H̃ᵀ
    precision = prior_term * jnp.eye(num_members) + einx.dot(  # → (N_E, N_E)
        "member obs, obs other -> member other", mapped_anomalies, whitened
    )
    # Λ_w = Φ diag(γ) Φᵀ
    eigenvalues, eigenvectors = jnp.linalg.eigh(precision)

    # user
    # ϕ(Λ_w) = Φ diag(ϕ(γ)) Φᵀ
    def spectral(values: Float[Array, " mode"]) -> Float[Array, "member other"]:
        scaled = einx.multiply(  # → (N_E, N_E)
            "member mode, mode -> member mode", eigenvectors, values
        )
        return einx.dot(  # → (N_E, N_E)
            "member mode, other mode -> member other", scaled, eigenvectors
        )

    # Σ_w, with ϕ(γ) = 1 / γ
    weight_covariance = spectral(1.0 / eigenvalues)  # → (N_E, N_E)
    # W, with ϕ(γ) = √(c / γ)
    transform = spectral(jnp.sqrt(prior_term / eigenvalues))  # → (N_E, N_E)
    # H̃ Σ_ε⁻¹ r̄
    projected = einx.dot("obs member, obs -> member", whitened, residual)  # → (N_E,)
    # w̄ = Σ_w H̃ Σ_ε⁻¹ r̄
    mean_weights = einx.dot(  # → (N_E,)
        "member other, other -> member", weight_covariance, projected
    )
    # x̄_t|t = x̄ + X̃ᵀ w̄
    analysis_mean = state_mean + einx.dot(  # → (D_x,)
        "member, member state -> state", mean_weights, state_anomalies
    )
    # X̃_t|t = W X̃
    analysis_anomalies = einx.dot(  # → (N_E, D_x)
        "member other, other state -> member state", transform, state_anomalies
    )
    # X_t|t = x̄_t|t + X̃_t|t
    return analysis_mean + analysis_anomalies
```

## Implementation in filterax

[`filterax.filters.ETKF`](https://jejjohnson.github.io/filterax/reference/filters/) is the Layer 1 analysis step: one analysis ensemble from one observation vector.
The example has $N_\mathcal{E} = 30$ members of a state with $D_x = 4$, drawn around 1 with standard deviation 0.5, and observes components 0 and 2 with noise variance 0.1.

```{code-block} python
:caption: One analysis step, which tightens the observed components and leaves the others nearly alone.
:label: code-etkf-analysis
import filterax as flx
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
# X_t|t, and log 𝒩(r̄; 𝟎, Σ̂_y)
result = flx.filters.ETKF().analysis(members, observation, observe, noise)
print(spread(members).round(3), spread(result.particles).round(3))
print(f"{result.log_likelihood:.3f}")
```

It prints `[0.43  0.528 0.531 0.522] [0.255 0.516 0.272 0.511]` and `-3.401`.
The observed components contract and the others move only through their sample correlation with them, which is why components 1 and 3 change a little.

`result.log_likelihood` is the Gaussian evidence of the residual under $\hat{\gmat{\Sigma}}_y$, which [chapter 13](13_diagnostics.md) uses for diagnostics.
The Layer 2 `filterax.ETKF` wraps this step with dynamics, inflation and the multi-window `assimilate` loop of [chapter 16](16_assimilation_cycle.md).
A rotated analysis comes from `filters.ETKF_Livings` in the same way.

```{code-block} python
:caption: The rotated analysis [](#eq-etkf-rotation) has the same mean and sample covariance, with different members.
:label: code-etkf-livings
# today (filterax)
# X̃_t|t = Q W X̃
rotated = flx.filters.ETKF_Livings(key=jr.key(1)).analysis(
    members, observation, observe, noise
)
# x̄ in each analysis
means = jnp.stack(
    [einx.mean("[member] state", e) for e in (result.particles, rotated.particles)]
)  # → (2, D_x)
# Σ̂_x|y in each analysis
covariances = [  # two of (D_x, D_x)
    flx.ensemble_covariance(e).as_matrix()
    for e in (result.particles, rotated.particles)
]
print(jnp.abs(means[0] - means[1]).max() < 1e-12)
print(jnp.abs(covariances[0] - covariances[1]).max() < 1e-12)
print(jnp.abs(rotated.particles - result.particles).max().round(2))
```

It prints `True`, `True` and `1.86`: the same mean and covariance, with members that have moved by up to 1.86.

## Agreement with the Kalman update

With a linear $h$, the analysis must match the Kalman update computed from the same sample covariances, and [](#code-etkf-algorithm) must match the production route.
[](#code-etkf-agreement) checks both against the dense gain form.

```{code-block} python
:caption: The transform form against the dense Kalman update on the sample covariance, in float64.
:label: code-etkf-agreement
# Shapes: N_E = 30 members; D_x = 4; D_y = 2

# h⁽ᵉ⁾ = H x⁽ᵉ⁾
mapped = jax.vmap(observe)(members)  # → (N_E, D_y)
# Σ̂_x
sample_covariance = flx.ensemble_covariance(members).as_matrix()  # → (D_x, D_x)
# Σ_ε
noise_matrix = 0.1 * jnp.eye(2)  # → (D_y, D_y)
# H Σ̂_x
mapped_covariance = obs_map @ sample_covariance  # → (D_y, D_x)
# Σ̂_y = H Σ̂_x Hᵀ + Σ_ε
innovation = einx.dot(  # → (D_y, D_y)
    "obs state, other state -> obs other", mapped_covariance, obs_map
) + noise_matrix
# Ĝᵀ = Σ̂_y⁻¹ H Σ̂_x
gain_transpose = jnp.linalg.solve(innovation, mapped_covariance)  # → (D_y, D_x)
# r̄ = y − h̄
residual = observation - einx.mean("[member] obs", mapped)  # → (D_y,)
# x̄ + Ĝ r̄
kalman_mean = einx.mean("[member] state", members) + einx.dot(  # → (D_x,)
    "obs state, obs -> state", gain_transpose, residual
)
# Σ̂_x − Ĝ H Σ̂_x
kalman_covariance = sample_covariance - einx.dot(  # → (D_x, D_x)
    "obs state, obs other -> state other", gain_transpose, mapped_covariance
)

analysis = result.particles  # → (N_E, D_x)
mean_error = jnp.abs(einx.mean("[member] state", analysis) - kalman_mean).max()
analysis_covariance = flx.ensemble_covariance(analysis).as_matrix()  # → (D_x, D_x)
covariance_error = jnp.abs(analysis_covariance - kalman_covariance).max()
by_hand = etkf_by_hand(members, mapped, observation, noise_matrix)  # → (N_E, D_x)
hand_error = jnp.abs(by_hand - analysis).max()
print(f"{mean_error:.0e} {covariance_error:.0e} {hand_error:.0e}")
```

It prints `4e-16 1e-16 2e-15`, which is float64 round-off in all three.

## Differentiating the transform

The ensemble-space route eigendecomposes $\gmat{\Lambda}_w = \gmat{\Phi} \op{diag}(\gvect{\gamma}) \gmat{\Phi}^\top$, and by [](#eq-etkf-low-rank) the eigenvalue $c$ is repeated $N_\mathcal{E} - \op{rank}(\tilde{\mat{H}})$ times, which is at least twice whenever $D_y \le N_\mathcal{E} - 2$.
JAX's rule for `eigh` divides by the gaps $\gamma_i - \gamma_j$, so differentiating through it gives a `NaN` gradient at exactly the matrices the filter always produces, and any training loop ([chapter 14](14_differentiable.md)) would fail on its first step.

The derivative of a spectral function needs no gaps once it is written in closed form.
For $\phi(\gmat{\Lambda}_w) = \gmat{\Phi} \phi(\op{diag}(\gvect{\gamma})) \gmat{\Phi}^\top$ the Daleckii-Krein formula gives

$$\mathrm{d} \phi(\gmat{\Lambda}_w) = \gmat{\Phi} \left(\mat{D} \circ \gmat{\Phi}^\top \mathrm{d} \gmat{\Lambda}_w \gmat{\Phi}\right) \gmat{\Phi}^\top, \qquad [\mat{D}]_{ij} = \frac{\phi(\gamma_i) - \phi(\gamma_j)}{\gamma_i - \gamma_j},$$ (eq-etkf-daleckii-krein)

where $\circ$ is the element-wise product and $[\mat{D}]_{ii} = \phi'(\gamma_i)$.
For the two functions needed here the difference cancels algebraically:

$$\frac{\gamma_i^{-1} - \gamma_j^{-1}}{\gamma_i - \gamma_j} = \frac{-1}{\gamma_i \gamma_j}, \qquad \frac{\gamma_i^{-1/2} - \gamma_j^{-1/2}}{\gamma_i - \gamma_j} = \frac{-1}{\sqrt{\gamma_i \gamma_j} \, (\sqrt{\gamma_i} + \sqrt{\gamma_j})},$$ (eq-etkf-divided-differences)

and both are finite however many eigenvalues coincide.
gaussx installs this as a custom derivative, and the rank-$D_y$ route applies the same rule in blocks on $\mat{U}$ and its complement.

```{code-block} python
:caption: The gradient with respect to the observation noise, where $\gmat{\Lambda}_w$ has a repeated eigenvalue, against a central difference.
:label: code-etkf-gradient
# Shapes: N_E = 30 members; D_x = 4; D_y = 2


# user
# 𝓛(σ²) = ∑ sin X_t|t, with Σ_ε = σ² I
def loss(variance: Float[Array, ""]) -> Float[Array, ""]:
    scaled = lx.DiagonalLinearOperator(variance * jnp.ones(2))  # (D_y, D_y) operator
    # today (filterax)
    analysis = flx.filters.ETKF().analysis(members, observation, observe, scaled)
    return jnp.sum(jnp.sin(analysis.particles))  # → ()


at = jnp.asarray(0.1)
step = 1e-6
central = (loss(at + step) - loss(at - step)) / (2.0 * step)  # → ()
print(f"{jax.grad(loss)(at):.6f} {central:.6f}")
```

It prints `78.248449 78.248449`.

## Float32 and strong observations

The eigenvalues of $\gmat{\Lambda}_w$ run from $c$ up to about $\Vert \tilde{\mat{H}} \gmat{\Sigma}_\varepsilon^{-1} \tilde{\mat{H}}^\top \Vert$, and an eigendecomposition resolves them only to about machine epsilon $\epsilon$ times the largest.
In float32, with $\epsilon \approx 10^{-7}$, a strong observation therefore loses the eigenvalues near $c$ altogether, and they can even come out negative.
Two things follow.

The bound $\gamma_i \ge c$ is exact, so the ensemble-space route floors the computed eigenvalues at $c$ before taking roots.
Without the floor, a rounded negative eigenvalue gives a non-finite transform.

The mean update cancels on the rank-$D_y$ route.
With the whitened residual $\vect{v} = \gmat{\Sigma}_\varepsilon^{-1} \bar{\vect{r}} \in \mathbb{R}^{D_y}$, reading $\mat{U}^\top \tilde{\mat{H}} \vect{v}$ directly lets a weak row of $\mat{U}^\top \tilde{\mat{H}}$ carry rounding from a strong mode into the large part of $\vect{v}$.
So gaussx reads the strong modes first and removes each from $\vect{v}$ by an oblique projection, which leaves the weak modes a well-scaled remainder.

[](#code-etkf-float32) measures the result against a float64 dense solve, with noise variances from $10^{-6}$ to $10^{6}$.

```{code-block} python
:caption: Float32 mean weights against a float64 dense solve, on both routes, with noise variances spanning twelve decades.
:label: code-etkf-float32
import gaussx
import numpy as np


# user
# w̄ = Λ_w⁻¹ H̃ Σ_ε⁻¹ r̄, by a dense float64 solve
def dense_weights(obs_members, target, variances):
    mean = obs_members.mean(0)
    anomalies = obs_members - mean
    precision = np.diag(1.0 / np.asarray(variances))
    weight_precision = (len(obs_members) - 1) * np.eye(len(obs_members))
    weight_precision = weight_precision + anomalies @ precision @ anomalies.T
    return np.linalg.solve(weight_precision, anomalies @ precision @ (target - mean))


rng = np.random.default_rng(1)
for num_members, num_obs, variances in [
    (10, 3, [1e-6, 1.0, 1e6]),
    (3, 3, [1e-6, 1.0, 1e6]),
    (10, 3, [1.0, 1.0, 1.0]),
]:
    obs_members = rng.normal(size=(num_members, num_obs))
    target = rng.normal(size=num_obs)
    expected = dense_weights(obs_members, target, variances)
    errors = []
    for low_rank in (True, False):
        # today (gaussx)
        weights, _ = gaussx.etkf_transform(
            jnp.asarray(obs_members, jnp.float32),
            jnp.asarray(target, jnp.float32),
            lx.DiagonalLinearOperator(jnp.asarray(variances, jnp.float32)),
            low_rank=low_rank,
        )
        error = np.abs(np.asarray(weights) - expected).max() / np.abs(expected).max()
        errors.append(f"{error:.0e}")
    print(num_members, num_obs, errors)
```

It prints:

| $N_\mathcal{E}$ | $D_y$ | Variances | Rank-$D_y$ route | Ensemble-space route |
| --- | --- | --- | --- | --- |
| 10 | 3 | $10^{-6}$ to $10^{6}$ | `8e-08` | `3e-01` |
| 3 | 3 | $10^{-6}$ to $10^{6}$ | not available | `1e-01` |
| 10 | 3 | all 1 | `2e-07` | `3e-07` |

Noise variances spanning many decades are the hard case, and with them the rank-$D_y$ route is the only accurate one in float32.
With $D_y \ge N_\mathcal{E}$ the default falls back to the ensemble-space route, whose error here is about 10% in the mean weights.
That limit is open as [gaussx#785](https://github.com/jejjohnson/gaussx/issues/785).
In float64 the two routes agree, and ordinary variances are fine in either precision.

## When it breaks

The transform is exact for the sample covariance, so it inherits every weakness of the sample covariance.
The anomalies of $\mat{W} \tilde{\mat{X}}$ span no more than those of $\tilde{\mat{X}}$, so the update cannot move the state outside the forecast's $N_\mathcal{E} - 1$ directions.
A small ensemble then underestimates its own error and trusts the forecast too much, the observations stop mattering, and the spread collapses while the error grows.

[](#code-etkf-divergence) shows this on Lorenz-96 with $D_x = 40$, half the sites observed with unit variance, and no inflation or localization.

```{code-block} python
:caption: Filter divergence on Lorenz-96. With 20 members the spread collapses far below the error.
:label: code-etkf-divergence
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
windows = [(observed[t], 0.05 * (t + 1)) for t in range(200)]
unit_noise = lx.DiagonalLinearOperator(jnp.ones(20))  # (D_y, D_y) operator

for num_members in (20, 40):
    # x⁽ᵉ⁾ ~ 𝒩(x₀, 4 I)
    start = spun_up + 2.0 * jr.normal(jr.key(2), (num_members, 40))  # → (N_E, D_x)
    # today (filterax)
    model = flx.ETKF(dynamics=rk4, obs_op=lambda state: state[sites])
    # today (filterax)
    history = model.assimilate(start, windows, unit_noise).analysis_history
    # the second half of the run, once the filter has settled
    settled = history[100:]  # → (T / 2, N_E, D_x)
    # x̄_t|t
    settled_mean = einx.mean("t [member] state", settled)  # → (T / 2, D_x)
    # RMSE of x̄_t|t against x_true
    rmse = jnp.sqrt(jnp.mean((settled_mean - truth[100:]) ** 2))  # → ()
    # x̃⁽ᵉ⁾_t|t
    anomalies = settled - settled_mean[:, None, :]  # → (T / 2, N_E, D_x)
    # √(mean diag Σ̂_x|y)
    mean_spread = jnp.sqrt(  # → ()
        jnp.sum(anomalies**2) / (100 * (num_members - 1) * 40)
    )
    print(f"N_E = {num_members}: RMSE {rmse:.2f}, spread {mean_spread:.2f}")
```

It prints `N_E = 20: RMSE 2.37, spread 0.26` and `N_E = 40: RMSE 0.31, spread 0.25`.
With 40 members the spread matches the error, which is a healthy filter, while with 20 the error is nine times the spread and is worse than the unit observation noise.
The treatments are localization ([chapter 8](08_letkf.md), [chapter 11](11_localization.md)), which stops distant sites competing for those few directions, and inflation ([chapter 12](12_inflation.md)), which restores the lost spread.

Three other limits are worth knowing:

- A strongly nonlinear $h$ makes [](#eq-etkf-analysis) a linearisation over the ensemble, and an iterated method ([chapter 10](10_smoothers.md)) does better.
- A non-Gaussian prior biases the update, and the bias does not shrink with ensemble size.
- The symmetric transform can leave a few outlying members over long runs, which is what `ETKF_Livings` addresses.

## When ETKF is the right answer

Use `filters.ETKF` when you want a deterministic, reproducible analysis, since no perturbation noise enters.
It suits small and moderate ensembles, where the $O(1 / \sqrt{N_\mathcal{E}})$ sampling error of the stochastic EnKF is the dominant error.
It also suits differentiable pipelines, and it is the inner solver of `LETKF`, which runs one ETKF per grid point.

Reach for something else when the state is much larger than the ensemble with no localization (use `LETKF`), or when you want to process observations one at a time, which is `EnSRF_Serial` in [chapter 7](07_ensrf_estkf.md).

## See also

- [Chapter 5](05_stochastic_enkf.md): the stochastic EnKF, the perturbed-observation alternative
- [Chapter 7](07_ensrf_estkf.md): EnSRF and ESTKF, the sibling square-root filters
- [Chapter 8](08_letkf.md): LETKF, one ETKF per grid point with weighted observations
- [Chapter 14](14_differentiable.md): why the gradient safety of the analysis matters
- [API: filters](https://jejjohnson.github.io/filterax/reference/filters/), [advanced filters](https://jejjohnson.github.io/filterax/reference/filters_advanced/) and [differentiable training](https://jejjohnson.github.io/filterax/reference/differentiable/)

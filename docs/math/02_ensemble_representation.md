---
numbering:
  enumerator: "2.%s"
---

# The ensemble as a low-rank covariance

An ensemble Kalman filter holds a finite sample of a density and not the density itself.
This chapter fixes how that sample is laid out and how its moments are defined, and then develops the fact everything downstream relies on.
The sample covariance of $N_\mathcal{E}$ members has rank at most $N_\mathcal{E} - 1$, so it is fully described by a thin factor and never needs to be formed.

Three things organise the chapter.
The rank bound follows from centring, and it also says where an update can and cannot go.
Sampling error is what the small rank costs, and it can be measured directly.
And `ensemble_covariance` returns a specific operator, whose abilities and limits are worth knowing exactly.

## The ensemble matrix

An ensemble of $N_\mathcal{E}$ states is a matrix with members as rows:

$$\mat{X} = \begin{bmatrix} \vect{x}^{(1)} & \cdots & \vect{x}^{(N_\mathcal{E})} \end{bmatrix}^\top \in \mathbb{R}^{N_\mathcal{E} \times D_x}.$$ (eq-ensemble-matrix)

Axis 0 is the ensemble axis and axis 1 the state axis, so `particles[e]` is member $e$ and `jax.vmap(fn)(particles)` maps a per-state function over the members.
Much of the data assimilation literature stacks members as columns, so its formulas appear here transposed.

The mean and the anomalies both come from one centring matrix $\gmat{\Pi} = \mat{I} - \vect{1} \vect{1}^\top / N_\mathcal{E} \in \mathbb{R}^{N_\mathcal{E} \times N_\mathcal{E}}$:

$$\bar{\vect{x}} = \frac{1}{N_\mathcal{E}} \mat{X}^\top \vect{1} \in \mathbb{R}^{D_x}, \qquad \tilde{\mat{X}} = \mat{X} - \vect{1} \bar{\vect{x}}^\top = \gmat{\Pi} \mat{X} \in \mathbb{R}^{N_\mathcal{E} \times D_x}.$$ (eq-ensemble-anomalies)

$\gmat{\Pi}$ is the orthogonal projector onto the complement of $\vect{1}$, so $\vect{1}^\top \tilde{\mat{X}} = \vect{0}$.
The pair $(\bar{\vect{x}}, \tilde{\mat{X}})$ holds the same information as $\mat{X}$, and most of the analysis algebra is written on it.

## The sample covariance

The sample covariance divides the anomaly scatter by $c = N_\mathcal{E} - 1$:

$$\hat{\gmat{\Sigma}}_x = \frac{1}{c} \tilde{\mat{X}}^\top \tilde{\mat{X}} \in \mathbb{R}^{D_x \times D_x}.$$ (eq-ensemble-covariance)

The divisor makes it unbiased, which a short calculation shows.
For members drawn independently with mean $\gvect{\mu}_x$ and covariance $\gmat{\Sigma}_x$, centring on the sample mean gives

$$\sum_e \tilde{\vect{x}}^{(e)} \tilde{\vect{x}}^{(e)\top} = \sum_e (\vect{x}^{(e)} - \gvect{\mu}_x)(\vect{x}^{(e)} - \gvect{\mu}_x)^\top - N_\mathcal{E} (\bar{\vect{x}} - \gvect{\mu}_x)(\bar{\vect{x}} - \gvect{\mu}_x)^\top.$$ (eq-ensemble-scatter)

The first term has expectation $N_\mathcal{E} \gmat{\Sigma}_x$ and the second $\gmat{\Sigma}_x$, because $\mathbb{V}[\bar{\vect{x}}] = \gmat{\Sigma}_x / N_\mathcal{E}$.
So $\mathbb{E}[\hat{\gmat{\Sigma}}_x] = \gmat{\Sigma}_x$, while the divisor $N_\mathcal{E}$ would be low by the factor $c / N_\mathcal{E}$.
Using $c$ is also why filterax needs at least two members.

## Rank, and where an update can go

Since $\tilde{\mat{X}} = \gmat{\Pi} \mat{X}$ and $\gmat{\Pi}$ has rank $N_\mathcal{E} - 1$,

$$\op{rank}(\hat{\gmat{\Sigma}}_x) = \op{rank}(\tilde{\mat{X}}) \le \min(N_\mathcal{E} - 1, D_x),$$ (eq-ensemble-rank)

however large $D_x$ is.
With 50 members and $D_x = 10^6$, the ensemble places all its uncertainty in a 49-dimensional subspace.
Any covariance-vector product lands in that subspace, because $\hat{\gmat{\Sigma}}_x \vect{v} = \tilde{\mat{X}}^\top (\tilde{\mat{X}} \vect{v}) / c$ is a combination of the rows of $\tilde{\mat{X}}$.
Every ensemble Kalman increment has this form:

$$\vect{x}^{(e)}_{t \mid t} - \vect{x}^{(e)}_{t \mid t-1} = \tilde{\mat{X}}^\top \vect{w}^{(e)}, \qquad \vect{w}^{(e)} \in \mathbb{R}^{N_\mathcal{E}}.$$ (eq-ensemble-span)

So an update cannot correct an error outside the span of the anomalies, and the filter does not even know such an error exists.
[](#code-ensemble-span) measures both sides of this: how much of an ETKF increment leaves the span, and how much of an unrelated direction lies inside it.

## The low-rank factor

With the rank bound in hand, the covariance needs only a thin factor:

$$\hat{\gmat{\Sigma}}_x = \mat{U}_x \mat{U}_x^\top, \qquad \mat{U}_x = \frac{1}{\sqrt{c}} \tilde{\mat{X}}^\top \in \mathbb{R}^{D_x \times N_\mathcal{E}}.$$ (eq-ensemble-factor)

A covariance-vector product $\mat{U}_x (\mat{U}_x^\top \vect{v})$ costs $O(N_\mathcal{E} D_x)$, and storing the factor takes $N_\mathcal{E} D_x$ numbers against $D_x^2$.
Adding a structured matrix keeps this useful, because the Woodbury identity then solves against the sum through an $N_\mathcal{E} \times N_\mathcal{E}$ system.
[Chapter 4](04_kalman_update.md) uses exactly that for the innovation covariance.

## Sampling error

The sample covariance is unbiased, but each entry is noisy.
For Gaussian members, $c \hat{\gmat{\Sigma}}_x$ is Wishart, and each entry has the variance

$$\mathbb{V}\big[[\hat{\gmat{\Sigma}}_x]_{ij}\big] = \frac{1}{c} \big([\gmat{\Sigma}_x]_{ii} [\gmat{\Sigma}_x]_{jj} + [\gmat{\Sigma}_x]_{ij}^2\big).$$ (eq-ensemble-wishart)

A pair of uncorrelated unit-variance components therefore shows a covariance of typical size $1 / \sqrt{c}$, and the same holds for the sample correlation.
The Kalman update acts on those values as if they were real, which is the spurious correlation that distance-based filtering suppresses {cite:p}`houtekamer1998data,hamill2001distance`.

The eigenvalues show a different symptom.
With $\gmat{\Sigma}_x = \mat{I}$ every true eigenvalue is 1, but for large $D_x$ and $c$ at a fixed ratio $\beta = D_x / c$ the sample eigenvalues $\hat{\gamma}_i$ spread over a known range {cite:p}`marcenko1967distribution`:

$$\big(1 - \sqrt{\beta}\big)^2 \le \hat{\gamma}_i \le \big(1 + \sqrt{\beta}\big)^2.$$ (eq-ensemble-mp)

When $\beta > 1$ the remaining $D_x - c$ eigenvalues are exactly zero.
So a small ensemble is not uniformly unsure.
It overstates the variance in a few directions, and it reports no variance at all in the rest.

## What `ensemble_covariance` returns

`flx.ensemble_covariance` returns a `gaussx.LowRankUpdate` with the Bessel divisor, built as $\mat{A}_0 + \mat{U} \op{diag}(\vect{d}) \mat{V}^\top$:

| Field | Value here | Shape |
| --- | --- | --- |
| `base` | a zero `lineax.DiagonalLinearOperator` | $D_x \times D_x$ operator |
| `U`, `V` | $\mat{U}_x$, shared | $D_x \times N_\mathcal{E}$ |
| `d` | ones | $N_\mathcal{E}$ |

`filterax.cross_covariance` returns the dense counterpart $\hat{\gmat{\Sigma}}_{xy} = \tilde{\mat{X}}^\top \tilde{\mat{H}} / c$, a $D_x \times D_y$ array, since $D_y$ is usually small.
All four primitives (`ensemble_mean`, `ensemble_anomalies`, `ensemble_covariance`, `cross_covariance`) are pure functions at Layer 0.

```{code-block} python
:caption: The operator for a 500-dimensional state, which is never formed.
:label: code-ensemble-operator
import einx
import filterax as flx
import gaussx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
from jaxtyping import Array, Float

jax.config.update("jax_enable_x64", True)

# Shapes: N_E = 20 members; D_x = 500 state components

# X
members = jr.normal(jr.key(0), (20, 500))  # → (N_E, D_x)
# today (filterax)
# x̄
mean = flx.ensemble_mean(members)  # → (D_x,)
# today (filterax)
# X̃ = X − 𝟏 x̄ᵀ
anomalies = flx.ensemble_anomalies(members)  # → (N_E, D_x)
# today (filterax)
# Σ̂_x = U_x U_xᵀ
covariance = flx.ensemble_covariance(members)  # (D_x, D_x) operator
# Σ̂_x 𝟏
product = covariance.mv(jnp.ones(500))  # → (D_x,)

print(type(covariance).__name__, type(covariance.base).__name__)
print(covariance.U.shape, covariance.d.shape, product.shape)
print(covariance.U.size, 500**2)
# 𝟏ᵀ X̃
print(f"{jnp.abs(einx.sum('[member] state', anomalies)).max():.0e}")
print(covariance.rank, lx.is_symmetric(covariance))
print(lx.is_positive_semidefinite(covariance))
```

It prints `LowRankUpdate DiagonalLinearOperator`, `(500, 20) (20,) (500,)`, `10000 250000`, `3e-15`, `20 True` and `False`.
The factor holds 10,000 numbers where the matrix would hold 250,000, and the anomalies sum to zero to round-off.
The operator is tagged symmetric but not positive semi-definite, because its zero diagonal base carries no such tag.

## Algorithm

[](#code-ensemble-by-hand) writes [](#eq-ensemble-anomalies)--[](#eq-ensemble-factor) in numpy, and checks each against the operator on a state small enough to form.

```{code-block} python
:caption: The moments written out in numpy, against the filterax primitives.
:label: code-ensemble-by-hand
# Shapes: N_E = 20 members; D_x = 50; D_y = 25


# user
# Dense reference, in numpy
def moments_by_hand(array):
    c = len(array) - 1.0
    mean = array.mean(axis=0)
    centred = array - mean
    factor = centred.T / np.sqrt(c)
    return mean, centred, factor, factor @ factor.T


small = jr.normal(jr.key(0), (20, 50))  # → (N_E, D_x)
ref_mean, ref_anomalies, ref_factor, ref_covariance = moments_by_hand(
    np.asarray(small)
)
# today (filterax)
small_covariance = flx.ensemble_covariance(small)  # (D_x, D_x) operator
errors = [
    # today (filterax)
    jnp.abs(flx.ensemble_mean(small) - ref_mean).max(),
    # today (filterax)
    jnp.abs(flx.ensemble_anomalies(small) - ref_anomalies).max(),
    jnp.abs(small_covariance.U - ref_factor).max(),
    jnp.abs(small_covariance.as_matrix() - ref_covariance).max(),
    jnp.abs(small_covariance.mv(jnp.ones(50)) - ref_covariance.sum(axis=1)).max(),
]
print([f"{error:.0e}" for error in errors])

# h(x) keeps every second component
mapped = small[:, ::2]  # → (N_E, D_y)
# today (filterax)
# Σ̂_xy = X̃ᵀ H̃ / c
cross = flx.cross_covariance(small, mapped)  # → (D_x, D_y)
print(cross.shape, f"{jnp.abs(cross - ref_covariance[:, ::2]).max():.0e}")

# Shapes: R = 20,000 repeats; N_E = 5 members; D_x = 1
trials = jr.normal(jr.key(3), (20000, 5, 1))  # → (R, N_E, 1)
# today (filterax)
# Σ̂_x with the divisor N_E − 1, for each repeat
sample_variances = jax.vmap(flx.cross_covariance)(trials, trials)  # → (R, 1, 1)
unbiased = sample_variances.mean()  # → ()
print(f"{unbiased:.3f} {unbiased * 4.0 / 5.0:.3f}")
```

It prints `['6e-17', '2e-16', '6e-17', '2e-16', '1e-15']`, `(50, 25) 4e-16` and `0.999 0.799`.
The first two lines are round-off.
The last averages 20,000 sample variances with $N_\mathcal{E} = 5$ and a true variance of 1.
The divisor $c$ gives `0.999` and the divisor $N_\mathcal{E}$ gives `0.799`, as [](#eq-ensemble-scatter) predicts.

## Measuring the sampling error

[](#code-ensemble-sampling) draws members with $\gmat{\Sigma}_x = \mat{I}$ and $D_x = 200$, so that every off-diagonal value is pure noise.

```{code-block} python
:caption: Sample correlations of uncorrelated components, against the prediction $1 / \sqrt{c}$.
:label: code-ensemble-sampling
# Shapes: N_E members; D_x = 200

for num_members in (10, 50, 250):
    draws = jr.normal(jr.key(0), (num_members, 200))  # → (N_E, D_x)
    # today (filterax)
    # Σ̂_x
    sample = flx.cross_covariance(draws, draws)  # → (D_x, D_x)
    # √(diag Σ̂_x)
    spread = jnp.sqrt(jnp.diag(sample))  # → (D_x,)
    # corr = Σ̂_x ⊘ (s sᵀ)
    correlation = sample / einx.multiply("a, b -> a b", spread, spread)
    # every pair once
    pairs = correlation[jnp.triu_indices(200, 1)]  # → (19,900,)
    rms = jnp.sqrt(jnp.mean(pairs**2))  # → ()
    prediction = 1.0 / np.sqrt(num_members - 1.0)
    largest = jnp.abs(pairs).max()  # → ()
    print(f"{num_members}: rms {rms:.3f}, 1/√c {prediction:.3f}, max {largest:.2f}")
```

It prints:

| $N_\mathcal{E}$ | RMS correlation | $1 / \sqrt{c}$ | Largest |
| --- | --- | --- | --- |
| 10 | `0.333` | `0.333` | `0.95` |
| 50 | `0.143` | `0.143` | `0.62` |
| 250 | `0.063` | `0.063` | `0.32` |

The typical size follows $1 / \sqrt{c}$ closely.
The largest value matters too, because a filter acts on every pair, and among 19,900 pairs some are far from zero.

The spectrum makes the second symptom concrete.

```{code-block} python
:caption: Sample eigenvalues against the edges [](#eq-ensemble-mp), with every true eigenvalue equal to 1.
:label: code-ensemble-spectrum
# Shapes: N_E members; D_x state components

for num_members, state_size in ((20, 200), (20, 40), (200, 40), (2000, 40)):
    draws = jr.normal(jr.key(1), (num_members, state_size))  # → (N_E, D_x)
    # today (filterax)
    centred = flx.ensemble_anomalies(draws)  # → (N_E, D_x)
    # eigenvalues of Σ̂_x, from the singular values of X̃
    singular = jnp.linalg.svd(centred, compute_uv=False)
    eigenvalues = singular**2 / (num_members - 1.0)
    nonzero = eigenvalues[eigenvalues > 1e-10]
    # β = D_x / c
    ratio = state_size / (num_members - 1.0)
    upper = (1.0 + np.sqrt(ratio)) ** 2
    lower = (1.0 - np.sqrt(ratio)) ** 2
    print(
        f"{num_members} {state_size}: {nonzero.size} nonzero, "
        f"top {nonzero.max():.2f} (edge {upper:.2f}), "
        f"low {nonzero.min():.2f} (edge {lower:.2f})"
    )
```

It prints:

| $N_\mathcal{E}$ | $D_x$ | Nonzero | Top (edge) | Lowest nonzero (edge) |
| --- | --- | --- | --- | --- |
| 20 | 200 | `19` | `16.78` (`18.02`) | `5.76` (`5.04`) |
| 20 | 40 | `19` | `5.34` (`6.01`) | `0.32` (`0.20`) |
| 200 | 40 | `40` | `1.87` (`2.10`) | `0.32` (`0.30`) |
| 2000 | 40 | `40` | `1.28` (`1.30`) | `0.75` (`0.74`) |

The ensemble of 20 in 200 dimensions reports a variance of about 17 in its leading direction, where the truth is 1, and 181 directions with none.
Even 2000 members in 40 dimensions leave eigenvalues from about 0.75 to 1.28.

## Numerics

Anomalies are the one place in this chapter where float32 loses digits, because subtracting the mean cancels.
A member with $|\bar{x}|$ large against its spread $s$ keeps only about $\epsilon |\bar{x}| / s$ of relative accuracy in its anomaly, with $\epsilon$ the machine epsilon.
A temperature near 300 with a small spread is the usual case.
[](#code-ensemble-float32) measures the relative error against float64, and then subtracts a reference value before the cast.

```{code-block} python
:caption: Anomalies in float32 when the mean is large against the spread.
:label: code-ensemble-float32
# Shapes: N_E = 30 members; D_x = 5

standard = jr.normal(jr.key(2), (30, 5))  # → (N_E, D_x)
for scale in (1.0, 0.01):
    # x⁽ᵉ⁾ = 300 + s z⁽ᵉ⁾
    double = 300.0 + scale * standard  # → (N_E, D_x)
    # today (filterax)
    reference = flx.ensemble_anomalies(double)  # → (N_E, D_x)
    single = jnp.asarray(double, jnp.float32)  # → (N_E, D_x)
    # today (filterax)
    raw = flx.ensemble_anomalies(single)  # → (N_E, D_x)
    shifted = jnp.asarray(double - 300.0, jnp.float32)  # → (N_E, D_x)
    # today (filterax)
    recentred = flx.ensemble_anomalies(shifted)  # → (N_E, D_x)
    size = jnp.abs(reference).max()  # → ()
    print(
        f"s = {scale}: raw {jnp.abs(raw - reference).max() / size:.0e}, "
        f"shifted {jnp.abs(recentred - reference).max() / size:.0e}"
    )
```

It prints `s = 1.0: raw 3e-05, shifted 4e-08` and `s = 0.01: raw 2e-03, shifted 5e-08`.
A spread of 0.01 about 300 costs a fifth of a percent in the anomalies, close to the estimate $\epsilon |\bar{x}| / s$.
Subtracting a reference value before the cast restores float32 accuracy, and that step is the caller's, since filterax takes the array it is given.
Keeping the factor and not the product helps downstream, since forming $\mat{U}_x \mat{U}_x^\top$ squares any condition number.
The analysis routes in [chapter 4](04_kalman_update.md) and [chapter 6](06_etkf.md) are built around that, and [chapter 18](18_numerics.md) collects the float32 limits.

## When it breaks

**A bare sample covariance cannot be solved.**
Woodbury needs to invert the `base`, and here the `base` is zero.
`gaussx.solve` and `gaussx.logdet` do not raise on this operator: they return `NaN`.
That holds with $N_\mathcal{E} \le D_x$, where the matrix is singular, and also with $N_\mathcal{E} > D_x$, where it is invertible.
Adding a positive definite noise covariance fixes it, and that is how filterax builds the innovation covariance.

```{code-block} python
:caption: Solving against the bare operator, and against the same factor over a noise base.
:label: code-ensemble-singular
# Shapes: N_E = 4 members; D_x = 6

few = jr.normal(jr.key(1), (4, 6))  # → (N_E, D_x)
# today (filterax)
bare = flx.ensemble_covariance(few)  # (D_x, D_x) operator
right = jnp.ones(6)  # → (D_x,)
# today (gaussx)
print(gaussx.solve(bare, right)[0], gaussx.logdet(bare))
print(bare.rank, jnp.linalg.matrix_rank(bare.as_matrix()))

# Shapes: N_E = 30 members; D_x = 3, so this Σ̂_x is invertible
tall = jr.normal(jr.key(6), (30, 3))  # → (N_E, D_x)
# today (filterax)
invertible = flx.ensemble_covariance(tall)  # (D_x, D_x) operator
# today (gaussx)
print(gaussx.solve(invertible, jnp.ones(3))[0], gaussx.logdet(invertible))
print(jnp.linalg.matrix_rank(invertible.as_matrix()))

# today (lineax)
base = lx.DiagonalLinearOperator(0.1 * jnp.ones(6))  # (D_x, D_x) operator
# today (gaussx)
# Σ̂_x + 0.1 I
regular = gaussx.LowRankUpdate(base, bare.U)  # (D_x, D_x) operator
# today (gaussx)
solved = gaussx.solve(regular, right)  # → (D_x,)
print(f"{jnp.abs(regular.as_matrix() @ solved - right).max():.0e}")
# today (gaussx)
logdet = gaussx.logdet(regular)  # → ()
print(f"{logdet - jnp.linalg.slogdet(regular.as_matrix())[1]:.0e}")

try:
    # today (filterax)
    flx.ensemble_covariance(jnp.ones((1, 3)))
except ValueError as error:
    print(type(error).__name__)
```

It prints `nan nan`, `4 3`, `nan nan`, `3`, `3e-15`, `2e-15` and `ValueError`.
The third and fourth values show the invertible case failing all the same, while the operator with a noise base solves to round-off and matches the dense log-determinant.
A single member raises a `ValueError`, because the divisor $c$ would be zero.
Note also that `rank` counts the columns of the factor, $N_\mathcal{E}$, while the rank of the matrix is $N_\mathcal{E} - 1$.

**`as_matrix` is for small problems.**
On this operator it forms an $N_\mathcal{E} \times N_\mathcal{E}$ matrix of weights on the way to a $D_x \times D_x$ result.
So it needs $8 N_\mathcal{E}^2$ bytes in float64 even for a small state.
With very many members and a small state, `cross_covariance(particles, particles)` gives the same dense matrix without that array.

**The span limits what can be learned.**
[](#code-ensemble-span) runs an ETKF analysis with $N_\mathcal{E} = 20$ and $D_x = 200$, and projects vectors onto the rows of the forecast anomalies.

```{code-block} python
:caption: An ETKF increment stays in the span of the anomalies, and random directions mostly do not.
:label: code-ensemble-span
# Shapes: N_E = 20 members; D_x = 200; D_y = 10; Q = 19 basis vectors

forecast = jr.normal(jr.key(4), (20, 200))  # → (N_E, D_x)
# today (lineax)
obs_noise = lx.DiagonalLinearOperator(0.5 * jnp.ones(10))  # (D_y, D_y) operator
# today (filterax)
analysed = flx.filters.ETKF().analysis(
    forecast, jnp.ones(10), lambda state: state[:10], obs_noise
).particles
# Orthonormal basis of the rows of X̃, dropping the null direction
# today (filterax)
forecast_anomalies = flx.ensemble_anomalies(forecast)  # → (N_E, D_x)
_, _, right_vectors = jnp.linalg.svd(forecast_anomalies, full_matrices=False)
basis = right_vectors[:19]  # → (Q, D_x)


# user
# ‖v − P v‖ / ‖v‖ for the projector P onto the span
def outside(vector: Float[Array, " state"]) -> Float[Array, ""]:
    # Qᵀ v
    coefficients = einx.dot("basis state, state -> basis", basis, vector)
    # P v
    projected = einx.dot("basis state, basis -> state", basis, coefficients)
    return jnp.linalg.norm(vector - projected) / jnp.linalg.norm(vector)


# one member's increment
increment = analysed[0] - forecast[0]  # → (D_x,)
# Shapes: M = 500 random directions
directions = jr.normal(jr.key(5), (500, 200))  # → (M, D_x)
outside_energy = jax.vmap(outside)(directions) ** 2  # → (M,)
print(f"{outside(increment):.0e} {outside_energy.mean():.3f}")
```

It prints `2e-15 0.905`.
The increment is inside the span to round-off, as [](#eq-ensemble-span) says.
Random directions have on average the fraction `0.905` of their energy outside it, which matches $1 - c / D_x$ with $c = 19$ and $D_x = 200$.
No update can reach that part.

## When a low-rank ensemble covariance is the right answer

Use `ensemble_covariance` when you need a covariance as an operator, for matrix-vector products or to pass into `gaussx`.
Use `cross_covariance` when you need a dense $D_x \times D_y$ array, or a dense covariance from very many members.
Use `ensemble_anomalies` when you are writing an update by hand, since most formulas are simplest on $\tilde{\mat{X}}$.

The representation itself suits a large $D_x$ with a covariance that depends on the flow and could never be stored.
If $D_x$ is small and the model is linear, keep a real covariance instead, with `SquareRootKF` ([chapter 17](17_exact_references.md)).
If $N_\mathcal{E} \ll D_x$, plan on localization ([chapter 11](11_localization.md)) and inflation ([chapter 12](12_inflation.md)), because the sampling error above will otherwise reach the analysis.

## See also

- [Chapter 1](01_problem_setting.md): why the filter works with samples at all
- [Chapter 3](03_observation_model.md): the same mean and anomaly algebra in observation space
- [Chapter 4](04_kalman_update.md): the gain, where the factor pays off through Woodbury
- [Chapter 11](11_localization.md) and [chapter 12](12_inflation.md): the treatments for sampling error
- [Chapter 18](18_numerics.md): float32 and conditioning across the library
- [API: primitives](https://jejjohnson.github.io/filterax/reference/primitives/)

---
numbering:
  enumerator: "18.%s"
---

# Numerics: where filters fail quietly

The algebra of a Kalman update fits on a page, and floating point is where it goes wrong.
The failures of interest raise nothing: the arrays stay finite, the covariance stays positive definite, and the answer is wrong.
A library that wants float32 and gradients has to design around them.

Seven themes recur across filterax and gaussx, and they organise this chapter.
Each one gets a small demonstration, and the chapter ends with the limits that are still open.
Throughout, $\epsilon$ is the machine epsilon, about $1.2 \times 10^{-7}$ in float32 and $2.2 \times 10^{-16}$ in float64.

## Forming a Gram matrix squares the condition number

Take anomalies $\tilde{\mat{X}} \in \mathbb{R}^{N_\mathcal{E} \times D_x}$ with the thin singular value decomposition $\tilde{\mat{X}} = \mat{U} \op{diag}(\gvect{\sigma}) \mat{V}^\top$, where $\gvect{\sigma} \in \mathbb{R}^R$ is non-negative and sorted down.
The Gram matrix is then

$$\tilde{\mat{X}} \tilde{\mat{X}}^\top = \mat{U} \op{diag}(\gvect{\sigma}^2) \mat{U}^\top, \qquad \kappa(\tilde{\mat{X}} \tilde{\mat{X}}^\top) = \kappa(\tilde{\mat{X}})^2,$$ (eq-numerics-gram)

where $\kappa$ is the ratio of the largest singular value to the smallest.
An eigendecomposition resolves eigenvalues only to about $\epsilon \sigma_1^2$.
So a singular value below about $\sqrt{\epsilon} \sigma_1$, which is $3.5 \times 10^{-4} \sigma_1$ in float32, is lost once the Gram matrix has been formed, however careful the solver.
Working from $\tilde{\mat{X}}$ itself resolves it down to about $\epsilon \sigma_1$.

[](#code-numerics-gram) builds anomalies with $\gvect{\sigma} = (1, 10^{-1}, \dots, 10^{-4})$ and reads the singular values both ways.
It then takes one backward smoother step, where the smoothed state differs from the forecast along the strongest and the weakest mode.
`EnKS` calls `gaussx.ensemble_rts_step`, which takes the pseudo-inverse from the singular values of the anomalies and differentiates it by the closed form of {cite:t}`golub1973differentiation`.
`EnsembleSqrtSmoother` forms the Gram matrix instead.

```{code-block} python
:caption: Singular values of an anomaly matrix in float32, from the matrix and from its Gram matrix, then the two smoothers built on each.
:label: code-numerics-gram
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

# Shapes: N_E = 12 members; D_x = 40 state components; R = 5 modes

rng = np.random.default_rng(0)
# centred left vectors, so the rows sum to zero as anomalies do
left, _, _ = np.linalg.svd(np.eye(12) - 1.0 / 12.0)
left = left[:, :5]  # → (N_E, R)
right, _ = np.linalg.qr(rng.normal(size=(40, 5)))  # → (D_x, R)
# σ = 1, 10⁻¹, …, 10⁻⁴
singular_values = 10.0 ** -np.arange(5.0)  # → (R,)
# X̃ = U diag(σ) Vᵀ, the forecast anomalies
forecast_anomalies = left * singular_values @ right.T  # → (N_E, D_x)
single = jnp.asarray(forecast_anomalies, jnp.float32)  # → (N_E, D_x)

# today (jax)
# σ from X̃
from_matrix = jnp.linalg.svdvals(single)[:5]  # → (R,)
# X̃ X̃ᵀ
gram = einx.dot("member state, other state -> member other", single, single)
# today (jax)
# σ² from X̃ X̃ᵀ, descending
from_gram = jnp.linalg.eigvalsh(gram)[::-1][:5]  # → (R,)
# √max(σ², 0)
from_gram = jnp.sqrt(jnp.maximum(from_gram, 0.0))  # → (R,)
print(np.abs(np.asarray(from_matrix) - singular_values) / singular_values)
print(np.abs(np.asarray(from_gram) - singular_values) / singular_values)

# A backward step from the last time to the first, with a smoothed shift along the
# strongest and weakest modes.
# X_t+1^f
forecast_next = forecast_anomalies + 1.0  # → (N_E, D_x)
# X_t^a
analysis_now = 0.7 * forecast_anomalies + 1.0  # → (N_E, D_x)
# X_t+1^s = X_t+1^f + the shift
smoothed_next = forecast_next + 0.1 * right[:, 0] + 0.01 * right[:, 4]  # → (N_E, D_x)


# user
# X_t^s from one backward step of a smoother, in one precision
def backward_step(smoother, dtype) -> Float[Array, " state"]:
    forecast = jnp.asarray(np.stack([forecast_next, forecast_next]), dtype)
    analysis = jnp.asarray(np.stack([analysis_now, smoothed_next]), dtype)
    # today (filterax)
    history = smoother.smooth(forecast, analysis).smoothed_history
    return einx.mean("[member] state", history[0])  # → (D_x,)


reference = backward_step(flx.EnKS(), jnp.float64)  # → (D_x,)
other = backward_step(flx.EnsembleSqrtSmoother(), jnp.float64)  # → (D_x,)
print(f"{jnp.abs(reference - other).max():.0e}")
shift = np.abs(np.asarray(reference) - analysis_now.mean(0)).max()
for smoother in (flx.EnKS(), flx.EnsembleSqrtSmoother()):
    error = jnp.abs(backward_step(smoother, jnp.float32) - reference).max()
    print(f"{type(smoother).__name__}: {error / shift:.1e}")
```

It prints the relative errors in $\gvect{\sigma}$, and then the smoothers:

| $\sigma_i$ | $1$ | $10^{-1}$ | $10^{-2}$ | $10^{-3}$ | $10^{-4}$ |
| --- | --- | --- | --- | --- | --- |
| From $\tilde{\mat{X}}$ | `6e-08` | `4e-07` | `2e-08` | `5e-08` | `4e-05` |
| From $\tilde{\mat{X}} \tilde{\mat{X}}^\top$ | `0` | `2e-06` | `9e-04` | `2e-04` | `2.3` |

The weakest mode comes back from the Gram matrix wrong by 230%, and from the matrix itself correct to `4e-05`.

The smoothers then print `7e-11`, `EnKS: 2.8e-04` and `EnsembleSqrtSmoother: 9.5e-02`.
In float64 the two means agree to `7e-11`.
In float32 `EnKS` misses the reference by `2.8e-04` of the size of the smoothed shift and `EnsembleSqrtSmoother` by `9.5e-02`.
The second drops the mode on purpose: its cutoff is $N_\mathcal{E} \epsilon \sigma_1^2$ on the eigenvalues, which at $N_\mathcal{E} = 12$ removes everything below about $1.2 \times 10^{-3} \sigma_1$.
So in float32, prefer `EnKS` when a weak direction matters.

The ETKF applies the same idea to observations.
Its rank-$D_y$ route takes a thin QR of $\tilde{\mat{H}}$ rather than decomposing $\gmat{\Lambda}_w$, as in [chapter 6](06_etkf.md).

## Finite and positive definite do not mean right

It is tempting to test a float32 filter by asserting that its output is finite and its covariance positive definite.
That check passes on broken code, because the covariance can be right while the mean beside it is not.
The test that catches it compares means and gradients with a float64 reference, and that is what `test_square_root_kf_float32_matches_a_float64_reference` does.

[](#code-numerics-means) has two cases.
The first is the ensemble-space route of the ETKF on noise variances spanning twelve decades, the setting of [gaussx#785](https://github.com/jejjohnson/gaussx/issues/785).
The second is the mean shift of the parallel filter from [chapter 17](17_exact_references.md).
With $\gmat{\Sigma} = \mat{L} \mat{L}^\top$ and $\gmat{\Lambda} = \mat{L}_\text{lik} \mat{L}_\text{lik}^\top$, both in $\mathbb{R}^{D_x \times D_x}$, and $\vect{b} \in \mathbb{R}^{D_x}$, two forms of the same vector are

$$(\mat{I} + \gmat{\Sigma} \gmat{\Lambda})^{-1} \gmat{\Sigma} \vect{b} = \gmat{\Sigma} \vect{b} - \mat{M} \mat{M}^\top \gmat{\Lambda} \gmat{\Sigma} \vect{b} = \mat{M} \mat{M}^\top \vect{b}, \qquad \mat{M} \mat{M}^\top = (\mat{I} + \gmat{\Sigma} \gmat{\Lambda})^{-1} \gmat{\Sigma},$$ (eq-numerics-shift)

where $\mat{M} = \mat{L} \gmat{\Xi}^{-\top}$ comes from a triangular factor $\gmat{\Xi}$ of a stack holding an identity, so it is well defined at any strength.
The first form is the push-through identity, and it subtracts two large vectors whose difference is small.
The second has nothing to cancel.

```{code-block} python
:caption: Two float32 computations that pass every check on finiteness and definiteness, one in the ETKF and one in the parallel filter.
:label: code-numerics-means
# Shapes: N_E = 10 members; D_y = 3 observations; D_x = 2 in the second part

# H̃ and y, with noise variances spanning twelve decades
obs_members = rng.normal(size=(10, 3))  # → (N_E, D_y)
target = rng.normal(size=3)  # → (D_y,)
variances = np.array([1e-6, 1.0, 1e6])  # → (D_y,)
# w̄ = Λ_w⁻¹ H̃ Σ_ε⁻¹ r̄, by a dense float64 solve
mapped_mean = obs_members.mean(0)
centred = obs_members - mapped_mean
weighted = centred / variances
dense_precision = 9.0 * np.eye(10) + weighted @ centred.T
expected = np.linalg.solve(dense_precision, weighted @ (target - mapped_mean))
# today (gaussx)
# the ensemble-space route, in float32
weights, transform = gaussx.etkf_transform(
    jnp.asarray(obs_members, jnp.float32),
    jnp.asarray(target, jnp.float32),
    lx.DiagonalLinearOperator(jnp.asarray(variances, jnp.float32)),
    low_rank=False,
)
eigenvalues = jnp.linalg.eigvalsh(transform)  # → (N_E,)
print(bool(jnp.isfinite(weights).all() and jnp.isfinite(transform).all()))
print(f"{eigenvalues.min():.1e} {eigenvalues.max():.3f}")
print(f"{np.abs(np.asarray(weights) - expected).max() / np.abs(expected).max():.2f}")


# user
# (I + Σ Λ)⁻¹ Σ b by the push-through form and by M Mᵀ b
def mean_shift(
    root: Float[Array, "state state"],
    lik_root: Float[Array, "state state"],
    information: Float[Array, " state"],
) -> tuple[Float[Array, " state"], Float[Array, " state"]]:
    # A ↦ Aᵀ
    flip = einx.id("row col -> col row", root)  # → (D_x, D_x)
    # Lᵀ L_lik
    cross = flip @ lik_root  # → (D_x, D_x)
    # [Lᵀ L_lik, I]
    stack = jnp.block([[cross, jnp.eye(2, dtype=root.dtype)]])  # → (D_x, 2 D_x)
    # Ξ = tria([Lᵀ L_lik, I]) = Rᵀ, for the QR of the stack's transpose
    upper = jnp.linalg.qr(einx.id("row col -> col row", stack), mode="r")
    xi = einx.id("row col -> col row", upper)  # → (D_x, D_x)
    # Mᵀ = Ξ⁻¹ Lᵀ
    factor_flip = jax.scipy.linalg.solve_triangular(xi, flip, lower=True)
    # M = L Ξ⁻ᵀ
    factor = einx.id("row col -> col row", factor_flip)  # → (D_x, D_x)
    # Σ b = L Lᵀ b, a large vector
    large = root @ (flip @ information)  # → (D_x,)
    # Λ Σ b = L_lik L_likᵀ Σ b
    pulled = lik_root @ einx.dot("state row, state -> row", lik_root, large)
    # (I + Σ Λ)⁻¹ v = v − M Mᵀ Λ v at v = Σ b, which cancels
    pushed = large - factor @ (factor_flip @ pulled)  # → (D_x,)
    # M Mᵀ b, which does not
    direct = factor @ (factor_flip @ information)  # → (D_x,)
    return pushed, direct


lik_root = np.array([[1.0, 0.0], [0.5, 1.0]])  # → (2, 2)
information = np.array([1.0, -2.0])  # → (2,)
base_root = np.array([[3.0, 0.0], [1.0, 2.0]])  # → (2, 2)
for scale in (1e1, 1e3, 1e5):
    root = scale * base_root  # → (2, 2)
    cov, precision = root @ root.T, lik_root @ lik_root.T
    # (I + Σ Λ)⁻¹ Σ b by dense float64 solve
    exact = np.linalg.solve(np.eye(2) + cov @ precision, cov @ information)  # → (2,)
    # the largest eigenvalue of Σ Λ
    strength = np.linalg.eigvals(cov @ precision).real.max()
    results = []
    for dtype in (jnp.float32, jnp.float64):
        pushed, direct = mean_shift(
            jnp.asarray(root, dtype),
            jnp.asarray(lik_root, dtype),
            jnp.asarray(information, dtype),
        )
        for computed in (pushed, direct):
            miss = np.abs(np.asarray(computed) - exact).max() / np.abs(exact).max()
            results.append(f"{miss:.0e}")
    print(f"{strength:.0e}", results)
```

The first case prints `True`, `1.0e-03 1.000` and `0.06`.
Every entry is finite and the eigenvalues of $\mat{W}$ lie in the range the theory allows, yet the mean weights are 6% wrong on this draw.

The second case prints the relative errors of the two forms:

| Largest eigenvalue of $\gmat{\Sigma} \gmat{\Lambda}$ | Push-through, float32 | $\mat{M} \mat{M}^\top$, float32 | Push-through, float64 | $\mat{M} \mat{M}^\top$, float64 |
| --- | --- | --- | --- | --- |
| `2e+03` | `2e-05` | `7e-08` | `1e-13` | `9e-16` |
| `2e+07` | `6e-01` | `2e-07` | `4e-10` | `0e+00` |
| `2e+11` | `7e+03` | `2e-07` | `6e-06` | `0e+00` |

The push-through error grows like $\epsilon$ times the strength, in either precision, while $\mat{M} \mat{M}^\top \vect{b}$ stays at round-off.
At strength near $1 / \epsilon$ the float32 push-through has lost the vector completely, and $\mat{M} \mat{M}^\top$, the covariance beside it, is unaffected.
This was the failure of `square_root=True` before gaussx v0.6.9, and [chapter 17](17_exact_references.md) shows the filter with the fix.

Gradients need the same test, against float64 and against a central difference.
[](#code-numerics-gradient) differentiates the log-likelihood of `SquareRootKF` on a strongly observed model with $D_y < D_x$, where the likelihood factors are rank deficient.
gaussx handles that with a custom derivative.
With $\mat{L} = \mat{A} \mat{Q}$ from a QR, the tangent is $\mathrm{d} \mat{L} = \mathrm{d} \mat{A} \mat{Q}$, which drops a term that only rotates $\mat{L}$ and so never changes $\mat{L} \mat{L}^\top$.
That needs no $\mat{R}^{-1}$, so it stays finite at low rank.
The same block compares the Jacobian of the ETKF mean weights in the two precisions, at noise variance $10^{-8}$.

```{code-block} python
:caption: A float32 gradient of the log-likelihood against float64 and a central difference, and the Jacobian of the mean weights.
:label: code-numerics-gradient
# Shapes: T = 32 times; D_x = 2; D_y = 1; N_E = 10 members in the second half

# F, H
velocity = jnp.array([[1.0, 0.1], [0.0, 1.0]])  # → (D_x, D_x)
position = jnp.array([[1.0, 0.0]])  # → (D_y, D_x)


# user
# xₜ = F xₜ₋₁ + ηₜ and yₜ = H xₜ + εₜ, with Σ_η = 10⁻¹⁰ I and Σ_ε = 10⁻⁸ I
def simulate(state, draw):
    state = velocity @ state + 1e-5 * draw[:2]
    return state, position @ state + 1e-4 * draw[2:]


# y_1:T
draws = jr.normal(jr.key(0), (32, 3))  # → (T, 3)
records = jax.lax.scan(simulate, jr.normal(jr.key(1), (2,)), draws)[1]  # → (T, D_y)


# user
# log p(y_1:T) as a function of log σ², with Σ_ε = σ² I
def log_evidence(log_variance: Float[Array, ""]) -> Float[Array, ""]:
    dtype = log_variance.dtype
    # today (filterax)
    model = flx.filters.SquareRootKF(
        velocity.astype(dtype),
        position.astype(dtype),
        (1e-10 * jnp.eye(2)).astype(dtype),
        jnp.exp(log_variance) * jnp.eye(1, dtype=dtype),
    )
    start, start_cov = jnp.zeros(2, dtype), jnp.eye(2, dtype=dtype)
    return model.filter(records.astype(dtype), start, start_cov).log_likelihood


at = jnp.log(jnp.asarray(1e-8))  # → ()
evidence_and_slope = jax.jit(jax.value_and_grad(log_evidence))
step = 1e-5
central = evidence_and_slope(at + step)[0] - evidence_and_slope(at - step)[0]
single_slope = evidence_and_slope(jnp.float32(at))[1]
double_slope = evidence_and_slope(at)[1]
print(f"{double_slope:.4f} {central / (2.0 * step):.4f} {single_slope:.4f}")

# H̃ and y for the mean weights
strong_members = jr.normal(jr.key(2), (10, 3))  # → (N_E, D_y)
strong_target = jr.normal(jr.key(3), (3,))  # → (D_y,)


# user
# w̄ as a function of σ², with Σ_ε = σ² I
def mean_weights(variance: Float[Array, ""]) -> Float[Array, " member"]:
    dtype = variance.dtype
    # today (gaussx)
    return gaussx.etkf_transform(
        strong_members.astype(dtype),
        strong_target.astype(dtype),
        lx.DiagonalLinearOperator(variance * jnp.ones(3, dtype)),
    )[0]


jacobian = jax.jit(jax.jacfwd(mean_weights))
double_jacobian = jacobian(jnp.asarray(1e-8))  # → (N_E,)
single_jacobian = jacobian(jnp.float32(1e-8))  # → (N_E,)
jacobian_gap = jnp.abs(single_jacobian - double_jacobian).max()
print(f"{jacobian_gap / jnp.abs(double_jacobian).max():.0e}")
```

It prints `5.0288 5.0288 5.0245` and `4e-07`.
The float64 gradient matches the central difference, the float32 gradient is 0.09% away, and the Jacobian of $\bar{\vect{w}}$ matches to float32 round-off.

## A repeated eigenvalue and gap-free derivatives

[Chapter 6](06_etkf.md) derives this in full, so here is only the point.
$\gmat{\Lambda}_w$ has the eigenvalue $c = N_\mathcal{E} - 1$ repeated whenever $D_y < N_\mathcal{E} - 1$, and JAX's rule for `eigh` divides by eigenvalue gaps.
A spectral function needs no such division once its divided differences are written in closed form.

```{code-block} python
:caption: The same transform differentiated through `eigh` and through the gap-free rule, at a repeated eigenvalue.
:label: code-numerics-repeated
# Shapes: N_E = 30 members; D_y = 2 observations

# h⁽ᵉ⁾
mapped = 1.0 + 0.5 * jr.normal(jr.key(0), (30, 2))  # → (N_E, D_y)
# H̃
mapped_anomalies = flx.ensemble_anomalies(mapped)  # → (N_E, D_y)
# H̃ H̃ᵀ
mapped_gram = einx.dot(  # → (N_E, N_E)
    "member obs, other obs -> member other", mapped_anomalies, mapped_anomalies
)


# user
# ∑ sin W, with W = (c Λ_w⁻¹)^½ through jax's rule for eigh
def through_eigh(variance: Float[Array, ""]) -> Float[Array, ""]:
    # Λ_w = c I + H̃ H̃ᵀ / σ²
    precision = 29.0 * jnp.eye(30) + mapped_gram / variance  # → (N_E, N_E)
    # today (jax)
    # Λ_w = Φ diag(γ) Φᵀ
    gamma, phi = jnp.linalg.eigh(precision)
    # Φ diag(√(c / γ))
    scaled = einx.multiply(  # → (N_E, N_E)
        "member mode, mode -> member mode", phi, jnp.sqrt(29.0 / gamma)
    )
    # W = Φ diag(√(c / γ)) Φᵀ
    root = einx.dot("member mode, other mode -> member other", scaled, phi)
    return jnp.sum(jnp.sin(root))  # → ()


# user
# ∑ sin W, with W from gaussx
def through_gaussx(variance: Float[Array, ""]) -> Float[Array, ""]:
    noise = lx.DiagonalLinearOperator(variance * jnp.ones(2))  # (D_y, D_y) operator
    # today (gaussx)
    _, root = gaussx.etkf_transform(mapped, jnp.zeros(2), noise, low_rank=False)
    return jnp.sum(jnp.sin(root))  # → ()


at = jnp.asarray(0.1)
step = 1e-6
central = (through_gaussx(at + step) - through_gaussx(at - step)) / (2.0 * step)
print(f"{through_eigh(at):.6f} {through_gaussx(at):.6f}")
print(f"{jax.grad(through_eigh)(at)} {jax.grad(through_gaussx)(at):.6f} {central:.6f}")
```

It prints `25.645979 25.645979`, then `nan -1.521080 -1.521080`.
The values agree, and the derivative through `eigh` is `nan` while the gap-free rule matches the central difference.

## Eigenvalue floors

Every eigenvalue of $\gmat{\Lambda}_w$ is at least $c$, exactly, because $\tilde{\mat{H}} \gmat{\Sigma}_\varepsilon^{-1} \tilde{\mat{H}}^\top$ is positive semi-definite.
A float32 eigendecomposition does not know that.
It resolves eigenvalues only to about $\epsilon$ times the largest, so a strong observation can return a computed $\gamma_i$ far below $c$, or below zero.
Then $\sqrt{c / \gamma_i}$ is `NaN`.
That was [gaussx#790](https://github.com/jejjohnson/gaussx/issues/790), now closed: the ensemble-space route floors the computed eigenvalues at $c$.

The floor is safe because it is a bound the exact matrix satisfies, so it can only move an eigenvalue toward the truth.
A floor that is not such a bound changes the answer without saying so.

[](#code-numerics-floor) reproduces the setting of that issue, with $D_y = 30$ observations of $N_\mathcal{E} = 8$ members, where the ensemble-space route is the only one.

```{code-block} python
:caption: An unfloored and a floored transform in float32, with both checked against a float64 solve on the same inputs.
:label: code-numerics-floor
# Shapes: N_E = 8 members; D_x = 12 state components; D_y = 30 observations

prior_term = 7.0
# s, r on a line
state_grid = jnp.linspace(0.0, 10.0, 12)  # → (D_x,)
obs_grid = jnp.linspace(0.0, 10.0, 30)  # → (D_y,)
# H, a Gaussian blur of the state
offsets = einx.subtract("obs, state -> obs state", obs_grid, state_grid)  # → (D_y, D_x)
blur = jnp.exp(-0.5 * (offsets / 1.5) ** 2).astype(jnp.float32)  # → (D_y, D_x)

for smallest in (1e-6, 1e-8):
    # Σ_ε, with variances from `smallest` up to 1
    log_range = jnp.linspace(jnp.log10(smallest), 0.0, 30)  # → (D_y,)
    variance = (10.0**log_range).astype(jnp.float32)  # → (D_y,)
    bare_failures, floored_failures, lowest, mean_error, root_error = 0, 0, [], [], []
    for seed in range(5):
        members = jr.normal(jr.key(seed), (8, 12)).astype(jnp.float32)  # → (N_E, D_x)
        obs_members = einx.dot("member state, obs state -> member obs", members, blur)
        goal = jr.normal(jr.key(100 + seed), (30,)).astype(jnp.float32)  # → (D_y,)
        # H̃
        centred = flx.ensemble_anomalies(obs_members)  # → (N_E, D_y)
        # H̃ Σ_ε⁻¹
        weighted = einx.divide("member obs, obs -> member obs", centred, variance)
        # Λ_w = c I + H̃ Σ_ε⁻¹ H̃ᵀ
        inner = einx.dot("member obs, other obs -> member other", weighted, centred)
        precision = prior_term * jnp.eye(8, dtype=jnp.float32) + inner  # → (N_E, N_E)
        # today (jax)
        # Λ_w = Φ diag(γ) Φᵀ
        gamma, phi = jnp.linalg.eigh(precision)
        lowest.append(float(gamma.min()))
        # W = Φ diag(√(c / γ)) Φᵀ, with no floor
        scaled = einx.multiply(
            "member mode, mode -> member mode", phi, jnp.sqrt(prior_term / gamma)
        )
        bare = einx.dot("member mode, other mode -> member other", scaled, phi)
        bare_failures += int(not jnp.isfinite(bare).all())
        # today (gaussx)
        # the same transform, with γ floored at c
        weights, root = gaussx.etkf_transform(
            obs_members, goal, lx.DiagonalLinearOperator(variance)
        )
        finite = jnp.isfinite(weights).all() and jnp.isfinite(root).all()
        floored_failures += int(not finite)
        # Λ_w, w̄ and W by dense float64 solve on the same float32 inputs
        dense_centred = np.asarray(centred, float)
        dense_mean = np.asarray(obs_members, float).mean(0)
        dense_scaled = dense_centred / np.asarray(variance, float)
        dense_precision = prior_term * np.eye(8) + dense_scaled @ dense_centred.T
        expected = np.linalg.solve(
            dense_precision, dense_scaled @ (np.asarray(goal, float) - dense_mean)
        )
        dense_gamma, dense_phi = np.linalg.eigh(dense_precision)
        expected_root = dense_phi * np.sqrt(prior_term / dense_gamma) @ dense_phi.T
        miss = np.abs(np.asarray(weights) - expected).max()
        mean_error.append(miss / np.abs(expected).max())
        root_error.append(np.abs(np.asarray(root) - expected_root).max())
    print(
        f"{smallest:g}: lowest γ {min(lowest):.2f}, no floor {bare_failures} of 5 "
        f"non-finite, gaussx {floored_failures} of 5, w̄ {max(mean_error):.2f}, "
        f"W {max(root_error):.2f}"
    )
```

It prints:

| Smallest variance | Lowest computed $\gamma$ | No floor, non-finite | gaussx, non-finite | Worst error in $\bar{\vect{w}}$ | Worst error in $\mat{W}$ |
| --- | --- | --- | --- | --- | --- |
| $10^{-6}$ | `6.39` | 0 of 5 | 0 of 5 | `0.12` | `0.03` |
| $10^{-8}$ | `-62.29` | 3 of 5 | 0 of 5 | `3.93` | `0.24` |

Here $c = 7$, so the computed eigenvalue is 69 below its exact bound on the harder row.
The floor fixes what it was meant to fix: nothing is non-finite.
It does not make the answer right: on that row the mean weights are off by nearly four times their own size.
That is relative to a float64 solve on the same inputs.
A floor restores finiteness and not accuracy, and this is the limit [gaussx#785](https://github.com/jejjohnson/gaussx/issues/785) tracks.

## Combine terms before evaluating them

A derivative written as a sum of terms can cancel just as a value can, and a custom derivative makes the choice of terms yours.
Two patterns appear in gaussx.

The first is a divided difference, which is what the Daleckii-Krein rule of [chapter 6](06_etkf.md) divides by.
For $\phi(\gamma) = 1 / \gamma$ it reduces by hand, and the closed form has no difference left in it:

$$\frac{a^{-1} - b^{-1}}{a - b} = \frac{-1}{a b}.$$ (eq-numerics-divided)

The second is a sum of derivative terms that are large and opposite.
Take the scalar mean weight $\bar{w} = \rho / \Lambda$ with $\rho = 1 / \sigma^2$ and $\Lambda = c + 1 / \sigma^2$, all scalars, for a strong observation.
The product rule gives two terms, each near $1 / \sigma^2$, whose sum is near $-c$:

$$\frac{\mathrm{d} \bar{w}}{\mathrm{d} \sigma^2} = \underbrace{-\Lambda^{-1} \frac{\mathrm{d} \Lambda}{\mathrm{d} \sigma^2} \Lambda^{-1} \rho}_{\approx 1 / \sigma^2} + \underbrace{\Lambda^{-1} \frac{\mathrm{d} \rho}{\mathrm{d} \sigma^2}}_{\approx -1 / \sigma^2} = \Lambda^{-1} \frac{\mathrm{d} \rho}{\mathrm{d} \sigma^2} (1 - \bar{w}).$$ (eq-numerics-terms)

Joining them is not enough, since $1 - \bar{w}$ is still a difference of nearly equal numbers.
The residual has to be reduced as well, and here it has a form with no subtraction:

$$1 - \bar{w} = \frac{c \sigma^2}{1 + c \sigma^2}.$$ (eq-numerics-residual)

gaussx does this in matrix form for the mean weights of the rank-$D_y$ route.
It combines the three terms of $\mathrm{d} \bar{\vect{w}}$ before evaluating any of them, around an analysis residual that is computed by deflating the strong modes rather than by subtracting.

```{code-block} python
:caption: A divided difference, and a derivative whose two terms cancel, each evaluated apart and in closed form in float32.
:label: code-numerics-cancellation
# Shapes: all scalars

# user
# a float32 scalar
def single(value: float) -> Float[Array, ""]:
    return jnp.asarray(value, jnp.float32)


first = single(1e4)
for gap in (1e-1, 1e-3):
    second = first + single(gap)
    # (a⁻¹ − b⁻¹) / (a − b), evaluated as written
    written = (1.0 / first - 1.0 / second) / (first - second)
    # −1 / (a b), which has no difference in it
    closed = -1.0 / (first * second)
    print(f"{written:.4e} {closed:.4e}")

prior = single(9.0)
for noise_variance in (1e-4, 1e-6, 1e-8, 1e-10):
    variance = single(noise_variance)
    # Λ = c + 1 / σ²
    precision = prior + 1.0 / variance
    # ρ = 1 / σ²
    rho = 1.0 / variance
    # dρ = dΛ = −1 / σ⁴
    d_rho = -1.0 / (variance * variance)
    # −Λ⁻¹ dΛ Λ⁻¹ ρ, the first term of dw̄
    term_one = -d_rho * rho / (precision * precision)
    # Λ⁻¹ dρ, the second term
    term_two = d_rho / precision
    # w̄ = Λ⁻¹ ρ
    weight = rho / precision
    # Λ⁻¹ dρ (1 − w̄), the terms joined but the residual still a difference
    joined = d_rho * (1.0 - weight) / precision
    # Λ⁻¹ dρ c σ² / (1 + c σ²), with the residual reduced by hand as well
    reduced = d_rho * (prior * variance / (1.0 + prior * variance)) / precision
    # dw̄/dσ² = −c / (1 + c σ²)², in float64
    exact = -9.0 / (1.0 + 9.0 * noise_variance) ** 2
    print(
        f"{noise_variance:g}: terms {term_one:.3g} {term_two:.3g}, "
        f"apart {term_one + term_two:.4f}, joined {joined:.4f}, "
        f"reduced {reduced:.4f}, exact {exact:.4f}"
    )
```

The divided difference prints `-1.0007e-08 -9.9999e-09` at a gap of $10^{-1}$, and `-7.4506e-09 -1.0000e-08` at a gap of $10^{-3}$, where the form as written is 25% wrong.

The sum prints the following, with the exact value computed in float64:

| $\sigma^2$ | First term | Second term | Apart | Joined, [](#eq-numerics-terms) | Reduced, [](#eq-numerics-residual) | Exact |
| --- | --- | --- | --- | --- | --- | --- |
| $10^{-4}$ | `9.98e+03` | `-9.99e+03` | `-8.9834` | `-8.9839` | `-8.9838` | `-8.9838` |
| $10^{-6}$ | `1e+06` | `-1e+06` | `-9.0625` | `-9.0002` | `-8.9998` | `-8.9998` |
| $10^{-8}$ | `1e+08` | `-1e+08` | `-8.0000` | `-5.9605` | `-9.0000` | `-9.0000` |
| $10^{-10}$ | `1e+10` | `-1e+10` | `0.0000` | `-0.0000` | `-9.0000` | `-9.0000` |

Evaluated apart the derivative decays and then vanishes, and joined it is better and then fails the same way.
Only the fully reduced form holds, and it holds at every row.
The `4e-07` of [](#code-numerics-gradient) is this idea at work in the library.

## Scale invariance: normalise before fourth powers

The Ledoit-Wolf intensity {cite:p}`ledoit2004well` is a ratio of two fourth powers of the anomalies,

$$\lambda = \min\left(1, \frac{b^2}{d^2}\right), \qquad d^2 = \Vert \hat{\gmat{\Sigma}}_x - \mu \mat{I} \Vert_F^2, \qquad b^2 = \frac{1}{N_\mathcal{E}^2} \sum_e \Vert \tilde{\vect{x}}^{(e)} \tilde{\vect{x}}^{(e) \top} - \hat{\gmat{\Sigma}}_x \Vert_F^2,$$ (eq-numerics-ledoit)

with $\mu = \op{tr}(\hat{\gmat{\Sigma}}_x) / D_x$.
Rescaling the ensemble by $s$ scales both $b^2$ and $d^2$ by $s^4$, so $\lambda$ does not change.
The floating-point range does not share that invariance.
Float32 holds about $10^{\pm 38}$, so a state of size $10^{12}$ overflows at the fourth power and one of size $10^{-12}$ underflows to zero.
So `gaussx.ledoit_wolf_shrinkage` divides the anomalies by their largest entry first, under a `stop_gradient`, because $\lambda$ does not depend on that scale.

```{code-block} python
:caption: Ledoit-Wolf shrinkage in float32 at several scales, written naively and from gaussx through filterax.
:label: code-numerics-scale
# Shapes: N_E = 40 members; D_x = 4 state components

# one shared factor plus noise, so that λ is well inside (0, 1)
shared = jr.normal(jr.key(0), (40, 1))  # → (N_E, 1)
extra = 0.5 * jr.normal(jr.key(1), (40, 4))  # → (N_E, D_x)
correlated = (shared + extra).astype(jnp.float32)  # → (N_E, D_x)


# user
# λ = min(1, b² / d²), evaluating the fourth powers as written
def naive_shrinkage(ensemble: Float[Array, "member state"]) -> Float[Array, ""]:
    # X̃
    anomalies = flx.ensemble_anomalies(ensemble)  # → (N_E, D_x)
    # Σ̂_x
    sample = einx.dot(  # → (D_x, D_x)
        "member state, member other -> state other", anomalies, anomalies
    ) / (ensemble.shape[0] - 1)
    # μ = tr(Σ̂_x) / D_x
    target_scale = jnp.trace(sample) / ensemble.shape[1]
    # d² = ‖Σ̂_x − μ I‖²_F
    distance = jnp.sum((sample - target_scale * jnp.eye(4, dtype=sample.dtype)) ** 2)
    # x̃⁽ᵉ⁾ x̃⁽ᵉ⁾ᵀ
    outer = einx.multiply(  # → (N_E, D_x, D_x)
        "member state, member other -> member state other", anomalies, anomalies
    )
    # b² = ∑ₑ ‖x̃⁽ᵉ⁾ x̃⁽ᵉ⁾ᵀ − Σ̂_x‖²_F / N_E²
    deviation = jnp.sum((outer - sample) ** 2) / ensemble.shape[0] ** 2
    return jnp.minimum(1.0, deviation / distance)  # → ()


for scale in (1e-12, 1e-6, 1.0, 1e6, 1e12):
    scaled = correlated * jnp.float32(scale)  # → (N_E, D_x)
    # today (filterax)
    # Σ̂_λ and λ
    _, intensity = flx.ledoit_wolf_shrinkage(scaled)
    print(f"{scale:g}: naive {naive_shrinkage(scaled):.4f}, filterax {intensity:.4f}")
```

It prints $\lambda$ for the naive form and for filterax:

| Scale | $10^{-12}$ | $10^{-6}$ | $1$ | $10^{6}$ | $10^{12}$ |
| --- | --- | --- | --- | --- | --- |
| Naive | `nan` | `0.0957` | `0.0957` | `0.0957` | `nan` |
| `flx.ledoit_wolf_shrinkage` | `0.0957` | `0.0957` | `0.0957` | `0.0957` | `0.0957` |

The rule is general: when a ratio is scale invariant, normalise before raising anything to a high power.

## Infinity times zero: mask before you divide

Automatic differentiation multiplies slopes, so an infinite slope meeting a zero gives `NaN` even where the answer is well defined.
Two patterns cause it.

The derivative of $\sqrt{\rho}$ at $\rho = 0$ is infinite, and a guard after the root does not help.
`jnp.where` still differentiates both branches, and the branch it discards contributes $\infty \times 0$.
The same goes for a division guarded after the fact.
The cure is to mask the argument first, so the unselected branch evaluates at a harmless value.

This pattern recurs across the code.
`_root_weights` in gaussx masks $\rho$ before the root, for a zero observation weight.
The Ledoit-Wolf code decides the clipped case before dividing, because $d^2 = 0$ would have an infinite slope.
`EKSStableScheduler` floors $\Vert \mat{S} \Vert_F^2$ inside the root, so its gradient stays finite at zero.
And `KalmanSmoother` replaces a masked datum with `jnp.where` rather than multiplying by the mask, since a `NaN` times zero is still `NaN`.

```{code-block} python
:caption: The slopes of √ρ and 1 / v at zero, the masked form of each, and a masked `NaN` through `KalmanSmoother`.
:label: code-numerics-masking
# Shapes: T = 6 times; D_x = 2; D_y = 2 in the last part


# user
# √ρ, guarded after the fact
def root_after(weight: Float[Array, ""]) -> Float[Array, ""]:
    return jnp.where(weight > 0.0, jnp.sqrt(weight), 0.0)


# user
# √ρ, with the argument masked first, as gaussx does
def root_before(weight: Float[Array, ""]) -> Float[Array, ""]:
    positive = weight > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, weight, 1.0)), 0.0)


# user
# 1 / v, guarded after the fact
def inverse_after(value: Float[Array, ""]) -> Float[Array, ""]:
    return jnp.where(value > 0.0, 1.0 / value, 0.0)


# user
# 1 / v, with the argument masked first
def inverse_before(value: Float[Array, ""]) -> Float[Array, ""]:
    positive = value > 0.0
    return jnp.where(positive, 1.0 / jnp.where(positive, value, 1.0), 0.0)


zero = jnp.asarray(0.0)
print(jax.grad(jnp.sqrt)(zero), jax.grad(root_after)(zero), jax.grad(root_before)(zero))
print(jax.grad(inverse_after)(zero), jax.grad(inverse_before)(zero))

# F, Σ_η and H
drift = jnp.array([[1.0, 0.1], [0.0, 1.0]])  # → (D_x, D_x)
small_noise = 0.01 * jnp.eye(2)  # → (D_x, D_x)
identity = jnp.eye(2)  # → (D_y, D_x)
# y, with one entry missing and one time missing entirely
readings = jr.normal(jr.key(0), (6, 2))  # → (T, D_y)
present = jnp.ones((6, 2), dtype=bool).at[2, 1].set(False).at[4, :].set(False)
holes = jnp.where(present, readings, jnp.nan)  # → (T, D_y)


# user
# log p(y) as a function of log σ², with Σ_ε = σ² I
def masked_evidence(log_variance: Float[Array, ""]) -> Float[Array, ""]:
    noise = jnp.exp(log_variance) * jnp.eye(2)  # → (D_y, D_y)
    # today (filterax)
    model = flx.KalmanSmoother(
        drift, small_noise, identity, noise, jnp.zeros(2), jnp.eye(2)
    )
    return model.smooth(holes, present).log_likelihood


at = jnp.log(jnp.asarray(0.1))
step = 1e-6
central = (masked_evidence(at + step) - masked_evidence(at - step)) / (2.0 * step)
print(f"{jax.grad(masked_evidence)(at):.6f} {central:.6f}")
```

It prints `inf nan 0.0`, then `nan 0.0`, then `19.608067 19.608067`.
The bare root has an infinite slope, the guard after it gives `nan`, and the mask before it gives `0.0`, and likewise for the division.
The gradient through `KalmanSmoother` with one `NaN` entry and one fully missing time matches a central difference.

## Known open limits

These four issues on gaussx are open as of 7 November 2026, checked with `gh issue view`.

- [gaussx#782](https://github.com/jejjohnson/gaussx/issues/782): `nonlinear_kalman_update` with `joseph=True` returns a covariance near $10^{19}$ at a rank-deficient predicted covariance, and its gradient is `NaN`. filterax's `UKI` passes `joseph=False` to `gaussx.sigma_point_kalman_update`.
- [gaussx#783](https://github.com/jejjohnson/gaussx/issues/783): `validate=True` on the same function rejects some well-posed float32 updates and misses indefiniteness at small scale, because its threshold does not follow the scale of the rounding. filterax never passes `validate=True`.
- [gaussx#785](https://github.com/jejjohnson/gaussx/issues/785): the ensemble-space route of `etkf_transform` loses the mean weights in float32 with noise variances spanning many decades, as measured above. With $D_y \ge N_\mathcal{E}$ there is no other route.
- [gaussx#787](https://github.com/jejjohnson/gaussx/issues/787): the derivative with respect to an observation weight of exactly zero is 0, where the one-sided derivative is not. The masking above makes it finite, and it does not make it the one-sided value. The issue notes that a compactly supported taper has zero slope at its support radius, so this reaches only a weight optimised directly.

[](#code-numerics-limits) reproduces the first and the last.

```{code-block} python
:caption: Two of the open limits, reproduced at the pinned gaussx.
:label: code-numerics-limits
# Shapes: D_x = 3 state components; D_y = 2 observations, in the first part


# Σ − G Σ_y Gᵀ at a rank-one Σ, against the dense form, for both updates
direction = jnp.array([1.0, 2.0, 0.5])  # → (D_x,)
# Σ = v vᵀ
rank_one = einx.multiply("state, other -> state other", direction, direction)
# H, Σ_ε, μ and y
blend = jnp.array([[1.0, 0.0, 0.5], [0.0, 1.0, -1.0]])  # → (D_y, D_x)
coloured = jnp.array([[0.5, 0.1], [0.1, 0.3]])  # → (D_y, D_y)
location = jnp.array([0.3, -0.2, 0.5])  # → (D_x,)
datum = jnp.array([0.4, 0.1])  # → (D_y,)
# Σ Hᵀ
cross = einx.dot("state other, obs other -> state obs", rank_one, blend)
# Σ_y = H Σ Hᵀ + Σ_ε
total = einx.dot("obs state, state other -> obs other", blend, cross) + coloured
# G Σ_y Gᵀ = Σ Hᵀ Σ_y⁻¹ H Σ
reduction = cross @ jnp.linalg.solve(total, einx.id("state obs -> obs state", cross))
dense_update = rank_one - reduction  # → (D_x, D_x)


# user
# ∑ of the updated covariance, as a function of the predicted one
def updated_total(cov: Float[Array, "state state"], joseph: bool) -> Float[Array, ""]:
    # today (gaussx)
    updated = gaussx.nonlinear_kalman_update(
        lambda state: blend @ state, location, cov, datum, coloured, joseph=joseph
    )[1]
    return jnp.sum(updated)  # → ()


for joseph in (True, False):
    # today (gaussx)
    updated = gaussx.nonlinear_kalman_update(
        lambda state: blend @ state, location, rank_one, datum, coloured, joseph=joseph
    )[1]
    finite = bool(jnp.isfinite(jax.grad(updated_total)(rank_one, joseph)).all())
    miss = jnp.abs(updated - dense_update).max()
    print(f"joseph={joseph}: error {miss:.1e}, gradient finite {finite}")

# H̃ and y, for the derivative at a zero weight
members = jr.normal(jr.key(0), (10, 3))  # → (N_E, D_y)
goal = jr.normal(jr.key(1), (3,))  # → (D_y,)
unit_noise = lx.DiagonalLinearOperator(0.2 * jnp.ones(3))  # (D_y, D_y) operator


# user
# ∑ sin W as a function of the observation weights ρ
def weighted_loss(rho: Float[Array, " obs"]) -> Float[Array, ""]:
    # today (gaussx)
    _, root = gaussx.etkf_transform(members, goal, unit_noise, obs_weights=rho)
    return jnp.sum(jnp.sin(root))  # → ()


rho = jnp.array([1.0, 0.5, 0.0])  # → (D_y,)
last = jnp.array([0.0, 0.0, 1.0])  # → (D_y,)
forward = (weighted_loss(rho + 1e-6 * last) - weighted_loss(rho)) / 1e-6
print(f"{jax.grad(weighted_loss)(rho)[2]:.4f} {forward:.4f}")
```

It prints `joseph=True: error 1.2e+19, gradient finite False`, `joseph=False: error 1.8e-15, gradient finite True` and `0.0000 0.6409`.
The last pair is the gradient at a zero weight and a forward difference with a step of $10^{-6}$.

## What to test

The themes above reduce to a short list for any new method.

- Compare float32 with a float64 reference, on the means and on the gradients, not only on finiteness or definiteness.
- Include low-rank inputs, noise variances spanning many decades, and a state of extreme scale.
- Check a custom derivative against a central difference in float64, at a repeated eigenvalue and at any boundary where a mask or floor acts.

## See also

- [Chapter 6](06_etkf.md): the repeated eigenvalue of $\gmat{\Lambda}_w$ and the Daleckii-Krein derivative
- [Chapter 14](14_differentiable.md): why these gradients matter for training
- [Chapter 17](17_exact_references.md): the square-root filter and its float32 means, against a float64 reference
- [API: filters](https://jejjohnson.github.io/filterax/reference/filters/), [smoothers](https://jejjohnson.github.io/filterax/reference/smoothers/) and [inflation](https://jejjohnson.github.io/filterax/reference/inflation/)

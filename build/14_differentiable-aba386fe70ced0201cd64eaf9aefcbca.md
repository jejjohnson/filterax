---
numbering:
  enumerator: "14.%s"
---

# Differentiable assimilation

A filter turns a model and a stream of observations into a likelihood of those observations, so it can train the model.
Every filterax analysis is a pure JAX computation, so `jax.grad` of that likelihood reaches dynamics parameters, decoder weights, noise amplitudes and the initial ensemble.
The question is not whether the gradient exists but which gradient to pay for.

Three ideas organise this chapter.
The exact gradient is a sum over products of cycle Jacobians, which shows exactly what each cheaper strategy throws away.
The strategies differ in what they store and what they drop, and the memory they use can be measured rather than assumed.
And the analysis contracts perturbations, so the gradient through a filter behaves very differently from the gradient through a free-running chaotic model.

## The loss

Let $\gvect{\theta} \in \mathbb{R}^{D_\theta}$ hold everything being learned.
One cycle applies the forward model to each member and then the analysis, including any inflator, written here as a map $a$:

$$\mat{X}_{t \mid t-1} = f(\mat{X}_{t-1 \mid t-1}; \gvect{\theta}), \qquad \mat{X}_{t \mid t} = a(\mat{X}_{t \mid t-1}, \vect{y}_t; \gvect{\theta}).$$ (eq-differentiable-cycle)

Both ensembles are $N_\mathcal{E} \times D_x$.
The evidence factorises over windows as $p(\vect{y}_{1:T}) = \prod_t p(\vect{y}_t \mid \vect{y}_{1:t-1})$, and the forecast ensemble gives each factor a Gaussian approximation ([chapter 13](13_diagnostics.md)).
The training loss is its negative logarithm:

$$\mathcal{L}(\gvect{\theta}) = \sum_{t=1}^{T} \mathcal{L}_t(\gvect{\theta}), \qquad \mathcal{L}_t = \frac{1}{2} \bar{\vect{r}}_t^\top \hat{\gmat{\Sigma}}_{y,t}^{-1} \bar{\vect{r}}_t + \frac{1}{2} \log \det \hat{\gmat{\Sigma}}_{y,t} + \frac{D_y}{2} \log 2 \pi.$$ (eq-differentiable-loss)

Here $\bar{\vect{r}}_t \in \mathbb{R}^{D_y}$ and $\hat{\gmat{\Sigma}}_{y,t} \in \mathbb{R}^{D_y \times D_y}$ are the mean residual and innovation covariance of the forecast ensemble $\mat{X}_{t \mid t-1}$, and `result.log_likelihoods[t]` is $-\mathcal{L}_t$.
This is a principled target for three kinds of $\gvect{\theta}$:

- Physical constants of the forward model, observation-operator gains and error-covariance amplitudes, where descent on $\mathcal{L}$ replaces hand-tuning.
- A neural surrogate in an `equinox` module, trained to predict the next observation from the filtered state {cite:p}`chen2022autodifferentiable`.
- The weights of a latent codec ([chapter 15](15_latent_da.md)).

## The exact gradient

Stack the analysis ensemble into a vector with $\op{vec}$, and define its sensitivity to the parameters, $\mat{S}_t = \partial \op{vec}(\mat{X}_{t \mid t}) / \partial \gvect{\theta} \in \mathbb{R}^{N_\mathcal{E} D_x \times D_\theta}$.
One cycle has two partial derivatives, taken with $\mat{X}_{t-1 \mid t-1}$ and $\gvect{\theta}$ held in turn:

$$\mat{J}_t = \frac{\partial \op{vec}(\mat{X}_{t \mid t})}{\partial \op{vec}(\mat{X}_{t-1 \mid t-1})}, \qquad \mat{D}_t = \frac{\partial \op{vec}(\mat{X}_{t \mid t})}{\partial \gvect{\theta}},$$ (eq-differentiable-partials)

with $\mat{J}_t \in \mathbb{R}^{N_\mathcal{E} D_x \times N_\mathcal{E} D_x}$ and $\mat{D}_t \in \mathbb{R}^{N_\mathcal{E} D_x \times D_\theta}$.
The chain rule gives a recursion for the sensitivity, and the same two partials of the window loss, $\vect{g}_t = \partial \mathcal{L}_t / \partial \gvect{\theta} \in \mathbb{R}^{D_\theta}$ and $\vect{v}_t = \partial \mathcal{L}_t / \partial \op{vec}(\mat{X}_{t-1 \mid t-1}) \in \mathbb{R}^{N_\mathcal{E} D_x}$, give the gradient:

$$\mat{S}_t = \mat{J}_t \mat{S}_{t-1} + \mat{D}_t, \qquad \nabla_{\gvect{\theta}} \mathcal{L} = \sum_{t=1}^{T} \left( \vect{g}_t + \mat{S}_{t-1}^\top \vect{v}_t \right).$$ (eq-differentiable-exact)

With a fixed initial ensemble $\mat{S}_0 = \mat{0}$, and unrolling the recursion shows the products that matter:

$$\mat{S}_t = \sum_{s=1}^{t} \left( \mat{J}_t \mat{J}_{t-1} \cdots \mat{J}_{s+1} \right) \mat{D}_s.$$ (eq-differentiable-unrolled)

Each term is the effect of $\gvect{\theta}$ at window $s$, carried forward to window $t$.
Every strategy in this chapter keeps a different subset of these terms, and whether the dropped terms matter depends on the size of those products.

Take a linear model $\mat{F}$ and a linear observation operator $\mat{H}$, and hold the gain $\mat{G}_t$ fixed.
The part of $\mat{J}_t$ acting on the ensemble mean then follows from $\bar{\vect{x}}_{t \mid t} = \mat{F} \bar{\vect{x}}_{t-1 \mid t-1} + \mat{G}_t (\vect{y}_t - \mat{H} \mat{F} \bar{\vect{x}}_{t-1 \mid t-1})$:

$$\frac{\partial \bar{\vect{x}}_{t \mid t}}{\partial \bar{\vect{x}}_{t-1 \mid t-1}} = (\mat{I} - \mat{G}_t \mat{H}) \mat{F}.$$ (eq-differentiable-contraction)

The model $\mat{F}$ may expand, but $\mat{I} - \mat{G}_t \mat{H}$ removes the observed part of a perturbation at every cycle.
This is the error propagator of the filter, and a filter that is tracking its observations keeps it contractive.

## Implementation in filterax

The Layer 2 models ([chapter 16](16_assimilation_cycle.md)) loop over windows in Python, which `jax.grad` differentiates, but the trace grows with $T$.
[`differentiable_assimilate`](https://jejjohnson.github.io/filterax/reference/differentiable/) runs the same cycle in one `jax.lax.scan`, with observations and times stacked on a leading axis.
Its result is the same `AssimilationResult`, and the gradient policy is the `adjoint=` argument.
[`road_enkf_loss_and_grad`](https://jejjohnson.github.io/filterax/reference/differentiable/) is a separate driver that accumulates local gradients of the dynamics.

The example learns a scalar in a linear model, $f(\vect{x}; \theta) = \theta \vect{x}$, where the truth has $\theta = 0.95$.
It has $N_\mathcal{E} = 12$ members and $D_x = 4$ components, of which $D_y = 2$ are observed for $T = 8$ windows with noise variance 0.1.

```{code-block} python
:caption: A twin experiment, the scan-based loss, and its gradient against a central difference.
:label: code-differentiable-twin
import einx
import equinox as eqx
import filterax as flx
import jax
import jax.numpy as jnp
import jax.random as jr
import jaxtyping as jt
import lineax as lx
import optax
from filterax.differentiable import (
    DirectAdjoint,
    RecursiveCheckpointAdjoint,
    TruncatedAdjoint,
    road_enkf_loss_and_grad,
)

jax.config.update("jax_enable_x64", True)

# Shapes: N_E = 12 members; D_x = 4; D_y = 2; T = 8 windows


# user
class ScaledDynamics(flx.AbstractDynamics):
    scale: jt.Float[jt.Array, ""]

    # f(x; θ) = θ x
    def __call__(self, state, t0, t1):
        return self.scale * state


# user
# h(x) selects components 0 and 1
def observe(state: jt.Float[jt.Array, " state"]) -> jt.Float[jt.Array, " obs"]:
    return state[:2]


# τₜ = t, the time stamp of window t
times = jnp.arange(1.0, 9.0)  # → (T,)
# x_true,t = 0.95ᵗ 𝟏
truth = einx.dot("t, state -> t state", 0.95**times, jnp.ones(4))  # → (T, D_x)
# yₜ = h(x_true,t) + εₜ, εₜ ~ 𝒩(0, 0.1 I)
observed = truth[:, :2] + jnp.sqrt(0.1) * jr.normal(jr.key(1), (8, 2))  # → (T, D_y)
# x⁽ᵉ⁾ ~ 𝒩(𝟏, 0.25 I)
members = 1.0 + 0.5 * jr.normal(jr.key(0), (12, 4))  # → (N_E, D_x)
# today (lineax)
# Σ_ε = 0.1 I
noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))  # (D_y, D_y) operator


# user
# 𝓛(θ) = −∑ₜ log 𝒩(r̄ₜ; 𝟎, Σ̂_y,t)
def loss(scale):
    # today (filterax)
    result = flx.differentiable_assimilate(
        flx.filters.ETKF(),
        ScaledDynamics(scale),
        observe,
        members,
        observed,
        times,
        noise,
    )
    return -jnp.sum(result.log_likelihoods)  # → ()


# 𝓛(θ) and ∇𝓛, compiled once
loss_and_grad = eqx.filter_jit(jax.value_and_grad(loss))
start = jnp.asarray(1.0)
step = 1e-6
value, exact = loss_and_grad(start)  # → (), ()
above, _ = loss_and_grad(start + step)  # → (), ()
below, _ = loss_and_grad(start - step)  # → (), ()
central = (above - below) / (2.0 * step)  # → ()
print(f"{loss_and_grad(jnp.asarray(0.95))[0]:.3f} {value:.3f}")
print(f"{exact:.6f} {central:.6f}")
```

It prints `3.854 4.782` and `34.163289 34.163289`.
The loss is lower at the true $\theta = 0.95$, the gradient at $\theta = 1$ points towards it, and it matches the central difference.

## Algorithm: local gradients by hand

ROAD-EnKF {cite:p}`chen2023reduced` jointly learns a latent surrogate and a decoder inside an ensemble Kalman filter.
For long windows it splits the sequence into short subsequences and backpropagates only within each, following truncated backpropagation through time.
`road_enkf_loss_and_grad` is the limit of that idea with subsequences of one window, and it differentiates the dynamics only.
Each window sees the previous analysis as a constant, which keeps $\vect{g}_t$ and drops $\mat{S}_{t-1}^\top \vect{v}_t$ from [](#eq-differentiable-exact):

$$\widehat{\nabla}_{\gvect{\theta}} \mathcal{L} = \sum_{t=1}^{T} \vect{g}_t.$$ (eq-differentiable-local)

[](#code-differentiable-local) writes this out with a dense loss [](#eq-differentiable-loss), so that it checks both the estimator and the likelihood the filter reports.

```{code-block} python
:caption: The local-gradient estimator [](#eq-differentiable-local) and the dense loss [](#eq-differentiable-loss), against `road_enkf_loss_and_grad`.
:label: code-differentiable-local
# Shapes: N_E = 12 members; D_x = 4; D_y = 2; T = 8 windows


# user
# 𝓛ₜ, and X_t|t, from X_t−1|t−1 held constant
def window_loss(scale, ensemble, target):
    # X_t|t−1 = θ X_t−1|t−1
    forecast = scale * ensemble  # → (N_E, D_x)
    # h⁽ᵉ⁾ = h(x⁽ᵉ⁾)
    mapped = jax.vmap(observe)(forecast)  # → (N_E, D_y)
    # h̄
    mapped_mean = einx.mean("[member] obs", mapped)  # → (D_y,)
    # H̃
    anomalies = mapped - mapped_mean  # → (N_E, D_y)
    # r̄ₜ = yₜ − h̄
    residual = target - mapped_mean  # → (D_y,)
    # Σ̂_y = H̃ᵀ H̃ / (N_E − 1) + Σ_ε
    innovation = einx.dot("member obs, member other -> obs other", anomalies, anomalies)
    innovation = innovation / 11.0 + 0.1 * jnp.eye(2)  # → (D_y, D_y)
    # Σ̂_y⁻¹ r̄ₜ
    solved = jnp.linalg.solve(innovation, residual)  # → (D_y,)
    # log det Σ̂_y
    _, log_det = jnp.linalg.slogdet(innovation)  # → ()
    # 𝓛ₜ = ½ r̄ₜᵀ Σ̂_y⁻¹ r̄ₜ + ½ log det Σ̂_y + (D_y / 2) log 2π
    value = 0.5 * (residual @ solved + log_det + 2.0 * jnp.log(2.0 * jnp.pi))  # → ()
    # today (filterax)
    analysis = flx.filters.ETKF().analysis(forecast, target, observe, noise)
    return value, analysis.particles


# gₜ = ∂𝓛ₜ / ∂θ
window_grad = jax.jit(jax.value_and_grad(window_loss, has_aux=True))


# user
# ∇̂𝓛 = ∑ₜ gₜ, with the ensemble carried between windows as a constant
def local_by_hand(scale):
    ensemble = members  # → (N_E, D_x)
    total_loss = 0.0
    total_grad = 0.0
    for target in observed:
        (value, analysis), grad = window_grad(scale, ensemble, target)
        total_loss = total_loss + value
        total_grad = total_grad + grad
        # X_t|t carried as a constant
        ensemble = jax.lax.stop_gradient(analysis)  # → (N_E, D_x)
    return total_loss, total_grad


by_hand_loss, by_hand_grad = local_by_hand(start)
# today (filterax)
road_loss, road_grad = road_enkf_loss_and_grad(
    ScaledDynamics(start), members, observed, times, noise, observe
)
print(f"{road_loss:.6f} {road_grad.scale:.6f}")
print(f"{abs(by_hand_loss - road_loss):.0e} {abs(by_hand_grad - road_grad.scale):.0e}")
print(f"{abs(road_loss - value):.0e}")
```

It prints `4.782471 14.383752`, then `2e-15 1e-14`, then `3e-15`.
The hand loop agrees with the library to round-off, in both the loss and the estimator.
The loss is the exact $\mathcal{L}$ of [](#code-differentiable-twin), because only the backward pass differs.
The local gradient of 14.38 is well under half the exact 34.16.

## The adjoint strategies

The `adjoint=` argument takes three strategies, whose names are shared with `pipekit_cycle.adjoints`; a structurally identical spec from that package is accepted too.
Forward values are identical under all of them.

- `DirectAdjoint`, the default, stores every step and computes [](#eq-differentiable-exact) exactly.
- `RecursiveCheckpointAdjoint` wraps the scan body in `jax.checkpoint`, so each step stores its inputs and recomputes the rest on the way back, and the gradient is exact {cite:p}`griewank2000algorithm`.
- `TruncatedAdjoint(k)` cuts the carry between windows with `stop_gradient`, for every window with $t \le T - k$.

The `checkpoints` field of `RecursiveCheckpointAdjoint` is carried but ignored, since the code applies one `jax.checkpoint` per step and has no schedule.
The older `checkpoint=True` flag selects the same strategy, with a `DeprecationWarning`.

The truncating cut sets $\mat{S}_t = \mat{0}$ for $t \le T - k$ in [](#eq-differentiable-exact), and it does not touch the per-window outputs.
So two consequences follow, and neither is a sliding window of depth $k$.
A loss summed over the history keeps $\vect{g}_t$ at every window, but only the windows with $t \ge T - k + 2$ see any cross-window term.
So `TruncatedAdjoint(1)` on the summed loss is [](#eq-differentiable-local) exactly, and window $t$ gains cross-window terms only once $k$ reaches $T - t + 2$.
A loss on the final ensemble depends on $\mat{S}_T$, and there the cut keeps the last $k$ terms of [](#eq-differentiable-unrolled).

```{code-block} python
:caption: The gradient under every strategy, for the summed loss and for a loss on the final ensemble.
:label: code-differentiable-strategies
# Shapes: N_E = 12 members; D_x = 4; D_y = 2; T = 8 windows


# user
# (𝓛(θ), 𝓛_final(θ)), with 𝓛_final(θ) = ‖X_T|T‖²
def both_losses(scale, adjoint):
    # today (filterax)
    result = flx.differentiable_assimilate(
        flx.filters.ETKF(),
        ScaledDynamics(scale),
        observe,
        members,
        observed,
        times,
        noise,
        adjoint=adjoint,
    )
    values = jnp.stack([-jnp.sum(result.log_likelihoods), jnp.sum(result.particles**2)])
    return values, values  # → (2,)


both_gradients = eqx.filter_jit(jax.jacrev(both_losses, has_aux=True))
for name, strategy in {
    "direct": DirectAdjoint(),
    "recursive": RecursiveCheckpointAdjoint(),
    "k=1": TruncatedAdjoint(k=1),
    "k=3": TruncatedAdjoint(k=3),
    "k=8": TruncatedAdjoint(k=8),
}.items():
    gradients, values = both_gradients(start, strategy)  # → (2,), (2,)
    print(name, f"{gradients[0]:.4f} {gradients[1]:.4f}", values.round(6))
```

It prints the following, and every row also prints the same forward values, `4.782471` and `42.442051`.

| Strategy | $\partial \mathcal{L} / \partial \theta$ | $\partial \mathcal{L}_\text{final} / \partial \theta$ |
| --- | --- | --- |
| `DirectAdjoint` | `34.1633` | `559.6458` |
| `RecursiveCheckpointAdjoint` | `34.1633` | `559.6458` |
| `TruncatedAdjoint(1)` | `14.3838` | `81.5554` |
| `TruncatedAdjoint(3)` | `13.3195` | `234.9817` |
| `TruncatedAdjoint(8)` | `34.1633` | `559.6458` |

`TruncatedAdjoint(1)` on the summed loss equals the local gradient of [](#code-differentiable-local), and $k = T$ equals the exact one.
Between them the summed loss does not approach the exact value monotonically.
The value at $k = 3$ is further away than at $k = 1$, because the extra terms it keeps can have either sign.
On the final ensemble the gradient builds up term by term as [](#eq-differentiable-unrolled) says.

At $T = 1$ there are no cross-window terms, so the local gradient equals the exact one, which `tests/test_road.py` pins.

## Numerics: what each strategy stores

The cost of a reverse pass is whatever the scan stores for the way back, and XLA can report it.
[](#code-differentiable-memory) compiles each gradient and reads the size of its temporary buffers at two window counts, on a larger problem of $N_\mathcal{E} = 20$ and $D_x = 30$.
The loss on the final ensemble is the case where truncation has the most to save.

```{code-block} python
:caption: Temporary buffers of the compiled gradient, in kilobytes, at two window counts.
:label: code-differentiable-memory
# Shapes: N_E = 20 members; D_x = 30; D_y = 10; T windows

# x⁽ᵉ⁾ ~ 𝒩(𝟎, I)
big_members = jr.normal(jr.key(0), (20, 30))  # → (N_E, D_x)
big_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(10))  # (D_y, D_y) operator


# user
# h(x) selects every third component
def observe_big(state: jt.Float[jt.Array, " state"]) -> jt.Float[jt.Array, " obs"]:
    return state[::3]


# user
# 𝓛_final(θ) = ‖X_T|T‖² over T windows
def big_loss(scale, adjoint, targets, stamps):
    # today (filterax)
    result = flx.differentiable_assimilate(
        flx.filters.ETKF(),
        ScaledDynamics(scale),
        observe_big,
        big_members,
        targets,
        stamps,
        big_noise,
        adjoint=adjoint,
    )
    return jnp.sum(result.particles**2)  # → ()


# user
# bytes of temporary buffers XLA allocates for a compiled function
def temp_kilobytes(function, argument):
    compiled = jax.jit(function).lower(argument).compile()
    return compiled.memory_analysis().temp_size_in_bytes / 1e3


for name, strategy in [
    ("direct", DirectAdjoint()),
    ("recursive", RecursiveCheckpointAdjoint()),
    ("k=1", TruncatedAdjoint(k=1)),
    ("local", None),
]:
    sizes = []
    for count in (20, 80):
        targets = jr.normal(jr.key(1), (count, 10))  # → (T, D_y)
        stamps = jnp.arange(1.0, count + 1.0)  # → (T,)
        if strategy is None:

            # user
            def function(scale):
                # today (filterax)
                return road_enkf_loss_and_grad(
                    ScaledDynamics(scale),
                    big_members,
                    targets,
                    stamps,
                    big_noise,
                    observe_big,
                )

        else:

            # user
            def function(scale):
                return jax.grad(big_loss)(scale, strategy, targets, stamps)

        sizes.append(temp_kilobytes(function, start))
    slope = (sizes[1] - sizes[0]) / 60.0
    print(name, [round(size) for size in sizes], round(slope, 1))
```

It prints:

| Strategy | $T = 20$ (kB) | $T = 80$ (kB) | Per window (kB) |
| --- | --- | --- | --- |
| `DirectAdjoint` | `553` | `2075` | `25.4` |
| `RecursiveCheckpointAdjoint` | `151` | `439` | `4.8` |
| `TruncatedAdjoint(1)` | `554` | `2076` | `25.4` |
| `road_enkf_loss_and_grad` | `31` | `31` | `0.0` |

Only the local driver is flat in $T$, because it finishes each window's backward pass before starting the next.
`RecursiveCheckpointAdjoint` is linear in $T$ with a smaller slope.
Its 4.8 kB per window is exactly one ensemble of $N_\mathcal{E} D_x = 600$ floats of 8 bytes, which is the step input that `jax.checkpoint` must keep.
A recursive schedule in the style of {cite:t}`griewank2000algorithm` would stop that growth, but the code does not implement one.
`TruncatedAdjoint(1)` costs the same as `DirectAdjoint`, even on this final-state loss with only one cycle of gradient.
The cut is a runtime `jnp.where` inside the scan body, so the scan still stores every step.
So its value is the bounded depth of the gradient and not its memory.
The docstrings that promise $O(\sqrt{T})$ and $O(1)$ overstate what these two strategies do today.

The other saving of the scan is at trace time.
[](#code-differentiable-trace) counts the operations JAX traces for each loop, including those inside the scan's body.

```{code-block} python
:caption: The trace of the Python loop grows with the number of windows, and the scan's does not.
:label: code-differentiable-trace
# Shapes: N_E = 12 members; D_x = 4; D_y = 2; T windows


# user
# number of traced operations, following sub-programs
def count_operations(program):
    total = 0
    for equation in program.eqns:
        total += 1
        for parameter in equation.params.values():
            inner = getattr(parameter, "jaxpr", parameter)
            if hasattr(inner, "eqns"):
                total += count_operations(inner)
    return total


for count in (2, 8):
    windows = list(zip(observed[:count], times[:count]))

    # user
    def python_loop(scale):
        # today (filterax)
        model = flx.ETKF(dynamics=ScaledDynamics(scale), obs_op=observe)
        return model.assimilate(members, windows, noise).log_likelihoods

    # user
    def scan_loop(scale):
        # today (filterax)
        return flx.differentiable_assimilate(
            flx.filters.ETKF(),
            ScaledDynamics(scale),
            observe,
            members,
            observed[:count],
            times[:count],
            noise,
        ).log_likelihoods

    python_size = count_operations(jax.make_jaxpr(python_loop)(start))
    scan_size = count_operations(jax.make_jaxpr(scan_loop)(start))
    print(count, python_size, scan_size)
```

It prints `2 346 171` and `8 1372 169`.
Four times the windows costs four times the operations in the Python loop, and nothing in the scan.

## Numerics: chaos and the filter

The exact adjoint of a free-running chaotic model grows like $e^{\lambda_\text{max} T \Delta t}$, with $\lambda_\text{max}$ the leading Lyapunov exponent and $\Delta t$ the time per window {cite:p}`lea2000sensitivity`.
That makes the gradient useless over long windows, and it is often quoted as a reason to truncate a filter's gradient.
But [](#eq-differentiable-contraction) says the analysis changes the picture, because $\mat{J}_t$ includes the contraction.
[](#code-differentiable-chaos) measures it on Lorenz-96 with $D_x = 20$ sites and the forcing $\theta = 8$ as the parameter.
It compares the free run with a filter of $N_\mathcal{E} = 25$ members, multiplicative inflation of 1.02 and unit noise variance.
The filter observes every second or every fifth site.

```{code-block} python
:caption: The gradient with respect to the forcing, in a free run and through the filter.
:label: code-differentiable-chaos
# Shapes: T = 400 windows; D_x = 20 sites; N_E = 25 members; D_y observed sites


# user
class Lorenz96(flx.AbstractDynamics):
    forcing: jt.Float[jt.Array, ""]

    # dxₖ/dt = (xₖ₊₁ − xₖ₋₂) xₖ₋₁ − xₖ + θ
    def tendency(self, state):
        advection = (jnp.roll(state, -1) - jnp.roll(state, 2)) * jnp.roll(state, 1)
        return advection - state + self.forcing

    # one RK4 step of Δt = 0.05
    def __call__(self, state, t0=None, t1=None):
        k1 = self.tendency(state)
        k2 = self.tendency(state + 0.025 * k1)
        k3 = self.tendency(state + 0.025 * k2)
        k4 = self.tendency(state + 0.05 * k3)
        return state + 0.05 / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


true_model = Lorenz96(jnp.asarray(8.0))
# x₀ on the attractor, after 500 steps from a perturbed rest state
spun_up = jax.lax.fori_loop(  # → (D_x,)
    0, 500, lambda _, state: true_model(state), 8.0 + jr.normal(jr.key(0), (20,))
)


# user
# xₜ₊₁ = f(xₜ; θ), stacked
def advance(state, _):
    next_state = true_model(state)
    return next_state, next_state


_, long_truth = jax.lax.scan(advance, spun_up, length=400)  # → (400, D_x)
# x⁽ᵉ⁾ ~ 𝒩(x₀, I)
start_ensemble = spun_up + jr.normal(jr.key(2), (25, 20))  # → (N_E, D_x)
# today (filterax)
inflator = flx.MultiplicativeInflator(factor=1.02)


# user
# ‖x_T‖² / D_x with no observations
def free_run_loss(forcing, count):
    model = Lorenz96(forcing)

    # user
    # xₜ₊₁ = f(xₜ; θ)
    def forward(state, _):
        return model(state), None

    final, _ = jax.lax.scan(forward, spun_up, length=count)  # → (D_x,)
    return jnp.mean(final**2)  # → ()


for count in (100, 400):
    free_gradient = jax.grad(free_run_loss)(jnp.asarray(8.0), count)
    print("free", count, f"{free_gradient:.1f}")

for stride in (2, 5):
    sites = jnp.arange(0, 20, stride)  # → (D_y,)
    size = sites.shape[0]
    # yₜ = H x_true,t + εₜ, εₜ ~ 𝒩(0, I)
    long_observed = long_truth[:, sites] + jr.normal(jr.key(1), (400, size))
    unit_noise = lx.DiagonalLinearOperator(jnp.ones(size))  # (D_y, D_y) operator
    stamps = 0.05 * jnp.arange(1.0, 401.0)  # → (T,)

    # user
    # h(x) selects the observed sites
    def observe_sites(state):
        return state[sites]

    # user
    # 𝓛(θ) over the 400 windows
    def filter_loss(forcing):
        # today (filterax)
        result = flx.differentiable_assimilate(
            flx.filters.ETKF(),
            Lorenz96(forcing),
            observe_sites,
            start_ensemble,
            long_observed,
            stamps,
            unit_noise,
            inflator=inflator,
        )
        return -jnp.sum(result.log_likelihoods)  # → ()

    exact_gradient = jax.grad(filter_loss)(jnp.asarray(8.0))
    print(stride, f"{exact_gradient:.1f}")
```

It prints `free 100 -286.9` and `free 400 903801938.6` for the free run.
For the filter over 400 windows it prints `2 -14.0` with every second site observed and `5 401.9` with every fifth.

The free run gains more than six orders of magnitude between $T = 100$ and $T = 400$, as the Lyapunov argument predicts.
The filter does not.
With half the sites observed, the exact gradient of a loss summed over 400 windows has magnitude 14, because each analysis contracts what the model expands.
So chaos alone is no reason to truncate a filter's gradient.
The growth returns when the contraction weakens.
With one site in five observed, the exact gradient reaches 401.9.

## Training with exact and local gradients

The local gradient is not the gradient of $\mathcal{L}$, so its zero need not be the minimum of $\mathcal{L}$.
[](#code-differentiable-training) reads both gradients on a grid of forcings, over $T = 100$ windows.
It then descends from $\theta = 6$ for 60 steps of Adam, with a learning rate of 0.1.
`road_enkf_grad_step` wraps one local step with an `optax` optimiser, and [chapter 15](15_latent_da.md) uses it.

```{code-block} python
:caption: Learning the Lorenz-96 forcing with the exact gradient and with the local gradient.
:label: code-differentiable-training
# Shapes: T = 100 windows; D_x = 20 sites; N_E = 25 members; D_y = 10 observed sites

sites = jnp.arange(0, 20, 2)  # → (D_y,)
# yₜ = H x_true,t + εₜ, εₜ ~ 𝒩(0, I)
train_observed = long_truth[:100, sites] + jr.normal(jr.key(1), (100, 10))  # → (T, D_y)
train_times = 0.05 * jnp.arange(1.0, 101.0)  # → (T,)
train_noise = lx.DiagonalLinearOperator(jnp.ones(10))  # (D_y, D_y) operator


# user
# h(x) selects every second site
def observe_train(state):
    return state[sites]


# user
# 𝓛(θ) as a function of the dynamics module
def train_loss(model):
    # today (filterax)
    result = flx.differentiable_assimilate(
        flx.filters.ETKF(),
        model,
        observe_train,
        start_ensemble,
        train_observed,
        train_times,
        train_noise,
        inflator=inflator,
    )
    return -jnp.sum(result.log_likelihoods)  # → ()


exact_fn = eqx.filter_jit(eqx.filter_value_and_grad(train_loss))


# user
# 𝓛(θ) and ∇̂𝓛, from the same arguments
def local_fn(model):
    # today (filterax)
    return road_enkf_loss_and_grad(
        model,
        start_ensemble,
        train_observed,
        train_times,
        train_noise,
        observe_train,
        inflator=inflator,
    )


local_fn = eqx.filter_jit(local_fn)
for forcing in (6.0, 7.5, 7.75, 8.0, 8.25, 10.0):
    model = Lorenz96(jnp.asarray(forcing))
    value, exact_grad = exact_fn(model)
    _, local_grad = local_fn(model)
    print(forcing, f"{value:.1f} {exact_grad.forcing:.1f} {local_grad.forcing:.2f}")

optimiser = optax.adam(0.1)
for name, gradient_fn in [("exact", exact_fn), ("local", local_fn)]:
    model = Lorenz96(jnp.asarray(6.0))
    state = optimiser.init(eqx.filter(model, eqx.is_inexact_array))
    for _ in range(60):
        _, gradient = gradient_fn(model)
        updates, state = optimiser.update(gradient, state)
        model = eqx.apply_updates(model, updates)
    print(name, f"{model.forcing:.3f} {exact_fn(model)[0]:.1f}")
```

It prints:

| $\theta$ | $\mathcal{L}$ | Exact gradient | Local gradient |
| --- | --- | --- | --- |
| 6.0 | `2132.0` | `-844.5` | `-10.93` |
| 7.5 | `1491.8` | `-86.7` | `-1.04` |
| 7.75 | `1472.3` | `-67.3` | `0.32` |
| 8.0 | `1462.8` | `4.4` | `1.20` |
| 8.25 | `1479.5` | `136.0` | `2.01` |
| 10.0 | `3569.7` | `2556.0` | `6.98` |

It then prints `exact 7.960 1462.9` and `local 7.651 1479.5`, the learned forcing and its loss.
Both gradients point the right way from far off, so both methods make good early progress from $\theta = 6$.
They differ near the optimum.
The exact gradient changes sign between 7.75 and 8.0, close to the true value of 8.
The local gradient changes sign between 7.5 and 7.75, and its descent ends near there.
The bias is modest here, but it is a bias in the answer and not only in the path.
The local gradient is also 77 times smaller at $\theta = 6$, which Adam's scaling hides and a fixed-step optimiser would not.

## Numerics: float32

The gradient inherits the precision of the analysis ([chapter 18](18_numerics.md)), and for ordinary variances that is enough.

[](#code-differentiable-float32) repeats the exact gradient at $\theta = 8$ of [](#code-differentiable-training) in float32, over 100 cycles of a chaotic model.

```{code-block} python
:caption: The exact gradient through 100 Lorenz-96 cycles in float32, against float64.
:label: code-differentiable-float32
# Shapes: T = 100 windows; D_x = 20 sites; N_E = 25 members; D_y = 10 observed sites


# user
# 𝓛(θ) in float32
def float32_loss(model):
    # today (filterax)
    result = flx.differentiable_assimilate(
        flx.filters.ETKF(),
        model,
        observe_train,
        jnp.asarray(start_ensemble, jnp.float32),
        jnp.asarray(train_observed, jnp.float32),
        jnp.asarray(train_times, jnp.float32),
        lx.DiagonalLinearOperator(jnp.ones(10, jnp.float32)),
        inflator=inflator,
    )
    return -jnp.sum(result.log_likelihoods)  # → ()


# today (equinox)
single = eqx.filter_grad(float32_loss)(Lorenz96(jnp.asarray(8.0, jnp.float32)))
_, double = exact_fn(Lorenz96(jnp.asarray(8.0)))
print(single.forcing.dtype, f"{single.forcing:.3f} {double.forcing:.3f}")
```

It prints `float32 4.367 4.366`, so the float32 gradient keeps three to four figures through 100 cycles.
Float32 limits on the analysis itself, with noise variances spanning many decades, are in [chapter 6](06_etkf.md).

## What is refused, and what is not

Two ingredients inject PRNG draws, which `jax.grad` would treat as constants, so the gradient would be that of one arbitrary noise realisation.
`differentiable_assimilate` and `road_enkf_loss_and_grad` both refuse them with a `ValueError` at call time:

- Among the filters, `StochasticEnKF` ([chapter 5](05_stochastic_enkf.md)) and `ETKF_Livings` ([chapter 6](06_etkf.md)), so use `ETKF`, `EnSRF`, `ESTKF`, `EnSRF_Serial` or `LETKF`.
- Among the inflators, `AdditiveInflator` ([chapter 12](12_inflation.md)), so use `MultiplicativeInflator`, `RTPS` or `RTPP`.

The check looks only for those three classes, so a stochastic filter or inflator of your own passes it silently.

The inflation parameters of the built-in inflators are static fields, so `eqx.filter_grad` over one sees no leaves and they cannot be trained as module fields.
To learn one, write a small `AbstractInflator` with the parameter as an array field and call the primitive, as in [](#code-differentiable-inflator).

The analysis itself is safe.
The textbook ETKF has a repeated eigenvalue whose `eigh` derivative is `NaN`, and [chapter 6](06_etkf.md) shows how gaussx removes it.

```{code-block} python
:caption: The refusals, and a trainable inflation factor against a central difference.
:label: code-differentiable-inflator
# Shapes: N_E = 12 members; D_x = 4; D_y = 2; T = 8 windows

for name, filter_, inflation in [
    ("StochasticEnKF", flx.filters.StochasticEnKF(), None),
    ("ETKF_Livings", flx.filters.ETKF_Livings(key=jr.key(0)), None),
    (
        "AdditiveInflator",
        flx.filters.ETKF(),
        flx.AdditiveInflator(
            noise_cov=lx.DiagonalLinearOperator(jnp.ones(4)), base_key=jr.key(0)
        ),
    ),
]:
    try:
        # today (filterax)
        flx.differentiable_assimilate(
            filter_,
            ScaledDynamics(start),
            observe,
            members,
            observed,
            times,
            noise,
            inflator=inflation,
        )
    except ValueError as error:
        print(name, "refused:", str(error)[:40])


# user
class TrainableInflation(flx.AbstractInflator):
    factor: jt.Float[jt.Array, ""]

    # X̃ ← ζ X̃
    def __call__(self, particles, forecast_particles=None, **_):
        # today (filterax)
        return flx.inflate_multiplicative(particles, self.factor)


# user
# 𝓛(ζ)
def inflation_loss(factor):
    # today (filterax)
    result = flx.differentiable_assimilate(
        flx.filters.ETKF(),
        ScaledDynamics(start),
        observe,
        members,
        observed,
        times,
        noise,
        inflator=TrainableInflation(factor),
    )
    return -jnp.sum(result.log_likelihoods)  # → ()


inflation_grad = eqx.filter_jit(jax.value_and_grad(inflation_loss))
zeta = jnp.asarray(1.05)
above, gradient = inflation_grad(zeta + step)  # → (), ()
below, _ = inflation_grad(zeta - step)  # → (), ()
print(f"{inflation_grad(zeta)[1]:.6f} {(above - below) / (2.0 * step):.6f}")
# today (filterax)
print(len(jax.tree.leaves(flx.MultiplicativeInflator(factor=1.05))))

```

It prints one refusal per ingredient, then `-0.848903 -0.848903`, then `0`.
The trainable factor matches its central difference, and the built-in inflator has no leaves for a gradient to reach.

## When it breaks

The gradient is only as healthy as the filter around it, and five limits are worth knowing.

- A filter that has stopped tracking loses the contraction of [](#eq-differentiable-contraction), and the exact gradient grows, as the sparse case of [](#code-differentiable-chaos) shows.
- The local gradient converges to its own zero and not to the minimum of $\mathcal{L}$, as [](#code-differentiable-training) shows.
- `road_enkf_loss_and_grad` differentiates the dynamics only, so decoder weights, noise amplitudes and inflation factors need `differentiable_assimilate`.
- `TruncatedAdjoint` saves no memory today, and its summed-loss gradient does not improve monotonically with $k$.
- `RTPS` gives a `NaN` gradient when a component has zero forecast spread, even though the forward pass stays finite.

The last of these is cheap to demonstrate.
Holding one component constant across the ensemble, as a masked or pinned variable would be, is enough.

```{code-block} python
:caption: `RTPS` gives a `NaN` gradient when one component has zero forecast spread.
:label: code-differentiable-rtps-nan
# Shapes: N_E = 12 members; D_x = 4; D_y = 2; T = 8 windows

# component 3 has zero spread
pinned = members.at[:, 3].set(1.0)  # → (N_E, D_x)


# user
# 𝓛(θ) with RTPS, or with RTPP, in the loop
def pinned_loss(scale, inflation):
    # today (filterax)
    result = flx.differentiable_assimilate(
        flx.filters.ETKF(),
        ScaledDynamics(scale),
        observe,
        pinned,
        observed,
        times,
        noise,
        inflator=inflation,
    )
    return -jnp.sum(result.log_likelihoods)  # → ()


# today (filterax)
for inflation in (flx.RTPS(alpha=0.5), flx.RTPP(alpha=0.5)):
    value, gradient = jax.value_and_grad(pinned_loss)(start, inflation)
    print(f"{value:.3f} {gradient:.3f}")
```

It prints `4.813 nan` and `4.806 19.830`.
`RTPS` takes a standard deviation per component, whose derivative is infinite at zero variance, and `RTPP` has no such step.
So keep `RTPS` out of a training loop when any component can be pinned, or use `RTPP`.

## When a differentiable filter is the right answer

Use `differentiable_assimilate` with `DirectAdjoint` when the windows are short enough to store, since its gradient is exact and nothing needs tuning.
`RecursiveCheckpointAdjoint` gives the same gradient for a constant-factor saving, about five times in [](#code-differentiable-memory), so reach for it when `DirectAdjoint` only just fails to fit.

Use `road_enkf_grad_step` when memory must stop growing with $T$, the dynamics are what is being learned, and a small bias in the optimum is acceptable.

Use `TruncatedAdjoint` to bound the depth of a gradient on something `road_enkf_loss_and_grad` cannot reach, such as a decoder.
For a summed loss on the dynamics alone, `TruncatedAdjoint(1)` computes the same gradient as the local driver at far higher memory.

Reach for something else when the forward model cannot be differentiated: the ensemble Kalman processes of [chapter 9](09_ensemble_kalman_processes.md) need only forward evaluations.
For a short experiment with no training at all, the Layer 2 loop of [chapter 16](16_assimilation_cycle.md) is simpler.

## See also

- [Chapter 5](05_stochastic_enkf.md): the perturbed-observation filter excluded from training
- [Chapter 6](06_etkf.md): the closed-form `eigh` derivative that makes the transform gradient-safe
- [Chapter 12](12_inflation.md): the inflators that may and may not appear in a training loop
- [Chapter 13](13_diagnostics.md): the innovation likelihood used as the loss
- [Chapter 15](15_latent_da.md): the same gradients flowing into a latent codec
- [Chapter 16](16_assimilation_cycle.md): the Layer 2 loop that `differentiable_assimilate` replaces
- [API: differentiable training](https://jejjohnson.github.io/filterax/reference/differentiable/)

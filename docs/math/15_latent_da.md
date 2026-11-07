---
numbering:
  enumerator: "15.%s"
---

# Latent-space ensemble assimilation

When the state is a high-resolution field, every member needs an expensive forecast.
The sample covariance also lives in a space far larger than the dynamics actually explore.
If the system evolves near a low-dimensional manifold, a codec can expose it, and the ensemble can be assimilated in those coordinates {cite:p}`peyron2021latent,amendola2021data`.

Three ideas organise this chapter.
The analysis is unchanged: only the observation operator and the forecast are rewired, so every ensemble filter works as it stands.
A linear codec gives an exact statement of what this buys and what it loses.
And a nonlinear decoder is not free, because it moves nonlinearity into the observation operator, where the update only linearises it.

## The construction

Take an encoder $g_\text{enc}: \mathbb{R}^{D_x} \to \mathbb{R}^{D_z}$ and a decoder $g_\text{dec}: \mathbb{R}^{D_z} \to \mathbb{R}^{D_x}$, usually with $D_z \ll D_x$ and $g_\text{dec}(g_\text{enc}(\vect{x})) \approx \vect{x}$ on the data.
The latent ensemble $\mat{Z} \in \mathbb{R}^{N_\mathcal{E} \times D_z}$ encodes each member, and the two pieces of wiring close the cycle:

$$\vect{z}^{(e)} = g_\text{enc}(\vect{x}^{(e)}), \qquad h_z = h \circ g_\text{dec}: \mathbb{R}^{D_z} \to \mathbb{R}^{D_y}.$$ (eq-latent-lift)

The lifted operator $h_z$ is what the filter sees.
Observations and $\gmat{\Sigma}_\varepsilon$ stay in observation space, untouched, and an ensemble filter never needs the Jacobian of $h_z$.

The forecast has two forms, which differ in cost.
A learned surrogate $f_z$ replaces the expensive model, and the alternative takes the state-space model $f$ through the codec:

$$\vect{z}_t = f_z(\vect{z}_{t-1}), \qquad \vect{z}_t = g_\text{enc}\big(f(g_\text{dec}(\vect{z}_{t-1}))\big).$$ (eq-latent-dynamics)

The second form saves nothing in the forecast, and the analysis still runs in $\mathbb{R}^{D_z}$.

### The analysis in latent coordinates

Take the update of [chapter 6](06_etkf.md) and substitute $\tilde{\mat{Z}}$ for $\tilde{\mat{X}}$.
Each member maps to $\vect{h}^{(e)} = h_z(\vect{z}^{(e)})$, and the mean residual $\bar{\vect{r}}$ and anomalies $\tilde{\mat{H}}$ follow as before.
The precision of the weights and the mean weights involve only observation-space quantities,

$$\gmat{\Lambda}_w = c \mat{I} + \tilde{\mat{H}} \gmat{\Sigma}_\varepsilon^{-1} \tilde{\mat{H}}^\top, \qquad \bar{\vect{w}} = \gmat{\Lambda}_w^{-1} \tilde{\mat{H}} \gmat{\Sigma}_\varepsilon^{-1} \bar{\vect{r}},$$ (eq-latent-weights)

with $c = N_\mathcal{E} - 1$, and the analysis applies them to the latent members:

$$\bar{\vect{z}}_{t \mid t} = \bar{\vect{z}} + \tilde{\mat{Z}}^\top \bar{\vect{w}}, \qquad \tilde{\mat{Z}}_{t \mid t} = \mat{W} \tilde{\mat{Z}}.$$ (eq-latent-analysis)

So $D_z$ enters only through the last two products, which cost $O(N_\mathcal{E}^2 D_z)$ in place of $O(N_\mathcal{E}^2 D_x)$ on the ensemble-space route.
The analysis was only ever linear in the state size, so that is not where most of the saving lies.
What changes is the rank of the sample covariance, which is at most $\min(N_\mathcal{E} - 1, D_z)$, and the forecast, when a surrogate is available.

### A linear codec is an exact reduction

Let $\mat{U} \in \mathbb{R}^{D_x \times D_z}$ have orthonormal columns, with $g_\text{enc}(\vect{x}) = \mat{U}^\top \vect{x}$ and $g_\text{dec}(\vect{z}) = \mat{U} \vect{z}$.
Suppose every member lies in the range of $\mat{U}$, so that $\mat{X} = \mat{Z} \mat{U}^\top$.
Then decoding recovers the members exactly, so $\tilde{\mat{H}}$ and $\bar{\vect{r}}$ are the same in both spaces, and so is everything in [](#eq-latent-weights).
Multiplying [](#eq-latent-analysis) by $\mat{U}^\top$ on the right gives

$$\mat{X}_{t \mid t} = \mat{Z}_{t \mid t} \mat{U}^\top,$$ (eq-latent-reduction)

which is the state-space ETKF, member for member.
In this case latent assimilation is purely a change of coordinates.

The encoded forecast in [](#eq-latent-dynamics) with a linear model $\mat{F}$ becomes $\mat{U}^\top \mat{F} \mat{U}$, a Galerkin projection.
That equals the true dynamics on the range of $\mat{U}$ only if $\mat{F}$ maps that range to itself.

Now suppose the members are not in the range.
Write $\vect{x}^{(e)} = \mat{U} \vect{z}^{(e)} + \vect{x}_\perp^{(e)}$ with $\mat{U}^\top \vect{x}_\perp^{(e)} = \vect{0}$.
The first encode discards $\vect{x}_\perp^{(e)}$ permanently, and so the codec acts in two ways.
Ensemble noise in those directions can no longer use up the $N_\mathcal{E} - 1$ directions of the update, which is the benefit.
Truth in those directions can never be recovered, which is the cost, so any part of $\vect{x}_\text{true}$ outside the range is an error floor:

$$\Vert \bar{\vect{x}}_{t \mid t} - \vect{x}_{\text{true},t} \Vert^2 \ge \Vert (\mat{I} - \mat{U} \mat{U}^\top) \vect{x}_{\text{true},t} \Vert^2.$$ (eq-latent-floor)

## Implementation in filterax

The wiring is thin composition over Layer 1, with no new analysis classes.

| Layer | Name | What it does |
| --- | --- | --- |
| 0 | `latent_ensemble(encoder, X)` | `vmap`s a callable encoder over members |
| 0 | `decode_ensemble(decoder, Z)` | `vmap`s a callable decoder over members |
| 0 | `IdentityLatentMap`, `identity_latent_map(dim)` | the codec $g_\text{enc} = g_\text{dec} = \op{id}$, for tests |
| 1 | `LatentDynamics(inner)` | an `AbstractDynamics` that calls `inner.step(z, t1 - t0)` |
| 1 | `EncodedDynamics(latent_map, inner)` | the second form of [](#eq-latent-dynamics) |
| 1 | `LiftedObs(decoder, inner)` | $h_z$ of [](#eq-latent-lift), an `AbstractObsOperator` |
| 2 | `LatentETKF`, `LatentLETKF` | encode, cycle with the ETKF, and decode the histories |

A codec is structural: any object with `.encode` and `.decode` works, which is also the shape of `pipekit_cycle.LatentMap`.
Note that `latent_ensemble` and `decode_ensemble` take a callable, so pass `codec.encode`, whereas `LiftedObs` takes the object and calls its `.decode`.
A trained autoencoder and a learned surrogate replace the toys below with nothing else changing.

The running example is a pattern advected around a periodic domain of $D_x = 100$ sites, moving one site per window.
The codec is $D_z = 6$ Fourier modes of wavenumbers 1 to 3.
In a mode of wavenumber $k$, a shift by $\delta$ rotates the sine and cosine coefficients $(a_k, b_k)$ by the angle $k \delta$:

$$a_k \leftarrow a_k \cos k \delta + b_k \sin k \delta, \qquad b_k \leftarrow b_k \cos k \delta - a_k \sin k \delta.$$ (eq-latent-rotation)

Every fifth site is observed, so $D_y = 20$, with noise variance 0.25.

```{code-block} python
:caption: The advection toy, its Fourier codec, its state-space and latent forward models, and the identity-codec check.
:label: code-latent-setup
import einx
import equinox as eqx
import filterax as flx
import jax
import jax.numpy as jnp
import jax.random as jr
import jaxtyping as jt
import lineax as lx
import numpy as np
import optax
from filterax.differentiable import road_enkf_grad_step

jax.config.update("jax_enable_x64", True)

# Shapes: D_x = 100 sites; D_z = 6 modes; D_y = 20 observed sites; N_E = 10 members

# sᵢ, the site coordinates
grid = 2.0 * jnp.pi * jnp.arange(100.0) / 100.0  # → (D_x,)
# k = 1, 2, 3
wavenumbers = jnp.arange(1.0, 4.0)  # → (3,)
# k s
phases = einx.dot("site, wave -> site wave", grid, wavenumbers)  # → (D_x, 3)
# U = √(2 / D_x) [sin(k s), cos(k s)]
basis = jnp.sqrt(2.0 / 100.0) * einx.id(  # → (D_x, D_z)
    "site a, site b -> site (a + b)", jnp.sin(phases), jnp.cos(phases)
)


# user
class LinearCodec(eqx.Module):
    basis: jt.Float[jt.Array, "site mode"]

    # g_enc(x) = Uᵀ x
    def encode(self, state):
        return einx.dot("site mode, site -> mode", self.basis, state)

    # g_dec(z) = U z
    def decode(self, latent):
        return einx.dot("site mode, mode -> site", self.basis, latent)


# user
class Advection(flx.AbstractDynamics):
    # x(s, t) = x(s − δ, t − 1), one site per window
    def __call__(self, state, t0, t1):
        return jnp.roll(state, 1)


# user
class RotatingModes(eqx.Module):
    speed: jt.Float[jt.Array, ""]

    # the rotation in Fourier coefficients for a shift of δ = speed × Δt
    def step(self, latent, dt):
        angles = wavenumbers * self.speed * dt  # → (3,)
        sines, cosines = latent[:3], latent[3:]  # → (3,), (3,)
        # aₖ ← aₖ cos kδ + bₖ sin kδ
        new_sines = sines * jnp.cos(angles) + cosines * jnp.sin(angles)
        # bₖ ← bₖ cos kδ − aₖ sin kδ
        new_cosines = cosines * jnp.cos(angles) - sines * jnp.sin(angles)
        return jnp.concatenate((new_sines, new_cosines))  # → (D_z,)


# user
# h(x) selects every fifth site
def observe(state: jt.Float[jt.Array, " site"]) -> jt.Float[jt.Array, " obs"]:
    return state[::5]


codec = LinearCodec(basis)
# δ = 2π / D_x radians per window
true_speed = jnp.asarray(2.0 * jnp.pi / 100.0)
latent_model = RotatingModes(true_speed)
# today (lineax)
# Σ_ε = 0.25 I
noise = lx.DiagonalLinearOperator(0.25 * jnp.ones(20))  # (D_y, D_y) operator

# z_true,0, with coefficients of standard deviation 3.5355
true_start = 3.5355 * jr.normal(jr.key(0), (6,))  # → (D_z,)


# user
# x_true,t = F x_true,t−1, stacked, from a given first state
def run_truth(first, count):
    # user
    def advance(state, _):
        next_state = jnp.roll(state, 1)
        return next_state, next_state

    _, states = jax.lax.scan(advance, first, length=count)
    return states  # → (T, D_x)


# user
# yₜ = h(x_true,t) + εₜ, packed as Layer 2 windows
def make_windows(states):
    count = states.shape[0]
    # εₜ ~ 𝒩(0, 0.25 I)
    errors = 0.5 * jr.normal(jr.key(1), (count, 20))  # → (T, D_y)
    values = states[:, ::5] + errors  # → (T, D_y)
    return [(values[t], float(t + 1)) for t in range(count)]


truth = run_truth(codec.decode(true_start), 30)  # → (T, D_x)
windows = make_windows(truth)  # T of ((D_y,), scalar)
# z⁽ᵉ⁾ ~ 𝒩(𝟎, 12.5 I), the same scale as the truth
start_latent = 3.5355 * jr.normal(jr.key(2), (10, 6))  # → (N_E, D_z)
# x⁽ᵉ⁾ = U z⁽ᵉ⁾, in the range of U
start_members = jax.vmap(codec.decode)(start_latent)  # → (N_E, D_x)

# today (filterax)
identity = flx.identity_latent_map(dim=100)
identity_run = flx.LatentETKF(
    latent_map=identity, dynamics=Advection(), obs_op=observe
).assimilate(start_members, windows[:5], noise)
# today (filterax)
plain_run = flx.ETKF(dynamics=Advection(), obs_op=observe).assimilate(
    start_members, windows[:5], noise
)
print(f"{jnp.abs(identity_run.particles - plain_run.particles).max():.0e}")
print(f"{jnp.abs(identity_run.particles_z - identity_run.particles).max():.0e}")
print(f"{jnp.sqrt(jnp.mean(truth**2)):.3f}")
```

It prints `0e+00`, `0e+00` and `0.719`.
With the identity codec, `LatentETKF` matches `ETKF` to the last bit, so the wrapper adds wiring and no arithmetic.
The last number is the root mean square of the true field.

## Algorithm

[](#code-latent-algorithm) writes out one latent cycle with the Layer 0 and Layer 1 pieces and a Layer 1 filter.
`LatentETKF` does the same thing inside `assimilate`.

```{code-block} python
:caption: The latent cycle by hand, against `LatentETKF`.
:label: code-latent-algorithm
# Shapes: T = 5 windows; D_x = 100; D_z = 6; D_y = 20; N_E = 10 members

# today (filterax)
# h_z = h ∘ g_dec
lifted = flx.LiftedObs(decoder=codec, inner=observe)
# today (filterax)
latent_dynamics = flx.LatentDynamics(inner=latent_model)
# today (filterax)
# z⁽ᵉ⁾ = g_enc(x⁽ᵉ⁾)
latent = flx.latent_ensemble(codec.encode, start_members)  # → (N_E, D_z)
previous_time = 0.0
for target, now in windows[:5]:
    # user
    # z_t|t−1 = f_z(z_t−1|t−1)
    def forecast_member(member):
        return latent_dynamics(member, jnp.asarray(previous_time), jnp.asarray(now))

    forecast = jax.vmap(forecast_member)(latent)  # → (N_E, D_z)
    # today (filterax)
    # Z_t|t, the ETKF of chapter 6 with h_z
    latent = flx.filters.ETKF().analysis(forecast, target, lifted, noise).particles
    previous_time = now
# today (filterax)
# X_T|T = g_dec(Z_T|T)
decoded = flx.decode_ensemble(codec.decode, latent)  # → (N_E, D_x)

# today (filterax)
latent_run = flx.LatentETKF(
    latent_map=codec, dynamics=latent_model, obs_op=observe
).assimilate(start_members, windows[:5], noise)
print(f"{jnp.abs(latent - latent_run.particles_z).max():.0e}")
print(f"{jnp.abs(decoded - latent_run.particles).max():.0e}")
```

It prints `0e+00` and `0e+00`.

The result is a `LatentAssimilationResult`.
The fields ending in `_z` are the primary representation, the arrays the filter updated.
The other fields are decoded views, so that smoothers, diagnostics and plots work unchanged.
The log-likelihoods are computed in observation space, so one array serves both views.

| Field | Shape | Meaning |
| --- | --- | --- |
| `particles_z` | $N_\mathcal{E} \times D_z$ | the terminal latent ensemble |
| `forecast_history_z`, `analysis_history_z` | $T \times N_\mathcal{E} \times D_z$ | the latent histories |
| `particles` | $N_\mathcal{E} \times D_x$ | the decoded terminal ensemble |
| `forecast_history`, `analysis_history` | $T \times N_\mathcal{E} \times D_x$ | the decoded histories |
| `log_likelihoods` | $T$ | $\log \mathcal{N}(\bar{\vect{r}}_t; \vect{0}, \hat{\gmat{\Sigma}}_{y,t})$ |

Every history is decoded and stacked as it is returned, so $T N_\mathcal{E} D_x$ numbers land in memory whatever the latent size.
For a very large field, compose the Layer 1 pieces and decode on demand, as in [](#code-latent-gradients).

With `in_x_space=False`, `assimilate` takes a ready-made $N_\mathcal{E} \times D_z$ ensemble and skips the first encode.
An `inflator` runs in latent space because the loop only ever sees $\mat{Z}$, so an additive inflator's covariance must be $D_z \times D_z$.

`LatentLETKF` has the same body as `LatentETKF`, with no radius and no localization.
Latent dimensions of a typical autoencoder are global modes with no notion of physical distance, so the class exists for wiring parity only.

## Agreement with the state-space ETKF

[](#eq-latent-reduction) says a linear orthonormal codec reproduces the state-space ETKF when the members lie in its range.
Advection maps Fourier modes to themselves, so the same holds through the forecast, and the three forecast routes must agree.

```{code-block} python
:caption: The reduction [](#eq-latent-reduction) over 10 windows, with the surrogate and encoded forecasts.
:label: code-latent-reduction
# Shapes: T = 10 windows; D_x = 100; D_z = 6; D_y = 20; N_E = 10 members

# today (filterax)
state_space = flx.ETKF(dynamics=Advection(), obs_op=observe).assimilate(
    start_members, windows[:10], noise
)
# today (filterax)
surrogate = flx.LatentETKF(
    latent_map=codec, dynamics=latent_model, obs_op=observe
).assimilate(start_members, windows[:10], noise)
# today (filterax)
# z_t = g_enc(f(g_dec(z_t−1)))
encoded = flx.LatentETKF(
    latent_map=codec,
    dynamics=flx.EncodedDynamics(latent_map=codec, inner=Advection()),
    obs_op=observe,
).assimilate(start_members, windows[:10], noise)
print(f"{jnp.abs(surrogate.analysis_history - state_space.analysis_history).max():.0e}")
print(f"{jnp.abs(surrogate.log_likelihoods - state_space.log_likelihoods).max():.0e}")
print(f"{jnp.abs(encoded.analysis_history_z - surrogate.analysis_history_z).max():.0e}")
```

It prints `5e-15`, `1e-14` and `1e-14`, all round-off.
So in this case the two state sizes give one answer, and the latent run differs only in what it costs.

## What the codec buys

The reduction holds only when the members are already in the range.
[](#code-latent-benefit) adds white noise of standard deviation 1 to every site of the initial members, which is larger than the true field's 0.719.
The state-space filter must spend its $N_\mathcal{E} - 1$ directions on it, while the first encode discards it.

```{code-block} python
:caption: The state-space and latent filters from an initial ensemble with noise outside the codec's range, over 30 windows.
:label: code-latent-benefit
# Shapes: T = 30 windows; D_x = 100; D_z = 6; D_y = 20; N_E = 10 members

# x⁽ᵉ⁾ = U z⁽ᵉ⁾ + noise
noisy_members = start_members + jr.normal(jr.key(3), (10, 100))  # → (N_E, D_x)


# user
# RMSE of the ensemble mean over the last 15 windows
def settled_rmse(history, reference):
    mean = einx.mean("t [member] site", history)  # → (T, D_x)
    return jnp.sqrt(jnp.mean((mean[15:] - reference[15:]) ** 2))  # → ()


# today (filterax)
noisy_state = flx.ETKF(dynamics=Advection(), obs_op=observe).assimilate(
    noisy_members, windows, noise
)
# today (filterax)
noisy_latent = flx.LatentETKF(
    latent_map=codec, dynamics=latent_model, obs_op=observe
).assimilate(noisy_members, windows, noise)
# rank X̃, and rank Z̃
state_rank = jnp.linalg.matrix_rank(flx.ensemble_anomalies(noisy_members))
latent_rank = jnp.linalg.matrix_rank(flx.ensemble_anomalies(noisy_latent.particles_z))
print(state_rank, latent_rank)
print(f"{settled_rmse(noisy_state.analysis_history, truth):.3f}")
print(f"{settled_rmse(noisy_latent.analysis_history, truth):.3f}")
```

It prints `9 6`, `0.652` and `0.084`.
The state-space anomalies have the full rank of $N_\mathcal{E} - 1 = 9$ in a space of 100 dimensions, so nothing in the sample covariance tells signal from noise.
The latent anomalies span all 6 of their dimensions, which makes this a well-sampled problem, and the error falls by a factor of nearly eight.

## Nonlinear decoders and Gaussianity

A common argument is that an encoder Gaussianises the state, so the Kalman update fits better in $\vect{z}$ than in $\vect{x}$.
This is half of the story, because a nonlinear decoder also makes $h_z$ nonlinear, and the update only linearises it over the ensemble.

[](#code-latent-nonlinear) tests this on a case with an exact answer.
A scalar state has a lognormal prior, $x = e^{z}$ with $z \sim \mathcal{N}(0, 0.64)$, and is observed directly with noise variance 0.25.
The prior is Gaussian in $z$, so the log codec is the ideal encoder, and a grid integration gives the true posterior mean.

```{code-block} python
:caption: The posterior mean of a lognormal state, from the state-space ETKF, the latent ETKF with a log codec, and a grid reference.
:label: code-latent-nonlinear
# Shapes: N_E = 2,000 members; D_x = D_z = D_y = 1


# user
class LogCodec(eqx.Module):
    # g_enc(x) = log x
    def encode(self, state):
        return jnp.log(state)

    # g_dec(z) = exp z
    def decode(self, latent):
        return jnp.exp(latent)


# user
class Persist(flx.AbstractDynamics):
    # f(x) = x
    def __call__(self, state, t0, t1):
        return state


# user
# E[x | y] by quadrature on a uniform grid in z
def grid_mean(value: float) -> float:
    z = np.linspace(-6.0, 6.0, 200001)
    weights = np.exp(-0.5 * z**2 / 0.64 - 0.5 * (value - np.exp(z)) ** 2 / 0.25)
    return float((weights * np.exp(z)).sum() / weights.sum())


# z⁽ᵉ⁾ ~ 𝒩(0, 0.64)
prior_latent = 0.8 * jr.normal(jr.key(7), (2000, 1))  # → (N_E, D_z)
# x⁽ᵉ⁾ = exp z⁽ᵉ⁾
prior_members = jnp.exp(prior_latent)  # → (N_E, D_x)
scalar_noise = lx.DiagonalLinearOperator(jnp.array([0.25]))  # (D_y, D_y) operator


# user
# h(x) = x
def observe_all(state):
    return state


for value in (0.1, 1.0, 3.0, 6.0):
    scalar_window = [(jnp.array([value]), 1.0)]
    # today (filterax)
    linear_run = flx.ETKF(dynamics=Persist(), obs_op=observe_all).assimilate(
        prior_members, scalar_window, scalar_noise
    )
    # today (filterax)
    log_run = flx.LatentETKF(
        latent_map=LogCodec(), dynamics=Persist(), obs_op=observe_all
    ).assimilate(prior_members, scalar_window, scalar_noise)
    # x̄ in state space, x̄ decoded, and g_dec(z̄)
    linear_mean = jnp.mean(linear_run.particles)  # → ()
    log_mean = jnp.mean(log_run.particles)  # → ()
    decoded_mean = jnp.exp(jnp.mean(log_run.particles_z))  # → ()
    print(value, f"{grid_mean(value):.3f} {linear_mean:.3f} {log_mean:.3f}", end=" ")
    print(f"{decoded_mean:.3f} {linear_run.particles.min():.3f}")
```

It prints:

| $y$ | Grid reference | State-space ETKF | Latent ETKF, $\bar{x}$ | Latent ETKF, $g_\text{dec}(\bar{z})$ | Smallest state-space member |
| --- | --- | --- | --- | --- | --- |
| 0.1 | `0.531` | `0.260` | `0.617` | `0.558` | `-0.210` |
| 1.0 | `0.915` | `1.051` | `0.931` | `0.841` | `0.581` |
| 3.0 | `2.765` | `2.810` | `2.318` | `2.096` | `2.339` |
| 6.0 | `5.839` | `5.447` | `9.112` | `8.239` | `4.977` |

Neither space wins everywhere.
Where the observation lies in the bulk of the prior, the log codec is the more accurate and it keeps every member positive.
The state-space update gives a negative member at $y = 0.1$.
Where the observation lies in the prior's tail, the state-space update is better, because its $h$ is linear and the likelihood dominates.
The latent update overshoots there, reaching `9.112` against a true `5.839`.
The cause is that $h_z = e^{z}$ is strongly curved, and the update fits one straight line through the prior ensemble.

The table also shows that the two views can disagree.
With a nonlinear decoder the mean of the decoded members is not the decoded mean, so state which of the two is being reported.

## Gradients through the codec

The whole construction is differentiable, so the codec and the surrogate can be trained through the filter ([chapter 14](14_differentiable.md)) {cite:p}`chen2023reduced`.
`LatentETKF.assimilate` is a Python loop, which `jax.grad` handles for short windows.
For long windows, compose `LatentDynamics` and `LiftedObs` with `differentiable_assimilate`, which runs the same arithmetic in a scan and returns the latent histories only.
Its `obs_op` closes over the decoder, so decoder weights receive a gradient.
`road_enkf_grad_step` differentiates only its `dynamics` argument, so it can learn the surrogate but not the decoder.

```{code-block} python
:caption: The scan-based latent cycle, a decoder gradient against a central difference, and learning the surrogate's speed with local gradients.
:label: code-latent-gradients
# Shapes: T = 30 windows; D_x = 100; D_z = 6; D_y = 20; N_E = 10 members

window_values = jnp.stack([value for value, _ in windows])  # → (T, D_y)
window_times = jnp.arange(1.0, 31.0)  # → (T,)


# user
# 𝓛(ν) for decoder and encoder weights ν, over the first T windows
def codec_loss(trainable):
    # today (filterax)
    result = flx.differentiable_assimilate(
        flx.filters.ETKF(),
        flx.LatentDynamics(inner=latent_model),
        flx.LiftedObs(decoder=trainable, inner=observe),
        start_latent,
        window_values,
        window_times,
        noise,
    )
    return -jnp.sum(result.log_likelihoods)  # → ()


scan_run = flx.differentiable_assimilate(
    flx.filters.ETKF(),
    flx.LatentDynamics(inner=latent_model),
    flx.LiftedObs(decoder=codec, inner=observe),
    start_latent,
    window_values[:5],
    window_times[:5],
    noise,
)
print(f"{jnp.abs(scan_run.particles - latent_run.particles_z).max():.0e}")

# today (equinox)
codec_gradient = eqx.filter_jit(eqx.filter_grad(codec_loss))(codec)
# a random direction in the decoder weights
direction = jr.normal(jr.key(9), (100, 6))  # → (D_x, D_z)
projected = einx.sum("[site mode]", codec_gradient.basis * direction)  # → ()
step = 1e-6
central = (
    codec_loss(LinearCodec(basis + step * direction))
    - codec_loss(LinearCodec(basis - step * direction))
) / (2.0 * step)
print(f"{projected:.4f} {central:.4f}")

# today (filterax)
trainable = flx.LatentDynamics(inner=RotatingModes(jnp.asarray(0.05)))
optimiser = optax.adam(0.001)
state = optimiser.init(eqx.filter(trainable, eqx.is_inexact_array))
# today (filterax)
descent = eqx.filter_jit(road_enkf_grad_step)
for _ in range(40):
    trainable, state, value = descent(
        trainable,
        optimiser,
        state,
        start_latent,
        window_values,
        window_times,
        noise,
        flx.LiftedObs(decoder=codec, inner=observe),
    )
print(f"{trainable.inner.speed:.4f} {true_speed:.4f} {value:.1f}")
```

It prints `4e-15`, then `-527.4808 -527.4808`, then `0.0625 0.0628 466.7`.
The scan matches `LatentETKF` and the decoder gradient matches its central difference.
Forty local steps from a speed of 0.05 land within 1% of the true speed.

## When it breaks

Whatever the codec cannot represent, the filter cannot correct.
[](#code-latent-floor) adds a pattern of wavenumber 10 to the truth, which the codec has no mode for, and compares the error with the floor [](#eq-latent-floor).

```{code-block} python
:caption: The error floor from truth outside the codec's range.
:label: code-latent-floor
# Shapes: T = 30 windows; D_x = 100; D_z = 6; D_y = 20; N_E = 10 members

# a pattern of wavenumber 10, outside the range of U
unresolved = jnp.sin(10.0 * grid)  # → (D_x,)
for amplitude in (0.0, 0.5):
    # x_true,0 = U z_true,0 + amplitude × pattern
    first = codec.decode(true_start) + amplitude * unresolved  # → (D_x,)
    states = run_truth(first, 30)  # → (T, D_x)
    # today (filterax)
    run = flx.LatentETKF(
        latent_map=codec, dynamics=latent_model, obs_op=observe
    ).assimilate(start_members, make_windows(states), noise)
    # (I − U Uᵀ) x_true,t
    outside = states - jax.vmap(codec.decode)(jax.vmap(codec.encode)(states))
    floor = jnp.sqrt(jnp.mean(outside[15:] ** 2))  # → ()
    print(amplitude, f"{settled_rmse(run.analysis_history, states):.3f} {floor:.3f}")
```

It prints `0.0 0.084 0.000` and `0.5 0.363 0.354`.
With the pattern present, almost all of the error is the floor, and no number of members or observations can lower it.

Three traps of the wiring are also worth knowing.
[](#code-latent-traps) measures them.

- `LatentETKF` wraps anything that is not an `AbstractDynamics` in `LatentDynamics`, so a plain function `f(z, t0, t1)` fails, although `flx.ETKF` accepts one.
- `EncodedDynamics` preserves the forecast only if $g_\text{enc} \circ g_\text{dec}$ is the identity on the ensemble, so a mismatched codec injects spurious dynamics in $\vect{z}$ even when $f$ does nothing.
- `LatentLETKF` takes no `radius`, so it cannot be swapped for `LETKF` by name.

```{code-block} python
:caption: The plain-function trap, spurious dynamics from a mismatched codec, and the `LatentLETKF` body.
:label: code-latent-traps
# Shapes: D_x = 100; D_z = 6; D_y = 20; N_E = 10 members


# user
# f_z(z) = z, with the Layer 2 dynamics signature
def plain_dynamics(latent, t0, t1):
    return latent


try:
    # today (filterax)
    trap = flx.LatentETKF(latent_map=codec, dynamics=plain_dynamics, obs_op=observe)
    trap.assimilate(start_members, windows[:1], noise)
except AttributeError as error:
    print(error)


# user
class MismatchedCodec(eqx.Module):
    weights_in: jt.Float[jt.Array, "mode site"]
    weights_out: jt.Float[jt.Array, "site mode"]

    # g_enc(x) = A x
    def encode(self, state):
        return einx.dot("mode site, site -> mode", self.weights_in, state)

    # g_dec(z) = B z
    def decode(self, latent):
        return einx.dot("site mode, mode -> site", self.weights_out, latent)


key_in, key_out = jr.split(jr.key(5))
# independent weights, scaled by √D_x and √D_z so that A B is of order one
mismatched = MismatchedCodec(
    jr.normal(key_in, (6, 100)) / 10.0, jr.normal(key_out, (100, 6)) / 2.45
)
# today (filterax)
ghost = flx.EncodedDynamics(latent_map=mismatched, inner=Persist())
one_step = ghost(jnp.ones(6), 0.0, 1.0)  # → (D_z,)
print(f"{jnp.linalg.norm(one_step - jnp.ones(6)):.3f}")

# today (filterax)
letkf_run = flx.LatentLETKF(
    latent_map=codec, dynamics=latent_model, obs_op=observe
).assimilate(start_members, windows[:5], noise)
difference = letkf_run.analysis_history_z - latent_run.analysis_history_z
print(f"{jnp.abs(difference).max():.0e}")
```

It prints `'function' object has no attribute 'step'`, then `3.039`, then `0e+00`.
The first is the error from the wrapped plain function.
The second is how far one step of "do nothing" moves $\vect{z} = \vect{1}$, whose norm is 2.449, so the codec itself is doing the forecasting.
The third confirms that `LatentLETKF` and `LatentETKF` are the same computation.

## When latent assimilation is the right answer

Use `LatentETKF` when the field has a low intrinsic dimension and a small ensemble cannot sample the full state, as in [](#code-latent-benefit).
It pays most when a learned surrogate replaces an expensive forward model, because that is where the large saving lies.
It is nearly free when a codec already exists for compression or emulation, and then the codec can be fine-tuned through the filter.
A transformation such as the log codec also keeps a constrained variable physical, if observations stay in the bulk of the prior.

Stay in state space when the truth has structure the codec cannot represent, since the floor [](#eq-latent-floor) is permanent.
Stay there too when localization matters, because `LatentLETKF` does not localize, and when a curved decoder meets a highly informative observation.
With a linear codec, members in its range and an exact model, the two give one answer, so choose by cost.

## See also

- [Chapter 6](06_etkf.md): the analysis running underneath, unchanged in $\vect{z}$
- [Chapter 8](08_letkf.md): why R-localization needs physical coordinates, and so why `LatentLETKF` omits it
- [Chapter 10](10_smoothers.md): the iterated methods for strongly nonlinear $h$
- [Chapter 14](14_differentiable.md): training a codec and a surrogate through the filter
- [Chapter 16](16_assimilation_cycle.md): the loop these wrappers reuse in $\vect{z}$, and `pipekit` wiring
- [API: advanced filters](https://jejjohnson.github.io/filterax/reference/filters_advanced/)

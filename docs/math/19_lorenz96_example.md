---
numbering:
  enumerator: "19.%s"
---

# Worked example: a Lorenz-96 twin experiment

A twin experiment is how an ensemble filter is judged honestly: simulate a truth, observe it with noise, assimilate, and compare the analysis with the truth you hid.
This chapter runs one end to end on the Lorenz-96 model {cite:p}`lorenz1998optimal`, which is chaotic and small enough to run in seconds.

Three things organise it.
Small ensembles need treating, and the treatments are inflation, localization, or both.
The honest report is RMSE against spread over several independent twins, including the settings that fail, because a filter that works on one seed may not work on the next.
And a smoother pass on top is not automatically an improvement.

## The model and the twin

The state is $\vect{x}_t \in \mathbb{R}^{D_x}$, with $D_x = 40$ sites on a circle.
filterax has no dynamical core, so the model is written out here.
With indices taken cyclically and forcing 8, its tendency at site $k$ is

$$\frac{\mathrm{d} x_k}{\mathrm{d} t} = (x_{k+1} - x_{k-2}) x_{k-1} - x_k + 8,$$ (eq-l96-tendency)

and a window advances the state by one fourth-order Runge-Kutta step of $\Delta t = 0.05$.
The twin observes every second site, $D_y = 20$, with unit noise:

$$\vect{y}_t = \mat{H} \vect{x}_t + \gvect{\varepsilon}_t, \qquad \gvect{\varepsilon}_t \sim \mathcal{N}(\vect{0}, \mat{I}).$$ (eq-l96-observation)

The first ensemble is the true initial state plus Gaussian noise of variance 4, with $N_\mathcal{E}$ members as rows of $\mat{X} \in \mathbb{R}^{N_\mathcal{E} \times D_x}$.
Every run has $T = 300$ windows, and the scores use the last 200, once the filter has settled.
Two scales judge the result: the observation noise of 1, and the standard deviation of the state itself.

```{code-block} python
:caption: Lorenz-96 by RK4, a true trajectory on the attractor, and noisy observations of every second site.
:label: code-l96-twin
import einx
import filterax as flx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float, Key

jax.config.update("jax_enable_x64", True)

# Shapes: T = 300 windows; D_x = 40 sites; D_y = 20 observed sites; N_E members

num_windows = 300
step_size = 0.05


# user
# dxₖ/dt = (xₖ₊₁ − xₖ₋₂) xₖ₋₁ − xₖ + 8, cyclic in k
def tendency(state: Float[Array, " state"]) -> Float[Array, " state"]:
    return (jnp.roll(state, -1) - jnp.roll(state, 2)) * jnp.roll(state, 1) - state + 8.0


# user
# one RK4 step of Δt = 0.05, with the (x, t₀, t₁) signature filterax calls
def rk4(state: Float[Array, " state"], _=None, __=None) -> Float[Array, " state"]:
    k1 = tendency(state)
    k2 = tendency(state + 0.5 * step_size * k1)
    k3 = tendency(state + 0.5 * step_size * k2)
    k4 = tendency(state + step_size * k3)
    return state + step_size / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# user
# xₜ₊₁ = f(xₜ), stacked
def advance(state: Float[Array, " state"], _) -> tuple[Array, Array]:
    next_state = rk4(state)
    return next_state, next_state


# x₀ on the attractor, after 1,000 steps from a perturbed rest state
spun_up = jax.lax.fori_loop(  # → (D_x,)
    0, 1000, lambda _, state: rk4(state), 8.0 + jr.normal(jr.key(0), (40,))
)
_, truth = jax.lax.scan(advance, spun_up, length=num_windows)  # → (T, D_x)
# window times
times = step_size * (jnp.arange(num_windows) + 1.0)  # → (T,)
# H selects every second site
sites = jnp.arange(0, 40, 2)  # → (D_y,)
# Σ_ε = I
unit_noise = lx.DiagonalLinearOperator(jnp.ones(20))  # (D_y, D_y) operator


# user
# h(x) = H x
def observe(state: Float[Array, " state"]) -> Float[Array, " obs"]:
    return state[sites]


# user
# yₜ = H xₜ + εₜ, and a first ensemble x⁽ᵉ⁾ ~ 𝒩(x₀, 4 I), both from one key
def twin_inputs(key: Key[Array, ""], num_members: int) -> tuple[Array, Array]:
    obs_key, start_key = jr.split(key)
    # yₜ
    records = truth[:, sites] + jr.normal(obs_key, (num_windows, 20))  # → (T, D_y)
    # x⁽ᵉ⁾
    start = spun_up + 2.0 * jr.normal(start_key, (num_members, 40))  # → (N_E, D_x)
    return records, start


# the variability of the state over the run
print(f"{jnp.std(truth):.2f}")
```

It prints `3.67`, the standard deviation of the truth.
An RMSE near that value means the filter has lost the trajectory and knows nothing beyond the attractor.
An RMSE below 1 means the filter beats the observations at the observed sites, and it must also fill in the unobserved half.

## One run, end to end

The Layer 2 class [`filterax.ETKF`](https://jejjohnson.github.io/filterax/reference/filters/) composes a dynamics function, the analysis and an optional inflator, and loops over windows in Python.
Its form of the RTPS inflator {cite:p}`whitaker2012evaluating` relaxes each site's spread back toward its forecast value,

$$\sigma^\text{target}_k = (1 - \alpha) \sigma^{t \mid t}_k + \alpha \sigma^{t \mid t-1}_k,$$ (eq-l96-rtps)

for $\alpha \in [0, 1]$, by rescaling the anomalies at site $k$ and leaving the mean alone.

A Python loop is fine for one run and costly for a sweep, since every window dispatches its own operations.
[`differentiable_assimilate`](https://jejjohnson.github.io/filterax/reference/differentiable/) runs the same cycle as one `jax.lax.scan`, so it can be compiled once and vectorised over twins.

```{code-block} python
:caption: The Layer 2 `ETKF` with `RTPS` as a Python loop, and the same cycle as a `jax.lax.scan`.
:label: code-l96-layer2
# Shapes: T = 30 windows in this block; D_x = 40; D_y = 20; N_E = 20

short = 30
records, start = twin_inputs(jr.key(100), 20)  # → (T, D_y), (N_E, D_x)
windows = [(records[t], times[t]) for t in range(short)]
# today (filterax)
# forecast, ETKF analysis, RTPS with α = 0.3, in a Python loop over windows
model = flx.ETKF(dynamics=rk4, obs_op=observe, inflator=flx.RTPS(alpha=0.3))
looped = model.assimilate(start, windows, unit_noise)
# today (filterax)
# the same cycle in one jax.lax.scan
scanned = flx.differentiable_assimilate(
    flx.filters.ETKF(),
    rk4,
    observe,
    start,
    records[:short],
    times[:short],
    unit_noise,
    inflator=flx.RTPS(alpha=0.3),
)
difference = jnp.abs(looped.analysis_history - scanned.analysis_history).max()
print(f"{difference:.0e}")
```

It prints `3e-14`, so the two loops are the same computation, and the sweep below uses the scan.

## ETKF, RTPS and LETKF against each other

[](#code-l96-comparison) runs eight independent twins for each setting, with the same truth and different observation noise and first ensembles.
It calls a twin healthy if its RMSE stays under the observation noise of 1.
For the healthy twins it reports the median RMSE and spread, and otherwise the worst RMSE and any `NaN`.

The LETKF {cite:p}`hunt2007efficient` needs coordinates, and it measures distance in Euclidean terms.
So the sites sit on a circle of circumference 40, with the observations at their sites, which makes distance respect the cyclic domain.
The radius $\ell$ below is therefore in sites, and the LETKF drops observations beyond it.

```{code-block} python
:caption: RMSE and spread over eight independent twins, for each filter and inflation setting.
:label: code-l96-comparison
# Shapes: T = 300 windows; D_x = 40; D_y = 20; N_E members; 8 independent twins

# the sites on a circle of circumference 40, so that Euclidean distance respects
# the cyclic domain; state and observation coordinates
angles = 2.0 * jnp.pi * jnp.arange(40.0) / 40.0  # → (D_x,)
state_coords = (40.0 / (2.0 * jnp.pi)) * jax.vmap(  # → (D_x, 2)
    lambda angle: jnp.array([jnp.cos(angle), jnp.sin(angle)])
)(angles)
obs_coords = state_coords[sites]  # → (D_y, 2)
twin_keys = jr.split(jr.key(100), 8)  # → (8,)
settled = 100


# user
# one twin experiment, as one scan
def run_twin(key, num_members, analysis_filter, inflator, **coordinates):
    records, start = twin_inputs(key, num_members)
    # today (filterax)
    return flx.differentiable_assimilate(
        analysis_filter,
        rk4,
        observe,
        start,
        records,
        times,
        unit_noise,
        inflator=inflator,
        **coordinates,
    )


# user
# RMSE of x̄ₜ and RMS spread, over windows `first` to `last`
def score(history, first=settled, last=num_windows):
    window = history[first:last]  # → (T′, N_E, D_x)
    # x̄ₜ
    means = einx.mean("t [member] state", window)  # → (T′, D_x)
    # today (filterax)
    errors = jax.vmap(flx.utils.rmse_vs_truth)(means, truth[first:last])  # → (T′,)
    # today (filterax)
    spreads = jax.vmap(flx.utils.rms_spread)(window)  # → (T′,)
    rmse = jnp.sqrt(jnp.mean(errors**2))  # → ()
    return rmse, jnp.sqrt(jnp.mean(spreads**2))


# user
# run the twins, then print how many stayed under the unit observation noise,
# the median RMSE and spread of those, and the worst RMSE of the rest
def report(label, num_members, analysis_filter, inflator=None, **coordinates):
    results = jax.jit(
        jax.vmap(
            lambda key: run_twin(
                key, num_members, analysis_filter, inflator, **coordinates
            )
        )
    )(twin_keys)
    histories = results.analysis_history  # → (8, T, N_E, D_x)
    finite = einx.all("twin t [member state]", jnp.isfinite(histories))  # → (8, T)
    survived = einx.all("twin [t]", finite)  # → (8,)
    rmse, spread = jax.vmap(score)(histories)  # → (8,), (8,)
    healthy = survived & (jnp.nan_to_num(rmse, nan=jnp.inf) < 1.0)  # → (8,)
    text = f"{label}: {int(healthy.sum())} of 8 healthy"
    if healthy.any():
        text += f", RMSE {jnp.median(rmse[healthy]):.2f}"
        text += f", spread {jnp.median(spread[healthy]):.2f}"
    if (survived & ~healthy).any():
        text += f", worst RMSE {jnp.max(rmse[survived]):.2f}"
    if (~survived).any():
        first_bad = jax.vmap(jnp.argmin)(finite)[~survived]
        text += f", {int((~survived).sum())} NaN at {sorted(first_bad.tolist())}"
    print(text)
    return results


letkf_coords = {"state_coords": state_coords, "obs_coords": obs_coords}
# today (filterax)
etkf_filter = flx.filters.ETKF()
report("N_E=20 ETKF", 20, etkf_filter)
for alpha in (0.3, 0.5, 0.7, 0.9):
    rtps = flx.RTPS(alpha=alpha)
    rtps_results = report(f"N_E=20 ETKF RTPS {alpha}", 20, etkf_filter, rtps)
    if alpha == 0.3:
        inflated = rtps_results
letkf_results = report(
    "N_E=20 LETKF 8", 20, flx.filters.LETKF(radius=8.0), **letkf_coords
)
calibrated_results = report(
    "N_E=20 LETKF 8 RTPS 0.2",
    20,
    flx.filters.LETKF(radius=8.0),
    flx.RTPS(alpha=0.2),
    **letkf_coords,
)
report(
    "N_E=20 LETKF 8 RTPS 0.9",
    20,
    flx.filters.LETKF(radius=8.0),
    flx.RTPS(alpha=0.9),
    **letkf_coords,
)
report("N_E=20 LETKF 4", 20, flx.filters.LETKF(radius=4.0), **letkf_coords)
report("N_E=10 ETKF RTPS 0.3", 10, etkf_filter, flx.RTPS(alpha=0.3))
report("N_E=10 LETKF 8", 10, flx.filters.LETKF(radius=8.0), **letkf_coords)
report("N_E=10 LETKF 4", 10, flx.filters.LETKF(radius=4.0), **letkf_coords)
report(
    "N_E=10 LETKF 8 RTPS 0.2",
    10,
    flx.filters.LETKF(radius=8.0),
    flx.RTPS(alpha=0.2),
    **letkf_coords,
)
```

It prints:

| $N_\mathcal{E}$ | Filter | Inflation | Healthy | RMSE | Spread | Otherwise |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | ETKF | none | 2 of 8 | `0.38` | `0.25` | worst RMSE `4.83` |
| 20 | ETKF | RTPS 0.3 | 7 of 8 | `0.33` | `0.36` | worst RMSE `3.21` |
| 20 | ETKF | RTPS 0.5 | 8 of 8 | `0.45` | `0.50` | |
| 20 | ETKF | RTPS 0.7 | 5 of 8 | `0.71` | `0.82` | worst RMSE `2.50` |
| 20 | ETKF | RTPS 0.9 | 0 of 8 | | | all `NaN`, from windows 29 to 64 |
| 20 | LETKF $\ell = 8$ | none | 8 of 8 | `0.32` | `0.28` | |
| 20 | LETKF $\ell = 8$ | RTPS 0.2 | 8 of 8 | `0.32` | `0.35` | |
| 20 | LETKF $\ell = 8$ | RTPS 0.9 | 0 of 8 | | | worst RMSE `1.35` |
| 20 | LETKF $\ell = 4$ | none | 8 of 8 | `0.40` | `0.33` | |
| 10 | ETKF | RTPS 0.3 | 0 of 8 | | | worst RMSE `5.09` |
| 10 | LETKF $\ell = 8$ | none | 2 of 8 | `0.54` | `0.28` | worst RMSE `2.96` |
| 10 | LETKF $\ell = 4$ | none | 7 of 8 | `0.47` | `0.32` | worst RMSE `2.14` |
| 10 | LETKF $\ell = 8$ | RTPS 0.2 | 8 of 8 | `0.33` | `0.33` | |

Four things stand out.

Without inflation or localization, 6 of 8 twins diverge at 20 members.
The two that survive have an RMSE of `0.38` against a spread of `0.25`, so they are overconfident as well.

Inflation alone has a narrow window at this size.
At $\alpha = 0.5$ every twin survives, with RMSE and spread nearly equal.
But $\alpha = 0.3$ loses one twin, $\alpha = 0.7$ loses three, and $\alpha = 0.9$ loses all of them.
Choosing $\alpha$ from a single seed would be unsafe here, because 0.3 gives the lowest RMSE of the three and also fails once.

Localization alone is enough at 20 members, with the lowest RMSE here and every twin healthy.
The spread of `0.28` is a little under the RMSE of `0.32`, and adding RTPS at $\alpha = 0.2$ brings it to `0.35`, at the same RMSE.

At 10 members a wide radius is not enough alone, as the LETKF at $\ell = 8$ without inflation fails 6 of 8.
A tighter radius recovers 7 of 8, and inflation with the wide radius recovers all 8, at an RMSE of `0.33`.
The global ETKF with RTPS at $\alpha = 0.3$ fails on all 8.

## What the failures look like

The `NaN` deserves a closer look, because nothing raises it.
RTPS at $\alpha = 0.9$ restores almost all the forecast spread after every analysis.
On the LETKF at the same $\alpha$ no twin goes non-finite, though none stays healthy either, with a worst RMSE of `1.35`, so the global analysis matters too.
[](#code-l96-failure) follows the first global twin to its first non-finite member.
It also repeats the best LETKF with its sites on a line, where sites 0 and 39 look 39 apart.

```{code-block} python
:caption: The run up to `NaN` for RTPS at $\alpha = 0.9$, and the effect of ignoring the cyclic domain in the LETKF.
:label: code-l96-failure
# Shapes: T = 300 windows; D_x = 40; D_y = 20; N_E = 20

# today (filterax)
blown = run_twin(twin_keys[0], 20, etkf_filter, flx.RTPS(alpha=0.9))
blown_history = blown.analysis_history  # → (T, N_E, D_x)
# the first window with a non-finite member
breaking = int(jnp.argmin(einx.all("t [member state]", jnp.isfinite(blown_history))))
print(f"NaN from window {breaking}")
for back in (breaking, 20, 10, 5, 2, 1):
    window = breaking - back
    members = blown_history[window]  # → (N_E, D_x)
    # today (filterax)
    width = flx.utils.rms_spread(members)
    # today (filterax)
    wrong = flx.utils.rmse_vs_truth(einx.mean("[member] state", members), truth[window])
    print(
        f"{window}: RMSE {wrong:.3g}, spread {width:.3g}, "
        f"largest |x| {jnp.abs(members).max():.3g}"
    )

# the same radius, with sites on a line instead of a circle
line = einx.id("site -> site 1", jnp.arange(40.0))  # → (D_x, 1)
report(
    "N_E=20 LETKF 8 on a line",
    20,
    flx.filters.LETKF(radius=8.0),
    state_coords=line,
    obs_coords=line[sites],
)
```

The first twin prints `NaN from window 29`, then:

| Window | RMSE | Spread | Largest $\vert x \vert$ |
| --- | --- | --- | --- |
| 0 | `0.609` | `1.94` | `13.7` |
| 9 | `1.35` | `2.08` | `12.8` |
| 19 | `2.48` | `3.13` | `22` |
| 24 | `5.66` | `3.67` | `35.9` |
| 27 | `6.2` | `53.7` | `630` |
| 28 | `1.01e+23` | `2.31e+13` | `4.66e+23` |

The error climbs above the observation noise within ten windows, and the spread reaches the variability of the state itself by window 24.
Members then leave the attractor, and at $\vert x \vert$ near 630 the quadratic term of [](#eq-l96-tendency) is far too stiff for an RK4 step of 0.05.
One window takes the state to $10^{23}$ and the next to `NaN`.
Neither `RTPS` nor `assimilate` checks for this, so the check belongs to you: watch the spread against the state's own variability.

On a line the LETKF prints `7 of 8 healthy, RMSE 0.35, spread 0.30, worst RMSE 1.07`.
Ignoring the cyclic domain costs a little RMSE and one twin.

## A smoother pass

A smoother uses later observations to refine earlier states.
`EnKS` takes the forecast and analysis histories and walks back over all the windows, and [`FixedLagSmoother`](https://jejjohnson.github.io/filterax/reference/smoothers/) stops each backward walk after `lag` steps.
With $\tilde{\mat{X}}_{t \mid t}$ and $\tilde{\mat{X}}_{t+1 \mid t}$ the analysis and forecast anomalies, one backward step on the members is

$$\mat{X}_{t \mid T} = \mat{X}_{t \mid t} + (\mat{X}_{t+1 \mid T} - \mat{X}_{t+1 \mid t}) \tilde{\mat{X}}_{t+1 \mid t}^{+} \tilde{\mat{X}}_{t \mid t},$$ (eq-l96-backward)

where $^{+}$ is the pseudo-inverse.
The sample gain here has no localization, whichever filter produced the history.

[](#code-l96-smoother) scores the windows from 100 to 250, so that every window has later data to use.
It reports medians over the eight twins, because one diverged twin dominates a mean.

```{code-block} python
:caption: `EnKS` and `FixedLagSmoother` on the same filter history, scored over the interior windows.
:label: code-l96-smoother
# Shapes: T = 300 windows; D_x = 40; N_E = 20; 8 twins

interior = (settled, 250)


# user
# smooth one twin and score it, over the interior windows
def smoothed_score(smoother, forecast, analysis):
    # today (filterax)
    history = smoother.smooth(forecast, analysis).smoothed_history
    return jnp.stack(score(history, *interior))  # → (2,)


# user
# median (RMSE, spread) over the eight twins, since one diverged twin dominates a mean
def median_over_twins(scores: Float[Array, "twin 2"]) -> Float[Array, " 2"]:
    return jax.vmap(jnp.median, in_axes=1)(scores)  # → (2,)


# user
# median (RMSE, spread) of a smoother
def smoother_score(smoother, results):
    scores = jax.vmap(lambda f, a: smoothed_score(smoother, f, a))(
        results.forecast_history, results.analysis_history
    )
    return median_over_twins(scores)  # → (2,)


# user
# median (RMSE, spread) of the filter, over the same windows
def filter_score(results):
    return median_over_twins(  # → (2,)
        jax.vmap(lambda history: jnp.stack(score(history, *interior)))(
            results.analysis_history
        )
    )


for label, results in [
    ("LETKF 8", letkf_results),
    ("LETKF 8 RTPS 0.2", calibrated_results),
]:
    rmse, spread = filter_score(results)
    print(f"{label} filter: {rmse:.3f} {spread:.3f}")
    rmse, spread = smoother_score(flx.EnKS(), results)
    print(f"{label} EnKS: {rmse:.3f} {spread:.3f}")
    for lag in (2, 10, 40):
        rmse, spread = smoother_score(flx.FixedLagSmoother(lag=lag), results)
        print(f"{label} lag {lag}: {rmse:.3f} {spread:.3f}")
# the global ETKF with RTPS at α = 0.3
rmse, spread = filter_score(inflated)
print(f"ETKF RTPS 0.3 filter: {rmse:.3f} {spread:.3f}")
for smoother in (flx.EnKS(), flx.FixedLagSmoother(lag=10)):
    rmse, spread = smoother_score(smoother, inflated)
    print(f"ETKF RTPS 0.3 {type(smoother).__name__}: {rmse:.3f} {spread:.3f}")
```

It prints:

| Filter | Pass | RMSE | Spread |
| --- | --- | --- | --- |
| LETKF $\ell = 8$ | filter | `0.323` | `0.280` |
| | lag 2 | `0.275` | `0.237` |
| | lag 10 | `0.210` | `0.156` |
| | lag 40 | `0.279` | `0.077` |
| | `EnKS` | `0.364` | `0.033` |
| LETKF $\ell = 8$, RTPS 0.2 | filter | `0.315` | `0.348` |
| | lag 2 | `0.266` | `0.295` |
| | lag 10 | `0.216` | `0.208` |
| | lag 40 | `0.329` | `0.147` |
| | `EnKS` | `0.543` | `0.119` |
| ETKF, RTPS 0.3 | filter | `0.345` | `0.353` |
| | lag 10 | `0.224` | `0.233` |
| | `EnKS` | `0.865` | `0.966` |

A lag of 10 windows helps, taking the RMSE from about `0.32` to about `0.21`, a third lower.
Longer lags give that back, and the full `EnKS` is worse than no smoother on all three filters.
The spread is no guide either: on the LETKF it keeps falling with the lag, to `0.033` against an RMSE of `0.364`.
A likely cause is that sample cross-covariances over many windows of a chaotic model are mostly noise.
The unlocalized gain of [](#eq-l96-backward) still acts on them, though we have not isolated this.

So on this problem use `FixedLagSmoother` at about 10 windows, and read its spread as a lower bound on the error.

## When this setup is the right answer

Start from the LETKF with light RTPS on a spatially extended chaotic model with few members.
It was healthy on every twin here, down to 10 members.
Use the global ETKF only when the ensemble is large against the error directions of the model, which [chapter 6](06_etkf.md) shows at 40 members on this problem.
Tune on several twins, and report the failures with the scores.

## See also

- [Chapter 6](06_etkf.md): the ETKF, and its divergence on this model
- [Chapter 8](08_letkf.md) and [chapter 11](11_localization.md): the LETKF and localization
- [Chapter 10](10_smoothers.md): the ensemble smoothers
- [Chapter 12](12_inflation.md): RTPS and the other inflators
- [Chapter 13](13_diagnostics.md): RMSE, spread and the spread-skill ratio
- [Chapter 16](16_assimilation_cycle.md): the assimilation cycle
- [API: filters](https://jejjohnson.github.io/filterax/reference/filters/), [inflation](https://jejjohnson.github.io/filterax/reference/inflation/), [smoothers](https://jejjohnson.github.io/filterax/reference/smoothers/) and [diagnostics](https://jejjohnson.github.io/filterax/reference/diagnostics/)

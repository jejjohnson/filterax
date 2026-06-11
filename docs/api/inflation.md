# Inflation

Counteract spread collapse from finite-ensemble, model-error, or
localization-side-effect sources. Two flavours of primitive:

* **Drop-in inflators** — `MultiplicativeInflator`, `RTPS`, `RTPP`,
  `AdditiveInflator` all implement
  [`AbstractInflator`][filterax.AbstractInflator] and plug straight
  into the Layer-2 assimilation loops (`filterax.ETKF`, `LETKF`, …).
* **Helper primitives** — `inflate_adaptive` returns a posterior on
  the multiplicative factor ``λ`` (the caller composes it with a
  fresh `MultiplicativeInflator(factor=μ_post)` on the next cycle);
  `ledoit_wolf_shrinkage` returns a regularised covariance matrix, not
  an inflated ensemble. Neither is an `AbstractInflator`, so they are
  driven from custom loops rather than dropped into L2 models.

The classic trio of functional primitives —
[`inflate_multiplicative`][filterax.inflate_multiplicative],
[`inflate_rtps`][filterax.inflate_rtps],
[`inflate_rtpp`][filterax.inflate_rtpp] — delegates to the corresponding
[gaussx](https://jejjohnson.github.io/gaussx/) primitives
(`gaussx.inflate_multiplicative`, `gaussx.inflate_rtps`,
`gaussx.inflate_rtpp`); filterax adds the EnKF-flavoured conventions and the
inflator classes on top.

## Picking an inflator

| Method | Mean-preserving? | Tunable | Use when |
|---|---|---|---|
| [`MultiplicativeInflator`][filterax.MultiplicativeInflator] / [`inflate_multiplicative`][filterax.inflate_multiplicative] | Yes | `factor` | Simple scalar inflation; production default |
| [`RTPS`][filterax.RTPS] / [`inflate_rtps`][filterax.inflate_rtps] | Yes (per-variable spread) | `alpha ∈ [0, 1]` | Per-variable adaptive recovery of forecast spread (Whitaker & Hamill 2012) |
| [`RTPP`][filterax.RTPP] / [`inflate_rtpp`][filterax.inflate_rtpp] | Yes | `alpha ∈ [0, 1]` | Full-anomaly blend that preserves inter-variable correlation (Zhang et al. 2004) |
| [`AdditiveInflator`][filterax.AdditiveInflator] / [`inflate_additive`][filterax.inflate_additive] | Yes (after recentring) | `Q_add` | Explicit model-error injection (Hamill & Whitaker 2005) |
| [`inflate_adaptive`][filterax.inflate_adaptive] | (Anderson 2009 prior update) | `prior (μ_λ, σ²_λ)` | Self-tuning multiplicative inflation; carry posterior across cycles |
| [`ledoit_wolf_shrinkage`][filterax.ledoit_wolf_shrinkage] | n/a (returns covariance) | none (analytic optimum) | Regularise the rank-deficient sample covariance when ``Nₑ ≪ Nₓ`` |

## Inflator classes

Drop-in [`AbstractInflator`][filterax.AbstractInflator] implementations,
accepted by every Layer-2 assimilation loop via its `inflator=` argument.

::: filterax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [MultiplicativeInflator, RTPS, RTPP, AdditiveInflator]

## Functional primitives

The pure functions underneath the classes, plus the two helpers
(`inflate_adaptive`, `ledoit_wolf_shrinkage`) that have no class form.

::: filterax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [inflate_multiplicative, inflate_rtps, inflate_rtpp, inflate_additive, inflate_adaptive, ledoit_wolf_shrinkage]

## Composition pattern

The L2 models (`filterax.ETKF`, `filterax.LETKF`, …) accept any
`AbstractInflator` instance. To carry stateful inflation (e.g. adaptive
λ) across cycles, drive the loop yourself with the L1 components:

```python
import filterax as flx

# Anderson 2009: carry (μ_λ, σ²_λ) across cycles.
mu, var = 1.0, 0.01
inflator = flx.MultiplicativeInflator(factor=mu)

for obs, time in observations:
    forecast = vmap_dynamics(particles, t_prev, time)
    result = flx.filters.ETKF().analysis(forecast, obs, obs_op, R)

    # Update the inflation belief from this cycle's innovation.
    S = flx.innovation_covariance(vmap(obs_op)(forecast), R).as_matrix()
    innov = obs - vmap(obs_op)(forecast).mean(axis=0)
    mu, var = flx.inflate_adaptive(mu, var, innov, S)
    inflator = flx.MultiplicativeInflator(factor=mu)

    particles = inflator(result.particles, forecast)
```

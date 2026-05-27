# Inflation

Counteract spread collapse from finite-ensemble, model-error, or
localization-side-effect sources. Three families of primitives, each
also exposed as a concrete [`AbstractInflator`][filterax.AbstractInflator]
class that drops straight into the Layer-2 assimilation loops.

## Picking an inflator

| Method | Mean-preserving? | Tunable | Use when |
|---|---|---|---|
| [`MultiplicativeInflator`][filterax.MultiplicativeInflator] / [`inflate_multiplicative`][filterax.inflate_multiplicative] | Yes | `factor` | Simple scalar inflation; production default |
| [`RTPS`][filterax.RTPS] / [`inflate_rtps`][filterax.inflate_rtps] | Yes (per-variable spread) | `alpha ∈ [0, 1]` | Per-variable adaptive recovery of forecast spread (Whitaker & Hamill 2012) |
| [`RTPP`][filterax.RTPP] / [`inflate_rtpp`][filterax.inflate_rtpp] | Yes | `alpha ∈ [0, 1]` | Full-anomaly blend that preserves inter-variable correlation (Zhang et al. 2004) |
| [`AdditiveInflator`][filterax.AdditiveInflator] / [`inflate_additive`][filterax.inflate_additive] | Yes (after recentring) | `Q_add` | Explicit model-error injection (Hamill & Whitaker 2005) |
| [`inflate_adaptive`][filterax.inflate_adaptive] | (Anderson 2009 prior update) | `prior (μ_λ, σ²_λ)` | Self-tuning multiplicative inflation; carry posterior across cycles |
| [`ledoit_wolf_shrinkage`][filterax.ledoit_wolf_shrinkage] | n/a (returns covariance) | none (analytic optimum) | Regularise the rank-deficient sample covariance when ``Nₑ ≪ Nₓ`` |

## Reference

### Multiplicative + relaxation

::: filterax.MultiplicativeInflator
::: filterax.RTPS
::: filterax.RTPP
::: filterax.inflate_multiplicative
::: filterax.inflate_rtps
::: filterax.inflate_rtpp

### Wave 4 additions

::: filterax.AdditiveInflator
::: filterax.inflate_additive
::: filterax.inflate_adaptive
::: filterax.ledoit_wolf_shrinkage

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

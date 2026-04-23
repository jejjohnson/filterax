# Layer 0 — Primitives

Pure functions on arrays and operators. Stateless, differentiable,
composable; compatible with `jax.jit`, `jax.grad`, and `eqx.filter_vmap`.

Heavy linear algebra (low-rank operators, Woodbury solves, matrix-determinant
lemma) is delegated to [`gaussx`](https://github.com/jejjohnson/gaussx);
`filterax` stays thin and opinionated.

Import from the top-level `filterax` namespace — the `_src` subpackage is an
internal implementation detail and not part of the stable API.

---

## Ensemble statistics

::: filterax.ensemble_mean
::: filterax.ensemble_anomalies
::: filterax.ensemble_covariance
::: filterax.cross_covariance

---

## Kalman gain

::: filterax.kalman_gain

---

## Likelihood and innovation diagnostics

::: filterax.log_likelihood
::: filterax.innovation_statistics
::: filterax.InnovationStatistics

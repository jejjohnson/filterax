# Layer 0 — Primitives

Pure functions on arrays and operators. Stateless, differentiable,
composable; compatible with `jax.jit`, `jax.grad`, and `eqx.filter_vmap`.

Heavy linear algebra (low-rank operators, Woodbury solves, matrix-determinant
lemma) is delegated to [`gaussx`](https://github.com/jejjohnson/gaussx);
`filterax` stays thin and opinionated.

---

## Ensemble statistics

::: filterax._src.statistics

---

## Kalman gain

::: filterax._src.gain

---

## Likelihood and innovation diagnostics

::: filterax._src.likelihood

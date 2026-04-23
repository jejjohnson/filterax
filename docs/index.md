# filterax

> Differentiable ensemble Kalman filters, smoothers, and processes for JAX.

`filterax` is a JAX-native library for ensemble data assimilation. It covers two complementary method families under one API:

- **Sequential filters** — Stochastic EnKF, ETKF, EnSRF, ESTKF, LETKF, and ensemble smoothers.
- **Iterative processes** — EKI, EKS, UKI, ETKI, GNKI for parameter estimation and posterior sampling.

Everything is pure JAX, so every filter is differentiable by construction — there is no separate "differentiable EnKF" mode.

---

## Three-layer architecture

| Layer | Role | Examples |
|-------|------|----------|
| **Layer 0 — Primitives** | Pure functions on arrays and operators | `ensemble_mean`, `ensemble_covariance`, `kalman_gain`, `log_likelihood` |
| **Layer 1 — Components** | Protocols and configurable building blocks | `AbstractSequentialFilter`, `AbstractProcess`, `LETKF`, `EKI` |
| **Layer 2 — Models** | Ready-to-run methods with sensible defaults | `filterax.LETKF(...).assimilate(...)`, `filterax.EKI(...).run(...)` |

Heavy linear algebra (structured covariance operators, Woodbury solves, log-determinants) is delegated to [`gaussx`](https://github.com/jejjohnson/gaussx); `filterax` stays thin and opinionated.

---

## Wave 1 status

The current release ships the Layer 0 primitives and the Layer 1 protocol surface. See [API Reference → Primitives](api/primitives.md). Concrete filters and processes land in later waves — the [design docs](design_docs/README.md) track the plan.

---

## Installation

```bash
uv add "filterax @ git+https://github.com/jejjohnson/filterax"
```

Requires Python 3.12+.

## Quickstart

```python
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import filterax as flx

key = jr.PRNGKey(0)
particles = jr.normal(key, (100, 20))
H = jr.normal(jr.split(key)[1], (5, 20))
R = lx.DiagonalLinearOperator(0.1 * jnp.ones(5))

K = flx.kalman_gain(particles, particles @ H.T, R)
P = flx.ensemble_covariance(particles)     # gaussx.LowRankUpdate
```

## Links

- [API Reference](api/reference.md) · [Primitives](api/primitives.md)
- [Design docs](design_docs/README.md)
- [Contributing](contributing.md)
- [Changelog](CHANGELOG.md)
- [GitHub](https://github.com/jejjohnson/filterax)

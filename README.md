# filterax

[![Tests](https://github.com/jejjohnson/filterax/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/filterax/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/filterax/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/filterax/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/filterax/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/filterax/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/filterax/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/filterax/actions/workflows/pages.yml)
[![codecov](https://codecov.io/gh/jejjohnson/filterax/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/filterax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

Differentiable ensemble Kalman filters, smoothers, and processes for JAX — built on [gaussx][gaussx], [lineax][lineax], [equinox][equinox], and [optax][optax].

Author: J. Emmanuel Johnson · Repo: <https://github.com/jejjohnson/filterax> · Docs: <https://jejjohnson.github.io/filterax>

---

## What is filterax?

`filterax` is a JAX-native library for ensemble data assimilation. It covers two complementary families of methods under one API:

- **Sequential filters** for state estimation — Stochastic EnKF, ETKF, EnSRF, ESTKF, LETKF, and ensemble smoothers.
- **Iterative processes** for parameter estimation and posterior sampling — EKI, EKS, UKI, ETKI, GNKI, and friends.

Everything is pure JAX, so every filter is differentiable by construction. There is no separate "differentiable EnKF" mode: wrap any filter in `jax.grad` and train dynamics parameters end-to-end through the assimilation window.

---

## Three-layer architecture

`filterax` is organised as a progressive-disclosure stack. Each layer is self-contained and useful on its own.

| Layer | Role | Examples |
|-------|------|----------|
| **Layer 0 — Primitives** | Pure functions on arrays and operators | `ensemble_mean`, `ensemble_covariance`, `kalman_gain`, `log_likelihood` |
| **Layer 1 — Components** | Protocols and configurable building blocks | `AbstractSequentialFilter`, `AbstractProcess`, `LETKF`, `EKI` |
| **Layer 2 — Models** | Ready-to-run methods with sensible defaults | `filterax.LETKF(...).assimilate(...)`, `filterax.EKI(...).run(...)` |

Lower layers are composable and pure; higher layers provide convenience. Use whichever layer matches the control you need.

---

## Wave 1 status

The **v0.0.0** release ships the Layer 0 foundations plus the Layer 1 protocol surface:

```python
import filterax as flx

# Layer 0 — pure ensemble algebra
flx.ensemble_mean, flx.ensemble_anomalies, flx.ensemble_covariance, flx.cross_covariance
flx.kalman_gain, flx.log_likelihood, flx.innovation_statistics

# Layer 1 — protocols and state/config types
flx.AbstractSequentialFilter, flx.AbstractProcess
flx.AbstractDynamics, flx.AbstractObsOperator
flx.AbstractNoise, flx.AbstractLocalizer, flx.AbstractInflator, flx.AbstractScheduler
flx.FilterState, flx.ProcessState, flx.UKIState, flx.AnalysisResult
flx.FilterConfig, flx.ProcessConfig
```

Concrete filters (ETKF, EnSRF, LETKF, …) and processes (EKI, EKS, UKI, …) land in later waves. See the [design docs](docs/design_docs/README.md) for the full roadmap.

---

## Installation

`filterax` is not yet on PyPI. Install from GitHub:

```bash
uv add "filterax @ git+https://github.com/jejjohnson/filterax"
# or
pip install git+https://github.com/jejjohnson/filterax
```

Requires Python 3.12+. Runtime dependencies: `jax`, `jaxlib`, `jaxtyping`, `equinox`, `lineax`, `optax`, `einops`, and [`gaussx`](https://github.com/jejjohnson/gaussx) (also resolved from GitHub until it ships to PyPI).

---

## Quick taste

```python
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import filterax as flx

key = jr.PRNGKey(0)
particles = jr.normal(key, (100, 20))        # 100-member ensemble in R^20
H = jr.normal(jr.split(key)[1], (5, 20))     # linear observation operator
R = lx.DiagonalLinearOperator(0.1 * jnp.ones(5))

obs_particles = particles @ H.T              # map ensemble to obs space
K = flx.kalman_gain(particles, obs_particles, R)  # (20, 5)
P = flx.ensemble_covariance(particles)       # gaussx.LowRankUpdate, rank <= 99
```

---

## Ecosystem

`filterax` sits alongside a family of JAX libraries. See [`docs/design_docs/boundaries.md`](docs/design_docs/boundaries.md) for the full ownership map.

| Library | Relationship |
|---------|--------------|
| [`gaussx`][gaussx] | Structured covariance operators, solves, logdet — required |
| [`lineax`][lineax] | Linear operator types — transitive through gaussx |
| [`equinox`][equinox] | Module system and PyTree compatibility |
| [`optax`][optax] | Iterative EKP processes expose an `optax.GradientTransformation` API |
| `somax` / `diffrax` | User-supplied forward models via `AbstractDynamics` |
| `vardax` | Sister library for variational DA (4DVar, 4DVarNet) |
| `xr_assimilate` | xarray-level orchestration that consumes `filterax` as a compute backend |

---

## Contributing

See [`docs/contributing.md`](docs/contributing.md) for the label taxonomy, epic model, and wave-backlog workflow. The [`design_docs`](docs/design_docs/README.md) capture the architecture, decisions, and per-wave plans.

---

## License

MIT — see [`LICENSE`](LICENSE).

[gaussx]: https://github.com/jejjohnson/gaussx
[lineax]: https://github.com/patrick-kidger/lineax
[equinox]: https://github.com/patrick-kidger/equinox
[optax]: https://github.com/google-deepmind/optax

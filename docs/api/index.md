# API Reference

filterax is a library of differentiable ensemble Kalman filters, smoothers,
and processes for JAX, built on [Equinox](https://github.com/patrick-kidger/equinox),
[lineax](https://github.com/patrick-kidger/lineax), and
[gaussx](https://github.com/jejjohnson/gaussx). The reference is organised by
the package's layered architecture — pure primitives at the bottom, one-shot
analysis components in the middle, full assimilation loops on top:

| Section | What's inside |
|---------|---------------|
| [Protocols & Types](protocols.md) | The `Abstract*` interfaces every component implements, plus the state / result / config containers they exchange |
| [Primitives](primitives.md) | Pure functions — ensemble statistics, Kalman gain, likelihood and innovation diagnostics, perturbed observations |
| [Localization](localization.md) | Taper functions (Gaspari-Cohn, Gaussian, SOAR, hard cutoff), Schur-product application, adaptive localization |
| [Inflation](inflation.md) | Multiplicative / RTPS / RTPP / additive inflators, adaptive inflation, Ledoit-Wolf shrinkage |
| [Filters](filters.md) | L1 analysis steps (`filterax.filters.ETKF`, …), the parametric `SquareRootKF`, L2 assimilation loops (`filterax.ETKF`, …), and latent-space DA |
| [Advanced filters](filters_advanced.md) | `ESTKF`, `ETKF_Livings`, `EnSRF_Serial` — specialised deterministic variants |
| [Smoothers](smoothers.md) | `EnKS`, `EnsembleRTS`, `EnsembleSqrtSmoother`, `FixedLagSmoother`, `IES` |
| [Processes](processes.md) | Ensemble Kalman processes for derivative-free inversion — EKI, EKS, UKI and friends |
| [Schedulers](schedulers.md) | Artificial-time step controllers for the EKP iteration |
| [optax integration](optax.md) | EKI / EKS / UKI as `optax.GradientTransformation`s |
| [Differentiable training](differentiable.md) | `differentiable_assimilate` (full-tape scan) and the ROAD-EnKF local-gradient strategy |
| [Diagnostics](diagnostics.md) | "Is the filter working?" — spread, innovations, rank histograms, Desroziers, CRPS |
| [pipekit integration](pipekit.md) | Adapters that plug filterax into pipekit-cycle's DA orchestration protocols |

## Import conventions

Everything documented here is importable from the top-level `filterax`
namespace or from one of the themed namespaces (`filterax.filters`,
`filterax.smoothers`, `filterax.processes`, `filterax.optax`,
`filterax.differentiable`, `filterax.utils`, `filterax.pipekit`).
Modules and subpackages with a leading underscore (`filterax._primitives`,
`filterax._filters`, …) are private implementation details and not part of
the stable API.

A name appearing in two places is deliberate: `filterax.filters.ETKF` is the
Layer-1 one-shot analysis step, while `filterax.ETKF` is the Layer-2
forecast–analyse–inflate loop built on top of it (see
[Filters](filters.md) for the full L1/L2 distinction).

## Linear algebra delegates to gaussx

filterax stays thin: heavy linear algebra — low-rank covariance operators,
Woodbury / matrix-determinant-lemma solves, taper functions, inflation
primitives — is delegated to
[gaussx](https://jejjohnson.github.io/gaussx/)
([GitHub](https://github.com/jejjohnson/gaussx)). Covariances are returned as
lineax operators (e.g. `gaussx.LowRankUpdate`), so solves dispatch on
structure rather than materialising dense `(N_x, N_x)` matrices, and any
function with a `solver=` keyword accepts a gaussx
`AbstractSolverStrategy` to override the numerical path.

# Primitives

Pure functions on arrays and operators. Stateless, differentiable,
composable; compatible with `jax.jit`, `jax.grad`, and `eqx.filter_vmap`.
Every filter and smoother in the package is assembled from these.

Heavy linear algebra is delegated to
[gaussx](https://jejjohnson.github.io/gaussx/):
[`ensemble_covariance`][filterax.ensemble_covariance] returns a
`gaussx.LowRankUpdate` operator (rank `N_e`, never a dense `(N_x, N_x)`
matrix), and the solves inside [`kalman_gain`][filterax.kalman_gain] and
[`log_likelihood`][filterax.log_likelihood] dispatch through gaussx's
structural solver — a Woodbury / matrix-determinant-lemma path for low-rank
updates — with an optional `solver=` keyword to override the strategy.

Localization tapers and inflation primitives have their own pages:
[Localization](localization.md) and [Inflation](inflation.md).

## Ensemble statistics

Bessel-corrected moments of an `(N_e, N_x)` ensemble: mean, centred
anomalies, and the (cross-)covariances built from them.

::: filterax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [ensemble_mean, ensemble_anomalies, ensemble_covariance, cross_covariance]

## Kalman gain

::: filterax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [kalman_gain]

## Likelihood & innovation diagnostics

The innovation covariance `S = Cᴴᴴ + R` as a low-rank update, the Gaussian
log-likelihood of an innovation under it, and the bundled per-cycle
innovation statistics.

::: filterax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [log_likelihood, innovation_covariance, innovation_statistics, InnovationStatistics]

## Perturbed observations

Stochastic-EnKF observation perturbations, drawn with an explicit PRNG key.

::: filterax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [perturbed_observations]

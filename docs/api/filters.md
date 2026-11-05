# Filters

Sequential ensemble filters come in two layers:

* **Layer 1 — analysis steps** (`filterax.filters.*`). A one-shot
  posterior update: forecast ensemble + observation + observation operator +
  noise covariance in, analysed ensemble out
  ([`AnalysisResult`][filterax.AnalysisResult]). All implement
  [`AbstractSequentialFilter`][filterax.AbstractSequentialFilter] and share
  the gaussx-backed ensemble-statistics primitives; they differ only in *how*
  the ensemble is updated.
* **Layer 2 — assimilation models** (top-level `filterax.*`, same names).
  The full forecast → analyse → inflate loop: each composes an
  [`AbstractDynamics`][filterax.AbstractDynamics] forecast (via
  `eqx.filter_vmap`), the matching L1 analysis step, and an optional
  [`AbstractInflator`][filterax.AbstractInflator] into a single
  `assimilate()` call over a sequence of observation windows, returning an
  [`AssimilationResult`][filterax.AssimilationResult].

So `filterax.filters.ETKF` is the one-shot analysis step, while
`filterax.ETKF` is the loop built around it. Drop down to Layer 1 when the
standard cycle does not fit (custom inflation schedules, pipekit cycling,
differentiable training).

!!! note "ETKF analysis core"
    `ETKF`, `EnSRF`, `ESTKF` and `ETKF_Livings` run `gaussx.etkf_analysis`. With fewer observations than members it takes a thin QR and an `(N_y, N_y)` symmetric eigendecomposition, and forms no `(N_e, N_e)` matrix; otherwise it takes one `(N_e, N_e)` eigendecomposition. Its derivative is taken in closed form, so it stays finite under `jax.grad` at the repeated eigenvalues of the transform (see [Differentiable training](differentiable.md)).

For the specialised deterministic variants (`ESTKF`, `ETKF_Livings`,
`EnSRF_Serial`) see [Advanced filters](filters_advanced.md).

## Layer-1 analysis steps

The everyday defaults: symmetric square-root ETKF, observation-space
localized LETKF, serial-free EnSRF, and the perturbed-observations
stochastic EnKF.

::: filterax.filters
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [ETKF, LETKF, EnSRF, StochasticEnKF]

## Parametric square-root filter

A non-ensemble reference: propagates `(μ, S)` with `P = S Sᵀ` instead of
particles, wrapping gaussx's parallel Kalman filter. Use it for
linear-Gaussian baselines and twin-experiment ground truths.

::: filterax.filters
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [SquareRootKF, SquareRootFilterResult]

## Layer-2 assimilation models

The full forecast–analyse–inflate loops. Each `assimilate()` call stacks
per-window forecasts, posteriors, and log-likelihoods along a leading time
axis in the returned [`AssimilationResult`][filterax.AssimilationResult].

::: filterax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [ETKF, LETKF, EnSRF, StochasticEnKF]

## Latent-space assimilation

Run the analysis in a learned (or prescribed) latent space `z = E(x)`:
`LatentETKF` / `LatentLETKF` are L2 loops that forecast and analyse in
z-space and decode back, returning a
[`LatentAssimilationResult`][filterax.LatentAssimilationResult]. The helper
wrappers lift the pieces of a latent problem into the standard protocols —
`LatentDynamics` wraps a z-space forward model, `LiftedObs` composes the
decoder with an x-space observation operator, and `EncodedDynamics` runs
x-space dynamics through an encode/decode round-trip.

::: filterax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [LatentETKF, LatentLETKF, latent_ensemble, decode_ensemble, identity_latent_map, IdentityLatentMap, LatentDynamics, LiftedObs, EncodedDynamics]

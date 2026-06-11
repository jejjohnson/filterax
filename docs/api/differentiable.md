# Differentiable training

Train dynamics (or observation operators) *through* the filter: run a full
assimilation over `T` windows, score it with a log-likelihood or
truth-matching loss, and differentiate with respect to the model
parameters. Two gradient strategies are provided:

* [`differentiable_assimilate`][filterax.differentiable_assimilate] — a
  fixed-shape `jax.lax.scan` over the forecast–analysis cycle with the full
  forward+backward tape (optionally `jax.checkpoint`-ed for `O(√T)` memory).
  Use when you want exact gradients, including the cross-time terms that
  flow through the analysis updates.
* `road_enkf_loss_and_grad` / `road_enkf_grad_step` — the ROAD-EnKF
  local-gradient strategy, with `jax.lax.stop_gradient` between cycles.
  Backward-pass memory is `O(Nₑ · Nₓ)` independent of `T`, at the cost of
  dropping cross-time gradient terms.

Both refuse stochastic filters (`StochasticEnKF`,
[`ETKF_Livings`](filters_advanced.md)) and the stochastic
`AdditiveInflator` at call time — randomness in the analysis breaks the
deterministic tape these strategies rely on. The deterministic
[ETKF analysis core](filters.md) (rank-`N_y` QR + symmetric `eigh`) is what
keeps the exact-gradient path differentiability-safe.

## Full-tape assimilation

::: filterax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [differentiable_assimilate]

## ROAD-EnKF local gradients

`road_enkf_loss_and_grad` evaluates the loss and its local gradient in one
pass; `road_enkf_grad_step` wraps it with an optax optimizer update for a
ready-made training step. Both live in `filterax.differentiable`.

::: filterax.differentiable
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [road_enkf_loss_and_grad, road_enkf_grad_step]

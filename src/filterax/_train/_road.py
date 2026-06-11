r"""ROAD-EnKF — reduced-order autodifferentiable EnKF (Wave 5.B+).

Memory-efficient gradient strategy for training through the filter on
long observation horizons. Where :func:`differentiable_assimilate` keeps
the full reverse-mode autodiff tape over ``T`` filter cycles (memory
``O(T · Nₑ · Nₓ)``, or ``O(√T · Nₑ · Nₓ)`` with ``jax.checkpoint``),
ROAD-EnKF takes the **local-gradient** approach:

1. At step ``t``, compute the per-step loss ``L_t = -log p(y_t | X^f_t)``
   *with* gradient enabled, getting a local gradient
   ``∇_θ L_t``.
2. Sum the local gradients across all steps:

   $$
   \nabla_\theta L \approx \sum_t \nabla_\theta L_t.
   $$

3. Advance the ensemble between steps with :func:`jax.lax.stop_gradient`
   so no autodiff tape spans more than one filter cycle.

Backward-pass memory stays ``O(Nₑ · Nₓ)`` regardless of ``T``. The
trade-off is that *cross-time* gradient terms are dropped — the local
gradient ``∇_θ L_t`` reflects how ``θ`` shapes the forecast / analysis
at step ``t``, but does **not** include the chain
``∂L_t / ∂X^a_{t-1} · ∂X^a_{t-1} / ∂θ``. In practice this is the
accepted ROAD-EnKF trade-off (Chen et al. 2023) — for long horizons the
cross-time terms decay through the filter's stability and the
gradient-bias-vs-memory trade is heavily favoured.

References
----------
* Chen, Y., Huang, D. Z. & Stuart, A. M. (2023). *ROAD-EnKF:
  Reduced-Order Autodiff Ensemble Kalman Filters.*
  https://github.com/ymchen0/ROAD-EnKF
* :mod:`filterax._train._differentiable` — companion full-backprop loop.
* ``design_docs/features/differentiable_da.md`` §6.D — algorithm
  derivation in the project's vocabulary.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optax
from jaxtyping import Array, Float, PyTree

from filterax._protocols import (
    AbstractDynamics,
    AbstractInflator,
    AbstractObsOperator,
    AbstractSequentialFilter,
)
from filterax._train._differentiable import _forecast, _reject_stochastic_components


def road_enkf_loss_and_grad(
    dynamics: AbstractDynamics,
    init_ensemble: Float[Array, "N_e N_x"],
    observations: Float[Array, "T N_y"],
    obs_times: Float[Array, " T"],
    obs_noise: lx.AbstractLinearOperator,
    obs_op: AbstractObsOperator,
    *,
    filter_: AbstractSequentialFilter | None = None,
    inflator: AbstractInflator | None = None,
    t0: float | Float[Array, ""] = 0.0,
    **analysis_extra: Any,
) -> tuple[Float[Array, ""], PyTree]:
    r"""Local-gradient NLL loss and dynamics gradient on a filter window.

    Per-step :func:`equinox.filter_value_and_grad` over the forecast +
    analysis cycle, with :func:`jax.lax.stop_gradient` between cycles.
    Backward-pass memory is ``O(Nₑ · Nₓ)`` independent of ``T`` (cf.
    :func:`differentiable_assimilate` whose tape grows linearly in
    ``T``).

    Args:
        dynamics: Forward model. Trainable: every array leaf of this
            ``AbstractDynamics`` PyTree contributes a gradient.
        init_ensemble: Prior ensemble ``(Nₑ, Nₓ)``.
        observations: Stacked observation history ``(T, Nᵧ)``.
        obs_times: Stacked timestamps ``(T,)`` — the forecast for window
            ``t`` propagates from ``obs_times[t-1]`` (or ``t0`` when
            ``t == 0``) to ``obs_times[t]``.
        obs_noise: Observation error covariance ``R``.
        obs_op: Observation operator ``H``.
        filter_: Deterministic L1 filter (``ETKF`` by default).
            Stochastic filters (``StochasticEnKF``, ``ETKF_Livings``)
            and the stochastic ``AdditiveInflator`` are rejected at
            call time — their PRNG draws break smooth gradients.
        inflator: Optional deterministic inflator
            (``MultiplicativeInflator`` / ``RTPS`` / ``RTPP``).
        t0: Starting time used for the first forecast window.
        **analysis_extra: Forwarded to ``filter_.analysis(...)`` (use
            for LETKF's ``state_coords`` / ``obs_coords``).

    Returns:
        ``(loss, dynamics_grad)`` where ``loss`` is the summed
        observation-space negative log-likelihood and ``dynamics_grad``
        is a PyTree of the same structure as ``dynamics`` (only its
        array leaves carry gradients; static / non-array fields are
        ``None``-like via ``eqx.filter_value_and_grad``).
    """
    # Use a deterministic default filter; import here to avoid circular import.
    if filter_ is None:
        from filterax._filters._sequential import ETKF

        filter_ = ETKF()
    _reject_stochastic_components(filter_, inflator)
    if observations.shape[0] != obs_times.shape[0]:
        raise ValueError(
            "observations and obs_times must have the same leading axis; "
            f"got {observations.shape[0]} and {obs_times.shape[0]}."
        )

    def local_step(
        dyn: AbstractDynamics,
        ensemble: Float[Array, "N_e N_x"],
        t_prev: Float[Array, ""],
        t_now: Float[Array, ""],
        obs_t: Float[Array, " N_y"],
    ) -> tuple[Float[Array, ""], Float[Array, "N_e N_x"]]:
        """One forecast + analysis with gradient tape on ``dyn`` only."""
        forecast = _forecast(dyn, ensemble, t_prev, t_now)
        result = filter_.analysis(forecast, obs_t, obs_op, obs_noise, **analysis_extra)
        analysis_particles = result.particles
        if inflator is not None:
            analysis_particles = inflator(analysis_particles, forecast)
        # Every deterministic filter returns a log-likelihood; the protocol's
        # ``Array | None`` slot exists for future stochastic variants.
        log_lik = result.log_likelihood
        if log_lik is None:
            log_lik = jnp.full((), jnp.nan, dtype=ensemble.dtype)
        return -log_lik, analysis_particles

    # ``filter_value_and_grad`` with ``has_aux`` returns the analysis
    # particles alongside the local loss, so we don't recompute the
    # forecast + analysis to advance the carry.
    grad_step = eqx.filter_value_and_grad(local_step, has_aux=True)

    # Unify the time dtype so the per-step ``t_prev`` carry stays consistent
    # regardless of caller-supplied dtypes — mirrors the same fix in
    # ``differentiable_assimilate``.
    t0_arr = jnp.asarray(t0)
    time_dtype = jnp.result_type(t0_arr, obs_times)
    t0_arr = t0_arr.astype(time_dtype)
    obs_times = obs_times.astype(time_dtype)

    # Seed the gradient accumulator with the right pytree shape — zeros
    # over *inexact* array leaves (float / complex) of ``dynamics``.
    # ``eqx.filter_value_and_grad`` only ever produces gradients for
    # inexact leaves, so we use ``eqx.is_inexact_array`` here to skip
    # integer/bool array leaves (indices, masks, PRNG-key metadata)
    # rather than seeding zero "gradients" optax would then try to
    # apply to them.
    total_grad_seed: PyTree = jax.tree.map(
        lambda leaf: jnp.zeros_like(leaf) if eqx.is_inexact_array(leaf) else None,
        eqx.filter(dynamics, eqx.is_inexact_array),
    )
    loss_seed = jnp.asarray(0.0, dtype=init_ensemble.dtype)

    # ``jax.lax.scan`` over the observation history. Each iteration runs
    # ``filter_value_and_grad`` on the per-step local loss; the resulting
    # gradient accumulates into a carry leaf of the same dynamics-shaped
    # PyTree. ``stop_gradient`` on the analysis particles between
    # iterations stops any reverse-mode tape from spanning steps — so the
    # backward-pass memory stays ``O(Nₑ · Nₓ)`` and, equally important,
    # the *forward* JIT trace stays ``O(1)`` in ``T`` (the loop body is
    # traced once and ``lax.scan`` carries it).
    def scan_step(carry, inputs):
        ensemble, total_loss, total_grad, t_prev = carry
        obs_t, t_now = inputs
        (loss_t, analysis_particles), grad_t = grad_step(
            dynamics, ensemble, t_prev, t_now, obs_t
        )
        new_total_loss = total_loss + loss_t
        new_total_grad = jax.tree.map(lambda acc, g: acc + g, total_grad, grad_t)
        new_ensemble = jax.lax.stop_gradient(analysis_particles)
        return (new_ensemble, new_total_loss, new_total_grad, t_now), None

    init_carry = (init_ensemble, loss_seed, total_grad_seed, t0_arr)
    (_, total_loss, total_grad, _), _ = jax.lax.scan(
        scan_step, init_carry, (observations, obs_times)
    )
    return total_loss, total_grad


def road_enkf_grad_step(
    dynamics: AbstractDynamics,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    init_ensemble: Float[Array, "N_e N_x"],
    observations: Float[Array, "T N_y"],
    obs_times: Float[Array, " T"],
    obs_noise: lx.AbstractLinearOperator,
    obs_op: AbstractObsOperator,
    *,
    filter_: AbstractSequentialFilter | None = None,
    inflator: AbstractInflator | None = None,
    t0: float | Float[Array, ""] = 0.0,
    **analysis_extra: Any,
) -> tuple[AbstractDynamics, optax.OptState, Float[Array, ""]]:
    """One ROAD-EnKF descent step composed with an :mod:`optax` optimizer.

    Convenience wrapper that runs :func:`road_enkf_loss_and_grad`, threads
    the gradient through ``optimizer.update``, and applies the updates
    to the dynamics module. Returns ``(new_dynamics, new_opt_state, loss)``
    so the caller can log loss and continue the training loop.

    Args:
        dynamics: Forward model to update.
        optimizer: Any ``optax.GradientTransformation``
            (``optax.adam(1e-3)``, ``optax.chain(...)``, …).
        opt_state: Optimizer state. Initialise with the *inexact*
            (gradient-eligible) leaves only — matches what the gradient
            update will write to::

                opt_state = optimizer.init(
                    eqx.filter(dynamics, eqx.is_inexact_array)
                )

            Filtering on ``eqx.is_array`` instead would let optax try
            to write float updates onto integer / boolean metadata
            (indices, masks, …) and either fail or silently corrupt
            those fields.
        init_ensemble: Prior ensemble ``(Nₑ, Nₓ)``.
        observations: Stacked observation history ``(T, Nᵧ)``.
        obs_times: Stacked timestamps ``(T,)``.
        obs_noise: Observation error covariance ``R``.
        obs_op: Observation operator ``H``.
        filter_: Deterministic L1 filter (default ``ETKF``).
        inflator: Optional deterministic inflator.
        t0: Starting time used for the first forecast window.
        **analysis_extra: Forwarded to ``filter_.analysis(...)``.

    Returns:
        ``(new_dynamics, new_opt_state, loss)`` — the updated module,
        the advanced optimizer state, and the loss at the current
        ``dynamics``.
    """
    loss, grads = road_enkf_loss_and_grad(
        dynamics,
        init_ensemble,
        observations,
        obs_times,
        obs_noise,
        obs_op,
        filter_=filter_,
        inflator=inflator,
        t0=t0,
        **analysis_extra,
    )
    # Pass only the *inexact* (gradient-eligible) array leaves of
    # ``dynamics`` to optax — matches the ``eqx.is_inexact_array`` filter
    # used to seed the gradient accumulator in
    # ``road_enkf_loss_and_grad`` and the convention callers should use
    # when constructing ``opt_state = optimizer.init(eqx.filter(dynamics,
    # eqx.is_inexact_array))``. Filtering on ``eqx.is_array`` instead
    # would let optax try to write updates onto integer / bool leaves.
    params = eqx.filter(dynamics, eqx.is_inexact_array)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_dynamics = eqx.apply_updates(dynamics, updates)
    return new_dynamics, new_opt_state, loss

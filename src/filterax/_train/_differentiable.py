"""Differentiable assimilation loop (Wave 5.B).

filterax filters are differentiable by construction (Decision D9 +
``design_docs/features/differentiable_da.md``) — every analysis step
is a pure JAX computation that ``jax.grad`` and ``jax.jit`` already
compose with. What this module adds is the *scan + remat* idiom the
design doc calls out: a fixed-shape :func:`jax.lax.scan` over a stacked
observation history, optionally wrapping the step in
:func:`jax.checkpoint` so reverse-mode AD over long ``T`` doesn't blow
out memory.

The L2 :class:`filterax.ETKF` / :class:`filterax.EnSRF` / … wrappers
use a Python ``for`` loop instead. That works under ``jax.grad`` for
short ``T`` but unrolls into one giant traced graph at compile time and
doesn't compose with ``jax.checkpoint``. Use
:func:`differentiable_assimilate` when training through the filter.

The design doc is explicit about which filter / inflator flavours play
well with ``jax.grad``:

* **Filters** — use deterministic square-root variants (``ETKF``,
  ``EnSRF``, ``ESTKF``, ``LETKF``, ``EnSRF_Serial``). Stochastic ones
  (``StochasticEnKF``, ``ETKF_Livings``) inject per-step PRNG draws
  that break smooth gradients.
* **Inflators** — use ``MultiplicativeInflator`` / ``RTPS`` / ``RTPP``.
  ``AdditiveInflator`` is stochastic.

We validate the stochastic-component constraint at call time so users
get a clear error instead of silently noisy gradients.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from filterax._protocols import (
    AbstractDynamics,
    AbstractInflator,
    AbstractObsOperator,
    AbstractSequentialFilter,
)
from filterax._types import AssimilationResult


def _forecast(
    dynamics: AbstractDynamics,
    particles: Float[Array, "N_e N_x"],
    t0: Float[Array, ""],
    t1: Float[Array, ""],
) -> Float[Array, "N_e N_x"]:
    """vmap dynamics over ensemble members — mirrors models._forecast."""
    return eqx.filter_vmap(lambda x: dynamics(x, t0, t1))(particles)


def _reject_stochastic_components(
    filter_: AbstractSequentialFilter,
    inflator: AbstractInflator | None,
) -> None:
    """Refuse filter / inflator flavours whose gradients are non-smooth.

    Deferred imports avoid a module-level cycle with
    :mod:`filterax._filters._sequential` and :mod:`filterax._primitives._inflators`.
    """
    from filterax._filters._sequential import ETKF_Livings, StochasticEnKF
    from filterax._primitives._inflators import AdditiveInflator

    if isinstance(filter_, (StochasticEnKF, ETKF_Livings)):
        raise ValueError(
            f"{type(filter_).__name__} uses per-step PRNG draws that break "
            "smooth gradients under jax.grad. Use a deterministic square-root "
            "filter (ETKF, EnSRF, ESTKF, EnSRF_Serial, LETKF) for "
            "differentiable training — see "
            "design_docs/features/differentiable_da.md §5.2."
        )
    if isinstance(inflator, AdditiveInflator):
        raise ValueError(
            "AdditiveInflator injects PRNG draws that break smooth "
            "gradients. Use MultiplicativeInflator / RTPS / RTPP for "
            "differentiable training — see "
            "design_docs/features/differentiable_da.md §5.5."
        )


def differentiable_assimilate(
    filter_: AbstractSequentialFilter,
    dynamics: AbstractDynamics,
    obs_op: AbstractObsOperator,
    init_ensemble: Float[Array, "N_e N_x"],
    observations: Float[Array, "T N_y"],
    obs_times: Float[Array, " T"],
    obs_noise: lx.AbstractLinearOperator,
    *,
    inflator: AbstractInflator | None = None,
    t0: float | Float[Array, ""] = 0.0,
    checkpoint: bool = False,
    **analysis_extra: Any,
) -> AssimilationResult:
    r"""Scan-based assimilation loop suitable for ``jax.grad`` / ``jax.jit``.

    Drop-in replacement for the L2 ``model.assimilate(...)`` Python loop
    when the caller needs a single fused XLA ``While`` (large ``T``,
    short compile, optional gradient checkpointing). Observations and
    timestamps are stacked along a leading time axis so the loop body
    has a static shape.

    Args:
        filter_: Deterministic L1 filter (``ETKF``, ``EnSRF``, ``ESTKF``,
            ``EnSRF_Serial``, or ``LETKF``). Stochastic filters raise.
        dynamics: Forward model applied with :func:`equinox.filter_vmap`
            over the ensemble.
        obs_op: Observation operator ``H``.
        init_ensemble: Prior ensemble ``(Nₑ, Nₓ)``.
        observations: Stacked obs ``(T, Nᵧ)``.
        obs_times: Stacked timestamps ``(T,)``. The forecast for window
            ``t`` propagates from ``obs_times[t-1]`` (or ``t0`` when
            ``t == 0``) to ``obs_times[t]``.
        obs_noise: Observation error covariance ``R``.
        inflator: Optional deterministic inflator
            (``MultiplicativeInflator`` / ``RTPS`` / ``RTPP``).
            ``AdditiveInflator`` raises.
        t0: Starting time used for the first forecast window. Default 0.
        checkpoint: When ``True``, wrap the scan body with
            :func:`jax.checkpoint`. Reduces backward-pass memory from
            ``O(T Nₑ Nₓ)`` to ``O(√T Nₑ Nₓ)`` at 2–3× compute cost — see
            §5.1 of the differentiable-DA design doc.
        **analysis_extra: Forwarded to ``filter_.analysis(...)``. Use
            this for filters that need extra context (``LETKF`` takes
            ``state_coords`` / ``obs_coords`` here).

    Returns:
        :class:`AssimilationResult` with the same fields as
        ``L2.assimilate(...)``. ``log_likelihoods`` is always populated
        because every deterministic filter returns one.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import filterax as flx
        >>> particles = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        >>> observations = jnp.array([[0.5], [0.4]])  # (T, N_y) = (2, 1)
        >>> obs_times = jnp.array([1.0, 2.0])
        >>> R = lx.DiagonalLinearOperator(0.1 * jnp.ones(1))
        >>> result = flx.differentiable_assimilate(
        ...     flx.filters.ETKF(),
        ...     lambda x, t0, t1: x,  # identity dynamics
        ...     lambda x: x[:1],  # observe the first component
        ...     particles,
        ...     observations,
        ...     obs_times,
        ...     R,
        ... )
        >>> result.analysis_history.shape, result.log_likelihoods.shape
        ((2, 3, 2), (2,))
    """
    _reject_stochastic_components(filter_, inflator)
    if observations.shape[0] != obs_times.shape[0]:
        raise ValueError(
            "observations and obs_times must have the same leading axis; "
            f"got {observations.shape[0]} and {obs_times.shape[0]}."
        )

    # ``lax.scan`` requires the carry's input and output dtypes to match
    # exactly. ``t_prev`` enters the carry from ``t0`` and exits it from
    # the per-step ``obs_times`` element, so cast both to a common dtype
    # up front — otherwise float32 ensembles with float64 (or integer)
    # ``obs_times`` raise at trace time even though the L2 Python loop
    # would accept that combination.
    t0_arr = jnp.asarray(t0)
    time_dtype = jnp.result_type(t0_arr, obs_times)
    t0_arr = t0_arr.astype(time_dtype)
    obs_times = obs_times.astype(time_dtype)

    def step(
        carry: tuple[Float[Array, "N_e N_x"], Float[Array, ""]],
        inputs: tuple[Float[Array, " N_y"], Float[Array, ""]],
    ) -> tuple[
        tuple[Float[Array, "N_e N_x"], Float[Array, ""]],
        tuple[Float[Array, "N_e N_x"], Float[Array, "N_e N_x"], Float[Array, ""]],
    ]:
        particles, t_prev = carry
        obs_t, t_now = inputs
        forecast = _forecast(dynamics, particles, t_prev, t_now)
        result = filter_.analysis(forecast, obs_t, obs_op, obs_noise, **analysis_extra)
        analysis_particles = result.particles
        if inflator is not None:
            analysis_particles = inflator(analysis_particles, forecast)
        # ``log_likelihood`` is typed ``Array | None`` on the protocol but
        # is populated by every deterministic filter we accept. The
        # Python-level guard is resolved at trace time, so when a filter
        # ever returns ``None`` we fall back to NaN rather than feeding
        # ``None`` into ``lax.scan``.
        log_lik = result.log_likelihood
        if log_lik is None:
            log_lik = jnp.full((), jnp.nan, dtype=forecast.dtype)
        return (analysis_particles, t_now), (
            forecast,
            analysis_particles,
            log_lik,
        )

    step_fn = jax.checkpoint(step) if checkpoint else step
    init_carry = (init_ensemble, t0_arr)
    (final_particles, _), (forecasts, analyses, logps) = jax.lax.scan(
        step_fn, init_carry, (observations, obs_times)
    )
    return AssimilationResult(
        particles=final_particles,
        forecast_history=forecasts,
        analysis_history=analyses,
        log_likelihoods=logps,
    )

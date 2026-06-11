"""Layer-2 high-level assimilation models.

Compose a forecast (via :func:`eqx.filter_vmap` over an
:class:`AbstractDynamics`), an analysis step (L1
:class:`AbstractSequentialFilter`), and an optional inflator into a
single ``assimilate`` call that loops over a sequence of
``(obs_values, obs_time)`` windows.

The L2 models are deliberately thin Python wrappers around the L1
components so users can drop down to ``filter.analysis(...)`` and roll
their own loop when the standard cycle does not fit.

The :class:`AssimilationResult` returned by every L2 model stacks the
per-window forecasts, posteriors, and log-likelihoods along a leading
time axis of length ``T = len(observations)``. The log-likelihood array
is *always* length ``T`` when present — missing entries are filled with
NaN so callers can index by window without re-aligning.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray

from filterax._filters._latent import (
    LatentDynamics,
    LiftedObs,
    _stack_decode,
    decode_ensemble,
    latent_ensemble,
)
from filterax._filters._sequential import (
    ETKF as _ETKFFilter,
    LETKF as _LETKFFilter,
    EnSRF as _EnSRFFilter,
    StochasticEnKF as _StochasticEnKFFilter,
)
from filterax._protocols import (
    AbstractDynamics,
    AbstractInflator,
    AbstractObsOperator,
    AbstractSequentialFilter,
)
from filterax._types import (
    AssimilationResult,
    FilterConfig,
    LatentAssimilationResult,
)


def _forecast(
    dynamics: AbstractDynamics,
    particles: Float[Array, "N_e N_x"],
    t0: Float[Array, ""],
    t1: Float[Array, ""],
) -> Float[Array, "N_e N_x"]:
    """Propagate every ensemble member through ``dynamics`` from ``t0`` to ``t1``."""
    return eqx.filter_vmap(lambda x: dynamics(x, t0, t1))(particles)


_FilterFactory = Callable[[int], AbstractSequentialFilter]


def _run_loop(
    filter_factory: _FilterFactory,
    dynamics: AbstractDynamics,
    obs_op: AbstractObsOperator,
    inflator: AbstractInflator | None,
    init_ensemble: Float[Array, "N_e N_x"],
    observations: Sequence[tuple[Float[Array, " N_y"], float | Float[Array, ""]]],
    obs_noise: lx.AbstractLinearOperator,
    t0: float | Float[Array, ""],
    analysis_extra: dict | None = None,
) -> AssimilationResult:
    """Generic forecast → analysis → inflate loop shared by every L2 model.

    ``filter_factory(step)`` produces the L1 filter for window ``step``.
    Stochastic filters override the factory to fold a fresh key per step;
    deterministic filters ignore the step argument and return the same
    instance.

    Log-likelihoods are kept length-``T`` (filling ``NaN`` when an L1
    filter does not produce one) so the time axis stays aligned with the
    forecast and analysis histories.
    """
    extra = analysis_extra or {}
    particles = init_ensemble

    forecasts: list[Float[Array, "N_e N_x"]] = []
    analyses: list[Float[Array, "N_e N_x"]] = []
    logps: list[Float[Array, ""] | None] = []

    t_prev = jnp.asarray(t0)
    for step, (obs_values, obs_time) in enumerate(observations):
        t_now = jnp.asarray(obs_time)
        forecast = _forecast(dynamics, particles, t_prev, t_now)
        forecasts.append(forecast)

        filter_ = filter_factory(step)
        result = filter_.analysis(forecast, obs_values, obs_op, obs_noise, **extra)
        analysis_particles = result.particles
        if inflator is not None:
            # ``step=`` lets stochastic inflators (AdditiveInflator) fold
            # a fresh PRNG sub-key per window. Deterministic inflators
            # accept the kwarg via **_ and ignore it.
            analysis_particles = inflator(analysis_particles, forecast, step=step)
        analyses.append(analysis_particles)
        logps.append(result.log_likelihood)

        particles = analysis_particles
        t_prev = t_now

    log_arr: Float[Array, " T"] | None
    if all(lp is None for lp in logps):
        log_arr = None
    else:
        # Pad missing entries with NaN so the array stays length T.
        log_arr = jnp.stack(
            [jnp.asarray(jnp.nan) if lp is None else lp for lp in logps]
        )

    empty = jnp.empty((0, *init_ensemble.shape))
    return AssimilationResult(
        particles=particles,
        forecast_history=jnp.stack(forecasts) if forecasts else empty,
        analysis_history=jnp.stack(analyses) if analyses else empty,
        log_likelihoods=log_arr,
    )


class ETKF(eqx.Module, strict=True):
    r"""Ensemble Transform Kalman Filter — full forecast-analysis loop.

    Composes ``dynamics → L1 ETKF analysis → optional inflator`` over
    each observation window. Localization is not applied at this level;
    use :class:`LETKF` for localized assimilation.

    Attributes:
        dynamics: Forward model applied with :func:`eqx.filter_vmap` over
            the ensemble.
        obs_op: Observation operator ``H``.
        inflator: Optional posterior inflator (e.g.
            :class:`filterax.RTPS`, :class:`filterax.MultiplicativeInflator`).
        config: Optional :class:`FilterConfig` reserved for future
            static configuration (ensemble size, diagnostics toggle, …).

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from filterax import ETKF
        >>> model = ETKF(dynamics=lambda x, t0, t1: x, obs_op=lambda x: x)
        >>> init = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        >>> obs = [(jnp.array([0.5, 0.5]), 1.0), (jnp.array([0.6, 0.4]), 2.0)]
        >>> R = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> result = model.assimilate(init, obs, R)
        >>> result.particles.shape, result.analysis_history.shape
        ((3, 2), (2, 3, 2))
    """

    dynamics: AbstractDynamics
    obs_op: AbstractObsOperator
    inflator: AbstractInflator | None = None
    config: FilterConfig | None = None

    def assimilate(
        self,
        init_ensemble: Float[Array, "N_e N_x"],
        observations: Sequence[tuple[Float[Array, " N_y"], float | Float[Array, ""]]],
        obs_noise: lx.AbstractLinearOperator,
        t0: float | Float[Array, ""] = 0.0,
    ) -> AssimilationResult:
        """Run forecast → ETKF analysis → optional inflation over each window."""
        return _run_loop(
            lambda _step: _ETKFFilter(),
            self.dynamics,
            self.obs_op,
            self.inflator,
            init_ensemble,
            observations,
            obs_noise,
            t0,
        )


class EnSRF(eqx.Module, strict=True):
    """Ensemble Square Root Filter — full forecast-analysis loop."""

    dynamics: AbstractDynamics
    obs_op: AbstractObsOperator
    inflator: AbstractInflator | None = None
    config: FilterConfig | None = None

    def assimilate(
        self,
        init_ensemble: Float[Array, "N_e N_x"],
        observations: Sequence[tuple[Float[Array, " N_y"], float | Float[Array, ""]]],
        obs_noise: lx.AbstractLinearOperator,
        t0: float | Float[Array, ""] = 0.0,
    ) -> AssimilationResult:
        return _run_loop(
            lambda _step: _EnSRFFilter(),
            self.dynamics,
            self.obs_op,
            self.inflator,
            init_ensemble,
            observations,
            obs_noise,
            t0,
        )


class StochasticEnKF(eqx.Module, strict=True):
    r"""Stochastic Ensemble Kalman Filter — full forecast-analysis loop.

    The PRNG key for window ``step`` is derived as
    ``jr.fold_in(base_key, step)`` so successive windows draw
    independent observation perturbations even when called inside the
    same assimilation. Pass ``seed`` (int) or ``base_key`` (PRNG array)
    at construction; ``base_key`` overrides ``seed`` when both are
    provided.

    Attributes:
        dynamics: Forward model.
        obs_op: Observation operator ``H``.
        inflator: Optional posterior inflator.
        seed: Integer used as ``jr.PRNGKey(seed)`` when ``base_key`` is
            not supplied.
        base_key: Explicit PRNG array. Takes precedence over ``seed``.
        config: Optional :class:`FilterConfig`.
    """

    dynamics: AbstractDynamics
    obs_op: AbstractObsOperator
    inflator: AbstractInflator | None = None
    seed: int = eqx.field(static=True, default=0)
    base_key: PRNGKeyArray | None = None
    config: FilterConfig | None = None

    def assimilate(
        self,
        init_ensemble: Float[Array, "N_e N_x"],
        observations: Sequence[tuple[Float[Array, " N_y"], float | Float[Array, ""]]],
        obs_noise: lx.AbstractLinearOperator,
        t0: float | Float[Array, ""] = 0.0,
    ) -> AssimilationResult:
        base_key = self.base_key if self.base_key is not None else jr.PRNGKey(self.seed)

        def factory(step: int) -> AbstractSequentialFilter:
            # jr.fold_in derives a fresh sub-key from (base, step) so each
            # window's perturbed observations are independent.
            return _StochasticEnKFFilter(key=jr.fold_in(base_key, step))

        return _run_loop(
            factory,
            self.dynamics,
            self.obs_op,
            self.inflator,
            init_ensemble,
            observations,
            obs_noise,
            t0,
        )


class LETKF(eqx.Module, strict=True):
    r"""Local ETKF — full forecast-analysis loop with R-localization.

    ``state_coords`` and ``obs_coords`` are passed at ``assimilate`` time
    because they describe the assimilation problem, not the filter
    configuration. ``radius`` (and the optional ``taper_fn``) are
    configured here.
    """

    dynamics: AbstractDynamics
    obs_op: AbstractObsOperator
    radius: float = eqx.field(static=True)
    inflator: AbstractInflator | None = None
    config: FilterConfig | None = None

    def assimilate(
        self,
        init_ensemble: Float[Array, "N_e N_x"],
        observations: Sequence[tuple[Float[Array, " N_y"], float | Float[Array, ""]]],
        obs_noise: lx.AbstractLinearOperator,
        state_coords: Float[Array, "N_x D"],
        obs_coords: Float[Array, "N_y D"],
        t0: float | Float[Array, ""] = 0.0,
    ) -> AssimilationResult:
        return _run_loop(
            lambda _step: _LETKFFilter(radius=self.radius),
            self.dynamics,
            self.obs_op,
            self.inflator,
            init_ensemble,
            observations,
            obs_noise,
            t0,
            analysis_extra={
                "state_coords": state_coords,
                "obs_coords": obs_coords,
            },
        )


# ──────────────────────────────────────────────────────────────────────
# Latent-space L2 wrappers (D17)
# ──────────────────────────────────────────────────────────────────────


def _coerce_latent_dynamics(d: Any) -> AbstractDynamics:
    """Auto-wrap a structural ``LatentForwardModel`` as an ``AbstractDynamics``.

    The design doc advertises both the wrapped form (any
    :class:`AbstractDynamics` — typically :class:`LatentDynamics` or
    :class:`EncodedDynamics`) and the raw structural form (any object
    with ``.step(z, dt) → z``). The L2 wrappers feed the dynamics
    into :func:`_forecast`, which calls ``dynamics(state, t0, t1)`` —
    so a raw ``.step``-only object would raise at trace time. We wrap
    here so users following either contract get the same ergonomics.
    """
    if isinstance(d, AbstractDynamics):
        return d
    return LatentDynamics(inner=d)


def _wrap_latent_result(
    latent_map: Any,
    result_z: AssimilationResult,
) -> LatentAssimilationResult:
    """Decode the z-space histories on a vanilla ``AssimilationResult``.

    Used by :class:`LatentETKF` / :class:`LatentLETKF` to surface the
    decoded ``x``-space view alongside the latent representations.
    """
    particles_x = decode_ensemble(latent_map.decode, result_z.particles)
    forecast_x = _stack_decode(latent_map.decode, result_z.forecast_history)
    analysis_x = _stack_decode(latent_map.decode, result_z.analysis_history)
    return LatentAssimilationResult(
        particles=particles_x,
        forecast_history=forecast_x,
        analysis_history=analysis_x,
        log_likelihoods=result_z.log_likelihoods,
        particles_z=result_z.particles,
        forecast_history_z=result_z.forecast_history,
        analysis_history_z=result_z.analysis_history,
    )


class LatentETKF(eqx.Module, strict=True):
    r"""ETKF analysis applied to a latent-space ensemble (D17).

    Thin composition over the existing :class:`ETKF` Layer-1 step:
    encode the initial ensemble (when supplied in ``x``-space), run
    the standard forecast → ETKF → optional inflation cycle on the
    latent ensemble using ``LiftedObs(decoder, x_obs_op)`` as the
    observation operator, then decode the histories back to
    ``x``-space alongside the latent representations.

    The ensemble math is unchanged — the wrapper exists for ergonomics
    and for compile-time correctness (it would be hard to accidentally
    feed an ``x``-space ensemble into a latent-space analysis step
    once the user builds a ``LatentETKF``).

    Attributes:
        latent_map: Object exposing ``.encode`` and ``.decode``.
            Typically a :class:`pipekit_cycle.LatentMap`-shaped autoencoder
            or :func:`identity_latent_map` for regression tests.
        dynamics: Latent-space dynamics — either an
            :class:`AbstractDynamics` (typically :class:`LatentDynamics`
            or :class:`EncodedDynamics`) or a structural
            ``LatentForwardModel`` exposing ``.step(z, dt) → z``. Raw
            ``.step``-only objects are auto-wrapped in
            :class:`LatentDynamics` at construction.
        obs_op: ``x``-space observation operator. Lifted internally
            via :class:`LiftedObs`.
        inflator: Optional deterministic inflator applied in latent
            space.
        config: Reserved :class:`FilterConfig`.
    """

    latent_map: Any
    dynamics: AbstractDynamics
    obs_op: AbstractObsOperator
    inflator: AbstractInflator | None
    config: FilterConfig | None

    def __init__(
        self,
        latent_map: Any,
        dynamics: Any,
        obs_op: AbstractObsOperator,
        inflator: AbstractInflator | None = None,
        config: FilterConfig | None = None,
    ):
        self.latent_map = latent_map
        self.dynamics = _coerce_latent_dynamics(dynamics)
        self.obs_op = obs_op
        self.inflator = inflator
        self.config = config

    def assimilate(
        self,
        init_ensemble: Float[Array, "N_e N_d"],
        observations: Sequence[tuple[Float[Array, " N_y"], float | Float[Array, ""]]],
        obs_noise: lx.AbstractLinearOperator,
        t0: float | Float[Array, ""] = 0.0,
        *,
        in_x_space: bool = True,
    ) -> LatentAssimilationResult:
        """Run forecast → ETKF analysis → optional inflation in latent space.

        Args:
            init_ensemble: Prior ensemble. Shape ``(Nₑ, Nₓ)`` when
                ``in_x_space=True`` (default); ``(Nₑ, N_z)`` when the
                caller already encoded.
            observations: Sequence of ``(y_t, t)`` pairs in
                ``x``-space coordinates.
            obs_noise: Observation error covariance ``R`` (operates in
                ``y``-space, untouched).
            t0: Starting time for the first forecast window.
            in_x_space: Encode the initial ensemble first. ``False``
                lets the caller skip the round-trip when their initial
                ensemble already lives in ``z``-space.

        Returns:
            :class:`LatentAssimilationResult` carrying both the
            latent and decoded ``x``-space ensembles.
        """
        z0 = (
            latent_ensemble(self.latent_map.encode, init_ensemble)
            if in_x_space
            else init_ensemble
        )
        result_z = _run_loop(
            lambda _step: _ETKFFilter(),
            self.dynamics,
            LiftedObs(decoder=self.latent_map, inner=self.obs_op),
            self.inflator,
            z0,
            observations,
            obs_noise,
            t0,
        )
        return _wrap_latent_result(self.latent_map, result_z)


class LatentLETKF(eqx.Module, strict=True):
    r"""Local ETKF in latent space (D17).

    Sister of :class:`LatentETKF` over :class:`LETKF`. v0.1 implements
    option (a) of `latent_da.md` §9.2 — *no* localization in latent
    space — because the latent dimensions of a typical autoencoder are
    global modes without an intrinsic notion of locality.
    Pseudo-localization-via-mode-coords (option c) is deferred.

    For the no-localization case, this class is provided for API
    parity with :class:`LatentETKF`: it gives users a ``LatentLETKF``
    name that they can wire into pipekit alongside ``LatentETKF``
    without having to fall back to ``LatentETKF`` themselves. The
    body is identical to :class:`LatentETKF`.

    Like :class:`LatentETKF`, ``dynamics`` may be either an
    :class:`AbstractDynamics` or a raw structural ``LatentForwardModel``
    (``.step(z, dt) → z``); the latter is auto-wrapped in
    :class:`LatentDynamics` at construction.
    """

    latent_map: Any
    dynamics: AbstractDynamics
    obs_op: AbstractObsOperator
    inflator: AbstractInflator | None
    config: FilterConfig | None

    def __init__(
        self,
        latent_map: Any,
        dynamics: Any,
        obs_op: AbstractObsOperator,
        inflator: AbstractInflator | None = None,
        config: FilterConfig | None = None,
    ):
        self.latent_map = latent_map
        self.dynamics = _coerce_latent_dynamics(dynamics)
        self.obs_op = obs_op
        self.inflator = inflator
        self.config = config

    def assimilate(
        self,
        init_ensemble: Float[Array, "N_e N_d"],
        observations: Sequence[tuple[Float[Array, " N_y"], float | Float[Array, ""]]],
        obs_noise: lx.AbstractLinearOperator,
        t0: float | Float[Array, ""] = 0.0,
        *,
        in_x_space: bool = True,
    ) -> LatentAssimilationResult:
        """Run forecast → ETKF analysis (no localization) → inflation in z-space."""
        z0 = (
            latent_ensemble(self.latent_map.encode, init_ensemble)
            if in_x_space
            else init_ensemble
        )
        result_z = _run_loop(
            lambda _step: _ETKFFilter(),
            self.dynamics,
            LiftedObs(decoder=self.latent_map, inner=self.obs_op),
            self.inflator,
            z0,
            observations,
            obs_noise,
            t0,
        )
        return _wrap_latent_result(self.latent_map, result_z)

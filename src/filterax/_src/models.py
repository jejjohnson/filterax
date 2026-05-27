"""Layer-2 high-level assimilation models.

Compose a forecast (via ``eqx.filter_vmap`` over an ``AbstractDynamics``),
an analysis step (L1 ``AbstractSequentialFilter``), and an optional
inflator into a single ``assimilate`` call that loops over a sequence of
``(obs_values, obs_time)`` windows.

The L2 models are deliberately thin Python wrappers around the L1
components so users can also drop down to ``filter.analysis(...)`` and
roll their own loop when the standard cycle does not fit.
"""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from filterax._src._protocols import (
    AbstractDynamics,
    AbstractInflator,
    AbstractObsOperator,
    AbstractSequentialFilter,
)
from filterax._src._types import AssimilationResult, FilterConfig
from filterax._src.sequential import (
    ETKF as _ETKFFilter,
    LETKF as _LETKFFilter,
    EnSRF as _EnSRFFilter,
    StochasticEnKF as _StochasticEnKFFilter,
)


def _forecast(
    dynamics: AbstractDynamics,
    particles: Float[Array, "N_e N_x"],
    t0: Float[Array, ""],
    t1: Float[Array, ""],
) -> Float[Array, "N_e N_x"]:
    """Apply ``dynamics`` to every member of an ensemble."""
    return eqx.filter_vmap(lambda x: dynamics(x, t0, t1))(particles)


def _run_loop(
    filter_: AbstractSequentialFilter,
    dynamics: AbstractDynamics,
    obs_op: AbstractObsOperator,
    inflator: AbstractInflator | None,
    init_ensemble: Float[Array, "N_e N_x"],
    observations: Sequence[tuple[Float[Array, " N_y"], float | Float[Array, ""]]],
    obs_noise: lx.AbstractLinearOperator,
    t0: float | Float[Array, ""],
    analysis_extra: dict | None = None,
) -> AssimilationResult:
    """Generic forecast-analysis-inflate loop shared by every L2 model."""
    extra = analysis_extra or {}
    particles = init_ensemble

    forecasts: list[Float[Array, "N_e N_x"]] = []
    analyses: list[Float[Array, "N_e N_x"]] = []
    logps: list[Float[Array, ""]] = []

    t_prev = jnp.asarray(t0)
    for obs_values, obs_time in observations:
        t_now = jnp.asarray(obs_time)
        forecast = _forecast(dynamics, particles, t_prev, t_now)
        forecasts.append(forecast)

        result = filter_.analysis(forecast, obs_values, obs_op, obs_noise, **extra)
        analysis_particles = result.particles
        if inflator is not None:
            analysis_particles = inflator(analysis_particles, forecast)
        analyses.append(analysis_particles)
        if result.log_likelihood is not None:
            logps.append(result.log_likelihood)
        particles = analysis_particles
        t_prev = t_now

    log_arr = jnp.stack(logps) if logps else None
    return AssimilationResult(
        particles=particles,
        forecast_history=jnp.stack(forecasts)
        if forecasts
        else jnp.empty((0, *init_ensemble.shape)),
        analysis_history=jnp.stack(analyses)
        if analyses
        else jnp.empty((0, *init_ensemble.shape)),
        log_likelihoods=log_arr,
    )


class ETKF(eqx.Module, strict=True):
    """Ensemble Transform Kalman Filter — full forecast-analysis loop.

    Composes ``dynamics → L1 ETKF analysis → optional inflator`` over each
    observation window. Localization is not applied at this level; use
    :class:`LETKF` for localized assimilation.
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
        """Run the full assimilation cycle over a sequence of windows."""
        return _run_loop(
            _ETKFFilter(),
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
            _EnSRFFilter(),
            self.dynamics,
            self.obs_op,
            self.inflator,
            init_ensemble,
            observations,
            obs_noise,
            t0,
        )


class StochasticEnKF(eqx.Module, strict=True):
    """Stochastic Ensemble Kalman Filter — full forecast-analysis loop.

    The PRNG key supplied at construction is used to draw observation
    perturbations on every analysis step; callers wanting per-step
    independence should pre-split the key.
    """

    dynamics: AbstractDynamics
    obs_op: AbstractObsOperator
    inflator: AbstractInflator | None = None
    seed: int = eqx.field(static=True, default=0)
    config: FilterConfig | None = None

    def assimilate(
        self,
        init_ensemble: Float[Array, "N_e N_x"],
        observations: Sequence[tuple[Float[Array, " N_y"], float | Float[Array, ""]]],
        obs_noise: lx.AbstractLinearOperator,
        t0: float | Float[Array, ""] = 0.0,
    ) -> AssimilationResult:
        return _run_loop(
            _StochasticEnKFFilter(key=self.seed),
            self.dynamics,
            self.obs_op,
            self.inflator,
            init_ensemble,
            observations,
            obs_noise,
            t0,
        )


class LETKF(eqx.Module, strict=True):
    """Local ETKF — full forecast-analysis loop with R-localization.

    ``state_coords`` and ``obs_coords`` are passed at ``assimilate`` time
    because they describe the assimilation problem, not the filter
    configuration. ``radius`` and the taper function are configured here.
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
            _LETKFFilter(radius=self.radius),
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

"""Layer-2 assimilation models: forecast, analysis and inflation over many windows.

Each model composes a forecast, the dynamics f mapped over the members, an analysis
step from `filterax._filters._sequential` and an optional inflator into one
``assimilate`` call over a sequence of ``(y_t, t)`` windows,

    x⁽ᵉ⁾_t|t−1 = f(x⁽ᵉ⁾_t−1|t−1),   x⁽ᵉ⁾_t|t = analysis(x⁽ᵉ⁾_t|t−1, y_t)

The models are thin Python loops, so ``filter_.analysis(...)`` is always there to
write your own loop when this cycle does not fit. The `AssimilationResult` they
return stacks the forecasts, analyses and log-likelihoods along a leading time axis
of length T, the number of windows. The log-likelihoods, when present, always have
length T, with NaN for a window whose filter reports none.
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
    """Forecast every member from ``t0`` to ``t1``, x⁽ᵉ⁾_t|t−1 = f(x⁽ᵉ⁾_t−1|t−1).

    Args:
        dynamics: Forward model f, applied to one state.
        particles: Analysis ensemble x⁽ᵉ⁾_t−1|t−1, one row per member.
        t0: Start time of the window.
        t1: End time of the window, the time of y_t.

    Returns:
        The forecast ensemble x⁽ᵉ⁾_t|t−1.
    """
    # x⁽ᵉ⁾_t|t−1 = f(x⁽ᵉ⁾_t−1|t−1)
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
    """Forecast, analysis and optional inflation over each window, for every model.

    ``filter_factory(step)`` builds the analysis step of window ``step``: a stochastic
    filter folds a fresh key per window, and a deterministic one ignores the step.
    The log-likelihoods keep length T, with NaN where a filter reports none, so that
    their time axis stays aligned with the histories.

    Algorithm:
        1. Forecast the members from the previous time to the window's time.
        2. Run the window's analysis step on the forecast.
        3. Inflate the analysis, passing the window as ``step=``.
        4. Stack the forecasts, analyses and log-likelihoods over the windows.

    Args:
        filter_factory: Analysis step of each window, from its index.
        dynamics: Forward model f, applied to one state.
        obs_op: Observation operator h, applied to one state.
        inflator: Inflator applied after each analysis, or ``None``.
        init_ensemble: Initial ensemble x⁽ᵉ⁾_0|0, one row per member.
        observations: Windows ``(y_t, t)``, in time order.
        obs_noise: Observation-error covariance Σ_ε, shared by the windows.
        t0: Time of the initial ensemble.
        analysis_extra: Extra keywords for every analysis, such as LETKF's
            coordinates.

    Returns:
        The final analysis and the stacked histories.
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
            # step= lets AdditiveInflator fold a fresh key per window; the
            # deterministic inflators swallow it in **_
            analysis_particles = inflator(analysis_particles, forecast, step=step)
        analyses.append(analysis_particles)
        logps.append(result.log_likelihood)

        particles = analysis_particles
        t_prev = t_now

    log_arr: Float[Array, " T"] | None
    if all(lp is None for lp in logps):
        log_arr = None
    else:
        # NaN where a window has no log-likelihood, so the array keeps length T
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
    """Ensemble transform Kalman filter over many windows.

        x⁽ᵉ⁾_t|t−1 = f(x⁽ᵉ⁾_t−1|t−1),   x̄_t|t = x̄_t|t−1 + X̃ᵀ w̄,   X̃_t|t = W X̃

    Each window forecasts with ``dynamics``, runs the analysis of
    `filterax.filters.ETKF` and applies the optional inflator. There is no
    localization; `LETKF` adds it.

    Attributes:
        dynamics: Forward model f, applied to one state and mapped over the members.
        obs_op: Observation operator h, applied to one state.
        inflator: Inflator applied after each analysis, such as `filterax.RTPS` or
            `filterax.MultiplicativeInflator`; ``None`` skips it.
        config: `FilterConfig`, reserved for static configuration and unused here.

    Examples:
        Identity dynamics, both components observed in two windows:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from filterax import ETKF
        >>> model = ETKF(dynamics=lambda x, t0, t1: x, obs_op=lambda x: x)
        >>> init = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        >>> obs = [(jnp.array([0.5, 0.5]), 1.0), (jnp.array([0.6, 0.4]), 2.0)]
        >>> obs_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> result = model.assimilate(init, obs, obs_noise)
        >>> result.particles.shape, result.analysis_history.shape
        ((3, 2), (2, 3, 2))
        >>> result.log_likelihoods.shape
        (2,)

        With inflation, the final spread is larger:

        >>> import filterax
        >>> inflated = ETKF(
        ...     dynamics=lambda x, t0, t1: x,
        ...     obs_op=lambda x: x,
        ...     inflator=filterax.MultiplicativeInflator(factor=1.5),
        ... ).assimilate(init, obs, obs_noise)
        >>> bool(jnp.all(inflated.particles.std(0) > result.particles.std(0)))
        True
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
        """Run forecast, ETKF analysis and optional inflation over each window.

        Args:
            init_ensemble: Initial ensemble x⁽ᵉ⁾_0|0, one row per member.
            observations: Windows ``(y_t, t)``, in time order.
            obs_noise: Observation-error covariance Σ_ε, shared by the windows.
            t0: Time of the initial ensemble.

        Returns:
            The final analysis, the stacked forecasts and analyses, and the
            log-likelihood of each window.
        """
        # x⁽ᵉ⁾_t|t from x⁽ᵉ⁾_t|t−1 and y_t, window by window
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
    """Ensemble square-root filter over many windows.

        x⁽ᵉ⁾_t|t−1 = f(x⁽ᵉ⁾_t−1|t−1),   x̄_t|t = x̄_t|t−1 + G r̄,   X̃_t|t = W X̃

    Each window forecasts with ``dynamics``, runs the analysis of
    `filterax.filters.EnSRF`, which is the ETKF update, and applies the optional
    inflator.

    Attributes:
        dynamics: Forward model f, applied to one state and mapped over the members.
        obs_op: Observation operator h, applied to one state.
        inflator: Inflator applied after each analysis; ``None`` skips it.
        config: `FilterConfig`, reserved for static configuration and unused here.

    Examples:
        The same cycle as `ETKF`:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from filterax import ETKF, EnSRF
        >>> init = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        >>> obs = [(jnp.array([0.5, 0.5]), 1.0), (jnp.array([0.6, 0.4]), 2.0)]
        >>> obs_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> result = EnSRF(
        ...     dynamics=lambda x, t0, t1: x, obs_op=lambda x: x
        ... ).assimilate(init, obs, obs_noise)
        >>> result.forecast_history.shape
        (2, 3, 2)
        >>> etkf = ETKF(
        ...     dynamics=lambda x, t0, t1: x, obs_op=lambda x: x
        ... ).assimilate(init, obs, obs_noise)
        >>> bool(jnp.array_equal(result.particles, etkf.particles))
        True
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
        """Run forecast, EnSRF analysis and optional inflation over each window.

        Args:
            init_ensemble: Initial ensemble x⁽ᵉ⁾_0|0, one row per member.
            observations: Windows ``(y_t, t)``, in time order.
            obs_noise: Observation-error covariance Σ_ε, shared by the windows.
            t0: Time of the initial ensemble.

        Returns:
            The final analysis, the stacked forecasts and analyses, and the
            log-likelihood of each window.
        """
        # x⁽ᵉ⁾_t|t from x⁽ᵉ⁾_t|t−1 and y_t, window by window
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
    """Stochastic ensemble Kalman filter over many windows.

        x⁽ᵉ⁾_t|t−1 = f(x⁽ᵉ⁾_t−1|t−1)
        x⁽ᵉ⁾_t|t = x⁽ᵉ⁾_t|t−1 + G (y_t + ε⁽ᵉ⁾_t − h⁽ᵉ⁾),   ε⁽ᵉ⁾_t ~ 𝒩(0, Σ_ε)

    Each window forecasts with ``dynamics``, runs the analysis of
    `filterax.filters.StochasticEnKF` and applies the optional inflator. The key of
    window ``step`` is ``jr.fold_in(base_key, step)``, so the windows draw independent
    perturbations, and the same model and data repeat the same run.

    Attributes:
        dynamics: Forward model f, applied to one state and mapped over the members.
        obs_op: Observation operator h, applied to one state.
        inflator: Inflator applied after each analysis; ``None`` skips it.
        seed: Seed of ``jr.PRNGKey(seed)``, used when ``base_key`` is ``None``.
        base_key: PRNG key the window keys are folded from; it takes precedence over
            ``seed``.
        config: `FilterConfig`, reserved for static configuration and unused here.

    Examples:
        The same seed repeats the run:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from filterax import StochasticEnKF
        >>> init = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        >>> obs = [(jnp.array([0.5, 0.5]), 1.0), (jnp.array([0.6, 0.4]), 2.0)]
        >>> obs_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> model = StochasticEnKF(
        ...     dynamics=lambda x, t0, t1: x, obs_op=lambda x: x, seed=3
        ... )
        >>> first = model.assimilate(init, obs, obs_noise)
        >>> first.analysis_history.shape
        (2, 3, 2)
        >>> second = model.assimilate(init, obs, obs_noise)
        >>> bool(jnp.array_equal(first.particles, second.particles))
        True
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
        """Run forecast, stochastic analysis and optional inflation over each window.

        Args:
            init_ensemble: Initial ensemble x⁽ᵉ⁾_0|0, one row per member.
            observations: Windows ``(y_t, t)``, in time order.
            obs_noise: Observation-error covariance Σ_ε, shared by the windows.
            t0: Time of the initial ensemble.

        Returns:
            The final analysis, the stacked forecasts and analyses, and the
            log-likelihood of each window.
        """
        base_key = self.base_key if self.base_key is not None else jr.PRNGKey(self.seed)

        def factory(step: int) -> AbstractSequentialFilter:
            # one key per window, so the windows' perturbations are independent
            return _StochasticEnKFFilter(key=jr.fold_in(base_key, step))

        # x⁽ᵉ⁾_t|t from x⁽ᵉ⁾_t|t−1 and y_t, window by window
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
    """Local ensemble transform Kalman filter over many windows.

        x⁽ᵉ⁾_t|t−1 = f(x⁽ᵉ⁾_t−1|t−1)
        x̄_t|t,i = x̄_t|t−1,i + X̃_iᵀ w̄_i,   X̃_t|t,i = W_i X̃_i

    Each window forecasts with ``dynamics``, runs the analysis of
    `filterax.filters.LETKF` with the half-width ℓ and the default Gaspari-Cohn taper,
    and applies the optional inflator. The coordinates go to ``assimilate`` rather
    than to the model, because they describe the problem rather than the filter.

    Attributes:
        dynamics: Forward model f, applied to one state and mapped over the members.
        obs_op: Observation operator h, applied to one state.
        radius: Localization half-width ℓ; observations farther than ℓ are dropped.
        inflator: Inflator applied after each analysis; ``None`` skips it.
        config: `FilterConfig`, reserved for static configuration and unused here.

    Examples:
        Three grid points on a line, the outer two observed:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from filterax import LETKF
        >>> model = LETKF(
        ...     dynamics=lambda x, t0, t1: x, obs_op=lambda x: x[::2], radius=1.5
        ... )
        >>> init = jax.random.normal(jax.random.key(0), (4, 3))
        >>> obs = [(jnp.array([0.1, -0.2]), 1.0), (jnp.array([0.0, 0.1]), 2.0)]
        >>> obs_noise = lx.DiagonalLinearOperator(0.5 * jnp.ones(2))
        >>> state_coords = jnp.array([[0.0], [1.0], [2.0]])
        >>> obs_coords = jnp.array([[0.0], [2.0]])
        >>> result = model.assimilate(init, obs, obs_noise, state_coords, obs_coords)
        >>> result.analysis_history.shape
        (2, 4, 3)
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
        """Run forecast, LETKF analysis and optional inflation over each window.

        Args:
            init_ensemble: Initial ensemble x⁽ᵉ⁾_0|0, one row per member.
            observations: Windows ``(y_t, t)``, in time order.
            obs_noise: Observation-error covariance Σ_ε, shared by the windows.
            state_coords: Coordinates s_i of the state components, one row each.
            obs_coords: Coordinates s_k of the observations, one row each.
            t0: Time of the initial ensemble.

        Returns:
            The final analysis, the stacked forecasts and analyses, and the
            log-likelihood of each window.
        """
        # x⁽ᵉ⁾_t|t from x⁽ᵉ⁾_t|t−1 and y_t, window by window
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


# --- latent-space models ---


def _coerce_latent_dynamics(d: Any) -> AbstractDynamics:
    """Wrap a raw latent forward model, with only ``.step(z, dt)``, as dynamics.

    The latent models take either an `AbstractDynamics`, typically `LatentDynamics`
    or `EncodedDynamics`, or a raw object with ``.step(z, dt)``. `_forecast` calls
    ``dynamics(state, t0, t1)``, which a raw object does not have, so it is wrapped
    here and both forms behave the same.

    Args:
        d: Latent dynamics, an `AbstractDynamics` or an object with ``.step``.

    Returns:
        ``d`` itself, or ``LatentDynamics(inner=d)``.
    """
    if isinstance(d, AbstractDynamics):
        return d
    return LatentDynamics(inner=d)


def _wrap_latent_result(
    latent_map: Any,
    result_z: AssimilationResult,
) -> LatentAssimilationResult:
    """Decode a latent result to state space, keeping the latent ensembles beside it.

        x⁽ᵉ⁾ = g_dec(z⁽ᵉ⁾)

    Args:
        latent_map: Codec with ``.decode``, the decoder g_dec.
        result_z: The result of the latent loop.

    Returns:
        The decoded ensembles and histories, with the latent ones alongside.
    """
    # x⁽ᵉ⁾ = g_dec(z⁽ᵉ⁾)
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
    """ETKF over many windows on a latent ensemble.

        z⁽ᵉ⁾_0|0 = g_enc(x⁽ᵉ⁾_0|0),   h_z = h ∘ g_dec,   x⁽ᵉ⁾_t|t = g_dec(z⁽ᵉ⁾_t|t)

    Encodes the initial ensemble, runs the cycle of `ETKF` on the latent members z⁽ᵉ⁾
    with the observation operator h ∘ g_dec, a `LiftedObs`, and decodes the histories
    back to state space beside the latent ones. The ensemble algebra is unchanged, and
    the class keeps a state-space ensemble out of a latent analysis by construction.

    Attributes:
        latent_map: Codec with ``.encode`` and ``.decode``, the encoder g_enc and the
            decoder g_dec, such as a pipekit ``LatentMap`` autoencoder or
            `identity_latent_map`.
        dynamics: Latent dynamics, an `AbstractDynamics` such as `LatentDynamics` or
            `EncodedDynamics`, or a raw object with ``.step(z, dt)``, which is wrapped
            in `LatentDynamics` at construction.
        obs_op: Observation operator h on the state, lifted to h ∘ g_dec.
        inflator: Deterministic inflator applied in latent space after each analysis;
            ``None`` skips it.
        config: `FilterConfig`, reserved for static configuration and unused here.

    Examples:
        The identity codec reproduces the state-space `ETKF`:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import filterax
        >>> init = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        >>> obs = [(jnp.array([0.5, 0.5]), 1.0), (jnp.array([0.6, 0.4]), 2.0)]
        >>> obs_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> class Persistence:
        ...     def step(self, z, dt):
        ...         return z
        >>> model = filterax.LatentETKF(
        ...     filterax.identity_latent_map(2), Persistence(), lambda x: x
        ... )
        >>> result = model.assimilate(init, obs, obs_noise)
        >>> result.particles_z.shape, result.analysis_history.shape
        ((3, 2), (2, 3, 2))
        >>> etkf = filterax.ETKF(
        ...     dynamics=lambda x, t0, t1: x, obs_op=lambda x: x
        ... ).assimilate(init, obs, obs_noise)
        >>> bool(jnp.allclose(result.particles, etkf.particles))
        True
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
        """Run forecast, ETKF analysis and optional inflation in latent space.

        Algorithm:
            1. Encode the initial ensemble, z⁽ᵉ⁾_0|0 = g_enc(x⁽ᵉ⁾_0|0), unless it is
               latent already.
            2. Run the ETKF cycle on the latent members, observing them through
               h ∘ g_dec.
            3. Decode the final ensemble and the histories with g_dec.

        Args:
            init_ensemble: Initial ensemble, one row per member: x⁽ᵉ⁾_0|0 of length
                D_x, or z⁽ᵉ⁾_0|0 of length D_z when ``in_x_space=False``.
            observations: Windows ``(y_t, t)``, in time order.
            obs_noise: Observation-error covariance Σ_ε, shared by the windows.
            t0: Time of the initial ensemble.
            in_x_space: Whether ``init_ensemble`` is in state space and needs
                encoding.

        Returns:
            The latent ensembles and histories, and their decoded state-space views.
        """
        # z⁽ᵉ⁾_0|0 = g_enc(x⁽ᵉ⁾_0|0)
        z0 = (
            latent_ensemble(self.latent_map.encode, init_ensemble)
            if in_x_space
            else init_ensemble
        )
        # z⁽ᵉ⁾_t|t from z⁽ᵉ⁾_t|t−1 and y_t, observed through h ∘ g_dec
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
    """Local ETKF in latent space, which today runs `LatentETKF` unlocalized.

        z⁽ᵉ⁾_0|0 = g_enc(x⁽ᵉ⁾_0|0),   h_z = h ∘ g_dec,   x⁽ᵉ⁾_t|t = g_dec(z⁽ᵉ⁾_t|t)

    The latent axes of a typical autoencoder are global modes with no notion of place,
    so this class applies no localization, option (a) of §9.2 of the latent design
    document, and localizing by mode coordinates is deferred. It exists so that a
    pipeline can name a ``LatentLETKF`` beside a ``LatentETKF``, and its body is that
    of `LatentETKF`.

    Attributes:
        latent_map: Codec with ``.encode`` and ``.decode``, the encoder g_enc and the
            decoder g_dec.
        dynamics: Latent dynamics, an `AbstractDynamics`, or a raw object with
            ``.step(z, dt)``, which is wrapped in `LatentDynamics` at construction.
        obs_op: Observation operator h on the state, lifted to h ∘ g_dec.
        inflator: Deterministic inflator applied in latent space after each analysis;
            ``None`` skips it.
        config: `FilterConfig`, reserved for static configuration and unused here.

    Examples:
        The same run as `LatentETKF`:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import filterax
        >>> init = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        >>> obs = [(jnp.array([0.5, 0.5]), 1.0)]
        >>> obs_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> codec = filterax.identity_latent_map(2)
        >>> class Persistence:
        ...     def step(self, z, dt):
        ...         return z
        >>> local = filterax.LatentLETKF(codec, Persistence(), lambda x: x)
        >>> result = local.assimilate(init, obs, obs_noise)
        >>> plain = filterax.LatentETKF(codec, Persistence(), lambda x: x)
        >>> other = plain.assimilate(init, obs, obs_noise)
        >>> bool(jnp.array_equal(result.particles_z, other.particles_z))
        True
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
        """Run forecast, unlocalized ETKF analysis and optional inflation on z⁽ᵉ⁾.

        Args:
            init_ensemble: Initial ensemble, one row per member: x⁽ᵉ⁾_0|0 of length
                D_x, or z⁽ᵉ⁾_0|0 of length D_z when ``in_x_space=False``.
            observations: Windows ``(y_t, t)``, in time order.
            obs_noise: Observation-error covariance Σ_ε, shared by the windows.
            t0: Time of the initial ensemble.
            in_x_space: Whether ``init_ensemble`` is in state space and needs
                encoding.

        Returns:
            The latent ensembles and histories, and their decoded state-space views.
        """
        # z⁽ᵉ⁾_0|0 = g_enc(x⁽ᵉ⁾_0|0)
        z0 = (
            latent_ensemble(self.latent_map.encode, init_ensemble)
            if in_x_space
            else init_ensemble
        )
        # z⁽ᵉ⁾_t|t from z⁽ᵉ⁾_t|t−1 and y_t, observed through h ∘ g_dec
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

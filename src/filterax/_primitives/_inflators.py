"""Inflators: the inflation primitives as modules for the assimilation loops.

`AdditiveInflator` is the only stochastic one, and it derives a key per window from
``step``, which the run loops pass through ``**kwargs``. The deterministic inflators
ignore ``**kwargs``.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray

from filterax._primitives._inflation import (
    inflate_additive,
    inflate_multiplicative,
    inflate_rtpp,
    inflate_rtps,
)
from filterax._protocols import AbstractInflator


class MultiplicativeInflator(AbstractInflator, strict=True):
    """Multiplicative inflation about the sample mean, as a module.

        x⁽ᵉ⁾ ← x̄ + ζ (x⁽ᵉ⁾ − x̄)

    Wraps `inflate_multiplicative`. It ignores ``forecast_particles`` and any extra
    keywords.

    Attributes:
        factor: Inflation factor ζ > 0; values in [1.01, 1.10] are typical.

    Examples:
        Doubling the spread keeps the mean:

        >>> import jax.numpy as jnp
        >>> import filterax
        >>> inflator = filterax.MultiplicativeInflator(factor=2.0)
        >>> inflator(jnp.array([[0.0], [2.0]])).tolist()
        [[-1.0], [3.0]]

        A forecast and the loop's ``step`` are ignored:

        >>> particles = jnp.array([[0.0], [2.0]])
        >>> inflator(particles, particles, step=3).tolist()
        [[-1.0], [3.0]]
    """

    factor: float = eqx.field(static=True)

    def __call__(
        self,
        particles: Float[Array, "N_e N_x"],
        forecast_particles: Float[Array, "N_e N_x"] | None = None,
        **_: Any,
    ) -> Float[Array, "N_e N_x"]:
        del forecast_particles
        # x⁽ᵉ⁾ ← x̄ + ζ (x⁽ᵉ⁾ − x̄)
        return inflate_multiplicative(particles, self.factor)


class RTPS(AbstractInflator, strict=True):
    """Relaxation to prior spread (RTPS), as a module.

        x̃⁽ᵉ⁾_t|t ← x̃⁽ᵉ⁾_t|t ((1 − α) + α σ_x,t|t−1 / σ_x,t|t)

    Wraps `inflate_rtps`. It needs the forecast ensemble rather than silently doing
    nothing without it.

    Attributes:
        alpha: Relaxation coefficient α ∈ [0, 1]. 0 keeps the analysis spread, and 1
            restores the forecast spread.

    Raises:
        ValueError: If ``forecast_particles`` is missing.

    Examples:
        A coefficient of one restores the forecast spread:

        >>> import jax.numpy as jnp
        >>> import filterax
        >>> inflator = filterax.RTPS(alpha=1.0)
        >>> forecast = jnp.array([[-2.0], [2.0]])
        >>> analysis = jnp.array([[-1.0], [1.0]])
        >>> bool(jnp.allclose(inflator(analysis, forecast), forecast))
        True

        The forecast ensemble is required:

        >>> inflator(analysis)
        Traceback (most recent call last):
        ...
        ValueError: RTPS requires the forecast ensemble.
    """

    alpha: float = eqx.field(static=True)

    def __call__(
        self,
        particles: Float[Array, "N_e N_x"],
        forecast_particles: Float[Array, "N_e N_x"] | None = None,
        **_: Any,
    ) -> Float[Array, "N_e N_x"]:
        if forecast_particles is None:
            raise ValueError("RTPS requires the forecast ensemble.")
        # x̃⁽ᵉ⁾_t|t ← x̃⁽ᵉ⁾_t|t ((1 − α) + α σ_x,t|t−1 / σ_x,t|t)
        return inflate_rtps(particles, forecast_particles, self.alpha)


class RTPP(AbstractInflator, strict=True):
    """Relaxation to prior perturbations (RTPP), as a module.

        x̃⁽ᵉ⁾_t|t ← (1 − α) x̃⁽ᵉ⁾_t|t + α x̃⁽ᵉ⁾_t|t−1

    Wraps `inflate_rtpp`. It needs the forecast ensemble.

    Attributes:
        alpha: Relaxation coefficient α ∈ [0, 1].

    Raises:
        ValueError: If ``forecast_particles`` is missing.

    Examples:
        Halfway back to the forecast anomalies:

        >>> import jax.numpy as jnp
        >>> import filterax
        >>> inflator = filterax.RTPP(alpha=0.5)
        >>> forecast = jnp.array([[-2.0], [2.0]])
        >>> analysis = jnp.array([[-1.0], [1.0]])
        >>> inflator(analysis, forecast).tolist()
        [[-1.5], [1.5]]

        The forecast ensemble is required:

        >>> inflator(analysis)
        Traceback (most recent call last):
        ...
        ValueError: RTPP requires the forecast ensemble.
    """

    alpha: float = eqx.field(static=True)

    def __call__(
        self,
        particles: Float[Array, "N_e N_x"],
        forecast_particles: Float[Array, "N_e N_x"] | None = None,
        **_: Any,
    ) -> Float[Array, "N_e N_x"]:
        if forecast_particles is None:
            raise ValueError("RTPP requires the forecast ensemble.")
        # x̃⁽ᵉ⁾_t|t ← (1 − α) x̃⁽ᵉ⁾_t|t + α x̃⁽ᵉ⁾_t|t−1
        return inflate_rtpp(particles, forecast_particles, self.alpha)


class AdditiveInflator(AbstractInflator, strict=True):
    """Additive Gaussian inflation with a fresh key per window, as a module.

        x⁽ᵉ⁾ ← x⁽ᵉ⁾ + η⁽ᵉ⁾ − η̄,   η⁽ᵉ⁾ ~ 𝒩(0, Σ_η)

    Wraps `inflate_additive` and adds the choice of key. Inside the run loops
    (`filterax.ETKF`, `filterax.LETKF`, …) the loop passes ``step=``, so successive
    windows draw independent noise from ``jr.fold_in(base_key, step)``. In your own
    loop, pass ``step=`` yourself, or an explicit ``key=``, which takes precedence.
    Calling with neither reuses ``base_key`` and yields the same draw every time,
    which is fine once but a trap across windows.

    Attributes:
        noise_cov: Model-error covariance Σ_η.
        base_key: PRNG key, folded with the per-call ``step``, or used as it is when
            neither ``step`` nor ``key`` is given.

    Examples:
        The noise is centred, so the ensemble mean stays where it was:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import filterax
        >>> inflator = filterax.AdditiveInflator(
        ...     noise_cov=lx.DiagonalLinearOperator(0.1 * jnp.ones(2)),
        ...     base_key=jr.key(0),
        ... )
        >>> first = inflator(jnp.zeros((4, 2)), step=0)
        >>> first.shape
        (4, 2)
        >>> bool(jnp.allclose(filterax.ensemble_mean(first), 0.0, atol=1e-6))
        True

        Another window draws other noise, and the same window repeats its draw:

        >>> second = inflator(jnp.zeros((4, 2)), step=1)
        >>> bool(jnp.allclose(first, second))
        False
        >>> bool(jnp.array_equal(first, inflator(jnp.zeros((4, 2)), step=0)))
        True
    """

    noise_cov: lx.AbstractLinearOperator
    base_key: PRNGKeyArray

    def __call__(
        self,
        particles: Float[Array, "N_e N_x"],
        forecast_particles: Float[Array, "N_e N_x"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
        step: int | jnp.ndarray | None = None,
        **_: Any,
    ) -> Float[Array, "N_e N_x"]:
        del forecast_particles
        if key is not None:
            call_key = key
        elif step is not None:
            call_step = jnp.asarray(step, dtype=jnp.int32)
            call_key = jr.fold_in(self.base_key, call_step)
        else:
            call_key = self.base_key
        # x⁽ᵉ⁾ ← x⁽ᵉ⁾ + η⁽ᵉ⁾ − η̄
        return inflate_additive(call_key, particles, self.noise_cov)

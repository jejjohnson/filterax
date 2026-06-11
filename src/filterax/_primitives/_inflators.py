"""Concrete inflator components.

Wrap the pure-function inflation primitives as ``AbstractInflator``
subclasses so they can be slotted into ``filterax.models`` assimilation
loops as configuration objects.

The :class:`AdditiveInflator` is the only stochastic inflator in the
set; it derives a per-cycle PRNG sub-key from ``jr.fold_in(self.base_key,
step)`` when the L2 run loop passes ``step=`` through ``**kwargs``.
The deterministic inflators (``MultiplicativeInflator``, ``RTPS``,
``RTPP``) ignore ``**kwargs`` entirely.
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
    r"""Multiplicative inflation :math:`X' \leftarrow \lambda X'`.

    Ignores ``forecast_particles`` and any extra kwargs.

    Attributes:
        factor: Inflation factor :math:`\lambda > 0`; values in
            ``[1.01, 1.10]`` are typical.
    """

    factor: float = eqx.field(static=True)

    def __call__(
        self,
        particles: Float[Array, "N_e N_x"],
        forecast_particles: Float[Array, "N_e N_x"] | None = None,
        **_: Any,
    ) -> Float[Array, "N_e N_x"]:
        del forecast_particles
        return inflate_multiplicative(particles, self.factor)


class RTPS(AbstractInflator, strict=True):
    r"""Relaxation to Prior Spread (Whitaker & Hamill 2012).

    Requires ``forecast_particles`` — raises ``ValueError`` when missing
    rather than silently no-opping.

    Attributes:
        alpha: Relaxation coefficient in ``[0, 1]``. ``0`` keeps the
            analysis spread; ``1`` restores the forecast spread.
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
        return inflate_rtps(particles, forecast_particles, self.alpha)


class RTPP(AbstractInflator, strict=True):
    r"""Relaxation to Prior Perturbations (Zhang et al. 2004).

    Requires ``forecast_particles``.

    Attributes:
        alpha: Relaxation coefficient in ``[0, 1]``.
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
        return inflate_rtpp(particles, forecast_particles, self.alpha)


class AdditiveInflator(AbstractInflator, strict=True):
    r"""Additive Gaussian inflation ``X′ ← X′ + ε, ε ~ 𝒩(0, Q_add)``.

    Wraps :func:`filterax.inflate_additive`. Per-call independence:

    * **Inside the L2 run loop** (``filterax.ETKF`` / ``LETKF`` / …):
      the loop passes ``step=`` through ``**kwargs`` and this inflator
      computes ``key = jr.fold_in(self.base_key, step)`` so successive
      assimilation windows draw independent Gaussian perturbations.
    * **In your own loop**: pass ``step=`` yourself (or pass an explicit
      ``key=`` kwarg, which takes precedence over ``base_key`` /
      ``step``). Calling without either reuses ``self.base_key`` and
      yields the *same* draw every time — fine for one-off use, but a
      footgun if you reuse the inflator across cycles.

    Attributes:
        noise_cov: Model-error covariance ``Q_add``.
        base_key: PRNG key seed; folded against the per-call ``step``
            (or used as-is when no step / key is supplied).
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
            call_key = jr.fold_in(self.base_key, jnp.asarray(step, dtype=jnp.int32))
        else:
            call_key = self.base_key
        return inflate_additive(call_key, particles, self.noise_cov)

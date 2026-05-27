"""Concrete inflator components.

Wrap the pure-function inflation primitives as ``AbstractInflator``
subclasses so they can be slotted into ``filterax.models`` assimilation
loops as configuration objects.
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float

from filterax._src._protocols import AbstractInflator
from filterax._src.inflation import (
    inflate_multiplicative,
    inflate_rtpp,
    inflate_rtps,
)


class MultiplicativeInflator(AbstractInflator, strict=True):
    r"""Multiplicative inflation :math:`X' \leftarrow \lambda X'`.

    Ignores ``forecast_particles``.

    Attributes:
        factor: Inflation factor :math:`\lambda > 0`; values in
            ``[1.01, 1.10]`` are typical.
    """

    factor: float = eqx.field(static=True)

    def __call__(
        self,
        particles: Float[Array, "N_e N_x"],
        forecast_particles: Float[Array, "N_e N_x"] | None = None,
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
    ) -> Float[Array, "N_e N_x"]:
        if forecast_particles is None:
            raise ValueError("RTPP requires the forecast ensemble.")
        return inflate_rtpp(particles, forecast_particles, self.alpha)

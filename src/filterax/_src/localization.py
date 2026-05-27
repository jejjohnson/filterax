r"""Covariance localization tapers.

Compactly-supported (Gaspari-Cohn) and infinite-support (Gaussian) taper
functions plus the Schur (Hadamard) product :func:`localize` used to apply a
taper to a covariance or gain matrix.

Localization suppresses spurious long-range correlations in the sample
covariance that arise from finite ensemble size. The localized covariance
is :math:`P_{\text{loc}} = \rho \circ P`, where :math:`\rho_{ij} = \rho(d_{ij}/r)`
is a positive-definite taper of the distance between grid points. The Schur
product of two PSD matrices is PSD (Schur product theorem), so
:math:`P_{\text{loc}}` remains a valid covariance for the Gaspari-Cohn and
Gaussian tapers.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def gaspari_cohn(
    distances: Float[Array, "..."],
    radius: float,
) -> Float[Array, "..."]:
    r"""Gaspari-Cohn 5th-order piecewise polynomial taper.

    .. math::

        \rho(z) = \begin{cases}
        -\tfrac{1}{4}z^5 + \tfrac{1}{2}z^4 + \tfrac{5}{8}z^3 - \tfrac{5}{3}z^2 + 1,
            & 0 \le z \le 1 \\
        \tfrac{1}{12}z^5 - \tfrac{1}{2}z^4 + \tfrac{5}{8}z^3 + \tfrac{5}{3}z^2
            - 5z + 4 - \tfrac{2}{3z}, & 1 < z \le 2 \\
        0, & z > 2
        \end{cases}

    where :math:`z = d / r`. Compactly supported (zero beyond ``2 * radius``),
    :math:`C^2` smooth at all transitions, and positive definite — the
    Schur product :math:`\rho \circ P` is guaranteed PSD when ``P`` is.

    Args:
        distances: Non-negative distance array, any shape.
        radius: Positive localization half-width. Compact support at
            ``2 * radius``.

    Returns:
        Taper weights in ``[0, 1]`` with the same shape as ``distances``.

    Reference:
        Gaspari, G. & Cohn, S. E. (1999). *Construction of correlation
        functions in two and three dimensions.* Q. J. R. Meteorol. Soc.,
        125, 723-757.
    """
    z = distances / radius
    # Guard the 2/(3z) branch against z = 0 — that branch is only evaluated
    # for z > 1, but jnp.where evaluates both arms unconditionally.
    z_safe = jnp.where(z > 0, z, 1.0)

    near = -0.25 * z**5 + 0.5 * z**4 + (5.0 / 8.0) * z**3 - (5.0 / 3.0) * z**2 + 1.0
    far = (
        (1.0 / 12.0) * z**5
        - 0.5 * z**4
        + (5.0 / 8.0) * z**3
        + (5.0 / 3.0) * z**2
        - 5.0 * z
        + 4.0
        - (2.0 / 3.0) / z_safe
    )
    out = jnp.where(z <= 1.0, near, jnp.where(z <= 2.0, far, 0.0))
    return jnp.where(z > 2.0, 0.0, out)


def gaussian_taper(
    distances: Float[Array, "..."],
    radius: float,
) -> Float[Array, "..."]:
    r"""Gaussian taper :math:`\rho(d) = \exp(-d^2 / (2 r^2))`.

    Infinitely smooth (:math:`C^\infty`) and positive definite, but **not**
    compactly supported — decays exponentially but never reaches zero.

    Args:
        distances: Non-negative distance array, any shape.
        radius: Positive characteristic length scale.

    Returns:
        Taper weights in ``(0, 1]`` with the same shape as ``distances``.
    """
    return jnp.exp(-(distances**2) / (2.0 * radius**2))


def hard_cutoff(
    distances: Float[Array, "..."],
    radius: float,
) -> Float[Array, "..."]:
    r"""Binary cutoff :math:`\rho(d) = \mathbf{1}\{d \le r\}`.

    Discontinuous at ``d = radius``; **not** positive definite in general.
    Useful only as a baseline / debugging primitive.

    Args:
        distances: Non-negative distance array, any shape.
        radius: Positive cutoff radius.

    Returns:
        Indicator weights (``0.0`` or ``1.0``) with the same shape as
        ``distances``.
    """
    return jnp.where(distances <= radius, 1.0, 0.0).astype(distances.dtype)


def localize(
    cov: Float[Array, "M N"],
    taper: Float[Array, "M N"],
) -> Float[Array, "M N"]:
    r"""Apply localization via Schur (element-wise) product.

    Computes :math:`P_{\text{loc}} = \rho \circ P`.

    Args:
        cov: Dense covariance or gain matrix of shape ``(M, N)``.
        taper: Taper matrix of the same shape; elementwise multipliers in
            ``[0, 1]``.

    Returns:
        Localized matrix of shape ``(M, N)``.
    """
    return cov * taper

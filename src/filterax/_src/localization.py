r"""Covariance localization tapers.

Finite-ensemble Kalman filters suffer from *spurious long-range
correlations* — a sample covariance of rank ``≤ Nₑ − 1`` assigns
non-zero correlation between physically distant variables purely from
sampling noise. The Kalman gain then draws information from distant,
uninformative observations and the filter diverges.

Localization suppresses these artefacts by tapering covariance entries
as a function of physical distance ``d``. The localized covariance is
the Schur (Hadamard / element-wise) product

``P_loc = ρ ∘ P,   ρᵢⱼ = ρ(dᵢⱼ / r)``

with ``r`` the localization half-width. By the Schur product theorem,
``ρ ∘ P`` is PSD whenever both factors are; the Gaspari-Cohn and
Gaussian tapers below are positive definite, so they preserve the
covariance structure. The hard cutoff is *not* PSD and is only useful
as a debugging baseline.

References: Gaspari & Cohn (1999); Houtekamer & Mitchell (2001).
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def gaspari_cohn(
    distances: Float[Array, "..."],
    radius: float,
) -> Float[Array, "..."]:
    r"""Gaspari-Cohn 5th-order piecewise polynomial taper.

    Define ``z = d / r``. Then

    ``ρ(z) = {  −¼ z⁵ + ½ z⁴ + ⅝ z³ − ⁵⁄₃ z² + 1,            0 ≤ z ≤ 1
                 ⅟₁₂ z⁵ − ½ z⁴ + ⅝ z³ + ⁵⁄₃ z² − 5 z + 4 − ⅔ / z,
                                                              1 < z ≤ 2
                 0,                                            z > 2 }``

    Properties:

    * **Compact support** — exactly zero for ``d > 2 r``.
    * **C² smoothness** — value, first and second derivatives all match
      at ``z = 1`` and ``z = 2``.
    * **Positive definite** — the gold standard for covariance
      localization in operational NWP and ocean DA.

    The ``2/(3 z)`` term is only evaluated on the far branch ``z > 1``,
    but :func:`jax.numpy.where` evaluates both arms unconditionally —
    we guard ``z`` away from zero so the unused branch doesn't return
    ``inf`` and contaminate the gradient.

    Args:
        distances: Non-negative distance array, any shape.
        radius: Positive localization half-width ``r``. Compact support
            at ``2 r``.

    Returns:
        Taper weights in ``[0, 1]`` with the same shape as ``distances``.

    Reference:
        Gaspari, G. & Cohn, S. E. (1999). *Construction of correlation
        functions in two and three dimensions.* Q. J. R. Meteorol. Soc.,
        125, 723–757.
    """
    z = distances / radius
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
    r"""Gaussian taper ``ρ(d) = exp(−d² / (2 r²))``.

    Infinitely smooth (``C^∞``) and positive definite, but **not**
    compactly supported — decays exponentially but never reaches zero.
    Useful as a simpler alternative to Gaspari-Cohn when compact support
    is not required (e.g., adjoint sensitivity studies).

    Args:
        distances: Non-negative distance array, any shape.
        radius: Positive characteristic length scale ``r``.

    Returns:
        Taper weights in ``(0, 1]`` with the same shape as ``distances``.
    """
    return jnp.exp(-(distances**2) / (2.0 * radius**2))


def hard_cutoff(
    distances: Float[Array, "..."],
    radius: float,
) -> Float[Array, "..."]:
    r"""Binary cutoff ``ρ(d) = 𝟙{d ≤ r}``.

    Discontinuous at ``d = r``; **not** positive definite in general
    (the resulting localized matrix may lose PSD-ness, which can cause
    filter instability). Provided as a debugging primitive only; prefer
    :func:`gaspari_cohn` or :func:`gaussian_taper` in production.

    Args:
        distances: Non-negative distance array, any shape.
        radius: Positive cutoff radius ``r``.

    Returns:
        Indicator weights (``0.0`` or ``1.0``) with the same shape as
        ``distances``, same dtype.
    """
    return jnp.where(distances <= radius, 1.0, 0.0).astype(distances.dtype)


def localize(
    cov: Float[Array, "M N"],
    taper: Float[Array, "M N"],
) -> Float[Array, "M N"]:
    r"""Apply localization via Schur (element-wise) product.

    Computes ``P_loc = ρ ∘ P`` where ``ρ ∘ P`` denotes
    elementwise multiplication. Works on covariance, gain, or any other
    dense ``(M, N)`` matrix; for structured operators apply localization
    closer to where the matrix is consumed.

    Args:
        cov: Dense covariance or gain matrix of shape ``(M, N)``.
        taper: Taper matrix of the same shape; elementwise multipliers
            in ``[0, 1]``.

    Returns:
        Localized matrix of shape ``(M, N)``.
    """
    return cov * taper

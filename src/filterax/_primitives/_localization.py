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

from collections.abc import Callable

import gaussx
import jax.numpy as jnp
from gaussx import (
    euclidean_distance as euclidean_distance,
    haversine_distance as haversine_distance,
)
from jaxtyping import Array, Float

from filterax._checks import check_ensemble_size


def localization_matrix(
    coords_a: Float[Array, "N_a D"],
    coords_b: Float[Array, "N_b D"],
    radius: float,
    *,
    metric: Callable[
        [Float[Array, "N_a D"], Float[Array, "N_b D"]], Float[Array, "N_a N_b"]
    ] = euclidean_distance,
) -> Float[Array, "N_a N_b"]:
    r"""Pairwise Gaspari-Cohn taper matrix ``ρᵢⱼ = ρ(d(aᵢ, bⱼ); r)``.

    Builds the dense localization matrix consumed by
    :func:`filterax.localize` and :func:`filterax.localized_kalman_gain`
    directly from coordinates, fusing the distance computation with the
    taper. Delegates to :func:`gaussx.localization_matrix` using
    filterax's half-width convention (compact support at ``2 r``), so
    ``localization_matrix(a, b, r)`` matches
    ``gaspari_cohn(metric(a, b), r)`` entry for entry.

    Args:
        coords_a: First coordinate set, shape ``(N_a, D)``.
        coords_b: Second coordinate set, shape ``(N_b, D)``.
        radius: Positive localization half-width ``r``; weights vanish
            beyond distance ``2 r``.
        metric: Pairwise distance function
            ``(coords_a, coords_b) -> (N_a, N_b)``. Defaults to
            :func:`filterax.euclidean_distance`; pass
            :func:`filterax.haversine_distance` for spherical
            (lat, lon)-in-radians grids.

    Returns:
        Taper matrix in ``[0, 1]`` of shape ``(N_a, N_b)``.

    Example:
        >>> import jax.numpy as jnp
        >>> from filterax import localization_matrix
        >>> grid = jnp.arange(3.0)[:, None]  # three points on a line
        >>> rho = localization_matrix(grid, grid, radius=1.0)
        >>> rho.shape, float(rho[0, 0]), float(rho[0, 2])
        ((3, 3), 1.0, 0.0)
    """
    return gaussx.localization_matrix(coords_a, coords_b, 2.0 * radius, metric=metric)


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

    Delegates to :func:`gaussx.gaspari_cohn`, which parameterises by
    the compact-support radius ``c = 2 r`` and guards the ``2/(3 z)``
    far-branch term so reverse-mode gradients stay finite at ``d = 0``.

    Args:
        distances: Non-negative distance array, any shape.
        radius: Positive localization half-width ``r``. Compact support
            at ``2 r``.

    Returns:
        Taper weights in ``[0, 1]`` with the same shape as ``distances``.

    Example:
        >>> import jax.numpy as jnp
        >>> from filterax import gaspari_cohn
        >>> d = jnp.array([0.0, 1.0, 2.0, 3.0])
        >>> w = gaspari_cohn(d, radius=1.0)
        >>> w[0], w[-1]  # full weight at zero distance, zero beyond 2r
        (Array(1., dtype=float32), Array(0., dtype=float32))

    Reference:
        Gaspari, G. & Cohn, S. E. (1999). *Construction of correlation
        functions in two and three dimensions.* Q. J. R. Meteorol. Soc.,
        125, 723–757.
    """
    return gaussx.gaspari_cohn(distances, 2.0 * radius)


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

    Example:
        >>> import jax.numpy as jnp
        >>> from filterax import gaussian_taper
        >>> gaussian_taper(jnp.array([0.0, 1.0]), radius=1.0)
        Array([1.        , 0.60653067], dtype=float32)
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

    Example:
        >>> import jax.numpy as jnp
        >>> from filterax import hard_cutoff
        >>> hard_cutoff(jnp.array([0.5, 1.0, 1.5]), radius=1.0)
        Array([1., 1., 0.], dtype=float32)
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

    Example:
        >>> import jax.numpy as jnp
        >>> from filterax import localize
        >>> cov = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        >>> taper = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        >>> localize(cov, taper)  # off-diagonal entries suppressed
        Array([[1., 0.],
               [0., 1.]], dtype=float32)
    """
    return cov * taper


def soar_taper(
    distances: Float[Array, "..."],
    radius: float,
) -> Float[Array, "..."]:
    r"""Second-Order Auto-Regressive taper (Thiebaux & Pedder 1987).

    ``ρ(d) = (1 + d/r) exp(−d/r)``

    Properties:

    * **C¹ smooth** — value and first derivative continuous, second is
      not (compare with Gaspari-Cohn's C²).
    * **Positive definite** — Schur product preserves PSD.
    * **Approximate compact support** — decays to ``< 0.01`` by
      ``d ≈ 5 r``; finite but exponentially small beyond that.

    Often used as a correlation model for background error covariance
    in operational variational systems (e.g. Met Office VAR). Simpler
    than Gaspari-Cohn while still being positive definite — useful as
    a localization taper when the strict compact support of
    :func:`gaspari_cohn` is not required.

    Args:
        distances: Non-negative distance array, any shape.
        radius: Positive characteristic length scale ``r``.

    Returns:
        Taper weights in ``(0, 1]`` with the same shape as ``distances``.

    Example:
        >>> import jax.numpy as jnp
        >>> from filterax import soar_taper
        >>> soar_taper(jnp.array([0.0, 1.0]), radius=1.0)
        Array([1.       , 0.7357589], dtype=float32)

    Reference:
        Thiebaux, H. J. & Pedder, M. A. (1987). *Spatial Objective
        Analysis.* Academic Press.
    """
    z = distances / radius
    return (1.0 + z) * jnp.exp(-z)


def adaptive_localization(
    state_particles: Float[Array, "N_e N_x"],
    obs_particles: Float[Array, "N_e N_y"],
    *,
    significance: float = 1.0,
) -> Float[Array, "N_x N_y"]:
    r"""Adaptive localization weights from ensemble correlations (Anderson 2007).

    Estimates a per-pair localization weight on the ``(Nₓ, Nᵧ)`` Kalman
    gain by checking whether each sample correlation
    ``r_{ij} = Cˣᴴ_{ij} / (σ_xᵢ σ_yⱼ)`` exceeds its sampling
    uncertainty:

    ``se(r) ≈ (1 − r²) / √(Nₑ − 2)``

    Correlations smaller than ``significance · se(r)`` are zeroed.
    Correlations above that threshold are kept at unit weight (a
    hard-mask variant). The resulting weight matrix is multiplied into
    the Kalman gain (Schur product) to suppress spurious long-range
    correlations driven by sampling noise rather than physical
    structure.

    Unlike a distance-based taper, this localizer is *data-driven* —
    no radius to tune. The trade-off: it requires enough ensemble
    members for the correlation noise floor to be informative
    (typically ``Nₑ ≥ 20``) and an extra ``O(Nₑ Nₓ Nᵧ)`` work per
    cycle.

    Args:
        state_particles: Prior ensemble in state space, ``(Nₑ, Nₓ)``.
        obs_particles: Prior ensemble in obs space ``(Nₑ, Nᵧ)`` —
            ``H`` applied to each member.
        significance: Threshold multiplier on ``se(r)``. Larger →
            more aggressive zeroing.

    Returns:
        Weight matrix ``ρ ∈ ℝ^{Nₓ × Nᵧ}`` with entries in ``{0, 1}``.

    Raises:
        ValueError: if ``Nₑ < 3`` (the ``√(Nₑ − 2)`` noise floor is
            undefined for the smallest ensembles).

    Example:
        >>> import jax
        >>> from filterax import adaptive_localization
        >>> state = jax.random.normal(jax.random.key(0), (20, 2))
        >>> obs_p = state[:, :1]  # obs perfectly correlated with dim 0
        >>> w = adaptive_localization(state, obs_p)
        >>> w.shape
        (2, 1)
        >>> w[0, 0]  # significant correlation kept at unit weight
        Array(1., dtype=float32)

    Reference:
        Anderson, J. L. (2007). *Exploring the need for localization
        in ensemble data assimilation using a hierarchical ensemble
        filter.* Physica D, 230, 99-111.
    """
    N_e = state_particles.shape[0]
    # √(N_e − 2) appears in the sampling-noise denominator; we also
    # rely on the Bessel-corrected std (N_e ≥ 2). Demand N_e ≥ 3.
    check_ensemble_size(N_e)
    if N_e < 3:
        raise ValueError(
            "adaptive_localization needs N_e ≥ 3 so the √(N_e − 2) "
            f"sampling-noise floor is defined; got N_e={N_e}."
        )
    state_mean = jnp.mean(state_particles, axis=0)
    obs_mean = jnp.mean(obs_particles, axis=0)
    state_anom = state_particles - state_mean[None, :]
    obs_anom = obs_particles - obs_mean[None, :]
    state_std = jnp.std(state_particles, axis=0, ddof=1)
    obs_std = jnp.std(obs_particles, axis=0, ddof=1)

    # Sample cross-correlation r_{ij} (Bessel-corrected).
    cross_cov = jnp.einsum("ex,ey->xy", state_anom, obs_anom) / (N_e - 1)
    denom = jnp.maximum(jnp.outer(state_std, obs_std), 1e-30)
    corr = cross_cov / denom
    # Sampling-noise floor; the (1 − r²) factor uses the absolute
    # value so the floor stays positive even for slightly noisy r.
    se = (1.0 - corr * corr) / jnp.sqrt(jnp.asarray(N_e - 2, dtype=corr.dtype))
    keep = jnp.abs(corr) > significance * jnp.maximum(se, 0.0)
    return keep.astype(state_particles.dtype)

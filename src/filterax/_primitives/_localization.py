"""Covariance localization: distance tapers, the Schur product and adaptive weights.

A sample covariance of rank at most N_ℰ − 1 shows spurious correlation between distant
variables, so localization tapers it with distance, Σ̂_x,loc = ρ ∘ Σ̂_x with
ρᵢⱼ = ρ(dᵢⱼ; ℓ), where ℓ is the half-width. The tapers wrap gaussx and keep filterax's
half-width: a compactly supported taper vanishes at 2 ℓ.
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


def localization_matrix(
    coords_a: Float[Array, "N_a D"],
    coords_b: Float[Array, "N_b D"],
    radius: float,
    *,
    metric: Callable[
        [Float[Array, "N_a D"], Float[Array, "N_b D"]], Float[Array, "N_a N_b"]
    ] = euclidean_distance,
) -> Float[Array, "N_a N_b"]:
    """Pairwise Gaspari-Cohn taper between two sets of coordinates.

        ρᵢⱼ = ρ(d(aᵢ, bⱼ); ℓ)

    Builds the dense ρ_xy and ρ_yy consumed by `localize` and `localized_kalman_gain`.
    Wraps `gaussx.localization_matrix`, whose radius is the compact support c, with
    c = 2 ℓ, so it matches ``gaspari_cohn(metric(a, b), radius)`` entry for entry.

    Args:
        coords_a: First set of points aᵢ, one row per point.
        coords_b: Second set of points bⱼ, one row per point.
        radius: Half-width ℓ > 0; the weights vanish beyond 2 ℓ.
        metric: Pairwise distance function d returning one row per point of a and one
            column per point of b. Defaults to `euclidean_distance`; pass
            `haversine_distance` for spherical grids with (latitude, longitude) in
            radians.

    Returns:
        The taper matrix ρ, with entries in [0, 1].

    Examples:
        Three points on a line, with a unit half-width:

        >>> import jax.numpy as jnp
        >>> import filterax
        >>> grid = jnp.array([[0.0], [1.0], [2.0]])
        >>> rho = filterax.localization_matrix(grid, grid, radius=1.0)
        >>> rho.shape, float(rho[0, 0]), float(rho[0, 2])
        ((3, 3), 1.0, 0.0)

        It is the taper of the distance matrix, entry for entry:

        >>> distance = filterax.euclidean_distance(grid, grid)
        >>> taper = filterax.gaspari_cohn(distance, radius=1.0)
        >>> bool(jnp.array_equal(rho, taper))
        True
    """
    # c = 2 ℓ
    support_radius = 2.0 * radius
    # ρᵢⱼ = ρ(d(aᵢ, bⱼ); c)
    return gaussx.localization_matrix(coords_a, coords_b, support_radius, metric=metric)


def gaspari_cohn(
    distances: Float[Array, "..."],
    radius: float,
) -> Float[Array, "..."]:
    """Gaspari-Cohn fifth-order compactly supported taper.

    With z = |d| / ℓ, the piecewise rational

        ρ(z) = −¼ z⁵ + ½ z⁴ + ⅝ z³ − (5/3) z² + 1                       0 ≤ z ≤ 1
        ρ(z) = (1/12) z⁵ − ½ z⁴ + ⅝ z³ + (5/3) z² − 5 z + 4 − 2 / (3 z)   1 < z < 2
        ρ(z) = 0                                                        z ≥ 2

    so ρ = 1 at d = 0 and ρ = 0 for |d| ≥ 2 ℓ. It is positive definite, and C² in d.
    Wraps `gaussx.gaspari_cohn`, which is parameterised by the compact support c and
    guards the 2 / (3 z) term so that reverse-mode gradients stay finite at d = 0. This
    function takes the half-width instead, with c = 2 ℓ.

    Args:
        distances: Distances d of any shape, e.g. a pairwise distance matrix.
        radius: Half-width ℓ > 0; ρ = 0 beyond |d| = 2 ℓ.

    Returns:
        Taper values in [0, 1], with the shape of ``distances``.

    Examples:
        One at zero distance and zero beyond twice the half-width:

        >>> import jax.numpy as jnp
        >>> import filterax
        >>> distances = jnp.array([0.0, 2.0, 3.0])
        >>> filterax.gaspari_cohn(distances, radius=1.0).tolist()
        [1.0, 0.0, 0.0]

        The gradient stays finite at d = 0:

        >>> import jax
        >>> slope = jax.grad(lambda d: filterax.gaspari_cohn(d, 1.0))(jnp.array(0.0))
        >>> bool(jnp.isfinite(slope))
        True

    References:
        Gaspari & Cohn (1999), *Q. J. R. Meteorol. Soc.* 125, 723–757.
    """
    # c = 2 ℓ
    support_radius = 2.0 * radius
    # ρ(z),   z = 2 |d| / c = |d| / ℓ
    return gaussx.gaspari_cohn(distances, support_radius)


def gaussian_taper(
    distances: Float[Array, "..."],
    radius: float,
) -> Float[Array, "..."]:
    """Gaussian localization taper.

        ρ(d) = exp(−d² / (2 ℓ²))

    Wraps `gaussx.gaussian_taper`, whose length scale L is ℓ here. It is infinitely
    smooth and positive definite, but not compactly supported: it decays until it
    rounds to zero. It is a simpler alternative to `gaspari_cohn` where compact support
    is not required, such as adjoint sensitivity studies.

    Args:
        distances: Distances d of any shape, e.g. a pairwise distance matrix.
        radius: Length scale ℓ > 0.

    Returns:
        Taper values in [0, 1], with the shape of ``distances``.

    Examples:
        One at zero distance, decaying with distance:

        >>> import jax.numpy as jnp
        >>> import filterax
        >>> values = filterax.gaussian_taper(jnp.array([0.0, 1.0]), radius=1.0)
        >>> [round(value, 4) for value in values.tolist()]
        [1.0, 0.6065]

        It never reaches zero at a moderate distance:

        >>> bool(filterax.gaussian_taper(jnp.array(4.0), radius=1.0) > 0.0)
        True
    """
    # ρ(d) = exp(−d² / (2 ℓ²))
    return gaussx.gaussian_taper(distances, radius)


def hard_cutoff(
    distances: Float[Array, "..."],
    radius: float,
) -> Float[Array, "..."]:
    """Binary taper, a debugging baseline.

        ρ(d) = 1 for d ≤ ℓ,   ρ(d) = 0 for d > ℓ

    It is discontinuous at d = ℓ and not positive definite in general, so the localized
    matrix may lose positive semi-definiteness, which can destabilise a filter. Prefer
    `gaspari_cohn` or `gaussian_taper` in production.

    Args:
        distances: Distances d of any shape.
        radius: Cutoff radius ℓ > 0.

    Returns:
        Indicator weights with the shape and dtype of ``distances``.

    Examples:
        The cutoff itself is kept:

        >>> import jax.numpy as jnp
        >>> import filterax
        >>> filterax.hard_cutoff(jnp.array([0.5, 1.0, 1.5]), radius=1.0).tolist()
        [1.0, 1.0, 0.0]

        The dtype of the distances is preserved:

        >>> distances = jnp.array([0.5], dtype=jnp.float32)
        >>> filterax.hard_cutoff(distances, radius=1.0).dtype == distances.dtype
        True
    """
    # d ≤ ℓ
    within_radius = distances <= radius
    # ρ(d)
    weights = jnp.where(within_radius, 1.0, 0.0)
    return weights.astype(distances.dtype)


def localize(
    cov: Float[Array, "M N"],
    taper: Float[Array, "M N"],
) -> Float[Array, "M N"]:
    """Apply a taper by the Schur, or element-wise, product.

        Σ̂_x,loc = ρ ∘ Σ̂_x

    Works on a covariance, a gain or any other dense matrix. For a structured operator,
    localize closer to where the matrix is consumed.

    Args:
        cov: Dense covariance or gain matrix.
        taper: Taper matrix ρ of the same shape, with multipliers in [0, 1].

    Returns:
        The localized matrix.

    Examples:
        A taper that decouples the two variables:

        >>> import jax.numpy as jnp
        >>> import filterax
        >>> covariance = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        >>> taper = jnp.eye(2)
        >>> filterax.localize(covariance, taper).tolist()
        [[1.0, 0.0], [0.0, 1.0]]

        A unit taper changes nothing:

        >>> unit = jnp.ones((2, 2))
        >>> bool(jnp.array_equal(filterax.localize(covariance, unit), covariance))
        True

    References:
        Houtekamer & Mitchell (2001).
    """
    # ρ ∘ Σ̂_x
    return cov * taper


def soar_taper(
    distances: Float[Array, "..."],
    radius: float,
) -> Float[Array, "..."]:
    """Second-order auto-regressive (SOAR) taper.

        ρ(d) = (1 + |d| / ℓ) exp(−|d| / ℓ)

    Wraps `gaussx.soar_taper`, whose length scale L is ℓ here. It is positive definite
    but not compactly supported, and it decays more slowly than `gaussian_taper`: ρ is
    still about 0.04 at d = 5 ℓ. It is a common correlation model for background error
    in operational variational systems, and a simple taper where the compact support of
    `gaspari_cohn` is not required.

    Args:
        distances: Distances d of any shape, e.g. a pairwise distance matrix.
        radius: Length scale ℓ > 0.

    Returns:
        Taper values in [0, 1], with the shape of ``distances``.

    Examples:
        One at zero distance, decaying with distance:

        >>> import jax.numpy as jnp
        >>> import filterax
        >>> values = filterax.soar_taper(jnp.array([0.0, 1.0]), radius=1.0)
        >>> [round(value, 4) for value in values.tolist()]
        [1.0, 0.7358]

        Still about 0.04 at five length scales:

        >>> round(float(filterax.soar_taper(jnp.array(5.0), radius=1.0)), 2)
        0.04

    References:
        Thiebaux & Pedder (1987), *Spatial Objective Analysis*, Academic Press.
    """
    # ρ(d) = (1 + |d| / ℓ) exp(−|d| / ℓ)
    return gaussx.soar_taper(distances, radius)


def adaptive_localization(
    state_particles: Float[Array, "N_e N_x"],
    obs_particles: Float[Array, "N_e N_y"],
    *,
    significance: float = 1.0,
) -> Float[Array, "N_x N_y"]:
    """Adaptive weights from the sampling noise of ensemble correlations.

        rᵢⱼ = [Σ̂_xy]ᵢⱼ / (σ_x,i σ_h,j),   se(r) ≈ (1 − r²) / √(N_ℰ − 2)
        ρᵢⱼ = 1 if |rᵢⱼ| > s se(rᵢⱼ),   ρᵢⱼ = 0 otherwise

    Keeps a pair at unit weight, a hard mask, when its sample correlation r exceeds s
    standard errors of sampling noise. The weights multiply the gain by the Schur
    product, which suppresses long-range correlations driven by noise rather than by
    physical structure.

    Unlike a distance taper, this has no half-width to tune. In exchange it needs enough
    members for the noise floor to be informative, typically N_ℰ ≥ 20, and costs an
    extra O(N_ℰ D_x D_y) per window.

    Algorithm:
        1. Check that N_ℰ ≥ 3, so that √(N_ℰ − 2) is defined.
        2. Take the sample spreads σ_x and σ_h with the divisor N_ℰ − 1.
        3. Divide Σ̂_xy by the outer product of the spreads to get r.
        4. Compare |r| with s se(r), floored at zero.

    Args:
        state_particles: Prior ensemble x⁽ᵉ⁾ in state space, one row per member.
        obs_particles: Its image h⁽ᵉ⁾ = h(x⁽ᵉ⁾) in observation space.
        significance: Threshold s on se(r); larger zeroes more pairs.

    Returns:
        The weight matrix ρ, with entries in {0, 1}, in the dtype of
        ``state_particles``.

    Raises:
        ValueError: If N_ℰ < 3, where the noise floor is undefined.

    Examples:
        An observation of the first component keeps that pair at unit weight:

        >>> import jax.random as jr
        >>> import filterax
        >>> state = jr.normal(jr.key(0), (20, 2))
        >>> weights = filterax.adaptive_localization(state, state[:, :1])
        >>> weights.shape, float(weights[0, 0])
        ((2, 1), 1.0)

        Two members cannot supply a noise floor:

        >>> filterax.adaptive_localization(
        ...     state[:2], state[:2, :1]
        ... )  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        ValueError: adaptive_localization needs N_e ≥ 3 so the √(N_e − 2)
        sampling-noise floor is defined; got N_e=2.

    References:
        Anderson (2007), *Physica D* 230, 99-111.
    """
    ensemble_size = state_particles.shape[0]
    # √(N_ℰ − 2) appears in the sampling-noise denominator, which also covers the
    # N_ℰ ≥ 2 the Bessel-corrected spread needs.
    if ensemble_size < 3:
        raise ValueError(
            "adaptive_localization needs N_e ≥ 3 so the √(N_e − 2) "
            f"sampling-noise floor is defined; got N_e={ensemble_size}."
        )
    # jnp, not einx: D_x = 0 or D_y = 0 returns an empty matrix here today, and einx
    # raises AxisSizeError on a zero-length axis.
    # σ_x
    state_spread = jnp.std(state_particles, axis=0, ddof=1)
    # σ_h
    obs_spread = jnp.std(obs_particles, axis=0, ddof=1)
    # Σ̂_xy
    cross_covariance = gaussx.ensemble_cross_covariance(
        state_particles, obs_particles, bessel=True
    )
    # σ_x σ_hᵀ
    spread_product = jnp.outer(state_spread, obs_spread)
    # Floored, so that a collapsed coordinate gives r = 0 rather than a division by 0.
    safe_spread_product = jnp.maximum(spread_product, 1e-30)
    # r
    correlation = cross_covariance / safe_spread_product
    # 1 − r²
    unexplained = 1.0 - correlation * correlation
    # √(N_ℰ − 2)
    root_members = jnp.sqrt(jnp.asarray(ensemble_size - 2, dtype=correlation.dtype))
    # se(r) = (1 − r²) / √(N_ℰ − 2)
    standard_error = unexplained / root_members
    # |r|
    correlation_size = jnp.abs(correlation)
    # max(se(r), 0), since round-off can take |r| just above 1
    floored_error = jnp.maximum(standard_error, 0.0)
    # s se(r)
    threshold = significance * floored_error
    # ρ = 1 where |r| > s se(r)
    keep = correlation_size > threshold
    return keep.astype(state_particles.dtype)

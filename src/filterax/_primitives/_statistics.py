"""Ensemble statistics: the sample mean, anomalies and covariances of an ensemble.

Pure functions over particles with one member x⁽ᵉ⁾, e ∈ ℰ, per row. The covariances
wrap gaussx with the ensemble Kalman divisor N_ℰ − 1.
"""

from __future__ import annotations

import einx
import gaussx
from jaxtyping import Array, Float


def ensemble_mean(
    particles: Float[Array, "N_e N_x"],
) -> Float[Array, " N_x"]:
    """Sample mean of an ensemble.

        x̄ = (1 / N_ℰ) ∑ₑ x⁽ᵉ⁾

    Args:
        particles: Ensemble members x⁽ᵉ⁾, one row per member.

    Returns:
        The ensemble mean x̄.

    Raises:
        AxisSizeError: From einx, if N_ℰ = 0 or D_x = 0.

    Examples:
        Two members in two dimensions:

        >>> import jax.numpy as jnp
        >>> import filterax
        >>> particles = jnp.array([[0.0, 1.0], [2.0, 3.0]])
        >>> filterax.ensemble_mean(particles).tolist()
        [1.0, 2.0]

        One member is its own mean:

        >>> filterax.ensemble_mean(particles[:1]).tolist()
        [0.0, 1.0]
    """
    # x̄ = (1 / N_ℰ) ∑ₑ x⁽ᵉ⁾
    return einx.mean("e x -> x", particles)


def ensemble_anomalies(
    particles: Float[Array, "N_e N_x"],
) -> Float[Array, "N_e N_x"]:
    """State anomalies of an ensemble about its sample mean.

        x̃⁽ᵉ⁾ = x⁽ᵉ⁾ − x̄

    The rows of X̃ sum to zero by construction. X̃ is the input to the covariance and
    gain recipes: every ensemble Kalman filter is some choice of square root applied
    to these anomalies.

    Args:
        particles: Ensemble members x⁽ᵉ⁾, one row per member.

    Returns:
        The anomalies X̃, with rows x̃⁽ᵉ⁾.

    Raises:
        AxisSizeError: From einx, if N_ℰ = 0 or D_x = 0.

    Examples:
        Two members about their mean:

        >>> import jax.numpy as jnp
        >>> import filterax
        >>> particles = jnp.array([[0.0, 1.0], [2.0, 3.0]])
        >>> filterax.ensemble_anomalies(particles).tolist()
        [[-1.0, -1.0], [1.0, 1.0]]

        One member has no spread:

        >>> filterax.ensemble_anomalies(particles[:1]).tolist()
        [[0.0, 0.0]]
    """
    # x̄ = (1 / N_ℰ) ∑ₑ x⁽ᵉ⁾
    mean = ensemble_mean(particles)
    # x̃⁽ᵉ⁾ = x⁽ᵉ⁾ − x̄
    return einx.subtract("e x, x -> e x", particles, mean)


def ensemble_covariance(
    particles: Float[Array, "N_e N_x"],
) -> gaussx.LowRankUpdate:
    """Sample covariance of an ensemble as a low-rank operator.

        Σ̂_x = X̃ᵀ X̃ / (N_ℰ − 1) = U Uᵀ,   U = X̃ᵀ / √(N_ℰ − 1)

    Wraps `gaussx.ensemble_covariance` and fixes ``bessel=True``, where gaussx itself
    defaults to the divisor N_ℰ. The result has rank ≤ N_ℰ − 1 and the (D_x, D_x)
    matrix is never formed. Construction costs O(N_ℰ D_x), and gaussx's structural
    dispatch solves it through Woodbury at O(N_ℰ² D_x + N_ℰ³) instead of O(D_x³).

    Args:
        particles: Ensemble members x⁽ᵉ⁾, one row per member.

    Returns:
        A `gaussx.LowRankUpdate` holding Σ̂_x.

    Raises:
        ValueError: If N_ℰ < 2, where the divisor N_ℰ − 1 is undefined.

    Examples:
        Three members along a diagonal:

        >>> import gaussx
        >>> import jax.numpy as jnp
        >>> import filterax
        >>> particles = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        >>> covariance = filterax.ensemble_covariance(particles)
        >>> isinstance(covariance, gaussx.LowRankUpdate)
        True
        >>> bool(jnp.allclose(covariance.as_matrix(), jnp.ones((2, 2))))
        True

        One member has no sample covariance:

        >>> filterax.ensemble_covariance(
        ...     particles[:1]
        ... )  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        ValueError: Bessel correction requires J >= 2 particles (divisor is J - 1);
        got J=1. Pass bessel=False for a maximum-likelihood divisor.
    """
    # Σ̂_x = U Uᵀ
    return gaussx.ensemble_covariance(particles, bessel=True)


def cross_covariance(
    particles: Float[Array, "N_e N_x"],
    obs_particles: Float[Array, "N_e N_y"],
) -> Float[Array, "N_x N_y"]:
    """Sample cross-covariance between an ensemble and its image in observation space.

        Σ̂_xy = X̃ᵀ H̃ / (N_ℰ − 1)

    Wraps `gaussx.ensemble_cross_covariance` and fixes ``bessel=True``. The result is
    dense because D_y is typically small, and it costs O(N_ℰ D_x D_y). For a nonlinear
    h it is the ensemble's derivative-free linearisation of h.

    Args:
        particles: Prior ensemble x⁽ᵉ⁾ in state space, one row per member.
        obs_particles: Its image h⁽ᵉ⁾ = h(x⁽ᵉ⁾) in observation space.

    Returns:
        The dense cross-covariance Σ̂_xy.

    Raises:
        ValueError: If N_ℰ < 2.

    Examples:
        Two members with a linear observation of half the state:

        >>> import jax.numpy as jnp
        >>> import filterax
        >>> particles = jnp.array([[0.0, 0.0], [2.0, 2.0]])
        >>> obs_particles = jnp.array([[0.0], [1.0]])
        >>> filterax.cross_covariance(particles, obs_particles).tolist()
        [[1.0], [1.0]]

        An ensemble against itself is `ensemble_covariance`:

        >>> dense = filterax.ensemble_covariance(particles).as_matrix()
        >>> bool(jnp.allclose(filterax.cross_covariance(particles, particles), dense))
        True

    References:
        Evensen (2003), §5.
    """
    # Σ̂_xy = X̃ᵀ H̃ / (N_ℰ − 1)
    return gaussx.ensemble_cross_covariance(particles, obs_particles, bessel=True)

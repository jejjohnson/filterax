"""Kalman gain from an ensemble, with or without Schur-product localization."""

from __future__ import annotations

import gaussx
import lineax as lx
from jaxtyping import Array, Float


def kalman_gain(
    particles: Float[Array, "N_e N_x"],
    obs_particles: Float[Array, "N_e N_y"],
    obs_noise: lx.AbstractLinearOperator,
    *,
    solver: gaussx.AbstractSolverStrategy | None = None,
    dense_innovation: bool | None = None,
) -> Float[Array, "N_x N_y"]:
    """Kalman gain from an ensemble and its image in observation space.

        G = Σ̂_xy Σ̂_y⁻¹,   Σ̂_y = Σ̂_h + Σ_ε

    Wraps `gaussx.ensemble_kalman_gain` and fixes ``bessel=True``, the divisor
    N_ℰ − 1. Σ̂_y is assembled by one of two routes, which agree to round-off. With
    U = H̃ᵀ / √(N_ℰ − 1), the Woodbury route solves through

        Σ̂_y⁻¹ = Σ_ε⁻¹ − Σ_ε⁻¹ U (I + Uᵀ Σ_ε⁻¹ U)⁻¹ Uᵀ Σ_ε⁻¹

    - **Woodbury**, the default when N_ℰ < D_y: Σ̂_y is a `gaussx.LowRankUpdate` of
      Σ_ε, solved by structural dispatch at O(N_ℰ² D_y + N_ℰ³), and a structured Σ_ε
      is never materialised.
    - **Dense**, the default when N_ℰ ≥ D_y: Σ̂_y is formed as a (D_y, D_y) matrix and
      solved directly at O(N_ℰ D_y² + D_y³), because the capacitance would be the
      larger matrix. It also handles a positive semi-definite Σ_ε, which Woodbury
      cannot.

    Algorithm:
        1. Choose the route for Σ̂_y from the shapes or ``dense_innovation``.
        2. Form the cross-covariance Σ̂_xy.
        3. Assemble Σ̂_y densely, or as a low-rank update of Σ_ε.
        4. Solve for the gain G = Σ̂_xy Σ̂_y⁻¹ row by row.

    Args:
        particles: Prior ensemble x⁽ᵉ⁾ in state space, one row per member.
        obs_particles: Its image h⁽ᵉ⁾ = h(x⁽ᵉ⁾) in observation space. A nonlinear h
            is linearised implicitly by the ensemble.
        obs_noise: Observation-error covariance Σ_ε.
        solver: Solver strategy for Σ̂_y; ``None`` uses structural dispatch. A
            matrix-free strategy wants ``dense_innovation=False``, so that it gets
            the structured operator.
        dense_innovation: Whether to form the (D_y, D_y) Σ̂_y densely. ``None``
            chooses by shape; ``True`` and ``False`` force the dense and the
            Woodbury route.

    Returns:
        The dense Kalman gain G. It is consumed once by the analysis and is usually
        small, since D_y counts the observations.

    Raises:
        ValueError: If the ensembles have different sizes, ``obs_noise`` is not
            (D_y, D_y), or N_ℰ < 2.

    Examples:
        Three members, with the first component observed:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import filterax
        >>> particles = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        >>> obs_particles = particles[:, :1]
        >>> obs_noise = lx.DiagonalLinearOperator(0.5 * jnp.ones(1))
        >>> gain = filterax.kalman_gain(particles, obs_particles, obs_noise)
        >>> gain.shape
        (2, 1)
        >>> bool(jnp.allclose(gain, jnp.array([[2.0 / 3.0], [0.0]])))
        True

        The two routes agree to round-off:

        >>> woodbury = filterax.kalman_gain(
        ...     particles, obs_particles, obs_noise, dense_innovation=False
        ... )
        >>> bool(jnp.allclose(gain, woodbury, atol=1e-4))
        True

        One member has no sample covariance:

        >>> filterax.kalman_gain(
        ...     particles[:1], obs_particles[:1], obs_noise
        ... )  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        ValueError: Bessel correction requires J >= 2 particles (divisor is J - 1);
        got J=1. Pass bessel=False for a maximum-likelihood divisor.
    """
    # G = Σ̂_xy Σ̂_y⁻¹
    return gaussx.ensemble_kalman_gain(
        particles,
        obs_particles,
        obs_noise,
        solver=solver,
        bessel=True,
        dense_innovation=dense_innovation,
    )


def localized_kalman_gain(
    particles: Float[Array, "N_e N_x"],
    obs_particles: Float[Array, "N_e N_y"],
    obs_noise: lx.AbstractLinearOperator,
    rho_xy: Float[Array, "N_x N_y"],
    rho_yy: Float[Array, "N_y N_y"],
    *,
    solver: gaussx.AbstractSolverStrategy | None = None,
) -> Float[Array, "N_x N_y"]:
    """Kalman gain with Schur-product localization of both sample covariances.

        G = (ρ_xy ∘ Σ̂_xy) (ρ_yy ∘ Σ̂_h + Σ_ε)⁻¹

    Wraps `gaussx.localized_kalman_gain` and fixes ``bessel=True``, the divisor
    N_ℰ − 1. Tapering suppresses spurious long-range sample correlations, and by the
    Schur product theorem ρ_yy ∘ Σ̂_h stays positive semi-definite when ρ_yy is. Build
    the tapers with `localization_matrix`.

    Unlike `kalman_gain`, the Schur product destroys the low-rank structure of Σ̂_h,
    so the (D_y, D_y) matrix is materialised, at O(N_ℰ D_x D_y + D_y³). The tapers
    ρ_xy = ρ_yy = 1 recover the unlocalized gain.

    Algorithm:
        1. Form the sample covariances Σ̂_xy and Σ̂_h.
        2. Taper them by ρ_xy and ρ_yy.
        3. Add Σ_ε and symmetrise.
        4. Solve for the gain G row by row.

    Args:
        particles: Prior ensemble x⁽ᵉ⁾ in state space, one row per member.
        obs_particles: Its image h⁽ᵉ⁾ = h(x⁽ᵉ⁾) in observation space.
        obs_noise: Observation-error covariance Σ_ε.
        rho_xy: State-observation taper ρ_xy.
        rho_yy: Observation-observation taper ρ_yy.
        solver: Solver strategy for the dense matrix; ``None`` uses structural
            dispatch.

    Returns:
        The dense localized Kalman gain G.

    Raises:
        ValueError: If the ensembles have different sizes, ``obs_noise`` or the
            tapers do not match D_x and D_y, or N_ℰ < 2.

    Examples:
        Unit tapers recover the unlocalized gain:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import filterax
        >>> particles = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        >>> obs_particles = particles[:, :1]
        >>> obs_noise = lx.DiagonalLinearOperator(0.5 * jnp.ones(1))
        >>> gain = filterax.localized_kalman_gain(
        ...     particles, obs_particles, obs_noise, jnp.ones((2, 1)), jnp.ones((1, 1))
        ... )
        >>> plain = filterax.kalman_gain(particles, obs_particles, obs_noise)
        >>> bool(jnp.allclose(gain, plain))
        True

        A zero taper removes that coupling from the gain:

        >>> rho_xy = jnp.array([[0.0], [1.0]])
        >>> gain = filterax.localized_kalman_gain(
        ...     particles, obs_particles, obs_noise, rho_xy, jnp.ones((1, 1))
        ... )
        >>> float(gain[0, 0])
        0.0

        A taper of the wrong shape is rejected:

        >>> filterax.localized_kalman_gain(
        ...     particles, obs_particles, obs_noise, jnp.ones((1, 1)), jnp.ones((1, 1))
        ... )  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        ValueError: rho_xy must have shape (2, 1) to match particles and obs_particles,
        got (1, 1).
    """
    # G = (ρ_xy ∘ Σ̂_xy) (ρ_yy ∘ Σ̂_h + Σ_ε)⁻¹
    return gaussx.localized_kalman_gain(
        particles, obs_particles, obs_noise, rho_xy, rho_yy, solver=solver, bessel=True
    )

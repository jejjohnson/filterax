"""Kalman gain from an ensemble."""

from __future__ import annotations

import gaussx
import lineax as lx
from jaxtyping import Array, Float

from filterax._checks import check_ensemble_size


def kalman_gain(
    particles: Float[Array, "N_e N_x"],
    obs_particles: Float[Array, "N_e N_y"],
    obs_noise: lx.AbstractLinearOperator,
    *,
    solver: gaussx.AbstractSolverStrategy | None = None,
) -> Float[Array, "N_x N_y"]:
    r"""Ensemble Kalman gain ``K = Cˣᴴ (Cᴴᴴ + R)⁻¹``.

    Thin wrapper over :func:`gaussx.ensemble_kalman_gain` with the EnKF
    Bessel convention. The innovation covariance ``S = Cᴴᴴ + R`` is
    assembled as a :class:`gaussx.LowRankUpdate` so structural dispatch
    routes the solve through the Woodbury identity:

    $$
    S^{-1} = R^{-1} - R^{-1} U
        \left( (N_{e} - 1) I + U^{\top} R^{-1} U \right)^{-1}
        U^{\top} R^{-1},
    \qquad
    U = \frac{(HX)^{\prime\top}}{\sqrt{N_{e} - 1}}.
    $$

    Total cost
    ``O(Nₑ² Nᵧ + Nₑ³)`` for low-rank Woodbury, instead of ``O(Nᵧ³)`` for
    the dense fallback. The dense ``(Nᵧ, Nᵧ)`` matrix is never materialised
    when ``R`` carries structure (diagonal, low-rank, Toeplitz, …).

    Args:
        particles: Prior ensemble in state space, shape ``(Nₑ, Nₓ)``.
        obs_particles: ``H`` applied to each member, shape ``(Nₑ, Nᵧ)``.
            For linear ``H``, this is ``X H ᵀ``; for nonlinear ``H``, the
            ensemble provides an implicit linearisation.
        obs_noise: Observation error covariance ``R`` as a linear operator.
        solver: Optional :class:`gaussx.AbstractSolverStrategy`. ``None``
            lets gaussx pick by operator type (Woodbury for low-rank +
            structured base, dense Cholesky otherwise).

    Returns:
        Dense Kalman gain of shape ``(Nₓ, Nᵧ)``. Materialising ``K`` is
        intentional — it is consumed once by the analysis update and is
        usually small (``Nᵧ`` is the observation count).

    Raises:
        ValueError: if ``Nₑ < 2``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from filterax import kalman_gain
        >>> particles = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        >>> obs_particles = particles[:, :1]  # H observes the first component
        >>> R = lx.DiagonalLinearOperator(0.5 * jnp.ones(1))
        >>> kalman_gain(particles, obs_particles, R)
        Array([[0.6666666],
               [0.       ]], dtype=float32)
    """
    check_ensemble_size(particles.shape[0])
    return gaussx.ensemble_kalman_gain(
        particles, obs_particles, obs_noise, solver=solver, bessel=True
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
    r"""Kalman gain with Schur-product (Hadamard) localization.

    Thin wrapper over :func:`gaussx.localized_kalman_gain` with the EnKF
    Bessel convention. Computes

    $$
    K = (\rho^{xy} \circ C^{xH})\,(\rho^{yy} \circ C^{HH} + R)^{-1}.
    $$

    Tapering both covariances with the Schur
    (element-wise) product suppresses spurious long-range correlations;
    by the Schur product theorem the tapered innovation covariance stays
    PSD when ``ρʸʸ`` is (use :func:`filterax.localization_matrix` /
    :func:`filterax.gaspari_cohn` to build the tapers).

    Unlike :func:`filterax.kalman_gain`, the Hadamard product destroys
    the low-rank structure of ``Cᴴᴴ``, so the innovation covariance is
    materialised densely — cost ``O(Nₑ Nₓ Nᵧ + Nᵧ³)``. With
    ``ρ ≡ 1`` this reduces exactly to the unlocalized gain.

    Args:
        particles: Prior ensemble in state space, shape ``(Nₑ, Nₓ)``.
        obs_particles: ``H`` applied to each member, shape ``(Nₑ, Nᵧ)``.
        obs_noise: Observation error covariance ``R`` as a linear operator.
        rho_xy: State-observation taper, shape ``(Nₓ, Nᵧ)``.
        rho_yy: Observation-observation taper, shape ``(Nᵧ, Nᵧ)``.
        solver: Optional :class:`gaussx.AbstractSolverStrategy`. ``None``
            uses structural dispatch on the dense innovation covariance.

    Returns:
        Dense localized Kalman gain of shape ``(Nₓ, Nᵧ)``.

    Raises:
        ValueError: if ``Nₑ < 2``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from filterax import kalman_gain, localized_kalman_gain
        >>> particles = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        >>> obs_particles = particles[:, :1]
        >>> R = lx.DiagonalLinearOperator(0.5 * jnp.ones(1))
        >>> ones = jnp.ones((2, 1)), jnp.ones((1, 1))
        >>> K_loc = localized_kalman_gain(particles, obs_particles, R, *ones)
        >>> bool(jnp.allclose(K_loc, kalman_gain(particles, obs_particles, R)))
        True
    """
    check_ensemble_size(particles.shape[0])
    return gaussx.localized_kalman_gain(
        particles, obs_particles, obs_noise, rho_xy, rho_yy, solver=solver, bessel=True
    )

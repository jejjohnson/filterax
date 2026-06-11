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

    ``S⁻¹ = R⁻¹ − R⁻¹ U ((Nₑ − 1) I + Uᵀ R⁻¹ U)⁻¹ Uᵀ R⁻¹``

    with ``U = (HX)′ᵀ / √(Nₑ − 1)``. Total cost
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

    Example:
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

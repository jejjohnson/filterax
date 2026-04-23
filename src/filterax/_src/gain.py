"""Kalman gain from an ensemble."""

from __future__ import annotations

import gaussx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from filterax._src.statistics import cross_covariance, ensemble_anomalies


def kalman_gain(
    particles: Float[Array, "N_e N_x"],
    obs_particles: Float[Array, "N_e N_y"],
    obs_noise: lx.AbstractLinearOperator,
    *,
    solver: gaussx.AbstractSolverStrategy | None = None,
) -> Float[Array, "N_x N_y"]:
    r"""Ensemble Kalman gain :math:`K = C^{xH} (C^{HH} + R)^{-1}`.

    The innovation covariance :math:`S = C^{HH} + R` is assembled as a
    :class:`gaussx.LowRankUpdate` with base ``R`` and rank-:math:`(N_e - 1)`
    update :math:`(HX)'^T / \sqrt{N_e - 1}`. Solves dispatch via gaussx, which
    applies the Woodbury identity when beneficial (cost
    :math:`O(N_e^2 N_y + N_e^3)` vs. :math:`O(N_y^3)` for a dense solve).

    Args:
        particles: Prior ensemble in state space, shape ``(N_e, N_x)``.
        obs_particles: Prior ensemble in observation space
            (``H`` applied to each member), shape ``(N_e, N_y)``.
        obs_noise: Observation error covariance :math:`R` as a linear
            operator of shape ``(N_y, N_y)``.
        solver: Optional :class:`gaussx.AbstractSolverStrategy`. When
            ``None``, structural dispatch picks an appropriate solver.

    Returns:
        Dense Kalman gain of shape ``(N_x, N_y)``.
    """
    N_e = particles.shape[0]
    Hxp = ensemble_anomalies(obs_particles)  # (N_e, N_y)
    C_xH = cross_covariance(particles, obs_particles)  # (N_x, N_y)

    # S = R + (1/(N_e-1)) (HX)'^T (HX)'  =  R + U U^T,  U = (HX)'^T / sqrt(N_e-1)
    U = Hxp.T / jnp.sqrt(N_e - 1)  # (N_y, N_e)
    S = gaussx.LowRankUpdate(obs_noise, U)

    # K = C_xH @ S^{-1}  — solve each row of C_xH against S.
    return gaussx.solve_rows(S, C_xH, solver=solver)

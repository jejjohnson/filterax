"""Kalman gain from an ensemble."""

from __future__ import annotations

import gaussx
import lineax as lx
from jaxtyping import Array, Float

from filterax._src._checks import check_ensemble_size


def kalman_gain(
    particles: Float[Array, "N_e N_x"],
    obs_particles: Float[Array, "N_e N_y"],
    obs_noise: lx.AbstractLinearOperator,
    *,
    solver: gaussx.AbstractSolverStrategy | None = None,
) -> Float[Array, "N_x N_y"]:
    r"""Ensemble Kalman gain :math:`K = C^{xH} (C^{HH} + R)^{-1}`.

    Thin wrapper over :func:`gaussx.ensemble_kalman_gain` (Bessel-corrected,
    EnKF convention). The innovation covariance :math:`S = C^{HH} + R` is
    assembled as a :class:`gaussx.LowRankUpdate` so structural dispatch can
    apply the Woodbury identity (cost :math:`O(N_e^2 N_y + N_e^3)` vs.
    :math:`O(N_y^3)`).

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

    Raises:
        ValueError: if ``particles`` has fewer than 2 ensemble members.
    """
    check_ensemble_size(particles.shape[0])
    return gaussx.ensemble_kalman_gain(
        particles, obs_particles, obs_noise, solver=solver, bessel=True
    )

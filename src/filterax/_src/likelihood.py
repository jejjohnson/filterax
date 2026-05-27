"""Innovation log-likelihood and diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import gaussx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from filterax._src._checks import check_ensemble_size
from filterax._src.statistics import ensemble_mean


def log_likelihood(
    innovation: Float[Array, " N_y"],
    innovation_cov: lx.AbstractLinearOperator,
    *,
    solver: gaussx.AbstractSolverStrategy | None = None,
) -> Float[Array, ""]:
    r"""Gaussian log-probability of an innovation vector.

    .. math::

        \log p(y \mid \text{forecast})
        = -\tfrac{1}{2}\left[N_y \log(2\pi) + \log|S| + v^T S^{-1} v\right]

    where :math:`v` is the innovation and :math:`S` the innovation covariance.
    ``logdet`` and ``solve`` dispatch via :mod:`gaussx`; when ``S`` is a
    :class:`gaussx.LowRankUpdate`, the matrix-determinant lemma and Woodbury
    identity are used automatically.

    Args:
        innovation: Innovation vector :math:`v = y - H\bar{x}`, shape ``(N_y,)``.
        innovation_cov: Innovation covariance :math:`S = H P H^T + R`.
        solver: Optional :class:`gaussx.AbstractSolverStrategy` (needs both
            solve and logdet). When ``None``, uses structural dispatch.

    Returns:
        Scalar log-probability.
    """
    zero = jnp.zeros_like(innovation)
    return gaussx.gaussian_log_prob(
        zero,
        innovation_cov,
        innovation,
        solver=solver,
    )


def innovation_covariance(
    obs_particles: Float[Array, "N_e N_y"],
    obs_noise: lx.AbstractLinearOperator,
) -> gaussx.LowRankUpdate:
    r"""Innovation covariance :math:`S = C^{HH} + R` as a low-rank update.

    Composes the Bessel-corrected ensemble covariance of the observation-space
    particles with the observation noise as base, so structural dispatch can
    apply Woodbury / matrix-determinant lemma on solves and logdets.
    """
    check_ensemble_size(obs_particles.shape[0])
    cov = gaussx.ensemble_covariance(obs_particles, bessel=True)
    return gaussx.LowRankUpdate(obs_noise, cov.U)


class InnovationStatistics(TypedDict):
    """Return type of :func:`innovation_statistics`."""

    innovation: Float[Array, " N_y"]
    innovation_cov: lx.AbstractLinearOperator
    normalized_innovation: Float[Array, " N_y"]
    log_likelihood: Float[Array, ""]


def innovation_statistics(
    particles: Float[Array, "N_e N_x"],
    obs: Float[Array, " N_y"],
    obs_op: Callable[[Float[Array, " N_x"]], Float[Array, " N_y"]],
    obs_noise: lx.AbstractLinearOperator,
    *,
    solver: gaussx.AbstractSolverStrategy | None = None,
) -> InnovationStatistics:
    r"""Innovation diagnostics for a forecast ensemble.

    Builds :math:`v = y - H\bar{x}` and :math:`S = C^{HH} + R`, then returns
    the innovation, its covariance, the normalised innovation
    :math:`S^{-1/2} v`, and the Gaussian log-likelihood.

    Args:
        particles: Forecast ensemble, shape ``(N_e, N_x)``.
        obs: Observation vector, shape ``(N_y,)``.
        obs_op: Observation operator applied to a single state vector.
        obs_noise: Observation error covariance :math:`R`.
        solver: Optional solver strategy.

    Returns:
        Mapping with keys ``innovation``, ``innovation_cov``,
        ``normalized_innovation``, and ``log_likelihood``.

    Raises:
        ValueError: if ``particles`` has fewer than 2 ensemble members.
    """
    check_ensemble_size(particles.shape[0])
    obs_particles = jax.vmap(obs_op)(particles)  # (N_e, N_y)
    innovation = obs - ensemble_mean(obs_particles)

    S = innovation_covariance(obs_particles, obs_noise)
    log_prob = log_likelihood(innovation, S, solver=solver)

    # Normalised innovation S^{-1/2} v via Cholesky of the (small) dense S.
    L = jnp.linalg.cholesky(S.as_matrix())
    normalized = jnp.linalg.solve(L, innovation)

    return InnovationStatistics(
        innovation=innovation,
        innovation_cov=S,
        normalized_innovation=normalized,
        log_likelihood=log_prob,
    )

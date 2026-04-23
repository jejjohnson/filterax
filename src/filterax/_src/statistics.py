"""Ensemble statistics primitives.

Pure functions computed on demand from particles ``(N_e, N_x)``. The
Bessel-corrected divisor :math:`1/(N_e-1)` is used throughout for consistency
with the EnKF literature (Evensen 1994, Vetra-Carvalho et al. 2018).

The covariance recipes delegate to :mod:`gaussx` for structure-exploiting
representations (``LowRankUpdate`` of rank ``<= N_e - 1``) and rescale its
``1/N_e`` output to ``1/(N_e - 1)``.
"""

from __future__ import annotations

import gaussx
import jax.numpy as jnp
from einops import reduce
from jaxtyping import Array, Float

from filterax._src._checks import check_ensemble_size


def ensemble_mean(
    particles: Float[Array, "N_e N_x"],
) -> Float[Array, " N_x"]:
    r"""Ensemble mean :math:`\bar{x} = (1/N_e) \sum_j x^{(j)}`."""
    return reduce(particles, "N_e N_x -> N_x", "mean")


def ensemble_anomalies(
    particles: Float[Array, "N_e N_x"],
) -> Float[Array, "N_e N_x"]:
    r"""Centred perturbations :math:`X' = X - \bar{x}`."""
    return particles - ensemble_mean(particles)[None, :]


def ensemble_covariance(
    particles: Float[Array, "N_e N_x"],
) -> gaussx.LowRankUpdate:
    r"""Sample covariance :math:`P = \frac{1}{N_e - 1} X'^T X'` as a low-rank operator.

    Returns a :class:`gaussx.LowRankUpdate` of rank ``<= N_e - 1``; never
    materialises the dense ``(N_x, N_x)`` matrix. Downstream ``solve``/``logdet``
    dispatch exploits the low-rank structure via Woodbury / matrix-determinant
    lemma.

    Args:
        particles: Ensemble of shape ``(N_e, N_x)``.

    Returns:
        :class:`gaussx.LowRankUpdate` representing ``P``.

    Raises:
        ValueError: if ``particles`` has fewer than 2 ensemble members.
    """
    N_e = particles.shape[0]
    check_ensemble_size(N_e)
    # gaussx uses the 1/N_e divisor; rescale the factor by sqrt(N_e/(N_e-1))
    # so the resulting operator represents 1/(N_e-1) X'^T X'.
    cov = gaussx.ensemble_covariance(particles)
    scale = jnp.sqrt(N_e / (N_e - 1))
    return gaussx.LowRankUpdate(cov.base, cov.U * scale)


def cross_covariance(
    particles: Float[Array, "N_e N_x"],
    obs_particles: Float[Array, "N_e N_y"],
) -> Float[Array, "N_x N_y"]:
    r"""Cross-covariance :math:`C^{xH} = \frac{1}{N_e - 1} X'^T (HX)'`.

    Returns a dense ``(N_x, N_y)`` array since ``N_y`` is typically small.
    For nonlinear observation operators, this is the ensemble's implicit,
    derivative-free linearisation of :math:`\nabla H`.

    Raises:
        ValueError: if either ensemble has fewer than 2 members.
    """
    N_e = particles.shape[0]
    check_ensemble_size(N_e)
    # gaussx returns 1/N_e; rescale to 1/(N_e-1).
    return gaussx.ensemble_cross_covariance(particles, obs_particles) * (
        N_e / (N_e - 1)
    )

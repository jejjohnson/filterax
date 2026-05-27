"""Ensemble statistics primitives.

Thin Bessel-corrected wrappers over :mod:`gaussx` recipes
(``gaussx.ensemble_covariance`` and ``gaussx.ensemble_cross_covariance``,
both with ``bessel=True``) so the EnKF convention :math:`1/(N_e-1)` is the
default in filterax. See Evensen 1994 and Vetra-Carvalho et al. 2018.
"""

from __future__ import annotations

import gaussx
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

    Delegates to :func:`gaussx.ensemble_covariance` with the EnKF Bessel
    convention. Returns a :class:`gaussx.LowRankUpdate` of rank ``<= N_e - 1``;
    never materialises the dense ``(N_x, N_x)`` matrix. Downstream
    ``solve``/``logdet`` dispatch exploits the low-rank structure via Woodbury
    and matrix-determinant lemma.

    Args:
        particles: Ensemble of shape ``(N_e, N_x)``.

    Returns:
        :class:`gaussx.LowRankUpdate` representing ``P``.

    Raises:
        ValueError: if ``particles`` has fewer than 2 ensemble members.
    """
    check_ensemble_size(particles.shape[0])
    return gaussx.ensemble_covariance(particles, bessel=True)


def cross_covariance(
    particles: Float[Array, "N_e N_x"],
    obs_particles: Float[Array, "N_e N_y"],
) -> Float[Array, "N_x N_y"]:
    r"""Cross-covariance :math:`C^{xH} = \frac{1}{N_e - 1} X'^T (HX)'`.

    Delegates to :func:`gaussx.ensemble_cross_covariance` with ``bessel=True``.
    Returns a dense ``(N_x, N_y)`` array since ``N_y`` is typically small.

    For nonlinear observation operators, this is the ensemble's implicit,
    derivative-free linearisation of :math:`\nabla H`.

    Raises:
        ValueError: if either ensemble has fewer than 2 members.
    """
    check_ensemble_size(particles.shape[0])
    return gaussx.ensemble_cross_covariance(particles, obs_particles, bessel=True)

"""Ensemble statistics primitives.

Bessel-corrected pure functions over a particle array of shape ``(Nₑ, Nₓ)``.
The covariance recipes delegate to ``gaussx.ensemble_covariance`` and
``gaussx.ensemble_cross_covariance`` (``bessel=True``) so the EnKF
convention ``1 / (Nₑ − 1)`` is the default and structured low-rank outputs
flow downstream without ever materialising a dense ``(Nₓ, Nₓ)`` block.

Symbol conventions (used throughout filterax):

* ``Nₑ`` — ensemble size
* ``Nₓ`` — state dimension
* ``X`` — particle matrix of shape ``(Nₑ, Nₓ)``, rows are members
* ``x̄ = Nₑ⁻¹ Σⱼ x⁽ʲ⁾`` — ensemble mean
* ``X′ = X − 𝟙 x̄ᵀ`` — centred anomaly matrix (rows sum to zero)
* ``P = (Nₑ − 1)⁻¹ X′ᵀ X′`` — sample covariance (rank ≤ Nₑ − 1)

References: Evensen (1994), Vetra-Carvalho et al. (2018).
"""

from __future__ import annotations

import einx
import gaussx
from jaxtyping import Array, Float

from filterax._checks import check_ensemble_size


def ensemble_mean(
    particles: Float[Array, "N_e N_x"],
) -> Float[Array, " N_x"]:
    r"""Ensemble mean ``x̄ = Nₑ⁻¹ Σⱼ x⁽ʲ⁾``.

    Computed as an ``einx`` reduction over the ensemble axis. ``O(Nₑ Nₓ)``.

    Args:
        particles: Ensemble of shape ``(Nₑ, Nₓ)`` with rows as members.

    Returns:
        Mean vector of shape ``(Nₓ,)``.
    """
    return einx.mean("e x -> x", particles)


def ensemble_anomalies(
    particles: Float[Array, "N_e N_x"],
) -> Float[Array, "N_e N_x"]:
    r"""Centred perturbations ``X′ = X − 𝟙 x̄ᵀ``.

    The rows of ``X′`` sum to zero by construction. ``X′`` is the
    fundamental input to the cross-covariance and Kalman-gain recipes:
    every ensemble Kalman filter is some choice of square root applied to
    these anomalies.

    Args:
        particles: Ensemble of shape ``(Nₑ, Nₓ)``.

    Returns:
        Centred anomaly matrix of shape ``(Nₑ, Nₓ)``.
    """
    mean = ensemble_mean(particles)
    return einx.subtract("e x, x -> e x", particles, mean)


def ensemble_covariance(
    particles: Float[Array, "N_e N_x"],
) -> gaussx.LowRankUpdate:
    r"""Sample covariance ``P = (Nₑ − 1)⁻¹ X′ᵀ X′`` as a low-rank operator.

    Delegates to :func:`gaussx.ensemble_covariance` with the EnKF Bessel
    convention. Returns a :class:`gaussx.LowRankUpdate` of rank
    ``≤ Nₑ − 1``; the dense ``(Nₓ, Nₓ)`` matrix is *never* materialised.
    Downstream ``solve`` / ``logdet`` calls exploit the low-rank structure
    via the Woodbury identity and matrix-determinant lemma — see
    :mod:`gaussx` dispatch.

    Complexity: ``O(Nₑ Nₓ)`` to construct; structure-aware solves cost
    ``O(Nₑ² Nₓ + Nₑ³)`` instead of ``O(Nₓ³)``.

    Args:
        particles: Ensemble of shape ``(Nₑ, Nₓ)``.

    Returns:
        :class:`gaussx.LowRankUpdate` representing ``P``.

    Raises:
        ValueError: if ``Nₑ < 2`` (the Bessel divisor is undefined).
    """
    check_ensemble_size(particles.shape[0])
    return gaussx.ensemble_covariance(particles, bessel=True)


def cross_covariance(
    particles: Float[Array, "N_e N_x"],
    obs_particles: Float[Array, "N_e N_y"],
) -> Float[Array, "N_x N_y"]:
    r"""Cross-covariance ``Cˣᴴ = (Nₑ − 1)⁻¹ X′ᵀ (HX)′``.

    Returned as a dense ``(Nₓ, Nᵧ)`` array because ``Nᵧ`` is typically
    small (instrument footprint, point observations). For nonlinear
    observation operators this *is* the ensemble's implicit derivative-free
    linearisation of ``∇H`` — see Evensen (2003) §5.

    Complexity: ``O(Nₑ Nₓ Nᵧ)``.

    Args:
        particles: Prior ensemble in state space, shape ``(Nₑ, Nₓ)``.
        obs_particles: Prior ensemble mapped to obs space, shape
            ``(Nₑ, Nᵧ)``.

    Returns:
        Dense cross-covariance of shape ``(Nₓ, Nᵧ)``.

    Raises:
        ValueError: if ``Nₑ < 2``.
    """
    check_ensemble_size(particles.shape[0])
    return gaussx.ensemble_cross_covariance(particles, obs_particles, bessel=True)

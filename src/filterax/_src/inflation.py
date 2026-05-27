"""Ensemble inflation primitives.

Counteract the systematic underestimation of uncertainty that occurs with
finite ensembles. Without inflation, the spread collapses over repeated
assimilation cycles, the filter becomes overconfident, and observations are
rejected (filter divergence).
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from filterax._src.statistics import ensemble_anomalies, ensemble_mean


def inflate_multiplicative(
    particles: Float[Array, "N_e N_x"],
    factor: float,
) -> Float[Array, "N_e N_x"]:
    r"""Multiplicative inflation.

    .. math::

        x^{(j)}_{\text{inflated}} = \bar{x} + \lambda (x^{(j)} - \bar{x})

    Equivalently :math:`X' \leftarrow \lambda X'`; the inflated covariance is
    :math:`\lambda^2 P`. Typical values ``factor in [1.01, 1.10]``.

    Args:
        particles: Ensemble of shape ``(N_e, N_x)``.
        factor: Multiplicative factor. ``1.0`` is identity; ``> 1`` inflates.

    Returns:
        Inflated ensemble of shape ``(N_e, N_x)``.

    Reference:
        Anderson, J. L. & Anderson, S. L. (1999). *A Monte Carlo
        implementation of the nonlinear filtering problem.* Mon. Wea. Rev.,
        127, 2741-2758.
    """
    mean = ensemble_mean(particles)
    return mean[None, :] + factor * (particles - mean[None, :])


def inflate_rtps(
    analysis_particles: Float[Array, "N_e N_x"],
    forecast_particles: Float[Array, "N_e N_x"],
    alpha: float,
) -> Float[Array, "N_e N_x"]:
    r"""Relaxation to Prior Spread (Whitaker & Hamill 2012).

    .. math::

        \sigma_{\text{relaxed},i} = (1 - \alpha) \sigma^a_i + \alpha \sigma^f_i

        x^{(j)}_{\text{relaxed}} = \bar{x}^a
            + \frac{\sigma_{\text{relaxed},i}}{\sigma^a_i} (x^{(j)}_a - \bar{x}^a)

    Per-variable inflation — spatially adaptive — that preserves the analysis
    mean. With ``alpha = 0`` the analysis is unchanged; with ``alpha = 1`` the
    full forecast spread is restored.

    Args:
        analysis_particles: Posterior ensemble ``(N_e, N_x)``.
        forecast_particles: Prior (forecast) ensemble ``(N_e, N_x)``.
        alpha: Relaxation coefficient in ``[0, 1]``.

    Returns:
        Relaxed analysis ensemble of shape ``(N_e, N_x)``.

    Reference:
        Whitaker, J. S. & Hamill, T. M. (2012). *Evaluating methods to
        account for system errors in ensemble data assimilation.* Mon. Wea.
        Rev., 140, 3078-3089.
    """
    sigma_a = analysis_particles.std(axis=0, ddof=1)
    sigma_f = forecast_particles.std(axis=0, ddof=1)
    sigma_target = (1.0 - alpha) * sigma_a + alpha * sigma_f
    # Avoid divide-by-zero where the analysis already collapsed in some
    # direction; leave that direction at its analysis spread (factor 1).
    scale = jnp.where(
        sigma_a > 0, sigma_target / jnp.where(sigma_a > 0, sigma_a, 1.0), 1.0
    )

    mean_a = ensemble_mean(analysis_particles)
    anomalies = analysis_particles - mean_a[None, :]
    return mean_a[None, :] + scale[None, :] * anomalies


def inflate_rtpp(
    analysis_particles: Float[Array, "N_e N_x"],
    forecast_particles: Float[Array, "N_e N_x"],
    alpha: float,
) -> Float[Array, "N_e N_x"]:
    r"""Relaxation to Prior Perturbations (Zhang et al. 2004).

    .. math::

        X'_{\text{relaxed}} = (1 - \alpha) X'_a + \alpha X'_f

    Interpolates the perturbation matrix directly — unlike RTPS this modifies
    inter-variable correlation structure, mixing it with the forecast.

    Args:
        analysis_particles: Posterior ensemble ``(N_e, N_x)``.
        forecast_particles: Prior (forecast) ensemble ``(N_e, N_x)``.
        alpha: Relaxation coefficient in ``[0, 1]``.

    Returns:
        Relaxed analysis ensemble of shape ``(N_e, N_x)``.

    Reference:
        Zhang, F., Snyder, C., & Sun, J. (2004). *Impacts of initial estimate
        and observation availability on convective-scale data assimilation.*
        Mon. Wea. Rev., 132, 1238-1253.
    """
    mean_a = ensemble_mean(analysis_particles)
    anom_a = ensemble_anomalies(analysis_particles)
    anom_f = ensemble_anomalies(forecast_particles)
    return mean_a[None, :] + (1.0 - alpha) * anom_a + alpha * anom_f

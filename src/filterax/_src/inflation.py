"""Ensemble inflation primitives.

Counteract the systematic spread collapse of finite-ensemble Kalman
filters. Three compounding effects underdisperse a raw ensemble: finite
``Nₑ``, model error not represented by the ensemble dynamics, and
localization side-effects. Without inflation the analysis grows
overconfident and rejects observations — *filter divergence*.

All three primitives here are pure functions of the ensemble matrix.
The :class:`filterax.AbstractInflator` wrappers in
``filterax._src.inflators`` lift them to module form for slotting into
L2 assimilation loops.
"""

from __future__ import annotations

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float

from filterax._src.statistics import ensemble_anomalies, ensemble_mean


def inflate_multiplicative(
    particles: Float[Array, "N_e N_x"],
    factor: float,
) -> Float[Array, "N_e N_x"]:
    r"""Multiplicative inflation ``X′ ← λ X′``.

    For each member ``x⁽ʲ⁾_inflated = x̄ + λ (x⁽ʲ⁾ − x̄)``. The inflated
    covariance is ``λ² P``; the analysis mean is preserved. Typical
    values ``λ ∈ [1.01, 1.10]`` for operational NWP.

    Args:
        particles: Ensemble of shape ``(Nₑ, Nₓ)``.
        factor: Inflation factor ``λ``. ``1.0`` is the identity;
            ``λ > 1`` inflates spread; ``λ < 1`` deflates.

    Returns:
        Inflated ensemble of shape ``(Nₑ, Nₓ)``.

    Reference:
        Anderson, J. L. & Anderson, S. L. (1999). *A Monte Carlo
        implementation of the nonlinear filtering problem.* Mon. Wea.
        Rev., 127, 2741–2758.
    """
    mean = ensemble_mean(particles)
    anom = einx.subtract("e x, x -> e x", particles, mean)
    return einx.add("e x, x -> e x", factor * anom, mean)


def inflate_rtps(
    analysis_particles: Float[Array, "N_e N_x"],
    forecast_particles: Float[Array, "N_e N_x"],
    alpha: float,
) -> Float[Array, "N_e N_x"]:
    r"""Relaxation to Prior Spread (Whitaker & Hamill 2012).

    Per-variable target spread is a convex blend of analysis and
    forecast standard deviations:

    ``σ_target,i = (1 − α) σᵃ_i + α σᶠ_i``

    Each analysis anomaly is then rescaled so the resulting standard
    deviation matches ``σ_target``:

    ``x⁽ʲ⁾_relaxed = x̄ᵃ + (σ_target,i / σᵃ_i) (x⁽ʲ⁾_a − x̄ᵃ)``

    Spatially adaptive (per-variable factor), mean-preserving, and
    parameterised by a single coefficient ``α``. ``α = 0`` is the
    identity; ``α = 1`` restores the full forecast spread.

    Args:
        analysis_particles: Posterior ensemble ``(Nₑ, Nₓ)``.
        forecast_particles: Prior (forecast) ensemble ``(Nₑ, Nₓ)``.
        alpha: Relaxation coefficient in ``[0, 1]``.

    Returns:
        Relaxed analysis ensemble of shape ``(Nₑ, Nₓ)``.

    Reference:
        Whitaker, J. S. & Hamill, T. M. (2012). *Evaluating methods to
        account for system errors in ensemble data assimilation.*
        Mon. Wea. Rev., 140, 3078–3089.
    """
    sigma_a = analysis_particles.std(axis=0, ddof=1)
    sigma_f = forecast_particles.std(axis=0, ddof=1)
    sigma_target = (1.0 - alpha) * sigma_a + alpha * sigma_f
    # Guard against directions where the analysis already collapsed
    # (σᵃ_i = 0). The relaxation factor is undefined there; we leave the
    # ensemble untouched (effective scale 1).
    scale = jnp.where(
        sigma_a > 0,
        sigma_target / jnp.where(sigma_a > 0, sigma_a, 1.0),
        1.0,
    )

    mean_a = ensemble_mean(analysis_particles)
    anomalies = einx.subtract("e x, x -> e x", analysis_particles, mean_a)
    rescaled = einx.multiply("e x, x -> e x", anomalies, scale)
    return einx.add("e x, x -> e x", rescaled, mean_a)


def inflate_rtpp(
    analysis_particles: Float[Array, "N_e N_x"],
    forecast_particles: Float[Array, "N_e N_x"],
    alpha: float,
) -> Float[Array, "N_e N_x"]:
    r"""Relaxation to Prior Perturbations (Zhang et al. 2004).

    Convex combination of analysis and forecast anomaly matrices:

    ``X′_relaxed = (1 − α) X′_a + α X′_f``

    Unlike RTPS (which acts on per-variable spread), RTPP operates on
    the full anomaly matrix and so blends inter-variable correlation
    structure between forecast and analysis. Mean-preserving.

    Args:
        analysis_particles: Posterior ensemble ``(Nₑ, Nₓ)``.
        forecast_particles: Prior (forecast) ensemble ``(Nₑ, Nₓ)``.
        alpha: Relaxation coefficient in ``[0, 1]``.

    Returns:
        Relaxed analysis ensemble of shape ``(Nₑ, Nₓ)``.

    Reference:
        Zhang, F., Snyder, C., & Sun, J. (2004). *Impacts of initial
        estimate and observation availability on convective-scale data
        assimilation.* Mon. Wea. Rev., 132, 1238–1253.
    """
    mean_a = ensemble_mean(analysis_particles)
    anom_a = ensemble_anomalies(analysis_particles)
    anom_f = ensemble_anomalies(forecast_particles)
    blended = (1.0 - alpha) * anom_a + alpha * anom_f
    return einx.add("e x, x -> e x", blended, mean_a)

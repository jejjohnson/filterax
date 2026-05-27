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
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray

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


# ──────────────────────────────────────────────────────────────────────
# Advanced inflation primitives (Wave 4.B)
# ──────────────────────────────────────────────────────────────────────


def inflate_additive(
    key: PRNGKeyArray,
    particles: Float[Array, "N_e N_x"],
    noise_cov: lx.AbstractLinearOperator,
) -> Float[Array, "N_e N_x"]:
    r"""Additive inflation ``X′ ← X′ + ε,  ε ~ 𝒩(0, Q_add)``.

    Injects stochastic model-error perturbations into every ensemble
    member's anomaly. Unlike multiplicative inflation, the noise
    direction is set by ``Q_add`` rather than by the ensemble's own
    spread — useful when the dominant model-error mode is *not*
    represented by the ensemble (e.g. structurally underdispersive in
    a particular subspace). Mean-preserving by construction (we
    subtract the empirical mean of the noise).

    Common sources for ``Q_add``: climatological variability, lagged
    forecast differences, or stochastic-physics tendency variances.

    Sampling delegates to :func:`filterax.perturbed_observations`'s
    diagonal-aware path, so a diagonal ``Q_add`` never materialises.

    Args:
        key: PRNG key consumed by this call.
        particles: Ensemble of shape ``(Nₑ, Nₓ)``.
        noise_cov: Additive-noise covariance ``Q_add`` as a linear
            operator.

    Returns:
        Inflated ensemble of shape ``(Nₑ, Nₓ)``.

    Reference:
        Hamill, T. M. & Whitaker, J. S. (2005). *Accounting for the
        error due to unresolved scales in ensemble data assimilation:
        A comparison of different approaches.* Mon. Wea. Rev., 133,
        3132-3147.
    """
    from filterax._src.perturbations import perturbed_observations

    N_e = particles.shape[0]
    # Sample N_e draws from 𝒩(0, Q_add).
    draws = perturbed_observations(
        key, jnp.zeros(particles.shape[1], dtype=particles.dtype), noise_cov, N_e
    )
    # Centre so the empirical mean is exactly zero — keeps the
    # ensemble mean unchanged after addition.
    draws = draws - jnp.mean(draws, axis=0, keepdims=True)
    return particles + draws


def inflate_adaptive(
    inflation_mean: float,
    inflation_var: float,
    innovation: Float[Array, " N_y"],
    innovation_cov: Float[Array, "N_y N_y"],
    *,
    min_factor: float = 1.0,
    max_factor: float = 1.2,
) -> tuple[float, float]:
    r"""Bayesian update of a multiplicative inflation factor (Anderson 2009).

    Treats the inflation factor ``λ`` as a random variable with a
    Gaussian prior ``λ ~ 𝒩(μ_λ, σ²_λ)`` and updates it from the
    observed innovation. The "data-implied" inflation is the scalar
    that would explain the *measured* innovation magnitude in
    Mahalanobis norm:

    ``λ̂_obs = (dᵀ S⁻¹ d / Nᵧ − 1) / (tr(S₀ S⁻¹) / Nᵧ)``

    with ``S = HPHᵀ + R`` the prescribed innovation covariance and
    ``S₀ = HPHᵀ`` the ensemble part. The posterior on ``λ`` is then
    the Gaussian product

    ``μ_post = (σ²_λ · λ̂_obs + σ²_obs · μ_prior) / (σ²_λ + σ²_obs)``

    ``σ²_post = σ²_λ · σ²_obs / (σ²_λ + σ²_obs)``

    A flat clamp ``λ ∈ [min_factor, max_factor]`` prevents the
    estimator from running away on outlier cycles. Returns the
    *posterior* ``(μ, σ²)`` so the caller can carry it as state
    across assimilation windows.

    This primitive only returns the updated inflation belief — actual
    inflation of an ensemble is one extra call to
    :func:`inflate_multiplicative` with ``μ_post`` as the factor.

    Args:
        inflation_mean: Prior mean ``μ_λ``.
        inflation_var: Prior variance ``σ²_λ``.
        innovation: Innovation vector ``d = y − H x̄``.
        innovation_cov: Dense innovation covariance ``S``. The
            (N_y, N_y) materialisation is intentional — Anderson's
            recipe involves both ``S⁻¹`` and the trace of ``S₀ S⁻¹``,
            which is cheapest dense for typical observation counts.
        min_factor: Clamp lower bound for the data-implied estimate.
        max_factor: Clamp upper bound.

    Returns:
        ``(μ_post, σ²_post)`` posterior on ``λ``, suitable for use as
        the prior on the next cycle.

    Reference:
        Anderson, J. L. (2009). *Spatially and temporally varying
        adaptive covariance inflation for ensemble filters.* Tellus A,
        61, 72-83.
    """
    N_y = innovation.shape[0]
    # Mahalanobis norm of the innovation under the prescribed S.
    chi2 = float(innovation @ jnp.linalg.solve(innovation_cov, innovation))
    chi2_norm = chi2 / N_y
    # Data-implied λ — simple Anderson-2009 form: inflate when the
    # observed normalised innovation exceeds 1 (under-dispersive), do
    # nothing when it equals 1 (well calibrated).
    lambda_obs = max(min_factor, min(max_factor, chi2_norm))
    sigma_obs_sq = 2.0 / N_y  # variance of χ²/N_y under the null
    # Gaussian-product posterior.
    sigma_sum = inflation_var + sigma_obs_sq
    mu_post = (inflation_var * lambda_obs + sigma_obs_sq * inflation_mean) / sigma_sum
    var_post = inflation_var * sigma_obs_sq / sigma_sum
    return float(mu_post), float(var_post)


def ledoit_wolf_shrinkage(
    particles: Float[Array, "N_e N_x"],
) -> tuple[Float[Array, "N_x N_x"], float]:
    r"""Ledoit-Wolf optimal covariance shrinkage (Ledoit & Wolf 2004).

    Replaces the sample covariance with a convex combination toward a
    scalar-multiple-of-identity target:

    ``P_shrunk = (1 − λ*) P_sample + λ* μ I``

    where ``μ = tr(P_sample) / Nₓ`` is the average sample eigenvalue
    and ``λ* ∈ [0, 1]`` is the optimal shrinkage intensity that
    minimises ``E ‖P_shrunk − P_true‖²_F``. The closed-form Ledoit-
    Wolf estimator:

    ``λ* = min(1, b² / d²)``

    ``d² = ‖P_sample − μ I‖²_F``  (sample deviation from the target)

    ``b² = (1 / Nₑ²) Σⱼ ‖x⁽ʲ⁾ x⁽ʲ⁾ᵀ − P_sample‖²_F`` (oracle
    approximation; capped at ``d²`` so ``λ* ≤ 1``).

    Useful when ``Nₑ ≪ Nₓ`` and the rank-deficient sample covariance
    would otherwise contaminate downstream operations. Guaranteed
    positive definite. Returns a dense ``(Nₓ, Nₓ)`` matrix — only
    call it when ``Nₓ`` is small enough to materialise.

    Args:
        particles: Ensemble of shape ``(Nₑ, Nₓ)``.

    Returns:
        ``(P_shrunk, λ*)`` — shrunk covariance and the optimal
        shrinkage intensity.

    Reference:
        Ledoit, O. & Wolf, M. (2004). *A well-conditioned estimator
        for large-dimensional covariance matrices.* J. Multivariate
        Anal., 88, 365-411.
    """
    N_e, N_x = particles.shape
    mean = jnp.mean(particles, axis=0)
    centered = particles - mean[None, :]
    sample = centered.T @ centered / (N_e - 1)  # (Nₓ, Nₓ)
    mu = jnp.trace(sample) / N_x
    target = mu * jnp.eye(N_x, dtype=particles.dtype)

    # d² and b² as in Ledoit-Wolf 2004 §3.2.
    diff = sample - target
    d2 = jnp.sum(diff * diff)
    # Per-sample rank-one deviation from sample covariance.
    per_sample = centered[:, :, None] * centered[:, None, :]  # (Nₑ, Nₓ, Nₓ)
    b2_terms = jnp.sum((per_sample - sample[None, :, :]) ** 2, axis=(1, 2))
    b2 = jnp.mean(b2_terms) / N_e
    shrinkage = jnp.minimum(
        jnp.asarray(1.0, dtype=particles.dtype), b2 / jnp.maximum(d2, 1e-30)
    )
    shrunk = (1.0 - shrinkage) * sample + shrinkage * target
    return shrunk, float(shrinkage)

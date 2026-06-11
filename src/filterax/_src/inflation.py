"""Ensemble inflation primitives.

Counteract the systematic spread collapse of finite-ensemble Kalman
filters. Three compounding effects underdisperse a raw ensemble: finite
``Nₑ``, model error not represented by the ensemble dynamics, and
localization side-effects. Without inflation the analysis grows
overconfident and rejects observations — *filter divergence*.

All primitives here are pure functions of the ensemble matrix. The
classic trio (multiplicative, RTPS, RTPP) delegates to the gaussx
ensemble-DA primitives; the advanced primitives (additive, adaptive,
Ledoit-Wolf) are filterax-specific. The
:class:`filterax.AbstractInflator` wrappers in
``filterax._src.inflators`` lift them to module form for slotting into
L2 assimilation loops.
"""

from __future__ import annotations

import gaussx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray

from filterax._src._checks import check_ensemble_size


def inflate_multiplicative(
    particles: Float[Array, "N_e N_x"],
    factor: float,
) -> Float[Array, "N_e N_x"]:
    r"""Multiplicative inflation ``X′ ← λ X′``.

    For each member ``x⁽ʲ⁾_inflated = x̄ + λ (x⁽ʲ⁾ − x̄)``. The inflated
    covariance is ``λ² P``; the analysis mean is preserved. Typical
    values ``λ ∈ [1.01, 1.10]`` for operational NWP. Delegates to
    :func:`gaussx.inflate_multiplicative`.

    Args:
        particles: Ensemble of shape ``(Nₑ, Nₓ)``.
        factor: Inflation factor ``λ``. ``1.0`` is the identity;
            ``λ > 1`` inflates spread; ``λ < 1`` deflates.

    Returns:
        Inflated ensemble of shape ``(Nₑ, Nₓ)``.

    Example:
        >>> import jax.numpy as jnp
        >>> from filterax import inflate_multiplicative
        >>> ens = jnp.array([[0.0, 0.0], [2.0, 2.0]])
        >>> inflate_multiplicative(ens, 1.5)  # mean (1, 1) preserved
        Array([[-0.5, -0.5],
               [ 2.5,  2.5]], dtype=float32)

    Reference:
        Anderson, J. L. & Anderson, S. L. (1999). *A Monte Carlo
        implementation of the nonlinear filtering problem.* Mon. Wea.
        Rev., 127, 2741–2758.
    """
    return gaussx.inflate_multiplicative(particles, factor)


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
    identity; ``α = 1`` restores the full forecast spread. Delegates to
    :func:`gaussx.inflate_rtps` (which calls the coefficient ``beta``
    and writes the equivalent form
    ``scale = (1 − β) + β σᶠ/σᵃ``).

    Args:
        analysis_particles: Posterior ensemble ``(Nₑ, Nₓ)``.
        forecast_particles: Prior (forecast) ensemble ``(Nₑ, Nₓ)``.
        alpha: Relaxation coefficient in ``[0, 1]``.

    Returns:
        Relaxed analysis ensemble of shape ``(Nₑ, Nₓ)``.

    Example:
        >>> import jax.numpy as jnp
        >>> from filterax import inflate_rtps
        >>> forecast = jnp.array([[-2.0], [2.0]])
        >>> analysis = jnp.array([[-1.0], [1.0]])
        >>> inflate_rtps(analysis, forecast, alpha=1.0)  # full prior spread
        Array([[-2.],
               [ 2.]], dtype=float32)

    Reference:
        Whitaker, J. S. & Hamill, T. M. (2012). *Evaluating methods to
        account for system errors in ensemble data assimilation.*
        Mon. Wea. Rev., 140, 3078–3089.
    """
    return gaussx.inflate_rtps(analysis_particles, forecast_particles, alpha)


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
    Delegates to :func:`gaussx.inflate_rtpp`.

    Args:
        analysis_particles: Posterior ensemble ``(Nₑ, Nₓ)``.
        forecast_particles: Prior (forecast) ensemble ``(Nₑ, Nₓ)``.
        alpha: Relaxation coefficient in ``[0, 1]``.

    Returns:
        Relaxed analysis ensemble of shape ``(Nₑ, Nₓ)``.

    Example:
        >>> import jax.numpy as jnp
        >>> from filterax import inflate_rtpp
        >>> forecast = jnp.array([[-2.0], [2.0]])
        >>> analysis = jnp.array([[-1.0], [1.0]])
        >>> inflate_rtpp(analysis, forecast, alpha=0.5)  # blend anomalies
        Array([[-1.5],
               [ 1.5]], dtype=float32)

    Reference:
        Zhang, F., Snyder, C., & Sun, J. (2004). *Impacts of initial
        estimate and observation availability on convective-scale data
        assimilation.* Mon. Wea. Rev., 132, 1238–1253.
    """
    return gaussx.inflate_rtpp(analysis_particles, forecast_particles, alpha)


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
    inflation_mean: Float[Array, ""] | float,
    inflation_var: Float[Array, ""] | float,
    innovation: Float[Array, " N_y"],
    innovation_cov: Float[Array, "N_y N_y"],
    *,
    min_factor: float = 1.0,
    max_factor: float = 1.2,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    r"""Bayesian update of a multiplicative inflation factor (Anderson 2009).

    Treats the inflation factor ``λ`` as a random variable with a
    Gaussian prior ``λ ~ 𝒩(μ_λ, σ²_λ)`` and updates it from the
    cycle's *normalised innovation*. The data-implied factor is the
    clamped Mahalanobis ratio

    ``λ̂_obs = clip(dᵀ S⁻¹ d / Nᵧ,  min_factor,  max_factor)``

    (the simplified Anderson-2009 form — under correct specification
    ``E[χ²/Nᵧ] = 1``; under-dispersive ensembles overshoot and pull
    ``λ̂_obs`` above 1). The posterior on ``λ`` is then the Gaussian
    product of the prior with an "observation likelihood" centred at
    ``λ̂_obs`` whose variance is the χ²-distribution variance
    ``σ²_obs = 2 / Nᵧ``:

    ``μ_post = (σ²_λ · λ̂_obs + σ²_obs · μ_prior) / (σ²_λ + σ²_obs)``

    ``σ²_post = σ²_λ · σ²_obs / (σ²_λ + σ²_obs)``

    Returns the *posterior* ``(μ, σ²)`` as **JAX scalars** so callers
    can carry the belief through ``jax.jit`` / ``jax.grad`` / ``lax.scan``
    state across assimilation windows. Actual inflation of an ensemble
    is one extra call to :func:`inflate_multiplicative` with ``μ_post``
    as the factor.

    Args:
        inflation_mean: Prior mean ``μ_λ`` (scalar or 0-d JAX array).
        inflation_var: Prior variance ``σ²_λ`` (scalar or 0-d JAX array).
        innovation: Innovation vector ``d = y − H x̄``.
        innovation_cov: Dense innovation covariance ``S``.
            ``(Nᵧ, Nᵧ)`` materialisation is acceptable because the
            recipe needs ``S⁻¹`` applied to ``d`` — for typical
            observation counts the dense solve is cheaper than
            assembling a structured operator. Pass the dense form when
            you have it; otherwise build it via ``S_op.as_matrix()`` at
            the call site.
        min_factor: Clamp lower bound for the data-implied estimate.
        max_factor: Clamp upper bound.

    Returns:
        ``(μ_post, σ²_post)`` JAX-scalar posterior on ``λ``, suitable
        for use as the prior on the next cycle.

    Reference:
        Anderson, J. L. (2009). *Spatially and temporally varying
        adaptive covariance inflation for ensemble filters.* Tellus A,
        61, 72-83.
    """
    N_y = innovation.shape[0]
    chi2 = innovation @ jnp.linalg.solve(innovation_cov, innovation)
    chi2_norm = chi2 / N_y
    # Data-implied λ; clamped to keep the estimator from running away
    # on outlier innovations.
    lambda_obs = jnp.clip(chi2_norm, min_factor, max_factor)
    sigma_obs_sq = jnp.asarray(2.0 / N_y, dtype=chi2_norm.dtype)
    prior_mean = jnp.asarray(inflation_mean, dtype=chi2_norm.dtype)
    prior_var = jnp.asarray(inflation_var, dtype=chi2_norm.dtype)
    sigma_sum = prior_var + sigma_obs_sq
    mu_post = (prior_var * lambda_obs + sigma_obs_sq * prior_mean) / sigma_sum
    var_post = prior_var * sigma_obs_sq / sigma_sum
    return mu_post, var_post


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
    would otherwise contaminate downstream operations. **Positive
    semidefinite** by construction (PSD, not strictly PD — when the
    ensemble is collapsed in some direction both the sample covariance
    and the target ``μ I`` are singular there). Returns a dense
    ``(Nₓ, Nₓ)`` matrix — only call it when ``Nₓ`` is small enough to
    materialise.

    Args:
        particles: Ensemble of shape ``(Nₑ, Nₓ)``.

    Returns:
        ``(P_shrunk, λ*)`` — shrunk covariance and the optimal
        shrinkage intensity.

    Raises:
        ValueError: if ``Nₑ < 2`` (the Bessel divisor is undefined).

    Reference:
        Ledoit, O. & Wolf, M. (2004). *A well-conditioned estimator
        for large-dimensional covariance matrices.* J. Multivariate
        Anal., 88, 365-411.
    """
    N_e, N_x = particles.shape
    check_ensemble_size(N_e)
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

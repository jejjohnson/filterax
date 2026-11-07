"""Inflation primitives: pure functions that restore the spread of an ensemble.

Finite N_ℰ, unrepresented model error and localization all underdisperse an ensemble,
and without inflation the analysis grows overconfident and rejects observations. The
modules in `filterax._primitives._inflators` wrap these for the assimilation loops.
"""

from __future__ import annotations

import gaussx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray

from filterax._checks import check_ensemble_size
from filterax._primitives._perturbations import perturbed_observations
from filterax._primitives._statistics import ensemble_anomalies


def inflate_multiplicative(
    particles: Float[Array, "N_e N_x"],
    factor: float,
) -> Float[Array, "N_e N_x"]:
    """Multiplicative inflation about the sample mean.

        x⁽ᵉ⁾ ← x̄ + ζ (x⁽ᵉ⁾ − x̄)

    Wraps `gaussx.inflate_multiplicative`. The mean is preserved and Σ̂_x becomes
    ζ² Σ̂_x. Values ζ ∈ [1.01, 1.10] are typical for operational NWP.

    Args:
        particles: Ensemble members x⁽ᵉ⁾, one row per member.
        factor: Inflation factor ζ. 1 is the identity, ζ > 1 inflates the spread and
            ζ < 1 deflates it.

    Returns:
        The inflated ensemble, with the same sample mean.

    Examples:
        The mean (1, 1) is preserved:

        >>> import jax.numpy as jnp
        >>> import filterax
        >>> particles = jnp.array([[0.0, 0.0], [2.0, 2.0]])
        >>> filterax.inflate_multiplicative(particles, 1.5).tolist()
        [[-0.5, -0.5], [2.5, 2.5]]

        A factor of one changes nothing:

        >>> inflated = filterax.inflate_multiplicative(particles, 1.0)
        >>> bool(jnp.allclose(inflated, particles))
        True

    References:
        Anderson & Anderson (1999), *Mon. Wea. Rev.* 127, 2741–2758.
    """
    # x⁽ᵉ⁾ ← x̄ + ζ (x⁽ᵉ⁾ − x̄)
    return gaussx.inflate_multiplicative(particles, factor)


def inflate_rtps(
    analysis_particles: Float[Array, "N_e N_x"],
    forecast_particles: Float[Array, "N_e N_x"],
    alpha: float,
) -> Float[Array, "N_e N_x"]:
    """Relaxation to prior spread (RTPS).

        x̃⁽ᵉ⁾_t|t ← x̃⁽ᵉ⁾_t|t ((1 − α) + α σ_x,t|t−1 / σ_x,t|t)

    Scales each analysis anomaly, per coordinate, so that the analysis spread relaxes
    toward the forecast spread, keeping the analysis mean. Wraps `gaussx.inflate_rtps`,
    which calls the coefficient β, takes σ_x with the divisor N_ℰ, and floors the
    denominator at 1e-12.

    Algorithm:
        1. Take the analysis sample mean and anomalies.
        2. Take the per-coordinate spread of both ensembles.
        3. Scale the anomalies and add the analysis mean back.

    Args:
        analysis_particles: Analysis ensemble x⁽ᵉ⁾_t|t.
        forecast_particles: Forecast ensemble x⁽ᵉ⁾_t|t−1.
        alpha: Relaxation coefficient α ∈ [0, 1]. 0 is the identity, and 1 restores
            the full forecast spread.

    Returns:
        The relaxed analysis ensemble, with the analysis sample mean.

    Examples:
        A coefficient of one restores the forecast spread:

        >>> import jax.numpy as jnp
        >>> import filterax
        >>> forecast = jnp.array([[-2.0], [2.0]])
        >>> analysis = jnp.array([[-1.0], [1.0]])
        >>> relaxed = filterax.inflate_rtps(analysis, forecast, alpha=1.0)
        >>> bool(jnp.allclose(relaxed, forecast))
        True

        A coefficient of zero keeps the analysis:

        >>> relaxed = filterax.inflate_rtps(analysis, forecast, alpha=0.0)
        >>> bool(jnp.allclose(relaxed, analysis))
        True

    References:
        Whitaker & Hamill (2012), *Mon. Wea. Rev.* 140, 3078–3089.
    """
    # x̃⁽ᵉ⁾_t|t ← x̃⁽ᵉ⁾_t|t ((1 − α) + α σ_x,t|t−1 / σ_x,t|t)
    return gaussx.inflate_rtps(analysis_particles, forecast_particles, alpha)


def inflate_rtpp(
    analysis_particles: Float[Array, "N_e N_x"],
    forecast_particles: Float[Array, "N_e N_x"],
    alpha: float,
) -> Float[Array, "N_e N_x"]:
    """Relaxation to prior perturbations (RTPP).

        x̃⁽ᵉ⁾_t|t ← (1 − α) x̃⁽ᵉ⁾_t|t + α x̃⁽ᵉ⁾_t|t−1

    Blends the analysis anomalies with the forecast ones, keeping the analysis mean.
    Unlike RTPS, which acts on the spread of each coordinate, RTPP acts on the whole
    anomalies, so it also blends the correlations between coordinates. Wraps
    `gaussx.inflate_rtpp`.

    Algorithm:
        1. Take the anomalies of each ensemble about its own sample mean.
        2. Blend them with weights 1 − α and α.
        3. Add the blend to the analysis sample mean.

    Args:
        analysis_particles: Analysis ensemble x⁽ᵉ⁾_t|t.
        forecast_particles: Forecast ensemble x⁽ᵉ⁾_t|t−1.
        alpha: Relaxation coefficient α ∈ [0, 1].

    Returns:
        The relaxed analysis ensemble, with the analysis sample mean.

    Examples:
        Halfway back to the forecast anomalies:

        >>> import jax.numpy as jnp
        >>> import filterax
        >>> forecast = jnp.array([[-2.0], [2.0]])
        >>> analysis = jnp.array([[-1.0], [1.0]])
        >>> filterax.inflate_rtpp(analysis, forecast, alpha=0.5).tolist()
        [[-1.5], [1.5]]

        A coefficient of zero keeps the analysis:

        >>> relaxed = filterax.inflate_rtpp(analysis, forecast, alpha=0.0)
        >>> bool(jnp.allclose(relaxed, analysis))
        True

    References:
        Zhang, Snyder & Sun (2004), *Mon. Wea. Rev.* 132, 1238–1253.
    """
    # x̃⁽ᵉ⁾_t|t ← (1 − α) x̃⁽ᵉ⁾_t|t + α x̃⁽ᵉ⁾_t|t−1
    return gaussx.inflate_rtpp(analysis_particles, forecast_particles, alpha)


def inflate_additive(
    key: PRNGKeyArray,
    particles: Float[Array, "N_e N_x"],
    noise_cov: lx.AbstractLinearOperator,
) -> Float[Array, "N_e N_x"]:
    """Additive inflation with centred model noise.

        x⁽ᵉ⁾ ← x⁽ᵉ⁾ + η⁽ᵉ⁾ − η̄,   η⁽ᵉ⁾ ~ 𝒩(0, Σ_η)

    Unlike multiplicative inflation, the direction of the noise comes from Σ_η rather
    than from the ensemble's own spread, which helps when the dominant model-error mode
    is not represented in the ensemble. Common sources of Σ_η are climatological
    variability, lagged forecast differences and stochastic-physics tendencies.

    Subtracting the sample mean η̄ of the noise preserves the ensemble mean exactly,
    so a single member is returned unchanged. Sampling goes through
    `perturbed_observations`, so a diagonal Σ_η is never materialised.

    Algorithm:
        1. Draw one η⁽ᵉ⁾ per member.
        2. Centre the draws on their sample mean η̄.
        3. Add them to the members.

    Args:
        key: PRNG key consumed by this call.
        particles: Ensemble members x⁽ᵉ⁾, one row per member.
        noise_cov: Model-error covariance Σ_η.

    Returns:
        The inflated ensemble, with the same sample mean.

    Raises:
        ValueError: If N_ℰ = 0, or ``noise_cov`` does not match D_x.

    Examples:
        Noise spreads a collapsed ensemble and keeps its mean:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import filterax
        >>> particles = jnp.ones((4, 2))
        >>> noise_cov = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> inflated = filterax.inflate_additive(jr.key(0), particles, noise_cov)
        >>> inflated.shape
        (4, 2)
        >>> bool(jnp.allclose(filterax.ensemble_mean(inflated), 1.0))
        True
        >>> bool(jnp.all(jnp.std(inflated, axis=0) > 0.0))
        True

        A single member is unchanged, since its noise is its own mean:

        >>> single = filterax.inflate_additive(jr.key(0), particles[:1], noise_cov)
        >>> single.tolist()
        [[1.0, 1.0]]

    References:
        Hamill & Whitaker (2005), *Mon. Wea. Rev.* 133, 3132-3147.
    """
    ensemble_size = particles.shape[0]
    state_size = particles.shape[1]
    zero_mean = jnp.zeros(state_size, dtype=particles.dtype)
    # η⁽ᵉ⁾ ~ 𝒩(0, Σ_η)
    noise = perturbed_observations(key, zero_mean, noise_cov, ensemble_size)
    # η⁽ᵉ⁾ − η̄, so that the ensemble mean does not move
    centred_noise = ensemble_anomalies(noise)
    # x⁽ᵉ⁾ ← x⁽ᵉ⁾ + η⁽ᵉ⁾ − η̄
    return particles + centred_noise


def inflate_adaptive(
    inflation_mean: Float[Array, ""] | float,
    inflation_var: Float[Array, ""] | float,
    innovation: Float[Array, " N_y"],
    innovation_cov: Float[Array, "N_y N_y"],
    *,
    min_factor: float = 1.0,
    max_factor: float = 1.2,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Bayesian update of a multiplicative inflation factor.

        ζ̂_obs = clip(r̄ᵀ Σ_y⁻¹ r̄ / D_y, ζ_min, ζ_max),   σ²_obs = 2 / D_y
        μ_ζ|y = (σ²_ζ ζ̂_obs + σ²_obs μ_ζ) / (σ²_ζ + σ²_obs)
        σ²_ζ|y = σ²_ζ σ²_obs / (σ²_ζ + σ²_obs)

    Treats ζ as a random variable with the Gaussian prior ζ ~ 𝒩(μ_ζ, σ²_ζ) and updates
    it from the normalised residual of the window, in a simplified form of Anderson
    (2009). A correctly specified Σ_y has 𝔼[χ² / D_y] = 1, and an underdispersive
    ensemble pulls ζ̂_obs above 1. The posterior is the Gaussian product of the prior
    with a likelihood centred at ζ̂_obs, whose variance σ²_obs is that of χ² / D_y.

    The posterior comes back as JAX scalars, so it can be carried through `jax.jit`,
    `jax.grad` and `jax.lax.scan` across windows. To inflate an ensemble, pass μ_ζ|y to
    `inflate_multiplicative`.

    Algorithm:
        1. Form χ² = r̄ᵀ Σ_y⁻¹ r̄ with one solve.
        2. Clamp χ² / D_y to [ζ_min, ζ_max] to get ζ̂_obs.
        3. Multiply the Gaussian prior by the likelihood.

    Args:
        inflation_mean: Prior mean μ_ζ.
        inflation_var: Prior variance σ²_ζ.
        innovation: Mean residual r̄ = y − h̄.
        innovation_cov: Dense innovation covariance Σ_y, taken as positive
            semi-definite. For typical D_y a dense solve is cheaper than assembling a
            structured operator; build it with ``as_matrix()`` at the call site.
        min_factor: Lower clamp ζ_min on the data-implied estimate.
        max_factor: Upper clamp ζ_max on the data-implied estimate.

    Returns:
        Tuple ``(posterior_mean, posterior_variance)`` holding μ_ζ|y and σ²_ζ|y, which
        serve as the prior of the next window.

    Raises:
        ZeroDivisionError: If D_y = 0.

    Examples:
        A residual far larger than Σ_y allows pulls the factor up, and the posterior
        variance shrinks:

        >>> import jax.numpy as jnp
        >>> import filterax
        >>> innovation = jnp.array([3.0, -3.0])
        >>> innovation_cov = jnp.eye(2)
        >>> mean, variance = filterax.inflate_adaptive(
        ...     1.0, 0.01, innovation, innovation_cov
        ... )
        >>> bool(mean > 1.0), bool(variance < 0.01)
        (True, True)

        With no prior variance the prior wins:

        >>> mean, variance = filterax.inflate_adaptive(
        ...     1.05, 0.0, innovation, innovation_cov
        ... )
        >>> bool(jnp.allclose(mean, 1.05)), float(variance)
        (True, 0.0)

        The posterior inflates an ensemble directly:

        >>> particles = jnp.array([[0.0], [2.0]])
        >>> filterax.inflate_multiplicative(particles, mean).shape
        (2, 1)

    References:
        Anderson (2009), *Tellus A* 61, 72-83.
    """
    obs_size = innovation.shape[0]
    covariance_op = lx.MatrixLinearOperator(
        innovation_cov, lx.positive_semidefinite_tag
    )
    # χ² = r̄ᵀ Σ_y⁻¹ r̄
    chi_squared = gaussx.quadratic_form(covariance_op, innovation)
    # χ² / D_y
    normalised_chi_squared = chi_squared / obs_size
    # ζ̂_obs, clamped to keep the estimate from running away on outlier residuals
    factor_estimate = jnp.clip(normalised_chi_squared, min_factor, max_factor)
    # σ²_obs = 2 / D_y
    obs_variance = jnp.asarray(2.0 / obs_size, dtype=normalised_chi_squared.dtype)
    # μ_ζ
    prior_mean = jnp.asarray(inflation_mean, dtype=normalised_chi_squared.dtype)
    # σ²_ζ
    prior_variance = jnp.asarray(inflation_var, dtype=normalised_chi_squared.dtype)
    # σ²_ζ + σ²_obs
    total_variance = prior_variance + obs_variance
    # σ²_ζ ζ̂_obs
    weighted_estimate = prior_variance * factor_estimate
    # σ²_obs μ_ζ
    weighted_prior = obs_variance * prior_mean
    # μ_ζ|y = (σ²_ζ ζ̂_obs + σ²_obs μ_ζ) / (σ²_ζ + σ²_obs)
    posterior_mean = (weighted_estimate + weighted_prior) / total_variance
    # σ²_ζ σ²_obs
    variance_product = prior_variance * obs_variance
    # σ²_ζ|y = σ²_ζ σ²_obs / (σ²_ζ + σ²_obs)
    posterior_variance = variance_product / total_variance
    return posterior_mean, posterior_variance


def ledoit_wolf_shrinkage(
    particles: Float[Array, "N_e N_x"],
) -> tuple[Float[Array, "N_x N_x"], Float[Array, ""]]:
    """Ledoit-Wolf shrinkage of the sample covariance, as a dense matrix.

        Σ̂_λ = (1 − λ) Σ̂_x + λ μ I,   μ = tr(Σ̂_x) / D_x
        d² = ‖Σ̂_x − μ I‖_F²
        b² = (1 / N_ℰ²) ∑ₑ ‖x̃⁽ᵉ⁾ x̃⁽ᵉ⁾ᵀ − Σ̂_x‖_F²
        λ = min(1, b² / d²)

    Wraps `gaussx.ledoit_wolf_shrinkage`, with the divisor N_ℰ − 1 in Σ̂_x, and adds
    two things: a check of N_ℰ against filterax's error, and the materialised matrix.
    It helps when N_ℰ ≪ D_x and a rank-deficient Σ̂_x would contaminate what follows.
    The result is positive semi-definite, not strictly definite: where the ensemble is
    collapsed in some direction, both Σ̂_x and μ I can be singular. Only call it when
    D_x is small enough to materialise; gaussx keeps the operator structured.

    Args:
        particles: Ensemble members x⁽ᵉ⁾, one row per member.

    Returns:
        Tuple ``(covariance, shrinkage)`` holding the dense Σ̂_λ and the intensity λ
        as a scalar array, so that the function composes with `jax.jit`, `jax.vmap`
        and `jax.grad`.

    Raises:
        ValueError: If N_ℰ < 2, where the divisor N_ℰ − 1 is undefined.

    Examples:
        Few members in a large state need heavy shrinkage:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import filterax
        >>> particles = jr.normal(jr.key(0), (10, 200))
        >>> covariance, shrinkage = filterax.ledoit_wolf_shrinkage(particles)
        >>> covariance.shape
        (200, 200)
        >>> bool(0.5 < shrinkage <= 1.0)
        True

        One member has no sample covariance:

        >>> filterax.ledoit_wolf_shrinkage(
        ...     particles[:1]
        ... )  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        ValueError: particles must contain at least 2 ensemble members for the
        Bessel-corrected estimators to be well-defined; got N_e=1.

    References:
        Ledoit & Wolf (2004), *J. Multivariate Anal.* 88, 365-411.
    """
    check_ensemble_size(particles.shape[0])
    # Σ̂_λ as an operator, and λ
    shrunk_operator, shrinkage = gaussx.ledoit_wolf_shrinkage(particles)
    # Σ̂_λ
    shrunk_covariance = shrunk_operator.as_matrix()
    return shrunk_covariance, shrinkage

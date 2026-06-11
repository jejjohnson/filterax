r"""Ensemble data-assimilation diagnostics.

Pure functions that answer "is the filter working correctly?" — they
*detect* problems, they don't fix them. Three layers of cost / value:

* **Per-cycle health checks** (cheap, run every analysis window):
  :func:`ensemble_spread`, :func:`rms_spread`, :func:`innovation`,
  :func:`normalized_innovation`, :func:`chi2_consistency`,
  :func:`chi2_normalized`, :func:`effective_ensemble_size`.

* **Calibration assessments** (rolling-window, need truth in twin
  experiments): :func:`rmse_vs_truth`, :func:`spread_skill_ratio`,
  :func:`rank_histogram`, :func:`rank_histogram_chi2`.

* **A-posteriori covariance diagnosis** (long accumulation):
  :func:`desroziers_R_estimate`,
  :func:`desroziers_innovation_cov`,
  :func:`desroziers_analysis_residual_cov`,
  :func:`dfs_from_gain`, :func:`dfs_from_ensemble`.

Plus the proper scoring rule :func:`crps_ensemble`, used for
forecast skill evaluation against observations.

All functions live as ``filterax.utils.*`` and take pre-computed
arrays — they don't run analyses or own state.

References at module bottom; per-function papers in each docstring.
"""

from __future__ import annotations

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


# ──────────────────────────────────────────────────────────────────────
# Phase 1 — per-cycle health checks
# ──────────────────────────────────────────────────────────────────────


def ensemble_spread(
    ensemble: Float[Array, "N_e N_x"],
) -> Float[Array, " N_x"]:
    r"""Per-variable ensemble standard deviation.

    ``σᵢ = √( (Nₑ − 1)⁻¹ Σⱼ (xᵢ⁽ʲ⁾ − x̄ᵢ)² )``

    Track over assimilation cycles. A well-calibrated filter has
    ``σ ≈ RMSE``; collapsing spread is the first sign of filter
    divergence.

    Args:
        ensemble: Ensemble of shape ``(Nₑ, Nₓ)``.

    Returns:
        Per-variable standard deviation of shape ``(Nₓ,)``.

    Example:
        >>> import jax.numpy as jnp
        >>> from filterax.utils import ensemble_spread
        >>> ens = jnp.array([[0.0, 0.0], [2.0, 4.0]])
        >>> ensemble_spread(ens)
        Array([1.4142135, 2.828427 ], dtype=float32)

    Reference:
        Whitaker, J. S. & Loughe, A. F. (1998). *The Relationship
        between Ensemble Spread and Ensemble Mean Skill.* Mon. Wea.
        Rev., 126(12), 3292–3302.
    """
    return jnp.std(ensemble, axis=0, ddof=1)


def rms_spread(
    ensemble: Float[Array, "N_e N_x"],
) -> Float[Array, ""]:
    r"""Scalar RMS ensemble spread ``√(Nₓ⁻¹ Σᵢ σᵢ²)``."""
    sigma = ensemble_spread(ensemble)
    return jnp.sqrt(jnp.mean(sigma * sigma))


def rmse_vs_truth(
    ensemble_mean_: Float[Array, " N_x"],
    x_true: Float[Array, " N_x"],
) -> Float[Array, ""]:
    r"""RMSE of an ensemble mean against the true state ``√(Nₓ⁻¹ Σᵢ (x̄ᵢ − xᵢ*)²)``.

    Only available in twin / OSSE experiments where ``x_true`` is
    known. Track over time to detect filter divergence (RMSE growing
    without bound).

    Example:
        >>> import jax.numpy as jnp
        >>> from filterax.utils import rmse_vs_truth
        >>> rmse_vs_truth(jnp.array([1.0, 2.0]), jnp.array([0.0, 2.0]))
        Array(0.70710677, dtype=float32)
    """
    diff = ensemble_mean_ - x_true
    return jnp.sqrt(jnp.mean(diff * diff))


def innovation(
    y_obs: Float[Array, " N_y"],
    Hx_forecast: Float[Array, " N_y"],
) -> Float[Array, " N_y"]:
    r"""Innovation vector ``d = y − H x̄_f``."""
    return y_obs - Hx_forecast


def normalized_innovation(
    y_obs: Float[Array, " N_y"],
    Hx_forecast: Float[Array, " N_y"],
    innovation_var: Float[Array, " N_y"],
) -> Float[Array, " N_y"]:
    r"""Normalised innovation ``d_i / √(HPHᵀ + R)_{ii}``.

    For a correctly-specified filter the normalised innovation is
    ``𝒩(0, 1)`` componentwise. Departures from unit variance signal
    misspecified ``P`` or ``R``.
    """
    d = y_obs - Hx_forecast
    return d / jnp.sqrt(jnp.maximum(innovation_var, 1e-30))


def chi2_consistency(
    d: Float[Array, " N_y"],
    S_inv_d: Float[Array, " N_y"],
) -> Float[Array, ""]:
    r"""``χ² = dᵀ S⁻¹ d``. Should be ``≈ Nᵧ`` under correct specification.

    Pre-compute ``S⁻¹ d`` once and pass it here so the diagnostic
    composes with whichever solver (gaussx Woodbury, dense Cholesky)
    you used for the analysis. Variance under the null is ``2 Nᵧ``.

    Reference:
        Mehra, R. K. (1970). *On the Identification of Variances and
        Adaptive Kalman Filtering.* IEEE Trans. Automatic Control,
        15(2), 175–184.
    """
    return d @ S_inv_d


def chi2_normalized(
    d: Float[Array, " N_y"],
    S_inv_d: Float[Array, " N_y"],
    n_obs: int,
) -> Float[Array, ""]:
    """Compact single-number consistency check ``χ² / Nᵧ`` (target ≈ 1)."""
    return (d @ S_inv_d) / n_obs


def effective_ensemble_size(
    weights: Float[Array, " N_e"],
) -> Float[Array, ""]:
    r"""Effective sample size ``Nₑff = 1 / Σⱼ wⱼ²``.

    For equally-weighted ensembles (the standard EnKF) this is
    exactly ``Nₑ``. For particle filters / weighted EnKF variants,
    ``Nₑff ≪ Nₑ`` signals weight degeneracy and the need to resample.

    Reference:
        Liu, J. S. & Chen, R. (1998). *Sequential Monte Carlo Methods
        for Dynamic Systems.* JASA, 93(443), 1032–1044.
    """
    return 1.0 / jnp.sum(weights * weights)


def weight_entropy(
    weights: Float[Array, " N_e"],
) -> Float[Array, ""]:
    r"""Shannon entropy ``−Σⱼ wⱼ log wⱼ`` of importance weights.

    Maximum ``log Nₑ`` for uniform weights, ``0`` for full
    degeneracy.
    """
    safe = jnp.maximum(weights, 1e-30)
    return -jnp.sum(weights * jnp.log(safe))


# ──────────────────────────────────────────────────────────────────────
# Phase 2 — calibration assessment
# ──────────────────────────────────────────────────────────────────────


def spread_skill_ratio(
    ensemble: Float[Array, "N_e N_x"],
    x_true: Float[Array, " N_x"],
) -> Float[Array, ""]:
    r"""Ratio of RMS spread to RMSE.

    ``SSR = √(Σᵢ σᵢ² / Nₓ) / √(Σᵢ (x̄ᵢ − xᵢ*)² / Nₓ)``

    * ``SSR ≈ 1`` — well calibrated.
    * ``SSR < 1`` — underdispersive (increase inflation).
    * ``SSR > 1`` — overdispersive (decrease inflation).

    Reference:
        Fortin, V., Abaza, M., Anctil, F., & Turcotte, R. (2014).
        *Why Should Ensemble Spread Match the RMSE of the Ensemble
        Mean?* J. Hydrometeorol., 15(4), 1708–1713.
    """
    rms = rms_spread(ensemble)
    err = rmse_vs_truth(jnp.mean(ensemble, axis=0), x_true)
    return rms / jnp.maximum(err, 1e-30)


def rank_histogram(
    ensemble_over_time: Float[Array, "T N_e N_x"],
    truth_over_time: Float[Array, "T N_x"],
) -> Int[Array, " N_eP1"]:
    r"""Rank histogram counts across all ``(t, i)`` pairs.

    For each variable ``i`` at each time ``t``, count how many of the
    ``Nₑ`` ensemble members are smaller than the truth; the resulting
    rank is in ``{0, 1, …, Nₑ}`` (``Nₑ + 1`` bins). Aggregate over
    time and variables. A reliable ensemble produces a roughly uniform
    histogram; U-shaped → underdispersive, dome-shaped → overdispersive,
    skewed → systematic bias.

    Args:
        ensemble_over_time: ``(T, Nₑ, Nₓ)`` ensemble snapshots.
        truth_over_time: ``(T, Nₓ)`` true states at the same times.

    Returns:
        ``(Nₑ + 1,)`` integer counts.

    Example:
        >>> import jax.numpy as jnp
        >>> from filterax.utils import rank_histogram
        >>> ens = jnp.array([[[0.0], [1.0]], [[0.0], [1.0]]])  # (T, N_e, N_x)
        >>> truth = jnp.array([[0.5], [2.0]])
        >>> rank_histogram(ens, truth)
        Array([0, 1, 1], dtype=int32)

    Reference:
        Hamill, T. M. (2001). *Interpretation of Rank Histograms for
        Verifying Ensemble Forecasts.* Mon. Wea. Rev., 129(3), 550–560.
    """
    _T, N_e, _N_x = ensemble_over_time.shape
    # Count members < truth, broadcast over (T, Nₑ, Nₓ).
    smaller = jnp.sum(
        ensemble_over_time < truth_over_time[:, None, :], axis=1
    )  # (T, Nₓ) — per-pair rank in {0, …, Nₑ}
    ranks = smaller.reshape(-1)
    return jnp.bincount(ranks, length=N_e + 1)


def rank_histogram_chi2(counts: Int[Array, " bins"]) -> Float[Array, ""]:
    r"""χ² flatness statistic for a rank histogram.

    ``χ² = Σ_k (O_k − E_k)² / E_k`` with ``E_k = N_total / N_bins``.
    Larger values indicate stronger departure from uniformity; under
    the null hypothesis of uniformity the test statistic is
    ``χ²(N_bins − 1)``.
    """
    n_bins = counts.shape[0]
    total = jnp.sum(counts)
    expected = total / n_bins
    diff = counts - expected
    return jnp.sum(diff * diff / jnp.maximum(expected, 1e-30))


# ──────────────────────────────────────────────────────────────────────
# Phase 3 — a-posteriori covariance diagnosis
# ──────────────────────────────────────────────────────────────────────


def desroziers_R_estimate(
    d_forecast: Float[Array, "T N_y"],
    d_analysis: Float[Array, "T N_y"],
) -> Float[Array, "N_y N_y"]:
    r"""A-posteriori estimate of ``R`` (Desroziers et al. 2005).

    ``R̂ = E[ d_a d_fᵀ ]`` over time. When ``R̂`` disagrees with the
    prescribed ``R`` the observation-error covariance is mis-specified.
    Accumulate over many cycles (100+) for stable estimates.
    """
    T = d_forecast.shape[0]
    return einx.dot("t i, t j -> i j", d_analysis, d_forecast) / T


def desroziers_innovation_cov(
    d_forecast: Float[Array, "T N_y"],
) -> Float[Array, "N_y N_y"]:
    r"""Empirical innovation covariance ``E[ d_f d_fᵀ ] ≈ HPHᵀ + R``."""
    T = d_forecast.shape[0]
    return einx.dot("t i, t j -> i j", d_forecast, d_forecast) / T


def desroziers_analysis_residual_cov(
    d_analysis: Float[Array, "T N_y"],
) -> Float[Array, "N_y N_y"]:
    r"""Empirical analysis-residual covariance ``E[ d_a d_aᵀ ] ≈ R − HAHᵀ``."""
    T = d_analysis.shape[0]
    return einx.dot("t i, t j -> i j", d_analysis, d_analysis) / T


def dfs_from_gain(
    K: Float[Array, "N_x N_y"],
    H: Float[Array, "N_y N_x"],
) -> Float[Array, ""]:
    r"""Degrees of freedom for signal from an explicit gain.

    ``DFS = tr(K H) ∈ [0, Nᵧ]``. Closer to ``Nᵧ`` means observations
    dominate the analysis; closer to ``0`` means the prior dominates.

    Reference:
        Cardinali, C., Pezzulli, S., & Andersson, E. (2004).
        *Influence-Matrix Diagnostic of a Data Assimilation System.*
        QJRMS, 130(603), 2767–2786.
    """
    return jnp.trace(K @ H)


def dfs_from_ensemble(
    ensemble_forecast: Float[Array, "N_e N_x"],
    ensemble_analysis: Float[Array, "N_e N_x"],
) -> Float[Array, ""]:
    r"""Ensemble-based DFS estimate from forecast / analysis perturbations.

    ``DFS ≈ tr( Cᵃᶠ / σ_f² )`` where ``Cᵃᶠ`` is the per-variable
    forecast-vs-analysis cross-covariance and ``σ_f²`` is the
    forecast variance — i.e. how much the analysis perturbations are
    *driven* by the forecast perturbations. ``DFS = Nₓ`` for an
    uninformative observation set; ``DFS < Nₓ`` measures how much
    constraint the observations contributed.

    Avoids forming ``K`` and ``H`` explicitly — useful in
    ensemble-only workflows.
    """
    N_e = ensemble_forecast.shape[0]
    mean_f = jnp.mean(ensemble_forecast, axis=0)
    mean_a = jnp.mean(ensemble_analysis, axis=0)
    anom_f = ensemble_forecast - mean_f[None, :]
    anom_a = ensemble_analysis - mean_a[None, :]
    # Per-variable cross-covariance and forecast variance.
    cross = jnp.sum(anom_a * anom_f, axis=0) / (N_e - 1)  # (Nₓ,)
    var_f = jnp.sum(anom_f * anom_f, axis=0) / (N_e - 1)  # (Nₓ,)
    ratio = cross / jnp.maximum(var_f, 1e-30)
    # Per-variable "signal share"; sum to a scalar DFS proxy.
    return jnp.sum(ratio)


# ──────────────────────────────────────────────────────────────────────
# Proper scoring rule — CRPS
# ──────────────────────────────────────────────────────────────────────


def crps_ensemble(
    ensemble: Float[Array, " N_e"],
    y_obs: Float[Array, ""],
) -> Float[Array, ""]:
    r"""Continuous Ranked Probability Score for a 1D ensemble forecast.

    ``CRPS = E|X − y| − ½ E|X − X′|``

    Computed via the sorted-ensemble identity (Hersbach 2000):

    ``½ E|X − X′| = Nₑ⁻² Σⱼ x_{(j)} (2j − 1 − Nₑ)``

    where ``x_{(j)}`` are the order statistics. Cost
    ``O(Nₑ log Nₑ)`` per observation. Lower is better; CRPS is a
    strictly proper scoring rule and has the same units as the
    observed variable.

    Args:
        ensemble: Single-variable ensemble of shape ``(Nₑ,)``.
        y_obs: Scalar observed value.

    Returns:
        Scalar CRPS.

    Example:
        >>> import jax.numpy as jnp
        >>> from filterax.utils import crps_ensemble
        >>> crps_ensemble(jnp.array([0.0, 1.0]), jnp.array(0.5))
        Array(0.25, dtype=float32)

    Reference:
        Gneiting, T. & Raftery, A. E. (2007). *Strictly Proper
        Scoring Rules, Prediction, and Estimation.* JASA, 102(477),
        359–378.
    """
    N_e = ensemble.shape[0]
    sorted_e = jnp.sort(ensemble)
    indices = jnp.arange(1, N_e + 1, dtype=ensemble.dtype)
    first_term = jnp.mean(jnp.abs(ensemble - y_obs))
    second_term = jnp.sum(sorted_e * (2.0 * indices - 1.0 - N_e)) / (N_e * N_e)
    return first_term - second_term


def crps_ensemble_batch(
    ensemble: Float[Array, "N_e N_y"],
    y_obs: Float[Array, " N_y"],
) -> Float[Array, ""]:
    r"""Mean CRPS over an observation vector.

    Applies :func:`crps_ensemble` to each observation component
    independently and averages the results.
    """
    import jax

    per_obs = jax.vmap(crps_ensemble, in_axes=(1, 0))(ensemble, y_obs)
    return jnp.mean(per_obs)

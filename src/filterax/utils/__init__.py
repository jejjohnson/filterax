"""Diagnostic and utility primitives.

Re-exports the ensemble-DA diagnostics from
:mod:`filterax._diagnostics` so users can do
``filterax.utils.rmse_vs_truth(...)``. See the module for
selection guidance (per-cycle health checks, calibration assessment,
a-posteriori covariance diagnosis, and the CRPS proper scoring rule).
"""

from filterax._diagnostics import (
    chi2_consistency as chi2_consistency,
    chi2_normalized as chi2_normalized,
    crps_ensemble as crps_ensemble,
    crps_ensemble_batch as crps_ensemble_batch,
    desroziers_analysis_residual_cov as desroziers_analysis_residual_cov,
    desroziers_innovation_cov as desroziers_innovation_cov,
    desroziers_R_estimate as desroziers_R_estimate,
    dfs_from_ensemble as dfs_from_ensemble,
    dfs_from_gain as dfs_from_gain,
    effective_ensemble_size as effective_ensemble_size,
    ensemble_spread as ensemble_spread,
    innovation as innovation,
    normalized_innovation as normalized_innovation,
    rank_histogram as rank_histogram,
    rank_histogram_chi2 as rank_histogram_chi2,
    rms_spread as rms_spread,
    rmse_vs_truth as rmse_vs_truth,
    spread_skill_ratio as spread_skill_ratio,
    weight_entropy as weight_entropy,
)

# Diagnostics

Pure functions that answer *"is the filter working correctly?"* — they
**detect** problems, they don't fix them. Live under
`filterax.utils.*`. Group them into a standard pipeline by cost:

## Phase 1 — per-cycle health checks (cheap)

Run on every analysis window; cost is at most ``O(N_e N_x + N_y)``.

| Function | What it tells you | Healthy value |
|---|---|---|
| [`ensemble_spread`][filterax.utils.ensemble_spread] | Per-variable σ | Stable across cycles |
| [`rms_spread`][filterax.utils.rms_spread] | Scalar RMS spread | Non-collapsing |
| [`innovation`][filterax.utils.innovation] | ``d = y − H x̄_f`` | Mean ≈ 0 over time |
| [`normalized_innovation`][filterax.utils.normalized_innovation] | ``d_i / √S_ii`` | ``~ 𝒩(0, 1)`` |
| [`chi2_consistency`][filterax.utils.chi2_consistency], [`chi2_normalized`][filterax.utils.chi2_normalized] | Mahalanobis innovation norm | ``χ² / N_y ≈ 1`` |
| [`effective_ensemble_size`][filterax.utils.effective_ensemble_size] | Weight degeneracy (particle filters) | ``≈ N_e`` |
| [`weight_entropy`][filterax.utils.weight_entropy] | Shannon entropy of weights | ``≈ log N_e`` |

## Phase 2 — calibration assessment (rolling window, needs truth)

Compute over 50–100 cycles in a twin / OSSE experiment.

| Function | What it tells you | Healthy value |
|---|---|---|
| [`rmse_vs_truth`][filterax.utils.rmse_vs_truth] | Accuracy of ensemble mean | Stable / decreasing |
| [`spread_skill_ratio`][filterax.utils.spread_skill_ratio] | Calibration of σ vs RMSE | ``≈ 1`` |
| [`rank_histogram`][filterax.utils.rank_histogram] | Distribution of truth's rank in the ensemble | Uniform |
| [`rank_histogram_chi2`][filterax.utils.rank_histogram_chi2] | Flatness test for the histogram | ``χ²(N_e)`` quantiles |

## Phase 3 — a-posteriori covariance diagnosis (long accumulation)

Run after a complete experiment (100+ cycles) to diagnose mis-specified
``P`` or ``R``.

| Function | What it tells you |
|---|---|
| [`desroziers_R_estimate`][filterax.utils.desroziers_R_estimate] | A-posteriori ``R̂`` from ``E[d_a d_fᵀ]`` |
| [`desroziers_innovation_cov`][filterax.utils.desroziers_innovation_cov] | Empirical ``HPHᵀ + R`` from ``E[d_f d_fᵀ]`` |
| [`desroziers_analysis_residual_cov`][filterax.utils.desroziers_analysis_residual_cov] | Empirical ``R − HAHᵀ`` from ``E[d_a d_aᵀ]`` |
| [`dfs_from_gain`][filterax.utils.dfs_from_gain] | Observation information content from ``tr(KH)`` |
| [`dfs_from_ensemble`][filterax.utils.dfs_from_ensemble] | Same, computed from ensemble perturbations |

## Phase 4 — forecast skill (verification against obs)

| Function | What it tells you |
|---|---|
| [`crps_ensemble`][filterax.utils.crps_ensemble], [`crps_ensemble_batch`][filterax.utils.crps_ensemble_batch] | Continuous Ranked Probability Score (lower is better; proper scoring rule) |

## Per-cycle reference

::: filterax.utils
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [ensemble_spread, rms_spread, innovation, normalized_innovation, chi2_consistency, chi2_normalized, effective_ensemble_size, weight_entropy]

## Calibration reference

::: filterax.utils
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [rmse_vs_truth, spread_skill_ratio, rank_histogram, rank_histogram_chi2]

## Covariance-diagnosis reference

::: filterax.utils
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [desroziers_R_estimate, desroziers_innovation_cov, desroziers_analysis_residual_cov, dfs_from_gain, dfs_from_ensemble]

## Forecast-skill reference

::: filterax.utils
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [crps_ensemble, crps_ensemble_batch]

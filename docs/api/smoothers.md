# Smoothers

Backward-pass refiners that turn a sequential-filter history into the
smoothing distribution `p(x_t | y_{1:T})` — conditioning every state on the
*whole* observation record rather than only the past. All smoothers consume
the stacked `forecast_history` / `analysis_history` produced by an
[`AssimilationResult`][filterax.AssimilationResult] and return a
[`SmoothingResult`][filterax.SmoothingResult].

Rough selection guide: `EnKS` (Evensen & van Leeuwen 2000) is the standard
single backward pass and the default choice; `EnsembleRTS` is the ensemble
analogue of the classical Rauch-Tung-Striebel recursion (it coincides with
`EnKS` for zero model error); `EnsembleSqrtSmoother` is the deterministic
square-root variant, paired with a forward square-root filter (ETKF, EnSRF,
ESTKF); `FixedLagSmoother` smooths only a trailing window of length `lag` —
the bounded-memory / streaming option, equivalent to `EnKS` when
`lag ≥ T − 1`; and `IES` is the standalone iterative ensemble smoother
(Chen & Oliver 2013) that solves an inverse problem over a single
observation window, for strongly nonlinear cases where one backward pass is
not enough.

These are importable both from the top level (`filterax.EnKS`) and from the
themed namespace (`filterax.smoothers.EnKS`).

## Reference

::: filterax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [EnKS, EnsembleRTS, EnsembleSqrtSmoother, FixedLagSmoother, IES]

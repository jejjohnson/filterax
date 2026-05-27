# Advanced Sequential Filters (Wave 4)

Wave 4 fills out the deterministic-filter zoo with three ensemble
variants and one parametric reference. Use the [Wave 2 filters](primitives.md)
(``ETKF``, ``EnSRF``, ``StochasticEnKF``, ``LETKF``) as the everyday
defaults; reach for these when you need the specific properties below.

## Picking an advanced filter

| Filter | Family | When to use |
|---|---|---|
| **`filterax.filters.ETKF_Livings`** | Deterministic, randomised | When the symmetric-sqrt ETKF develops preferred ensemble directions over long runs — the random mean-preserving rotation averages out the degeneracy. |
| **`filterax.filters.EnSRF_Serial`** | Deterministic, scalar-serial | When ``R`` is diagonal and you want to avoid any ``N_y × N_y`` solve. Cost ``O(N_e N_x N_y)`` with no matrix inversion. |
| **`filterax.filters.ESTKF`** | Deterministic, reduced-rank | When you want exact mean preservation by construction and a modest constant-factor speedup over ETKF — the eigendecomposition is ``(N_e − 1)³`` rather than ``N_e³``. |
| **`filterax.filters.SquareRootKF`** | Parametric (non-ensemble) | Linear-Gaussian baselines, twin-experiment ground truths, and any setting where carrying ``S`` (with ``P = S Sᵀ``) is cheaper than carrying an ensemble. Wraps gaussx's parallel KF. |

All four implement [`AbstractSequentialFilter`][filterax.AbstractSequentialFilter]
except `SquareRootKF`, which has its own ``filter()`` entry point because
it propagates ``(μ, S)`` rather than particles.

## Reference

::: filterax.filters.ETKF_Livings
::: filterax.filters.EnSRF_Serial
::: filterax.filters.ESTKF
::: filterax.filters.SquareRootKF
::: filterax.filters.SquareRootFilterResult

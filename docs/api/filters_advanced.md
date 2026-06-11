# Advanced Sequential Filters

Specialised deterministic variants of the everyday
[Layer-1 filters](filters.md) (`ETKF`, `EnSRF`, `StochasticEnKF`, `LETKF`).
Use those as the defaults; reach for these when you need the specific
properties below. All three implement
[`AbstractSequentialFilter`][filterax.AbstractSequentialFilter], so they
slot into the same Layer-2 loops and pipekit adapters.

## Picking an advanced filter

| Filter | Family | When to use |
|---|---|---|
| [`filterax.filters.ETKF_Livings`][filterax.filters.ETKF_Livings] | Deterministic, randomised | When the symmetric-sqrt ETKF develops preferred ensemble directions over long runs — the random mean-preserving rotation averages out the degeneracy. |
| [`filterax.filters.EnSRF_Serial`][filterax.filters.EnSRF_Serial] | Deterministic, scalar-serial | When ``R`` is diagonal and you want to avoid any ``N_y × N_y`` solve. Cost ``O(N_e N_x N_y)`` with no matrix inversion. |
| [`filterax.filters.ESTKF`][filterax.filters.ESTKF] | Deterministic, reduced-rank | When you want exact mean preservation by construction and a modest constant-factor speedup over ETKF — the eigendecomposition is ``(N_e − 1)³`` rather than ``N_e³``. |

For the parametric (non-ensemble)
[`SquareRootKF`][filterax.filters.SquareRootKF] — linear-Gaussian baselines
and twin-experiment ground truths — see the [Filters](filters.md) page.

Note that `ETKF_Livings` is stochastic (the rotation draws from a PRNG key),
so it is rejected by the [differentiable training](differentiable.md)
entry points, like `StochasticEnKF`.

## Reference

::: filterax.filters
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [ESTKF, ETKF_Livings, EnSRF_Serial]

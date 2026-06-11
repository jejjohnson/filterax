# Ensemble Kalman Processes

Iterative ensemble methods for derivative-free parameter estimation. All
methods share the same outer loop — the caller supplies forward-model
evaluations `G(θ⁽ʲ⁾)` each step, the process supplies the ensemble
update rule. The Layer-2 wrappers (`filterax.EKI`, `filterax.EKS`,
`filterax.UKI`) handle the loop for you; the artificial-time step `Δtₙ` is
controlled by a [scheduler](schedulers.md), and the iteration states
([`ProcessState`][filterax.ProcessState], [`UKIState`][filterax.UKIState],
[`ProcessConfig`][filterax.ProcessConfig]) are documented with the
[Protocols & Types](protocols.md).

## Picking a process

| Method | Output | UQ | Forward evals / step | Use when |
|---|---|---|---|---|
| **`filterax.EKI`** | Collapsing point estimate | None | `J` | Default calibration; underdetermined OK (`J ≥ p + 1`) |
| **`filterax.EKS`** | Posterior samples | Full ensemble | `J` | Posterior uncertainty matters; longer runs |
| **`filterax.UKI`** | Parametric `(μ, Σ)` | Calibrated `Σ` | `2 Nₚ + 1` (fixed) | Moderate `Nₚ ≲ 100`, deterministic, reproducible |
| **`filterax.processes.ETKI`** | MLE | None | `J` | Same as EKI — semantic alias; output-scalable variant for `N_d ≫ J` (current impl delegates to EKI's gaussx-Woodbury solve) |
| **`filterax.processes.GNKI`** | MAP + ensemble spread | Via spread | `J` | Well-conditioned, fast convergence, requires `J > Nₚ` |
| **`filterax.processes.SparseInversion`** | Sparse point estimate | None | `J` | Variable selection, sparse parameter recovery |
| **`filterax.processes.TEKI`** | MAP | None | `J` | Ill-posed problems where vanilla EKI drifts |

### Decision tree

1. Need posterior samples? → **EKS**
2. Need calibrated parametric covariance? → **UKI** (if `Nₚ ≲ 100`)
3. Sparse parameters? → **SparseInversion**
4. Ill-posed / need prior regularisation? → **TEKI**
5. Well-conditioned, want fast convergence? → **GNKI** (if `J > Nₚ`)
6. Otherwise → **EKI** (simple, robust default)

## Layer-2 API (run-loop wrappers)

These are the canonical entry points. Each composes the forward model,
a Layer-1 process, and a scheduler into a single `.run()` call returning a
`ProcessResult`.

::: filterax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [EKI, EKS, UKI, ProcessResult]

## Layer-1 components (advanced)

Drop down to these when you need per-step control of the loop (custom
convergence checks, JAX `scan` integration, etc.). All implement
[`AbstractProcess`][filterax.AbstractProcess].

::: filterax.processes
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [EKI, EKS_Process, UKI, ETKI, GNKI, SparseInversion, TEKI]

## Sigma-point utilities

The unscented quadrature used by `UKI` is also exposed for direct use.

::: filterax.processes
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [sigma_points]

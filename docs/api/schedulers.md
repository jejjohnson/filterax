# Schedulers

Each EKP iteration is parameterised by an artificial-time step
`Δtₙ ∈ ℝ₊`. The scheduler picks `Δtₙ` from the current
[`ProcessState`][filterax.ProcessState] — either constant, misfit-adaptive,
or stability-controlled for the EKS Langevin SDE. All implement
[`AbstractScheduler`][filterax.AbstractScheduler].

By convention `algo_time = Σₙ Δtₙ`. Schedulers that drive convergence
arrange for `algo_time → 1`; the Layer-2 run loops break out when
`algo_time ≥ 1` (the standard EKI termination rule; Iglesias & Yang 2021).

## Picking a scheduler

| Scheduler | Use with | Behaviour |
|---|---|---|
| [`FixedScheduler`][filterax.FixedScheduler] | Any process | Constant `Δt`; you tune manually |
| [`DataMisfitController`][filterax.DataMisfitController] | EKI, ETKI, GNKI, TEKI | Adaptive `Δt` from the Iglesias & Yang (2021) rule; clamps so `algo_time` lands at `1` |
| [`EKSStableScheduler`][filterax.EKSStableScheduler] | EKS | Bounds `Δt` by `‖Cᶿᶿ‖` and by the posterior's stiffness, which sets the sampling bias |

## Reference

::: filterax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [FixedScheduler, DataMisfitController, EKSStableScheduler]

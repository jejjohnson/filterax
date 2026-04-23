# [Wave 3] Ensemble Kalman Processes and Optax

## Shared Context
This wave uses `features/processes.md`, `features/optax_ekp.md`,
`api/components.md`, `api/models.md`, and `examples/components.md` /
`examples/integration.md`.

The process side of `filterax` is the iterative inverse-problem counterpart to
the sequential filter side:

- EKI for point-estimate inversion
- EKS for posterior sampling
- UKI for parametric uncertainty propagation
- transform/advanced variants for scalability or regularization
- optax wrappers as a first-class package promise

## Mathematical Baseline
The common EKP update is:

$$\theta_{n+1}^{(j)} = \theta_n^{(j)} + \Delta t_n\, C_n^{\theta G}\left(C_n^{GG} + \Delta t_n^{-1}\Gamma\right)^{-1}(y - G(\theta_n^{(j)}))$$

UKI replaces an explicit ensemble with sigma-point quadrature and mean/covariance
updates.

The optax layer treats the change in the ensemble mean as the `updates` output,
while the full ensemble or UKI state lives inside optimizer state.

## Canonical API Snapshot
High-level process loop:

```python
process = filterax.EKI(
    forward_fn=my_forward_model,
    obs=observations,
    noise_cov=Gamma,
    scheduler=filterax.DataMisfitController(),
)
result = process.run(init_particles)
```

Optax wrapper:

```python
tx = filterax.optax.eki(
    forward_fn=my_forward_model,
    obs=observations,
    noise_cov=Gamma,
    scheduler=filterax.DataMisfitController(),
)
state = tx.init(params)
updates, state = tx.update(None, state, params)
params = optax.apply_updates(params, updates)
```

# [Wave 3] Ensemble Kalman Processes and Optax
Draft ID: `FLX-26`
## Goal
Ship the first process/inversion side of `filterax`, including high-level
EKP models and optax transforms.

## Wave / Milestone
- Wave: `wave:3`
- Milestone: `v0.3-processes-optax`

## Canonical Epics
- [ ] FLX-27 [Epic] 3.A Process Schedulers and Core Algorithms
- [ ] FLX-28 [Epic] 3.B Process Models and Optax Transforms
- [ ] FLX-29 [Epic] 3.C Process Verification and API Docs

## Sequential Dependencies
- `FLX-27 -> FLX-28 -> FLX-29`.

## Definition of Done
- `filterax` can run inversion/sampling workflows and expose them as optax
  transforms.

## Relationships
- Blocked by FLX-06.
- Blocks FLX-37 and FLX-48.

---

# [Epic] 3.A Process Schedulers and Core Algorithms
Draft ID: `FLX-27`
## Theme
Implement the process-side schedulers and algorithm cores.

## Parent Wave
- Wave epic: `FLX-26`
- Wave label: `wave:3`
- Milestone: `v0.3-processes-optax`

## Parallelism
- Can run in parallel with: limited prep only
- Must complete before: FLX-28

---

# [Epic] 3.B Process Models and Optax Transforms
Draft ID: `FLX-28`
## Theme
Wrap the process algorithms into user-facing models and optax transforms.

## Parent Wave
- Wave epic: `FLX-26`
- Wave label: `wave:3`
- Milestone: `v0.3-processes-optax`

## Parallelism
- Blocked by (inside this wave): FLX-27
- Must complete before: FLX-29

---

# [Epic] 3.C Process Verification and API Docs
Draft ID: `FLX-29`
## Theme
Verify the process side and document the process/optax API surface without making this the main tutorial wave.

## Parent Wave
- Wave epic: `FLX-26`
- Wave label: `wave:3`
- Milestone: `v0.3-processes-optax`

## Parallelism
- Blocked by (inside this wave): FLX-28

---

# processes(schedulers): implement FixedScheduler, DataMisfitController, and EKSStableScheduler
Draft ID: `FLX-30`
## Problem / Request
Implement the process scheduler surface used by EKI/EKS/UKI and the optax
wrappers.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/api/components.md`, `design_docs/filterX/features/optax_ekp.md`

## Implementation Steps
- [ ] Implement `FixedScheduler`.
- [ ] Implement `DataMisfitController`.
- [ ] Implement `EKSStableScheduler`.

## Relationships
- Parent epic: FLX-27.
- Blocked by FLX-10 and FLX-11.

---

# processes(core): implement EKI and EKS_Process
Draft ID: `FLX-31`
## Problem / Request
Implement the core ensemble-process algorithms for inversion and posterior
sampling.

## Mathematical Notes
EKI uses the standard deterministic ensemble update.

EKS adds a covariance-scaled stochastic term so the ensemble does not collapse.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/features/processes.md`

## Implementation Steps
- [ ] Implement `EKI`.
- [ ] Implement `EKS_Process`.
- [ ] Reuse the shared ensemble statistics and gain-side algebra from Wave 1.

## Relationships
- Parent epic: FLX-27.
- Blocked by FLX-12, FLX-13, and FLX-30.

---

# processes(uki): implement UKI and sigma-point utilities
Draft ID: `FLX-32`
## Problem / Request
Implement Unscented Kalman Inversion and the sigma-point utilities it needs.

## Motivation
UKI is a distinct parametric branch of the process story and deserves a clean
issue rather than being buried inside EKI logic.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/features/processes.md`

## Implementation Steps
- [ ] Implement sigma-point generation and weighting helpers.
- [ ] Implement `UKI` and `UKIState`.
- [ ] Keep the mean/covariance state representation explicit.

## Relationships
- Parent epic: FLX-27.
- Blocked by FLX-11 and FLX-30.

---

# processes(advanced): implement ETKI, GNKI, SparseInversion, and TEKI
Draft ID: `FLX-33`
## Problem / Request
Implement the higher-variant process algorithms that extend the basic EKI/EKS
surface.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/features/processes.md`

## Implementation Steps
- [ ] Implement `ETKI`.
- [ ] Implement `GNKI`.
- [ ] Implement `SparseInversion`.
- [ ] Implement `TEKI`.

## Definition of Done
- The advanced process catalog exists as real algorithm classes, not only
  research notes.

## Relationships
- Parent epic: FLX-27.
- Blocked by FLX-31 and FLX-32.

---

# models(processes): implement high-level EKI, EKS, and UKI run loops
Draft ID: `FLX-34`
## Problem / Request
Wrap the process algorithms into ready-to-run Layer 2 models with convergence
handling and results containers.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/api/models.md`, `design_docs/filterX/examples/models.md`

## Implementation Steps
- [ ] Implement high-level `EKI`, `EKS`, and `UKI` model classes.
- [ ] Implement `run(...)` and result/history handling.
- [ ] Return `ProcessResult` consistently across process families.

## Relationships
- Parent epic: FLX-28.
- Blocked by FLX-31 and FLX-32.

---

# optax(processes): implement eki, eks, and uki as GradientTransformations
Draft ID: `FLX-35`
## Problem / Request
Implement the optax wrappers that expose EKP as `optax.GradientTransformation`.

## Proposed API
```python
filterax.optax.eki(...)
filterax.optax.eks(...)
filterax.optax.uki(...)
```

## Motivation
The docs explicitly position this as core identity, parallel to `optax_bayes`.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/features/optax_ekp.md`, `design_docs/filterX/examples/integration.md`

## Implementation Steps
- [ ] Implement the optax state types and init/update logic.
- [ ] Keep the ensemble or UKI state inside optimizer state.
- [ ] Return updates as mean-parameter deltas.
- [ ] Support `optax.chain` composition patterns from the examples.

## Relationships
- Parent epic: FLX-28.
- Blocked by FLX-30, FLX-31, and FLX-32.

---

# docs/tests: add inverse-problem verification and Wave 3 process examples
Draft ID: `FLX-36`
## Problem / Request
Verify process convergence on toy inverse problems and document the process and
optax API workflows.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/examples/components.md`, `design_docs/filterX/examples/models.md`, `design_docs/filterX/examples/integration.md`

## Implementation Steps
- [ ] Add linear or mildly nonlinear inverse-problem checks for EKI/EKS/UKI.
- [ ] Add optax composition smoke tests.
- [ ] Document when to choose EKI vs EKS vs UKI in API-reference form.
- [ ] Defer longer narrative tutorials and integration-heavy notebooks to Wave 6.

## Relationships
- Parent epic: FLX-29.
- Blocked by FLX-34 and FLX-35.


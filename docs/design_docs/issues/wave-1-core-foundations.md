# [Wave 1] Core Foundations

## Shared Context
This grouped draft uses `architecture.md`, `api/README.md`,
`api/primitives.md`, `api/components.md`, `api/models.md`, and `decisions.md`
from the source docs. This is the dedicated core wave for `filterax`.

If this wave is wrong, the package is not coherent:

- protocols are the stable contract between user models and the library
- state/result types define what moves across layers
- ensemble statistics, Kalman gain, and innovation likelihood are the shared
  algebra reused by filters, processes, smoothers, diagnostics, and
  differentiable training

The key design decisions that belong in this wave are:

- one library, not split filter/process repos
- two protocol families: sequential filters and iterative processes
- particles-only state where possible; derive statistics on demand
- `gaussx` as a required dependency for covariance operators and solves
- `optax` as a required dependency because EKP as transforms is core, not extra
- ensemble axis always leading

## Mathematical Baseline
The wave should pin down the main low-level equations:

$$\bar{x} = \frac{1}{N_e}\sum_{j=1}^{N_e} x^{(j)}$$

$$P = \frac{1}{N_e-1} X'^\top X'$$

$$C^{xH} = \frac{1}{N_e-1} X'^\top (HX)'$$

$$K = C^{xH}(C^{HH} + R)^{-1}$$

$$\log p(y \mid \text{forecast}) = -\tfrac{1}{2}\left[N_y\log(2\pi) + \log|S| + v^\top S^{-1}v\right]$$

## Core Public Surface
At the end of this wave, later issues should be able to assume:

- `AbstractSequentialFilter`, `AbstractProcess`
- `AbstractDynamics`, `AbstractObsOperator`, `AbstractLocalizer`,
  `AbstractInflator`, `AbstractNoise`, `AbstractScheduler`
- `FilterState`, `ProcessState`, `UKIState`, `AnalysisResult`,
  `FilterConfig`, `ProcessConfig`
- `ensemble_mean`, `ensemble_anomalies`, `ensemble_covariance`,
  `cross_covariance`, `kalman_gain`, `log_likelihood`, `innovation_statistics`

# [Wave 1] Core Foundations
Draft ID: `FLX-06`
## Goal
Implement the shared protocol, type, and primitive algebra surface that every
later filter/process/smoother issue builds on.

## Why This Wave Exists
This is the release gate for the package itself. If protocols, types, or basic
ensemble algebra are wrong, every later algorithm issue either forks its own
logic or rewrites the contract.

## Wave / Milestone
- Wave: `wave:1`
- Milestone: `v0.1-core`

## Canonical Epics
- [ ] FLX-07 [Epic] 1.A Protocols and State Containers
- [ ] FLX-08 [Epic] 1.B Primitive Statistics, Gain, and Likelihood
- [ ] FLX-09 [Epic] 1.C Core Verification and Docs

## Sequential Dependencies
- `FLX-07` and `FLX-08` can begin in parallel after Wave 0.
- `FLX-09` should follow once the public names stabilize.
- Later algorithm waves should not start implementation until this wave is
  credible.

## Definition of Done
- Protocols and state containers are importable and tested.
- Ensemble statistics, gain, and likelihood primitives are pure, JIT-able, and
  reusable.
- The core docs/tests teach the package contract without relying on private
  notes.

## Relationships
- Blocked by FLX-01.
- Blocks every later wave.

---

# [Epic] 1.A Protocols and State Containers
Draft ID: `FLX-07`
## Theme
Land the protocol families and shared result/state/config types.

## Parent Wave
- Wave epic: `FLX-06`
- Wave label: `wave:1`
- Milestone: `v0.1-core`

## Parallelism
- Can run in parallel with: FLX-08
- Blocked by (inside this wave): none
- Must complete before: FLX-09 and all later algorithm waves

## Definition of Done
- The protocol surface and state/result types are stable enough that later
  filters, processes, and smoothers can implement against them directly.

## Relationships
- Parent wave: FLX-06.

---

# [Epic] 1.B Primitive Statistics, Gain, and Likelihood
Draft ID: `FLX-08`
## Theme
Implement the pure ensemble/Kalman algebra that later algorithms reuse.

## Parent Wave
- Wave epic: `FLX-06`
- Wave label: `wave:1`
- Milestone: `v0.1-core`

## Parallelism
- Can run in parallel with: FLX-07
- Blocked by (inside this wave): none
- Must complete before: FLX-09 and all later algorithm waves

## Definition of Done
- Core ensemble statistics and gain/likelihood primitives exist as pure
  functions with `gaussx`/`lineax` operator support where appropriate.

## Relationships
- Parent wave: FLX-06.

---

# [Epic] 1.C Core Verification and Docs
Draft ID: `FLX-09`
## Theme
Verify the core surface numerically and document the package contract.

## Parent Wave
- Wave epic: `FLX-06`
- Wave label: `wave:1`
- Milestone: `v0.1-core`

## Parallelism
- Can run in parallel with: limited prep only
- Blocked by (inside this wave): FLX-07 and FLX-08
- Must complete before: Wave 2

## Definition of Done
- Linear-Gaussian primitive checks, JAX-transform smoke coverage, and core docs
  examples are in place.

## Relationships
- Parent wave: FLX-06.

---

# core(protocols): implement filter/process protocols and extension points
Draft ID: `FLX-10`
## Problem / Request
Implement the Layer 1 protocol surface for sequential filters, iterative
processes, and pluggable user components.

## User Story
As a user or later implementer, I want stable abstract interfaces for dynamics,
observation operators, localizers, inflators, noise models, and schedulers so
algorithms can stay composable.

## Proposed API
```python
class AbstractSequentialFilter(eqx.Module): ...
class AbstractProcess(eqx.Module): ...
class AbstractDynamics(eqx.Module): ...
class AbstractObsOperator(eqx.Module): ...
class AbstractLocalizer(eqx.Module): ...
class AbstractInflator(eqx.Module): ...
class AbstractNoise(eqx.Module): ...
class AbstractScheduler(eqx.Module): ...
```

## Motivation
These protocols are the stable contract that separates user-owned models from
library-owned ensemble algebra.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/api/components.md`, `design_docs/filterX/architecture.md`

## Implementation Steps
- [ ] Implement the abstract protocol classes as `eqx.Module` subclasses.
- [ ] Keep method signatures aligned with the design docs.
- [ ] Preserve pytree/JIT friendliness and documentation clarity.

## Definition of Done
- Later algorithm issues can inherit from these protocols without redesigning
  signatures.

## Relationships
- Parent epic: FLX-07.
- Blocked by FLX-03 and FLX-04.

---

# core(types): implement shared state, result, and config containers
Draft ID: `FLX-11`
## Problem / Request
Implement the shared state and result containers used across filters, EKP, and
high-level models.

## Proposed API
```python
class FilterState(eqx.Module): ...
class ProcessState(eqx.Module): ...
class UKIState(eqx.Module): ...
class AnalysisResult(eqx.Module): ...
class FilterConfig(eqx.Module): ...
class ProcessConfig(eqx.Module): ...
class AssimilationResult(eqx.Module): ...
class ProcessResult(eqx.Module): ...
```

## Motivation
These containers define what crosses layer boundaries, what gets stored during
filtering/smoothing, and what downstream code can count on.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/architecture.md`, `design_docs/filterX/api/models.md`

## Implementation Steps
- [ ] Implement the shared state/result/config types as `eqx.Module` data
      containers.
- [ ] Preserve static fields like `step` and config counts where intended.
- [ ] Keep optional diagnostics/log-likelihood fields explicit.

## Definition of Done
- Filters, processes, and models can all share the same state/result types.

## Relationships
- Parent epic: FLX-07.
- Blocked by FLX-10.

---

# primitives(statistics): implement ensemble_mean, anomalies, covariance, and cross-covariance
Draft ID: `FLX-12`
## Problem / Request
Implement the core Layer 0 ensemble statistics as pure JAX functions.

## Mathematical Notes
$$\bar{x} = \frac{1}{N_e}\sum_j x^{(j)}$$

$$X' = X - \bar{x}$$

$$P = \frac{1}{N_e-1}X'^\top X'$$

$$C^{xH} = \frac{1}{N_e-1}X'^\top(HX)'$$

## Motivation
Everything else in the package depends on these statistics, so they should live
as simple pure functions rather than being recomputed ad hoc inside every
algorithm.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/api/primitives.md`

## Implementation Steps
- [ ] Implement `ensemble_mean` and `ensemble_anomalies`.
- [ ] Implement `ensemble_covariance` using a `gaussx` low-rank operator rather
      than dense covariance materialization.
- [ ] Implement `cross_covariance` as a dense state/observation cross term.

## Definition of Done
- These functions are JIT-safe, shape-stable, and numerically pinned by tests.

## Relationships
- Parent epic: FLX-08.
- Blocked by FLX-04.

---

# primitives(gain-likelihood): implement kalman_gain, log_likelihood, and innovation_statistics
Draft ID: `FLX-13`
## Problem / Request
Implement the shared Kalman-gain and innovation-likelihood primitives using the
operator conventions from the design docs.

## Mathematical Notes
$$K = C^{xH}(C^{HH}+R)^{-1}$$

$$S = HPH^\top + R$$

$$\log p(y \mid \text{forecast}) = -\tfrac{1}{2}\left[N_y\log(2\pi) + \log|S| + v^\top S^{-1}v\right]$$

## Motivation
This is the algebraic bridge between ensemble statistics and every downstream
analysis/process update.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/api/primitives.md`

## Implementation Steps
- [ ] Implement `kalman_gain` with `gaussx`/`lineax` solve support.
- [ ] Implement `log_likelihood` against structured innovation covariance.
- [ ] Implement `innovation_statistics` as a reusable diagnostics payload.

## Definition of Done
- The gain and innovation functions work on linear-Gaussian toy problems and
  expose the same conventions later filters/processes expect.

## Relationships
- Parent epic: FLX-08.
- Blocked by FLX-12.

---

# docs/tests: add Core-wave verification and package-contract docs for filterax
Draft ID: `FLX-14`
## Problem / Request
Replace the remaining template docs/tests surface with Core-wave verification
and package-contract documentation.

## Motivation
The package needs a credible contract surface before algorithm waves land on
top of it.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/examples/primitives.md`, `design_docs/filterX/examples/components.md`

## Implementation Steps
- [ ] Add linear-Gaussian tests for statistics, gain, and likelihood.
- [ ] Add protocol/type smoke tests under `jax.jit`, `jax.grad`, and `eqx.filter_vmap`.
- [ ] Update docs pages to introduce Layer 0 / 1 contracts.

## Definition of Done
- Core docs and tests are strong enough to support later waves.

## Relationships
- Parent epic: FLX-09.
- Blocked by FLX-10, FLX-11, FLX-12, and FLX-13.


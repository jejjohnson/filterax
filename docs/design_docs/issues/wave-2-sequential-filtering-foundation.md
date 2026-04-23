# [Wave 2] Sequential Filtering Foundation

## Shared Context
This wave draws on `features/filters.md`, `features/localization_inflation.md`,
`api/components.md`, `api/models.md`, and `examples/primitives.md` /
`examples/models.md`.

The design docs divide sequential filtering into three layers:

- Layer 0 primitives for localization, inflation, perturbations, and patches
- Layer 1 analysis-step implementations
- Layer 2 forecast-analysis-inflate loops

Wave 2 should prove the sequential state-estimation side of `filterax` on the
standard algorithms before the more advanced variants and calibration tooling
arrive.

## Mathematical Baseline
Sequential filters all share:

$$C^{xH} = X'Y'^\top, \qquad C^{HH} = Y'Y'^\top$$

$$K = C^{xH}(C^{HH}+R)^{-1}$$

ETKF transform-space update:

$$\tilde{C} = (N_e - 1)I + Y'^\top R^{-1}Y'$$

$$\bar{w}_a = \tilde{C}^{-1} Y'^\top R^{-1}(y - \bar{y}_f)$$

Localization by Schur product:

$$P_{\text{loc}} = \rho \circ P$$

## Wave 2 Public Surface
After this wave, later work should be able to assume:

- `gaspari_cohn`, `gaussian_taper`, `hard_cutoff`, `localize`
- `inflate_multiplicative`, `inflate_rtps`, `inflate_rtpp`,
  `perturbed_observations`
- `create_patches`, `assign_obs_to_patches`, `blend_patches`
- `StochasticEnKF`, `ETKF`, `EnSRF`, `LETKF`
- high-level `ETKF`, `EnSRF`, and `LETKF` assimilation models

## Canonical API Snapshot
The design docs suggest two main user-facing entry patterns in this wave.

High-level assimilation loop:

```python
filter = filterax.LETKF(
    dynamics=my_dynamics,
    obs_op=my_obs_op,
    localizer=filterax.GaspariCohn(radius=radius),
    inflator=filterax.RTPS(alpha=0.7),
    config=filterax.FilterConfig(n_ensemble=50),
)
result = filter.assimilate(init_ensemble, observations, obs_noise=R)
```

Lower-level component usage:

```python
forecast = eqx.filter_vmap(my_dynamics)(particles, t0, t1)
analysis = filterax.filters.ETKF().analysis(forecast, obs, my_obs_op, R)
updated = filterax.inflate_rtps(analysis.particles, forecast, alpha=0.7)
```

# [Wave 2] Sequential Filtering Foundation
Draft ID: `FLX-15`
## Goal
Ship the first end-to-end sequential filtering slice of `filterax`: analysis
primitives, core filter components, and high-level assimilation loops.

## Wave / Milestone
- Wave: `wave:2`
- Milestone: `v0.2-sequential-filters`

## Canonical Epics
- [ ] FLX-16 [Epic] 2.A Localization, Inflation, and Patch Primitives
- [ ] FLX-17 [Epic] 2.B Sequential Filter Components and Models
- [ ] FLX-18 [Epic] 2.C Filter Verification and API Docs

## Sequential Dependencies
- `FLX-16 -> FLX-17 -> FLX-18`.
- Patch/localization primitives should stabilize before LETKF and high-level
  localized models.

## Definition of Done
- The package can run credible ETKF/EnSRF/LETKF assimilation loops on toy
  state-estimation problems.

## Relationships
- Blocked by FLX-06.
- Blocks FLX-37 and FLX-48.

---

# [Epic] 2.A Localization, Inflation, and Patch Primitives
Draft ID: `FLX-16`
## Theme
Implement the reusable analysis-side primitives that sequential filters need.

## Parent Wave
- Wave epic: `FLX-15`
- Wave label: `wave:2`
- Milestone: `v0.2-sequential-filters`

## Parallelism
- Can run in parallel with: none inside this wave
- Must complete before: FLX-17

## Definition of Done
- Localization, inflation, perturbations, and patch decomposition primitives
  are available and tested.

## Relationships
- Parent wave: FLX-15.

---

# [Epic] 2.B Sequential Filter Components and Models
Draft ID: `FLX-17`
## Theme
Implement the first usable sequential analysis components and high-level
forecast-analysis loops.

## Parent Wave
- Wave epic: `FLX-15`
- Wave label: `wave:2`
- Milestone: `v0.2-sequential-filters`

## Parallelism
- Can run in parallel with: limited prep only
- Blocked by (inside this wave): FLX-16
- Must complete before: FLX-18

## Definition of Done
- Foundation filters and models exist with stable interfaces.

## Relationships
- Parent wave: FLX-15.

---

# [Epic] 2.C Filter Verification and API Docs
Draft ID: `FLX-18`
## Theme
Numerically verify the sequential filters and present them clearly in API docs
and reference-level usage notes.

## Parent Wave
- Wave epic: `FLX-15`
- Wave label: `wave:2`
- Milestone: `v0.2-sequential-filters`

## Parallelism
- Can run in parallel with: docs prep only
- Blocked by (inside this wave): FLX-17
- Must complete before: Wave 3 and Wave 4

## Relationships
- Parent wave: FLX-15.

---

# primitives(localization): implement Gaspari-Cohn, Gaussian, cutoff, and localize
Draft ID: `FLX-19`
## Problem / Request
Implement the basic covariance-localization primitives for ensemble DA.

## Mathematical Notes
$$P_{\text{loc}} = \rho \circ P$$

The taper families needed in this wave are:

- Gaspari-Cohn
- Gaussian
- hard cutoff

## References & Existing Code
- Design doc / spec: `design_docs/filterX/features/localization_inflation.md`, `design_docs/filterX/api/primitives.md`

## Implementation Steps
- [ ] Implement `gaspari_cohn`, `gaussian_taper`, and `hard_cutoff`.
- [ ] Implement `localize(cov, taper)`.
- [ ] Keep the functions pure and array-level.

## Definition of Done
- Basic localization tapers work and preserve symmetry/compact-support behavior
  where expected.

## Relationships
- Parent epic: FLX-16.
- Blocked by FLX-12.

---

# primitives(inflation-perturbations): implement multiplicative inflation, RTPS, RTPP, and perturbed observations
Draft ID: `FLX-20`
## Problem / Request
Implement the foundation inflation and stochastic-observation utilities used by
the first sequential filters.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/features/localization_inflation.md`, `design_docs/filterX/api/primitives.md`

## Implementation Steps
- [ ] Implement `inflate_multiplicative`, `inflate_rtps`, and `inflate_rtpp`.
- [ ] Implement `perturbed_observations`.
- [ ] Preserve ensemble-mean behavior where the design docs expect it.

## Definition of Done
- Foundation inflation and stochastic update utilities work for Wave 2
  filters.

## Relationships
- Parent epic: FLX-16.
- Blocked by FLX-12 and FLX-13.

---

# primitives(patches): implement create_patches, assign_obs_to_patches, and blend_patches
Draft ID: `FLX-21`
## Problem / Request
Implement array-level patch decomposition primitives for large-domain localized
filtering.

## Motivation
The design docs treat patch-based decomposition as owned by `filterax` at the
array layer, while xarray orchestration is delegated elsewhere.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/api/primitives.md`, `design_docs/filterX/examples/primitives.md`

## Implementation Steps
- [ ] Implement `create_patches`.
- [ ] Implement `assign_obs_to_patches`.
- [ ] Implement `blend_patches`.

## Definition of Done
- The patch roundtrip is testable at the pure-array level and ready for LETKF
  workflows.

## Relationships
- Parent epic: FLX-16.
- Blocked by FLX-19.

---

# filters(components-basic): implement StochasticEnKF, ETKF, and EnSRF analysis steps
Draft ID: `FLX-22`
## Problem / Request
Implement the first Layer 1 sequential filter components.

## Mathematical Notes
Stochastic EnKF:

$$x_a^{(j)} = x_f^{(j)} + K(y + \varepsilon^{(j)} - Hx_f^{(j)})$$

ETKF and EnSRF should use deterministic square-root style updates.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/features/filters.md`, `design_docs/filterX/api/components.md`

## Implementation Steps
- [ ] Implement `StochasticEnKF`.
- [ ] Implement `ETKF`.
- [ ] Implement `EnSRF`.
- [ ] Reuse Wave 1 primitives instead of re-deriving ensemble statistics inline.

## Definition of Done
- The first deterministic and stochastic analysis-step implementations exist and
  satisfy the sequential-filter protocol.

## Relationships
- Parent epic: FLX-17.
- Blocked by FLX-19 and FLX-20.

---

# filters(localized): implement LETKF analysis and localized observation handling
Draft ID: `FLX-23`
## Problem / Request
Implement the localized ETKF analysis path and the coordination with
localization/patch primitives.

## Motivation
LETKF is central to the state-estimation story in the design docs, especially
for large spatial domains.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/features/filters.md`, `design_docs/filterX/examples/models.md`

## Implementation Steps
- [ ] Implement `LETKF` as a Layer 1 analysis component.
- [ ] Integrate localization radius / coordinate handling cleanly.
- [ ] Support patch-style workflows where appropriate.

## Definition of Done
- Localized filtering exists as a first-class component, not an ad hoc example.

## Relationships
- Parent epic: FLX-17.
- Blocked by FLX-19, FLX-21, and FLX-22.

---

# models(sequential): implement high-level ETKF, EnSRF, and LETKF assimilation loops
Draft ID: `FLX-24`
## Problem / Request
Wrap the Layer 1 components into ready-to-use Layer 2 sequential models.

## Proposed API
```python
class LETKF(eqx.Module):
    def assimilate(...): ...
    def assimilate_and_smooth(...): ...

class ETKF(eqx.Module):
    def assimilate(...): ...

class EnSRF(eqx.Module):
    def assimilate(...): ...
```

## Motivation
The design docs promise high-level models with minimal boilerplate, not just
analysis-step components.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/api/models.md`, `design_docs/filterX/examples/models.md`

## Implementation Steps
- [ ] Implement the forecast-analysis-inflate loops.
- [ ] Return `AssimilationResult` with history and optional log-likelihood.
- [ ] Keep user-owned dynamics and observation operators as pluggable modules.

## Definition of Done
- A newcomer can run a high-level ETKF/EnSRF/LETKF workflow without stitching
  the loop manually.

## Relationships
- Parent epic: FLX-17.
- Blocked by FLX-22 and FLX-23.

---

# docs/tests: add linear-Gaussian checks and Wave 2 filter API docs
Draft ID: `FLX-25`
## Problem / Request
Verify the first sequential filters against known behavior and document the
state-estimation API surface without turning this wave into the main tutorial phase.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/examples/components.md`, `design_docs/filterX/examples/models.md`

## Implementation Steps
- [ ] Add linear-Gaussian RMSE-improvement tests.
- [ ] Add API-reference examples for ETKF/LETKF/EnSRF focused on signatures and return types.
- [ ] Add JAX compatibility coverage for the filter loop.

## Definition of Done
- Wave 2 filters are numerically credible and documented at the API-reference level.
- Tutorial-heavy or narrative examples are deferred to Wave 6.

## Relationships
- Parent epic: FLX-18.
- Blocked by FLX-24.


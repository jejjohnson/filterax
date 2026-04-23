# [Wave 6] Examples, Tutorials, and Integration Playbooks

## Shared Context
This final grouped draft is intentionally example-heavy. It draws on
`examples/primitives.md`, `examples/components.md`, `examples/models.md`, and
`examples/integration.md`, plus the surrounding ecosystem notes in
`boundaries.md`.

The earlier waves should already have landed the actual runtime surfaces. That
lets this final wave focus on teaching, comparison, and workflow packaging
rather than using examples to compensate for missing APIs.

## Example Inventory To Cover
The design docs suggest a fairly rich tutorial surface by the end:

- Layer 0 primitives: statistics, gain, localization, inflation, patches
- Layer 1 components: ETKF analysis step, EKI step, custom localizer/inflator
- Layer 2 models: LETKF, EKI, EKS, UKI
- Integrations: optax, differentiable training, somax, gaussx, geo_toolz, xr_assimilate

## Canonical API Snapshot
The final wave should give readers clear worked examples such as:

```python
# Layer 2 filtering
result = filterax.LETKF(...).assimilate(init_ensemble, observations, obs_noise=R)

# Layer 2 inversion
process = filterax.EKI(forward_fn=G, obs=y, noise_cov=Gamma)
posterior = process.run(init_particles)

# optax integration
tx = filterax.optax.eki(forward_fn=G, obs=y, noise_cov=Gamma)
```

# [Wave 6] Examples, Tutorials, and Integration Playbooks
Draft ID: `FLX-58`
## Goal
Consolidate the tutorial-heavy documentation, worked examples, and ecosystem
playbooks after the implementation waves are stable.

## Wave / Milestone
- Wave: `wave:6`
- Milestone: `v0.6-examples-playbooks`

## Canonical Epics
- [ ] FLX-59 [Epic] 6.A Layered API Walkthroughs
- [ ] FLX-60 [Epic] 6.B Process, Optax, and Differentiable Training Tutorials
- [ ] FLX-61 [Epic] 6.C Ecosystem and Xarray Playbooks

## Sequential Dependencies
- This wave should start only after the implementation surfaces from Waves 1–5 are credible.
- The goal is breadth and teaching value, not to backfill missing runtime functionality.

## Relationships
- Blocked by FLX-48.

---

# [Epic] 6.A Layered API Walkthroughs
Draft ID: `FLX-59`
## Theme
Turn the Layer 0 / Layer 1 / Layer 2 surfaces into coherent walkthrough docs.

## Parent Wave
- Wave epic: `FLX-58`
- Wave label: `wave:6`
- Milestone: `v0.6-examples-playbooks`

## Parallelism
- Can run in parallel with: FLX-60 and FLX-61 once APIs are stable

---

# [Epic] 6.B Process, Optax, and Differentiable Training Tutorials
Draft ID: `FLX-60`
## Theme
Package the process, optax, and differentiable-DA stories into worked tutorials.

## Parent Wave
- Wave epic: `FLX-58`
- Wave label: `wave:6`
- Milestone: `v0.6-examples-playbooks`

## Parallelism
- Can run in parallel with: FLX-59 and FLX-61 once APIs are stable

---

# [Epic] 6.C Ecosystem and Xarray Playbooks
Draft ID: `FLX-61`
## Theme
Collect the ecosystem composition recipes into honest playbooks.

## Parent Wave
- Wave epic: `FLX-58`
- Wave label: `wave:6`
- Milestone: `v0.6-examples-playbooks`

## Parallelism
- Can run in parallel with: FLX-59 and FLX-60 once APIs are stable

---

# examples(layered): add primitives, components, and model walkthrough notebooks/docs
Draft ID: `FLX-62`
## Problem / Request
Create the main layered walkthroughs that teach users how to move from pure
primitives to component-level composition to high-level models.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/examples/primitives.md`, `design_docs/filterX/examples/components.md`, `design_docs/filterX/examples/models.md`

## Implementation Steps
- [ ] Add Layer 0 primitive walkthroughs.
- [ ] Add Layer 1 component walkthroughs.
- [ ] Add Layer 2 model walkthroughs.
- [ ] Keep the examples aligned with the actual shipped APIs from earlier waves.

## Relationships
- Parent epic: FLX-59.
- Blocked by FLX-15, FLX-26, and FLX-48.

---

# examples(integrations): add optax, somax, gaussx, geo_toolz, and xr_assimilate playbooks
Draft ID: `FLX-63`
## Problem / Request
Create the example-heavy integration playbooks that show how `filterax` composes
with the surrounding ecosystem.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/examples/integration.md`, `design_docs/filterX/boundaries.md`

## Implementation Steps
- [ ] Add `optax` and differentiable-training tutorials.
- [ ] Add `somax` and `gaussx` integration playbooks.
- [ ] Add `geo_toolz` and `xr_assimilate` orchestration examples.
- [ ] Keep the docs honest about what is core runtime versus external composition.

## Relationships
- Parent epic: FLX-61.
- Related to FLX-60.
- Blocked by FLX-35, FLX-54, FLX-55, and FLX-56.

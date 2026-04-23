# [Wave 5] Smoothers, Differentiable DA, and Ecosystem Integrations

## Shared Context
This wave uses `features/smoothers.md`, `features/differentiable_da.md`,
`examples/integration.md`, and the ecosystem boundary material in
`boundaries.md`.

By this point, the package should already have:

- a stable sequential filter surface
- a stable process and optax surface
- calibration/diagnostics tooling

Wave 5 is where `filterax` becomes the differentiable DA and ensemble-smoothing
library promised by the vision doc.

This wave should stay focused on algorithmic capability and minimal verification.
The large tutorial-style example and integration-documentation push is deferred
to a final Wave 6 so these issues stay implementation-first.

## Mathematical Baseline
Ensemble smoothers reuse the backward smoother gain:

$$G_t = C^{a,f}_{t,t+1}(C^{f,f}_{t+1})^{-1}$$

$$X_t^s = X_t^a + G_t(X_{t+1}^s - X_{t+1}^f)$$

Differentiable DA relies on the filter loop remaining a pure JAX computation so
gradients can flow through forecast, analysis, likelihood, and hyperparameter
paths.

## Canonical API Snapshot
Backward smoothing:

```python
smoothed = filterax.EnKS().smooth(filter_results=history, forecast_particles=forecast_history)
```

Differentiable training pattern:

```python
def loss_fn(dyn_params, ensemble, observations):
    dynamics = MyDynamics(params=dyn_params)
    model = filterax.ETKF(dynamics=dynamics, obs_op=my_obs_op)
    result = model.assimilate(ensemble, observations, obs_noise=R)
    return -result.log_likelihood
```

# [Wave 5] Smoothers, Differentiable DA, and Ecosystem Integrations
Draft ID: `FLX-48`
## Goal
Complete the package with smoother support and differentiable-training patterns, while keeping tutorial-heavy integrations for the final examples wave.

## Wave / Milestone
- Wave: `wave:5`
- Milestone: `v0.5-smoothers-diffda-integrations`

## Canonical Epics
- [ ] FLX-49 [Epic] 5.A Ensemble Smoothers
- [ ] FLX-50 [Epic] 5.B Differentiable DA and Training Patterns
- [ ] FLX-51 [Epic] 5.C Smoother Verification and Minimal Reference Surfaces

## Sequential Dependencies
- `FLX-49` depends on the sequential filter surface.
- `FLX-50` depends on core/filter likelihood pathways.
- `FLX-51` can progress in parallel once the underlying surfaces exist.

## Relationships
- Blocked by FLX-15, FLX-26, and FLX-37.
- Blocks FLX-58.

---

# [Epic] 5.A Ensemble Smoothers
Draft ID: `FLX-49`
## Theme
Implement the backward-pass smoother family on top of the sequential-filter
history surface.

## Parent Wave
- Wave epic: `FLX-48`
- Wave label: `wave:5`
- Milestone: `v0.5-smoothers-diffda-integrations`

---

# [Epic] 5.B Differentiable DA and Training Patterns
Draft ID: `FLX-50`
## Theme
Make the differentiable data-assimilation story explicit and usable.

## Parent Wave
- Wave epic: `FLX-48`
- Wave label: `wave:5`
- Milestone: `v0.5-smoothers-diffda-integrations`

---

# [Epic] 5.C Smoother Verification and Minimal Reference Surfaces
Draft ID: `FLX-51`
## Theme
Verify the smoother surface and keep only the thinnest reference/integration seams needed before the final examples wave.

## Parent Wave
- Wave epic: `FLX-48`
- Wave label: `wave:5`
- Milestone: `v0.5-smoothers-diffda-integrations`

---

# smoothers(core): implement EnKS, EnsembleRTS, and FixedLagSmoother
Draft ID: `FLX-52`
## Problem / Request
Implement the main smoother surface that refines filtering results with future
observations.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/features/smoothers.md`, `design_docs/filterX/api/components.md`

## Implementation Steps
- [ ] Implement `EnKS`.
- [ ] Implement `EnsembleRTS`.
- [ ] Implement `FixedLagSmoother`.

## Relationships
- Parent epic: FLX-49.
- Blocked by FLX-24.

---

# smoothers(advanced): implement EnsembleSqrtSmoother and IES
Draft ID: `FLX-53`
## Problem / Request
Implement the advanced deterministic and iterative smoother variants from the
design docs.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/features/smoothers.md`

## Implementation Steps
- [ ] Implement `EnsembleSqrtSmoother`.
- [ ] Implement `IES`.
- [ ] Document how these differ from the standard backward-pass smoothers.

## Relationships
- Parent epic: FLX-49.
- Blocked by FLX-52.

---

# differentiable(training): implement log-likelihood training patterns and JAX transform guidance
Draft ID: `FLX-54`
## Problem / Request
Turn the differentiable-DA design notes into concrete implementation and docs
artifacts.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/features/differentiable_da.md`, `design_docs/filterX/examples/integration.md`

## Implementation Steps
- [ ] Add end-to-end differentiable filter-loss examples.
- [ ] Encode the `scan`, `vmap`, `remat`, and JIT patterns that the docs call
      out.
- [ ] Add smoke tests for learning dynamics, observation operators, or filter
      hyperparameters through the filter.

## Relationships
- Parent epic: FLX-50.
- Blocked by FLX-13, FLX-24, and FLX-25.

---

# integrations(minimal): add thin integration seams for optax, somax, and structured noise
Draft ID: `FLX-55`
## Problem / Request
Add the minimal integration-facing surfaces needed before the final tutorial wave, without turning this issue into a giant example pack.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/examples/integration.md`, `design_docs/filterX/boundaries.md`

## Implementation Steps
- [ ] Add minimal reference snippets for `optax` composition.
- [ ] Add minimal reference snippets for `somax` dynamics wrappers.
- [ ] Add minimal reference snippets for `gaussx` structured noise usage.
- [ ] Defer full tutorial notebooks and integration playbooks to Wave 6.

## Relationships
- Parent epic: FLX-51.
- Blocked by FLX-24, FLX-35, and FLX-54.

---

# references(zoo): add minimal geo_toolz/xr_assimilate seams and zoo/reference surfaces
Draft ID: `FLX-56`
## Problem / Request
Document the minimal xarray and zoo-facing seams needed before the final examples wave, while keeping clear that these are not core runtime ownership.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/examples/integration.md`, `design_docs/filterX/boundaries.md`

## Implementation Steps
- [ ] Add `geo_toolz` and `xr_assimilate` composition docs/examples.
- [ ] Add or document the `zoo/` reference surface for continuous-time filters
      and toy dynamical systems.
- [ ] Keep the distinction between maintained core API and educational reference
      code explicit.

## Relationships
- Parent epic: FLX-51.
- Blocked by FLX-24 and FLX-55.

---

# docs/tests: add smoother verification and minimal end-to-end smoke workflows
Draft ID: `FLX-57`
## Problem / Request
Close the loop with smoother verification and minimal end-to-end smoke workflows, while deferring the large example-doc push to Wave 6.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/examples/components.md`, `design_docs/filterX/examples/integration.md`

## Implementation Steps
- [ ] Add smoother-improves-on-filter tests for toy problems.
- [ ] Add minimal end-to-end smoke workflows spanning filtering, smoothing, and differentiable training.
- [ ] Reserve the large narrative example suite and tutorial reading path for Wave 6.

## Relationships
- Parent epic: FLX-48.
- Blocked by FLX-52, FLX-53, FLX-54, FLX-55, and FLX-56.

# [Wave 4] Advanced Filtering and Diagnostics

## Shared Context
This wave draws from `features/filters.md`,
`features/localization_inflation.md`, and `features/diagnostics.md`.

The source docs go beyond the foundation filters and ask for:

- advanced deterministic variants
- adaptive/domain-aware localization and inflation
- a serious DA diagnostics surface for calibration and evaluation

This wave should keep those ideas separate so the package does not end up with
one giant “advanced utilities” issue.

## Mathematical Baseline
Several diagnostics and regularization ideas deserve explicit formulas in the
wave doc so the issues remain self-contained.

Normalized innovation:

$$
u = S^{-1/2}(y - H\bar{x}_f)$$

Desroziers consistency target:

$$\mathbb{E}[d^a (d^o)^\top] \approx R, \qquad \mathbb{E}[d^o (d^f)^\top] \approx HP_fH^\top$$

Spread-skill ratio target:

$$\text{SSR} = \frac{\sqrt{\text{ensemble spread}}}{\text{RMSE}} \approx 1$$

## Canonical API Snapshot
Representative advanced-usage surface in this wave:

```python
localizer = filterax.AdaptiveLocalizer(base_radius=radius0)
inflator = filterax.AdaptiveInflator(min_factor=1.0, max_factor=1.3)
report = filterax.utils.diagnostics.rank_histogram(ensemble, truth)
```

# [Wave 4] Advanced Filtering and Diagnostics
Draft ID: `FLX-37`
## Goal
Expand the sequential DA side with advanced filter variants, adaptive
regularization tooling, and diagnostics for calibration/verification.

## Wave / Milestone
- Wave: `wave:4`
- Milestone: `v0.4-advanced-filters-diagnostics`

## Canonical Epics
- [ ] FLX-38 [Epic] 4.A Advanced Sequential Filter Variants
- [ ] FLX-39 [Epic] 4.B Adaptive Localization, Inflation, and Diagnostics
- [ ] FLX-40 [Epic] 4.C Verification and Calibration Workflows

## Sequential Dependencies
- `FLX-38` and `FLX-39` can run in parallel after Wave 2.
- `FLX-40` should follow the public API stabilization from the first two epics.

## Relationships
- Blocked by FLX-15 and FLX-26.
- Blocks FLX-48.

---

# [Epic] 4.A Advanced Sequential Filter Variants
Draft ID: `FLX-38`
## Theme
Implement the advanced filter variants not required for the first foundation
wave.

## Parent Wave
- Wave epic: `FLX-37`
- Wave label: `wave:4`
- Milestone: `v0.4-advanced-filters-diagnostics`

---

# [Epic] 4.B Adaptive Localization, Inflation, and Diagnostics
Draft ID: `FLX-39`
## Theme
Implement the adaptive regularization surface and the diagnostics catalog.

## Parent Wave
- Wave epic: `FLX-37`
- Wave label: `wave:4`
- Milestone: `v0.4-advanced-filters-diagnostics`

---

# [Epic] 4.C Verification and Calibration Workflows
Draft ID: `FLX-40`
## Theme
Tie the advanced filter and diagnostics surface together into calibration and
verification workflows.

## Parent Wave
- Wave epic: `FLX-37`
- Wave label: `wave:4`
- Milestone: `v0.4-advanced-filters-diagnostics`

---

# filters(advanced): implement ETKF_Livings, EnSRF_Serial, and ESTKF
Draft ID: `FLX-41`
## Problem / Request
Implement the advanced deterministic sequential filter variants from the design
catalog.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/features/filters.md`

## Implementation Steps
- [ ] Implement `ETKF_Livings`.
- [ ] Implement `EnSRF_Serial`.
- [ ] Implement `ESTKF`.

## Relationships
- Parent epic: FLX-38.
- Blocked by FLX-22.

---

# filters(parametric): implement SquareRootKF and domain-aware localized workflows
Draft ID: `FLX-42`
## Problem / Request
Implement the parametric square-root Kalman branch and its place in the
sequential-filter surface.

## Motivation
The design docs include a parametric square-root filter alongside the ensemble
family, and it should have an explicit slot in the backlog.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/features/filters.md`

## Implementation Steps
- [ ] Implement `SquareRootKF`.
- [ ] Define how it fits the Layer 1 / Layer 2 filter surface.
- [ ] Document where it differs from the ensemble variants.

## Relationships
- Parent epic: FLX-38.
- Blocked by FLX-13 and FLX-24.

---

# localization(advanced): implement SOAR, adaptive localization, and domain localization for LETKF
Draft ID: `FLX-43`
## Problem / Request
Implement the advanced localization strategies called out in the localization
feature doc.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/features/localization_inflation.md`

## Implementation Steps
- [ ] Implement SOAR tapering.
- [ ] Implement adaptive localization.
- [ ] Implement explicit domain localization workflows for LETKF.

## Relationships
- Parent epic: FLX-39.
- Blocked by FLX-19 and FLX-23.

---

# inflation(advanced): implement additive, adaptive, and shrinkage-based inflation
Draft ID: `FLX-44`
## Problem / Request
Implement the more advanced inflation strategies and shrinkage regularization
from the design docs.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/features/localization_inflation.md`

## Implementation Steps
- [ ] Implement additive inflation.
- [ ] Implement adaptive inflation.
- [ ] Implement Ledoit-Wolf shrinkage style regularization where appropriate.

## Relationships
- Parent epic: FLX-39.
- Blocked by FLX-20.

---

# diagnostics(basic): implement spread, RMSE, rank histogram, and innovation checks
Draft ID: `FLX-45`
## Problem / Request
Implement the first diagnostics surface for routine filter-health checks.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/features/diagnostics.md`

## Implementation Steps
- [ ] Implement spread.
- [ ] Implement RMSE vs truth.
- [ ] Implement rank histogram.
- [ ] Implement innovation-based checks.

## Relationships
- Parent epic: FLX-39.
- Blocked by FLX-14 and FLX-25.

---

# diagnostics(advanced): implement Desroziers, ESS, chi-squared, spread-skill, CRPS, and DFS
Draft ID: `FLX-46`
## Problem / Request
Implement the advanced diagnostics catalog used for calibration assessment and
covariance diagnosis.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/features/diagnostics.md`

## Implementation Steps
- [ ] Implement Desroziers consistency diagnostics.
- [ ] Implement effective ensemble size and chi-squared checks.
- [ ] Implement spread-skill ratio, CRPS, and degrees of freedom for signal.

## Relationships
- Parent epic: FLX-39.
- Blocked by FLX-45.

---

# docs/tests: add advanced filter verification and diagnostic workflows
Draft ID: `FLX-47`
## Problem / Request
Add the test and docs surface that proves the advanced filters and diagnostics
compose into usable calibration workflows.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/features/diagnostics.md`, `design_docs/filterX/features/localization_inflation.md`

## Implementation Steps
- [ ] Add advanced-filter smoke and regression coverage.
- [ ] Add diagnostic-workflow examples matching the phased workflow in the
      diagnostics doc.
- [ ] Document how diagnostics guide localization/inflation tuning.

## Relationships
- Parent epic: FLX-40.
- Blocked by FLX-41, FLX-42, FLX-43, FLX-44, FLX-45, and FLX-46.


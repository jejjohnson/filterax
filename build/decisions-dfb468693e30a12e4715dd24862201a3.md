---
status: draft
version: 0.1.0
---

# ekalmX — Design Decisions

## Overview

Architecture Decision Records for ekalmX. Each records the context, options considered, decision, and consequences.

---

## D1: One library, not two

**Status:** Accepted

**Context:** The `overview.md` listed two separate names — "FilterProX" (ensemble Kalman processes / inversion) and "EnsFilterX" (ensemble Kalman filters / sequential DA). They serve different users: EKP for calibration/inference, EnKF for state estimation.

**Options:**
- (A) Two separate libraries with shared primitives extracted to a third package
- (B) One library covering both, with shared L0 primitives and separate L1 filter/process families

**Decision:** Option B — one library. Filters and processes share the same Kalman machinery (ensemble statistics, Kalman gain, covariance localization, gaussx integration). Splitting would duplicate infrastructure and force users who need both modes to manage two dependencies.

**Consequences:**
- Larger API surface in a single package
- Two protocol families (`AbstractSequentialFilter`, `AbstractProcess`) keep the modes cleanly separated
- Shared L0 primitives avoid duplication
- Users who only need EKI don't have to learn about LETKF and vice versa (progressive disclosure)

---

## D2: Name — ekalmX

**Status:** Accepted

**Context:** Needed a name covering ensemble Kalman filters, processes, and joint estimation. Candidates: filterX, ensfilterX, filterproX, enkfX, kalmX, ekalmX.

**Decision:** ekalmX — punchy, captures "ensemble Kalman" in 5 characters, follows the ecosystem `*X` convention (somax, gaussx, vardax).

**Consequences:**
- Directory is still `filterX` in the design docs repo; rename when standalone repo is created
- Import: `import ekalmx`

---

## D3: Two protocol families — sequential filters and iterative processes

**Status:** Accepted

**Context:** EnKF (one analysis step per assimilation window) and EKI (iterate until convergence) have different outer loops. Forcing them into a single `AbstractFilter` with `init/update` would be awkward.

**Options:**
- (A) Unified `AbstractFilter` protocol for both
- (B) Two protocol families: `AbstractSequentialFilter` (forecast + analysis) and `AbstractProcess` (init + update)
- (C) No protocols — bare functions, L2 models wrap them

**Decision:** Option B. Sequential filters and iterative processes have genuinely different lifecycles. Shared infrastructure lives at L0 (ensemble statistics, Kalman gain), not in the protocol hierarchy.

**Consequences:**
- Users learn one protocol for their use case, not a lowest-common-denominator abstraction
- Generic code can still be written per-family (e.g., "run any sequential filter on this problem")
- Shared L0 primitives prevent duplication despite separate protocols

---

## D4: Particles-only state — compute statistics on demand

**Status:** Accepted

**Context:** `EnsembleState` could store precomputed mean, anomalies, and covariance alongside particles. This avoids recomputation but creates staleness risk (derived quantities can become inconsistent with particles).

**Options:**
- (A) Store particles + mean + anomalies + covariance
- (B) Store particles only, compute derived quantities via L0 primitives on demand

**Decision:** Option B. Keep state minimal — just `particles: Float[Array, "N_e N_x"]`. Compute `ensemble_mean`, `ensemble_anomalies`, `ensemble_covariance` via L0 functions when needed.

**Consequences:**
- No staleness bugs
- Natural for `eqx.filter_vmap` over ensemble members (leading axis is `N_e`)
- Slight recomputation cost — acceptable since these are O(N_e × N_x) operations
- JAX's caching/XLA fusion will often eliminate redundant computation anyway

---

## D5: gaussx is a required dependency

**Status:** Accepted

**Context:** ekalmX needs ensemble covariance (as low-rank operator), Kalman gain (via Woodbury), log-likelihood (via logdet), and noise models (via lineax operators). These could be reimplemented in ekalmX or delegated to gaussx.

**Options:**
- (A) gaussx optional — ekalmX ships basic implementations, gaussx adds structured ops
- (B) gaussx required — ekalmX delegates all structured covariance operations

**Decision:** Option B. gaussx is the shared linear algebra layer across the ecosystem (also used by optax_bayes, pyrox_gp, vardax). Reimplementing covariance ops in ekalmX would duplicate work and diverge over time.

**Consequences:**
- ekalmX is thin on covariance math — calls `gaussx.recipes.ensemble_covariance`, `gaussx.recipes.kalman_gain`, `gaussx.ops.logdet`
- Users get structured operators (low-rank, diagonal, Kronecker) for free
- ekalmX installation pulls in gaussx (and transitively lineax)
- gaussx must be stable before ekalmX can ship

---

## D6: optax is required — EKP as GradientTransformation

**Status:** Accepted

**Context:** Ensemble Kalman Processes (EKI, EKS, UKI) are iterative update rules that can be framed as optax `GradientTransformation`s. This parallels optax_bayes, where the Bayesian Learning Rule is also an optax transform. The caller provides forward evaluations instead of gradients.

**Options:**
- (A) optax optional — EKP has standalone API only, optax wrapper in examples
- (B) optax required — EKP ships both standalone and optax interfaces

**Decision:** Option B. The optax interface gives users composability for free (schedules, clipping, logging, `optax.chain`). optax is already ubiquitous in the JAX ecosystem. Making it required keeps the optax wrappers in the core library rather than relegating them to examples.

**Consequences:**
- Users can use EKI as a drop-in optimizer in any optax-based training loop
- Parallel API to optax_bayes — both are structured update rules in the optax pattern
- optax is a lightweight dependency (already required by most JAX projects)

---

## D7: Continuous-time filters live in zoo/, not core

**Status:** Accepted

**Context:** The existing `kf_continuous.py` has 9 continuous-time filter variants (Kalman-Bucy, UKF-Bucy, etc.) implemented as `eqx.Module` subclasses integrated via diffrax. These are educational and useful for comparison but serve a different audience than the discrete-time ensemble methods.

**Options:**
- (A) Core library — continuous-time filters as L1/L2 components alongside discrete filters
- (B) Zoo — reference implementations, not maintained to core API standard
- (C) Separate library

**Decision:** Option B. Continuous-time filters go in `zoo/continuous/`. They're valuable as reference implementations and baselines but don't share the ensemble Kalman protocol structure. Including them in core would dilute the API surface and add diffrax as a required dependency.

**Consequences:**
- diffrax stays optional (only needed for zoo and users with ODE-based dynamics)
- Zoo code is tested with smoke tests only (runs without crashing), not full correctness suites
- If demand grows, continuous-time filters could be promoted to core in a later phase

---

## D8: Smoothers are in scope (core library)

**Status:** Accepted

**Context:** Ensemble smoothers (EnKS, ensemble RTS, fixed-lag) are the backward-pass complement to forward filtering. They share ensemble infrastructure and are essential for many DA workflows (reanalysis, parameter estimation, offline state estimation).

**Options:**
- (A) Core — smoothers alongside filters in L1/L2
- (B) Separate module or future phase
- (C) Delegate to gaussx (which already has `rts_smoother` in recipes)

**Decision:** Option A. Smoothers use the same ensemble state, covariance, and gain primitives as filters. They belong in the same library. gaussx's `rts_smoother` is for parametric (mean + covariance) Kalman smoothing; ekalmX owns the ensemble variant.

**Consequences:**
- Adds `_src/filters/` smoothing components (EnKS, ensemble RTS, fixed-lag)
- L2 models can expose `filter_and_smooth()` convenience methods
- Smoother tests verify improvement over filter-only estimates

---

## D9: All filters are differentiable by construction

**Status:** Accepted

**Context:** torchEnKF introduces a "DifferentiableEnKF" as a distinct class because PyTorch requires explicit autograd support. In JAX, if everything is pure functions on arrays, differentiability is automatic via `jax.grad`.

**Decision:** There is no separate `DifferentiableEnKF`. Every filter in ekalmX is differentiable by construction because all operations are pure JAX. Users backpropagate through any filter by wrapping it in `jax.grad`. The `log_likelihood` primitive at L0 provides the training signal.

**Consequences:**
- No special "differentiable" vs "non-differentiable" mode — simplifies the API
- `log_likelihood` is an optional output (only computed when users need it for training)
- Joint state + parameter estimation is just "use `jax.grad` on a filter" — no new concepts
- Users coming from torchEnKF get the same capability with less API surface

---

## D10: Ensemble dimension is the leading axis

**Status:** Accepted

**Context:** Particles can be stored as `(N_e, N_x)` (ensemble leading) or `(N_x, N_e)` (state leading). The existing `enskf_zoo.py` uses `(N_x, N_e)` following the matrix convention (columns are ensemble members).

**Options:**
- (A) `(N_x, N_e)` — matrix convention, columns are members
- (B) `(N_e, N_x)` — batch convention, leading axis is ensemble

**Decision:** Option B. `(N_e, N_x)` is the natural layout for `eqx.filter_vmap` over ensemble members (vmap over axis 0). It aligns with JAX's batch-first convention and makes broadcasting intuitive.

**Consequences:**
- `eqx.filter_vmap(dynamics)(particles, t0, t1)` just works (vmap over leading axis)
- Existing `enskf_zoo.py` code needs transposition during migration
- Matrix operations (cross-covariance, Kalman gain) transpose internally where needed
- Consistent with torchEnKF's convention `(*batch, N_ensem, x_dim)`

---

## D11: pipekit-cycle protocol compatibility — structural only

**Status:** Accepted

**Context:** `pipekit-cycle` defines three `@runtime_checkable` Protocols that overlap with filterax's abstractions:

- `pipekit_cycle.ForwardModel.step(state, dt) -> state` ≈ filterax `AbstractDynamics.__call__(state, t0, t1)`
- `pipekit_cycle.ObservationOperator.__call__(state) -> obs` + `linearize` ≈ filterax `AbstractObsOperator.__call__`
- `pipekit_cycle.AnalysisStep.__call__(forecast, obs, *, obs_op, obs_err_cov) -> analysis` ≈ filterax `AbstractSequentialFilter.analysis`

filterax filters need to slot into pipekit `Sequential` / `Graph` / `Cycle` pipelines without forcing pipekit-cycle as a dependency on every filterax user.

**Options:**
- (A) Inherit — filterax abstracts subclass pipekit-cycle Protocols (requires pipekit import)
- (B) Structural — same method names/shapes, no import, satisfies Protocols via duck typing
- (C) Optional integration module — `filterax.integrations.pipekit` re-declares classes as protocol satisfiers

**Decision:** Option B. filterax abstracts have method **shapes** (argument types, return types) that map cleanly onto the pipekit-cycle Protocols, even though method names diverge where DA-domain conventions warrant it (`analysis` vs `__call__`; `(state, t0, t1)` vs `(state, dt)`). Bridging is a one-line user-side wrapper per concept; filterax does not import pipekit and does not ship the wrappers. `@runtime_checkable` then makes `isinstance(wrapper, pipekit_cycle.AnalysisStep)` pass at runtime without inheritance.

This mirrors pipekit's own rule: algorithm libraries do not import pipekit. The same discipline keeps filterax usable on its own and droppable into pipekit graphs without coupling.

**Consequences:**
- Method shapes on filterax abstracts are chosen with pipekit-cycle bridge clarity in mind (see `integrations/pipekit.md`).
- filterax filters do **not** themselves satisfy `isinstance(filter, pipekit_cycle.AnalysisStep)` — the wrapper does. Wave 5's compatibility test (FLX-55A) asserts `isinstance(FilterAsAnalysisStep(filter), AnalysisStep)`, not the bare filter.
- Compatibility is tested from outside filterax (in pipekit-cycle's own test suite, or in a downstream integration test repo). filterax has no pipekit imports.
- Users who want pipekit's `StatefulOperator` wrapping (e.g., to drive a `Cycle`) write a 5-line wrapper in their own code; filterax does not ship the wrapper.
- If pipekit-cycle's Protocol signatures change, filterax compatibility may need a one-line wrapper update on the user side. We accept this in exchange for zero coupling.

---

## D12: Multi-instrument fusion — both joint and sequential surfaces

**Status:** Accepted

**Context:** The plumax/geostack motivating use case (Tier IV methane attribution) assimilates observations from TROPOMI, EMIT, and GHGSat at different times, native resolutions, and noise characteristics. The original design's `AbstractObsOperator` is single-instrument.

**Options:**
- (A) Sequential only — user composes single-instrument analyses in a loop
- (B) Joint only — `JointObsOperator(ops, noise_covs)` combinator wraps a list as one obs operator
- (C) Both — sequential wrapper for temporally separated independent overpasses, joint combinator for shared-state simultaneous observations

**Decision:** Option C. Ship both:

- `JointObsOperator(ops: tuple[AbstractObsOperator, ...], noise_covs: tuple[AbstractLinearOperator, ...])` — concatenates outputs of each `op(state)` into one stacked observation vector, returns a block-diagonal noise covariance. One analysis call, correct cross-instrument covariance when noise is block-diagonal.
- `SequentialAssimilation(filter: AbstractSequentialFilter)` — Layer 2 helper that loops over a list of `(obs, obs_op, obs_noise)` tuples, calling `filter.analysis` once per instrument. Used when overpasses are temporally separated and there is no benefit to a joint update.

**Consequences:**
- `AbstractObsOperator` stays single-instrument; joint behaviour is composition, not a new protocol.
- Block-diagonal noise representation lives in `gaussx` (`BlockDiagonalLinearOperator`). filterax does not reimplement.
- Partial obs masking (e.g., missing pixels in one overpass) handled by per-instrument operators with NaN-aware reduction; filterax provides a `MaskedObsOperator` wrapper as a primitive in Wave 2.
- Documented in `features/multi_instrument.md`; lands in Wave 2 alongside the foundation filters.

---

## D13: Coordinate-aware carriers via adapter, not core type

**Status:** Accepted

**Context:** Downstream consumers (geostack, plumax) want ensemble particles that carry coordinate metadata (CRS, affine transform, dim names) — `coordax.Array` or `GeoTensor`. The original design's `FilterState.particles: Float[Array, "N_e N_x"]` is a bare JAX array.

**Options:**
- (A) Adapter pattern — core stays JAX-array only; a `CarrierAdapter` flattens/unflattens coordax↔array around analysis calls
- (B) PyTree leaves — relax type to allow particles to be any JAX PyTree; primitives gain an internal flatten step
- (C) Defer

**Decision:** Option A. `FilterState.particles` is `Float[Array, "N_e N_x"]` in the core type. A `CarrierAdapter` interface lives in `filterax.integrations`:

```python
class CarrierAdapter(eqx.Module):
    def flatten(self, carrier) -> tuple[Float[Array, "N_e N_x"], CarrierMeta]: ...
    def unflatten(self, particles: Float[Array, "N_e N_x"], meta: CarrierMeta) -> Carrier: ...
```

filterax ships **one** reference implementation — `DictAdapter` for `dict[str, Array]` schemas — because it has zero optional dependencies and covers the common mixed-shape state case. Adapters for `coordax.Array`, `GeoTensor`, or other coordinate-aware carriers are **downstream-owned** (geostack, plumax, user code); filterax does not pull `coordax` or `pyproj` for the carrier path.

Filters never see metadata. Adapters are user-facing convenience; the math layer stays minimal.

**Consequences:**
- Core stays lean, JAX-pure, and free of optional carrier dependencies.
- Coordinate awareness costs one flatten/unflatten per analysis call — cheap relative to the analysis itself.
- filterax owns the `CarrierAdapter` interface and the `DictAdapter` reference. Coordinate-aware adapters (e.g. `CoordaxAdapter`, `GeoTensorAdapter`) live in the libraries that own those carriers; `integrations/geostack.md` documents the expected shape so downstream authors can build them consistently.
- Documented in `integrations/geostack.md`.

---

## D14: Geospatial localization owned by filterax

**Status:** Accepted

**Context:** The plumax Tier IV use case needs Gaspari–Cohn tapering over **geographic distance** (great-circle or projected), not grid indices. This requires:

1. A way to express coordinates of state grid points and observations in a shared frame (`LocalFrame`: CRS + projection origin, built with pyproj — non-JAX, static).
2. A distance kernel inside `jax.jit` that consumes precomputed pairwise distances and applies a taper.

Geostack already disclaims responsibility for filtering math (master plan §9). pyproj produces static metadata, not JAX-traceable computation.

**Options:**
- (A) Filterax owns `GeoLocalizer`; consumes pyproj-built `LocalFrame` as static metadata
- (B) Geostack owns `GeoLocalizer`; filterax stays generic
- (C) Generic localizer only; users build geographic distance matrices themselves

**Decision:** Option A. filterax adds `GeoLocalizer(coords, frame, radius, taper="gaspari_cohn")` to the localizer catalogue in Wave 4. `coords` are static `(N_x, 2)` lon/lat arrays; `frame` is a `LocalFrame` built once with pyproj (held as static metadata on the localizer, not inside `jax.jit`). Pairwise distances are precomputed at construction; the JIT path is just `taper(distances / radius)`.

**Consequences:**
- Adds a soft (optional) dependency on `pyproj` for `LocalFrame` construction. The localizer itself is JAX-pure once built; pyproj is not on the analysis path.
- Distance-based covariance tapering remains filterax's core competency.
- Documented in `features/localization_inflation.md`; lands in Wave 4.
- Compatible with patcher-based LETKF (D16): the patcher provides per-patch coordinate slices, `GeoLocalizer` provides the per-patch taper.

---

## D15: Filter state is serializable

**Status:** Accepted

**Context:** Operational consumers (plumax alert service, long forecast cycles) need to checkpoint and restore filter state across process boundaries. The cold-start budget for the alert service is ≤5 s, so warm-started ensembles must round-trip from disk.

**Decision:** `FilterState`, `ProcessState`, `UKIState`, `AnalysisResult`, `FilterConfig`, and `ProcessConfig` are fully `eqx.tree_serialise_leaves`-compatible. Wave 4 adds two thin helpers:

```python
filterax.save_state(path: str | Path, state: FilterState) -> None
filterax.load_state(path: str | Path, like: FilterState) -> FilterState
```

These wrap `eqx.tree_serialise_leaves` / `eqx.tree_deserialise_leaves` with a versioned magic-byte header so future schema changes can be detected.

**Consequences:**
- All static-field choices on state types must be JSON-serializable scalars (int, str, bool, None). Custom classes in static fields are rejected.
- `AbstractLinearOperator` fields inside `ProcessState` are serialized by their PyTree structure; structure must be reconstructible by gaussx.
- Documented in `features/state_persistence.md`; lands in Wave 4.

---

## D16: Patcher-based localized DA lives in filterax

**Status:** Accepted

**Context:** Large spatial domains (continental, basin-scale) cannot fit a full ensemble × full state in device memory. The original ekalmX design includes ~1,379 lines of patch decomposition code as filterax-owned primitives. Geostack has a separate `geotoolz.patch` (incubating) for general-purpose spatial windowing.

**Options:**
- (A) filterax owns patcher implementation (sampler + stitcher) plus patcher-aware filters
- (B) filterax consumes `geotoolz.patch` as a dependency; ships only patcher-aware filters
- (C) Hybrid — filterax ships a minimal in-house patcher; promotes to `geotoolz.patch` consumption once that API stabilizes

**Decision:** Option C. Wave 2 lands `create_patches`, `assign_obs_to_patches`, `blend_patches` as L0 primitives inside filterax (as originally planned). Wave 4 adds `LocalEnKF` / patcher-LETKF as L2 models. Once `geotoolz.patch` stabilizes (post-v0.1), filterax migrates to consuming it via an `AbstractPatcher` protocol; the in-house primitives are kept as a reference implementation and as a fallback for users who don't want a geostack dependency.

**Consequences:**
- Wave 2/4 deliverables are unchanged.
- `AbstractPatcher` protocol added to the extension-point set in Wave 4 (mirrors `AbstractLocalizer` / `AbstractInflator`).
- Documented in `features/localization_inflation.md` (patcher-LETKF section) and `integrations/geostack.md`.

---

## D17: Latent ensemble DA via composition + dedicated wrappers

**Status:** Accepted (v0.1, design only — implementation lands with the
`pipekit_cycle.latent` foundation).

**Context:** Latent data assimilation (Peyron 2021, Cheng 2023) runs
the EnKF analysis on a low-dimensional latent ensemble $\{z^{(i)}\}$
obtained from an autoencoder $(\varphi, \psi)$. The wins are
(a) cheaper Kalman gain ($O(N_z^3)$ vs $O(N_y^3)$ when $N_z < N_y$),
(b) smaller backprop tape, (c) implicit smoothness regularisation from
$\psi$. Filterax's core ensemble math is dimension-agnostic, so latent
filtering should *not* require new analysis-step classes — only
ergonomic glue.

**Options.**

(A) Add new abstract protocols `AbstractLatentFilter` and
    `AbstractLatentDynamics`; reimplement ETKF/LETKF in z-space.
(B) Compose existing `AbstractDynamics` / `AbstractObsOperator` with
    thin adapter classes; add Layer-2 wrappers (`LatentETKF`,
    `LatentLETKF`) that pre-bind those adapters.
(C) Document the pattern but ship no code — let users write the
    glue inline.

**Decision:** Option B.

**Rationale.**

* The ensemble math in `statistics.py`, `gain.py`, `localization.py`,
  `inflation.py` is dimension-agnostic. Re-implementing it as
  "z-versions" (Option A) would duplicate code without changing the
  math.
* Option C (no code) leaves every project to reinvent the same five
  adapter classes. The ergonomics matter — `LatentETKF(latent_map,
  dynamics, obs_op)` in one line is the difference between latent DA
  being a research curiosity and being a default offering.
* `pipekit_cycle.LatentMap` (new) gives a substrate-neutral name for
  the AE; consuming it structurally preserves D11 (no pipekit import
  in filterax core).

**Consequences.**

* New Layer-1 components: `LatentDynamics`, `LiftedObs`,
  `EncodedDynamics` in `filterax/_src/latent.py`.
* New Layer-2 wrappers: `LatentETKF`, `LatentLETKF` in
  `filterax/_src/models.py`.
* New Layer-0 helpers: `latent_ensemble`, `decode_ensemble`,
  `identity_latent_map`.
* No new abstract protocols; the existing `AbstractDynamics`,
  `AbstractObsOperator`, `AbstractSequentialFilter` cover everything.
* D11 (structural pipekit-cycle compatibility) extends transparently:
  `pipekit_cycle.LatentMap` and `LatentForwardModel` are consumed
  structurally; the optional `pytest.importorskip("pipekit_cycle")`
  compat suite gains a `test_latent_pipekit_compat.py` module.
* Localization in latent space is offered with a documented caveat —
  latent dimensions are global modes by default, so the default is
  "no localization"; mode-coordinate localization is marked
  experimental in v0.1.
* Documented in `features/latent_da.md`.

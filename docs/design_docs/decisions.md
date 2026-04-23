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

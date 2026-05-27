---
status: draft
version: 0.1.0
---

# Research — operational data assimilation patterns

**Subject:** Sketches of how filterax fits into latency-sensitive, streaming, and online DA settings — the operational tail of the research-to-production arc.

**Status:** Research note, not API spec. The concrete API commitments live in `features/`, `integrations/`, and `decisions.md`.

---

## 1  Why this doc exists

filterax's design is shaped by both research workflows (notebooks, papers, sweeps) and operational workflows (alert services, scheduled cycles, retraining loops). The research side is well-covered by `vision.md` and the feature docs; the operational side is more dispersed. This note collects the operational requirements in one place and traces them to filterax's design commitments.

---

## 2  Operational workflows of interest

### 2.1  Methane attribution alert service (plumax)

A FastAPI service ingests satellite L2 retrievals as they land, runs an ensemble DA cycle, and emits an alert payload (location, source rate estimate, uncertainty) within a fixed latency budget.

| Characteristic | Value |
|---|---|
| Latency budget | ≤5 s per alert |
| Trigger | New L2 retrieval available on the catalog |
| State carrier | `coordax.Array` or `GeoTensor` (coordinate-aware) |
| State dimension | 20 (Tier I Gaussian plume) — 10⁴ (Tier III gridded transport) |
| Ensemble size | 50–1000 |
| Observation dimension | 100–1000 pixels per overpass |
| Restart frequency | Every new overpass; warm-started from disk |

filterax design responses:

- `save_state` / `load_state` for warm-start (D15, `features/state_persistence.md`).
- `JointObsOperator` for multi-instrument fusion (D12, `features/multi_instrument.md`).
- `GeoLocalizer` for geographic localization (D14, `features/localization_inflation.md`).
- `CarrierAdapter` for coordinate-aware particles (D13, `integrations/geostack.md`).

### 2.2  Continental reanalysis

A scheduled job processes a year of satellite observations as a chain of cycles, producing a reanalysis dataset for downstream science. Latency does not matter; throughput, reproducibility, and post-hoc smoothing do.

| Characteristic | Value |
|---|---|
| Latency budget | Hours per cycle is fine |
| State dimension | 10⁴–10⁶ (continental grid) |
| Ensemble size | 200–500 |
| Smoother | Fixed-lag or full RTS over week-scale windows |
| Output | xarray / zarr dataset of analysis + smoother + uncertainty |

filterax design responses:

- `LocalEnKF` (patcher-LETKF) for domain decomposition (D16).
- Smoothers as a first-class deliverable, not an extension (D8, `features/smoothers.md`).
- xarray orchestration via `xr_assimilate` — outside filterax (`boundaries.md`).

### 2.3  Online retraining of neural surrogates

A neural RTM is continuously retrained against new L1 radiance as it lands. The training signal is observation-space log-likelihood through the full filter.

| Characteristic | Value |
|---|---|
| Loss | `-log_likelihood` over a rolling time window |
| Gradient path | RTM params → obs operator → analysis → log-likelihood |
| Training cadence | Daily or per-batch |
| Checkpoint frequency | After each batch |

filterax design responses:

- Differentiability by construction (D9, `features/differentiable_da.md`).
- Filter state checkpointing (D15).
- No special "training mode" — `jax.grad` on a filter call is the API.

---

## 3  Cross-cutting concerns

### 3.1  Cold-start budget

The ≤5 s budget breaks down approximately as:

| Cost | Budget |
|------|--------|
| Process start + Python imports | ~1 s |
| `load_state` from disk | ~0.1 s |
| JIT-compile the analysis graph | 1–3 s (first call only) |
| Actual analysis | < 0.5 s for 200-member, 10⁴-state ensemble |

JIT compilation is the dominant cost on cold start. Mitigations:

- **Persistent JIT cache.** `jax.experimental.compilation_cache.set_cache_dir(...)` keeps the compiled XLA computation on disk across process restarts.
- **AoT compilation.** `jax.jit(fn).lower(args).compile()` produces a compiled object that can be pickled and reloaded. Experimental but increasingly stable.
- **Long-lived workers.** If the deployment can keep a Python process alive (uvicorn workers, long-running pods), JIT only happens once.

filterax doesn't ship a recipe for any of these — they're deployment concerns. We commit to *not making them harder*: every JIT-able path through filterax is a single static-shape function, no per-overpass recompilation.

### 3.2  Streaming observations

The alert service consumes observations one at a time as they land. Each analysis is independent (modulo warm-state). The pattern is:

```
loop:
    wait for new overpass
    load filter state from disk
    propagate ensemble through dynamics
    analysis(forecast, new_obs, op, noise)
    save filter state to disk
    emit alert
```

filterax supports this pattern directly via the `analysis` + `save_state` / `load_state` surfaces. There is no "streaming filter" abstraction — the streaming loop is user-owned.

For pipekit integration, the streaming loop wraps a filterax filter in `StatefulOperator` (`integrations/pipekit.md` §4.3). pipekit's `Cycle` then drives the loop.

### 3.3  Async observation fetching

Real alert services fetch L1B / L2 data from cloud storage asynchronously. filterax has nothing to say about async — every filterax function is sync, pure-JAX. The async fetch happens in user code; filterax sees only the resulting array.

### 3.4  Hybrid ensemble-variational for tight latency

For latency-critical applications (alert service), pure ensemble methods are usually fast enough at moderate ensemble sizes (~100 members). When they're not, the operational pattern is:

- Use the ensemble for **background covariance** (flow-dependent, captures uncertainty).
- Run L-BFGS or another gradient method for the analysis update (deterministic, fast).

This is hybrid EnVar; it crosses the boundary between filterax and `vardax`. We don't ship it. The open-questions section of `boundaries.md` tracks the eventual cross-library design.

---

## 4  What filterax commits to and what it doesn't

### Commitments

- **Pure JAX analysis path.** No Python control flow per analysis step; everything inside `jax.jit`.
- **Serializable state.** D15 — `save_state` / `load_state` round-trip every state type.
- **Static-shape API.** Observation vector shape is fixed at JIT time; masking handled via `MaskedObsOperator` with dynamic mask.
- **No hidden mutation.** Every filter is pure; warm-start from disk is bit-identical.

### Non-commitments

- **No latency guarantees.** filterax does not measure or budget cold-start time. That's deployment-side.
- **No async API.** Sync, pure-JAX. Async happens outside.
- **No metric emission.** Logs, traces, Prometheus metrics — all user-owned.
- **No "operational mode."** There's no flag that flips filterax into operational behavior. The same library serves notebooks and alert services.

---

## 5  Open questions

1. **AoT compilation for filter graphs.** `jax.experimental.serialize_executable` is maturing. When stable, filterax could ship a recipe for compiling an analysis graph once and reloading per process.
2. **Static-typed observation schema.** A `JointObsOperator` over a fixed instrument tuple has a fixed observation shape. Could filterax provide a `@filter_for(joint_op)` decorator that pre-computes shapes? Defer until pressure is concrete.
3. **Distributed analysis.** Patcher-LETKF across multiple devices via `jax.pmap` or `shard_map`. Defer to Wave 4+ once the in-house patcher lands.

These all live in the research column for now. They become design changes only when a downstream consumer (plumax, geostack) puts concrete pressure on them.

---

## 6  References

- `vision.md` — motivating use case and downstream pressures
- `integrations/plumax.md` — worked Tier IV recipe
- `features/state_persistence.md` — full serialization contract
- `features/multi_instrument.md` — fusion patterns
- `features/differentiable_da.md` — gradient pipe for neural surrogates
- pipekit `pipekit-train` design docs — analogue patterns for training loops

---
status: draft
version: 0.1.0
---

# filterax — State persistence

**Subject:** Serialization contract for `FilterState`, `ProcessState`, `UKIState`, and related types. Warm-start patterns for operational deployments.

**Decision anchor:** [D15](../decisions.md#d15-filter-state-is-serializable)

**Lands in:** Wave 4 (`issues/wave-4-*.md`, task `FLX-46A`)

---

## 1  Why

Operational consumers — the plumax alert service, long forecast cycles, retraining loops — need to checkpoint and restore filter state across process boundaries. Concretely:

- **Plumax alert service.** Cold-start budget is ≤5 s. Re-initializing the ensemble from scratch would blow that budget; warm-starting from disk does not.
- **Long forecast cycles.** A multi-week reanalysis is run as a chain of jobs, each picking up where the previous left off.
- **Retraining loops.** A differentiable filter is checkpointed mid-training to recover from crashes.

filterax addresses these uniformly: every state type is fully `eqx.tree_serialise_leaves`-compatible.

## 2  The contract

Every filterax state type satisfies the following:

1. **Pure PyTree.** All non-static fields are JAX arrays or other filterax state types. No Python objects with side effects.
2. **JSON-serializable statics.** All static fields are `int`, `str`, `bool`, `float`, `None`, or tuples thereof. No `numpy.ndarray`, no custom classes, no closures in statics. This is enforced at construction time.
3. **Round-trippable.** `load_state(save_state(s)) == s` for every state `s`, modulo float-bit-identity.

The types covered:

| Type | Status |
|------|--------|
| `FilterState` | Required |
| `ProcessState` | Required |
| `UKIState` | Required |
| `AnalysisResult` | Required |
| `FilterConfig` | Required |
| `ProcessConfig` | Required |
| `AbstractDynamics` instances | Best-effort (user-defined; check their static-field types) |
| `AbstractObsOperator` instances | Best-effort |

The "best-effort" classes are user-supplied; filterax has no way to enforce the contract for them. The save/load helpers detect violations and raise a clear error.

## 3  The helpers

Wave 4 adds two thin wrappers around `eqx.tree_serialise_leaves`:

```python
def save_state(path: str | Path, state) -> None:
    """Serialize a filter / process state (and friends) to disk.

    Writes a 4-byte magic header (b"FAX1") followed by the eqx-serialized leaves.
    Versioning lets future schema changes raise a clear error on load.
    """

def load_state(path: str | Path, like):
    """Deserialize a state from disk using `like` as a structure template.

    `like` must be the same type as the saved state, with leaves of matching shape
    and dtype but arbitrary values (the standard eqx idiom).
    """
```

Both raise `filterax.SerializationError` on:

- Missing or wrong magic header (file is not a filterax state, or a future version).
- PyTree structure mismatch between file and `like`.
- Static-field type validation failure at construction time.

## 4  Recipe — operational alert warm-start

```python
from pathlib import Path

import filterax
import jax.numpy as jnp

STATE_PATH = Path("/var/run/plumax/last_state.fax")

# Template for load — empty arrays of the right shape/dtype.
# FilterState.step is a scalar JAX integer array (Int[Array, ""]), not a Python int.
template = filterax.FilterState(
    particles=jnp.zeros((N_ENSEMBLE, N_STATE)),
    step=jnp.array(0),
)

# Warm start
state = filterax.load_state(STATE_PATH, like=template) if STATE_PATH.exists() else template

# ... assimilate latest overpass into `state` ...

# Persist for next run
filterax.save_state(STATE_PATH, state)
```

The JIT-compiled analysis graph is preserved by the Python process if it stays alive; warm-start covers the case where the process is recycled (typical for serverless / FaaS deployments).

## 5  Recipe — checkpointed training loop

```python
import filterax
import equinox as eqx

@eqx.filter_jit
def step(state, batch):
    ...
    return new_state, loss

state = init_state
for epoch in range(N_EPOCHS):
    for batch in dataloader:
        state, loss = step(state, batch)

    # Checkpoint every epoch
    filterax.save_state(f"checkpoints/epoch_{epoch:03d}.fax", state)
```

Recovery picks the most recent checkpoint, deserializes, and resumes.

## 6  Versioning

The 4-byte magic header (`b"FAX1"`) is the only schema version filterax commits to. Schema-breaking changes (renamed fields, restructured PyTrees) bump the magic byte. Old checkpoints raise `SerializationError` on load with a clear message describing the version mismatch and pointing at a migration script (when one exists).

We do **not** commit to forward compatibility — a newer filterax must be able to read older checkpoints only when there is no PyTree change.

## 7  Things to be careful about

### 7.1  `AbstractLinearOperator` in `ProcessState`

`ProcessState.noise_cov` is a gaussx / lineax operator. Serialization works as long as the operator is itself a PyTree with JSON-serializable statics (the gaussx ones are). User-defined operators must follow the same discipline.

### 7.2  PRNG keys

Filter state does **not** include a PRNG key. Stochastic filters (`StochasticEnKF`) take the key as an argument to `analysis`. Users who want deterministic restart must persist their key separately.

### 7.3  JIT cache

`save_state` / `load_state` move **state**, not compiled code. After a fresh process, the first analysis will re-JIT. If startup latency matters, persist the JIT cache via `jax.experimental.compilation_cache`.

### 7.4  Cross-platform compatibility

filterax checkpoints are JAX-array bytestreams. They are **not** portable across float dtypes (a `float64` checkpoint won't load into a `float32` template). They **are** portable across CPU / GPU / TPU.

## 8  Out of scope

- **Streaming to remote stores** (S3, GCS) — use `fsspec` in user code; filterax takes a local path.
- **Incremental / append-mode checkpoints** — every save is a full state dump.
- **Schema migration tools** — when D15 needs to change, we ship a one-off migration script; not a general framework.

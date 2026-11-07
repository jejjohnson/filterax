---
status: draft
version: 0.1.0
---

# filterax × pipekit-cycle

**Subject:** How filterax slots into `pipekit-cycle` pipelines without importing pipekit.

**Decision anchor:** [D11 — Structural Protocol satisfaction](../decisions.md#d11-pipekit-cycle-protocol-compatibility--structural-only)

---

## 1  The problem

`pipekit-cycle` defines three `@runtime_checkable` Protocols that describe a generic data-assimilation cycle:

```python
# from pipekit_cycle.protocols
@runtime_checkable
class ForwardModel(Protocol):
    def step(self, state, dt) -> state: ...
    @property
    def dt(self) -> float: ...
    @property
    def state_signature(self) -> Any: ...

@runtime_checkable
class ObservationOperator(Protocol):
    def __call__(self, state) -> obs: ...
    def linearize(self, state) -> Callable: ...

@runtime_checkable
class AnalysisStep(Protocol):
    def __call__(self, forecast, obs, *, obs_op, obs_err_cov) -> analysis: ...
```

These overlap with filterax's `AbstractDynamics`, `AbstractObsOperator`, and `AbstractSequentialFilter`. A filterax filter should drop into a pipekit `Sequential` / `Graph` / `Cycle` without:

- forcing every filterax user to install pipekit, or
- forcing filterax to track pipekit's protocol churn.

## 2  The decision

filterax classes **structurally satisfy** the pipekit Protocols — same method names, same call shapes — but filterax itself **never imports pipekit**. The compatibility is duck-typed; `@runtime_checkable` makes `isinstance(filter, pipekit_cycle.AnalysisStep)` work at runtime without inheritance.

This mirrors pipekit's own rule (algorithm libraries do not import pipekit). It also matches how `vardax`, `pyrox_gp`, and other ecosystem libraries integrate.

## 3  Compatibility table

| filterax surface | pipekit Protocol | Adapter required? |
|------------------|------------------|-------------------|
| `AbstractDynamics.__call__(state, t0, t1)` | `ForwardModel.step(state, dt)` | One-line wrapper (see §4.1) |
| `AbstractObsOperator.__call__(state)` | `ObservationOperator.__call__(state)` | None — identical surface |
| `AbstractObsOperator.linearize(state)` *(optional)* | `ObservationOperator.linearize(state)` | None when present |
| `AbstractSequentialFilter.analysis(forecast, obs, obs_op, obs_noise)` | `AnalysisStep.__call__(forecast, obs, *, obs_op, obs_err_cov)` | One-line wrapper (see §4.2) |

The Dynamics and AnalysisStep surfaces differ in argument names and time-stepping convention; both are bridged by trivial wrappers users write in their own code.

## 4  Recipes

### 4.1  Use a filterax filter as a pipekit `AnalysisStep`

```python
# user code — neither filterax nor pipekit-cycle imports the other
import filterax
import pipekit_cycle as pkc

filter_ = filterax.ETKF()  # or LETKF, EnSRF, ...

class FilterAsAnalysisStep:
    """Bridge filterax.AbstractSequentialFilter.analysis -> pkc.AnalysisStep."""

    def __init__(self, filter_):
        self._filter = filter_

    def __call__(self, forecast, obs, *, obs_op, obs_err_cov):
        return self._filter.analysis(forecast, obs, obs_op, obs_err_cov)

step = FilterAsAnalysisStep(filter_)
assert isinstance(step, pkc.AnalysisStep)   # passes — runtime_checkable
```

### 4.2  Use a filterax dynamics inside a pipekit `Cycle`

filterax `AbstractDynamics.__call__(state, t0, t1)` carries explicit start and end times; pipekit's `step(state, dt)` carries a step size. The bridge is one line:

```python
class DynamicsAsForwardModel:
    def __init__(self, dyn, dt):
        self._dyn = dyn
        self.dt = dt

    def step(self, state, dt=None):
        dt = self.dt if dt is None else dt
        return self._dyn(state, 0.0, dt)
```

`state_signature` is optional in pipekit; provide it if you want pipekit's shape inference. filterax does not require it.

### 4.3  Wrap a filter as a pipekit `StatefulOperator`

pipekit threads `(carrier, state) -> (carrier, state)` through `Sequential` graphs. A filterax filter does not own its history; the user does. The wrapper is again user-side:

```python
from typing import Any

import filterax
import pipekit as pk

class FilterCycle(pk.StatefulOperator):
    """A filterax filter wrapped as a stateful pipekit operator."""

    _is_stateful = True

    filter_: filterax.AbstractSequentialFilter
    obs_op: filterax.AbstractObsOperator
    obs_noise: Any                # gaussx / lineax operator

    def _apply(self, carrier, state):
        # carrier: (forecast_ensemble, obs) tuple
        # state: filterax.FilterState
        forecast, obs = carrier
        result = self.filter_.analysis(forecast, obs, self.obs_op, self.obs_noise)
        new_state = filterax.FilterState(
            particles=result.particles,
            step=state.step + 1,
        )
        return result, new_state
```

## 5  Naming and signature alignment

filterax keeps method names and argument orders in step with pipekit's Protocols whenever there is no domain reason to diverge. Where they do diverge:

- **Time arguments.** filterax dynamics take `(state, t0, t1)` (start, end). pipekit takes `(state, dt)` (step size). Bridge in user code; do not force one convention on the other.
- **Noise covariance.** filterax accepts an `AbstractLinearOperator` (gaussx-shaped). pipekit accepts any array-shaped covariance. The wrapper passes through; gaussx operators behave like arrays for elementwise use.

## 6  What we do *not* ship

- `filterax.integrations.pipekit` is **not** a module. There is no Python code in filterax that mentions pipekit.
- The wrappers above live in user code or in pipekit-cycle's own examples.

## 7  Testing the contract

filterax does not test pipekit compatibility directly. Wave 5 adds an opt-in test guarded by `pytest.importorskip("pipekit_cycle")`:

```python
def test_filter_satisfies_pipekit_analysis_step():
    import pipekit_cycle as pkc
    filter_ = filterax.ETKF()
    step = FilterAsAnalysisStep(filter_)
    assert isinstance(step, pkc.AnalysisStep)
```

This catches accidental signature drift without coupling the libraries at import time. See `issues/wave-5-*.md` task `FLX-55A`.

## 8  Why not just import pipekit?

- **Optionality.** filterax should run for users who don't use pipekit at all.
- **Versioning.** pipekit's Protocols are still pre-1.0. Importing them locks filterax to a moving target.
- **Symmetry.** `vardax`, `pyrox_gp`, and other algorithm libraries follow the same rule. The ecosystem decision is consistent.

If the cost of duck typing becomes untenable (e.g., pipekit grows non-trivial helper functions every filterax user wants), revisit by adding `filterax[pipekit]` extras with a thin adapter module. That is a future change, not a current commitment.

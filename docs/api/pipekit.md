# pipekit integration

[pipekit-cycle](https://github.com/jejjohnson/pipekit) orchestrates
data-assimilation cycles through three runtime-checkable Protocols —
`ForwardModel`, `ObservationOperator`, and `AnalysisStep` — and algorithm
libraries plug in *structurally*, by matching the protocol signatures.
filterax never imports pipekit: the adapters in `filterax.pipekit` simply
lift filterax components into the right shapes, so conformance is checked by
pipekit's `isinstance` against its runtime-checkable Protocols, with no
dependency in either direction.

Two impedance mismatches are handled at the boundary. pipekit's
`EnsembleDACycle` represents the ensemble as a Python list of member states,
while filterax stacks the ensemble along axis 0 of a single array —
`FilterAnalysisStep` converts between the two and returns the same container
kind it received. And pipekit never passes a PRNG key to the analysis step,
so stochastic filters (e.g. `StochasticEnKF`) draw from the key they were
constructed with; prefer deterministic filters (ETKF, EnSRF) when cycling
through pipekit, or reconstruct the filter per cycle.

## Wiring an `EnsembleDACycle`

```python
import jax.numpy as jnp
from pipekit_cycle import DAState, EnsembleDACycle

from filterax.filters import ETKF
from filterax.pipekit import (
    DynamicsForwardModel,
    FilterAnalysisStep,
    LinearizableObsOperator,
)

cycle = EnsembleDACycle(
    forward_model=DynamicsForwardModel(lorenz96_step, dt=0.05),
    obs_op=LinearizableObsOperator(lambda x: x[::2]),  # observe every 2nd var
    analysis_step=FilterAnalysisStep(ETKF()),
    obs_source=obs_source,  # pipekit Operator: cycle index -> observation
    n_steps=100,
    n_members=20,
)

members = [x0 + 0.1 * eps for eps in perturbations]  # pipekit's list carrier
state = DAState(obs_err_cov=0.5 * jnp.eye(20))       # R rides on the DA state
members, state = cycle(members, state)
```

The observation-error covariance travels on the pipekit `DAState`
(`obs_err_cov`) and may be a dense `(N_y, N_y)` array or a lineax operator;
`FilterAnalysisStep` coerces it to the tagged operator filterax expects.

## Adapters

::: filterax.pipekit
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [FilterAnalysisStep, DynamicsForwardModel, LinearizableObsOperator]

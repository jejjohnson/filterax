# The Assimilation Cycle

Chapters 4–8 derived analysis steps in isolation: one forecast
ensemble in, one posterior ensemble out. Real assimilation is a
*cycle* — forecast, analyse, inflate, repeat — and this chapter
covers how filterax layers that cycle and how it plugs into external
orchestration through pipekit-cycle.

## Two layers

filterax separates the mathematics from the loop:

**L1 — the analysis step.** A stateless `AbstractSequentialFilter`
(`filters.ETKF()`, `filters.EnSRF()`, `filters.LETKF(radius=...)`, …)
whose single method

$$
\texttt{analysis}(X^f,\, y,\, \mathcal{H},\, R)
\;\longmapsto\; \texttt{AnalysisResult}
$$

maps a forecast ensemble to a posterior ensemble. `AnalysisResult`
carries the posterior `particles` $(N_e, N_x)$, the scalar
`log_likelihood` $\log p(y \mid \text{forecast})$ (populated by every
deterministic filter), and an optional `diagnostics` dict. L1 owns
no time, no dynamics, no state.

**L2 — the forecast–analyse–inflate loop.** The high-level models
(`filterax.ETKF`, `EnSRF`, `LETKF`, `StochasticEnKF`, and the latent
wrappers of chapter 15) compose three configuration objects —
`dynamics`, `obs_op`, optional `inflator` — and run, for each
observation window $(y_t, t)$:

$$
X^f_t = \mathcal{M}_{t-1 \to t}\big(X^a_{t-1}\big)
\;\longrightarrow\;
X^a_t = \text{analysis}\big(X^f_t, y_t\big)
\;\longrightarrow\;
X^a_t \leftarrow \text{inflate}\big(X^a_t, X^f_t\big),
$$

with the dynamics `vmap`-ed over members and the inflator receiving
both ensembles so RTPS/RTPP can relax toward the prior (chapter 12).
The result is an `AssimilationResult`: terminal `particles`, plus
`forecast_history` and `analysis_history` stacked $(T, N_e, N_x)$
along a leading time axis, plus length-$T$ `log_likelihoods`
(NaN-padded where a filter produces none, so the time axes stay
aligned).

The L2 loop is a deliberately thin Python `for` — easy to read, easy
to replace. When you need a fused `lax.scan` (long $T$, gradient
checkpointing), `differentiable_assimilate` is the drop-in
(chapter 14); when you need external orchestration, the adapters
below are.

## Cycling through pipekit-cycle

[pipekit-cycle](https://github.com/jejjohnson/pipekit) orchestrates
DA cycles for *any* algorithm library through three
runtime-checkable Protocols:

| Protocol | Contract |
|---|---|
| `AnalysisStep` | `__call__(forecast, obs, *, obs_op, obs_err_cov)` |
| `ForwardModel` | `step(state, dt)` + `dt` / `state_signature` attributes |
| `ObservationOperator` | `__call__(state)` + `linearize(state)` |

Conformance is **structural** — filterax never imports pipekit, and
pipekit never imports filterax. The adapters in `filterax.pipekit`
lift filterax components into the protocol shapes, and pipekit's
`isinstance` checks against its runtime-checkable Protocols pass:

- **`FilterAnalysisStep`** wraps any `AbstractSequentialFilter` as an
  `AnalysisStep`. It absorbs the main impedance mismatch: pipekit's
  `EnsembleDACycle` represents the ensemble as a Python **list of
  member states**, while filterax stacks members along axis 0 of a
  single array. The adapter stacks a list on the way in and returns
  the same container kind it received — list in, list out; array in,
  array out. It also coerces a dense `(N_y, N_y)` `obs_err_cov`
  array into a tagged lineax operator (an explicit error is raised
  when it is missing). Extra per-cycle analysis kwargs (LETKF's
  `state_coords` / `obs_coords`) ride along via `analysis_kwargs`.
- **`DynamicsForwardModel`** wraps an `AbstractDynamics` as a
  `ForwardModel`. pipekit does not thread absolute time into `step`,
  so the dynamics are treated as autonomous and integrated over
  $[0, dt]$.
- **`LinearizableObsOperator`** wraps an `AbstractObsOperator` (or
  plain callable) as an `ObservationOperator`, deriving `linearize`
  from `jax.jacfwd` — any JAX-traceable $\mathcal{H}$ gets an exact
  $(N_y, N_x)$ Jacobian for free.

**Randomness note.** pipekit never passes a PRNG key to the analysis
step, so a stochastic filter wrapped in `FilterAnalysisStep` draws
from the key it was constructed with — the *same* perturbations every
cycle. Prefer deterministic filters (ETKF, EnSRF) when cycling
through pipekit, or reconstruct the filter per cycle.

## Implementation in filterax

The adapter layer is pure filterax and runs without pipekit
installed:

```python
import jax.numpy as jnp

from filterax.filters import ETKF
from filterax.pipekit import (
    DynamicsForwardModel,
    FilterAnalysisStep,
    LinearizableObsOperator,
)

# Analysis step: list-of-members in, list-of-members out (pipekit's
# EnsembleDACycle container), or stacked array in / array out.
step = FilterAnalysisStep(ETKF())
members = [jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]), jnp.array([2.0, 0.5])]
analysed = step(
    members,
    jnp.array([0.5, 0.5]),
    obs_op=lambda x: x,
    obs_err_cov=0.1 * jnp.eye(2),       # dense array — coerced to lineax
)
print("container preserved:", type(analysed).__name__, len(analysed))

# Forward model: autonomous step(state, dt) + dt attribute.
fwd = DynamicsForwardModel(lambda x, t0, t1: x + 0.1 * (t1 - t0), dt=0.5)
print("step:", fwd.step(jnp.array([1.0, 2.0]), fwd.dt))

# Observation operator: __call__ + linearize via jax.jacfwd.
H = LinearizableObsOperator(lambda x: x[:1] ** 2)
print("H(x):", H(jnp.array([3.0, 4.0])), " dH/dx:", H.linearize(jnp.array([3.0, 4.0])))
```

```
container preserved: list 3
step: [1.05 2.05]
H(x): [9.]  dH/dx: [[6. 0.]]
```

### Wiring an `EnsembleDACycle`

The full cycle wiring requires `pipekit_cycle` to be installed, so
the block below is **illustrative only** (not executed here; the
filterax-side adapter behaviour it relies on is exactly what ran
above, and `tests/test_pipekit_integration.py` runs the real
`isinstance` conformance checks whenever `pipekit_cycle` is
importable):

```python
# Illustrative — requires `pipekit_cycle` to be installed.
import jax.numpy as jnp
from pipekit_cycle import DAState, EnsembleDACycle

from filterax.filters import ETKF
from filterax.pipekit import (
    DynamicsForwardModel,
    FilterAnalysisStep,
    LinearizableObsOperator,
)

cycle = EnsembleDACycle(
    forward=DynamicsForwardModel(my_dynamics, dt=0.5),
    obs_op=LinearizableObsOperator(lambda x: x[::4]),
    analysis=FilterAnalysisStep(ETKF()),
)
state = DAState(
    ensemble=[x0 + perturbation(j) for j in range(N_e)],   # list of members
    obs_err_cov=0.1 * jnp.eye(N_y),
)
for y_t in observation_stream:
    state = cycle.run(state, obs=y_t)
```

pipekit owns the loop, the scheduling, and the state container;
filterax owns the ensemble mathematics inside `FilterAnalysisStep`.
Swapping `ETKF()` for `EnSRF()`, `LETKF(radius=...)` (with
`analysis_kwargs={"state_coords": ..., "obs_coords": ...}`), or an
analysis step from an entirely different library changes nothing
else in the pipeline — that is the point of the protocols.

## Choosing your loop

| Need | Use |
|---|---|
| Quick experiment, short $T$ | L2 `model.assimilate(...)` Python loop |
| Training through the filter, long $T$ | `differentiable_assimilate` (scan + checkpoint) |
| $T$-independent gradient memory | `road_enkf_loss_and_grad` |
| External orchestration / mixed libraries | pipekit-cycle + `filterax.pipekit` adapters |
| Custom cycling logic | roll your own around L1 `filter.analysis(...)` |

All five paths call the same L1 analysis code, so results are
consistent across them — the differentiable loop is pinned
step-for-step to the L2 loop in `tests/test_differentiable.py`, and
the adapter to the direct analysis call in
`tests/test_pipekit_integration.py`.

## Where next

- [Chapter 1 — Problem setting](01_problem_setting.md): the
  filtering recursion this cycle implements.
- [Chapter 12 — Inflation](12_inflation.md): the inflate leg of the
  L2 loop.
- [Chapter 14 — Differentiable assimilation](14_differentiable.md):
  the scan-based replacement for the Python loop.
- [Chapter 15 — Latent-space ensemble DA](15_latent_da.md): the L2
  latent wrappers that reuse this exact loop in $z$-space.
- [API: pipekit integration](../api/pipekit.md) — adapter reference
  and a full `EnsembleDACycle` example.
- [API: Protocols & Types](../api/protocols.md) — `AnalysisResult`,
  `AssimilationResult`, and the abstract component classes.

## References

- Carrassi, A., Bocquet, M., Bertino, L., & Evensen, G. (2018).
  *Data assimilation in the geosciences: An overview of methods,
  issues, and perspectives.* WIREs Climate Change, 9(5), e535.
- Asch, M., Bocquet, M., & Nodet, M. (2016). *Data Assimilation:
  Methods, Algorithms, and Applications.* SIAM. (Ch. 1 frames the
  forecast–analysis cycle.)
- Evensen, G. (2009). *Data Assimilation: The Ensemble Kalman
  Filter.* 2nd ed., Springer.

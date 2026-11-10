---
status: draft
version: 0.1.0
---

# filterax × plumax — Tier IV multi-instrument methane attribution

**Subject:** Worked API sketch for fusing TROPOMI + EMIT + GHGSat observations into a joint posterior over methane source rates, positions, wind, and instrument biases.

**Decision anchors:** [D8 (smoothers in core)](../decisions.md#d8-smoothers-are-in-scope-core-library), [D12 (multi-instrument)](../decisions.md#d12-multi-instrument-fusion--both-joint-and-sequential-surfaces), [D13 (carrier adapter)](../decisions.md#d13-coordinate-aware-carriers-via-adapter-not-core-type), [D14 (GeoLocalizer)](../decisions.md#d14-geospatial-localization-owned-by-filterax), [D15 (state persistence)](../decisions.md#d15-filter-state-is-serializable)

> This document is **API-shape only**. It does not import plumax; plumax does not ship yet. The point is to validate that filterax surfaces compose into the target use case without further design changes.

---

## 1  The Tier IV problem

The plumax stack fuses three satellite observation streams into a joint posterior over a methane attribution state vector:

| Symbol | Meaning | Dimensionality |
|--------|---------|----------------|
| `Q(t)` | Source rates over `K` candidate locations | `K × T` |
| `x₀` | Source positions | `K × 2` |
| `ū`, `θ_wind` | Wind speed and direction | `~10` |
| `c_bg` | Background concentration | `~1` |
| `b_inst` | Per-instrument bias offset (TROPOMI, EMIT, GHGSat) | `3` |
| `A_surf`, `AOD` | Surface albedo, aerosol optical depth | `~10` |

State dimensionality: ~20 (Tier I Gaussian plume) to ~10⁴ (Tier III Eulerian transport on a regional grid). Ensemble size 50–1000.

Observations arrive **at different times** (TROPOMI 24 h revisit, EMIT ISS cadence, GHGSat tasked) **at different native resolutions** (TROPOMI 7×7 km, EMIT 60 m, GHGSat 25 m) with **different noise models** (retrieval covariance + representation error + temporal-alignment error per instrument).

## 2  Mapping plumax components to filterax surfaces

| plumax component | filterax surface | Notes |
|------------------|------------------|-------|
| Tier II/III transport model | `AbstractDynamics` | Stepped via `eqx.filter_vmap` over ensemble |
| RTM + averaging kernel application | `AbstractObsOperator` (per instrument) | One per satellite |
| Per-instrument noise (R_retr + R_repr + R_align) | `AbstractNoise` (per instrument) → gaussx `LowRankUpdate + Diagonal` | Block-diagonal for joint analysis |
| TROPOMI ⊕ EMIT ⊕ GHGSat fusion | `JointObsOperator(ops, noises)` *or* `SequentialAssimilation(filter)` | D12 |
| Geographic Gaspari–Cohn over plume footprint | `GeoLocalizer(coords, frame, radius_km=50)` | D14 |
| Continental / basin domain decomposition | `LocalEnKF` (patcher-LETKF) | D16 |
| Coordinate-aware ensembles | `CarrierAdapter` for `coordax.Array` | D13 |
| Multi-day event reconstruction | `FixedLagSmoother(lag=5)` or `EnsembleRTS` | Wave 5 |
| Operational alert warm-start | `save_state` / `load_state` | D15 |

## 3  Joint-analysis recipe (simultaneous overpass)

When two instruments observe overlapping pixels within the same analysis window (e.g., a coincident TROPOMI + EMIT overpass) a single joint analysis is correct:

```python
import filterax
from plumax.obs import TROPOMIOp, EMITOp, GHGSatOp        # user / plumax-side
from plumax.noise import TROPOMINoise, EMITNoise, GHGSatNoise

joint_op = filterax.JointObsOperator(
    ops=(TROPOMIOp(ak_path=...), EMITOp(...), GHGSatOp(...)),
    noise_covs=(TROPOMINoise(...), EMITNoise(...), GHGSatNoise(...)),
)

filter_ = filterax.LETKF(
    localizer=filterax.GeoLocalizer(
        coords=state_lonlat,
        frame=filterax.LocalFrame.from_crs("EPSG:4326", origin=(lon0, lat0)),
        radius_km=50.0,
    ),
    inflator=filterax.RTPS(alpha=0.8),
    config=filterax.FilterConfig(n_ensemble=200),
)

result = filter_.analysis(
    particles=forecast_ensemble,
    obs=joint_op.stack_observations(y_trop, y_emit, y_ghgsat),
    obs_op=joint_op,
    obs_noise=joint_op.noise(),       # block-diagonal
)
```

`JointObsOperator.stack_observations` concatenates the three observation vectors; `joint_op.noise()` returns a block-diagonal `gaussx.BlockDiagonalLinearOperator` over the three per-instrument noise operators.

## 4  Sequential-analysis recipe (temporally separated overpasses)

When overpasses are an hour apart and there is no benefit to joint treatment:

```python
import jax.numpy as jnp

overpasses = [
    (y_trop_t0,   TROPOMIOp(...), TROPOMINoise(...)),
    (y_emit_t1,   EMITOp(...),    EMITNoise(...)),
    (y_ghgsat_t2, GHGSatOp(...),  GHGSatNoise(...)),
]

cycle = filterax.SequentialAssimilation(filter_)
state = filterax.FilterState(particles=forecast_ensemble, step=jnp.array(0))
final_state, results = cycle(state, overpasses)
```

`SequentialAssimilation.__call__(state, overpasses)` consumes the full list of `(obs, obs_op, obs_noise)` tuples in one call (see `features/multi_instrument.md` §2.2) and returns the final `FilterState` plus the trail of `AnalysisResult` for diagnostics. The user dynamics are still responsible for propagating the ensemble between overpasses — see §5 for the inter-overpass case.

## 5  Multi-day event reconstruction (fixed-lag smoother)

A plume event spans 12–24 hours and 3–5 overpasses with non-trivial dynamics in between. Operational triage uses the filter trail; the post-hoc attribution report uses a fixed-lag smoother to incorporate future observations. When dynamics matter between overpasses, run one analysis per step (rather than batching them through `SequentialAssimilation`):

```python
filter_results = []
forecast_history = []
state = filterax.FilterState(particles=init, step=jnp.array(0))
t_prev = 0.0
for t, (obs, op, noise) in zip(overpass_times, overpasses):
    forecast = dynamics_step(state.particles, t_prev, t)
    forecast_history.append(forecast)
    result = filter_.analysis(forecast, obs, op, noise)
    state = filterax.FilterState(particles=result.particles, step=state.step + 1)
    filter_results.append(result)
    t_prev = t

smoother = filterax.FixedLagSmoother(lag=5)
smoothed = smoother.smooth(filter_results, forecast_history)
```

## 6  Operational alert warm-start

Cold-start budget for the alert service is ≤5 s. The warm ensemble is loaded from disk per D15:

```python
state = filterax.load_state("/var/run/plumax/last_state.fax", like=template_state)
forecast = dynamics_step(state.particles, state.step, state.step + 1)
# ... assimilate latest overpass ...
filterax.save_state("/var/run/plumax/last_state.fax", new_state)
```

The first analysis after warm-start reuses the JIT-compiled graph from the previous run; the only cost is the disk read plus the analysis itself.

## 7  Differentiable RTM (Tier IV v2+)

Tier IV v2 replaces the look-up-table RTM with a neural surrogate trained against L1 radiance. The training signal is the observation-space log-likelihood through the full filter:

```python
@eqx.filter_jit
@eqx.filter_value_and_grad
def loss(rtm_params, ensemble, observations):
    obs_op = NeuralRTMObsOp(rtm_params)
    state = filterax.FilterState(ensemble, 0)
    nll = 0.0
    for obs in observations:
        result = filter_.analysis(state.particles, obs, obs_op, noise)
        nll = nll - result.log_likelihood
        state = filterax.FilterState(result.particles, state.step + 1)
    return nll

value, grad = loss(rtm_params, ensemble, observations)
```

filterax's only contribution is staying differentiable. The neural network and training loop are user-owned.

## 8  Open seams

These are concrete things plumax will pressure-test once both libraries mature:

1. **Block-diagonal noise representation.** `JointObsOperator.noise()` relies on gaussx's `BlockDiagonalLinearOperator`. Confirm gaussx ships that surface before Wave 2 closes.
2. **Partial obs masking.** Real overpasses have cloud-masked pixels. `MaskedObsOperator` (Wave 2) wraps a per-instrument op with a NaN-aware reduction. Plumax should validate the masking semantics on real TROPOMI L2.
3. **`coordax.Array` adapter.** filterax ships only the interface plus a `dict[str, Array]` reference adapter. Plumax (or geostack) needs to author the `coordax.Array` adapter once coordax stabilizes.
4. **Patcher API.** filterax ships an in-house patcher in Wave 2 (D16). Migration to `geotoolz.patch` is deferred until geostack's patcher stabilizes.

None of these change the filterax design — they're all downstream / sibling-library deliverables.

## 9  Acceptance for filterax

filterax considers the plumax integration complete when:

- Joint and sequential analyses on a synthetic two-instrument linear Gaussian problem produce identical posteriors (when noise is block-diagonal and obs are independent).
- `GeoLocalizer` produces a Gaspari–Cohn taper that matches a reference NumPy implementation to 1e-6 on a 100-point lon/lat grid.
- A warm-start cycle (load_state → analysis → save_state) preserves the ensemble bit-for-bit.
- The differentiable Tier IV v2 loss above produces finite gradients on a synthetic problem.

These land as smoke + correctness tests in Wave 5 (see `issues/wave-5-*.md`, task `FLX-55B`).

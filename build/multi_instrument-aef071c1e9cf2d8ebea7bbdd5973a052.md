---
status: draft
version: 0.1.0
---

# filterax — Multi-instrument observation fusion

**Subject:** API design for assimilating observations from multiple instruments with heterogeneous noise, resolution, and overpass times.

**Decision anchor:** [D12](../decisions.md#d12-multi-instrument-fusion--both-joint-and-sequential-surfaces)

**Lands in:** Wave 2 (`issues/wave-2-*.md`, task `FLX-24A`)

---

## 1  Scope

Real-world data assimilation typically fuses observations from multiple sources — different satellites, different instruments on the same satellite, in-situ sensors mixed with remote sensing. These streams differ in:

- **Resolution** (TROPOMI 7×7 km vs GHGSat 25 m)
- **Noise characteristics** (retrieval covariance + representation error + temporal misalignment, all instrument-specific)
- **Timing** (24 h revisit vs tasked-on-demand)
- **State pieces observed** (column XCH₄ vs surface flux vs radiance)

filterax does not bake multi-instrument fusion into its protocols. Instead, it provides two composition surfaces — **joint** (one analysis call) and **sequential** (one analysis call per overpass) — and lets the user choose based on the physics.

**In scope:** `JointObsOperator`, `MaskedObsOperator`, `SequentialAssimilation`.

**Out of scope:** instrument-specific obs operators (user / domain library code), forward modelling, averaging-kernel application (user / domain library code), 4D-Var-style joint cost over time windows (vardax).

---

## 2  Two surfaces, two regimes

### 2.1  Joint analysis — `JointObsOperator`

When multiple instruments observe the same analysis window and noise can be modelled as block-diagonal across instruments, a single joint analysis is statistically correct and computationally efficient (one Kalman gain, one update).

```python
class JointObsOperator(AbstractObsOperator):
    """Concatenate multiple single-instrument observation operators."""

    ops: tuple[AbstractObsOperator, ...]
    noise_covs: tuple[AbstractLinearOperator, ...]

    def __call__(self, state: Float[Array, "N_x"]) -> Float[Array, "N_y"]:
        """Stack per-instrument predictions into one observation vector."""
        return jnp.concatenate([op(state) for op in self.ops])

    def noise(self) -> AbstractLinearOperator:
        """Return block-diagonal noise covariance (via gaussx)."""
        return gaussx.BlockDiagonalLinearOperator(self.noise_covs)

    def stack_observations(self, *ys: Float[Array, "..."]) -> Float[Array, "N_y"]:
        """Helper: stack per-instrument observation vectors in the same order as ops."""
        return jnp.concatenate(ys)
```

**Use when:**

- Overpasses are simultaneous (or within the analysis-step time tolerance).
- Per-instrument noise is independent (no cross-instrument correlation), so block-diagonal R is correct.
- The state pieces observed overlap, so a joint update captures cross-instrument constraint (e.g., TROPOMI XCH₄ + EMIT XCH₄ both constrain the same column).

**Cost:** one Kalman gain solve on a `(N_y, N_y)` system where `N_y = Σ N_y_i`. The block-diagonal structure of R is exploited by gaussx.

### 2.2  Sequential analysis — `SequentialAssimilation`

When overpasses are temporally separated (an hour apart, a day apart) the joint approach is wrong: there should be a dynamics step between observations. Sequential analysis treats each overpass as its own filter step.

```python
class SequentialAssimilation(eqx.Module):
    """Layer 2 helper: loop a filter's analysis over a list of overpasses."""

    filter_: AbstractSequentialFilter

    def __call__(
        self,
        state: FilterState,
        overpasses: list[tuple[Array, AbstractObsOperator, AbstractLinearOperator]],
    ) -> tuple[FilterState, list[AnalysisResult]]:
        results = []
        for obs, op, noise in overpasses:
            result = self.filter_.analysis(state.particles, obs, op, noise)
            state = FilterState(particles=result.particles, step=state.step + 1)
            results.append(result)
        return state, results
```

**Use when:**

- Overpasses are time-staggered and there is meaningful dynamics evolution between them.
- Per-instrument observation operators do not share state pieces (independence makes sequential equivalent to joint up to the dynamics step).
- Memory or compute pressure makes one big `(N_y, N_y)` solve unattractive.

**Cost:** `K` analyses on smaller `(N_y_i, N_y_i)` systems. Often cheaper than the joint solve when `N_y_i ≪ Σ N_y_i`.

### 2.3  Equivalence

When (a) noise is block-diagonal, (b) per-instrument obs operators are independent linear functions of disjoint state pieces, and (c) no dynamics step happens between overpasses — joint and sequential produce **identical posteriors**. Wave 2 ships an equivalence test on this case.

When the obs operators share state pieces or noise has cross-instrument correlation, joint is correct and sequential is approximate.

---

## 3  Partial observations — `MaskedObsOperator`

Real observations have missing pixels (cloud, snow, AOD rejection, retrieval QA failures). Filtering out masked pixels at construction time loses the fixed JIT shape; doing it inside the analysis path breaks `jax.jit`.

filterax's resolution is a `MaskedObsOperator` wrapper that:

1. Computes the full instrument prediction `op(state)` (fixed shape).
2. Multiplies the observation residual by a static valid-pixel mask.
3. Uses gaussx's noise operator to inflate the noise on masked pixels to infinity (or a large finite value).

```python
class MaskedObsOperator(AbstractObsOperator):
    """Wrap an obs operator with a static valid-pixel mask."""

    op: AbstractObsOperator
    mask: Bool[Array, "N_y"] = eqx.field(static=False)   # not static — varies per overpass

    def __call__(self, state):
        return self.op(state)

    def masked_noise(self, base_noise: AbstractLinearOperator) -> AbstractLinearOperator:
        """Inflate noise on masked pixels."""
        # Add a large multiple of (I - diag(mask)) to base_noise
        ...
```

The mask is **not static** for `jax.jit` — it varies overpass to overpass — but the shape of the observation vector is. This pattern works under `jit` as long as `mask.shape == op(state).shape`.

---

## 4  Picking between joint and sequential

| Scenario | Recommendation |
|----------|----------------|
| Two coincident overpasses, independent noise | Joint |
| Two coincident overpasses, correlated noise across instruments | Joint with full (non-block-diag) noise; gaussx supports this |
| Three overpasses spaced 1 h apart, dynamics non-trivial | Sequential with dynamics between |
| Three overpasses spaced 1 h apart, dynamics negligible | Either; joint is one solve, sequential is three smaller solves |
| Joint, but one instrument is unmasked (partial pixels) | Joint with `MaskedObsOperator` per instrument |
| Very large `N_y` (many pixels), small ensemble | Sequential — avoids one big `(N_y, N_y)` solve |

The library does not auto-choose. The user decides; both surfaces are equally first-class.

---

## 5  Testing strategy

Wave 2 ships:

- **Joint shape test.** `JointObsOperator(ops, noises)(state)` shape matches `concat([op(state) for op in ops])`.
- **Block-diagonal noise test.** `JointObsOperator.noise()` matches the dense block-diagonal reference.
- **Equivalence test.** On a two-instrument linear Gaussian problem with block-diagonal noise and independent obs operators, joint and sequential produce posteriors that agree to numerical precision.
- **Masking test.** A masked observation produces identical posterior to running the unmasked analysis on the surviving pixels alone.

Wave 5 adds the multi-day, multi-instrument plumax smoke test (`FLX-55B`).

---

## 6  What we do not ship

- **Cross-instrument noise correlation.** Block-diagonal noise covers the common case. For genuine cross-instrument correlation, users construct the full noise operator via gaussx directly and pass it to `filter_.analysis` — `JointObsOperator.noise()` is a convenience, not a constraint.
- **Temporal-alignment error decomposition.** That belongs to the per-instrument noise model in user / plumax code.
- **Auto-selection between joint and sequential.** Domain-specific; not filterax's call to make.

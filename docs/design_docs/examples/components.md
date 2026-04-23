---
status: draft
version: 0.1.0
---

# Layer 1 — Component Examples

Usage patterns for building filters, processes, and custom protocol implementations.

---

## Building an ETKF Analysis Step

### Defining dynamics and observation operator, running a single analysis

```python
from __future__ import annotations
import jax.numpy as jnp
from jaxtyping import Float, Array
import equinox as eqx
import somax
import ekalmx

# Define observation operator
class LinearObs(ekalmx.AbstractObsOperator):
    H: Float[Array, "N_y N_x"]

    def __call__(self, state):
        return self.H @ state

# Define dynamics (wrapping a somax model)
class SWMDynamics(ekalmx.AbstractDynamics):
    model: somax.ShallowWaterModel
    dt: float = eqx.field(static=True)

    def __call__(self, state, t0, t1):
        sol = self.model.integrate(state, t0=t0, t1=t1, dt=self.dt)
        return sol.ys[-1]

# Instantiate dynamics and observation operator
dynamics = SWMDynamics(model=somax.ShallowWaterModel(grid=grid, params=params), dt=60.0)
obs_op = LinearObs(H=H_matrix)

# Use the ETKF filter component directly
etkf = ekalmx.filters.ETKF()

# Forecast: vmap dynamics over ensemble members
forecast = eqx.filter_vmap(dynamics)(particles, t0, t1)

# Analysis: single step
result = etkf.analysis(forecast, obs, obs_op, obs_noise=R)
# result.particles: (N_e, N_x) — updated ensemble
```

---

## Building an EKI Process Step

### Initializing and iterating an Ensemble Kalman Inversion process

```python
import ekalmx

# Define forward model (black-box simulator)
def simulator(params):
    """Expensive forward model: params → predictions."""
    return run_climate_model(params)

# Initialize EKI process
eki = ekalmx.processes.EKI(
    scheduler=ekalmx.FixedScheduler(dt=1.0),
)

state = eki.init(
    particles=init_ensemble,   # (J, N_p) — 100 particles, 50 params
    obs=observations,          # (N_d,)
    noise_cov=R,               # lineax operator
)

# Iterate
for step in range(n_iterations):
    # Caller evaluates forward model (possibly expensive, non-JAX)
    G_ensemble = eqx.filter_vmap(simulator)(state.particles)  # (J, N_d)
    state = eki.update(state, forward_evals=G_ensemble)

# Extract calibrated parameters
best_params = ekalmx.ensemble_mean(state.particles)
```

---

## Custom Localizer

### Adaptive distance-based localization with per-variable radius

```python
import ekalmx

class AdaptiveLocalizer(ekalmx.AbstractLocalizer):
    """Distance-based localization with per-variable radius."""
    radii: Float[Array, "N_x"]
    coords: Float[Array, "N_x 2"]  # (lat, lon) per state element

    def __call__(self, cov, coords):
        distances = self._compute_distances(coords)
        # Per-variable Gaspari-Cohn with different radii
        taper = eqx.filter_vmap(ekalmx.gaspari_cohn, in_axes=(0, 0))(
            distances, self.radii
        )
        return ekalmx.localize(cov, taper)
```

---

## Custom Inflator

### Anderson (2007) adaptive multiplicative inflation

```python
import ekalmx

class AdaptiveInflator(ekalmx.AbstractInflator):
    """Anderson (2007) adaptive multiplicative inflation."""
    min_factor: float = eqx.field(static=True, default=1.0)
    max_factor: float = eqx.field(static=True, default=1.2)

    def __call__(self, particles, obs=None, obs_op=None, obs_noise=None):
        # Estimate optimal inflation from innovation statistics
        factor = self._estimate_factor(particles, obs, obs_op, obs_noise)
        factor = jnp.clip(factor, self.min_factor, self.max_factor)
        return ekalmx.inflate_multiplicative(particles, factor)
```

---

## Ensemble Smoother (EnKS)

### Forward filter followed by backward smoother pass

```python
import ekalmx

# Run forward filter first
filter = ekalmx.filters.ETKF()
filter_results = []

state = ekalmx.FilterState(particles=ensemble, step=0)
for t, obs in enumerate(observations):
    forecast = eqx.filter_vmap(dynamics)(state.particles, t, t + dt)
    result = filter.analysis(forecast, obs, obs_op, R)
    filter_results.append(result)
    state = ekalmx.FilterState(particles=result.particles, step=t + 1)

# Backward smoother pass
smoother = ekalmx.smoothers.EnKS()
smoothed = smoother.smooth(filter_results)
# smoothed: list of AnalysisResult with improved state estimates
```

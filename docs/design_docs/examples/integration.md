---
status: draft
version: 0.1.0
---

# Layer 3 — Integration Examples

Cross-library patterns showing how ekalmX composes with the broader ecosystem.

---

## With optax (EKP as optimizer)

### EKI as a drop-in optax optimizer

```python
import ekalmx
import optax

# EKI as drop-in optimizer — parallel to optax_bayes
optimizer = ekalmx.optax.eki(
    forward_fn=simulator,
    obs=observations,
    noise_cov=R,
    scheduler=ekalmx.DataMisfitController(),
    n_ensemble=100,
)

# Compose with optax ecosystem
optimizer = optax.chain(
    optimizer,
    optax.clip_by_global_norm(1.0),
)

# Standard optax loop
state = optimizer.init(params)
for step in range(n_steps):
    updates, state = optimizer.update(grads=None, state=state, params=params)
    params = optax.apply_updates(params, updates)
```

---

## With JAX/Equinox (Differentiable training)

### Learning neural dynamics by backpropagating through the filter

```python
import jax
import optax
import ekalmx
import equinox as eqx

# Neural ODE dynamics
class NeuralDynamics(ekalmx.AbstractDynamics):
    net: eqx.nn.MLP

    def __call__(self, state, t0, t1):
        # Simple Euler step with learned RHS
        dt = t1 - t0
        return state + dt * self.net(state)

dynamics = NeuralDynamics(net=eqx.nn.MLP(in_size=40, out_size=40, ...))

# Loss: negative log-likelihood from filter
def loss_fn(dynamics, ensemble, obs_sequence, obs_op, R):
    filter = ekalmx.filters.ETKF()
    particles = ensemble
    total_ll = 0.0

    for obs_val, t0, t1 in obs_sequence:
        # Forecast — differentiable through neural ODE
        particles = eqx.filter_vmap(dynamics)(particles, t0, t1)
        # Analysis — differentiable through Kalman update
        result = filter.analysis(particles, obs_val, obs_op, R)
        particles = result.particles
        total_ll = total_ll + result.log_likelihood

    return -total_ll

# Train with standard JAX/optax loop
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(dynamics, eqx.is_array))

@eqx.filter_jit
def train_step(dynamics, opt_state, ensemble, obs_sequence, obs_op, R):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(
        dynamics, ensemble, obs_sequence, obs_op, R
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    dynamics = eqx.apply_updates(dynamics, updates)
    return dynamics, opt_state, loss
```

---

## With somax (Ocean state estimation)

### Wrapping a somax shallow water model as ekalmX dynamics

```python
import somax
import ekalmx

# somax forward model
grid = somax.ArakawaCGrid2D(nx=128, ny=128, dx=10e3, dy=10e3)
swm = somax.ShallowWaterModel(grid=grid, params=somax.SWMParams())

# Wrap as ekalmX dynamics
class OceanDynamics(ekalmx.AbstractDynamics):
    model: somax.ShallowWaterModel
    dt: float = eqx.field(static=True, default=300.0)

    def __call__(self, state, t0, t1):
        return self.model.integrate(state, t0=t0, t1=t1, dt=self.dt).ys[-1]

# Sparse observation operator (satellite tracks)
class SatelliteObs(ekalmx.AbstractObsOperator):
    obs_indices: Int[Array, "N_y"]

    def __call__(self, state):
        # Extract SSH at satellite footprint locations
        return state[self.obs_indices]

# Run LETKF
letkf = ekalmx.LETKF(
    dynamics=OceanDynamics(model=swm),
    obs_op=SatelliteObs(obs_indices=sat_indices),
    localizer=ekalmx.GaspariCohn(radius=300e3),
    inflator=ekalmx.RTPS(alpha=0.85),
    config=ekalmx.FilterConfig(n_ensemble=40),
)
result = letkf.assimilate(init_ensemble, observations, t0=0.0, dt=3600.0)
```

---

## With gaussx (Structured noise)

### Diagonal and low-rank noise models via gaussx operators

```python
import lineax as lx
import gaussx
import ekalmx

# Diagonal observation noise
R_diag = lx.DiagonalLinearOperator(jnp.full(n_obs, 0.01))

# Structured process noise (low-rank + diagonal)
Q = gaussx.operators.low_rank_plus_diag(
    W=basis_vectors,      # (N_x, r) — r dominant modes
    d=jnp.full(n_x, 1e-4),  # small isotropic baseline
)

# Both plug directly into ekalmX
result = filter.analysis(particles, obs, obs_op, obs_noise=R_diag)
```

---

## With geo_toolz (Pre/post-processing pipeline)

### Preprocessing satellite data and evaluating filter output

```python
import geo_toolz as gt
import ekalmx

# --- Preprocessing ---
raw_obs = xr.open_dataset("satellite_ssh.nc")

pipeline = gt.Sequential([
    gt.validation.harmonize_coords(),
    gt.subset.bounding_box(lon=(-80, 0), lat=(20, 60)),
    gt.regrid.linear(target_grid=model_grid),
    gt.detrend.remove_climatology(climatology=ssh_clim),
])
obs_dataset = pipeline(raw_obs)

# Convert xarray → JAX arrays for ekalmX
obs_values = jnp.array(obs_dataset["ssh_anomaly"].values)

# --- Run ekalmX ---
result = letkf.assimilate(ensemble, obs_values, ...)

# --- Postprocessing ---
analysis_ds = xr.Dataset({"ssh": (["ens", "y", "x"], result.particles)})

metrics = gt.Sequential([
    gt.metrics.rmse(reference=truth_dataset),
    gt.spectral.psd(dim="x"),
])
evaluation = metrics(analysis_ds)
```

---

## With xr_assimilate (xarray orchestration)

### Using ekalmX as a compute backend for xarray-level DA

```python
import xr_assimilate as xra
import ekalmx

# ekalmX provides array-level compute
filter_backend = ekalmx.LETKF(
    dynamics=ocean_dynamics,
    obs_op=satellite_obs,
    localizer=ekalmx.GaspariCohn(radius=300e3),
    config=ekalmx.FilterConfig(n_ensemble=40),
)

# xr_assimilate handles xarray bookkeeping
assimilator = xra.Assimilator(
    backend=filter_backend,
    observations=obs_dataset,
    background=background_dataset,
    output_dir="./analysis/",
)
result_ds = assimilator.run()  # xr.Dataset with analysis fields
```

---

## Tutorial Index

Full end-to-end tutorials (in `notebooks/`):

| Tutorial | Layer | Topic |
|----------|-------|-------|
| 01_primitives | L0 | Ensemble statistics, Kalman gain, localization from scratch |
| 02_linear_gaussian | L1 | EnKF on a linear Gaussian system — verify against analytic KF |
| 03_lorenz96_etkf | L1-L2 | ETKF on Lorenz-96 with localization and inflation |
| 04_eki_calibration | L1-L2 | Calibrate L96 forcing parameter F via EKI |
| 05_differentiable_enkf | L1 | Learn neural dynamics by backpropagating through EnKF |
| 06_ocean_letkf | L2 | LETKF with somax shallow water model |
| 07_eki_optax | L2 | EKI as optax transform, composed with schedules |
| 08_smoother | L1-L2 | EnKS backward pass, compare filter vs smoother |

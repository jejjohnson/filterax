---
status: draft
version: 0.1.0
---

# Layer 2 — Model Examples

High-level API usage with minimal boilerplate. Configure, call, get results.

---

## LETKF for State Estimation

### Minimal boilerplate ocean state estimation

```python
import ekalmx

# One-shot: configure and run
letkf = ekalmx.LETKF(
    dynamics=SWMDynamics(model=swm),
    obs_op=LinearObs(H=H_matrix),
    localizer=ekalmx.GaspariCohn(radius=500e3),
    inflator=ekalmx.RTPS(alpha=0.9),
    config=ekalmx.FilterConfig(n_ensemble=50),
)

# Assimilate a sequence of observations
result = letkf.assimilate(
    init_ensemble=ensemble,        # (50, N_x)
    observations=obs_sequence,     # list of (obs_values, obs_time)
    t0=0.0,
    dt=3600.0,                     # 1-hour windows
)
# result.particles: final analysis ensemble
# result.history: list of per-window AnalysisResult
```

---

## EKI for Parameter Estimation

### Calibrate a black-box simulator

```python
import ekalmx

# Configure and run
eki = ekalmx.EKI(
    forward_fn=expensive_simulator,
    obs=observations,
    noise_cov=R,
    scheduler=ekalmx.DataMisfitController(),
    config=ekalmx.ProcessConfig(n_iterations=50),
)

result = eki.run(init_particles=init_ensemble)  # (100, N_p)
# result.particles: calibrated parameter ensemble
# result.mean: best estimate
# result.covariance: uncertainty estimate
```

---

## EKS for Posterior Sampling

### Approximate posterior samples via Ensemble Kalman Sampler

```python
import ekalmx

# Same interface, different process — produces posterior samples
eks = ekalmx.EKS(
    forward_fn=simulator,
    obs=observations,
    noise_cov=R,
    scheduler=ekalmx.EKSStableScheduler(),
    config=ekalmx.ProcessConfig(n_iterations=500),
)

result = eks.run(init_particles=init_ensemble)
# result.particles: approximate posterior samples
```

---

## Configuration Patterns

| Pattern | Configuration | Use Case |
|---|---|---|
| State estimation (localized) | `LETKF(localizer=GaspariCohn(...), inflator=RTPS(...))` | Ocean/atmosphere DA |
| State estimation (global) | `ETKF(inflator=RTPS(...))` | Low-dimensional systems |
| Parameter calibration | `EKI(scheduler=DataMisfitController())` | Black-box simulators |
| Posterior sampling | `EKS(scheduler=EKSStableScheduler())` | Uncertainty quantification |
| Unscented inversion | `UKI(alpha=1.0, beta=2.0, kappa=0.0)` | Deterministic, parametric |

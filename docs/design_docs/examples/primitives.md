---
status: draft
version: 0.1.0
---

# Layer 0 — Primitive Examples

Usage patterns for ensemble statistics, Kalman gain, localization, inflation, likelihood, perturbations, and patch decomposition.

---

## Ensemble Statistics

### Computing on demand from particles

```python
import jax.numpy as jnp
import ekalmx

# particles: (N_e, N_x) — ensemble of 50 members, state dim 100
particles = jnp.ones((50, 100))

# Compute on demand — no precomputed state object needed
mean = ekalmx.ensemble_mean(particles)                    # (N_x,)
anomalies = ekalmx.ensemble_anomalies(particles)          # (N_e, N_x)
cov = ekalmx.ensemble_covariance(particles)               # LowRankUpdate operator
cross_cov = ekalmx.cross_covariance(particles, H_particles)  # (N_x, N_y) array
```

---

## Kalman Gain

### gaussx-backed structured solve

```python
import ekalmx

# gaussx-backed: exploits low-rank structure when N_e << N_x
K = ekalmx.kalman_gain(
    particles=forecast_particles,   # (N_e, N_x)
    obs_particles=H_forecast,       # (N_e, N_y)
    obs_noise=R,                    # lineax.DiagonalLinearOperator
)
# K: (N_x, N_y)
```

---

## Localization

### Gaspari-Cohn taper and application

```python
import ekalmx

# Gaspari-Cohn taper
taper = ekalmx.gaspari_cohn(distances, radius=500e3)  # (N_x, N_y)

# Apply to covariance (Hadamard product)
cov_localized = ekalmx.localize(cov, taper)
```

---

## Inflation

### Multiplicative and relaxation strategies

```python
import ekalmx

# Multiplicative inflation
inflated = ekalmx.inflate_multiplicative(particles, factor=1.05)

# Relaxation to prior spread (RTPS)
inflated = ekalmx.inflate_rtps(
    analysis_particles, forecast_particles, alpha=0.9
)
```

---

## Log-Likelihood

### Innovation-based log-likelihood for differentiable training

```python
import ekalmx

# Innovation-based log-likelihood for differentiable training
ll = ekalmx.log_likelihood(
    innovation=obs - H_mean,            # (N_y,)
    innovation_cov=S,                    # gaussx operator (H P H^T + R)
)
# ll: scalar — use with jax.grad for parameter learning
```

---

## Perturbed Observations

### Stochastic perturbations for stochastic EnKF

```python
import jax
import ekalmx

# For stochastic EnKF
key = jax.random.PRNGKey(0)
obs_perturbed = ekalmx.perturbed_observations(
    key, obs, obs_noise=R, n_ensemble=50
)
# obs_perturbed: (N_e, N_y)
```

---

## Patch-Based Decomposition

### Spatial decomposition for large domains

```python
import ekalmx

# Create overlapping patches for large spatial domains
patches, metadata = ekalmx.create_patches(
    particles,             # (N_e, lat, lon)
    patch_size=(64, 64),
    stride=(48, 48),       # overlap = 16
)

# Assign observations to patches (with buffer zone)
patch_obs = ekalmx.assign_obs_to_patches(
    obs_coords, obs_values, metadata, buffer=8
)

# After per-patch analysis, blend back
merged = ekalmx.blend_patches(
    analyzed_patches, metadata, taper_fn=ekalmx.gaspari_cohn
)
```

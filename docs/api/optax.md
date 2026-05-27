# optax integration

EKI, EKS, and UKI are exposed as `optax.GradientTransformation` pairs
so you can mix derivative-free ensemble updates with the rest of the
optax ecosystem (schedules, gradient clipping, hybrid pipelines with
gradient-based optimisers, …).

Three constructors live in `filterax.optax`:

::: filterax.optax.eki
::: filterax.optax.eks
::: filterax.optax.uki

## Contract

The `grads` argument to `update(grads, state, params)` is **ignored** —
the ensemble (or sigma-point quadrature) provides search directions.
The returned `updates` is the change in the *ensemble mean*, unflattened
back into the original `params` PyTree shape so `optax.apply_updates`
works without modification.

The ensemble itself lives in the optax `OptState`; `params` only ever
holds the running mean. Pytree-shaped params (nested dicts,
NamedTuples, …) round-trip via `jax.flatten_util.ravel_pytree`.

## Patterns

### Basic EKI loop

```python
import filterax as flx
import optax

transform = flx.optax.eki(
    forward_fn=simulator,
    obs=observations,
    noise_cov=Gamma,
    scheduler=flx.DataMisfitController(),
)

params = initial_params
state = transform.init(params)
for _ in range(50):
    updates, state = transform.update(None, state, params)
    params = optax.apply_updates(params, updates)
```

### Posterior sampling with EKS

```python
sampler = flx.optax.eks(
    forward_fn=simulator,
    obs=observations,
    noise_cov=Gamma,
    n_ensemble=200,
)
state = sampler.init(params)
samples = []
for step in range(500):
    updates, state = sampler.update(None, state, params)
    params = optax.apply_updates(params, updates)
    if step > burnin:
        samples.append(state.particles)  # (200, Nₚ) per step
```

### Hybrid: gradient warm-start + EKI refinement

```python
# Phase 1 — Adam handles fast initial descent with gradients.
adam = optax.adam(1e-3)
state_a = adam.init(params)
for _ in range(1000):
    grads = jax.grad(loss_fn)(params)
    updates, state_a = adam.update(grads, state_a, params)
    params = optax.apply_updates(params, updates)

# Phase 2 — switch to EKI for gradient-free refinement.
phase2 = flx.optax.eki(
    forward_fn=simulator,
    obs=observations,
    noise_cov=Gamma,
    init_spread=0.1,  # tight cluster around the warm-started mean
)
state_b = phase2.init(params)
for _ in range(50):
    updates, state_b = phase2.update(None, state_b, params)
    params = optax.apply_updates(params, updates)
```

### Composition with other optax transforms

```python
optimizer = optax.chain(
    flx.optax.eki(forward_fn=sim, obs=y, noise_cov=R),
    optax.clip_by_global_norm(1.0),
)
state = optimizer.init(params)
updates, state = optimizer.update(None, state, params)
params = optax.apply_updates(params, updates)
```

## Internal state types

These are returned from the optax transforms' `init` and threaded
through `update`. Inspect them directly for the full ensemble
(EKI / EKS) or parametric belief (UKI) — useful for diagnostics and
posterior sampling.

::: filterax.optax.EKIOptaxState
::: filterax.optax.EKSOptaxState
::: filterax.optax.UKIOptaxState

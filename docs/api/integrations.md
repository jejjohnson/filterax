# Integrations

filterax composes with the rest of the stack through its protocols, and it imports none of these packages except gaussx and optax, which it depends on.
These are minimal snippets, and each one runs as written: `tests/test_integration_snippets.py` executes this page.

## optax

An ensemble process is an `optax.GradientTransformation`, so it chains with the rest of optax.
The `grads` argument is ignored, since the ensemble provides the search directions.
See [optax integration](optax.md) for the contract and more patterns.

```python
import filterax as flx
import jax.numpy as jnp
import lineax as lx
import optax

obs_map = jnp.array([[1.0, 0.5], [-0.3, 1.2], [0.7, 0.1]])

optimizer = optax.chain(
    flx.optax.eki(
        forward_fn=lambda theta: obs_map @ theta,
        obs=jnp.array([0.4, 0.1, 0.3]),
        noise_cov=lx.DiagonalLinearOperator(0.1 * jnp.ones(3)),
        n_ensemble=50,
        scheduler=flx.FixedScheduler(dt=0.5),
    ),
    optax.clip_by_global_norm(1.0),
)

params = jnp.zeros(2)
state = optimizer.init(params)
for _ in range(10):
    updates, state = optimizer.update(None, state, params)
    params = optax.apply_updates(params, updates)
```

## gaussx

Observation noise is a lineax operator, so a structured gaussx operator passes straight through, and its solves keep their structure.
Here the noise is a Kronecker product of a channel covariance and a site covariance.
`LETKF` is the exception, since its localisation assumes a diagonal noise covariance.

```python
import filterax as flx
import gaussx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

particles = jr.normal(jr.key(0), (30, 6))  # (N_e, N_x)
channels = lx.DiagonalLinearOperator(jnp.array([0.5, 1.0]))
sites = lx.MatrixLinearOperator(
    jnp.array([[1.0, 0.2, 0.0], [0.2, 1.0, 0.2], [0.0, 0.2, 1.0]]),
    lx.positive_semidefinite_tag,
)
noise = gaussx.Kronecker(channels, sites)  # (6, 6), never formed

result = flx.filters.ETKF().analysis(particles, jnp.zeros(6), lambda x: x, noise)
```

The ensemble covariance is a gaussx operator too, so gaussx's primitives apply to it without forming a state-sized matrix.

```python
import filterax as flx
import gaussx
import jax.random as jr

particles = jr.normal(jr.key(0), (30, 500))
covariance = flx.ensemble_covariance(particles)  # gaussx.LowRankUpdate
norm = gaussx.frobenius_norm(covariance)
```

## somax

A somax model becomes filterax dynamics through a small wrapper, since `AbstractDynamics` only asks for a map from one state at `t0` to its value at `t1`.
somax is not a filterax dependency, so install it alongside filterax to run this one.

```python
import equinox as eqx
import filterax as flx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from somax.models import L96State, Lorenz96


class Lorenz96Dynamics(flx.AbstractDynamics):
    """Advance one flat state from t0 to t1 through a somax model."""

    model: Lorenz96
    dt: float = eqx.field(static=True, default=0.01)

    def __call__(self, state, t0, t1):
        solution = self.model.integrate(L96State(x=state), t0, t1, self.dt)
        return solution.ys.x[-1]


class EveryOtherSite(flx.AbstractObsOperator):
    def __call__(self, state):
        return state[::2]


filter_ = flx.ETKF(
    dynamics=Lorenz96Dynamics(model=Lorenz96.create(F=8.0)),
    obs_op=EveryOtherSite(),
)
particles = 8.0 + jr.normal(jr.key(0), (20, 40))  # (N_e, N_x)
observations = [(jnp.full(20, 8.0), 0.05), (jnp.full(20, 8.0), 0.10)]  # (y, t)
noise = lx.DiagonalLinearOperator(0.5 * jnp.ones(20))
result = filter_.assimilate(particles, observations, noise)
```

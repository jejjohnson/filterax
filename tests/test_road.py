"""Tests for the ROAD-EnKF local-gradient training surface.

Two layers of coverage:

1. **Algebraic correctness** — at ``T = 1`` ROAD-EnKF has no cross-time
   terms to drop, so its loss + gradient must equal what
   :func:`differentiable_assimilate` + :func:`jax.value_and_grad`
   produces from the full-tape path. Provides a tight algebraic pin.
2. **Differentiable-DA smoke** — gradient descent through ROAD-EnKF
   moves a learnable scalar dynamics parameter the right way; the
   convenience ``road_enkf_grad_step`` integrates with an ``optax``
   optimiser; JIT + grad compose; stochastic components are rejected.

All assertions are correctness invariants. No Monte-Carlo convergence.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import optax
import pytest

import filterax as flx


# ──────────────────────────────────────────────────────────────────────
# Helpers (mirror tests/test_differentiable.py)
# ──────────────────────────────────────────────────────────────────────


class _LinearObs(flx.AbstractObsOperator):
    H: jnp.ndarray

    def __call__(self, state):
        return self.H @ state


class _IdentityDynamics(flx.AbstractDynamics):
    def __call__(self, state, t0, t1):
        return state


class _LinearDynamics(flx.AbstractDynamics):
    M: jnp.ndarray

    def __call__(self, state, t0, t1):
        return self.M @ state


def _setup(getkey, T: int = 4, N_e: int = 20, N_x: int = 3, N_y: int = 2):
    """Same shape as the differentiable-assimilate test fixture."""
    particles = jr.normal(getkey(), (N_e, N_x))
    H = jnp.zeros((N_y, N_x)).at[jnp.arange(N_y), jnp.arange(N_y)].set(1.0)
    obs_op = _LinearObs(H=H)
    R = lx.DiagonalLinearOperator(jnp.full((N_y,), 0.1))
    obs = jr.normal(getkey(), (T, N_y))
    times = jnp.arange(1.0, T + 1.0)
    return particles, obs, times, obs_op, R


# ──────────────────────────────────────────────────────────────────────
# Algebraic parity: T=1 ROAD matches full-tape value_and_grad
# ──────────────────────────────────────────────────────────────────────


def test_road_one_step_matches_full_value_and_grad(getkey):
    """At ``T = 1`` ROAD has no cross-time terms to drop, so its loss and
    gradient must equal the full-tape ``differentiable_assimilate``
    path under :func:`jax.value_and_grad`."""
    particles, obs, times, obs_op, R = _setup(getkey, T=1)

    def full_loss(dyn):
        result = flx.differentiable_assimilate(
            flx.filters.ETKF(), dyn, obs_op, particles, obs, times, R
        )
        return -jnp.sum(result.log_likelihoods)

    init_M = 0.95 * jnp.eye(particles.shape[1])
    dyn = _LinearDynamics(M=init_M)

    full_loss_val, full_grad = eqx.filter_value_and_grad(full_loss)(dyn)
    road_loss_val, road_grad = flx.differentiable.road_enkf_loss_and_grad(
        dyn, particles, obs, times, R, obs_op
    )

    np.testing.assert_allclose(float(road_loss_val), float(full_loss_val), atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(road_grad.M), np.asarray(full_grad.M), atol=1e-10
    )


def test_road_grad_structure_matches_dynamics_pytree(getkey):
    """The gradient PyTree mirrors the dynamics PyTree — every array leaf
    has a same-shape gradient leaf; static fields are filtered out."""
    particles, obs, times, obs_op, R = _setup(getkey)
    dyn = _LinearDynamics(M=jnp.eye(particles.shape[1]))
    _, grads = flx.differentiable.road_enkf_loss_and_grad(
        dyn, particles, obs, times, R, obs_op
    )
    assert grads.M.shape == dyn.M.shape
    assert grads.M.dtype == dyn.M.dtype


# ──────────────────────────────────────────────────────────────────────
# Smoke: training pattern works end-to-end
# ──────────────────────────────────────────────────────────────────────


def test_road_pattern_a_descent_reduces_loss():
    """Same Pattern A toy as ``test_pattern_a_learn_dynamics_params`` —
    true ``M = 0.95 I``, starting at ``M = 1.0 I``. One ROAD-EnKF
    descent step must reduce the loss."""
    N_e, N_x, N_y, T = 20, 2, 2, 6
    rng = np.random.default_rng(0)
    truth = np.zeros((T, N_x))
    state = np.array([1.0, -0.5])
    H_np = np.eye(N_y, N_x)
    for t in range(T):
        state = 0.95 * state
        truth[t] = state
    obs = jnp.asarray(truth @ H_np.T + 0.05 * rng.standard_normal((T, N_y)))
    times = jnp.arange(1.0, T + 1.0)
    particles = jnp.asarray(rng.standard_normal((N_e, N_x))) + jnp.array([1.0, -0.5])
    H = jnp.eye(N_y, N_x)
    obs_op = _LinearObs(H=H)
    R = lx.DiagonalLinearOperator(jnp.full((N_y,), 0.05**2))

    dyn_init = _LinearDynamics(M=jnp.eye(N_x))
    loss_init, grads = flx.differentiable.road_enkf_loss_and_grad(
        dyn_init, particles, obs, times, R, obs_op
    )
    # Normalise the gradient before stepping — the raw gradient on the
    # full ``M`` matrix has large norm in some entries and a fixed-step
    # update overshoots the local quadratic. A unit-norm descent step is
    # what proves the gradient direction is correct without depending on
    # the loss-surface curvature.
    grad_norm = jnp.linalg.norm(grads.M)
    step = 0.01
    dyn_after = eqx.tree_at(
        lambda d: d.M, dyn_init, dyn_init.M - step * grads.M / grad_norm
    )
    loss_after, _ = flx.differentiable.road_enkf_loss_and_grad(
        dyn_after, particles, obs, times, R, obs_op
    )
    assert float(loss_after) < float(loss_init)


def test_road_grad_step_with_optax_optimizer():
    """``road_enkf_grad_step`` composes with an :mod:`optax` optimiser —
    multi-step descent reduces the loss."""
    N_e, N_x, N_y, T = 16, 2, 2, 6
    rng = np.random.default_rng(1)
    truth = np.zeros((T, N_x))
    state = np.array([1.0, -0.5])
    for t in range(T):
        state = 0.9 * state
        truth[t] = state
    H_np = np.eye(N_y, N_x)
    obs = jnp.asarray(truth @ H_np.T + 0.05 * rng.standard_normal((T, N_y)))
    times = jnp.arange(1.0, T + 1.0)
    particles = jnp.asarray(rng.standard_normal((N_e, N_x))) + jnp.array([1.0, -0.5])
    H = jnp.eye(N_y, N_x)
    obs_op = _LinearObs(H=H)
    R = lx.DiagonalLinearOperator(jnp.full((N_y,), 0.05**2))

    dyn = _LinearDynamics(M=1.1 * jnp.eye(N_x))
    optimizer = optax.adam(0.05)
    opt_state = optimizer.init(eqx.filter(dyn, eqx.is_inexact_array))

    losses = []
    for _ in range(5):
        dyn, opt_state, loss = flx.differentiable.road_enkf_grad_step(
            dyn, optimizer, opt_state, particles, obs, times, R, obs_op
        )
        losses.append(float(loss))
    assert losses[-1] < losses[0]


def test_road_skips_integer_array_leaves():
    """Regression for the gradient-accumulator predicate: a dynamics
    module carrying an integer index leaf alongside a trainable float
    matrix must produce a ``None`` slot for the integer leaf so optax
    can't try to write float updates onto it."""

    class _IndexedDynamics(flx.AbstractDynamics):
        M: jnp.ndarray  # trainable
        obs_indices: jnp.ndarray  # non-trainable int metadata

        def __call__(self, state, t0, t1):
            # `obs_indices` is carried but never differentiated.
            return self.M @ state

    N_e, N_x, N_y, T = 12, 3, 2, 3
    particles = jr.normal(jr.PRNGKey(4), (N_e, N_x))
    obs = jr.normal(jr.PRNGKey(5), (T, N_y))
    times = jnp.arange(1.0, T + 1.0)
    H = jnp.eye(N_y, N_x)
    obs_op = _LinearObs(H=H)
    R = lx.DiagonalLinearOperator(jnp.full((N_y,), 0.1))
    dyn = _IndexedDynamics(M=jnp.eye(N_x), obs_indices=jnp.arange(N_y, dtype=jnp.int32))
    _, grads = flx.differentiable.road_enkf_loss_and_grad(
        dyn, particles, obs, times, R, obs_op
    )
    assert grads.M is not None
    assert grads.obs_indices is None
    # And the full optimizer-update path doesn't choke on the int leaf.
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(eqx.filter(dyn, eqx.is_inexact_array))
    new_dyn, _, _ = flx.differentiable.road_enkf_grad_step(
        dyn, optimizer, opt_state, particles, obs, times, R, obs_op
    )
    # Integer field stayed untouched.
    np.testing.assert_array_equal(
        np.asarray(new_dyn.obs_indices), np.asarray(dyn.obs_indices)
    )


def test_road_grad_step_with_optax_chain():
    """``road_enkf_grad_step`` works under ``optax.chain`` (gradient
    clipping + Adam) — proves the gradient piping is standard-optax-
    compatible."""
    particles = jr.normal(jr.PRNGKey(2), (12, 2))
    obs = jr.normal(jr.PRNGKey(3), (4, 1))
    times = jnp.arange(1.0, 5.0)
    H = jnp.asarray([[1.0, 0.0]])
    obs_op = _LinearObs(H=H)
    R = lx.DiagonalLinearOperator(jnp.full((1,), 0.1))

    dyn = _LinearDynamics(M=jnp.eye(2))
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(0.01))
    opt_state = optimizer.init(eqx.filter(dyn, eqx.is_inexact_array))

    dyn, opt_state, loss = flx.differentiable.road_enkf_grad_step(
        dyn, optimizer, opt_state, particles, obs, times, R, obs_op
    )
    assert jnp.isfinite(loss)
    assert bool(jnp.all(jnp.isfinite(dyn.M)))


# ──────────────────────────────────────────────────────────────────────
# JIT & memory sanity
# ──────────────────────────────────────────────────────────────────────


def test_road_jit_compiles(getkey):
    particles, obs, times, obs_op, R = _setup(getkey)
    dyn = _LinearDynamics(M=jnp.eye(particles.shape[1]))

    @eqx.filter_jit
    def run(d):
        return flx.differentiable.road_enkf_loss_and_grad(
            d, particles, obs, times, R, obs_op
        )

    loss, grads = run(dyn)
    assert jnp.isfinite(loss)
    assert bool(jnp.all(jnp.isfinite(grads.M)))


def test_road_long_horizon_compiles(getkey):
    """ROAD's ``O(1)``-per-step backward-pass tape lets a long ``T``
    compile that would be unwieldy under full-tape backprop. Smoke check
    at ``T = 30``."""
    particles, _, _, obs_op, R = _setup(getkey, T=4)
    T = 30
    obs = jr.normal(getkey(), (T, 2))
    times = jnp.arange(1.0, T + 1.0)
    dyn = _LinearDynamics(M=jnp.eye(particles.shape[1]))
    loss, grads = flx.differentiable.road_enkf_loss_and_grad(
        dyn, particles, obs, times, R, obs_op
    )
    assert jnp.isfinite(loss)
    assert bool(jnp.all(jnp.isfinite(grads.M)))


# ──────────────────────────────────────────────────────────────────────
# Validation & dtype handling
# ──────────────────────────────────────────────────────────────────────


def test_road_rejects_stochastic_enkf(getkey):
    particles, obs, times, obs_op, R = _setup(getkey)
    dyn = _IdentityDynamics()
    with pytest.raises(ValueError, match="StochasticEnKF"):
        flx.differentiable.road_enkf_loss_and_grad(
            dyn,
            particles,
            obs,
            times,
            R,
            obs_op,
            filter_=flx.filters.StochasticEnKF(0),
        )


def test_road_rejects_additive_inflator(getkey):
    particles, obs, times, obs_op, R = _setup(getkey)
    dyn = _IdentityDynamics()
    inflator = flx.AdditiveInflator(
        noise_cov=lx.DiagonalLinearOperator(jnp.full(particles.shape[1], 0.01)),
        base_key=jr.PRNGKey(0),
    )
    with pytest.raises(ValueError, match="AdditiveInflator"):
        flx.differentiable.road_enkf_loss_and_grad(
            dyn,
            particles,
            obs,
            times,
            R,
            obs_op,
            inflator=inflator,
        )


def test_road_rejects_mismatched_obs_and_times(getkey):
    particles, obs, times, obs_op, R = _setup(getkey, T=4)
    dyn = _IdentityDynamics()
    with pytest.raises(ValueError, match="same leading axis"):
        flx.differentiable.road_enkf_loss_and_grad(
            dyn,
            particles,
            obs,
            times[:3],
            R,
            obs_op,
        )


def test_road_handles_mixed_time_dtypes(getkey):
    """Same mixed-dtype regression as
    ``test_diff_assimilate_handles_mixed_time_dtypes`` — integer obs
    times paired with a float ensemble must trace cleanly."""
    particles, obs, _, obs_op, R = _setup(getkey)
    dyn = _LinearDynamics(M=jnp.eye(particles.shape[1]))
    int_times = jnp.arange(1, obs.shape[0] + 1, dtype=jnp.int32)
    loss, grads = flx.differentiable.road_enkf_loss_and_grad(
        dyn,
        particles,
        obs,
        int_times,
        R,
        obs_op,
        t0=0,
    )
    assert jnp.isfinite(loss)
    assert bool(jnp.all(jnp.isfinite(grads.M)))

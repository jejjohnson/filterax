"""Tests for the Wave 5.B differentiable assimilation loop.

Two layers of coverage:

1. **Algebraic correctness** — :func:`differentiable_assimilate` must
   match the L2 :class:`filterax.ETKF` / :class:`filterax.EnSRF` Python-
   loop assimilation step-for-step on the same inputs.
2. **Differentiable-DA smoke tests** — gradient descent through the
   filter must reduce the observation-space NLL when learning
   dynamics, observation operator, or a scalar inflation parameter
   (Patterns A / B / C from
   ``design_docs/features/differentiable_da.md`` §6).

All assertions are correctness invariants — we check that the gradient
moves the loss in the right direction, not how close it gets to a
specific minimiser.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

import filterax as flx


# ──────────────────────────────────────────────────────────────────────
# Helpers
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
    """Linear-Gaussian assimilation problem with identity dynamics."""
    particles = jr.normal(getkey(), (N_e, N_x))
    H = jnp.zeros((N_y, N_x)).at[jnp.arange(N_y), jnp.arange(N_y)].set(1.0)
    obs_op = _LinearObs(H=H)
    dyn = _IdentityDynamics()
    R = lx.DiagonalLinearOperator(jnp.full((N_y,), 0.1))
    obs = jr.normal(getkey(), (T, N_y))
    times = jnp.arange(1.0, T + 1.0)
    obs_list = [(obs[t], float(times[t])) for t in range(T)]
    return particles, obs, times, obs_list, obs_op, dyn, R


# ──────────────────────────────────────────────────────────────────────
# Parity with the L2 Python-loop assimilation
# ──────────────────────────────────────────────────────────────────────


def test_diff_assimilate_matches_l2_etkf(getkey):
    """Scan-based loop and L2 Python-loop must produce identical history."""
    particles, obs, times, obs_list, obs_op, dyn, R = _setup(getkey)
    ref = flx.ETKF(dynamics=dyn, obs_op=obs_op).assimilate(particles, obs_list, R)
    got = flx.differentiable_assimilate(
        flx.filters.ETKF(), dyn, obs_op, particles, obs, times, R
    )
    np.testing.assert_allclose(
        np.asarray(got.forecast_history),
        np.asarray(ref.forecast_history),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(got.analysis_history),
        np.asarray(ref.analysis_history),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(got.log_likelihoods),
        np.asarray(ref.log_likelihoods),
        atol=1e-10,
    )


def test_diff_assimilate_matches_l2_ensrf(getkey):
    """Same parity check for EnSRF."""
    particles, obs, times, obs_list, obs_op, dyn, R = _setup(getkey)
    ref = flx.EnSRF(dynamics=dyn, obs_op=obs_op).assimilate(particles, obs_list, R)
    got = flx.differentiable_assimilate(
        flx.filters.EnSRF(), dyn, obs_op, particles, obs, times, R
    )
    np.testing.assert_allclose(
        np.asarray(got.analysis_history),
        np.asarray(ref.analysis_history),
        atol=1e-10,
    )


def test_diff_assimilate_with_inflator_matches_l2(getkey):
    """Deterministic inflator threads through the same way."""
    particles, obs, times, obs_list, obs_op, dyn, R = _setup(getkey)
    inflator = flx.MultiplicativeInflator(factor=1.05)
    ref = flx.ETKF(dynamics=dyn, obs_op=obs_op, inflator=inflator).assimilate(
        particles, obs_list, R
    )
    got = flx.differentiable_assimilate(
        flx.filters.ETKF(),
        dyn,
        obs_op,
        particles,
        obs,
        times,
        R,
        inflator=inflator,
    )
    np.testing.assert_allclose(
        np.asarray(got.analysis_history),
        np.asarray(ref.analysis_history),
        atol=1e-10,
    )


def test_diff_assimilate_letkf_with_coords(getkey):
    """LETKF needs ``state_coords`` / ``obs_coords`` — they thread through
    ``analysis_extra`` and the scan loop matches the L2 LETKF."""
    N_e, N_x, N_y, T = 20, 4, 2, 3
    particles = jr.normal(getkey(), (N_e, N_x))
    state_coords = jnp.arange(N_x, dtype=jnp.float64)[:, None]
    obs_coords = jnp.asarray([[0.5], [2.5]])
    H = jnp.zeros((N_y, N_x)).at[jnp.arange(N_y), jnp.asarray([0, 2])].set(1.0)
    obs_op = _LinearObs(H=H)
    dyn = _IdentityDynamics()
    R = lx.DiagonalLinearOperator(jnp.full((N_y,), 0.3))
    obs = jr.normal(getkey(), (T, N_y))
    times = jnp.arange(1.0, T + 1.0)
    obs_list = [(obs[t], float(times[t])) for t in range(T)]

    ref = flx.LETKF(dynamics=dyn, obs_op=obs_op, radius=1.5).assimilate(
        particles, obs_list, R, state_coords=state_coords, obs_coords=obs_coords
    )
    got = flx.differentiable_assimilate(
        flx.filters.LETKF(radius=1.5),
        dyn,
        obs_op,
        particles,
        obs,
        times,
        R,
        state_coords=state_coords,
        obs_coords=obs_coords,
    )
    np.testing.assert_allclose(
        np.asarray(got.analysis_history),
        np.asarray(ref.analysis_history),
        atol=1e-10,
    )


def test_diff_assimilate_checkpoint_is_value_equivalent(getkey):
    """``checkpoint=True`` is a memory/compute trade-off, not a numerical
    change — the output values must be identical."""
    particles, obs, times, _, obs_op, dyn, R = _setup(getkey)
    eager = flx.differentiable_assimilate(
        flx.filters.ETKF(), dyn, obs_op, particles, obs, times, R, checkpoint=False
    )
    ckpt = flx.differentiable_assimilate(
        flx.filters.ETKF(), dyn, obs_op, particles, obs, times, R, checkpoint=True
    )
    np.testing.assert_allclose(
        np.asarray(ckpt.analysis_history),
        np.asarray(eager.analysis_history),
        atol=1e-12,
    )


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────


def test_rejects_stochastic_enkf(getkey):
    particles, obs, times, _, obs_op, dyn, R = _setup(getkey)
    with pytest.raises(ValueError, match="StochasticEnKF"):
        flx.differentiable_assimilate(
            flx.filters.StochasticEnKF(0), dyn, obs_op, particles, obs, times, R
        )


def test_rejects_etkf_livings(getkey):
    particles, obs, times, _, obs_op, dyn, R = _setup(getkey)
    with pytest.raises(ValueError, match="ETKF_Livings"):
        flx.differentiable_assimilate(
            flx.filters.ETKF_Livings(0), dyn, obs_op, particles, obs, times, R
        )


def test_rejects_additive_inflator(getkey):
    particles, obs, times, _, obs_op, dyn, R = _setup(getkey)
    inflator = flx.AdditiveInflator(
        noise_cov=lx.DiagonalLinearOperator(jnp.full(particles.shape[1], 0.01)),
        base_key=jr.PRNGKey(0),
    )
    with pytest.raises(ValueError, match="AdditiveInflator"):
        flx.differentiable_assimilate(
            flx.filters.ETKF(),
            dyn,
            obs_op,
            particles,
            obs,
            times,
            R,
            inflator=inflator,
        )


def test_rejects_mismatched_obs_and_times(getkey):
    particles, obs, times, _, obs_op, dyn, R = _setup(getkey, T=4)
    with pytest.raises(ValueError, match="same leading axis"):
        flx.differentiable_assimilate(
            flx.filters.ETKF(), dyn, obs_op, particles, obs, times[:3], R
        )


# ──────────────────────────────────────────────────────────────────────
# Differentiable-DA training patterns (smoke)
# ──────────────────────────────────────────────────────────────────────


def _nll(result):
    """Negative log-likelihood, summed over the assimilation window."""
    return -jnp.sum(result.log_likelihoods)


def test_pattern_a_learn_dynamics_params(getkey):
    """Pattern A: backprop the observation-space NLL through the filter
    moves a learnable scalar dynamics parameter toward the truth.

    ``true_M = 0.95 * I``; the loss-gradient at ``M = 1.0 * I`` should
    have the right sign to push the parameter *down* (toward 0.95).
    """
    N_e, N_x, N_y, T = 20, 2, 2, 6
    rng = np.random.default_rng(0)
    truth_traj = np.zeros((T, N_x))
    state = np.array([1.0, -0.5])
    H_np = np.eye(N_y, N_x)
    for t in range(T):
        state = 0.95 * state
        truth_traj[t] = state
    obs = jnp.asarray(truth_traj @ H_np.T + 0.05 * rng.standard_normal((T, N_y)))
    times = jnp.arange(1.0, T + 1.0)
    particles = jnp.asarray(rng.standard_normal((N_e, N_x))) + jnp.array([1.0, -0.5])

    H = jnp.eye(N_y, N_x)
    obs_op = _LinearObs(H=H)
    R = lx.DiagonalLinearOperator(jnp.full((N_y,), 0.05**2))
    filter_ = flx.filters.ETKF()

    def loss(scale):
        dyn = _LinearDynamics(M=scale * jnp.eye(N_x))
        result = flx.differentiable_assimilate(
            filter_, dyn, obs_op, particles, obs, times, R
        )
        return _nll(result)

    grad_at_one = jax.grad(loss)(1.0)
    # NLL increases when M is too large (true M = 0.95); gradient at
    # ``scale = 1.0`` should be positive so a descent step *decreases* scale.
    assert float(grad_at_one) > 0
    # One descent step lowers the loss.
    step = 0.01
    loss_before = float(loss(1.0))
    loss_after = float(loss(1.0 - step * jnp.sign(grad_at_one)))
    assert loss_after < loss_before


def test_pattern_b_learn_observation_operator_params(getkey):
    """Pattern B: learn the observation operator (a scalar gain on the
    first state component) end-to-end through the filter."""
    N_e, N_x, N_y, T = 20, 2, 1, 4
    rng = np.random.default_rng(1)
    # True observation: y = 2.0 * x[0]; we'll start at gain = 1.0.
    truth = np.array([1.5, -0.7])
    truth_traj = np.tile(truth, (T, 1))
    obs = jnp.asarray(2.0 * truth_traj[:, :1] + 0.05 * rng.standard_normal((T, N_y)))
    times = jnp.arange(1.0, T + 1.0)
    particles = jnp.asarray(rng.standard_normal((N_e, N_x))) + truth
    R = lx.DiagonalLinearOperator(jnp.full((N_y,), 0.05**2))
    filter_ = flx.filters.ETKF()
    dyn = _IdentityDynamics()

    def loss(gain):
        H = jnp.asarray([[gain, 0.0]])
        obs_op = _LinearObs(H=H)
        result = flx.differentiable_assimilate(
            filter_, dyn, obs_op, particles, obs, times, R
        )
        return _nll(result)

    grad_at_one = jax.grad(loss)(1.0)
    # True gain is 2.0; loss should be lower for larger gain, so the
    # gradient at gain = 1.0 must be negative.
    assert float(grad_at_one) < 0
    step = 0.05
    loss_before = float(loss(1.0))
    loss_after = float(loss(1.0 - step * grad_at_one / jnp.abs(grad_at_one)))
    assert loss_after < loss_before


def test_pattern_c_learn_inflation_factor(getkey):
    """Pattern C: optimise a learnable multiplicative inflation factor.

    The differentiable path uses the low-level ``inflate_multiplicative``
    primitive (which is gradient-friendly in ``factor``) rather than
    ``MultiplicativeInflator``'s static field. Loss should drop when we
    step in the negative-gradient direction.
    """
    N_e, N_x, N_y, T = 20, 3, 2, 4
    particles, obs, times, _, obs_op, dyn, R = _setup(
        getkey, T=T, N_e=N_e, N_x=N_x, N_y=N_y
    )
    filter_ = flx.filters.ETKF()

    def loss(log_factor):
        factor = jnp.exp(log_factor)

        # Hand-rolled differentiable step using the package's primitives.
        def step(carry, inputs):
            ens, t_prev = carry
            obs_t, t_now = inputs
            forecast = jax.vmap(lambda x: dyn(x, t_prev, t_now))(ens)
            result = filter_.analysis(forecast, obs_t, obs_op, R)
            analysed = flx.inflate_multiplicative(result.particles, factor)
            return (analysed, t_now), result.log_likelihood

        _, logps = jax.lax.scan(step, (particles, jnp.asarray(0.0)), (obs, times))
        return -jnp.sum(logps)

    log_factor_0 = jnp.log(1.2)
    grad = jax.grad(loss)(log_factor_0)
    assert jnp.isfinite(grad)
    assert grad != 0.0
    step = 0.05
    loss_before = float(loss(log_factor_0))
    loss_after = float(loss(log_factor_0 - step * jnp.sign(grad)))
    assert loss_after < loss_before


def test_diff_assimilate_under_jit_and_grad(getkey):
    """JIT-compiled gradient of the NLL through the scan loop runs and is
    finite — required composition of jit + grad + scan."""
    particles, obs, times, _, obs_op, _dyn, R = _setup(getkey)
    filter_ = flx.filters.ETKF()

    @jax.jit
    @jax.grad
    def grad_loss(scale):
        dyn = _LinearDynamics(M=scale * jnp.eye(particles.shape[1]))
        result = flx.differentiable_assimilate(
            filter_, dyn, obs_op, particles, obs, times, R
        )
        return _nll(result)

    g = grad_loss(1.0)
    assert jnp.isfinite(g)


def test_diff_assimilate_grad_with_checkpoint(getkey):
    """``checkpoint=True`` must round-trip through ``jax.grad`` (the whole
    reason it exists)."""
    particles, obs, times, _, obs_op, _dyn, R = _setup(getkey)
    filter_ = flx.filters.ETKF()

    def loss(scale):
        dyn = _LinearDynamics(M=scale * jnp.eye(particles.shape[1]))
        result = flx.differentiable_assimilate(
            filter_,
            dyn,
            obs_op,
            particles,
            obs,
            times,
            R,
            checkpoint=True,
        )
        return _nll(result)

    g = jax.grad(loss)(1.0)
    assert jnp.isfinite(g)

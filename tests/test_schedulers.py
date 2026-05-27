"""Tests for the EKP step-size schedulers."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import filterax as flx


def _state_with_evals(particles, evals, noise_cov, algo_time=0.0):
    return flx.ProcessState(
        particles=particles,
        forward_evals=evals,
        obs=jnp.zeros(evals.shape[1]),
        noise_cov=noise_cov,
        step=jnp.asarray(0, dtype=jnp.int32),
        algo_time=jnp.asarray(algo_time),
    )


def test_fixed_scheduler_returns_constant():
    sched = flx.FixedScheduler(dt=0.42)
    state = _state_with_evals(
        jnp.zeros((4, 3)), jnp.zeros((4, 2)), lx.DiagonalLinearOperator(jnp.ones(2))
    )
    assert float(sched.get_dt(state)) == pytest.approx(0.42)


def test_data_misfit_controller_clamps_to_remaining_algo_time():
    # When obs ≈ G(θ), the misfit is tiny and dt would be huge — but the
    # controller clamps to (1 - algo_time).
    sched = flx.DataMisfitController()
    evals = jnp.zeros((5, 2))
    state = _state_with_evals(
        jnp.zeros((5, 3)),
        evals,
        lx.DiagonalLinearOperator(jnp.ones(2)),
        algo_time=0.9,
    )
    dt = float(sched.get_dt(state))
    assert dt == pytest.approx(0.1, abs=1e-6)


def test_data_misfit_controller_shrinks_when_misfit_is_large():
    # Far from the data: misfit is huge → dt small.
    sched = flx.DataMisfitController(target_misfit=1.0)
    evals = 10.0 * jnp.ones((5, 2))
    state = _state_with_evals(
        jnp.zeros((5, 3)),
        evals,
        lx.DiagonalLinearOperator(0.01 * jnp.ones(2)),
    )
    # Misfit = mean(Σ_d (0 - 10)^2 / 0.01) = 2 * 100 / 0.01 = 20000.
    # dt = 1 / 20000 = 5e-5.
    dt = float(sched.get_dt(state))
    np.testing.assert_allclose(dt, 5e-5, rtol=1e-3)


def test_eks_stable_scheduler_clips_to_max_dt():
    sched = flx.EKSStableScheduler(max_dt=0.1, target=1.0)
    # Tight ensemble → small ‖Cᶿᶿ‖ → would give large dt, but max_dt caps.
    particles = jnp.eye(2)[None, :, :].repeat(5, axis=0).reshape(5, 4)[:, :2]
    # Add a sliver of variance.
    particles = particles + 0.001 * jnp.arange(5)[:, None]
    state = _state_with_evals(
        particles, jnp.zeros((5, 2)), lx.DiagonalLinearOperator(jnp.ones(2))
    )
    assert float(sched.get_dt(state)) == pytest.approx(0.1, abs=1e-9)


def test_eks_stable_scheduler_shrinks_for_spread_ensembles():
    # Large ensemble spread → large ‖Cᶿᶿ‖ → small dt.
    import jax.random as jr

    sched = flx.EKSStableScheduler(max_dt=10.0, target=1.0)
    particles = 5.0 * jr.normal(jr.PRNGKey(0), (50, 4))
    state = _state_with_evals(
        particles, jnp.zeros((50, 2)), lx.DiagonalLinearOperator(jnp.ones(2))
    )
    dt = float(sched.get_dt(state))
    assert 0 < dt < 1.0  # well below max_dt

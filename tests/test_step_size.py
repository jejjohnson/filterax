"""The step size Δt enters the tempered Kalman updates exactly once (#106).

A tempered update with step ``Δt`` replaces ``Γ`` by ``Δt⁻¹ Γ`` in the
innovation covariance and applies the plain Kalman gain. Two exact
consequences pin that down, which a one-step ``Δt = 1`` test cannot:

* **Splitting.** For a linear forward model, a Gaussian belief updated by
  steps whose sizes sum to one ends at the exact posterior however the
  sum is split. UKI carries the belief exactly, so it must match the
  closed form for every split.
* **Small-step limit.** As ``Δt → 0`` the EKI increment tends to
  ``Δt Cᶿᴳ Γ⁻¹ (y − G⁽ʲ⁾)``, the Euler step of ``θ̇ = Cᶿᴳ Γ⁻¹ (y − G)``.
  Every process built on the shared increment inherits it.

The regression problem has more data than parameters and a noise level
at which the old double-Δt update left ``DataMisfitController`` at
``algo_time ≈ 0.14`` after 200 iterations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optax
import pytest

import filterax as flx
from filterax import ProcessConfig
from filterax.processes import EKI as EKIProcess, ETKI, SparseInversion


N_P, N_D, GAMMA = 6, 15, 1e-2


def _problem():
    """Overdetermined linear-Gaussian problem with its closed-form posterior."""
    F = jax.random.uniform(jax.random.key(0), (N_D, N_P))
    theta_true = jnp.linspace(0.5, 1.5, N_P)
    noise = jax.random.normal(jax.random.key(1), (N_D,))
    y = F @ theta_true + jnp.sqrt(GAMMA) * noise
    m0 = jnp.ones(N_P)
    S0 = 0.5 * jnp.eye(N_P)
    post_cov = jnp.linalg.inv(jnp.linalg.inv(S0) + F.T @ F / GAMMA)
    post_mean = m0 + post_cov @ F.T @ (y - F @ m0) / GAMMA
    Gamma = lx.MatrixLinearOperator(GAMMA * jnp.eye(N_D), lx.positive_semidefinite_tag)
    return F, y, Gamma, m0, S0, post_mean, post_cov


SPLITS = [(1.0, 1), (0.5, 2), (0.1, 10)]


@pytest.mark.parametrize(("dt", "n"), SPLITS)
def test_uki_recovers_posterior_for_any_split(dt, n):
    F, y, Gamma, m0, S0, post_mean, post_cov = _problem()
    uki = flx.UKI(
        forward_fn=lambda th: F @ th,
        obs=y,
        noise_cov=Gamma,
        config=ProcessConfig(scheduler=flx.FixedScheduler(dt=dt), n_iterations=n),
    )
    result = uki.run(m0, lx.MatrixLinearOperator(S0, lx.positive_semidefinite_tag))
    np.testing.assert_allclose(result.mean, post_mean, atol=1e-10)
    np.testing.assert_allclose(result.covariance.as_matrix(), post_cov, atol=1e-10)


@pytest.mark.parametrize(("dt", "n"), SPLITS)
def test_uki_optax_recovers_posterior_for_any_split(dt, n):
    F, y, Gamma, m0, S0, post_mean, post_cov = _problem()
    transform = flx.optax.uki(
        forward_fn=lambda th: F @ th,
        obs=y,
        noise_cov=Gamma,
        init_cov=S0,
        scheduler=flx.FixedScheduler(dt=dt),
    )
    params = m0
    state = transform.init(params)
    for _ in range(n):
        updates, state = transform.update(None, state, params)
        params = optax.apply_updates(params, updates)
    np.testing.assert_allclose(params, post_mean, atol=1e-10)
    np.testing.assert_allclose(state.covariance.as_matrix(), post_cov, atol=1e-10)


@pytest.mark.parametrize(
    "make_process",
    [
        pytest.param(lambda s: EKIProcess(scheduler=s), id="EKI"),
        pytest.param(lambda s: ETKI(scheduler=s), id="ETKI"),
        pytest.param(
            lambda s: SparseInversion(scheduler=s, penalty_weight=0.0),
            id="SparseInversion",
        ),
    ],
)
def test_increment_tends_to_euler_step(make_process):
    F, y, Gamma, m0, S0, _, _ = _problem()
    particles = m0 + jax.random.normal(jax.random.key(2), (40, N_P)) @ jnp.sqrt(S0)
    evals = particles @ F.T
    dt = 1e-8
    process = make_process(flx.FixedScheduler(dt=dt))
    state = process.init(particles, y, Gamma)
    delta = process.update(state, evals).particles - particles
    # Δt Cᶿᴳ Γ⁻¹ (y − G⁽ʲ⁾)
    anom_p = particles - particles.mean(0)
    anom_g = evals - evals.mean(0)
    C_pg = anom_p.T @ anom_g / (particles.shape[0] - 1)
    euler = dt * (y[None, :] - evals) @ C_pg.T / GAMMA
    # The gap is O(Δt ‖Cᴳᴳ‖ / γ) relative to the step: ≈ 1e-5 here. The old
    # double-Δt increment was a further factor Δt smaller, so its gap was ≈ 1.
    gap = jnp.max(jnp.abs(delta - euler)) / jnp.max(jnp.abs(euler))
    assert float(gap) < 1e-4


def test_data_misfit_controller_reaches_algo_time_one():
    F, y, Gamma, m0, S0, _, _ = _problem()
    particles = m0 + jax.random.normal(jax.random.key(2), (400, N_P)) @ jnp.sqrt(S0)
    eki = flx.EKI(
        forward_fn=lambda th: F @ th,
        obs=y,
        noise_cov=Gamma,
        config=ProcessConfig(scheduler=flx.DataMisfitController(), n_iterations=200),
    )
    result = eki.run(particles)
    assert result.converged
    assert int(result.n_iterations) <= 50

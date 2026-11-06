"""Tests for the EKP step-size schedulers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
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


def _iglesias_yang_candidates(residuals, noise_variance):
    """Closed-form candidates of Iglesias & Yang (2021, eq. 14), for diagonal Γ."""
    n_obs = residuals.shape[1]
    misfits = 0.5 * np.sum(np.asarray(residuals) ** 2 / noise_variance, axis=1)
    accuracy = n_obs / (2.0 * misfits.mean())
    uncertainty = np.sqrt(n_obs / (2.0 * misfits.var(ddof=1)))
    return accuracy, uncertainty


@pytest.mark.slow
def test_data_misfit_controller_matches_iglesias_yang_closed_form():
    """Regression for #116: the old step was M times too small on this problem."""
    n_members, n_params, n_obs = 50, 10, 100
    key_g, key_u, key_y = jr.split(jr.key(0), 3)
    G = jr.normal(key_g, (n_obs, n_params))
    particles = jr.normal(key_u, (n_members, n_params))
    obs = G @ jnp.ones(n_params) + jr.normal(key_y, (n_obs,))
    evals = particles @ G.T
    state = flx.ProcessState(
        particles=particles,
        forward_evals=evals,
        obs=obs,
        noise_cov=lx.DiagonalLinearOperator(jnp.ones(n_obs)),
        step=jnp.asarray(0, dtype=jnp.int32),
        algo_time=jnp.asarray(0.0),
    )
    accuracy, uncertainty = _iglesias_yang_candidates(obs[None, :] - evals, 1.0)
    expected = min(max(accuracy, uncertainty), 1.0)
    dt = flx.DataMisfitController().get_dt(state)
    np.testing.assert_allclose(float(dt), expected, rtol=1e-12)


@pytest.mark.parametrize(
    ("residuals", "accuracy_binds"),
    [
        # Nearly equal misfits: σ²_Φ is small, so the √ candidate is the larger.
        ([[30.0, 0.0], [30.1, 0.0], [30.2, 0.0]], False),
        # Very unequal misfits: σ²_Φ is large, so M / 2Φ̄ is the larger.
        ([[1.0, 0.0], [10.0, 0.0], [100.0, 0.0]], True),
    ],
)
def test_data_misfit_controller_takes_the_larger_candidate(residuals, accuracy_binds):
    residuals = jnp.asarray(residuals)
    state = _state_with_evals(
        jnp.zeros((3, 2)), -residuals, lx.DiagonalLinearOperator(jnp.ones(2))
    )
    accuracy, uncertainty = _iglesias_yang_candidates(residuals, 1.0)
    assert (accuracy > uncertainty) == accuracy_binds
    dt = float(flx.DataMisfitController().get_dt(state))
    np.testing.assert_allclose(dt, max(accuracy, uncertainty), rtol=1e-12)


def test_data_misfit_controller_equal_misfits_take_the_remaining_budget():
    """σ²_Φ = 0 makes the second candidate infinite, so only 1 − tₙ binds."""
    state = _state_with_evals(
        jnp.zeros((5, 3)),
        10.0 * jnp.ones((5, 2)),
        lx.DiagonalLinearOperator(0.01 * jnp.ones(2)),
        algo_time=0.25,
    )
    assert float(flx.DataMisfitController().get_dt(state)) == pytest.approx(0.75)


def test_data_misfit_controller_is_zero_past_the_budget():
    """algo_time can overshoot 1, and a negative step must not reach the update."""
    evals = jnp.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    state = _state_with_evals(
        jnp.zeros((3, 2)),
        evals,
        lx.DiagonalLinearOperator(jnp.ones(2)),
        algo_time=1.5,
    )
    assert float(flx.DataMisfitController().get_dt(state)) == 0.0


def test_data_misfit_controller_runs_under_jit():
    evals = jnp.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    state = _state_with_evals(
        jnp.zeros((3, 2)), evals, lx.DiagonalLinearOperator(jnp.ones(2))
    )
    controller = flx.DataMisfitController()
    eager = controller.get_dt(state)
    np.testing.assert_allclose(jax.jit(controller.get_dt)(state), eager, rtol=1e-12)


@pytest.mark.slow
def test_eki_with_data_misfit_controller_reaches_time_one_in_few_iterations():
    """Regression for #116: at M = 100 the old controller needed about M times more."""
    n_members, n_params, n_obs = 50, 10, 100
    key_g, key_u, key_y = jr.split(jr.key(0), 3)
    G = jr.normal(key_g, (n_obs, n_params))
    particles = jr.normal(key_u, (n_members, n_params))
    obs = G @ jnp.ones(n_params) + jr.normal(key_y, (n_obs,))
    eki = flx.EKI(
        forward_fn=lambda theta: G @ theta,
        obs=obs,
        noise_cov=lx.DiagonalLinearOperator(jnp.ones(n_obs)),
        config=flx.ProcessConfig(scheduler=flx.DataMisfitController(), n_iterations=20),
    )
    result = eki.run(particles)
    assert result.converged


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_eks_stable_scheduler_uses_the_frobenius_norm_of_the_covariance():
    sched = flx.EKSStableScheduler(max_dt=1e6, target=1.0, target_stiffness=1e6)
    particles = jr.normal(jr.PRNGKey(0), (6, 20))
    state = _state_with_evals(
        particles, jnp.zeros((6, 2)), lx.DiagonalLinearOperator(jnp.ones(2))
    )
    anomalies = particles - particles.mean(axis=0)
    covariance = anomalies.T @ anomalies / 5
    expected = 1.0 / jnp.sqrt(jnp.sum(covariance * covariance))
    # Round-off through one (20, 20) product against one (6, 6) product
    assert jnp.allclose(sched.get_dt(state), expected, rtol=1e-5)


def _stiffness_state(gamma=0.1, prior_variance=None, scale=1.0, num_members=40):
    """A linear model's state, with its dense stiffness ``S`` and ``Cᶿᶿ H``."""
    F = jnp.asarray([[1.0, 0.5], [-0.3, 1.2], [0.7, 0.1]]) / scale
    particles = scale * jr.normal(jr.PRNGKey(1), (num_members, 2))
    evals = particles @ F.T
    prior_cov = None if prior_variance is None else prior_variance * jnp.eye(2)
    state = flx.ProcessState(
        particles=particles,
        forward_evals=evals,
        obs=jnp.zeros(3),
        noise_cov=lx.DiagonalLinearOperator(gamma * jnp.ones(3)),
        step=jnp.asarray(0, dtype=jnp.int32),
        algo_time=jnp.asarray(0.0),
        prior_cov=None
        if prior_cov is None
        else lx.MatrixLinearOperator(prior_cov, lx.positive_semidefinite_tag),
    )
    anomalies = particles - particles.mean(0)
    obs_anomalies = evals - evals.mean(0)
    stiffness = obs_anomalies @ obs_anomalies.T / gamma
    H = F.T @ F / gamma
    if prior_cov is not None:
        stiffness = stiffness + anomalies @ jnp.linalg.inv(prior_cov) @ anomalies.T
        H = H + jnp.linalg.inv(prior_cov)
    C = anomalies.T @ anomalies / (num_members - 1)
    return state, stiffness / (num_members - 1), C @ H


_ONLY_STIFFNESS = {"max_dt": 1e6, "target": 1e6}


@pytest.mark.slow
@pytest.mark.parametrize("prior_variance", [None, 0.5], ids=["flat", "prior"])
def test_eks_stable_scheduler_is_bounded_by_the_dense_stiffness(prior_variance):
    state, stiffness, _ = _stiffness_state(prior_variance=prior_variance)
    sched = flx.EKSStableScheduler(target_stiffness=0.3, **_ONLY_STIFFNESS)
    expected = 0.3 / jnp.linalg.norm(stiffness)
    assert float(sched.get_dt(state)) == pytest.approx(float(expected), rel=1e-10)


@pytest.mark.parametrize("prior_variance", [None, 0.5], ids=["flat", "prior"])
def test_eks_stable_scheduler_stiffness_has_the_spectrum_of_cov_times_hessian(
    prior_variance,
):
    """``‖S‖²_F = tr((Cᶿᶿ H)²)`` for a linear model, the Euler stiffness."""
    state, _, cov_hessian = _stiffness_state(prior_variance=prior_variance)
    sched = flx.EKSStableScheduler(target_stiffness=1.0, **_ONLY_STIFFNESS)
    expected = 1.0 / jnp.sqrt(jnp.trace(cov_hessian @ cov_hessian))
    assert float(sched.get_dt(state)) == pytest.approx(float(expected), rel=1e-10)


@pytest.mark.parametrize(("gamma", "scale"), [(1e-4, 1.0), (100.0, 1.0), (0.1, 30.0)])
def test_eks_stable_scheduler_settles_at_the_same_step_at_any_scale(gamma, scale):
    """An ensemble with ``Cᶿᶿ = H⁻¹`` has ``‖S‖_F = √Nₚ`` whatever the units."""
    num_members = 40
    F = jnp.asarray([[1.0, 0.5], [-0.3, 1.2], [0.7, 0.1]]) / scale
    H = F.T @ F / gamma + jnp.eye(2) / scale**2
    # Rows with exactly identity sample covariance, coloured by H^{-1/2}.
    raw = jr.normal(jr.PRNGKey(2), (num_members, 2))
    raw = raw - raw.mean(0)
    whitened = jnp.linalg.qr(raw)[0] * jnp.sqrt(num_members - 1.0)
    particles = whitened @ jnp.linalg.cholesky(jnp.linalg.inv(H)).T
    state = flx.ProcessState(
        particles=particles,
        forward_evals=particles @ F.T,
        obs=jnp.zeros(3),
        noise_cov=lx.DiagonalLinearOperator(gamma * jnp.ones(3)),
        step=jnp.asarray(0, dtype=jnp.int32),
        algo_time=jnp.asarray(0.0),
        prior_cov=lx.MatrixLinearOperator(
            scale**2 * jnp.eye(2), lx.positive_semidefinite_tag
        ),
    )
    sched = flx.EKSStableScheduler(**_ONLY_STIFFNESS)
    assert float(sched.get_dt(state)) == pytest.approx(0.1 / jnp.sqrt(2.0), rel=1e-8)


@pytest.mark.slow
def test_eks_stable_scheduler_gradient_is_finite_at_zero_stiffness():
    particles = jr.normal(jr.PRNGKey(3), (10, 2))
    noise_cov = lx.DiagonalLinearOperator(jnp.ones(3))
    sched = flx.EKSStableScheduler()

    def dt_of(evals):
        return sched.get_dt(_state_with_evals(particles, evals, noise_cov))

    assert bool(jnp.all(jnp.isfinite(jax.grad(dt_of)(jnp.zeros((10, 3))))))

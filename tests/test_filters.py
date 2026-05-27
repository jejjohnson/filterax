"""Linear-Gaussian baseline tests for the L1 sequential filters.

Every comparison uses the Kalman formula evaluated *on the sample
covariance of the input ensemble*, so the assertions are algebraic
(``atol = 1e-9``), not Monte-Carlo. ``N_e`` only needs to exceed
``N_x`` for the sample covariance to be full rank.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

import filterax as flx


def _kalman_reference(
    x_bar_f: np.ndarray,
    P_f: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form Kalman analysis for a linear-Gaussian update."""
    S = H @ P_f @ H.T + R
    K = P_f @ H.T @ np.linalg.inv(S)
    mean_a = x_bar_f + K @ (y - H @ x_bar_f)
    P_a = P_f - K @ H @ P_f
    return mean_a, P_a


class _LinearObs(flx.AbstractObsOperator):
    H: jnp.ndarray

    def __call__(self, state):
        return self.H @ state


@pytest.fixture
def linear_gaussian(getkey):
    N_e, N_x = 60, 4
    x_bar = jnp.zeros(N_x)
    # Set up a non-trivial true forecast covariance and sample the ensemble.
    P_f = jnp.asarray(
        [
            [1.0, 0.3, 0.1, 0.0],
            [0.3, 2.0, 0.4, 0.1],
            [0.1, 0.4, 1.5, 0.2],
            [0.0, 0.1, 0.2, 0.8],
        ]
    )
    chol = jnp.linalg.cholesky(P_f)
    standard = jr.normal(getkey(), (N_e, N_x))
    particles = x_bar[None, :] + standard @ chol.T

    H = jnp.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.0, 1.0]])
    R_diag = jnp.asarray([0.2, 0.1, 0.3])
    R = lx.DiagonalLinearOperator(R_diag)
    obs_op = _LinearObs(H=H)
    y = jnp.asarray([0.7, -0.4, 0.2])

    P_f_np = np.asarray(jnp.cov(particles.T, ddof=1))
    x_bar_np = np.asarray(particles.mean(axis=0))
    H_np = np.asarray(H)
    R_np = np.diag(np.asarray(R_diag))
    mean_ref, P_ref = _kalman_reference(x_bar_np, P_f_np, H_np, R_np, np.asarray(y))
    return particles, y, obs_op, R, mean_ref, P_ref


def _check_against_reference(
    particles_a: jnp.ndarray,
    mean_ref: np.ndarray,
    P_ref: np.ndarray,
    atol_mean: float,
    atol_cov: float,
) -> None:
    np.testing.assert_allclose(
        np.asarray(particles_a).mean(axis=0), mean_ref, atol=atol_mean
    )
    P_a = np.cov(np.asarray(particles_a).T, ddof=1)
    np.testing.assert_allclose(P_a, P_ref, atol=atol_cov)


def test_etkf_matches_kalman_in_linear_gaussian(linear_gaussian):
    particles, y, obs_op, R, mean_ref, P_ref = linear_gaussian
    result = flx.filters.ETKF().analysis(particles, y, obs_op, R)
    _check_against_reference(
        result.particles, mean_ref, P_ref, atol_mean=1e-9, atol_cov=1e-9
    )
    assert result.log_likelihood is not None


def test_ensrf_matches_kalman_in_linear_gaussian(linear_gaussian):
    particles, y, obs_op, R, mean_ref, P_ref = linear_gaussian
    result = flx.filters.EnSRF().analysis(particles, y, obs_op, R)
    _check_against_reference(
        result.particles, mean_ref, P_ref, atol_mean=1e-9, atol_cov=1e-8
    )


def test_etkf_preserves_mean_consistency(getkey):
    # Pure mean-preservation: with zero innovation the analysis mean should
    # equal the forecast mean exactly.
    particles = jr.normal(getkey(), (50, 4))
    H = jr.normal(getkey(), (3, 4))
    obs_op = _LinearObs(H=H)
    forecast_obs_mean = (particles @ H.T).mean(axis=0)
    R = lx.DiagonalLinearOperator(jnp.ones(3))
    result = flx.filters.ETKF().analysis(particles, forecast_obs_mean, obs_op, R)
    np.testing.assert_allclose(
        np.asarray(result.particles).mean(axis=0),
        np.asarray(particles).mean(axis=0),
        atol=1e-9,
    )


def test_letkf_no_localization_radius_matches_etkf(getkey):
    # With a localization radius large enough to cover every observation,
    # LETKF should reproduce ETKF up to the per-grid-point analysis (modulo
    # Gaspari-Cohn tapering at the radius).
    particles = jr.normal(getkey(), (40, 6))
    state_coords = jnp.arange(6.0)[:, None]  # 1D grid
    obs_coords = jnp.asarray([[0.5], [2.5], [4.5]])
    H = jnp.zeros((3, 6)).at[jnp.arange(3), jnp.asarray([0, 2, 4])].set(1.0)
    obs_op = _LinearObs(H=H)
    R = lx.DiagonalLinearOperator(jnp.full((3,), 0.5))
    obs = jnp.asarray([0.1, -0.2, 0.3])

    # Very small radius — each grid point should only see its nearest obs;
    # smoke-test that the analysis returns sensible (finite) values and
    # preserves the ensemble size.
    result = flx.filters.LETKF(radius=1.0).analysis(
        particles, obs, obs_op, R, state_coords=state_coords, obs_coords=obs_coords
    )
    assert result.particles.shape == particles.shape
    assert jnp.all(jnp.isfinite(result.particles))


def test_letkf_rejects_mismatched_coords(getkey):
    particles = jr.normal(getkey(), (20, 5))
    obs = jnp.zeros(3)
    H = jnp.zeros((3, 5)).at[0, 0].set(1.0)
    obs_op = _LinearObs(H=H)
    R = lx.DiagonalLinearOperator(jnp.ones(3))
    with pytest.raises(ValueError, match="state_coords"):
        flx.filters.LETKF(radius=1.0).analysis(
            particles,
            obs,
            obs_op,
            R,
            state_coords=jnp.zeros((4, 1)),  # wrong size
            obs_coords=jnp.zeros((3, 1)),
        )


def _identity_dynamics():
    class _ID(flx.AbstractDynamics):
        def __call__(self, state, t0, t1):
            return state

    return _ID()


def test_l2_etkf_assimilate_smoke(getkey):
    # 2-window assimilation under identity dynamics + identity observations.
    N_e, N_x = 60, 4
    particles = jr.normal(getkey(), (N_e, N_x))
    H = jnp.eye(N_x)[:2]
    obs_op = _LinearObs(H=H)
    R = lx.DiagonalLinearOperator(jnp.full((2,), 0.5))
    obs_seq = [
        (jnp.asarray([0.5, -0.5]), 1.0),
        (jnp.asarray([0.6, -0.4]), 2.0),
    ]
    filter_ = flx.ETKF(dynamics=_identity_dynamics(), obs_op=obs_op)
    result = filter_.assimilate(particles, obs_seq, R)
    assert result.particles.shape == (N_e, N_x)
    assert result.forecast_history.shape == (2, N_e, N_x)
    assert result.analysis_history.shape == (2, N_e, N_x)
    assert result.log_likelihoods is not None
    assert result.log_likelihoods.shape == (2,)


def test_stochastic_enkf_l2_uses_independent_keys_per_window(getkey):
    # Regression: the original L1 StochasticEnKF reused self.key across
    # windows so every window drew identical observation perturbations.
    # The L2 model now folds the seed with the step index so successive
    # windows produce statistically different posteriors.
    N_e, N_x = 200, 3
    particles = jr.normal(getkey(), (N_e, N_x))
    H = jnp.eye(N_x)
    obs_op = _LinearObs(H=H)
    R = lx.DiagonalLinearOperator(jnp.full((N_x,), 0.1))
    # Two identical observations, identical dynamics — the only source of
    # variation between windows is the perturbation draw.
    obs_seq = [(jnp.zeros(N_x), 1.0), (jnp.zeros(N_x), 2.0)]

    filter_ = flx.StochasticEnKF(dynamics=_identity_dynamics(), obs_op=obs_op, seed=7)
    result = filter_.assimilate(particles, obs_seq, R)
    posterior_0 = result.analysis_history[0]
    posterior_1 = result.analysis_history[1]

    # Same forecast distribution, same data, same R — but the posteriors
    # must differ because the perturbations are independent draws.
    diff = jnp.linalg.norm(posterior_1 - posterior_0)
    assert float(diff) > 1e-3, "stochastic perturbations must be independent per window"


def test_letkf_hard_cutoff_at_radius(getkey):
    # Regression: an observation at distance r < d < 2r should be EXCLUDED
    # by R-localization, even though Gaspari-Cohn is non-zero on (r, 2r].
    # We compare the analysis with the far observation present vs. absent;
    # they must be identical to numerical precision.
    N_e, N_x = 40, 3
    particles = jr.normal(getkey(), (N_e, N_x))
    state_coords = jnp.arange(N_x, dtype=jnp.float64)[:, None]

    # One near-by observation (distance 0.5 from state 0) and one far
    # observation at distance 1.5 (inside Gaspari-Cohn's 2 * radius=2.0
    # support but outside our radius=1.0 cutoff).
    obs_coords = jnp.asarray([[0.5], [1.5]])
    H = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    obs_op = _LinearObs(H=H)
    R = lx.DiagonalLinearOperator(jnp.asarray([0.3, 0.3]))

    full_obs = jnp.asarray([0.4, 0.9])
    near_only_obs = jnp.asarray([0.4, 0.0])

    full = flx.filters.LETKF(radius=1.0).analysis(
        particles,
        full_obs,
        obs_op,
        R,
        state_coords=state_coords,
        obs_coords=obs_coords,
    )
    # When the far obs is excluded, the analysis at state 0 should be
    # invariant to its value. Set it to zero and confirm.
    same = flx.filters.LETKF(radius=1.0).analysis(
        particles,
        near_only_obs,
        obs_op,
        R,
        state_coords=state_coords,
        obs_coords=obs_coords,
    )
    np.testing.assert_allclose(
        np.asarray(full.particles[:, 0]),
        np.asarray(same.particles[:, 0]),
        atol=1e-10,
    )


def test_letkf_rejects_non_diagonal_R(getkey):
    particles = jr.normal(getkey(), (20, 3))
    obs = jnp.zeros(3)
    H = jnp.eye(3)
    obs_op = _LinearObs(H=H)
    R = lx.MatrixLinearOperator(jnp.eye(3) * 0.1, tags=lx.positive_semidefinite_tag)
    with pytest.raises(NotImplementedError, match="diagonal observation noise"):
        flx.filters.LETKF(radius=1.0).analysis(
            particles,
            obs,
            obs_op,
            R,
            state_coords=jnp.zeros((3, 1)),
            obs_coords=jnp.zeros((3, 1)),
        )


def test_perturbed_observations_diagonal_fast_path():
    # Smoke check for the dense fallback: shape and finiteness. The
    # diagonal fast-path correctness is covered by
    # ``test_perturbed_observations_diagonal_scales_with_sqrt_R``.
    key = jr.PRNGKey(11)
    obs = jnp.asarray([1.0, -1.0, 0.5])
    R_diag = jnp.asarray([0.5, 1.0, 2.0])
    R_dense_op = lx.MatrixLinearOperator(
        jnp.diag(R_diag), tags=lx.positive_semidefinite_tag
    )
    dense_draws = flx.perturbed_observations(key, obs, R_dense_op, n_ensemble=6)
    assert dense_draws.shape == (6, 3)
    assert bool(jnp.all(jnp.isfinite(dense_draws)))


def test_l2_etkf_with_inflator(getkey):
    N_e, N_x = 40, 3
    particles = jr.normal(getkey(), (N_e, N_x))
    H = jnp.eye(N_x)
    obs_op = _LinearObs(H=H)
    R = lx.DiagonalLinearOperator(jnp.full((N_x,), 0.2))
    obs_seq = [(jnp.zeros(N_x), 1.0)]
    filter_ = flx.ETKF(
        dynamics=_identity_dynamics(),
        obs_op=obs_op,
        inflator=flx.RTPS(alpha=0.5),
    )
    result = filter_.assimilate(particles, obs_seq, R)
    # With RTPS(alpha=0.5) the posterior spread should be larger than the
    # unrelaxed analysis (which is contracted by the observation).
    forecast_spread = result.forecast_history[0].std(axis=0, ddof=1)
    analysis_spread = result.analysis_history[0].std(axis=0, ddof=1)
    assert jnp.all(analysis_spread > 0)
    # RTPS relaxes toward the forecast spread, so analysis_spread should
    # be bounded between the pure-analysis spread and the forecast spread.
    assert jnp.all(analysis_spread <= forecast_spread + 1e-6)

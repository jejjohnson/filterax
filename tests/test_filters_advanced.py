"""Tests for the Wave 4 advanced sequential filter variants."""

from __future__ import annotations

import warnings

import gaussx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

import filterax as flx


# ──────────────────────────────────────────────────────────────────────
# Linear-Gaussian reference + ensemble setup shared by all three new
# filters. Tight tolerances match the existing Wave 2 baselines.
# ──────────────────────────────────────────────────────────────────────


class _LinearObs(flx.AbstractObsOperator):
    H: jnp.ndarray

    def __call__(self, state):
        return self.H @ state


@pytest.fixture
def linear_gaussian():
    # N_e only has to exceed N_x for the sample covariance to be full
    # rank; the references are evaluated on the sample stats, so the
    # comparison is algebraic (atol = 1e-9), not Monte Carlo.
    N_e, N_x = 60, 4
    P_f = jnp.asarray(
        [
            [1.0, 0.3, 0.1, 0.0],
            [0.3, 2.0, 0.4, 0.1],
            [0.1, 0.4, 1.5, 0.2],
            [0.0, 0.1, 0.2, 0.8],
        ]
    )
    chol = jnp.linalg.cholesky(P_f)
    particles = jr.normal(jr.PRNGKey(0), (N_e, N_x)) @ chol.T

    H = jnp.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.0, 1.0]])
    R_diag = jnp.asarray([0.2, 0.1, 0.3])
    R = lx.DiagonalLinearOperator(R_diag)
    obs_op = _LinearObs(H=H)
    y = jnp.asarray([0.7, -0.4, 0.2])

    P_f_np = np.asarray(jnp.cov(particles.T, ddof=1))
    x_bar_np = np.asarray(particles.mean(axis=0))
    H_np = np.asarray(H)
    R_np = np.diag(np.asarray(R_diag))
    S = H_np @ P_f_np @ H_np.T + R_np
    K = P_f_np @ H_np.T @ np.linalg.inv(S)
    mean_ref = x_bar_np + K @ (np.asarray(y) - H_np @ x_bar_np)
    P_ref = P_f_np - K @ H_np @ P_f_np
    return particles, y, obs_op, R, mean_ref, P_ref


def _check_against_kalman(particles_a, mean_ref, P_ref, atol_mean, atol_cov):
    arr = np.asarray(particles_a)
    np.testing.assert_allclose(arr.mean(axis=0), mean_ref, atol=atol_mean)
    np.testing.assert_allclose(np.cov(arr.T, ddof=1), P_ref, atol=atol_cov)


# ──────────────────────────────────────────────────────────────────────
# ETKF_Livings
# ──────────────────────────────────────────────────────────────────────


def test_etkf_livings_matches_kalman_in_linear_gaussian(linear_gaussian):
    particles, y, obs_op, R, mean_ref, P_ref = linear_gaussian
    result = flx.filters.ETKF_Livings(key=jr.PRNGKey(0)).analysis(
        particles, y, obs_op, R
    )
    _check_against_kalman(
        result.particles, mean_ref, P_ref, atol_mean=1e-9, atol_cov=1e-9
    )


def test_etkf_livings_different_keys_yield_different_ensembles(linear_gaussian):
    particles, y, obs_op, R, *_ = linear_gaussian
    a = flx.filters.ETKF_Livings(key=jr.PRNGKey(1)).analysis(particles, y, obs_op, R)
    b = flx.filters.ETKF_Livings(key=jr.PRNGKey(2)).analysis(particles, y, obs_op, R)
    # Same posterior covariance (the rotation is mean-preserving and
    # cov-preserving) but different ensemble realisations.
    arr_a = np.asarray(a.particles)
    arr_b = np.asarray(b.particles)
    np.testing.assert_allclose(
        np.cov(arr_a.T, ddof=1), np.cov(arr_b.T, ddof=1), atol=1e-9
    )
    assert np.linalg.norm(arr_a - arr_b) > 1e-3


# ──────────────────────────────────────────────────────────────────────
# EnSRF_Serial
# ──────────────────────────────────────────────────────────────────────


def test_ensrf_serial_matches_kalman_in_linear_gaussian(linear_gaussian):
    particles, y, obs_op, R, mean_ref, P_ref = linear_gaussian
    result = flx.filters.EnSRF_Serial().analysis(particles, y, obs_op, R)
    _check_against_kalman(
        result.particles, mean_ref, P_ref, atol_mean=1e-8, atol_cov=1e-7
    )


def test_ensrf_serial_rejects_non_diagonal_R(linear_gaussian):
    particles, y, obs_op, _, *_ = linear_gaussian
    R_dense = lx.MatrixLinearOperator(
        0.1 * jnp.eye(3), tags=lx.positive_semidefinite_tag
    )
    with pytest.raises(NotImplementedError, match="diagonal observation noise"):
        flx.filters.EnSRF_Serial().analysis(particles, y, obs_op, R_dense)


# ──────────────────────────────────────────────────────────────────────
# ESTKF
# ──────────────────────────────────────────────────────────────────────


def test_estkf_matches_kalman_in_linear_gaussian(linear_gaussian):
    particles, y, obs_op, R, mean_ref, P_ref = linear_gaussian
    result = flx.filters.ESTKF().analysis(particles, y, obs_op, R)
    _check_against_kalman(
        result.particles, mean_ref, P_ref, atol_mean=1e-9, atol_cov=1e-9
    )


def test_estkf_preserves_ensemble_mean_consistency():
    # Pure mean-preservation: with zero innovation, the analysis mean
    # should equal the forecast mean exactly.
    particles = jr.normal(jr.PRNGKey(7), (50, 4))
    H = jr.normal(jr.PRNGKey(8), (3, 4))
    obs_op = _LinearObs(H=H)
    forecast_obs_mean = (particles @ H.T).mean(axis=0)
    R = lx.DiagonalLinearOperator(jnp.ones(3))
    result = flx.filters.ESTKF().analysis(particles, forecast_obs_mean, obs_op, R)
    np.testing.assert_allclose(
        np.asarray(result.particles).mean(axis=0),
        np.asarray(particles).mean(axis=0),
        atol=1e-9,
    )


# ──────────────────────────────────────────────────────────────────────
# SquareRootKF (parametric)
# ──────────────────────────────────────────────────────────────────────


def test_square_root_kf_tracks_random_walk():
    """Parametric KF on a 2-D random walk with one observed component.

    Algebraic invariants (independent of trajectory realisation):
    shape, finite log-likelihood, PSD filtered covariances, and the
    structural fact that the *unobserved* component's posterior
    variance must exceed the *observed* one's at every step — the
    filter never sees ``x[1]``, so it cannot shrink its uncertainty
    there while ``x[0]`` is constrained by ``H``.
    """
    N, M, T = 2, 1, 4
    F = jnp.eye(N)
    H = jnp.array([[1.0, 0.0]])
    Q = jnp.eye(N) * 0.01
    R = jnp.array([[0.05]])
    x_true = jnp.cumsum(0.1 * jr.normal(jr.PRNGKey(0), (T, N)), axis=0)
    y = x_true @ H.T + jnp.sqrt(0.05) * jr.normal(jr.PRNGKey(1), (T, M))

    kf = flx.filters.SquareRootKF(
        transition=F, obs_model=H, process_noise=Q, obs_noise=R
    )
    result = kf.filter(y, init_mean=jnp.zeros(N), init_cov=jnp.eye(N))

    assert result.filtered_means.shape == (T, N)
    assert result.filtered_covs.shape == (T, N, N)
    assert bool(jnp.isfinite(result.log_likelihood))
    eigs = jnp.linalg.eigvalsh(result.filtered_covs)
    assert float(eigs.min()) > -1e-9
    # Observed component is constrained by H every step; the unobserved
    # one only sees the process noise — so var_unobs > var_obs at every t.
    var_obs = result.filtered_covs[:, 0, 0]
    var_unobs = result.filtered_covs[:, 1, 1]
    assert bool(jnp.all(var_unobs > var_obs))


def _square_root_model(dtype=None):
    transition = jnp.asarray([[1.0, 0.1, 0.0], [0.0, 0.9, 0.1], [0.05, 0.0, 0.95]])
    obs_model = jnp.asarray([[1.0, 0.0, 0.5], [0.0, 1.0, -0.3]])
    process_noise = 0.1 * jnp.eye(3) + 0.02
    obs_noise = 0.2 * jnp.eye(2)
    observations = jr.normal(jr.PRNGKey(3), (12, 2))
    return tuple(
        jnp.asarray(array, dtype)
        for array in (transition, obs_model, process_noise, obs_noise, observations)
    )


def test_square_root_kf_matches_the_covariance_form():
    transition, obs_model, process_noise, obs_noise, observations = _square_root_model()
    init_mean, init_cov = jnp.zeros(3), jnp.eye(3)
    expected = gaussx.parallel_kalman_filter(
        transition,
        obs_model,
        process_noise,
        obs_noise,
        observations,
        init_mean,
        init_cov,
    )
    result = flx.filters.SquareRootKF(
        transition=transition,
        obs_model=obs_model,
        process_noise=process_noise,
        obs_noise=obs_noise,
    ).filter(observations, init_mean, init_cov)
    for name in (
        "filtered_means",
        "filtered_covs",
        "predicted_means",
        "predicted_covs",
        "log_likelihood",
    ):
        np.testing.assert_allclose(
            getattr(result, name), getattr(expected, name), atol=1e-10
        )


def test_square_root_kf_raises_no_gaussx_deprecation():
    transition, obs_model, process_noise, obs_noise, observations = _square_root_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        flx.filters.SquareRootKF(
            transition=transition,
            obs_model=obs_model,
            process_noise=process_noise,
            obs_noise=obs_noise,
        ).filter(observations, jnp.zeros(3), jnp.eye(3))


def _strong_observation_model(process_var, obs_var, prior_var, dtype):
    """A constant-velocity model with data drawn from the model itself."""
    num_steps = 64
    transition = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    obs_model = jnp.array([[1.0, 0.0]])
    state_key, noise_key, obs_key = jr.split(jr.PRNGKey(0), 3)
    state = jnp.sqrt(prior_var) * jr.normal(state_key, (2,))
    observations = []
    for step in range(num_steps):
        process_draw = jr.normal(jr.fold_in(noise_key, step), (2,))
        state = transition @ state + jnp.sqrt(process_var) * process_draw
        obs_draw = jr.normal(jr.fold_in(obs_key, step), (1,))
        observations.append(obs_model @ state + jnp.sqrt(obs_var) * obs_draw)
    return flx.filters.SquareRootKF(
        transition=transition.astype(dtype),
        obs_model=obs_model.astype(dtype),
        process_noise=(process_var * jnp.eye(2)).astype(dtype),
        obs_noise=(obs_var * jnp.eye(1)).astype(dtype),
    ).filter(
        jnp.stack(observations).astype(dtype),
        jnp.zeros(2, dtype),
        (prior_var * jnp.eye(2)).astype(dtype),
    )


@pytest.mark.parametrize(
    ("process_var", "obs_var", "prior_var", "max_mean_error", "max_loglik_error"),
    [
        (1e-8, 1e-6, 1.0, 0.1, 0.1),
        (1e-10, 1e-8, 1.0, 1.0, 0.1),
        (1e-12, 1e-10, 100.0, 10.0, 2.0),
    ],
)
def test_square_root_kf_float32_matches_a_float64_reference(
    process_var, obs_var, prior_var, max_mean_error, max_loglik_error
):
    """Means and log-likelihood, not only a PSD covariance.

    Before gaussx v0.6.9 the means here were up to billions of standard
    deviations off, while the covariances still looked healthy (#141).
    """
    reference = _strong_observation_model(process_var, obs_var, prior_var, jnp.float64)
    single = _strong_observation_model(process_var, obs_var, prior_var, jnp.float32)
    assert single.filtered_means.dtype == jnp.float32

    variances = jnp.stack([reference.filtered_covs[:, i, i] for i in range(2)], -1)
    mean_error = jnp.abs(single.filtered_means - reference.filtered_means)
    assert jnp.max(mean_error / jnp.sqrt(variances)) < max_mean_error
    assert jnp.abs(single.log_likelihood - reference.log_likelihood) < max_loglik_error
    # The covariance form returns an indefinite covariance at the last of these.
    assert jnp.min(jnp.linalg.eigvalsh(single.filtered_covs)) > 0.0


def test_square_root_kf_is_accurate_in_float32_on_a_well_conditioned_model():
    """Close agreement at ordinary conditioning, beyond the strong cases above."""
    transition, obs_model, process_noise, obs_noise, observations = _square_root_model()

    def run(dtype):
        return flx.filters.SquareRootKF(
            transition=jnp.asarray(transition, dtype),
            obs_model=jnp.asarray(obs_model, dtype),
            process_noise=jnp.asarray(process_noise, dtype),
            obs_noise=jnp.asarray(obs_noise, dtype),
        ).filter(
            jnp.asarray(observations, dtype),
            jnp.zeros(3, dtype),
            jnp.eye(3, dtype=dtype),
        )

    single, double = run(jnp.float32), run(jnp.float64)
    np.testing.assert_allclose(single.filtered_means, double.filtered_means, atol=1e-4)
    np.testing.assert_allclose(single.filtered_covs, double.filtered_covs, atol=1e-4)
    np.testing.assert_allclose(single.log_likelihood, double.log_likelihood, atol=1e-3)

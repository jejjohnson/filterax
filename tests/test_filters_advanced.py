"""Tests for the Wave 4 advanced sequential filter variants."""

from __future__ import annotations

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
    N_e, N_x = 2000, 4
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
    """Parametric KF on a 2-D random walk with scalar observations.

    Asserts shape + finiteness + that the observed component of the
    filtered mean is closer to truth than the *unobserved* component
    (the hidden state) is — i.e. observations actually constrain x[0].
    """
    N, M, T = 2, 1, 30
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
    # The observed component should be more accurate than the hidden
    # component (the filter sees x[0] every step, x[1] never).
    rmse_obs = float(
        jnp.sqrt(jnp.mean((result.filtered_means[:, 0] - x_true[:, 0]) ** 2))
    )
    rmse_hidden = float(
        jnp.sqrt(jnp.mean((result.filtered_means[:, 1] - x_true[:, 1]) ** 2))
    )
    assert rmse_obs < rmse_hidden
    # Marginal log-likelihood is finite.
    assert bool(jnp.isfinite(result.log_likelihood))
    # Filtered covariances are PSD (Cholesky-form propagation guards this).
    for t in range(T):
        eigs = jnp.linalg.eigvalsh(result.filtered_covs[t])
        assert float(eigs.min()) > -1e-9

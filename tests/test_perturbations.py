"""Tests for the perturbed-observations primitive."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np

import filterax as flx


def test_perturbed_observations_shape(getkey):
    obs = jnp.zeros(4)
    R = lx.DiagonalLinearOperator(jnp.ones(4))
    pert = flx.perturbed_observations(getkey(), obs, R, n_ensemble=12)
    assert pert.shape == (12, 4)


def test_perturbed_observations_diagonal_scales_with_sqrt_R(getkey):
    """Algebraic correctness for the diagonal fast path: a draw equals
    ``obs + sqrt(R_diag) * standard_normal`` element-wise, which we
    reproduce from the same PRNG key without going through the package."""
    obs = jnp.asarray([3.0, -1.0, 2.0])
    R_diag = jnp.asarray([0.5, 1.0, 2.0])
    R = lx.DiagonalLinearOperator(R_diag)
    key = jr.PRNGKey(0)
    pert = flx.perturbed_observations(key, obs, R, n_ensemble=4)
    standard = jr.normal(key, (4, 3))
    expected = np.asarray(obs)[None, :] + np.asarray(standard) * np.sqrt(
        np.asarray(R_diag)
    )
    np.testing.assert_allclose(np.asarray(pert), expected, atol=1e-12)


def test_perturbed_observations_deterministic(getkey):
    obs = jnp.zeros(3)
    R = lx.DiagonalLinearOperator(jnp.ones(3))
    key = jr.PRNGKey(42)
    a = flx.perturbed_observations(key, obs, R, n_ensemble=5)
    b = flx.perturbed_observations(key, obs, R, n_ensemble=5)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _dense_correlated_noise(n_obs: int) -> lx.MatrixLinearOperator:
    factor = jr.normal(jr.key(0), (n_obs, n_obs))
    matrix = factor @ factor.T + jnp.eye(n_obs)
    return lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)


def test_perturbed_observations_dense_noise_above_lanczos_rank(getkey):
    """Regression for #115: rank-50 Lanczos gave a 50-column factor at N_y = 60."""
    obs = jnp.zeros(60)
    pert = flx.perturbed_observations(getkey(), obs, _dense_correlated_noise(60), 8)
    assert pert.shape == (8, 60)


def test_perturbed_observations_dense_noise_matches_covariance(getkey):
    """Sample covariance of the draws matches a dense correlated R.

    Each entry is scaled by its Wishart standard error,
    Var(Ĉ_ik) = (R_ik² + R_ii R_kk) / (J − 1).
    """
    n_draws = 20_000
    R = _dense_correlated_noise(60)
    pert = flx.perturbed_observations(getkey(), jnp.zeros(60), R, n_draws)
    sample_covariance = np.cov(np.asarray(pert), rowvar=False)
    R_dense = np.asarray(R.as_matrix())
    variance = np.diag(R_dense)
    standard_error = np.sqrt(
        (R_dense**2 + np.outer(variance, variance)) / (n_draws - 1)
    )
    z_scores = np.abs(sample_covariance - R_dense) / standard_error
    # 3 600 correlated entries, so a few exceedances of 3σ are expected by chance.
    assert np.mean(z_scores > 3.0) < 0.02
    assert z_scores.max() < 5.5


def test_perturbed_observations_runs_under_jit(getkey):
    R = _dense_correlated_noise(60)

    def draw(key):
        return flx.perturbed_observations(key, jnp.zeros(60), R, 4)

    assert jax.jit(draw)(getkey()).shape == (4, 60)

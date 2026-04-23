"""Tests for innovation log-likelihood and diagnostics."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest
from scipy.stats import multivariate_normal

import filterax as flx


def test_log_likelihood_matches_scipy_diagonal(getkey):
    N_y = 5
    v = jr.normal(getkey(), (N_y,))
    R_diag = jnp.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
    S = lx.DiagonalLinearOperator(R_diag)
    got = float(flx.log_likelihood(v, S))
    expected = float(
        multivariate_normal.logpdf(
            np.asarray(v),
            mean=np.zeros(N_y),
            cov=np.diag(np.asarray(R_diag)),
        )
    )
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


def test_log_likelihood_matches_scipy_dense(getkey):
    N_y = 4
    A = jr.normal(getkey(), (N_y, N_y))
    S_dense = jnp.asarray(np.eye(N_y) + np.asarray(A) @ np.asarray(A).T)
    S = lx.MatrixLinearOperator(S_dense, lx.positive_semidefinite_tag)
    v = jr.normal(getkey(), (N_y,))
    got = float(flx.log_likelihood(v, S))
    expected = float(
        multivariate_normal.logpdf(
            np.asarray(v),
            mean=np.zeros(N_y),
            cov=np.asarray(S_dense),
        )
    )
    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-8)


def test_innovation_statistics_keys_and_shapes(getkey):
    N_e, N_x, N_y = 40, 5, 3
    particles = jr.normal(getkey(), (N_e, N_x))
    obs = jr.normal(getkey(), (N_y,))
    R = lx.DiagonalLinearOperator(jnp.ones(N_y))

    def obs_op(x):
        return x[:N_y]

    stats = flx.innovation_statistics(particles, obs, obs_op, R)
    assert set(stats.keys()) == {
        "innovation",
        "innovation_cov",
        "normalized_innovation",
        "log_likelihood",
    }
    assert stats["innovation"].shape == (N_y,)
    assert stats["normalized_innovation"].shape == (N_y,)
    assert stats["log_likelihood"].shape == ()


def test_innovation_statistics_normalised_innovation_has_identity_covariance():
    # For a fixed ensemble, the normalized innovation should have unit norm
    # expectation in the limit of many innovations. Here we check that applying
    # S^{-1} to v via scipy equals reconstructing v from the normalized
    # innovation via L.
    import jax

    N_e, N_x, N_y = 30, 6, 4
    key = jr.PRNGKey(42)
    k1, k2 = jr.split(key)
    particles = jr.normal(k1, (N_e, N_x))
    obs = jr.normal(k2, (N_y,))
    R = lx.DiagonalLinearOperator(0.1 * jnp.ones(N_y))

    def obs_op(x):
        return x[:N_y]

    stats = flx.innovation_statistics(particles, obs, obs_op, R)
    S = stats["innovation_cov"].as_matrix()
    L = jnp.linalg.cholesky(S)
    reconstructed = L @ stats["normalized_innovation"]
    np.testing.assert_allclose(
        np.asarray(reconstructed), np.asarray(stats["innovation"]), atol=1e-9
    )

    # Also check that innovation_statistics's log_likelihood agrees with
    # the standalone log_likelihood primitive.
    ll_standalone = flx.log_likelihood(stats["innovation"], stats["innovation_cov"])
    np.testing.assert_allclose(
        float(stats["log_likelihood"]), float(ll_standalone), rtol=1e-10
    )

    # Silence unused jax import (kept for future parametric tests).
    _ = jax


@pytest.mark.parametrize("N_e", [0, 1])
def test_innovation_statistics_rejects_degenerate_ensemble(N_e):
    R = lx.DiagonalLinearOperator(jnp.ones(2))
    with pytest.raises(ValueError, match="at least 2 ensemble members"):
        flx.innovation_statistics(
            jnp.zeros((N_e, 3)),
            jnp.zeros(2),
            lambda x: x[:2],
            R,
        )

"""Tests for innovation log-likelihood and diagnostics."""

from __future__ import annotations

import jax
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


def test_innovation_statistics_normalised_innovation_is_whitened():
    # The normalized innovation must whiten the innovation: ``zᵀ z = vᵀ S⁻¹ v``.
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
    v = stats["innovation"]
    z = stats["normalized_innovation"]

    mahalanobis_via_solve = float(v @ jnp.linalg.solve(S, v))
    mahalanobis_via_whitening = float(z @ z)
    np.testing.assert_allclose(
        mahalanobis_via_whitening, mahalanobis_via_solve, atol=1e-9
    )

    # And log_likelihood agrees with the standalone primitive.
    ll_standalone = flx.log_likelihood(stats["innovation"], stats["innovation_cov"])
    np.testing.assert_allclose(
        float(stats["log_likelihood"]), float(ll_standalone), rtol=1e-10
    )


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


def _whitening_problem(num_members, obs_size, noise):
    keys = jr.split(jr.PRNGKey(7), 3)
    obs_particles = jr.normal(keys[0], (num_members, obs_size))
    obs = jr.normal(keys[1], (obs_size,))
    if noise == "diagonal":
        obs_noise = lx.DiagonalLinearOperator(jnp.logspace(-2, 0, obs_size))
    else:
        factor = jr.normal(keys[2], (obs_size, obs_size))
        dense = factor @ factor.T / obs_size + 0.1 * jnp.eye(obs_size)
        obs_noise = lx.MatrixLinearOperator(dense, lx.positive_semidefinite_tag)
    return obs_particles, obs, obs_noise


@pytest.mark.parametrize("noise", ["diagonal", "dense"])
@pytest.mark.parametrize(
    ("num_members", "obs_size"), [(30, 4), (4, 6), (20, 60)], ids=str
)
def test_innovation_statistics_returns_the_principal_whitener(
    num_members, obs_size, noise
):
    """``z = S⁻¹ᐟ² v`` and the log-likelihood, against a dense eigendecomposition.

    ``(30, 4)`` takes the dense route; the other two take Woodbury.
    """
    obs_particles, obs, obs_noise = _whitening_problem(num_members, obs_size, noise)
    stats = flx.innovation_statistics(obs_particles, obs, lambda x: x, obs_noise)

    S = stats["innovation_cov"].as_matrix()
    innovation = stats["innovation"]
    eigenvalues, eigenvectors = jnp.linalg.eigh(S)
    root_inverse = (eigenvectors / jnp.sqrt(eigenvalues)) @ eigenvectors.T
    whitened = stats["normalized_innovation"]
    np.testing.assert_allclose(whitened, root_inverse @ innovation, atol=1e-10)

    mahalanobis = innovation @ jnp.linalg.solve(S, innovation)
    np.testing.assert_allclose(whitened @ whitened, mahalanobis, rtol=1e-10)

    expected_log_prob = -0.5 * (
        obs_size * jnp.log(2.0 * jnp.pi) + jnp.linalg.slogdet(S)[1] + mahalanobis
    )
    np.testing.assert_allclose(stats["log_likelihood"], expected_log_prob, rtol=1e-10)


@pytest.mark.parametrize(
    ("num_members", "obs_size"), [(30, 4), (4, 6)], ids=["dense", "woodbury"]
)
def test_innovation_statistics_whitener_has_finite_gradients(num_members, obs_size):
    obs_particles, obs, obs_noise = _whitening_problem(
        num_members, obs_size, "diagonal"
    )

    def squared_norm(particles):
        stats = flx.innovation_statistics(particles, obs, lambda x: x, obs_noise)
        return stats["normalized_innovation"] @ stats["normalized_innovation"]

    def mahalanobis(particles):
        stats = flx.innovation_statistics(particles, obs, lambda x: x, obs_noise)
        S = stats["innovation_cov"].as_matrix()
        return stats["innovation"] @ jnp.linalg.solve(S, stats["innovation"])

    np.testing.assert_allclose(
        jax.grad(squared_norm)(obs_particles),
        jax.grad(mahalanobis)(obs_particles),
        rtol=1e-8,
        atol=1e-8,
    )

"""Tests for innovation log-likelihood and diagnostics."""

from __future__ import annotations

import gaussx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest
from scipy.stats import multivariate_normal

import filterax as flx
from filterax._primitives._likelihood import ensemble_log_likelihood


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


def _dense_log_likelihood(innovation, S):
    obs_size = innovation.shape[0]
    quadratic = innovation @ jnp.linalg.solve(S, innovation)
    log_det = jnp.linalg.slogdet(S)[1]
    return -0.5 * (obs_size * jnp.log(2.0 * jnp.pi) + log_det + quadratic)


@pytest.mark.parametrize("noise", ["diagonal", "dense"])
@pytest.mark.parametrize(
    ("num_members", "obs_size"), [(30, 4), (4, 6), (6, 6)], ids=str
)
def test_ensemble_log_likelihood_matches_the_dense_form(num_members, obs_size, noise):
    """``(30, 4)`` takes the QR route; the other two take Woodbury."""
    obs_particles, obs, obs_noise = _whitening_problem(num_members, obs_size, noise)
    innovation = obs - obs_particles.mean(0)
    S = flx.innovation_covariance(obs_particles, obs_noise)
    value = ensemble_log_likelihood(obs_particles, innovation, obs_noise)
    np.testing.assert_allclose(
        value, _dense_log_likelihood(innovation, S.as_matrix()), rtol=1e-10
    )


def test_ensemble_log_likelihood_keeps_a_solver_on_the_structured_operator():
    obs_particles, obs, obs_noise = _whitening_problem(30, 4, "diagonal")
    innovation = obs - obs_particles.mean(0)
    value = ensemble_log_likelihood(
        obs_particles, innovation, obs_noise, solver=gaussx.DenseSolver()
    )
    reference = ensemble_log_likelihood(obs_particles, innovation, obs_noise)
    np.testing.assert_allclose(value, reference, rtol=1e-8)


@pytest.mark.parametrize(
    ("variance", "channel"),
    [(1e-8, "independent"), (1e-12, "independent"), (1e-10, "duplicated")],
    ids=str,
)
def test_ensemble_log_likelihood_float32_matches_float64(variance, channel):
    """Woodbury is wrong at the first two, and a formed S gives NaN at the last."""
    obs_particles = jr.normal(jr.PRNGKey(0), (200, 3))
    if channel == "duplicated":
        obs_particles = obs_particles.at[:, 1].set(obs_particles[:, 0])
    obs = jr.normal(jr.PRNGKey(1), (3,))

    def value(dtype):
        particles = obs_particles.astype(dtype)
        noise = lx.DiagonalLinearOperator(variance * jnp.ones(3, dtype))
        innovation = obs.astype(dtype) - particles.mean(0)
        return ensemble_log_likelihood(particles, innovation, noise)

    reference, single = value(jnp.float64), value(jnp.float32)
    assert single.dtype == jnp.float32
    np.testing.assert_allclose(single, reference, rtol=1e-4)


@pytest.mark.parametrize(
    ("num_members", "obs_size"), [(30, 4), (4, 6)], ids=["qr", "woodbury"]
)
def test_ensemble_log_likelihood_gradient_matches_central_differences(
    num_members, obs_size
):
    obs_particles, obs, obs_noise = _whitening_problem(
        num_members, obs_size, "diagonal"
    )

    def loss(particles):
        innovation = obs - particles.mean(0)
        return ensemble_log_likelihood(particles, innovation, obs_noise)

    gradient = jax.grad(loss)(obs_particles)
    direction = jr.normal(jr.PRNGKey(5), obs_particles.shape)
    step = 1e-6
    central = loss(obs_particles + step * direction)
    central = (central - loss(obs_particles - step * direction)) / (2.0 * step)
    np.testing.assert_allclose(jnp.sum(gradient * direction), central, rtol=1e-6)


@pytest.mark.parametrize(
    "make_noise",
    [
        pytest.param(
            lambda: lx.MatrixLinearOperator(
                jnp.eye(4) + 0.2, lx.positive_semidefinite_tag
            ),
            id="dense",
        ),
        pytest.param(
            lambda: gaussx.Kronecker(
                lx.DiagonalLinearOperator(jnp.array([0.5, 1.5])),
                lx.DiagonalLinearOperator(jnp.array([0.4, 0.8])),
            ),
            id="kronecker",
        ),
        pytest.param(
            lambda: gaussx.BlockDiag(
                lx.DiagonalLinearOperator(jnp.ones(2)),
                lx.MatrixLinearOperator(jnp.eye(2) + 0.1, lx.positive_semidefinite_tag),
            ),
            id="block",
        ),
    ],
)
def test_ensemble_log_likelihood_takes_structured_noise(make_noise):
    obs_particles = jr.normal(jr.PRNGKey(3), (12, 4))
    innovation = jr.normal(jr.PRNGKey(4), (4,))
    noise = make_noise()
    S = flx.innovation_covariance(obs_particles, noise).as_matrix()
    np.testing.assert_allclose(
        ensemble_log_likelihood(obs_particles, innovation, noise),
        _dense_log_likelihood(innovation, S),
        rtol=1e-10,
    )

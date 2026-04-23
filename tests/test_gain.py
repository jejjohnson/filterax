"""Tests for the ensemble Kalman gain primitive."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np

import filterax as flx


def _dense_gain(P, H, R):
    """Reference K = P H^T (H P H^T + R)^{-1}."""
    S = H @ P @ H.T + R
    return P @ H.T @ np.linalg.inv(S)


def test_kalman_gain_matches_analytic_linear(getkey):
    N_e, N_x, N_y = 200, 6, 3
    particles = jr.normal(getkey(), (N_e, N_x))
    H = jr.normal(getkey(), (N_y, N_x))
    obs_particles = particles @ H.T
    R_diag = 0.5 * jnp.ones(N_y)
    R = lx.DiagonalLinearOperator(R_diag)

    K_got = np.asarray(flx.kalman_gain(particles, obs_particles, R))

    P = np.asarray(flx.ensemble_covariance(particles).as_matrix())
    K_exp = _dense_gain(P, np.asarray(H), np.diag(np.asarray(R_diag)))
    np.testing.assert_allclose(K_got, K_exp, atol=1e-7, rtol=1e-5)


def test_kalman_gain_shape(getkey):
    particles = jr.normal(getkey(), (20, 5))
    obs_particles = jr.normal(getkey(), (20, 3))
    R = lx.DiagonalLinearOperator(jnp.ones(3))
    K = flx.kalman_gain(particles, obs_particles, R)
    assert K.shape == (5, 3)


def test_kalman_gain_zero_ensemble_spread_gives_zero_gain(getkey):
    # If the ensemble collapses in state space, C^{xH} = 0 and K = 0.
    N_e, N_x, N_y = 30, 4, 3
    mean = jr.normal(getkey(), (N_x,))
    particles = jnp.broadcast_to(mean, (N_e, N_x))
    obs_particles = jr.normal(getkey(), (N_e, N_y))
    R = lx.DiagonalLinearOperator(jnp.ones(N_y))
    K = flx.kalman_gain(particles, obs_particles, R)
    np.testing.assert_allclose(np.asarray(K), 0.0, atol=1e-9)

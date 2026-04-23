"""Tests for the ensemble statistics primitives."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import filterax as flx


def test_ensemble_mean_matches_numpy(getkey):
    particles = jr.normal(getkey(), (32, 7))
    got = np.asarray(flx.ensemble_mean(particles))
    expected = np.asarray(particles).mean(axis=0)
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_ensemble_mean_shape(getkey):
    assert flx.ensemble_mean(jr.normal(getkey(), (10, 5))).shape == (5,)


def test_ensemble_anomalies_rows_sum_to_zero(getkey):
    anom = flx.ensemble_anomalies(jr.normal(getkey(), (40, 6)))
    assert anom.shape == (40, 6)
    np.testing.assert_allclose(np.asarray(anom).sum(axis=0), 0.0, atol=1e-10)


def test_ensemble_covariance_matches_bessel_numpy(getkey):
    particles = jr.normal(getkey(), (50, 5))
    got = np.asarray(flx.ensemble_covariance(particles).as_matrix())
    expected = np.cov(np.asarray(particles).T, ddof=1)  # Bessel-corrected
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_ensemble_covariance_symmetric(getkey):
    cov = flx.ensemble_covariance(jr.normal(getkey(), (20, 4))).as_matrix()
    np.testing.assert_allclose(np.asarray(cov), np.asarray(cov).T, atol=1e-10)


def test_ensemble_covariance_psd(getkey):
    # Eigenvalues should be non-negative (up to numerical noise).
    cov = flx.ensemble_covariance(jr.normal(getkey(), (30, 6))).as_matrix()
    eigs = jnp.linalg.eigvalsh(cov)
    assert float(eigs.min()) > -1e-10


def test_cross_covariance_matches_bessel_numpy(getkey):
    theta = jr.normal(getkey(), (60, 4))
    G = jr.normal(getkey(), (60, 3))
    got = np.asarray(flx.cross_covariance(theta, G))
    dev_t = np.asarray(theta) - np.asarray(theta).mean(axis=0)
    dev_g = np.asarray(G) - np.asarray(G).mean(axis=0)
    expected = dev_t.T @ dev_g / (60 - 1)
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_cross_covariance_linear_H(getkey):
    # For linear H, C^{xH} should equal P @ H.T.
    key = getkey()
    particles = jr.normal(key, (80, 5))
    H = jr.normal(getkey(), (3, 5))
    obs_particles = particles @ H.T
    C_xH = np.asarray(flx.cross_covariance(particles, obs_particles))
    P = np.asarray(flx.ensemble_covariance(particles).as_matrix())
    np.testing.assert_allclose(C_xH, P @ np.asarray(H).T, atol=1e-9)


@pytest.mark.parametrize("N_e", [3, 10, 100])
def test_ensemble_mean_shape_parametrized(getkey, N_e):
    assert flx.ensemble_mean(jr.normal(getkey(), (N_e, 4))).shape == (4,)


@pytest.mark.parametrize("N_e", [0, 1])
def test_ensemble_covariance_rejects_degenerate_ensemble(N_e):
    # N_e < 2 would divide by zero in the Bessel correction; fail fast.
    with pytest.raises(ValueError, match="at least 2 ensemble members"):
        flx.ensemble_covariance(jnp.zeros((N_e, 4)))


@pytest.mark.parametrize("N_e", [0, 1])
def test_cross_covariance_rejects_degenerate_ensemble(N_e):
    with pytest.raises(ValueError, match="at least 2 ensemble members"):
        flx.cross_covariance(jnp.zeros((N_e, 4)), jnp.zeros((N_e, 3)))

"""Tests for the perturbed-observations primitive."""

from __future__ import annotations

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


def test_perturbed_observations_empirical_mean_and_cov(getkey):
    obs = jnp.asarray([3.0, -1.0, 2.0])
    R_diag = jnp.asarray([0.5, 1.0, 2.0])
    R = lx.DiagonalLinearOperator(R_diag)

    pert = flx.perturbed_observations(getkey(), obs, R, n_ensemble=20000)
    # Mean converges to obs.
    np.testing.assert_allclose(
        np.asarray(pert).mean(axis=0), np.asarray(obs), atol=2e-2
    )
    # Empirical covariance close to diag(R) at large sample sizes.
    centred = np.asarray(pert) - np.asarray(obs)
    emp_cov = centred.T @ centred / (pert.shape[0] - 1)
    np.testing.assert_allclose(np.diag(emp_cov), np.asarray(R_diag), rtol=0.05)


def test_perturbed_observations_deterministic(getkey):
    obs = jnp.zeros(3)
    R = lx.DiagonalLinearOperator(jnp.ones(3))
    key = jr.PRNGKey(42)
    a = flx.perturbed_observations(key, obs, R, n_ensemble=5)
    b = flx.perturbed_observations(key, obs, R, n_ensemble=5)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

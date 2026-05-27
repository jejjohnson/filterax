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

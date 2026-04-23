"""Smoke tests that L0 primitives compose with ``jit``, ``grad``, and ``vmap``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

import filterax as flx


def test_ensemble_mean_jit(getkey):
    f = jax.jit(flx.ensemble_mean)
    p = jr.normal(getkey(), (10, 4))
    assert jnp.allclose(f(p), flx.ensemble_mean(p))


def test_kalman_gain_jit(getkey):
    R = lx.DiagonalLinearOperator(jnp.ones(3))

    def run(particles, obs_particles):
        return flx.kalman_gain(particles, obs_particles, R)

    run_jit = jax.jit(run)
    p = jr.normal(getkey(), (20, 5))
    o = jr.normal(getkey(), (20, 3))
    out = run_jit(p, o)
    assert out.shape == (5, 3)
    assert jnp.allclose(out, run(p, o), atol=1e-6)


def test_log_likelihood_grad(getkey):
    """Gradient of log-likelihood wrt innovation equals ``-S^{-1} v``."""
    v = jr.normal(getkey(), (4,))
    R_diag = 0.1 * jnp.ones(4)
    S = lx.DiagonalLinearOperator(R_diag)
    grad_v = jax.grad(lambda vv: flx.log_likelihood(vv, S))(v)
    expected = -v / R_diag
    assert jnp.allclose(grad_v, expected, atol=1e-8)


def test_ensemble_covariance_vmap(getkey):
    """Batch several independent ensembles via vmap."""
    batched = jr.normal(getkey(), (3, 15, 4))  # 3 ensembles each (15, 4)

    def cov_matrix(p):
        return flx.ensemble_covariance(p).as_matrix()

    batched_cov = jax.vmap(cov_matrix)(batched)
    assert batched_cov.shape == (3, 4, 4)

    # Each slice should match the non-vmapped primitive.
    for i in range(3):
        ref = flx.ensemble_covariance(batched[i]).as_matrix()
        assert jnp.allclose(batched_cov[i], ref, atol=1e-10)


def test_kalman_gain_differentiable_wrt_particles(getkey):
    """A scalar function of K must differentiate cleanly wrt the ensemble."""
    R = lx.DiagonalLinearOperator(jnp.ones(3))

    def scalar(particles, obs_particles):
        return flx.kalman_gain(particles, obs_particles, R).sum()

    p = jr.normal(getkey(), (20, 4))
    o = jr.normal(getkey(), (20, 3))
    g = jax.grad(scalar, argnums=0)(p, o)
    assert g.shape == p.shape
    assert jnp.isfinite(g).all()

"""Tests for state/result/config containers."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

import filterax as flx


def test_filter_state_is_pytree():
    state = flx.FilterState(particles=jnp.zeros((4, 3)), step=0)
    leaves, treedef = jax.tree.flatten(state)
    restored = jax.tree.unflatten(treedef, leaves)
    assert jnp.allclose(restored.particles, state.particles)
    assert restored.step == state.step


def test_process_state_is_pytree():
    state = flx.ProcessState(
        particles=jnp.zeros((5, 3)),
        forward_evals=jnp.zeros((5, 2)),
        obs=jnp.zeros(2),
        noise_cov=lx.DiagonalLinearOperator(jnp.ones(2)),
        step=0,
        algo_time=jnp.asarray(0.0),
    )
    leaves, treedef = jax.tree.flatten(state)
    restored = jax.tree.unflatten(treedef, leaves)
    assert restored.particles.shape == state.particles.shape


def test_uki_state_is_pytree():
    state = flx.UKIState(
        mean=jnp.zeros(4),
        covariance=lx.DiagonalLinearOperator(jnp.ones(4)),
        step=2,
    )
    leaves, treedef = jax.tree.flatten(state)
    restored = jax.tree.unflatten(treedef, leaves)
    assert restored.step == 2


def test_analysis_result_defaults_and_fields():
    res = flx.AnalysisResult(particles=jnp.zeros((3, 4)))
    assert res.log_likelihood is None
    assert res.diagnostics is None

    res2 = flx.AnalysisResult(
        particles=jnp.zeros((3, 4)),
        log_likelihood=jnp.asarray(-1.0),
        diagnostics={"spread": jnp.asarray(0.5)},
    )
    assert res2.diagnostics is not None


def test_filter_state_jit_roundtrip():
    @eqx.filter_jit
    def bump(state: flx.FilterState) -> flx.FilterState:
        return flx.FilterState(particles=state.particles + 1.0, step=state.step + 1)

    start = flx.FilterState(particles=jnp.zeros((2, 3)), step=0)
    end = bump(start)
    assert end.step == 1
    assert jnp.allclose(end.particles, 1.0)


def test_filter_config_static_fields():
    cfg = flx.FilterConfig(n_ensemble=50)
    assert cfg.n_ensemble == 50
    assert cfg.localizer is None
    assert cfg.inflator is None

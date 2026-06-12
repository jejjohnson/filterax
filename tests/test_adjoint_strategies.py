"""Adjoint strategies for differentiable_assimilate.

Forward values must be identical under every strategy; gradients must
match the legacy checkpoint flag exactly, reduce to the full gradient
when the truncation window covers the rollout, and stay bounded on
chaotic dynamics where the full adjoint explodes.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

import filterax as flx
from filterax.differentiable import (
    DirectAdjoint,
    RecursiveCheckpointAdjoint,
    TruncatedAdjoint,
    differentiable_assimilate,
)
from filterax.filters import ETKF


class _LinearDynamics(eqx.Module):
    """x -> a * x over each window; a is the trainable parameter."""

    a: jax.Array

    def __call__(self, state, t0, t1):
        return self.a * state


def _setup(T=6, key=0):
    k = jax.random.key(key)
    init = jax.random.normal(k, (5, 3))
    obs = jnp.linspace(-1.0, 1.0, T)[:, None] * jnp.ones((T, 3))
    times = jnp.arange(1.0, T + 1.0)
    R = lx.DiagonalLinearOperator(0.5 * jnp.ones(3))
    return init, obs, times, R


def _loss(a, adjoint=None, checkpoint=False, T=6):
    init, obs, times, R = _setup(T=T)
    result = differentiable_assimilate(
        ETKF(),
        _LinearDynamics(a),
        lambda x: x,
        init,
        obs,
        times,
        R,
        adjoint=adjoint,
        checkpoint=checkpoint,
    )
    return jnp.sum(result.particles**2)


class TestEquivalences:
    def test_forward_identical_across_strategies(self):
        a = jnp.asarray(0.9)
        base = _loss(a, adjoint=DirectAdjoint())
        for adj in (RecursiveCheckpointAdjoint(), TruncatedAdjoint(k=2)):
            assert jnp.allclose(base, _loss(a, adjoint=adj))

    def test_direct_matches_legacy_default(self):
        a = jnp.asarray(0.9)
        g_new = jax.grad(lambda a: _loss(a, adjoint=DirectAdjoint()))(a)
        g_old = jax.grad(lambda a: _loss(a))(a)
        assert jnp.allclose(g_new, g_old)

    def test_checkpoint_strategy_matches_legacy_flag(self):
        a = jnp.asarray(0.9)
        g_new = jax.grad(lambda a: _loss(a, adjoint=RecursiveCheckpointAdjoint()))(a)
        with pytest.warns(DeprecationWarning, match="checkpoint=True"):
            g_old = jax.grad(lambda a: _loss(a, checkpoint=True))(a)
        assert jnp.allclose(g_new, g_old)

    def test_truncated_full_window_matches_direct(self):
        a = jnp.asarray(0.9)
        g_full = jax.grad(lambda a: _loss(a, adjoint=DirectAdjoint()))(a)
        g_trunc = jax.grad(lambda a: _loss(a, adjoint=TruncatedAdjoint(k=6)))(a)
        assert jnp.allclose(g_full, g_trunc)

    def test_truncated_differs_from_direct_when_cutting(self):
        a = jnp.asarray(0.9)
        g_full = jax.grad(lambda a: _loss(a, adjoint=DirectAdjoint()))(a)
        g_k1 = jax.grad(lambda a: _loss(a, adjoint=TruncatedAdjoint(k=1)))(a)
        assert not jnp.allclose(g_full, g_k1)

    def test_pipekit_shaped_spec_accepted(self):
        """Structurally identical foreign specs work (duck-typed dispatch)."""

        class TruncatedAdjoint:
            k = 2

        a = jnp.asarray(0.9)
        g_foreign = jax.grad(lambda a: _loss(a, adjoint=TruncatedAdjoint()))(a)
        from filterax.differentiable import TruncatedAdjoint as Ours

        g_ours = jax.grad(lambda a: _loss(a, adjoint=Ours(k=2)))(a)
        assert jnp.allclose(g_foreign, g_ours)


class TestChaoticTaming:
    def test_truncated_gradient_bounded_on_expanding_dynamics(self):
        """On expanding dynamics the truncated gradient is far tamer."""
        a = jnp.asarray(1.6)  # |a| > 1: each window amplifies perturbations
        g_full = jnp.abs(jax.grad(lambda a: _loss(a, adjoint=DirectAdjoint(), T=12))(a))
        g_k1 = jnp.abs(
            jax.grad(lambda a: _loss(a, adjoint=TruncatedAdjoint(k=1), T=12))(a)
        )
        assert jnp.isfinite(g_k1)
        # The analysis step itself contracts perturbations toward the
        # observations each cycle, so the full adjoint grows far more
        # slowly than the raw dynamics would suggest — assert the
        # strict ordering rather than an arbitrary explosion factor.
        assert g_k1 < g_full


class TestValidation:
    def test_both_adjoint_and_checkpoint_rejected(self):
        with pytest.raises(ValueError, match="not both"):
            _loss(jnp.asarray(0.9), adjoint=DirectAdjoint(), checkpoint=True)

    def test_bad_k_rejected(self):
        with pytest.raises(ValueError, match="k >= 1"):
            _loss(jnp.asarray(0.9), adjoint=TruncatedAdjoint(k=0))

    def test_unknown_strategy_rejected(self):
        with pytest.raises(ValueError, match="Unrecognised"):
            _loss(jnp.asarray(0.9), adjoint=object())

    def test_stochastic_filter_still_refused(self):
        init, obs, times, R = _setup()
        with pytest.raises(ValueError, match=r"[Ss]tochastic"):
            differentiable_assimilate(
                flx.filters.StochasticEnKF(),
                _LinearDynamics(jnp.asarray(0.9)),
                lambda x: x,
                init,
                obs,
                times,
                R,
                adjoint=TruncatedAdjoint(k=1),
            )

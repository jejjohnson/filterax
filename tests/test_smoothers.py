"""Tests for the Wave 5.A ensemble smoothers (EnKS, EnsembleRTS, FixedLagSmoother).

All tests are *algebraic* — they pin the smoother to the formula in the
design doc on tiny synthetic histories rather than chasing Monte-Carlo
convergence against an analytic Kalman RTS smoother. The expensive
"ensemble mean matches the analytic posterior" check needs a 1000+
member ensemble to be meaningful and adds nothing the algebraic check
doesn't already cover.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import filterax as flx


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _synth_history(
    getkey, T: int = 3, N_e: int = 8, N_x: int = 3
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Random ``(T, N_e, N_x)`` forecast / analysis histories.

    The smoother is a pure linear-algebra map over its inputs — the
    invariants we test don't care whether those inputs came from a real
    filter pass, so we skip running ETKF and avoid the per-test JIT
    compile cost.
    """
    forecast = jr.normal(getkey(), (T, N_e, N_x))
    analysis = jr.normal(getkey(), (T, N_e, N_x))
    return forecast, analysis


def _ref_smoother_step(
    smoothed_next: np.ndarray,
    analysis_t: np.ndarray,
    forecast_t1: np.ndarray,
) -> np.ndarray:
    """Plain-numpy reference for one EnKS backward step.

    Uses the *state-space* form ``G = A^T F (F^T F)^+`` with
    :func:`numpy.linalg.pinv`. The package implementation uses the dual
    ``F F^T`` ensemble-space form; agreement between the two formulations
    is what proves the implementation correct.
    """
    A = analysis_t - analysis_t.mean(axis=0)
    F = forecast_t1 - forecast_t1.mean(axis=0)
    G = A.T @ F @ np.linalg.pinv(F.T @ F)
    return analysis_t + (smoothed_next - forecast_t1) @ G.T


def _ref_enks_backward(analysis: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Full backward EnKS pass in numpy."""
    T = analysis.shape[0]
    smoothed = np.empty_like(analysis)
    smoothed[-1] = analysis[-1]
    for t in range(T - 2, -1, -1):
        smoothed[t] = _ref_smoother_step(smoothed[t + 1], analysis[t], forecast[t + 1])
    return smoothed


# ──────────────────────────────────────────────────────────────────────
# EnKS — algebraic correctness & invariants
# ──────────────────────────────────────────────────────────────────────


def test_enks_matches_numpy_reference(getkey):
    """The package backward pass must match a plain-numpy implementation
    of the same formula written in the dual (state-space) form."""
    forecast, analysis = _synth_history(getkey, T=4, N_e=8, N_x=3)
    got = flx.EnKS().smooth(forecast, analysis).smoothed_history
    expected = _ref_enks_backward(np.asarray(analysis), np.asarray(forecast))
    np.testing.assert_allclose(np.asarray(got), expected, atol=1e-10)


def test_enks_preserves_final_time_analysis(getkey):
    """X^s_{T-1} == X^a_{T-1} by definition of the backward pass."""
    forecast, analysis = _synth_history(getkey)
    smoothed = flx.EnKS().smooth(forecast, analysis)
    np.testing.assert_allclose(
        np.asarray(smoothed.smoothed_history[-1]),
        np.asarray(analysis[-1]),
        atol=1e-12,
    )


def test_enks_zero_innovation_is_identity(getkey):
    """``forecast == analysis`` at every step → ``D == 0`` and the
    smoothed history coincides with the analysis history."""
    _, analysis = _synth_history(getkey, T=5, N_e=6, N_x=4)
    smoothed = flx.EnKS().smooth(analysis, analysis)
    np.testing.assert_allclose(
        np.asarray(smoothed.smoothed_history), np.asarray(analysis), atol=1e-10
    )


def test_enks_t_equals_one_is_identity(getkey):
    """T=1 → no backward step; smoothed history == analysis."""
    forecast, analysis = _synth_history(getkey, T=1)
    smoothed = flx.EnKS().smooth(forecast, analysis)
    np.testing.assert_allclose(
        np.asarray(smoothed.smoothed_history), np.asarray(analysis), atol=1e-12
    )


def test_enks_smoothing_result_particles_is_terminal(getkey):
    """``particles == smoothed_history[-1]`` matches the
    :class:`AssimilationResult.particles` convention."""
    forecast, analysis = _synth_history(getkey)
    smoothed = flx.EnKS().smooth(forecast, analysis)
    np.testing.assert_array_equal(
        np.asarray(smoothed.particles), np.asarray(smoothed.smoothed_history[-1])
    )
    np.testing.assert_array_equal(
        np.asarray(smoothed.particles), np.asarray(analysis[-1])
    )


# ──────────────────────────────────────────────────────────────────────
# EnsembleRTS
# ──────────────────────────────────────────────────────────────────────


def test_ensemble_rts_matches_enks(getkey):
    """In the cross-covariance form (zero model error) the two coincide."""
    forecast, analysis = _synth_history(getkey)
    enks = flx.EnKS().smooth(forecast, analysis)
    rts = flx.EnsembleRTS().smooth(forecast, analysis)
    np.testing.assert_allclose(
        np.asarray(rts.smoothed_history),
        np.asarray(enks.smoothed_history),
        atol=1e-12,
    )


# ──────────────────────────────────────────────────────────────────────
# FixedLagSmoother
# ──────────────────────────────────────────────────────────────────────


def test_fixed_lag_zero_returns_analysis(getkey):
    forecast, analysis = _synth_history(getkey)
    out = flx.FixedLagSmoother(lag=0).smooth(forecast, analysis)
    np.testing.assert_allclose(
        np.asarray(out.smoothed_history), np.asarray(analysis), atol=1e-12
    )


def test_fixed_lag_full_lookahead_matches_enks(getkey):
    """``lag >= T-1`` recovers the full EnKS pass at every position."""
    forecast, analysis = _synth_history(getkey)
    T = analysis.shape[0]
    enks = flx.EnKS().smooth(forecast, analysis)
    out = flx.FixedLagSmoother(lag=T - 1).smooth(forecast, analysis)
    np.testing.assert_allclose(
        np.asarray(out.smoothed_history),
        np.asarray(enks.smoothed_history),
        atol=1e-10,
    )


def test_fixed_lag_partial_window_differs_from_enks(getkey):
    """A short lag changes the earliest position (had room for some
    correction) but less than the full EnKS pass."""
    forecast, analysis = _synth_history(getkey, T=6)
    enks = flx.EnKS().smooth(forecast, analysis)
    fl = flx.FixedLagSmoother(lag=2).smooth(forecast, analysis)
    assert not np.allclose(np.asarray(fl.smoothed_history[0]), np.asarray(analysis[0]))
    assert not np.allclose(
        np.asarray(fl.smoothed_history[0]), np.asarray(enks.smoothed_history[0])
    )


def test_fixed_lag_rejects_negative_lag():
    with pytest.raises(ValueError, match="lag must be a non-negative int"):
        flx.FixedLagSmoother(lag=-1)


def test_fixed_lag_t_equals_one_with_positive_lag(getkey):
    """T==1 + lag>0 has no real backward window. The effective lag must
    clamp to ``T-1 == 0`` — earlier versions indexed
    ``forecast_history[1]`` and crashed."""
    forecast, analysis = _synth_history(getkey, T=1)
    out = flx.FixedLagSmoother(lag=5).smooth(forecast, analysis)
    np.testing.assert_array_equal(
        np.asarray(out.smoothed_history), np.asarray(analysis)
    )
    np.testing.assert_array_equal(np.asarray(out.particles), np.asarray(analysis[-1]))


def test_fixed_lag_huge_lag_matches_enks(getkey):
    """``lag >> T-1`` is clamped to ``T-1`` and still matches EnKS."""
    forecast, analysis = _synth_history(getkey, T=4)
    enks = flx.EnKS().smooth(forecast, analysis)
    out = flx.FixedLagSmoother(lag=1000).smooth(forecast, analysis)
    np.testing.assert_allclose(
        np.asarray(out.smoothed_history),
        np.asarray(enks.smoothed_history),
        atol=1e-10,
    )


# ──────────────────────────────────────────────────────────────────────
# JIT / grad compatibility — required for Wave 5.B differentiable DA
# ──────────────────────────────────────────────────────────────────────


def test_enks_smooth_under_jit(getkey):
    forecast, analysis = _synth_history(getkey)
    smoother = flx.EnKS()
    eager = smoother.smooth(forecast, analysis).smoothed_history
    jitted = jax.jit(lambda f, a: smoother.smooth(f, a).smoothed_history)(
        forecast, analysis
    )
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), atol=1e-12)


def test_enks_smooth_is_differentiable(getkey):
    """The backward pass is a chain of linear-algebra primitives — it
    must be JAX-differentiable for Wave 5.B."""
    forecast, analysis = _synth_history(getkey)
    smoother = flx.EnKS()

    def loss(scale):
        out = smoother.smooth(scale * forecast, scale * analysis)
        return jnp.sum(out.smoothed_history**2)

    grad = jax.grad(loss)(1.0)
    assert jnp.isfinite(grad)
    assert grad != 0.0


def test_fixed_lag_smooth_under_jit(getkey):
    forecast, analysis = _synth_history(getkey, T=5)
    fl = flx.FixedLagSmoother(lag=3)
    eager = fl.smooth(forecast, analysis).smoothed_history
    jitted = jax.jit(lambda f, a: fl.smooth(f, a).smoothed_history)(forecast, analysis)
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), atol=1e-12)


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────


def test_smoother_rejects_mismatched_shapes(getkey):
    a = jr.normal(getkey(), (4, 6, 3))
    f = jr.normal(getkey(), (4, 6, 4))
    with pytest.raises(ValueError, match="same shape"):
        flx.EnKS().smooth(f, a)


def test_smoother_rejects_two_d_input(getkey):
    a = jr.normal(getkey(), (6, 3))
    f = jr.normal(getkey(), (6, 3))
    with pytest.raises(ValueError, match=r"\(T, N_e, N_x\)"):
        flx.EnKS().smooth(f, a)


def test_smoother_rejects_empty_history():
    a = jnp.zeros((0, 5, 3))
    f = jnp.zeros((0, 5, 3))
    with pytest.raises(ValueError, match="at least one filter window"):
        flx.EnKS().smooth(f, a)


def test_smoother_rejects_degenerate_ensemble():
    a = jnp.zeros((3, 1, 4))
    f = jnp.zeros((3, 1, 4))
    with pytest.raises(ValueError, match="at least 2 ensemble members"):
        flx.EnKS().smooth(f, a)

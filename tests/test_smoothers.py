"""Tests for the Wave 5.A ensemble smoothers (EnKS, EnsembleRTS, FixedLagSmoother).

The linear-Gaussian baseline compares the ensemble smoother mean and
covariance against the analytic Kalman RTS recursion with a large
ensemble (Monte Carlo error tolerated). The remaining tests cover the
boundary cases that follow from the formula directly: final-time
identity, fixed-lag limits, JIT/grad compatibility, and shape checks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

import filterax as flx


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


class _LinearObs(flx.AbstractObsOperator):
    H: jnp.ndarray

    def __call__(self, state):
        return self.H @ state


class _LinearDynamics(flx.AbstractDynamics):
    M: jnp.ndarray

    def __call__(self, state, t0, t1):
        return self.M @ state


def _kalman_filter_smoother(
    x0: np.ndarray,
    P0: np.ndarray,
    M: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    ys: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Closed-form Kalman filter + RTS smoother on a linear-Gaussian model.

    Returns smoothed means and covariances at every time step.
    """
    T = len(ys)
    # Forward filter
    means_f, covs_f = [], []
    means_a, covs_a = [], []
    mean, cov = x0.copy(), P0.copy()
    for y in ys:
        # Forecast
        mean_f = M @ mean
        cov_f = M @ cov @ M.T  # no model error
        means_f.append(mean_f)
        covs_f.append(cov_f)
        # Analysis
        S = H @ cov_f @ H.T + R
        K = cov_f @ H.T @ np.linalg.inv(S)
        mean = mean_f + K @ (y - H @ mean_f)
        cov = cov_f - K @ H @ cov_f
        means_a.append(mean)
        covs_a.append(cov)
    # Backward RTS smoother
    means_s = [means_a[-1]]
    covs_s = [covs_a[-1]]
    for t in range(T - 2, -1, -1):
        # gain from analysis(t) → forecast(t+1)
        G = covs_a[t] @ M.T @ np.linalg.inv(covs_f[t + 1])
        mean_s = means_a[t] + G @ (means_s[0] - means_f[t + 1])
        cov_s = covs_a[t] + G @ (covs_s[0] - covs_f[t + 1]) @ G.T
        means_s.insert(0, mean_s)
        covs_s.insert(0, cov_s)
    return means_s, covs_s


def _run_etkf(
    key,
    init_ensemble,
    obs_seq,
    dynamics,
    obs_op,
    R,
):
    """Run the filterax ETKF assimilation and return ``AssimilationResult``."""
    del key  # ETKF is deterministic; key reserved for future stochastic mode
    model = flx.ETKF(dynamics=dynamics, obs_op=obs_op)
    return model.assimilate(init_ensemble, obs_seq, R)


@pytest.fixture
def linear_gauss_setup(getkey):
    """Linear-Gaussian assimilation problem with a large ensemble."""
    N_e, N_x, N_y, T = 4000, 3, 2, 5
    rng = np.random.default_rng(42)
    M_np = np.eye(N_x) + 0.05 * rng.standard_normal((N_x, N_x))
    H_np = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.5]])
    R_np = np.diag([0.2, 0.1])
    x0 = np.zeros(N_x)
    P0 = np.asarray([[1.0, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 1.0]])

    # Draw initial ensemble from N(x0, P0).
    chol = np.linalg.cholesky(P0)
    standard = jr.normal(getkey(), (N_e, N_x))
    particles = jnp.asarray(x0)[None, :] + jnp.asarray(standard) @ jnp.asarray(chol.T)

    # Synthetic observation trajectory.
    truth = x0.copy()
    ys = []
    for _ in range(T):
        truth = M_np @ truth
        ys.append(H_np @ truth + np.sqrt(np.diag(R_np)) * rng.standard_normal(N_y))

    dynamics = _LinearDynamics(M=jnp.asarray(M_np))
    obs_op = _LinearObs(H=jnp.asarray(H_np))
    R_op = lx.DiagonalLinearOperator(jnp.asarray(np.diag(R_np)))
    obs_seq = [(jnp.asarray(y), float(t + 1)) for t, y in enumerate(ys)]

    means_s, covs_s = _kalman_filter_smoother(x0, P0, M_np, H_np, R_np, ys)
    return {
        "particles": particles,
        "obs_seq": obs_seq,
        "dynamics": dynamics,
        "obs_op": obs_op,
        "R_op": R_op,
        "means_s": means_s,
        "covs_s": covs_s,
        "T": T,
        "N_e": N_e,
    }


# ──────────────────────────────────────────────────────────────────────
# EnKS — correctness & invariants
# ──────────────────────────────────────────────────────────────────────


def test_enks_matches_kalman_rts_linear_gaussian(linear_gauss_setup, getkey):
    """EnKS ensemble mean/cov approach the analytic RTS smoother."""
    s = linear_gauss_setup
    result = _run_etkf(
        getkey(), s["particles"], s["obs_seq"], s["dynamics"], s["obs_op"], s["R_op"]
    )
    smoothed = flx.EnKS().smooth(result.forecast_history, result.analysis_history)

    for t in range(s["T"]):
        mean_got = np.asarray(smoothed.smoothed_history[t].mean(axis=0))
        cov_got = np.cov(np.asarray(smoothed.smoothed_history[t]).T, ddof=1)
        np.testing.assert_allclose(mean_got, s["means_s"][t], atol=0.08)
        np.testing.assert_allclose(cov_got, s["covs_s"][t], atol=0.1)


def test_enks_preserves_final_time_analysis(getkey):
    """X^s_{T-1} == X^a_{T-1} by definition of the backward pass."""
    _, forecast_hist, analysis_hist = _toy_filter_result(getkey)
    smoothed = flx.EnKS().smooth(forecast_hist, analysis_hist)
    np.testing.assert_allclose(
        np.asarray(smoothed.smoothed_history[-1]),
        np.asarray(analysis_hist[-1]),
        atol=1e-12,
    )


def test_enks_preserves_ensemble_mean_under_zero_innovation(getkey):
    """When the smoother input has equal analysis and forecast ensembles
    at every time, the backward correction is zero and the smoothed
    history coincides with the analysis history."""
    N_e, N_x, T = 30, 4, 5
    analysis = jr.normal(getkey(), (T, N_e, N_x))
    # Set forecast == analysis at every step → D = smoothed_next - forecast
    # eventually collapses with the analyses if the analysis chain is shared.
    forecast = analysis
    smoothed = flx.EnKS().smooth(forecast, analysis)
    np.testing.assert_allclose(
        np.asarray(smoothed.smoothed_history),
        np.asarray(analysis),
        atol=1e-10,
    )


def test_enks_t_equals_one_is_identity(getkey):
    """T=1 → there is no backward step; smoothed history == analysis."""
    N_e, N_x = 20, 3
    analysis = jr.normal(getkey(), (1, N_e, N_x))
    forecast = jr.normal(getkey(), (1, N_e, N_x))
    smoothed = flx.EnKS().smooth(forecast, analysis)
    np.testing.assert_allclose(
        np.asarray(smoothed.smoothed_history),
        np.asarray(analysis),
        atol=1e-12,
    )


def test_enks_smoothing_result_particles_is_terminal(getkey):
    """``particles == smoothed_history[-1]`` matches the
    :class:`AssimilationResult.particles` convention (terminal ensemble
    for chaining into a follow-up forecast). For backward smoothers the
    terminal smoothed ensemble equals the filter's last analysis."""
    _, forecast_hist, analysis_hist = _toy_filter_result(getkey)
    smoothed = flx.EnKS().smooth(forecast_hist, analysis_hist)
    np.testing.assert_array_equal(
        np.asarray(smoothed.particles), np.asarray(smoothed.smoothed_history[-1])
    )
    np.testing.assert_array_equal(
        np.asarray(smoothed.particles), np.asarray(analysis_hist[-1])
    )


# ──────────────────────────────────────────────────────────────────────
# EnsembleRTS
# ──────────────────────────────────────────────────────────────────────


def test_ensemble_rts_matches_enks(getkey):
    """In the cross-covariance form (zero model error) the two coincide."""
    _, forecast_hist, analysis_hist = _toy_filter_result(getkey)
    enks = flx.EnKS().smooth(forecast_hist, analysis_hist)
    rts = flx.EnsembleRTS().smooth(forecast_hist, analysis_hist)
    np.testing.assert_allclose(
        np.asarray(rts.smoothed_history),
        np.asarray(enks.smoothed_history),
        atol=1e-12,
    )


# ──────────────────────────────────────────────────────────────────────
# FixedLagSmoother
# ──────────────────────────────────────────────────────────────────────


def test_fixed_lag_zero_returns_analysis(getkey):
    _, forecast_hist, analysis_hist = _toy_filter_result(getkey)
    fl = flx.FixedLagSmoother(lag=0)
    out = fl.smooth(forecast_hist, analysis_hist)
    np.testing.assert_allclose(
        np.asarray(out.smoothed_history), np.asarray(analysis_hist), atol=1e-12
    )


def test_fixed_lag_full_lookahead_matches_enks(getkey):
    """``lag >= T-1`` recovers the full EnKS pass at every position."""
    _, forecast_hist, analysis_hist = _toy_filter_result(getkey)
    T = analysis_hist.shape[0]
    enks = flx.EnKS().smooth(forecast_hist, analysis_hist)
    fl = flx.FixedLagSmoother(lag=T - 1)
    out = fl.smooth(forecast_hist, analysis_hist)
    np.testing.assert_allclose(
        np.asarray(out.smoothed_history),
        np.asarray(enks.smoothed_history),
        atol=1e-10,
    )


def test_fixed_lag_partial_window_differs_from_enks(getkey):
    """A short lag must change at least the earliest position relative to
    the analysis (it had room for some correction) and relative to EnKS
    (it had less lookahead)."""
    _, forecast_hist, analysis_hist = _toy_filter_result(getkey, T=6)
    enks = flx.EnKS().smooth(forecast_hist, analysis_hist)
    fl = flx.FixedLagSmoother(lag=2).smooth(forecast_hist, analysis_hist)
    # Position 0 saw some backward correction from positions 1 and 2.
    assert not np.allclose(
        np.asarray(fl.smoothed_history[0]), np.asarray(analysis_hist[0])
    )
    # But less correction than the full EnKS pass.
    assert not np.allclose(
        np.asarray(fl.smoothed_history[0]), np.asarray(enks.smoothed_history[0])
    )


def test_fixed_lag_rejects_negative_lag():
    with pytest.raises(ValueError, match="lag must be a non-negative int"):
        flx.FixedLagSmoother(lag=-1)


def test_fixed_lag_t_equals_one_with_positive_lag(getkey):
    """T==1 + lag>0 has no real backward window. The effective lag must
    clamp to ``T-1 == 0`` and return the analysis unchanged — earlier
    versions indexed ``forecast_history[1]`` and crashed."""
    N_e, N_x = 12, 3
    analysis = jr.normal(getkey(), (1, N_e, N_x))
    forecast = jr.normal(getkey(), (1, N_e, N_x))
    out = flx.FixedLagSmoother(lag=5).smooth(forecast, analysis)
    np.testing.assert_array_equal(
        np.asarray(out.smoothed_history), np.asarray(analysis)
    )
    np.testing.assert_array_equal(np.asarray(out.particles), np.asarray(analysis[-1]))


def test_fixed_lag_huge_lag_matches_enks_without_extra_work(getkey):
    """``lag >> T-1`` is clamped to ``T-1`` and still matches EnKS."""
    _, forecast_hist, analysis_hist = _toy_filter_result(getkey, T=4)
    enks = flx.EnKS().smooth(forecast_hist, analysis_hist)
    fl = flx.FixedLagSmoother(lag=1000).smooth(forecast_hist, analysis_hist)
    np.testing.assert_allclose(
        np.asarray(fl.smoothed_history),
        np.asarray(enks.smoothed_history),
        atol=1e-10,
    )


# ──────────────────────────────────────────────────────────────────────
# JIT / grad compatibility
# ──────────────────────────────────────────────────────────────────────


def test_enks_smooth_under_jit(getkey):
    _, forecast_hist, analysis_hist = _toy_filter_result(getkey)
    smoother = flx.EnKS()

    @jax.jit
    def go(f, a):
        return smoother.smooth(f, a).smoothed_history

    eager = smoother.smooth(forecast_hist, analysis_hist).smoothed_history
    jitted = go(forecast_hist, analysis_hist)
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), atol=1e-12)


def test_enks_smooth_is_differentiable(getkey):
    """The backward pass is a chain of linear-algebra primitives -- it
    must be JAX-differentiable to support Wave 5.B differentiable-DA
    training patterns."""
    _, forecast_hist, analysis_hist = _toy_filter_result(getkey)
    smoother = flx.EnKS()

    def loss(scale):
        out = smoother.smooth(scale * forecast_hist, scale * analysis_hist)
        return jnp.sum(out.smoothed_history**2)

    grad = jax.grad(loss)(1.0)
    assert jnp.isfinite(grad)
    assert grad != 0.0


def test_fixed_lag_smooth_under_jit(getkey):
    _, forecast_hist, analysis_hist = _toy_filter_result(getkey, T=5)
    fl = flx.FixedLagSmoother(lag=3)

    @jax.jit
    def go(f, a):
        return fl.smooth(f, a).smoothed_history

    eager = fl.smooth(forecast_hist, analysis_hist).smoothed_history
    jitted = go(forecast_hist, analysis_hist)
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), atol=1e-12)


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────


def test_smoother_rejects_mismatched_shapes(getkey):
    a = jr.normal(getkey(), (4, 10, 3))
    f = jr.normal(getkey(), (4, 10, 4))
    with pytest.raises(ValueError, match="same shape"):
        flx.EnKS().smooth(f, a)


def test_smoother_rejects_two_d_input(getkey):
    a = jr.normal(getkey(), (10, 3))
    f = jr.normal(getkey(), (10, 3))
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


# ──────────────────────────────────────────────────────────────────────
# Smoothing improves on the filter for a toy problem
# ──────────────────────────────────────────────────────────────────────


def test_enks_reduces_or_matches_rmse_vs_filter(linear_gauss_setup, getkey):
    """Smoothing with future observations should not make the analysis
    worse on average -- the smoothed RMSE against the analytic Kalman
    posterior mean is at most the filter's RMSE."""
    s = linear_gauss_setup
    result = _run_etkf(
        getkey(), s["particles"], s["obs_seq"], s["dynamics"], s["obs_op"], s["R_op"]
    )
    smoothed = flx.EnKS().smooth(result.forecast_history, result.analysis_history)

    # Compare ensemble means against the analytic RTS / filter means.
    rmse_filter = 0.0
    rmse_smoother = 0.0
    for t in range(s["T"]):
        m_a = np.asarray(result.analysis_history[t].mean(axis=0))
        m_s = np.asarray(smoothed.smoothed_history[t].mean(axis=0))
        rmse_filter += np.sum((m_a - s["means_s"][t]) ** 2)
        rmse_smoother += np.sum((m_s - s["means_s"][t]) ** 2)
    rmse_filter = np.sqrt(rmse_filter / s["T"])
    rmse_smoother = np.sqrt(rmse_smoother / s["T"])

    # Allow a small Monte Carlo slack -- the analytic posterior is the
    # *smoothed* mean, so the smoothed RMSE should be lower in
    # expectation. With finite ensembles we allow a tolerance.
    assert rmse_smoother <= rmse_filter + 1e-3


# ──────────────────────────────────────────────────────────────────────
# Shared toy-filter fixture (function so tests can vary T cheaply)
# ──────────────────────────────────────────────────────────────────────


def _toy_filter_result(getkey, T: int = 4):
    """Run a short ETKF assimilation and return its history arrays.

    Used to seed smoother tests that don't need the linear-Gaussian
    fixture's large ensemble.
    """
    N_e, N_x, N_y = 50, 3, 2
    rng = np.random.default_rng(0)
    H = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    M = jnp.eye(N_x) + 0.05 * jnp.asarray(rng.standard_normal((N_x, N_x)))
    R = lx.DiagonalLinearOperator(jnp.asarray([0.1, 0.1]))
    obs_op = _LinearObs(H=H)
    dynamics = _LinearDynamics(M=M)
    particles = jr.normal(getkey(), (N_e, N_x))
    ys = [(jr.normal(getkey(), (N_y,)), float(t + 1)) for t in range(T)]
    model = flx.ETKF(dynamics=dynamics, obs_op=obs_op)
    result = model.assimilate(particles, ys, R)
    return result.particles, result.forecast_history, result.analysis_history

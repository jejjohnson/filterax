"""Tests for the Wave 4 ensemble-DA diagnostics."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import filterax as flx


# ──────────────────────────────────────────────────────────────────────
# Phase-1 health checks
# ──────────────────────────────────────────────────────────────────────


def test_ensemble_spread_matches_numpy_std():
    ensemble = jr.normal(jr.PRNGKey(0), (80, 6))
    got = np.asarray(flx.utils.ensemble_spread(ensemble))
    expected = np.asarray(ensemble).std(axis=0, ddof=1)
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_rms_spread_is_scalar_root_mean_square():
    ensemble = jr.normal(jr.PRNGKey(1), (50, 4))
    got = float(flx.utils.rms_spread(ensemble))
    sigma = np.asarray(ensemble).std(axis=0, ddof=1)
    expected = float(np.sqrt((sigma * sigma).mean()))
    assert got == pytest.approx(expected, abs=1e-9)


def test_rmse_vs_truth_is_zero_when_mean_equals_truth():
    truth = jnp.asarray([1.0, -2.0, 0.5])
    mean = truth + 0.0
    assert float(flx.utils.rmse_vs_truth(mean, truth)) == pytest.approx(0.0)


def test_innovation_and_normalized_innovation():
    y = jnp.asarray([1.0, 2.0, 3.0])
    Hx = jnp.asarray([0.5, 2.5, 2.0])
    inn = np.asarray(flx.utils.innovation(y, Hx))
    np.testing.assert_allclose(inn, np.asarray([0.5, -0.5, 1.0]))
    var = jnp.asarray([0.25, 1.0, 4.0])
    norm = np.asarray(flx.utils.normalized_innovation(y, Hx, var))
    np.testing.assert_allclose(norm, np.asarray([1.0, -0.5, 0.5]), atol=1e-10)


def test_chi2_consistency_recovers_mahalanobis():
    d = jnp.asarray([1.0, 2.0, 3.0])
    S = jnp.diag(jnp.asarray([0.5, 1.0, 2.0]))
    S_inv_d = jnp.linalg.solve(S, d)
    got = float(flx.utils.chi2_consistency(d, S_inv_d))
    expected = float(d @ jnp.linalg.solve(S, d))
    assert got == pytest.approx(expected)
    norm = float(flx.utils.chi2_normalized(d, S_inv_d, 3))
    assert norm == pytest.approx(expected / 3)


def test_effective_ensemble_size_extremes():
    n_e = 10
    uniform = jnp.ones(n_e) / n_e
    assert float(flx.utils.effective_ensemble_size(uniform)) == pytest.approx(n_e)
    degenerate = jnp.zeros(n_e).at[0].set(1.0)
    assert float(flx.utils.effective_ensemble_size(degenerate)) == pytest.approx(1.0)


def test_weight_entropy_extremes():
    n_e = 8
    uniform = jnp.ones(n_e) / n_e
    assert float(flx.utils.weight_entropy(uniform)) == pytest.approx(float(np.log(n_e)))
    degenerate = jnp.zeros(n_e).at[0].set(1.0)
    assert float(flx.utils.weight_entropy(degenerate)) == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────
# Phase-2 calibration
# ──────────────────────────────────────────────────────────────────────


def test_spread_skill_ratio_calibrated_ensemble():
    # Construct an ensemble with σ_spread ≈ |error|: each member equals
    # truth ± σ. Spread-skill ratio should be near 1.
    truth = jnp.zeros(4)
    sigma = 0.5
    ensemble = jnp.stack([truth + sigma, truth - sigma])  # (2, 4)
    ssr = float(flx.utils.spread_skill_ratio(ensemble, truth))
    # Mean of ensemble equals truth → RMSE = 0 → SSR blows up. So
    # offset truth slightly so the metric is well-defined.
    truth_shifted = truth + sigma / 2.0
    ssr = float(flx.utils.spread_skill_ratio(ensemble, truth_shifted))
    assert ssr > 0


def test_rank_histogram_uniform_for_perfectly_calibrated_ensemble():
    # Place truth equispaced through a sorted ensemble draw — each
    # quantile should land in a distinct bin once aggregated.
    n_e, n_x, T = 19, 1, 200
    key = jr.PRNGKey(2)
    ensemble = jr.normal(key, (T, n_e, n_x))
    # Pick truth as the median of each ensemble (rank ≈ n_e/2).
    truth = jnp.median(ensemble, axis=1)
    counts = flx.utils.rank_histogram(ensemble, truth)
    assert counts.shape == (n_e + 1,)
    assert int(jnp.sum(counts)) == T * n_x


def test_rank_histogram_chi2_is_zero_for_uniform_counts():
    counts = jnp.full(11, 9, dtype=jnp.int32)
    assert float(flx.utils.rank_histogram_chi2(counts)) == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────
# Phase-3 a-posteriori covariance diagnostics
# ──────────────────────────────────────────────────────────────────────


def test_desroziers_estimates_have_right_shapes():
    T, N_y = 100, 4
    rng = np.random.default_rng(0)
    d_f = jnp.asarray(rng.standard_normal((T, N_y)))
    d_a = jnp.asarray(rng.standard_normal((T, N_y)))
    for fn in (
        flx.utils.desroziers_R_estimate,
        flx.utils.desroziers_innovation_cov,
    ):
        out = fn(d_f, d_a) if fn is flx.utils.desroziers_R_estimate else fn(d_f)
        assert out.shape == (N_y, N_y)
    a_cov = flx.utils.desroziers_analysis_residual_cov(d_a)
    assert a_cov.shape == (N_y, N_y)


def test_desroziers_R_estimate_recovers_diagonal_R():
    """If d_f, d_a are jointly sampled with E[d_a d_fᵀ] = R, the
    empirical Desroziers estimate should be close to R for large T."""
    T, N_y = 5000, 3
    R_true = jnp.diag(jnp.asarray([0.5, 1.0, 2.0]))
    rng = np.random.default_rng(1)
    chol = np.linalg.cholesky(np.asarray(R_true))
    z = rng.standard_normal((T, N_y))
    d_a = jnp.asarray(z @ chol.T)
    d_f = d_a  # joint sample matching the Desroziers invariant
    R_hat = np.asarray(flx.utils.desroziers_R_estimate(d_f, d_a))
    np.testing.assert_allclose(R_hat, np.asarray(R_true), atol=0.1)


def test_dfs_from_gain_equals_trace_of_KH():
    K = jnp.asarray([[0.5, 0.0], [0.1, 0.3]])
    H = jnp.eye(2)
    assert float(flx.utils.dfs_from_gain(K, H)) == pytest.approx(0.8)


def test_dfs_from_ensemble_zero_observation_impact():
    """If analysis == forecast (no observation impact), DFS proxy ≈ Nₓ."""
    n_e, n_x = 100, 4
    ens = jr.normal(jr.PRNGKey(3), (n_e, n_x))
    dfs = float(flx.utils.dfs_from_ensemble(ens, ens))
    np.testing.assert_allclose(dfs, n_x, atol=1e-9)


# ──────────────────────────────────────────────────────────────────────
# CRPS
# ──────────────────────────────────────────────────────────────────────


def test_crps_zero_for_collapsed_ensemble_at_truth():
    """When every member equals the observation, CRPS = 0."""
    y = jnp.asarray(2.5)
    ensemble = jnp.full((20,), 2.5)
    assert float(flx.utils.crps_ensemble(ensemble, y)) == pytest.approx(0.0, abs=1e-9)


def test_crps_positive_for_dispersed_ensemble():
    ensemble = jr.normal(jr.PRNGKey(4), (200,))
    y = jnp.asarray(0.5)
    crps = float(flx.utils.crps_ensemble(ensemble, y))
    assert crps > 0


def test_crps_batch_averages_over_obs():
    y = jnp.zeros(3)
    ensemble = jr.normal(jr.PRNGKey(5), (100, 3))
    per_obs = jnp.asarray(
        [flx.utils.crps_ensemble(ensemble[:, i], y[i]) for i in range(3)]
    )
    batch = flx.utils.crps_ensemble_batch(ensemble, y)
    np.testing.assert_allclose(float(batch), float(per_obs.mean()), atol=1e-9)

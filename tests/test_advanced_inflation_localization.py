"""Tests for the Wave 4 advanced inflation + localization primitives."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

import filterax as flx


# ──────────────────────────────────────────────────────────────────────
# SOAR taper
# ──────────────────────────────────────────────────────────────────────


def test_soar_taper_value_at_zero_and_radius():
    rho_0 = float(flx.soar_taper(jnp.zeros(()), radius=1.0))
    assert rho_0 == pytest.approx(1.0)
    rho_r = float(flx.soar_taper(jnp.asarray(1.0), radius=1.0))
    # (1 + 1) e^{-1} = 2 / e ≈ 0.7358
    assert rho_r == pytest.approx(2.0 / np.e, abs=1e-7)


def test_soar_taper_monotone_decreasing():
    d = jnp.linspace(0.0, 5.0, 50)
    out = np.asarray(flx.soar_taper(d, radius=1.0))
    diffs = np.diff(out)
    assert np.all(diffs <= 1e-12)


# ──────────────────────────────────────────────────────────────────────
# Adaptive localization
# ──────────────────────────────────────────────────────────────────────


def test_adaptive_localization_zeros_uncorrelated_pairs():
    # Construct ensemble where state and obs are independent draws
    # → cross-correlations are pure noise → most weights should be 0.
    n_e, n_x, n_y = 100, 5, 4
    key = jr.PRNGKey(0)
    k1, k2 = jr.split(key)
    state = jr.normal(k1, (n_e, n_x))
    obs = jr.normal(k2, (n_e, n_y))
    weights = np.asarray(flx.adaptive_localization(state, obs, significance=2.0))
    # Should mostly zero out (tight significance threshold).
    assert weights.mean() < 0.5


def test_adaptive_localization_keeps_correlated_pairs():
    # Identical ensembles → correlation = 1 → all weights kept.
    n_e, n = 100, 4
    ensemble = jr.normal(jr.PRNGKey(1), (n_e, n))
    weights = np.asarray(
        flx.adaptive_localization(ensemble, ensemble, significance=1.0)
    )
    # Diagonal entries (perfect correlation with self) must be kept.
    np.testing.assert_array_equal(np.diag(weights), np.ones(n))


# ──────────────────────────────────────────────────────────────────────
# Additive inflation
# ──────────────────────────────────────────────────────────────────────


def test_inflate_additive_preserves_ensemble_mean():
    ensemble = jr.normal(jr.PRNGKey(2), (50, 4))
    R = lx.DiagonalLinearOperator(jnp.full((4,), 0.1))
    inflated = flx.inflate_additive(jr.PRNGKey(3), ensemble, R)
    np.testing.assert_allclose(
        np.asarray(inflated.mean(axis=0)),
        np.asarray(ensemble.mean(axis=0)),
        atol=1e-10,
    )


def test_additive_inflator_class_matches_function():
    ensemble = jr.normal(jr.PRNGKey(4), (30, 3))
    R = lx.DiagonalLinearOperator(jnp.full((3,), 0.05))
    key = jr.PRNGKey(5)
    direct = flx.inflate_additive(key, ensemble, R)
    via_class = flx.AdditiveInflator(noise_cov=R, base_key=key)(ensemble)
    np.testing.assert_allclose(np.asarray(direct), np.asarray(via_class), atol=1e-12)


def test_additive_inflator_folds_step_into_key():
    """Regression: AdditiveInflator must derive a fresh per-step key
    when the L2 _run_loop passes ``step=`` through kwargs, otherwise
    every assimilation window draws identical Gaussian perturbations.
    """
    ensemble = jr.normal(jr.PRNGKey(20), (30, 3))
    R = lx.DiagonalLinearOperator(jnp.full((3,), 0.05))
    inflator = flx.AdditiveInflator(noise_cov=R, base_key=jr.PRNGKey(21))
    a = np.asarray(inflator(ensemble, step=0))
    b = np.asarray(inflator(ensemble, step=1))
    # Different steps must produce different draws.
    assert np.linalg.norm(a - b) > 1e-3
    # Same step is reproducible.
    a_again = np.asarray(inflator(ensemble, step=0))
    np.testing.assert_array_equal(a, a_again)


def test_inflate_adaptive_returns_jax_scalars():
    """Regression: inflate_adaptive must return JAX scalars so callers
    can carry the (μ, σ²) belief through jit / grad / lax.scan."""
    import jax

    S = jnp.eye(3) * 0.1
    d = jnp.ones(3)

    def run_one(mu_var):
        mu, var = mu_var
        return flx.inflate_adaptive(mu, var, d, S)

    # JIT compiles only when both return values are JAX arrays.
    mu_post, var_post = jax.jit(run_one)((1.0, 0.01))
    assert hasattr(mu_post, "dtype")
    assert hasattr(var_post, "dtype")
    assert bool(jnp.isfinite(mu_post))


# ──────────────────────────────────────────────────────────────────────
# Adaptive inflation (Anderson 2009)
# ──────────────────────────────────────────────────────────────────────


def test_inflate_adaptive_increases_factor_for_underdispersive_innovation():
    """χ²/Nᵧ > 1 should pull the posterior λ above the prior mean."""
    N_y = 3
    S = jnp.eye(N_y) * 0.1
    # Innovation 10× the noise std → big Mahalanobis norm.
    d = jnp.full((N_y,), 1.0)
    mu_prior, var_prior = 1.0, 0.01
    mu_post, var_post = flx.inflate_adaptive(mu_prior, var_prior, d, S)
    assert mu_post > mu_prior
    assert var_post > 0
    # Clamping should keep λ ≤ max_factor (default 1.2).
    assert mu_post <= 1.2


def test_inflate_adaptive_preserves_factor_for_calibrated_innovation():
    """χ²/Nᵧ ≈ 1 should leave λ near the prior mean."""
    N_y = 3
    S = jnp.eye(N_y)
    # Innovation with norm² == N_y exactly.
    d = jnp.ones(N_y)
    mu_post, _var_post = flx.inflate_adaptive(1.0, 0.01, d, S)
    assert mu_post == pytest.approx(1.0, abs=0.05)


# ──────────────────────────────────────────────────────────────────────
# Ledoit-Wolf shrinkage
# ──────────────────────────────────────────────────────────────────────


def test_ledoit_wolf_returns_psd_covariance():
    ensemble = jr.normal(jr.PRNGKey(6), (8, 6))  # underdetermined (Nₑ < Nₓ)
    P, lam = flx.ledoit_wolf_shrinkage(ensemble)
    assert P.shape == (6, 6)
    # Should be symmetric.
    np.testing.assert_allclose(np.asarray(P), np.asarray(P).T, atol=1e-10)
    # Positive eigenvalues (PSD).
    eigs = np.linalg.eigvalsh(np.asarray(P))
    assert eigs.min() > -1e-10
    # Shrinkage in [0, 1].
    assert 0.0 <= lam <= 1.0


def test_ledoit_wolf_reduces_condition_number():
    """Shrinking the sample covariance must produce a better-conditioned
    matrix than the raw sample (the whole point of regularisation)."""
    rng = np.random.default_rng(0)
    ensemble = jnp.asarray(rng.standard_normal((8, 20)))  # rank-deficient
    P_shrunk, lam = flx.ledoit_wolf_shrinkage(ensemble)
    sample = (
        (np.asarray(ensemble) - np.asarray(ensemble).mean(0)).T
        @ (np.asarray(ensemble) - np.asarray(ensemble).mean(0))
        / (8 - 1)
    )
    sample_cond = float(np.linalg.cond(sample))
    shrunk_cond = float(np.linalg.cond(np.asarray(P_shrunk)))
    assert shrunk_cond < sample_cond
    assert 0.0 <= lam <= 1.0

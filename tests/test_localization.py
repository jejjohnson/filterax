"""Tests for the localization tapers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import filterax as flx


def test_gaspari_cohn_compact_support():
    d = jnp.linspace(0.0, 3.0, 100)
    rho = np.asarray(flx.gaspari_cohn(d, radius=1.0))
    # Compact support at 2 * radius == 2.0
    assert np.all(rho[np.asarray(d) > 2.0] == 0.0)


def test_gaspari_cohn_at_zero_is_one():
    rho = float(flx.gaspari_cohn(jnp.zeros(()), radius=1.0))
    assert rho == pytest.approx(1.0, abs=1e-12)


def test_gaspari_cohn_continuity_at_branch_points():
    # Should be C^2; we check zeroth-order continuity at z = 1 and z = 2.
    eps = 1e-6
    for z in (1.0, 2.0):
        rho_lo = float(flx.gaspari_cohn(jnp.asarray(z - eps), radius=1.0))
        rho_hi = float(flx.gaspari_cohn(jnp.asarray(z + eps), radius=1.0))
        np.testing.assert_allclose(rho_lo, rho_hi, atol=1e-5)


def test_gaspari_cohn_monotone_decreasing():
    d = jnp.linspace(0.0, 1.5, 30)
    rho = np.asarray(flx.gaspari_cohn(d, radius=1.0))
    assert np.all(np.diff(rho) <= 1e-10)


def test_gaussian_taper_at_zero_is_one():
    val = float(flx.gaussian_taper(jnp.zeros(()), radius=1.0))
    assert val == pytest.approx(1.0)


def test_gaussian_taper_decay():
    d = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    rho = np.asarray(flx.gaussian_taper(d, radius=1.0))
    expected = np.exp(-(np.asarray(d) ** 2) / 2.0)
    np.testing.assert_allclose(rho, expected, atol=1e-12)


def test_hard_cutoff_binary():
    d = jnp.asarray([0.0, 0.5, 1.0, 1.5])
    out = np.asarray(flx.hard_cutoff(d, radius=1.0))
    np.testing.assert_array_equal(out, np.asarray([1.0, 1.0, 1.0, 0.0]))


def test_localize_schur_product():
    cov = jnp.arange(12.0).reshape(3, 4)
    taper = jnp.ones((3, 4)) * 0.5
    np.testing.assert_allclose(
        np.asarray(flx.localize(cov, taper)), np.asarray(cov) * 0.5
    )

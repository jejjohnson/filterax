"""Tests for inflation primitives and concrete inflators."""

from __future__ import annotations

import jax.random as jr
import numpy as np
import pytest

import filterax as flx


def test_multiplicative_inflation_preserves_mean(getkey):
    particles = jr.normal(getkey(), (40, 5))
    inflated = flx.inflate_multiplicative(particles, 1.5)
    np.testing.assert_allclose(
        np.asarray(inflated).mean(axis=0),
        np.asarray(particles).mean(axis=0),
        atol=1e-10,
    )


def test_multiplicative_inflation_scales_variance(getkey):
    particles = jr.normal(getkey(), (200, 4))
    factor = 1.3
    inflated = flx.inflate_multiplicative(particles, factor)
    var_in = np.asarray(particles).var(axis=0, ddof=1)
    var_out = np.asarray(inflated).var(axis=0, ddof=1)
    np.testing.assert_allclose(var_out, factor**2 * var_in, atol=1e-10)


def test_multiplicative_identity_is_noop(getkey):
    particles = jr.normal(getkey(), (20, 5))
    np.testing.assert_allclose(
        np.asarray(flx.inflate_multiplicative(particles, 1.0)),
        np.asarray(particles),
        atol=1e-12,
    )


def test_rtps_alpha_zero_returns_analysis(getkey):
    analysis = jr.normal(getkey(), (30, 6))
    forecast = jr.normal(getkey(), (30, 6))
    out = flx.inflate_rtps(analysis, forecast, alpha=0.0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(analysis), atol=1e-12)


def test_rtps_alpha_one_matches_forecast_spread(getkey):
    analysis = jr.normal(getkey(), (200, 4)) * 0.5
    forecast = jr.normal(getkey(), (200, 4)) * 1.5
    out = flx.inflate_rtps(analysis, forecast, alpha=1.0)
    np.testing.assert_allclose(
        np.asarray(out).std(axis=0, ddof=1),
        np.asarray(forecast).std(axis=0, ddof=1),
        atol=1e-10,
    )
    # Mean preserved.
    np.testing.assert_allclose(
        np.asarray(out).mean(axis=0),
        np.asarray(analysis).mean(axis=0),
        atol=1e-10,
    )


def test_rtpp_alpha_zero_is_analysis(getkey):
    analysis = jr.normal(getkey(), (25, 5))
    forecast = jr.normal(getkey(), (25, 5))
    out = flx.inflate_rtpp(analysis, forecast, alpha=0.0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(analysis), atol=1e-12)


def test_rtpp_alpha_one_uses_forecast_anomalies(getkey):
    analysis = jr.normal(getkey(), (30, 4))
    forecast = jr.normal(getkey(), (30, 4))
    out = flx.inflate_rtpp(analysis, forecast, alpha=1.0)
    mean_a = np.asarray(analysis).mean(axis=0)
    forecast_anom = np.asarray(forecast) - np.asarray(forecast).mean(axis=0)
    expected = mean_a + forecast_anom
    np.testing.assert_allclose(np.asarray(out), expected, atol=1e-10)


def test_inflator_classes_delegate(getkey):
    particles = jr.normal(getkey(), (30, 4))
    forecast = jr.normal(getkey(), (30, 4))

    mult = flx.MultiplicativeInflator(factor=1.1)
    np.testing.assert_allclose(
        np.asarray(mult(particles)),
        np.asarray(flx.inflate_multiplicative(particles, 1.1)),
    )

    rtps = flx.RTPS(alpha=0.5)
    np.testing.assert_allclose(
        np.asarray(rtps(particles, forecast)),
        np.asarray(flx.inflate_rtps(particles, forecast, 0.5)),
    )

    rtpp = flx.RTPP(alpha=0.3)
    np.testing.assert_allclose(
        np.asarray(rtpp(particles, forecast)),
        np.asarray(flx.inflate_rtpp(particles, forecast, 0.3)),
    )


def test_rtps_rtpp_require_forecast(getkey):
    particles = jr.normal(getkey(), (10, 3))
    with pytest.raises(ValueError, match="RTPS requires"):
        flx.RTPS(alpha=0.5)(particles)
    with pytest.raises(ValueError, match="RTPP requires"):
        flx.RTPP(alpha=0.5)(particles)

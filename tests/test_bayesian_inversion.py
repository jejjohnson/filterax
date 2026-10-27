"""Layer-2 Bayesian inversion: TEKI targets the MAP, EKS samples the posterior.

Both are checked against the closed-form linear-Gaussian posterior

    ``Σ* = (Σ₀⁻¹ + Fᵀ Γ⁻¹ F)⁻¹``,   ``θ* = m₀ + Σ* Fᵀ Γ⁻¹ (y − F m₀)``.

EKS runs at a fixed small step: ``EKSStableScheduler`` bounds ``Δt`` by the
parameter covariance, not by the posterior's stiffness, and overdisperses
on this problem (#109).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import filterax as flx
from filterax import ProcessConfig
from filterax.processes import TEKI as TEKIProcess, EKS_Process


def _psd(matrix):
    return lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)


def _overdetermined():
    """15 data, 6 parameters: the TEKI convergence problem."""
    F = jax.random.uniform(jax.random.key(0), (15, 6))
    gamma = 1e-2
    noise = jax.random.normal(jax.random.key(1), (15,))
    y = F @ jnp.linspace(0.5, 1.5, 6) + jnp.sqrt(gamma) * noise
    m0, S0 = jnp.ones(6), 0.5 * jnp.eye(6)
    cov = jnp.linalg.inv(jnp.linalg.inv(S0) + F.T @ F / gamma)
    mean = m0 + cov @ F.T @ (y - F @ m0) / gamma
    return F, y, _psd(gamma * jnp.eye(15)), m0, S0, mean, cov


def _two_parameter():
    """3 data, 2 parameters: the EKS sampling problem."""
    F = jnp.asarray([[1.0, 0.5], [-0.3, 1.2], [0.7, 0.1]])
    y = jnp.asarray([0.9, -1.4, 0.6])
    gamma = 0.1
    m0, S0 = jnp.zeros(2), jnp.eye(2)
    cov = jnp.linalg.inv(jnp.linalg.inv(S0) + F.T @ F / gamma)
    mean = m0 + cov @ F.T @ (y - F @ m0) / gamma
    return F, y, lx.DiagonalLinearOperator(gamma * jnp.ones(3)), m0, S0, mean, cov


# ──────────────────────────────────────────────────────────────────────
# TEKI
# ──────────────────────────────────────────────────────────────────────


def test_teki_map_is_a_fixed_point():
    """An ensemble centred exactly on θ* has a zero mean increment.

    With J > Nₚ, Woodbury turns the increment's mean into
    ``(Cᶿᶿ⁻¹ + Δt H̃ᵀ Γ̃⁻¹ H̃)⁻¹ H̃ᵀ Γ̃⁻¹ (ỹ − H̃ θ*)``, and the MAP
    condition makes ``H̃ᵀ Γ̃⁻¹ (ỹ − H̃ θ*)`` vanish.
    """
    F, y, Gamma, m0, S0, mean, _ = _overdetermined()
    noise = jax.random.normal(jax.random.key(2), (40, 6))
    particles = mean + 0.1 * (noise - noise.mean(0))
    process = TEKIProcess(
        scheduler=flx.FixedScheduler(dt=1.0), prior_mean=m0, prior_cov=_psd(S0)
    )
    state = process.init(particles, y, Gamma)
    new = process.update(state, particles @ F.T).particles
    np.testing.assert_allclose(new.mean(0), mean, atol=1e-10)


def test_teki_converges_to_map():
    F, y, Gamma, m0, S0, mean, _ = _overdetermined()
    init = m0 + jax.random.normal(jax.random.key(2), (200, 6)) @ jnp.sqrt(S0)
    teki = flx.TEKI(
        forward_fn=lambda th: F @ th,
        obs=y,
        noise_cov=Gamma,
        prior_mean=m0,
        prior_cov=_psd(S0),
    )
    short = flx.TEKI(
        forward_fn=lambda th: F @ th,
        obs=y,
        noise_cov=Gamma,
        prior_mean=m0,
        prior_cov=_psd(S0),
        config=ProcessConfig(scheduler=flx.FixedScheduler(dt=1.0), n_iterations=10),
    ).run(init)
    full = teki.run(init)
    err_short = float(jnp.max(jnp.abs(short.mean - mean)))
    err_full = float(jnp.max(jnp.abs(full.mean - mean)))
    # The deterministic ensemble contracts onto θ* and the error tracks
    # its spread (≈ 5e-3 after the default 200 steps on this problem).
    assert err_full < err_short
    assert err_full < 1e-2


# ──────────────────────────────────────────────────────────────────────
# EKS
# ──────────────────────────────────────────────────────────────────────


def _eks_moments(prior: bool):
    F, y, Gamma, m0, S0, mean, cov = _two_parameter()
    if not prior:
        cov = jnp.linalg.inv(F.T @ F / Gamma.diagonal[0])
        mean = cov @ F.T @ y / Gamma.diagonal[0]
    eks = flx.EKS(
        forward_fn=lambda th: F @ th,
        obs=y,
        noise_cov=Gamma,
        prior_mean=m0 if prior else None,
        prior_cov=_psd(S0) if prior else None,
        config=ProcessConfig(scheduler=flx.FixedScheduler(dt=0.02), n_iterations=600),
    )
    result = eks.run(jax.random.normal(jax.random.key(10), (200, 2)))
    z = jnp.abs(result.mean - mean) / jnp.sqrt(jnp.diag(cov))
    ratio = jnp.diag(result.covariance.as_matrix()) / jnp.diag(cov)
    return z, ratio


@pytest.mark.parametrize("prior", [True, False], ids=["prior", "flat"])
def test_eks_samples_the_posterior(prior):
    """Mean within 0.25 σ, variances within [0.75, 1.3] of the closed form.

    With J = 200 the sample mean's standard error is ≈ 0.07 σ and the
    sample variance's ≈ 0.10 relative; the bounds are about 3.5 of each,
    plus the O(Δt) Euler bias at Δt = 0.02.
    """
    z, ratio = _eks_moments(prior)
    assert float(jnp.max(z)) < 0.25
    assert bool(jnp.all((ratio > 0.75) & (ratio < 1.3)))


def test_eks_prior_needs_both_mean_and_covariance():
    with pytest.raises(ValueError, match="both prior_mean and prior_cov"):
        EKS_Process(scheduler=flx.FixedScheduler(dt=0.1), prior_mean=jnp.zeros(2))


# ──────────────────────────────────────────────────────────────────────
# EKS prior term: implicit, so a tight prior cannot destabilise it
# ──────────────────────────────────────────────────────────────────────


def _uninformative_eks(scheduler, n_iterations):
    """Constant forward model, prior 100× tighter than the initial ensemble."""
    m0 = jnp.asarray([0.3, -0.2])
    eks = flx.EKS(
        forward_fn=lambda th: jnp.zeros(3),
        obs=jnp.zeros(3),
        noise_cov=lx.DiagonalLinearOperator(jnp.ones(3)),
        prior_mean=m0,
        prior_cov=_psd(0.01 * jnp.eye(2)),
        config=ProcessConfig(scheduler=scheduler, n_iterations=n_iterations),
    )
    return m0, eks.run(jax.random.normal(jax.random.key(0), (200, 2)))


def test_eks_tight_prior_is_stable_with_default_scheduler():
    """The default scheduler picks Δt ≈ 1 here; an explicit prior drift
    multiplied offsets by 1 − Δt/σ₀² = −99 per step and diverged."""
    m0, result = _uninformative_eks(flx.EKSStableScheduler(), 60)
    assert bool(jnp.all(jnp.isfinite(result.particles)))
    assert float(jnp.max(jnp.abs(result.mean - m0))) < 0.05
    ratio = jnp.diag(result.covariance.as_matrix()) / 0.01
    assert bool(jnp.all((ratio > 0.5) & (ratio < 2.0)))


def test_eks_uninformative_data_returns_the_prior():
    m0, result = _uninformative_eks(flx.FixedScheduler(dt=0.02), 600)
    z = jnp.abs(result.mean - m0) / 0.1
    ratio = jnp.diag(result.covariance.as_matrix()) / 0.01
    assert float(jnp.max(z)) < 0.25
    assert bool(jnp.all((ratio > 0.75) & (ratio < 1.3)))


def test_eks_implicit_prior_step_matches_dense_formula():
    """θ = θ* − (I + Δt C Σ₀⁻¹)⁻¹ Δt C Σ₀⁻¹ (θ* − m₀), formed densely."""
    J, n_p, dt = 30, 4, 0.7
    particles = jax.random.normal(jax.random.key(3), (J, n_p))
    m0 = jnp.linspace(-1.0, 1.0, n_p)
    S0 = jnp.diag(jnp.asarray([0.01, 0.1, 1.0, 10.0]))
    # A constant forward model zeroes the data drift; zero ensemble noise is
    # not reachable, so compare against the same key's explicit update.
    process = EKS_Process(
        scheduler=flx.FixedScheduler(dt=dt), key=5, prior_mean=m0, prior_cov=_psd(S0)
    )
    flat = EKS_Process(scheduler=flx.FixedScheduler(dt=dt), key=5)
    noise_cov = lx.DiagonalLinearOperator(jnp.ones(2))
    evals = jnp.zeros((J, 2))
    explicit = flat.update(flat.init(particles, jnp.zeros(2), noise_cov), evals)
    implicit = process.update(process.init(particles, jnp.zeros(2), noise_cov), evals)
    anom = particles - particles.mean(0)
    C = anom.T @ anom / (J - 1)
    A = C @ jnp.linalg.inv(S0)
    theta_star = explicit.particles
    expected = (
        theta_star
        - jnp.linalg.solve(jnp.eye(n_p) + dt * A, dt * A @ (theta_star - m0).T).T
    )
    np.testing.assert_allclose(implicit.particles, expected, rtol=1e-10, atol=1e-12)

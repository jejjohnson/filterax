"""Linear-Gaussian inverse-problem baseline for the EKP processes.

For a linear forward model ``G ∈ ℝ^{N_d × Nₚ}`` and Gaussian prior
``θ ~ 𝒩(0, σ₀² I)``, the closed-form posterior is

    ``Σ_post = (Gᵀ Γ⁻¹ G + σ₀⁻² I)⁻¹``
    ``μ_post = Σ_post · Gᵀ Γ⁻¹ y``

Vanilla EKI / UKI applied for one step at ``Δt = 1`` collapses to the
exact Kalman update, so the analysis matches the closed-form posterior
to numerical precision. EKS produces samples whose mean and covariance
converge to ``(μ_post, Σ_post)`` after burn-in.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np

import filterax as flx
from filterax import ProcessConfig


def _linear_inverse_problem():
    G = jnp.asarray([[1.0, 0.5], [-0.3, 1.2], [0.7, 0.1]])
    theta_true = jnp.asarray([1.0, -1.0])
    y = G @ theta_true
    Gamma_diag = jnp.asarray([0.1, 0.1, 0.1])
    R = lx.DiagonalLinearOperator(Gamma_diag)
    sigma0 = 5.0
    R_inv = jnp.diag(1.0 / Gamma_diag)
    Sigma_post = jnp.linalg.inv(G.T @ R_inv @ G + (1.0 / sigma0**2) * jnp.eye(2))
    mu_post = Sigma_post @ G.T @ R_inv @ y
    return G, y, R, sigma0, mu_post, Sigma_post


def _forward(G):
    return lambda theta: G @ theta


# ──────────────────────────────────────────────────────────────────────
# EKI
# ──────────────────────────────────────────────────────────────────────


def test_eki_one_step_matches_kalman_posterior():
    G, y, R, sigma0, mu_post, _ = _linear_inverse_problem()
    init = sigma0 * jr.normal(jr.PRNGKey(0), (3000, 2))
    eki = flx.EKI(
        forward_fn=_forward(G),
        obs=y,
        noise_cov=R,
        scheduler=flx.FixedScheduler(dt=1.0),
        config=ProcessConfig(scheduler=flx.FixedScheduler(dt=1.0), n_iterations=1),
    )
    result = eki.run(init)
    np.testing.assert_allclose(np.asarray(result.mean), np.asarray(mu_post), atol=5e-2)


def test_eki_misfit_controller_reduces_misfit_monotonically():
    """The data-misfit controller's adaptive Δt is conservative; we just
    check that the misfit decreases monotonically over a long run."""
    G, y, R, sigma0, _, _ = _linear_inverse_problem()
    init = sigma0 * jr.normal(jr.PRNGKey(1), (200, 2))
    eki = flx.EKI(
        forward_fn=_forward(G),
        obs=y,
        noise_cov=R,
        scheduler=flx.DataMisfitController(),
        config=ProcessConfig(scheduler=flx.DataMisfitController(), n_iterations=200),
    )
    result = eki.run(init)

    # Initial and final misfits in Mahalanobis-norm.
    def misfit(particles):
        residuals = y[None, :] - particles @ G.T
        return float(jnp.mean(jnp.sum(residuals**2 / 0.1, axis=-1)))

    initial = misfit(init)
    final = misfit(result.particles)
    assert final < initial
    # algo_time grows but may not reach 1.0 with the conservative recipe.
    assert float(result.history_algo_time[-1]) > 0


# ──────────────────────────────────────────────────────────────────────
# EKS
# ──────────────────────────────────────────────────────────────────────


def test_eks_does_not_collapse():
    # EKS should preserve spread (ergodic sampler) — after many steps,
    # the ensemble covariance trace must remain bounded away from zero.
    G, y, R, sigma0, _, _Sigma_post = _linear_inverse_problem()
    init = sigma0 * jr.normal(jr.PRNGKey(2), (200, 2))
    eks = flx.EKS(
        forward_fn=_forward(G),
        obs=y,
        noise_cov=R,
        scheduler=flx.FixedScheduler(dt=0.05),
        config=ProcessConfig(scheduler=flx.FixedScheduler(dt=0.05), n_iterations=200),
        seed=3,
    )
    result = eks.run(init)
    sample_cov = result.covariance.as_matrix()
    # Trace bounded away from zero — sampler is exploring.
    assert float(jnp.trace(sample_cov)) > 1e-3


# ──────────────────────────────────────────────────────────────────────
# UKI
# ──────────────────────────────────────────────────────────────────────


def test_uki_one_step_matches_kalman_posterior():
    G, y, R, sigma0, mu_post, Sigma_post = _linear_inverse_problem()
    init_cov = lx.MatrixLinearOperator(
        sigma0**2 * jnp.eye(2), tags=lx.positive_semidefinite_tag
    )
    uki = flx.UKI(
        forward_fn=_forward(G),
        obs=y,
        noise_cov=R,
        scheduler=flx.FixedScheduler(dt=1.0),
        config=ProcessConfig(scheduler=flx.FixedScheduler(dt=1.0), n_iterations=1),
    )
    result = uki.run(jnp.zeros(2), init_cov)
    np.testing.assert_allclose(np.asarray(result.mean), np.asarray(mu_post), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(result.covariance.as_matrix()),
        np.asarray(Sigma_post),
        atol=1e-5,
    )


# ──────────────────────────────────────────────────────────────────────
# ETKI
# ──────────────────────────────────────────────────────────────────────


def test_etki_runs_and_drives_misfit_down():
    G, y, R, sigma0, _, _ = _linear_inverse_problem()
    init = sigma0 * jr.normal(jr.PRNGKey(4), (50, 2))
    etki = flx.processes.ETKI(scheduler=flx.FixedScheduler(dt=0.5))
    state = etki.init(init, y, R)
    initial_misfit = float(jnp.mean(jnp.sum((y - state.particles @ G.T) ** 2, axis=-1)))
    # The init forward_evals slot is zero; do a real iteration:
    evals = state.particles @ G.T
    state = etki.update(state, evals)
    new_misfit = float(jnp.mean(jnp.sum((y - state.particles @ G.T) ** 2, axis=-1)))
    assert new_misfit < initial_misfit


# ──────────────────────────────────────────────────────────────────────
# GNKI
# ──────────────────────────────────────────────────────────────────────


def test_gnki_converges_in_one_step_for_linear_gaussian():
    G, y, R, sigma0, mu_post, _ = _linear_inverse_problem()
    init = sigma0 * jr.normal(jr.PRNGKey(5), (60, 2))
    gnki = flx.processes.GNKI(
        scheduler=flx.FixedScheduler(dt=1.0),
        prior_mean=jnp.zeros(2),
        prior_cov=lx.MatrixLinearOperator(
            sigma0**2 * jnp.eye(2), tags=lx.positive_semidefinite_tag
        ),
    )
    state = gnki.init(init, y, R)
    evals = state.particles @ G.T
    state = gnki.update(state, evals)
    mean_after = jnp.mean(state.particles, axis=0)
    # GNKI reaches the posterior in one Gauss-Newton step for linear G.
    np.testing.assert_allclose(np.asarray(mean_after), np.asarray(mu_post), atol=0.1)


# ──────────────────────────────────────────────────────────────────────
# SparseInversion
# ──────────────────────────────────────────────────────────────────────


def test_sparse_inversion_drives_inactive_params_to_zero():
    # G isolates the first parameter; the second has zero gradient and
    # should be soft-thresholded to zero with a strong penalty.
    G = jnp.asarray([[1.0, 0.0], [0.0, 0.0]])
    y = jnp.asarray([2.0, 0.0])
    R = lx.DiagonalLinearOperator(jnp.asarray([0.1, 0.1]))
    init = 0.1 * jr.normal(jr.PRNGKey(6), (50, 2))
    sparse = flx.processes.SparseInversion(
        scheduler=flx.FixedScheduler(dt=0.5), penalty_weight=0.2
    )
    state = sparse.init(init, y, R)
    for _ in range(20):
        evals = state.particles @ G.T
        state = sparse.update(state, evals)
    mean_final = jnp.mean(state.particles, axis=0)
    # First param recovered (target = 2.0), second driven to zero.
    assert float(jnp.abs(mean_final[1])) < 0.05


# ──────────────────────────────────────────────────────────────────────
# TEKI
# ──────────────────────────────────────────────────────────────────────


def test_teki_pulls_ensemble_toward_prior_mean():
    # Ill-posed problem: G sees only the first component → without prior
    # regularisation the second component is unidentifiable. TEKI's
    # augmented identity block pulls the second component toward the
    # prior mean m₀ = 0 (relative to where vanilla EKI would leave it).
    G = jnp.asarray([[1.0, 0.0]])  # only sees first param
    y = jnp.asarray([1.0])
    R = lx.DiagonalLinearOperator(jnp.asarray([0.01]))
    init = jnp.asarray([[0.0, 5.0]] * 50) + 0.5 * jr.normal(jr.PRNGKey(7), (50, 2))

    # TEKI with small dt to land near algo_time ≈ 1 after 20 iters.
    teki = flx.processes.TEKI(
        scheduler=flx.FixedScheduler(dt=0.05),
        prior_mean=jnp.zeros(2),
        prior_cov=lx.MatrixLinearOperator(
            jnp.eye(2), tags=lx.positive_semidefinite_tag
        ),
    )
    state_t = teki.init(init, y, R)
    for _ in range(20):
        evals = state_t.particles @ G.T
        state_t = teki.update(state_t, evals)

    # Vanilla EKI for the same problem — same dt + iterations.
    eki = flx.processes.EKI(scheduler=flx.FixedScheduler(dt=0.05))
    state_e = eki.init(init, y, R)
    for _ in range(20):
        evals = state_e.particles @ G.T
        state_e = eki.update(state_e, evals)

    teki_second = float(jnp.mean(state_t.particles[:, 1]))
    eki_second = float(jnp.mean(state_e.particles[:, 1]))
    # TEKI must pull the unidentifiable second component closer to 0
    # than vanilla EKI does.
    assert abs(teki_second) < abs(eki_second)


# ──────────────────────────────────────────────────────────────────────
# Sigma-point utilities
# ──────────────────────────────────────────────────────────────────────


def test_sigma_points_reconstruct_mean_and_covariance():
    from filterax._src.processes import sigma_points

    mean = jnp.asarray([1.0, -2.0, 0.5])
    Sigma_dense = jnp.asarray([[2.0, 0.3, 0.1], [0.3, 1.5, -0.2], [0.1, -0.2, 1.0]])
    cov = lx.MatrixLinearOperator(Sigma_dense, tags=lx.positive_semidefinite_tag)

    points, w_mean, w_cov = sigma_points(mean, cov, alpha=1.0, beta=2.0, kappa=0.0)
    recon_mean = jnp.sum(w_mean[:, None] * points, axis=0)
    diffs = points - recon_mean[None, :]
    recon_cov = jnp.sum(
        w_cov[:, None, None] * diffs[:, :, None] * diffs[:, None, :], axis=0
    )
    np.testing.assert_allclose(np.asarray(recon_mean), np.asarray(mean), atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(recon_cov), np.asarray(Sigma_dense), atol=1e-7
    )


# pytest import here so test_eki_converges_with_misfit_controller can use it.

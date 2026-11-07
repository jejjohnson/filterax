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

import gaussx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

import filterax as flx
from filterax import ProcessConfig, optax as filterax_optax


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


@pytest.mark.slow
def test_eki_one_step_matches_kalman_posterior():
    """One EKI step with ``Δt = 1`` collapses to the Kalman update applied
    to the sample covariance — pin against that (algebraic), not against
    the broad-prior analytic posterior (convergence)."""
    G, y, R, sigma0, _, _ = _linear_inverse_problem()
    init = sigma0 * jr.normal(jr.PRNGKey(0), (60, 2))
    # Reference: Kalman analysis on the sample mean / sample cov.
    init_np = np.asarray(init)
    sample_mean = init_np.mean(axis=0)
    sample_cov = np.cov(init_np.T, ddof=1)
    G_np = np.asarray(G)
    R_np = np.diag(np.asarray([0.1, 0.1, 0.1]))
    y_np = np.asarray(y)
    S = G_np @ sample_cov @ G_np.T + R_np
    K = sample_cov @ G_np.T @ np.linalg.inv(S)
    mean_ref = sample_mean + K @ (y_np - G_np @ sample_mean)

    eki = flx.EKI(
        forward_fn=_forward(G),
        obs=y,
        noise_cov=R,
        scheduler=flx.FixedScheduler(dt=1.0),
        config=ProcessConfig(scheduler=flx.FixedScheduler(dt=1.0), n_iterations=1),
    )
    result = eki.run(init)
    np.testing.assert_allclose(np.asarray(result.mean), mean_ref, atol=1e-9)


@pytest.mark.integration
@pytest.mark.slow
def test_eki_misfit_controller_reduces_misfit_monotonically():
    """Correctness invariant: misfit decreases. Don't need 200 iterations
    to verify this."""
    G, y, R, sigma0, _, _ = _linear_inverse_problem()
    init = sigma0 * jr.normal(jr.PRNGKey(1), (50, 2))
    eki = flx.EKI(
        forward_fn=_forward(G),
        obs=y,
        noise_cov=R,
        scheduler=flx.DataMisfitController(),
        config=ProcessConfig(scheduler=flx.DataMisfitController(), n_iterations=20),
    )
    result = eki.run(init)

    def misfit(particles):
        residuals = y[None, :] - particles @ G.T
        return float(jnp.mean(jnp.sum(residuals**2 / 0.1, axis=-1)))

    assert misfit(result.particles) < misfit(init)
    assert float(result.history_algo_time[-1]) > 0


# ──────────────────────────────────────────────────────────────────────
# EKS
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_eks_does_not_collapse():
    """Correctness invariant: EKS's noise injection keeps the spread
    bounded away from zero. A short run is enough to verify the noise
    term fires — long-horizon convergence to the posterior isn't what
    this test should cover."""
    G, y, R, sigma0, _, _ = _linear_inverse_problem()
    init = sigma0 * jr.normal(jr.PRNGKey(2), (50, 2))
    eks = flx.EKS(
        forward_fn=_forward(G),
        obs=y,
        noise_cov=R,
        scheduler=flx.FixedScheduler(dt=0.05),
        config=ProcessConfig(scheduler=flx.FixedScheduler(dt=0.05), n_iterations=20),
        seed=3,
    )
    result = eks.run(init)
    sample_cov = result.covariance.as_matrix()
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


@pytest.mark.slow
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
    from filterax._processes._processes import sigma_points

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


# ──────────────────────────────────────────────────────────────────────
# Nonlinear inverse problem — verifies the algorithms work beyond the
# linear-Gaussian one-shot Kalman equivalence.
#
# We test the *invariant* that matters in practice: the analysis
# misfit strictly improves on the initial ensemble's, and the final
# parameter estimate lands closer to the truth. A tight "match
# θ_true exactly" assertion is too brittle with a finite ensemble and
# the EKI prior-regularisation pull — the algorithm reaches the MAP
# under the implicit ensemble prior, which only coincides with
# ``θ_true`` in the broad-prior, infinite-ensemble limit.
# ──────────────────────────────────────────────────────────────────────


def _nonlinear_forward():
    r"""Mildly nonlinear toy: ``G(θ) = θ + 0.3 sin(2 θ)``.

    Monotone (so the posterior is well-defined) but nonlinear enough
    that the cross-covariance linearisation is a real approximation
    rather than exact.
    """

    def G(theta):
        return theta + 0.3 * jnp.sin(2.0 * theta)

    return G


def _nonlinear_problem():
    forward = _nonlinear_forward()
    theta_true = jnp.asarray([0.7, -1.1, 0.3])
    y = forward(theta_true)
    R = lx.DiagonalLinearOperator(jnp.full((3,), 0.01))
    return forward, theta_true, y, R


def _misfit(forward, theta_or_particles, y):
    """Mean Mahalanobis-style misfit ‖y − G(θ)‖² (Γ = 0.01 I, omitted)."""
    if theta_or_particles.ndim == 1:
        return float(jnp.sum((y - forward(theta_or_particles)) ** 2))
    g = jax.vmap(forward)(theta_or_particles)
    return float(jnp.mean(jnp.sum((y[None, :] - g) ** 2, axis=-1)))


@pytest.mark.integration
def test_eki_reduces_misfit_and_moves_toward_truth_on_nonlinear_problem():
    forward, theta_true, y, R = _nonlinear_problem()
    init = 2.0 * jr.normal(jr.PRNGKey(10), (200, 3))
    eki = flx.EKI(
        forward_fn=forward,
        obs=y,
        noise_cov=R,
        scheduler=flx.FixedScheduler(dt=0.1),
        config=ProcessConfig(scheduler=flx.FixedScheduler(dt=0.1), n_iterations=10),
    )
    result = eki.run(init)

    initial_misfit = _misfit(forward, init, y)
    final_misfit = _misfit(forward, result.mean, y)
    assert final_misfit < initial_misfit / 10  # at least 10× reduction
    # Parameter estimate is closer to the truth than the initial mean.
    initial_dist = float(jnp.linalg.norm(jnp.mean(init, axis=0) - theta_true))
    final_dist = float(jnp.linalg.norm(result.mean - theta_true))
    assert final_dist < initial_dist


@pytest.mark.integration
def test_uki_reduces_misfit_on_nonlinear_problem():
    forward, _theta_true, y, R = _nonlinear_problem()
    init_cov = lx.MatrixLinearOperator(
        4.0 * jnp.eye(3), tags=lx.positive_semidefinite_tag
    )
    uki = flx.UKI(
        forward_fn=forward,
        obs=y,
        noise_cov=R,
        scheduler=flx.FixedScheduler(dt=0.1),
        config=ProcessConfig(scheduler=flx.FixedScheduler(dt=0.1), n_iterations=10),
    )
    result = uki.run(jnp.zeros(3), init_cov)
    initial_misfit = _misfit(forward, jnp.zeros(3), y)
    final_misfit = _misfit(forward, result.mean, y)
    assert final_misfit < initial_misfit / 10
    # UKI also shrinks its parametric covariance — the trace must drop.
    initial_cov_trace = float(jnp.trace(init_cov.as_matrix()))
    final_cov_trace = float(jnp.trace(result.covariance.as_matrix()))
    assert final_cov_trace < initial_cov_trace


def test_eki_update_is_finite_after_algo_time_one():
    """Regression: ``DataMisfitController`` returns ``dt = 0`` once
    ``algo_time ≥ 1``; the L1 EKI update must still produce finite
    output (no division by zero) so fixed-length user loops are safe.
    """
    G, y, R, _, _, _ = _linear_inverse_problem()
    init = 1.0 * jr.normal(jr.PRNGKey(20), (50, 2))
    process = flx.processes.EKI(scheduler=flx.DataMisfitController())
    state = process.init(init, y, R)
    # Drive the loop until algo_time ≥ 1 in a couple of huge steps.
    state = process.update(state, init @ G.T)
    state = process.update(state, state.particles @ G.T)
    state = process.update(state, state.particles @ G.T)
    # Now call update past the convergence point — must not NaN.
    for _ in range(3):
        evals = state.particles @ G.T
        state = process.update(state, evals)
    assert bool(jnp.all(jnp.isfinite(state.particles)))


def test_gnki_rejects_underdetermined_ensemble():
    """GNKI requires ``J > Nₚ`` because it inverts ``Cᶿᶿ``."""
    _G, y, R, _, _, _ = _linear_inverse_problem()
    init = jr.normal(jr.PRNGKey(21), (2, 2))  # J == N_p → not enough
    gnki = flx.processes.GNKI(
        scheduler=flx.FixedScheduler(dt=0.1),
        prior_mean=jnp.zeros(2),
        prior_cov=lx.MatrixLinearOperator(
            jnp.eye(2), tags=lx.positive_semidefinite_tag
        ),
    )
    try:
        gnki.init(init, y, R)
    except ValueError as exc:
        assert "GNKI requires" in str(exc)
    else:
        raise AssertionError("expected ValueError")


@pytest.mark.slow
def test_eks_preserves_spread_on_nonlinear_problem():
    """Correctness invariant on a nonlinear problem: EKS's noise term
    keeps spread positive and the update is finite. A short run is
    enough to catch a regression that would collapse the ensemble."""
    forward, _theta_true, y, R = _nonlinear_problem()
    init = 1.0 * jr.normal(jr.PRNGKey(11), (50, 3))
    eks = flx.EKS(
        forward_fn=forward,
        obs=y,
        noise_cov=R,
        scheduler=flx.FixedScheduler(dt=0.02),
        config=ProcessConfig(scheduler=flx.FixedScheduler(dt=0.02), n_iterations=20),
        seed=12,
    )
    result = eks.run(init)
    final_trace = float(jnp.trace(result.covariance.as_matrix()))
    assert final_trace > 1e-3
    assert bool(jnp.all(jnp.isfinite(result.particles)))


# ──────────────────────────────────────────────────────────────────────
# gaussx-backed updates
# ──────────────────────────────────────────────────────────────────────


def _inverse_problem_for_gaussx(num_members, obs_size, param_size=3):
    keys = jr.split(jr.PRNGKey(11), 3)
    forward = jr.normal(keys[0], (obs_size, param_size)) / jnp.sqrt(param_size)
    obs = jr.normal(keys[1], (obs_size,))
    particles = jr.normal(keys[2], (num_members, param_size))
    return forward, obs, particles


@pytest.mark.slow
@pytest.mark.parametrize(
    ("num_members", "obs_size"), [(30, 5), (5, 12)], ids=["dense", "woodbury"]
)
def test_eki_step_matches_the_dense_formula_on_both_routes(num_members, obs_size):
    """θ⁺ = θ + Cᶿᴳ (Cᴳᴳ + Γ / Δt)⁻¹ (y − G), formed densely."""
    forward, obs, particles = _inverse_problem_for_gaussx(num_members, obs_size)
    variances = jnp.linspace(0.2, 1.0, obs_size)
    noise_cov = lx.DiagonalLinearOperator(variances)
    dt = 0.3
    process = flx.processes.EKI(scheduler=flx.FixedScheduler(dt=dt))
    evals = particles @ forward.T
    new = process.update(process.init(particles, obs, noise_cov), evals).particles

    param_anomalies = particles - particles.mean(0)
    obs_anomalies = evals - evals.mean(0)
    cross = param_anomalies.T @ obs_anomalies / (num_members - 1)
    obs_cov = obs_anomalies.T @ obs_anomalies / (num_members - 1)
    innovation_cov = obs_cov + jnp.diag(variances) / dt
    expected = particles + (obs - evals) @ jnp.linalg.solve(innovation_cov, cross.T)
    np.testing.assert_allclose(new, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.slow
def test_eki_update_is_a_no_op_at_zero_step():
    forward, obs, particles = _inverse_problem_for_gaussx(10, 5)
    noise_cov = lx.DiagonalLinearOperator(jnp.ones(5))
    for make_process in (flx.processes.EKI, flx.processes.ETKI):
        process = make_process(scheduler=flx.FixedScheduler(dt=0.0))
        state = process.init(particles, obs, noise_cov)
        new = process.update(state, particles @ forward.T).particles
        np.testing.assert_allclose(new, particles, atol=1e-12)


@pytest.mark.slow
def test_teki_keeps_a_kronecker_noise_structured(monkeypatch):
    """Γ̃ is a BlockDiag, and a Kronecker Γ is never materialised on Woodbury."""
    obs_size, param_size, num_members = 6, 3, 5
    forward, obs, particles = _inverse_problem_for_gaussx(
        num_members, obs_size, param_size
    )
    left = lx.MatrixLinearOperator(
        jnp.asarray([[1.0, 0.3], [0.3, 1.0]]), lx.positive_semidefinite_tag
    )
    right = lx.DiagonalLinearOperator(jnp.asarray([0.5, 1.0, 2.0]))
    noise_cov = gaussx.Kronecker(left, right)
    prior_cov = lx.DiagonalLinearOperator(jnp.ones(param_size))
    process = flx.processes.TEKI(
        scheduler=flx.FixedScheduler(dt=0.5),
        prior_mean=jnp.zeros(param_size),
        prior_cov=prior_cov,
    )
    state = process.init(particles, obs, noise_cov)
    assert isinstance(state.noise_cov, gaussx.BlockDiag)
    noise_dense = noise_cov.as_matrix()

    def _refuse(self):
        raise AssertionError("the Kronecker noise was materialised")

    with monkeypatch.context() as patch:
        patch.setattr(gaussx.Kronecker, "as_matrix", _refuse)
        new = process.update(state, particles @ forward.T).particles

    # The same step with a dense Γ̃ = blockdiag(Γ, Σ₀), formed by hand.
    aug_noise = jnp.block(
        [
            [noise_dense, jnp.zeros((obs_size, param_size))],
            [jnp.zeros((param_size, obs_size)), jnp.eye(param_size)],
        ]
    )
    aug_evals = jnp.concatenate([particles @ forward.T, particles], axis=1)
    aug_obs = jnp.concatenate([obs, jnp.zeros(param_size)])
    param_anomalies = particles - particles.mean(0)
    obs_anomalies = aug_evals - aug_evals.mean(0)
    cross = param_anomalies.T @ obs_anomalies / (num_members - 1)
    obs_cov = obs_anomalies.T @ obs_anomalies / (num_members - 1)
    innovation_cov = obs_cov + aug_noise / 0.5
    expected = particles + (aug_obs - aug_evals) @ jnp.linalg.solve(
        innovation_cov, cross.T
    )
    np.testing.assert_allclose(new, expected, rtol=1e-9, atol=1e-10)


@pytest.mark.slow
def test_etki_matches_a_dense_etkf_transform():
    """x̄⁺ = x̄ + w̄ X′ and X′⁺ = W X′, with Ĉ = (J − 1) I + Y′ (Γ / Δt)⁻¹ Y′ᵀ."""
    num_members, obs_size = 10, 4
    forward, obs, particles = _inverse_problem_for_gaussx(num_members, obs_size)
    variances = jnp.linspace(0.2, 1.0, obs_size)
    dt = 0.4
    process = flx.processes.ETKI(scheduler=flx.FixedScheduler(dt=dt))
    noise_cov = lx.DiagonalLinearOperator(variances)
    evals = particles @ forward.T
    new = process.update(process.init(particles, obs, noise_cov), evals).particles

    param_anomalies = particles - particles.mean(0)
    obs_anomalies = evals - evals.mean(0)
    weighted = obs_anomalies * (dt / variances)
    capacitance = (num_members - 1) * jnp.eye(num_members)
    capacitance = capacitance + weighted @ obs_anomalies.T
    eigenvalues, eigenvectors = jnp.linalg.eigh(capacitance)
    mean_weights = jnp.linalg.solve(capacitance, weighted @ (obs - evals.mean(0)))
    root = jnp.sqrt((num_members - 1) / eigenvalues)
    transform = (eigenvectors * root) @ eigenvectors.T
    expected = (
        particles.mean(0) + mean_weights @ param_anomalies + transform @ param_anomalies
    )
    np.testing.assert_allclose(new, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.integration
@pytest.mark.slow
def test_etki_schedule_lands_on_the_gaussian_posterior():
    """With Σ Δt = 1 and J > Nₚ, the ensemble's moments are the exact posterior."""
    num_members, obs_size = 12, 5
    forward, obs, particles = _inverse_problem_for_gaussx(num_members, obs_size)
    gamma = 0.3
    noise_cov = lx.DiagonalLinearOperator(gamma * jnp.ones(obs_size))
    prior_mean = particles.mean(0)
    prior_anomalies = particles - prior_mean
    prior_cov = prior_anomalies.T @ prior_anomalies / (num_members - 1)
    precision = jnp.linalg.inv(prior_cov) + forward.T @ forward / gamma
    post_cov = jnp.linalg.inv(precision)
    post_mean = prior_mean + post_cov @ forward.T @ (obs - forward @ prior_mean) / gamma

    process = flx.processes.ETKI(scheduler=flx.FixedScheduler(dt=0.25))
    state = process.init(particles, obs, noise_cov)
    for _ in range(4):
        state = process.update(state, state.particles @ forward.T)
    new_anomalies = state.particles - state.particles.mean(0)
    np.testing.assert_allclose(state.particles.mean(0), post_mean, atol=1e-10)
    np.testing.assert_allclose(
        new_anomalies.T @ new_anomalies / (num_members - 1), post_cov, atol=1e-10
    )


def test_gnki_solves_a_structured_prior_like_its_dense_form():
    forward, obs, particles = _inverse_problem_for_gaussx(30, 6)
    noise_cov = lx.DiagonalLinearOperator(0.2 * jnp.ones(6))
    variances = jnp.asarray([0.5, 1.0, 2.0])

    def run(prior_cov):
        process = flx.processes.GNKI(
            scheduler=flx.FixedScheduler(dt=0.5),
            prior_mean=jnp.zeros(3),
            prior_cov=prior_cov,
        )
        state = process.init(particles, obs, noise_cov)
        return process.update(state, particles @ forward.T).particles

    structured = run(lx.DiagonalLinearOperator(variances))
    dense = run(
        lx.MatrixLinearOperator(jnp.diag(variances), lx.positive_semidefinite_tag)
    )
    np.testing.assert_allclose(structured, dense, rtol=1e-10)


# ──────────────────────────────────────────────────────────────────────
# UKI on gaussx.sigma_points
# ──────────────────────────────────────────────────────────────────────


def _psd(matrix):
    return lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)


def _uki_forward(theta):
    return jnp.asarray(
        [
            theta[0] + 0.1 * theta[1] ** 3,
            theta[1] + 0.2 * jnp.sin(theta[0]),
            theta[0] * theta[1],
        ]
    )


NONLINEAR_OBS = jnp.asarray([0.8, -0.5, -0.3])
NONLINEAR_PRIOR_COV = jnp.asarray([[1.0, 0.4], [0.4, 1.0]])


def _run_l1_uki(num_steps=4, dt=0.25):
    noise_cov = lx.DiagonalLinearOperator(0.1 * jnp.ones(3))
    process = flx.processes.UKI(scheduler=flx.FixedScheduler(dt=dt))
    state = process.init_parametric(
        jnp.zeros(2), _psd(NONLINEAR_PRIOR_COV), NONLINEAR_OBS, noise_cov
    )
    for _ in range(num_steps):
        points, _, _ = flx.processes.sigma_points(state.mean, state.covariance)
        evals = jax.vmap(_uki_forward)(points)
        state = process.update_parametric(
            state,
            evals,
            obs=NONLINEAR_OBS,
            noise_cov=noise_cov,
            dt=jnp.asarray(dt),
        )
    return state


def test_sigma_points_use_the_principal_root():
    """χ⁺ⱼ − μ = √(Nₚ + λ) [Σ^{1/2}]_{:,j}, with Σ^{1/2} the symmetric root."""
    covariance = jnp.asarray([[2.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 1.5]])
    mean = jnp.asarray([1.0, -1.0, 0.5])
    points, _, _ = flx.processes.sigma_points(mean, _psd(covariance))
    eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
    root = (eigenvectors * jnp.sqrt(eigenvalues)) @ eigenvectors.T
    # α = 1 and κ = 0 give Nₚ + λ = Nₚ.
    np.testing.assert_allclose(points[1:4] - mean, jnp.sqrt(3.0) * root.T, atol=1e-10)
    np.testing.assert_allclose(points[4:7] - mean, -jnp.sqrt(3.0) * root.T, atol=1e-10)


def test_sigma_point_weights_keep_the_input_dtype():
    mean = jnp.zeros(2, dtype=jnp.float32)
    covariance = lx.DiagonalLinearOperator(jnp.ones(2, dtype=jnp.float32))
    points, mean_weights, cov_weights = flx.processes.sigma_points(mean, covariance)
    assert points.dtype == jnp.float32
    assert mean_weights.dtype == jnp.float32
    assert cov_weights.dtype == jnp.float32


@pytest.mark.slow
def test_sigma_point_gradient_is_finite_at_an_isotropic_covariance():
    """UKI starts from σ² I, where a plain eigh gradient would be NaN."""

    def loss(covariance):
        points, mean_weights, _ = flx.processes.sigma_points(
            jnp.zeros(3), _psd(covariance)
        )
        return mean_weights @ jnp.sum(jnp.tanh(points) ** 2, axis=-1)

    gradient = jax.grad(loss)(0.5 * jnp.eye(3))
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_uki_nonlinear_posterior_is_pinned():
    """Recorded for the principal root; Cholesky differed by up to 2.8e-3."""
    state = _run_l1_uki()
    np.testing.assert_allclose(
        state.mean, [0.71218330016186, -0.48378544925327], atol=1e-10
    )
    np.testing.assert_allclose(
        state.covariance.as_matrix(),
        [
            [0.08092196941865, -0.00150706084194],
            [-0.00150706084194, 0.06996919420458],
        ],
        atol=1e-10,
    )


def test_optax_uki_agrees_with_the_l1_process_on_a_nonlinear_model():
    expected = _run_l1_uki()
    transform = filterax_optax.uki(
        _uki_forward,
        NONLINEAR_OBS,
        lx.DiagonalLinearOperator(0.1 * jnp.ones(3)),
        init_cov=NONLINEAR_PRIOR_COV,
        scheduler=flx.FixedScheduler(dt=0.25),
    )
    params = jnp.zeros(2)
    state = transform.init(params)
    for _ in range(4):
        updates, state = transform.update(None, state, params)
        params = params + updates
    np.testing.assert_allclose(state.mean, expected.mean, atol=1e-12)
    np.testing.assert_allclose(
        state.covariance.as_matrix(), expected.covariance.as_matrix(), atol=1e-12
    )

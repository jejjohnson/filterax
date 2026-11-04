"""Tests for the Wave 5.A ensemble smoothers.

Covers ``EnKS``, ``EnsembleRTS``, ``FixedLagSmoother``,
``EnsembleSqrtSmoother``, and ``IES``. All tests are *algebraic* — they
pin each smoother to the formula in the design doc on tiny synthetic
histories rather than chasing Monte-Carlo convergence against an
analytic Kalman RTS smoother.
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


# ──────────────────────────────────────────────────────────────────────
# EnsembleSqrtSmoother — algebraic correctness & invariants
# ──────────────────────────────────────────────────────────────────────


def test_sqrt_smoother_mean_matches_enks(getkey):
    """``EnsembleSqrtSmoother`` decomposes mean + perturbation updates
    explicitly; the mean update is identical to ``EnKS``."""
    forecast, analysis = _synth_history(getkey, T=4)
    enks = flx.EnKS().smooth(forecast, analysis)
    sqrt_s = flx.EnsembleSqrtSmoother().smooth(forecast, analysis)
    enks_means = np.asarray(enks.smoothed_history).mean(axis=1)
    sqrt_means = np.asarray(sqrt_s.smoothed_history).mean(axis=1)
    np.testing.assert_allclose(sqrt_means, enks_means, atol=1e-10)


def test_sqrt_smoother_preserves_final_time_analysis(getkey):
    forecast, analysis = _synth_history(getkey)
    sqrt_s = flx.EnsembleSqrtSmoother().smooth(forecast, analysis)
    np.testing.assert_allclose(
        np.asarray(sqrt_s.smoothed_history[-1]),
        np.asarray(analysis[-1]),
        atol=1e-12,
    )


def test_sqrt_smoother_perturbations_in_analysis_column_space(getkey):
    """The sqrt smoother applies ``Wᵀ A`` to produce smoothed anomalies,
    so the smoothed perts live in the row span of the analysis anomalies.
    The unobserved-by-A directions are unchanged from the analysis."""
    forecast, analysis = _synth_history(getkey, T=3, N_e=5, N_x=8)
    sqrt_s = flx.EnsembleSqrtSmoother().smooth(forecast, analysis)
    # Anomalies at each t.
    sm = np.asarray(sqrt_s.smoothed_history)
    an = np.asarray(analysis)
    for t in range(sm.shape[0]):
        sm_perts = sm[t] - sm[t].mean(axis=0)
        an_perts = an[t] - an[t].mean(axis=0)
        # Smoothed perts must live in the row span of the analysis
        # anomalies — directions orthogonal to that span are unchanged
        # from the analysis. Take the null space of an_perts^T an_perts
        # and check the projection of sm_perts onto it is ~0.
        rank = min(an_perts.shape[0] - 1, an_perts.shape[1])
        if rank < an_perts.shape[1]:
            _eigvals, eigvecs_xx = np.linalg.eigh(an_perts.T @ an_perts)
            null_basis = eigvecs_xx[:, : an_perts.shape[1] - rank]
            projected_null = sm_perts @ null_basis
            np.testing.assert_allclose(projected_null, 0.0, atol=1e-9)


def test_sqrt_smoother_t_equals_one_is_identity(getkey):
    forecast, analysis = _synth_history(getkey, T=1)
    out = flx.EnsembleSqrtSmoother().smooth(forecast, analysis)
    np.testing.assert_allclose(
        np.asarray(out.smoothed_history), np.asarray(analysis), atol=1e-12
    )


def test_sqrt_smoother_zero_innovation_is_identity(getkey):
    """``forecast == analysis`` → smoother gain has no innovation to
    propagate and the smoothed history equals the analysis history."""
    _, analysis = _synth_history(getkey, T=4, N_e=6, N_x=3)
    out = flx.EnsembleSqrtSmoother().smooth(analysis, analysis)
    np.testing.assert_allclose(
        np.asarray(out.smoothed_history), np.asarray(analysis), atol=1e-9
    )


def test_sqrt_smoother_under_jit(getkey):
    forecast, analysis = _synth_history(getkey)
    smoother = flx.EnsembleSqrtSmoother()
    eager = smoother.smooth(forecast, analysis).smoothed_history
    jitted = jax.jit(lambda f, a: smoother.smooth(f, a).smoothed_history)(
        forecast, analysis
    )
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), atol=1e-12)


# ──────────────────────────────────────────────────────────────────────
# IES — Iterative Ensemble Smoother
# ──────────────────────────────────────────────────────────────────────


def _ref_ies_step(
    particles_i: np.ndarray,
    particles_0: np.ndarray,
    obs: np.ndarray,
    R_diag: np.ndarray,
    forward: np.ndarray,
    step_size: float,
    y_pert: np.ndarray,
) -> np.ndarray:
    """Plain-numpy reference for one IES Chen-Oliver step (linear ``G``).

    Uses the dense state-space gain form so any agreement between the
    package's structural Woodbury solve and this dense reference is a
    real correctness check, not the same code in numpy.
    """
    evals = particles_i @ forward.T  # (J, N_d)
    centred_p = particles_i - particles_i.mean(axis=0)
    centred_g = evals - evals.mean(axis=0)
    J = particles_i.shape[0]
    C_theta_G = centred_p.T @ centred_g / (J - 1)  # (N_p, N_d)
    C_GG = centred_g.T @ centred_g / (J - 1)  # (N_d, N_d)
    S = C_GG + np.diag(R_diag)  # (N_d, N_d)
    innovations = y_pert - evals  # (J, N_d)
    S_inv_innov = innovations @ np.linalg.inv(S).T  # (J, N_d)
    K_r = S_inv_innov @ C_theta_G.T  # (J, N_p)
    target = particles_0 + K_r
    return (1.0 - step_size) * particles_i + step_size * target


def test_ies_single_step_matches_numpy_reference(getkey):
    """One IES iteration on a linear inverse problem must equal the
    Chen-Oliver formula computed in numpy with the same PRNG draws."""
    G = jnp.asarray([[1.0, 0.5], [-0.3, 1.2], [0.7, 0.1]])
    R_diag = jnp.asarray([0.1, 0.1, 0.1])
    R = lx.DiagonalLinearOperator(R_diag)
    y = jnp.asarray([0.5, -0.3, 0.2])
    init = jr.normal(getkey(), (12, 2))

    seed = 11
    ies = flx.IES(n_iterations=1, step_size=1.0, seed=seed)
    out = ies.solve(init, y, R, lambda theta: G @ theta)

    # Reproduce the package's perturbed-observation draw exactly.
    y_pert = flx.perturbed_observations(jr.PRNGKey(seed), y, R, init.shape[0])

    expected = _ref_ies_step(
        np.asarray(init),
        np.asarray(init),
        np.asarray(y),
        np.asarray(R_diag),
        np.asarray(G),
        step_size=1.0,
        y_pert=np.asarray(y_pert),
    )
    np.testing.assert_allclose(np.asarray(out.particles), expected, atol=1e-9)


def _linear_gaussian_problem(n_members: int, n_params: int = 2):
    """Linear G(θ) = A θ, prior 𝒩(0, I), noise 𝒩(0, 0.1 I), with exact posterior."""
    if n_params == 2:
        A = jnp.asarray([[1.0, 0.5], [0.0, 1.0], [1.0, -1.0]])
        y = jnp.asarray([1.0, 2.0, 0.5])
    else:
        A = jr.normal(jr.key(3), (3, n_params))
        y = jr.normal(jr.key(4), (3,))
    noise_variance = 0.1
    R = lx.DiagonalLinearOperator(noise_variance * jnp.ones(3))
    prior = jr.normal(jr.key(0), (n_members, n_params))
    # Σ_post = (I + Aᵀ A / σ²)⁻¹, μ_post = Σ_post Aᵀ y / σ²
    posterior_covariance = jnp.linalg.inv(jnp.eye(n_params) + A.T @ A / noise_variance)
    posterior_mean = posterior_covariance @ (A.T @ y) / noise_variance
    return A, y, R, prior, posterior_mean, posterior_covariance


@pytest.mark.parametrize("n_iterations", [1, 2, 5, 20])
def test_ies_converges_to_linear_gaussian_posterior_mean(n_iterations):
    """Regression for #117: the iteration fell back toward the prior mean."""
    A, y, R, prior, posterior_mean, _ = _linear_gaussian_problem(4000)
    out = flx.IES(n_iterations=n_iterations).solve(prior, y, R, lambda th: A @ th)
    error = jnp.max(jnp.abs(out.particles.mean(axis=0) - posterior_mean))
    assert error < 0.02


def test_ies_matches_linear_gaussian_posterior_covariance():
    A, y, R, prior, _, posterior_covariance = _linear_gaussian_problem(4000)
    out = flx.IES(n_iterations=5).solve(prior, y, R, lambda th: A @ th)
    sample_covariance = jnp.cov(out.particles, rowvar=False)
    relative_error = jnp.abs(sample_covariance - posterior_covariance) / jnp.abs(
        posterior_covariance
    )
    assert jnp.max(relative_error) < 0.05


@pytest.mark.parametrize(("n_members", "n_params"), [(50, 2), (8, 20)])
def test_ies_first_iterate_is_a_fixed_point_for_linear_forward(n_members, n_params):
    """For linear G every member's innovation is d^j − A θ_0^j at all iterations,
    with more parameters than members too."""
    A, y, R, prior, _, _ = _linear_gaussian_problem(n_members, n_params)
    forward = lambda th: A @ th
    first = flx.IES(n_iterations=1).solve(prior, y, R, forward).particles
    later = flx.IES(n_iterations=10).solve(prior, y, R, forward).particles
    np.testing.assert_allclose(np.asarray(later), np.asarray(first), atol=1e-7)


def test_ies_damped_iteration_reaches_the_undamped_fixed_point():
    A, y, R, prior, _, _ = _linear_gaussian_problem(100)
    forward = lambda th: A @ th
    undamped = flx.IES(n_iterations=1).solve(prior, y, R, forward).particles
    damped = flx.IES(n_iterations=40, step_size=0.5).solve(prior, y, R, forward)
    np.testing.assert_allclose(
        np.asarray(damped.particles), np.asarray(undamped), atol=1e-7
    )


def test_ies_recovers_nonlinear_posterior_against_grid_oracle():
    """G(θ) = θ + 0.3 θ³ per component, so the posterior factorises on a 1-D grid."""
    truth = jnp.asarray([1.2, -0.7])
    forward = lambda th: th + 0.3 * th**3
    y = forward(truth)
    noise_variance = 0.01
    R = lx.DiagonalLinearOperator(noise_variance * jnp.ones(2))
    prior = jr.normal(jr.key(0), (2000, 2))

    grid = np.linspace(-4.0, 4.0, 40_001)
    log_density = (
        -0.5 * grid[None, :] ** 2
        - 0.5
        * (np.asarray(y)[:, None] - np.asarray(forward(grid))) ** 2
        / noise_variance
    )
    weights = np.exp(log_density - log_density.max(axis=1, keepdims=True))
    weights = weights / weights.sum(axis=1, keepdims=True)
    exact_mean = (weights * grid).sum(axis=1)
    exact_std = np.sqrt((weights * (grid - exact_mean[:, None]) ** 2).sum(axis=1))

    out = flx.IES(n_iterations=20).solve(prior, y, R, forward).particles
    np.testing.assert_allclose(np.asarray(out.mean(axis=0)), exact_mean, atol=0.01)
    np.testing.assert_allclose(np.asarray(out.std(axis=0)), exact_std, rtol=0.1)


@pytest.mark.parametrize(("n_members", "n_params"), [(50, 2), (8, 20)])
def test_ies_gradient_matches_central_differences(n_members, n_params):
    A, y, R, prior, _, _ = _linear_gaussian_problem(n_members, n_params)
    forward = lambda th: A @ th + 0.05 * jnp.sin(A @ th)

    def loss(obs):
        out = flx.IES(n_iterations=3).solve(prior, obs, R, forward)
        return jnp.sum(out.particles**2)

    gradient = jax.grad(loss)(y)
    h = 1e-5
    central = jnp.stack(
        [(loss(y.at[i].add(h)) - loss(y.at[i].add(-h))) / (2 * h) for i in range(3)]
    )
    np.testing.assert_allclose(np.asarray(gradient), np.asarray(central), rtol=1e-6)


def test_ies_step_size_zero_is_rejected():
    with pytest.raises(ValueError, match=r"step_size must lie in \(0, 1\]"):
        flx.IES(n_iterations=5, step_size=0.0)


def test_ies_step_size_above_one_is_rejected():
    with pytest.raises(ValueError, match=r"step_size must lie in \(0, 1\]"):
        flx.IES(n_iterations=5, step_size=1.5)


def test_ies_rejects_zero_iterations():
    with pytest.raises(ValueError, match="n_iterations must be a positive int"):
        flx.IES(n_iterations=0)


def test_ies_rejects_degenerate_ensemble():
    """``J < 2`` would divide by zero in the Bessel-corrected sample cov."""
    G = jnp.asarray([[1.0, 0.0]])
    R = lx.DiagonalLinearOperator(jnp.asarray([0.1]))
    y = jnp.zeros(1)
    init = jnp.zeros((1, 2))
    with pytest.raises(ValueError, match="at least 2 ensemble members"):
        flx.IES(n_iterations=2).solve(init, y, R, lambda theta: G @ theta)


def test_ies_explicit_base_key_overrides_seed(getkey):
    """``base_key`` takes precedence over ``seed`` — the two runs with the
    same explicit key and different seeds must agree."""
    G = jnp.eye(2)
    R = lx.DiagonalLinearOperator(jnp.ones(2))
    y = jnp.asarray([0.5, -0.3])
    init = jr.normal(getkey(), (10, 2))
    forward = lambda t: G @ t
    key = jr.PRNGKey(42)
    out_a = flx.IES(n_iterations=3, seed=0, base_key=key).solve(init, y, R, forward)
    out_b = flx.IES(n_iterations=3, seed=999, base_key=key).solve(init, y, R, forward)
    np.testing.assert_array_equal(
        np.asarray(out_a.particles), np.asarray(out_b.particles)
    )


def test_ies_solve_under_jit(getkey):
    G = jnp.eye(2)
    R = lx.DiagonalLinearOperator(jnp.ones(2))
    y = jnp.zeros(2)
    init = jr.normal(getkey(), (10, 2))
    forward = lambda t: G @ t
    ies = flx.IES(n_iterations=3, seed=1)
    eager = ies.solve(init, y, R, forward).particles
    jitted = jax.jit(lambda p: ies.solve(p, y, R, forward).particles)(init)
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), atol=1e-12)

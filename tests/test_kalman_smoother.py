"""``KalmanSmoother`` against the dense Gaussian conditional.

An Ornstein–Uhlenbeck prior on irregular times t₁ … t_T has covariance
σ² exp(−|tᵢ − tⱼ| / ℓ), and on a gap Δt it discretises to
F = exp(−Δt / ℓ), Σ_η = σ² (1 − exp(−2 Δt / ℓ)). The smoothed marginals
must equal the conditional of the stacked x₁:T given y₁:T, and the
log-likelihood the Gaussian marginal of the observed entries. x64 is on
for the whole suite (``conftest.py``).
"""

from __future__ import annotations

import gaussx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import filterax as flx


TIMES = np.array([0.0, 0.3, 1.1, 1.4, 2.9, 3.0, 4.2, 5.5])
OBS = np.array([0.4, 0.1, -0.6, -0.2, 0.9, 1.2, 0.3, -0.5])
OBS_VAR = 0.05


def _ou_smoother(sigma, length, times=TIMES):
    gaps = jnp.diff(jnp.asarray(times))
    decay = jnp.exp(-gaps / length)
    return flx.KalmanSmoother(
        # F = e^{−Δt/ℓ}, Σ_η = σ² (1 − e^{−2Δt/ℓ})
        transition=decay[:, None, None],
        process_noise=(sigma**2 * (1 - decay**2))[:, None, None],
        obs_model=jnp.ones((1, 1)),
        obs_noise=OBS_VAR * jnp.ones((1, 1)),
        init_mean=jnp.zeros(1),
        init_cov=sigma**2 * jnp.ones((1, 1)),
    )


def _dense_conditional(prior, observed):
    """Mean, marginal variance and log-likelihood of x₁:T given the observed y."""
    H = jnp.eye(prior.shape[0])[observed]
    y = jnp.asarray(OBS)[observed]
    obs_cov = H @ prior @ H.T + OBS_VAR * jnp.eye(int(observed.sum()))
    gain = prior @ H.T @ jnp.linalg.inv(obs_cov)
    mean = gain @ y
    cov = prior - gain @ H @ prior
    log_likelihood = jax.scipy.stats.multivariate_normal.logpdf(
        y, jnp.zeros_like(y), obs_cov
    )
    return mean, jnp.diag(cov), log_likelihood


def _ou_prior(sigma, length, times=TIMES):
    times = jnp.asarray(times)
    return sigma**2 * jnp.exp(-jnp.abs(times[:, None] - times[None, :]) / length)


def _obs():
    return jnp.asarray(OBS)[:, None]


def test_matches_the_dense_conditional():
    result = _ou_smoother(0.8, 1.5).smooth(_obs())
    mean, variance, log_likelihood = _dense_conditional(
        _ou_prior(0.8, 1.5), np.ones(8, bool)
    )
    np.testing.assert_allclose(result.smoothed_means[:, 0], mean, rtol=1e-10)
    np.testing.assert_allclose(result.smoothed_covs[:, 0, 0], variance, rtol=1e-10)
    np.testing.assert_allclose(result.log_likelihood, log_likelihood, rtol=1e-10)
    # the last filtered marginal already conditions on every observation
    np.testing.assert_allclose(result.filtered_means[-1, 0], mean[-1], rtol=1e-10)


def test_from_sde_matches_the_hand_discretisation():
    sde = gaussx.MaternSDE(
        variance=jnp.asarray(0.64), lengthscale=jnp.asarray(1.5), order=0
    )
    smoother = flx.KalmanSmoother.from_sde(
        sde,
        jnp.asarray(TIMES),
        obs_model=jnp.ones((1, 1)),
        obs_noise=OBS_VAR * jnp.ones((1, 1)),
    )
    np.testing.assert_allclose(
        smoother.smooth(_obs()).smoothed_means,
        _ou_smoother(0.8, 1.5).smooth(_obs()).smoothed_means,
        rtol=1e-10,
    )


def test_from_sde_matern32_matches_the_dense_gp():
    # Matérn-3/2: k(τ) = σ² (1 + √3 τ / ℓ) exp(−√3 τ / ℓ), state (f, f′)
    variance, length = 0.64, 1.5
    sde = gaussx.MaternSDE(
        variance=jnp.asarray(variance), lengthscale=jnp.asarray(length), order=1
    )
    smoother = flx.KalmanSmoother.from_sde(
        sde,
        jnp.asarray(TIMES),
        obs_model=jnp.array([[1.0, 0.0]]),
        obs_noise=OBS_VAR * jnp.ones((1, 1)),
    )
    times = jnp.asarray(TIMES)
    scaled = jnp.sqrt(3.0) * jnp.abs(times[:, None] - times[None, :]) / length
    prior = variance * (1 + scaled) * jnp.exp(-scaled)
    mean, variance_f, log_likelihood = _dense_conditional(prior, np.ones(8, bool))
    result = smoother.smooth(_obs())
    np.testing.assert_allclose(result.smoothed_means[:, 0], mean, rtol=1e-8)
    np.testing.assert_allclose(result.smoothed_covs[:, 0, 0], variance_f, rtol=1e-8)
    np.testing.assert_allclose(result.log_likelihood, log_likelihood, rtol=1e-8)


def test_one_time_is_the_blue_analysis():
    B = jnp.array([[1.0, 0.3], [0.3, 0.5]])
    H = jnp.array([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
    R = jnp.diag(jnp.array([0.1, 0.2, 0.3]))
    y = jnp.array([0.5, -0.2, 0.8])
    smoother = flx.KalmanSmoother(
        transition=jnp.zeros((0, 2, 2)),
        process_noise=jnp.zeros((0, 2, 2)),
        obs_model=H,
        obs_noise=R,
        init_mean=jnp.zeros(2),
        init_cov=B,
    )
    result = smoother.smooth(y[None])
    # x_a = B Hᵀ (H B Hᵀ + R)⁻¹ y, P_a = B − B Hᵀ (H B Hᵀ + R)⁻¹ H B
    gain = B @ H.T @ jnp.linalg.inv(H @ B @ H.T + R)
    np.testing.assert_allclose(result.smoothed_means[0], gain @ y, rtol=1e-10)
    np.testing.assert_allclose(
        result.smoothed_covs[0], B - gain @ H @ B, rtol=1e-10, atol=1e-14
    )


def test_a_fully_masked_time_is_skipped_and_filled():
    observed = np.ones(8, bool)
    observed[3] = False
    # NaN at the missing time, which the filter never reads
    gappy = jnp.where(jnp.asarray(observed), jnp.asarray(OBS), jnp.nan)[:, None]
    result = _ou_smoother(0.8, 1.5).smooth(gappy, jnp.asarray(observed)[:, None])
    # the filter only predicts at the gap: x₄|₃ = F₄ x₃|₃
    decay = jnp.exp(-(TIMES[3] - TIMES[2]) / 1.5)
    np.testing.assert_allclose(
        result.filtered_means[3], decay * result.filtered_means[2], rtol=1e-10
    )
    mean, variance, log_likelihood = _dense_conditional(_ou_prior(0.8, 1.5), observed)
    assert jnp.all(jnp.isfinite(result.smoothed_means))
    np.testing.assert_allclose(result.smoothed_means[:, 0], mean, rtol=1e-10)
    np.testing.assert_allclose(result.smoothed_covs[:, 0, 0], variance, rtol=1e-10)
    np.testing.assert_allclose(result.log_likelihood, log_likelihood, rtol=1e-10)


def test_a_masked_channel_equals_deleting_it():
    # two sensors see the same OU state; the second drops out at times 2 and 5
    smoother = _ou_smoother(0.8, 1.5)
    two_channel = flx.KalmanSmoother(
        transition=smoother.transition,
        process_noise=smoother.process_noise,
        obs_model=jnp.ones((2, 1)),
        obs_noise=OBS_VAR * jnp.eye(2),
        init_mean=smoother.init_mean,
        init_cov=smoother.init_cov,
    )
    channel = np.ones((8, 2), bool)
    channel[[2, 5], 1] = False
    obs = jnp.stack([jnp.asarray(OBS), 0.5 * jnp.asarray(OBS)], axis=-1)
    gappy = jnp.where(jnp.asarray(channel), obs, jnp.nan)
    result = two_channel.smooth(gappy, jnp.asarray(channel))

    # dense oracle on the 8 + 6 observed entries
    prior = _ou_prior(0.8, 1.5)
    H = jnp.concatenate([jnp.eye(8), jnp.eye(8)[channel[:, 1]]])
    y = jnp.concatenate([obs[:, 0], obs[channel[:, 1], 1]])
    obs_cov = H @ prior @ H.T + OBS_VAR * jnp.eye(14)
    gain = prior @ H.T @ jnp.linalg.inv(obs_cov)
    log_likelihood = jax.scipy.stats.multivariate_normal.logpdf(
        y, jnp.zeros(14), obs_cov
    )
    np.testing.assert_allclose(result.smoothed_means[:, 0], gain @ y, rtol=1e-10)
    np.testing.assert_allclose(
        result.smoothed_covs[:, 0, 0], jnp.diag(prior - gain @ H @ prior), rtol=1e-10
    )
    np.testing.assert_allclose(result.log_likelihood, log_likelihood, rtol=1e-10)


def test_jit_and_vmap_match_a_loop():
    smoother = _ou_smoother(0.8, 1.5)
    obs = jnp.asarray(OBS)
    rows = jnp.stack([obs, -obs, 2.0 * obs])[:, :, None]
    batched = jax.jit(jax.vmap(smoother.smooth))(rows)
    for i in range(3):
        one = smoother.smooth(rows[i])
        np.testing.assert_allclose(
            batched.smoothed_means[i], one.smoothed_means, rtol=1e-10
        )
        np.testing.assert_allclose(
            batched.log_likelihood[i], one.log_likelihood, rtol=1e-10
        )


def test_gradients_match_central_differences():
    def loss(params):
        sigma, length = params
        result = _ou_smoother(sigma, length).smooth(_obs())
        return jnp.sum(result.smoothed_means**2) - result.log_likelihood

    params = jnp.array([0.8, 1.5])
    grad = jax.grad(loss)(params)
    step = 1e-6
    for i in range(2):
        shift = jnp.zeros(2).at[i].set(step)
        central = (loss(params + shift) - loss(params - shift)) / (2 * step)
        np.testing.assert_allclose(grad[i], central, rtol=1e-6)


def test_a_wrong_time_axis_is_rejected():
    smoother = _ou_smoother(0.8, 1.5)
    with pytest.raises(ValueError, match="transition must stack 6"):
        smoother.smooth(_obs()[:7])


def test_a_field_that_is_not_a_matrix_is_rejected():
    with pytest.raises(ValueError, match="obs_model"):
        flx.KalmanSmoother(
            transition=jnp.eye(1),
            process_noise=jnp.eye(1),
            obs_model=jnp.ones(1),
            obs_noise=jnp.eye(1),
            init_mean=jnp.zeros(1),
            init_cov=jnp.eye(1),
        )


def test_exported_from_the_smoothers_namespace():
    assert flx.smoothers.KalmanSmoother is flx.KalmanSmoother
    assert flx.smoothers.KalmanSmoothingResult is flx.KalmanSmoothingResult

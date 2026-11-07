"""Smoothers against the filter, and a filter-smoother-training workflow.

Both run on one linear-Gaussian model, where the exact Kalman filter and RTS
smoother give the improvement to expect.
"""

from __future__ import annotations

import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import optax
import pytest

import filterax as flx


pytestmark = [pytest.mark.integration, pytest.mark.slow]

TRANSITION = jnp.array(
    [
        [0.95, 0.1, 0.0, 0.0],
        [-0.1, 0.95, 0.0, 0.0],
        [0.0, 0.0, 0.9, 0.2],
        [0.0, 0.0, -0.2, 0.9],
    ]
)
PROCESS_VARIANCE = 0.1
OBS_MAP = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
OBS_VARIANCE = 0.25
OBS_NOISE = lx.DiagonalLinearOperator(OBS_VARIANCE * jnp.ones(2))
NUM_TIMES, NUM_MEMBERS = 40, 100


def _simulate(seed):
    """A truth and its observations, shapes (T, N_x) and (T, N_y)."""
    keys = jr.split(jr.key(seed), 2 * NUM_TIMES + 1)
    state = jr.normal(keys[0], (4,))
    truth, observations = [], []
    for time in range(NUM_TIMES):
        process = jnp.sqrt(PROCESS_VARIANCE) * jr.normal(keys[2 * time + 1], (4,))
        state = TRANSITION @ state + process
        noise = jnp.sqrt(OBS_VARIANCE) * jr.normal(keys[2 * time + 2], (2,))
        truth.append(state)
        observations.append(OBS_MAP @ state + noise)
    return jnp.stack(truth), jnp.stack(observations)


def _filter_histories(seed, observations, scale=1.0):
    """Forecast and analysis ensembles from an ETKF, shapes (T, N_e, N_x).

    The forecast carries the model's process noise. A smoother with none takes
    the model as exact and propagates its errors backwards through the inverse
    dynamics, which on this truth does worse than the filter.
    """
    keys = jr.split(jr.key(1000 + seed), NUM_TIMES + 1)
    members = jr.normal(keys[0], (NUM_MEMBERS, 4))
    filter_ = flx.filters.ETKF()
    forecasts, analyses = [], []
    for time in range(NUM_TIMES):
        process = jnp.sqrt(PROCESS_VARIANCE) * jr.normal(
            keys[time + 1], (NUM_MEMBERS, 4)
        )
        forecast = scale * members @ TRANSITION.T + process
        members = filter_.analysis(
            forecast, observations[time], lambda x: OBS_MAP @ x, OBS_NOISE
        ).particles
        forecasts.append(forecast)
        analyses.append(members)
    return jnp.stack(forecasts), jnp.stack(analyses)


def _squared_error(history, truth):
    """Mean squared error of the ensemble mean, before the final time.

    Every smoother leaves the final time at the analysis, so it says nothing.
    """
    return jnp.mean((jnp.mean(history[:-1], axis=1) - truth[:-1]) ** 2)


def _exact_variance_ratio(observations):
    """Smoothed over filtered variance from the exact Kalman smoother."""
    state = gaussx.kalman_filter(
        TRANSITION,
        OBS_MAP,
        PROCESS_VARIANCE * jnp.eye(4),
        OBS_VARIANCE * jnp.eye(2),
        observations,
        jnp.zeros(4),
        jnp.eye(4),
    )
    _, smoothed_covariances = gaussx.rts_smoother(state, TRANSITION)
    middle = NUM_TIMES // 2
    smoothed_trace = jnp.trace(smoothed_covariances[middle])
    return smoothed_trace / jnp.trace(state.filtered_covs[middle])


@pytest.fixture(scope="module")
def runs():
    """Truth and filter histories on three independent draws."""
    results = []
    for seed in range(3):
        truth, observations = _simulate(seed)
        results.append((truth, observations, *_filter_histories(seed, observations)))
    return results


@pytest.mark.parametrize(
    "smoother",
    [flx.EnKS(), flx.EnsembleRTS(), flx.EnsembleSqrtSmoother()],
    ids=["enks", "ensemble_rts", "sqrt"],
)
def test_a_smoother_beats_the_filter_as_much_as_the_exact_smoother(runs, smoother):
    smoothed_error, filtered_error = 0.0, 0.0
    for truth, _, forecasts, analyses in runs:
        history = smoother.smooth(forecasts, analyses).smoothed_history
        smoothed_error += _squared_error(history, truth)
        filtered_error += _squared_error(analyses, truth)
    ratio = smoothed_error / filtered_error
    expected = _exact_variance_ratio(runs[0][1])
    # Measured at 0.78 against the exact 0.68. The margin covers sampling noise from
    # three draws of 39 times, and 100 members' error in the gain.
    assert ratio < 1.0
    assert jnp.abs(ratio - expected) < 0.2


def test_a_fixed_lag_smoother_lies_between_the_filter_and_the_full_smoother(runs):
    filtered, lagged, full = 0.0, 0.0, 0.0
    for truth, _, forecasts, analyses in runs:
        filtered += _squared_error(analyses, truth)
        lagged += _squared_error(
            flx.FixedLagSmoother(lag=3).smooth(forecasts, analyses).smoothed_history,
            truth,
        )
        full += _squared_error(
            flx.EnKS().smooth(forecasts, analyses).smoothed_history, truth
        )
    assert full < lagged < filtered


class _ScaledDynamics(flx.AbstractDynamics):
    """x ↦ a F x with the known F, for a learnable scalar a."""

    log_scale: jax.Array

    def __call__(self, state, t0, t1):
        return jnp.exp(self.log_scale) * TRANSITION @ state


class _Observe(flx.AbstractObsOperator):
    def __call__(self, state):
        return OBS_MAP @ state


def test_train_filter_and_smooth_end_to_end():
    truth, observations = _simulate(0)
    times = jnp.arange(1.0, NUM_TIMES + 1.0)
    members = jr.normal(jr.key(5), (NUM_MEMBERS, 4))
    # Training runs without process noise, so the spread needs keeping up.
    inflator = flx.RTPS(alpha=0.9)

    def negative_log_likelihood(dynamics):
        result = flx.differentiable_assimilate(
            flx.filters.ETKF(),
            dynamics,
            _Observe(),
            members,
            observations,
            times,
            OBS_NOISE,
            inflator=inflator,
        )
        return -jnp.sum(result.log_likelihoods)

    start = 0.7
    dynamics = _ScaledDynamics(log_scale=jnp.log(jnp.asarray(start)))
    optimizer = optax.adam(0.05)
    state = optimizer.init(eqx.filter(dynamics, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(negative_log_likelihood)(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        return eqx.apply_updates(model, updates), opt_state, loss

    losses = []
    for _ in range(25):
        dynamics, state, loss = step(dynamics, state)
        losses.append(loss)
    learned = jnp.exp(dynamics.log_scale)
    assert losses[-1] < losses[0]
    # The truth has a scale of 1, and this measured 1.01 after 25 steps.
    assert jnp.abs(learned - 1.0) < 0.1

    def errors(scale):
        forecasts, analyses = _filter_histories(0, observations, scale=scale)
        smoothed = flx.EnKS().smooth(forecasts, analyses).smoothed_history
        return _squared_error(analyses, truth), _squared_error(smoothed, truth)

    filtered, smoothed = errors(learned)
    _, smoothed_at_start = errors(start)
    assert smoothed < filtered
    assert smoothed < smoothed_at_start

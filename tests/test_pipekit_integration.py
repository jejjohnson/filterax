"""pipekit-cycle protocol adapters.

Conformance is structural — filterax never imports pipekit — so most
tests exercise the adapter signatures directly. ``pipekit-cycle`` is a
dev-group dependency, so the isinstance checks against its real
runtime-checkable Protocols and the cycled Kalman oracle (roadmap gate
D2) run in CI.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from pipekit import Const
from pipekit_cycle import EnsembleDACycle, protocols
from pipekit_cycle.state import DAState

from filterax.filters import ETKF, EnSRF
from filterax.pipekit import (
    DynamicsForwardModel,
    FilterAnalysisStep,
    LinearizableObsOperator,
)


def _identity_obs(x):
    return x


@pytest.fixture
def members():
    return [
        jnp.array([0.0, 0.0]),
        jnp.array([1.0, 1.0]),
        jnp.array([2.0, 0.5]),
        jnp.array([0.5, 2.0]),
    ]


@pytest.fixture
def obs():
    return jnp.array([1.0, 1.0])


@pytest.fixture
def obs_err_cov():
    return 0.1 * jnp.eye(2)


class TestFilterAnalysisStep:
    def test_signature_matches_analysis_step_protocol(self):
        sig = inspect.signature(FilterAnalysisStep.__call__)
        params = list(sig.parameters.values())[1:]  # drop self
        assert [p.name for p in params] == ["forecast", "obs", "obs_op", "obs_err_cov"]
        assert params[2].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[3].kind is inspect.Parameter.KEYWORD_ONLY

    def test_list_in_list_out(self, members, obs, obs_err_cov):
        step = FilterAnalysisStep(ETKF())
        analysed = step(members, obs, obs_op=_identity_obs, obs_err_cov=obs_err_cov)
        assert isinstance(analysed, list)
        assert len(analysed) == len(members)
        assert all(m.shape == (2,) for m in analysed)

    def test_array_in_array_out(self, members, obs, obs_err_cov):
        particles = jnp.stack(members)
        step = FilterAnalysisStep(EnSRF())
        analysed = step(particles, obs, obs_op=_identity_obs, obs_err_cov=obs_err_cov)
        assert analysed.shape == particles.shape

    def test_matches_direct_filter_analysis(self, members, obs, obs_err_cov):
        particles = jnp.stack(members)
        noise = lx.MatrixLinearOperator(
            obs_err_cov, (lx.symmetric_tag, lx.positive_semidefinite_tag)
        )
        direct = ETKF().analysis(particles, obs, _identity_obs, noise).particles
        adapted = FilterAnalysisStep(ETKF())(
            particles, obs, obs_op=_identity_obs, obs_err_cov=obs_err_cov
        )
        assert jnp.allclose(direct, adapted, atol=1e-6)

    def test_accepts_operator_obs_err_cov(self, members, obs, obs_err_cov):
        op = lx.MatrixLinearOperator(
            obs_err_cov, (lx.symmetric_tag, lx.positive_semidefinite_tag)
        )
        analysed = FilterAnalysisStep(ETKF())(
            members, obs, obs_op=_identity_obs, obs_err_cov=op
        )
        assert len(analysed) == len(members)

    def test_none_obs_err_cov_raises(self, members, obs):
        step = FilterAnalysisStep(ETKF())
        with pytest.raises(ValueError, match="obs_err_cov"):
            step(members, obs, obs_op=_identity_obs, obs_err_cov=None)

    def test_analysis_kwargs_forwarded(self, members, obs, obs_err_cov):
        # ETKF ignores extras per the AbstractSequentialFilter contract.
        step = FilterAnalysisStep(ETKF(), analysis_kwargs={"unused_extra": 0})
        analysed = step(members, obs, obs_op=_identity_obs, obs_err_cov=obs_err_cov)
        assert len(analysed) == len(members)


class TestDynamicsForwardModel:
    def test_step_integrates_autonomously(self):
        fwd = DynamicsForwardModel(lambda x, t0, t1: x + (t1 - t0), dt=0.5)
        out = fwd.step(jnp.array([1.0, 2.0]), fwd.dt)
        assert jnp.allclose(out, jnp.array([1.5, 2.5]))

    def test_protocol_attributes(self):
        fwd = DynamicsForwardModel(lambda x, t0, t1: x, dt=2.0)
        assert fwd.dt == 2.0
        assert fwd.state_signature is None


class TestLinearizableObsOperator:
    def test_call_delegates(self):
        H = LinearizableObsOperator(lambda x: x[:1] ** 2)
        assert jnp.allclose(H(jnp.array([3.0, 4.0])), jnp.array([9.0]))

    def test_linearize_is_jacobian(self):
        H = LinearizableObsOperator(lambda x: x[:1] ** 2)
        jac = H.linearize(jnp.array([3.0, 4.0]))
        assert jnp.allclose(jac, jnp.array([[6.0, 0.0]]))


class TestRuntimeProtocolConformance:
    """Real isinstance checks against pipekit-cycle's Protocols."""

    def test_isinstance_against_pipekit_protocols(self, obs_err_cov):
        assert isinstance(FilterAnalysisStep(ETKF()), protocols.AnalysisStep)
        assert isinstance(
            DynamicsForwardModel(lambda x, t0, t1: x), protocols.ForwardModel
        )
        assert isinstance(
            LinearizableObsOperator(_identity_obs), protocols.ObservationOperator
        )


def _kalman_mean(
    x_bar_f: np.ndarray, P_f: np.ndarray, H: np.ndarray, R: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Kalman analysis mean x̄ᵃ = x̄ᶠ + P H^T (H P H^T + R)⁻¹ (y − H x̄ᶠ)."""
    S = H @ P_f @ H.T + R
    return x_bar_f + P_f @ H.T @ np.linalg.solve(S, y - H @ x_bar_f)


@pytest.mark.parametrize("filt", [ETKF(), EnSRF()], ids=["ETKF", "EnSRF"])
def test_cycled_analysis_mean_matches_kalman(filt):
    """Roadmap gate D2: one EnsembleDACycle step equals the Kalman update.

    Linear 𝓜 and H, no inflation, no localisation: the square-root
    analysis mean must match the Kalman update built from the forecast
    ensemble's own sample mean and covariance (ddof=1, as
    ``tests/test_filters.py::_kalman_reference``) to 1e-6 in float64.
    """
    N_e, N_x, N_y = 20, 4, 3
    rng = np.random.default_rng(0)
    # 𝓜: x ↦ A x, a damped rotation-plus-shear (well conditioned).
    A = 0.9 * np.eye(N_x) + 0.1 * rng.standard_normal((N_x, N_x))
    H = rng.standard_normal((N_y, N_x))  # (N_y, N_x)
    R = np.diag([0.2, 0.5, 0.3])  # (N_y, N_y)
    x0 = rng.standard_normal((N_e, N_x)) + np.arange(N_x)  # (N_e, N_x)
    y = rng.standard_normal(N_y)  # (N_y,)

    cycle = EnsembleDACycle(
        forward_model=DynamicsForwardModel(lambda x, t0, t1: jnp.asarray(A) @ x),
        obs_op=LinearizableObsOperator(lambda x: jnp.asarray(H) @ x),
        analysis_step=FilterAnalysisStep(filt),
        obs_source=Const(jnp.asarray(y)),
        n_steps=1,
        n_members=N_e,
    )
    members = [jnp.asarray(m) for m in x0]
    analysed, state = cycle(members, DAState(obs_err_cov=jnp.asarray(R)))
    particles_a = np.stack([np.asarray(m) for m in analysed])  # (N_e, N_x)
    assert particles_a.dtype == np.float64
    assert state.cycle_count == 1

    # Independent reference: forecast each member with A, then Kalman-update
    # the forecast ensemble's sample moments.
    x_f = x0 @ A.T  # (N_e, N_x)
    mean_ref = _kalman_mean(x_f.mean(axis=0), np.cov(x_f.T, ddof=1), H, R, y)
    np.testing.assert_allclose(particles_a.mean(axis=0), mean_ref, rtol=0, atol=1e-6)

"""Latent-space ensemble DA (D17) — algebraic correctness + smoke tests.

The wrapping layer adds no new ensemble math: identity-codec wrappers
must match their state-space counterparts bit-for-bit; the L1 wrappers
(:class:`LatentDynamics`, :class:`LiftedObs`, :class:`EncodedDynamics`)
must satisfy the obvious round-trip identities under an identity codec.
Differentiability and JIT compatibility are covered as smoke tests.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np

import filterax as flx


class _LinearObs(flx.AbstractObsOperator):
    H: jnp.ndarray

    def __call__(self, state):
        return self.H @ state


class _IdentityDynamics(flx.AbstractDynamics):
    def __call__(self, state, t0, t1):
        return state


class _LinearXDynamics(flx.AbstractDynamics):
    """Non-trivial ``x``-space linear dynamics ``xₙ₊₁ = A xₙ``."""

    A: jnp.ndarray

    def __call__(self, state, t0, t1):
        return self.A @ state


class _LinearLatentForward(eqx.Module):
    """Toy ``LatentForwardModel`` — duck-typed ``.step(z, dt)`` only."""

    A: jnp.ndarray

    def step(self, z, dt):
        del dt
        return self.A @ z


def _make_obs_problem(getkey, N_x=4, N_y=2, N_e=30, n_windows=2):
    particles = jr.normal(getkey(), (N_e, N_x))
    H = jnp.eye(N_x)[:N_y]
    obs_op = _LinearObs(H=H)
    R = lx.DiagonalLinearOperator(jnp.full((N_y,), 0.4))
    obs_seq = [
        (jr.normal(getkey(), (N_y,)) * 0.5, float(t + 1)) for t in range(n_windows)
    ]
    return particles, obs_op, R, obs_seq


# ──────────────────────────────────────────────────────────────────────
# L0: latent_ensemble, decode_ensemble, identity_latent_map
# ──────────────────────────────────────────────────────────────────────


def test_identity_latent_map_round_trip(getkey):
    lm = flx.identity_latent_map(dim=4)
    x = jr.normal(getkey(), (4,))
    np.testing.assert_array_equal(np.asarray(lm.encode(x)), np.asarray(x))
    np.testing.assert_array_equal(np.asarray(lm.decode(x)), np.asarray(x))


def test_latent_ensemble_vmaps_encoder(getkey):
    lm = flx.identity_latent_map(dim=3)
    ens = jr.normal(getkey(), (5, 3))
    np.testing.assert_array_equal(
        np.asarray(flx.latent_ensemble(lm.encode, ens)), np.asarray(ens)
    )


def test_decode_ensemble_vmaps_decoder(getkey):
    lm = flx.identity_latent_map(dim=3)
    ens = jr.normal(getkey(), (5, 3))
    np.testing.assert_array_equal(
        np.asarray(flx.decode_ensemble(lm.decode, ens)), np.asarray(ens)
    )


# ──────────────────────────────────────────────────────────────────────
# L1: LiftedObs, LatentDynamics, EncodedDynamics
# ──────────────────────────────────────────────────────────────────────


def test_lifted_obs_identity_decoder_matches_x_space(getkey):
    H = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    obs_op = _LinearObs(H=H)
    lm = flx.identity_latent_map(dim=3)
    lifted = flx.LiftedObs(decoder=lm, inner=obs_op)
    x = jr.normal(getkey(), (3,))
    np.testing.assert_allclose(np.asarray(lifted(x)), np.asarray(obs_op(x)), atol=1e-12)


def test_encoded_dynamics_identity_codec_matches_inner(getkey):
    A = jr.normal(getkey(), (4, 4))
    inner = _LinearXDynamics(A=A)
    lm = flx.identity_latent_map(dim=4)
    wrapped = flx.EncodedDynamics(latent_map=lm, inner=inner)
    x = jr.normal(getkey(), (4,))
    t0, t1 = jnp.asarray(0.0), jnp.asarray(1.0)
    np.testing.assert_allclose(
        np.asarray(wrapped(x, t0, t1)),
        np.asarray(inner(x, t0, t1)),
        atol=1e-12,
    )


def test_latent_dynamics_delegates_to_inner_step(getkey):
    A = jr.normal(getkey(), (3, 3))
    inner = _LinearLatentForward(A=A)
    wrapped = flx.LatentDynamics(inner=inner)
    z = jr.normal(getkey(), (3,))
    t0, t1 = jnp.asarray(0.5), jnp.asarray(1.5)
    np.testing.assert_allclose(
        np.asarray(wrapped(z, t0, t1)),
        np.asarray(inner.step(z, t1 - t0)),
        atol=1e-12,
    )


# ──────────────────────────────────────────────────────────────────────
# L2: identity-codec parity with ETKF / LETKF
# ──────────────────────────────────────────────────────────────────────


def test_latent_etkf_identity_codec_matches_etkf(getkey):
    # With ``φ = ψ = id`` the latent wrapper must reproduce the
    # plain ETKF posterior bit-for-bit at double precision.
    particles, obs_op, R, obs_seq = _make_obs_problem(getkey)
    N_x = particles.shape[1]
    dynamics = _IdentityDynamics()

    plain = flx.ETKF(dynamics=dynamics, obs_op=obs_op).assimilate(particles, obs_seq, R)
    lm = flx.identity_latent_map(dim=N_x)
    latent = flx.LatentETKF(latent_map=lm, dynamics=dynamics, obs_op=obs_op).assimilate(
        particles, obs_seq, R
    )

    np.testing.assert_allclose(
        np.asarray(latent.particles), np.asarray(plain.particles), atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(latent.particles_z),
        np.asarray(plain.particles),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(latent.forecast_history),
        np.asarray(plain.forecast_history),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(latent.analysis_history),
        np.asarray(plain.analysis_history),
        atol=1e-12,
    )
    assert latent.log_likelihoods is not None
    np.testing.assert_allclose(
        np.asarray(latent.log_likelihoods),
        np.asarray(plain.log_likelihoods),
        atol=1e-12,
    )


def test_latent_letkf_identity_codec_matches_etkf(getkey):
    # v0.1 has no localization in latent space (§9.2 option a), so
    # ``LatentLETKF`` must collapse to plain ``ETKF`` under identity codec.
    particles, obs_op, R, obs_seq = _make_obs_problem(getkey)
    N_x = particles.shape[1]
    dynamics = _IdentityDynamics()

    plain = flx.ETKF(dynamics=dynamics, obs_op=obs_op).assimilate(particles, obs_seq, R)
    lm = flx.identity_latent_map(dim=N_x)
    latent = flx.LatentLETKF(
        latent_map=lm, dynamics=dynamics, obs_op=obs_op
    ).assimilate(particles, obs_seq, R)

    np.testing.assert_allclose(
        np.asarray(latent.particles), np.asarray(plain.particles), atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(latent.analysis_history),
        np.asarray(plain.analysis_history),
        atol=1e-12,
    )


def test_latent_etkf_in_x_space_false_accepts_pre_encoded_ensemble(getkey):
    particles, obs_op, R, obs_seq = _make_obs_problem(getkey)
    N_x = particles.shape[1]
    lm = flx.identity_latent_map(dim=N_x)
    dynamics = _IdentityDynamics()

    res_x = flx.LatentETKF(latent_map=lm, dynamics=dynamics, obs_op=obs_op).assimilate(
        particles, obs_seq, R, in_x_space=True
    )
    res_z = flx.LatentETKF(latent_map=lm, dynamics=dynamics, obs_op=obs_op).assimilate(
        flx.latent_ensemble(lm.encode, particles), obs_seq, R, in_x_space=False
    )
    np.testing.assert_allclose(
        np.asarray(res_x.particles_z),
        np.asarray(res_z.particles_z),
        atol=1e-12,
    )


def test_latent_etkf_result_shapes(getkey):
    # Codec changes dimension: encoder ``ℝ^4 → ℝ^2``, decoder ``ℝ^2 → ℝ^4``.
    N_e, N_x, N_z, N_y = 30, 4, 2, 2
    particles = jr.normal(getkey(), (N_e, N_x))
    H = jnp.eye(N_x)[:N_y]
    obs_op = _LinearObs(H=H)
    R = lx.DiagonalLinearOperator(jnp.full((N_y,), 0.4))
    obs_seq = [(jr.normal(getkey(), (N_y,)) * 0.3, float(t + 1)) for t in range(2)]

    class _Codec(eqx.Module):
        W_enc: jnp.ndarray
        W_dec: jnp.ndarray

        def encode(self, x):
            return self.W_enc @ x

        def decode(self, z):
            return self.W_dec @ z

    lm = _Codec(
        W_enc=jr.normal(getkey(), (N_z, N_x)),
        W_dec=jr.normal(getkey(), (N_x, N_z)),
    )
    A_z = jr.normal(getkey(), (N_z, N_z))
    dynamics = flx.LatentDynamics(inner=_LinearLatentForward(A=A_z))

    res = flx.LatentETKF(latent_map=lm, dynamics=dynamics, obs_op=obs_op).assimilate(
        particles, obs_seq, R
    )

    assert res.particles.shape == (N_e, N_x)
    assert res.particles_z.shape == (N_e, N_z)
    assert res.forecast_history.shape == (2, N_e, N_x)
    assert res.forecast_history_z.shape == (2, N_e, N_z)
    assert res.analysis_history.shape == (2, N_e, N_x)
    assert res.analysis_history_z.shape == (2, N_e, N_z)
    assert res.log_likelihoods is not None
    assert res.log_likelihoods.shape == (2,)


# ──────────────────────────────────────────────────────────────────────
# Differentiability + JIT
# ──────────────────────────────────────────────────────────────────────


def test_latent_etkf_assimilate_is_jit_safe(getkey):
    # ``LatentETKF.assimilate`` must compose cleanly with ``eqx.filter_jit``
    # — observations are a Python list of tuples so the static structure is
    # baked in at trace time, just like the other L2 wrappers.
    particles, obs_op, R, obs_seq = _make_obs_problem(getkey)
    N_x = particles.shape[1]
    lm = flx.identity_latent_map(dim=N_x)
    filter_ = flx.LatentETKF(latent_map=lm, dynamics=_IdentityDynamics(), obs_op=obs_op)

    @eqx.filter_jit
    def run(p):
        return filter_.assimilate(p, obs_seq, R).particles

    out = run(particles)
    assert out.shape == particles.shape
    assert bool(jnp.all(jnp.isfinite(out)))


def test_latent_etkf_differentiable_wrt_codec_weights(getkey):
    """Gradient w.r.t. trainable codec weights flows through the L2 loop.

    Confirms the wrapping layer is end-to-end differentiable, which is
    the whole point of the latent DA primitive — the user can train the
    codec by backpropagating through the assimilation.
    """
    N_e, N_x = 20, 3
    particles = jr.normal(getkey(), (N_e, N_x))
    H = jnp.eye(N_x)[:2]
    obs_op = _LinearObs(H=H)
    R = lx.DiagonalLinearOperator(jnp.full((2,), 0.5))
    obs_seq = [
        (jr.normal(getkey(), (2,)) * 0.3, 1.0),
        (jr.normal(getkey(), (2,)) * 0.3, 2.0),
    ]

    class _Codec(eqx.Module):
        W: jnp.ndarray

        def encode(self, x):
            return self.W @ x

        def decode(self, z):
            return self.W.T @ z

    lm0 = _Codec(W=jnp.eye(N_x))

    def loss(lm):
        filter_ = flx.LatentETKF(
            latent_map=lm, dynamics=_IdentityDynamics(), obs_op=obs_op
        )
        out = filter_.assimilate(particles, obs_seq, R)
        return jnp.mean(out.particles**2)

    grads = eqx.filter_grad(loss)(lm0)
    assert grads.W.shape == lm0.W.shape
    assert bool(jnp.all(jnp.isfinite(grads.W)))
    assert float(jnp.abs(grads.W).max()) > 0.0


def test_lifted_obs_satisfies_obs_operator_protocol():
    # Structural check: the wrapping classes must register as their
    # abstract types so they slot into the existing filters and any
    # ``isinstance`` checks in downstream code.
    lm = flx.identity_latent_map(dim=2)
    obs_op = _LinearObs(H=jnp.eye(2))
    lifted = flx.LiftedObs(decoder=lm, inner=obs_op)
    assert isinstance(lifted, flx.AbstractObsOperator)

    dyn = flx.LatentDynamics(inner=_LinearLatentForward(A=jnp.eye(2)))
    assert isinstance(dyn, flx.AbstractDynamics)

    enc_dyn = flx.EncodedDynamics(latent_map=lm, inner=_IdentityDynamics())
    assert isinstance(enc_dyn, flx.AbstractDynamics)


def test_latent_etkf_auto_wraps_raw_latent_forward_model(getkey):
    # The design doc advertises the structural ``LatentForwardModel`` —
    # ``.step(z, dt) → z`` only. The L2 wrappers must auto-wrap so a raw
    # ``.step``-only object does not fail at trace time inside
    # ``_forecast`` (which calls ``dynamics(state, t0, t1)``).
    particles, obs_op, R, obs_seq = _make_obs_problem(getkey)
    N_x = particles.shape[1]
    lm = flx.identity_latent_map(dim=N_x)
    raw = _LinearLatentForward(A=jnp.eye(N_x))

    filt = flx.LatentETKF(latent_map=lm, dynamics=raw, obs_op=obs_op)
    assert isinstance(filt.dynamics, flx.LatentDynamics)
    assert filt.dynamics.inner is raw

    out = filt.assimilate(particles, obs_seq, R)
    assert out.particles.shape == particles.shape
    assert bool(jnp.all(jnp.isfinite(out.particles)))

    # An already-wrapped AbstractDynamics is passed through unchanged.
    wrapped = flx.LatentDynamics(inner=raw)
    filt_wrapped = flx.LatentETKF(latent_map=lm, dynamics=wrapped, obs_op=obs_op)
    assert filt_wrapped.dynamics is wrapped


def test_latent_letkf_auto_wraps_raw_latent_forward_model(getkey):
    particles, obs_op, R, obs_seq = _make_obs_problem(getkey)
    N_x = particles.shape[1]
    lm = flx.identity_latent_map(dim=N_x)
    raw = _LinearLatentForward(A=jnp.eye(N_x))

    filt = flx.LatentLETKF(latent_map=lm, dynamics=raw, obs_op=obs_op)
    assert isinstance(filt.dynamics, flx.LatentDynamics)
    out = filt.assimilate(particles, obs_seq, R)
    assert out.particles.shape == particles.shape
    assert bool(jnp.all(jnp.isfinite(out.particles)))


def test_latent_etkf_grad_skips_non_trainable_codec_fields(getkey):
    # ``identity_latent_map`` carries only a static ``int`` field, so
    # ``eqx.filter_grad`` should treat it as having no trainable leaves.
    particles, obs_op, R, obs_seq = _make_obs_problem(getkey)
    N_x = particles.shape[1]
    lm = flx.identity_latent_map(dim=N_x)
    filter_ = flx.LatentETKF(latent_map=lm, dynamics=_IdentityDynamics(), obs_op=obs_op)

    def loss(p):
        return jnp.mean(filter_.assimilate(p, obs_seq, R).particles ** 2)

    grad = jax.grad(loss)(particles)
    assert grad.shape == particles.shape
    assert bool(jnp.all(jnp.isfinite(grad)))

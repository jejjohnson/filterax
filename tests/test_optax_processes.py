"""Contract + smoke tests for the optax-flavoured EKP transforms."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optax

import filterax as flx


def _problem():
    G = jnp.asarray([[1.0, 0.5], [-0.3, 1.2]])
    theta_true = jnp.asarray([1.0, -1.0])
    y = G @ theta_true
    R = lx.DiagonalLinearOperator(jnp.asarray([0.1, 0.1]))
    sigma0 = 5.0
    R_inv = jnp.diag(jnp.asarray([10.0, 10.0]))
    Sigma_post = jnp.linalg.inv(G.T @ R_inv @ G + (1 / sigma0**2) * jnp.eye(2))
    mu_post = Sigma_post @ G.T @ R_inv @ y
    return G, y, R, sigma0, mu_post, Sigma_post


def test_eki_optax_contract():
    """`init` + repeated `update` + `apply_updates` mirrors the L2 EKI run."""
    G, y, R, sigma0, mu_post, _ = _problem()
    forward = lambda theta: G @ theta

    transform = flx.optax.eki(
        forward_fn=forward,
        obs=y,
        noise_cov=R,
        n_ensemble=500,
        init_spread=sigma0,
        scheduler=flx.FixedScheduler(dt=1.0),
    )
    params = jnp.zeros(2)  # starting point for the mean
    state = transform.init(params)
    for _ in range(3):
        updates, state = transform.update(None, state, params)
        params = optax.apply_updates(params, updates)
    np.testing.assert_allclose(np.asarray(params), np.asarray(mu_post), atol=1e-1)


def test_eks_optax_advances_key_per_step():
    G, y, R, _, _, _ = _problem()
    forward = lambda theta: G @ theta

    transform = flx.optax.eks(
        forward_fn=forward,
        obs=y,
        noise_cov=R,
        n_ensemble=100,
        scheduler=flx.FixedScheduler(dt=0.01),
    )
    params = jnp.zeros(2)
    state = transform.init(params)
    # Two consecutive updates with the same params should produce
    # different particle sets (Brownian noise advances).
    _, state_a = transform.update(None, state, params)
    _, state_b = transform.update(None, state_a, params)
    diff = jnp.linalg.norm(state_a.particles - state_b.particles)
    assert float(diff) > 0


def test_uki_optax_one_step_matches_posterior():
    G, y, R, sigma0, mu_post, Sigma_post = _problem()
    forward = lambda theta: G @ theta

    transform = flx.optax.uki(
        forward_fn=forward,
        obs=y,
        noise_cov=R,
        init_cov=sigma0,
        scheduler=flx.FixedScheduler(dt=1.0),
    )
    params = jnp.zeros(2)
    state = transform.init(params)
    updates, state = transform.update(None, state, params)
    params = optax.apply_updates(params, updates)
    np.testing.assert_allclose(np.asarray(params), np.asarray(mu_post), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(state.covariance.as_matrix()),
        np.asarray(Sigma_post),
        atol=1e-5,
    )


def test_eki_optax_composes_with_optax_chain():
    """A simple chain with gradient clipping leaves the EKI step intact."""
    G, y, R, _, _, _ = _problem()
    forward = lambda theta: G @ theta

    optimizer = optax.chain(
        flx.optax.eki(
            forward_fn=forward,
            obs=y,
            noise_cov=R,
            n_ensemble=80,
            scheduler=flx.FixedScheduler(dt=0.2),
        ),
        optax.clip_by_global_norm(10.0),
    )
    params = jnp.zeros(2)
    state = optimizer.init(params)
    updates, state = optimizer.update(None, state, params)
    params = optax.apply_updates(params, updates)
    # Just check we got finite, two-element output.
    assert params.shape == (2,)
    assert jnp.all(jnp.isfinite(params))


def test_eki_optax_handles_pytree_params():
    """Params can be a dict; the unflatten machinery returns the same tree."""
    G, _y, R, _, _, _ = _problem()
    forward = lambda flat: G @ flat
    # Use a pytree param of total size 2 by raveling internally.
    params_tree = {"a": jnp.zeros(1), "b": jnp.zeros(1)}

    def forward_tree(theta):
        flat, _ = jax.flatten_util.ravel_pytree(theta)
        return forward(flat)

    transform = flx.optax.eki(
        forward_fn=forward_tree,
        obs=jnp.zeros(2),
        noise_cov=R,
        n_ensemble=40,
        scheduler=flx.FixedScheduler(dt=0.1),
    )
    state = transform.init(params_tree)
    updates, state = transform.update(None, state, params_tree)
    # Updates must be the same tree shape as params.
    assert set(updates.keys()) == {"a", "b"}
    assert updates["a"].shape == (1,)
    assert updates["b"].shape == (1,)


def test_adam_to_eki_hybrid_optimisation():
    r"""Two-phase optimisation: Adam (gradient-based) → EKI (derivative-free).

    Demonstrates the hybrid pattern from
    ``docs/design_docs/features/optax_ekp.md`` §6: Adam handles a fast
    warm-start on a smooth differentiable loss; EKI then refines
    without gradients. Both phases use the same ``optax`` driver loop,
    just with different transforms.
    """
    G, y, R, _, mu_post, _ = _problem()

    def loss_fn(theta):
        residual = G @ theta - y
        return 0.5 * jnp.sum(residual**2 / 0.1)

    forward = lambda theta: G @ theta

    # ── Phase 1: Adam warm-start with explicit gradients ────────────
    adam = optax.adam(learning_rate=0.2)
    params = jnp.asarray([3.0, 3.0])  # far from the truth
    state = adam.init(params)
    for _ in range(30):
        grads = jax.grad(loss_fn)(params)
        updates, state = adam.update(grads, state, params)
        params = optax.apply_updates(params, updates)

    adam_error = float(jnp.linalg.norm(params - mu_post))

    # ── Phase 2: EKI refinement (grads=None) ────────────────────────
    eki_transform = flx.optax.eki(
        forward_fn=forward,
        obs=y,
        noise_cov=R,
        n_ensemble=100,
        init_spread=0.1,  # tight cluster around the warm-started mean
        scheduler=flx.FixedScheduler(dt=1.0),
    )
    state = eki_transform.init(params)
    for _ in range(2):
        updates, state = eki_transform.update(None, state, params)
        params = optax.apply_updates(params, updates)

    hybrid_error = float(jnp.linalg.norm(params - mu_post))

    # Adam alone gets close, EKI refines it further (within Monte Carlo).
    assert hybrid_error <= adam_error + 0.1
    # And both phases produced finite output.
    assert bool(jnp.all(jnp.isfinite(params)))

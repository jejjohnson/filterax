"""EKP processes wrapped as ``optax.GradientTransformation`` objects.

Each constructor returns the standard ``(init, update)`` pair from
optax:

* ``init(params) -> OptState`` flattens the params tree and seeds an
  ensemble (EKI / EKS) or a parametric belief (UKI).
* ``update(grads, state, params) -> (updates, OptState)`` ignores the
  ``grads`` argument — the ensemble provides search directions — and
  returns the increment to apply to ``params`` via
  :func:`optax.apply_updates`.

The ensemble (or sigma-point sequence) lives in the optax ``OptState``;
the ``params`` tree only ever holds the running mean estimate. This
matches the contract documented in ``docs/design_docs/features/optax_ekp.md``.

Forward models are captured by the closure of the constructor, not
stored in state. Anyone who has used ``optax.inject_hyperparams`` will
recognise the pattern.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import einx
import equinox as eqx
import jax
import jax.flatten_util  # ensure the submodule is loaded for typecheck
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from filterax._processes._processes import (
    _eki_delta,
    _safe_inv_dt,
    sigma_points,
)
from filterax._processes._schedulers import (
    DataMisfitController,
    EKSStableScheduler,
    FixedScheduler,
)
from filterax._protocols import AbstractScheduler
from filterax._types import ProcessState, UKIState


ForwardFn = Callable[[Float[Array, " N_p"]], Float[Array, " N_d"]]


# ──────────────────────────────────────────────────────────────────────
# State containers (optax expects NamedTuple-ish containers; eqx.Module
# is pytree-compatible and JIT-friendly, so we use those).
# ──────────────────────────────────────────────────────────────────────


class EKIOptaxState(eqx.Module, strict=True):
    """Internal optax state for :func:`eki`."""

    particles: Float[Array, "J N_p"]
    algo_time: Float[Array, ""]
    step: jnp.ndarray
    unravel: Callable[[Float[Array, " N_p"]], Any] = eqx.field(static=True)


class EKSOptaxState(eqx.Module, strict=True):
    """Internal optax state for :func:`eks`."""

    particles: Float[Array, "J N_p"]
    algo_time: Float[Array, ""]
    step: jnp.ndarray
    key: PRNGKeyArray
    unravel: Callable[[Float[Array, " N_p"]], Any] = eqx.field(static=True)


class UKIOptaxState(eqx.Module, strict=True):
    """Internal optax state for :func:`uki`."""

    mean: Float[Array, " N_p"]
    covariance: lx.AbstractLinearOperator
    algo_time: Float[Array, ""]
    step: jnp.ndarray
    unravel: Callable[[Float[Array, " N_p"]], Any] = eqx.field(static=True)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _flatten(params: Any) -> tuple[Float[Array, " N_p"], Callable]:
    """Ravel a PyTree of params to a flat vector + its unravel callback."""
    flat, unravel = jax.flatten_util.ravel_pytree(params)
    return flat, unravel


def _scheduler_state_shim(
    particles: Float[Array, "J N_p"],
    evals: Float[Array, "J N_d"],
    obs: Float[Array, " N_d"],
    noise_cov: lx.AbstractLinearOperator,
    step: jnp.ndarray,
    algo_time: Float[Array, ""],
) -> ProcessState:
    """Build a transient ProcessState so a scheduler can see the evals."""
    return ProcessState(
        particles=particles,
        forward_evals=evals,
        obs=obs,
        noise_cov=noise_cov,
        step=step,
        algo_time=algo_time,
    )


# ──────────────────────────────────────────────────────────────────────
# EKI as optax
# ──────────────────────────────────────────────────────────────────────


def eki(
    forward_fn: ForwardFn,
    obs: Float[Array, " N_d"],
    noise_cov: lx.AbstractLinearOperator,
    *,
    n_ensemble: int = 50,
    init_spread: float = 1.0,
    scheduler: AbstractScheduler | None = None,
    key: PRNGKeyArray | int = 0,
) -> optax.GradientTransformation:
    r"""Ensemble Kalman Inversion as an ``optax.GradientTransformation``.

    The ``grads`` argument to :meth:`update` is ignored — the ensemble
    provides search directions via the cross-covariance. The increment
    returned is the change in the *ensemble mean*, unflattened back to
    the ``params`` tree shape so :func:`optax.apply_updates` works as
    usual.

    Args:
        forward_fn: ``θ → G(θ)`` mapping. Closure-captured (not stored
            in state) so it doesn't have to be a pytree leaf.
        obs: Observation vector ``y ∈ ℝ^{N_d}``.
        noise_cov: Observation noise covariance ``Γ``.
        n_ensemble: Ensemble size ``J``.
        init_spread: Standard deviation of the initial ensemble
            perturbation around ``params`` (the seed mean).
        scheduler: Step-size strategy. Defaults to
            :class:`~filterax.DataMisfitController`.
        key: PRNG seed (int or PRNG array) for the initial ensemble.

    Returns:
        ``optax.GradientTransformation`` whose ``update`` advances the
        ensemble and returns the per-step mean delta as the optax
        update.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import optax
        >>> from filterax import FixedScheduler
        >>> from filterax.optax import eki
        >>> G = jnp.array([[1.0, 0.5], [-0.3, 1.2]])
        >>> transform = eki(
        ...     forward_fn=lambda theta: G @ theta,
        ...     obs=jnp.array([0.5, 0.9]),
        ...     noise_cov=lx.DiagonalLinearOperator(0.1 * jnp.ones(2)),
        ...     n_ensemble=8,
        ...     scheduler=FixedScheduler(dt=1.0),
        ... )
        >>> params = jnp.zeros(2)
        >>> state = transform.init(params)
        >>> updates, state = transform.update(None, state, params)
        >>> params = optax.apply_updates(params, updates)
        >>> params.shape
        (2,)
    """
    sched = scheduler if scheduler is not None else DataMisfitController()
    base_key = jr.PRNGKey(key) if isinstance(key, int) else key

    def init_fn(params: Any) -> EKIOptaxState:
        mean, unravel = _flatten(params)
        # θ⁽ʲ⁾ ~ 𝒩(mean, init_spread² I), recentred so that
        # ``particles.mean(axis=0) == mean`` exactly — otherwise the
        # first ``update`` would return a non-zero delta driven purely
        # by finite-sample sampling noise rather than by the data.
        noise = init_spread * jr.normal(
            base_key, (n_ensemble, mean.shape[0]), dtype=mean.dtype
        )
        noise = noise - jnp.mean(noise, axis=0, keepdims=True)
        particles = mean[None, :] + noise
        return EKIOptaxState(
            particles=particles,
            algo_time=jnp.asarray(0.0, dtype=mean.dtype),
            step=jnp.asarray(0, dtype=jnp.int32),
            unravel=unravel,
        )

    def update_fn(
        grads: Any,
        state: EKIOptaxState,
        params: Any | None = None,
    ) -> tuple[Any, EKIOptaxState]:
        del grads
        if params is None:
            raise ValueError(
                "filterax.optax.eki requires `params` to compute the delta."
            )
        prev_mean, _ = _flatten(params)

        evals = jax.vmap(forward_fn)(state.particles)
        shim = _scheduler_state_shim(
            state.particles, evals, obs, noise_cov, state.step, state.algo_time
        )
        dt = sched.get_dt(shim)
        delta = _eki_delta(state.particles, evals, obs, noise_cov, dt)
        particles_new = state.particles + delta
        new_mean = jnp.mean(particles_new, axis=0)
        # Unflatten the mean increment back into the params tree.
        mean_update = state.unravel(new_mean - prev_mean)
        return mean_update, EKIOptaxState(
            particles=particles_new,
            algo_time=state.algo_time + dt,
            step=state.step + 1,
            unravel=state.unravel,
        )

    # optax's TransformInitFn/TransformUpdateFn protocols assume the
    # state is array-typed; our richer eqx.Module state is functionally
    # compatible (pytree + JIT-stable) but the narrow stub rejects it.
    return optax.GradientTransformation(init_fn, update_fn)  # ty: ignore[invalid-argument-type]


# ──────────────────────────────────────────────────────────────────────
# EKS as optax
# ──────────────────────────────────────────────────────────────────────


def eks(
    forward_fn: ForwardFn,
    obs: Float[Array, " N_d"],
    noise_cov: lx.AbstractLinearOperator,
    *,
    n_ensemble: int = 100,
    init_spread: float = 1.0,
    scheduler: AbstractScheduler | None = None,
    key: PRNGKeyArray | int = 0,
) -> optax.GradientTransformation:
    r"""Ensemble Kalman Sampler as an ``optax.GradientTransformation``.

    Adds Brownian noise so the ensemble samples the posterior rather
    than collapsing. Access ``state.particles`` after burn-in for
    posterior samples. The PRNG key advances at every step via
    :func:`jr.fold_in` so successive draws are independent.

    Args:
        forward_fn: ``θ → G(θ)``.
        obs: Observation vector.
        noise_cov: Observation noise covariance ``Γ``.
        n_ensemble: Ensemble size ``J`` (recommended ≥ 100 for sampling).
        init_spread: Standard deviation of the initial ensemble
            perturbation around ``params``.
        scheduler: Step-size strategy. Use
            :class:`~filterax.EKSStableScheduler` for stability.
        key: PRNG seed.
    """
    sched = scheduler if scheduler is not None else EKSStableScheduler()
    base_key = jr.PRNGKey(key) if isinstance(key, int) else key
    init_key, noise_key = jr.split(base_key)

    def init_fn(params: Any) -> EKSOptaxState:
        mean, unravel = _flatten(params)
        noise = init_spread * jr.normal(
            init_key, (n_ensemble, mean.shape[0]), dtype=mean.dtype
        )
        particles = mean[None, :] + noise
        return EKSOptaxState(
            particles=particles,
            algo_time=jnp.asarray(0.0, dtype=mean.dtype),
            step=jnp.asarray(0, dtype=jnp.int32),
            key=noise_key,
            unravel=unravel,
        )

    def update_fn(
        grads: Any,
        state: EKSOptaxState,
        params: Any | None = None,
    ) -> tuple[Any, EKSOptaxState]:
        del grads
        if params is None:
            raise ValueError(
                "filterax.optax.eks requires `params` to compute the delta."
            )
        prev_mean, _ = _flatten(params)

        J, N_p = state.particles.shape
        evals = jax.vmap(forward_fn)(state.particles)
        shim = _scheduler_state_shim(
            state.particles, evals, obs, noise_cov, state.step, state.algo_time
        )
        dt = sched.get_dt(shim)

        # EKI-style drift + mean-pull + ensemble-preconditioned noise.
        drift = _eki_delta(state.particles, evals, obs, noise_cov, dt)
        mean = jnp.mean(state.particles, axis=0)
        anom = state.particles - mean[None, :]
        drift_pull = -((N_p + 1) / J) * dt * anom
        step_key = jr.fold_in(state.key, state.step)
        xi = jr.normal(step_key, (J, J), dtype=state.particles.dtype)
        brownian = (
            jnp.sqrt(2.0 * dt) * einx.dot("j k, k p -> j p", xi, anom) / jnp.sqrt(J - 1)
        )

        particles_new = state.particles + drift + drift_pull + brownian
        new_mean = jnp.mean(particles_new, axis=0)
        mean_update = state.unravel(new_mean - prev_mean)
        return mean_update, EKSOptaxState(
            particles=particles_new,
            algo_time=state.algo_time + dt,
            step=state.step + 1,
            key=state.key,
            unravel=state.unravel,
        )

    # optax's TransformInitFn/TransformUpdateFn protocols assume the
    # state is array-typed; our richer eqx.Module state is functionally
    # compatible (pytree + JIT-stable) but the narrow stub rejects it.
    return optax.GradientTransformation(init_fn, update_fn)  # ty: ignore[invalid-argument-type]


# ──────────────────────────────────────────────────────────────────────
# UKI as optax
# ──────────────────────────────────────────────────────────────────────


def uki(
    forward_fn: ForwardFn,
    obs: Float[Array, " N_d"],
    noise_cov: lx.AbstractLinearOperator,
    *,
    init_cov: float | Float[Array, "N_p N_p"] = 1.0,
    scheduler: AbstractScheduler | None = None,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> optax.GradientTransformation:
    r"""Unscented Kalman Inversion as an ``optax.GradientTransformation``.

    Parametric: state carries ``(μ, Σ)`` plus the unflatten callback.
    Each ``update`` regenerates ``2 Nₚ + 1`` sigma points, evaluates
    ``forward_fn`` on them, and applies the unscented Kalman update.
    Access ``state.covariance`` for the posterior covariance estimate.

    Args:
        forward_fn: ``θ → G(θ)``.
        obs: Observation vector.
        noise_cov: Observation noise covariance ``Γ``.
        init_cov: Initial parameter covariance. Scalar → ``σ² I``;
            matrix → used directly.
        scheduler: Step-size strategy.
        alpha: Sigma-point spread parameter (Wan & van der Merwe 2000).
        beta: Prior-moment weighting (``2.0`` is optimal for Gaussian).
        kappa: Secondary scaling parameter; typically ``0`` or ``3 − Nₚ``.
    """
    sched = scheduler if scheduler is not None else DataMisfitController()

    def init_fn(params: Any) -> UKIOptaxState:
        mean, unravel = _flatten(params)
        if jnp.ndim(jnp.asarray(init_cov)) == 0:
            Sigma = (init_cov**2) * jnp.eye(mean.shape[0], dtype=mean.dtype)
        else:
            Sigma = jnp.asarray(init_cov)
        cov_op = lx.MatrixLinearOperator(Sigma, tags=lx.positive_semidefinite_tag)
        return UKIOptaxState(
            mean=mean,
            covariance=cov_op,
            algo_time=jnp.asarray(0.0, dtype=mean.dtype),
            step=jnp.asarray(0, dtype=jnp.int32),
            unravel=unravel,
        )

    def update_fn(
        grads: Any,
        state: UKIOptaxState,
        params: Any | None = None,
    ) -> tuple[Any, UKIOptaxState]:
        del grads
        if params is None:
            raise ValueError(
                "filterax.optax.uki requires `params` to compute the delta."
            )
        prev_mean, _ = _flatten(params)

        points, w_mean, w_cov = sigma_points(
            state.mean,
            state.covariance,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
        )
        evals = jax.vmap(forward_fn)(points)
        shim = _scheduler_state_shim(
            points, evals, obs, noise_cov, state.step, state.algo_time
        )
        dt = sched.get_dt(shim)

        # Unscented Kalman update.
        y_mean = einx.dot("k, k d -> d", w_mean, evals)
        theta_anom = einx.subtract("k p, p -> k p", points, state.mean)
        y_anom = einx.subtract("k d, d -> k d", evals, y_mean)
        weighted_theta = einx.multiply("k p, k -> k p", theta_anom, w_cov)
        C_theta_y = einx.dot("k p, k d -> p d", weighted_theta, y_anom)
        weighted_y = einx.multiply("k d, k -> k d", y_anom, w_cov)
        S_dense = einx.dot("k a, k b -> a b", weighted_y, y_anom)
        S_tempered = S_dense + _safe_inv_dt(dt) * noise_cov.as_matrix()
        innovation = obs - y_mean
        K_gain = jnp.linalg.solve(S_tempered.T, C_theta_y.T).T

        new_mean = state.mean + dt * einx.dot("p d, d -> p", K_gain, innovation)
        # Σ_{n+1} = Σ̂ − Δt · K · S_tempered · K^T  (Huang 2022 eq. 14).
        Sigma_dense = state.covariance.as_matrix()
        K_S = einx.dot("p a, a b -> p b", K_gain, S_tempered)
        Sigma_new = Sigma_dense - dt * einx.dot("p d, q d -> p q", K_S, K_gain)
        Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
        cov_new = lx.MatrixLinearOperator(Sigma_new, tags=lx.positive_semidefinite_tag)

        mean_update = state.unravel(new_mean - prev_mean)
        return mean_update, UKIOptaxState(
            mean=new_mean,
            covariance=cov_new,
            algo_time=state.algo_time + dt,
            step=state.step + 1,
            unravel=state.unravel,
        )

    # optax's TransformInitFn/TransformUpdateFn protocols assume the
    # state is array-typed; our richer eqx.Module state is functionally
    # compatible (pytree + JIT-stable) but the narrow stub rejects it.
    return optax.GradientTransformation(init_fn, update_fn)  # ty: ignore[invalid-argument-type]


__all__ = [
    "EKIOptaxState",
    "EKSOptaxState",
    "UKIOptaxState",
    "eki",
    "eks",
    "uki",
]

# Silence the unused-import warning for FixedScheduler — it is part of
# the public scheduler API and re-exported from filterax/__init__.py.
_ = FixedScheduler
_ = UKIState

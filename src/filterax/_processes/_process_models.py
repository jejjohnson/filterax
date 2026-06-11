"""Layer-2 high-level run loops for ensemble Kalman processes.

Compose a forward model with a Layer-1 process (:class:`EKI`,
:class:`EKS_Process`, :class:`UKI`, …) and a scheduler into a single
``run`` call. Mirrors :mod:`filterax._filters._models` for the sequential
filters.

The ``run`` loop iterates until either ``n_iterations`` steps elapse or
the scheduler reports ``algo_time ≥ 1.0``. Per-step history (particles,
forward evals, algo_time) is stacked along a leading time axis.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray

from filterax._processes._processes import (
    EKI as _EKI,
    UKI as _UKI,
    EKS_Process as _EKS,
)
from filterax._processes._schedulers import (
    DataMisfitController,
    EKSStableScheduler,
    FixedScheduler,
)
from filterax._protocols import AbstractScheduler
from filterax._types import ProcessConfig, ProcessState, UKIState


ForwardFn = Callable[[Float[Array, " N_p"]], Float[Array, " N_d"]]


class ProcessResult(eqx.Module, strict=True):
    r"""Output of a full EKP run.

    Attributes:
        particles: Final ensemble (``None`` for UKI — use ``mean`` /
            ``covariance`` instead).
        mean: Final posterior-mean estimate.
        covariance: Final posterior covariance (UKI's parametric state;
            the ensemble sample covariance for EKI / EKS).
        history_particles: Particles at every iteration, shape
            ``(T+1, J, Nₚ)`` (initial + each post-update state).
        history_algo_time: ``algo_time`` after each iteration, shape
            ``(T,)``.
        n_iterations: Actual number of iterations run.
        converged: ``True`` when the scheduler reached ``algo_time ≥ 1``
            before ``n_iterations`` was exhausted.
    """

    particles: Float[Array, "J N_p"] | None
    mean: Float[Array, " N_p"]
    covariance: lx.AbstractLinearOperator
    history_particles: Float[Array, "Tp1 J N_p"] | None
    history_algo_time: Float[Array, " T"]
    n_iterations: int = eqx.field(static=True)
    converged: bool = eqx.field(static=True)


def _vmap_forward(
    forward_fn: ForwardFn, particles: Float[Array, "J N_p"]
) -> Float[Array, "J N_d"]:
    """vmap the forward model over an ensemble."""
    return jax.vmap(forward_fn)(particles)


def _resolve_scheduler(
    scheduler: AbstractScheduler, config: ProcessConfig | None
) -> AbstractScheduler:
    """If ``config`` is supplied, its ``scheduler`` field wins over the
    module-level field. This matches the ``ProcessConfig`` contract:
    it is the single source of static configuration when present.

    ``ProcessConfig.scheduler`` is forward-referenced as the looser
    :class:`equinox.Module` to avoid a circular import; we narrow back
    to :class:`AbstractScheduler` here with an explicit runtime check.
    """
    if config is None:
        return scheduler
    if not isinstance(config.scheduler, AbstractScheduler):
        raise TypeError(
            "ProcessConfig.scheduler must be an AbstractScheduler subclass; "
            f"got {type(config.scheduler).__name__}."
        )
    return config.scheduler


def _resolve_n_iterations(config: ProcessConfig | None, default: int) -> int:
    return config.n_iterations if config is not None else default


def _ensemble_cov_dense(
    particles: Float[Array, "J N_p"],
) -> Float[Array, "N_p N_p"]:
    """Materialise the (small) ensemble covariance for the result struct."""
    J, _ = particles.shape
    mean = jnp.mean(particles, axis=0)
    anom = particles - mean[None, :]
    return anom.T @ anom / (J - 1)


# ──────────────────────────────────────────────────────────────────────
# Ensemble-typed L2 wrappers — EKI, EKS
# ──────────────────────────────────────────────────────────────────────


def _run_ensemble(
    process: _EKI | _EKS,
    forward_fn: ForwardFn,
    init_particles: Float[Array, "J N_p"],
    obs: Float[Array, " N_d"],
    noise_cov: lx.AbstractLinearOperator,
    n_iterations: int,
    stop_at_algo_time_one: bool,
) -> ProcessResult:
    r"""Generic init → forward eval → process update loop."""
    state = process.init(init_particles, obs, noise_cov)

    history = [state.particles]
    algo_times: list[Float[Array, ""]] = []
    converged = False
    actual_steps = 0
    for _ in range(n_iterations):
        evals = _vmap_forward(forward_fn, state.particles)
        state = process.update(state, evals)
        history.append(state.particles)
        algo_times.append(state.algo_time)
        actual_steps += 1
        if stop_at_algo_time_one and float(state.algo_time) >= 1.0 - 1e-9:
            converged = True
            break

    particles_final = state.particles
    mean = jnp.mean(particles_final, axis=0)
    cov_dense = _ensemble_cov_dense(particles_final)
    return ProcessResult(
        particles=particles_final,
        mean=mean,
        covariance=lx.MatrixLinearOperator(
            cov_dense, tags=lx.positive_semidefinite_tag
        ),
        history_particles=jnp.stack(history),
        history_algo_time=jnp.stack(algo_times) if algo_times else jnp.zeros(0),
        n_iterations=actual_steps,
        converged=converged,
    )


class EKI(eqx.Module, strict=True):
    r"""Ensemble Kalman Inversion — full ``init → update`` run loop.

    Composes a :class:`forward_fn` (the simulator) with a Layer-1
    :class:`~filterax._processes._processes.EKI` and a scheduler. Returns the
    final parameter ensemble plus per-iteration history.

    Attributes:
        forward_fn: ``θ → G(θ)`` mapping. Applied with :func:`jax.vmap`
            over the ensemble.
        obs: Observation vector ``y``.
        noise_cov: Observation noise covariance ``Γ``.
        scheduler: Step-size strategy. Defaults to
            :class:`DataMisfitController` which terminates when
            ``algo_time → 1``. Ignored when ``config`` is provided —
            see ``config`` below.
        config: Optional :class:`ProcessConfig`. When provided, **both**
            its ``scheduler`` and ``n_iterations`` fields take precedence
            over the module-level ``scheduler`` and the ``50``-iteration
            default. The ``scheduler`` attribute on the L2 module then
            acts as a fallback used only when ``config`` is ``None``.
    """

    forward_fn: ForwardFn
    obs: Float[Array, " N_d"]
    noise_cov: lx.AbstractLinearOperator
    scheduler: AbstractScheduler = eqx.field(default_factory=DataMisfitController)
    config: ProcessConfig | None = None

    def run(self, init_particles: Float[Array, "J N_p"]) -> ProcessResult:
        sched = _resolve_scheduler(self.scheduler, self.config)
        n_iter = _resolve_n_iterations(self.config, default=50)
        return _run_ensemble(
            _EKI(scheduler=sched),
            self.forward_fn,
            init_particles,
            self.obs,
            self.noise_cov,
            n_iterations=n_iter,
            stop_at_algo_time_one=True,
        )


class EKS(eqx.Module, strict=True):
    r"""Ensemble Kalman Sampler — full run loop.

    Like :class:`EKI` but produces approximate posterior samples (the
    ensemble does *not* collapse). Uses
    :class:`~filterax._processes._processes.EKS_Process` with
    :class:`EKSStableScheduler` by default; runs the full
    ``n_iterations`` (no algo-time termination — burn-in + sampling).

    Attributes:
        forward_fn: ``θ → G(θ)``.
        obs: Observation vector ``y``.
        noise_cov: Observation noise covariance ``Γ``.
        scheduler: Step-size strategy. Use :class:`EKSStableScheduler`.
            Ignored when ``config`` is provided.
        seed: PRNG seed for the Brownian draws.
        config: Optional :class:`ProcessConfig`. When provided, **both**
            its ``scheduler`` and ``n_iterations`` fields take precedence
            over the module-level ``scheduler`` and the ``500``-iteration
            default.
    """

    forward_fn: ForwardFn
    obs: Float[Array, " N_d"]
    noise_cov: lx.AbstractLinearOperator
    scheduler: AbstractScheduler = eqx.field(default_factory=EKSStableScheduler)
    seed: int = eqx.field(static=True, default=0)
    config: ProcessConfig | None = None

    def run(self, init_particles: Float[Array, "J N_p"]) -> ProcessResult:
        sched = _resolve_scheduler(self.scheduler, self.config)
        n_iter = _resolve_n_iterations(self.config, default=500)
        return _run_ensemble(
            _EKS(scheduler=sched, key=self.seed),
            self.forward_fn,
            init_particles,
            self.obs,
            self.noise_cov,
            n_iterations=n_iter,
            stop_at_algo_time_one=False,
        )


# ──────────────────────────────────────────────────────────────────────
# UKI L2 (parametric)
# ──────────────────────────────────────────────────────────────────────


class UKI(eqx.Module, strict=True):
    r"""Unscented Kalman Inversion — full parametric run loop.

    Maintains ``μₙ`` and ``Σₙ`` rather than a random ensemble; uses
    ``2 Nₚ + 1`` deterministic sigma points per step.

    Attributes:
        forward_fn: ``θ → G(θ)``.
        obs: Observation vector ``y``.
        noise_cov: Observation noise covariance ``Γ``.
        scheduler: Step-size strategy. Ignored when ``config`` is
            provided.
        alpha, beta, kappa: Unscented tuning parameters; see
            :func:`~filterax._processes._processes.sigma_points`.
        config: Optional :class:`ProcessConfig`. When provided, **both**
            its ``scheduler`` and ``n_iterations`` fields take precedence
            over the module-level ``scheduler`` and the ``50``-iteration
            default.
    """

    forward_fn: ForwardFn
    obs: Float[Array, " N_d"]
    noise_cov: lx.AbstractLinearOperator
    scheduler: AbstractScheduler = eqx.field(default_factory=DataMisfitController)
    alpha: float = eqx.field(static=True, default=1.0)
    beta: float = eqx.field(static=True, default=2.0)
    kappa: float = eqx.field(static=True, default=0.0)
    config: ProcessConfig | None = None

    def run(
        self,
        init_mean: Float[Array, " N_p"],
        init_cov: lx.AbstractLinearOperator,
    ) -> ProcessResult:
        sched = _resolve_scheduler(self.scheduler, self.config)
        n_iter = _resolve_n_iterations(self.config, default=50)
        process = _UKI(
            scheduler=sched,
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa,
        )
        state = process.init_parametric(init_mean, init_cov, self.obs, self.noise_cov)
        # Drive the loop in parametric form for clarity; sample-point
        # evaluation happens once per iteration.
        from filterax._processes._processes import sigma_points

        algo_time = jnp.asarray(0.0)
        algo_times: list[Float[Array, ""]] = []
        converged = False
        actual = 0
        for _ in range(n_iter):
            points, _wm, _wc = sigma_points(
                state.mean,
                state.covariance,
                alpha=self.alpha,
                beta=self.beta,
                kappa=self.kappa,
            )
            evals = _vmap_forward(self.forward_fn, points)
            # Determine dt via the scheduler — wrap the parametric state
            # in a ProcessState shim so DataMisfitController can see the
            # forward evals.
            shim = ProcessState(
                particles=points,
                forward_evals=evals,
                obs=self.obs,
                noise_cov=self.noise_cov,
                step=state.step,
                algo_time=algo_time,
            )
            dt = sched.get_dt(shim)
            state = process.update_parametric(
                state, evals, obs=self.obs, noise_cov=self.noise_cov, dt=dt
            )
            algo_time = algo_time + dt
            algo_times.append(algo_time)
            actual += 1
            if float(algo_time) >= 1.0 - 1e-9:
                converged = True
                break

        return ProcessResult(
            particles=None,
            mean=state.mean,
            covariance=state.covariance,
            history_particles=None,
            history_algo_time=jnp.stack(algo_times) if algo_times else jnp.zeros(0),
            n_iterations=actual,
            converged=converged,
        )


# Re-exports used by filterax.__init__ — keep here for symmetry with the
# L2 filter wrappers in :mod:`filterax._filters._models`.
__all__ = ["EKI", "EKS", "UKI", "ProcessResult"]

# ``_`` markers below silence the unused-import warnings for symbols
# (PRNGKeyArray, FixedScheduler) that are part of the public surface
# but only referenced via re-export by ``filterax/__init__.py``.
_ = PRNGKeyArray
_ = FixedScheduler
_ = UKIState

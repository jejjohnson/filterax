"""Layer-1 ensemble Kalman processes (EKI, EKS, UKI, …).

Iterative ensemble methods for derivative-free parameter estimation. All
methods share the same outer loop — the caller supplies forward-model
evaluations ``G(θ⁽ʲ⁾)`` each step; the process supplies the ensemble
update rule.

Notation
--------
* ``J`` — ensemble (or sigma-point) size
* ``Nₚ`` — parameter dimension
* ``N_d`` — observation / data dimension
* ``Θ ∈ ℝ^{J×Nₚ}``  — ensemble of parameter vectors, rows are members
* ``Y = G(Θ) ∈ ℝ^{J×N_d}``  — forward evaluations, rows are members
* ``Θ′, Y′``  — centred anomaly matrices (rows sum to zero)
* ``θ̄, ȳ``  — ensemble means
* ``Γ``  — observation noise covariance (``state.noise_cov``)
* ``Δtₙ``  — step size (``scheduler.get_dt(state)``)
* ``S = Cᴳᴳ + Δtₙ⁻¹ Γ`` — Tikhonov-tempered innovation covariance

The Bessel-corrected ensemble statistics flow through
:func:`gaussx.ensemble_covariance` / :func:`gaussx.ensemble_cross_covariance`
so structured ``Γ`` (diagonal, low-rank, Toeplitz, …) is never densified.

References:
    Iglesias, M. A., Law, K. J. H., & Stuart, A. M. (2013).
    *Ensemble Kalman methods for inverse problems.* Inverse Problems,
    29(4), 045001.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import einx
import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray

from filterax._checks import check_ensemble_size
from filterax._primitives._statistics import ensemble_anomalies, ensemble_mean
from filterax._protocols import AbstractProcess, AbstractScheduler
from filterax._types import ProcessState, UKIState


# ──────────────────────────────────────────────────────────────────────
# shared helpers
# ──────────────────────────────────────────────────────────────────────


def _with_evals(state: ProcessState, evals: Float[Array, "J N_d"]) -> ProcessState:
    """Drop fresh forward evaluations into a state so the scheduler sees them."""
    return eqx.tree_at(lambda s: s.forward_evals, state, evals)


# When the scheduler reports ``dt = 0`` (DataMisfitController past
# ``algo_time = 1``), downstream ``1 / dt`` would blow up. The L2 run
# loops break out before that happens, but L1 ``update()`` and the
# optax wrappers can still be called in a fixed-length loop. We floor
# ``1 / dt`` at ``1 / _DT_EPSILON`` — large enough that the tempered
# noise ``Δt⁻¹ Γ`` dominates ``S``, so the EKI / UKI per-member delta
# (which is itself ``∝ dt``) ends up identically zero. Repeated calls
# after convergence are therefore no-ops rather than NaNs.
_DT_EPSILON: float = 1e-30


def _safe_inv_dt(dt: Float[Array, ""]) -> Float[Array, ""]:
    """Reciprocal of ``dt`` with a tiny floor so post-convergence calls are stable."""
    return 1.0 / jnp.maximum(dt, _DT_EPSILON)


def _scale_noise(
    noise_cov: lx.AbstractLinearOperator, factor: Float[Array, ""]
) -> lx.AbstractLinearOperator:
    """Return ``factor · Γ`` preserving structural type when possible.

    Diagonal ``Γ`` scales its diagonal in place (no matrix formed); any
    other operator falls back to a dense ``MatrixLinearOperator`` since
    lineax's structural solvers don't dispatch through
    :class:`gaussx.ScaledOperator`.
    """
    if isinstance(noise_cov, lx.DiagonalLinearOperator):
        return lx.DiagonalLinearOperator(factor * lx.diagonal(noise_cov))
    dense = factor * noise_cov.as_matrix()
    return lx.MatrixLinearOperator(dense, tags=lx.positive_semidefinite_tag)


def _innovation_covariance(
    obs_anom: Float[Array, "J N_d"],
    noise_cov: lx.AbstractLinearOperator,
    inv_dt: Float[Array, ""],
) -> gaussx.LowRankUpdate:
    r"""Build ``S = Cᴳᴳ + Δt⁻¹ Γ`` as a low-rank update.

    The ensemble obs-space covariance ``Cᴳᴳ`` is rank ``≤ J − 1`` and is
    composed with the (Δt-scaled) noise base so gaussx dispatches solves
    via the Woodbury identity. ``_scale_noise`` keeps the base
    structural when ``Γ`` is diagonal — no dense ``(N_d, N_d)`` block is
    formed.
    """
    cov = gaussx.ensemble_covariance(obs_anom, bessel=True)
    scaled_noise = _scale_noise(noise_cov, inv_dt)
    return gaussx.LowRankUpdate(scaled_noise, cov.U)


def _eki_delta(
    particles: Float[Array, "J N_p"],
    forward_evals: Float[Array, "J N_d"],
    obs: Float[Array, " N_d"],
    noise_cov: lx.AbstractLinearOperator,
    dt: Float[Array, ""],
) -> Float[Array, "J N_p"]:
    r"""Per-member EKI increment.

    $$
    \delta\theta^{(j)} = \Delta t\, C^{\theta G} S^{-1} \big(y - G^{(j)}\big)
    $$

    ``S`` is built as a low-rank update so the solve is Woodbury when
    ``Γ`` is structured. The cross-covariance ``Cᶿᴳ`` is a dense
    ``(Nₚ, N_d)`` matrix because ``N_d`` is typically small.

    Cost: ``O(J² N_d + J Nₚ N_d)`` per step.
    """
    C_theta_G = gaussx.ensemble_cross_covariance(
        particles, forward_evals, bessel=True
    )  # (Nₚ, N_d)
    obs_anom = ensemble_anomalies(forward_evals)
    S = _innovation_covariance(obs_anom, noise_cov, _safe_inv_dt(dt))
    # innovations[j, d] = y[d] − G[j, d]
    innovations = einx.subtract("d, j d -> j d", obs, forward_evals)
    # Solve once for each per-member innovation: S⁻¹ rⱼ.
    S_inv_innov = gaussx.solve_rows(S, innovations)  # (J, N_d)
    # δθⱼ = Δt · Cᶿᴳ (S⁻¹ rⱼ).
    return dt * einx.dot("p d, j d -> j p", C_theta_G, S_inv_innov)


# ──────────────────────────────────────────────────────────────────────
# EKI — Ensemble Kalman Inversion
# ──────────────────────────────────────────────────────────────────────


class EKI(AbstractProcess, strict=True):
    r"""Ensemble Kalman Inversion (Iglesias, Law & Stuart 2013).

    Iterative ensemble method that drives ``θ⁽ʲ⁾`` toward a MAP /
    regularised-least-squares solution. The update for each member:

    $$
    \theta^{(j)}_{n+1} = \theta^{(j)}_n + \Delta t_n\, C^{\theta G}_n
        \big(C^{G G}_n + \Delta t_n^{-1} \Gamma\big)^{-1}
        \big(y - G(\theta^{(j)}_n)\big)
    $$

    The ensemble collapses to a point as ``algo_time → 1``: spread → 0,
    no posterior uncertainty (see :class:`EKS_Process` if you need
    samples). Works in the underdetermined regime ``J < Nₚ``.

    Attributes:
        scheduler: Step-size strategy. Use
            :class:`~filterax.DataMisfitController` for the standard
            adaptive recipe.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from filterax import FixedScheduler
        >>> from filterax.processes import EKI
        >>> G = jnp.array([[1.0, 0.5], [-0.3, 1.2]])
        >>> obs = jnp.array([0.5, 0.9])
        >>> noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> eki = EKI(scheduler=FixedScheduler(dt=1.0))
        >>> particles = jax.random.normal(jax.random.key(0), (4, 2))
        >>> state = eki.init(particles, obs, noise)
        >>> evals = jax.vmap(lambda theta: G @ theta)(state.particles)
        >>> state = eki.update(state, evals)
        >>> state.particles.shape, int(state.step)
        ((4, 2), 1)
    """

    scheduler: AbstractScheduler

    def init(
        self,
        particles: Float[Array, "J N_p"],
        obs: Float[Array, " N_d"],
        noise_cov: lx.AbstractLinearOperator,
    ) -> ProcessState:
        check_ensemble_size(particles.shape[0])
        N_d = obs.shape[0]
        # Forward evaluations are filled in on the first update(); seed
        # with zeros so the state shape is JIT-stable.
        zero_evals = jnp.zeros((particles.shape[0], N_d), dtype=particles.dtype)
        return ProcessState(
            particles=particles,
            forward_evals=zero_evals,
            obs=obs,
            noise_cov=noise_cov,
            step=jnp.asarray(0, dtype=jnp.int32),
            algo_time=jnp.asarray(0.0, dtype=particles.dtype),
        )

    def update(
        self,
        state: ProcessState,
        forward_evals: Float[Array, "J N_d"],
        **_: Any,
    ) -> ProcessState:
        # Drop the fresh evals into state so the scheduler's misfit
        # computation sees them.
        state_now = _with_evals(state, forward_evals)
        dt = self.scheduler.get_dt(state_now)
        delta = _eki_delta(
            state.particles, forward_evals, state.obs, state.noise_cov, dt
        )
        particles_new = state.particles + delta
        return ProcessState(
            particles=particles_new,
            forward_evals=forward_evals,
            obs=state.obs,
            noise_cov=state.noise_cov,
            step=state.step + 1,
            algo_time=state.algo_time + dt,
        )


# ──────────────────────────────────────────────────────────────────────
# EKS — Ensemble Kalman Sampler (ALDI form)
# ──────────────────────────────────────────────────────────────────────


class EKS_Process(AbstractProcess, strict=True):
    r"""Ensemble Kalman Sampler / ALDI (Garbuno-Inigo et al. 2020).

    Interacting-Langevin ensemble that does **not** collapse — at
    stationarity the particles are approximate samples from the
    posterior ``p(θ | y)``. The per-step update is the EKI drift plus a
    finite-sample mean correction and an ensemble-preconditioned
    Brownian noise:

    $$
    \mathrm{d}\theta^{(j)} =
        C^{\theta G} \Gamma^{-1} \big(y - G^{(j)}\big)\, \mathrm{d}t
        - (N_p + 1)\, J^{-1} \big(\theta^{(j)} - \bar{\theta}\big)\,
          \mathrm{d}t
        + \sqrt{2\, C^{\theta\theta}}\; \mathrm{d}W^{(j)}
    $$

    The discrete-time recipe used here matches Garbuno-Inigo §4: an
    EKI-style mean update (with ``Δt⁻¹ Γ`` tempering), the
    ``(Nₚ + 1)/J`` drift term, and a noise term scaled by the
    *ensemble-space* sqrt of ``Cᶿᶿ`` (which avoids the dense ``Nₚ × Nₚ``
    Cholesky — see comments below).

    Use :class:`~filterax.EKSStableScheduler` to keep ``Δt`` inside the
    stability region.

    Attributes:
        scheduler: Step-size controller. Prefer EKSStableScheduler.
        key: Base PRNG key for the Brownian draws. Each step uses
            ``jr.fold_in(key, state.step)`` so successive updates are
            independent without manual key splitting.
    """

    scheduler: AbstractScheduler
    key: PRNGKeyArray

    def __init__(
        self,
        scheduler: AbstractScheduler,
        key: PRNGKeyArray | int = 0,
    ):
        self.scheduler = scheduler
        self.key = jr.PRNGKey(key) if isinstance(key, int) else key

    def init(
        self,
        particles: Float[Array, "J N_p"],
        obs: Float[Array, " N_d"],
        noise_cov: lx.AbstractLinearOperator,
    ) -> ProcessState:
        check_ensemble_size(particles.shape[0])
        N_d = obs.shape[0]
        zero_evals = jnp.zeros((particles.shape[0], N_d), dtype=particles.dtype)
        return ProcessState(
            particles=particles,
            forward_evals=zero_evals,
            obs=obs,
            noise_cov=noise_cov,
            step=jnp.asarray(0, dtype=jnp.int32),
            algo_time=jnp.asarray(0.0, dtype=particles.dtype),
        )

    def update(
        self,
        state: ProcessState,
        forward_evals: Float[Array, "J N_d"],
        **_: Any,
    ) -> ProcessState:
        state_now = _with_evals(state, forward_evals)
        dt = self.scheduler.get_dt(state_now)
        J, N_p = state.particles.shape

        # 1. EKI-style drift toward the data.
        drift_data = _eki_delta(
            state.particles, forward_evals, state.obs, state.noise_cov, dt
        )

        # 2. Mean-pull term ``− (Nₚ + 1) / J (θ − θ̄) Δt``.
        anom = ensemble_anomalies(state.particles)
        drift_pull = -((N_p + 1) / J) * dt * anom

        # 3. Ensemble-preconditioned noise:
        #     η⁽ʲ⁾ = √(2 Δt) · sqrt(Cᶿᶿ) · ξ⁽ʲ⁾,    ξ ~ 𝒩(0, I_{Nₚ}).
        # Using the ensemble sqrt ``sqrt(Cᶿᶿ) = X′ᵀ / √(J − 1)`` (i.e.
        # the LowRankUpdate's ``U`` factor) lets us draw the noise in
        # ensemble space (J-dimensional) and lift it: cost ``O(J²·Nₚ)``
        # instead of a ``Nₚ × Nₚ`` dense Cholesky.
        step_key = jr.fold_in(self.key, state.step)
        xi = jr.normal(step_key, (J, J), dtype=state.particles.dtype)
        # The ensemble sqrt has shape (Nₚ, J): U = X′ᵀ / √(J−1).
        # So η = √(2 Δt) · ξ U ᵀ, applied row-wise: η⁽ʲ⁾ = √(2 Δt) Σₖ ξⱼₖ Uᵀₖ
        # which collapses to ``ξ @ X′ / √(J−1)``.
        noise = (
            jnp.sqrt(2.0 * dt) * einx.dot("j k, k p -> j p", xi, anom) / jnp.sqrt(J - 1)
        )

        particles_new = state.particles + drift_data + drift_pull + noise
        return ProcessState(
            particles=particles_new,
            forward_evals=forward_evals,
            obs=state.obs,
            noise_cov=state.noise_cov,
            step=state.step + 1,
            algo_time=state.algo_time + dt,
        )


# ──────────────────────────────────────────────────────────────────────
# UKI — Unscented Kalman Inversion (parametric)
# ──────────────────────────────────────────────────────────────────────


def sigma_points(
    mean: Float[Array, " N_p"],
    cov: lx.AbstractLinearOperator,
    *,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> tuple[
    Float[Array, "twoNp1 N_p"],
    Float[Array, " twoNp1"],
    Float[Array, " twoNp1"],
]:
    r"""Generate ``2 Nₚ + 1`` deterministic sigma points (Wan & van der Merwe 2000).

    For mean ``μ`` and covariance ``Σ``:

    $$
    \lambda = \alpha^2 (N_p + \kappa) - N_p, \qquad
    c = \sqrt{N_p + \lambda}
    $$

    $$
    \chi^0 = \mu, \qquad
    \chi^{+}_j = \mu + c \, [\sqrt{\Sigma}]_{:,j}, \qquad
    \chi^{-}_j = \mu - c \, [\sqrt{\Sigma}]_{:,j}
    $$

    with weights for the mean and covariance:

    $$
    \begin{aligned}
    W_m^0 &= \frac{\lambda}{N_p + \lambda}, &
    W_c^0 &= W_m^0 + (1 - \alpha^2 + \beta), \\
    W_m^i &= W_c^i = \frac{1}{2 (N_p + \lambda)} & &\text{for } i \neq 0.
    \end{aligned}
    $$

    ``√Σ`` comes from :func:`gaussx.root_decomposition` so structured
    ``Σ`` (diagonal, low-rank, …) skips the dense Cholesky.

    Returns:
        ``(points, mean_weights, cov_weights)`` with shapes
        ``(2 Nₚ + 1, Nₚ)``, ``(2 Nₚ + 1,)``, ``(2 Nₚ + 1,)``.
    """
    N_p = mean.shape[0]
    lam = alpha**2 * (N_p + kappa) - N_p
    c = jnp.sqrt(N_p + lam)

    # Sigma offsets — columns of c · √Σ. UKI is parametric so Σ is
    # always a (Nₚ, Nₚ) dense or diagonal block, and Cholesky is both
    # cheap and exact. (The Lanczos default in gaussx is meant for huge
    # operators and produces NaN here on small exactly-isotropic Σ.)
    root = gaussx.root_decomposition(cov, method="cholesky").root
    offsets = c * root  # (Nₚ, Nₚ); column j is c · [√Σ]_{:,j}.

    # χ⁺ⱼ = μ + [c√Σ]_{:,j} — transpose so each row is one sigma point.
    plus = offsets.T + mean[None, :]
    minus = -offsets.T + mean[None, :]
    points = jnp.concatenate([mean[None, :], plus, minus], axis=0)

    w0_mean = lam / (N_p + lam)
    w0_cov = w0_mean + (1.0 - alpha**2 + beta)
    w_other = 1.0 / (2.0 * (N_p + lam))
    weights_mean = jnp.concatenate([jnp.asarray([w0_mean]), jnp.full(2 * N_p, w_other)])
    weights_cov = jnp.concatenate([jnp.asarray([w0_cov]), jnp.full(2 * N_p, w_other)])
    return points, weights_mean, weights_cov


class UKI(AbstractProcess, strict=True):
    r"""Unscented Kalman Inversion (Huang, Schneider & Stuart 2022).

    Parametric inversion that maintains a Gaussian belief
    ``θ ~ 𝒩(μₙ, Σₙ)`` instead of an explicit ensemble. Each step
    generates ``2 Nₚ + 1`` sigma points, evaluates the forward model on
    them, and applies a Kalman update on the unscented statistics:

    $$
    \begin{aligned}
    \mu_{n+1} &= \mu_n
        + \Delta t\, C^{\theta y} S_n^{-1} \big(y - \hat{y}_0\big) \\
    \Sigma_{n+1} &= \Sigma_n
        - \Delta t\, C^{\theta y} S_n^{-1} C^{\theta y \top}
    \end{aligned}
    $$

    Deterministic (no sampling noise), no ensemble collapse, and the
    covariance ``Σₙ`` gives a calibrated posterior in the linear-Gaussian
    case.

    The L1 protocol uses :class:`ProcessState` for compatibility with
    ``DataMisfitController`` — the field ``particles`` carries the sigma
    points (last 2 Nₚ + 1 evaluations); the parametric state lives in
    :class:`UKIState` and is returned by :meth:`init_parametric` /
    :meth:`update_parametric` for callers that prefer the explicit form.

    Attributes:
        scheduler: Step-size strategy.
        alpha: Unscented spread parameter; usually small (e.g. ``1e-3``)
            for highly nonlinear ``G``, ``1.0`` for benign problems.
        beta: Prior moment weighting; ``2.0`` is optimal for Gaussian
            priors.
        kappa: Secondary scaling, usually ``0`` or ``3 − Nₚ``.
    """

    scheduler: AbstractScheduler
    alpha: float = eqx.field(static=True, default=1.0)
    beta: float = eqx.field(static=True, default=2.0)
    kappa: float = eqx.field(static=True, default=0.0)

    # --- parametric API ---------------------------------------------------

    def init_parametric(
        self,
        mean: Float[Array, " N_p"],
        covariance: lx.AbstractLinearOperator,
        obs: Float[Array, " N_d"],
        noise_cov: lx.AbstractLinearOperator,
    ) -> UKIState:
        """Initialise the parametric ``(μ, Σ)`` state."""
        return UKIState(
            mean=mean,
            covariance=covariance,
            step=jnp.asarray(0, dtype=jnp.int32),
        )

    def update_parametric(
        self,
        state: UKIState,
        forward_evals: Float[Array, "twoNp1 N_d"],
        *,
        obs: Float[Array, " N_d"],
        noise_cov: lx.AbstractLinearOperator,
        dt: Float[Array, ""],
    ) -> UKIState:
        r"""One UKI step on the parametric state.

        Args:
            state: Current ``(μₙ, Σₙ)`` belief.
            forward_evals: Forward-model evaluations at the
                ``2 Nₚ + 1`` sigma points (in the order returned by
                :func:`sigma_points`).
            obs: Observation vector.
            noise_cov: Observation noise covariance ``Γ``.
            dt: Step size for this iteration.

        Returns:
            Updated :class:`UKIState`.
        """
        points, w_mean, w_cov = sigma_points(
            state.mean,
            state.covariance,
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa,
        )
        # Weighted obs-space mean.
        y_mean = einx.dot("k, k d -> d", w_mean, forward_evals)
        # Weighted cross-covariance Cᶿʸ = Σₖ Wcₖ (χₖ − μ) (ŷₖ − ȳ)ᵀ.
        theta_anom = einx.subtract("k p, p -> k p", points, state.mean)
        y_anom = einx.subtract("k d, d -> k d", forward_evals, y_mean)
        weighted_theta = einx.multiply("k p, k -> k p", theta_anom, w_cov)
        C_theta_y = einx.dot("k p, k d -> p d", weighted_theta, y_anom)
        # Sᵧᵧ = Σₖ Wcₖ (ŷₖ − ȳ)(ŷₖ − ȳ)ᵀ + Δt⁻¹ Γ.
        weighted_y = einx.multiply("k d, k -> k d", y_anom, w_cov)
        S_dense = einx.dot("k a, k b -> a b", weighted_y, y_anom)
        # Tempered Kalman solve.
        S_tempered = S_dense + _safe_inv_dt(dt) * noise_cov.as_matrix()
        innovation = obs - y_mean
        K = jnp.linalg.solve(S_tempered.T, C_theta_y.T).T  # (Nₚ, N_d)

        mean_new = state.mean + dt * einx.dot("p d, d -> p", K, innovation)
        # Σ_{n+1} = Σ̂ − Δt · K · S_tempered · K^T  (Huang 2022 eq. 14).
        Sigma_dense = state.covariance.as_matrix()
        K_S = einx.dot("p a, a b -> p b", K, S_tempered)  # (Nₚ, N_d)
        Sigma_new = Sigma_dense - dt * einx.dot("p d, q d -> p q", K_S, K)
        Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
        return UKIState(
            mean=mean_new,
            covariance=lx.MatrixLinearOperator(
                Sigma_new, tags=lx.positive_semidefinite_tag
            ),
            step=state.step + 1,
        )

    # --- AbstractProcess (ensemble-typed) API ---------------------------------

    def init(
        self,
        particles: Float[Array, "J N_p"],
        obs: Float[Array, " N_d"],
        noise_cov: lx.AbstractLinearOperator,
    ) -> ProcessState:
        r"""Initialise from an ensemble proxy for the parametric belief.

        Computes the empirical mean and covariance of ``particles`` and
        stores them in the leading sigma point block. Provided so UKI
        slots into the same ``init / update`` loop as the
        ensemble-typed processes; for the cleaner parametric API use
        :meth:`init_parametric` / :meth:`update_parametric`.
        """
        check_ensemble_size(particles.shape[0])
        mean = ensemble_mean(particles)
        cov_op = gaussx.ensemble_covariance(particles, bessel=True)
        cov = lx.MatrixLinearOperator(
            cov_op.as_matrix(), tags=lx.positive_semidefinite_tag
        )
        points, _wm0, _wc0 = sigma_points(
            mean, cov, alpha=self.alpha, beta=self.beta, kappa=self.kappa
        )
        N_d = obs.shape[0]
        zero_evals = jnp.zeros((points.shape[0], N_d), dtype=particles.dtype)
        return ProcessState(
            particles=points,
            forward_evals=zero_evals,
            obs=obs,
            noise_cov=noise_cov,
            step=jnp.asarray(0, dtype=jnp.int32),
            algo_time=jnp.asarray(0.0, dtype=particles.dtype),
        )

    def update(
        self,
        state: ProcessState,
        forward_evals: Float[Array, "twoNp1 N_d"],
        **_: Any,
    ) -> ProcessState:
        state_now = _with_evals(state, forward_evals)
        dt = self.scheduler.get_dt(state_now)
        # Reconstruct the parametric state from the sigma points stored
        # on ``state.particles``: the first point is the mean, and the
        # sample covariance of the remaining 2 Nₚ points recovers Σ /
        # (N_p + λ).
        points = state.particles
        N_p = (points.shape[0] - 1) // 2
        mean = points[0]
        # Recover Σ from the ``Nₚ`` positive-offset sigma points alone:
        # by construction (see :func:`sigma_points`) we have
        #   χ⁺ⱼ − μ = c · [√Σ]_{:,j},   c = √(Nₚ + λ),
        # so  Σ_k (χ⁺ⱼ − μ)(χ⁺ⱼ − μ)ᵀ = c² Σ = (Nₚ + λ) Σ.
        # Dividing by ``(Nₚ + λ)`` therefore returns Σ exactly. The
        # negative-offset block is the reflection and contributes
        # nothing new, so we skip it.
        lam = self.alpha**2 * (N_p + self.kappa) - N_p
        scale = N_p + lam
        offsets = points[1 : 1 + N_p] - mean[None, :]  # (Nₚ, Nₚ)
        Sigma_mat = einx.dot("k p, k q -> p q", offsets, offsets) / scale
        Sigma_mat = 0.5 * (Sigma_mat + Sigma_mat.T)
        cov_op = lx.MatrixLinearOperator(Sigma_mat, tags=lx.positive_semidefinite_tag)
        parametric = UKIState(
            mean=mean,
            covariance=cov_op,
            step=state.step,
        )
        new_parametric = self.update_parametric(
            parametric, forward_evals, obs=state.obs, noise_cov=state.noise_cov, dt=dt
        )
        new_points, _wm, _wc = sigma_points(
            new_parametric.mean,
            new_parametric.covariance,
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa,
        )
        return ProcessState(
            particles=new_points,
            forward_evals=forward_evals,
            obs=state.obs,
            noise_cov=state.noise_cov,
            step=state.step + 1,
            algo_time=state.algo_time + dt,
        )


# ──────────────────────────────────────────────────────────────────────
# Advanced processes
# ──────────────────────────────────────────────────────────────────────


class ETKI(AbstractProcess, strict=True):
    r"""Ensemble Transform Kalman Inversion.

    The "transform" variant of EKI. Conceptually identical to
    :class:`EKI` on the analysis ensemble (in batch form both write the
    same linear-Gaussian posterior); the *intended* distinction is that
    ETKI rewrites the per-step solve in the ``J × J`` ensemble subspace,
    yielding ``O(J² N_d)`` cost instead of ``O(N_d³)`` when ``N_d ≫ J``.

    The current implementation defers to the same Bessel-corrected EKI
    delta as :class:`EKI` — gaussx routes the inner solve through the
    Woodbury identity, so the dense ``(N_d, N_d)`` block is *not* formed
    even though the outer call signature is identical. A future
    optimisation can specialise the transform path explicitly when
    ``N_d`` is extreme; for now the two classes coexist so users can
    select the conceptually-appropriate one without paying a different
    cost.

    Attributes:
        scheduler: Step-size strategy.
    """

    scheduler: AbstractScheduler

    def init(
        self,
        particles: Float[Array, "J N_p"],
        obs: Float[Array, " N_d"],
        noise_cov: lx.AbstractLinearOperator,
    ) -> ProcessState:
        check_ensemble_size(particles.shape[0])
        N_d = obs.shape[0]
        zero_evals = jnp.zeros((particles.shape[0], N_d), dtype=particles.dtype)
        return ProcessState(
            particles=particles,
            forward_evals=zero_evals,
            obs=obs,
            noise_cov=noise_cov,
            step=jnp.asarray(0, dtype=jnp.int32),
            algo_time=jnp.asarray(0.0, dtype=particles.dtype),
        )

    def update(
        self,
        state: ProcessState,
        forward_evals: Float[Array, "J N_d"],
        **_: Any,
    ) -> ProcessState:
        state_now = _with_evals(state, forward_evals)
        dt = self.scheduler.get_dt(state_now)
        delta = _eki_delta(
            state.particles, forward_evals, state.obs, state.noise_cov, dt
        )
        return ProcessState(
            particles=state.particles + delta,
            forward_evals=forward_evals,
            obs=state.obs,
            noise_cov=state.noise_cov,
            step=state.step + 1,
            algo_time=state.algo_time + dt,
        )


class GNKI(AbstractProcess, strict=True):
    r"""Gauss-Newton Kalman Inversion.

    Estimates the Jacobian explicitly from ensemble perturbations
    (``J̃ ≈ Cᶿᴳ (Cᶿᶿ)⁻¹``) and applies a Gauss-Newton update with prior
    pull-back:

    $$
    \delta\theta^{(j)} = K \big(y - G^{(j)}\big)
        + \big(I - K \tilde{J}\big)\big(m_0 - \theta^{(j)}\big), \qquad
    K = \big(\tilde{J}^{\top} \Gamma^{-1} \tilde{J}
        + \Sigma_0^{-1}\big)^{-1} \tilde{J}^{\top} \Gamma^{-1}.
    $$

    Faster convergence
    than :class:`EKI` for well-conditioned problems and requires
    ``J > Nₚ`` so ``Cᶿᶿ`` is invertible. In the linear-Gaussian limit
    GNKI recovers the exact posterior mean *and* covariance.

    Attributes:
        scheduler: Step-size strategy.
        prior_mean: Prior mean ``m₀ ∈ ℝ^{Nₚ}``.
        prior_cov: Prior covariance ``Σ₀``.
        jitter: Tikhonov regularisation added to ``Cᶿᶿ`` before
            inversion. Needed when ``J`` is barely larger than ``Nₚ``.
    """

    scheduler: AbstractScheduler
    prior_mean: Float[Array, " N_p"]
    prior_cov: lx.AbstractLinearOperator
    jitter: float = eqx.field(static=True, default=1e-6)

    def init(
        self,
        particles: Float[Array, "J N_p"],
        obs: Float[Array, " N_d"],
        noise_cov: lx.AbstractLinearOperator,
    ) -> ProcessState:
        J, N_p = particles.shape
        check_ensemble_size(J)
        # GNKI inverts the sample parameter covariance Cᶿᶿ ∈ ℝ^{Nₚ×Nₚ}.
        # With J ≤ Nₚ that matrix has rank ≤ J − 1 < Nₚ and is singular
        # — the jitter regulariser would dominate and silently corrupt
        # the Gauss-Newton step. Fail fast so users know to switch to
        # EKI (which works in the underdetermined regime).
        if J <= N_p:  # noqa: SIM300 — phrasing matches "J > Nₚ" requirement
            raise ValueError(
                "GNKI requires J > Nₚ so the sample parameter covariance "
                f"Cᶿᶿ is invertible; got J={J} and Nₚ={N_p}. Use "
                "filterax.EKI for the underdetermined regime."
            )
        N_d = obs.shape[0]
        zero_evals = jnp.zeros((J, N_d), dtype=particles.dtype)
        return ProcessState(
            particles=particles,
            forward_evals=zero_evals,
            obs=obs,
            noise_cov=noise_cov,
            step=jnp.asarray(0, dtype=jnp.int32),
            algo_time=jnp.asarray(0.0, dtype=particles.dtype),
        )

    def update(
        self,
        state: ProcessState,
        forward_evals: Float[Array, "J N_d"],
        **_: Any,
    ) -> ProcessState:
        state_now = _with_evals(state, forward_evals)
        dt = self.scheduler.get_dt(state_now)
        N_p = state.particles.shape[1]

        # Ensemble Jacobian via cross-covariance solve.
        C_theta_theta = gaussx.ensemble_covariance(
            state.particles, bessel=True
        ).as_matrix() + self.jitter * jnp.eye(N_p)
        C_theta_G = gaussx.ensemble_cross_covariance(
            state.particles, forward_evals, bessel=True
        )  # (Nₚ, N_d)
        J_tilde = jnp.linalg.solve(C_theta_theta, C_theta_G)  # (Nₚ, N_d)

        # Gain K = (J̃ᵀ Γ⁻¹ J̃ + Σ₀⁻¹)⁻¹ J̃ᵀ Γ⁻¹.
        # solve_rows treats each row of J_tilde as an N_d vector and
        # returns (Γ⁻¹ J̃)_k row-by-row → shape (Nₚ, N_d).
        Gamma_inv_J = gaussx.solve_rows(state.noise_cov, J_tilde)  # (Nₚ, N_d)
        precision = einx.dot("p d, q d -> p q", J_tilde, Gamma_inv_J) + jnp.linalg.inv(
            self.prior_cov.as_matrix()
        )
        precision = 0.5 * (precision + precision.T)
        K_gain = jnp.linalg.solve(precision, Gamma_inv_J)  # (Nₚ, N_d)

        # Per-member update.
        residuals = einx.subtract("d, j d -> j d", state.obs, forward_evals)
        prior_pull = einx.subtract("p, j p -> j p", self.prior_mean, state.particles)
        data_step = einx.dot("p d, j d -> j p", K_gain, residuals)
        # (I − K J̃) m₀-pull term, expanded as m₀-pull − K J̃ (m₀-pull).
        J_pull = einx.dot("p d, j p -> j d", J_tilde, prior_pull)
        prior_step = prior_pull - einx.dot("p d, j d -> j p", K_gain, J_pull)
        delta = dt * (data_step + prior_step)
        particles_new = state.particles + delta
        return ProcessState(
            particles=particles_new,
            forward_evals=forward_evals,
            obs=state.obs,
            noise_cov=state.noise_cov,
            step=state.step + 1,
            algo_time=state.algo_time + dt,
        )


class SparseInversion(AbstractProcess, strict=True):
    r"""Sparse Ensemble Kalman Inversion (Schneider, Stuart & Wu 2022).

    Standard EKI step followed by an L¹ proximal soft-threshold:

    $$
    \mathrm{prox}_{\lambda \|\cdot\|_1}(z)_i =
        \mathrm{sign}(z_i) \cdot \max\big(|z_i| - \lambda,\, 0\big)
    $$

    drives inactive parameters exactly to zero. Useful for variable
    selection / sparse physics discovery / sensor placement.

    Attributes:
        scheduler: Step-size strategy.
        penalty_weight: ``λ > 0``. Larger ``λ`` → more sparsity.
    """

    scheduler: AbstractScheduler
    penalty_weight: float = eqx.field(static=True, default=0.1)

    def init(
        self,
        particles: Float[Array, "J N_p"],
        obs: Float[Array, " N_d"],
        noise_cov: lx.AbstractLinearOperator,
    ) -> ProcessState:
        check_ensemble_size(particles.shape[0])
        N_d = obs.shape[0]
        zero_evals = jnp.zeros((particles.shape[0], N_d), dtype=particles.dtype)
        return ProcessState(
            particles=particles,
            forward_evals=zero_evals,
            obs=obs,
            noise_cov=noise_cov,
            step=jnp.asarray(0, dtype=jnp.int32),
            algo_time=jnp.asarray(0.0, dtype=particles.dtype),
        )

    def update(
        self,
        state: ProcessState,
        forward_evals: Float[Array, "J N_d"],
        **_: Any,
    ) -> ProcessState:
        state_now = _with_evals(state, forward_evals)
        dt = self.scheduler.get_dt(state_now)
        delta = _eki_delta(
            state.particles, forward_evals, state.obs, state.noise_cov, dt
        )
        # Soft-threshold proximal operator scaled by Δt · λ.
        lam = self.penalty_weight * dt
        z = state.particles + delta
        particles_new = jnp.sign(z) * jnp.maximum(jnp.abs(z) - lam, 0.0)
        return ProcessState(
            particles=particles_new,
            forward_evals=forward_evals,
            obs=state.obs,
            noise_cov=state.noise_cov,
            step=state.step + 1,
            algo_time=state.algo_time + dt,
        )


class TEKI(AbstractProcess, strict=True):
    r"""Tikhonov-regularised Ensemble Kalman Inversion (Chada et al. 2019).

    Standard EKI on an augmented system that includes the parameters in
    observation space:

    $$
    \tilde{G}(\theta) = \big(G(\theta),\, \theta\big)^{\top}, \qquad
    \tilde{y} = (y,\, m_0)^{\top}, \qquad
    \tilde{\Gamma} = \mathrm{blockdiag}(\Gamma,\, \Sigma_0)
    $$

    The augmented identity block in ``Cᶿᴳ̃`` pulls particles toward the
    prior mean ``m₀`` and prevents the ensemble from drifting arbitrarily
    far from the prior — useful for ill-posed problems where vanilla EKI
    diverges. Converges to the MAP estimate (not MLE).

    Attributes:
        scheduler: Step-size strategy.
        prior_mean: Prior mean ``m₀ ∈ ℝ^{Nₚ}``.
        prior_cov: Prior covariance ``Σ₀``.
    """

    scheduler: AbstractScheduler
    prior_mean: Float[Array, " N_p"]
    prior_cov: lx.AbstractLinearOperator

    def init(
        self,
        particles: Float[Array, "J N_p"],
        obs: Float[Array, " N_d"],
        noise_cov: lx.AbstractLinearOperator,
    ) -> ProcessState:
        check_ensemble_size(particles.shape[0])
        N_d = obs.shape[0]
        N_p = particles.shape[1]
        zero_evals = jnp.zeros((particles.shape[0], N_d + N_p), dtype=particles.dtype)
        # Store the augmented obs ỹ = (y, m₀) and augmented Γ̃ = blockdiag.
        aug_obs = jnp.concatenate([obs, self.prior_mean])
        aug_noise = (
            lx.BlockDiagonalLinearOperator([noise_cov, self.prior_cov])
            if hasattr(lx, "BlockDiagonalLinearOperator")
            else _block_diag(noise_cov, self.prior_cov)
        )
        return ProcessState(
            particles=particles,
            forward_evals=zero_evals,
            obs=aug_obs,
            noise_cov=aug_noise,
            step=jnp.asarray(0, dtype=jnp.int32),
            algo_time=jnp.asarray(0.0, dtype=particles.dtype),
        )

    def update(
        self,
        state: ProcessState,
        forward_evals: Float[Array, "J N_d"],
        **_: Any,
    ) -> ProcessState:
        # Augment evals with the particles themselves: G̃(θ⁽ʲ⁾) = (G⁽ʲ⁾, θ⁽ʲ⁾).
        aug_evals = jnp.concatenate([forward_evals, state.particles], axis=1)
        state_now = _with_evals(state, aug_evals)
        dt = self.scheduler.get_dt(state_now)
        delta = _eki_delta(state.particles, aug_evals, state.obs, state.noise_cov, dt)
        particles_new = state.particles + delta
        return ProcessState(
            particles=particles_new,
            forward_evals=aug_evals,
            obs=state.obs,
            noise_cov=state.noise_cov,
            step=state.step + 1,
            algo_time=state.algo_time + dt,
        )


def _block_diag(
    A: lx.AbstractLinearOperator, B: lx.AbstractLinearOperator
) -> lx.MatrixLinearOperator:
    """Fallback dense block-diagonal for the augmented Γ̃ in TEKI.

    Used only when lineax does not expose ``BlockDiagonalLinearOperator``.
    The matrix is small (``N_d + Nₚ`` per side) so densifying is cheap.
    """
    A_mat = A.as_matrix()
    B_mat = B.as_matrix()
    n, m = A_mat.shape[0], B_mat.shape[0]
    out = jnp.zeros((n + m, n + m), dtype=A_mat.dtype)
    out = out.at[:n, :n].set(A_mat)
    out = out.at[n:, n:].set(B_mat)
    return lx.MatrixLinearOperator(out, tags=lx.positive_semidefinite_tag)


# ──────────────────────────────────────────────────────────────────────
# Convenience helper used by the L2 models.
# ──────────────────────────────────────────────────────────────────────


def forward_evaluate(
    forward_fn: Callable[[Float[Array, " N_p"]], Float[Array, " N_d"]],
    particles: Float[Array, "J N_p"],
) -> Float[Array, "J N_d"]:
    """vmap a forward model over an ensemble of parameter vectors."""
    return jax.vmap(forward_fn)(particles)

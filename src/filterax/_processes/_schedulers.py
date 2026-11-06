"""Step-size schedulers for ensemble Kalman processes.

Every EKP iteration is parameterised by an artificial-time step
``Δtₙ ∈ ℝ₊``. The role of the scheduler is to pick ``Δtₙ`` from the
current :class:`ProcessState` — either a constant, a misfit-adaptive
rule (Iglesias & Yang 2021), or a stability-controlled rule for the EKS
Langevin dynamics.

All schedulers implement :class:`AbstractScheduler` so they slot into
the L1 processes (``EKI``, ``EKS_Process``, ``UKI``, …) and the L2 run
loops without further glue. Schedulers are stateless ``eqx.Module``
subclasses — they receive the state and return a scalar.

Convention: ``algo_time = Σₙ Δtₙ``. Schedulers that drive convergence
arrange for ``algo_time → 1`` (Iglesias & Yang 2021); the L2 run loop can
break out when ``algo_time ≥ 1``.
"""

from __future__ import annotations

import einx
import equinox as eqx
import gaussx
import jax.numpy as jnp
from jaxtyping import Array, Float

from filterax._primitives._statistics import ensemble_anomalies
from filterax._protocols import AbstractScheduler
from filterax._types import ProcessState


class FixedScheduler(AbstractScheduler, strict=True):
    r"""Constant step size ``Δtₙ = dt``.

    The simplest scheduler. Requires manual tuning of ``dt`` to balance
    convergence speed against ensemble blow-up — too large and EKI's
    update blows past the data, too small and convergence is glacial.

    Attributes:
        dt: Positive scalar step size.

    Examples:
        >>> from filterax import FixedScheduler
        >>> sched = FixedScheduler(dt=0.5)
        >>> float(sched.get_dt(state=None))  # constant, ignores the state
        0.5
    """

    dt: float = eqx.field(static=True)

    def get_dt(self, state: ProcessState) -> Float[Array, ""]:
        del state
        return jnp.asarray(self.dt)


class DataMisfitController(AbstractScheduler, strict=True):
    r"""Data misfit controller of Iglesias & Yang (2021, eq. 14).

    With per-particle misfits ``Φⱼ = ½ ‖y − G(θ⁽ʲ⁾)‖²_{Γ⁻¹}``, their ensemble
    mean ``Φ̄`` and variance ``σ²_Φ``, and ``M`` observations:

    $$
    \Delta t_n = \min\left(
        \max\left(\frac{M}{2 \bar\Phi},\;
            \sqrt{\frac{M}{2 \sigma^2_\Phi}}\right),\;
        1 - t_n\right)
    $$

    Both candidates come from ``‖Γ^{-1/2} r‖² ~ χ²_M`` under the correct
    model, which has mean ``M`` and variance ``2M``. The rule has no tuning
    parameter. The ``min`` clamps the final step so ``algo_time`` lands
    exactly at ``1.0``, which is the standard EKI termination criterion;
    subsequent calls return ``Δt = 0`` (the L1 processes guard their ``1/Δt``
    paths with a small floor so further calls are no-ops rather than NaNs).

    An ensemble whose misfits are all equal has ``σ²_Φ = 0``, so the second
    candidate is infinite and the step is the whole remaining budget.

    The computation is :func:`gaussx.discrepancy_step_size`, which applies
    ``Γ⁻¹`` through a structured solve, so diagonal or low-rank ``Γ`` is never
    densified.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from filterax import DataMisfitController, ProcessState
        >>> state = ProcessState(
        ...     particles=jnp.zeros((3, 2)),
        ...     forward_evals=jnp.array([[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]),
        ...     obs=jnp.zeros(2),
        ...     noise_cov=lx.DiagonalLinearOperator(jnp.ones(2)),
        ...     step=jnp.asarray(0),
        ...     algo_time=jnp.asarray(0.0),
        ... )
        >>> dt = DataMisfitController().get_dt(state)
        >>> bool(0.0 < dt <= 1.0)
        True

    Reference:
        Iglesias, M. A. & Yang, Y. (2021). *Adaptive regularisation for
        ensemble Kalman inversion.* Inverse Problems, 37(2), 025008.
    """

    def get_dt(self, state: ProcessState) -> Float[Array, ""]:
        # 1 − tₙ, floored because a negative budget is not a step
        remaining = jnp.maximum(1.0 - state.algo_time, 0.0)
        # Δtₙ = min(max(M / 2Φ̄, √(M / 2σ²_Φ)), 1 − tₙ)
        return gaussx.discrepancy_step_size(
            state.forward_evals, state.obs, state.noise_cov, remaining=remaining
        )


def _posterior_stiffness_norm(state: ProcessState) -> Float[Array, ""]:
    """Frobenius norm of the ensemble-space posterior stiffness ``S``, floored."""
    num_members = state.particles.shape[0]
    # ΔG
    obs_anomalies = ensemble_anomalies(state.forward_evals)
    # Γ⁻¹ ΔGᵀ, row-wise
    precision_obs_anomalies = gaussx.solve_rows(state.noise_cov, obs_anomalies)
    # S = ΔG Γ⁻¹ ΔGᵀ / (J − 1)
    stiffness = einx.dot("j d, k d -> j k", obs_anomalies, precision_obs_anomalies) / (
        num_members - 1
    )
    if state.prior_cov is not None:
        # X′
        anomalies = ensemble_anomalies(state.particles)
        # Σ₀⁻¹ X′ᵀ, row-wise
        precision_anomalies = gaussx.solve_rows(state.prior_cov, anomalies)
        # S ← S + X′ Σ₀⁻¹ X′ᵀ / (J − 1)
        stiffness = stiffness + einx.dot(
            "j p, k p -> j k", anomalies, precision_anomalies
        ) / (num_members - 1)
    # ‖S‖²_F, floored inside the root so its gradient stays finite at S = 0
    squared_norm = einx.sum("[j k]", stiffness * stiffness)
    return jnp.sqrt(jnp.maximum(squared_norm, 1e-24))


class EKSStableScheduler(AbstractScheduler, strict=True):
    r"""Stability-aware step size for the EKS Langevin dynamics.

    Bounds ``Δt`` by the stiffness of the posterior as well as by the size of
    the ensemble preconditioner ``Cᶿᶿ``:

    $$
    \Delta t_n = \min\left(\text{max\_dt},\;
        \frac{\text{target}}{\|C^{\theta\theta}_n\|_F},\;
        \frac{\text{target\_stiffness}}{\|S_n\|_F}\right),
    \qquad
    S_n = \frac{\Delta G_n \Gamma^{-1} \Delta G_n^\top
        + X'_n \Sigma_0^{-1} X_n'^\top}{J - 1}
    $$

    ``ΔG_n`` and ``X′_n`` hold the anomalies of the forward evaluations and
    the particles, as rows. For a linear ``G`` with ``H = Gᵀ Γ⁻¹ G + Σ₀⁻¹``,
    ``S_n`` has the nonzero spectrum of ``Cᶿᶿ H``, the stiffness of the Euler-
    Maruyama step. The prior term is present when ``state.prior_cov`` is set.
    At the exact posterior ``Cᶿᶿ H = I``, so ``‖S‖_F = √Nₚ`` whatever the scale
    of the problem, and the step settles at about ``target_stiffness / √Nₚ``.

    The step sets the sampling bias. On a scalar linear model with a flat
    prior, the update's stationary variance solves
    ``C = C / (1 + Δt C H)² + 2 Δt C``, which gives

    $$
    \frac{C}{H^{-1}} = \frac{(1 - 2 \Delta t)^{-1/2} - 1}{\Delta t}
        \approx 1 + \tfrac{3}{2} \Delta t,
    $$

    with no stationary solution for ``Δt ≥ 1/2``. The prior step is implicit
    and also shrinks the noise, so a direction set by the prior alone has
    ``C / H⁻¹ = ((1 + 2 Δt)^{1/2} - 1) / Δt ≈ 1 - Δt / 2`` and is stable at any
    ``Δt``. The default ``target_stiffness`` aims for an excess variance of
    about 10% on a low-dimensional, data-dominated problem. All of this is
    computed in the ``(J, J)`` ensemble space, and ``Γ`` and ``Σ₀`` are applied
    through structured solves.

    Attributes:
        max_dt: Hard ceiling on the step size.
        target: Desired ``Δt × ‖Cᶿᶿ‖_F`` product.
        target_stiffness: Desired ``Δt × ‖S‖_F`` product.
    """

    max_dt: float = eqx.field(static=True, default=1.0)
    target: float = eqx.field(static=True, default=1.0)
    target_stiffness: float = eqx.field(static=True, default=0.1)

    def get_dt(self, state: ProcessState) -> Float[Array, ""]:
        # The Frobenius bound on top eigenvalue: ‖A‖₂ ≤ ‖A‖_F, computed by gaussx
        # in the (J, J) ensemble space.
        spectral_bound = gaussx.frobenius_norm(
            gaussx.ensemble_covariance(state.particles, bessel=True)
        )
        dt_stable = self.target / jnp.maximum(spectral_bound, 1e-12)
        # target_stiffness / ‖S‖_F
        dt_stiffness = self.target_stiffness / _posterior_stiffness_norm(state)
        dt_bound = jnp.minimum(dt_stable, dt_stiffness)
        return jnp.minimum(jnp.asarray(self.max_dt), dt_bound)

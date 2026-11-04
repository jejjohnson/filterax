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

import equinox as eqx
import gaussx
import jax.numpy as jnp
from jaxtyping import Array, Float

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


class EKSStableScheduler(AbstractScheduler, strict=True):
    r"""Stability-aware step size for the EKS Langevin dynamics.

    The EKS / ALDI SDE develops instability if ``Δt`` is too large
    relative to the spectral norm of the ensemble preconditioner
    ``Cᶿᶿ`` (Garbuno-Inigo et al. 2020 §5). We clip ``Δt`` so the
    drift term is bounded:

    $$
    \Delta t_n = \min\big(\text{max\_dt},\;
        \text{target} / \|C^{\theta\theta}_n\|_2\big)
    $$

    We bound the spectral norm via the Frobenius norm of the
    ``Nₑ × Nₑ`` Gram of the anomalies — ``‖A‖₂ ≤ ‖A‖_F`` for any
    matrix, so this is a *conservative* (slightly tighter than
    necessary) cap on ``Δt``. A true top-eigenvalue estimate would
    require an extra ``Nₑ × Nₑ`` eigendecomposition per step; the
    Frobenius bound costs only a single dot product and is cheap
    enough to evaluate every iteration.

    Attributes:
        max_dt: Hard ceiling on the step size.
        target: Desired ``Δt × ‖Cᶿᶿ‖_F`` product.
    """

    max_dt: float = eqx.field(static=True, default=1.0)
    target: float = eqx.field(static=True, default=1.0)

    def get_dt(self, state: ProcessState) -> Float[Array, ""]:
        # Cheap spectral-norm estimate via the (J, J) Gram of the
        # anomalies — the nonzero eigenvalues of (J−1)⁻¹ X′ᵀ X′ match
        # those of (J−1)⁻¹ X′ X′ᵀ.
        particles = state.particles
        J = particles.shape[0]
        mean = jnp.mean(particles, axis=0)
        anom = particles - mean[None, :]  # (J, N_p)
        gram = anom @ anom.T / (J - 1)  # (J, J)
        # The Frobenius bound on top eigenvalue: ‖A‖₂ ≤ ‖A‖_F.
        spectral_bound = jnp.sqrt(jnp.sum(gram * gram))
        dt_stable = self.target / jnp.maximum(spectral_bound, 1e-12)
        return jnp.minimum(jnp.asarray(self.max_dt), dt_stable)

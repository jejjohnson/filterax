"""Step-size schedulers for ensemble Kalman processes.

Every EKP iteration is parameterised by an artificial-time step
``Δtₙ ∈ ℝ₊``. The role of the scheduler is to pick ``Δtₙ`` from the
current :class:`ProcessState` — either a constant, a misfit-adaptive
rule (Iglesias 2016), or a stability-controlled rule for the EKS
Langevin dynamics.

All schedulers implement :class:`AbstractScheduler` so they slot into
the L1 processes (``EKI``, ``EKS_Process``, ``UKI``, …) and the L2 run
loops without further glue. Schedulers are stateless ``eqx.Module``
subclasses — they receive the state and return a scalar.

Convention: ``algo_time = Σₙ Δtₙ``. Schedulers that drive convergence
arrange for ``algo_time → 1`` (Iglesias 2016 §3); the L2 run loop can
break out when ``algo_time ≥ 1``.
"""

from __future__ import annotations

import equinox as eqx
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
    r"""Data-misfit-adaptive step size (Iglesias 2016 §3.2).

    Picks ``Δtₙ`` so that the **current** ensemble's misfit contributes
    a uniform fraction of the total artificial-time interval ``[0, 1]``.
    The recipe:

    $$
    \Delta t_n = \min\big(\text{target\_misfit} / \Phi_n,\;
        1 - \text{algo\_time}_n\big)
    $$

    with the per-step misfit norm

    $$
    \Phi_n = J^{-1} \sum_{j} \big\| y - G(\theta^{(j)}) \big\|^2_{\Gamma^{-1}}
    $$

    computed in observation space — ``state.noise_cov`` carries ``Γ``
    itself, and we apply ``Γ⁻¹`` internally via :func:`gaussx.solve_rows`
    so structured ``Γ`` (diagonal, low-rank, …) is never densified.
    The ``min`` clamps the final step so ``algo_time`` lands exactly at
    ``1.0``, which is the standard EKI termination criterion; subsequent
    calls return ``Δt = 0`` (the L1 processes guard their ``1/Δt``
    paths with a small floor so further calls are no-ops rather than
    NaNs).

    Attributes:
        target_misfit: Desired misfit-weighted increment. ``1.0`` is the
            default and matches Iglesias' "noise-level" stopping rule.
        eps: Small floor on the misfit denominator so the initial step
            (where the ensemble has not yet been corrected toward the
            data) doesn't divide by zero.
    """

    target_misfit: float = eqx.field(static=True, default=1.0)
    eps: float = eqx.field(static=True, default=1e-8)

    def get_dt(self, state: ProcessState) -> Float[Array, ""]:
        # Mahalanobis misfit Φ = (1/J) Σⱼ ‖y − Gⱼ‖²_{Γ⁻¹}.
        residuals = state.obs[None, :] - state.forward_evals  # (J, N_d)
        # Solve Γ z = r for each row — gaussx dispatches by structure of Γ.
        from gaussx import solve_rows

        Gamma_inv_r = solve_rows(state.noise_cov, residuals)  # (J, N_d)
        # Φ = (1/J) Σⱼ rⱼ · (Γ⁻¹ rⱼ).
        misfit = jnp.mean(jnp.sum(residuals * Gamma_inv_r, axis=-1))
        dt_target = self.target_misfit / jnp.maximum(misfit, self.eps)
        remaining = 1.0 - state.algo_time
        return jnp.minimum(dt_target, jnp.maximum(remaining, 0.0))


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

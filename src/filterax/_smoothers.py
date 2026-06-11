"""Ensemble backward-pass smoothers (Wave 5.A).

Smoothers refine a filter's history into the smoothing distribution
``p(x_t | y_{1:T})`` for ``T > t`` by walking backward over the stored
analysis and forecast ensembles. Every smoother here shares the same
ensemble-space gain construction; they differ only in *which slice* of
the history each backward correction is allowed to see.

Notation
--------
* ``Nₑ``           — ensemble size
* ``Nₓ``           — state dimension
* ``T``            — number of filter windows
* ``X^a_t``        — analysis ensemble at time ``t``, shape ``(Nₑ, Nₓ)``
* ``X^f_{t+1}``    — forecast ensemble at time ``t+1`` (propagation of ``X^a_t``)
* ``X^s_t``        — smoothed ensemble at time ``t``
* ``A_t, F_{t+1}`` — centred anomaly matrices of ``X^a_t`` / ``X^f_{t+1}``
* ``G_t``          — state-space smoother gain ``C^{af}_{t,t+1} (C^{ff}_{t+1})^{-1}``

The state-space gain is rank ``≤ Nₑ − 1`` (the forecast covariance is
estimated from ``Nₑ`` members), so the inverse is taken in the Moore-
Penrose sense. We never materialise the ``Nₓ × Nₓ`` covariance — the
backward step factors through ``F Fᵀ ∈ ℝ^{Nₑ × Nₑ}`` instead, giving
``O(Nₑ² Nₓ + Nₑ³)`` per step.
"""

from __future__ import annotations

from collections.abc import Callable

import einx
import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float, Int, PRNGKeyArray

from filterax._checks import check_ensemble_size
from filterax._primitives._perturbations import perturbed_observations
from filterax._primitives._statistics import ensemble_anomalies, ensemble_mean
from filterax._types import AnalysisResult, SmoothingResult


def _pinv_threshold(eigvals: Float[Array, " N_e"]) -> Float[Array, ""]:
    """Numpy-style relative cutoff for the rank-deficient PSD pseudoinverse.

    Matches numpy's default ``rcond = max(M.shape) * eps``: eigenvalues
    below ``size × eps × λ_max`` are treated as null modes. Picking up
    the dtype's epsilon means a float32 ensemble where many off-mean
    directions are roundoff-positive doesn't accidentally invert them
    (and the float64 threshold stays as tight as before).
    """
    eps = jnp.finfo(eigvals.dtype).eps
    return jnp.maximum(eigvals[-1], 0.0) * eigvals.shape[0] * eps


def _smoother_step(
    smoothed_next: Float[Array, "N_e N_x"],
    analysis_t: Float[Array, "N_e N_x"],
    forecast_t1: Float[Array, "N_e N_x"],
) -> Float[Array, "N_e N_x"]:
    r"""One backward smoother step.

    Applies the per-member update
    ``X^s_t[j] = X^a_t[j] + G_t (X^s_{t+1}[j] - X^f_{t+1}[j])`` using the
    ensemble-space factorisation

    ``G_t (X^s_{t+1} - X^f_{t+1}) = D Fᵀ (F Fᵀ)⁺ A``

    where ``A = X^a_t − x̄^a_t``, ``F = X^f_{t+1} − x̄^f_{t+1}``, and
    ``D = X^s_{t+1} − X^f_{t+1}`` are arranged with rows as members. The
    pseudoinverse is computed via :func:`jax.numpy.linalg.eigh` on the
    symmetrised ``F Fᵀ`` with a dtype-aware relative threshold to
    discard the rank-deficient null direction.
    """
    # Anomaly matrices share the analysis / forecast row layout.
    a_mean = einx.mean("e x -> x", analysis_t)
    A = einx.subtract("e x, x -> e x", analysis_t, a_mean)
    f_mean = einx.mean("e x -> x", forecast_t1)
    F = einx.subtract("e x, x -> e x", forecast_t1, f_mean)

    # Per-member discrepancy used as input to the smoother gain.
    D = smoothed_next - forecast_t1

    # F Fᵀ is rank ≤ Nₑ − 1; eigh + relative threshold handles the null direction.
    M = einx.dot("e x, f x -> e f", F, F)
    M = 0.5 * (M + jnp.swapaxes(M, -1, -2))
    eigvals, eigvecs = jnp.linalg.eigh(M)
    # eigh returns eigvals in ascending order; eigvals[-1] is the largest.
    threshold = _pinv_threshold(eigvals)
    inv_lambda = jnp.where(eigvals > threshold, 1.0 / eigvals, 0.0)

    # B = D Fᵀ in ensemble space.
    B = einx.dot("e x, f x -> e f", D, F)

    # G̃ = B (F Fᵀ)⁺ = B U diag(1/λ) Uᵀ. Three factored matmuls.
    BU = einx.dot("e a, a b -> e b", B, eigvecs)
    BU_scaled = einx.multiply("e a, a -> e a", BU, inv_lambda)
    G_tilde = einx.dot("e a, b a -> e b", BU_scaled, eigvecs)

    delta = einx.dot("e f, f x -> e x", G_tilde, A)
    return einx.add("e x, e x -> e x", analysis_t, delta)


def _enks_backward(
    analysis_history: Float[Array, "T N_e N_x"],
    forecast_history: Float[Array, "T N_e N_x"],
) -> Float[Array, "T N_e N_x"]:
    r"""Full backward pass over a stored filter history.

    Initialises ``X^s_{T-1} = X^a_{T-1}`` and runs :func:`_smoother_step`
    for ``t = T-2, …, 0``. Implemented as a reversed :func:`jax.lax.scan`
    so the whole pass stays JIT-friendly.
    """
    T = analysis_history.shape[0]
    init = analysis_history[-1]

    def step(
        carry: Float[Array, "N_e N_x"], idx: Int[Array, ""]
    ) -> tuple[Float[Array, "N_e N_x"], Float[Array, "N_e N_x"]]:
        smoothed_t = _smoother_step(
            carry,
            analysis_history[idx],
            forecast_history[idx + 1],
        )
        return smoothed_t, smoothed_t

    # Reverse scan over indices 0..T-2 — carry starts at T-1, ends at 0.
    indices = jnp.arange(T - 1)
    _, smoothed_partial = jax.lax.scan(step, init, indices, reverse=True)
    # Append the unchanged final-time analysis so output matches T.
    return jnp.concatenate([smoothed_partial, init[None, :, :]], axis=0)


def _validate_history(
    analysis_history: Float[Array, "T N_e N_x"],
    forecast_history: Float[Array, "T N_e N_x"],
) -> None:
    """Shape, time-length, and ensemble-size checks for smoother inputs."""
    if analysis_history.ndim != 3 or forecast_history.ndim != 3:
        raise ValueError(
            "analysis_history and forecast_history must be (T, N_e, N_x) "
            "arrays; got shapes "
            f"{analysis_history.shape} and {forecast_history.shape}."
        )
    if analysis_history.shape != forecast_history.shape:
        raise ValueError(
            "analysis_history and forecast_history must have the same "
            f"shape; got {analysis_history.shape} vs {forecast_history.shape}."
        )
    if analysis_history.shape[0] < 1:
        raise ValueError("smoother requires at least one filter window.")
    check_ensemble_size(analysis_history.shape[1])


class EnKS(eqx.Module, strict=True):
    r"""Ensemble Kalman Smoother — Evensen & van Leeuwen (2000).

    Standard single backward pass over a stored filter history. The
    smoother gain at step ``t`` is the sample cross-covariance times the
    pseudoinverse of the sample forecast covariance,

    ``G_t = (Nₑ − 1)⁻¹ A_tᵀ F_{t+1} · ((Nₑ − 1)⁻¹ F_{t+1}ᵀ F_{t+1})⁺``,

    factored through the ``Nₑ × Nₑ`` Gram matrix ``F Fᵀ`` so the dense
    ``Nₓ × Nₓ`` forecast covariance is never materialised.

    Compatible with any :class:`AbstractSequentialFilter` (StochasticEnKF,
    ETKF, EnSRF, LETKF, ETKF_Livings, EnSRF_Serial, ESTKF). Pass the
    ``forecast_history`` and ``analysis_history`` produced by an
    :class:`AssimilationResult`.

    Complexity: ``O(T (Nₑ² Nₓ + Nₑ³))`` for the whole backward pass.

    Example:
        >>> import jax
        >>> from filterax import EnKS
        >>> k1, k2 = jax.random.split(jax.random.key(0))
        >>> forecast = jax.random.normal(k1, (3, 4, 2))  # (T, N_e, N_x)
        >>> analysis = jax.random.normal(k2, (3, 4, 2))
        >>> result = EnKS().smooth(forecast, analysis)
        >>> result.smoothed_history.shape
        (3, 4, 2)
    """

    def smooth(
        self,
        forecast_history: Float[Array, "T N_e N_x"],
        analysis_history: Float[Array, "T N_e N_x"],
    ) -> SmoothingResult:
        """Run the full backward pass.

        Args:
            forecast_history: Stacked forecast ensembles
                ``X^f_0, …, X^f_{T-1}`` from the filter pass.
            analysis_history: Stacked analysis ensembles
                ``X^a_0, …, X^a_{T-1}`` from the filter pass.

        Returns:
            :class:`SmoothingResult` with ``smoothed_history[t] = X^s_t``
            and ``particles = X^s_{T-1}`` — the terminal ensemble that
            chains into a follow-up forecast, matching the
            ``AssimilationResult.particles`` convention.
        """
        _validate_history(analysis_history, forecast_history)
        smoothed = _enks_backward(analysis_history, forecast_history)
        return SmoothingResult(particles=smoothed[-1], smoothed_history=smoothed)


class EnsembleRTS(eqx.Module, strict=True):
    r"""Ensemble Rauch-Tung-Striebel smoother.

    Conceptual ensemble analog of the classical RTS recursion. With the
    cross-covariance form used here — ``G_t = C^{af}_{t,t+1}
    (C^{ff}_{t+1})⁺`` — the algorithm coincides with :class:`EnKS` for
    zero model error (Evensen 2003, §5; see also the smoothers design
    doc). The class exists so callers can name the RTS interpretation
    explicitly; the implementation is shared with :class:`EnKS`.

    Compatible with any :class:`AbstractSequentialFilter`. Complexity:
    ``O(T (Nₑ² Nₓ + Nₑ³))``.
    """

    def smooth(
        self,
        forecast_history: Float[Array, "T N_e N_x"],
        analysis_history: Float[Array, "T N_e N_x"],
    ) -> SmoothingResult:
        """Run the backward RTS pass — see :meth:`EnKS.smooth`."""
        _validate_history(analysis_history, forecast_history)
        smoothed = _enks_backward(analysis_history, forecast_history)
        return SmoothingResult(particles=smoothed[-1], smoothed_history=smoothed)


class FixedLagSmoother(eqx.Module, strict=True):
    r"""Fixed-lag ensemble smoother (Anderson & Anderson 1999; Nerger et al. 2012).

    At each position ``s``, runs the backward smoother only over the
    window ``[s, min(T-1, s + lag)]``. Equivalent to :class:`EnKS` when
    ``lag ≥ T-1``; equivalent to "no smoothing" when ``lag == 0``.

    The batch interface is the one exposed here — for an online /
    streaming setting (where states older than ``lag`` are finalised and
    discarded from memory) the user maintains the rolling buffer
    explicitly and calls :meth:`smooth` over the current window.

    Attributes:
        lag: Number of backward steps. ``lag == 0`` returns the filter
            analysis unchanged; large ``lag`` recovers the full EnKS.

    Complexity: ``O(T · lag · (Nₑ² Nₓ + Nₑ³))`` for the batch sweep.
    """

    lag: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        if not isinstance(self.lag, int) or self.lag < 0:
            raise ValueError(f"lag must be a non-negative int; got {self.lag!r}.")

    def smooth(
        self,
        forecast_history: Float[Array, "T N_e N_x"],
        analysis_history: Float[Array, "T N_e N_x"],
    ) -> SmoothingResult:
        """Run the windowed backward pass.

        Args:
            forecast_history: Stacked forecast ensembles.
            analysis_history: Stacked analysis ensembles.

        Returns:
            :class:`SmoothingResult` whose ``smoothed_history[s]`` was
            produced by an EnKS-style backward pass restricted to the
            window ``[s, min(T-1, s + lag)]``.
        """
        _validate_history(analysis_history, forecast_history)
        T = analysis_history.shape[0]
        # Effective lag: clamp at ``T - 1`` (max usable lookahead) so a
        # large user-supplied ``lag`` does no extra work and the
        # ``T == 1, lag > 0`` case can never index ``forecast_history[1]``.
        lag_eff = min(self.lag, max(T - 1, 0))

        if lag_eff == 0:
            return SmoothingResult(
                particles=analysis_history[-1],
                smoothed_history=analysis_history,
            )

        def smooth_one_anchor(s: Int[Array, ""]) -> Float[Array, "N_e N_x"]:
            """Smoothed estimate at position ``s`` using lookahead ≤ ``lag_eff``."""
            end = jnp.minimum(s + lag_eff, T - 1)
            init = analysis_history[end]

            def step(
                carry: Float[Array, "N_e N_x"], k: Int[Array, ""]
            ) -> tuple[Float[Array, "N_e N_x"], None]:
                # k counts backward steps from the window end (0 = first step).
                # The nominal index processed at this step is `end - 1 - k`.
                t = end - 1 - k
                valid = t >= s
                # Gather safely even for out-of-window steps; we mask the carry.
                t_safe = jnp.maximum(t, 0)
                analysis_t = analysis_history[t_safe]
                forecast_t1 = forecast_history[t_safe + 1]
                candidate = _smoother_step(carry, analysis_t, forecast_t1)
                next_carry = jnp.where(valid, candidate, carry)
                return next_carry, None

            final, _ = jax.lax.scan(step, init, jnp.arange(lag_eff))
            return final

        smoothed = jax.vmap(smooth_one_anchor)(jnp.arange(T))
        return SmoothingResult(particles=smoothed[-1], smoothed_history=smoothed)


def _sqrt_smoother_step(
    smoothed_next: Float[Array, "N_e N_x"],
    analysis_t: Float[Array, "N_e N_x"],
    forecast_t1: Float[Array, "N_e N_x"],
) -> Float[Array, "N_e N_x"]:
    r"""One deterministic square-root backward step.

    Decomposes the update into:

    * **Mean** — ``m^s_t = m^a_t + G_t (m^s_{t+1} - m^f_{t+1})`` via the
      dual ensemble form ``G_t v = A_tᵀ (F Fᵀ)⁺ F v``.
    * **Perturbations** — apply a symmetric square-root transform in
      ensemble space ``W = Λ^{1/2}`` with
      ``Λ = I + K_e (Dᵀ D − Fᵀ F) K_eᵀ`` and
      ``K_e = (F Fᵀ)⁺ F``. New anomalies = ``Wᵀ A_t``.

    Negative eigenvalues of ``Λ`` (sample-noise artefacts when the
    smoother reduces variance) are clamped to zero before the sqrt.
    """
    m_a = ensemble_mean(analysis_t)
    A = ensemble_anomalies(analysis_t)
    m_f = ensemble_mean(forecast_t1)
    F = ensemble_anomalies(forecast_t1)
    m_s_next = ensemble_mean(smoothed_next)
    D = ensemble_anomalies(smoothed_next)

    # F Fᵀ eigendecomposition shared by the mean and perturbation updates.
    M = einx.dot("e x, f x -> e f", F, F)
    M = 0.5 * (M + jnp.swapaxes(M, -1, -2))
    eigvals, eigvecs = jnp.linalg.eigh(M)
    threshold = _pinv_threshold(eigvals)
    inv_lambda = jnp.where(eigvals > threshold, 1.0 / eigvals, 0.0)

    # Mean update: G_t (m^s_{t+1} - m^f_{t+1}) = Aᵀ (F Fᵀ)⁺ F (m^s_{t+1} - m^f_{t+1}).
    delta_mean = m_s_next - m_f
    Fv = einx.dot("e x, x -> e", F, delta_mean)
    Ut_Fv = einx.dot("e a, e -> a", eigvecs, Fv)
    z = einx.dot("e a, a -> e", eigvecs, inv_lambda * Ut_Fv)
    mean_correction = einx.dot("e x, e -> x", A, z)
    m_s = einx.add("x, x -> x", m_a, mean_correction)

    # Perturbation update: ensemble-space transform Λ such that
    # AᵀΛA = (Nₑ−1) P^s_t. K_e = (F Fᵀ)⁺ F appears only through
    # B = K_e Dᵀ and (K_e Fᵀ K_e F) = projector onto col(F).
    B = einx.dot("a b, e b -> a e", eigvecs * inv_lambda[None, :], eigvecs)
    # B is now (F Fᵀ)⁺ in eigh form. Multiply by F → K_e (N_e, N_x):
    K_e = einx.dot("e a, a x -> e x", B, F)
    # B_D = K_e Dᵀ (Nₑ, Nₑ); projector_F = K_e Fᵀ = (F Fᵀ)⁺ (F Fᵀ).
    B_D = einx.dot("e x, f x -> e f", K_e, D)
    projector_F = einx.dot("e x, f x -> e f", K_e, F)
    # Λ = I + B_D B_Dᵀ − projector_F.  Symmetrise + clamp negative eigvals.
    I_ne = jnp.eye(F.shape[0], dtype=F.dtype)
    Lambda = I_ne + einx.dot("e a, f a -> e f", B_D, B_D) - projector_F
    Lambda = 0.5 * (Lambda + jnp.swapaxes(Lambda, -1, -2))
    lam_eigvals, lam_eigvecs = jnp.linalg.eigh(Lambda)
    lam_eigvals = jnp.maximum(lam_eigvals, 0.0)
    sqrt_lambda = jnp.sqrt(lam_eigvals)
    # W = Λ^{1/2} = U diag(√λ) Uᵀ. New anomalies = Wᵀ A = W A (symmetric).
    Ut_A = einx.dot("e a, e x -> a x", lam_eigvecs, A)
    scaled = einx.multiply("a x, a -> a x", Ut_A, sqrt_lambda)
    new_anom = einx.dot("e a, a x -> e x", lam_eigvecs, scaled)

    return einx.add("x, e x -> e x", m_s, new_anom)


def _sqrt_backward(
    analysis_history: Float[Array, "T N_e N_x"],
    forecast_history: Float[Array, "T N_e N_x"],
) -> Float[Array, "T N_e N_x"]:
    """Full backward pass using :func:`_sqrt_smoother_step`."""
    T = analysis_history.shape[0]
    init = analysis_history[-1]

    def step(
        carry: Float[Array, "N_e N_x"], idx: Int[Array, ""]
    ) -> tuple[Float[Array, "N_e N_x"], Float[Array, "N_e N_x"]]:
        smoothed_t = _sqrt_smoother_step(
            carry,
            analysis_history[idx],
            forecast_history[idx + 1],
        )
        return smoothed_t, smoothed_t

    indices = jnp.arange(T - 1)
    _, partial = jax.lax.scan(step, init, indices, reverse=True)
    return jnp.concatenate([partial, init[None, :, :]], axis=0)


class EnsembleSqrtSmoother(eqx.Module, strict=True):
    r"""Ensemble Square Root Smoother — Tippett et al. (2003); Whitaker & Compo (2002).

    Deterministic backward pass paired with a forward square-root filter
    (ETKF, EnSRF, ESTKF). Differs from :class:`EnKS` in *how* the
    smoothed perturbations are produced: instead of the raw per-member
    correction ``δ_j = G_t (X^s_{t+1}[j] - X^f_{t+1}[j])``, the
    smoothed mean and perturbations are updated separately, with the
    perturbation half applied through a *symmetric square root* of the
    ensemble-space cov-update transform

    ``Λ = I + K_e (Dᵀ D − Fᵀ F) K_eᵀ,    K_e = (F Fᵀ)⁺ F``.

    This avoids accumulating cross-member coupling across many backward
    steps — the same reason :class:`ETKF` uses a symmetric square-root
    transform on the forward pass.

    Mathematically agrees with :class:`EnKS` in the infinite-ensemble
    limit; with finite ensembles the *mean* is identical (both produce
    ``m^a_t + G_t (m^s_{t+1} - m^f_{t+1})``) and the *perturbations*
    differ only in their ensemble-space rotation.

    Complexity: ``O(T (Nₑ² Nₓ + Nₑ³))`` — same as :class:`EnKS`.
    """

    def smooth(
        self,
        forecast_history: Float[Array, "T N_e N_x"],
        analysis_history: Float[Array, "T N_e N_x"],
    ) -> SmoothingResult:
        """Run the square-root backward pass — see :meth:`EnKS.smooth`."""
        _validate_history(analysis_history, forecast_history)
        smoothed = _sqrt_backward(analysis_history, forecast_history)
        return SmoothingResult(particles=smoothed[-1], smoothed_history=smoothed)


# ──────────────────────────────────────────────────────────────────────
# IES — Iterative Ensemble Smoother (Chen & Oliver 2013; Evensen 2019)
# ──────────────────────────────────────────────────────────────────────


def _ies_step(
    particles_i: Float[Array, "J N_p"],
    particles_0: Float[Array, "J N_p"],
    obs: Float[Array, " N_d"],
    noise_cov: lx.AbstractLinearOperator,
    forward_model: Callable[[Float[Array, " N_p"]], Float[Array, " N_d"]],
    step_size: Float[Array, ""],
    key: PRNGKeyArray,
) -> Float[Array, "J N_p"]:
    r"""One Chen-Oliver IES iteration.

    ``θ_{i+1}^j = (1 − α) θ_i^j + α [θ_0^j + K_i (y + ε^j − G(θ_i^j))]``

    where ``K_i = C^{θG}_i (C^{GG}_i + Γ_y)⁻¹`` is the sample-cov gain
    at the current iterate. With ``α = 1`` this is the pure
    Chen-Oliver update — particles snap to ``θ_0 + K_i r_i`` each step;
    with ``α < 1`` the iteration is damped (Levenberg-Marquardt-style).

    The gain solve runs through gaussx's structural Woodbury so
    diagonal / low-rank ``Γ_y`` never densifies.
    """
    J = particles_i.shape[0]
    evals = jax.vmap(forward_model)(particles_i)  # (J, N_d)
    # Perturbed obs y + ε^j with ε^j ~ 𝒩(0, Γ_y) via the structure-aware draw.
    y_pert = perturbed_observations(key, obs, noise_cov, J)
    innovations = y_pert - evals  # (J, N_d)
    # K_i applied to innovations via the ensemble-cov Kalman recipe.
    # S = C^GG + Γ_y as a low-rank update for structural Woodbury dispatch.
    C_GG = gaussx.ensemble_covariance(evals, bessel=True)
    S = gaussx.LowRankUpdate(noise_cov, C_GG.U)
    S_inv_innov = gaussx.solve_rows(S, innovations)  # (J, N_d)
    C_theta_G = gaussx.ensemble_cross_covariance(
        particles_i, evals, bessel=True
    )  # (N_p, N_d)
    K_r = einx.dot("p d, j d -> j p", C_theta_G, S_inv_innov)  # (J, N_p)
    target = particles_0 + K_r
    return (1.0 - step_size) * particles_i + step_size * target


class IES(eqx.Module, strict=True):
    r"""Iterative Ensemble Smoother — Chen & Oliver (2013); Evensen et al. (2019).

    Standalone iterative ensemble method that solves an inverse problem
    in a single window of observations. At each iteration:

    ``θ_{i+1}^j = (1 − α) θ_i^j + α [θ_0^j + K_i (y + ε^j − G(θ_i^j))]``

    The anchor to ``θ_0^j`` (the *initial* ensemble member) is what
    distinguishes IES from :class:`~filterax.EKI`'s drift-style
    iteration: rather than walking ``θ`` toward a MAP with a step-size
    schedule, IES re-targets ``θ_0`` plus the current Kalman correction
    every iteration. ``α = 1`` is the pure Chen-Oliver update;
    ``α < 1`` damps the iteration for strongly nonlinear ``G``.

    Use case: history matching, reservoir simulation, and any inverse
    problem where the forward model maps parameters to a *full*
    observation time series and a single backward pass on a sequential
    filter would be insufficient. For sequential filtering followed by a
    backward refinement, use :class:`EnKS` instead.

    Attributes:
        n_iterations: Static iteration count.
        step_size: Damping factor ``α ∈ (0, 1]``. Default 1.0
            (pure Chen-Oliver).
        seed: Integer for the default per-iteration PRNG key.
        base_key: Optional explicit PRNG key (takes precedence over
            ``seed``).
    """

    n_iterations: int = eqx.field(static=True)
    step_size: float = eqx.field(static=True, default=1.0)
    seed: int = eqx.field(static=True, default=0)
    base_key: PRNGKeyArray | None = None

    def __check_init__(self) -> None:
        if not isinstance(self.n_iterations, int) or self.n_iterations < 1:
            raise ValueError(
                f"n_iterations must be a positive int; got {self.n_iterations!r}."
            )
        if not (0.0 < float(self.step_size) <= 1.0):
            raise ValueError(f"step_size must lie in (0, 1]; got {self.step_size!r}.")

    def solve(
        self,
        particles: Float[Array, "J N_p"],
        obs: Float[Array, " N_d"],
        noise_cov: lx.AbstractLinearOperator,
        forward_model: Callable[[Float[Array, " N_p"]], Float[Array, " N_d"]],
    ) -> AnalysisResult:
        """Run the iterative smoother to its fixed iteration count.

        Args:
            particles: Initial ensemble ``(J, Nₚ)``. Doubles as the
                anchor ``θ_0`` against which every subsequent iterate is
                compared.
            obs: Observation vector ``y ∈ ℝ^{N_d}``.
            noise_cov: Observation noise covariance ``Γ_y``.
            forward_model: ``G(θ): ℝ^{Nₚ} → ℝ^{N_d}`` applied to a single
                particle; vectorised internally with :func:`jax.vmap`.

        Returns:
            :class:`AnalysisResult` with the final iterate's particles.
        """
        check_ensemble_size(particles.shape[0])
        base_key = self.base_key if self.base_key is not None else jr.PRNGKey(self.seed)
        step_size = jnp.asarray(self.step_size, dtype=particles.dtype)

        def step(
            carry: Float[Array, "J N_p"], i: Int[Array, ""]
        ) -> tuple[Float[Array, "J N_p"], None]:
            # Fresh per-iteration sub-key so the perturbation draws are
            # independent across iterations.
            sub = jr.fold_in(base_key, i)
            new_carry = _ies_step(
                carry,
                particles,
                obs,
                noise_cov,
                forward_model,
                step_size,
                sub,
            )
            return new_carry, None

        final, _ = jax.lax.scan(step, particles, jnp.arange(self.n_iterations))
        return AnalysisResult(particles=final)

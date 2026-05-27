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

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from filterax._src._checks import check_ensemble_size
from filterax._src._types import SmoothingResult


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

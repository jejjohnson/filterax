"""Parametric (non-ensemble) Kalman filters.

These are the textbook Kalman filters in which a single multivariate
Gaussian belief ``𝒩(μₙ, Pₙ)`` is propagated through a linear-Gaussian
state-space model. They are included alongside the ensemble filters so
linear-Gaussian baselines, twin-experiment ground truths, and
square-root reference implementations are available without leaving
filterax.

The :class:`SquareRootKF` here is a thin wrapper over
:func:`gaussx.parallel_kalman_filter` with ``form='sqrt'``: it
propagates the Cholesky factor ``S`` such that ``P = S Sᵀ``, never
materialising ``P`` directly so the analysis covariance stays PSD
under floating-point error (Maybeck 1979 §7.4).

The :class:`KalmanSmoother` here is the exact counterpart of the
ensemble smoothers: the Kalman filter and Rauch–Tung–Striebel smoother
over irregular observation times, through :func:`gaussx.kalman_filter`
and :func:`gaussx.rts_smoother`.
"""

from __future__ import annotations

import math

import einx
import equinox as eqx
import gaussx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float


class SquareRootKF(eqx.Module, strict=True):
    r"""Square-root parametric Kalman filter (Maybeck 1979 / gaussx).

    Propagates a Gaussian belief through a linear-Gaussian model:

    $$
    \begin{aligned}
    x_n &= \Phi x_{n-1} + w_n, & w_n &\sim \mathcal{N}(0, Q) \\
    y_n &= H x_n + v_n, & v_n &\sim \mathcal{N}(0, R)
    \end{aligned}
    $$

    The forecast and analysis steps are both rewritten in *square-root*
    form on the Cholesky factor ``S`` of ``P`` so the covariance stays
    PSD by construction. Internally this delegates to
    :func:`gaussx.parallel_kalman_filter` (``form='sqrt'``), which runs
    Blelloch's parallel scan along the time axis — cost
    ``O(log T · (Nₓ² Nᵧ + Nₓ³))`` on hardware that can exploit the
    parallelism.

    The constructor stores the (time-invariant) ``Φ``, ``H``, ``Q``,
    ``R`` and lets the user push observations through :meth:`filter`.
    For time-varying systems pass leading-time-axis arrays directly to
    :func:`gaussx.parallel_kalman_filter` and skip this wrapper.

    Attributes:
        transition: State transition operator ``Φ`` (matrix or
            :class:`lineax.AbstractLinearOperator`).
        obs_model: Observation operator ``H``.
        process_noise: Process-noise covariance ``Q``.
        obs_noise: Observation-noise covariance ``R``.
    """

    transition: Float[Array, "N N"] | lx.AbstractLinearOperator
    obs_model: Float[Array, "M N"] | lx.AbstractLinearOperator
    process_noise: Float[Array, "N N"] | lx.AbstractLinearOperator
    obs_noise: Float[Array, "M M"] | lx.AbstractLinearOperator

    def filter(
        self,
        observations: Float[Array, "T M"],
        init_mean: Float[Array, " N"],
        init_cov: Float[Array, "N N"],
    ) -> "SquareRootFilterResult":
        r"""Run the square-root Kalman filter over ``observations``.

        Args:
            observations: Observation sequence ``(T, M)``.
            init_mean: Initial belief mean ``μ₀ ∈ ℝᴺ``.
            init_cov: Initial belief covariance ``P₀ ∈ ℝ^{N×N}``.

        Returns:
            :class:`SquareRootFilterResult` containing the per-step
            filtered means, filtered covariances, and the marginal
            log-likelihood ``log p(y₁…y_T)``.
        """
        state = gaussx.parallel_kalman_filter(
            self.transition,
            self.obs_model,
            self.process_noise,
            self.obs_noise,
            observations,
            init_mean,
            init_cov,
            form="sqrt",
        )
        return SquareRootFilterResult(
            filtered_means=state.filtered_means,
            filtered_covs=state.filtered_covs,
            predicted_means=state.predicted_means,
            predicted_covs=state.predicted_covs,
            log_likelihood=state.log_likelihood,
        )


class SquareRootFilterResult(eqx.Module, strict=True):
    """Output of :meth:`SquareRootKF.filter`.

    Mirrors :class:`gaussx.FilterState` so callers don't depend on the
    internal gaussx structure: per-step filtered / predicted means and
    covariances along a leading time axis, plus the marginal
    log-likelihood of the full observation sequence.

    Attributes:
        filtered_means: ``μ_{n|n}`` for ``n = 1, …, T``, shape ``(T, N)``.
        filtered_covs: ``P_{n|n}`` for ``n = 1, …, T``, shape ``(T, N, N)``.
        predicted_means: ``μ_{n|n−1}`` (forecast means before update),
            shape ``(T, N)``.
        predicted_covs: ``P_{n|n−1}`` (forecast covariances), shape
            ``(T, N, N)``.
        log_likelihood: Scalar marginal log-likelihood.
    """

    filtered_means: Float[Array, "T N"]
    filtered_covs: Float[Array, "T N N"]
    predicted_means: Float[Array, "T N"]
    predicted_covs: Float[Array, "T N N"]
    log_likelihood: Float[Array, ""]


class KalmanSmoothingResult(eqx.Module, strict=True):
    """Output of :meth:`KalmanSmoother.smooth`.

    The Gaussian marginals of the state at every observation time, both
    filtered (conditioned on the past) and smoothed (conditioned on the
    whole record), along a leading time axis. Cross-time covariances are
    not kept.

    Attributes:
        filtered_means: ``μ_x,k|k`` for ``k = 1, …, T``, shape ``(T, N)``.
        filtered_covs: ``Σ_x,k|k``, shape ``(T, N, N)``.
        smoothed_means: ``μ_x,k|T``, shape ``(T, N)``.
        smoothed_covs: ``Σ_x,k|T``, shape ``(T, N, N)``.
        log_likelihood: Marginal log-likelihood ``log p(y_1:T)`` of the
            observed entries; masked entries do not contribute.
    """

    filtered_means: Float[Array, "T N"]
    filtered_covs: Float[Array, "T N N"]
    smoothed_means: Float[Array, "T N"]
    smoothed_covs: Float[Array, "T N N"]
    log_likelihood: Float[Array, ""]


class KalmanSmoother(eqx.Module, strict=True):
    r"""Exact Kalman filter and Rauch–Tung–Striebel smoother over irregular times.

    The linear-Gaussian counterpart of :class:`filterax.EnKS` and
    :class:`filterax.EnsembleRTS`: a state observed at times
    ``t_1 < … < t_T`` evolves as a Gauss–Markov process between them,

    $$
    x_k = F_k x_{k-1} + \eta_k, \quad \eta_k \sim \mathcal{N}(0, \Sigma_{\eta,k}),
    \qquad
    y_k = H_k x_k + \varepsilon_k, \quad
    \varepsilon_k \sim \mathcal{N}(0, \Sigma_{\varepsilon,k}),
    $$

    with ``x_1 ~ 𝒩(μ_x, Σ_x)`` at the first observation time, and
    :meth:`smooth` returns the exact marginals ``p(x_k | y_1:k)`` and
    ``p(x_k | y_1:T)``. ``F_k`` and ``Σ_η,k`` act over the actual gap
    ``t_k − t_{k−1}``, so irregular times need no regridding;
    :meth:`from_sde` builds them from a gaussx SDE kernel.

    Unlike :class:`SquareRootKF`, whose prior sits one transition before
    the first observation, the prior here is the state at ``t_1``, so
    there are ``T − 1`` transitions for ``T`` observation times.

    Algorithm:
        1. Prepend ``F_1 = I`` and ``Σ_η,1 = 0``, so that gaussx's
           predict-then-update filter observes ``x_1`` under the prior
           ``𝒩(μ_x, Σ_x)`` itself.
        2. Make masked observation entries inert (zero row of ``H_k``,
           unit variance in ``Σ_ε,k``, zero datum), which equals
           deleting them; a time with every entry masked only predicts.
        3. Filter forward with :func:`gaussx.kalman_filter`.
        4. Smooth backward with :func:`gaussx.rts_smoother`,
           ``μ_x,k|T = μ_x,k|k + G_k (μ_x,k+1|T − μ_x,k+1|k)`` with the
           smoother gain ``G_k = Σ_x,k|k F_k+1ᵀ Σ_x,k+1|k⁻¹``.

    Attributes:
        transition: ``F_k`` from time ``k − 1`` to ``k``, shape
            ``(T − 1, N, N)``, or ``(N, N)`` for every gap.
        process_noise: ``Σ_η,k``, the same shapes as ``transition``.
        obs_model: ``H_k``, shape ``(T, M, N)``, or ``(M, N)`` for every
            time.
        obs_noise: ``Σ_ε,k``, shape ``(T, M, M)`` or ``(M, M)``.
        init_mean: Prior mean ``μ_x`` of ``x_1``, shape ``(N,)``.
        init_cov: Prior covariance ``Σ_x`` of ``x_1``, shape ``(N, N)``.

    Complexity: ``O(T (N³ + M³))``, sequential in ``T``.

    Examples:
        A random walk observed at three times, the middle one missing;
        the smoother fills the gap from both neighbours:

        >>> import jax.numpy as jnp
        >>> from filterax import KalmanSmoother
        >>> smoother = KalmanSmoother(
        ...     transition=jnp.eye(1),
        ...     process_noise=jnp.eye(1),
        ...     obs_model=jnp.eye(1),
        ...     obs_noise=0.01 * jnp.eye(1),
        ...     init_mean=jnp.zeros(1),
        ...     init_cov=jnp.eye(1),
        ... )
        >>> obs = jnp.array([[1.0], [jnp.nan], [3.0]])
        >>> mask = jnp.array([[True], [False], [True]])
        >>> result = smoother.smooth(obs, mask)
        >>> result.smoothed_means.shape
        (3, 1)
        >>> bool(jnp.isclose(result.smoothed_means[1, 0], 2.0, atol=0.05))
        True

    References:
        Rauch, Tung & Striebel (1965), AIAA Journal 3(8), 1445–1450.
        Särkkä (2013), Bayesian Filtering and Smoothing, ch. 8.
    """

    transition: Float[Array, "*G N N"]
    process_noise: Float[Array, "*G N N"]
    obs_model: Float[Array, "*T M N"]
    obs_noise: Float[Array, "*T M M"]
    init_mean: Float[Array, " N"]
    init_cov: Float[Array, "N N"]

    def __check_init__(self) -> None:
        for name in ("transition", "process_noise", "obs_model", "obs_noise"):
            ndim = getattr(self, name).ndim
            if ndim not in (2, 3):
                raise ValueError(
                    f"{name} must be one matrix or a (T, …) stack of them; "
                    f"got {ndim} dimensions."
                )

    @classmethod
    def from_sde(
        cls,
        sde: gaussx.SDEKernel,
        times: Float[Array, " T"],
        obs_model: Float[Array, "*T M N"],
        obs_noise: Float[Array, "*T M M"],
        init_mean: Float[Array, " N"] | None = None,
    ) -> KalmanSmoother:
        r"""Discretise a gaussx SDE kernel over the gaps of ``times``.

        ``F_k = exp(A Δt_k)`` and ``Σ_η,k = P_∞ − F_k P_∞ F_kᵀ`` on
        ``Δt_k = t_k − t_{k−1}``, where ``A`` is the SDE's drift and
        ``P_∞`` its stationary covariance, which is also the prior
        covariance of ``x_1``.

        Args:
            sde: A gaussx SDE kernel, e.g. ``gaussx.MaternSDE``.
            times: Observation times ``t_1 < … < t_T``, shape ``(T,)``.
            obs_model: ``H_k`` on the SDE state, shape ``(T, M, N)`` or
                ``(M, N)``.
            obs_noise: ``Σ_ε,k``, shape ``(T, M, M)`` or ``(M, M)``.
            init_mean: Prior mean ``μ_x`` of ``x_1``. Default ``None``:
                zero.

        Returns:
            The smoother for those times.
        """
        # Δt_k = t_k − t_{k−1}
        gaps = jnp.diff(times)
        # F_k = exp(A Δt_k), Σ_η,k = P_∞ − F_k P_∞ F_kᵀ
        transition, process_noise = sde.discretise_sequence(gaps)
        # Σ_x = P_∞
        init_cov = sde.sde_params().P_inf
        if init_mean is None:
            init_mean = jnp.zeros(init_cov.shape[0], init_cov.dtype)
        return cls(
            transition=transition,
            process_noise=process_noise,
            obs_model=obs_model,
            obs_noise=obs_noise,
            init_mean=init_mean,
            init_cov=init_cov,
        )

    def smooth(
        self,
        observations: Float[Array, "T M"],
        mask: Bool[Array, "T M"] | None = None,
    ) -> KalmanSmoothingResult:
        """Run the filter forward and the RTS smoother backward.

        Args:
            observations: ``y_1, …, y_T``, shape ``(T, M)``. Masked
                entries are never read, so they may be ``NaN``.
            mask: ``True`` where an entry was observed, shape ``(T, M)``.
                Default ``None``: every entry observed.

        Returns:
            :class:`KalmanSmoothingResult` with the filtered and
            smoothed marginals at every time and the log-likelihood.

        Raises:
            ValueError: If a stacked field's time axis does not match
                ``T`` (``T − 1`` for ``transition`` and
                ``process_noise``).
        """
        num_steps = observations.shape[0]
        if mask is None:
            mask = jnp.ones(observations.shape, dtype=bool)

        # --- sequences over T steps ---
        transition = _over(self.transition, num_steps - 1, "transition")
        process_noise = _over(self.process_noise, num_steps - 1, "process_noise")
        identity = jnp.eye(self.init_mean.shape[0], dtype=transition.dtype)
        # F_1 = I, Σ_η,1 = 0: x_1 is observed under the prior itself
        first_transition = einx.id("row col -> 1 row col", identity)
        transition = jnp.concatenate([first_transition, transition])
        first_noise = jnp.zeros_like(first_transition)
        process_noise = jnp.concatenate([first_noise, process_noise])
        obs_model = _over(self.obs_model, num_steps, "obs_model")
        obs_noise = _over(self.obs_noise, num_steps, "obs_noise")

        # --- missing data ---
        obs_model, obs_noise, observations = _inert_masked_entries(
            obs_model, obs_noise, observations, mask
        )
        # a time with no observed entry only predicts
        step_observed = einx.any("T M -> T", mask)

        # --- filter and smoother ---
        state = gaussx.kalman_filter(
            transition,
            obs_model,
            process_noise,
            obs_noise,
            observations,
            self.init_mean,
            self.init_cov,
            mask=step_observed,
            solver=gaussx.DenseSolver(),
        )
        smoothed_means, smoothed_covs = gaussx.rts_smoother(
            state, transition, process_noise, solver=gaussx.DenseSolver()
        )

        # --- log-likelihood ---
        # each inert entry of an updated step adds −½ log 2π to gaussx's sum
        missing_per_step = einx.sum("T M -> T", ~mask)
        missing_updated = jnp.where(step_observed, missing_per_step, 0)
        num_missing = einx.sum("T ->", missing_updated)
        log_likelihood = state.log_likelihood + 0.5 * math.log(2 * math.pi) * (
            num_missing
        )
        return KalmanSmoothingResult(
            filtered_means=state.filtered_means,
            filtered_covs=state.filtered_covs,
            smoothed_means=smoothed_means,
            smoothed_covs=smoothed_covs,
            log_likelihood=log_likelihood,
        )


def _over(
    stack: Float[Array, "*T R C"], length: int, name: str
) -> Float[Array, "T R C"]:
    """``stack`` with a leading time axis of ``length``, broadcasting one matrix."""
    if stack.ndim == 2:
        return jnp.broadcast_to(stack, (length, *stack.shape))
    if stack.shape[0] != length:
        raise ValueError(
            f"{name} must stack {length} matrices along the time axis; "
            f"got {stack.shape[0]}."
        )
    return stack


def _inert_masked_entries(
    obs_model: Float[Array, "T M N"],
    obs_noise: Float[Array, "T M M"],
    observations: Float[Array, "T M"],
    mask: Bool[Array, "T M"],
) -> tuple[Float[Array, "T M N"], Float[Array, "T M M"], Float[Array, "T M"]]:
    """Substitute ``(H_k, Σ_ε,k, y_k)`` so masked entries carry no information.

    Zeroing row ``i`` of ``H_k``, replacing row and column ``i`` of
    ``Σ_ε,k`` with the unit basis vector and zeroing ``y_k,i`` splits the
    innovation covariance as ``S_k = blockdiag(S_k,obs, I)``. Column ``i``
    of the gain ``P Hᵀ S⁻¹`` then vanishes, so the update is exactly the
    one with entry ``i`` deleted.
    """
    # H_k,i = 0 for a masked entry i
    obs_model = einx.where("T M, T M N, -> T M N", mask, obs_model, 0.0)
    # Σ_ε,k keeps an entry only where both its row and column are observed
    both_observed = einx.logical_and("T row, T col -> T row col", mask, mask)
    identity = jnp.eye(mask.shape[1], dtype=obs_noise.dtype)
    obs_noise = einx.where(
        "T row col, T row col, row col -> T row col",
        both_observed,
        obs_noise,
        identity,
    )
    # where, not a product, so a NaN at a masked entry never enters
    observations = jnp.where(mask, observations, 0.0)
    return obs_model, obs_noise, observations


# Silence unused-import warning — jnp is imported for the eqx.field
# defaults if we ever add scalar dtype fields. Keep the import so the
# module is self-contained for future extension.
_ = jnp

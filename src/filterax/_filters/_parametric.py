"""Parametric Kalman filters, which carry one Gaussian 𝒩(μ_x, Σ_x), not an ensemble.

They sit beside the ensemble filters as linear-Gaussian baselines, as the ground truth
of twin experiments and as square-root references, without leaving filterax.

`SquareRootKF` wraps `gaussx.parallel_kalman_filter` with ``square_root=True``, which
carries a factor L_x with Σ_x = L_x L_xᵀ through the scan, so every covariance stays
positive semi-definite under rounding (Yaghoobi et al. 2022). `KalmanSmoother` is the
exact counterpart of the ensemble smoothers: the Kalman filter and the
Rauch-Tung-Striebel smoother over irregular observation times, through
`gaussx.kalman_filter` and `gaussx.rts_smoother`.
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
    """Square-root Kalman filter for a time-invariant linear-Gaussian model.

        x_t = F x_t−1 + η_t,   η_t ~ 𝒩(0, Σ_η)
        y_t = H x_t + ε_t,     ε_t ~ 𝒩(0, Σ_ε)

    with x₀ ~ 𝒩(μ_x,0, Σ_x,0) and t = 1, …, T. Both the prediction and the update are
    written on factors L_x with Σ_x = L_x L_xᵀ, so every covariance is positive
    semi-definite by construction. The filter is `gaussx.parallel_kalman_filter` with
    ``square_root=True``, an associative scan over time at O(log T (D_x² D_y + D_x³))
    depth on parallel hardware; the class adds the stored model and a result type of
    its own. The factors stay inside gaussx, and the result reports L_x L_xᵀ.

    It matches the covariance form to rounding in float64 on a well-conditioned model.
    In float32 it stays closer to a float64 reference than the covariance form under
    strong conditioning, and stays positive definite where the covariance form does
    not. For a time-varying model, pass stacked arrays to
    `gaussx.parallel_kalman_filter` directly.

    Attributes:
        transition: Linear dynamics F, as a matrix or a lineax operator.
        obs_model: Observation operator H.
        process_noise: Model-error covariance Σ_η.
        obs_noise: Observation-error covariance Σ_ε.

    Examples:
        A random walk observed four times:

        >>> import jax.numpy as jnp
        >>> from filterax.filters import SquareRootKF
        >>> filter_ = SquareRootKF(
        ...     transition=jnp.eye(1),
        ...     obs_model=jnp.eye(1),
        ...     process_noise=0.1 * jnp.eye(1),
        ...     obs_noise=0.5 * jnp.eye(1),
        ... )
        >>> observations = jnp.array([[1.0], [1.2], [0.9], [1.1]])
        >>> result = filter_.filter(observations, jnp.zeros(1), jnp.eye(1))
        >>> result.filtered_means.shape, result.filtered_covs.shape
        ((4, 1), (4, 1, 1))

        The filtered variance never exceeds the predicted one:

        >>> filtered = result.filtered_covs[:, 0, 0]
        >>> bool(jnp.all(filtered <= result.predicted_covs[:, 0, 0]))
        True

    References:
        Yaghoobi, Corenflos, Hassan & Särkkä (2022), *Parallel square-root
        statistical linear regression for inference in nonlinear state space
        models*, arXiv:2207.00426.
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
        """Run the square-root Kalman filter over the observations.

        Args:
            observations: Observations y_1:T, one row per step.
            init_mean: Prior mean μ_x,0 of x₀, one transition before y_1.
            init_cov: Prior covariance Σ_x,0 of x₀.

        Returns:
            The filtered and predicted moments at every step and the log-likelihood
            log p(y_1:T).
        """
        # μ_x,t|t, Σ_x,t|t = L_x,t|t L_x,t|tᵀ, through the factor scan
        state = gaussx.parallel_kalman_filter(
            self.transition,
            self.obs_model,
            self.process_noise,
            self.obs_noise,
            observations,
            init_mean,
            init_cov,
            square_root=True,
        )
        return SquareRootFilterResult(
            filtered_means=state.filtered_means,
            filtered_covs=state.filtered_covs,
            predicted_means=state.predicted_means,
            predicted_covs=state.predicted_covs,
            log_likelihood=state.log_likelihood,
        )


class SquareRootFilterResult(eqx.Module, strict=True):
    """Output of `SquareRootKF.filter`, the moments of x_t at every step.

    It mirrors `gaussx.FilterState`, so that callers do not depend on gaussx's
    internals: the filtered and predicted moments along a leading time axis, and the
    log-likelihood of the whole sequence.

    Attributes:
        filtered_means: Filtered means μ_x,t|t for t = 1, …, T.
        filtered_covs: Filtered covariances Σ_x,t|t.
        predicted_means: Predicted means μ_x,t|t−1, before each update.
        predicted_covs: Predicted covariances Σ_x,t|t−1.
        log_likelihood: Log-likelihood log p(y_1:T).
    """

    filtered_means: Float[Array, "T N"]
    filtered_covs: Float[Array, "T N N"]
    predicted_means: Float[Array, "T N"]
    predicted_covs: Float[Array, "T N N"]
    log_likelihood: Float[Array, ""]


class KalmanSmoothingResult(eqx.Module, strict=True):
    """Output of `KalmanSmoother.smooth`, the marginals of x_k at every time.

    The Gaussian marginals of the state at every observation time, filtered on the
    past and smoothed on the whole record, along a leading time axis. Covariances
    across times are not kept.

    Attributes:
        filtered_means: Filtered means μ_x,k|k for k = 1, …, T.
        filtered_covs: Filtered covariances Σ_x,k|k.
        smoothed_means: Smoothed means μ_x,k|T.
        smoothed_covs: Smoothed covariances Σ_x,k|T.
        log_likelihood: Log-likelihood log p(y_1:T) of the observed entries; masked
            entries do not contribute.
    """

    filtered_means: Float[Array, "T N"]
    filtered_covs: Float[Array, "T N N"]
    smoothed_means: Float[Array, "T N"]
    smoothed_covs: Float[Array, "T N N"]
    log_likelihood: Float[Array, ""]


class KalmanSmoother(eqx.Module, strict=True):
    """Exact Kalman filter and Rauch-Tung-Striebel smoother over irregular times.

        x_k = F_k x_k−1 + η_k,   η_k ~ 𝒩(0, Σ_η,k)
        y_k = H_k x_k + ε_k,     ε_k ~ 𝒩(0, Σ_ε,k)

    The linear-Gaussian counterpart of `filterax.EnKS` and `filterax.EnsembleRTS`: a
    state observed at the times t_1 < … < t_T evolves as a Gauss-Markov process between
    them, with x_1 ~ 𝒩(μ_x, Σ_x) at the first observation time, and `smooth` returns
    the exact marginals p(x_k ∣ y_1:k) and p(x_k ∣ y_1:T). F_k and Σ_η,k span the
    actual gap t_k − t_k−1, so irregular times need no regridding, and `from_sde`
    builds them from a gaussx SDE kernel. The filter and smoother are
    `gaussx.kalman_filter` and `gaussx.rts_smoother`; the class adds the prior at the
    first observation, the exact treatment of masked entries and their share of the
    log-likelihood.

    Unlike `SquareRootKF`, whose prior sits one transition before the first
    observation, the prior here is the state at t_1, so there are T − 1 transitions for
    T observation times. The cost is O(T (D_x³ + D_y³)), sequential in T.

    Algorithm:
        1. Prepend F_1 = I and Σ_η,1 = 0, so that gaussx's predict-then-update filter
           observes x_1 under the prior 𝒩(μ_x, Σ_x) itself.
        2. Make masked entries inert, with a zero row of H_k, unit variance in Σ_ε,k
           and a zero datum, which equals deleting them; a time with every entry
           masked only predicts.
        3. Filter forward with `gaussx.kalman_filter`.
        4. Smooth backward with `gaussx.rts_smoother`,
           μ_x,k|T = μ_x,k|k + G_smooth,k (μ_x,k+1|T − μ_x,k+1|k), with the smoother
           gain G_smooth,k = Σ_x,k|k F_k+1ᵀ Σ_x,k+1|k⁻¹.

    Attributes:
        transition: Linear dynamics F_k from t_k−1 to t_k, stacked over the T − 1
            gaps, or one matrix for every gap.
        process_noise: Model-error covariance Σ_η,k, in the shapes of
            ``transition``.
        obs_model: Observation operator H_k, stacked over the T times, or one matrix
            for every time.
        obs_noise: Observation-error covariance Σ_ε,k, in the shapes of
            ``obs_model``.
        init_mean: Prior mean μ_x of x_1.
        init_cov: Prior covariance Σ_x of x_1.

    Raises:
        ValueError: If a model field is neither one matrix nor a stack of them.

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

        A vector where a matrix belongs is rejected:

        >>> KalmanSmoother(
        ...     transition=jnp.ones(1),
        ...     process_noise=jnp.eye(1),
        ...     obs_model=jnp.eye(1),
        ...     obs_noise=jnp.eye(1),
        ...     init_mean=jnp.zeros(1),
        ...     init_cov=jnp.eye(1),
        ... )  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        ValueError: transition must be one matrix or a (T, …) stack of them;
        got 1 dimensions.

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
        """Discretise a gaussx SDE kernel over the gaps between the times.

            F_k = exp(F_c Δt_k),   Σ_η,k = Σ_x,∞ − F_k Σ_x,∞ F_kᵀ,   Δt_k = t_k − t_k−1

        where F_c is the drift of the SDE and Σ_x,∞ its stationary covariance, which is
        also the prior covariance Σ_x of x_1.

        Args:
            sde: A gaussx SDE kernel, such as `gaussx.MaternSDE`.
            times: Observation times t_1 < … < t_T.
            obs_model: Observation operator H_k on the SDE state, stacked over the
                times or one matrix for all.
            obs_noise: Observation-error covariance Σ_ε,k, stacked or one matrix.
            init_mean: Prior mean μ_x of x_1; ``None`` is zero.

        Returns:
            The smoother for those times.

        Raises:
            ValueError: If the SDE has no stationary covariance.

        Examples:
            An Ornstein-Uhlenbeck process at three uneven times:

            >>> import gaussx
            >>> import jax.numpy as jnp
            >>> from filterax import KalmanSmoother
            >>> sde = gaussx.MaternSDE(
            ...     variance=jnp.asarray(0.64), lengthscale=jnp.asarray(1.5), order=0
            ... )
            >>> smoother = KalmanSmoother.from_sde(
            ...     sde, jnp.array([0.0, 0.4, 1.5]), jnp.ones((1, 1)), 0.1 * jnp.eye(1)
            ... )
            >>> smoother.transition.shape, smoother.init_cov.shape
            ((2, 1, 1), (1, 1))
            >>> bool(jnp.allclose(smoother.init_cov, 0.64))
            True
        """
        # Δt_k = t_k − t_{k−1}
        gaps = jnp.diff(times)
        # F_k = exp(F_c Δt_k), Σ_η,k = Σ_x,∞ − F_k Σ_x,∞ F_kᵀ
        transition, process_noise = sde.discretise_sequence(gaps)
        # Σ_x = Σ_x,∞
        init_cov = sde.sde_params().P_inf
        if init_cov is None:
            raise ValueError(
                f"{type(sde).__name__}.sde_params() reports P_inf=None (no "
                "stationary covariance), so from_sde cannot set the prior."
            )
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
        """Run the Kalman filter forward and the RTS smoother backward.

        Algorithm:
            1. Broadcast the model to T steps and prepend F_1 = I and Σ_η,1 = 0.
            2. Make the masked entries inert.
            3. Filter forward, then smooth backward.
            4. Remove the −½ log 2π that each inert entry adds to the log-likelihood.

        Args:
            observations: Observations y_1, …, y_T, one row per time. Masked entries
                are never read, so they may be NaN.
            mask: ``True`` where an entry was observed; ``None`` observes every
                entry.

        Returns:
            The filtered and smoothed marginals at every time and the log-likelihood
            log p(y_1:T) of the observed entries.

        Raises:
            ValueError: If a stacked field's time axis does not have T entries, or
                T − 1 for ``transition`` and ``process_noise``.

        Examples:
            A stack of the wrong length is rejected:

            >>> import jax.numpy as jnp
            >>> from filterax import KalmanSmoother
            >>> smoother = KalmanSmoother(
            ...     transition=jnp.stack([jnp.eye(1)] * 3),
            ...     process_noise=jnp.eye(1),
            ...     obs_model=jnp.eye(1),
            ...     obs_noise=jnp.eye(1),
            ...     init_mean=jnp.zeros(1),
            ...     init_cov=jnp.eye(1),
            ... )
            >>> smoother.smooth(jnp.ones((2, 1)))
            Traceback (most recent call last):
            ...
            ValueError: transition must stack 1 matrices along the time axis; got 3.

            With every entry observed, the mask changes nothing:

            >>> smoother = KalmanSmoother(
            ...     transition=jnp.eye(1),
            ...     process_noise=jnp.eye(1),
            ...     obs_model=jnp.eye(1),
            ...     obs_noise=jnp.eye(1),
            ...     init_mean=jnp.zeros(1),
            ...     init_cov=jnp.eye(1),
            ... )
            >>> plain = smoother.smooth(jnp.ones((2, 1)))
            >>> masked = smoother.smooth(jnp.ones((2, 1)), jnp.ones((2, 1), bool))
            >>> bool(jnp.allclose(plain.log_likelihood, masked.log_likelihood))
            True
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
            state, transition, solver=gaussx.DenseSolver()
        )

        # --- log-likelihood ---
        # each inert entry of an updated step adds −½ log 2π to gaussx's sum
        missing_per_step = einx.sum("T M -> T", ~mask)
        missing_updated = jnp.where(step_observed, missing_per_step, 0)
        num_missing = einx.sum("T ->", missing_updated)
        # ½ log 2π
        half_log_two_pi = 0.5 * math.log(2 * math.pi)
        # ½ log 2π per inert entry
        inert_log_density = half_log_two_pi * num_missing
        # log p(y_1:T) of the observed entries
        log_likelihood = state.log_likelihood + inert_log_density
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
    """A model field with a leading time axis of ``length``, broadcasting one matrix.

    Args:
        stack: One matrix, or a stack of them over time.
        length: Number of steps the stack must have.
        name: Field name, for the error message.

    Returns:
        The stack, with ``length`` matrices.

    Raises:
        ValueError: If a stack does not have ``length`` matrices.
    """
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
    """Substitute H_k, Σ_ε,k and y_k so that masked entries carry no information.

        Σ_y,k = blockdiag(Σ_y,k,obs, I)

    Zeroing row i of H_k, replacing row and column i of Σ_ε,k by the unit vector and
    zeroing y_k,i splits the innovation covariance Σ_y,k = H_k Σ_x,k|k−1 H_kᵀ + Σ_ε,k
    as above. Column i of the gain G_k = Σ_x,k|k−1 H_kᵀ Σ_y,k⁻¹ then vanishes, so the
    update is exactly the one with entry i deleted.

    Args:
        obs_model: Observation operators H_k, stacked over time.
        obs_noise: Observation-error covariances Σ_ε,k, stacked over time.
        observations: Observations y_k, one row per time.
        mask: ``True`` where an entry was observed.

    Returns:
        Tuple ``(obs_model, obs_noise, observations)`` with the masked entries inert.
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

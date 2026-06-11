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
"""

from __future__ import annotations

import equinox as eqx
import gaussx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float


class SquareRootKF(eqx.Module, strict=True):
    r"""Square-root parametric Kalman filter (Maybeck 1979 / gaussx).

    Propagates a Gaussian belief through a linear-Gaussian model:

    ``xₙ = Φ xₙ₋₁ + wₙ,  wₙ ~ 𝒩(0, Q)``
    ``yₙ = H xₙ + vₙ,    vₙ ~ 𝒩(0, R)``

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


# Silence unused-import warning — jnp is imported for the eqx.field
# defaults if we ever add scalar dtype fields. Keep the import so the
# module is self-contained for future extension.
_ = jnp

"""Layer-1 sequential filter components.

Each filter implements ``AbstractSequentialFilter.analysis`` and produces a
single posterior ensemble from a forecast ensemble, observation vector,
observation operator, and observation noise covariance. The forecast and
inflation steps are handled at Layer 2 (``filterax.models``).

All four filters share the same ensemble-statistics primitives from
``filterax._src.statistics`` and the Bessel-corrected gaussx Kalman gain
recipe; they differ in how the ensemble is *updated* given those statistics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray

from filterax._src._checks import check_ensemble_size
from filterax._src._protocols import (
    AbstractObsOperator,
    AbstractSequentialFilter,
)
from filterax._src._types import AnalysisResult
from filterax._src.gain import kalman_gain
from filterax._src.likelihood import innovation_covariance, log_likelihood
from filterax._src.localization import gaspari_cohn
from filterax._src.perturbations import perturbed_observations
from filterax._src.statistics import ensemble_anomalies, ensemble_mean


ObsCallable = Callable[[Float[Array, " N_x"]], Float[Array, " N_y"]]


def _apply_obs_op(
    obs_op: AbstractObsOperator | ObsCallable,
    particles: Float[Array, "N_e N_x"],
) -> Float[Array, "N_e N_y"]:
    """Vmap ``obs_op`` over an ensemble."""
    return jax.vmap(obs_op)(particles)


class StochasticEnKF(AbstractSequentialFilter, strict=True):
    """Stochastic Ensemble Kalman Filter (Evensen 1994).

    Perturbed-observation update — each member sees an independent draw from
    the observation noise:

    .. math::

        \\varepsilon^{(j)} \\sim \\mathcal{N}(0, R), \\qquad
        x_a^{(j)} = x_f^{(j)} + K (y + \\varepsilon^{(j)} - H x_f^{(j)})

    Simple and robust but introduces sampling noise in the analysis ensemble.

    Attributes:
        key: PRNG key used to draw observation perturbations. Treated as a
            JAX array (not static) so callers can split / vmap freely.
    """

    key: PRNGKeyArray

    def __init__(self, key: PRNGKeyArray | int = 0):
        if isinstance(key, int):
            key = jr.PRNGKey(key)
        self.key = key

    def analysis(
        self,
        particles: Float[Array, "N_e N_x"],
        obs: Float[Array, " N_y"],
        obs_op: AbstractObsOperator | ObsCallable,
        obs_noise: lx.AbstractLinearOperator,
        **_: Any,
    ) -> AnalysisResult:
        N_e = particles.shape[0]
        check_ensemble_size(N_e)
        obs_particles = _apply_obs_op(obs_op, particles)  # (N_e, N_y)

        K = kalman_gain(particles, obs_particles, obs_noise)  # (N_x, N_y)

        y_pert = perturbed_observations(self.key, obs, obs_noise, N_e)
        innovations = y_pert - obs_particles  # (N_e, N_y)
        particles_a = particles + innovations @ K.T  # (N_e, N_x)

        S = innovation_covariance(obs_particles, obs_noise)
        log_p = log_likelihood(obs - ensemble_mean(obs_particles), S)
        return AnalysisResult(particles=particles_a, log_likelihood=log_p)


class ETKF(AbstractSequentialFilter, strict=True):
    r"""Ensemble Transform Kalman Filter (Bishop, Etherton & Majumdar 2001).

    Deterministic square-root update in the :math:`N_e`-dimensional ensemble
    subspace. Eigendecomposes the transform precision
    :math:`\tilde{C} = (N_e - 1)I + Y'^T R^{-1} Y'`, takes its symmetric
    inverse square root, and applies it to the ensemble anomalies.

    .. math::

        \bar{w}_a = T Y'^T R^{-1} (y - \bar{y}_f), \qquad
        W_a = \sqrt{(N_e - 1) T}

        X_a = \bar{x}_f \mathbf{1}^T + X'_f (\bar{w}_a \mathbf{1}^T + W_a)
    """

    def analysis(
        self,
        particles: Float[Array, "N_e N_x"],
        obs: Float[Array, " N_y"],
        obs_op: AbstractObsOperator | ObsCallable,
        obs_noise: lx.AbstractLinearOperator,
        **_: Any,
    ) -> AnalysisResult:
        N_e = particles.shape[0]
        check_ensemble_size(N_e)
        obs_particles = _apply_obs_op(obs_op, particles)

        anom = ensemble_anomalies(particles)  # (N_e, N_x)
        obs_anom = ensemble_anomalies(obs_particles)  # (N_e, N_y)
        innovation = obs - ensemble_mean(obs_particles)  # (N_y,)

        # R^{-1} Y'  via gaussx solve (dispatches on R's structure).
        R_inv_Y = gaussx.solve_rows(obs_noise, obs_anom)  # (N_e, N_y)

        # C_tilde = (N_e - 1) I + Y'  R^{-1} Y'^T  — shape (N_e, N_e).
        C_tilde = (N_e - 1) * jnp.eye(N_e) + obs_anom @ R_inv_Y.T
        eigvals, eigvecs = jnp.linalg.eigh(C_tilde)
        # T = U diag(1/lambda) U^T, W = U diag(sqrt((N_e-1)/lambda)) U^T.
        inv_lambda = 1.0 / eigvals
        T = (eigvecs * inv_lambda) @ eigvecs.T
        W = (eigvecs * jnp.sqrt((N_e - 1) * inv_lambda)) @ eigvecs.T

        # w_bar = T @ Y' @ R^{-1} @ d  — shape (N_e,).
        w_bar = T @ (R_inv_Y @ innovation)

        x_bar = ensemble_mean(particles)
        # Mean update: x_bar_a = x_bar + sum_k w_bar[k] anom[k, :].
        mean_a = x_bar + w_bar @ anom  # (N_x,)
        # Perturbation update: A_a = W @ anom — symmetric W gives symmetric
        # square root so W.T == W.
        particles_a = mean_a[None, :] + W @ anom

        S = innovation_covariance(obs_particles, obs_noise)
        log_p = log_likelihood(innovation, S)
        return AnalysisResult(particles=particles_a, log_likelihood=log_p)


class EnSRF(AbstractSequentialFilter, strict=True):
    r"""Ensemble Square Root Filter (Tippett 2003 unified form).

    Separate mean and perturbation updates that produce the exact analysis
    covariance :math:`P_a = (I - KH) P_f` (in the linear-Gaussian limit)
    without perturbed observations:

    .. math::

        \bar{x}_a = \bar{x}_f + K (y - \bar{y}_f)

        X'_a = W_{\text{EnSRF}}^T X'_f

    The transform shares the ETKF eigendecomposition of
    :math:`\tilde{C} = (N_e - 1) I + Y'^T R^{-1} Y'` but uses the
    *one-sided* square root :math:`W = \mathrm{diag}\sqrt{(N_e-1)/\lambda_k}\,U^T`
    rather than ETKF's symmetric :math:`U\,\mathrm{diag}\sqrt{\cdot}\,U^T`.
    The two filters share the same analysis covariance — they differ only in
    the particular ensemble realisation of that covariance (Tippett et al.
    2003 show that both choices are "ensemble square root filters" in this
    sense). EnSRF's one-sided sqrt is the choice originally proposed by
    Whitaker & Hamill (2002) for serial scalar observations.
    """

    def analysis(
        self,
        particles: Float[Array, "N_e N_x"],
        obs: Float[Array, " N_y"],
        obs_op: AbstractObsOperator | ObsCallable,
        obs_noise: lx.AbstractLinearOperator,
        **_: Any,
    ) -> AnalysisResult:
        N_e = particles.shape[0]
        check_ensemble_size(N_e)
        obs_particles = _apply_obs_op(obs_op, particles)

        x_bar = ensemble_mean(particles)
        y_bar = ensemble_mean(obs_particles)
        anom = ensemble_anomalies(particles)  # (N_e, N_x)
        obs_anom = ensemble_anomalies(obs_particles)  # (N_e, N_y)
        innovation = obs - y_bar

        # Mean update uses the standard ensemble gain (full Bessel form).
        K = kalman_gain(particles, obs_particles, obs_noise)  # (N_x, N_y)
        mean_a = x_bar + K @ innovation

        # Perturbation update via the one-sided ensemble-space square root.
        R_inv_Y = gaussx.solve_rows(obs_noise, obs_anom)  # (N_e, N_y)
        C_tilde = (N_e - 1) * jnp.eye(N_e) + obs_anom @ R_inv_Y.T
        eigvals, eigvecs = jnp.linalg.eigh(C_tilde)
        sqrt_scale = jnp.sqrt((N_e - 1) / eigvals)
        # W = diag(sqrt_scale) @ U^T  (non-symmetric one-sided sqrt of T).
        # Pert update with rows-as-members: particles_a = mean + W^T @ anom
        # where W^T = U @ diag(sqrt_scale).
        pert_correction = (eigvecs * sqrt_scale) @ (eigvecs.T @ anom)
        particles_a = mean_a[None, :] + pert_correction

        S_op = innovation_covariance(obs_particles, obs_noise)
        log_p = log_likelihood(innovation, S_op)
        return AnalysisResult(particles=particles_a, log_likelihood=log_p)


class LETKF(AbstractSequentialFilter, strict=True):
    r"""Local Ensemble Transform Kalman Filter (Hunt, Kostelich & Szunyogh 2007).

    Runs an independent ETKF analysis at each state grid point using only
    the observations within ``localizer.radius`` (R-localization). For
    point :math:`i`:

    1. Pick observation indices :math:`\mathcal{I}_i` with
       :math:`d(x_i, y_k) \le r`.
    2. Inflate the local observation noise by :math:`1/\rho_k` where
       :math:`\rho_k = \rho_{GC}(d_{ik}/r)`.
    3. Run ETKF on the local subset and write back the analysis at point
       :math:`i`.

    The local analyses are embarrassingly parallel and are mapped with
    :func:`jax.vmap` over grid points. ``state_coords`` and ``obs_coords``
    are required keyword arguments.

    Attributes:
        radius: Localization half-width. Observations beyond this distance
            from a state point are zero-weighted (Gaspari-Cohn supports up
            to ``2 * radius`` but local-obs selection uses ``radius`` as
            the hard cutoff).
        taper_fn: Distance-to-weight function. Defaults to
            :func:`gaspari_cohn`. Must have signature
            ``(distances, radius) -> weights``.
    """

    radius: float = eqx.field(static=True)
    taper_fn: Callable[[Float[Array, "..."], float], Float[Array, "..."]] = eqx.field(
        static=True, default=gaspari_cohn
    )

    def analysis(
        self,
        particles: Float[Array, "N_e N_x"],
        obs: Float[Array, " N_y"],
        obs_op: AbstractObsOperator | ObsCallable,
        obs_noise: lx.AbstractLinearOperator,
        *,
        state_coords: Float[Array, "N_x D"] | None = None,
        obs_coords: Float[Array, "N_y D"] | None = None,
        **_: Any,
    ) -> AnalysisResult:
        N_e, N_x = particles.shape
        check_ensemble_size(N_e)
        if state_coords is None or obs_coords is None:
            raise ValueError(
                "LETKF.analysis requires state_coords and obs_coords keyword "
                "arguments — pass them through the L2 model or call site."
            )
        if state_coords.shape[0] != N_x:
            raise ValueError(
                "state_coords must have one row per state component; got "
                f"{state_coords.shape[0]} vs N_x={N_x}."
            )
        if obs_coords.shape[0] != obs.shape[0]:
            raise ValueError(
                "obs_coords must have one row per observation; got "
                f"{obs_coords.shape[0]} vs N_y={obs.shape[0]}."
            )

        obs_particles = _apply_obs_op(obs_op, particles)  # (N_e, N_y)
        obs_anom = ensemble_anomalies(obs_particles)
        y_bar = ensemble_mean(obs_particles)
        anom = ensemble_anomalies(particles)
        x_bar = ensemble_mean(particles)
        innovation = obs - y_bar

        # Observation-noise diagonal — we localize by inflating per-obs variance.
        # Materialising R's diagonal accepts dense and DiagonalLinearOperator alike.
        R_diag = jnp.diag(obs_noise.as_matrix())

        def per_point(
            state_point: Float[Array, " D"],
        ) -> tuple[Float[Array, "N_e 1"], Float[Array, "N_e N_e"]]:
            # Distances from this state point to every observation.
            diff = obs_coords - state_point[None, :]
            dist = jnp.sqrt(jnp.sum(diff * diff, axis=1))
            rho = self.taper_fn(dist, self.radius)  # (N_y,)

            # R-localization: scale per-observation variance by 1/rho. When
            # rho == 0 the observation is fully discounted — we set its
            # inverse-variance contribution to zero.
            safe_rho = jnp.where(rho > 0, rho, 1.0)
            inv_R_loc = jnp.where(rho > 0, rho / R_diag, 0.0)  # (N_y,)

            # Local ETKF in ensemble (N_e) space.
            # R^{-1}_loc Y'  with Y' shape (N_e, N_y).
            R_inv_Y = obs_anom * inv_R_loc[None, :]  # (N_e, N_y)
            C_tilde = (N_e - 1) * jnp.eye(N_e) + obs_anom @ R_inv_Y.T
            eigvals, eigvecs = jnp.linalg.eigh(C_tilde)
            inv_lambda = 1.0 / eigvals
            T = (eigvecs * inv_lambda) @ eigvecs.T
            W = (eigvecs * jnp.sqrt((N_e - 1) * inv_lambda)) @ eigvecs.T
            w_bar = T @ (R_inv_Y @ innovation)
            # Reference the safe_rho to keep jaxtype consistency; logically a no-op.
            _ = safe_rho
            return w_bar[:, None], W  # (N_e, 1), (N_e, N_e)

        # vmap over state grid points — for each point we get (w_bar, W).
        w_bars, Ws = jax.vmap(per_point)(state_coords)
        # w_bars: (N_x, N_e, 1) -> (N_x, N_e). Ws: (N_x, N_e, N_e).
        w_bars = w_bars[..., 0]

        # Apply per-point updates: anomalies are *global* (every grid point
        # shares the same ensemble anomaly column), so x_a[i] = x_bar[i] +
        # sum_k (w_bar[i, k] + W[i, :, k]) * anom[k, i].
        # We use einsum to express both terms compactly.
        mean_correction = jnp.einsum("ik,ki->i", w_bars, anom)  # (N_x,)
        pert_correction = jnp.einsum("ijk,ki->ji", Ws, anom)  # (N_e, N_x)

        mean_a = x_bar + mean_correction
        particles_a = mean_a[None, :] + pert_correction

        S = innovation_covariance(obs_particles, obs_noise)
        log_p = log_likelihood(innovation, S)
        return AnalysisResult(particles=particles_a, log_likelihood=log_p)

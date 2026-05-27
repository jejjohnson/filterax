"""Layer-1 sequential filter components.

Each filter implements :meth:`AbstractSequentialFilter.analysis` and
produces a single posterior ensemble from a forecast ensemble, an
observation vector, an observation operator, and an observation noise
covariance. The forecast and inflation steps live at Layer 2
(``filterax.models``).

All four filters share the same ensemble-statistics primitives from
:mod:`filterax._src.statistics` and the Bessel-corrected gaussx Kalman-gain
recipe; they differ only in *how the ensemble is updated* given those
statistics.

Notation
--------
* ``Nₑ``           — ensemble size
* ``Nₓ``           — state dimension
* ``Nᵧ``           — observation dimension
* ``X ∈ ℝ^{Nₑ×Nₓ}``  — forecast ensemble, rows are members
* ``Y = H(X) ∈ ℝ^{Nₑ×Nᵧ}``  — ensemble in obs space (linearised when ``H`` is nonlinear)
* ``X′, Y′``        — centred anomaly matrices
* ``x̄, ȳ``         — ensemble means
* ``R``             — observation error covariance
* ``S = Cᴴᴴ + R``   — innovation covariance (``Cᴴᴴ = (Nₑ−1)⁻¹ Y′ᵀ Y′``)
* ``K = Cˣᴴ S⁻¹``  — Kalman gain
* ``v = y − ȳ``    — innovation
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

# Floor applied to eigenvalues of the (Nₑ, Nₑ) transform precision before
# inversion / square root. The eigenvalues are mathematically bounded
# below by Nₑ − 1 (≥ 2 for any valid ensemble), so a tiny absolute floor
# only kicks in if a near-zero negative slipped through due to round-off.
_EIG_FLOOR: float = 1e-12


def _apply_obs_op(
    obs_op: AbstractObsOperator | ObsCallable,
    particles: Float[Array, "N_e N_x"],
) -> Float[Array, "N_e N_y"]:
    """vmap ``obs_op`` over an ensemble. Cost ``O(Nₑ · cost(H))``."""
    return jax.vmap(obs_op)(particles)


def _symmetrize(matrix: Float[Array, "N N"]) -> Float[Array, "N N"]:
    """Return ``½ (M + Mᵀ)`` — kills accumulated antisymmetric round-off."""
    return 0.5 * (matrix + matrix.T)


def _transform_eig(
    obs_anom: Float[Array, "N_e N_y"],
    R_inv_Y: Float[Array, "N_e N_y"],
    N_e: int,
) -> tuple[Float[Array, " N_e"], Float[Array, "N_e N_e"]]:
    r"""Eigendecomposition of the ETKF transform precision.

    ``C̃ = (Nₑ − 1) I + Y′ R⁻¹ Y′ᵀ ∈ ℝ^{Nₑ×Nₑ}``

    Symmetrises ``C̃`` to absorb floating-point antisymmetry, calls
    :func:`jax.numpy.linalg.eigh`, and clamps eigenvalues to a tiny
    positive floor so the downstream ``1/λ`` and ``√λ`` stay finite.

    Returns ``(λ, U)`` where ``C̃ = U diag(λ) Uᵀ`` and all ``λ_k > 0``.
    Cost ``O(Nₑ³)``.
    """
    C_tilde = (N_e - 1) * jnp.eye(N_e) + einx.dot("e a, f a -> e f", obs_anom, R_inv_Y)
    eigvals, eigvecs = jnp.linalg.eigh(_symmetrize(C_tilde))
    eigvals = jnp.maximum(eigvals, _EIG_FLOOR)
    return eigvals, eigvecs


class StochasticEnKF(AbstractSequentialFilter, strict=True):
    r"""Stochastic Ensemble Kalman Filter (Evensen 1994).

    Perturbed-observation update — each member sees an independent draw
    from the observation noise:

    ``ε⁽ʲ⁾ ~ 𝒩(0, R),  x⁽ʲ⁾_a = x⁽ʲ⁾_f + K (y + ε⁽ʲ⁾ − H x⁽ʲ⁾_f)``

    Simple and robust; the perturbations introduce extra Monte Carlo
    variance whose magnitude scales as ``1/√Nₑ``.

    PRNG key handling. The key stored on the filter is the *default*
    source of randomness. ``analysis(..., key=subkey)`` overrides it for
    a single call — this is how the Layer-2 :class:`filterax.StochasticEnKF`
    threads a fresh sub-key per assimilation window so consecutive
    perturbations stay independent. If you call ``analysis`` directly,
    pass a freshly split key each cycle or you will get identical
    observation perturbations every window.

    Attributes:
        key: Default PRNG key used when no ``key`` is supplied in the
            ``analysis`` kwargs.
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
        *,
        key: PRNGKeyArray | None = None,
        **_: Any,
    ) -> AnalysisResult:
        r"""Stochastic EnKF analysis step.

        Args:
            particles: Forecast ensemble ``(Nₑ, Nₓ)``.
            obs: Observation ``y`` of shape ``(Nᵧ,)``.
            obs_op: ``H`` applied to a single state vector.
            obs_noise: Observation covariance ``R``.
            key: Per-step PRNG key. When ``None``, falls back to
                ``self.key`` — the same draw is repeated across calls
                unless callers split externally.

        Returns:
            :class:`AnalysisResult` with the perturbed-observation
            posterior ensemble and the Gaussian log-likelihood of the
            innovation under ``S = Cᴴᴴ + R``.
        """
        N_e = particles.shape[0]
        check_ensemble_size(N_e)
        step_key = self.key if key is None else key

        obs_particles = _apply_obs_op(obs_op, particles)  # Y ∈ ℝ^{Nₑ×Nᵧ}

        # K = Cˣᴴ (Cᴴᴴ + R)⁻¹ via gaussx-dispatched Woodbury solve.
        K = kalman_gain(particles, obs_particles, obs_noise)  # (Nₓ, Nᵧ)

        # y_pert[j] = y + ε⁽ʲ⁾ with ε⁽ʲ⁾ ~ 𝒩(0, R) — sample structure-
        # aware so diagonal R never densifies.
        y_pert = perturbed_observations(step_key, obs, obs_noise, N_e)
        innovations = y_pert - obs_particles  # v⁽ʲ⁾ = y + ε⁽ʲ⁾ − H x⁽ʲ⁾
        # x⁽ʲ⁾_a = x⁽ʲ⁾ + K v⁽ʲ⁾, summed over the obs axis.
        increment = einx.dot("e y, x y -> e x", innovations, K)
        particles_a = einx.add("e x, e x -> e x", particles, increment)

        S = innovation_covariance(obs_particles, obs_noise)
        log_p = log_likelihood(obs - ensemble_mean(obs_particles), S)
        return AnalysisResult(particles=particles_a, log_likelihood=log_p)


class ETKF(AbstractSequentialFilter, strict=True):
    r"""Ensemble Transform Kalman Filter (Bishop, Etherton & Majumdar 2001).

    Deterministic square-root update working entirely in the
    ``Nₑ``-dimensional ensemble subspace. Define the transform precision
    in the ensemble space and its symmetric inverse square root:

    ``C̃ = (Nₑ − 1) I + Y′ R⁻¹ Y′ᵀ``
    ``T = C̃⁻¹,  Wₐ = √((Nₑ − 1) T)``

    Mean and perturbation weights:

    ``w̄ₐ = T Y′ R⁻¹ v,   X_a = x̄ 𝟙ᵀ + X′ᵀ (w̄ₐ 𝟙ᵀ + Wₐ)``

    The eigendecomposition is taken on the symmetrised ``C̃`` with a
    positive eigenvalue floor to keep the square root well-defined under
    floating-point error.

    Complexity ``O(Nₑ² Nᵧ + Nₑ³)`` per analysis — the inner solve is
    routed through gaussx so structured ``R`` does not densify.
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

        anom = ensemble_anomalies(particles)  # X′ ∈ ℝ^{Nₑ×Nₓ}
        obs_anom = ensemble_anomalies(obs_particles)  # Y′ ∈ ℝ^{Nₑ×Nᵧ}
        x_bar = ensemble_mean(particles)
        innovation = obs - ensemble_mean(obs_particles)

        # R⁻¹ Y′ — gaussx structural dispatch (no dense Nᵧ × Nᵧ matrix
        # for DiagonalLinearOperator, Toeplitz, etc.).
        R_inv_Y = gaussx.solve_rows(obs_noise, obs_anom)  # (Nₑ, Nᵧ)

        eigvals, eigvecs = _transform_eig(obs_anom, R_inv_Y, N_e)

        # Symmetric inverse square root: Wₐ = U diag(√((Nₑ−1)/λ)) Uᵀ.
        # ``eigvecs`` is U; columns are eigenvectors, so eigvecs[:, k] is the
        # k-th eigenvector. Equivalently, U[i, k] is the i-th component of
        # the k-th eigenvector — when we sum over the FIRST axis we apply Uᵀ.
        inv_lambda = 1.0 / eigvals
        sqrt_scale = jnp.sqrt((N_e - 1) * inv_lambda)

        # y = Y′ R⁻¹ v ∈ ℝ^{Nₑ}.
        Y_R_inv_v = einx.dot("e y, y -> e", R_inv_Y, innovation)
        # w̄ = T y = U diag(1/λ) Uᵀ y. Sum the FIRST axis of U for Uᵀ.
        Ut_y = einx.dot("e a, e -> a", eigvecs, Y_R_inv_v)
        w_bar = einx.dot("e a, a -> e", eigvecs, inv_lambda * Ut_y)

        # Mean update: x̄_a = x̄ + Σ_k w̄[k] X′[k, :].
        mean_correction = einx.dot("e, e x -> x", w_bar, anom)

        # Wₐ X′ = U diag(sqrt_scale) Uᵀ X′ — three factored matmuls.
        Ut_anom = einx.dot("e a, e x -> a x", eigvecs, anom)
        scaled = einx.multiply("a x, a -> a x", Ut_anom, sqrt_scale)
        pert_correction = einx.dot("e a, a x -> e x", eigvecs, scaled)

        particles_a = (
            einx.add("x, x -> x", x_bar, mean_correction)[None, :] + pert_correction
        )

        S = innovation_covariance(obs_particles, obs_noise)
        log_p = log_likelihood(innovation, S)
        return AnalysisResult(particles=particles_a, log_likelihood=log_p)


class EnSRF(AbstractSequentialFilter, strict=True):
    r"""Ensemble Square Root Filter (Whitaker & Hamill 2002, batch form).

    Separate mean and perturbation updates that produce the exact
    analysis covariance ``Pₐ = (I − KH) P_f`` in the linear-Gaussian
    limit, without perturbed observations:

    * Mean from the Kalman gain: ``x̄ₐ = x̄_f + K (y − ȳ_f)``.
    * Perturbations via the symmetric ETKF transform applied to anomalies.

    Why "EnSRF" and not "ETKF" then? In *batch* form (this class) the two
    are numerically equivalent on the analysis ensemble — both choices
    of ensemble square root produce the same posterior covariance and
    the symmetric sqrt is the only PSD- and mean-preserving choice for
    rows-as-members anomalies (Tippett et al. 2003). The classical
    Whitaker & Hamill (2002) distinction — a *one-sided* reduced gain on
    perturbations — only matters when observations are assimilated one at
    a time. That serial variant lives in Wave 4 as
    ``filterax.filters.EnSRF_Serial``.

    For now, :class:`EnSRF` gives users the more familiar API (mean from
    ``K``, perturbations from a sqrt-of-``T``) while remaining
    mathematically equivalent to :class:`ETKF`. Pick whichever phrasing
    is more idiomatic for your domain.

    Complexity matches ETKF: ``O(Nₑ² Nᵧ + Nₑ³)``.
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
        anom = ensemble_anomalies(particles)  # (Nₑ, Nₓ)
        obs_anom = ensemble_anomalies(obs_particles)  # (Nₑ, Nᵧ)
        innovation = obs - ensemble_mean(obs_particles)

        # Mean update via the standard Bessel-corrected gain.
        K = kalman_gain(particles, obs_particles, obs_noise)
        mean_a = einx.add(
            "x, x -> x",
            x_bar,
            einx.dot("x y, y -> x", K, innovation),
        )

        # Perturbation update via the symmetric ETKF sqrt
        # ``Wₐ = U diag(√((Nₑ−1)/λ)) Uᵀ`` applied to anomalies. Symmetric
        # because that is the only sqrt of ``T`` that maps mean-zero
        # anomalies to mean-zero anomalies, leaving the analysis mean
        # equal to ``x̄ + K v`` (see Tippett et al. 2003 §3).
        R_inv_Y = gaussx.solve_rows(obs_noise, obs_anom)
        eigvals, eigvecs = _transform_eig(obs_anom, R_inv_Y, N_e)
        sqrt_scale = jnp.sqrt((N_e - 1) / eigvals)
        Ut_anom = einx.dot("e a, e x -> a x", eigvecs, anom)
        scaled = einx.multiply("a x, a -> a x", Ut_anom, sqrt_scale)
        pert_correction = einx.dot("e a, a x -> e x", eigvecs, scaled)

        particles_a = mean_a[None, :] + pert_correction

        S = innovation_covariance(obs_particles, obs_noise)
        log_p = log_likelihood(innovation, S)
        return AnalysisResult(particles=particles_a, log_likelihood=log_p)


class LETKF(AbstractSequentialFilter, strict=True):
    r"""Local Ensemble Transform Kalman Filter (Hunt, Kostelich & Szunyogh 2007).

    Runs an independent ETKF analysis at each state grid point using only
    observations whose distance to that grid point is ``≤ radius``
    (R-localization). For grid point ``i``:

    1. Compute distances ``dᵢₖ = ‖xᵢ − yₖ‖₂`` and tapers
       ``ρᵢₖ = ρ(dᵢₖ / r)`` via ``taper_fn`` (default Gaspari-Cohn).
    2. **Hard cutoff at ``r``** — observations with ``dᵢₖ > r`` are
       dropped entirely. (Gaspari-Cohn has compact support out to ``2r``
       so a purely taper-based mask would let *every* nonzero-tapered
       observation participate. We use the explicit distance cutoff to
       match the documented selection rule.)
    3. Inflate the local observation variance by ``1/ρᵢₖ``:
       ``R⁻¹_loc[k] = ρᵢₖ / Rₖₖ`` (zero when ``ρᵢₖ = 0``).
    4. Solve the local ``Nₑ × Nₑ`` ETKF transform and write back the
       analysis at grid point ``i``.

    R-localization assumes ``R`` is **diagonal** (Hunt et al. 2007 §2).
    We accept :class:`lineax.DiagonalLinearOperator` directly via
    :func:`lineax.diagonal` (no densification); anything else raises
    :class:`NotImplementedError`.

    Local analyses are embarrassingly parallel; we ``jax.vmap`` over
    state grid points and accumulate the per-point ``(w̄ᵢ, Wᵢ)`` weights
    in a single call.

    Attributes:
        radius: Localization half-width ``r``. Observations beyond ``r``
            are excluded entirely.
        taper_fn: Distance-to-weight function with signature
            ``(distances, radius) -> weights``. Defaults to
            :func:`filterax.gaspari_cohn`.
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
                "arguments — pass them through the L2 model or the call site."
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
        if not isinstance(obs_noise, lx.DiagonalLinearOperator):
            raise NotImplementedError(
                "LETKF R-localization assumes diagonal observation noise "
                "(Hunt et al. 2007 §2). Pass a lineax.DiagonalLinearOperator "
                "or decorrelate observations beforehand."
            )

        obs_particles = _apply_obs_op(obs_op, particles)  # Y ∈ ℝ^{Nₑ×Nᵧ}
        obs_anom = ensemble_anomalies(obs_particles)
        anom = ensemble_anomalies(particles)
        x_bar = ensemble_mean(particles)
        innovation = obs - ensemble_mean(obs_particles)

        # Diagonal of R extracted without ever materialising the dense matrix.
        R_diag = lx.diagonal(obs_noise)
        radius = self.radius

        def per_point(
            state_point: Float[Array, " D"],
        ) -> tuple[Float[Array, " N_e"], Float[Array, "N_e N_e"]]:
            """Local ETKF weights at one grid point.

            Returns ``(w̄ᵢ, Wᵢ)`` with shapes ``(Nₑ,)`` and ``(Nₑ, Nₑ)``.
            """
            # ‖xᵢ − yₖ‖₂ — Euclidean distance to every observation.
            diff = einx.subtract("y d, d -> y d", obs_coords, state_point)
            dist = jnp.sqrt(einx.sum("y d -> y", diff * diff))

            # Distance-based hard cutoff at the radius; taper inside.
            within = dist <= radius
            rho = jnp.where(within, self.taper_fn(dist, radius), 0.0)

            # R⁻¹_loc = ρ ⊙ R⁻¹ (diagonal, no matrix materialisation).
            inv_R_loc = jnp.where(within, rho / R_diag, 0.0)  # (Nᵧ,)

            # Local ETKF in ensemble space.
            # R⁻¹_loc Y′ — broadcasted elementwise over the obs axis.
            R_inv_Y = einx.multiply("e y, y -> e y", obs_anom, inv_R_loc)
            eigvals_loc, eigvecs_loc = _transform_eig(obs_anom, R_inv_Y, N_e)
            inv_lambda = 1.0 / eigvals_loc
            sqrt_scale = jnp.sqrt((N_e - 1) * inv_lambda)
            # w̄ᵢ = U diag(1/λ) Uᵀ (Y′ R⁻¹_loc v). Sum the FIRST axis of U
            # to apply Uᵀ — see ETKF.analysis comments.
            Y_R_inv_v = einx.dot("e y, y -> e", R_inv_Y, innovation)
            Ut_y = einx.dot("e a, e -> a", eigvecs_loc, Y_R_inv_v)
            w_bar_i = einx.dot("e a, a -> e", eigvecs_loc, inv_lambda * Ut_y)
            # Wᵢ = U diag(sqrt_scale) Uᵀ — symmetric square root.
            W_i = einx.dot(
                "i a, j a -> i j",
                eigvecs_loc * sqrt_scale[None, :],
                eigvecs_loc,
            )
            return w_bar_i, W_i

        # vmap over state grid points → per-point (w̄ᵢ, Wᵢ).
        w_bars, Ws = jax.vmap(per_point)(state_coords)  # (Nₓ, Nₑ), (Nₓ, Nₑ, Nₑ)

        # Apply per-point updates with anomaly columns shared globally:
        # ``x_a[i] = x̄[i] + Σₖ w̄[i, k] X′[k, i]`` for the mean and
        # ``x⁽ʲ⁾_a[i] = x̄_a[i] + Σₖ W[i, j, k] X′[k, i]`` for members.
        mean_correction = einx.dot("x e, e x -> x", w_bars, anom)
        pert_correction = einx.dot("x j k, k x -> j x", Ws, anom)

        mean_a = einx.add("x, x -> x", x_bar, mean_correction)
        particles_a = mean_a[None, :] + pert_correction

        S = innovation_covariance(obs_particles, obs_noise)
        log_p = log_likelihood(innovation, S)
        return AnalysisResult(particles=particles_a, log_likelihood=log_p)

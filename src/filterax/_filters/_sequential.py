"""Layer-1 sequential filter components.

Each filter implements :meth:`AbstractSequentialFilter.analysis` and
produces a single posterior ensemble from a forecast ensemble, an
observation vector, an observation operator, and an observation noise
covariance. The forecast and inflation steps live at Layer 2
(``filterax.models``).

All four filters share the same ensemble-statistics primitives from
:mod:`filterax._primitives._statistics` and the Bessel-corrected gaussx Kalman-gain
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

from filterax._checks import check_ensemble_size
from filterax._primitives._gain import kalman_gain
from filterax._primitives._likelihood import innovation_covariance, log_likelihood
from filterax._primitives._localization import gaspari_cohn
from filterax._primitives._perturbations import perturbed_observations
from filterax._primitives._statistics import ensemble_anomalies, ensemble_mean
from filterax._protocols import (
    AbstractObsOperator,
    AbstractSequentialFilter,
)
from filterax._types import AnalysisResult


ObsCallable = Callable[[Float[Array, " N_x"]], Float[Array, " N_y"]]


def _apply_obs_op(
    obs_op: AbstractObsOperator | ObsCallable,
    particles: Float[Array, "N_e N_x"],
) -> Float[Array, "N_e N_y"]:
    """vmap ``obs_op`` over an ensemble. Cost ``O(Nₑ · cost(H))``."""
    return jax.vmap(obs_op)(particles)


def _etkf_inner_spectrum(
    obs_anom: Float[Array, "K N_y"],
    R_inv_Y: Float[Array, "K N_y"],
) -> tuple[Float[Array, " N_y"], Float[Array, "K N_y"]]:
    r"""Rank-``Nᵧ`` spectrum of the inner product ``Y′ R⁻¹ Y′ᵀ``.

    The decomposition

    $$
    Y' R^{-1} Y'^{\top} = U_y \,\mathrm{diag}(\lambda_i)\, U_y^{\top}
    $$

    holds with ``U_y ∈ ℝ^{K × Nᵧ}`` an
    orthonormal basis for the column span of ``Y′`` (the only directions
    where the outer product has non-trivial action). The remaining
    ``K − Nᵧ`` eigenvalues are all zero; they correspond to a degenerate
    null subspace that downstream matrix-function evaluations short-
    circuit through ``g(K_base)`` and never need explicit eigenvectors
    for.

    Computed via thin QR + a small ``(Nᵧ, Nᵧ)`` eigh:

    1. ``Y′ = Q R_qr`` (thin QR; ``Q ∈ ℝ^{K × Nᵧ}``, ``R_qr ∈ ℝ^{Nᵧ × Nᵧ}``).
    2. ``M_y = Qᵀ (Y′ R⁻¹) R_qrᵀ = R_qr R⁻¹ R_qrᵀ`` (symmetric PSD,
       ``Nᵧ × Nᵧ``).
    3. ``M_y = V_M diag(λ_i) V_Mᵀ`` via eigh on the small matrix.
    4. ``U_y = Q V_M``.

    This avoids the ``(K, K)`` eigh of the design-doc form
    ``(Nₑ − 1) I + Y′ R⁻¹ Y′ᵀ`` whose ``K − Nᵧ`` structurally-repeated
    eigenvalues give ``NaN`` reverse-mode gradients in JAX (issue #82).

    Used in two places:

    * ETKF-style filters with ``K = Nₑ`` (``ETKF``, ``EnSRF``,
      ``ETKF_Livings``; ``LETKF`` per-grid-point).
    * ESTKF with ``K = Nₑ − 1`` (reduced subspace).

    Assumes ``Nᵧ ≤ K``; gradient-stable when ``Y′`` has full column
    rank (i.e. observations are linearly independent across the
    ensemble), which is the typical ensemble-filtering setting.
    """
    Q, R_qr = jnp.linalg.qr(obs_anom)  # Q: (K, N_y), R_qr: (N_y, N_y)
    # Qᵀ (Y′ R⁻¹) = R_qr R⁻¹ (since Qᵀ Y′ = R_qr).
    R_qr_R_inv = einx.dot("e a, e b -> a b", Q, R_inv_Y)  # (N_y, N_y)
    M_y = einx.dot("a b, c b -> a c", R_qr_R_inv, R_qr)  # (N_y, N_y)
    M_y = 0.5 * (M_y + jnp.swapaxes(M_y, -1, -2))
    eigvals, V_M = jnp.linalg.eigh(M_y)
    # ``M_y`` is PSD by construction; tiny negative round-off is clipped.
    eigvals = jnp.maximum(eigvals, 0.0)
    U_y = einx.dot("e a, a b -> e b", Q, V_M)
    return eigvals, U_y


def _apply_ctilde_func(
    U_y: Float[Array, "K N_y"],
    g_diff: Float[Array, " N_y"],
    g_base: Float[Array, ""],
    v: Float[Array, "K ..."],
) -> Float[Array, "K ..."]:
    r"""Apply ``g(C̃)`` to ``v`` via the rank-``Nᵧ`` correction form.

    $$
    g(\tilde{C}) = g_{\mathrm{base}} \cdot I
        + U_y \,\mathrm{diag}(g_{\mathrm{diff}})\, U_y^{\top}
    $$

    with ``g_diff[i] = g((Nₑ − 1) + λ_i) − g_base`` and
    ``g_base = g(Nₑ − 1)``. The caller supplies the precomputed
    differences so the same ``U_y`` / ``λ`` decomposition can be reused
    across multiple matrix functions in one analysis step.

    ``v`` is a vector ``(K,)`` or matrix ``(K, …)``; the leading axis is
    the one ``g(C̃)`` acts on.
    """
    Ut_v = einx.dot("e a, e ... -> a ...", U_y, v)
    scaled = einx.multiply("a, a ... -> a ...", g_diff, Ut_v)
    correction = einx.dot("e a, a ... -> e ...", U_y, scaled)
    return g_base * v + correction


class StochasticEnKF(AbstractSequentialFilter, strict=True):
    r"""Stochastic Ensemble Kalman Filter (Evensen 1994).

    Perturbed-observation update — each member sees an independent draw
    from the observation noise:

    $$
    \epsilon^{(j)} \sim \mathcal{N}(0, R), \qquad
    x^{(j)}_a = x^{(j)}_f + K \big(y + \epsilon^{(j)} - H x^{(j)}_f\big)
    $$

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

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> from filterax.filters import StochasticEnKF
        >>> particles = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        >>> R = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> filt = StochasticEnKF(key=jr.key(0))
        >>> result = filt.analysis(
        ...     particles, jnp.array([1.0, 0.5]), lambda x: x, R, key=jr.key(1)
        ... )
        >>> result.particles.shape
        (3, 2)
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

    $$
    \tilde{C} = (N_e - 1) I + Y' R^{-1} Y'^{\top}, \qquad
    T = \tilde{C}^{-1}, \qquad
    W_a = \sqrt{(N_e - 1)\, T}
    $$

    Mean and perturbation weights:

    $$
    \bar{w}_a = T\, Y' R^{-1} v, \qquad
    X_a = \bar{x} \mathbf{1}^{\top}
        + X'^{\top} \big(\bar{w}_a \mathbf{1}^{\top} + W_a\big)
    $$

    The eigendecomposition is taken on the symmetrised ``C̃`` with a
    positive eigenvalue floor to keep the square root well-defined under
    floating-point error.

    Complexity ``O(Nₑ² Nᵧ + Nₑ³)`` per analysis — the inner solve is
    routed through gaussx so structured ``R`` does not densify.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from filterax.filters import ETKF
        >>> particles = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        >>> R = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> result = ETKF().analysis(
        ...     particles, jnp.array([1.0, 0.5]), lambda x: x, R
        ... )
        >>> result.particles.shape
        (3, 2)
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

        # Rank-Nᵧ spectrum of Y′ R⁻¹ Y′ᵀ via QR + small eigh; avoids the
        # gradient-NaN trap of an eigh on the full (Nₑ, Nₑ) C̃ matrix.
        eigvals, U_y = _etkf_inner_spectrum(obs_anom, R_inv_Y)

        # T = C̃⁻¹ = (1/(Nₑ−1)) I + U_y diag(inv_diff) U_yᵀ.
        inv_base = jnp.asarray(1.0 / (N_e - 1), dtype=anom.dtype)
        inv_diff = 1.0 / ((N_e - 1) + eigvals) - inv_base
        # Wₐ = √((Nₑ−1) C̃⁻¹) = I + U_y diag(sqrt_diff) U_yᵀ.
        sqrt_base = jnp.asarray(1.0, dtype=anom.dtype)
        sqrt_diff = jnp.sqrt((N_e - 1) / ((N_e - 1) + eigvals)) - sqrt_base

        # Mean: w̄ = T (Y′ R⁻¹ v); then mean_correction = w̄ᵀ X′.
        Y_R_inv_v = einx.dot("e y, y -> e", R_inv_Y, innovation)
        w_bar = _apply_ctilde_func(U_y, inv_diff, inv_base, Y_R_inv_v)
        mean_correction = einx.dot("e, e x -> x", w_bar, anom)

        # Perturbations: Wₐ X′.
        pert_correction = _apply_ctilde_func(U_y, sqrt_diff, sqrt_base, anom)

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

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from filterax.filters import EnSRF
        >>> particles = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        >>> R = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> result = EnSRF().analysis(
        ...     particles, jnp.array([1.0, 0.5]), lambda x: x, R
        ... )
        >>> result.particles.shape
        (3, 2)
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
        eigvals, U_y = _etkf_inner_spectrum(obs_anom, R_inv_Y)
        sqrt_base = jnp.asarray(1.0, dtype=anom.dtype)
        sqrt_diff = jnp.sqrt((N_e - 1) / ((N_e - 1) + eigvals)) - sqrt_base
        pert_correction = _apply_ctilde_func(U_y, sqrt_diff, sqrt_base, anom)

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

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from filterax.filters import LETKF
        >>> particles = jax.random.normal(jax.random.key(0), (4, 3))
        >>> state_coords = jnp.arange(3.0)[:, None]  # 1-D grid
        >>> obs_coords = jnp.array([[0.0], [2.0]])
        >>> R = lx.DiagonalLinearOperator(0.5 * jnp.ones(2))
        >>> result = LETKF(radius=1.5).analysis(
        ...     particles,
        ...     jnp.array([0.1, -0.2]),
        ...     lambda x: x[::2],  # observe grid points 0 and 2
        ...     R,
        ...     state_coords=state_coords,
        ...     obs_coords=obs_coords,
        ... )
        >>> result.particles.shape
        (4, 3)
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
            eigvals_loc, U_y_loc = _etkf_inner_spectrum(obs_anom, R_inv_Y)

            inv_base = jnp.asarray(1.0 / (N_e - 1), dtype=anom.dtype)
            inv_diff = 1.0 / ((N_e - 1) + eigvals_loc) - inv_base
            sqrt_base = jnp.asarray(1.0, dtype=anom.dtype)
            sqrt_diff = jnp.sqrt((N_e - 1) / ((N_e - 1) + eigvals_loc)) - sqrt_base

            # w̄ᵢ = T (Y′ R⁻¹_loc v); T = (1/(Nₑ−1)) I + rank-Nᵧ correction.
            Y_R_inv_v = einx.dot("e y, y -> e", R_inv_Y, innovation)
            w_bar_i = _apply_ctilde_func(U_y_loc, inv_diff, inv_base, Y_R_inv_v)
            # Wᵢ = I + U_y_loc diag(sqrt_diff) U_y_locᵀ — materialise the
            # (Nₑ, Nₑ) transform so the outer ``einx`` can fuse over grid
            # points without re-invoking per-point spectra.
            scaled_U = einx.multiply("e a, a -> e a", U_y_loc, sqrt_diff)
            correction = einx.dot("e a, f a -> e f", scaled_U, U_y_loc)
            W_i = jnp.eye(N_e, dtype=anom.dtype) + correction
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


# ──────────────────────────────────────────────────────────────────────
# Advanced sequential filter variants (Wave 4.A)
# ──────────────────────────────────────────────────────────────────────


def _mean_preserving_rotation(
    key: PRNGKeyArray, n_ensemble: int, dtype: jnp.dtype = jnp.float64
) -> Float[Array, "N_e N_e"]:
    r"""Random orthogonal ``Θ ∈ O(Nₑ)`` with ``Θ 𝟙 = 𝟙`` (Livings 2008).

    Construct an orthogonal basis ``V ∈ ℝ^{Nₑ × (Nₑ−1)}`` for the
    complement of ``𝟙``, draw a random rotation ``Q ∈ O(Nₑ−1)`` via
    the QR-of-Gaussian construction, and lift back to ``Nₑ`` space:

    $$
    \Theta = \tfrac{1}{N_e} \mathbf{1} \mathbf{1}^{\top} + V Q V^{\top}
    $$

    By construction ``Θ 𝟙 = 𝟙`` (the constant vector is the +1
    eigenvector) and ``Θᵀ Θ = I`` (the orthogonal complement is
    preserved). Cost ``O(Nₑ³)`` per draw; negligible compared with the
    ETKF transform that produced ``W_a``.
    """
    # Basis for {𝟙}⊥: take Householder reflector that maps 𝟙/√Nₑ → e₀
    # and keep its last Nₑ−1 columns (orthogonal to the all-ones vector).
    ones = jnp.ones(n_ensemble, dtype=dtype) / jnp.sqrt(n_ensemble)
    e0 = jnp.zeros(n_ensemble, dtype=dtype).at[0].set(1.0)
    v = ones - e0
    v_norm = jnp.linalg.norm(v)
    # Guard against the degenerate Nₑ=1 case (already rejected by
    # check_ensemble_size upstream).
    v = jnp.where(v_norm > 0, v / jnp.maximum(v_norm, 1e-30), v)
    H = jnp.eye(n_ensemble, dtype=dtype) - 2.0 * jnp.outer(v, v)
    V = H[:, 1:]  # (Nₑ, Nₑ−1)
    # Random rotation in the (Nₑ−1)-dim complement: Q = qr(𝒩(0, I))[0].
    g = jr.normal(key, (n_ensemble - 1, n_ensemble - 1), dtype=dtype)
    Q, R = jnp.linalg.qr(g)
    # Make Q sign-canonical so it's drawn uniformly from O(Nₑ−1).
    signs = jnp.sign(jnp.diag(R))
    signs = jnp.where(signs == 0, 1.0, signs)
    Q = Q * signs[None, :]
    # Lift: Θ = projector onto 𝟙 + V Q Vᵀ.
    proj_ones = einx.dot("a, b -> a b", ones, ones)
    return proj_ones + V @ Q @ V.T


class ETKF_Livings(AbstractSequentialFilter, strict=True):
    r"""ETKF with a mean-preserving random rotation (Livings et al. 2008).

    ETKF's symmetric square root ``W_a = √((Nₑ − 1) T)`` is the unique
    PSD- and mean-preserving choice, but it produces a *deterministic*
    transform that can develop preferred directions over many cycles —
    the ensemble loses rank to a small invariant subspace. Livings et
    al. break this symmetry by composing with a random orthogonal
    matrix:

    $$
    W_a^{\mathrm{rot}} = W_a\, \Theta, \qquad
    \Theta \in O(N_e), \quad \Theta \mathbf{1} = \mathbf{1}
    $$

    The mean-preserving constraint ``Θ 𝟙 = 𝟙`` keeps the analysis
    mean and the rank-deficiency-against-𝟙 property; the random factor
    averages out preferred-direction artefacts over time.

    PRNG key handling. The constructor's ``self.key`` is the *default*
    source of randomness; ``analysis(..., key=subkey)`` overrides it
    for a single call. Pattern-match the :class:`StochasticEnKF` API:
    if you call ``analysis`` directly inside a loop, pass a freshly
    split key each cycle or you will get the *same* rotation every
    window — which defeats the whole purpose of the Livings recipe.

    Attributes:
        key: Default PRNG key for the random rotation. Used when no
            explicit ``key=`` kwarg is passed to ``analysis``.
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
        N_e = particles.shape[0]
        check_ensemble_size(N_e)
        rot_key = self.key if key is None else key
        obs_particles = _apply_obs_op(obs_op, particles)

        anom = ensemble_anomalies(particles)
        obs_anom = ensemble_anomalies(obs_particles)
        x_bar = ensemble_mean(particles)
        innovation = obs - ensemble_mean(obs_particles)

        R_inv_Y = gaussx.solve_rows(obs_noise, obs_anom)
        eigvals, U_y = _etkf_inner_spectrum(obs_anom, R_inv_Y)
        inv_base = jnp.asarray(1.0 / (N_e - 1), dtype=anom.dtype)
        inv_diff = 1.0 / ((N_e - 1) + eigvals) - inv_base
        sqrt_base = jnp.asarray(1.0, dtype=anom.dtype)
        sqrt_diff = jnp.sqrt((N_e - 1) / ((N_e - 1) + eigvals)) - sqrt_base

        # ETKF weight + symmetric square root, same as the base ETKF.
        Y_R_inv_v = einx.dot("e y, y -> e", R_inv_Y, innovation)
        w_bar = _apply_ctilde_func(U_y, inv_diff, inv_base, Y_R_inv_v)
        mean_correction = einx.dot("e, e x -> x", w_bar, anom)

        W_anom = _apply_ctilde_func(U_y, sqrt_diff, sqrt_base, anom)  # (Nₑ, Nₓ)

        # Apply the mean-preserving rotation Θ on the *left* of W_a X′:
        # Θ leaves 𝟙 fixed, so the column-mean (== analysis mean) is
        # invariant under the rotation.
        Theta = _mean_preserving_rotation(rot_key, N_e, dtype=anom.dtype)
        pert_correction = einx.dot("i j, j x -> i x", Theta, W_anom)

        particles_a = (
            einx.add("x, x -> x", x_bar, mean_correction)[None, :] + pert_correction
        )
        S = innovation_covariance(obs_particles, obs_noise)
        log_p = log_likelihood(innovation, S)
        return AnalysisResult(particles=particles_a, log_likelihood=log_p)


class EnSRF_Serial(AbstractSequentialFilter, strict=True):
    r"""Serial Ensemble Square Root Filter (Whitaker & Hamill 2002).

    Processes observations **one at a time** with the classical W&H
    scalar reduced-gain formula:

    $$
    \begin{aligned}
    K_k &= C^{x H_k} \big/ \big(C^{H_k H_k} + R_{kk}\big) \\
    \bar{x}_a^{(k)} &= \bar{x}_a^{(k-1)}
        + K_k \big(y_k - H_k \bar{x}_a^{(k-1)}\big) \\
    \alpha_k &= 1 \Big/ \Big(1 + \sqrt{R_{kk} / (C^{H_k H_k} + R_{kk})}\Big) \\
    X'^{(k)}_a &= X'^{(k-1)}_a - \alpha_k K_k \big(H_k X'^{(k-1)}_a\big)
    \end{aligned}
    $$

    No matrix inversion is required — each scalar update is an inner
    product. Cost ``O(Nₑ Nₓ Nᵧ)`` total. Requires **diagonal** ``R``
    (uncorrelated obs); use :class:`ETKF` / :class:`EnSRF` for general
    ``R`` or pre-decorrelate observations.

    The implementation reapplies ``obs_op`` to the running ensemble
    inside each scalar update. That keeps the serial trajectory
    consistent for **nonlinear** ``H`` (where the relationship between
    state-space and obs-space anomalies is updated by ``H`` itself, not
    by linear book-keeping). Cost is ``O(Nᵧ × Nₑ × cost(H))`` —
    fractionally more than the linear-only shortcut but correct for
    every ``H`` filterax supports.
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
        if not isinstance(obs_noise, lx.DiagonalLinearOperator):
            raise NotImplementedError(
                "EnSRF_Serial requires diagonal observation noise; pre-"
                "decorrelate or use filterax.filters.EnSRF for general Γ."
            )
        R_diag = lx.diagonal(obs_noise)
        initial_obs_particles = _apply_obs_op(obs_op, particles)

        def step(parts, k):
            # Re-evaluate H on the *current* ensemble so nonlinear obs
            # ops remain consistent across the serial sweep.
            parts_obs = _apply_obs_op(obs_op, parts)  # (Nₑ, Nᵧ)
            y_k = obs[k]
            R_kk = R_diag[k]
            obs_col = parts_obs[:, k]  # (Nₑ,)
            obs_mean = jnp.mean(obs_col)
            obs_anom_col = obs_col - obs_mean
            obs_var = jnp.sum(obs_anom_col * obs_anom_col) / (N_e - 1)  # Cᴴₖᴴₖ
            innovation_var = obs_var + R_kk

            mean_parts = jnp.mean(parts, axis=0)
            state_anom = parts - mean_parts[None, :]  # (Nₑ, Nₓ)
            C_xH = einx.dot("e, e x -> x", obs_anom_col, state_anom) / (N_e - 1)

            K_k = C_xH / innovation_var  # (Nₓ,)

            # Mean update.
            mean_parts_new = mean_parts + K_k * (y_k - obs_mean)
            # Reduced-gain factor for the perturbation half of the
            # square-root update.
            alpha = 1.0 / (1.0 + jnp.sqrt(R_kk / innovation_var))
            pert_update_state = alpha * einx.dot("x, e -> e x", K_k, obs_anom_col)
            parts_new = mean_parts_new[None, :] + (state_anom - pert_update_state)
            return parts_new, None

        N_y = obs.shape[0]
        particles_a, _ = jax.lax.scan(step, particles, jnp.arange(N_y))

        S = innovation_covariance(initial_obs_particles, obs_noise)
        log_p = log_likelihood(obs - ensemble_mean(initial_obs_particles), S)
        return AnalysisResult(particles=particles_a, log_likelihood=log_p)


class ESTKF(AbstractSequentialFilter, strict=True):
    r"""Error Subspace Transform Kalman Filter (Nerger et al. 2012).

    Projects ETKF into the ``(Nₑ − 1)``-dimensional error subspace via
    a mean-preserving orthogonal map ``L ∈ ℝ^{Nₑ × (Nₑ − 1)}`` with
    ``Lᵀ L = I_{Nₑ − 1}`` and ``Lᵀ 𝟙 = 0``. Anomalies are rank
    ``≤ Nₑ − 1`` to begin with (their rows sum to zero), so ``L``
    captures them losslessly.

    Reduced anomalies and transform precision:

    $$
    \tilde{X} = L^{\top} X' \in \mathbb{R}^{(N_e-1) \times N_x}, \qquad
    \tilde{Y} = L^{\top} Y' \in \mathbb{R}^{(N_e-1) \times N_y}
    $$

    $$
    A = (N_e - 1) I + \tilde{Y} R^{-1} \tilde{Y}^{\top}
        \in \mathbb{R}^{(N_e-1) \times (N_e-1)}
    $$

    Eigendecompose ``A = U Λ Uᵀ``:

    $$
    \tilde{w} = U \Lambda^{-1} U^{\top} \tilde{Y} R^{-1} d, \qquad
    \tilde{W} = U \sqrt{(N_e - 1) \Lambda^{-1}}\, U^{\top}
    $$

    Lift back to the full ensemble:

    $$
    \bar{x}_a = \bar{x}_f + \tilde{w}^{\top} \tilde{X}, \qquad
    X_a = \bar{x}_a \mathbf{1}^{\top} + L \tilde{W} \tilde{X}
    $$

    Mean-preserving by construction; PSD analysis covariance;
    eigendecomposition is ``(Nₑ − 1)³`` rather than ``Nₑ³``.
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

        # Mean-preserving projection L ∈ ℝ^{Nₑ × (Nₑ−1)}: Householder
        # reflector mapping 𝟙/√Nₑ → e₀, then drop the first column.
        dtype = anom.dtype
        ones = jnp.ones(N_e, dtype=dtype) / jnp.sqrt(N_e)
        e0 = jnp.zeros(N_e, dtype=dtype).at[0].set(1.0)
        v = ones - e0
        v_norm = jnp.linalg.norm(v)
        v = jnp.where(v_norm > 0, v / jnp.maximum(v_norm, 1e-30), v)
        H_mat = jnp.eye(N_e, dtype=dtype) - 2.0 * jnp.outer(v, v)
        L = H_mat[:, 1:]  # (Nₑ, Nₑ−1)

        # Reduce both anomaly matrices to (Nₑ−1) ensemble coordinates.
        X_tilde = einx.dot("e r, e x -> r x", L, anom)  # (Nₑ−1, Nₓ)
        Y_tilde = einx.dot("e r, e y -> r y", L, obs_anom)  # (Nₑ−1, Nᵧ)

        # Rank-Nᵧ spectrum of Ỹ R⁻¹ Ỹᵀ in the reduced subspace; same
        # rank-deficiency story as ETKF, so the QR + small-eigh form
        # keeps gradients finite.
        R_inv_Y = gaussx.solve_rows(obs_noise, Y_tilde)  # (Nₑ−1, Nᵧ)
        eigvals, U_y = _etkf_inner_spectrum(Y_tilde, R_inv_Y)

        inv_base = jnp.asarray(1.0 / (N_e - 1), dtype=dtype)
        inv_diff = 1.0 / ((N_e - 1) + eigvals) - inv_base
        sqrt_base = jnp.asarray(1.0, dtype=dtype)
        sqrt_diff = jnp.sqrt((N_e - 1) / ((N_e - 1) + eigvals)) - sqrt_base

        # Reduced weights w̃ = T̃ (Ỹ R⁻¹ d)  ∈ ℝ^{Nₑ−1}.
        rhs = einx.dot("r y, y -> r", R_inv_Y, innovation)
        w_tilde = _apply_ctilde_func(U_y, inv_diff, inv_base, rhs)

        # Symmetric square-root applied to X̃: W̃ X̃ = (I + correction) X̃.
        WX_tilde = _apply_ctilde_func(U_y, sqrt_diff, sqrt_base, X_tilde)

        # Lift back to Nₑ space and assemble the analysis.
        mean_shift = einx.dot("r, r x -> x", w_tilde, X_tilde)  # (Nₓ,)
        pert_full = einx.dot("e r, r x -> e x", L, WX_tilde)  # (Nₑ, Nₓ)

        particles_a = (x_bar + mean_shift)[None, :] + pert_full

        S = innovation_covariance(obs_particles, obs_noise)
        log_p = log_likelihood(innovation, S)
        return AnalysisResult(particles=particles_a, log_likelihood=log_p)

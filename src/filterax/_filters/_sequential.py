"""Sequential filters: one analysis step from a forecast ensemble to a posterior.

Each filter implements `AbstractSequentialFilter.analysis`, from a forecast ensemble X,
an observation y, an observation operator h and an observation-error covariance Σ_ε,
and reports the log-likelihood log 𝒩(r̄; 0, Σ̂_y) of the mean residual r̄ = y − h̄
alongside the analysis. The forecast and the inflation live in the Layer-2 models of
`filterax._filters._models`.

The filters share the sample statistics of `filterax._primitives._statistics` and the
ensemble Kalman divisor N_ℰ − 1, and differ only in how the ensemble is moved: by the
perturbed-observation gain of `gaussx.enkf_analysis`, by the ensemble-space transform
of `gaussx.etkf_analysis` and `gaussx.etkf_transform`, or one observation at a time.
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
from filterax._primitives._likelihood import ensemble_log_likelihood
from filterax._primitives._localization import gaspari_cohn
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
    """Map every member to observation space, h⁽ᵉ⁾ = h(x⁽ᵉ⁾), at N_ℰ calls of h.

    Args:
        obs_op: Observation operator h, applied to one state.
        particles: Ensemble members x⁽ᵉ⁾, one row per member.

    Returns:
        The mapped members h⁽ᵉ⁾, one row per member.
    """
    # h⁽ᵉ⁾ = h(x⁽ᵉ⁾)
    return jax.vmap(obs_op)(particles)


def _innovation_log_likelihood(
    obs_particles: Float[Array, "N_e N_y"],
    obs: Float[Array, " N_y"],
    obs_noise: lx.AbstractLinearOperator,
) -> Float[Array, ""]:
    """Log-likelihood of the mean residual that every analysis reports.

        log 𝒩(r̄; 0, Σ̂_y),   r̄ = y − h̄,   Σ̂_y = Σ̂_h + Σ_ε

    Args:
        obs_particles: Forecast members h⁽ᵉ⁾ in observation space.
        obs: Observation y.
        obs_noise: Observation-error covariance Σ_ε.

    Returns:
        The scalar log-likelihood.
    """
    # h̄ = (1 / N_ℰ) ∑ₑ h⁽ᵉ⁾
    obs_mean = ensemble_mean(obs_particles)
    # r̄ = y − h̄
    residual = obs - obs_mean
    # log 𝒩(r̄; 0, Σ̂_h + Σ_ε)
    return ensemble_log_likelihood(obs_particles, residual, obs_noise)


def _etkf_analysis(
    particles: Float[Array, "N_e N_x"],
    obs: Float[Array, " N_y"],
    obs_op: AbstractObsOperator | ObsCallable,
    obs_noise: lx.AbstractLinearOperator,
) -> AnalysisResult:
    """The batch ETKF analysis that `ETKF`, `EnSRF` and `ESTKF` all compute.

        x̄_t|t = x̄_t|t−1 + X̃ᵀ w̄,   X̃_t|t = W X̃

    Args:
        particles: Forecast ensemble x⁽ᵉ⁾_t|t−1, one row per member.
        obs: Observation y.
        obs_op: Observation operator h, applied to one state.
        obs_noise: Observation-error covariance Σ_ε.

    Returns:
        The analysis ensemble and the log-likelihood of r̄.
    """
    # h⁽ᵉ⁾ = h(x⁽ᵉ⁾)
    obs_particles = _apply_obs_op(obs_op, particles)
    # x⁽ᵉ⁾_t|t = x̄_t|t−1 + X̃ᵀ w̄ + (W X̃)⁽ᵉ⁾
    analysis_particles = gaussx.etkf_analysis(particles, obs_particles, obs, obs_noise)
    # log 𝒩(r̄; 0, Σ̂_y)
    log_likelihood = _innovation_log_likelihood(obs_particles, obs, obs_noise)
    return AnalysisResult(particles=analysis_particles, log_likelihood=log_likelihood)


class StochasticEnKF(AbstractSequentialFilter, strict=True):
    """Stochastic ensemble Kalman filter, with perturbed observations.

        x⁽ᵉ⁾_t|t = x⁽ᵉ⁾_t|t−1 + G (y + ε⁽ᵉ⁾ − h⁽ᵉ⁾),   ε⁽ᵉ⁾ ~ 𝒩(0, Σ_ε)
        G = Σ̂_xy Σ̂_y⁻¹,   Σ̂_y = Σ̂_h + Σ_ε

    Each member sees its own draw of the observation noise, which keeps the analysis
    spread right at the cost of Monte Carlo noise of order 1 / √N_ℰ. The update is
    `gaussx.enkf_analysis` with the divisor N_ℰ − 1; the filter adds the map to
    observation space, the choice of key and the log-likelihood of r̄.

    The stored ``key`` is only the default. ``analysis(..., key=step_key)`` overrides
    it for one call, which is how the Layer-2 `filterax.StochasticEnKF` draws fresh
    perturbations in every window. Calling ``analysis`` yourself without a fresh key
    repeats the same perturbations in every window.

    Attributes:
        key: Default PRNG key, used when ``analysis`` gets no ``key``. An integer is
            turned into ``jr.PRNGKey(key)``.

    Examples:
        Three members, both components observed:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> from filterax.filters import StochasticEnKF
        >>> particles = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        >>> obs_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> filter_ = StochasticEnKF(key=jr.key(0))
        >>> result = filter_.analysis(
        ...     particles, jnp.array([1.0, 0.5]), lambda x: x, obs_noise, key=jr.key(1)
        ... )
        >>> result.particles.shape
        (3, 2)

        The same key repeats the same draw, and another key moves it:

        >>> again = filter_.analysis(
        ...     particles, jnp.array([1.0, 0.5]), lambda x: x, obs_noise, key=jr.key(1)
        ... )
        >>> bool(jnp.array_equal(result.particles, again.particles))
        True
        >>> other = filter_.analysis(
        ...     particles, jnp.array([1.0, 0.5]), lambda x: x, obs_noise, key=jr.key(2)
        ... )
        >>> bool(jnp.array_equal(result.particles, other.particles))
        False

    References:
        Evensen (1994), *J. Geophys. Res.* 99(C5), 10143–10162.
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
        """Stochastic EnKF analysis step.

        Algorithm:
            1. Map the members to observation space, h⁽ᵉ⁾ = h(x⁽ᵉ⁾).
            2. Draw ε⁽ᵉ⁾ ~ 𝒩(0, Σ_ε) and add G (y + ε⁽ᵉ⁾ − h⁽ᵉ⁾) to each member.
            3. Score r̄ under 𝒩(0, Σ̂_y).

        Args:
            particles: Forecast ensemble x⁽ᵉ⁾_t|t−1, one row per member.
            obs: Observation y.
            obs_op: Observation operator h, applied to one state.
            obs_noise: Observation-error covariance Σ_ε.
            key: PRNG key for this step's perturbations; ``None`` uses ``self.key``,
                which repeats the same draw across calls.

        Returns:
            The analysis ensemble and the log-likelihood log 𝒩(r̄; 0, Σ̂_y).

        Raises:
            ValueError: From gaussx, if N_ℰ < 2, or ``obs`` or ``obs_noise`` does not
                match D_y.
        """
        step_key = self.key if key is None else key
        # h⁽ᵉ⁾ = h(x⁽ᵉ⁾)
        obs_particles = _apply_obs_op(obs_op, particles)
        # x⁽ᵉ⁾_t|t = x⁽ᵉ⁾_t|t−1 + G (y + ε⁽ᵉ⁾ − h⁽ᵉ⁾),   ε⁽ᵉ⁾ ~ 𝒩(0, Σ_ε)
        analysis_particles = gaussx.enkf_analysis(
            particles, obs_particles, obs, obs_noise, key=step_key
        )
        # log 𝒩(r̄; 0, Σ̂_y)
        log_likelihood = _innovation_log_likelihood(obs_particles, obs, obs_noise)
        return AnalysisResult(
            particles=analysis_particles, log_likelihood=log_likelihood
        )


class ETKF(AbstractSequentialFilter, strict=True):
    """Ensemble transform Kalman filter, a deterministic square-root analysis.

        Λ_w = (N_ℰ − 1) I + H̃ Σ_ε⁻¹ H̃ᵀ,   Σ_w = Λ_w⁻¹
        w̄ = Σ_w H̃ Σ_ε⁻¹ r̄,   W = ((N_ℰ − 1) Σ_w)^½
        x̄_t|t = x̄_t|t−1 + X̃ᵀ w̄,   X̃_t|t = W X̃

    The update works in the N_ℰ-dimensional space of the members, with the symmetric
    square root W, which keeps the analysis mean exact. It is `gaussx.etkf_analysis`,
    and the filter adds the map to observation space and the log-likelihood of r̄.

    With D_y < N_ℰ gaussx takes a thin QR of H̃ and a (D_y, D_y) eigendecomposition, at
    O(N_ℰ D_y² + N_ℰ D_y D_x), and forms no (N_ℰ, N_ℰ) matrix. Otherwise it
    eigendecomposes Λ_w at O(N_ℰ² D_y + N_ℰ³). Both routes have closed-form
    derivatives that stay finite at repeated eigenvalues, and Σ_ε⁻¹ goes through
    gaussx's solves, so a structured Σ_ε is never densified.

    Examples:
        Three members, both components observed:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from filterax.filters import ETKF
        >>> particles = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        >>> obs_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> result = ETKF().analysis(
        ...     particles, jnp.array([1.0, 0.5]), lambda x: x, obs_noise
        ... )
        >>> result.particles.shape
        (3, 2)

        The analysis anomalies still sum to zero:

        >>> anomalies = result.particles - result.particles.mean(0)
        >>> bool(jnp.allclose(anomalies.sum(0), 0.0, atol=1e-5))
        True

    References:
        Bishop, Etherton & Majumdar (2001), *Mon. Wea. Rev.* 129, 420–436.
    """

    def analysis(
        self,
        particles: Float[Array, "N_e N_x"],
        obs: Float[Array, " N_y"],
        obs_op: AbstractObsOperator | ObsCallable,
        obs_noise: lx.AbstractLinearOperator,
        **_: Any,
    ) -> AnalysisResult:
        """ETKF analysis step.

        Algorithm:
            1. Map the members to observation space, h⁽ᵉ⁾ = h(x⁽ᵉ⁾).
            2. Form w̄ and W in ensemble space and apply them to X̃.
            3. Score r̄ under 𝒩(0, Σ̂_y).

        Args:
            particles: Forecast ensemble x⁽ᵉ⁾_t|t−1, one row per member.
            obs: Observation y.
            obs_op: Observation operator h, applied to one state.
            obs_noise: Observation-error covariance Σ_ε.

        Returns:
            The analysis ensemble and the log-likelihood log 𝒩(r̄; 0, Σ̂_y).

        Raises:
            ValueError: From gaussx, if N_ℰ < 2, or ``obs`` or ``obs_noise`` does not
                match D_y.
        """
        return _etkf_analysis(particles, obs, obs_op, obs_noise)


class EnSRF(AbstractSequentialFilter, strict=True):
    """Ensemble square-root filter in batch form, which is the ETKF update.

        x̄_t|t = x̄_t|t−1 + G (y − h̄),   X̃_t|t = W X̃

    The mean moves by the gain, G r̄ = X̃ᵀ w̄, and the anomalies by a square root that
    gives the analysis covariance (I − G H) Σ̂_x of the linear-Gaussian case without
    perturbed observations. With the members as rows, the symmetric square root W of
    the ETKF is the one that is also mean-preserving (Tippett et al. 2003), so in batch
    form this class runs exactly the analysis of `ETKF`, through
    `gaussx.etkf_analysis`, and keeps the name for those who think of it this way.

    The one-sided reduced gain of Whitaker & Hamill (2002) only differs when the
    observations are assimilated one at a time; that variant is `EnSRF_Serial`. The
    cost is that of `ETKF`.

    Examples:
        The same analysis as `ETKF`:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from filterax.filters import ETKF, EnSRF
        >>> particles = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        >>> obs_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> result = EnSRF().analysis(
        ...     particles, jnp.array([1.0, 0.5]), lambda x: x, obs_noise
        ... )
        >>> result.particles.shape
        (3, 2)
        >>> etkf = ETKF().analysis(
        ...     particles, jnp.array([1.0, 0.5]), lambda x: x, obs_noise
        ... )
        >>> bool(jnp.array_equal(result.particles, etkf.particles))
        True

    References:
        Whitaker & Hamill (2002), *Mon. Wea. Rev.* 130, 1913–1924.
        Tippett et al. (2003), *Mon. Wea. Rev.* 131, 1485–1490.
    """

    def analysis(
        self,
        particles: Float[Array, "N_e N_x"],
        obs: Float[Array, " N_y"],
        obs_op: AbstractObsOperator | ObsCallable,
        obs_noise: lx.AbstractLinearOperator,
        **_: Any,
    ) -> AnalysisResult:
        """EnSRF analysis step, the ETKF update.

        Algorithm:
            1. Map the members to observation space, h⁽ᵉ⁾ = h(x⁽ᵉ⁾).
            2. Form w̄ and W in ensemble space and apply them to X̃.
            3. Score r̄ under 𝒩(0, Σ̂_y).

        Args:
            particles: Forecast ensemble x⁽ᵉ⁾_t|t−1, one row per member.
            obs: Observation y.
            obs_op: Observation operator h, applied to one state.
            obs_noise: Observation-error covariance Σ_ε.

        Returns:
            The analysis ensemble and the log-likelihood log 𝒩(r̄; 0, Σ̂_y).

        Raises:
            ValueError: From gaussx, if N_ℰ < 2, or ``obs`` or ``obs_noise`` does not
                match D_y.
        """
        return _etkf_analysis(particles, obs, obs_op, obs_noise)


class LETKF(AbstractSequentialFilter, strict=True):
    """Local ensemble transform Kalman filter, with R-localization.

        Λ_w,i = (N_ℰ − 1) I + H̃ D_ρ,i^½ Σ_ε⁻¹ D_ρ,i^½ H̃ᵀ,   D_ρ,i = diag(ρ_i)
        x̄_t|t,i = x̄_i + X̃_iᵀ w̄_i,   X̃_t|t,i = W_i X̃_i

    An independent ETKF analysis at each state component i, with its own weights w̄_i
    and transform W_i, from only the observations near it. Component i sits at
    s_i and observation k at s_k, at the distance d_ik = ‖s_i − s_k‖₂, and the
    observation enters with the weight

        ρ_ik = ρ(d_ik; ℓ) if d_ik ≤ ℓ, else 0

    from the taper ρ, the Gaspari-Cohn function by default. The cutoff at ℓ is explicit,
    because Gaspari-Cohn reaches out to 2 ℓ and would otherwise keep every observation
    with a nonzero taper. A weight divides the variance, ρ_ik / Σ_ε,kk.

    The weights at each i are `gaussx.etkf_transform` with ``obs_weights=ρ_i``, mapped
    over the components with `jax.vmap`. The filter adds the distances, the taper and
    its cutoff, and applies the local weights to the global anomalies X̃ column by
    column. R-localization needs a diagonal Σ_ε (Hunt et al. 2007, §2), which is read
    with `lineax.diagonal` and never densified.

    Attributes:
        radius: Localization half-width ℓ; observations farther than ℓ are dropped.
        taper_fn: Taper ρ, called as ``taper_fn(distances, radius)``; defaults to
            `filterax.gaspari_cohn`.

    Examples:
        Three grid points on a line, the outer two observed:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from filterax.filters import LETKF
        >>> particles = jax.random.normal(jax.random.key(0), (4, 3))
        >>> state_coords = jnp.array([[0.0], [1.0], [2.0]])
        >>> obs_coords = jnp.array([[0.0], [2.0]])
        >>> obs_noise = lx.DiagonalLinearOperator(0.5 * jnp.ones(2))
        >>> result = LETKF(radius=1.5).analysis(
        ...     particles,
        ...     jnp.array([0.1, -0.2]),
        ...     lambda x: x[::2],
        ...     obs_noise,
        ...     state_coords=state_coords,
        ...     obs_coords=obs_coords,
        ... )
        >>> result.particles.shape
        (4, 3)

        A point with no observation within ℓ keeps its forecast:

        >>> far = LETKF(radius=0.5).analysis(
        ...     particles,
        ...     jnp.array([0.1, -0.2]),
        ...     lambda x: x[::2],
        ...     obs_noise,
        ...     state_coords=state_coords,
        ...     obs_coords=obs_coords,
        ... )
        >>> bool(jnp.allclose(far.particles[:, 1], particles[:, 1], atol=1e-5))
        True

        The coordinates are required:

        >>> LETKF(radius=1.5).analysis(
        ...     particles, jnp.array([0.1, -0.2]), lambda x: x[::2], obs_noise
        ... )  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        ValueError: LETKF.analysis requires state_coords and obs_coords keyword
        arguments — pass them through the L2 model or the call site.

    References:
        Hunt, Kostelich & Szunyogh (2007), *Physica D* 230, 112–126.
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
        """LETKF analysis step.

        Algorithm:
            1. Check the coordinates and that Σ_ε is diagonal.
            2. Map the members to observation space and take the state anomalies X̃.
            3. At each component i, taper the observations by distance and form the
               local weights w̄_i and transform W_i.
            4. Apply them to the anomalies of component i and add its mean back.
            5. Score r̄ under 𝒩(0, Σ̂_y), unlocalized.

        Args:
            particles: Forecast ensemble x⁽ᵉ⁾_t|t−1, one row per member.
            obs: Observation y.
            obs_op: Observation operator h, applied to one state.
            obs_noise: Diagonal observation-error covariance Σ_ε.
            state_coords: Coordinates s_i of the state components, one row each.
            obs_coords: Coordinates s_k of the observations, one row each.

        Returns:
            The analysis ensemble and the log-likelihood log 𝒩(r̄; 0, Σ̂_y).

        Raises:
            ValueError: If N_ℰ < 2, if a coordinate array is missing, or if it does
                not have one row per state component or per observation.
            NotImplementedError: If ``obs_noise`` is not a
                `lineax.DiagonalLinearOperator`.
        """
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

        # h⁽ᵉ⁾ = h(x⁽ᵉ⁾)
        obs_particles = _apply_obs_op(obs_op, particles)
        # x̃⁽ᵉ⁾ = x⁽ᵉ⁾ − x̄
        anomalies = ensemble_anomalies(particles)
        # x̄ = (1 / N_ℰ) ∑ₑ x⁽ᵉ⁾
        mean = ensemble_mean(particles)
        radius = self.radius

        def local_weights(
            state_point: Float[Array, " D"],
        ) -> tuple[Float[Array, " N_e"], Float[Array, "N_e N_e"]]:
            """Local weights w̄_i and transform W_i at one state component."""
            # s_k − s_i
            offsets = einx.subtract("y d, d -> y d", obs_coords, state_point)
            # (s_k − s_i)²
            squared_offsets = offsets * offsets
            # d_ik² = ‖s_i − s_k‖₂²
            squared_distances = einx.sum("y d -> y", squared_offsets)
            # d_ik = ‖s_i − s_k‖₂
            distances = jnp.sqrt(squared_distances)
            # the hard cutoff at ℓ, inside Gaspari-Cohn's support of 2 ℓ
            within = distances <= radius
            # ρ(d_ik; ℓ)
            tapered = self.taper_fn(distances, radius)
            # ρ_ik = ρ(d_ik; ℓ) if d_ik ≤ ℓ, else 0
            taper = jnp.where(within, tapered, 0.0)
            # w̄_i, W_i with Σ_ε⁻¹ ← D_ρ,i^½ Σ_ε⁻¹ D_ρ,i^½
            return gaussx.etkf_transform(
                obs_particles, obs, obs_noise, obs_weights=taper
            )

        # (w̄_i, W_i) for every component i
        mean_weights, transforms = jax.vmap(local_weights)(state_coords)

        # X̃_iᵀ w̄_i = ∑ₑ w̄_i,e x̃⁽ᵉ⁾_i
        mean_increment = einx.dot("x e, e x -> x", mean_weights, anomalies)
        # (W_i X̃_i)⁽ᵉ⁾ = ∑_e′ W_i,ee′ x̃⁽ᵉ′⁾_i
        analysis_anomalies = einx.dot(
            "x member other, other x -> member x", transforms, anomalies
        )

        # x̄_t|t,i = x̄_i + X̃_iᵀ w̄_i
        analysis_mean = einx.add("x, x -> x", mean, mean_increment)
        # x⁽ᵉ⁾_t|t = x̄_t|t + x̃⁽ᵉ⁾_t|t
        analysis_particles = einx.add(
            "x, e x -> e x", analysis_mean, analysis_anomalies
        )

        # log 𝒩(r̄; 0, Σ̂_y)
        log_likelihood = _innovation_log_likelihood(obs_particles, obs, obs_noise)
        return AnalysisResult(
            particles=analysis_particles, log_likelihood=log_likelihood
        )


def _mean_preserving_rotation(
    key: PRNGKeyArray, n_ensemble: int, dtype: jnp.dtype = jnp.float64
) -> Float[Array, "N_e N_e"]:
    """Random orthogonal Q with Q 1 = 1, drawn uniformly (Livings et al. 2008).

        Q = (1 / N_ℰ) 1 1ᵀ + V Q_⊥ Vᵀ

    V holds an orthonormal basis of the complement of 1 as columns, and Q_⊥ is a random
    rotation of that (N_ℰ − 1)-dimensional complement. Then Q 1 = 1, so 1 is an
    eigenvector with eigenvalue 1, and Qᵀ Q = I. A draw costs O(N_ℰ³), small next to
    the transform it rotates.

    Algorithm:
        1. Take V from the Householder reflection that maps 1 / √N_ℰ to the first unit
           vector, dropping its first column.
        2. Draw Q_⊥ as the sign-fixed QR factor of a Gaussian matrix.
        3. Lift it, Q = (1 / N_ℰ) 1 1ᵀ + V Q_⊥ Vᵀ.

    Args:
        key: PRNG key for Q_⊥.
        n_ensemble: Ensemble size N_ℰ.
        dtype: Dtype of Q.

    Returns:
        The rotation Q.
    """
    # u = 1 / √N_ℰ
    ones = jnp.ones(n_ensemble, dtype=dtype) / jnp.sqrt(n_ensemble)
    # e₀
    first_axis = jnp.zeros(n_ensemble, dtype=dtype).at[0].set(1.0)
    # v = u − e₀
    reflection_axis = ones - first_axis
    # ‖v‖₂
    reflection_norm = jnp.linalg.norm(reflection_axis)
    # v ← v / ‖v‖₂; v = 0 only at N_ℰ = 1, which callers reject
    reflection_axis = jnp.where(
        reflection_norm > 0,
        reflection_axis / jnp.maximum(reflection_norm, 1e-30),
        reflection_axis,
    )
    # P = I − 2 v vᵀ, with P u = e₀
    householder = jnp.eye(n_ensemble, dtype=dtype) - 2.0 * jnp.outer(
        reflection_axis, reflection_axis
    )
    # V: the last N_ℰ − 1 columns of P, orthogonal to 1
    complement_basis = householder[:, 1:]
    # Q_⊥ R_⊥ = qr(𝒩(0, I))
    gaussian = jr.normal(key, (n_ensemble - 1, n_ensemble - 1), dtype=dtype)
    rotation, triangular = jnp.linalg.qr(gaussian)
    # sign(diag R_⊥) makes Q_⊥ uniform on the orthogonal group
    signs = jnp.sign(jnp.diag(triangular))
    signs = jnp.where(signs == 0, 1.0, signs)
    rotation = rotation * signs[None, :]
    # (1 / N_ℰ) 1 1ᵀ
    ones_projector = einx.dot("a, b -> a b", ones, ones)
    # V Q_⊥
    rotated_basis = complement_basis @ rotation
    # V Q_⊥ Vᵀ
    complement_rotation = rotated_basis @ complement_basis.T
    # Q = (1 / N_ℰ) 1 1ᵀ + V Q_⊥ Vᵀ
    return ones_projector + complement_rotation


class ETKF_Livings(AbstractSequentialFilter, strict=True):
    """ETKF followed by a mean-preserving random rotation (Livings et al. 2008).

        X̃_t|t = Q W X̃,   Qᵀ Q = I,   Q 1 = 1

    The symmetric square root W of `ETKF` is deterministic, and over many cycles it can
    develop preferred directions, so that the ensemble collapses onto a small invariant
    subspace. A random rotation Q of the analysis anomalies breaks that symmetry. Since
    Q 1 = 1 implies Qᵀ 1 = 1, the anomalies still sum to zero and the analysis mean is
    unchanged. The analysis is `gaussx.etkf_analysis`, and the filter adds the
    rotation.

    The stored ``key`` is only the default, as for `StochasticEnKF`:
    ``analysis(..., key=step_key)`` overrides it for one call. Calling ``analysis`` in
    a loop without a fresh key repeats the same rotation in every window, which defeats
    its purpose.

    Attributes:
        key: Default PRNG key for Q, used when ``analysis`` gets no ``key``. An integer
            is turned into ``jr.PRNGKey(key)``.

    Examples:
        The rotation keeps the ETKF mean:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> from filterax.filters import ETKF, ETKF_Livings
        >>> particles = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [1.0, 2.0]])
        >>> obs_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> result = ETKF_Livings(key=0).analysis(
        ...     particles, jnp.array([1.0, 0.5]), lambda x: x, obs_noise, key=jr.key(3)
        ... )
        >>> result.particles.shape
        (4, 2)
        >>> etkf = ETKF().analysis(
        ...     particles, jnp.array([1.0, 0.5]), lambda x: x, obs_noise
        ... )
        >>> mean, etkf_mean = result.particles.mean(0), etkf.particles.mean(0)
        >>> bool(jnp.allclose(mean, etkf_mean, atol=1e-5))
        True

    References:
        Livings, Dance & Nichols (2008), *Physica D* 237, 1021–1028.
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
        """ETKF analysis step with a random rotation of the anomalies.

        Algorithm:
            1. Map the members to observation space and run the ETKF analysis.
            2. Split the analysis into its mean and anomalies.
            3. Draw Q and rotate the anomalies, then add the mean back.
            4. Score r̄ under 𝒩(0, Σ̂_y).

        Args:
            particles: Forecast ensemble x⁽ᵉ⁾_t|t−1, one row per member.
            obs: Observation y.
            obs_op: Observation operator h, applied to one state.
            obs_noise: Observation-error covariance Σ_ε.
            key: PRNG key for this step's rotation; ``None`` uses ``self.key``, which
                repeats the same rotation across calls.

        Returns:
            The analysis ensemble and the log-likelihood log 𝒩(r̄; 0, Σ̂_y).

        Raises:
            ValueError: From gaussx, if N_ℰ < 2, or ``obs`` or ``obs_noise`` does not
                match D_y.
        """
        rotation_key = self.key if key is None else key
        # h⁽ᵉ⁾ = h(x⁽ᵉ⁾)
        obs_particles = _apply_obs_op(obs_op, particles)
        # x̄_t|t = x̄_t|t−1 + X̃ᵀ w̄,   X̃_t|t = W X̃
        analysis = gaussx.etkf_analysis(particles, obs_particles, obs, obs_noise)
        # x̄_t|t
        analysis_mean = ensemble_mean(analysis)
        # X̃_t|t
        analysis_anomalies = ensemble_anomalies(analysis)

        # Q, with Qᵀ 1 = 1, so the rotated anomalies still sum to zero
        rotation = _mean_preserving_rotation(
            rotation_key, particles.shape[0], dtype=analysis_anomalies.dtype
        )
        # X̃_t|t ← Q X̃_t|t
        rotated_anomalies = einx.dot(
            "member other, other x -> member x", rotation, analysis_anomalies
        )

        # x⁽ᵉ⁾_t|t = x̄_t|t + x̃⁽ᵉ⁾_t|t
        analysis_particles = einx.add("x, e x -> e x", analysis_mean, rotated_anomalies)
        # log 𝒩(r̄; 0, Σ̂_y)
        log_likelihood = _innovation_log_likelihood(obs_particles, obs, obs_noise)
        return AnalysisResult(
            particles=analysis_particles, log_likelihood=log_likelihood
        )


class EnSRF_Serial(AbstractSequentialFilter, strict=True):
    """Serial ensemble square-root filter, one observation at a time.

        g_k = Σ̂_xy,k / (Σ̂_h,kk + Σ_ε,kk)
        x̄ ← x̄ + g_k (y_k − h̄_k)
        α_k = 1 / (1 + √(Σ_ε,kk / (Σ̂_h,kk + Σ_ε,kk)))
        x̃⁽ᵉ⁾ ← x̃⁽ᵉ⁾ − α_k g_k h̃⁽ᵉ⁾_k

    Each observation k is a scalar update with the gain column g_k and the reduced gain
    α_k g_k for the anomalies (Whitaker & Hamill 2002), so there is no matrix inverse,
    only inner products. The observations must be uncorrelated, so Σ_ε must be
    diagonal; use `ETKF` or `EnSRF` for a general Σ_ε, or decorrelate first. gaussx has
    no serial update, so this filter is filterax's own.

    h is reapplied to the current ensemble before every scalar update, so the sweep
    stays consistent for a nonlinear h, where the observation-space anomalies cannot
    be updated by linear bookkeeping. That costs D_y N_ℰ calls of h on top of the
    O(N_ℰ D_x D_y) of the updates. The log-likelihood is of the forecast, before the
    sweep.

    Examples:
        One observation agrees with `ETKF`:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from filterax.filters import ETKF, EnSRF_Serial
        >>> particles = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        >>> obs_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(1))
        >>> result = EnSRF_Serial().analysis(
        ...     particles, jnp.array([1.5]), lambda x: x[:1], obs_noise
        ... )
        >>> result.particles.shape
        (3, 2)
        >>> etkf = ETKF().analysis(
        ...     particles, jnp.array([1.5]), lambda x: x[:1], obs_noise
        ... )
        >>> bool(jnp.allclose(result.particles, etkf.particles, atol=1e-5))
        True

        A correlated Σ_ε is rejected:

        >>> dense = lx.MatrixLinearOperator(jnp.eye(1))
        >>> EnSRF_Serial().analysis(
        ...     particles, jnp.array([1.5]), lambda x: x[:1], dense
        ... )  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        NotImplementedError: EnSRF_Serial requires diagonal observation noise;
        pre-decorrelate or use filterax.filters.EnSRF for general Γ.

    References:
        Whitaker & Hamill (2002), *Mon. Wea. Rev.* 130, 1913–1924.
    """

    def analysis(
        self,
        particles: Float[Array, "N_e N_x"],
        obs: Float[Array, " N_y"],
        obs_op: AbstractObsOperator | ObsCallable,
        obs_noise: lx.AbstractLinearOperator,
        **_: Any,
    ) -> AnalysisResult:
        """Serial EnSRF analysis step.

        Algorithm:
            1. Check that Σ_ε is diagonal and read its variances Σ_ε,kk.
            2. For each observation k in turn, map the current members, form g_k and
               α_k, and update the mean and the anomalies.
            3. Score the forecast's r̄ under 𝒩(0, Σ̂_y).

        Args:
            particles: Forecast ensemble x⁽ᵉ⁾_t|t−1, one row per member.
            obs: Observation y.
            obs_op: Observation operator h, applied to one state.
            obs_noise: Diagonal observation-error covariance Σ_ε.

        Returns:
            The analysis ensemble and the log-likelihood log 𝒩(r̄; 0, Σ̂_y) of the
            forecast.

        Raises:
            ValueError: If N_ℰ < 2.
            NotImplementedError: If ``obs_noise`` is not a
                `lineax.DiagonalLinearOperator`.
        """
        N_e = particles.shape[0]
        check_ensemble_size(N_e)
        if not isinstance(obs_noise, lx.DiagonalLinearOperator):
            raise NotImplementedError(
                "EnSRF_Serial requires diagonal observation noise; pre-"
                "decorrelate or use filterax.filters.EnSRF for general Γ."
            )
        # Σ_ε,kk
        obs_variances = lx.diagonal(obs_noise)
        # h⁽ᵉ⁾ = h(x⁽ᵉ⁾) of the forecast, for the log-likelihood
        initial_obs_particles = _apply_obs_op(obs_op, particles)

        # jnp, not einx, in the step: einx raises AxisSizeError on D_x = 0 with
        # another message than today's
        # the carry stays named parts: JAX quotes its name in a carry dtype error
        def assimilate_one(parts, k):
            # h⁽ᵉ⁾ = h(x⁽ᵉ⁾) of the current members, so a nonlinear h stays consistent
            current_obs = _apply_obs_op(obs_op, parts)
            obs_k = obs[k]
            obs_variance_k = obs_variances[k]
            # h⁽ᵉ⁾_k
            obs_column = current_obs[:, k]
            # h̄_k = (1 / N_ℰ) ∑ₑ h⁽ᵉ⁾_k
            obs_mean = jnp.mean(obs_column)
            # h̃⁽ᵉ⁾_k = h⁽ᵉ⁾_k − h̄_k
            obs_anomaly = obs_column - obs_mean
            # Σ̂_h,kk = ∑ₑ (h̃⁽ᵉ⁾_k)² / (N_ℰ − 1)
            obs_sample_variance = jnp.sum(obs_anomaly * obs_anomaly) / (N_e - 1)
            # Σ̂_h,kk + Σ_ε,kk
            innovation_variance = obs_sample_variance + obs_variance_k

            # x̄ = (1 / N_ℰ) ∑ₑ x⁽ᵉ⁾
            mean = jnp.mean(parts, axis=0)
            # x̃⁽ᵉ⁾ = x⁽ᵉ⁾ − x̄
            anomalies = parts - mean[None, :]
            # ∑ₑ h̃⁽ᵉ⁾_k x̃⁽ᵉ⁾
            cross_sum = einx.dot("e, e x -> x", obs_anomaly, anomalies)
            # Σ̂_xy,k = ∑ₑ h̃⁽ᵉ⁾_k x̃⁽ᵉ⁾ / (N_ℰ − 1)
            cross_covariance = cross_sum / (N_e - 1)
            # g_k = Σ̂_xy,k / (Σ̂_h,kk + Σ_ε,kk)
            gain_column = cross_covariance / innovation_variance

            # x̄ ← x̄ + g_k (y_k − h̄_k)
            new_mean = mean + gain_column * (obs_k - obs_mean)
            # α_k = 1 / (1 + √(Σ_ε,kk / (Σ̂_h,kk + Σ_ε,kk)))
            reduction = 1.0 / (1.0 + jnp.sqrt(obs_variance_k / innovation_variance))
            # g_k h̃⁽ᵉ⁾_k
            gain_outer = einx.dot("x, e -> e x", gain_column, obs_anomaly)
            # α_k g_k h̃⁽ᵉ⁾_k
            anomaly_update = reduction * gain_outer
            # x⁽ᵉ⁾ ← x̄ + x̃⁽ᵉ⁾ − α_k g_k h̃⁽ᵉ⁾_k
            updated = new_mean[None, :] + (anomalies - anomaly_update)
            return updated, None

        N_y = obs.shape[0]
        analysis_particles, _ = jax.lax.scan(assimilate_one, particles, jnp.arange(N_y))

        # log 𝒩(r̄; 0, Σ̂_y) of the forecast
        log_likelihood = _innovation_log_likelihood(
            initial_obs_particles, obs, obs_noise
        )
        return AnalysisResult(
            particles=analysis_particles, log_likelihood=log_likelihood
        )


class ESTKF(AbstractSequentialFilter, strict=True):
    """Error-subspace transform Kalman filter, which is the ETKF update.

        Λ_L = (N_ℰ − 1) I + Lᵀ H̃ Σ_ε⁻¹ H̃ᵀ L,   w̄_L = Λ_L⁻¹ Lᵀ H̃ Σ_ε⁻¹ r̄
        W_L = ((N_ℰ − 1) Λ_L⁻¹)^½
        x̄_t|t = x̄_t|t−1 + X̃ᵀ L w̄_L,   X̃_t|t = L W_L Lᵀ X̃

    Nerger et al. (2012) write the ETKF in the (N_ℰ − 1)-dimensional error subspace,
    through a projection L of shape (N_ℰ, N_ℰ − 1) with Lᵀ L = I and Lᵀ 1 = 0. The rows
    of X̃ and H̃ sum to zero, so their columns are orthogonal to 1 and L Lᵀ acts on them
    as the identity. The subspace form therefore gives the same w̄ = L w̄_L and
    W X̃ = L W_L Lᵀ X̃ as `ETKF`, and this class runs that analysis through
    `gaussx.etkf_analysis`, keeping the name and formulation for readers of the
    error-subspace literature.

    Examples:
        The same analysis as `ETKF`:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from filterax.filters import ESTKF, ETKF
        >>> particles = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        >>> obs_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> result = ESTKF().analysis(
        ...     particles, jnp.array([1.0, 0.5]), lambda x: x, obs_noise
        ... )
        >>> etkf = ETKF().analysis(
        ...     particles, jnp.array([1.0, 0.5]), lambda x: x, obs_noise
        ... )
        >>> bool(jnp.array_equal(result.particles, etkf.particles))
        True

    References:
        Nerger et al. (2012), *Mon. Wea. Rev.* 140, 2335–2345.
    """

    def analysis(
        self,
        particles: Float[Array, "N_e N_x"],
        obs: Float[Array, " N_y"],
        obs_op: AbstractObsOperator | ObsCallable,
        obs_noise: lx.AbstractLinearOperator,
        **_: Any,
    ) -> AnalysisResult:
        """ESTKF analysis step, the ETKF update.

        Algorithm:
            1. Map the members to observation space, h⁽ᵉ⁾ = h(x⁽ᵉ⁾).
            2. Form w̄ and W in ensemble space and apply them to X̃.
            3. Score r̄ under 𝒩(0, Σ̂_y).

        Args:
            particles: Forecast ensemble x⁽ᵉ⁾_t|t−1, one row per member.
            obs: Observation y.
            obs_op: Observation operator h, applied to one state.
            obs_noise: Observation-error covariance Σ_ε.

        Returns:
            The analysis ensemble and the log-likelihood log 𝒩(r̄; 0, Σ̂_y).

        Raises:
            ValueError: From gaussx, if N_ℰ < 2, or ``obs`` or ``obs_noise`` does not
                match D_y.
        """
        return _etkf_analysis(particles, obs, obs_op, obs_noise)

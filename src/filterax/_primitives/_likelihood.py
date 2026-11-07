"""Innovation log-likelihood and diagnostics of a forecast ensemble.

The innovation covariance Σ̂_y = Σ̂_h + Σ_ε is a `gaussx.LowRankUpdate`, so its
log-determinant and solves dispatch through the matrix-determinant lemma and Woodbury.
With more members than observations the (N_ℰ, N_ℰ) capacitance is the larger system,
so the ensemble log-likelihood takes a QR of the stacked factors of Σ̂_y instead.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TypedDict

import einx
import gaussx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from filterax._checks import check_ensemble_size
from filterax._primitives._statistics import ensemble_mean


def log_likelihood(
    innovation: Float[Array, " N_y"],
    innovation_cov: lx.AbstractLinearOperator,
    *,
    solver: gaussx.AbstractSolverStrategy | None = None,
) -> Float[Array, ""]:
    """Gaussian log-probability of a mean residual.

        log 𝒩(r̄; 0, Σ_y) = −½ (D_y log 2π + log det Σ_y + r̄ᵀ Σ_y⁻¹ r̄)

    Wraps `gaussx.gaussian_log_prob`, so both terms go through gaussx's dispatch: when
    Σ_y is a `gaussx.LowRankUpdate`, the standard case, the log-determinant uses the
    matrix-determinant lemma and the solve uses Woodbury, each at
    O(N_ℰ² D_y + N_ℰ³).

    This is the training signal of differentiable data assimilation: its gradient with
    respect to dynamics or observation-model parameters gives an end-to-end learning
    loop without ever forming Σ_y.

    Args:
        innovation: Mean residual r̄ = y − h̄.
        innovation_cov: Innovation covariance Σ_y, typically from
            `innovation_covariance`.
        solver: Solver strategy for Σ_y; ``None`` uses structural dispatch.

    Returns:
        The scalar log-probability.

    Examples:
        A unit covariance leaves −½ (2 log 2π + ‖r̄‖²):

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import filterax
        >>> innovation = jnp.array([0.5, -0.5])
        >>> innovation_cov = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> value = filterax.log_likelihood(innovation, innovation_cov)
        >>> round(float(value), 4)
        -2.0879

        A dense and a structured Σ_y agree:

        >>> dense = lx.MatrixLinearOperator(jnp.eye(2), lx.positive_semidefinite_tag)
        >>> bool(jnp.allclose(filterax.log_likelihood(innovation, dense), value))
        True
    """
    zero_mean = jnp.zeros_like(innovation)
    # log 𝒩(r̄; 0, Σ_y)
    return gaussx.gaussian_log_prob(
        zero_mean, innovation_cov, innovation, solver=solver
    )


def innovation_covariance(
    obs_particles: Float[Array, "N_e N_y"],
    obs_noise: lx.AbstractLinearOperator,
) -> gaussx.LowRankUpdate:
    """Innovation covariance of an ensemble as a low-rank update of Σ_ε.

        Σ̂_y = Σ̂_h + Σ_ε = Σ_ε + U Uᵀ,   U = H̃ᵀ / √(N_ℰ − 1)

    Takes the factor U from `gaussx.ensemble_covariance` with ``bessel=True`` and
    composes it with Σ_ε, keeping Σ_ε's structure. Downstream linear algebra exploits
    the term of rank at most N_ℰ − 1 without densifying.

    Args:
        obs_particles: Forecast ensemble h⁽ᵉ⁾ in observation space, one row per member.
        obs_noise: Observation-error covariance Σ_ε.

    Returns:
        A `gaussx.LowRankUpdate` holding Σ̂_y.

    Raises:
        ValueError: If N_ℰ < 2, or ``obs_noise`` is not (D_y, D_y).

    Examples:
        Two members with Σ̂_h = 0.5 and Σ_ε = 0.1:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import filterax
        >>> obs_particles = jnp.array([[0.0], [1.0]])
        >>> obs_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(1))
        >>> covariance = filterax.innovation_covariance(obs_particles, obs_noise)
        >>> bool(jnp.allclose(covariance.as_matrix(), jnp.array([[0.6]])))
        True

        It solves without being materialised:

        >>> import gaussx
        >>> gaussx.solve(covariance, jnp.ones(1)).shape
        (1,)
    """
    # Σ̂_h = U Uᵀ,   U = H̃ᵀ / √(N_ℰ − 1)
    obs_ensemble_covariance = gaussx.ensemble_covariance(obs_particles, bessel=True)
    # Σ̂_y = Σ_ε + U Uᵀ
    return gaussx.LowRankUpdate(obs_noise, obs_ensemble_covariance.U)


def ensemble_log_likelihood(
    obs_particles: Float[Array, "N_e N_y"],
    innovation: Float[Array, " N_y"],
    obs_noise: lx.AbstractLinearOperator,
    *,
    solver: gaussx.AbstractSolverStrategy | None = None,
) -> Float[Array, ""]:
    """Log-probability under the ensemble's Σ̂_y, by the cheaper and safer route.

        log 𝒩(r̄; 0, Σ̂_y),   Σ̂_y = U Uᵀ + Σ_ε

    With N_ℰ > D_y, a thin QR of the stacked factors [U, Σ_ε^½] gives a triangular T
    with Tᵀ T = Σ̂_y at O(N_ℰ D_y²):

        log det Σ̂_y = 2 ∑ₙ log |Tₙₙ|,   r̄ᵀ Σ̂_y⁻¹ r̄ = ‖T⁻ᵀ r̄‖²

    This avoids two failures in float32. Woodbury factors an (N_ℰ, N_ℰ) capacitance
    and is wrong once Σ_ε is small against Σ̂_h. Forming Σ̂_y squares the condition
    number, and gives NaN when channels are duplicated and Σ_ε is tiny. With
    N_ℰ ≤ D_y, or with a ``solver``, which needs the structured operator, this is
    `log_likelihood` on `innovation_covariance`.

    Algorithm:
        1. With N_ℰ ≤ D_y, or with a ``solver``, delegate to `log_likelihood`.
        2. Stack the factors U and Σ_ε^½, and take the triangular factor T.
        3. Whiten r̄ with T, and read the log-determinant off T's diagonal.
        4. Sum the terms of the log-probability.

    Args:
        obs_particles: Forecast ensemble h⁽ᵉ⁾ in observation space, one row per member.
        innovation: Mean residual r̄ = y − h̄.
        obs_noise: Observation-error covariance Σ_ε.
        solver: Solver strategy for Σ̂_y; given one, this takes the structured route.

    Returns:
        The scalar log-probability.

    Raises:
        ValueError: If N_ℰ < 2.

    Examples:
        The QR route agrees with the structured one:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import filterax
        >>> from filterax._primitives._likelihood import ensemble_log_likelihood
        >>> obs_particles = jnp.array([[0.0, 1.0], [2.0, 0.0], [1.0, 3.0], [4.0, 1.0]])
        >>> innovation = jnp.array([0.5, -0.5])
        >>> obs_noise = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> value = ensemble_log_likelihood(obs_particles, innovation, obs_noise)
        >>> covariance = filterax.innovation_covariance(obs_particles, obs_noise)
        >>> bool(jnp.allclose(value, filterax.log_likelihood(innovation, covariance)))
        True

        With fewer members than observations, it is the structured route itself:

        >>> value = ensemble_log_likelihood(obs_particles[:2], innovation, obs_noise)
        >>> covariance = filterax.innovation_covariance(obs_particles[:2], obs_noise)
        >>> bool(value == filterax.log_likelihood(innovation, covariance))
        True
    """
    ensemble_size, obs_size = obs_particles.shape
    if solver is not None or ensemble_size <= obs_size:
        obs_covariance = innovation_covariance(obs_particles, obs_noise)
        return log_likelihood(innovation, obs_covariance, solver=solver)
    # U = H̃ᵀ / √(N_ℰ − 1)
    obs_ensemble_covariance = gaussx.ensemble_covariance(obs_particles, bessel=True)
    ensemble_factor = obs_ensemble_covariance.U
    # Σ_ε^½
    noise_sqrt = gaussx.sqrt(obs_noise)
    noise_factor = noise_sqrt.as_matrix()
    # [U, Σ_ε^½]
    factors = jnp.hstack([ensemble_factor, noise_factor])
    # [U, Σ_ε^½]ᵀ, so that its triangular factor T has Tᵀ T = U Uᵀ + Σ_ε
    stacked = einx.id("obs col -> col obs", factors)
    # T
    upper = jnp.linalg.qr(stacked, mode="r")
    # T⁻ᵀ r̄
    whitened = jax.scipy.linalg.solve_triangular(upper, innovation, trans=1)
    # Tₙₙ
    diagonal = jnp.diagonal(upper)
    # log |Tₙₙ|, the sign of Tₙₙ being arbitrary
    log_diagonal = jnp.log(jnp.abs(diagonal))
    # ½ log det Σ̂_y = ∑ₙ log |Tₙₙ|
    half_logdet = einx.sum("[obs]", log_diagonal)
    # r̄ᵀ Σ̂_y⁻¹ r̄ = ‖T⁻ᵀ r̄‖²
    quadratic = einx.dot("obs, obs ->", whitened, whitened)
    # D_y log 2π
    normalisation = obs_size * math.log(2.0 * math.pi)
    # log det Σ̂_y
    logdet = 2.0 * half_logdet
    # −½ (D_y log 2π + log det Σ̂_y + r̄ᵀ Σ̂_y⁻¹ r̄)
    return -0.5 * (normalisation + logdet + quadratic)


class InnovationStatistics(TypedDict):
    """Return type of `innovation_statistics`.

    Attributes:
        innovation: Mean residual r̄ = y − h̄.
        innovation_cov: Innovation covariance Σ̂_y as a `gaussx.LowRankUpdate`.
        normalized_innovation: Whitened residual Σ̂_y^−½ r̄.
        log_likelihood: Gaussian log-probability log 𝒩(r̄; 0, Σ̂_y).
    """

    innovation: Float[Array, " N_y"]
    innovation_cov: lx.AbstractLinearOperator
    normalized_innovation: Float[Array, " N_y"]
    log_likelihood: Float[Array, ""]


def innovation_statistics(
    particles: Float[Array, "N_e N_x"],
    obs: Float[Array, " N_y"],
    obs_op: Callable[[Float[Array, " N_x"]], Float[Array, " N_y"]],
    obs_noise: lx.AbstractLinearOperator,
    *,
    solver: gaussx.AbstractSolverStrategy | None = None,
) -> InnovationStatistics:
    """Innovation diagnostics of a forecast ensemble.

        r̄ = y − h̄,   Σ̂_y = Σ̂_h + Σ_ε,   Σ̂_y^−½ r̄,   log 𝒩(r̄; 0, Σ̂_y)

    The whitened residual serves χ² and Desroziers-style consistency checks: under a
    correctly specified Σ̂_y each component is a unit-variance Gaussian. It uses the
    principal root through `gaussx.sqrt_inv_matmul`, a contour quadrature of shifted
    solves, so it does not depend on the order of the observations. With 30 nodes, its
    squared norm matches r̄ᵀ Σ̂_y⁻¹ r̄ to about 1e-10 (relative, float64) up to a
    condition number of 1e6.

    With N_ℰ ≤ D_y, the whitener goes through Woodbury and Σ̂_y is never materialised.
    Otherwise each Woodbury solve would factor the larger (N_ℰ, N_ℰ) capacitance, so
    Σ̂_y is formed densely. The log-likelihood comes from `ensemble_log_likelihood`.

    Algorithm:
        1. Map the members through h, and form r̄ and Σ̂_y.
        2. Choose the operator for the whitener from the shapes.
        3. Evaluate the log-likelihood.
        4. Whiten r̄ with 30 quadrature nodes.

    Args:
        particles: Forecast ensemble x⁽ᵉ⁾, one row per member.
        obs: Observation vector y.
        obs_op: Observation operator h on one state, mapped over the ensemble here.
        obs_noise: Observation-error covariance Σ_ε.
        solver: Solver strategy for the log-likelihood's log-determinant and solve.
            The whitened residual always uses structural dispatch.

    Returns:
        An `InnovationStatistics` mapping with ``innovation``, ``innovation_cov``,
        ``normalized_innovation`` and ``log_likelihood``.

    Raises:
        ValueError: If N_ℰ < 2.

    Examples:
        Two members, with the first component observed:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import filterax
        >>> particles = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        >>> obs = jnp.array([0.5])
        >>> obs_noise = lx.DiagonalLinearOperator(jnp.ones(1))
        >>> stats = filterax.innovation_statistics(
        ...     particles, obs, lambda state: state[:1], obs_noise
        ... )
        >>> stats["innovation"].tolist()
        [0.0]
        >>> round(float(stats["log_likelihood"]), 4)
        -1.1217

        One member cannot supply a sample covariance:

        >>> filterax.innovation_statistics(
        ...     particles[:1], obs, lambda state: state[:1], obs_noise
        ... )  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        ValueError: particles must contain at least 2 ensemble members for the
        Bessel-corrected estimators to be well-defined; got N_e=1.
    """
    # ensemble_mean runs first, and einx cannot size an empty ensemble.
    check_ensemble_size(particles.shape[0])
    # h⁽ᵉ⁾ = h(x⁽ᵉ⁾)
    obs_particles = jax.vmap(obs_op)(particles)
    # h̄ = (1 / N_ℰ) ∑ₑ h⁽ᵉ⁾
    obs_mean = ensemble_mean(obs_particles)
    # r̄ = y − h̄
    residual = obs - obs_mean

    # Σ̂_y = Σ_ε + U Uᵀ
    obs_covariance = innovation_covariance(obs_particles, obs_noise)
    ensemble_size, obs_size = obs_particles.shape
    # At N_ℰ = 1000, D_y = 3 the 30 Woodbury solves below take 100 s against
    # milliseconds for the (D_y, D_y) matrix.
    solve_operator = obs_covariance
    if solver is None and ensemble_size > obs_size:
        dense_covariance = obs_covariance.as_matrix()
        solve_operator = lx.MatrixLinearOperator(
            dense_covariance, lx.positive_semidefinite_tag
        )
    # log 𝒩(r̄; 0, Σ̂_y)
    log_prob = ensemble_log_likelihood(
        obs_particles, residual, obs_noise, solver=solver
    )

    residual_column = einx.id("obs -> obs 1", residual)
    # Σ̂_y^−½ r̄. The default 15 nodes leave a relative error of 1e-5 in its squared
    # norm at a condition number of 6.5e6.
    whitened_column = gaussx.sqrt_inv_matmul(
        solve_operator,
        residual_column,
        num_quadrature=30,
    )
    whitened = einx.id("obs 1 -> obs", whitened_column)

    return InnovationStatistics(
        innovation=residual,
        innovation_cov=obs_covariance,
        normalized_innovation=whitened,
        log_likelihood=log_prob,
    )

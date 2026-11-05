"""Innovation log-likelihood and diagnostics.

The innovation covariance ``S = H P H ᵀ + R`` is built as a
:class:`gaussx.LowRankUpdate`, so logdets and solves can dispatch through the
matrix-determinant lemma and Woodbury identity respectively. Only
:func:`innovation_statistics` forms the ``(Nᵧ, Nᵧ)`` matrix, and only when it
is smaller than the ``(Nₑ, Nₑ)`` Woodbury capacitance.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import einx
import gaussx
import jax
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
    r"""Gaussian log-probability of an innovation vector.

    $$
    \log p(y \mid \text{forecast}) = -\tfrac{1}{2} \left[
        N_{y} \log(2\pi) + \log\lvert S \rvert + v^{\top} S^{-1} v
    \right],
    $$

    where ``v`` is the innovation and ``S`` the innovation covariance.
    Both ``log|S|`` and ``S⁻¹ v`` flow through :mod:`gaussx` dispatch —
    when ``S`` is a :class:`gaussx.LowRankUpdate` (the standard case),
    log-determinants use the matrix-determinant lemma and solves use
    Woodbury, both at ``O(Nₑ² Nᵧ + Nₑ³)``.

    This is the training signal for differentiable data assimilation: the
    gradient of this scalar with respect to dynamics or observation-model
    parameters provides an end-to-end learning loop without ever forming
    ``S`` explicitly. See Decision D9 in the design docs.

    Args:
        innovation: Innovation ``v = y − H x̄``, shape ``(Nᵧ,)``.
        innovation_cov: Innovation covariance ``S = H P H ᵀ + R`` as a
            linear operator (typically :class:`gaussx.LowRankUpdate`).
        solver: Optional :class:`gaussx.AbstractSolverStrategy`. ``None``
            uses structural dispatch.

    Returns:
        Scalar log-probability.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from filterax import log_likelihood
        >>> innovation = jnp.array([0.5, -0.5])
        >>> S = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> log_likelihood(innovation, S)
        Array(-2.087877, dtype=float32)
    """
    zero = jax.numpy.zeros_like(innovation)
    return gaussx.gaussian_log_prob(zero, innovation_cov, innovation, solver=solver)


def innovation_covariance(
    obs_particles: Float[Array, "N_e N_y"],
    obs_noise: lx.AbstractLinearOperator,
) -> gaussx.LowRankUpdate:
    r"""Innovation covariance ``S = Cᴴᴴ + R`` as a low-rank update.

    Wraps :func:`gaussx.ensemble_covariance` (Bessel-corrected) to build
    the low-rank factor ``U = (HX)′ᵀ / √(Nₑ − 1)`` and composes it with the
    structured ``R`` base. The result is a :class:`gaussx.LowRankUpdate`
    so all downstream linear algebra (solve, logdet, sample, …) takes
    advantage of the rank-``(Nₑ − 1)`` ensemble term without densifying.

    Args:
        obs_particles: ``H`` applied to each ensemble member, shape
            ``(Nₑ, Nᵧ)``.
        obs_noise: Observation error covariance ``R``.

    Returns:
        :class:`gaussx.LowRankUpdate` representing ``S``.

    Raises:
        ValueError: if ``Nₑ < 2``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from filterax import innovation_covariance
        >>> obs_particles = jnp.array([[0.0], [1.0]])  # Cᴴᴴ = 0.5
        >>> R = lx.DiagonalLinearOperator(0.1 * jnp.ones(1))
        >>> S = innovation_covariance(obs_particles, R)
        >>> S.as_matrix()  # 0.5 + 0.1
        Array([[0.6]], dtype=float32)
    """
    cov = gaussx.ensemble_covariance(obs_particles, bessel=True)
    return gaussx.LowRankUpdate(obs_noise, cov.U)


class InnovationStatistics(TypedDict):
    """Return type of :func:`innovation_statistics`.

    Keys:
        innovation: ``v = y − H x̄``.
        innovation_cov: ``S = Cᴴᴴ + R`` as a :class:`gaussx.LowRankUpdate`.
        normalized_innovation: ``S⁻¹ᐟ² v`` (whitened residual).
        log_likelihood: Scalar Gaussian log-probability of ``v`` under ``S``.
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
    r"""Innovation diagnostics for a forecast ensemble.

    Returns ``v``, ``S``, the whitened residual ``S⁻¹ᐟ² v``, and
    ``log p(y | forecast)``. The whitened residual is useful for χ² /
    Desroziers-style consistency checks (under correctly-specified ``S``
    each component is unit-variance Gaussian).

    The whitened residual uses the principal root through
    :func:`gaussx.sqrt_inv_matmul`, a contour quadrature of shifted solves, so
    it does not depend on the order of the observations. With 30 nodes, its
    squared norm matches ``vᵀ S⁻¹ v`` to about ``1e-10`` (relative, float64) up
    to a condition number of ``1e6``.

    With fewer members than observations, every solve goes through Woodbury and
    ``S`` is never materialised. Otherwise each Woodbury solve would factor the
    larger ``(Nₑ, Nₑ)`` capacitance, so ``S`` is formed densely, the same rule
    as :func:`gaussx.ensemble_kalman_gain`. Passing ``solver`` keeps the
    structured operator for the log-likelihood.

    Args:
        particles: Forecast ensemble, shape ``(Nₑ, Nₓ)``.
        obs: Observation vector, shape ``(Nᵧ,)``.
        obs_op: ``H`` applied to a single state vector. ``vmap``'d over
            the ensemble internally.
        obs_noise: Observation error covariance ``R``.
        solver: Optional solver strategy for the log-likelihood's logdet and
            solve. The whitened residual always uses structural dispatch.

    Returns:
        :class:`InnovationStatistics` mapping with ``innovation``,
        ``innovation_cov``, ``normalized_innovation``, ``log_likelihood``.

    Raises:
        ValueError: if ``Nₑ < 2``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from filterax import innovation_statistics
        >>> particles = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        >>> obs = jnp.array([0.5])
        >>> R = lx.DiagonalLinearOperator(jnp.ones(1))
        >>> stats = innovation_statistics(particles, obs, lambda x: x[:1], R)
        >>> stats["innovation"]  # y − H x̄ = 0.5 − 0.5
        Array([0.], dtype=float32)
        >>> stats["log_likelihood"]
        Array(-1.1216711, dtype=float32)
    """
    # ensemble_mean runs first, and einx cannot size an empty ensemble.
    check_ensemble_size(particles.shape[0])
    obs_particles = jax.vmap(obs_op)(particles)
    innovation = obs - ensemble_mean(obs_particles)

    S = innovation_covariance(obs_particles, obs_noise)
    num_members, obs_size = obs_particles.shape
    # At Nₑ = 1000, Nᵧ = 3 the 30 Woodbury solves below take 100 s against
    # milliseconds for the (Nᵧ, Nᵧ) matrix.
    solve_operator = S
    if solver is None and num_members >= obs_size:
        solve_operator = lx.MatrixLinearOperator(
            S.as_matrix(), lx.positive_semidefinite_tag
        )
    log_prob = log_likelihood(innovation, solve_operator, solver=solver)

    # z = S⁻¹ᐟ² v. The default 15 nodes leave a relative error of 1e-5 in zᵀ z
    # at a condition number of 6.5e6.
    normalized = gaussx.sqrt_inv_matmul(
        solve_operator,
        einx.id("obs -> obs 1", innovation),
        num_quadrature=30,
    )
    normalized = einx.id("obs 1 -> obs", normalized)

    return InnovationStatistics(
        innovation=innovation,
        innovation_cov=S,
        normalized_innovation=normalized,
        log_likelihood=log_prob,
    )

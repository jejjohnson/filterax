"""Random perturbation generation for stochastic ensemble methods.

The single primitive here, :func:`perturbed_observations`, is the Monte
Carlo Burgers-van-Leeuwen-Evensen (1998) draw used by the stochastic
EnKF analysis. When ``R`` is diagonal — the overwhelmingly common case
for instrument noise — we skip the dense Cholesky entirely; otherwise we
fall back to gaussx's structured root decomposition so dense matrices are
still avoided when ``R`` carries low-rank or Toeplitz structure.
"""

from __future__ import annotations

import einx
import gaussx
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray


def perturbed_observations(
    key: PRNGKeyArray,
    obs: Float[Array, " N_y"],
    obs_noise: lx.AbstractLinearOperator,
    n_ensemble: int,
) -> Float[Array, "N_e N_y"]:
    r"""Generate perturbed observations for the stochastic EnKF.

    $$
    y^{(j)}_{\text{pert}} = y + \varepsilon^{(j)},
    \qquad
    \varepsilon^{(j)} \sim \mathcal{N}(0, R),
    \qquad
    j = 1, \dots, N_{e}.
    $$

    The sampling path depends on the structure of ``R``:

    * :class:`lineax.DiagonalLinearOperator`: draws are computed as
      ``ε = z ⊙ √diag(R)`` for ``z ~ 𝒩(0, I)``. No matrix is ever
      formed; cost is ``O(Nₑ Nᵧ)``.
    * General ``R``: a structural square root from
      :func:`gaussx.root_decomposition` is applied to standard normals.
      For dense ``R`` this still costs ``O(Nᵧ³)`` for the factorisation,
      but the cost is amortised across ``Nₑ`` samples; structured ``R``
      (low-rank, Toeplitz, …) avoids the cubic term entirely.

    Args:
        key: PRNG key consumed by this call.
        obs: Observation vector ``(Nᵧ,)``.
        obs_noise: Observation error covariance ``R`` as a linear operator.
        n_ensemble: Number of perturbed copies ``Nₑ``.

    Returns:
        Perturbed observation matrix of shape ``(Nₑ, Nᵧ)``.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from filterax import perturbed_observations
        >>> obs = jnp.array([1.0, 2.0])
        >>> R = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> y_pert = perturbed_observations(jax.random.key(0), obs, R, 4)
        >>> y_pert.shape
        (4, 2)

    Reference:
        Burgers, G., van Leeuwen, P. J., & Evensen, G. (1998). *Analysis
        scheme in the ensemble Kalman filter.* Mon. Wea. Rev., 126,
        1719–1724.
    """
    n_obs = obs.shape[0]
    standard = jr.normal(key, (n_ensemble, n_obs))

    if isinstance(obs_noise, lx.DiagonalLinearOperator):
        # ε⁽ʲ⁾_k = z⁽ʲ⁾_k √R_kk — pointwise scaling along the obs axis.
        sigma = lx.diagonal(obs_noise) ** 0.5
        eps = einx.multiply("e y, y -> e y", standard, sigma)
    else:
        # Structured square root: R = L Lᵀ with L = root_decomposition(R).root.
        # For LowRankUpdate / Toeplitz / Kronecker, gaussx avoids the dense
        # Cholesky and returns an operator-typed factor.
        root = gaussx.root_decomposition(obs_noise).root  # L with L Lᵀ = R
        # ε⁽ʲ⁾ = L z⁽ʲ⁾ — sum the latent axis ``a``; ``y`` is the output axis.
        eps = einx.dot("e a, y a -> e y", standard, root)

    return einx.add("e y, y -> e y", eps, obs)

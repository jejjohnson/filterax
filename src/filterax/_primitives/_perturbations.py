"""Random perturbation generation for stochastic ensemble methods.

The single primitive here, :func:`perturbed_observations`, is the Monte
Carlo Burgers-van-Leeuwen-Evensen (1998) draw used by the stochastic
EnKF analysis. Sampling is delegated to :func:`gaussx.sample_mvn`, which draws
exactly from ``R`` and dispatches on its structure.
"""

from __future__ import annotations

import gaussx
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

    Sampling delegates to :func:`gaussx.sample_mvn`. Every path draws
    ``ε = L z`` with ``z ~ 𝒩(0, I)`` and ``L Lᵀ = R`` exactly, and only the
    factor depends on the structure of ``R``:

    * Diagonal ``R``: ``ε = √diag(R) ⊙ z``. No matrix is ever formed; cost
      is ``O(Nₑ Nᵧ)``.
    * ``gaussx`` block-diagonal, Kronecker, Toeplitz and low-rank-update
      ``R``: a factor that keeps the structure, avoiding the cubic cost.
    * Any other ``R``: a dense symmetric square root. It costs ``O(Nᵧ³)``
      once per call and also handles a singular ``R``.

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
    # y⁽ʲ⁾ = y + L z⁽ʲ⁾, L Lᵀ = R
    return gaussx.sample_mvn(obs, obs_noise, key=key, num_samples=n_ensemble)

"""Random perturbation generation for stochastic ensemble methods."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray


def perturbed_observations(
    key: PRNGKeyArray,
    obs: Float[Array, " N_y"],
    obs_noise: lx.AbstractLinearOperator,
    n_ensemble: int,
) -> Float[Array, "N_e N_y"]:
    r"""Generate perturbed observations for the stochastic EnKF.

    .. math::

        y^{(j)}_{\text{pert}} = y + \varepsilon^{(j)}, \qquad
        \varepsilon^{(j)} \sim \mathcal{N}(0, R)

    Samples are drawn with a Cholesky factor of ``R.as_matrix()``. For
    diagonal ``R`` this is exact and cheap; for dense ``R`` the cost is the
    one-time :math:`O(N_y^3)` Cholesky plus :math:`O(N_e N_y^2)` for the
    apply. Use a structured operator on the caller side if ``N_y`` is large.

    Args:
        key: PRNG key.
        obs: Observation vector ``(N_y,)``.
        obs_noise: Observation error covariance :math:`R` as a linear operator.
        n_ensemble: Number of perturbed copies to draw.

    Returns:
        Perturbed observation matrix of shape ``(n_ensemble, N_y)``.

    Reference:
        Burgers, G., van Leeuwen, P. J., & Evensen, G. (1998). *Analysis
        scheme in the ensemble Kalman filter.* Mon. Wea. Rev., 126,
        1719-1724.
    """
    import jax.random as jr

    n_obs = obs.shape[0]
    # Cholesky of the dense materialisation; diagonal R is the typical case
    # and incurs no real cost. Callers with structured R can pre-decompose.
    cov_dense = obs_noise.as_matrix()
    chol = jnp.linalg.cholesky(cov_dense)
    standard = jr.normal(key, (n_ensemble, n_obs))
    eps = standard @ chol.T  # (N_e, N_y)
    return obs[None, :] + eps

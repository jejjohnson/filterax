"""Perturbed observations for the stochastic ensemble Kalman analysis."""

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
    """One noisy copy of the observations per ensemble member.

        y⁽ᵉ⁾ = y + ε⁽ᵉ⁾,   ε⁽ᵉ⁾ ~ 𝒩(0, Σ_ε),   e ∈ ℰ

    Wraps `gaussx.sample_mvn`, which draws ε = L z with z ~ 𝒩(0, I) and L Lᵀ = Σ_ε
    exactly, and chooses L by the structure of Σ_ε:

    - Diagonal: ε = √diag(Σ_ε) ⊙ z. No matrix is formed, and it costs O(N_ℰ D_y).
    - `gaussx` block-diagonal, Kronecker, Toeplitz and low-rank-update: a factor that
      keeps the structure, avoiding the cubic cost.
    - Anything else: a dense symmetric square root. It costs O(D_y³) once per call
      and also handles a singular Σ_ε.

    Args:
        key: PRNG key consumed by this call.
        obs: Observation vector y.
        obs_noise: Observation-error covariance Σ_ε.
        n_ensemble: Number of perturbed copies N_ℰ.

    Returns:
        The perturbed observations y⁽ᵉ⁾, one row per member.

    Raises:
        ValueError: If ``n_ensemble`` is below 1, or ``obs`` does not match Σ_ε.

    Examples:
        Four copies of two observations:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import filterax
        >>> obs = jnp.array([1.0, 2.0])
        >>> obs_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> perturbed = filterax.perturbed_observations(jr.key(0), obs, obs_noise, 4)
        >>> perturbed.shape
        (4, 2)

        Zero noise returns the observations themselves:

        >>> quiet = lx.DiagonalLinearOperator(jnp.zeros(2))
        >>> copies = filterax.perturbed_observations(jr.key(0), obs, quiet, 2)
        >>> copies.tolist()
        [[1.0, 2.0], [1.0, 2.0]]

        At least one copy is needed:

        >>> filterax.perturbed_observations(jr.key(0), obs, obs_noise, 0)
        Traceback (most recent call last):
        ...
        ValueError: num_samples must be at least 1, got 0.

    References:
        Burgers, van Leeuwen & Evensen (1998), *Mon. Wea. Rev.* 126, 1719–1724.
    """
    # y⁽ᵉ⁾ = y + L z⁽ᵉ⁾,   L Lᵀ = Σ_ε
    return gaussx.sample_mvn(obs, obs_noise, key=key, num_samples=n_ensemble)

"""Differentiable data-assimilation surface.

Two gradient strategies for training through the filter:

* :func:`differentiable_assimilate` — fixed-shape :func:`jax.lax.scan`
  with the full forward+backward tape (optional :func:`jax.checkpoint`
  for ``O(√T)`` memory). Use when you want exact gradients including
  cross-time terms.
* :func:`road_enkf_loss_and_grad` / :func:`road_enkf_grad_step` —
  ROAD-EnKF local-gradient strategy with :func:`jax.lax.stop_gradient`
  between cycles. Backward-pass memory ``O(Nₑ · Nₓ)`` independent of
  ``T``; drops cross-time gradient terms.

Both refuse stochastic filters (``StochasticEnKF``, ``ETKF_Livings``)
and the stochastic ``AdditiveInflator`` at call time — see
``design_docs/features/differentiable_da.md`` §5.2 / §5.5.
"""

from filterax._train._differentiable import (
    differentiable_assimilate as differentiable_assimilate,
)
from filterax._train._road import (
    road_enkf_grad_step as road_enkf_grad_step,
    road_enkf_loss_and_grad as road_enkf_loss_and_grad,
)

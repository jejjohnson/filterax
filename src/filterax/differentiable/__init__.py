"""Differentiable data-assimilation surface.

Gradient strategies for training through the filter:

* :func:`differentiable_assimilate` — fixed-shape :func:`jax.lax.scan`
  whose gradient strategy is selected with ``adjoint=``:
  :class:`DirectAdjoint` (exact, full tape),
  :class:`RecursiveCheckpointAdjoint` (exact, ``O(√T)`` memory via
  recomputation), or :class:`TruncatedAdjoint` (biased: trailing-``k``
  cycles only — ``O(1)`` memory in ``T`` and tolerant of chaotic
  gradient explosion). The names match the shared pipekit / diffrax
  vocabulary; ``pipekit_cycle.adjoints`` specs are accepted
  interchangeably.
* :func:`road_enkf_loss_and_grad` / :func:`road_enkf_grad_step` —
  ROAD-EnKF local-gradient strategy with :func:`jax.lax.stop_gradient`
  between cycles. Backward-pass memory ``O(Nₑ · Nₓ)`` independent of
  ``T``; drops cross-time gradient terms.

Both refuse stochastic filters (``StochasticEnKF``, ``ETKF_Livings``)
and the stochastic ``AdditiveInflator`` at call time — see
``design_docs/features/differentiable_da.md`` §5.2 / §5.5.
"""

from filterax._train._adjoints import (
    DirectAdjoint as DirectAdjoint,
    RecursiveCheckpointAdjoint as RecursiveCheckpointAdjoint,
    TruncatedAdjoint as TruncatedAdjoint,
)
from filterax._train._differentiable import (
    differentiable_assimilate as differentiable_assimilate,
)
from filterax._train._road import (
    road_enkf_grad_step as road_enkf_grad_step,
    road_enkf_loss_and_grad as road_enkf_loss_and_grad,
)

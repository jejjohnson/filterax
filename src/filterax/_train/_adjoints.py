r"""Adjoint strategies for the differentiable assimilation loop.

How should reverse-mode gradients flow through the ``T``-cycle scan of
:func:`filterax.differentiable.differentiable_assimilate`? Three
declarative strategies, named to match the shared pipekit / diffrax
vocabulary (filterax does not depend on pipekit — specs are matched
structurally, so the identically-named classes from
``pipekit_cycle.adjoints`` are accepted interchangeably):

* :class:`DirectAdjoint` — plain scan: exact gradients, the whole
  forward tape is stored, memory $O(T)$. The default.
* :class:`RecursiveCheckpointAdjoint` — :func:`jax.checkpoint` around
  the scan body: exact gradients, memory $O(\sqrt{T})$-ish at the cost
  of recomputation.
* :class:`TruncatedAdjoint` — cross-cycle gradient flow stops more
  than ``k`` cycles back: carries leaving earlier cycles run under
  :func:`jax.lax.stop_gradient`. Per-cycle *outputs* keep their local
  gradients deliberately — a loss summed over the returned histories
  (e.g. ``-sum(log_likelihoods)``) therefore reproduces the ROAD-EnKF
  local-gradient estimator, where every window contributes its own
  term but no cross-window adjoint products form. Biased but
  *chaos-tolerant*: the exact adjoint of a chaotic rollout grows like
  $e^{\lambda_{\max} T}$, so truncation acts as gradient
  regularisation. Backward memory is $O(1)$ in ``T`` for losses on
  the final state; for history-summed losses the activation memory
  matches the plain scan, and the win is the bounded gradient depth.
  (See :func:`filterax.differentiable.road_enkf_loss_and_grad` for
  the sequential ROAD-EnKF driver whose memory is $O(N_e N_x)$
  regardless of the loss.)

Forward values are identical under every strategy — only the gradient
computation differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DirectAdjoint:
    """Exact gradients with the full forward tape stored ($O(T)$ memory)."""


@dataclass(frozen=True)
class RecursiveCheckpointAdjoint:
    """Exact gradients via :func:`jax.checkpoint` recomputation.

    Attributes:
        checkpoints: Reserved for schedule control; the current
            implementation uses :func:`jax.checkpoint`'s default policy
            and ignores this value beyond carrying it as config.
    """

    checkpoints: int | None = None


@dataclass(frozen=True)
class TruncatedAdjoint:
    """Gradients flow through the trailing ``k`` cycles only.

    Cross-cycle flow through the ensemble carry is cut more than
    ``k`` cycles back; per-cycle outputs keep their local gradients,
    so history-summed losses accumulate one local term per window
    (the ROAD-EnKF estimator at ``k=1``) instead of exploding
    cross-window products.

    Attributes:
        k: Number of trailing cycles that propagate gradients through
            the ensemble carry. Must be at least 1; ``k >= T`` is
            equivalent to :class:`DirectAdjoint`.
    """

    k: int = 1


def resolve_adjoint(adjoint: Any, checkpoint: bool) -> Any:
    """Resolve the ``adjoint=`` / deprecated ``checkpoint=`` pair.

    Args:
        adjoint: A strategy object (any of the spec classes above, or a
            structurally identical object such as the pipekit specs),
            or ``None`` to derive the strategy from ``checkpoint``.
        checkpoint: The deprecated boolean flag.

    Returns:
        The effective strategy object.

    Raises:
        ValueError: if both ``adjoint`` and ``checkpoint=True`` are
            supplied, or the strategy is unrecognised / invalid.
    """
    if adjoint is None:
        return RecursiveCheckpointAdjoint() if checkpoint else DirectAdjoint()
    if checkpoint:
        raise ValueError(
            "Pass either adjoint= or the deprecated checkpoint=True, not both."
        )
    name = type(adjoint).__name__
    if name in ("DirectAdjoint", "RecursiveCheckpointAdjoint"):
        return adjoint
    if name == "TruncatedAdjoint":
        k = getattr(adjoint, "k", None)
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"TruncatedAdjoint needs an integer k >= 1; got {k!r}.")
        return adjoint
    raise ValueError(
        f"Unrecognised adjoint strategy: {adjoint!r}. Expected DirectAdjoint, "
        "RecursiveCheckpointAdjoint, or TruncatedAdjoint (filterax or "
        "pipekit_cycle spec objects)."
    )

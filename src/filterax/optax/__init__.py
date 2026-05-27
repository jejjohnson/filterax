"""Ensemble Kalman processes exposed as optax gradient transformations.

Each constructor returns an ``optax.GradientTransformation`` with the
ensemble (or sigma-point belief) living inside the optax ``OptState``
and the running mean estimate flowing back through ``apply_updates``.

* :func:`eki` — Ensemble Kalman Inversion.
* :func:`eks` — Ensemble Kalman Sampler (posterior sampling).
* :func:`uki` — Unscented Kalman Inversion (parametric).

Compose with ``optax.chain`` for clipping, schedules, hybrid pipelines
with gradient-based optimisers, etc. See
``docs/design_docs/features/optax_ekp.md`` for the contract and
examples.
"""

from filterax._src.optax_processes import (
    EKIOptaxState as EKIOptaxState,
    EKSOptaxState as EKSOptaxState,
    UKIOptaxState as UKIOptaxState,
    eki as eki,
    eks as eks,
    uki as uki,
)

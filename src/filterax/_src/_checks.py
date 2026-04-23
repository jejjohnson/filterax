"""Preconditions shared by the Bessel-corrected primitives.

The ensemble size ``N_e`` is a static (shape-level) quantity, so this check
runs at trace time and does not introduce a runtime branch.
"""

from __future__ import annotations


def check_ensemble_size(n_ensemble: int, *, name: str = "particles") -> None:
    """Raise ``ValueError`` if ``n_ensemble < 2``.

    All Bessel-corrected estimators divide by ``N_e - 1`` and the ensemble
    Kalman gain additionally divides by ``sqrt(N_e - 1)``; a degenerate
    one-member ensemble would silently produce ``inf`` / ``nan``. We fail
    fast at the boundary instead.
    """
    if n_ensemble < 2:
        raise ValueError(
            f"{name} must contain at least 2 ensemble members for the "
            f"Bessel-corrected estimators to be well-defined; got N_e={n_ensemble}."
        )

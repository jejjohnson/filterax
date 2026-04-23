"""Shared test fixtures.

Enables x64 globally so analytic comparisons against numpy/scipy do not
fail on float32 rounding.
"""

from __future__ import annotations

import equinox.internal as eqxi
import jax
import pytest


jax.config.update("jax_enable_x64", True)


@pytest.fixture
def getkey():
    return eqxi.GetKey()

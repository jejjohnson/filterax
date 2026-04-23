"""Shared test fixtures.

Enables x64 globally so analytic comparisons against numpy/scipy do not
fail on float32 rounding. Uses only public :mod:`jax.random` APIs for the
key generator.
"""

from __future__ import annotations

import jax
import jax.random as jr
import pytest


jax.config.update("jax_enable_x64", True)


@pytest.fixture
def getkey():
    """Return a zero-arg callable that yields fresh PRNG keys per invocation.

    Each test gets its own split stream seeded at 0, so calls are independent
    across tests but deterministic within a test.
    """
    state = {"key": jr.PRNGKey(0)}

    def _next() -> jax.Array:
        state["key"], sub = jr.split(state["key"])
        return sub

    return _next

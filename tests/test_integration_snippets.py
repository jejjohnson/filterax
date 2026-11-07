"""Every snippet on the integrations docs page runs as written."""

from __future__ import annotations

import pathlib
import re

import pytest


PAGE = pathlib.Path(__file__).parents[1] / "docs" / "api" / "integrations.md"
SNIPPETS = re.findall(r"```python\n(.*?)```", PAGE.read_text(), flags=re.DOTALL)


def test_the_page_has_its_snippets():
    assert len(SNIPPETS) == 4


# The optax and somax snippets run a whole loop, so they are integration tests too.
CASES = [
    pytest.param(SNIPPETS[0], id="optax", marks=pytest.mark.integration),
    pytest.param(SNIPPETS[1], id="kronecker"),
    pytest.param(SNIPPETS[2], id="norm"),
    pytest.param(SNIPPETS[3], id="somax", marks=pytest.mark.integration),
]


@pytest.mark.slow
@pytest.mark.parametrize("snippet", CASES)
def test_snippet_runs(snippet):
    # somax is not a filterax dependency, so its snippet needs it installed.
    if "somax" in snippet:
        pytest.importorskip("somax")
    exec(compile(snippet, str(PAGE), "exec"), {"__name__": "snippet"})

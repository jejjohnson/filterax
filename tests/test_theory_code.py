"""Every code block in the maths reference runs as written, chapter by chapter."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


CHAPTERS = sorted((Path(__file__).parents[1] / "docs" / "math").glob("[0-9]*.md"))
# A fence, optionally MyST's code-block directive with its option lines.
BLOCK = re.compile(
    r"^```(?:python|\{code-block\} python)\n(?::[a-z]+:[^\n]*\n)*(.*?)^```",
    flags=re.DOTALL | re.MULTILINE,
)


@pytest.mark.slow
@pytest.mark.parametrize("chapter", CHAPTERS, ids=lambda path: path.stem)
def test_chapter_code_runs(chapter):
    # A chapter's blocks build on each other, so they share one namespace.
    namespace = {"__name__": f"chapter_{chapter.stem}"}
    for block in BLOCK.findall(chapter.read_text()):
        # A sketch with placeholders says so on its first line.
        if block.startswith("# not run"):
            continue
        exec(compile(block, str(chapter), "exec"), namespace)

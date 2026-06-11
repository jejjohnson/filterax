"""Docstring quality gates for the rendered API docs.

mkdocstrings renders docstrings through griffe's Google-style parser,
and two classes of mistake silently degrade the built docs:

* Doctest code under a header griffe does not recognise as an examples
  section (e.g. singular ``Example:``) is parsed as a generic
  admonition, and the ``>>>`` lines collapse into a run-together
  paragraph instead of a highlighted code block.
* Unbalanced ``$$`` math fences leave raw TeX in the rendered page.

These tests parse every docstring in the package exactly the way the
docs build does and fail on either mistake, so the problems cannot
reach the published site again.
"""

from __future__ import annotations

import griffe
import pytest
from griffe import Docstring, DocstringSectionKind, Parser


def _iter_docstrings(obj, seen):
    """Yield (path, Docstring) for every object defined in filterax."""
    if obj.path in seen:
        return
    seen.add(obj.path)
    if obj.docstring is not None:
        yield obj.path, obj.docstring
    members = getattr(obj, "members", {})
    for member in members.values():
        # Skip aliases (re-exports): gaussx-owned docstrings are tested
        # upstream, and filterax-owned ones are reached at their
        # definition site.
        if member.is_alias:
            continue
        yield from _iter_docstrings(member, seen)


@pytest.fixture(scope="module")
def all_docstrings() -> list[tuple[str, Docstring]]:
    pkg = griffe.load("filterax", search_paths=["src"])
    docstrings = list(_iter_docstrings(pkg, set()))
    assert len(docstrings) > 50  # the walk found the real package
    return docstrings


def test_doctests_parse_as_examples_sections(all_docstrings):
    """Every ``>>>`` snippet must land in a real examples section.

    A doctest inside a text or admonition section means the section
    header is one griffe does not recognise (the ``Example:`` vs
    ``Examples:`` trap) and the code will render as prose.
    """
    offenders = []
    for path, docstring in all_docstrings:
        for section in docstring.parse(Parser.google):
            if section.kind is DocstringSectionKind.examples:
                continue
            if section.kind is DocstringSectionKind.admonition:
                text = section.value.contents
            elif section.kind is DocstringSectionKind.text:
                text = section.value
            else:
                continue
            if ">>>" in str(text):
                offenders.append(f"{path} ({section.kind.value})")
    assert not offenders, (
        "Doctest code outside an examples section — use the plural "
        f"'Examples:' header so mkdocstrings renders it as code: {offenders}"
    )


def test_examples_sections_contain_code(all_docstrings):
    """Parsed examples sections must carry at least one code part."""
    offenders = []
    for path, docstring in all_docstrings:
        for section in docstring.parse(Parser.google):
            if section.kind is not DocstringSectionKind.examples:
                continue
            kinds = [kind for kind, _ in section.value]
            if DocstringSectionKind.examples not in kinds:
                offenders.append(path)
    assert not offenders, f"Examples sections without doctest code: {offenders}"


def test_math_fences_are_balanced(all_docstrings):
    """``$$`` math fences must come in pairs, or raw TeX leaks into the docs."""
    offenders = [
        path
        for path, docstring in all_docstrings
        if docstring.value.count("$$") % 2 != 0
    ]
    assert not offenders, f"Unbalanced $$ math fences: {offenders}"

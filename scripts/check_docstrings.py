#!/usr/bin/env python
"""Catch docstrings that would fail the `-W` docs build, without running Sphinx.

    scripts/check_docstrings.py
    scripts/check_docstrings.py --verbose

The docs build treats every warning as an error
(``sphinx-build -b html -W --keep-going docs docs/_build/html``), and autodoc
inserts a docstring as **nested** content. That nesting is the whole problem: a
section title or a malformed table is legal at the top level of a document and
illegal one level in, so a docstring can look fine in an editor, render fine in
a plain docutils run, and take the site down.

This runs each docstring the way autodoc will -- through napoleon, then indented
under a directive -- and reports what docutils says. It found seven real defects
when it was written, in six files:

- ``FeatureDict``'s ``Keys/----`` block, a section title napoleon does not know
  (fixed in ``conf.py`` with ``napoleon_custom_sections``, not in the source).
- ``SCHEMA_VERSION``'s ``History/-------`` ``#:`` block, the same shape.
- Two simple tables whose column border was narrower than its header cell.
- An unescaped ``|r|``, read as an undefined substitution reference.
- Two numpydoc ``Returns``/``Raises`` sections written as free prose with no
  type line, which makes napoleon treat the prose *as* the type and mangle it
  into bullets -- the one class of defect here that emits only a warning while
  destroying three whole sections.

**Why it is a fast test and not a docs-build concern.** The docs build runs in
its own workflow, on push and on pull requests, and a docstring edit that breaks
it would otherwise be found a workflow later by someone else. This is the same
reasoning as the standing rule that a guard against a silently wrong number
belongs in the fast suite.

**It is not a substitute for the docs build.** It sees one docstring at a time,
so it cannot catch a duplicated ``automodule``, an orphan page, or a broken
toctree -- only defects inside a docstring.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "visbench"

#: Sphinx's own attribute-comment pattern (``sphinx/pycode/parser.py``), so a
#: ``#:`` block is read exactly the way autodoc will read it.
COMMENT = re.compile(r"^\s*#: ?(.*)$")

#: Roles and directives that only exist inside Sphinx. docutils alone cannot
#: know them, so these are noise rather than findings -- and this is the one
#: filter here, deliberately narrow: anything else docutils says is a real
#: defect.
NOISE = re.compile(r"Unknown interpreted text role|Unknown directive type|No role entry")

#: Where a docutils message starts. Needed because a message is several lines
#: -- the severity line, then the offending source as context -- and filtering
#: line by line keeps the context of a message it just dropped. The first draft
#: did exactly that and reported two files' worth of `.. attribute::` bodies as
#: findings, which is a checker crying wolf: the least useful failure mode for
#: something whose whole job is to be trusted in the fast suite.
MESSAGE = re.compile(r"^<string>:\d+: \((?:INFO|WARNING|ERROR|SEVERE)/\d\)")


def as_autodoc_sees_it(text: str) -> str:
    """napoleon's output, indented under a directive the way autodoc inserts it.

    The indent is the entire trick. Un-nested, a section title is legal and
    docutils says nothing; nested, it is a SEVERE. A checker that skips this
    step reports a clean bill of health on a docstring that fails the build.
    """
    from sphinx.ext.napoleon import Config
    from sphinx.ext.napoleon.docstring import NumpyDocstring

    config = Config(
        napoleon_google_docstring=False,
        napoleon_numpy_docstring=True,
        napoleon_use_rtype=False,
        napoleon_preprocess_types=True,
        napoleon_custom_sections=[("Keys", "params_style")],
    )
    converted = str(NumpyDocstring(text, config))
    body = "\n".join("   " + line if line.strip() else "" for line in converted.split("\n"))
    return f".. admonition:: nested\n\n{body}\n"


def findings(text: str) -> list[str]:
    """What docutils reports about one docstring, Sphinx-only noise removed."""
    from docutils.core import publish_doctree
    from docutils.utils import SystemMessage

    collected: list[str] = []
    try:
        publish_doctree(
            as_autodoc_sees_it(text),
            settings_overrides={
                "report_level": 2,
                "halt_level": 5,
                "warning_stream": _Collector(collected),
                "traceback": True,
                "input_encoding": "unicode",
            },
        )
    except SystemMessage as error:  # pragma: no cover - halt_level makes this rare
        collected.append(str(error))
    return _real_messages(collected)


def _real_messages(lines: list[str]) -> list[str]:
    """Group docutils output into whole messages, then drop the Sphinx-only ones.

    A message is its severity line plus every line until the next one, so the
    unit that gets filtered has to be the group -- not the line.
    """
    groups: list[list[str]] = []
    for line in lines:
        if MESSAGE.match(line) or not groups:
            groups.append([line])
        else:
            groups[-1].append(line)

    kept: list[str] = []
    for group in groups:
        head = group[0]
        if not head.strip() or NOISE.search(head):
            continue
        if not MESSAGE.match(head):
            # Unclassified chatter, printed by docutils rather than reported
            # through its reporter, so it carries no severity and no location.
            # The only one this package produces is smartquotes' "malformed
            # string literal (missing closing quote)" on a docstring that opens
            # a quote and closes it on the next line. Sphinx reports the same
            # thing at INFO, which -W does not act on, so it is genuinely not a
            # finding -- but it is dropped here by rule rather than by accident.
            continue
        kept.append("\n    ".join(part for part in group if part.strip()))
    return kept


class _Collector:
    """A write-only stream docutils reports into."""

    def __init__(self, into: list[str]) -> None:
        self._into = into

    def write(self, text: str) -> None:
        self._into.extend(text.split("\n"))

    def flush(self) -> None:
        """docutils calls this; nothing is buffered."""


def attribute_comments(source: str) -> list[tuple[int, str]]:
    """Every ``#:`` block, as (line number, dedented text).

    Read from the raw source rather than the AST, because a ``#:`` comment is a
    comment -- the AST has already discarded it, and autodoc recovers it by
    re-parsing the file for exactly this reason.
    """
    blocks: list[tuple[int, str]] = []
    current: list[str] = []
    start = 0
    for number, line in enumerate(source.split("\n"), start=1):
        match = COMMENT.match(line)
        if match:
            if not current:
                start = number
            current.append(match.group(1))
        elif current:
            blocks.append((start, "\n".join(current)))
            current = []
    if current:
        blocks.append((start, "\n".join(current)))
    return blocks


def docstrings(path: Path) -> list[tuple[int, str, str]]:
    """Every docstring in a file, as (line, qualified-ish name, text)."""
    source = path.read_text()
    found: list[tuple[int, str, str]] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        text = ast.get_docstring(node)
        if text:
            name = getattr(node, "name", "<module>")
            found.append((getattr(node, "lineno", 1), name, text))
    for line, text in attribute_comments(source):
        found.append((line, "#: comment", text))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--verbose", action="store_true", help="print every file checked")
    parser.add_argument(
        "paths", nargs="*", type=Path, default=None, help="files to check (default: visbench/)"
    )
    args = parser.parse_args()

    files = args.paths or sorted(PACKAGE.rglob("*.py"))
    problems = 0
    for path in files:
        for line, name, text in docstrings(path):
            reported = findings(text)
            if not reported:
                continue
            problems += 1
            relative = path.relative_to(REPO) if path.is_absolute() else path
            print(f"\n{relative}:{line}  {name}")
            for message in reported:
                print(f"    {message.strip()}")
        if args.verbose:
            print(f"checked {path}")

    if problems:
        print(f"\n{problems} docstring(s) would warn in the docs build.", file=sys.stderr)
        return 1
    print(f"{len(files)} files: every docstring parses clean as nested content.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

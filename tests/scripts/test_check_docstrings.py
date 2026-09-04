"""Every docstring must survive the nested parse autodoc puts it through.

The docs site is built with `-W`, so a docstring that docutils complains about
takes the site down. But the docs build runs in **its own workflow**
(`docs.yml`), not in `ci.yml` — so without this, a docstring edit breaks the
site a workflow later, and for whoever pushes next rather than whoever wrote it.
That is the same reasoning as the standing rule that a guard against a silently
wrong number belongs in the fast suite: `scripts/check_docstrings.py` runs in
about a second and needs no Sphinx, so there is no reason to learn this late.

It is deliberately **not** a substitute for the docs build. It sees one
docstring at a time, so it cannot see a duplicated `automodule`, an orphan page
or a broken toctree — only defects inside a docstring.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_docstrings.py"


def test_every_docstring_parses_as_nested_content():
    """The guard itself, run the way CI would.

    A subprocess rather than an import, matching how `tests/test_readme.py`
    runs `render_tables.py --check`: the thing being tested is the script's
    exit code, which is what a workflow would act on.
    """
    result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, (
        "A docstring would warn in the -W docs build.\n"
        "Run scripts/check_docstrings.py for the locations.\n"
        f"{result.stdout}{result.stderr}"
    )


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("check_docstrings", SCRIPT)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_it_would_catch_a_section_title(module):
    """The defect class that motivated the script, and the one it is easiest to
    write by accident: a heading in a docstring.

    Legal at the top level of a document, a SEVERE one level in — which is
    where autodoc puts it. Two of these shipped (`FeatureDict`'s `Keys` and
    `SCHEMA_VERSION`'s `History`) and neither was visible in an editor.
    """
    assert module.findings("Heading\n-------\n\nbody.\n")


def test_it_would_catch_a_malformed_table(module):
    """A column border narrower than its own header cell. Two shipped."""
    bad = "====  ====\nlonger  b\n====  ====\nx     y\n====  ====\n"
    assert module.findings(bad)


def test_it_passes_a_clean_docstring(module):
    """The anti-vacuity half: a checker that flags everything is no checker.

    This is the shape most of the package's docstrings have — numpydoc
    sections, a literal, a cross-reference role.
    """
    clean = (
        "One line.\n\n"
        "Parameters\n----------\n"
        "value : int\n    A number.\n\n"
        "Returns\n-------\n"
        "bool\n    Whether it worked. See :meth:`fit`, and ``literal`` text.\n"
    )
    assert module.findings(clean) == []


def test_the_indent_is_what_makes_it_work(module):
    """The whole trick, pinned — because removing it looks like a simplification.

    `as_autodoc_sees_it` indents napoleon's output under a dummy directive,
    because that is how autodoc inserts a docstring. Un-nested, a section title
    is *legal* and docutils reports nothing, so a checker without the indent
    passes every docstring in the package including the two that broke the
    build.
    """
    from docutils.core import publish_doctree

    heading = "Heading\n-------\n\nbody.\n"
    collected: list[str] = []
    publish_doctree(
        heading,
        settings_overrides={
            "report_level": 2,
            "halt_level": 5,
            "warning_stream": module._Collector(collected),
            "input_encoding": "unicode",
        },
    )
    assert module._real_messages(collected) == [], "un-nested, a title is legal"
    assert module.findings(heading), "nested, it must be a finding"


def test_it_drops_sphinx_only_roles_whole(module):
    """A message is several lines, so the unit filtered has to be the message.

    The first draft filtered line by line: it dropped an "unknown directive"
    severity line and kept that message's *context* lines, reporting two files'
    worth of `.. attribute::` bodies as findings. A checker that cries wolf is
    the least useful thing to put in a gating suite.
    """
    assert module.findings("See :meth:`fit` and :nosuchrole:`x`.\n") == []
    assert module.findings(".. attribute:: thing\n\n   Its documentation.\n") == []

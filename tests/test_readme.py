"""The README is package metadata, and PyPI resolves its links differently.

``pyproject.toml`` names ``README.md`` as the long description, so every link
and image in it is rendered on the project page as well as on GitHub. PyPI
serves that page from ``pypi.org``, where a relative path resolves against
nothing useful — the image 404s and the link leads somewhere that does not
exist. GitHub resolves the same path against the repository and looks fine, so
the mistake is invisible in every place it is normally read.

CI's ``build`` job runs ``twine check``, which is the renderer PyPI itself uses,
and it does **not** catch this: a relative link is valid markdown and renders
without complaint. It only points nowhere. A version number on PyPI can never
be reused, so a broken link ships until the next release — this was fixed by
hand for v0.2.0 and nothing stopped it coming back.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"

# ``[text](target)``, and its ``![alt](target)`` image form, which differs only
# by a leading ``!`` and so is matched by the same pattern.
MARKDOWN_LINK = re.compile(r"\]\(([^)\s]+)")

# The README opens with a raw HTML block for the logo and badges, which the
# markdown pattern above cannot see.
HTML_ATTR = re.compile(r'(?:src|href)="([^"]+)"')

# An anchor is resolved by the browser against the page already being shown, so
# it is correct on both hosts. A mailto has no host to resolve against at all.
ALLOWED_PREFIXES = ("https://", "http://", "mailto:", "#")


def readme_targets() -> list[str]:
    text = README.read_text(encoding="utf-8")
    return MARKDOWN_LINK.findall(text) + HTML_ATTR.findall(text)


@pytest.mark.parametrize("target", readme_targets())
def test_readme_link_is_absolute(target):
    """No link or image in the README may be relative.

    If this fails, do not "tidy" the path back to a relative one — point it at
    ``https://github.com/turhancan97/VisBench/blob/main/...`` (or
    ``raw.githubusercontent.com`` for an image), which is what the rest of the
    file does and what renders on both hosts.
    """
    assert target.startswith(ALLOWED_PREFIXES), (
        f"README links must be absolute so they work on PyPI as well as GitHub; "
        f"{target!r} is relative and would 404 on the project page."
    )


def test_the_pattern_finds_the_links_it_claims_to():
    """A regex that matched nothing would pass the test above forever.

    The guard this file exists to provide is only as good as its ability to see
    a link at all — the same failure as a warning filter matching a phrase that
    is never emitted. So assert the extraction found both syntaxes and roughly
    the number of links the README actually has.
    """
    targets = readme_targets()
    assert len(targets) > 20, "the README has more links than this; the pattern missed some"
    assert any("arxiv.org" in t for t in targets), "no markdown link found"
    assert any("raw.githubusercontent.com" in t for t in targets), "no HTML image found"


# --------------------------------------------------------------------------
# Generated tables (step 6e-3)
# --------------------------------------------------------------------------


def test_the_generated_tables_match_the_corpus():
    """The README's measured numbers must equal what the records say.

    Every one of these was hand-copied from a terminal until step 6e-3, and one
    had drifted by the time anyone noticed. This is the guard that makes drift
    impossible rather than merely unlikely, and it belongs in the *fast* suite:
    a wrong number ships to PyPI on the next release, and a version there can
    never be reused.

    If this fails, run ``scripts/render_tables.py`` — do not edit the tables by
    hand, since the next regeneration would overwrite the edit.
    """
    script = Path(__file__).resolve().parent.parent / "scripts" / "render_tables.py"
    result = subprocess.run(
        [sys.executable, str(script), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"README tables are stale against results/corpus/visbench.jsonl.\n"
        f"Run scripts/render_tables.py to regenerate.\n{result.stdout}{result.stderr}"
    )


def _marked_files():
    """The files the generator itself scans, not a second list that can drift."""
    import importlib.util

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "render_tables", root / "scripts" / "render_tables.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MARKED_FILES


def test_every_marker_is_closed_and_carries_a_task():
    """A malformed marker renders nothing and fails silently.

    The regex simply would not match it, so the stale table underneath survives
    and ``--check`` passes — which is the failure mode the generator exists to
    remove, reintroduced by a typo.
    """
    for path in _marked_files():
        text = path.read_text()
        opens = re.findall(r"<!-- visbench:board ([^>]*?) -->", text)
        assert len(opens) == text.count("<!-- /visbench:board -->"), f"unbalanced in {path.name}"
        assert opens, f"no generated boards in {path.name}"
        for attrs in opens:
            assert "task=" in attrs, f"marker without a task in {path.name}: {attrs!r}"


def test_the_readme_links_to_the_docs_it_split_into():
    """The reorganisation is only safe if the pointers exist.

    Moving 500 lines into docs/ helps nobody if the README does not say where
    they went, and a typo in one of those paths is invisible on GitHub until
    someone clicks it.
    """
    root = Path(__file__).resolve().parent.parent
    readme = (root / "README.md").read_text()
    for name in ("docs/tasks.md", "docs/roadmap.md", "LEADERBOARD.md", "CHANGELOG.md"):
        assert (root / name).is_file(), f"{name} does not exist"
        assert name in readme, f"the README never points at {name}"


def _docs_links():
    """Every markdown link target under docs/, recursively, with its page."""
    root = Path(__file__).resolve().parent.parent / "docs"
    found = []
    for page in sorted(root.rglob("*.md")):
        for target in re.findall(r"\]\(([^)\s]+)\)", page.read_text()):
            found.append((page, target))
    return found


def test_no_docs_page_escapes_the_sphinx_tree():
    """`docs/` is a Sphinx source directory, so `../` cannot resolve.

    MyST leaves an unresolvable relative path as a literal href and emits no
    warning, so the `-W` build will not catch it either — it simply 404s on the
    published site. Anything outside the tree must be an absolute URL.
    """
    for page, target in _docs_links():
        assert not target.startswith("../"), (
            f"{page.name} links to {target!r}, which escapes the docs tree and "
            "cannot resolve on the site. Use an absolute GitHub URL instead."
        )


def test_relative_docs_links_resolve():
    """Intra-site links must point at a page that exists."""
    for page, target in _docs_links():
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        resolved = (page.parent / target.split("#")[0]).resolve()
        assert resolved.exists(), f"{page.name} -> {target}"


def test_the_docs_link_pattern_finds_something():
    """Both tests above pass trivially if the extraction sees no links.

    This is not hypothetical: the previous version of this check matched only
    `../`-prefixed targets, and became vacuous the moment those were converted
    to absolute URLs for the Sphinx build. It passed while checking nothing.
    """
    links = _docs_links()
    assert len(links) > 15, f"only {len(links)} links found under docs/; the pattern missed some"
    assert any(t.startswith("https://") for _, t in links), "no absolute link found"
    assert any(not t.startswith(("http", "#")) for _, t in links), "no intra-site link found"

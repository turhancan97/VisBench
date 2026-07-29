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

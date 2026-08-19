"""Every gallery photograph is one this repository may redistribute, and credits it.

The docs gallery is the only place VisBench ships third-party imagery. That is
allowed because the frames are CC BY 2.0 — and CC BY is not a licence you comply
with by intending to: it requires attribution, per work, wherever the work is
redistributed. So the obligation is exactly the kind of thing that rots quietly.
Someone adds a frame to `scripts/fetch_gallery_frames.py`, the figure renders,
the page looks right, and the repository is out of compliance with nothing to
show for it.

These tests are the guard. They read what is *committed* rather than re-running
the fetcher, because what is committed is what is redistributed.
"""

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FRAMES = REPO / "assets" / "gallery_frames"
CREDITS = FRAMES / "CREDITS.md"
MANIFEST = FRAMES / "frames.json"

#: Kept in step with `ALLOWED_LICENCES` in the fetcher by the test below. Two
#: copies on purpose: this one is what the repository *claims*, and duplicating
#: it is what makes widening the fetcher's allowlist a visible change here.
ALLOWED = {
    "https://creativecommons.org/licenses/by/2.0/",
    "https://creativecommons.org/publicdomain/zero/1.0/",
    "https://creativecommons.org/publicdomain/mark/1.0/",
}

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="gallery frames are not fetched; run scripts/fetch_gallery_frames.py",
)


def _images() -> list[str]:
    return sorted(path.stem for path in (FRAMES / "images").glob("*.jpg"))


def _credited() -> set[str]:
    return set(re.findall(r"^\| `([0-9a-f]+)`", CREDITS.read_text(), re.MULTILINE))


def test_every_committed_photograph_is_credited():
    """The CC BY obligation, as a build failure.

    A frame present in `images/` and absent from `CREDITS.md` is a photograph
    this repository redistributes without the attribution its licence requires.
    """
    missing = sorted(set(_images()) - _credited())
    assert not missing, f"photographs with no credit: {missing}"


def test_nothing_is_credited_that_is_not_here():
    """The other direction, which is staleness rather than breach.

    A credit for a frame that has been removed is a claim about content that is
    not in the repository, and it is how the file drifts into being decorative.
    """
    extra = sorted(_credited() - set(_images()))
    assert not extra, f"credits for photographs that are not here: {extra}"


def test_every_credit_names_an_allowed_licence():
    body = CREDITS.read_text()
    rows = [line for line in body.splitlines() if line.startswith("| `")]
    assert rows, "CREDITS.md lists no photographs"
    for row in rows:
        assert any(licence in row for licence in ALLOWED), row


def test_every_credit_carries_an_author_and_a_source():
    """Attribution is a name and a link, not a licence badge on its own."""
    for row in (line for line in CREDITS.read_text().splitlines() if line.startswith("| `")):
        cells = [cell.strip() for cell in row.split("|")[1:-1]]
        assert len(cells) == 4, row
        _photograph, author, _licence, source = cells
        assert author, f"no author: {row}"
        assert "http" in source, f"no source link: {row}"


def test_the_manifest_and_the_images_agree():
    """A manifest naming a frame that was never fetched renders a broken page."""
    manifest = json.loads(MANIFEST.read_text())
    assert sorted(manifest) == _images()


def test_the_fetchers_allowlist_is_the_one_documented_here():
    """Widening the fetcher must be a visible change, not a silent one."""
    from scripts.fetch_gallery_frames import ALLOWED_LICENCES

    assert ALLOWED_LICENCES == ALLOWED

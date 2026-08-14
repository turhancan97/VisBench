"""The committed figure gallery, and that it covers every probe.

The figures themselves are not regenerated here — that needs a backbone and a
scene render, which is `scripts/render_gallery.py`'s job. What the fast suite
can guarantee is that the set on disk matches the set of probes, so adding a
probe and forgetting its figure fails a build rather than shipping a docs page
with a gap in it. The same posture as the generated leaderboard tables.
"""

from pathlib import Path

import pytest

import visbench

GALLERY = Path(__file__).resolve().parent.parent / "docs" / "_static" / "gallery"
SHOW_DOC = Path(__file__).resolve().parent.parent / "docs" / "show.md"
README = Path(__file__).resolve().parent.parent / "README.md"

#: Where the README points, which must be the repository's own raw host. The
#: README is package metadata rendered by PyPI, so a relative path there would
#: render without complaint and point nowhere -- the rule `test_readme.py`
#: enforces generally, and these figures are the newest thing to obey it.
RAW = "https://raw.githubusercontent.com/turhancan97/VisBench/main/docs/_static/gallery"


def test_there_is_a_figure_for_every_probe():
    """A missing figure is a hole in the docs that nothing else would catch."""
    drawn = {path.stem for path in GALLERY.glob("*.png")}
    assert drawn == set(visbench.list_probes())


def test_the_gallery_holds_nothing_else():
    """A figure for a probe that no longer exists would quietly go stale."""
    for path in GALLERY.iterdir():
        assert path.suffix == ".png", path
        assert path.stem in visbench.list_probes(), path


@pytest.mark.parametrize("probe", sorted(visbench.list_probes()))
def test_the_docs_page_shows_each_one(probe):
    assert f"_static/gallery/{probe}.png" in SHOW_DOC.read_text()


def test_the_docs_page_uses_relative_paths():
    """Sphinx cannot follow a path that escapes its source tree.

    `../assets/...` is valid markdown, renders locally, and breaks the built
    site without a warning -- MyST does not report it, so `-W` would not catch
    it either. Keeping the figures under `docs/_static` is what avoids it.
    """
    body = SHOW_DOC.read_text()
    assert "../assets" not in body
    assert f"]({RAW}" not in body, "the docs site should not fetch its own figures over the network"


def test_the_readme_points_at_the_raw_host():
    body = README.read_text()
    referenced = [line for line in body.splitlines() if "_static/gallery" in line]
    assert referenced, "the README shows no figures"
    for line in referenced:
        assert RAW in line, line


def test_every_readme_figure_exists():
    body = README.read_text()
    for probe in visbench.list_probes():
        if f"{RAW}/{probe}.png" in body:
            assert (GALLERY / f"{probe}.png").is_file()


def test_the_figures_are_small_enough_to_commit():
    """Documentation, not data. A figure that balloons should be noticed here.

    The sdist excludes this directory precisely because it is not needed to
    install the package, but the repository still carries it.
    """
    total = sum(path.stat().st_size for path in GALLERY.glob("*.png"))
    assert total < 4_000_000, f"the gallery is {total / 1e6:.1f} MB"

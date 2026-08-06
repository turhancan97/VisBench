"""Citation metadata has to describe the version it ships with.

A DOI is permanent and a released archive cannot be edited, so a `CITATION.cff`
naming the previous version is wrong in the one place being wrong is expensive:
someone cites v0.6.1 and gets an archive of v0.7.0's code, or the reverse. The
failure is silent — GitHub renders the button, Zenodo mints the DOI, and
everything looks fine.

These are the same class of check as `tests/test_readme.py`: metadata nobody
executes, so nothing else would catch it.
"""

import json
import re
from pathlib import Path

import pytest

import visbench

ROOT = Path(__file__).resolve().parents[1]
CITATION = ROOT / "CITATION.cff"
ZENODO = ROOT / ".zenodo.json"
ORCID = "0000-0002-6751-4773"


@pytest.fixture(scope="module")
def citation() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(CITATION.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def zenodo() -> dict:
    return json.loads(ZENODO.read_text(encoding="utf-8"))


def test_the_cited_version_is_the_shipped_version(citation):
    """The one that actually breaks: a bump that touches only `__init__.py`.

    `uv.lock` already forces a re-lock on a version bump, and this forces the
    citation with it.
    """
    assert str(citation["version"]) == visbench.__version__, (
        f"CITATION.cff cites {citation['version']}, but visbench.__version__ is "
        f"{visbench.__version__}. Bump both in the release commit."
    )


def test_the_release_date_is_a_real_date(citation):
    """`date-released` is what Zenodo shows as the publication date."""
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(citation["date-released"]))


def test_the_author_carries_an_orcid(citation, zenodo):
    """An ORCID is what disambiguates an author across papers and archives.

    Checked on both files because they are read by different consumers —
    GitHub's citation button and Zenodo's deposit form — and a name without an
    identifier is what makes a software citation hard to credit later.
    """
    (author,) = citation["authors"]
    assert author["orcid"] == f"https://orcid.org/{ORCID}"

    (creator,) = zenodo["creators"]
    # Zenodo wants the bare identifier; CFF wants the resolvable URL. Neither
    # accepts the other's form, which is exactly why this is pinned.
    assert creator["orcid"] == ORCID


def test_the_two_files_agree_on_the_work_they_describe(citation, zenodo):
    """Two files, one deposit. Zenodo reads .zenodo.json in preference to
    CITATION.cff, so a divergence between them is invisible until the archive
    is already published under a title nobody chose."""
    assert citation["title"] == zenodo["title"]
    assert citation["license"] == zenodo["license"]


def test_the_licence_is_the_one_in_the_repository(citation):
    """MIT here and something else in LICENSE would be a licensing claim, not
    a typo."""
    assert citation["license"] == "MIT"
    assert "MIT License" in (ROOT / "LICENSE").read_text(encoding="utf-8")


def test_the_readme_tells_people_how_to_cite(citation):
    """The button is only found by people who already know it exists."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Citing VisBench" in readme, "the README has no citation section"
    assert citation["title"] in readme, "the README's BibTeX title drifted from CITATION.cff"


def test_zenodo_metadata_is_complete_enough_to_deposit(zenodo):
    """Zenodo rejects a deposit missing any of these, and the failure arrives
    as a broken webhook after the release is already tagged."""
    for field in ("title", "description", "upload_type", "access_right", "license", "creators"):
        assert zenodo.get(field), f".zenodo.json is missing {field!r}"
    assert zenodo["upload_type"] == "software"

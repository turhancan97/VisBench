"""The Cars split is VisBench's own, so what is in it has to be checkable.

`vehicle_classification` cannot claim Stanford Cars' official 8,144/8,041 split:
the copy on this machine holds 8,148 train images, eleven of which are the same
image filed under two class directories, seven more such pairs in test, and one
blank placeholder. `scripts/stage_cars_split.py` pins a cleaned **8,125 / 8,026**
split and writes `data/cars_split_manifest.json` naming every exclusion.

That manifest is the protocol — two people's numbers are comparable only if they
staged the same files, exactly as `corner`'s frame set is pinned. So it is
committed and checked here, and when the raw dataset is present the exclusions
are **recomputed from it** rather than trusted, because a copy that is
*differently* broken must not be staged as though it were this one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "cars_split_manifest.json"
STAGED = ROOT / "data" / "cars_split"

EXPECTED = {"train": {"raw_files": 8148, "kept": 8125}, "val": {"raw_files": 8041, "kept": 8026}}


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not MANIFEST.is_file():
        pytest.skip(f"no manifest at {MANIFEST}")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_the_manifest_records_the_split_the_probe_claims(manifest):
    for split, counts in EXPECTED.items():
        got = manifest["splits"][split]
        assert got["raw_files"] == counts["raw_files"], split
        assert got["kept"] == counts["kept"], split
        assert got["raw_files"] - len(got["excluded"]) == got["kept"], split


def test_every_exclusion_states_why(manifest):
    """A dropped file with no reason is a split nobody can audit."""
    for split, got in manifest["splits"].items():
        for entry in got["excluded"]:
            assert entry["reason"], f"{split}/{entry['path']} dropped without a reason"
            assert entry["reason"].startswith(("same image under classes", "duplicate of")) or (
                entry["reason"] == "blank image"
            ), entry


def test_the_split_is_not_the_official_one_and_says_so(manifest):
    """The whole point of the manifest: 8,148 is not Stanford Cars' 8,144.

    If someone ever stages the official split here, this test fails and the
    docstring, the docs page and the probe's own docstring all need rewriting —
    which is the correct amount of friction for a claim about comparability.
    """
    assert manifest["splits"]["train"]["raw_files"] != 8144
    source = ROOT / "scripts" / "stage_cars_split.py"
    text = source.read_text(encoding="utf-8")
    assert "not comparable with" in text.lower() or "NOT comparable" in text


@pytest.mark.slow
def test_the_exclusions_recompute_from_the_raw_dataset(manifest):
    """Trust the dataset, not the file: a differently broken copy must not pass."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import stage_cars_split as staging

    data = Path(manifest["source"])
    if not data.is_dir():
        pytest.skip(f"no Stanford Cars copy at {data}")
    for split, folder in staging.SOURCES.items():
        report = staging.audit(data / folder)
        drop = staging.excluded(report)
        recorded = {entry["path"] for entry in manifest["splits"][split]["excluded"]}
        assert {str(p.relative_to(data / folder)) for p in drop} == recorded, split


def test_the_staged_split_holds_only_symlinks():
    """Symlinks, not copies: a staged image shares its cache entry with the original."""
    if not STAGED.is_dir():
        pytest.skip(f"nothing staged at {STAGED}")
    for split in ("train", "val"):
        links = list((STAGED / split).rglob("*.png"))
        assert links, split
        assert all(p.is_symlink() for p in links[:200]), f"{split} holds real files"

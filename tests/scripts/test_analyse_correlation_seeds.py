"""The correlation audit (22b), and the prose that quotes it.

`scripts/analyse_correlation_seeds.py` re-reads every correlational claim in
`CORPUS_FINDINGS.md` on the committed seed sweeps. Its numbers went straight
into three files, which is exactly how every count in this project has gone
stale: so the counts are recomputed here, through the script itself rather
than a second copy of its arithmetic, and checked against the files that quote
them.

The verdicts are pinned as well. If a future sweep or a new backbone column
turns a robust claim fragile, the prose that calls it robust is now wrong, and
nothing else would say so.

The script is loaded by path because `scripts/` is not a package.
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analyse_correlation_seeds.py"
FINDINGS = ROOT / "CORPUS_FINDINGS.md"
GUIDE = ROOT / "docs" / "guides" / "reading-a-board.md"

#: Fewer than the script's default: enough to separate "every draw" from "most",
#: few enough for the fast suite.
DRAWS = 300

#: The one claim 22b found unsupported, and why it is listed rather than
#: derived: the prose was rewritten around it, so if it ever becomes robust the
#: correction is what needs revisiting.
FAILING = {"vehicle_closest_is_place"}


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("analyse_correlation_seeds", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: a dataclass resolves its own module by name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    data = module.load()
    views = module.draws(data, DRAWS, 22)
    return {
        "module": module,
        "data": data,
        "views": views,
        "claims": {r["key"]: r for r in module.evaluate_claims(data, views)},
        "matrix": module.matrix(data),
    }


def test_every_claim_but_the_corrected_one_holds_on_every_draw(audit):
    verdict = audit["module"].verdict
    robust = {k for k, r in audit["claims"].items() if verdict(r) == "robust"}
    assert robust == set(audit["claims"]) - FAILING, (
        "a claim CORPUS_FINDINGS.md reads as holding on every re-fit no longer does, "
        f"or the corrected one now does: robust = {sorted(robust)}"
    )


def test_the_corrected_claim_is_a_tie_on_the_seed_mean(audit):
    row = audit["claims"]["vehicle_closest_is_place"]
    assert abs(row["seed_mean"]) < 0.01
    assert row["holds"] < 0.95


def test_the_claim_count_matches_the_prose(audit):
    total = len(audit["claims"])
    holding = total - len(FAILING)
    findings = " ".join(FINDINGS.read_text(encoding="utf-8").split())
    guide = " ".join(GUIDE.read_text(encoding="utf-8").split())
    assert f"**{holding} of the {total} claims tested hold on every" in findings
    # The guide spells them out; a new claim must update both files and this map.
    spelled = {15: "fifteen", 16: "sixteen"}
    assert f"Of the {spelled[total]} correlational readings tested, {spelled[holding]} hold" in (
        guide
    )


def test_the_matrix_totals_match_the_prose(audit):
    varying = [r for r in audit["matrix"] if r["n"] > 1]
    flips = sum(1 for r in varying if r["sign_flips"])
    strong = [r for r in varying if abs(r["published"]) >= 0.5]
    strong_flips = sum(1 for r in strong if r["sign_flips"])
    median = statistics.median(r["width"] for r in varying)
    findings = " ".join(FINDINGS.read_text(encoding="utf-8").split())
    guide = " ".join(GUIDE.read_text(encoding="utf-8").split())

    assert f"**{flips} of {len(varying)}**" in findings
    assert f"{flips} of {len(varying)} pairs change sign" in guide
    assert f"**{median:.3f}**" in findings
    assert f"**{median:.2f}**" in guide
    assert strong_flips == 0, "the prose says no |rho| >= 0.5 pair changes sign"
    assert f"none of the {len(strong)} published at |rho| ≥ 0.5" in findings


def test_the_published_draws_are_not_biased_high(audit):
    """The handoff that proposed 22b suspected they were, from two pairs."""
    varying = [r for r in audit["matrix"] if r["n"] > 1]
    mean = statistics.fmean(r["percentile"] for r in varying)
    assert 0.45 <= mean <= 0.55
    assert f"mean percentile of **{mean:.2f}**" in " ".join(
        FINDINGS.read_text(encoding="utf-8").split()
    )


def test_zero_shot_boards_are_identical_in_every_draw(audit):
    data = audit["data"]
    for view in audit["views"][:20]:
        for task in audit["module"].ZERO_SHOT:
            assert view.boards[task] == data.published[task]


def test_the_box_half_takes_the_mask_half_seed(audit):
    """They are the same runs; drawing them independently would decouple a pair
    that the instance entry quotes at +0.986 precisely because it is coupled."""
    module, data = audit["module"], audit["data"]
    for view in audit["views"][:50]:
        mask = view.boards["instance_segmentation"]
        (seed,) = [s for s in module.SEEDS if data.seeded["instance_segmentation"][s] is mask]
        assert view.boards[module.BOX] is data.seeded[module.BOX][seed]


def test_a_trained_board_without_a_sweep_is_refused(audit, tmp_path, monkeypatch):
    module = audit["module"]
    monkeypatch.setattr(module, "sweep_paths", lambda: {})
    with pytest.raises(SystemExit, match="has no committed sweep"):
        module.load()

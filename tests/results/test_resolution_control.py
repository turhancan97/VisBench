"""The resolution control, pinned as a finding.

`results/controls/resolution.jsonl` exists to answer one question the corpus
cannot: feature resolution is the strongest correlate of every dense board
(rho +0.50 to +0.96), and the only backbones carrying 256 tokens are the two
DINOv2s -- so grid size, the DINOv2 objective and LVD-142M pretraining were one
variable.

Two properties are tested here, and they pull in opposite directions on
purpose. The control must be *rankable* against the corpus, or it answers
nothing. It must also stay *out* of the corpus, or it becomes a thirteenth
competitor in every generated table. Both are easy to break by accident, and
neither breaks loudly.
"""

import json
from pathlib import Path

import pytest

from visbench.results.leaderboard import comparability_key, latest_per_backbone
from visbench.results.writer import read_records

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"
CONTROL = ROOT / "results" / "controls" / "resolution.jsonl"

#: board -> (headline metric, the 224px number, the 196px number).
#: Literals rather than a re-read, so a corpus edit that moves either side has
#: to be acknowledged here instead of silently redefining the finding.
MEASURED: dict[str, tuple[str, float, float]] = {
    "generic_segmentation": ("iou", 0.7556, 0.7407),
    "depth": ("d1", 0.7851, 0.7791),
    "surface_normal": ("mean", 30.1143, 30.6556),
    "edge": ("edge_correlation", 0.4481, 0.4363),
    "corner": ("corner_correlation", 0.6526, 0.6349),
}


@pytest.fixture(scope="module")
def control():
    return read_records(CONTROL)


@pytest.fixture(scope="module")
def corpus():
    return read_records(CORPUS)


def test_the_control_is_not_in_the_corpus(corpus):
    """A control in the corpus is a thirteenth row in every generated table.

    `scripts/render_tables.py` and `LEADERBOARD.md` read the corpus, so this is
    the whole mechanism keeping a control out of the published boards. Nothing
    else would fail if the two files were concatenated -- the tables would
    simply regenerate, correctly, around a row that answers a different
    question from the twelve beside it.
    """
    assert not any(r.backbone == "dinov2_vitb14_196" for r in corpus)


def test_the_control_is_rankable_against_the_corpus(control, corpus):
    """Kept apart editorially, not because the rules refuse it.

    If these ever landed in different comparability groups the control would be
    measuring something else -- a different split, schedule or protocol -- and
    the comparison in the README would be meaningless rather than merely
    unpublished.
    """
    for record in control:
        against = [r for r in corpus if r.task == record.task]
        assert against, f"no corpus board for {record.task}"
        assert {comparability_key(r).short_id() for r in against} == {
            comparability_key(record).short_id()
        }, f"{record.task}: the control is not in its board's comparability group"


def test_the_control_adds_a_row_rather_than_evicting_one(control, corpus):
    """The failure the `name=` override exists to prevent, end to end.

    Before `DINOv2` took a name, this configuration reported `dinov2_vitb14`
    and `latest_per_backbone` -- which keeps the newest per name -- would have
    dropped the corpus's 224px number from all five boards.
    """
    for task in MEASURED:
        rows = [r for r in corpus if r.task == task]
        merged = latest_per_backbone(rows + [r for r in control if r.task == task])
        # Against the corpus's *backbone* count, not its line count. The corpus is
        # append-only and a re-run board carries two lines per backbone -- which
        # `latest_per_backbone` is precisely what collapses. Comparing against
        # `len(rows)` passed only while no board had ever been re-run, and broke
        # the first time one was (the five low-level boards, regenerated to carry
        # `ceiling_*`). The claim being made here was never about lines.
        assert len(merged) == len(latest_per_backbone(rows)) + 1
        assert {"dinov2_vitb14", "dinov2_vitb14_196"} <= {r.backbone for r in merged}


@pytest.mark.parametrize("task", sorted(MEASURED))
def test_the_recorded_numbers_are_what_the_control_measured(task, control, corpus):
    metric, at224, at196 = MEASURED[task]
    got224 = next(r for r in corpus if r.task == task and r.backbone == "dinov2_vitb14")
    got196 = next(r for r in control if r.task == task)
    assert got224.metrics[metric] == pytest.approx(at224, abs=5e-5)
    assert got196.metrics[metric] == pytest.approx(at196, abs=5e-5)


def test_matching_the_grid_costs_dinov2_under_three_percent():
    """The finding itself: the token correlation is mostly not causal here.

    Stated as a bound rather than five numbers, because the claim that matters
    is the size of the effect, not its exact value on any one board. If a
    future re-run pushes any board past 3% this should be updated deliberately
    -- the write-up in `results/controls/README.md` says "under 3%" and would
    be wrong.
    """
    for task, (_, at224, at196) in MEASURED.items():
        relative = abs(at196 - at224) / abs(at224)
        assert relative < 0.03, f"{task} moved {relative:.1%}, which the write-up denies"


def test_dinov2_keeps_its_lead_on_the_boards_it_led(corpus):
    """21% of the lead on generic_segmentation, 7% on depth -- not all of it.

    These are the only two boards where DINOv2-B led, and therefore the only
    two where the confound was ever real. `mae_vitb16` is ahead on the other
    three, so there was no lead for resolution to explain.
    """
    for task, metric, rival in (
        ("generic_segmentation", "iou", "dino_vitb16"),
        ("depth", "d1", "mae_vitb16"),
    ):
        _, at224, at196 = MEASURED[task]
        best_rival = max(
            r.metrics[metric]
            for r in corpus
            if r.task == task and not r.backbone.startswith("dinov2")
        )
        assert at196 > best_rival, f"{task}: the control fell into the pack"
        explained = (at224 - at196) / (at224 - best_rival)
        assert 0.05 < explained < 0.25, f"{task}: grid explains {explained:.0%}"
        assert best_rival == max(
            r.metrics[metric] for r in corpus if r.task == task and r.backbone == rival
        )


def test_every_control_record_names_a_distinct_backbone():
    """A control whose backbone forgot its name is indistinguishable from a re-run."""
    rows = [
        json.loads(line)
        for line in CONTROL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows, "the control file is empty"
    assert {r["backbone"] for r in rows} == {"dinov2_vitb14_196"}
    assert {r["backbone_key"] for r in rows} == {"dinov2/dinov2_vitb14/196/7764ea0f912e"}

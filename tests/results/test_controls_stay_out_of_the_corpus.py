"""No control record may appear in the corpus, and a merge nearly put three there.

`results/controls/` holds records that are deliberately **not** corpus records.
Some answer a different question with the same probe (`detection_split` on other
images, `dpt_head` with another head); one is a backbone under a different
configuration (`resolution`); one is an unregistered probe (`relative_depth`);
and one -- `hardware_a100` -- is three cells of a re-run that **did not
reproduce** the value they were re-running, held out because publishing them
would have dropped `convnext_base` below `resnet50` on the CUB board, a ranking
change caused by a variable no record carried.

**Why this file exists.** On 2026-09-16, widening the corpus with a new backbone
ran `scripts/merge_corpus.sh` as documented and it added **58** records where 19
were expected. `results/corpus/parts/` still held the 2026-09-10 files from that
re-run, including all three A100 cells, and the merge takes everything it finds
there. It deduplicates by exact JSON line, which stops it re-adding what is
*already* in the corpus and does nothing whatever about records deliberately
kept *out* of it -- so the one protection it has is blind to the one mistake
that matters. It was caught by diffing the merge against the corpus before
trusting it, which is a habit rather than a mechanism.

**The check has to come from somewhere other than the merge.** A guard inside
`merge_corpus.sh` would read the same `parts/` directory it is merging, and a
directory that contains an excluded record is self-consistent -- the same shape
as the corpus-matrix guard that could not see its own short probe list. This
asserts the *outcome* instead: whatever route a record took, a control record
and a corpus record are never the same record.

Two strengths of match, because they fail differently:

* **Exact line** is what a re-merge from `parts/` produces, byte for byte.
* **(task, backbone, timestamp)** catches the same run arriving with a field
  re-serialised -- a record edited by hand, or written by a schema that has
  since gained a field. A held-out record that came back under a new schema
  would be invisible to the exact check and is exactly as wrong.

Neither is a substitute for the other, and both were verified to hold across
all six control files before this was written -- a rejection criterion is
calibrated against what already passes before it is allowed to reject anything.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"
CONTROLS = sorted((ROOT / "results" / "controls").glob("*.jsonl"))


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _run_key(record: dict) -> tuple[str, str, str]:
    """What identifies one *run*, independent of how its JSON is spelled."""
    return (record["task"], record["backbone"], record["timestamp"])


@pytest.fixture(scope="module")
def corpus_lines() -> set[str]:
    return {line.strip() for line in CORPUS.read_text().splitlines() if line.strip()}


@pytest.fixture(scope="module")
def corpus_runs() -> set[tuple[str, str, str]]:
    return {_run_key(r) for r in _records(CORPUS)}


def test_there_are_control_files_to_check():
    """A glob that matches nothing passes forever while checking nothing."""
    assert CONTROLS, "no control files found; has results/controls/ moved?"


@pytest.mark.parametrize("path", CONTROLS, ids=lambda p: p.name)
def test_no_control_record_appears_in_the_corpus_verbatim(path, corpus_lines):
    """The exact-line check -- what a re-merge from `parts/` would produce."""
    shared = [line for line in path.read_text().splitlines() if line.strip() in corpus_lines]
    assert not shared, (
        f"{path.name}: {len(shared)} record(s) are in the corpus verbatim. "
        "A control is not a corpus record; if a merge put them there, revert "
        "the corpus and re-merge from only the parts the step produced."
    )


@pytest.mark.parametrize("path", CONTROLS, ids=lambda p: p.name)
def test_no_control_run_appears_in_the_corpus_under_any_spelling(path, corpus_runs):
    """The run-identity check -- catches a re-serialised or re-schema'd copy."""
    shared = [_run_key(r) for r in _records(path) if _run_key(r) in corpus_runs]
    assert not shared, (
        f"{path.name}: {len(shared)} run(s) appear in the corpus under a "
        f"different spelling: {shared[:3]}. Same task, backbone and timestamp "
        "is the same run, however its JSON is written."
    )


def test_the_three_held_out_cells_are_still_held_out(corpus_runs):
    """The specific exclusion the general checks above were written for.

    Named rather than left to the glob, because this is one of the two
    *dangerous* kinds: `hardware_a100`'s records share task, backbone,
    protocol, seed and fingerprint with published corpus rows and differ only
    in the silicon, which the schema did not record when they were made. They
    merge invisibly and move a published ranking.

    **`pose_seeds.jsonl` is the other, and is not covered here** -- see
    `test_pose_seed_sweep.py`. It carries the published configuration exactly
    and differs only in `seed`, which `comparability_key` does not read, so its
    records land in the *identical* group as the board. Every other control file
    could not collide even if merged: a different backbone name, head or probe
    keeps it in its own comparability group.
    """
    held = _records(ROOT / "results" / "controls" / "hardware_a100.jsonl")
    assert len(held) == 3, f"expected the three re-run cells, found {len(held)}"
    assert {r["task"] for r in held} == {"fine_grained_classification"}
    assert {r["backbone"] for r in held} == {"convnext_base", "dino_vitb16", "dinov2_vitb14"}
    for record in held:
        assert _run_key(record) not in corpus_runs, (
            f"{record['backbone']}'s A100 cell is in the corpus. Merging it drops "
            "convnext_base below resnet50 on the CUB board -- a ranking change "
            "caused by a variable no record carries."
        )

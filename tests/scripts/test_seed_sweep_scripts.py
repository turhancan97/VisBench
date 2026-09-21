"""The seed-sweep matrix, against the two things it has to agree with.

`slurm/seed_sweep.sbatch` re-fits published boards at several seeds (20b). Like
every array here it defines its matrix as shell arrays, and this repository has
already shipped one that was quietly short by a probe for a whole release --
because the guard inside the array sized itself from its *own* list, which is
self-consistent whatever it omits.

So the check has to come from outside the file:

* the **backbones** must be the ones on the board being swept -- read out of
  the corpus, not out of another script, because `slurm/corpus.sbatch` takes
  its backbones from an environment variable and has no list to compare with.
  A sweep short of a backbone is invisible afterwards: every probe present
  still holds every backbone it ran; and
* the **probes** must be exactly the ones with a committed sweep file, so a
  probe cannot be added to the matrix and left unanalysed, or a file kept after
  the matrix stopped producing it.

One more, derived rather than listed: a zero-shot probe trains nothing, so a
seed cannot move it and sweeping it would spend a board's compute measuring
zero. Which probes those are is read off the corpus (`training is None`) rather
than named here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from visbench import list_probes
from visbench.results.writer import iter_records

ROOT = Path(__file__).resolve().parents[2]
SWEEP_SBATCH = ROOT / "slurm" / "seed_sweep.sbatch"
SEEDS_DIR = ROOT / "results" / "controls" / "seeds"
CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"


def _bash_array(text: str, name: str) -> list[str]:
    match = re.search(rf"^{name}=\(\n(.*?)^\)$", text, re.MULTILINE | re.DOTALL)
    assert match is not None, f"no {name}=( ... ) array found"
    entries: list[str] = []
    for line in match.group(1).splitlines():
        line = line.split("#", 1)[0].strip()
        entries.extend(line.split())
    return entries


@pytest.fixture(scope="module")
def sweep_probes() -> list[str]:
    return _bash_array(SWEEP_SBATCH.read_text(), "PROBES")


def test_the_sweep_covers_every_backbone_on_the_boards_it_sweeps(sweep_probes):
    """Same backbones as the boards, read from the corpus rather than another script.

    A separability verdict is about adjacent rows, so a sweep missing a row
    cannot describe the pairs on either side of it -- and nothing downstream
    would say so, because the pairs it *did* measure are all well formed.
    """
    sweep = set(_bash_array(SWEEP_SBATCH.read_text(), "BACKBONE_LIST"))
    for probe in sweep_probes:
        board = {r.backbone for r in iter_records(CORPUS) if r.task == probe}
        assert board, f"no published {probe} board to sweep"
        assert sweep == board, (
            f"the sweep runs {sorted(sweep)} where the {probe} board has "
            f"{sorted(board)}; missing: {sorted(board - sweep)}"
        )


def test_every_swept_probe_is_registered(sweep_probes):
    unknown = sorted(set(sweep_probes) - set(list_probes()))
    assert not unknown, f"unregistered probe(s) in the sweep matrix: {unknown}"


def test_the_matrix_and_the_committed_sweeps_are_the_same_set(sweep_probes):
    """A probe in the matrix with no file is an unanalysed run; the reverse is a stale file."""
    committed = sorted(p.stem for p in SEEDS_DIR.glob("*.jsonl")) if SEEDS_DIR.is_dir() else []
    assert sorted(sweep_probes) == committed, (
        f"matrix has {sorted(sweep_probes)}, results/controls/seeds/ has {committed}"
    )


def test_no_zero_shot_probe_is_swept(sweep_probes):
    """A probe that fits nothing cannot move with the seed, so sweeping it measures zero."""
    zero_shot = {record.task for record in iter_records(CORPUS) if record.training is None} - {
        record.task for record in iter_records(CORPUS) if record.training is not None
    }
    swept_zero_shot = sorted(set(sweep_probes) & zero_shot)
    assert not swept_zero_shot, (
        f"{swept_zero_shot} train no head, so every seed would return the same number"
    )


class TestHeldOutSweepsAreRefusedByTheMerge:
    """A sweep that does not reproduce its board must never land under `seeds/`.

    `merge_controls.sh` derives its probe list from the parts present, which is
    right in general -- a probe missing from a hand-written table would have its
    parts silently skipped. But the parts directory is an *archive*, so a sweep
    held out on purpose is still sitting in it, and 21b's merge duly wrote
    `results/controls/seeds/scene_classification.jsonl`: a sweep whose seed-0
    rows miss their published cells by up to 0.0102, published into the
    directory every consumer reads as a board's measured noise.

    The gate in `tests/results/test_seed_sweeps.py` catches it, but only once
    the file exists. These pin the refusal at the merge.
    """

    def test_the_merge_script_names_the_held_out_sweep(self):
        text = (ROOT / "scripts" / "merge_controls.sh").read_text()
        assert "HELD_OUT_SWEEPS=" in text, "the held-out list is gone from merge_controls.sh"
        listed = text.split("HELD_OUT_SWEEPS=")[1].split("\n")[0]
        assert "scene_classification" in listed, (
            "scene_classification is held out at results/controls/scene_classification_seeds"
            ".jsonl and must not be merged into seeds/"
        )

    def test_the_held_out_sweep_is_not_committed_under_seeds(self):
        """The file this guards against, checked where it would appear."""
        stray = ROOT / "results" / "controls" / "seeds" / "scene_classification.jsonl"
        assert not stray.exists(), (
            f"{stray} exists -- a sweep that does not reproduce its board cannot be read "
            "as that board's noise; it belongs at results/controls/scene_classification_seeds.jsonl"
        )

    def test_the_held_out_sweep_is_still_committed_where_it_belongs(self):
        """Refusing the merge must not be confused with deleting the evidence."""
        kept = ROOT / "results" / "controls" / "scene_classification_seeds.jsonl"
        assert kept.is_file(), f"{kept} is missing; 20c/20d's held-out sweep is the evidence"

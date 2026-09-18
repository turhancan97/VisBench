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

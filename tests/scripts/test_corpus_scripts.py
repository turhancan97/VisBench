"""The corpus matrix is defined in two files, and they must agree.

`scripts/build_corpus.sh` holds one `probe_<name>` function per probe and lists
them in `ALL_PROBES`; `slurm/corpus.sbatch` holds its own `PROBES` array, which
it multiplies by the backbone list to size the array job. Neither file can see
the other, and the failure when they disagree is the one the sbatch's own guard
was written to prevent and cannot catch: a probe missing from `PROBES` is simply
never scheduled, the matrix stays *self-consistently* the wrong size, and every
comparability group in the resulting corpus still holds every backbone.

That is not hypothetical. `corner` shipped in 8b, was added to `ALL_PROBES` and
not to `PROBES`, and sat unschedulable through v0.9.0.

These are string-scraping tests because the definitions are shell arrays. That
is worth it here: the alternative is a third place defining the matrix.
"""

import re
from pathlib import Path

import pytest

from visbench import list_probes

ROOT = Path(__file__).resolve().parents[2]
BUILD_CORPUS = ROOT / "scripts" / "build_corpus.sh"
SBATCH = ROOT / "slurm" / "corpus.sbatch"


def _bash_array(text: str, name: str) -> list[str]:
    """Read `NAME=(\n  a\n  b\n)` out of a shell script, comments stripped."""
    match = re.search(rf"^{name}=\(\n(.*?)^\)$", text, re.MULTILINE | re.DOTALL)
    assert match is not None, f"no {name}=( ... ) array found"
    entries = []
    for line in match.group(1).splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            entries.append(line)
    return entries


@pytest.fixture(scope="module")
def all_probes() -> list[str]:
    return _bash_array(BUILD_CORPUS.read_text(), "ALL_PROBES")


@pytest.fixture(scope="module")
def sbatch_probes() -> list[str]:
    return _bash_array(SBATCH.read_text(), "PROBES")


def test_the_two_probe_lists_are_identical(all_probes, sbatch_probes):
    """Same probes, same order.

    Order matters as well as membership: the sbatch derives its probe from
    `index / len(backbones)`, so a reordering would silently relabel which task
    ran which probe when a partial array is resubmitted against logged indices.
    """
    assert sbatch_probes == all_probes


def test_the_corpus_covers_every_registered_probe(all_probes):
    """A probe that ships without a corpus row is absent from every board.

    `list_probes()` is the registry the CLI and the leaderboard both read, so
    this is the check that a new probe cannot be added without deciding what it
    would be measured on.
    """
    assert sorted(all_probes) == sorted(list_probes())


def test_every_listed_probe_has_a_builder(all_probes):
    """`build_corpus.sh` warns and continues on an unknown probe, rather than
    failing — the right behaviour in a loop, and one that turns a typo in
    `ALL_PROBES` into a probe silently missing from the corpus."""
    text = BUILD_CORPUS.read_text()
    for probe in all_probes:
        assert f"probe_{probe}()" in text, f"ALL_PROBES names {probe}, no probe_{probe}()"


# -- merge_corpus.sh ---------------------------------------------------------
#
# The corpus is widened one board at a time, so `results/corpus/parts/` holds
# only the most recent board. `merge_corpus.sh` originally rebuilt the corpus
# with `cat parts/*.jsonl > corpus`, which was right exactly once -- at 6e-2,
# when one array produced the whole matrix -- and became silently destructive
# as soon as the corpus outgrew a single run. Measured on 2026-08-28: 12
# orientation parts against a 180-record, 15-board corpus, so a rebuild would
# have dropped 168 records and 14 boards without an error. The generated tables
# would simply have rendered fewer boards, and every board still present would
# still have held every backbone -- the same shape of invisible gap the sbatch
# array guard exists for.
#
# These run the real script, because the failure was in what it *did* and a
# string-scraping test would have passed against the destructive version.

MERGE_CORPUS = ROOT / "scripts" / "merge_corpus.sh"
CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"


@pytest.fixture
def corpus_lines() -> list[str]:
    """Real records, so the script's own validation pass has something to parse."""
    lines = CORPUS.read_text().splitlines()
    assert len(lines) >= 4, "the committed corpus is unexpectedly small"
    return lines


def _run_merge(parts: Path, corpus: Path, **env_extra: str):
    import os
    import subprocess

    env = {**os.environ, "PARTS": str(parts), "CORPUS": str(corpus), **env_extra}
    return subprocess.run(
        ["bash", str(MERGE_CORPUS)], capture_output=True, text=True, env=env, check=False
    )


def test_merging_a_single_board_does_not_delete_the_others(tmp_path, corpus_lines):
    """The regression: parts/ holds one board, the corpus holds many."""
    parts = tmp_path / "parts"
    parts.mkdir()
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(corpus_lines[:4]) + "\n")
    (parts / "probe__backbone.jsonl").write_text("\n".join(corpus_lines[4:6]) + "\n")

    result = _run_merge(parts, corpus)
    assert result.returncode == 0, result.stderr

    merged = corpus.read_text().splitlines()
    for line in corpus_lines[:4]:
        assert line in merged, "merging a single board deleted records already in the corpus"
    for line in corpus_lines[4:6]:
        assert line in merged, "the new part was not merged in"


def test_merging_twice_adds_nothing(tmp_path, corpus_lines):
    """Idempotent by exact line, which is what makes a re-merge safe to run."""
    parts = tmp_path / "parts"
    parts.mkdir()
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(corpus_lines[:4]) + "\n")
    (parts / "probe__backbone.jsonl").write_text("\n".join(corpus_lines[4:6]) + "\n")

    assert _run_merge(parts, corpus).returncode == 0
    once = corpus.read_text()
    assert _run_merge(parts, corpus).returncode == 0
    assert corpus.read_text() == once


def test_rebuild_is_still_available_and_explicit(tmp_path, corpus_lines):
    """The 6e-2 behaviour is kept for the case it was written for, behind a flag.

    It has to stay reachable -- a full-matrix array with a corpus to replace is
    a real workflow -- but it must be asked for, since it is the destructive one.
    """
    parts = tmp_path / "parts"
    parts.mkdir()
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(corpus_lines[:4]) + "\n")
    (parts / "probe__backbone.jsonl").write_text("\n".join(corpus_lines[4:6]) + "\n")

    assert _run_merge(parts, corpus, REBUILD="1").returncode == 0
    merged = corpus.read_text().splitlines()
    assert merged == corpus_lines[4:6], "REBUILD should replace the corpus with the parts"

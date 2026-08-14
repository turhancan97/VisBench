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

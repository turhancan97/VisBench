"""The probe and board counts that user-facing prose repeats in the present tense.

Every count that has gone stale in this project went stale the same way: it was
carried forward through a release that added a probe or a board. The generated
tables have always been right -- `LEADERBOARD.md` is rendered from the corpus
and a fast test fails if it drifts -- and only the prose *around* them moves,
which is the half no test reads.

It has now happened twice. `scene_classification` and `fine_grained_classification`
took the corpus from fourteen probes to sixteen and the front page kept up; the
seventeenth, `instance_segmentation`, did not, and `README.md`, `docs/index.md`
and `docs/probes/overview.md` all still said "Sixteen probes" five releases
later. The README is package metadata, so a stale count there ships to PyPI and
a version can never be re-uploaded.

So the counts are read from the things that *are* generated -- `list_probes()`
for the probes, `LEADERBOARD.md`'s rendered sections for the boards -- and the
present-tense claims must agree with them.

**Only present-tense claims.** `CHANGELOG.md`, `ENGINEERING_LOG.md` and
`docs/roadmap.md` state counts *as of a release* -- "Sixteen probes against
twelve backbones, 192 records" is a correct fact about v0.13 and rewriting it
would falsify the history. They are excluded by name, and that exclusion is the
reason this guard is keyed to phrasings rather than to any number word before
"probes": a subset count ("five probes and nine ViTs", "ten probes against
DINOv2-S/14") is ordinary prose and must not fail.
"""

import re
from pathlib import Path

import pytest

from visbench import list_probes

ROOT = Path(__file__).resolve().parent.parent

#: Where the current state is claimed. These describe VisBench as it is now, so
#: a count in one of them is either right or stale -- never history.
CURRENT_STATE = (
    "README.md",
    "docs/index.md",
    "docs/probes/overview.md",
    "docs/probes/leaderboard.md",
    "docs/guides/reading-a-board.md",
    "docs/guides/dense-probes.md",
    "docs/probes/mid-level/correspondence.md",
    "docs/probes/high-level/semantic_segmentation.md",
)

#: The idioms that carry a *total*, as these files actually write it. Each is
#: anchored so that a subset count cannot match: "across five probes and nine
#: ViTs" puts the count before "across" rather than after "probes", and "the
#: five boards" (the high-level cluster) lacks the "of" that a total takes.
TOTAL_PROBES = re.compile(
    r"\bof the ([A-Za-z]+) probes\b|\b([A-Za-z]+) probes(?: across| against|, three)"
)
TOTAL_BOARDS = re.compile(r"\bof the ([A-Za-z]+) boards\b|\b([A-Za-z]+) boards, twelve\b")

#: Only as far as this project has had to count. A count word beyond it means
#: the corpus grew past what this table knows, which should fail loudly.
WORDS = {
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
}


def _word(count: int) -> str:
    assert count in WORDS, f"no spelling for {count}; extend WORDS"
    return WORDS[count]


@pytest.fixture(scope="module")
def probe_word() -> str:
    """How many probes there are, from the registry rather than from prose."""
    return _word(len(list_probes()))


@pytest.fixture(scope="module")
def board_word() -> str:
    """How many boards there are, read off the generated leaderboard.

    `CLAUDE.md`'s standing rule is to read a count off `LEADERBOARD.md` and
    never out of prose, so this test does exactly that rather than assuming one
    board per probe -- which is true today and is not a property of anything.
    """
    text = (ROOT / "LEADERBOARD.md").read_text(encoding="utf-8")
    boards = re.findall(r"^### (\S+)", text, flags=re.MULTILINE)
    assert boards, "LEADERBOARD.md rendered no boards"
    return _word(len(boards))


def _claims(pattern: re.Pattern[str]) -> list[tuple[str, int, str]]:
    found = []
    for name in CURRENT_STATE:
        path = ROOT / name
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in pattern.finditer(line):
                word = next(group for group in match.groups() if group)
                found.append((name, number, word.lower()))
    return found


def test_every_searched_file_exists():
    """A renamed page must not silently drop out of this guard's reach."""
    for name in CURRENT_STATE:
        assert (ROOT / name).is_file(), f"{name} is gone; update CURRENT_STATE"


def test_the_guard_reaches_something():
    """A pattern that matches nothing passes forever while checking nothing."""
    assert _claims(TOTAL_PROBES), "no total-probe claim found; the idiom changed"
    assert _claims(TOTAL_BOARDS), "no total-board claim found; the idiom changed"


@pytest.mark.parametrize(("path", "line", "word"), _claims(TOTAL_PROBES))
def test_probe_total_matches_the_registry(path, line, word, probe_word):
    assert word == probe_word, (
        f"{path}:{line} says '{word} probes' where the registry has "
        f"{len(list_probes())} ({probe_word})"
    )


@pytest.mark.parametrize(("path", "line", "word"), _claims(TOTAL_BOARDS))
def test_board_total_matches_the_leaderboard(path, line, word, board_word):
    assert word == board_word, (
        f"{path}:{line} says '{word} boards' where LEADERBOARD.md renders {board_word}"
    )

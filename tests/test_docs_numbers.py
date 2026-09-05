"""The one measured number that prose repeats, and that prose got wrong.

Most numbers in this project are generated -- the boards, `LEADERBOARD.md`, the
gallery -- and a generated number cannot drift. The pooling-mismatch pair is
the exception: it is a single measurement from 6e-4, and it is quoted by hand
in the CLI's help, an example, two guides, an API page and both archives.

It had drifted **three ways** by the time the docs were restructured: the
README said 0.9540/0.9830, `docs/hub.md` said 0.9620/0.9895 and `docs/show.md`
said 0.9620/0.9820. Two of the three were wrong, and nothing failed -- which is
the standing lesson in `CLAUDE.md` that only the prose around a generated table
goes stale, arriving on the one number that has no table.

So the pair is read from `ENGINEERING_LOG.md`, which is the record of the run
that produced it, and every other site must agree with it.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: The engineering log's sentence, which is the record. Reading the expected
#: value out of it rather than typing it here is the point: a test carrying its
#: own copy of a number is a fourth place for that number to drift.
RECORD = re.compile(r"scores \*\*(0\.\d{4}) against (0\.\d{4})\*\*")

#: How a mismatch pair is written anywhere else -- "0.9620 against 0.9820",
#: with or without emphasis, or hyphenated as in a table cell.
QUOTED = re.compile(r"(0\.\d{4})\**[ -]against[ -]\**(0\.\d{4})")

#: Everything that is not generated, not a build artifact and not this file.
SEARCHED = ("*.md", "visbench/**/*.py", "examples/*.py", "scripts/*.py")


def _texts() -> list[tuple[Path, str]]:
    found = []
    for pattern in SEARCHED:
        for path in sorted(ROOT.glob(pattern)):
            found.append((path, path.read_text()))
    for path in sorted((ROOT / "docs").rglob("*.md")):
        if "_build" not in path.parts:
            found.append((path, path.read_text()))
    return found


@pytest.fixture(scope="module")
def recorded() -> tuple[str, str]:
    """The pair as the engineering log records it."""
    log = (ROOT / "ENGINEERING_LOG.md").read_text()
    matches = RECORD.findall(log)
    fitted = [pair for pair in matches if pair[0] == "0.9620"]
    assert fitted, "the pooling-mismatch measurement is no longer in the log"
    return fitted[0]


def test_the_log_records_the_pair(recorded):
    """If this changes, the measurement was re-run and every quote must move."""
    assert recorded == ("0.9620", "0.9820")


def test_every_quote_of_the_pair_agrees_with_the_record(recorded):
    """The failure this exists for: a hand-copied number quietly rewritten."""
    seen = 0
    for path, text in _texts():
        if path.name == Path(__file__).name:
            continue
        for pair in QUOTED.findall(text):
            if pair[0] != recorded[0]:
                continue  # a different measurement that happens to be a pair
            seen += 1
            assert pair == recorded, f"{path.relative_to(ROOT)} quotes {pair}"
    assert seen >= 4, "the quotes moved or changed shape; re-point this guard"


# --- The corpus counts CLAUDE.md quotes -------------------------------------
#
# Every count that has gone stale in this project went stale by being carried
# forward in prose through a release that added a column or a board -- the
# standing rule in `CLAUDE.md` says so, and the corpus counts there had drifted
# again by 13a ("192 records" for a 252-record file). These are pinned rather
# than the fast-suite count because they move only when a probe, a backbone or
# a board lands, which already touches a dozen files by the documented edit
# list; a guard on the *test* count would turn every pull request that adds a
# test into a CLAUDE.md edit, on a file with a size budget.

CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"
CLAUDE = ROOT / "CLAUDE.md"


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    import json

    return [json.loads(line) for line in CORPUS.read_text().splitlines() if line.strip()]


def test_claude_md_quotes_the_corpus_file_size(corpus):
    """252 is the file; it is append-only, so it exceeds the coverage."""
    assert f"**The corpus file is {len(corpus)} records" in CLAUDE.read_text()


def test_claude_md_quotes_the_board_coverage(corpus):
    """192 is boards x backbones -- what `latest_per_backbone` resolves to."""
    boards = {record["task"] for record in corpus}
    backbones = {record["backbone"] for record in corpus}
    assert f"resolving to {len(boards) * len(backbones)} board cells" in CLAUDE.read_text()


def test_the_leaderboard_renders_a_board_for_every_probe_in_the_corpus(corpus):
    """The count above is only meaningful if the boards actually exist."""
    rendered = re.findall(r"^### (\S+)$", (ROOT / "LEADERBOARD.md").read_text(), re.M)
    assert set(rendered) == {record["task"] for record in corpus}

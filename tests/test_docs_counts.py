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
#: The second alternative is the front page's shape -- "nineteen boards, thirteen
#: backbones each". It was written as `boards, twelve` and so stopped matching
#: the moment `dino_vitb8` made that line say "thirteen": the README then carried
#: "eighteen boards" through the release that rendered nineteen, and nothing
#: failed. An idiom anchored on a count *beside* the one it checks is anchored on
#: something that moves.
TOTAL_BOARDS = re.compile(r"\bof the ([A-Za-z]+) boards\b|\b([A-Za-z]+) boards, [a-z]+ backbones\b")

#: "a committed corpus covering **247 board cells**" -- the one count on the
#: front page that is neither probes nor boards, and the one that went stale
#: without either guard above having an opinion about it.
TOTAL_CELLS = re.compile(r"\*{0,2}(\d+) board cells\*{0,2}")

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


@pytest.fixture(scope="module")
def cell_count() -> int:
    """How many board cells there are, counted off the generated leaderboard.

    A cell is one backbone's row on one board, which is what the front page
    means by the number: it is the size of the corpus as *rendered*, not the
    line count of the record file, which is larger because the corpus is
    append-only.
    """
    text = (ROOT / "LEADERBOARD.md").read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.startswith("| `")]
    assert rows, "LEADERBOARD.md rendered no board rows"
    return len(rows)


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
    assert _cell_claims(), "no board-cell claim found; the idiom changed"


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


def _cell_claims() -> list[tuple[str, int, str]]:
    """Board-cell claims, matched across a line break.

    The front page wraps this one mid-phrase -- "covering **247 board / cells**"
    -- and it is inside a blockquote, so the continuation carries a `> ` that a
    naive join puts in the middle of the number's own sentence. Both are stripped
    before matching, exactly as the leader claim below joins two lines. The digit
    has to appear on the reported line so a claim is not found twice.
    """
    found = []
    for name in CURRENT_STATE:
        lines = [
            re.sub(r"^\s*>\s?", "", line)
            for line in (ROOT / name).read_text(encoding="utf-8").splitlines()
        ]
        for number, line in enumerate(lines, 1):
            window = " ".join(lines[number - 1 : number + 1])
            for match in TOTAL_CELLS.finditer(window):
                if match.group(1) in line:
                    found.append((name, number, match.group(1)))
    return found


@pytest.mark.parametrize(("path", "line", "word"), _cell_claims())
def test_board_cell_total_matches_the_leaderboard(path, line, word, cell_count):
    assert int(word) == cell_count, (
        f"{path}:{line} says '{word} board cells' where LEADERBOARD.md renders {cell_count}"
    )


#: "`mae_vitb16` is first on five of the eighteen boards and last on four" —
#: the one claim here that is a *count over the corpus* rather than a count of
#: it, and the one that has gone stale without anything noticing.
#:
#: It was published as "six ... four", which was right at twelve backbones and
#: wrong from `dino_vitb8` onward: the new backbone took `corner` and
#: `correspondence` off `mae_vitb16`, leaving four, and `relative_pose` later
#: gave one back. Nothing failed, because the guard above pins the number of
#: *boards* and had no opinion about the number beside it.
LEADER_CLAIM = re.compile(
    r"`(?P<backbone>\w+)` is first on \*{0,2}(?P<first>[A-Za-z]+)\*{0,2} "
    r"of the (?P<boards>[A-Za-z]+) boards and last on \*{0,2}(?P<last>[A-Za-z]+)\*{0,2}"
)

#: Small counts, for the leader claim. Separate from WORDS, which starts at
#: twelve because it spells *totals*.
SMALL_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
}


def _leader_counts() -> dict[str, tuple[int, int]]:
    """How many boards each backbone leads and trails, from the corpus.

    Computed the way `LEADERBOARD.md` is — one board per comparability group,
    ranked by that task's headline metric — rather than parsed back out of the
    rendered tables, because the rendered ordering is what this claim is *about*
    and reading it from the same place twice would check nothing.
    """
    from collections import Counter

    from visbench.results.leaderboard import group_comparable, latest_per_backbone, rank
    from visbench.results.render import HEADLINE_METRICS
    from visbench.results.writer import iter_records

    records = list(iter_records(ROOT / "results" / "corpus" / "visbench.jsonl"))
    first: Counter = Counter()
    last: Counter = Counter()
    for key, group in group_comparable(records).items():
        ordered = [row for row, _ in rank(latest_per_backbone(group), HEADLINE_METRICS[key.task])]
        first[ordered[0].backbone] += 1
        last[ordered[-1].backbone] += 1
    return {name: (first[name], last[name]) for name in set(first) | set(last)}


def _leader_claims() -> list[tuple[str, int, str, str, str, str]]:
    found = []
    for name in CURRENT_STATE:
        text = (ROOT / name).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            # The claim spans two lines in two of the three files, so match on
            # the joined text and report the line the backbone appears on.
            window = "\n".join(text.splitlines()[number - 1 : number + 1]).replace("\n", " ")
            match = LEADER_CLAIM.search(window)
            if match and match.group("backbone") in line:
                found.append(
                    (
                        name,
                        number,
                        match.group("backbone"),
                        match.group("first").lower(),
                        match.group("boards").lower(),
                        match.group("last").lower(),
                    )
                )
    return found


def test_the_leader_guard_reaches_something():
    assert _leader_claims(), "no 'first on N boards' claim found; the idiom changed"


@pytest.mark.parametrize(("path", "line", "backbone", "first", "boards", "last"), _leader_claims())
def test_the_leader_count_matches_the_corpus(path, line, backbone, first, boards, last, board_word):
    """A count over the corpus is a fact about that corpus, not a backbone."""
    leads, trails = _leader_counts()[backbone]
    assert boards == board_word, f"{path}:{line} says '{boards} boards'"
    assert first == SMALL_WORDS[leads], (
        f"{path}:{line} says {backbone} is first on '{first}' boards; the corpus says {leads}"
    )
    assert last == SMALL_WORDS[trails], (
        f"{path}:{line} says {backbone} is last on '{last}' boards; the corpus says {trails}"
    )

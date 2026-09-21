"""The placement counts are prose too, and they will go stale the same way.

22a put two numbers into `CORPUS_FINDINGS.md` and `docs/guides/reading-a-board.md`:
**how many of the twenty boards have a first place that survives a paired
re-fit, and how many have a last place that does.** Both move whenever a board
is added, a backbone column is added, or a sweep is re-run — which is every
route by which a count has gone stale in this project.

`tests/results/test_sweep_totals.py` guards the *adjacent-pair* totals from the
same study and has no opinion about these, which is this project's most
repeated failure in its usual shape: a guard on a total is not a guard on the
count beside it. So these are recomputed here from the committed sweeps, through
`scripts/analyse_placements.py` itself rather than through a second copy of its
arithmetic — a reimplementation here would be free to drift from the script the
prose tells a reader to run.

The script is loaded by path because `scripts/` is not a package, the way
`tests/scripts/test_analyse_training_diagnostics.py` loads its subject.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analyse_placements.py"

#: Where the two counts are quoted, and the shapes those files write them in.
#: "17 first places and 13 last places survive a paired re-fit" and
#: "**17 first places and 13 last places**".
QUOTING = ("CORPUS_FINDINGS.md", "docs/guides/reading-a-board.md")

CLAIM = re.compile(
    r"\*{0,2}(?P<first>\d+)\s+of\s+(?P<boards>\d+)\s+first\s+places[^.]*?"
    r"(?P<last>\d+)\s+of\s+(?P<boards2>\d+)\s+last\s+places"
    r"|\*{0,2}(?P<first2>\d+)\s+first\s+places\s+and\s+(?P<last2>\d+)\s+last\s+places"
)


@pytest.fixture(scope="module")
def placements():
    if not SCRIPT.is_file():
        pytest.skip(f"no script at {SCRIPT}")
    spec = importlib.util.spec_from_file_location("analyse_placements", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.audit()
    solid = ("ordered", "exact")
    return {
        "boards": len(rows),
        "first": sum(1 for r in rows if r["first_verdict"] in solid),
        "last": sum(1 for r in rows if r["last_verdict"] in solid),
        "rows": rows,
    }


def _claims() -> list[tuple[str, int, re.Match]]:
    found = []
    for name in QUOTING:
        lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
        for number, _ in enumerate(lines, 1):
            # Collapse the run of spaces an indented continuation line
            # contributes; without this "17\n  first places" joins to
            # "17   first places" and a single-space pattern misses it.
            window = re.sub(r"\s+", " ", " ".join(lines[number - 1 : number + 2]))
            match = CLAIM.search(window)
            if match and match.group(0).split()[0].strip("*") in lines[number - 1]:
                found.append((name, number, match))
    return found


def test_the_guard_reaches_every_file_that_quotes_the_counts():
    """Not merely "reaches something" -- that is what let this one under-reach.

    The first version of this pattern matched the guide and silently missed both
    claims in `CORPUS_FINDINGS.md`, because they wrap onto an indented
    continuation line. `assert _claims()` passed the whole time. **A guard that
    asserts it found *a* claim cannot tell you it missed most of them**, so this
    asserts every file in `QUOTING` is reached by name.
    """
    reached = {name for name, _, _ in _claims()}
    assert reached == set(QUOTING), f"not reached: {sorted(set(QUOTING) - reached)}"


@pytest.mark.parametrize(("path", "line", "match"), _claims(), ids=lambda v: str(v)[:44])
def test_the_placement_counts_match_the_sweeps(path, line, match, placements):
    first = int(match.group("first") or match.group("first2"))
    last = int(match.group("last") or match.group("last2"))
    assert (first, last) == (placements["first"], placements["last"]), (
        f"{path}:{line} quotes {first} first / {last} last places surviving, where the "
        f"committed sweeps give {placements['first']} / {placements['last']}"
    )
    for group in ("boards", "boards2"):
        if match.group(group):
            assert int(match.group(group)) == placements["boards"], (
                f"{path}:{line} states {match.group(group)} boards, not {placements['boards']}"
            )


def test_last_places_are_the_weaker_half(placements):
    """The asymmetry the entry is *about*, pinned as a direction rather than a value.

    If a future corpus made last places the safer half, the write-up would be
    telling a reader to hedge the wrong one -- and no count above would notice,
    because both numbers would still be internally consistent.
    """
    assert placements["last"] < placements["first"], (
        f"last places ({placements['last']}) now survive at least as often as first "
        f"({placements['first']}); CORPUS_FINDINGS.md's asymmetry entry needs rewriting"
    )


def test_a_reversed_placement_is_reported_as_such(placements):
    """A reversal must never be summarised as merely 'not separable'.

    `relative_pose`'s last place leans the other way (t = -3.06). A verdict that
    collapsed REVERSED into tied would lose the one distinction 20a exists to
    make, while every count in this file stayed correct.
    """
    verdicts = {r["task"]: r["last_verdict"] for r in placements["rows"]}
    assert verdicts["relative_pose"] == "REVERSED", (
        "relative_pose's last place is one of 20a's three documented reversals; "
        f"this run reports {verdicts['relative_pose']}"
    )

"""The separability totals are prose, and prose is the half that drifts.

`CLAUDE.md`, `CORPUS_FINDINGS.md` and `results/controls/README.md` each quote
how many adjacent pairs the seed sweeps order, tie and reverse — "119 of 180
pairs ordered, 58 tied, 3 reversed" — and how many boards have a largest
*unordered* gap above their smallest *ordered* one. Every one of those moves
when a board is swept, and **nothing recomputed them**: `tests/test_docs_counts.py`
pins probe, board and cell totals in the seven user-facing files, and these are
neither those numbers nor those files.

That is this project's most repeated failure — a guard on a total is not a guard
on the count beside it — arriving a third time, so the totals are recomputed
here from the committed sweeps rather than trusted. The arithmetic is the one
`scripts/analyse_seeds.py` prints, kept in one place so the prose and the script
cannot disagree.

**`scene_classification` is excluded deliberately**: its sweep does not
reproduce its board (20c/20d), so it is held out of `results/controls/seeds/`
and its pairs are not part of any published total. The exclusion is structural —
this file globs the directory, and that sweep is not in it.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

import pytest

from visbench.results.leaderboard import group_comparable, latest_per_backbone, metric_direction
from visbench.results.render import HEADLINE_METRICS
from visbench.results.writer import iter_records

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"
SWEEP_DIR = ROOT / "results" / "controls" / "seeds"
POSE_SWEEP = ROOT / "results" / "controls" / "pose_seeds.jsonl"

#: Where the totals are quoted. `CHANGELOG.md` and `docs/roadmap.md` are absent
#: for the reason `test_docs_counts.py` excludes them: a number stated *as of a
#: step* is correct history, and rewriting it would falsify the record.
QUOTING = ("CLAUDE.md", "CORPUS_FINDINGS.md", "results/controls/README.md")

#: "119 of 180 pairs ordered, 58 tied, 3 reversed", in the shapes those files
#: actually write it.
TOTALS = re.compile(
    r"\*{0,2}(?P<ordered>\d+) of (?P<pairs>\d+)(?: adjacent)? pairs ordered,?\*{0,2}\s+"
    r"\*{0,2}(?P<tied>\d+) tied,?\*{0,2}\s+(?:and )?\*{0,2}(?P<reversed>\d+) reversed"
)

#: The same three totals again, one per **table row**, which is how
#: `results/controls/README.md` writes them. A pattern built only for the prose
#: form reads that file and finds nothing -- which is this project's own
#: "anchored on something that moves", so both forms are matched.
TABLE_ROWS = {
    "ordered": re.compile(r"\|\s*adjacent pairs \*{0,2}ordered\*{0,2}\s*\|\s*\*{0,2}(\d+)"),
    "tied": re.compile(r"\|\s*\*{0,2}tied\*{0,2}[^|]*\|\s*\*{0,2}(\d+)"),
    "reversed": re.compile(r"\|\s*\*{0,2}reversed\*{0,2}[^|]*\|\s*\*{0,2}(\d+)"),
}

#: "on 10 of 15 boards the largest ..." (prose) and "| boards where the largest
#: ... | **10 of 15** |" (table).
THRESHOLDLESS = re.compile(
    r"\*{0,2}(?P<boards>\d+) of (?P<total>\d+)\*{0,2} boards the largest"
    r"|boards where the largest[^|]*\|\s*\*{0,2}(?P<boards2>\d+) of (?P<total2>\d+)"
)

#: A count spelled as a word is the anchor that has already broken a guard here
#: -- the README's board idiom was pinned on `boards, twelve` and stopped
#: matching when a thirteenth backbone arrived. `results/controls/README.md`
#: heads its summary "over fifteen boards", so the word is checked too.
NUMBER_WORDS = {
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
BOARD_WORD = re.compile(r"over (?P<word>[a-z]+) boards")

T_CRITICAL_N5 = 2.776


def _sweeps() -> dict[str, Path]:
    found = {path.stem: path for path in SWEEP_DIR.glob("*.jsonl")} if SWEEP_DIR.is_dir() else {}
    if POSE_SWEEP.is_file():
        found["relative_pose"] = POSE_SWEEP
    return found


def _verdicts() -> dict[str, dict]:
    """Per board: ordered / tied / reversed, and whether a threshold could sort it."""
    corpus = list(iter_records(CORPUS))
    out: dict[str, dict] = {}
    for task, path in _sweeps().items():
        metric = HEADLINE_METRICS[task]
        higher = metric_direction(metric) == "higher"
        (group,) = group_comparable([r for r in corpus if r.task == task]).values()
        cells = {c.backbone: c.metrics[metric] for c in latest_per_backbone(group)}
        scores: dict[str, dict[int, float]] = {}
        for record in iter_records(path):
            if record.task == task:
                scores.setdefault(record.backbone, {})[record.seed] = record.metrics[metric]

        order = sorted(cells, key=lambda b: cells[b], reverse=higher)
        sign = 1.0 if higher else -1.0
        counts = {"ordered": 0, "tied": 0, "reversed": 0}
        ordered_gaps, unordered_gaps = [], []
        for better, worse in zip(order, order[1:], strict=False):
            shared = sorted(set(scores.get(better, {})) & set(scores.get(worse, {})))
            if len(shared) < 2:
                continue
            diffs = [sign * (scores[better][s] - scores[worse][s]) for s in shared]
            sd = statistics.stdev(diffs)
            t = statistics.mean(diffs) / (sd / len(diffs) ** 0.5) if sd else float("inf")
            gap = abs(cells[better] - cells[worse])
            if t >= T_CRITICAL_N5:
                counts["ordered"] += 1
                ordered_gaps.append(gap)
            else:
                counts["reversed" if t <= -T_CRITICAL_N5 else "tied"] += 1
                unordered_gaps.append(gap)
        counts["thresholdless"] = bool(
            ordered_gaps and unordered_gaps and max(unordered_gaps) > min(ordered_gaps)
        )
        out[task] = counts
    return out


@pytest.fixture(scope="module")
def totals() -> dict:
    verdicts = _verdicts()
    if not verdicts:
        pytest.skip(f"no sweeps under {SWEEP_DIR}")
    return {
        "boards": len(verdicts),
        "ordered": sum(v["ordered"] for v in verdicts.values()),
        "tied": sum(v["tied"] for v in verdicts.values()),
        "reversed": sum(v["reversed"] for v in verdicts.values()),
        "pairs": sum(v["ordered"] + v["tied"] + v["reversed"] for v in verdicts.values()),
        "thresholdless": sum(v["thresholdless"] for v in verdicts.values()),
    }


def _claims(pattern: re.Pattern[str]) -> list[tuple[str, int, re.Match]]:
    found = []
    for name in QUOTING:
        lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
        for number, _ in enumerate(lines, 1):
            window = " ".join(lines[number - 1 : number + 1])
            match = pattern.search(window)
            if match and match.group(0).split()[0].strip("*") in lines[number - 1]:
                found.append((name, number, match))
    return found


def test_the_guard_reaches_something():
    """A pattern matching nothing passes forever while checking nothing."""
    assert _claims(TOTALS), "no ordered/tied/reversed claim found; the idiom changed"
    assert _claims(THRESHOLDLESS), "no thresholdless-board claim found; the idiom changed"


@pytest.mark.parametrize(("path", "line", "match"), _claims(TOTALS), ids=lambda v: str(v)[:40])
def test_the_separability_totals_match_the_sweeps(path, line, match, totals):
    got = {key: int(match.group(key)) for key in ("ordered", "pairs", "tied", "reversed")}
    assert got == {
        "ordered": totals["ordered"],
        "pairs": totals["pairs"],
        "tied": totals["tied"],
        "reversed": totals["reversed"],
    }, f"{path}:{line} quotes {got} where the committed sweeps give {totals}"


@pytest.mark.parametrize(
    ("path", "line", "match"), _claims(THRESHOLDLESS), ids=lambda v: str(v)[:40]
)
def test_the_thresholdless_board_count_matches(path, line, match, totals):
    boards = match.group("boards") or match.group("boards2")
    total = match.group("total") or match.group("total2")
    assert int(boards) == totals["thresholdless"], (
        f"{path}:{line} says {boards} boards where the sweeps give {totals['thresholdless']}"
    )
    assert int(total) == totals["boards"], (
        f"{path}:{line} says {total} swept boards where there are {totals['boards']}"
    )


@pytest.mark.parametrize("label", sorted(TABLE_ROWS))
def test_the_summary_table_rows_match_the_sweeps(label, totals):
    """`results/controls/README.md` states each total as its own table row."""
    text = (ROOT / "results" / "controls" / "README.md").read_text(encoding="utf-8")
    found = TABLE_ROWS[label].search(text)
    assert found, f"no `{label}` row in the summary table; the table was reshaped"
    assert int(found.group(1)) == totals[label], (
        f"the summary table's `{label}` row says {found.group(1)} where the "
        f"committed sweeps give {totals[label]}"
    )


def test_the_board_count_spelled_as_a_word_matches():
    """A count spelled out is still a count, and is how a guard here failed before."""
    text = (ROOT / "results" / "controls" / "README.md").read_text(encoding="utf-8")
    found = BOARD_WORD.search(text)
    assert found, "no 'over <word> boards' heading; the idiom changed"
    expected = NUMBER_WORDS[len(_sweeps())]
    assert found.group("word") == expected, (
        f"the summary heading says 'over {found.group('word')} boards' where "
        f"{len(_sweeps())} sweeps are committed ('{expected}')"
    )

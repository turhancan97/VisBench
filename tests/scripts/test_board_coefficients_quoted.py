"""The board-pair coefficients the shipped pages quote, recomputed from the corpus.

23a re-ran nine of the thirteen columns and moved every coefficient those
columns touch, and nothing noticed: the generated tables were right, and the
prose around them kept the cropped-board values for a whole step. 23c re-read
them by hand. This pins the ones a reader meets first — the probe pages and the
overview, which ship with the docs site, and the vehicle docstring, which ships
in the wheel — against `scripts/analyse_board_correlates.py` itself, so the next
column or board that moves them turns this red instead of leaving the pages to
drift.

It is deliberately a list of *quoted* numbers rather than every coefficient: a
coefficient nobody quotes cannot go stale. `CORPUS_FINDINGS.md` is not pinned
here, because its entries are dated narratives that keep earlier values on
purpose; its current-value blocks say which corpus they were read on.
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analyse_board_correlates.py"
PAGES = ROOT / "docs" / "probes"


@pytest.fixture(scope="module")
def corpus():
    spec = importlib.util.spec_from_file_location("analyse_board_correlates", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    boards = module.load_boards(module.CORPUS)
    levels = module.load_levels(module.CORPUS)

    def rho(a: str, b: str) -> float:
        return module.spearman(boards[a], boards[b])

    def tier_mean(task: str, level: str) -> float:
        others = [o for o in boards if levels[o] == level and o != task]
        return statistics.fmean(rho(task, o) for o in others)

    def against(task: str, prop: str) -> float:
        return module.spearman(boards[task], module._property(prop, sorted(boards[task]), 10e9))

    return {"rho": rho, "tier_mean": tier_mean, "against": against}


def _text(path: Path) -> str:
    """The file with its line breaks collapsed, so a quote may wrap."""
    return " ".join(path.read_text(encoding="utf-8").split())


def _signed(value: float) -> str:
    return f"{value:+.3f}".replace("-", "−")


VEHICLE = "high-level/vehicle_classification.md"

#: (page, the board pair it quotes)
PAIRS = [
    (VEHICLE, ("vehicle_classification", "fine_grained_classification")),
    (VEHICLE, ("vehicle_classification", "scene_classification")),
    ("overview.md", ("vehicle_classification", "scene_classification")),
    ("overview.md", ("vehicle_classification", "fine_grained_classification")),
    ("overview.md", ("detection", "semantic_segmentation")),
    ("overview.md", ("classification", "retrieval")),
    ("overview.md", ("keypoints2d", "orientation")),
    ("high-level/scene_parsing.md", ("scene_parsing", "semantic_segmentation")),
    ("high-level/scene_parsing.md", ("scene_parsing", "fine_grained_classification")),
    ("high-level/scene_parsing.md", ("scene_parsing", "depth")),
    ("mid-level/relative_pose.md", ("relative_pose", "occlusion_edge")),
    ("mid-level/relative_pose.md", ("relative_pose", "surface_normal")),
]


@pytest.mark.parametrize(("page", "pair"), PAIRS, ids=lambda v: str(v))
def test_a_quoted_board_pair_is_the_corpus_value(corpus, page, pair):
    value = corpus["rho"](*pair)
    text = _text(PAGES / page)
    assert _signed(value) in text or f"{value:+.3f}" in text, (
        f"{page} should quote {pair[0]} / {pair[1]} at {value:+.3f}; re-read it off "
        "scripts/analyse_board_correlates.py"
    )


TIER_MEANS = [
    (VEHICLE, "vehicle_classification"),
    ("high-level/scene_parsing.md", "scene_parsing"),
    ("mid-level/relative_pose.md", "relative_pose"),
]


@pytest.mark.parametrize(("page", "task"), TIER_MEANS, ids=lambda v: str(v))
def test_a_quoted_tier_mean_is_the_corpus_value(corpus, page, task):
    text = _text(PAGES / page)
    for level in ("high_level", "mid_level", "low_level"):
        value = corpus["tier_mean"](task, level)
        assert _signed(value) in text or f"{value:+.3f}" in text, (
            f"{page} should quote {task}'s mean rho against {level} at {value:+.3f}"
        )


@pytest.mark.parametrize(
    ("page", "task", "prop"),
    [
        ("high-level/semantic_segmentation.md", "semantic_segmentation", "tokens"),
        ("high-level/semantic_segmentation.md", "semantic_segmentation", "pretrain_images"),
        ("high-level/semantic_segmentation.md", "generic_segmentation", "tokens"),
        ("mid-level/generic_segmentation.md", "generic_segmentation", "tokens"),
        ("overview.md", "generic_segmentation", "tokens"),
        ("overview.md", "depth", "tokens"),
    ],
    ids=lambda v: str(v),
)
def test_a_quoted_structural_coefficient_is_the_corpus_value(corpus, page, task, prop):
    value = corpus["against"](task, prop)
    assert f"{value:+.3f}" in _text(PAGES / page), (
        f"{page} should quote {task} against {prop} at {value:+.3f}"
    )


def test_the_vehicle_docstring_ships_the_corpus_values(corpus):
    """The one place these numbers reach the wheel rather than the docs site."""
    text = _text(ROOT / "visbench" / "tasks" / "high_level" / "vehicle_classification.py")
    for pair in (
        ("vehicle_classification", "fine_grained_classification"),
        ("vehicle_classification", "scene_classification"),
    ):
        assert f"{corpus['rho'](*pair):+.3f}" in text, pair

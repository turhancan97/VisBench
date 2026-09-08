"""The correlate script duplicates two tables, so both are pinned here.

It copies ``HEADLINE_METRICS`` and re-states which metrics are errors, so that
it runs without importing visbench -- and therefore without torch, which is
what lets it run on a login node where the project venv does not resolve. A
copy that drifts would rank a board on a metric nobody chose, or rank an error
metric upside down and print the result as a finding.

These are the same class of check as ``tests/test_readme.py``: nothing executes
the duplicated table, so nothing else would catch it.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import pytest

from visbench.results.leaderboard import metric_direction
from visbench.results.render import HEADLINE_METRICS

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analyse_board_correlates.py"


def _load():
    spec = importlib.util.spec_from_file_location("analyse_board_correlates", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load()


def test_the_copied_headline_table_matches_the_real_one(script):
    """The one that actually breaks: a probe added with a new headline metric."""
    assert script.HEADLINE_METRICS == HEADLINE_METRICS


def test_lower_is_better_agrees_with_metric_direction(script):
    """Restating a direction is the error that reads as a finding.

    Checked against ``metric_direction`` rather than a second literal, so the
    source of truth is the same function the leaderboard ranks with.
    """
    for task, metric in HEADLINE_METRICS.items():
        expected = metric_direction(metric) == "lower"
        assert (task in script.LOWER_IS_BETTER) is expected, (
            f"{task}'s headline metric {metric!r} is "
            f"{metric_direction(metric)}-is-better, which LOWER_IS_BETTER disagrees with"
        )


def test_every_backbone_in_the_corpus_has_a_structure_entry(script):
    """A new column reaches the corpus before it reaches the structural table.

    Correlating against a structure that does not exist would drop the backbone
    from every coefficient silently, so the script exits 1 -- and this fails
    first, in the fast suite, rather than after someone reads a stale number.
    """
    boards = script.load_boards(script.CORPUS)
    seen = {backbone for board in boards.values() for backbone in board}
    assert seen <= set(script.STRUCTURE), (
        f"no STRUCTURE entry for {', '.join(sorted(seen - set(script.STRUCTURE)))}"
    )


def test_spearman_on_the_cases_with_known_answers(script):
    """Perfect agreement, perfect inversion, and the sign in between."""
    same = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert script.spearman(same, same) == pytest.approx(1.0)
    assert script.spearman(same, {"a": 1.0, "b": 2.0, "c": 3.0}) == pytest.approx(-1.0)
    assert script.spearman(same, {"a": 3.0, "b": 1.0, "c": 2.0}) == pytest.approx(0.5)


def test_spearman_correlates_only_the_shared_backbones(script):
    """A half-filled board shrinks the comparison rather than raising.

    The corpus is grown one backbone column at a time, so a board that is
    missing a row is a normal intermediate state, not an error.
    """
    left = {"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.0}
    right = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert script.spearman(left, right) == pytest.approx(1.0)


def test_spearman_refuses_a_sample_too_small_to_mean_anything(script):
    with pytest.raises(ValueError, match="at least 3"):
        script.spearman({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.0})


def test_error_metrics_are_oriented_before_ranking(script):
    """surface_normal's headline is angular error, so the board must be negated.

    Without this the normals board ranks upside down, and every correlation
    drawn from it flips sign -- which looks like a result about geometry.
    """
    boards = script.load_boards(script.CORPUS)
    normals = boards["surface_normal"]
    best = max(normals, key=lambda k: normals[k])
    assert normals[best] < 0, "an error metric should have been negated"
    assert best == "mae_vitb16", "mae_vitb16 has the lowest mean angular error in the corpus"


def test_levels_come_from_the_records_not_the_task_name(script):
    """`similarity` is the trap this reads from the corpus to avoid.

    It is mid-level image similarity, deliberately distinct from high-level
    retrieval, and counting it as high-level is a mistake that shipped here for
    a commit. Pinned explicitly because a tier analysis keyed on the wrong
    level produces a plausible number rather than an error.
    """
    levels = script.load_levels(script.CORPUS)
    assert levels["similarity"] == "mid_level"
    assert levels["retrieval"] == "high_level"
    assert levels["semantic_segmentation"] == "high_level"
    assert sorted(set(levels.values())) == ["high_level", "low_level", "mid_level"]


def test_a_task_appearing_at_two_levels_is_refused(script, tmp_path):
    """Two levels for one task would be averaged into both tiers silently."""
    corpus = tmp_path / "c.jsonl"
    corpus.write_text(
        '{"task":"edge","level":"low_level","backbone":"a","metrics":{"edge_correlation":1}}\n'
        '{"task":"edge","level":"mid_level","backbone":"b","metrics":{"edge_correlation":2}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="two levels"):
        script.load_levels(corpus)


def test_agreement_covers_every_unordered_pair_once(script):
    boards = script.load_boards(script.CORPUS)
    pairs = script.agreement(boards)
    n = len(boards)
    assert len(pairs) == n * (n - 1) // 2
    assert all(a < b for a, b in pairs), "keys must be sorted, so lookup needs no ordering rule"


def test_tier_summary_splits_within_from_across(script):
    """A hand-checkable case: one tight tier, one loose one, and the mean between."""
    levels = {"a": "high", "b": "high", "c": "low"}
    pairs = {("a", "b"): 0.9, ("a", "c"): 0.1, ("b", "c"): 0.3}
    within, across = script.tier_summary(pairs, levels)
    assert within == {"high": pytest.approx(0.9)}
    assert across == pytest.approx(0.2)
    assert "low" not in within, "a tier with one member has no within-tier pair"


def test_the_high_level_tier_is_two_clusters_not_one(script):
    """The stable finding is the structure, not the sign of tier-mean minus cross.

    Two clusters that barely correlate with each other: image-level
    categorisation (`classification`, `retrieval`) and localised / spatial-
    context prediction (`detection`, `semantic_segmentation`,
    `scene_classification`, `fine_grained_classification`). That has held at
    six, nine and twelve backbones and from thirteen to sixteen boards.

    The tier *mean* has not been stable and is deliberately not pinned tightly
    here. It was below the cross-tier mean at thirteen boards, marginally above
    at fourteen and fifteen, and +0.373 against +0.276 at sixteen -- the CUB
    board lifted it more than any single board has, not by making the tier
    coherent but by joining the larger of its two clusters. An earlier version
    of this test asserted `abs(high - across) < 0.1` and passed at 0.097, which
    is a test one board away from failing for a reason that is not a
    regression. What is pinned instead is the *ordering* of the three tier
    means, which is the structural consequence of high-level being bimodal.
    """
    boards = script.load_boards(script.CORPUS)
    levels = script.load_levels(script.CORPUS)
    pairs = script.agreement(boards)

    def rho(a, b):
        return pairs[tuple(sorted((a, b)))]

    image_level = ("classification", "retrieval")
    localised = (
        "detection",
        "semantic_segmentation",
        "scene_classification",
        "fine_grained_classification",
    )

    # Each cluster is tight,
    assert rho(*image_level) > 0.6
    for a, b in itertools.combinations(localised, 2):
        assert rho(a, b) > 0.5, f"{a}/{b} = {rho(a, b):+.3f} breaks the localised cluster"
    # and the two ignore each other.
    for a in image_level:
        for b in localised:
            assert rho(a, b) < 0.4, f"{a}/{b} = {rho(a, b):+.3f} bridges the two clusters"

    # The replication, which is what makes this a property of the cluster
    # rather than a curiosity about Places365: both probes that are *mechanically*
    # object classification with a different folder rank with the localised
    # cluster instead of with the object board they subclass.
    assert rho("scene_classification", "detection") > 0.5
    assert rho("fine_grained_classification", "detection") > 0.5
    assert rho("fine_grained_classification", "classification") < 0.4

    means, across = script.tier_summary(pairs, levels)
    assert means["mid_level"] > across
    assert means["low_level"] > across
    # High-level is the *loosest* tier because it is two clusters averaged
    # together. That ordering is the robust claim; the gap to `across` is not.
    assert means["high_level"] < means["mid_level"] < means["low_level"]


def test_every_board_in_the_corpus_has_a_source(script):
    """SOURCE_IMAGES is hand-written, so a new probe silently falls out of it."""
    boards = script.load_boards(script.CORPUS)
    assert set(boards) <= set(script.SOURCE_IMAGES), (
        f"no SOURCE_IMAGES entry for {', '.join(sorted(set(boards) - set(script.SOURCE_IMAGES)))}"
    )


def test_sharing_images_is_not_what_makes_two_boards_agree(script):
    """The VOC confound test, **reformulated at 14a-4 because the old one broke.**

    Until the seventeenth board there were two boards on the identical 1449
    images and this test asserted two things: that their pair was the weakest of
    the three VOC pairs, and that `generic_segmentation`'s nearest neighbour
    read a different dataset. `instance_segmentation` reads those same 1449
    images and agrees with `generic_segmentation` at **+0.909** -- its nearest
    neighbour of all, and a VOC sibling. The second assertion failed, exactly as
    its own message said it would, and "shared pixels drive agreement" was back
    on the table.

    It does not survive the numbers, and what replaces it is sharper. The three
    same-image pairs span **+0.378 to +0.909** -- a range of 0.531, wider than
    any source group's -- and the *lowest of all six* VOC pairs is a same-image
    one (`instance_segmentation` / `semantic_segmentation`, +0.378). So reading
    identical pixels is not sufficient. And `generic_segmentation` reaches
    +0.881, +0.867 and +0.853 with three boards on three *other* datasets, so it
    is not necessary either.

    What the three same-image boards actually differ in is what they are
    sensitive to: against feature grid they read +0.958, +0.902 and **+0.545**.
    The two resolution-driven ones agree at +0.909; the one that is not agrees
    with neither. That is consistent with the standing corpus finding that
    feature resolution is the strongest correlate of every dense board, and with
    `semantic_segmentation` being the board that ranks by neither objective nor
    resolution.

    Deliberately **not** asserted: that agreement is monotonic in the
    grid-correlation gap. It is not -- `generic`/`semantic` differ by 0.413 and
    agree at +0.538, while `instance`/`semantic` differ by 0.357 and agree at
    +0.378 -- and pinning a mechanism this corpus cannot support is how the
    previous formulation got itself retired.
    """
    boards = script.load_boards(script.CORPUS)
    pairs = script.agreement(boards)

    def rho(a, b):
        return pairs[tuple(sorted((a, b)))]

    same = sorted(script.SAME_IMAGES & set(boards))
    trio = [rho(a, b) for i, a in enumerate(same) for b in same[i + 1 :]]
    assert len(trio) >= 3, "this finding needs at least three same-image boards"

    # Not sufficient: identical pixels span most of the achievable range, and
    # the weakest VOC pair of all is one of them.
    assert max(trio) - min(trio) > 0.4, f"same-image pairs span only {max(trio) - min(trio):.3f}"
    voc = sorted(t for t in boards if script.SOURCE_IMAGES[t] == "VOC")
    voc_pairs = {(a, b): rho(a, b) for i, a in enumerate(voc) for b in voc[i + 1 :]}
    weakest = min(voc_pairs, key=lambda k: voc_pairs[k])
    assert set(weakest) <= script.SAME_IMAGES, (
        f"the weakest VOC pair is {weakest}, which no longer reads identical images -- "
        "the 'not sufficient' half of this finding needs re-deriving"
    )

    # Not necessary: agreement at the same level is reached across datasets.
    others = [
        rho("generic_segmentation", t)
        for t in boards
        if t != "generic_segmentation" and script.SOURCE_IMAGES[t] != "VOC"
    ]
    assert max(others) > 0.8, (
        f"generic_segmentation's best non-VOC partner is only {max(others):+.3f}; "
        "shared pixels would then be the simplest explanation left"
    )


def test_imagenette_is_the_second_counterexample(script, capsys):
    """Three boards on one corpus that barely agree at all.

    classification, retrieval and correspondence all read Imagenette, and their
    mean pairwise rho is far below the within-source average. Sharing a dataset
    is plainly not sufficient for agreement, which is the independent check on
    the VOC result above.
    """
    boards = script.load_boards(script.CORPUS)
    pairs = script.agreement(boards)
    imagenette = [t for t in boards if script.SOURCE_IMAGES[t] == "Imagenette"]
    rhos = [
        pairs[tuple(sorted((a, b)))]
        for i, a in enumerate(sorted(imagenette))
        for b in sorted(imagenette)[i + 1 :]
    ]
    assert sum(rhos) / len(rhos) < 0.3, "Imagenette's boards should not cohere"

    script.report_sources(boards, script.load_levels(script.CORPUS))
    assert "same images" in capsys.readouterr().out

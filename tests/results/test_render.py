"""Rendering rules for comparable records.

The renderer is the last place a number can start meaning something it does not.
Every test here pins a way a *correct* corpus could still produce a misleading
table — which is a different failure from the one `test_leaderboard.py` covers,
and not one the ranking rules can catch.
"""

import pytest

from visbench.results import ResultRecord
from visbench.results.leaderboard import UnknownMetric, comparability_key, group_comparable
from visbench.results.render import (
    CAVEATS,
    HEADLINE_METRICS,
    board_columns,
    render_board,
    render_leaderboard,
)
from visbench.results.schema import utc_timestamp


def make_record(**overrides) -> ResultRecord:
    payload = dict(
        backbone="dinov2_vits14",
        backbone_key="dinov2/dinov2_vits14/224/abcdef012345",
        task="semantic_segmentation",
        level="high_level",
        dataset="voc2012",
        split="val",
        pooling="mean",
        pooling_requested="mean",
        feature_mode="dense_only",
        metrics={"miou": 0.7328, "pixel_acc": 0.9267},
        timestamp=utc_timestamp(),
        visbench_version="0.5.0",
        dataset_fingerprint="deadbeef",
        task_params={"protocol": "visbench_semantic_seg", "epochs": 10},
    )
    payload.update(overrides)
    return ResultRecord(**payload)


def pair(metric_a, metric_b, **kw):
    """Two records differing only in backbone and metrics."""
    return [
        make_record(backbone="small", metrics=metric_a, **kw),
        make_record(backbone="base", metrics=metric_b, **kw),
    ]


# --------------------------------------------------------------------------
# Ordering is declared, never guessed
# --------------------------------------------------------------------------


def test_every_registered_probe_has_a_headline_metric():
    """A probe reaching the renderer without one cannot be rendered at all.

    Checked against the registry rather than a copy of it, so adding a probe and
    forgetting the table fails here instead of at the next corpus run.
    """
    import visbench

    assert set(visbench.list_probes()) <= set(HEADLINE_METRICS)


def test_a_task_with_no_headline_is_refused():
    """Ordering by whatever sorted first asserts a ranking nobody chose."""
    records = pair({"top1": 0.5}, {"top1": 0.6}, task="brand_new_probe")
    with pytest.raises(UnknownMetric, match="No headline metric"):
        render_board(records)


def test_a_board_missing_its_own_headline_is_refused():
    records = pair({"pixel_acc": 0.9}, {"pixel_acc": 0.95})
    with pytest.raises(UnknownMetric, match="not present on every record"):
        render_board(records)


def test_rows_are_ordered_by_the_headline_not_by_input_order():
    records = pair({"miou": 0.60, "pixel_acc": 0.9}, {"miou": 0.75, "pixel_acc": 0.8})
    body = render_board(records)
    assert body.index("`base`") < body.index("`small`")
    assert "Ordered by `miou`" in body


def test_lower_is_better_orders_upwards():
    """Angular error in degrees. The renderer must not assume bigger wins."""
    records = pair(
        {"mean": 29.5, "d1": 0.21},
        {"mean": 30.1, "d1": 0.22},
        task="surface_normal",
        level="mid_level",
    )
    body = render_board(records)
    assert body.index("`small`") < body.index("`base`")


# --------------------------------------------------------------------------
# What a table must not hide
# --------------------------------------------------------------------------


def test_the_winner_is_bolded_per_column_not_per_row():
    """A board whose metrics disagree must show it in the cells.

    Bolding the whole winning row would assert that one backbone won outright,
    which on `edge` is false three different ways.
    """
    records = pair(
        {"edge_correlation": 0.4558, "mae": 0.5028},
        {"edge_correlation": 0.4481, "mae": 0.4972},
        task="edge",
        level="low_level",
        task_params={"protocol": "visbench_edge_regression"},
    )
    body = render_board(records)
    assert "**0.4558**" in body  # small wins the correlation
    assert "**0.4972**" in body  # base wins the MAE
    assert "**0.4481**" not in body
    assert "**0.5028**" not in body


def test_a_disagreeing_headline_says_so():
    records = pair(
        {"edge_correlation": 0.4558, "mae": 0.5028},
        {"edge_correlation": 0.4481, "mae": 0.4972},
        task="edge",
        level="low_level",
        task_params={"protocol": "visbench_edge_regression"},
    )
    body = render_board(records)
    assert "disagrees with `mae`" in body


def test_agreement_says_nothing_rather_than_claiming_consensus():
    records = pair({"miou": 0.60, "pixel_acc": 0.80}, {"miou": 0.75, "pixel_acc": 0.90})
    body = render_board(records)
    assert "disagrees" not in body


def test_diagnostics_are_shown_as_columns_though_they_cannot_be_ranked():
    """`num_matches` is the denominator its own recall is averaged over.

    `rank` refuses it, correctly. A table that then omits it presents a
    comparison whose terms differ as though they did not — which is how the
    corpus nearly published "ResNet-18 is the best correspondence backbone" off
    5.6x fewer, easier candidates.
    """
    records = pair(
        {"recall@5px": 0.8927, "num_matches": 4911.0, "ceiling_recall@5px": 0.9762},
        {"recall@5px": 0.7594, "num_matches": 27590.0, "ceiling_recall@5px": 0.9471},
        task="correspondence",
        level="mid_level",
    )
    body = render_board(records)
    assert "`num_matches`" in body
    assert "4,911" in body and "27,590" in body
    # Shown, but never bolded: "best" is not defined for a denominator.
    assert "**4,911**" not in body and "**27,590**" not in body


def test_context_metrics_are_shown_but_not_ranked():
    records = pair(
        {"recall@5px": 0.89, "ceiling_recall@5px": 0.9762, "num_matches": 4911.0},
        {"recall@5px": 0.76, "ceiling_recall@5px": 0.9471, "num_matches": 27590.0},
        task="correspondence",
        level="mid_level",
    )
    body = render_board(records)
    assert "`ceiling_recall@5px`" in body
    assert "**0.9762**" not in body


def test_a_caveat_travels_with_the_board_it_qualifies():
    records = pair(
        {"recall@5px": 0.89, "num_matches": 4911.0},
        {"recall@5px": 0.76, "num_matches": 27590.0},
        task="correspondence",
        level="mid_level",
    )
    body = render_board(records)
    assert "Read this first" in body
    assert "only unit two backbones can be compared in" in body


def test_every_caveat_names_a_real_task():
    """A caveat keyed on a typo silently never renders."""
    assert set(CAVEATS) <= set(HEADLINE_METRICS)


# --------------------------------------------------------------------------
# Narrowing a board must not launder it
# --------------------------------------------------------------------------


def test_metrics_narrows_the_rankable_columns():
    records = pair(
        {"recall@5px": 0.89, "auc@5px": 0.51, "recall@10px": 0.97, "num_matches": 4911.0},
        {"recall@5px": 0.76, "auc@5px": 0.41, "recall@10px": 0.96, "num_matches": 27590.0},
        task="correspondence",
        level="mid_level",
    )
    body = render_board(records, metrics=["recall@5px"])
    assert "`recall@5px`" in body
    assert "`auc@5px`" not in body


def test_narrowing_cannot_drop_the_denominator():
    """The width fix must not be able to remove what qualifies the numbers.

    Asserted against the header row and the cells, not against the whole page:
    the correspondence caveat *mentions* `num_matches` in prose, so a substring
    check over the rendered board passes whether or not the column is there.
    This test was written that way first and caught nothing.
    """
    records = pair(
        {"recall@5px": 0.89, "auc@5px": 0.51, "num_matches": 4911.0},
        {"recall@5px": 0.76, "auc@5px": 0.41, "num_matches": 27590.0},
        task="correspondence",
        level="mid_level",
    )
    header = render_board(records, metrics=["recall@5px"]).splitlines()[2]
    assert "`num_matches`" in header
    assert "`auc@5px`" not in header


def test_narrowing_cannot_hide_a_disagreement():
    """Trimming for width must not make a board look more consistent."""
    records = pair(
        {"edge_correlation": 0.4558, "mae": 0.5028},
        {"edge_correlation": 0.4481, "mae": 0.4972},
        task="edge",
        level="low_level",
        task_params={"protocol": "visbench_edge_regression"},
    )
    body = render_board(records, metrics=["edge_correlation"])
    assert "`mae`" not in body.split("Ordered by")[0]
    assert "disagrees with `mae`" in body


def test_narrowing_keeps_the_headline_even_if_unlisted():
    records = pair({"miou": 0.60, "pixel_acc": 0.80}, {"miou": 0.75, "pixel_acc": 0.90})
    body = render_board(records, metrics=["pixel_acc"])
    assert "`miou`" in body


def test_narrowing_to_an_unavailable_metric_is_refused():
    records = pair({"miou": 0.6}, {"miou": 0.7})
    with pytest.raises(UnknownMetric, match="Not rankable"):
        render_board(records, metrics=["nonexistent"])


# --------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------


def test_board_columns_splits_rankable_from_qualifying():
    records = pair(
        {"recall@5px": 0.89, "ceiling_recall@5px": 0.97, "num_matches": 4911.0},
        {"recall@5px": 0.76, "ceiling_recall@5px": 0.94, "num_matches": 27590.0},
        task="correspondence",
        level="mid_level",
    )
    ranked, extra = board_columns(records)
    assert ranked == ["recall@5px"]
    assert extra == ["ceiling_recall@5px", "num_matches"]


def test_a_column_with_a_hole_is_not_shown():
    """A missing cell invites reading the gap as a zero."""
    records = [
        make_record(backbone="small", metrics={"miou": 0.6, "pixel_acc": 0.9}),
        make_record(backbone="base", metrics={"miou": 0.7}),
    ]
    ranked, _ = board_columns(records)
    assert ranked == ["miou"]


def test_a_count_keeps_its_integer_form_and_a_rate_does_not():
    """A saturated ceiling of exactly 1.0 must not render as a count."""
    records = pair(
        {"recall@5px": 0.89, "ceiling_recall@5px": 1.0, "num_matches": 4911.0},
        {"recall@5px": 0.76, "ceiling_recall@5px": 1.0, "num_matches": 27590.0},
        task="correspondence",
        level="mid_level",
    )
    body = render_board(records)
    assert "1.0000" in body
    assert "| 1 |" not in body
    assert "4,911" in body


def test_a_diagnostic_that_is_a_rate_keeps_its_decimals():
    """`tie_rate` is a diagnostic and a rate; 0.0 must not render as `0`.

    Formatting on "is a diagnostic" got this wrong on the real similarity board,
    where no triplet tied on any of six backbones.
    """
    records = pair(
        {"accuracy": 0.8701, "tie_rate": 0.0},
        {"accuracy": 0.8580, "tie_rate": 0.0},
        task="similarity",
        level="mid_level",
    )
    body = render_board(records)
    assert "0.0000" in body
    assert "| 0 |" not in body


def test_a_fractional_count_column_is_not_truncated():
    """`detections_per_image` is a mean, not a count."""
    records = pair(
        {"map_50": 0.2895, "detections_per_image": 88.5217, "classes_scored": 20.0},
        {"map_50": 0.2291, "detections_per_image": 83.0333, "classes_scored": 20.0},
        task="detection",
        level="high_level",
    )
    body = render_board(records)
    assert "88.5217" in body
    assert "| 20 |" in body


# --------------------------------------------------------------------------
# The whole corpus
# --------------------------------------------------------------------------


def test_render_leaderboard_emits_one_board_per_group():
    seg = pair({"miou": 0.60}, {"miou": 0.75})
    edge = pair(
        {"edge_correlation": 0.45},
        {"edge_correlation": 0.44},
        task="edge",
        level="low_level",
        dataset="taskonomy",
        task_params={"protocol": "visbench_edge_regression"},
    )
    body = render_leaderboard([*seg, *edge])
    assert body.count("### ") == 2
    assert len(group_comparable([*seg, *edge])) == 2


def test_groups_are_emitted_low_to_high():
    """Fixed order, so an unchanged corpus regenerates to an unchanged file."""
    seg = pair({"miou": 0.60}, {"miou": 0.75})
    edge = pair(
        {"edge_correlation": 0.45},
        {"edge_correlation": 0.44},
        task="edge",
        level="low_level",
        dataset="taskonomy",
        task_params={"protocol": "visbench_edge_regression"},
    )
    body = render_leaderboard([*seg, *edge])
    assert body.index("### edge") < body.index("### semantic_segmentation")


def test_rendering_is_deterministic():
    records = pair({"miou": 0.60}, {"miou": 0.75})
    assert render_leaderboard(records) == render_leaderboard(list(reversed(records)))


def test_a_single_backbone_group_still_renders():
    """It ranks nothing, but dropping it makes a gap look like an absence."""
    body = render_leaderboard([make_record(metrics={"miou": 0.73})])
    assert "### semantic_segmentation" in body
    assert "`dinov2_vits14`" in body


def test_the_group_identity_travels_with_the_board():
    """`describe()` is lossy, so the digest is what distinguishes two boards."""
    records = pair({"miou": 0.60}, {"miou": 0.75})
    key = comparability_key(records[0])
    assert key.short_id() in render_board(records, key=key)


def test_repeats_collapse_to_the_newest():
    """The corpus really does contain a probe run twice."""
    old = make_record(backbone="small", metrics={"miou": 0.60}, timestamp="2026-01-01T00:00:00Z")
    new = make_record(backbone="small", metrics={"miou": 0.75}, timestamp="2026-06-01T00:00:00Z")
    body = render_board([old, new])
    assert "0.7500" in body
    assert "0.6000" not in body

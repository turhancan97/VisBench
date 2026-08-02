"""Comparability rules for result records.

Every test here pins a confusion this codebase has actually had, or came within
one merge of shipping. The module exists because those rules were prose, and
prose does not fail a build.
"""

import pytest

from visbench.results import ResultRecord
from visbench.results.leaderboard import (
    IncomparableRecords,
    UnknownMetric,
    comparability_key,
    group_comparable,
    is_context_metric,
    latest_per_backbone,
    metric_direction,
    rank,
    ranking_disagreements,
    shared_metrics,
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
        pooling="cls",
        feature_mode="dense_only",
        metrics={"miou": 0.732, "miou_per_image": 0.683, "pixel_acc": 0.9267},
        timestamp=utc_timestamp(),
        visbench_version="0.5.0",
        dataset_fingerprint="deadbeef",
        task_params={"protocol": "visbench_semantic_seg", "epochs": 10},
    )
    payload.update(overrides)
    return ResultRecord(**payload)


# --------------------------------------------------------------------------
# Metric directions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("miou", "higher"),
        ("top1", "higher"),
        ("d1", "higher"),
        ("map_50", "higher"),
        ("edge_correlation", "higher"),
        ("rmse", "lower"),
        ("mae", "lower"),
        ("abs_rel", "lower"),
    ],
)
def test_metric_direction(metric, expected):
    assert metric_direction(metric) == expected


def test_angular_error_is_lower_better_despite_its_name():
    """`mean` and `median` are surface-normal angular error, in degrees.

    Nothing about either word says "lower is better", which is exactly why the
    table lists names instead of inferring from them. A heuristic reading
    "mean" as a score would rank the normals leaderboard upside down.
    """
    assert metric_direction("mean") == "lower"
    assert metric_direction("median") == "lower"


@pytest.mark.parametrize(
    "metric",
    ["recall@1", "recall@5", "recall@10", "recall@1p", "recall@0.5p", "auc@1p", "auc@4p"],
)
def test_parametrised_metric_names_have_a_direction(metric):
    """Retrieval and correspondence emit every metric with a parameter attached.

    The real corpus is what caught this: both probes ranked *nothing*, because
    `shared_metrics` skips any name it cannot direct and none of `recall@1`,
    `recall@1p` or `auc@4p` was in the exact table. Two of twelve probes
    produced an empty leaderboard section rather than an error.
    """
    assert metric_direction(metric) == "higher"


def test_parametrised_lookup_is_a_format_not_a_heuristic():
    """Only listed stems resolve; an unknown one still raises."""
    with pytest.raises(UnknownMetric, match="No recorded direction"):
        metric_direction("mystery@5")


def test_context_prefix_still_wins_over_parametrised_stem():
    """`ceiling_recall@1p` is context, even though `recall@` is directable."""
    with pytest.raises(UnknownMetric, match="context"):
        metric_direction("ceiling_recall@1p")


def test_count_metrics_are_diagnostics():
    """These say how much the probe emitted, not how good any of it was."""
    for name in ("detections_per_image", "num_matches"):
        with pytest.raises(UnknownMetric, match="diagnostic"):
            metric_direction(name)


def test_retrieval_and_correspondence_rank_end_to_end():
    """The regression in the shape the corpus actually produced."""
    retrieval = [
        make_record(task="retrieval", backbone="s", metrics={"recall@1": 0.90, "mAP": 0.71}),
        make_record(task="retrieval", backbone="b", metrics={"recall@1": 0.95, "mAP": 0.80}),
    ]
    assert shared_metrics(retrieval) == ["mAP", "recall@1"]
    assert [r.backbone for r, _ in rank(retrieval, "recall@1")] == ["b", "s"]

    correspondence = [
        make_record(
            task="correspondence",
            backbone="s",
            metrics={"recall@1p": 0.78, "auc@1p": 0.6, "ceiling_recall@1p": 0.95},
        ),
        make_record(
            task="correspondence",
            backbone="b",
            metrics={"recall@1p": 0.81, "auc@1p": 0.7, "ceiling_recall@1p": 0.95},
        ),
    ]
    # The ceiling travels with the score and must not become a rankable column.
    assert shared_metrics(correspondence) == ["auc@1p", "recall@1p"]


def test_unknown_metric_raises_rather_than_defaulting():
    with pytest.raises(UnknownMetric, match="No recorded direction"):
        metric_direction("some_new_score")


def test_context_metrics_are_not_rankable():
    """Correspondence's ceiling describes the split, not the representation."""
    assert is_context_metric("ceiling_recall@1px")
    assert not is_context_metric("recall@1px")
    with pytest.raises(UnknownMetric, match="context"):
        metric_direction("ceiling_recall@1px")


def test_diagnostics_are_not_rankable():
    with pytest.raises(UnknownMetric, match="diagnostic"):
        metric_direction("classes_scored")


# --------------------------------------------------------------------------
# Comparability
# --------------------------------------------------------------------------


def test_backbone_is_not_part_of_the_key():
    """The backbone is the thing being compared, so it must not split groups."""
    small = make_record(backbone="dinov2_vits14")
    base = make_record(backbone="dinov2_vitb14", backbone_key="dinov2/vitb/224/ffff")
    assert comparability_key(small) == comparability_key(base)


def test_frozen_and_finetuned_are_never_comparable():
    """A frozen score and a fine-tuned one answer different questions.

    "What does this representation already carry" against "what can it be
    adapted into". Schema v6 added `finetune` precisely so a leaderboard could
    not pool them; this is that field doing its job.
    """
    frozen = make_record(metrics={"miou": 0.7328})
    tuned = make_record(
        metrics={"miou": 0.7758},
        finetune={"blocks": 2, "backbone_lr": 5e-6, "trainable_params": 14_000_000},
    )
    assert comparability_key(frozen) != comparability_key(tuned)
    with pytest.raises(IncomparableRecords):
        rank([frozen, tuned], "miou")


def test_trainable_params_does_not_split_a_finetuned_comparison():
    """ViT-S and ViT-B unfreeze the same *setting* and differ in parameter count.

    Including `trainable_params` in the key would make the one comparison
    fine-tuning exists to support — small against large, same blocks — look
    incomparable.
    """
    small = make_record(
        backbone="dinov2_vits14",
        metrics={"miou": 0.7758},
        finetune={"blocks": 2, "backbone_lr": 5e-6, "trainable_params": 7_000_000},
    )
    base = make_record(
        backbone="dinov2_vitb14",
        metrics={"miou": 0.7992},
        finetune={"blocks": 2, "backbone_lr": 5e-6, "trainable_params": 14_000_000},
    )
    assert comparability_key(small) == comparability_key(base)
    assert [r.backbone for r, _ in rank([small, base], "miou")] == [
        "dinov2_vitb14",
        "dinov2_vits14",
    ]


def test_differing_protocol_is_incomparable():
    """v0.4.0 let `edge` record a keypoint number under the edge protocol.

    The CLI fix stops that being produced; this stops two such records being
    ranked together if one already exists on disk.
    """
    edge = make_record(task="edge", task_params={"protocol": "visbench_edge_regression"})
    keypoint = make_record(task="edge", task_params={"protocol": "visbench_keypoint2d_regression"})
    assert comparability_key(edge) != comparability_key(keypoint)


def test_target_transform_splits_the_group():
    """The log1p occlusion-edge number is not a linear-space one.

    `edge_occlusion` is loaded in log space and nothing else is, because a
    log-space correlation is not the same measurement. `dataset_params` records
    it so the two can never be pooled — this is the check that they are not.
    """
    linear = make_record(task="occlusion_edge", dataset_params={"domain": "edge_occlusion"})
    logged = make_record(
        task="occlusion_edge",
        dataset_params={"domain": "edge_occlusion", "target_transform": "log1p"},
    )
    assert comparability_key(linear) != comparability_key(logged)


def test_training_budget_splits_the_group():
    """0.16 IoU at ten epochs and 0.87 at forty are the same features.

    Ranking a forty-epoch probe above a ten-epoch one would report a schedule
    difference as a representation difference.
    """
    short = make_record(task_params={"protocol": "visbench_binary_seg", "epochs": 10})
    long = make_record(task_params={"protocol": "visbench_binary_seg", "epochs": 40})
    assert comparability_key(short) != comparability_key(long)


def test_dataset_fingerprint_splits_a_limited_run_from_a_full_one():
    full = make_record(dataset_fingerprint="aaaa", dataset_size=1449)
    limited = make_record(dataset_fingerprint="bbbb", dataset_size=600)
    assert comparability_key(full) != comparability_key(limited)


def test_a_vit_and_a_cnn_that_both_asked_for_default_are_comparable():
    """The whole reason schema v7 exists.

    ``pooling="default"`` resolves to ``cls`` on a ViT and ``mean`` on a CNN, so
    a key reading the *resolved* value split every pooled-feature probe along an
    architectural line. Measured on the six-backbone corpus: classification,
    retrieval, correspondence and similarity each became two groups, and
    `resnet50` — which tops two of those boards — could not be ranked against
    DINOv2 at all.
    """
    vit = make_record(backbone="dinov2_vitb14", pooling="cls", pooling_requested="default")
    cnn = make_record(backbone="resnet50", pooling="mean", pooling_requested="default")
    assert comparability_key(vit) == comparability_key(cnn)


def test_explicitly_different_pooling_is_still_incomparable():
    """The request is the protocol, so asking for two different things splits.

    Without this the fix above would be indistinguishable from dropping pooling
    from the key entirely, which would silently rank a CLS-pooled retrieval
    number against a mean-pooled one.
    """
    cls = make_record(pooling="cls", pooling_requested="cls")
    mean = make_record(pooling="mean", pooling_requested="mean")
    assert comparability_key(cls) != comparability_key(mean)


def test_a_v6_record_falls_back_to_its_resolved_pooling():
    """v6 recorded only the resolution, so that is all it can be keyed on.

    Two such records still group with each other exactly as they did before v7 —
    the point of an additive schema is that old files keep their meaning.
    """
    old = make_record(schema_version=6, pooling="cls", pooling_requested=None)
    also_old = make_record(
        schema_version=6, pooling="cls", pooling_requested=None, backbone="other"
    )
    assert comparability_key(old) == comparability_key(also_old)
    assert comparability_key(old).pooling == "cls"


def test_a_v6_record_does_not_silently_join_a_v7_group_that_asked_for_something_else():
    """A v6 `cls` record cannot say whether `cls` was asked for or resolved to.

    So it groups with a v7 record that asked for `cls`, and not with one that
    asked for `default`. That is the conservative direction: the alternative
    would merge a run whose request is unknown into a group defined by a request.
    """
    unknown = make_record(schema_version=6, pooling="cls", pooling_requested=None)
    asked_default = make_record(pooling="cls", pooling_requested="default")
    asked_cls = make_record(pooling="cls", pooling_requested="cls")
    assert comparability_key(unknown) != comparability_key(asked_default)
    assert comparability_key(unknown) == comparability_key(asked_cls)


def test_ignore_relaxes_a_named_setting():
    a = make_record(task_params={"protocol": "p", "epochs": 10, "batch_size": 8})
    b = make_record(task_params={"protocol": "p", "epochs": 10, "batch_size": 16})
    assert comparability_key(a) != comparability_key(b)
    relaxed = {"ignore": ["batch_size"]}
    assert comparability_key(a, **relaxed) == comparability_key(b, **relaxed)


def test_params_compare_by_value_not_by_dict_order():
    a = make_record(dataset_params={"image_size": 224, "domain": "edge_texture"})
    b = make_record(dataset_params={"domain": "edge_texture", "image_size": 224})
    assert comparability_key(a) == comparability_key(b)


def test_unhashable_param_values_do_not_break_the_key():
    """`iou_thresholds` is a list and `layers` is a list; both live in params."""
    record = make_record(task_params={"protocol": "p", "iou_thresholds": [0.5, 0.75]})
    assert comparability_key(record) == comparability_key(record)


def test_group_comparable_partitions():
    frozen_s = make_record(backbone="dinov2_vits14")
    frozen_b = make_record(backbone="dinov2_vitb14")
    tuned = make_record(backbone="dinov2_vits14", finetune={"blocks": 2, "backbone_lr": 5e-6})
    groups = group_comparable([frozen_s, frozen_b, tuned])
    assert len(groups) == 2
    assert sorted(len(v) for v in groups.values()) == [1, 2]


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


def test_rank_higher_is_better():
    small = make_record(backbone="dinov2_vits14", metrics={"miou": 0.732})
    base = make_record(backbone="dinov2_vitb14", metrics={"miou": 0.753})
    assert [r.backbone for r, _ in rank([small, base], "miou")] == [
        "dinov2_vitb14",
        "dinov2_vits14",
    ]


def test_rank_lower_is_better():
    """Taskonomy normals: DINOv2-S wins on mean angular error."""
    small = make_record(task="surface_normal", backbone="dinov2_vits14", metrics={"mean": 26.66})
    base = make_record(task="surface_normal", backbone="dinov2_vitb14", metrics={"mean": 27.37})
    assert [r.backbone for r, _ in rank([small, base], "mean")] == [
        "dinov2_vits14",
        "dinov2_vitb14",
    ]


def test_rank_refuses_incomparable_records():
    a = make_record(dataset="voc2012")
    b = make_record(dataset="ade20k")
    with pytest.raises(IncomparableRecords, match="same question"):
        rank([a, b], "miou")


def test_rank_refuses_a_missing_metric_rather_than_dropping_the_record():
    """Silently ranking the subset would present a partial comparison as whole."""
    complete = make_record(backbone="dinov2_vits14", metrics={"miou": 0.7})
    partial = make_record(backbone="dinov2_vitb14", metrics={"pixel_acc": 0.9})
    with pytest.raises(KeyError, match="missing"):
        rank([complete, partial], "miou")


def test_rank_refuses_when_classes_scored_disagrees():
    """mAP over 18 classes and mAP over 20 are averages of different things.

    `classes_scored` is mAP's real denominator and is not always `num_classes`:
    a class with no non-difficult objects is excluded entirely.
    """
    a = make_record(
        task="detection",
        backbone="dinov2_vits14",
        metrics={"map_50": 0.2127, "classes_scored": 20},
    )
    b = make_record(
        task="detection",
        backbone="dinov2_vitb14",
        metrics={"map_50": 0.2616, "classes_scored": 18},
    )
    with pytest.raises(IncomparableRecords, match="classes_scored"):
        rank([a, b], "map_50")


def test_rank_allows_matching_classes_scored():
    a = make_record(task="detection", backbone="s", metrics={"map_50": 0.21, "classes_scored": 20})
    b = make_record(task="detection", backbone="b", metrics={"map_50": 0.26, "classes_scored": 20})
    assert [r.backbone for r, _ in rank([a, b], "map_50")] == ["b", "s"]


def test_rank_refuses_a_context_metric():
    a = make_record(task="correspondence", backbone="s", metrics={"ceiling_recall@1px": 0.951})
    b = make_record(task="correspondence", backbone="b", metrics={"ceiling_recall@1px": 0.951})
    with pytest.raises(UnknownMetric, match="context"):
        rank([a, b], "ceiling_recall@1px")


def test_rank_of_nothing_is_empty():
    assert rank([], "miou") == []


def test_ties_keep_input_order():
    first = make_record(backbone="alpha", metrics={"miou": 0.5})
    second = make_record(backbone="beta", metrics={"miou": 0.5})
    assert [r.backbone for r, _ in rank([first, second], "miou")] == ["alpha", "beta"]


# --------------------------------------------------------------------------
# Disagreement between metrics
# --------------------------------------------------------------------------


def test_ranking_disagreements_catches_a_task_disagreeing_with_itself():
    """The measured Taskonomy normals case, which is why this function exists.

    DINOv2-S wins on mean angular error; DINOv2-B wins on the 11.25-degree
    threshold. Quoting one and dropping the other manufactures a result.
    """
    small = make_record(
        task="surface_normal",
        backbone="dinov2_vits14",
        metrics={"mean": 26.66, "d1": 0.2727},
    )
    base = make_record(
        task="surface_normal",
        backbone="dinov2_vitb14",
        metrics={"mean": 27.37, "d1": 0.2787},
    )
    disagreements = ranking_disagreements([small, base])
    assert ("d1", "mean") in disagreements
    by_d1, by_mean = disagreements[("d1", "mean")]
    assert by_d1 == ["dinov2_vitb14", "dinov2_vits14"]
    assert by_mean == ["dinov2_vits14", "dinov2_vitb14"]


def test_no_disagreement_is_an_answer_not_an_absence():
    small = make_record(backbone="s", metrics={"miou": 0.732, "pixel_acc": 0.9267})
    base = make_record(backbone="b", metrics={"miou": 0.753, "pixel_acc": 0.9316})
    assert ranking_disagreements([small, base]) == {}


def test_disagreements_need_two_records():
    assert ranking_disagreements([make_record()]) == {}


def test_shared_metrics_excludes_context_and_unknowns():
    a = make_record(metrics={"miou": 0.7, "ceiling_x": 1.0, "mystery": 3.0, "pixel_acc": 0.9})
    b = make_record(metrics={"miou": 0.8, "ceiling_x": 1.0, "mystery": 4.0, "pixel_acc": 0.9})
    assert shared_metrics([a, b]) == ["miou", "pixel_acc"]


def test_shared_metrics_requires_presence_on_every_record():
    a = make_record(metrics={"miou": 0.7, "pixel_acc": 0.9})
    b = make_record(metrics={"miou": 0.8})
    assert shared_metrics([a, b]) == ["miou"]


# --------------------------------------------------------------------------
# Repeats
# --------------------------------------------------------------------------


def test_latest_per_backbone_keeps_the_newest():
    """6a alone wrote five records for one VOC configuration chasing a timing."""
    old = make_record(backbone="dinov2_vits14", timestamp="2026-07-29T10:00:00+00:00")
    new = make_record(backbone="dinov2_vits14", timestamp="2026-07-31T10:00:00+00:00")
    other = make_record(backbone="dinov2_vitb14", timestamp="2026-07-30T10:00:00+00:00")
    kept = latest_per_backbone([old, new, other])
    assert len(kept) == 2
    assert {r.backbone: r.timestamp for r in kept}["dinov2_vits14"] == "2026-07-31T10:00:00+00:00"


def test_describe_distinguishes_groups_that_read_alike():
    """The real corpus contains this pair, and it is why `short_id` exists.

    Two `edge` groups agreeing on task, dataset, split, protocol and
    frozen-ness, differing only in `target_scale` — 6d-1's sweep, where 65535
    scored 0.047 and 1000 scored 0.456. Without the digest they render as one
    group listed twice.
    """
    coarse = make_record(task="edge", dataset_params={"target_scale": 65535.0})
    fine = make_record(task="edge", dataset_params={"target_scale": 1000.0})
    key_coarse = comparability_key(coarse)
    key_fine = comparability_key(fine)

    assert key_coarse != key_fine
    assert key_coarse.describe(with_id=False) == key_fine.describe(with_id=False)
    assert key_coarse.describe() != key_fine.describe()


def test_short_id_is_stable_across_equal_keys():
    first = comparability_key(make_record())
    second = comparability_key(make_record())
    assert first.short_id() == second.short_id()


def test_older_schema_records_group_together():
    """A v1 record predates `dataset_fingerprint`; it must still be readable.

    Refusing them would throw away exactly the history a benchmark accumulates.
    """
    old = make_record(schema_version=1, dataset_fingerprint=None)
    also_old = make_record(schema_version=1, dataset_fingerprint=None, backbone="other")
    assert comparability_key(old) == comparability_key(also_old)

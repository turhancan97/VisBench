"""Tests for the anchor-free detection probe — v0.3, step 6c-3.

The load-bearing ones, in the order they matter:

* :func:`test_decoding_an_oracle_head_output_reproduces_the_boxes` — the
  geometry round trip. Cell centres, the exp/stride parameterisation and the
  decode all have to agree, and a disagreement between any two of them shifts
  every box while still training.
* :func:`test_the_probe_learns_boxes_it_is_given_the_answer_to` — the whole
  pipeline against itself. Features that *encode* the target make a perfect
  score reachable by a linear head, so anything below 1.0 mAP means assignment,
  loss, decoding and metric do not agree about what a box is. Fake backbones
  cannot show a training-dynamics problem, but they can show this.
"""

import math

import pytest
import torch

from visbench.heads import build_head, list_heads, register_head
from visbench.heads.detection import DetectionHead
from visbench.tasks.high_level.detection import DetectionTask

IMAGE_SIZE = 112
GRID = 8
NUM_CLASSES = 3
STRIDE = IMAGE_SIZE / GRID


def make_task(**kwargs):
    defaults = {
        "num_classes": NUM_CLASSES,
        "image_size": IMAGE_SIZE,
        "epochs": 2,
        "batch_size": 4,
        "warmup_epochs": 0.5,
    }
    return DetectionTask(**{**defaults, **kwargs})


def annotation(box, label=0, difficult=False):
    return {
        "boxes": torch.tensor([box], dtype=torch.float32),
        "labels": torch.tensor([label], dtype=torch.int64),
        "difficult": torch.tensor([difficult], dtype=torch.bool),
    }


def teachable(task, box, label):
    """A feature map from which a *linear* head can recover ``box`` exactly.

    Channels ``0..num_classes-1`` are a one-hot class indicator over the cells
    inside the box; the last four are ``log(ltrb / stride)``, which is precisely
    the quantity the regression branch exponentiates. So the ideal head is the
    identity on both branches and a perfect score is reachable — which is what
    makes a *less* than perfect score evidence of an inconsistency rather than
    of a hard learning problem.
    """
    centres = task._centres((GRID, GRID))
    inside = (
        (centres[:, 0] > box[0])
        & (centres[:, 0] < box[2])
        & (centres[:, 1] > box[1])
        & (centres[:, 1] < box[3])
    )
    features = torch.zeros(NUM_CLASSES + 4, GRID * GRID)
    features[label][inside] = 1.0
    ltrb = torch.stack(
        [
            centres[:, 0] - box[0],
            centres[:, 1] - box[1],
            box[2] - centres[:, 0],
            box[3] - centres[:, 1],
        ],
        dim=1,
    )
    features[NUM_CLASSES:, inside] = torch.log(ltrb[inside].clamp(min=1e-3) / STRIDE).T
    return features.reshape(NUM_CLASSES + 4, GRID, GRID)


def teachable_split(task, count=24):
    features, targets = [], []
    for index in range(count):
        label = index % NUM_CLASSES
        x1 = float(5 + (index * 7) % 40)
        y1 = float(5 + (index * 11) % 40)
        box = [x1, y1, x1 + 45.0, y1 + 50.0]
        features.append(teachable(task, box, label))
        targets.append(annotation(box, label))
    return torch.stack(features), targets


# -- geometry ----------------------------------------------------------------


def test_cell_centres_are_row_major_and_offset_by_half_a_stride():
    centres = make_task()._centres((GRID, GRID))
    assert centres.shape == (GRID * GRID, 2)
    assert torch.allclose(centres[0], torch.tensor([STRIDE / 2, STRIDE / 2]))
    # Row-major: the second entry moves in x, not y. A transposed grid would
    # put every prediction in the wrong place and still train.
    assert torch.allclose(centres[1], torch.tensor([1.5 * STRIDE, STRIDE / 2]))
    assert torch.allclose(centres[GRID], torch.tensor([STRIDE / 2, 1.5 * STRIDE]))


def test_distances_are_exponentiated_and_stride_scaled():
    task = make_task()
    raw = torch.zeros(1, GRID * GRID, 4)
    distances = task._distances(raw, (GRID, GRID))
    # exp(0) * stride: one cell wide in every direction, the deliberate
    # scale-free starting point the head's zero bias init produces.
    assert torch.allclose(distances, torch.full_like(distances, STRIDE))


def test_the_distance_exponent_is_clamped_so_one_bad_step_cannot_produce_inf():
    task = make_task()
    distances = task._distances(torch.full((1, 1, 4), 1e4), (GRID, GRID))
    assert torch.isfinite(distances).all()


def test_decoding_an_oracle_head_output_reproduces_the_boxes():
    """Hand-built "perfect" head output must decode back to the exact box.

    The detection counterpart of 6c-2's oracle check: any off-by-one in the
    cell centres, the stride or the corner arithmetic lands *near* the box
    without reaching it, and looks like a weak probe rather than a bug.
    """
    task = make_task()
    centres = task._centres((GRID, GRID))
    box = torch.tensor([10.0, 8.0, 50.0, 48.0])

    inside = (
        (centres[:, 0] > box[0])
        & (centres[:, 0] < box[2])
        & (centres[:, 1] > box[1])
        & (centres[:, 1] < box[3])
    )
    logits = torch.full((GRID * GRID, NUM_CLASSES), -20.0)
    logits[inside, 0] = 20.0
    distances = torch.stack(
        [
            centres[:, 0] - box[0],
            centres[:, 1] - box[1],
            box[2] - centres[:, 0],
            box[3] - centres[:, 1],
        ],
        dim=1,
    )

    decoded = task._decode(logits, distances, centres)
    assert decoded["boxes"].shape == (1, 4)
    assert torch.allclose(decoded["boxes"][0], box, atol=1e-4)
    assert int(decoded["labels"][0]) == 0


def test_decoding_returns_correctly_shaped_empties_when_nothing_clears_the_threshold():
    task = make_task()
    centres = task._centres((GRID, GRID))
    decoded = task._decode(
        torch.full((GRID * GRID, NUM_CLASSES), -50.0),
        torch.ones(GRID * GRID, 4),
        centres,
    )
    assert decoded["boxes"].shape == (0, 4)
    assert decoded["scores"].shape == (0,)
    assert decoded["labels"].shape == (0,)


def test_decoded_boxes_are_clipped_to_the_frame():
    task = make_task()
    centres = task._centres((GRID, GRID))
    logits = torch.full((GRID * GRID, NUM_CLASSES), -20.0)
    logits[0, 0] = 20.0
    decoded = task._decode(logits, torch.full((GRID * GRID, 4), 1e3), centres)
    assert float(decoded["boxes"].min()) >= 0.0
    assert float(decoded["boxes"].max()) <= IMAGE_SIZE


# -- NMS ---------------------------------------------------------------------


def test_nms_suppresses_an_overlapping_duplicate_of_the_same_class():
    task = make_task(nms_iou=0.5)
    boxes = torch.tensor([[0.0, 0.0, 20.0, 20.0], [1.0, 1.0, 21.0, 21.0]])
    keep = task._nms(boxes, torch.tensor([0.9, 0.8]), torch.tensor([0, 0]))
    assert keep.tolist() == [0]


def test_nms_never_suppresses_across_classes():
    """Two classes on the same box are two detections, not one.

    The offset trick is what guarantees it; getting the offset smaller than the
    frame would let a high-scoring car quietly delete the bus underneath it.
    """
    task = make_task(nms_iou=0.5)
    boxes = torch.tensor([[0.0, 0.0, 20.0, 20.0], [0.0, 0.0, 20.0, 20.0]])
    keep = task._nms(boxes, torch.tensor([0.9, 0.8]), torch.tensor([0, 1]))
    assert sorted(keep.tolist()) == [0, 1]


def test_nms_returns_indices_in_descending_score_order():
    task = make_task(nms_iou=0.5)
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
    keep = task._nms(boxes, torch.tensor([0.2, 0.9]), torch.tensor([0, 0]))
    assert keep.tolist() == [1, 0]


# -- assignment --------------------------------------------------------------


def test_a_cell_whose_centre_is_inside_a_box_is_positive_for_its_class():
    task = make_task()
    centres = task._centres((GRID, GRID))
    box = [0.0, 0.0, 3 * STRIDE, 3 * STRIDE]
    class_target, distance_target, positives = task._assign(annotation(box, label=2), centres)

    expected = (
        (centres[:, 0] > box[0])
        & (centres[:, 0] < box[2])
        & (centres[:, 1] > box[1])
        & (centres[:, 1] < box[3])
    )
    assert torch.equal(positives, expected)
    assert torch.equal(class_target[positives].argmax(dim=1), torch.full((int(expected.sum()),), 2))
    # Background cells carry no class at all: focal loss reads all-zeros as
    # "negative for every class", not as "class 0".
    assert float(class_target[~positives].sum()) == 0.0
    # The distances must reconstruct the box from the cell centre.
    first = int(torch.nonzero(positives)[0])
    assert math.isclose(float(centres[first, 0] - distance_target[first, 0]), box[0], abs_tol=1e-4)
    assert math.isclose(float(centres[first, 1] + distance_target[first, 3]), box[3], abs_tol=1e-4)


def test_an_ambiguous_cell_takes_the_smaller_box():
    task = make_task()
    centres = task._centres((GRID, GRID))
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 30.0, 30.0]]),
        "labels": torch.tensor([0, 1]),
        "difficult": torch.tensor([False, False]),
    }
    class_target, _, positives = task._assign(target, centres)
    inside_small = (centres[:, 0] < 30.0) & (centres[:, 1] < 30.0)
    assert positives[inside_small].all()
    assert (class_target[inside_small].argmax(dim=1) == 1).all()


def test_difficult_objects_are_dropped_from_training_targets():
    """Training and scoring make different use of ``difficult``, deliberately.

    The metric needs them *present* so a detection matching one can be ignored
    (VOC's rule, worth 4.3 mAP on VOC val). Training against an object the
    annotators called unreasonable is a separate question, so assignment drops
    them whatever the dataset kept — which is what lets one dataset built with
    ``include_difficult=True`` serve both halves of a run.
    """
    task = make_task()
    centres = task._centres((GRID, GRID))
    _, _, positives = task._assign(
        annotation([0.0, 0.0, 100.0, 100.0], label=0, difficult=True), centres
    )
    assert not bool(positives.any())


def test_an_image_with_no_objects_assigns_nothing_and_does_not_raise():
    task = make_task()
    centres = task._centres((GRID, GRID))
    empty = {
        "boxes": torch.zeros((0, 4)),
        "labels": torch.zeros(0, dtype=torch.int64),
        "difficult": torch.zeros(0, dtype=torch.bool),
    }
    class_target, _, positives = task._assign(empty, centres)
    assert not bool(positives.any())
    assert float(class_target.sum()) == 0.0


# -- losses ------------------------------------------------------------------


def test_giou_loss_is_zero_for_a_perfect_prediction():
    predicted = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
    assert float(DetectionTask._giou_loss(predicted, predicted.clone())) == pytest.approx(
        0.0, abs=1e-5
    )


def test_giou_loss_stays_finite_and_positive_for_disjoint_boxes():
    """The reason it is GIoU and not IoU.

    Plain IoU loss is flat at 1.0 for every disjoint pair, so it has no
    gradient in exactly the state every box starts in. GIoU keeps rising as the
    boxes separate, which is what gives the first epochs something to descend.
    """
    near = DetectionTask._giou_loss(
        torch.tensor([[1.0, 1.0, 1.0, 1.0]]), torch.tensor([[1.0, 1.0, 30.0, 1.0]])
    )
    far = DetectionTask._giou_loss(
        torch.tensor([[1.0, 1.0, 1.0, 1.0]]), torch.tensor([[1.0, 1.0, 300.0, 1.0]])
    )
    assert 0.0 < float(near) < float(far) < math.inf


def test_focal_loss_punishes_a_confident_mistake_far_more_than_a_confident_hit():
    task = make_task()
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    hit = task._focal_loss(torch.tensor([[6.0, -6.0, -6.0]]), targets)
    miss = task._focal_loss(torch.tensor([[-6.0, -6.0, -6.0]]), targets)
    assert float(miss) > 100 * float(hit)


# -- the head ----------------------------------------------------------------


def test_the_detection_head_is_registered_and_emits_num_classes_plus_four():
    assert "detection" in list_heads()
    head = build_head("detection", in_channels=16, num_classes=NUM_CLASSES)
    assert head.out_channels == NUM_CLASSES + 4
    out = head(torch.randn(2, 16, GRID, GRID))
    assert out.shape == (2, NUM_CLASSES + 4, GRID, GRID)


def test_the_classification_bias_starts_at_the_focal_prior():
    """Without the prior a dense head spends its schedule learning that
    background is common, and can sit at ~0 mAP long enough to look broken."""
    head = DetectionHead(in_channels=8, num_classes=NUM_CLASSES, prior_probability=0.01)
    assert torch.allclose(
        torch.sigmoid(head.classifier.bias), torch.full((NUM_CLASSES,), 0.01), atol=1e-6
    )
    assert torch.allclose(head.regressor.bias, torch.zeros(4))


def test_the_head_refuses_a_multi_layer_input():
    head = DetectionHead(in_channels=8, num_classes=NUM_CLASSES)
    with pytest.raises(TypeError, match="single-scale"):
        head([torch.randn(1, 8, GRID, GRID), torch.randn(1, 8, GRID, GRID)])


@register_head("_test_wrong_width_detection_head")
class _WrongWidthHead(DetectionHead):
    """A contributor's head that emits the wrong number of channels.

    The realistic version of this mistake: a head that accepts every argument
    the task passes and quietly produces one channel too few, so the class
    dimension and the box dimension are off by one and every box is decoded
    from a classification logit.
    """

    def __init__(self, in_channels: int, num_classes: int, **kwargs) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes, **kwargs)
        self.out_channels = num_classes + 3


def test_the_task_refuses_a_head_of_the_wrong_width():
    task = make_task(head="_test_wrong_width_detection_head")
    with pytest.raises(ValueError, match="emits"):
        task._build_head(16)


# -- training and scoring ----------------------------------------------------


def test_the_probe_learns_boxes_it_is_given_the_answer_to():
    """A perfect score must be *reachable*, or the parts disagree.

    Assignment, the two losses, the exp/stride decoding and VOC's AP all have to
    describe the same box for this to reach 1.0. It is the cheapest available
    end-to-end check that they do — the real proof is ``examples/detect.py`` on
    a real backbone, which this cannot replace.
    """
    torch.manual_seed(0)
    task = make_task(epochs=30, lr=5e-2, warmup_epochs=1.0)
    features, targets = teachable_split(task)
    task.fit(features, targets)

    metrics = task.evaluate(features, targets)
    assert metrics["map_50"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["map_50_95"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["classes_scored"] == float(NUM_CLASSES)
    assert task.train_loss is not None and task.train_loss < 0.5


def test_predict_returns_one_entry_per_image_in_dataset_order():
    torch.manual_seed(0)
    task = make_task()
    features, targets = teachable_split(task, count=8)
    task.fit(features, targets)
    predictions = task.predict(features, targets)
    assert len(predictions) == len(targets)
    for prediction in predictions:
        assert set(prediction) == {"boxes", "scores", "labels"}
        assert prediction["boxes"].shape[0] == prediction["scores"].shape[0]


def test_evaluate_is_deterministic_across_repeated_calls():
    torch.manual_seed(0)
    task = make_task()
    features, targets = teachable_split(task, count=8)
    task.fit(features, targets)
    assert task.evaluate(features, targets) == task.evaluate(features, targets)


def test_no_prediction_ever_exceeds_max_detections():
    torch.manual_seed(0)
    task = make_task(max_detections=3, score_threshold=0.0)
    features, targets = teachable_split(task, count=8)
    task.fit(features, targets)
    assert all(len(p["scores"]) <= 3 for p in task.predict(features, targets))


# -- guards ------------------------------------------------------------------


def test_a_box_beyond_the_declared_image_size_is_refused():
    """The silent failure this probe is most exposed to.

    Boxes are absolute post-transform pixels, so a probe and a dataset that
    disagree about ``image_size`` put every cell centre in the wrong place —
    and the run trains, and scores badly, and reads as a weak backbone.
    """
    task = make_task()
    with pytest.raises(ValueError, match="image_size"):
        task._assign(annotation([0.0, 0.0, 400.0, 400.0]), task._centres((GRID, GRID)))


def test_a_label_outside_num_classes_is_refused():
    task = make_task()
    with pytest.raises(ValueError, match="num_classes"):
        task._assign(annotation([1.0, 1.0, 40.0, 40.0], label=99), task._centres((GRID, GRID)))


def test_evaluating_before_fitting_raises():
    task = make_task()
    features, targets = teachable_split(task, count=4)
    with pytest.raises(RuntimeError, match="not been fitted"):
        task.evaluate(features, targets)


def test_features_on_a_different_grid_than_the_fit_are_refused():
    torch.manual_seed(0)
    task = make_task()
    features, targets = teachable_split(task, count=4)
    task.fit(features, targets)
    with pytest.raises(ValueError, match="grid"):
        task.evaluate(torch.randn(4, NUM_CLASSES + 4, GRID + 2, GRID + 2), targets)


def test_fitting_without_annotations_raises():
    task = make_task()
    with pytest.raises(ValueError, match="annotations"):
        task.fit(torch.randn(4, 8, GRID, GRID))


def test_fitting_on_an_empty_split_raises():
    task = make_task()
    with pytest.raises(ValueError, match="empty"):
        task.fit(torch.zeros(0, 8, GRID, GRID), [])


def test_multi_layer_features_are_refused_rather_than_silently_truncated():
    task = make_task()
    with pytest.raises(TypeError, match="single-scale"):
        task.fit([torch.randn(4, 8, GRID, GRID)] * 2, [annotation([1.0, 1.0, 40.0, 40.0])] * 4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_classes": 0},
        {"image_size": 0},
        {"batch_size": 0},
        {"epochs": 0},
        {"warmup_epochs": 10.0},
        {"score_threshold": 1.0},
        {"nms_iou": 0.0},
        {"max_detections": 0},
        {"focal_alpha": 0.0},
        {"focal_gamma": -1.0},
    ],
)
def test_nonsense_settings_are_refused_at_construction(kwargs):
    with pytest.raises(ValueError):
        make_task(**kwargs)


# -- provenance --------------------------------------------------------------


def test_describe_records_the_decoding_settings_and_does_not_claim_probe3d():
    params = make_task().describe()["task_params"]
    # probe3d has no detection task and VOC defines a metric, not this head.
    assert params["protocol"] == "visbench_anchor_free_det"
    for key in ("score_threshold", "nms_iou", "max_detections", "image_size", "num_classes"):
        assert key in params


def test_the_probe_declares_itself_dense_frozen_and_not_pairwise():
    task = make_task()
    assert task.uses_dense is True
    assert task.uses_pairs is False
    assert task.zero_shot is False
    # Fine-tuning (6a/6b) lives on DenseTrainingTask; detection does not
    # inherit it, and must not claim a finetune record it never produced.
    assert task.finetune_blocks == 0
    assert task.finetune() is None

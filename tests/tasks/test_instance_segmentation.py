"""Detect-then-segment instance segmentation — step 14a-3.

The load-bearing test is :func:`test_teachable_features_reach_a_perfect_score`,
which is 6c-3's argument applied one level up: features that *are* the head's
ideal output make a perfect score reachable, so **anything less is evidence of
an inconsistency in the pipeline rather than of a hard learning problem**. It
caught the only real bug in this step — an earlier fixture whose boxes were
smaller than two grid cells, which trained to 0.0 mAP and looked like broken
code.

The mask half of that fixture is deliberately **not** a rectangle. A mask
identical to its own box is all-ones inside the RoI, so predicting all-ones
scores perfectly and the mask branch is never tested at all; the fixture
removes a cell-aligned quadrant so there is something to recover.

Everything else guards a silent failure. Masks are paired with boxes by row, so
filtering one alone pairs an instance's box with another's outline while every
length still matches. RoIAlign takes a single spatial scale, so a rectangular
grid would be sampled at the wrong scale on one axis with no shape disagreeing.
And the mask branch lives *inside* the head rather than beside it, because
fitted state outside ``self.head`` is a bug this codebase has already shipped
once (9a).
"""

import pytest
import torch
import torch.nn as nn

from visbench.heads import build_head, list_heads
from visbench.heads.instance import InstanceHead
from visbench.tasks.high_level.instance_segmentation import InstanceSegmentationTask

IMAGE_SIZE = 112
GRID = 8
NUM_CLASSES = 3
STRIDE = IMAGE_SIZE / GRID
CHANNELS = NUM_CLASSES + 5


def make_task(**kwargs):
    defaults = {
        "num_classes": NUM_CLASSES,
        "image_size": IMAGE_SIZE,
        "epochs": 2,
        "batch_size": 4,
        "warmup_epochs": 0.5,
        "mask_size": 14,
    }
    return InstanceSegmentationTask(**{**defaults, **kwargs})


def quadrant_mask(box):
    """The box, minus its cell-aligned top-left quarter.

    Not a rectangle, so a mask branch predicting all-ones inside the RoI scores
    badly — which is what makes the mask half of the perfect-score test mean
    anything.
    """
    x_min, y_min, x_max, y_max = (int(value) for value in box)
    mask = torch.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=torch.bool)
    mask[y_min:y_max, x_min:x_max] = True
    mask[y_min : (y_min + y_max) // 2, x_min : (x_min + x_max) // 2] = False
    return mask


def annotation(box, label=0, mask=None, difficult=False, ignore=None):
    return {
        "boxes": torch.tensor([box], dtype=torch.float32),
        "labels": torch.tensor([label], dtype=torch.int64),
        "masks": (quadrant_mask(box) if mask is None else mask)[None],
        "difficult": torch.tensor([difficult], dtype=torch.bool),
        "ignore": torch.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=torch.bool)
        if ignore is None
        else ignore,
    }


def teachable(task, box, label, mask):
    """Features from which a *linear* head can recover the box and the mask.

    Channels ``0..num_classes-1`` are a one-hot class indicator over the cells
    inside the box and the next four are ``log(ltrb / stride)`` — exactly what
    ``DetectionTask``'s own fixture builds, because the box half of this probe
    is that class unchanged. The last channel carries the mask **pooled to the
    grid**, which is what a per-patch feature can actually say about an outline.
    """
    centres = task._centres((GRID, GRID))
    inside = (
        (centres[:, 0] > box[0])
        & (centres[:, 0] < box[2])
        & (centres[:, 1] > box[1])
        & (centres[:, 1] < box[3])
    )
    features = torch.zeros(CHANNELS, GRID * GRID)
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
    features[NUM_CLASSES : NUM_CLASSES + 4, inside] = torch.log(
        ltrb[inside].clamp(min=1e-3) / STRIDE
    ).T
    grid = features.reshape(CHANNELS, GRID, GRID)
    grid[NUM_CLASSES + 4] = nn.functional.adaptive_avg_pool2d(
        mask[None, None].float(), (GRID, GRID)
    )[0, 0]
    return grid


def teachable_split(task, count=24):
    features, targets = [], []
    for index in range(count):
        label = index % NUM_CLASSES
        x_min = float(5 + (index * 7) % 40)
        y_min = float(5 + (index * 11) % 40)
        box = [x_min, y_min, x_min + 56.0, y_min + 56.0]
        mask = quadrant_mask(box)
        features.append(teachable(task, box, label, mask))
        targets.append(annotation(box, label, mask=mask))
    return torch.stack(features), targets


class TestTheHead:
    def test_it_is_registered_and_emits_detection_channels(self):
        assert "instance" in list_heads()
        head = build_head("instance", in_channels=16, num_classes=NUM_CLASSES)
        assert head.out_channels == NUM_CLASSES + 4
        output = head(torch.randn(2, 16, GRID, GRID))
        assert output.shape == (2, NUM_CLASSES + 4, GRID, GRID)

    def test_mask_logits_are_one_channel_and_class_agnostic(self):
        """One channel, not num_classes: the class is already decided."""
        head = InstanceHead(in_channels=16, num_classes=NUM_CLASSES)
        logits = head.mask_logits(torch.randn(5, 16, 14, 14))
        assert logits.shape == (5, 1, 14, 14)

    def test_the_default_mask_branch_is_a_single_1x1_convolution(self):
        """What makes a difference between backbones a difference of features."""
        head = InstanceHead(in_channels=16, num_classes=NUM_CLASSES)
        assert isinstance(head.mask_stem, nn.Identity)
        assert isinstance(head.mask_predictor, nn.Conv2d)
        assert head.mask_predictor.kernel_size == (1, 1)
        # in_channels * 1 output + 1 bias, and nothing else in the branch.
        parameters = sum(p.numel() for p in head.mask_predictor.parameters())
        assert parameters == 16 + 1

    def test_the_mask_bias_starts_at_zero_not_at_the_focal_prior(self):
        """A RoI is a detected object's box, so half its pixels are foreground.

        Copying the classification branch's ``-log((1-pi)/pi)`` prior here would
        start every mask empty.
        """
        head = InstanceHead(in_channels=8, num_classes=NUM_CLASSES)
        assert float(head.mask_predictor.bias.detach()) == 0.0
        assert float(head.detection.classifier.bias.detach()[0]) < -4.0

    def test_an_empty_roi_batch_returns_a_shaped_empty(self):
        head = InstanceHead(in_channels=8, num_classes=NUM_CLASSES)
        assert head.mask_logits(torch.zeros(0, 8, 14, 14)).shape == (0, 1, 14, 14)

    def test_wrong_roi_channels_are_refused(self):
        head = InstanceHead(in_channels=8, num_classes=NUM_CLASSES)
        with pytest.raises(ValueError, match="built for 8"):
            head.mask_logits(torch.randn(2, 16, 14, 14))

    def test_both_branches_are_in_one_state_dict(self):
        """The 9a lesson: fitted state outside ``self.head`` does not round-trip.

        A mask convolution held beside the head would be absent from
        ``head.state_dict()``, so a saved probe would load its boxes and predict
        blank masks.
        """
        head = InstanceHead(in_channels=8, num_classes=NUM_CLASSES)
        keys = set(head.state_dict())
        assert any(key.startswith("mask_predictor") for key in keys)
        assert any(key.startswith("detection.") for key in keys)

    def test_a_negative_mask_hidden_dim_is_refused(self):
        with pytest.raises(ValueError, match="mask_hidden_dim"):
            InstanceHead(in_channels=8, num_classes=1, mask_hidden_dim=-1)


class TestTheProbeIsConfiguredHonestly:
    def test_it_is_registered_under_its_own_name(self):
        """Registered at 14a-4, with the board that gives its tables something to say.

        The name is what a dozen fixed tables key on, so this asserts the
        registry agrees with the class rather than merely that the name exists:
        a probe registered under one name whose class declares another would
        write records no board could find.
        """
        from visbench import get_probe, list_probes

        assert "instance_segmentation" in list_probes()
        assert InstanceSegmentationTask.name == "instance_segmentation"
        probe = get_probe("instance_segmentation", num_classes=NUM_CLASSES)
        assert isinstance(probe, InstanceSegmentationTask)
        assert probe.level == "high_level"

    def test_task_params_record_everything_that_shaped_the_number(self):
        params = make_task(mask_size=10, mask_weight=2.0).describe()["task_params"]
        assert params["protocol"] == "visbench_anchor_free_instance"
        assert params["mask_size"] == 10
        assert params["mask_weight"] == 2.0
        assert params["mask_threshold"] == 0.5
        assert params["mask_train_boxes"] == "ground_truth"
        assert params["mask_classes"] == "agnostic"
        # Inherited detection settings must still be there.
        assert params["nms_iou"] == 0.5
        assert params["score_threshold"] == 0.05

    def test_the_protocol_does_not_claim_mask_rcnn(self):
        """Four convolutions on an FPN is not what this is; overclaiming is
        worse than saying nothing."""
        protocol = make_task().describe()["task_params"]["protocol"]
        assert "rcnn" not in protocol.lower()
        assert protocol.startswith("visbench_")

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"mask_size": 0}, "mask_size"),
            ({"mask_weight": -1.0}, "mask_weight"),
            ({"mask_threshold": 0.0}, "mask_threshold"),
            ({"mask_threshold": 1.0}, "mask_threshold"),
            ({"mask_hidden_dim": -1}, "mask_hidden_dim"),
        ],
    )
    def test_out_of_range_settings_are_refused(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            make_task(**kwargs)

    def test_a_head_without_a_mask_branch_is_refused_by_name(self):
        """Rather than an AttributeError partway through the first batch."""
        task = make_task(head="detection")
        with pytest.raises(ValueError, match="no mask_logits"):
            task._build_head(CHANNELS)


class TestGeometry:
    def test_a_non_square_feature_grid_is_refused(self):
        """roi_align takes one scalar scale, so a rectangular grid would be
        sampled at the wrong scale on one axis with no shape disagreeing."""
        task = make_task()
        with pytest.raises(ValueError, match="square feature grid"):
            task._spatial_scale((8, 16))

    def test_the_spatial_scale_is_cells_per_pixel(self):
        assert make_task()._spatial_scale((GRID, GRID)) == GRID / IMAGE_SIZE

    def test_a_mask_equal_to_its_box_gives_an_all_ones_target(self):
        """The RoIAlign target path, on the case whose answer is known.

        Also states why the perfect-score fixture uses a non-rectangular mask:
        this target is all ones, so predicting all ones would score perfectly.
        """
        task = make_task()
        box = torch.tensor([[20.0, 20.0, 76.0, 76.0]])
        mask = torch.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=torch.bool)
        mask[20:76, 20:76] = True
        target, weight = task._mask_targets(mask[None], torch.zeros_like(mask), box)
        assert target.shape == (1, 1, 14, 14)
        assert float(target.min()) > 0.99
        assert float(weight.min()) > 0.99

    def test_void_pixels_become_zero_loss_weight_not_background(self):
        task = make_task()
        box = torch.tensor([[20.0, 20.0, 76.0, 76.0]])
        mask = torch.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=torch.bool)
        mask[20:76, 20:76] = True
        ignore = torch.zeros_like(mask)
        ignore[20:48, 20:76] = True  # the top half of the box is void
        _, weight = task._mask_targets(mask[None], ignore, box)
        assert float(weight[0, 0, :6].max()) < 0.01  # void rows carry no weight
        assert float(weight[0, 0, -1].min()) > 0.99  # labelled rows carry full

    def test_empty_targets_give_shaped_empties(self):
        task = make_task()
        target, weight = task._mask_targets(
            torch.zeros((0, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.bool),
            torch.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=torch.bool),
            torch.zeros((0, 4)),
        )
        assert target.shape == (0, 1, 14, 14)
        assert weight.shape == (0, 1, 14, 14)


class TestPastingMasksBack:
    def test_a_confident_roi_fills_its_box_and_nothing_else(self):
        task = make_task()
        boxes = torch.tensor([[20.0, 30.0, 60.0, 70.0]])
        probabilities = torch.ones((1, 1, 14, 14))
        pasted = task._paste_masks(probabilities, boxes)
        assert pasted.shape == (1, IMAGE_SIZE, IMAGE_SIZE)
        assert bool(pasted[0, 40, 40])
        assert not bool(pasted[0, 10, 10])
        assert not bool(pasted[0, 90, 90])

    def test_the_threshold_is_applied_after_resizing(self):
        task = make_task(mask_threshold=0.5)
        boxes = torch.tensor([[20.0, 20.0, 60.0, 60.0]])
        assert not task._paste_masks(torch.full((1, 1, 14, 14), 0.4), boxes).any()
        assert task._paste_masks(torch.full((1, 1, 14, 14), 0.6), boxes).any()

    def test_no_detections_gives_a_shaped_empty(self):
        pasted = make_task()._paste_masks(torch.zeros((0, 1, 14, 14)), torch.zeros((0, 4)))
        assert pasted.shape == (0, IMAGE_SIZE, IMAGE_SIZE)
        assert pasted.dtype == torch.bool

    def test_a_degenerate_box_is_skipped_rather_than_raising(self):
        task = make_task()
        boxes = torch.tensor([[50.0, 50.0, 50.0, 50.0]])
        pasted = task._paste_masks(torch.ones((1, 1, 14, 14)), boxes)
        assert pasted.shape == (1, IMAGE_SIZE, IMAGE_SIZE)


class TestAnnotationsArePairedByRow:
    def test_masks_are_filtered_with_their_boxes(self):
        """The hazard: filtering one alone pairs a box with another's outline."""
        task = make_task()
        first = quadrant_mask([5.0, 5.0, 40.0, 40.0])
        second = quadrant_mask([60.0, 60.0, 100.0, 100.0])
        target = {
            "boxes": torch.tensor([[5.0, 5.0, 40.0, 40.0], [60.0, 60.0, 100.0, 100.0]]),
            "labels": torch.tensor([0, 1]),
            "masks": torch.stack([first, second]),
            "difficult": torch.tensor([True, False]),
            "ignore": torch.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=torch.bool),
        }
        boxes, labels, masks, _ = task._boxes_labels_masks(target, drop_difficult=True)
        assert boxes.shape[0] == labels.shape[0] == masks.shape[0] == 1
        assert labels.tolist() == [1]
        # The surviving mask must be the SECOND one, not the first.
        assert torch.equal(masks[0], second)

    def test_a_row_count_disagreement_is_refused(self):
        task = make_task()
        target = {
            "boxes": torch.tensor([[5.0, 5.0, 40.0, 40.0]]),
            "labels": torch.tensor([0]),
            "masks": torch.zeros((2, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.bool),
        }
        with pytest.raises(ValueError, match="paired by row"):
            task._boxes_labels_masks(target)

    def test_an_annotation_without_masks_names_the_dataset_that_has_them(self):
        task = make_task()
        with pytest.raises(ValueError, match="VOCInstanceDataset"):
            task._boxes_labels_masks(
                {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)}
            )

    def test_an_ignore_map_of_the_wrong_size_is_refused(self):
        task = make_task()
        target = {
            "boxes": torch.tensor([[5.0, 5.0, 40.0, 40.0]]),
            "labels": torch.tensor([0]),
            "masks": torch.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.bool),
            "ignore": torch.zeros((8, 8), dtype=torch.bool),
        }
        with pytest.raises(ValueError, match="same frame"):
            task._boxes_labels_masks(target)


class TestEndToEnd:
    def test_teachable_features_reach_a_perfect_score(self):
        """The load-bearing test: a perfect score is reachable, so a low one is a bug.

        6c-3's argument one level up. The masks are non-rectangular, so this
        exercises the mask branch rather than rewarding an all-ones prediction.
        """
        task = make_task(epochs=60, lr=1e-2, warmup_epochs=1.0)
        features, targets = teachable_split(task)
        task.fit(features, targets)
        metrics = task.evaluate(features, targets)
        assert metrics["mask_map_50"] == pytest.approx(1.0)
        assert metrics["box_map_50"] == pytest.approx(1.0)

    def test_predict_emits_masks_beside_boxes(self):
        task = make_task()
        features, targets = teachable_split(task, count=8)
        task.fit(features, targets)
        predictions = task.predict(features, targets)
        assert len(predictions) == 8
        for prediction in predictions:
            assert set(prediction) == {"boxes", "scores", "labels", "masks"}
            count = prediction["boxes"].shape[0]
            assert prediction["masks"].shape == (count, IMAGE_SIZE, IMAGE_SIZE)
            # Stored as bool, which is what makes collecting a whole split's
            # predictions feasible: 5.4 GB over VOC val at the rate DINOv2-S
            # actually decodes, against 21.6 GB as float32.
            assert prediction["masks"].dtype == torch.bool

    def test_evaluate_reports_the_box_number_beside_the_mask_one(self):
        """So a low mask AP is attributable to outlines or to localisation."""
        task = make_task()
        features, targets = teachable_split(task, count=8)
        task.fit(features, targets)
        metrics = task.evaluate(features, targets)
        for key in ("mask_map_50", "mask_map_50_95", "box_map_50", "detections_per_image"):
            assert key in metrics

    def test_the_mask_loss_is_reported_separately_and_only_after_fit(self):
        task = make_task()
        assert task.training_summary() is None
        features, targets = teachable_split(task, count=8)
        task.fit(features, targets)
        summary = task.training_summary()
        assert summary is not None
        assert "train_loss" in summary and "train_mask_loss" in summary
        assert summary["train_mask_loss"] >= 0.0

    def test_predicting_before_fitting_raises(self):
        task = make_task()
        features, targets = teachable_split(task, count=4)
        with pytest.raises(RuntimeError, match="has not been fitted"):
            task.predict(features, targets)

    def test_probe_state_carries_the_grid_and_the_head_carries_the_mask_branch(self):
        """Together these are everything a saved probe needs to predict again."""
        task = make_task()
        features, targets = teachable_split(task, count=8)
        task.fit(features, targets)
        state = task.probe_state()
        assert "grid_hw" in state
        assert state["grid_hw"].tolist() == [GRID, GRID]

        restored = make_task()
        restored.load_probe_state(state)
        assert restored.grid_hw == (GRID, GRID)
        restored.head = task.head
        # Predicting works from restored state alone.
        assert len(restored.predict(features, targets)) == 8

    def test_a_head_without_a_mask_branch_is_refused_by_name(self):
        """The mask half of this probe cannot run on a plain DetectionHead.

        Reachable only through a `probe_state` round trip, which is exactly how
        9a's `grid_hw` bug shipped: the failure without this check is
        `AttributeError` deep inside a batch loop, or — worse — a head that
        satisfies `nn.Module.__getattr__` and returns something that is not a
        mask.
        """
        task = make_task()
        features, targets = teachable_split(task, count=4)
        task.fit(features, targets)
        task.head = build_head("detection", in_channels=CHANNELS, num_classes=NUM_CLASSES)
        with pytest.raises(TypeError, match="no mask branch"):
            task.predict(features, targets)

    def test_an_image_whose_instances_were_all_cropped_away_still_trains(self):
        """Legitimate on VOC, and it must not divide by zero or drop a branch."""
        task = make_task()
        features, targets = teachable_split(task, count=8)
        empty = {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros(0, dtype=torch.int64),
            "masks": torch.zeros((0, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.bool),
            "difficult": torch.zeros(0, dtype=torch.bool),
            "ignore": torch.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=torch.bool),
        }
        targets = [empty] + targets[1:]
        task.fit(features, targets)
        assert task.train_loss is not None

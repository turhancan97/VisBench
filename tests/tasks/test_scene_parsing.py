"""Scene parsing — a thin subclass of the semantic-segmentation probe.

The mechanics (the head, the schedule, cross-entropy with `ignore_index`, both
mIoU reductions) are the parent's and are covered by
`test_semantic_segmentation.py`. What this module pins is the part that is *not*
inherited — the distinct name, level and protocol string that keep the two
boards apart — and the one thing this probe decides for itself: that the class
count defaults to the label set it is named for.

**The label convention is the thing that would be silent when wrong**, so the
fixtures here use NYUv2's shape rather than a convenient one: contiguous
zero-indexed classes with 255 marking void. Reading 0 as unlabelled would
discard a fifth of every real image and train the probe never to answer `wall`;
leaving 255 as a class would ask a forty-class head to predict index 255.
"""

import numpy as np
import pytest
import torch
from PIL import Image

import visbench
from visbench.cli.datasets import showable_probes, supported_probes
from visbench.data.dense import DenseFolderDataset, load_label_map
from visbench.results.render import HEADLINE_METRICS
from visbench.tasks.high_level.scene_parsing import NYU40_CLASSES, NYU40_VOID, SceneParsingTask
from visbench.tasks.high_level.semantic_segmentation import IGNORE_INDEX
from visbench.viz import show_probes


@pytest.fixture
def probe():
    return visbench.get_probe("scene_parsing", device="cpu")


def test_it_is_a_registered_probe():
    assert "scene_parsing" in visbench.list_probes()


def test_it_is_wired_through_every_fixed_table():
    """A new probe must appear in all four sets or the show/CLI tests fail."""
    assert "scene_parsing" in supported_probes()
    assert "scene_parsing" in showable_probes()
    assert "scene_parsing" in show_probes()
    assert "scene_parsing" in HEADLINE_METRICS


def test_identity_is_distinct_from_semantic_segmentation(probe):
    """The name is what gives it its own board; the protocol says what it is.

    Both matter and they fail differently: a shared *name* makes the VOC board
    unrenderable, while a shared protocol only misdescribes the record.
    """
    voc = visbench.get_probe("semantic_segmentation", num_classes=21, device="cpu")
    assert probe.name == "scene_parsing" and voc.name == "semantic_segmentation"
    assert probe.level == voc.level == "high_level"
    assert probe.describe()["task_params"]["protocol"] == "visbench_scene_parsing"
    assert voc.describe()["task_params"]["protocol"] == "visbench_semantic_seg"
    assert isinstance(probe, type(voc))


def test_the_class_count_defaults_to_the_label_set_it_is_named_for(probe):
    """The base refuses a default because the count belongs to the dataset.

    This probe *is* named for a label set, so the count is a protocol pin —
    the `corner` argument — and it still travels in `task_params`, so a run at
    another count lands in its own comparability group.
    """
    assert probe.num_classes == NYU40_CLASSES == 40
    assert visbench.get_probe("scene_parsing", num_classes=13).num_classes == 13
    assert probe.describe()["task_params"]["num_classes"] == 40


def test_headline_metric_is_the_dataset_level_miou():
    """Not `miou_per_image`: the dataset-level one is what the segmentation
    literature defines and the only one comparable with published numbers."""
    assert HEADLINE_METRICS["scene_parsing"] == "miou"


def test_the_void_value_becomes_the_ignore_index_before_the_loss_sees_it(tmp_path):
    """255 in, IGNORE_INDEX out — the wiring the whole probe rests on.

    Left unmapped, a forty-class head would be asked to predict index 255; a
    real run's loss would either raise or train against a category that is not
    in the label set.
    """
    images, targets = tmp_path / "images", tmp_path / "labels"
    images.mkdir(parents=True)
    targets.mkdir()
    label = np.zeros((16, 16), dtype=np.uint8)
    label[:4] = 39  # the last real class
    label[-4:] = NYU40_VOID  # the border margin, as NYUv2 stores it
    Image.new("RGB", (16, 16), (12, 34, 56)).save(images / "a.png")
    Image.fromarray(label).save(targets / "a.png")

    dataset = DenseFolderDataset(
        tmp_path,
        target_dir="labels",
        target_loader=lambda path: load_label_map(path, ignore_index=NYU40_VOID),
        image_size=16,
    )
    target = dataset.target(0)
    assert int(target.max()) == 39, "the real classes must survive untouched"
    assert int(target.min()) == IGNORE_INDEX, "255 must arrive as the ignore index"
    assert not (target == NYU40_VOID).any(), "no raw void value may reach the loss"


def test_zero_is_a_real_class_and_is_not_masked(tmp_path):
    """The half of the convention that is easy to get backwards.

    NYU40's 0 is `wall` — 21.4% of all pixels in the real split and 28.6% of
    interior ones. Treating it as unlabelled, which is the other common
    convention for this dataset, would discard a fifth of every image.
    """
    images, targets = tmp_path / "images", tmp_path / "labels"
    images.mkdir(parents=True)
    targets.mkdir()
    Image.new("RGB", (8, 8), (12, 34, 56)).save(images / "a.png")
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(targets / "a.png")

    dataset = DenseFolderDataset(
        tmp_path,
        target_dir="labels",
        target_loader=lambda path: load_label_map(path, ignore_index=NYU40_VOID),
        image_size=8,
    )
    assert int(dataset.target(0).unique().item()) == 0, "class 0 must stay class 0"


def test_it_fits_and_scores_like_its_parent():
    """One fit on features that encode the answer, so a failure here is the
    subclass rather than the representation.

    ``epochs=60, lr=5e-2`` rather than the defaults, for the reason
    `test_semantic_segmentation.py` uses ``epochs=200, lr=2e-1`` on its own
    learnability fixtures: the ten-epoch schedule assumes NYUv2-sized data, and
    six synthetic images at 5e-4 measures the schedule rather than the probe.
    """
    torch.manual_seed(0)
    grid, classes = 8, 4
    targets = torch.randint(0, classes, (6, grid, grid))
    features = torch.zeros(6, classes, grid, grid)
    features.scatter_(1, targets.unsqueeze(1), 5.0)

    probe = SceneParsingTask(num_classes=classes, epochs=60, lr=5e-2, batch_size=2, device="cpu")
    probe.fit({"dense": features}, targets)
    metrics = probe.evaluate({"dense": features}, targets)
    assert metrics["miou"] > 0.5
    assert set(metrics) >= {"miou", "miou_per_image", "pixel_acc", "mean_acc"}
    assert probe.training_summary()["train_loss"] > 0

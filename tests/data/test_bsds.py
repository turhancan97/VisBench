"""BSDS500 — the dataset that keeps several people's answers instead of one.

Every other dense dataset here pairs an image with *the* target. This one pairs
it with four to nine people's boundary maps, which they drew differently: over
the real 500, the densest annotator marks a median of 1.92x as many boundary
pixels as the sparsest on the same image. Two consequences are what these tests
pin.

**The annotator set is the ground truth and the consensus is a convenience.**
`target()` returns the mean because a training loop needs one map per image;
scoring against that mean would measure agreement with an average person nobody
is, which is not the published benchmark.

**Nothing resizes.** A BSDS number exists to be comparable with the literature,
which scores at native resolution, so the 224 centre-crop every other dense
dataset applies would forfeit the only reason to add this data. The geometry
tests below are the guard, and the transpose case is the one that would be
silent: PIL reports `(width, height)` and a target is `(height, width)`.

The fixtures are synthetic `.mat` files in the real layout, so the fast suite
needs none of the 60 MB `scripts/fetch_bsds500.py` downloads. One test reads the
real tree when it happens to be present.
"""

from pathlib import Path

import numpy as np
import pytest
import scipy.io
import torch
from PIL import Image

from visbench.data import BSDS500Dataset

REAL_ROOT = Path(__file__).resolve().parents[2] / "data" / "bsds500"


def write_truth(root: Path, split: str, stem: str, width: int, height: int, annotators: int):
    """One `.mat` in the real layout: a 1xA cell of structs with Boundaries."""
    truths = root / "groundTruth" / split
    truths.mkdir(parents=True, exist_ok=True)
    cell = np.empty((1, annotators), dtype=object)
    for index in range(annotators):
        boundaries = np.zeros((height, width), dtype=np.uint8)
        # One extra marked row per annotator, so the maps differ and the
        # consensus is a genuine fraction rather than 0/1 everywhere.
        boundaries[: index + 1, :] = 1
        cell[0, index] = {
            "Segmentation": np.ones((height, width), dtype=np.uint16),
            "Boundaries": boundaries,
        }
    scipy.io.savemat(truths / f"{stem}.mat", {"groundTruth": cell})


def write_split(root: Path, split: str, spec: dict[str, tuple[int, int, int]]) -> None:
    """``spec`` maps stem -> (width, height, annotators). Writes images and truths."""
    images = root / "images" / split
    images.mkdir(parents=True, exist_ok=True)
    for stem, (width, height, annotators) in spec.items():
        Image.new("RGB", (width, height), (10, 20, 30)).save(images / f"{stem}.jpg")
        write_truth(root, split, stem, width, height, annotators)


@pytest.fixture
def tree(tmp_path):
    """Two orientations and a varying annotator count, like the real thing."""
    write_split(
        tmp_path,
        "val",
        {
            "a": (8, 6, 3),  # landscape
            "b": (8, 6, 5),  # landscape, more annotators
            "c": (6, 8, 4),  # portrait
        },
    )
    return tmp_path


# -- what it hands back -------------------------------------------------------


def test_annotations_keep_every_annotator(tree):
    """4 to 9 people per image in the real data; nothing may collapse them."""
    dataset = BSDS500Dataset(tree, split="val")
    counts = {dataset.stems[i]: dataset.annotations(i).shape[0] for i in range(len(dataset))}
    assert counts == {"a": 3, "b": 5, "c": 4}


def test_the_annotator_count_may_differ_between_images(tree):
    """Ragged across images, which is why no fixed `A` is promised anywhere."""
    dataset = BSDS500Dataset(tree, split="val")
    shapes = {dataset.annotations(i).shape[0] for i in range(len(dataset))}
    assert len(shapes) > 1


def test_the_consensus_is_the_fraction_of_annotators_who_marked_each_pixel(tree):
    """Hand-computed: annotator k marks rows 0..k, so row r is marked by A-r of them."""
    dataset = BSDS500Dataset(tree, split="val")
    index = dataset.stems.index("b")  # 5 annotators, 6 rows
    consensus = dataset.target(index)

    assert consensus.dtype == torch.float32
    assert float(consensus[0, 0]) == pytest.approx(1.0)  # every annotator marked row 0
    assert float(consensus[1, 0]) == pytest.approx(4 / 5)
    assert float(consensus[4, 0]) == pytest.approx(1 / 5)
    assert float(consensus[5, 0]) == pytest.approx(0.0)  # nobody marked the last row


def test_the_consensus_is_the_mean_of_the_annotations(tree):
    """Stated once, so a future 'smarter' consensus cannot arrive unnoticed."""
    dataset = BSDS500Dataset(tree, split="val")
    for index in range(len(dataset)):
        expected = dataset.annotations(index).to(torch.float32).mean(dim=0)
        assert torch.equal(dataset.target(index), expected)


def test_getitem_pairs_the_image_with_its_own_consensus(tree):
    dataset = BSDS500Dataset(tree, split="val")
    image, target = dataset[0]
    assert isinstance(image, Image.Image)
    assert torch.equal(target, dataset.target(0))


# -- geometry -----------------------------------------------------------------


def test_nothing_is_resized_or_cropped(tree):
    """The whole reason this dataset exists: published numbers are at native size."""
    dataset = BSDS500Dataset(tree, split="val")
    for index in range(len(dataset)):
        image, target = dataset[index]
        width, height = image.size
        assert tuple(target.shape) == (height, width)


def test_the_target_is_not_transposed(tree):
    """PIL is `(width, height)` and a tensor is `(height, width)`.

    A transpose here is invisible on the real data's landscape majority only in
    the sense that it raises later rather than scoring wrongly — but on a square
    fixture it would be undetectable, so the fixture is deliberately non-square
    in both orientations.
    """
    dataset = BSDS500Dataset(tree, split="val")
    portrait = dataset.stems.index("c")  # 6 wide, 8 tall
    image, target = dataset[portrait]
    assert image.size == (6, 8)
    assert tuple(target.shape) == (8, 6)
    assert tuple(dataset.annotations(portrait).shape) == (4, 8, 6)


def test_group_by_orientation_separates_the_two_shapes(tree):
    """A batch mixing them cannot be collated; rotating them would change the data."""
    groups = BSDS500Dataset(tree, split="val").group_by_orientation()
    assert {size: sorted(indices) for size, indices in groups.items()} == {
        (8, 6): [0, 1],
        (6, 8): [2],
    }


# -- pairing and identity -----------------------------------------------------


def test_an_image_without_an_annotation_is_refused(tmp_path):
    """A silently short split is the failure every loader here guards against."""
    write_split(tmp_path, "val", {"a": (8, 6, 3)})
    Image.new("RGB", (8, 6)).save(tmp_path / "images" / "val" / "orphan.jpg")
    with pytest.raises(ValueError, match="not both"):
        BSDS500Dataset(tmp_path, split="val")


def test_subset_keeps_images_and_annotations_in_lockstep(tree):
    """Three parallel lists; slicing one alone pairs an image with another's truth."""
    dataset = BSDS500Dataset(tree, split="val")
    taken = dataset.subset([2, 0])

    assert taken.stems == ["c", "a"]
    assert [p.stem for p in taken.image_paths] == ["c", "a"]
    assert [p.stem for p in taken.truth_paths] == ["c", "a"]
    assert taken.annotations(0).shape[0] == 4  # c's annotator count, not a's


def test_the_fingerprint_moves_when_the_annotations_change(tree):
    """Same images, different people's answers, is a different measurement.

    Only the `.mat` is rewritten — `write_truth`, not `write_split` — or this
    would pass on the image having changed and prove nothing about annotations.
    """
    dataset = BSDS500Dataset(tree, split="val")
    before = dataset.fingerprint()

    write_truth(tree, "val", "a", width=8, height=6, annotators=7)
    after = BSDS500Dataset(tree, split="val").fingerprint()
    assert before != after


def test_cache_identity_ignores_the_annotation_file(tree):
    """Features come from pixels; re-annotating must not invalidate the cache.

    The same single-file rewrite as above, which is what makes the two tests a
    pair: one asserts the fingerprint moves, this one that the cache key does
    not, on *identical* filesystem changes.
    """
    dataset = BSDS500Dataset(tree, split="val")
    before = [dataset.cache_identity(i) for i in range(len(dataset))]

    write_truth(tree, "val", "a", width=8, height=6, annotators=7)
    reloaded = BSDS500Dataset(tree, split="val")
    assert [reloaded.cache_identity(i) for i in range(len(reloaded))] == before


def test_stems_select_and_order(tree):
    dataset = BSDS500Dataset(tree, split="val", stems=["c", "a"])
    assert dataset.stems == ["c", "a"]


def test_an_absent_stem_raises(tree):
    with pytest.raises(ValueError, match="absent from val"):
        BSDS500Dataset(tree, split="val", stems=["a", "nope"])


@pytest.mark.parametrize("split", ["", "trainval", "Train"])
def test_an_unofficial_split_is_refused(tree, split):
    """The official partition is what every published number uses."""
    with pytest.raises(ValueError, match="split must be one of"):
        BSDS500Dataset(tree, split=split)


def test_a_missing_tree_names_the_fetch_script(tmp_path):
    (tmp_path / "images").mkdir()
    with pytest.raises(NotADirectoryError, match="fetch_bsds500"):
        BSDS500Dataset(tmp_path, split="val")


def test_describe_records_the_two_things_that_make_it_unusual(tree):
    described = BSDS500Dataset(tree, split="val").describe()
    assert described["dataset"] == "bsds500"
    assert described["annotators"] == "per_image"
    assert described["geometry"] == "native"


# -- the real data, when it is there ------------------------------------------


@pytest.mark.skipif(
    not (REAL_ROOT / "groundTruth" / "val").is_dir(),
    reason="run scripts/fetch_bsds500.py to populate data/bsds500",
)
def test_the_real_split_is_the_official_one():
    """Sizes and geometry as published, so a partial download cannot pass quietly."""
    dataset = BSDS500Dataset(REAL_ROOT, split="val")
    assert len(dataset) == 100

    groups = dataset.group_by_orientation()
    assert set(groups) == {(481, 321), (321, 481)}

    annotators = {dataset.annotations(i).shape[0] for i in range(0, len(dataset), 10)}
    assert annotators <= {4, 5, 6, 7, 8, 9}
    assert min(annotators) >= 4

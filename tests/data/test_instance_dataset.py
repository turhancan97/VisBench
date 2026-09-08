"""Per-instance mask datasets — step 14a-1.

Three failure modes shape this file, and none of them raises on its own.

**The palette trap**, which this codebase has now paid for twice. A
``SegmentationObject`` PNG is mode ``P`` and its raw bytes *are* the instance
indices; any mode conversion resolves the palette and returns unrelated greys
that load, train and score against supervision meaning nothing. So there is a
test that reads a *real* palette file — written with a palette, not just a mode
``P`` array — and asserts the indices come back as written.

**The class lookup**, which is exact on VOC and must stay a lookup rather than
becoming a vote. Over all 6,934 instances in train and val, every instance
covers exactly one semantic class with majority share 1.000000. A majority vote
would return a confident class for genuinely inconsistent annotation, so a
mixed instance must *raise*, and that is asserted in both directions.

**The geometry**, tested on a non-square frame for the reason
``test_detection_dataset`` gives: an axis swap, a missed rescale and a missed
crop shift each produce a visibly different answer, and only arithmetic on a
known geometry shows it. The instance case adds one of its own — masks are
resampled with ``interpolate``, which takes ``(height, width)``, while the image
is resized by PIL, which takes ``(width, height)``. Writing one from the other
transposes silently on any frame that is not square.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from visbench.data import (
    ANNOTATION_KEYS,
    VOC_CLASSES,
    VOCInstanceDataset,
    load_instance_map,
)

# A palette that is deliberately NOT the identity, so a mode conversion is
# visible: index 1 resolves to grey 40, index 2 to grey 90, and so on. Reading
# with convert("L") would return those greys as if they were instance ids.
_PALETTE: list[int] = []
for value in range(256):
    grey = {0: 0, 1: 40, 2: 90, 3: 150, 255: 220}.get(value, value)
    _PALETTE += [grey, grey, grey]


def write_indexed(path, array):
    """Write ``array`` as a palette PNG whose raw bytes are the given indices.

    ``frombytes`` rather than ``fromarray(..., mode="P")``: the latter is
    deprecated in Pillow 13, and this states more plainly that the bytes going
    in are the indices coming out.
    """
    array = np.ascontiguousarray(array.astype(np.uint8))
    height, width = array.shape
    image = Image.frombytes("P", (width, height), array.tobytes())
    image.putpalette(_PALETTE)
    image.save(path)


def build(root, *, size=(8, 8), instances=None, semantic=None, void=None, stems=("a",)):
    """Lay out a VOC-shaped folder and return the root.

    ``size`` is ``(width, height)``. ``instances`` maps stem to a list of
    ``(instance_index, class_value, (row_slice, col_slice))``; the semantic map
    is derived from it unless ``semantic`` overrides one stem outright.
    """
    width, height = size
    for sub in ("JPEGImages", "SegmentationObject", "SegmentationClass"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    for stem in stems:
        rng = np.random.default_rng(abs(hash(stem)) % 2**32)
        Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8)).save(
            root / "JPEGImages" / f"{stem}.jpg"
        )
        inst = np.zeros((height, width), dtype=np.uint8)
        sem = np.zeros((height, width), dtype=np.uint8)
        for index, class_value, (rows, cols) in (instances or {}).get(stem, []):
            inst[rows, cols] = index
            sem[rows, cols] = class_value
        if void is not None:
            rows, cols = void
            inst[rows, cols] = 255
            sem[rows, cols] = 255
        write_indexed(root / "SegmentationObject" / f"{stem}.png", inst)
        write_indexed(
            root / "SegmentationClass" / f"{stem}.png",
            sem if semantic is None else semantic,
        )
    return root


def one_instance(tmp_path, **kwargs):
    """A single 4x4 instance of class 1 in an 8x8 frame."""
    return build(
        tmp_path,
        instances={"a": [(1, 1, (slice(2, 6), slice(1, 5)))]},
        **kwargs,
    )


class TestLoadInstanceMap:
    def test_palette_indices_survive_the_read(self, tmp_path):
        """The trap: convert("L") would return the palette's greys instead.

        Index 2 resolves to grey 90 under ``_PALETTE``, so a converted read
        gives 90 where the annotation says instance 2.
        """
        array = np.zeros((4, 4), dtype=np.uint8)
        array[0, 0] = 1
        array[1, 1] = 2
        array[2, 2] = 3
        path = tmp_path / "inst.png"
        write_indexed(path, array)

        indices, void = load_instance_map(path)
        assert indices.dtype == torch.int64
        assert [int(indices[i, i]) for i in range(3)] == [1, 2, 3]
        assert not void.any()

        # Prove the trap is real rather than hypothetical.
        with Image.open(path) as handle:
            converted = np.array(handle.convert("L"))
        assert converted[1, 1] == 90

    def test_void_is_split_out_and_is_not_an_instance(self, tmp_path):
        array = np.zeros((4, 4), dtype=np.uint8)
        array[0, 0] = 1
        array[3, 3] = 255
        path = tmp_path / "inst.png"
        write_indexed(path, array)

        indices, void = load_instance_map(path)
        assert void[3, 3] and void.sum() == 1
        # Void must not read back as instance 255, nor be silently background
        # in the index map only -- it is carried by the mask instead.
        assert int(indices[3, 3]) == 0
        assert int(indices.max()) == 1

    def test_an_rgb_file_is_refused(self, tmp_path):
        """Inherited from load_label_map: a palette cannot be guessed."""
        path = tmp_path / "inst.png"
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(path)
        with pytest.raises(ValueError, match="RGB"):
            load_instance_map(path)


class TestTargetStructure:
    def test_rows_are_paired_and_instances_are_disjoint(self, tmp_path):
        root = build(
            tmp_path,
            instances={
                "a": [
                    (1, 1, (slice(0, 4), slice(0, 4))),
                    (2, 1, (slice(4, 8), slice(4, 8))),
                    (3, 7, (slice(0, 2), slice(6, 8))),
                ]
            },
        )
        target = VOCInstanceDataset(root, image_size=8).target(0)
        assert target["masks"].shape == (3, 8, 8)
        assert target["labels"].shape == (3,)
        assert target["boxes"].shape == (3, 4)
        assert target["num_original"] == 3
        # One index map cannot assign a pixel to two instances.
        assert int(target["masks"].long().sum(0).max()) == 1

    def test_labels_are_zero_based_class_ids(self, tmp_path):
        """Palette value 1 is aeroplane, which is class id 0."""
        root = build(
            tmp_path,
            instances={
                "a": [
                    (1, 1, (slice(0, 2), slice(0, 2))),
                    (2, 20, (slice(4, 6), slice(4, 6))),
                ]
            },
        )
        target = VOCInstanceDataset(root, image_size=8).target(0)
        assert target["labels"].tolist() == [0, 19]
        assert VOC_CLASSES[0] == "aeroplane"
        assert VOC_CLASSES[19] == "tvmonitor"

    def test_an_image_with_no_instances_still_carries_every_key(self, tmp_path):
        root = build(tmp_path, instances={"a": []})
        target = VOCInstanceDataset(root, image_size=8).target(0)
        # The declared contract, on the case most likely to omit a field.
        assert set(target) == set(ANNOTATION_KEYS)
        assert target["masks"].shape == (0, 8, 8)
        assert target["labels"].shape == (0,)
        assert target["boxes"].shape == (0, 4)
        assert target["num_original"] == 0
        assert target["ignore"].shape == (8, 8)

    def test_getitem_returns_image_and_target_at_one_geometry(self, tmp_path):
        root = one_instance(tmp_path)
        image, target = VOCInstanceDataset(root, image_size=8)[0]
        assert image.size == (8, 8)
        assert target["masks"].shape[-2:] == (8, 8)


class TestBoxesComeFromTheirMask:
    def test_box_is_the_tight_half_open_extent(self, tmp_path):
        root = build(
            tmp_path,
            instances={"a": [(1, 1, (slice(2, 6), slice(1, 5)))]},
        )
        target = VOCInstanceDataset(root, image_size=8).target(0)
        # rows 2..5 and cols 1..4 inclusive -> xyxy (1, 2, 5, 6), exclusive far edge
        assert target["boxes"][0].tolist() == [1.0, 2.0, 5.0, 6.0]

    def test_a_single_pixel_instance_has_positive_area(self, tmp_path):
        """The half-open convention exists so this box is not degenerate."""
        root = build(
            tmp_path,
            instances={"a": [(1, 1, (slice(3, 4), slice(5, 6)))]},
        )
        box = VOCInstanceDataset(root, image_size=8).target(0)["boxes"][0]
        assert box.tolist() == [5.0, 3.0, 6.0, 4.0]
        assert box[2] > box[0] and box[3] > box[1]

    def test_every_box_matches_its_own_mask(self, tmp_path):
        """Derived, not transformed, so the two cannot drift apart."""
        root = build(
            tmp_path,
            size=(11, 7),
            instances={
                "a": [
                    (1, 3, (slice(1, 5), slice(2, 9))),
                    (2, 3, (slice(5, 7), slice(0, 3))),
                ]
            },
        )
        target = VOCInstanceDataset(root, image_size=6).target(0)
        for mask, box in zip(target["masks"], target["boxes"], strict=True):
            ys, xs = torch.nonzero(mask, as_tuple=True)
            expected = [
                float(xs.min()),
                float(ys.min()),
                float(xs.max()) + 1,
                float(ys.max()) + 1,
            ]
            assert box.tolist() == expected


class TestVoidIsIgnoredNotBackground:
    def test_void_pixels_belong_to_no_instance_and_are_flagged(self, tmp_path):
        root = build(
            tmp_path,
            instances={"a": [(1, 1, (slice(2, 6), slice(2, 6)))]},
            void=(slice(0, 1), slice(0, 8)),
        )
        target = VOCInstanceDataset(root, image_size=8).target(0)
        assert target["ignore"][0].all()
        assert not target["ignore"][1:].any()
        # No instance claims a void pixel.
        assert not target["masks"][:, 0, :].any()

    def test_ignore_is_not_folded_into_a_mask_or_dropped(self, tmp_path):
        """Void is 6% of VOC's pixels, far too much to treat as either class."""
        root = build(
            tmp_path,
            instances={"a": [(1, 1, (slice(4, 8), slice(0, 8)))]},
            void=(slice(0, 2), slice(0, 8)),
        )
        target = VOCInstanceDataset(root, image_size=8).target(0)
        assert int(target["ignore"].sum()) == 16
        assert int(target["masks"].sum()) == 32


class TestTheClassLookupIsExactNotAVote:
    def test_an_instance_spanning_two_classes_raises(self, tmp_path):
        """A vote would return a confident class for broken annotation."""
        root = build(tmp_path, instances={"a": [(1, 1, (slice(0, 4), slice(0, 4)))]})
        # Repaint half the instance's semantic pixels as a different class.
        sem = np.array(Image.open(root / "SegmentationClass" / "a.png"))
        sem[0:2, 0:4] = 7
        write_indexed(root / "SegmentationClass" / "a.png", sem)

        dataset = VOCInstanceDataset(root, image_size=8)
        with pytest.raises(ValueError, match="semantic classes"):
            dataset.target(0)

    def test_an_instance_with_no_class_pixels_raises(self, tmp_path):
        root = build(tmp_path, instances={"a": [(1, 1, (slice(0, 4), slice(0, 4)))]})
        write_indexed(root / "SegmentationClass" / "a.png", np.zeros((8, 8), dtype=np.uint8))
        dataset = VOCInstanceDataset(root, image_size=8)
        with pytest.raises(ValueError, match="expected exactly one"):
            dataset.target(0)

    def test_maps_of_different_sizes_raise(self, tmp_path):
        root = build(tmp_path, instances={"a": [(1, 1, (slice(0, 4), slice(0, 4)))]})
        write_indexed(root / "SegmentationClass" / "a.png", np.zeros((6, 6), dtype=np.uint8))
        dataset = VOCInstanceDataset(root, image_size=6)
        with pytest.raises(ValueError, match="same image"):
            dataset.target(0)

    def test_an_instance_cropped_entirely_away_does_not_raise(self, tmp_path):
        """This is what reading the class *before* the crop buys.

        The instance sits outside the centre crop, so after cropping it has no
        pixels at all -- and therefore no semantic pixels either. Were the class
        looked up on the cropped maps, this would hit the "expected exactly one"
        path and raise on an image whose annotation is perfectly valid. Read at
        full resolution, the class resolves and the instance is simply dropped.

        A 24x8 frame at image_size=8 keeps columns 8..15; the instance is at
        columns 0..1.
        """
        root = build(
            tmp_path,
            size=(24, 8),
            instances={"a": [(1, 12, (slice(0, 8), slice(0, 2)))]},
        )
        target = VOCInstanceDataset(root, image_size=8, min_instance_pixels=1).target(0)
        assert target["masks"].shape[0] == 0
        assert target["num_original"] == 1


class TestTheCropDropsInstances:
    def test_an_instance_outside_the_crop_is_dropped_but_counted(self, tmp_path):
        """A centre crop genuinely removes objects; num_original says so."""
        root = build(
            tmp_path,
            size=(24, 8),
            instances={
                "a": [
                    (1, 5, (slice(0, 8), slice(0, 2))),
                    (2, 5, (slice(2, 6), slice(10, 14))),
                ]
            },
        )
        target = VOCInstanceDataset(root, image_size=8).target(0)
        assert target["num_original"] == 2
        assert target["masks"].shape[0] < 2
        assert target["masks"].shape[0] == target["labels"].shape[0]

    def test_min_instance_pixels_filters_and_keeps_rows_aligned(self, tmp_path):
        root = build(
            tmp_path,
            instances={
                "a": [
                    (1, 1, (slice(0, 4), slice(0, 4))),
                    (2, 9, (slice(7, 8), slice(7, 8))),
                ]
            },
        )
        kept = VOCInstanceDataset(root, image_size=8, min_instance_pixels=1).target(0)
        assert kept["masks"].shape[0] == 2

        filtered = VOCInstanceDataset(root, image_size=8, min_instance_pixels=2).target(0)
        assert filtered["masks"].shape[0] == 1
        assert filtered["labels"].tolist() == [0]
        assert filtered["num_original"] == 2

    def test_min_instance_pixels_below_one_is_refused(self, tmp_path):
        root = one_instance(tmp_path)
        with pytest.raises(ValueError, match="min_instance_pixels"):
            VOCInstanceDataset(root, image_size=8, min_instance_pixels=0)


class TestGeometryOnANonSquareFrame:
    def test_masks_are_square_and_at_the_requested_size(self, tmp_path):
        root = build(
            tmp_path,
            size=(20, 10),
            instances={"a": [(1, 1, (slice(2, 8), slice(4, 16)))]},
        )
        target = VOCInstanceDataset(root, image_size=6).target(0)
        assert target["masks"].shape == (1, 6, 6)
        assert target["ignore"].shape == (6, 6)

    def test_the_mask_lands_where_the_arithmetic_says(self, tmp_path):
        """A transpose, a missed rescale or a missed shift each move this.

        A 20x10 frame at image_size=10 resizes to 20x10 (the short side is
        already 10) and centre-crops columns 5..14. An instance spanning columns
        6..9 of the original therefore lands at columns 1..4 of the crop, at the
        same rows.
        """
        root = build(
            tmp_path,
            size=(20, 10),
            instances={"a": [(1, 1, (slice(3, 7), slice(6, 10)))]},
        )
        target = VOCInstanceDataset(root, image_size=10).target(0)
        assert target["boxes"][0].tolist() == [1.0, 3.0, 5.0, 7.0]

    def test_a_portrait_frame_is_not_transposed(self, tmp_path):
        """The same case with the axes swapped, which a transposed
        ``interpolate`` size would pass on the landscape test and fail here."""
        root = build(
            tmp_path,
            size=(10, 20),
            instances={"a": [(1, 1, (slice(6, 10), slice(3, 7)))]},
        )
        target = VOCInstanceDataset(root, image_size=10).target(0)
        assert target["boxes"][0].tolist() == [3.0, 1.0, 7.0, 5.0]


class TestSplitsAndIdentity:
    def test_stems_set_the_order_and_the_length(self, tmp_path):
        root = build(
            tmp_path,
            instances={s: [(1, 1, (slice(0, 4), slice(0, 4)))] for s in ("a", "b", "c")},
            stems=("a", "b", "c"),
        )
        dataset = VOCInstanceDataset(root, image_size=8, stems=["c", "a"])
        assert dataset.stems == ["c", "a"]
        assert len(dataset) == 2

    def test_a_missing_stem_is_named(self, tmp_path):
        root = one_instance(tmp_path)
        with pytest.raises(ValueError, match="missing"):
            VOCInstanceDataset(root, image_size=8, stems=["a", "nope"])

    def test_a_repeated_stem_is_refused(self, tmp_path):
        root = one_instance(tmp_path)
        with pytest.raises(ValueError, match="more than once"):
            VOCInstanceDataset(root, image_size=8, stems=["a", "a"])

    def test_subset_reindexes_all_four_parallel_lists(self, tmp_path):
        """Slicing one alone would pair an image with another image's instances."""
        root = build(
            tmp_path,
            instances={
                "a": [(1, 1, (slice(0, 4), slice(0, 4)))],
                "b": [(1, 5, (slice(4, 8), slice(4, 8)))],
                "c": [(1, 9, (slice(0, 2), slice(0, 2)))],
            },
            stems=("a", "b", "c"),
        )
        dataset = VOCInstanceDataset(root, image_size=8, stems=["a", "b", "c"])
        assert set(VOCInstanceDataset._parallel_attrs) == {
            "stems",
            "image_paths",
            "instance_paths",
            "class_paths",
        }

        subset = dataset.subset([2, 0])
        assert subset.stems == ["c", "a"]
        for attribute in VOCInstanceDataset._parallel_attrs:
            values = getattr(subset, attribute)
            assert len(values) == 2
        # The pairing survives: item 0 is c's image with c's instance class.
        assert subset.image_paths[0].stem == "c"
        assert subset.instance_paths[0].stem == "c"
        assert subset.class_paths[0].stem == "c"
        assert subset.target(0)["labels"].tolist() == [8]

    def test_cache_identity_ignores_the_annotations(self, tmp_path):
        """Features depend on the image, so re-exporting a mask must not
        invalidate an extraction that is still valid."""
        root = one_instance(tmp_path)
        dataset = VOCInstanceDataset(root, image_size=8)
        before = dataset.cache_identity(0)

        sem = np.array(Image.open(root / "SegmentationClass" / "a.png"))
        sem[sem == 1] = 4
        write_indexed(root / "SegmentationClass" / "a.png", sem)

        assert VOCInstanceDataset(root, image_size=8).cache_identity(0) == before

    def test_fingerprint_follows_the_settings_that_shape_the_targets(self, tmp_path):
        root = one_instance(tmp_path)
        base = VOCInstanceDataset(root, image_size=8).fingerprint()
        assert VOCInstanceDataset(root, image_size=6).fingerprint() != base
        assert VOCInstanceDataset(root, image_size=8, min_instance_pixels=4).fingerprint() != base

    def test_fingerprint_follows_the_annotations(self, tmp_path):
        """Unlike cache_identity: the question here is which data scored."""
        root = one_instance(tmp_path)
        before = VOCInstanceDataset(root, image_size=8).fingerprint()
        inst = np.array(Image.open(root / "SegmentationObject" / "a.png"))
        inst[0, :] = 3
        write_indexed(root / "SegmentationObject" / "a.png", inst)
        assert VOCInstanceDataset(root, image_size=8).fingerprint() != before

    def test_describe_carries_the_target_shaping_settings(self, tmp_path):
        root = one_instance(tmp_path)
        info = VOCInstanceDataset(root, image_size=8, split="val").describe()
        assert info["split"] == "val"
        assert info["image_size"] == 8
        assert info["min_instance_pixels"] == 1
        assert info["num_classes"] == 20

    def test_a_missing_directory_names_which_one(self, tmp_path):
        root = one_instance(tmp_path)
        (root / "SegmentationClass").rename(root / "elsewhere")
        with pytest.raises(NotADirectoryError, match="class_dir"):
            VOCInstanceDataset(root, image_size=8)

    def test_duplicate_file_stems_are_refused(self, tmp_path):
        """Pairing would otherwise depend on directory iteration order."""
        root = one_instance(tmp_path)
        write_indexed(root / "SegmentationObject" / "a.tiff", np.zeros((8, 8), dtype=np.uint8))
        with pytest.raises(ValueError, match="two files with stem"):
            VOCInstanceDataset(root, image_size=8)

    def test_labels_returns_one_annotation_per_item_in_order(self, tmp_path):
        root = build(
            tmp_path,
            instances={
                "a": [(1, 1, (slice(0, 4), slice(0, 4)))],
                "b": [(1, 6, (slice(0, 4), slice(0, 4)))],
            },
            stems=("a", "b"),
        )
        dataset = VOCInstanceDataset(root, image_size=8, stems=["a", "b"])
        annotations = dataset.labels()
        assert len(annotations) == 2
        assert [a["labels"].tolist() for a in annotations] == [[0], [5]]

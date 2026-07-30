"""Box-target datasets — step 6c.

Almost every test here is about **geometry**, for a sharper version of the
reason the dense tests are: a box does not resample with its image, it has to be
rescaled and shifted by hand, and when that is skipped nothing raises. The boxes
simply describe the original frame while the tensor is 224x224 and the probe
trains against supervision that is wrong everywhere.

So the load-bearing test is :func:`test_boxes_land_in_post_transform_pixels`,
on a non-square image where a missed rescale, a missed shift or a swapped axis
each produce a visibly different answer. The fake-backbone tests elsewhere in
this suite cannot show that; only arithmetic on a known geometry can.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from visbench.data import VOC_CLASSES, DetectionFolderDataset, load_voc_boxes


def write_xml(path, size, objects):
    """Write a VOC-shaped annotation.

    ``size`` is ``(width, height)``; ``objects`` is a list of
    ``(name, xmin, ymin, xmax, ymax, difficult)`` with coordinates **1-indexed**,
    exactly as VOC stores them, so the tests exercise the conversion rather than
    assuming it away.
    """
    width, height = size
    parts = [
        "<annotation>",
        f"  <size><width>{width}</width><height>{height}</height><depth>3</depth></size>",
    ]
    for name, x_min, y_min, x_max, y_max, difficult in objects:
        parts += [
            "  <object>",
            f"    <name>{name}</name>",
            f"    <difficult>{int(difficult)}</difficult>",
            f"    <bndbox><xmin>{x_min}</xmin><ymin>{y_min}</ymin>"
            f"<xmax>{x_max}</xmax><ymax>{y_max}</ymax></bndbox>",
            "  </object>",
        ]
    parts.append("</annotation>")
    path.write_text("\n".join(parts))


def build(root, items, image_size=(400, 200)):
    """Write an image/annotation folder pair. ``items`` maps stem -> objects."""
    (root / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (root / "Annotations").mkdir(parents=True, exist_ok=True)
    width, height = image_size
    for stem, objects in items.items():
        array = np.random.default_rng(abs(hash(stem)) % 2**31).integers(
            0, 255, (height, width, 3), dtype=np.uint8
        )
        Image.fromarray(array).save(root / "JPEGImages" / f"{stem}.jpg")
        write_xml(root / "Annotations" / f"{stem}.xml", image_size, objects)
    return root


# -- the convention ---------------------------------------------------------


def test_boxes_land_in_post_transform_pixels(tmp_path):
    """The load-bearing test: boxes match the tensor they are returned beside.

    A 400x200 original at ``image_size=100`` resizes the short side to 100,
    giving 200x100, then centre-crops to 100x100 at ``left=50, top=0``. So a
    box maps ``x' = x * 0.5 - 50``, ``y' = y * 0.5``.

    The 1-indexed input ``(201, 41, 300, 140)`` becomes 0-indexed
    ``(200, 40, 299, 139)`` and must come back as ``(50, 20, 99.5, 69.5)``.
    Each failure mode gives something else: no rescale clips to
    ``(100, 40, 100, 100)``, no shift gives ``(100, 20, ...)``, and swapping the
    axes gives ``(20, 50, ...)``. The expected value is asymmetric in x and y
    precisely so the swap cannot pass.
    """
    build(tmp_path, {"a": [("cat", 201, 41, 300, 140, 0)]}, image_size=(400, 200))
    dataset = DetectionFolderDataset(tmp_path, image_size=100)

    image, annotation = dataset[0]

    assert image.size == (100, 100)
    assert annotation["boxes"].shape == (1, 4)
    assert torch.allclose(annotation["boxes"][0], torch.tensor([50.0, 20.0, 99.5, 69.5]), atol=1e-5)


def test_voc_boxes_are_zero_indexed_on_read(tmp_path):
    """VOC's 1-indexed corners lose exactly one on read, and nothing else."""
    write_xml(tmp_path / "a.xml", (500, 375), [("dog", 1, 1, 500, 375, 0)])

    annotation = load_voc_boxes(tmp_path / "a.xml")

    # 1 -> 0 and 500 -> 499: the whole frame, in 0-indexed corners.
    assert torch.equal(annotation["boxes"][0], torch.tensor([0.0, 0.0, 499.0, 374.0]))
    assert annotation["size"] == (500, 375)


def test_boxes_are_xyxy_not_xywh(tmp_path):
    """Assert the corner convention, because a swapped pair loads and scores.

    ``(11, 21, 60, 90)`` 1-indexed is a box from (10, 20) to (59, 89). Read as
    ``xywh`` it would instead be 60 wide and 90 tall starting at (10, 20), so
    the third coordinate distinguishes the two unambiguously.
    """
    write_xml(tmp_path / "a.xml", (200, 200), [("car", 11, 21, 60, 90, 0)])

    box = load_voc_boxes(tmp_path / "a.xml")["boxes"][0]

    assert box.tolist() == [10.0, 20.0, 59.0, 89.0]
    assert (box[2] - box[0]).item() == 49.0  # a width, derived — not stored


def test_labels_index_into_voc_classes(tmp_path):
    write_xml(tmp_path / "a.xml", (100, 100), [("aeroplane", 2, 2, 50, 50, 0)])
    first = load_voc_boxes(tmp_path / "a.xml")
    write_xml(tmp_path / "b.xml", (100, 100), [("tvmonitor", 2, 2, 50, 50, 0)])
    last = load_voc_boxes(tmp_path / "b.xml")

    assert first["labels"].tolist() == [0]
    assert last["labels"].tolist() == [len(VOC_CLASSES) - 1]
    assert first["labels"].dtype == torch.int64


# -- difficult objects ------------------------------------------------------


def test_difficult_objects_are_excluded_by_default(tmp_path):
    """VOC's protocol excludes them, so the default must too.

    Counting the 4,462 flagged objects as false negatives depresses mAP against
    every published number while raising nothing.
    """
    build(
        tmp_path,
        {"a": [("cat", 11, 11, 90, 90, 0), ("dog", 21, 21, 80, 80, 1)]},
        image_size=(200, 200),
    )

    default = DetectionFolderDataset(tmp_path, image_size=100).target(0)
    kept = DetectionFolderDataset(tmp_path, image_size=100, include_difficult=True).target(0)

    assert default["boxes"].shape[0] == 1
    assert default["labels"].tolist() == [VOC_CLASSES.index("cat")]
    assert kept["boxes"].shape[0] == 2
    # num_original reports what the file held, so "no objects" and "all objects
    # filtered" stay distinguishable.
    assert default["num_original"] == 2


def test_loader_returns_difficult_rather_than_filtering(tmp_path):
    """Filtering in the loader would make the exclusion invisible to the record."""
    write_xml(tmp_path / "a.xml", (100, 100), [("cat", 2, 2, 50, 50, 1)])

    annotation = load_voc_boxes(tmp_path / "a.xml")

    assert annotation["boxes"].shape[0] == 1
    assert annotation["difficult"].tolist() == [True]


def test_include_difficult_is_recorded(tmp_path):
    build(tmp_path, {"a": [("cat", 11, 11, 90, 90, 0)]}, image_size=(200, 200))

    info = DetectionFolderDataset(tmp_path, image_size=100, include_difficult=True).describe()

    assert info["include_difficult"] is True
    assert info["num_classes"] == 20


# -- the crop removes objects -----------------------------------------------


def test_a_box_outside_the_crop_is_dropped(tmp_path):
    """A centre crop genuinely removes objects; scoring against one is measuring nothing."""
    # 400x200 at image_size=100 keeps original x in [100, 300). A box at x 1..40
    # is entirely left of the crop.
    build(
        tmp_path,
        {"a": [("cat", 1, 41, 40, 140, 0), ("dog", 201, 41, 300, 140, 0)]},
        image_size=(400, 200),
    )

    annotation = DetectionFolderDataset(tmp_path, image_size=100).target(0)

    assert annotation["boxes"].shape[0] == 1
    assert annotation["labels"].tolist() == [VOC_CLASSES.index("dog")]
    assert annotation["num_original"] == 2


def test_a_straddling_box_is_clipped_to_the_crop(tmp_path):
    """The visible part of a partly visible object is the correct target."""
    # Original x 101..300 maps to 0..99.5; a box from x 1 to 300 clips at 0.
    build(tmp_path, {"a": [("cat", 1, 41, 300, 140, 0)]}, image_size=(400, 200))

    box = DetectionFolderDataset(tmp_path, image_size=100).target(0)["boxes"][0]

    assert box[0].item() == 0.0
    assert box[2].item() == pytest.approx(99.5)
    assert torch.all(box >= 0.0) and torch.all(box <= 100.0)


def test_an_image_with_no_surviving_boxes_is_not_an_error(tmp_path):
    """A negative image is legitimate and must not crash the loader."""
    build(tmp_path, {"a": [("cat", 1, 41, 40, 140, 0)]}, image_size=(400, 200))

    image, annotation = DetectionFolderDataset(tmp_path, image_size=100)[0]

    assert image.size == (100, 100)
    assert annotation["boxes"].shape == (0, 4)
    assert annotation["labels"].shape == (0,)
    assert annotation["difficult"].shape == (0,)
    assert annotation["num_original"] == 1


def test_rows_stay_aligned_when_boxes_are_dropped(tmp_path):
    """boxes, labels and difficult are indexed by one mask, so they cannot drift."""
    build(
        tmp_path,
        {
            "a": [
                ("cat", 1, 41, 30, 140, 0),  # dropped, off the left
                ("dog", 201, 41, 300, 140, 0),  # kept
                ("bus", 371, 41, 400, 140, 0),  # dropped, off the right
            ]
        },
        image_size=(400, 200),
    )

    annotation = DetectionFolderDataset(tmp_path, image_size=100).target(0)

    assert annotation["boxes"].shape[0] == annotation["labels"].shape[0] == 1
    assert annotation["labels"].tolist() == [VOC_CLASSES.index("dog")]


# -- image geometry matches the dense dataset -------------------------------


def test_image_geometry_matches_the_dense_convention(tmp_path):
    """The image half must be pixel-identical to DenseFolderDataset's crop.

    The box transform is derived from this geometry, so if the two ever diverge
    the boxes shift and nothing raises.
    """
    from visbench.data.dense import DenseFolderDataset

    build(tmp_path, {"a": [("cat", 11, 11, 90, 90, 0)]}, image_size=(320, 240))
    # A dense dataset over the same images, with a throwaway target.
    (tmp_path / "targets").mkdir()
    np.save(tmp_path / "targets" / "a.npy", np.ones((240, 320), dtype=np.float32))

    detection = DetectionFolderDataset(tmp_path, image_size=112)[0][0]
    dense = DenseFolderDataset(tmp_path, "JPEGImages", "targets", image_size=112)[0][0]

    assert detection.size == dense.size
    assert np.array_equal(np.array(detection), np.array(dense))


# -- splits, subsetting, identity -------------------------------------------


def test_stems_from_a_split_file(tmp_path):
    build(tmp_path, {stem: [("cat", 11, 11, 90, 90, 0)] for stem in ("a", "b", "c")})
    (tmp_path / "split.txt").write_text("c\na\n")

    dataset = DetectionFolderDataset(tmp_path, stems=tmp_path / "split.txt", image_size=100)

    assert dataset.stems == ["c", "a"]  # order preserved: targets travel by index
    assert len(dataset) == 2


def test_a_two_column_split_file_reads_the_stem(tmp_path):
    """VOC's Main/<class>_train.txt carries "<stem> <±1>" rather than bare stems."""
    build(tmp_path, {stem: [("cat", 11, 11, 90, 90, 0)] for stem in ("a", "b")})
    (tmp_path / "split.txt").write_text("a  1\nb -1\n")

    dataset = DetectionFolderDataset(tmp_path, stems=tmp_path / "split.txt", image_size=100)

    assert dataset.stems == ["a", "b"]


def test_a_missing_stem_raises(tmp_path):
    build(tmp_path, {"a": [("cat", 11, 11, 90, 90, 0)]})

    with pytest.raises(ValueError, match="missing"):
        DetectionFolderDataset(tmp_path, stems=["a", "nope"], image_size=100)


def test_a_repeated_stem_raises(tmp_path):
    build(tmp_path, {"a": [("cat", 11, 11, 90, 90, 0)]})

    with pytest.raises(ValueError, match="more than once"):
        DetectionFolderDataset(tmp_path, stems=["a", "a"], image_size=100)


def test_subset_reindexes_all_three_lists_together(tmp_path):
    """Slicing one alone would pair an image with another image's boxes."""
    build(tmp_path, {stem: [("cat", 11, 11, 90, 90, 0)] for stem in ("a", "b", "c")})
    dataset = DetectionFolderDataset(tmp_path, image_size=100)

    short = dataset.subset([2, 0])

    assert isinstance(short, DetectionFolderDataset)
    assert short.stems == ["c", "a"]
    assert [path.stem for path in short.image_paths] == ["c", "a"]
    assert [path.stem for path in short.annotation_paths] == ["c", "a"]
    assert len(dataset) == 3  # the original is untouched


def test_subset_changes_the_fingerprint(tmp_path):
    build(tmp_path, {stem: [("cat", 11, 11, 90, 90, 0)] for stem in ("a", "b", "c")})
    dataset = DetectionFolderDataset(tmp_path, image_size=100)

    assert dataset.subset(2).fingerprint() != dataset.fingerprint()


def test_fingerprint_follows_the_target_settings(tmp_path):
    """Two runs whose targets differ must not share a record or a cache entry."""
    build(tmp_path, {"a": [("cat", 11, 11, 90, 90, 1)]}, image_size=(200, 200))

    base = DetectionFolderDataset(tmp_path, image_size=100)
    prints = {
        base.fingerprint(),
        DetectionFolderDataset(tmp_path, image_size=112).fingerprint(),
        DetectionFolderDataset(tmp_path, image_size=100, include_difficult=True).fingerprint(),
        DetectionFolderDataset(tmp_path, image_size=100, min_box_size=8.0).fingerprint(),
    }

    assert len(prints) == 4


def test_cache_identity_ignores_the_annotation(tmp_path):
    """Cached features depend on the image alone, so editing an XML must not invalidate them."""
    build(tmp_path, {"a": [("cat", 11, 11, 90, 90, 0)]}, image_size=(200, 200))
    dataset = DetectionFolderDataset(tmp_path, image_size=100)
    before = dataset.cache_identity(0)

    write_xml(tmp_path / "Annotations" / "a.xml", (200, 200), [("dog", 21, 21, 80, 80, 0)])

    assert DetectionFolderDataset(tmp_path, image_size=100).cache_identity(0) == before


def test_labels_returns_one_annotation_per_item(tmp_path):
    build(tmp_path, {stem: [("cat", 11, 11, 90, 90, 0)] for stem in ("a", "b")}, (200, 200))

    labels = DetectionFolderDataset(tmp_path, image_size=100).labels()

    assert len(labels) == 2
    assert all(entry["boxes"].shape == (1, 4) for entry in labels)


# -- refusals ---------------------------------------------------------------


def test_an_unknown_class_name_raises(tmp_path):
    write_xml(tmp_path / "a.xml", (100, 100), [("unicorn", 2, 2, 50, 50, 0)])

    with pytest.raises(ValueError, match="unicorn"):
        load_voc_boxes(tmp_path / "a.xml")


def test_a_degenerate_box_raises(tmp_path):
    """A zero-area box cannot be matched or scored, so it is refused on read."""
    write_xml(tmp_path / "a.xml", (100, 100), [("cat", 30, 30, 30, 60, 0)])

    with pytest.raises(ValueError, match="degenerate"):
        load_voc_boxes(tmp_path / "a.xml")


def test_a_missing_size_element_raises(tmp_path):
    (tmp_path / "a.xml").write_text("<annotation></annotation>")

    with pytest.raises(ValueError, match="<size>"):
        load_voc_boxes(tmp_path / "a.xml")


def test_float_coordinates_are_accepted(tmp_path):
    """Some VOC redistributions write "174.0"; int() would raise on those."""
    (tmp_path / "a.xml").write_text(
        "<annotation><size><width>100</width><height>100</height></size>"
        "<object><name>cat</name><bndbox><xmin>11.0</xmin><ymin>21.0</ymin>"
        "<xmax>60.0</xmax><ymax>90.0</ymax></bndbox></object></annotation>"
    )

    box = load_voc_boxes(tmp_path / "a.xml")["boxes"][0]

    assert box.tolist() == [10.0, 20.0, 59.0, 89.0]


def test_duplicate_stems_in_a_directory_raise(tmp_path):
    """Two files sharing a stem would make the pairing depend on iteration order."""
    build(tmp_path, {"a": [("cat", 11, 11, 90, 90, 0)]}, image_size=(200, 200))
    Image.fromarray(np.zeros((200, 200, 3), np.uint8)).save(tmp_path / "JPEGImages" / "a.png")

    with pytest.raises(ValueError, match="two files with stem"):
        DetectionFolderDataset(tmp_path, image_size=100)


def test_a_missing_directory_raises(tmp_path):
    """NotADirectoryError, matching DenseFolderDataset rather than inventing a second convention."""
    with pytest.raises(NotADirectoryError, match="image_dir"):
        DetectionFolderDataset(tmp_path, image_size=100)

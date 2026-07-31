"""Taskonomy loading — the edge-map reader and the split-indexed dataset.

Two hazards carry these tests, and both are silent when got wrong.

The **16-bit read**: an ``edge_texture`` map is mode ``I;16`` with values well
past 255, so any mode conversion quantises it and the probe trains on a
coarsened target without anything raising. That is the palette-PNG failure
`load_label_map` guards against, in a different container.

The **geometry**: image and target must survive the same resize and crop. A
misaligned dense target does not raise, it merely scores badly, and this
codebase has already paid for that once at recall@1px = 0.003.
"""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from visbench.data import (
    TASKONOMY_DOMAINS,
    DenseFolderDataset,
    TaskonomyDataset,
    load_edge_map,
    load_taskonomy_split,
)

BUILDINGS = ("allensville", "beechwood")


def _write_edge_png(path: Path, array: np.ndarray) -> None:
    """Write a uint16 single-channel PNG, as Taskonomy ships them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # No mode=: Pillow infers I;16 from the uint16 dtype, and passing it
    # explicitly is deprecated from Pillow 13.
    Image.fromarray(array.astype(np.uint16)).save(path)


def _stripe(height: int, width: int, column: int, value: int) -> np.ndarray:
    """A blank field with one bright vertical stripe — a locatable feature."""
    array = np.zeros((height, width), dtype=np.uint16)
    array[:, column] = value
    return array


@pytest.fixture
def taskonomy(tmp_path: Path) -> Path:
    """A miniature Taskonomy tree: two buildings, two frames each, one domain.

    Frames are deliberately **non-square** (64 x 96). Step 6c-1 settled that a
    missed rescale is invisible on a square image — the resize factor is the
    same on both axes, so getting it wrong still lands the feature in the right
    place. Only a non-square frame separates the two.
    """
    rows = []
    for building in BUILDINGS:
        for point, view in ((0, 0), (1, 2)):
            rgb = np.zeros((64, 96, 3), dtype=np.uint8)
            rgb[:, 20] = 255
            image_path = tmp_path / "rgb" / building / f"point_{point}_view_{view}_domain_rgb.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(image_path)

            _write_edge_png(
                tmp_path
                / "edge_texture"
                / building
                / f"point_{point}_view_{view}_domain_edge_texture.png",
                _stripe(64, 96, 20, 9000),
            )
            rows.append({"building": building, "point": str(point), "view": str(view)})

    splits = tmp_path / "splits"
    splits.mkdir()
    with (splits / "tiny_train.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["building", "point", "view"])
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


# -- load_edge_map -----------------------------------------------------------


def test_sixteen_bit_values_survive_the_read(tmp_path: Path):
    """The whole reason this is not `load_mask` or a `convert("L")`.

    Values run to ~10,500 in real data. Converting to 8-bit greyscale would
    rescale them into 0-255, discarding six bits of edge magnitude, and the
    probe would train and score against the coarsened version silently.
    """
    _write_edge_png(tmp_path / "e.png", np.array([[0, 300, 9000], [65535, 1, 2]]))
    edges = load_edge_map(tmp_path / "e.png", scale=1.0)
    assert edges.tolist() == [[0.0, 300.0, 9000.0], [65535.0, 1.0, 2.0]]


def test_scale_divides_into_the_container_range(tmp_path: Path):
    """The default 65535 puts a uint16 map on [0, 1]."""
    _write_edge_png(tmp_path / "e.png", np.array([[0, 65535], [32768, 6554]]))
    edges = load_edge_map(tmp_path / "e.png")
    assert edges[0, 0].item() == 0.0
    assert edges[0, 1].item() == pytest.approx(1.0)
    assert edges[1, 0].item() == pytest.approx(0.5, abs=1e-4)


def test_zero_is_kept_as_a_real_reading(tmp_path: Path):
    """The convention that differs from depth, and the one most easily assumed away.

    For depth and normals, 0 means "no ground truth". Here it means "no edge" —
    a genuine measurement covering most of most frames. Nothing may drop it, or
    the probe is scored only where an edge already is.
    """
    _write_edge_png(tmp_path / "e.png", np.zeros((4, 4), dtype=np.uint16))
    edges = load_edge_map(tmp_path / "e.png")
    assert edges.shape == (4, 4)
    assert torch.equal(edges, torch.zeros(4, 4))


def test_an_rgb_file_is_refused(tmp_path: Path):
    """Single-channel by definition; a three-channel file means the wrong domain."""
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(tmp_path / "e.png")
    with pytest.raises(ValueError, match="single-channel"):
        load_edge_map(tmp_path / "e.png")


def test_a_non_2d_npy_is_refused(tmp_path: Path):
    np.save(tmp_path / "e.npy", np.zeros((3, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="2D"):
        load_edge_map(tmp_path / "e.npy")


# -- load_taskonomy_split ----------------------------------------------------


def test_split_rows_are_read_by_column_name(taskonomy: Path):
    """By name, not position. A reordered file would otherwise pair wrong frames."""
    rows = load_taskonomy_split(taskonomy, "train")
    assert len(rows) == 4
    assert ("allensville", "0", "0") in rows


def test_a_missing_split_file_names_the_partition(taskonomy: Path):
    """The partition is part of what a number means, so it is not guessed."""
    with pytest.raises(FileNotFoundError, match="partition"):
        load_taskonomy_split(taskonomy, "val")


# -- TaskonomyDataset --------------------------------------------------------


def test_pairs_rgb_to_the_domain_across_different_filenames(taskonomy: Path):
    """The reason DenseFolderDataset's stem pairing cannot be reused.

    The two halves of a Taskonomy pair never share a filename — the domain is
    in it — so pairing has to come from the split list instead.
    """
    dataset = TaskonomyDataset(taskonomy, split="train", image_size=32)
    assert len(dataset) == 4
    for image_path, target_path in zip(dataset.image_paths, dataset.target_paths, strict=True):
        assert image_path.name.endswith("_domain_rgb.png")
        assert target_path.name.endswith("_domain_edge_texture.png")
        assert image_path.parent.name == target_path.parent.name


def test_stems_carry_the_building(taskonomy: Path):
    """point/view numbering restarts per building, so a bare stem collides.

    Two buildings both have `point_0_view_0`. A stem list that collapsed them
    would make the fingerprint claim two different splits were the same one.
    """
    dataset = TaskonomyDataset(taskonomy, split="train", image_size=32)
    assert len(set(dataset.stems)) == len(dataset.stems)
    assert all("/" in stem for stem in dataset.stems)


def test_image_and_target_stay_aligned_through_the_crop(taskonomy: Path):
    """The failure that does not raise.

    The fixture puts one bright stripe at the same column in both halves. If the
    image took a different resize or crop from the target, the two stripes land
    in different columns — which is what a dense probe would then be trained to
    reproduce, scoring badly for a reason that looks like a weak backbone.
    """
    dataset = TaskonomyDataset(taskonomy, split="train", image_size=32)
    image, target = dataset[0]

    image_column = int(np.array(image.convert("L")).sum(axis=0).argmax())
    target_column = int(target.sum(dim=0).argmax())
    assert image_column == target_column


def test_geometry_matches_dense_folder_dataset(tmp_path: Path, taskonomy: Path):
    """The crop is inherited, not copied — this asserts it stayed that way.

    Step 6c-1 kept the detection dataset's crop byte-identical to
    DenseFolderDataset's and tested it. Here the code is literally shared, so
    the test is cheap; it earns its place by failing loudly if someone later
    overrides `_crop_image` on one side only.
    """
    flat = tmp_path / "flat"
    (flat / "images").mkdir(parents=True)
    (flat / "depths").mkdir()
    source = Image.open(taskonomy / "rgb" / "allensville" / "point_0_view_0_domain_rgb.png")
    source.save(flat / "images" / "a.png")
    np.save(flat / "depths" / "a.npy", np.ones((64, 96), dtype=np.float32))

    reference = DenseFolderDataset(flat, image_size=32)
    dataset = TaskonomyDataset(taskonomy, split="train", image_size=32)
    assert np.array_equal(np.array(dataset[0][0]), np.array(reference[0][0]))


def test_a_reconstruction_derived_domain_is_refused(taskonomy: Path):
    """Those have invalid regions in mask_valid/ that this does not read.

    Scoring against reprojection holes would depress every backbone equally and
    silently, which is worse than not supporting the domain.
    """
    with pytest.raises(NotImplementedError, match="mask_valid"):
        TaskonomyDataset(taskonomy, domain="normal", split="train")


def test_an_unknown_domain_is_refused(taskonomy: Path):
    with pytest.raises(ValueError, match="Unknown Taskonomy domain"):
        TaskonomyDataset(taskonomy, domain="edges", split="train")


def test_edge_texture_is_not_treated_as_needing_a_valid_mask(taskonomy: Path):
    """It is computed from the RGB frame, so every pixel is a real measurement.

    The complement of the test above: the gate must let this domain through,
    or the one supported low-level target would be refused along with the rest.
    """
    assert "edge_texture" in TASKONOMY_DOMAINS
    assert (
        len(TaskonomyDataset(taskonomy, domain="edge_texture", split="train", image_size=32)) == 4
    )


def test_max_images_truncates_the_split(taskonomy: Path):
    """The tiny train list is 272k rows; building all of them to keep 600 is waste."""
    assert len(TaskonomyDataset(taskonomy, split="train", max_images=2, image_size=32)) == 2


def test_buildings_filters_and_rejects_unknown_ones(taskonomy: Path):
    dataset = TaskonomyDataset(taskonomy, split="train", buildings=["allensville"], image_size=32)
    assert len(dataset) == 2
    assert all(stem.startswith("allensville/") for stem in dataset.stems)

    with pytest.raises(ValueError, match="disjoint by building"):
        TaskonomyDataset(taskonomy, split="train", buildings=["nowhere"], image_size=32)


def test_subset_reindexes_the_parallel_lists_together(taskonomy: Path):
    """Slicing one alone pairs a target with the wrong image, silently."""
    dataset = TaskonomyDataset(taskonomy, split="train", image_size=32)
    smaller = dataset.subset(2)
    assert len(smaller.stems) == len(smaller.image_paths) == len(smaller.target_paths) == 2
    for stem, image_path in zip(smaller.stems, smaller.image_paths, strict=True):
        assert image_path.parent.name == stem.split("/")[0]


def test_fingerprint_distinguishes_size_and_needs_no_stat(taskonomy: Path):
    """Overridden to skip stat-ing 2N files; it must still separate two splits."""
    full = TaskonomyDataset(taskonomy, split="train", image_size=32)
    assert full.fingerprint() != full.subset(2).fingerprint()
    assert (
        full.fingerprint()
        != TaskonomyDataset(taskonomy, split="train", image_size=16).fingerprint()
    )


def test_target_scale_actually_reaches_the_loader(taskonomy: Path):
    """A parameter that is recorded but does nothing is the QuickGELU failure.

    `DenseFolderDataset.target()` applies `target_scale` only on its default
    depth path, so handing it a bare custom loader silently drops the value —
    while `describe()` still reports it and the fingerprint still folds it in.
    Written after a scale sweep returned four identical numbers.
    """
    coarse = TaskonomyDataset(taskonomy, split="train", image_size=32, target_scale=100.0)
    fine = TaskonomyDataset(taskonomy, split="train", image_size=32, target_scale=10.0)
    assert fine.target(0).max().item() == pytest.approx(10.0 * coarse.target(0).max().item())


def test_a_custom_target_loader_overrides_the_scale(taskonomy: Path):
    """The escape hatch stays an escape hatch: the caller's loader wins outright."""
    dataset = TaskonomyDataset(
        taskonomy,
        split="train",
        image_size=32,
        target_scale=100.0,
        target_loader=lambda path: torch.full((32, 32), 7.0),
    )
    assert dataset.target(0).unique().tolist() == [7.0]


def test_describe_records_the_domain_and_partition(taskonomy: Path):
    """Both change what the number means, so both belong in the record."""
    info = TaskonomyDataset(taskonomy, split="train", image_size=32).describe()
    assert info["domain"] == "edge_texture"
    assert info["partition"] == "tiny"
    assert info["dataset"] == "taskonomy_edge_texture"

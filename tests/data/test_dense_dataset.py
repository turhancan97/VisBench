"""Dense-target datasets.

Most of these are about **geometry**. A depth target that has been resized or
cropped differently from its image produces a probe that trains against
misaligned supervision, and the only symptom is that the numbers come out bad —
no error, no warning. The correspondence task already paid for this lesson once
(a homography in original pixels against features from a 224 crop scored
recall@1px at 0.003), which is why it gets tested directly here.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from visbench.data import DenseFolderDataset, load_depth_map, load_normal_map


def build(root, depths, images=None, image_suffix=".png", target_suffix=".npy"):
    """Write a folder pair from a list of (H, W) arrays."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "depths").mkdir(parents=True, exist_ok=True)
    for index, depth in enumerate(depths):
        stem = f"s{index:03d}"
        picture = images[index] if images is not None else np.stack([depth] * 3, -1)
        Image.fromarray(picture.astype(np.uint8)).save(root / "images" / f"{stem}{image_suffix}")
        if target_suffix == ".npy":
            np.save(root / "depths" / f"{stem}.npy", depth.astype(np.float32))
        else:
            Image.fromarray(depth.astype(np.uint16)).save(
                root / "depths" / f"{stem}{target_suffix}"
            )
    return root


@pytest.fixture
def ramp_dataset(tmp_path):
    """Four images whose depth is a horizontal ramp, 1..5 metres."""
    ramp = np.linspace(1.0, 5.0, 64)[None, :].repeat(64, 0)
    return DenseFolderDataset(build(tmp_path, [ramp] * 4), image_size=64)


# -- pairing ------------------------------------------------------------------


class TestPairing:
    def test_pairs_by_stem(self, ramp_dataset):
        assert len(ramp_dataset) == 4
        assert [p.stem for p in ramp_dataset.image_paths] == ramp_dataset.stems

    def test_a_missing_target_raises(self, tmp_path):
        root = build(tmp_path, [np.ones((8, 8))] * 3)
        (root / "depths" / "s001.npy").unlink()
        with pytest.raises(ValueError, match="only one of"):
            DenseFolderDataset(root, image_size=8)

    def test_a_missing_image_raises(self, tmp_path):
        root = build(tmp_path, [np.ones((8, 8))] * 3)
        (root / "images" / "s002.png").unlink()
        with pytest.raises(ValueError, match="only one of"):
            DenseFolderDataset(root, image_size=8)

    def test_the_message_names_the_offenders(self, tmp_path):
        root = build(tmp_path, [np.ones((8, 8))] * 3)
        (root / "depths" / "s001.npy").unlink()
        with pytest.raises(ValueError, match="s001"):
            DenseFolderDataset(root, image_size=8)

    def test_a_missing_directory_says_which(self, tmp_path):
        (tmp_path / "images").mkdir()
        with pytest.raises(NotADirectoryError, match="target_dir"):
            DenseFolderDataset(tmp_path, image_size=8)


# -- geometry: the whole reason this class exists -----------------------------


class TestMatchedGeometry:
    def test_image_and_target_come_out_the_same_size(self, tmp_path):
        tall = np.linspace(0, 255, 100 * 60).reshape(100, 60)
        dataset = DenseFolderDataset(build(tmp_path, [tall]), image_size=32)
        image, target = dataset[0]
        assert image.size == (32, 32)
        assert tuple(target.shape) == (32, 32)

    def test_a_feature_lands_in_the_same_place_in_both(self, tmp_path):
        """The test that would have caught the correspondence bug.

        A sharp vertical step, in the image and in the depth map, at the same
        column. After resize and centre-crop the step must still be at the same
        column in both — otherwise the probe trains against supervision that is
        offset from the features it is given.
        """
        height, width = 90, 150
        step_column = 100
        depth = np.zeros((height, width), dtype=np.float32)
        depth[:, step_column:] = 5.0
        picture = np.stack([np.where(depth > 0, 255, 0)] * 3, -1)

        dataset = DenseFolderDataset(build(tmp_path, [depth], images=[picture]), image_size=48)
        image, target = dataset[0]

        image_column = np.array(image.convert("L"), dtype=np.float32).mean(axis=0)
        target_column = target.numpy().mean(axis=0)

        image_edge = int(np.abs(np.diff(image_column)).argmax())
        target_edge = int(np.abs(np.diff(target_column)).argmax())
        assert abs(image_edge - target_edge) <= 1, (
            f"image step at column {image_edge}, depth step at {target_edge}"
        )

    def test_non_square_input_is_centre_cropped_not_squashed(self, tmp_path):
        """Squashing would change the aspect ratio, and with it the geometry a
        mid-level probe exists to measure."""
        wide = np.tile(np.linspace(0, 255, 200), (50, 1))
        dataset = DenseFolderDataset(build(tmp_path, [wide]), image_size=32)
        target = dataset.target(0)
        # The short side sets the scale, so the crop keeps the middle columns:
        # a squashed map would still span the full 0..255 range.
        assert target.min() > 0
        assert target.max() < 255


class TestNearestNeighbourResampling:
    def test_no_new_values_are_invented(self, tmp_path):
        """Bilinear resampling averages across depth discontinuities, inventing
        surfaces no sensor saw. Every resampled value must be one of the
        originals."""
        depth = np.zeros((40, 40), dtype=np.float32)
        depth[20:, :] = 7.0
        dataset = DenseFolderDataset(build(tmp_path, [depth]), image_size=16)
        assert set(dataset.target(0).unique().tolist()) <= {0.0, 7.0}

    def test_holes_do_not_bleed_into_valid_depth(self, tmp_path):
        """The worse half of the same problem: interpolating a zero hole against
        valid depth produces a halo of plausible wrong depths that the valid
        mask no longer excludes."""
        depth = np.full((40, 40), 3.0, dtype=np.float32)
        depth[:, :20] = 0.0
        dataset = DenseFolderDataset(build(tmp_path, [depth]), image_size=16)
        values = dataset.target(0).unique().tolist()
        assert all(value in (0.0, 3.0) for value in values), values


# -- target values ------------------------------------------------------------


class TestTargetValues:
    def test_npy_is_taken_at_face_value(self, ramp_dataset):
        target = ramp_dataset.target(0)
        assert target.min() == pytest.approx(1.0, abs=0.1)
        assert target.max() == pytest.approx(5.0, abs=0.1)

    def test_integer_png_is_divided_by_the_scale(self, tmp_path):
        """Millimetres in a 16-bit container, the usual depth-dataset layout."""
        millimetres = np.full((16, 16), 2500, dtype=np.uint16)
        root = build(tmp_path, [millimetres], target_suffix=".png")
        dataset = DenseFolderDataset(root, image_size=16, target_scale=1000.0)
        assert dataset.target(0).mean().item() == pytest.approx(2.5)

    def test_max_target_marks_invalid_rather_than_clamping(self, tmp_path):
        """Clamping would train and score against a wall of fabricated depth at
        exactly the cap; a pixel beyond sensor range is unknown, not distant."""
        depth = np.array([[1.0, 50.0]], dtype=np.float32).repeat(16, 0).repeat(8, 1)
        dataset = DenseFolderDataset(build(tmp_path, [depth]), image_size=16, max_target=10.0)
        values = dataset.target(0)
        assert values.max().item() <= 10.0
        assert (values == 0).any(), "out-of-range depth should be invalid, not clamped"
        assert not (values == 10.0).any(), "clamped to the cap instead of zeroed"

    def test_targets_stacks_every_map(self, ramp_dataset):
        assert tuple(ramp_dataset.targets().shape) == (4, 64, 64)

    def test_labels_matches_targets(self, ramp_dataset):
        assert torch.equal(torch.stack(ramp_dataset.labels()), ramp_dataset.targets())

    def test_a_custom_loader_is_used(self, tmp_path):
        root = build(tmp_path, [np.ones((16, 16))])
        dataset = DenseFolderDataset(
            root, image_size=16, target_loader=lambda path: torch.full((16, 16), 4.0)
        )
        assert dataset.target(0).mean().item() == pytest.approx(4.0)

    def test_a_three_dimensional_target_file_is_refused(self, tmp_path):
        root = build(tmp_path, [np.ones((8, 8))])
        np.save(root / "depths" / "s000.npy", np.ones((8, 8, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="2D"):
            DenseFolderDataset(root, image_size=8).target(0)


# -- identity -----------------------------------------------------------------


class TestIdentity:
    def test_cache_identity_tracks_the_image_only(self, tmp_path, ramp_dataset):
        """Cached features depend on the image and nothing else, so editing a
        depth map must not invalidate a perfectly good extraction."""
        before = ramp_dataset.cache_identity(0)
        np.save(ramp_dataset.target_paths[0], np.zeros((64, 64), dtype=np.float32))
        assert ramp_dataset.cache_identity(0) == before

    def test_fingerprint_does_track_the_target(self, tmp_path):
        """The record must distinguish two runs whose targets differ."""
        depth = np.ones((16, 16), dtype=np.float32)
        first = DenseFolderDataset(build(tmp_path / "a", [depth]), image_size=16)
        second = DenseFolderDataset(
            build(tmp_path / "b", [depth], target_suffix=".png"), image_size=16
        )
        assert first.fingerprint() != second.fingerprint()

    def test_fingerprint_includes_the_geometry(self, tmp_path):
        """image_size decides the crop, so one folder at two resolutions is two
        different sets of targets and the records must not collide."""
        root = build(tmp_path, [np.ones((32, 32), dtype=np.float32)])
        small = DenseFolderDataset(root, image_size=16)
        large = DenseFolderDataset(root, image_size=32)
        assert small.fingerprint() != large.fingerprint()

    def test_fingerprint_is_stable(self, ramp_dataset):
        assert ramp_dataset.fingerprint() == ramp_dataset.fingerprint()

    def test_describe_carries_the_working_resolution(self, ramp_dataset):
        described = ramp_dataset.describe()
        assert described["image_size"] == 64
        assert described["dataset_size"] == 4


# -- loader -------------------------------------------------------------------


def test_load_depth_map_reads_npy(tmp_path):
    path = tmp_path / "d.npy"
    np.save(path, np.full((4, 4), 2.0, dtype=np.float32))
    assert load_depth_map(path).mean().item() == pytest.approx(2.0)


def test_load_depth_map_scales_integers(tmp_path):
    path = tmp_path / "d.png"
    Image.fromarray(np.full((4, 4), 3000, dtype=np.uint16)).save(path)
    assert load_depth_map(path, scale=1000.0).mean().item() == pytest.approx(3.0)


def test_load_depth_map_returns_float32(tmp_path):
    path = tmp_path / "d.png"
    Image.fromarray(np.full((4, 4), 3000, dtype=np.uint16)).save(path)
    assert load_depth_map(path).dtype == torch.float32


# -- vector targets -----------------------------------------------------------


def build_normals(root, maps, size=32):
    """Write a folder pair whose targets are (3, H, W) normal maps."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "normals").mkdir(parents=True, exist_ok=True)
    for index, normals in enumerate(maps):
        stem = f"s{index:03d}"
        Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8)).save(
            root / "images" / f"{stem}.png"
        )
        np.save(root / "normals" / f"{stem}.npy", normals.astype(np.float32))
    return root


class TestVectorTargets:
    """Surface normals are the first target that is not one number per pixel."""

    def test_a_three_channel_target_survives_the_geometry(self, tmp_path):
        normals = np.zeros((3, 40, 60), dtype=np.float32)
        normals[2] = 1.0
        root = build_normals(tmp_path, [normals], size=60)
        dataset = DenseFolderDataset(
            root, target_dir="normals", image_size=32, target_loader=load_normal_map
        )
        assert dataset.target(0).shape == (3, 32, 32)
        assert dataset.targets().shape == (1, 3, 32, 32)

    def test_the_crop_matches_the_image(self, tmp_path):
        """The same sharp-step check the depth path gets: a normal map cropped
        differently from its image trains a probe against a shifted world."""
        normals = np.zeros((3, 32, 64), dtype=np.float32)
        normals[2, :, :32] = 1.0
        normals[0, :, 32:] = 1.0
        root = build_normals(tmp_path, [normals], size=64)
        dataset = DenseFolderDataset(
            root, target_dir="normals", image_size=32, target_loader=load_normal_map
        )
        target = dataset.target(0)
        # A 32x64 map centre-cropped to 32x32 keeps columns 16..47: half of
        # each half, so the step lands exactly in the middle.
        assert target[2, :, 15].mean().item() == pytest.approx(1.0)
        assert target[0, :, 16].mean().item() == pytest.approx(1.0)

    def test_nearest_resampling_keeps_vectors_unit(self, tmp_path):
        """Bilinear would average two unit vectors across an edge into
        something that is not merely wrong but not even unit length."""
        rng = np.random.RandomState(0)
        raw = rng.randn(3, 64, 64).astype(np.float32)
        raw /= np.linalg.norm(raw, axis=0, keepdims=True)
        root = build_normals(tmp_path, [raw], size=64)
        dataset = DenseFolderDataset(
            root, target_dir="normals", image_size=16, target_loader=load_normal_map
        )
        lengths = dataset.target(0).norm(dim=0)
        assert torch.allclose(lengths, torch.ones_like(lengths), atol=1e-5)

    def test_max_target_is_refused_for_a_vector_map(self, tmp_path):
        """It caps a scalar quantity; silently applying it per channel would
        zero the x component of every steep normal."""
        normals = np.zeros((3, 32, 32), dtype=np.float32)
        normals[2] = 1.0
        root = build_normals(tmp_path, [normals])
        dataset = DenseFolderDataset(
            root,
            target_dir="normals",
            image_size=32,
            max_target=10.0,
            target_loader=load_normal_map,
        )
        with pytest.raises(ValueError, match="caps a scalar quantity"):
            dataset.target(0)

    def test_a_four_dimensional_target_is_refused(self, tmp_path):
        root = build_normals(tmp_path, [np.zeros((3, 32, 32), dtype=np.float32)])
        dataset = DenseFolderDataset(
            root,
            target_dir="normals",
            image_size=32,
            target_loader=lambda p: torch.zeros(1, 3, 4, 4),
        )
        with pytest.raises(ValueError, match="expected .*H, W"):
            dataset.target(0)


class TestLoadNormalMap:
    def test_it_reads_channels_first_npy(self, tmp_path):
        path = tmp_path / "n.npy"
        array = np.zeros((3, 4, 4), dtype=np.float32)
        array[2] = 1.0
        np.save(path, array)
        assert load_normal_map(path).shape == (3, 4, 4)
        assert load_normal_map(path)[2].mean().item() == pytest.approx(1.0)

    def test_it_reads_channels_last_npy(self, tmp_path):
        """(H, W, 3) is at least as common on disk as (3, H, W)."""
        path = tmp_path / "n.npy"
        array = np.zeros((4, 4, 3), dtype=np.float32)
        array[..., 2] = 1.0
        np.save(path, array)
        assert load_normal_map(path).shape == (3, 4, 4)
        assert load_normal_map(path)[2].mean().item() == pytest.approx(1.0)

    def test_it_decodes_the_eight_bit_convention(self, tmp_path):
        """GeoNet and every other published NYU normal set store 2*v/255 - 1."""
        path = tmp_path / "n.png"
        encoded = np.zeros((4, 4, 3), dtype=np.uint8)
        encoded[..., 2] = 255  # z = +1
        encoded[..., 0] = 128  # x ~ 0
        encoded[..., 1] = 128
        Image.fromarray(encoded).save(path)
        normals = load_normal_map(path)
        assert normals[2].mean().item() == pytest.approx(1.0, abs=1e-2)
        assert normals[0].abs().max().item() == pytest.approx(0.0, abs=1e-2)

    def test_output_is_unit_length(self, tmp_path):
        path = tmp_path / "n.npy"
        rng = np.random.RandomState(1)
        np.save(path, (rng.randn(3, 8, 8) * 7.0).astype(np.float32))
        lengths = load_normal_map(path).norm(dim=0)
        assert torch.allclose(lengths, torch.ones_like(lengths), atol=1e-5)

    def test_the_grey_invalid_pixel_becomes_zero(self, tmp_path):
        """(128, 128, 128) decodes to a length of about 0.007 — no direction at
        all. Zeroing it is what makes the metric's default mask correct."""
        path = tmp_path / "n.png"
        Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8)).save(path)
        assert load_normal_map(path).abs().max().item() == 0.0

    def test_a_two_dimensional_array_is_refused(self, tmp_path):
        path = tmp_path / "n.npy"
        np.save(path, np.zeros((4, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="a normal map is 3D"):
            load_normal_map(path)

    def test_an_array_with_no_length_three_axis_is_refused(self, tmp_path):
        path = tmp_path / "n.npy"
        np.save(path, np.zeros((4, 4, 5), dtype=np.float32))
        with pytest.raises(ValueError, match="length-3 axis"):
            load_normal_map(path)

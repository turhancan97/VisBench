"""NAVI — the pair set a relative-pose probe is scored on.

The class under test presents *unique frames* and puts the pairing in
`labels()` as indices into itself, which is the move `TwoAFCDataset` makes for
triplets. At eight partners per training anchor the real split holds 50,519
pairs over 8,217 frames, so the alternative — presenting pairs — would extract
every frame eight times to hold one copy of it.

Four properties carry the protocol and each is silent when wrong, which is what
these tests are for.

**The pair count is the protocol.** Rotation error keeps falling as partners
are added, so two pose numbers are comparable only if they drew the same pairs.
The fingerprint therefore has to move with every sampler parameter, and the
validation split has to stay at one partner however high `partners` goes — it
is the yardstick the no-feature floor is measured on.

**The direction of the relative pose**, which is the same magnitude and the
wrong sign if reversed.

**Millimetres become metres at the loader**, or an MSE loss optimises
translation alone and never learns rotation.

**EXIF orientation is applied**, because 327 of the release's 8,217 frames
carry a half-turn tag whose camera pose describes the turned image. Reading
them as stored supervises 4.0% of the data against its own negation.

The fixtures are synthetic scenes in the real layout, so the fast suite needs
none of NAVI. One test reads the real release when it happens to be present.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from visbench.data import NaviPoseDataset
from visbench.data.dense import DenseFolderDataset
from visbench.metrics.pose import quaternion_to_matrix, rotation_angle_deg

REAL_ROOT = Path("/shared/sets/datasets/vision/probing_3D/navi_v1")


def quaternion_about_z(degrees: float) -> list[float]:
    half = math.radians(degrees) / 2
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def write_scene(
    root: Path,
    object_id: str,
    scene: str,
    cameras: list[tuple[float, list[float], str]],
    size: tuple[int, int] = (40, 30),
) -> Path:
    """One multiview scene: ``cameras`` is ``(z_rotation_deg, translation_mm, split)``."""
    directory = root / object_id / scene
    (directory / "images").mkdir(parents=True, exist_ok=True)
    records = []
    for index, (degrees, translation, split) in enumerate(cameras):
        filename = f"{index:03d}.jpg"
        Image.new("RGB", size, (index * 7 % 255, 40, 90)).save(directory / "images" / filename)
        records.append(
            {
                "object_id": object_id,
                "camera": {
                    "q": quaternion_about_z(degrees),
                    "t": translation,
                    "focal_length": 1000.0,
                    "camera_model": "test",
                },
                "filename": filename,
                "image_size": list(size),
                "scene_name": scene,
                "split": split,
            }
        )
    (directory / "annotations.json").write_text(json.dumps(records))
    return directory


def simple_root(tmp_path: Path, partners_available: int = 3) -> Path:
    """One scene whose views sit 30 degrees apart, plus a far one at 170."""
    cameras = [(30.0 * i, [0.0, 0.0, 1000.0 * i], "train") for i in range(partners_available + 1)]
    cameras.append((170.0, [0.0, 0.0, 0.0], "val"))
    write_scene(tmp_path, "widget", "multiview_00_test", cameras)
    return tmp_path


class TestTheShapeOfIt:
    def test_frames_are_unique_and_pairs_index_into_them(self, tmp_path):
        dataset = NaviPoseDataset(simple_root(tmp_path), split="train", image_size=16)
        indices, pose = dataset.labels()

        assert len(dataset.paths) == len(set(dataset.paths))
        assert indices.dtype == torch.long
        assert int(indices.max()) < len(dataset)
        assert pose.shape == (len(indices), 7)
        assert dataset.labels() == dataset.pairs

    def test_an_item_is_an_image_and_no_label(self, tmp_path):
        dataset = NaviPoseDataset(simple_root(tmp_path), split="train", image_size=16)
        image, label = dataset[0]
        assert image.size == (16, 16)
        assert label is None

    def test_an_anchor_is_never_its_own_partner(self, tmp_path):
        dataset = NaviPoseDataset(simple_root(tmp_path), split="train", image_size=16)
        indices = dataset.labels().indices
        assert not bool((indices[:, 0] == indices[:, 1]).any())

    def test_a_scene_with_one_usable_frame_is_skipped(self, tmp_path):
        """A missing file leaves an annotation that cannot be paired.

        Dropped rather than raising, because NAVI ships scenes whose images are
        not all present and a probe should read what is there — but a scene
        reduced to one frame has no pair to contribute.
        """
        write_scene(tmp_path, "lonely", "multiview_00_test", [(0.0, [0.0] * 3, "train")])
        write_scene(
            tmp_path,
            "pair",
            "multiview_00_test",
            [(0.0, [0.0] * 3, "train"), (30.0, [0.0] * 3, "train")],
        )
        dataset = NaviPoseDataset(tmp_path, split="train", image_size=16)
        assert len(dataset.labels().indices) == 2
        assert all("pair" in str(path) for path in dataset.paths)

    def test_a_split_with_no_pair_raises_rather_than_being_empty(self, tmp_path):
        write_scene(
            tmp_path,
            "widget",
            "multiview_00_test",
            [(0.0, [0.0] * 3, "train"), (30.0, [0.0] * 3, "train")],
        )
        with pytest.raises(ValueError, match="No pairs in split 'val'"):
            NaviPoseDataset(tmp_path, split="val", image_size=16)


class TestTheGeometry:
    def test_translation_arrives_in_metres(self, tmp_path):
        """1000 mm is 1 m, and getting this wrong does not look like a units bug.

        It looks like every backbone scoring worse than a constant, because an
        MSE loss over ``[quat, trans]`` then optimises a translation four
        orders of magnitude larger than the quaternion and never learns the
        rotation being scored.
        """
        write_scene(
            tmp_path,
            "widget",
            "multiview_00_test",
            [(0.0, [0.0, 0.0, 0.0], "train"), (30.0, [0.0, 0.0, 1000.0], "train")],
        )
        dataset = NaviPoseDataset(tmp_path, split="train", image_size=16)
        assert float(dataset.labels().pose[:, 4:].norm(dim=-1).max()) == pytest.approx(
            1.0, abs=1e-5
        )

    def test_the_pose_is_the_partner_with_respect_to_the_anchor(self, tmp_path):
        """Pinned because the opposite composition is equally plausible and
        reads as an ordinary number."""
        write_scene(
            tmp_path,
            "widget",
            "multiview_00_test",
            [(0.0, [0.0] * 3, "train"), (40.0, [0.0] * 3, "train")],
        )
        dataset = NaviPoseDataset(tmp_path, split="train", image_size=16)
        indices, pose = dataset.labels()

        anchor_first = indices[0, 0] < indices[0, 1]
        turn = 40.0 if anchor_first else -40.0
        expected = quaternion_to_matrix(torch.tensor(quaternion_about_z(turn), dtype=torch.float32))
        assert (quaternion_to_matrix(pose[0, :4]) - expected).abs().max() < 1e-5

    def test_partners_are_drawn_within_max_angle(self, tmp_path):
        dataset = NaviPoseDataset(
            simple_root(tmp_path, partners_available=4),
            split="train",
            max_angle=60.0,
            image_size=16,
        )
        angles = dataset.relative_rotation_deg()
        assert float(angles.max()) <= 60.0 + 1e-3
        assert float(angles.min()) > 0.0

    def test_max_angle_changes_which_partners_are_eligible(self, tmp_path):
        """A view 170 degrees away is admitted at 180 and not at 120, so the
        angle is a protocol parameter rather than a filter detail: it moves the
        floor, since the floor is the mean pose of whatever was drawn."""
        write_scene(
            tmp_path,
            "widget",
            "multiview_00_test",
            [(0.0, [0.0] * 3, "train"), (170.0, [0.0] * 3, "train")],
        )
        with pytest.raises(ValueError, match="No pairs"):
            NaviPoseDataset(tmp_path, split="train", max_angle=120.0, image_size=16)
        wide = NaviPoseDataset(tmp_path, split="train", max_angle=180.0, image_size=16)
        assert float(rotation_angle_deg(wide.labels().pose[:, :4]).max()) == pytest.approx(
            170.0, abs=1e-2
        )

    def test_the_crop_agrees_with_the_dense_dataset_pixel_for_pixel(self, tmp_path):
        """The fourth copy of one crop, pinned like the other three.

        A pose number and a dense number over the same frames must differ in
        what is asked, not in which pixels were asked about. The image is
        non-square, which is where a missed rescale shows.
        """
        rng = np.random.default_rng(1)
        array = rng.integers(0, 255, (90, 140, 3), dtype=np.uint8)

        scene = write_scene(
            tmp_path / "navi",
            "widget",
            "multiview_00_test",
            [(0.0, [0.0] * 3, "train"), (30.0, [0.0] * 3, "train")],
        )
        Image.fromarray(array).save(scene / "images" / "000.jpg")

        dense_root = tmp_path / "dense"
        (dense_root / "images").mkdir(parents=True)
        (dense_root / "depths").mkdir()
        Image.fromarray(array).save(dense_root / "images" / "a.jpg")
        np.save(dense_root / "depths" / "a.npy", rng.random((90, 140)).astype(np.float32))

        navi = NaviPoseDataset(tmp_path / "navi", split="train", image_size=32)
        dense = DenseFolderDataset(root=dense_root, image_size=32)
        frame = next(i for i, path in enumerate(navi.paths) if path.name == "000.jpg")
        assert np.array_equal(np.asarray(navi[frame][0]), np.asarray(dense[0][0]))

    def test_exif_orientation_is_applied(self, tmp_path):
        """327 of NAVI's 8,217 frames carry a half-turn tag, and the camera
        pose describes the turned image — checked on the one scene where tagged
        and untagged frames sit together. Reading them as stored hands the
        backbone an upside-down image beside a pose that says otherwise.
        """
        scene = write_scene(
            tmp_path,
            "widget",
            "multiview_00_test",
            [(0.0, [0.0] * 3, "train"), (30.0, [0.0] * 3, "train")],
        )
        rng = np.random.default_rng(2)
        array = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
        image = Image.fromarray(array)
        exif = image.getexif()
        exif[274] = 3  # rotate 180
        image.save(scene / "images" / "000.jpg", exif=exif)

        dataset = NaviPoseDataset(tmp_path, split="train", image_size=48)
        frame = next(i for i, path in enumerate(dataset.paths) if path.name == "000.jpg")
        stored = np.asarray(Image.open(scene / "images" / "000.jpg").convert("RGB"))
        assert np.array_equal(np.asarray(dataset[frame][0]), stored[::-1, ::-1])


class TestThePairCountIsProtocol:
    def test_extra_partners_grow_only_the_training_split(self, tmp_path):
        root = simple_root(tmp_path, partners_available=4)
        one = NaviPoseDataset(root, split="train", partners=1, image_size=16)
        many = NaviPoseDataset(root, split="train", partners=3, image_size=16)
        assert len(many.labels().indices) > len(one.labels().indices)

    def test_the_first_partner_of_a_many_partner_draw_is_the_one_partner_draw(self, tmp_path):
        """What makes the pair-count curve one experiment rather than five.

        The extras come from a second generator precisely so the first draw is
        untouched; without that, adding partners would resample everything and
        a change in the score could not be attributed to the extra data.
        """
        root = simple_root(tmp_path, partners_available=4)
        one = NaviPoseDataset(root, split="train", partners=1, image_size=16)
        many = NaviPoseDataset(root, split="train", partners=3, image_size=16)
        first = len(one.labels().indices)
        assert torch.allclose(many.labels().pose[:first], one.labels().pose)

    def test_validation_stays_at_one_partner(self, tmp_path):
        """The yardstick must not move. Growing both halves at once would
        change the measurement and the thing measured together."""
        cameras = [(30.0 * i, [0.0] * 3, "val") for i in range(5)]
        write_scene(tmp_path, "widget", "multiview_00_test", cameras)
        one = NaviPoseDataset(tmp_path, split="val", partners=1, image_size=16)
        asked = NaviPoseDataset(tmp_path, split="val", partners=8, image_size=16)

        assert torch.equal(one.labels().indices, asked.labels().indices)
        assert torch.equal(one.labels().pose, asked.labels().pose)
        assert one.fingerprint() == asked.fingerprint()

    def test_a_val_split_reports_the_partners_it_holds_and_the_one_asked_for(self, tmp_path):
        """``partners_requested`` beside ``partners``, the way a record keeps
        ``pooling_requested`` beside the resolved pooling."""
        cameras = [(30.0 * i, [0.0] * 3, "val") for i in range(5)]
        write_scene(tmp_path, "widget", "multiview_00_test", cameras)
        described = NaviPoseDataset(tmp_path, split="val", partners=8, image_size=16).describe()
        assert described["partners"] == 1
        assert described["partners_requested"] == 8

    def test_stride_shortens_the_draw(self, tmp_path):
        root = simple_root(tmp_path, partners_available=4)
        full = NaviPoseDataset(root, split="train", image_size=16)
        strided = NaviPoseDataset(root, split="train", stride=2, image_size=16)
        assert len(strided.labels().indices) < len(full.labels().indices)

    def test_max_pairs_shortens_without_slicing_the_frames(self, tmp_path):
        root = simple_root(tmp_path, partners_available=4)
        short = NaviPoseDataset(root, split="train", max_pairs=2, image_size=16)
        assert len(short.labels().indices) == 2
        assert int(short.labels().indices.max()) < len(short)

    def test_subset_is_refused(self, tmp_path):
        dataset = NaviPoseDataset(simple_root(tmp_path), split="train", image_size=16)
        with pytest.raises(NotImplementedError, match="max_pairs"):
            dataset.subset(2)


class TestTheRecord:
    def test_describe_carries_every_sampler_parameter(self, tmp_path):
        """``dataset_size`` counts frames here, so the pair count and the
        sampler have to be stated — they are the protocol a second run matches."""
        dataset = NaviPoseDataset(
            simple_root(tmp_path, partners_available=4),
            split="train",
            partners=2,
            max_angle=90.0,
            pair_seed=3,
            stride=1,
            image_size=16,
        )
        described = dataset.describe()
        assert described["dataset_size"] == len(dataset.paths)
        assert described["num_pairs"] == len(dataset.labels().indices)
        assert described["partners"] == 2
        assert described["max_angle"] == 90.0
        assert described["pair_seed"] == 3
        assert described["stride"] == 1
        assert described["image_size"] == 16

    def test_the_fingerprint_is_stable_across_constructions(self, tmp_path):
        root = simple_root(tmp_path, partners_available=4)
        first = NaviPoseDataset(root, split="train", partners=2, image_size=16)
        second = NaviPoseDataset(root, split="train", partners=2, image_size=16)
        assert first.fingerprint() == second.fingerprint()

    @pytest.mark.parametrize(
        "overrides",
        [
            {"partners": 3},
            {"max_angle": 90.0},
            {"pair_seed": 9},
            {"stride": 2},
            {"image_size": 24},
        ],
    )
    def test_every_sampler_parameter_moves_the_fingerprint(self, tmp_path, overrides):
        """Two pair counts over the same frames are two measurements — error
        falls by degrees between them — so they must not share a comparability
        group."""
        root = simple_root(tmp_path, partners_available=4)
        base = {"split": "train", "partners": 2, "image_size": 16}
        reference = NaviPoseDataset(root, **base).fingerprint()
        assert NaviPoseDataset(root, **{**base, **overrides}).fingerprint() != reference

    def test_cache_identity_follows_the_file(self, tmp_path):
        dataset = NaviPoseDataset(simple_root(tmp_path), split="train", image_size=16)
        identity = dataset.cache_identity(0)
        assert identity is not None and str(dataset.paths[0]) in identity


class TestTheArguments:
    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"split": "test"}, "Unknown split"),
            ({"partners": 0}, "partners"),
            ({"stride": 0}, "stride"),
            ({"image_size": 0}, "image_size"),
            ({"max_angle": 0.0}, "max_angle"),
            ({"max_angle": 200.0}, "max_angle"),
        ],
    )
    def test_bad_arguments_are_refused(self, tmp_path, kwargs, message):
        root = simple_root(tmp_path)
        with pytest.raises(ValueError, match=message):
            NaviPoseDataset(root, **{"image_size": 16, **kwargs})

    def test_a_missing_root_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            NaviPoseDataset(tmp_path / "absent")


@pytest.mark.slow
@pytest.mark.skipif(not REAL_ROOT.is_dir(), reason="NAVI is not on this machine")
def test_the_real_release_reproduces_the_premeasured_pair_set():
    """The numbers `scripts/premeasure_pose.py` parked, from the shipped class.

    The pair set is the protocol, so this is the test that says the probe to
    come will measure what the pre-measurement measured: 6,477 training pairs
    at one partner, 50,519 at eight, 1,740 validation pairs either way, and a
    no-feature floor of 66.94 / 66.85 degrees.
    """
    from visbench.metrics.pose import mean_pose_floor

    val = NaviPoseDataset(REAL_ROOT, split="val")
    train = NaviPoseDataset(REAL_ROOT, split="train", partners=1)
    assert len(val.labels().indices) == 1740
    assert len(train.labels().indices) == 6477
    floor = mean_pose_floor(train.labels().pose, val.labels().pose)
    assert floor["floor_rotation_error_deg"] == pytest.approx(66.94, abs=0.01)

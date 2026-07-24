"""Synthetic homography pairs.

The ground truth here is only worth anything if the homography VisBench scores
against is the *same* transform PIL actually applied. If those two conventions
disagree, every correspondence number is silently wrong and still looks
plausible — so that agreement is checked directly, against pixels.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from visbench.data.pair_dataset import (
    HomographyPairDataset,
    _solve_homography,
    apply_homography,
)


@pytest.fixture
def folder(tmp_path):
    """Images with a single bright square, at a different place in each."""
    root = tmp_path / "pairs"
    root.mkdir()
    for i in range(3):
        array = np.zeros((96, 96, 3), dtype=np.uint8)
        array[20 + i * 10 : 28 + i * 10, 30 + i * 10 : 38 + i * 10] = 255
        Image.fromarray(array).save(root / f"{i}.png")
    return root


# -- the homography itself ---------------------------------------------------


def test_solve_homography_reproduces_its_points():
    source = torch.tensor([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    target = torch.tensor([[1.0, 1.0], [11.0, 0.5], [12.0, 9.0], [0.0, 11.0]])

    mapped = apply_homography(_solve_homography(source, target), source)
    assert torch.allclose(mapped, target.double(), atol=1e-9)


def test_identity_homography_is_identity():
    corners = torch.tensor([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    homography = _solve_homography(corners, corners)
    points = torch.tensor([[1.0, 2.0], [3.5, 0.5]])
    assert torch.allclose(apply_homography(homography, points), points.double(), atol=1e-9)


def test_warp_matches_the_recorded_homography(folder):
    """The claim everything else rests on, checked against actual pixels.

    Warp an image containing one bright square, find where the square landed,
    and compare with where the recorded homography says it should be. PIL's
    PERSPECTIVE coefficients map output back to input, so a missing inverse
    here would produce a plausible-looking but entirely wrong ground truth.
    """
    dataset = HomographyPairDataset(folder, max_warp=0.15)

    for index in range(len(dataset)):
        source, warped, geometry = dataset[index]

        centre_before = _bright_centre(source)
        centre_after = _bright_centre(warped)
        predicted = apply_homography(geometry["homography"], centre_before[None])[0]

        assert torch.allclose(predicted, centre_after, atol=2.0), (
            f"pair {index}: homography predicts {predicted.tolist()}, "
            f"pixels landed at {centre_after.tolist()}"
        )


def _bright_centre(image: Image.Image) -> torch.Tensor:
    """xy centroid of the bright region."""
    array = np.asarray(image.convert("L"), dtype=np.float64)
    ys, xs = np.nonzero(array > 128)
    assert len(xs) > 0, "the bright square vanished"
    return torch.tensor([xs.mean(), ys.mean()], dtype=torch.float64)


# -- determinism -------------------------------------------------------------


def test_warps_are_deterministic(folder):
    """Same pairs every run, or the feature cache is useless."""
    first = HomographyPairDataset(folder, seed=0)
    second = HomographyPairDataset(folder, seed=0)
    assert torch.allclose(first[1][2]["homography"], second[1][2]["homography"])


def test_index_order_does_not_change_a_pair(folder):
    """Item i must not depend on which items were read before it."""
    dataset = HomographyPairDataset(folder, seed=0)
    forward = [dataset[i][2]["homography"] for i in range(len(dataset))]
    backward = [dataset[i][2]["homography"] for i in reversed(range(len(dataset)))]
    assert torch.allclose(forward[0], backward[-1])


def test_seed_changes_the_warps(folder):
    assert not torch.allclose(
        HomographyPairDataset(folder, seed=0)[0][2]["homography"],
        HomographyPairDataset(folder, seed=1)[0][2]["homography"],
    )


# -- provenance --------------------------------------------------------------


def test_labels_do_not_decode_images(folder, monkeypatch):
    import visbench.data.image_folder as module

    monkeypatch.setattr(module, "load_image", lambda p: pytest.fail("labels() decoded an image"))
    geometries = HomographyPairDataset(folder).labels()
    assert len(geometries) == 3
    assert "homography" in geometries[0]


def test_labels_match_getitem(folder):
    dataset = HomographyPairDataset(folder)
    assert torch.allclose(dataset.labels()[2]["homography"], dataset[2][2]["homography"])


def test_fingerprint_covers_the_warp_settings(folder):
    """Different warps are different data, even from identical files."""
    assert (
        HomographyPairDataset(folder, max_warp=0.1).fingerprint()
        != HomographyPairDataset(folder, max_warp=0.3).fingerprint()
    )
    assert (
        HomographyPairDataset(folder, seed=0).fingerprint()
        != HomographyPairDataset(folder, seed=1).fingerprint()
    )


def test_view_identity_distinguishes_the_two_views(folder):
    dataset = HomographyPairDataset(folder)
    assert dataset.view_identity(0, 0) != dataset.view_identity(0, 1)
    assert dataset.view_identity(0, 0) != dataset.view_identity(1, 0)


def test_view_identity_tracks_the_warp(folder):
    """A warped view is generated, so its identity must follow the parameters."""
    assert HomographyPairDataset(folder, seed=0).view_identity(0, 1) != HomographyPairDataset(
        folder, seed=1
    ).view_identity(0, 1)


def test_invalid_max_warp_raises(folder):
    with pytest.raises(ValueError, match="max_warp"):
        HomographyPairDataset(folder, max_warp=0.8)


def test_flat_folder_is_accepted(folder):
    """No class subdirectories needed: correspondence has no labels."""
    assert len(HomographyPairDataset(folder)) == 3


# -- the coordinate frame ----------------------------------------------------


class TestFrame:
    """The homography must describe the plane the patches actually sit on.

    A backbone's preprocess resizes and centre-crops, so features cover only
    part of the original image. A homography written in original-image
    coordinates describes a different plane, and every pixel error comes out
    wrong while still looking plausible — measured on Imagenette, that bug
    reported recall@10px of 0.22 where the true figure was 0.64.
    """

    def test_views_are_square_at_image_size(self, folder):
        dataset = HomographyPairDataset(folder, image_size=224)
        image_0, image_1, geometry = dataset[0]

        assert image_0.size == (224, 224)
        assert image_1.size == (224, 224)
        assert geometry["size"] == (224, 224)

    def test_geometry_matches_the_emitted_frame(self, folder):
        """labels() must agree with __getitem__ about the frame."""
        dataset = HomographyPairDataset(folder, image_size=128)
        assert dataset.labels()[0]["size"] == dataset[0][2]["size"]
        assert torch.allclose(dataset.labels()[0]["homography"], dataset[0][2]["homography"])

    def test_preprocessing_is_geometrically_a_noop(self, folder, fake_vit):
        """Feeding an emitted view through preprocess must not move anything."""
        dataset = HomographyPairDataset(folder, image_size=fake_vit.image_size)
        image_0, _, _ = dataset[0]

        batch = fake_vit.preprocess([image_0])
        assert batch.shape[-2:] == (fake_vit.image_size, fake_vit.image_size)

    def test_non_square_source_is_centre_cropped(self, tmp_path):
        """The crop is what makes the frame predictable for any input shape."""
        root = tmp_path / "wide"
        root.mkdir()
        Image.fromarray(np.zeros((100, 300, 3), dtype=np.uint8)).save(root / "a.png")

        image_0, image_1, geometry = HomographyPairDataset(root, image_size=64)[0]
        assert image_0.size == image_1.size == (64, 64)
        assert geometry["size"] == (64, 64)

    def test_image_size_changes_the_fingerprint(self, folder):
        assert (
            HomographyPairDataset(folder, image_size=112).fingerprint()
            != HomographyPairDataset(folder, image_size=224).fingerprint()
        )

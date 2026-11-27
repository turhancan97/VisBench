"""Dense targets land on their images (#173).

Before #173 every dense target was resized with torch's left-aligned
``interpolate(mode="nearest")`` while its image went through PIL's
centre-aligned bicubic, so a target sat a sub-pixel from its image. The fix
resamples targets with PIL's own nearest-neighbour. These tests pin the three
things that make that true and safe: the PIL path *is* PIL's geometry; a
coordinate-encoded image and target agree under it and disagree under the old
path (the check CLAUDE.md asks for whenever a resize claims to be harmless); and
the old path stays reproducible and can never share a fingerprint with the new.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from visbench.data.dense import (
    DEFAULT_TARGET_RESAMPLE,
    LEGACY_TARGET_RESAMPLE,
    DenseFolderDataset,
    resample_target,
)

#: (height, width) of the frames the dense boards read: NYUv2, a VOC frame,
#: Taskonomy, and an awkward odd size.
FRAMES = [(480, 640), (375, 500), (512, 512), (333, 517)]


def _resized(height: int, width: int, size: int = 224) -> tuple[int, int]:
    scale = size / min(height, width)
    return max(size, round(height * scale)), max(size, round(width * scale))


@pytest.mark.parametrize(("height", "width"), FRAMES)
def test_the_default_is_exactly_pils_nearest_neighbour(height, width):
    labels = np.random.default_rng(0).integers(0, 40, (height, width)).astype(np.uint8)
    out_h, out_w = _resized(height, width)
    reference = np.asarray(Image.fromarray(labels).resize((out_w, out_h), Image.Resampling.NEAREST))
    ours = resample_target(torch.from_numpy(labels)[None].float(), (out_h, out_w))
    assert np.array_equal(ours[0].numpy().astype(np.uint8), reference)


@pytest.mark.parametrize(("height", "width"), FRAMES)
def test_a_coordinate_encoded_target_lands_on_its_image(height, width):
    """The image's bicubic resize of a column ramp says which source column each
    output pixel is centred on; a target on its image must agree to within the
    half-pixel nearest-neighbour rounding, and on average to well under that."""
    out_h, out_w = _resized(height, width)
    ramp = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    image_columns = np.asarray(
        Image.fromarray(ramp).resize((out_w, out_h), Image.Resampling.BICUBIC)
    )[:, 2:-2]  # bicubic overshoots at the border

    def offset(method: str) -> np.ndarray:
        target = resample_target(torch.from_numpy(ramp)[None], (out_h, out_w), method)
        return target[0].numpy()[:, 2:-2] - image_columns

    scale = width / out_w
    aligned = offset(DEFAULT_TARGET_RESAMPLE)
    assert np.abs(aligned).max() <= 0.5 * scale + 1e-3
    assert abs(aligned.mean()) < 0.05 * scale

    legacy = offset(LEGACY_TARGET_RESAMPLE)
    if scale != 1.0:
        # The old path is biased towards the left: a systematic, not a random,
        # sub-pixel offset -- which is what #173 was.
        assert legacy.mean() < -0.2 * scale


def test_holes_survive_the_resample():
    target = torch.ones(1, 8, 8)
    target[0, :, :4] = float("nan")
    out = resample_target(target, (4, 4))
    assert torch.isnan(out[0, :, :2]).all()
    assert (out[0, :, 2:] == 1).all()


def test_the_legacy_path_is_the_pre_173_call():
    target = torch.rand(3, 31, 47)
    expected = torch.nn.functional.interpolate(target[None], size=(16, 24), mode="nearest")[0]
    assert torch.equal(resample_target(target, (16, 24), LEGACY_TARGET_RESAMPLE), expected)


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError, match="target_resample"):
        resample_target(torch.zeros(1, 4, 4), (2, 2), "bilinear")


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "depths").mkdir()
    for index in range(3):
        Image.fromarray(np.full((40, 60, 3), 50 * index, dtype=np.uint8)).save(
            tmp_path / "images" / f"s{index}.png"
        )
        np.save(tmp_path / "depths" / f"s{index}.npy", np.full((40, 60), 1.0 + index, np.float32))
    return tmp_path


def test_the_two_geometries_never_share_a_fingerprint(folder):
    aligned = DenseFolderDataset(folder, image_size=32)
    legacy = DenseFolderDataset(folder, image_size=32, target_resample=LEGACY_TARGET_RESAMPLE)
    assert aligned.fingerprint() != legacy.fingerprint()
    assert aligned.describe()["target_resample"] == DEFAULT_TARGET_RESAMPLE
    # The legacy dataset describes itself exactly as every pre-#173 record did.
    assert "target_resample" not in legacy.describe()


def test_the_default_is_aligned(folder):
    assert DenseFolderDataset(folder, image_size=32).target_resample == "pil_nearest"

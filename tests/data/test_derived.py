"""The first target this library computes rather than reads.

A stored target is wrong loudly — a missing file raises. A *derived* target is
wrong quietly: it is always present, always the right shape, and always
plausible, so nothing downstream can tell a correct corner response from a
subtly broken one. Everything here exists because of that asymmetry.

The load-bearing case is `test_a_straight_edge_scores_exactly_zero`. Shi-Tomasi
cornerness is the *smaller* eigenvalue of the structure tensor, so a straight
edge — one direction of variation — has to give exactly 0.0, not merely less
than a corner. Any of the plausible mistakes (taking the larger eigenvalue, the
trace, the determinant, the gradient magnitude) still peaks at corners and would
pass a test that only ranked corners above edges.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from visbench.data.dense import DenseFolderDataset
from visbench.data.derived import DerivedTargetDataset, ShiTomasiResponse, structure_tensor


def square_image(size: int = 64, lo: int = 0, hi: int = 255) -> Image.Image:
    """A light square on a dark ground: four corners, four straight edges."""
    array = np.full((size, size), lo, dtype=np.uint8)
    array[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = hi
    return Image.fromarray(array).convert("RGB")


def write_images(directory, count: int = 4, size: int = 64) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for index in range(count):
        array = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        Image.fromarray(array).save(directory / f"img_{index:03d}.png")


# -- the operator -------------------------------------------------------------


def test_a_straight_edge_scores_exactly_zero():
    """The claim that separates this operator from every plausible substitute.

    lambda_min is the variation in the *weaker* direction, and a straight edge
    has none, so this is exact rather than approximate. The larger eigenvalue,
    the trace, the determinant and plain gradient magnitude are all large on an
    edge; each would still rank corners highest and pass a weaker test.
    """
    response = ShiTomasiResponse(transform="none", scale=1.0)(square_image())
    array = response.numpy()

    def local_max(y: int, x: int, radius: int = 3) -> float:
        window = array[y - radius : y + radius + 1, x - radius : x + radius + 1]
        return float(window.max())

    # Mid-edge, on the contour but away from any corner.
    for y, x in ((16, 32), (47, 32), (32, 16), (32, 47)):
        assert local_max(y, x) == 0.0

    for y, x in ((16, 16), (16, 47), (47, 16), (47, 47)):
        assert local_max(y, x) > 0.0


def test_a_flat_image_scores_exactly_zero_including_at_the_border():
    """This is the padding test, and zero padding is the failure it catches.

    Padding with zeros manufactures an intensity step around the whole frame.
    For most targets that is a cosmetic border artifact; for this one a step is
    precisely what the operator responds to, so the outer ring would carry a
    fabricated response that the probe would be trained and scored on. With
    replicate padding a uniform image is uniformly zero, border included.
    """
    flat = Image.fromarray(np.full((32, 32), 128, dtype=np.uint8)).convert("RGB")
    response = ShiTomasiResponse(transform="none", scale=1.0)(flat)

    assert torch.all(response == 0.0)


def test_the_four_corners_of_a_square_score_equally():
    """A gradient kernel applied with the wrong transpose is a rotation away
    from correct, and asymmetric corner scores are the symptom."""
    array = ShiTomasiResponse(transform="none", scale=1.0)(square_image()).numpy()
    peaks = [
        float(array[y - 3 : y + 4, x - 3 : x + 4].max())
        for y, x in ((16, 16), (16, 47), (47, 16), (47, 47))
    ]
    assert max(peaks) - min(peaks) < 1e-6 * max(peaks)


def test_the_structure_tensor_matches_a_hand_computed_gradient():
    """The operator is checked against arithmetic, not only against its own
    probe's score. An operator whose only test is a downstream number is an
    operator nobody has checked."""
    # A horizontal ramp: dI/dx constant, dI/dy zero.
    ramp = torch.arange(16, dtype=torch.float32).repeat(16, 1).view(1, 1, 16, 16)
    ixx, ixy, iyy = structure_tensor(ramp, sigma=1.0)

    interior = (slice(None), slice(None), slice(4, 12), slice(4, 12))
    assert torch.allclose(ixx[interior], torch.ones_like(ixx[interior]), atol=1e-5)
    assert torch.allclose(ixy[interior], torch.zeros_like(ixy[interior]), atol=1e-6)
    assert torch.allclose(iyy[interior], torch.zeros_like(iyy[interior]), atol=1e-6)


def test_the_response_is_deterministic():
    """A benchmark target that varied between calls would make every number
    unreproducible while looking entirely normal."""
    generator = ShiTomasiResponse()
    image = square_image()
    assert torch.equal(generator(image), generator(image))


@pytest.mark.parametrize(
    ("field", "value"),
    [("sigma", 1.0), ("scale", 1e3), ("transform", "none")],
)
def test_changing_a_parameter_changes_the_target(field, value):
    """A parameter that is recorded and does nothing is the QuickGELU failure.

    Step 6d-1 shipped exactly that: `target_scale` was accepted, described and
    fingerprinted while having no effect, and the sweep that should have found
    it returned four identical rows instead of raising.
    """
    image = square_image()
    default = ShiTomasiResponse()(image)
    changed = ShiTomasiResponse(**{field: value})(image)
    assert not torch.equal(default, changed)


@pytest.mark.parametrize(
    ("field", "value"),
    [("sigma", 0.0), ("sigma", -1.0), ("scale", 0.0), ("transform", "sqrt")],
)
def test_an_invalid_generator_setting_raises(field, value):
    with pytest.raises(ValueError):
        ShiTomasiResponse(**{field: value})


def test_describe_names_everything_that_moves_the_target():
    """A derived target has no file to point at, so `describe()` is its whole
    identity. The fixed choices are named too: if the gradient kernel ever
    changed, a record omitting it would silently mean something else."""
    described = ShiTomasiResponse().describe()
    for key in (
        "target_operator",
        "target_sigma",
        "target_transform",
        "target_scale",
        "target_gradient",
        "target_luma",
    ):
        assert key in described
    assert described["target_operator"] == "shi_tomasi"


def test_two_generators_that_differ_have_different_tokens():
    assert ShiTomasiResponse().token() != ShiTomasiResponse(sigma=1.0).token()
    assert ShiTomasiResponse().token() == ShiTomasiResponse().token()


# -- the dataset --------------------------------------------------------------


def test_the_crop_agrees_with_the_dense_dataset_pixel_for_pixel(tmp_path):
    """A corner number and an edge number over the same frames must differ in
    the operator and nothing else.

    Step 6c-1 made the same guarantee for the detection dataset's crop. The
    image is deliberately non-square, which is where a missed rescale shows.
    """
    images = tmp_path / "images"
    targets = tmp_path / "depths"
    images.mkdir()
    targets.mkdir()
    rng = np.random.default_rng(1)
    array = rng.integers(0, 255, (90, 140, 3), dtype=np.uint8)
    Image.fromarray(array).save(images / "a.png")
    np.save(targets / "a.npy", rng.random((90, 140)).astype(np.float32))

    derived = DerivedTargetDataset(root=tmp_path, image_size=32)
    dense = DenseFolderDataset(root=tmp_path, image_size=32)

    assert np.array_equal(np.asarray(derived[0][0]), np.asarray(dense[0][0]))


def test_the_target_is_generated_at_the_working_resolution(tmp_path):
    """No resize and no resampling of the response: it is computed from the
    already-cropped image, which is what removes the alignment hazard entirely
    rather than testing for its absence."""
    write_images(tmp_path / "images", count=2, size=90)
    dataset = DerivedTargetDataset(root=tmp_path, image_size=32)

    image, target = dataset[0]
    assert image.size == (32, 32)
    assert target.shape == (32, 32)


def test_the_target_matches_the_image_the_backbone_sees(tmp_path):
    """The pairing claim, stated as an equality rather than trusted.

    `__getitem__` must return a target generated from *that* image, not from
    the file it came from — the difference is the crop, and it is invisible in
    the shapes.
    """
    write_images(tmp_path / "images", count=1, size=90)
    dataset = DerivedTargetDataset(root=tmp_path, image_size=32)

    image, target = dataset[0]
    assert torch.equal(target, dataset.generator(image))


def test_the_fingerprint_follows_the_generator(tmp_path):
    """Two operators over one folder are two different sets of targets, and two
    records that collided would be indistinguishable."""
    write_images(tmp_path / "images", count=3)
    default = DerivedTargetDataset(root=tmp_path)
    other = DerivedTargetDataset(root=tmp_path, generator=ShiTomasiResponse(sigma=1.0))

    assert default.fingerprint() != other.fingerprint()


def test_the_cache_identity_does_not_follow_the_generator(tmp_path):
    """Cached *features* depend on the image alone.

    Changing the corner operator must not invalidate an extraction that is
    still perfectly valid — the same reasoning that keeps the target out of
    `DenseFolderDataset.cache_identity`.
    """
    write_images(tmp_path / "images", count=2)
    default = DerivedTargetDataset(root=tmp_path)
    other = DerivedTargetDataset(root=tmp_path, generator=ShiTomasiResponse(sigma=1.0))

    assert default.cache_identity(0) == other.cache_identity(0)


def test_describe_carries_the_generator_into_dataset_params(tmp_path):
    """This is what puts two sigmas into two comparability groups without
    anyone having to notice — the mechanism step 6f used for threshold units."""
    write_images(tmp_path / "images", count=2)
    described = DerivedTargetDataset(root=tmp_path).describe()

    assert described["target_operator"] == "shi_tomasi"
    assert described["target_sigma"] == 2.0
    assert described["image_size"] == 224


def test_subset_reindexes_and_moves_the_fingerprint(tmp_path):
    write_images(tmp_path / "images", count=6)
    dataset = DerivedTargetDataset(root=tmp_path)
    short = dataset.subset(2)

    assert len(short) == 2
    assert short.stems == dataset.stems[:2]
    assert short.fingerprint() != dataset.fingerprint()
    assert len(dataset) == 6


def test_max_images_takes_a_prefix(tmp_path):
    write_images(tmp_path / "images", count=5)
    assert len(DerivedTargetDataset(root=tmp_path, max_images=3)) == 3


def test_a_missing_image_directory_raises(tmp_path):
    (tmp_path / "elsewhere").mkdir()
    with pytest.raises(NotADirectoryError):
        DerivedTargetDataset(root=tmp_path)


def test_an_empty_image_directory_raises(tmp_path):
    (tmp_path / "images").mkdir()
    with pytest.raises(ValueError, match="No images"):
        DerivedTargetDataset(root=tmp_path)


def test_a_stem_named_but_absent_raises(tmp_path):
    write_images(tmp_path / "images", count=2)
    with pytest.raises(ValueError, match="absent"):
        DerivedTargetDataset(root=tmp_path, stems=["img_000", "nope"])

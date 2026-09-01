"""The oracle gate — the gauntlet check that asks whether a target is *recoverable*.

Every other check a new derived target passes is about the target's own
distribution, or about its relation to a target that already ships. None of them
asks whether a dense probe could recover it *at all*, and photometric
superpixels is what that cost: it passed the tail check, passed the overlap
check, was built in full, and then scored 0.02-0.04 because a 1px SLIC boundary
in a flat region is placed by the seeding lattice, which is not in a 14px patch
token in principle.

`evaluate_oracle` asks it directly and needs no backbone, no features and no
fitted head — the whole point being that it costs one pass over a split rather
than a board. So it is testable in the *fast* suite, which is where a check that
guards a silently wrong decision belongs.

The two anchor cases below are the whole idea in miniature: a target that is
constant within each patch survives the grid exactly, and a target that flips
every pixel does not survive it at all.
"""

import pytest
import torch

from visbench.tasks.dense_base import DenseTrainingTask, pool_to_grid
from visbench.tasks.high_level import SemanticSegmentationTask
from visbench.tasks.low_level import CornerTask, EdgeTask, Keypoint2DTask, OrientationTask
from visbench.tasks.mid_level import DepthTask, OcclusionEdgeTask
from visbench.utils import set_seed

MAGNITUDE_PROBES = (EdgeTask, Keypoint2DTask, OcclusionEdgeTask, CornerTask)

SIZE = 32
GRID = 8


def smooth_ramp(batch: int = 3, size: int = SIZE) -> torch.Tensor:
    """A target with nothing in it finer than a patch — recoverable in full."""
    axis = torch.linspace(0.0, 1.0, size)
    ramp = axis[:, None] + axis[None, :]
    return ramp.expand(batch, 1, size, size).clone()


def grid_constant(batch: int = 3, grid: int = GRID, size: int = SIZE) -> torch.Tensor:
    """A target that is constant within each cell of a ``grid`` x ``grid`` grid."""
    set_seed(0)
    cells = torch.rand(batch, 1, grid, grid)
    return cells.repeat_interleave(size // grid, dim=2).repeat_interleave(size // grid, dim=3)


def pixel_checkerboard(batch: int = 3, size: int = SIZE) -> torch.Tensor:
    """A target whose every pixel differs from its neighbours — pure sub-patch signal."""
    row = torch.arange(size)
    board = ((row[:, None] + row[None, :]) % 2).float()
    return board.expand(batch, 1, size, size).clone()


# -- pool_to_grid -------------------------------------------------------------


def test_pool_to_grid_is_the_cell_mean():
    """The bottleneck is an area average, and a divisible grid makes that exact."""
    targets = torch.arange(64, dtype=torch.float32).view(1, 1, 8, 8)
    pooled = pool_to_grid(targets, (2, 2))
    assert pooled.shape == (1, 1, 2, 2)
    assert pooled[0, 0, 0, 0] == pytest.approx(float(targets[0, 0, :4, :4].mean()))
    assert pooled[0, 0, 1, 1] == pytest.approx(float(targets[0, 0, 4:, 4:].mean()))


def test_pooling_excludes_invalid_pixels_rather_than_averaging_them_in():
    """A hole must not drag its whole cell down, or the oracle scores a different split.

    `NaN` is the magnitude probes' invalid marker, and a plain mean over a cell
    holding one of them returns `NaN` for the cell — which would shrink the
    population the oracle covers relative to the one `evaluate` covers, and the
    two numbers would stop qualifying each other.
    """
    targets = torch.full((1, 1, 4, 4), 2.0)
    targets[0, 0, 0, 0] = float("nan")

    pooled = pool_to_grid(targets, (2, 2))
    assert torch.isfinite(pooled).all()
    assert pooled[0, 0, 0, 0] == pytest.approx(2.0)


def test_a_cell_with_no_valid_pixel_stays_invalid():
    """It carries the marker its own pixels carried — it is not filled with a fiction."""
    targets = torch.full((1, 1, 4, 4), 2.0)
    targets[0, 0, :2, :2] = float("nan")

    pooled = pool_to_grid(targets, (2, 2))
    assert torch.isnan(pooled[0, 0, 0, 0])
    assert torch.isfinite(pooled[0, 0, 1, 1])


def test_pooling_refuses_a_grid_finer_than_the_target():
    """A finer grid is not a bound on anything, so it is a mistake and not a no-op."""
    with pytest.raises(ValueError, match="finer than the target"):
        pool_to_grid(torch.zeros(1, 1, 4, 4), (8, 8))


# -- the two anchors ----------------------------------------------------------


@pytest.mark.parametrize("probe_class", MAGNITUDE_PROBES)
def test_a_target_with_no_sub_patch_structure_survives_the_grid(probe_class):
    """The upper anchor: nothing about this target is finer than a patch."""
    probe = probe_class()
    metrics = probe.evaluate_oracle(smooth_ramp(), GRID, batch_size=2)
    assert metrics[probe.correlation_key] > 0.99


def test_the_oracle_faces_the_head_s_own_interpolation_and_not_a_kinder_one():
    """A patch-constant target does *not* score 1.0, and that is the design.

    `LinearHead` is a 1x1 convolution on the grid followed by a **bilinear**
    upsample, so a real head cannot emit a piecewise-constant map however
    perfect its features are — the block edges get smoothed. The oracle
    upsamples the same way and inherits the same limit, landing around 0.88 on
    a target made of hard grid cells.

    Substituting a nearest upsample here would raise every oracle and make the
    gate more permissive than the heads it is protecting, which is the wrong
    direction for a check that exists to reject things.
    """
    scored = EdgeTask().evaluate_oracle(grid_constant(), GRID)["edge_correlation"]
    assert 0.8 < scored < 0.95


@pytest.mark.parametrize("probe_class", MAGNITUDE_PROBES)
def test_a_target_that_flips_every_pixel_does_not_survive_it(probe_class):
    """The lower anchor, and the finding the gate exists to produce.

    Every patch mean is identical, so the pooled target carries *no* information
    about the checkerboard. This is the superpixel case in its pure form: the
    signal is entirely below the grid, and no backbone can be ranked on it.
    """
    probe = probe_class()
    metrics = probe.evaluate_oracle(pixel_checkerboard(), GRID, batch_size=2)
    assert metrics[probe.correlation_key] == pytest.approx(0.0, abs=1e-5)


def test_the_oracle_is_the_target_itself_when_the_grid_is_the_image():
    """A sanity bound in the other direction: no bottleneck, no loss."""
    set_seed(0)
    probe = EdgeTask()
    targets = torch.rand(2, 1, SIZE, SIZE)
    metrics = probe.evaluate_oracle(targets, SIZE)
    assert metrics["edge_correlation"] == pytest.approx(1.0, abs=1e-4)


def test_a_coarser_grid_never_scores_higher():
    """Monotonic in the grid, which is what makes it a statement about resolution."""
    probe = EdgeTask()
    set_seed(1)
    targets = torch.rand(3, 1, SIZE, SIZE)
    fine = probe.evaluate_oracle(targets, 16)["edge_correlation"]
    coarse = probe.evaluate_oracle(targets, 4)["edge_correlation"]
    assert coarse < fine


# -- what it does and does not need -------------------------------------------


def test_the_oracle_needs_no_fitted_head():
    """The whole cost argument: a candidate target is measured before a probe exists.

    `evaluate` on the same probe raises, which is the contrast worth pinning —
    if the oracle ever started needing a head it would stop being a *pre*-check
    and nothing else would notice.
    """
    probe = EdgeTask()
    assert probe.head is None
    probe.evaluate_oracle(grid_constant(), GRID)

    with pytest.raises(RuntimeError, match="not been fitted"):
        probe.evaluate(torch.rand(2, 4, GRID, GRID), torch.rand(2, 1, SIZE, SIZE))


def test_targets_arrive_as_a_tensor_a_sequence_or_a_dataset():
    """A candidate is usually measured straight off the dataset that generates it."""
    probe = EdgeTask()
    targets = grid_constant(batch=4)
    expected = probe.evaluate_oracle(targets, GRID)["edge_correlation"]

    as_list = [targets[i, 0] for i in range(len(targets))]
    assert probe.evaluate_oracle(as_list, GRID)["edge_correlation"] == pytest.approx(expected)

    class _Dataset:
        def __len__(self) -> int:
            return len(targets)

        def target(self, index: int) -> torch.Tensor:
            return targets[index, 0]

    assert probe.evaluate_oracle(_Dataset(), GRID)["edge_correlation"] == pytest.approx(expected)


def test_the_batch_size_does_not_move_the_number():
    """Per-image metrics weighted by batch size, exactly as `evaluate` averages.

    Shared code rather than a parallel copy, so this pins that the sharing holds:
    a ceiling averaged over a different population from its score is the mistake
    `CorrespondenceTask.evaluate_ceiling` records having made once.
    """
    set_seed(2)
    probe = EdgeTask()
    targets = torch.rand(6, 1, SIZE, SIZE)
    one = probe.evaluate_oracle(targets, GRID, batch_size=1)
    many = probe.evaluate_oracle(targets, GRID, batch_size=6)
    assert one["edge_correlation"] == pytest.approx(many["edge_correlation"], abs=1e-6)


def test_a_hole_does_not_poison_the_pixels_around_it():
    """An all-invalid cell is filled before upsampling, or bilinear spreads the NaN.

    The metric masks the hole's own pixels either way; what must not happen is
    the whole frame's correlation coming back `NaN` because a neighbouring cell
    borrowed from it.
    """
    targets = smooth_ramp(batch=2)
    targets[:, :, : SIZE // GRID, : SIZE // GRID] = float("nan")

    metrics = OcclusionEdgeTask().evaluate_oracle(targets, GRID)
    assert torch.isfinite(torch.tensor(metrics["occlusion_edge_correlation"]))
    assert metrics["occlusion_edge_correlation"] > 0.99


def test_an_empty_target_set_is_refused():
    with pytest.raises(ValueError, match="empty target set"):
        EdgeTask().evaluate_oracle(torch.zeros(0, 1, SIZE, SIZE), GRID)


# -- who may have one ---------------------------------------------------------


@pytest.mark.parametrize("probe_class", (SemanticSegmentationTask, DepthTask))
def test_a_probe_whose_target_does_not_average_declares_no_oracle(probe_class):
    """Listed opt-in, no fallback — the posture `TARGET_STYLES` and `METRIC_DIRECTIONS` take.

    Mean-pooling a class-index map is meaningless (the mean of classes 1 and 15
    is class 8) and a bin-expectation depth target is not the quantity its head
    emits. A silently-defaulting oracle would return a confident number about
    nothing, which is worse here than no number: this gate exists to *stop* a
    probe being built.
    """
    probe = probe_class(num_classes=3) if probe_class is SemanticSegmentationTask else probe_class()
    with pytest.raises(NotImplementedError, match="declares no oracle"):
        probe.evaluate_oracle(torch.zeros(2, 1, SIZE, SIZE), GRID)


def test_the_refusal_names_the_way_to_opt_in():
    """A contributor adding a fifth magnitude probe should not have to read this file."""
    with pytest.raises(NotImplementedError, match="_averaging_oracle"):
        DepthTask().evaluate_oracle(torch.zeros(1, 1, SIZE, SIZE), GRID)


# -- orientation, the vector case ---------------------------------------------


def test_the_orientation_oracle_is_a_unit_field():
    """It must be what the head emits, which `_activate` normalises."""
    probe = OrientationTask()
    set_seed(3)
    targets = torch.randn(2, 2, SIZE, SIZE)
    prediction = probe.oracle_prediction(targets, (GRID, GRID))
    assert prediction.shape == targets.shape
    assert prediction.norm(dim=1).allclose(torch.ones(2, SIZE, SIZE), atol=1e-5)


def test_a_single_orientation_survives_the_grid_exactly():
    """No structure below a patch, so a perfect backbone recovers it perfectly."""
    targets = torch.zeros(2, 2, SIZE, SIZE)
    targets[:, 0] = 1.0
    metrics = OrientationTask().evaluate_oracle(targets, GRID)
    assert metrics["orientation_error"] == pytest.approx(0.0, abs=1e-3)


def test_orientations_that_cancel_within_a_patch_are_not_recoverable():
    """The vector average is a *resultant*: opposed directions inside one cell cancel.

    That is the honest answer rather than a defect — a patch with no dominant
    orientation has none to report, and the coherence weighting is what keeps
    the cost of saying so proportionate.
    """
    columns = torch.arange(SIZE)
    alternating = torch.zeros(2, 2, SIZE, SIZE)
    # (cos 2t, sin 2t) flipping between +x and -x every column: mean is zero.
    alternating[:, 0] = torch.where(columns % 2 == 0, 1.0, -1.0).expand(SIZE, SIZE)

    metrics = OrientationTask().evaluate_oracle(alternating, GRID)
    assert metrics["orientation_error"] > 40.0


# -- the interface ------------------------------------------------------------


def test_every_dense_probe_answers_the_oracle_question_one_way_or_the_other():
    """Either it implements one or it refuses by name; neither may be an AttributeError."""
    for probe in (EdgeTask(), OrientationTask(), DepthTask()):
        assert callable(probe.oracle_prediction)
        assert isinstance(probe, DenseTrainingTask)

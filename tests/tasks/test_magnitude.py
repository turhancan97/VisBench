"""The shared magnitude probe, and the masking that separates its three targets.

`tests/tasks/test_edge.py` already carries the unmasked half — the identity
activation, the correlation-over-RMSE argument, and an end-to-end run. What is
here is what step 6d-2 added around it.

**The fourth validity convention.** Depth marks holes 0, normals mark them with
a zero-length vector, label maps mark them negative. A magnitude map has no
value it can spare: 0 means "no edge here", a real reading. So the two
reconstruction-derived magnitude targets mark holes `NaN`, and the loss and the
metric must mask on exactly the same predicate — a probe optimised over pixels
it is not scored on is a probe whose training loss and score describe different
splits.

**Three probes, one implementation.** That is the honest description, and it is
also the hazard: the only thing keeping an occlusion-edge number from being
pooled with a texture-edge number is that they carry different metric keys and
different protocol strings. Those are asserted here rather than assumed.
"""

import pytest
import torch

import visbench
from visbench.metrics.dense import edge_metrics, magnitude_metrics
from visbench.tasks.low_level import EdgeTask, Keypoint2DTask
from visbench.tasks.magnitude_base import DenseMagnitudeTask
from visbench.tasks.mid_level import OcclusionEdgeTask
from visbench.utils import set_seed

MAGNITUDE_PROBES = (EdgeTask, Keypoint2DTask, OcclusionEdgeTask)


# -- masking in the metric ----------------------------------------------------


def test_masking_reduces_to_the_unmasked_computation():
    """The image-derived probes must pay nothing for a mask they never use.

    If this drifted, every published `edge_correlation` would move without the
    edge probe changing at all.
    """
    set_seed(0)
    pred, target = torch.rand(4, 1, 12, 12), torch.rand(4, 1, 12, 12)
    masked = magnitude_metrics(pred, target, correlation_key="edge_correlation")
    assert masked == pytest.approx(edge_metrics(pred, target))


def test_masked_pixels_are_scored_as_if_absent():
    """The strong form: masking half a frame equals scoring only the other half.

    A weaker test — "the number changes when I mask" — passes for an
    implementation that merely mixes fabricated zeros in at a different weight.
    """
    set_seed(0)
    pred, target = torch.rand(3, 1, 16, 16), torch.rand(3, 1, 16, 16)
    holed = target.clone()
    holed[:, :, 8:, :] = float("nan")

    scored = magnitude_metrics(pred, holed, correlation_key="c")
    cropped = magnitude_metrics(pred[:, :, :8, :], target[:, :, :8, :], correlation_key="c")
    for key in scored:
        assert scored[key] == pytest.approx(cropped[key], abs=1e-6)


def test_a_fully_invalid_image_scores_zero_rather_than_nan():
    """It has no structure to recover, and a NaN would poison the split mean.

    Scoring 0 keeps every image weighted equally, which is what lets
    `DenseTrainingTask.evaluate` recover the split number from batch means.
    """
    scores = magnitude_metrics(
        torch.rand(2, 1, 8, 8), torch.full((2, 1, 8, 8), float("nan")), correlation_key="c"
    )
    assert scores["c"] == 0.0
    assert all(value == value for value in scores.values())  # no NaN


# -- masking in the loss ------------------------------------------------------


def test_the_loss_ignores_invalid_pixels_entirely():
    """Loss and metric must drop the same pixels, or the probe is misoptimised."""
    set_seed(0)
    pred, target = torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)
    holed = target.clone()
    holed[:, :, 4:, :] = float("nan")

    task = OcclusionEdgeTask()
    masked = task._loss(pred, holed)
    cropped = task._loss(pred[:, :, :4, :], target[:, :, :4, :])
    assert torch.isfinite(masked)
    assert masked.item() == pytest.approx(cropped.item(), abs=1e-6)


def test_an_unmasked_loss_would_have_gone_nan():
    """Why NaN was chosen over an in-band sentinel: it fails loudly.

    A fabricated 0 in a hole trains quietly and merely scores badly, which is
    indistinguishable from a weak backbone. This pins that the marker really is
    the kind of value that cannot pass unnoticed.
    """
    pred = torch.rand(1, 1, 4, 4)
    holed = torch.rand(1, 1, 4, 4)
    holed[:, :, 2:, :] = float("nan")
    assert torch.isnan(torch.nn.functional.l1_loss(pred, holed))
    assert torch.isfinite(OcclusionEdgeTask()._loss(pred, holed))


def test_the_loss_keeps_a_gradient_when_every_pixel_is_invalid():
    """A whole batch of holes must be a no-op step, not a broken graph."""
    pred = torch.rand(1, 1, 4, 4, requires_grad=True)
    loss = OcclusionEdgeTask()._loss(pred, torch.full((1, 1, 4, 4), float("nan")))
    loss.backward()
    assert loss.item() == 0.0
    assert pred.grad is not None and torch.equal(pred.grad, torch.zeros_like(pred))


# -- the three probes ---------------------------------------------------------


@pytest.mark.parametrize("task_class", MAGNITUDE_PROBES)
def test_every_magnitude_probe_shares_the_base(task_class):
    assert issubclass(task_class, DenseMagnitudeTask)
    assert task_class().out_channels == 1


def test_the_three_probes_cannot_have_their_numbers_pooled():
    """The only thing separating three identical implementations.

    Same activation, same loss, same metric, three targets that are not
    comparable. Distinct metric keys and distinct protocol strings are what
    stop a leaderboard averaging them.
    """
    keys = {task_class.correlation_key for task_class in MAGNITUDE_PROBES}
    protocols = {task_class.protocol for task_class in MAGNITUDE_PROBES}
    assert len(keys) == len(MAGNITUDE_PROBES)
    assert len(protocols) == len(MAGNITUDE_PROBES)
    assert not any("probe3d" in protocol or "bsds" in protocol for protocol in protocols)


@pytest.mark.parametrize(
    ("name", "level"),
    [("edge", "low_level"), ("keypoints2d", "low_level"), ("occlusion_edge", "mid_level")],
)
def test_each_probe_is_registered_at_its_level(name, level):
    """The occlusion-edge probe is mid-level because recovering it needs geometry.

    Its implementation is identical to the low-level one, so the level attribute
    is the only thing placing either in the taxonomy.
    """
    assert name in visbench.list_probes()
    assert visbench.get_probe(name).level == level


@pytest.mark.parametrize("task_class", MAGNITUDE_PROBES)
def test_the_reported_metric_key_matches_the_probe(task_class):
    """A record whose key said `edge_correlation` for a keypoint target would
    be readable, plausible and wrong."""
    task = task_class()
    scores = task._batch_metrics(torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8))
    assert task.correlation_key in scores
    assert task.describe()["task_params"]["protocol"] == task.protocol


def test_a_masked_probe_recovers_the_signal_it_can_see():
    """End to end with holes: the whole path has to agree about which pixels exist.

    The features encode the answer, so the ceiling is ~1. Half of every target
    is `NaN`. If the loss trained against fabricated values, or the metric
    scored them, this could not converge — the same argument as the edge
    probe's end-to-end test, with the mask added.
    """
    set_seed(0)
    size, grid, count = 16, 8, 12

    coarse = torch.rand(count, 1, grid, grid) * 0.13
    targets = torch.nn.functional.interpolate(coarse, size=size, mode="bilinear")[:, 0]
    targets[:, size // 2 :, :] = float("nan")
    features = torch.cat([coarse, torch.rand(count, 3, grid, grid) * 0.01], dim=1)

    task = OcclusionEdgeTask(epochs=60, lr=5e-2, batch_size=4)
    task.fit({"dense": features}, targets)
    scores = task.evaluate({"dense": features}, targets)

    assert scores["occlusion_edge_correlation"] > 0.9
    assert task.train_loss is not None and task.train_loss < 0.02

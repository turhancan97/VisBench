"""Edge detection — the metric's conventions and the probe end to end.

Two things here are deliberately unlike the three dense tasks that came before,
and both would pass a careless test suite while being wrong.

**Nothing is masked.** Depth holes and zero-length normals mean "no ground
truth"; an edge map's zeros mean "no edge", a real reading covering most of a
frame. A metric that reused the earlier convention would score the probe only
where an edge already is.

**Correlation leads, not RMSE.** Edge magnitude sits near zero almost
everywhere, so a probe that ignores its input and predicts the split mean gets a
small RMSE. `test_a_constant_prediction_scores_zero` is the guard that makes
that failure visible rather than flattering.
"""

import pytest
import torch

import visbench
from visbench.metrics.dense import edge_metrics
from visbench.tasks.low_level import EdgeTask
from visbench.utils import set_seed

# -- edge_metrics ------------------------------------------------------------


def test_a_perfect_prediction_scores_one():
    target = torch.rand(3, 1, 16, 16)
    scores = edge_metrics(target.clone(), target)
    assert scores["edge_correlation"] == pytest.approx(1.0, abs=1e-5)
    assert scores["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert scores["mae"] == pytest.approx(0.0, abs=1e-6)


def test_a_constant_prediction_scores_zero():
    """The failure the headline metric exists to catch.

    A frame's mean edge magnitude is about 0.011 of the container range, so a
    probe emitting that constant everywhere has a small RMSE and has learned
    nothing. Correlation is 0 for it by construction.
    """
    target = torch.rand(3, 1, 16, 16) * 0.13
    scores = edge_metrics(torch.full_like(target, 0.011), target)
    assert scores["edge_correlation"] == 0.0
    assert scores["rmse"] < 0.1  # small, which is exactly the point


def test_an_inverted_prediction_scores_minus_one():
    """Correlation is signed, so predicting edges where there are none is visible."""
    target = torch.rand(2, 1, 8, 8)
    assert edge_metrics(-target, target)["edge_correlation"] == pytest.approx(-1.0, abs=1e-5)


def test_correlation_is_invariant_to_scale_and_offset():
    """The property that makes it the right headline, stated as a test.

    A prediction with the right structure at the wrong magnitude scores 1.0 —
    which is why rmse and mae are reported next to it rather than instead.
    """
    target = torch.rand(2, 1, 12, 12)
    scores = edge_metrics(3.0 * target + 5.0, target)
    assert scores["edge_correlation"] == pytest.approx(1.0, abs=1e-5)
    assert scores["rmse"] > 1.0


def test_zeros_are_scored_rather_than_masked():
    """The convention that differs from depth and normals.

    Here the target is zero everywhere except one pixel. Under depth's rule all
    but that pixel would be dropped and the prediction would look perfect; under
    this one the wrong zeros count against it.
    """
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 3, 3] = 1.0
    wrong = torch.zeros(1, 1, 8, 8)
    wrong[0, 0, 5, 5] = 1.0

    assert edge_metrics(wrong, target)["mae"] > 0.0
    assert edge_metrics(wrong, target)["edge_correlation"] < 0.0


def test_metrics_are_per_image_then_averaged():
    """The codebase-wide rule, and it is observable here.

    One image scores 1.0 and one scores 0.0, so the per-image mean is 0.5. A
    metric that pooled both images' pixels into one correlation would not
    generally land there.
    """
    good = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    target = torch.stack([good, good])[:, None]
    pred = torch.stack([good, torch.full_like(good, 0.5)])[:, None]

    assert edge_metrics(pred, target)["edge_correlation"] == pytest.approx(0.5, abs=1e-5)


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        edge_metrics(torch.rand(2, 1, 8, 8), torch.rand(2, 1, 4, 4))


# -- EdgeTask ----------------------------------------------------------------


def test_the_probe_is_registered_as_low_level():
    """The first entry in a folder that was a placeholder from v0.1."""
    assert "edge" in visbench.list_probes()
    assert visbench.get_probe("edge").level == "low_level"


def test_the_activation_is_the_identity_and_does_not_rectify():
    """Non-negativity is learned here, not imposed, and that was measured.

    Rectifying looks obviously right — an edge magnitude cannot be negative —
    and destroys the probe, because the target sits in a narrow band just above
    zero. On features that encode the answer, ReLU scores 0.0000 (dead, zero
    prediction variance) and softplus -0.9851 (collapsed to a constant), against
    0.9997 for the identity. See `_activate`.

    Pinned as a test because reinstating a rectifier is the natural "tidy-up",
    and its cost shows up only as a mediocre score.
    """
    raw = torch.linspace(-50, 50, 101).reshape(1, 1, 101, 1)
    assert torch.equal(EdgeTask()._activate(raw), raw)


def test_the_loss_refuses_a_shape_mismatch():
    with pytest.raises(ValueError, match="must match"):
        EdgeTask()._loss(torch.rand(2, 1, 8, 8), torch.rand(2, 1, 4, 4))


def test_the_protocol_claims_neither_probe3d_nor_bsds():
    """The field exists to say what a number may be compared with.

    probe3d has no edge task, and BSDS's ODS/OIS/AP is a correspondence metric
    this does not implement. Claiming either would be worse than no record.
    """
    params = EdgeTask().describe()["task_params"]
    assert params["protocol"] == "visbench_edge_regression"
    assert "bsds" not in params["protocol"].lower()


def test_a_frozen_probe_reports_no_finetune_record():
    """Frozen and fine-tuned are different measurements; this keeps them apart."""
    assert EdgeTask().finetune() is None
    assert EdgeTask(finetune_blocks=2).finetune()["blocks"] == 2


def test_a_probe_recovers_edges_the_features_encode():
    """End to end: activation, L1 loss and the metric must describe one quantity.

    The features here literally contain the answer in one channel, so a linear
    head can reach it. If `_activate` and `_loss` disagreed about what the model
    predicts, or the metric read a different scale, this could not converge —
    the same argument that makes detection's 1.0-mAP test worth having.
    """
    set_seed(0)
    size, grid, count = 16, 8, 12

    # The target is generated *at* the feature grid's resolution and upsampled,
    # so it is representable from that grid. Drawing it at full resolution
    # instead would make this a test of how much of an i.i.d. random field
    # survives 2x2 pooling — which is little, and nothing to do with the probe.
    coarse = torch.rand(count, 1, grid, grid) * 0.13
    targets = torch.nn.functional.interpolate(coarse, size=size, mode="bilinear")[:, 0]
    features = torch.cat([coarse, torch.rand(count, 3, grid, grid) * 0.01], dim=1)

    task = EdgeTask(epochs=60, lr=5e-2, batch_size=4)
    task.fit({"dense": features}, targets)
    scores = task.evaluate({"dense": features}, targets)

    assert scores["edge_correlation"] > 0.9
    assert task.train_loss is not None and task.train_loss < 0.02


def test_predict_returns_one_channel_per_image():
    set_seed(0)
    features = torch.rand(4, 6, 8, 8)
    targets = torch.rand(4, 16, 16) * 0.1

    task = EdgeTask(epochs=1, warmup_epochs=0)
    task.fit({"dense": features}, targets)
    predictions = task.predict({"dense": features})

    assert predictions.shape == (4, 1, 16, 16)

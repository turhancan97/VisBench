"""Gradient orientation — the metric, the loss, and the probe end to end.

The fourth low-level probe and the first whose target is a *direction*, so it is
the first that could not reuse ``DenseMagnitudeTask``. What is tested here is the
part specific to orientation: the doubled-angle metric (chance is 45 degrees,
not 0), the coherence weighting, and that ``_activate``, ``_loss`` and the
metric all describe one quantity.
"""

import math

import pytest
import torch

import visbench
from visbench.metrics.dense import orientation_metrics
from visbench.tasks.low_level import OrientationTask
from visbench.utils import set_seed


def _field(two_theta: torch.Tensor, coherence: torch.Tensor) -> torch.Tensor:
    """``(1, 2, H, W)`` target from ``(H, W)`` inputs: coherence * (cos 2t, sin 2t)."""
    return torch.stack([coherence * two_theta.cos(), coherence * two_theta.sin()])[None]


# -- orientation_metrics ----------------------------------------------------


def test_a_perfect_prediction_scores_zero_error():
    two_theta = torch.rand(8, 8) * 2 * math.pi
    target = _field(two_theta, torch.ones(8, 8))
    scores = orientation_metrics(target.clone(), target)
    assert scores["orientation_error"] == pytest.approx(0.0, abs=1e-4)
    assert scores["d1"] == pytest.approx(1.0)
    assert scores["d2"] == pytest.approx(1.0)


def test_a_perpendicular_prediction_scores_ninety_degrees():
    """Orientation error tops out at 90, not 180 — theta and theta+90 are as
    wrong as it gets, and in the doubled angle that is a sign flip."""
    two_theta = torch.zeros(4, 4)
    target = _field(two_theta, torch.ones(4, 4))
    wrong = _field(two_theta + math.pi, torch.ones(4, 4))  # theta -> theta + 90
    assert orientation_metrics(wrong, target)["orientation_error"] == pytest.approx(90.0, abs=1e-3)


def test_a_random_prediction_sits_near_chance():
    set_seed(0)
    two_theta = torch.rand(16, 12, 12) * 2 * math.pi
    target = torch.stack([two_theta.cos(), two_theta.sin()], dim=1)
    pred = torch.nn.functional.normalize(torch.randn(16, 2, 12, 12), dim=1)
    assert orientation_metrics(pred, target)["orientation_error"] == pytest.approx(45.0, abs=3.0)


def test_incoherent_pixels_barely_count():
    """A wrong prediction on a zero-coherence pixel must not move the score:
    the weighting is what lets the probe predict everywhere without a mask."""
    two_theta = torch.zeros(2, 2)
    target = _field(two_theta, torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    wrong = _field(two_theta + math.pi / 2, torch.ones(2, 2))
    # Only the two coherent pixels are wrong (by 45 deg); the incoherent two are ignored.
    assert orientation_metrics(wrong, target)["orientation_error"] == pytest.approx(45.0, abs=1e-3)


def test_the_metric_is_invariant_to_prediction_length():
    two_theta = torch.rand(6, 6) * 2 * math.pi
    target = _field(two_theta, torch.ones(6, 6))
    assert orientation_metrics(3.0 * target, target)["orientation_error"] == pytest.approx(
        0.0, abs=1e-2
    )


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        orientation_metrics(torch.rand(2, 2, 8, 8), torch.rand(2, 2, 4, 4))
    with pytest.raises(ValueError):
        orientation_metrics(torch.rand(2, 2, 8, 8), torch.rand(2, 3, 8, 8))


# -- OrientationTask ------------------------------------------------------


def test_the_probe_is_registered_as_low_level():
    assert "orientation" in visbench.list_probes()
    assert visbench.get_probe("orientation").level == "low_level"
    assert visbench.get_probe("orientation").out_channels == 2


def test_activate_returns_a_unit_field():
    raw = torch.randn(2, 2, 5, 5) * 4.0
    activated = OrientationTask()._activate(raw)
    assert torch.allclose(activated.norm(dim=1), torch.ones(2, 5, 5), atol=1e-5)


def test_the_loss_is_zero_for_a_matching_direction_and_positive_otherwise():
    two_theta = torch.rand(6, 6) * 2 * math.pi
    target = _field(two_theta, torch.ones(6, 6))
    pred = torch.stack([two_theta.cos(), two_theta.sin()])[None]
    # The eps clamp in the loss floors a perfect match at acos(1 - 1e-4) ~ 0.014.
    assert OrientationTask()._loss(pred, target).item() < 0.02
    assert OrientationTask()._loss(pred[:, [1, 0]], target).item() > 0.5


def test_the_protocol_claims_neither_probe3d_nor_bsds():
    params = OrientationTask().describe()["task_params"]
    assert params["protocol"] == "visbench_structure_tensor_orientation_regression"
    assert "probe3d" not in params["protocol"]


def test_a_probe_recovers_orientation_the_features_encode():
    """End to end: the features carry the answer in two channels, so a linear
    head can reach it — which only converges if _activate, _loss and the metric
    agree on what is predicted."""
    set_seed(0)
    size, grid, count = 16, 8, 16

    two_theta = torch.rand(count, grid, grid) * 2 * math.pi
    coherence = torch.rand(count, grid, grid) * 0.5 + 0.5
    coarse = torch.stack([coherence * two_theta.cos(), coherence * two_theta.sin()], dim=1)
    targets = torch.nn.functional.interpolate(coarse, size=size, mode="bilinear")
    features = torch.cat([coarse, torch.rand(count, 3, grid, grid) * 0.01], dim=1)

    task = OrientationTask(epochs=80, lr=5e-2, batch_size=4)
    task.fit({"dense": features}, targets)
    scores = task.evaluate({"dense": features}, targets)

    assert scores["orientation_error"] < 15.0
    assert task.train_loss is not None


def test_predict_returns_two_channels_per_image():
    set_seed(0)
    features = torch.rand(4, 6, 8, 8)
    two_theta = torch.rand(4, 16, 16) * 2 * math.pi
    targets = torch.stack([two_theta.cos(), two_theta.sin()], dim=1)
    task = OrientationTask(epochs=1, warmup_epochs=0, batch_size=2)
    task.fit({"dense": features}, targets)
    predictions = task.predict({"dense": features})
    assert predictions.shape == (4, 2, 16, 16)

"""Classification linear probe — the first task that trains something.

Retrieval no-ops in ``fit``, so this module is where the trained half of the
:class:`BaseTask` contract is actually exercised: fitting, refusing to predict
unfitted, reproducibility under a seed, and keeping train and test features
apart.
"""

import pytest
import torch

import visbench
from visbench.metrics.classification import top_k_accuracy
from visbench.utils import set_seed


@pytest.fixture
def separable():
    """A problem a linear model provably can solve, split into train and test."""
    set_seed(0)
    features = torch.randn(400, 16)
    weights = torch.randn(16, 4)
    labels = (features @ weights).argmax(dim=1)
    return features[:300], labels[:300], features[300:], labels[300:]


@pytest.fixture
def probe():
    return visbench.get_probe("classification", device="cpu")


# -- metrics -----------------------------------------------------------------


def test_top_k_accuracy_hand_computed():
    logits = torch.tensor([[3.0, 1.0, 2.0], [1.0, 3.0, 2.0]])
    targets = torch.tensor([0, 2])  # rank 1 and rank 2 respectively

    metrics = top_k_accuracy(logits, targets, ks=(1, 2))
    assert metrics["top1"] == 0.5
    assert metrics["top2"] == 1.0


def test_degenerate_k_is_dropped():
    """top-5 on 3 classes is 1.0 by construction; a constant metric is noise."""
    metrics = top_k_accuracy(torch.rand(4, 3), torch.zeros(4, dtype=torch.long), ks=(1, 5))
    assert "top1" in metrics
    assert "top5" not in metrics


def test_all_ks_degenerate_still_returns_top1():
    metrics = top_k_accuracy(torch.rand(4, 2), torch.zeros(4, dtype=torch.long), ks=(5,))
    assert set(metrics) == {"top1"}


# -- fitting -----------------------------------------------------------------


def test_learns_a_separable_problem(probe, separable):
    train_x, train_y, test_x, test_y = separable
    probe.fit(train_x, train_y)

    assert probe.evaluate(test_x, test_y)["top1"] > 0.8


def test_defaults_actually_converge(probe, separable):
    """The default lr must fit data the model provably can separate.

    lr=1e-3 reached 0.66 train accuracy here — underfitting that would have
    silently understated every backbone.
    """
    train_x, train_y, _, _ = separable
    probe.fit(train_x, train_y)

    assert probe.train_top1 > 0.95, f"probe underfitted: train_top1={probe.train_top1}"


def test_train_diagnostics_distinguish_underfit_from_hard_data(separable):
    train_x, train_y, test_x, test_y = separable

    starved = visbench.get_probe("classification", epochs=1, lr=1e-4, device="cpu")
    starved.fit(train_x, train_y)

    assert starved.train_top1 < 0.9
    assert starved.train_loss > 0.5
    # Same data, converged probe — so a low score from `starved` is the
    # optimiser's fault, not the representation's.
    assert (
        visbench.get_probe("classification", device="cpu").fit(train_x, train_y).train_top1 > 0.95
    )


def test_num_classes_is_inferred(probe, separable):
    train_x, train_y, _, _ = separable
    assert probe.num_classes is None
    probe.fit(train_x, train_y)
    assert probe.num_classes == 4


def test_explicit_num_classes_is_respected(separable):
    """A class absent from the training split still needs an output unit."""
    train_x, train_y, _, _ = separable
    probe = visbench.get_probe("classification", num_classes=10, device="cpu")
    probe.fit(train_x, train_y)
    assert probe.logits(train_x).shape[1] == 10


def test_label_beyond_num_classes_raises(separable):
    train_x, train_y, _, _ = separable
    probe = visbench.get_probe("classification", num_classes=2, device="cpu")
    with pytest.raises(ValueError, match="out of range"):
        probe.fit(train_x, train_y)


def test_single_class_raises(probe):
    with pytest.raises(ValueError, match="single class"):
        probe.fit(torch.rand(10, 8), torch.zeros(10, dtype=torch.long))


def test_fit_is_chainable_and_returns_self(probe, separable):
    train_x, train_y, _, _ = separable
    assert probe.fit(train_x, train_y) is probe


# -- reproducibility ---------------------------------------------------------


def test_same_seed_gives_the_same_probe(separable):
    """Training is stochastic, so the recorded seed must fully determine it."""
    train_x, train_y, test_x, test_y = separable

    results = []
    for _ in range(2):
        set_seed(1234)
        probe = visbench.get_probe("classification", device="cpu")
        results.append(probe.fit(train_x, train_y).evaluate(test_x, test_y))

    assert results[0] == results[1]


def test_different_seeds_can_differ(separable):
    """Not a guarantee of difference — just that the seed is actually used."""
    train_x, train_y, _, _ = separable

    weights = []
    for seed in (1, 2):
        set_seed(seed)
        probe = visbench.get_probe("classification", epochs=1, device="cpu")
        weights.append(probe.fit(train_x, train_y).head.weight.clone())

    assert not torch.allclose(weights[0], weights[1])


# -- unfitted and mismatched use --------------------------------------------


def test_predict_before_fit_raises(probe, separable):
    _, _, test_x, _ = separable
    with pytest.raises(RuntimeError, match="has not been fitted"):
        probe.predict(test_x)


def test_evaluate_before_fit_raises(probe, separable):
    _, _, test_x, test_y = separable
    with pytest.raises(RuntimeError, match="has not been fitted"):
        probe.evaluate(test_x, test_y)


def test_feature_dim_mismatch_names_the_cause(probe, separable):
    train_x, train_y, _, _ = separable
    probe.fit(train_x, train_y)

    with pytest.raises(ValueError, match="same backbone"):
        probe.predict(torch.rand(5, 99))


def test_predict_returns_class_indices(probe, separable):
    train_x, train_y, test_x, _ = separable
    probe.fit(train_x, train_y)

    predictions = probe.predict(test_x)
    assert predictions.shape == (100,)
    assert predictions.min() >= 0
    assert predictions.max() < 4


# -- standardisation ---------------------------------------------------------


def test_standardize_uses_training_statistics(separable):
    """Test features must be scaled by train stats, never their own.

    Using test statistics would leak the test distribution into the score.
    """
    train_x, train_y, test_x, test_y = separable
    probe = visbench.get_probe("classification", standardize=True, device="cpu")
    probe.fit(train_x, train_y)

    assert probe._mean.shape == (1, 16)
    shifted = probe.evaluate(test_x + 100.0, test_y)["top1"]
    unshifted = probe.evaluate(test_x, test_y)["top1"]
    assert shifted != unshifted, "a shifted test set was silently re-centred"


def test_standardize_rescues_an_awkward_feature_scale(separable):
    train_x, train_y, _, _ = separable
    scaled_train = train_x * 1e-3

    set_seed(0)
    off = visbench.get_probe("classification", device="cpu")
    off.fit(scaled_train, train_y)

    set_seed(0)
    on = visbench.get_probe("classification", standardize=True, device="cpu")
    on.fit(scaled_train, train_y)

    assert on.train_top1 > off.train_top1


# -- provenance --------------------------------------------------------------


def test_hyperparameters_reach_the_record(probe):
    """A probe accuracy without its optimiser settings is not reproducible."""
    params = probe.describe()["task_params"]

    assert params["optimizer"] == "adamw"
    assert params["lr"] == 1e-2
    assert params["epochs"] == 200
    assert params["weight_decay"] == 1e-4
    assert params["batch_size"] == 256
    assert params["standardize"] is False


def test_zero_shot_tasks_report_no_params():
    assert visbench.get_probe("retrieval").describe()["task_params"] == {}


def test_task_is_not_zero_shot(probe):
    assert probe.zero_shot is False
    assert probe.level == "high_level"


def test_invalid_configuration_raises():
    with pytest.raises(ValueError, match="num_classes must be >= 2"):
        visbench.get_probe("classification", num_classes=1)
    with pytest.raises(ValueError, match="epochs must be >= 1"):
        visbench.get_probe("classification", epochs=0)
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        visbench.get_probe("classification", batch_size=0)

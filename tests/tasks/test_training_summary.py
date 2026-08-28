"""Every trained probe reports how its fit went; every zero-shot one reports None.

Schema v8. The field exists because a low score has two opposite readings — an
underfitting probe, which *understates* a backbone, or a representation that
genuinely does not carry the answer — and `train_loss` is what separates them.
`GenericSegmentationTask` on 80 training images reads 0.16 IoU at the defaults
and 0.87 at `epochs=40` on identical features: same number, opposite conclusion.

Every trained probe already computed this and dropped it before the record, so a
corpus of 156 trained runs could not answer "did this underfit?" without
re-running them. Found while writing up the CUB board, where the claim held for
the six backbones run by hand and could not be checked for the six run on the
cluster.

The parametrised test over `list_probes()` is the point: it fails when a *new*
probe trains a head and does not report it, which is the way this gets lost.
"""

import pytest
import torch

import visbench
from visbench.utils import set_seed

#: Probes that fit nothing. `training_summary()` is None for these by design,
#: which is the same value every pre-v8 record carries by absence.
ZERO_SHOT = {"retrieval", "correspondence", "similarity"}

#: Probes whose constructor demands an argument, because a wrong class count
#: does not raise -- it trains and scores against labels that mean nothing.
REQUIRED_KWARGS: dict[str, dict] = {
    "semantic_segmentation": {"num_classes": 3},
    "detection": {"num_classes": 3},
}


def _probe(name: str, **kwargs):
    return visbench.get_probe(name, **{**REQUIRED_KWARGS.get(name, {}), **kwargs})


def test_the_zero_shot_set_matches_the_probes_own_declaration():
    """A probe declares `zero_shot`; this module's copy must not drift from it."""
    declared = {name for name in visbench.list_probes() if _probe(name).zero_shot}
    assert declared == ZERO_SHOT


@pytest.mark.parametrize("name", sorted(ZERO_SHOT))
def test_a_zero_shot_probe_reports_none(name):
    """None rather than {}: there is no fit to describe.

    That is a different statement from "trained and reported nothing about it",
    and it is the convention `finetune` already uses.
    """
    assert _probe(name).training_summary() is None


@pytest.mark.parametrize("name", sorted(set(visbench.list_probes()) - ZERO_SHOT))
def test_a_trained_probe_reports_none_before_it_is_fitted(name):
    """A record must never claim a fit that did not happen."""
    assert _probe(name).training_summary() is None


class TestAfterFitting:
    """The classification family is the one that reports an accuracy too."""

    @pytest.fixture
    def separable(self):
        set_seed(0)
        features = torch.randn(240, 16)
        weights = torch.randn(16, 4)
        return features, (features @ weights).argmax(dim=1)

    @pytest.mark.parametrize(
        "name", ["classification", "scene_classification", "fine_grained_classification"]
    )
    def test_the_classification_family_reports_loss_and_top1(self, name, separable):
        features, labels = separable
        probe = visbench.get_probe(name, epochs=30, device="cpu")
        set_seed(0)
        probe.fit(features, labels)

        summary = probe.training_summary()
        assert summary is not None
        assert set(summary) == {"train_loss", "train_top1"}
        assert 0.0 <= summary["train_top1"] <= 1.0
        assert summary["train_loss"] >= 0.0
        # It must be the *training* number, not a copy of the evaluation one.
        assert summary["train_top1"] == probe.train_top1

    def test_the_values_are_json_primitives(self, separable):
        """These land in a JSONL record, so a tensor here would fail at write time."""
        features, labels = separable
        probe = visbench.get_probe("classification", epochs=5, device="cpu")
        set_seed(0)
        probe.fit(features, labels)
        for key, value in probe.training_summary().items():
            assert isinstance(key, str)
            assert isinstance(value, (int, float)) and not isinstance(value, bool)

    def test_an_underfitted_probe_is_visible_in_the_summary(self, separable):
        """The whole purpose: the diagnostic has to actually move.

        A probe given one epoch at a tiny learning rate cannot fit data it
        provably can separate. If `train_loss` did not distinguish that from a
        converged run, the field would be decoration.
        """
        features, labels = separable
        set_seed(0)
        starved = visbench.get_probe("classification", epochs=1, lr=1e-6, device="cpu")
        starved.fit(features, labels)
        set_seed(0)
        converged = visbench.get_probe("classification", epochs=200, device="cpu")
        converged.fit(features, labels)

        assert starved.training_summary()["train_loss"] > converged.training_summary()["train_loss"]
        assert starved.training_summary()["train_top1"] < converged.training_summary()["train_top1"]

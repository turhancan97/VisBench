"""Retrieval task and its metrics.

The metrics are tested against hand-computed values rather than a reference
implementation — a retrieval number that is quietly wrong is exactly the kind
of bug a benchmark library must not ship.
"""

import pytest
import torch

import visbench
from visbench.metrics.retrieval import mean_average_precision, recall_at_k


@pytest.fixture
def probe():
    return visbench.get_probe("retrieval")


@pytest.fixture
def separable():
    """Three tight clusters of four; perfect retrieval is achievable."""
    centres = torch.eye(3) * 10
    features = torch.cat([centres[i].repeat(4, 1) for i in range(3)])
    features = features + torch.randn(12, 3, generator=torch.manual_seed(0)) * 0.01
    labels = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    return features, labels


# -- metrics against hand-computed values ------------------------------------


def test_recall_at_k_hand_computed():
    # Query 0: first hit at rank 1. Query 1: first hit at rank 3.
    ranked = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0]])
    queries = torch.tensor([1, 1])

    metrics = recall_at_k(ranked, queries, ks=(1, 2, 3))
    assert metrics["recall@1"] == 0.5
    assert metrics["recall@2"] == 0.5
    assert metrics["recall@3"] == 1.0


def test_recall_skips_k_larger_than_the_gallery():
    """recall@10 over 4 candidates is a different number wearing the same name."""
    ranked = torch.tensor([[1, 0, 0, 0]])
    metrics = recall_at_k(ranked, torch.tensor([1]), ks=(1, 5, 10))

    assert "recall@1" in metrics
    assert "recall@5" not in metrics
    assert "recall@10" not in metrics


def test_map_hand_computed():
    # Relevant at ranks 1 and 3: AP = (1/1 + 2/3) / 2 = 0.8333...
    ranked = torch.tensor([[1, 0, 1, 0]])
    result = mean_average_precision(ranked, torch.tensor([1]))
    assert result == pytest.approx((1.0 + 2 / 3) / 2)


def test_map_is_one_for_a_perfect_ranking():
    ranked = torch.tensor([[1, 1, 0, 0]])
    assert mean_average_precision(ranked, torch.tensor([1])) == pytest.approx(1.0)


def test_map_handles_a_query_with_no_relevant_item():
    """A singleton class must contribute 0, not NaN."""
    ranked = torch.tensor([[0, 0, 0]])
    result = mean_average_precision(ranked, torch.tensor([1]))
    assert result == 0.0


def test_map_penalises_late_hits_where_recall_saturates():
    early = torch.tensor([[1, 1, 0, 0]])
    late = torch.tensor([[1, 0, 0, 1]])
    queries = torch.tensor([1])

    # recall@1 cannot tell these apart; mAP can.
    assert recall_at_k(early, queries, ks=(1,)) == recall_at_k(late, queries, ks=(1,))
    assert mean_average_precision(early, queries) > mean_average_precision(late, queries)


# -- the task ----------------------------------------------------------------


def test_perfect_retrieval_on_separable_clusters(probe, separable):
    features, labels = separable
    metrics = probe.evaluate(features, labels)

    assert metrics["recall@1"] == 1.0
    assert metrics["mAP"] == pytest.approx(1.0)


def test_random_features_score_near_chance(probe):
    """A sanity floor: 10 classes, so recall@1 should land nowhere near 1.0."""
    torch.manual_seed(0)
    features = torch.randn(200, 32)
    labels = torch.arange(200) % 10

    metrics = probe.evaluate(features, labels)
    assert metrics["recall@1"] < 0.35


def test_self_match_is_excluded(probe):
    """Leaving the query in its own ranking makes recall@1 trivially 1.0."""
    torch.manual_seed(0)
    features = torch.randn(20, 8)
    # Every image its own class: with self-matching, recall@1 would be 1.0.
    labels = torch.arange(20)

    metrics = probe.evaluate(features, labels)
    assert metrics["recall@1"] == 0.0


def test_ranking_excludes_self_column(probe, separable):
    features, _ = separable
    ranking = probe.predict(features)

    assert ranking.shape == (12, 11)
    assert not (ranking == torch.arange(12)[:, None]).any(), "a query ranked itself"


def test_cosine_ignores_magnitude(probe, separable):
    """Cosine must score a scaled copy identically; l2 must not."""
    features, labels = separable
    scaled = features * 5

    cosine = visbench.get_probe("retrieval", metric="cosine")
    assert cosine.evaluate(features, labels) == cosine.evaluate(scaled, labels)


def test_l2_metric_runs_and_agrees_when_separable(separable):
    features, labels = separable
    l2 = visbench.get_probe("retrieval", metric="l2")
    assert l2.evaluate(features, labels)["recall@1"] == 1.0


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="Unknown metric"):
        visbench.get_probe("retrieval", metric="manhattan")


def test_empty_topk_raises():
    with pytest.raises(ValueError, match="at least one k"):
        visbench.get_probe("retrieval", topk=())


def test_single_image_cannot_be_ranked(probe):
    with pytest.raises(ValueError, match="at least 2 images"):
        probe.predict(torch.rand(1, 8))


# -- explicit query/gallery split --------------------------------------------


def test_separate_gallery(probe):
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gallery = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    query_labels = torch.tensor([0, 1])
    gallery_labels = torch.tensor([0, 1, 2])

    metrics = probe.evaluate(
        queries,
        query_labels,
        gallery_features=gallery,
        gallery_labels=gallery_labels,
    )
    assert metrics["recall@1"] == 1.0


def test_separate_gallery_keeps_all_columns(probe):
    """No self-match to drop when the gallery is a different set."""
    ranking = probe.predict(torch.rand(2, 4), gallery_features=torch.rand(5, 4))
    assert ranking.shape == (2, 5)


def test_mismatched_feature_dims_raise(probe):
    with pytest.raises(ValueError, match="same backbone"):
        probe.predict(torch.rand(2, 8), gallery_features=torch.rand(5, 16))

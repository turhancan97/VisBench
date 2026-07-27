"""Mid-level image similarity — zero-shot 2AFC.

The probe is two cosine similarities and a comparison, so most of what can go
wrong is structural rather than numerical: reading the wrong feature for a
triplet, silently accepting features and triplets from different datasets, or
letting ties decide a number invisibly. That is what these cover.

The metric itself is checked against scikit-learn in
``tests/tasks/test_2afc_metrics.py``.
"""

import pytest
import torch

import visbench
from visbench.tasks.mid_level.similarity import MidLevelSimilarityTask


def probe(**kwargs):
    return visbench.get_probe("similarity", **kwargs)


@pytest.fixture
def planted():
    """Features where the answer is unambiguous, and triplets that say so.

    Four unit vectors. Triplet 0: reference matches image 1 exactly, so "left"
    wins and the vote is 0. Triplet 1: reference matches image 3, so "right"
    wins and the vote is 1.
    """
    pooled = torch.tensor(
        [
            [1.0, 0.0],  # 0: reference for both triplets
            [1.0, 0.0],  # 1: identical to it
            [0.0, 1.0],  # 2: orthogonal
            [1.0, 0.0],  # 3: identical again
        ]
    )
    triplets = torch.tensor(
        [
            [0, 1, 2, 0],  # left is the match, humans said left
            [0, 2, 3, 1],  # right is the match, humans said right
        ]
    )
    return {"pooled": pooled}, triplets


class TestShape:
    def test_it_is_zero_shot_and_mid_level(self):
        task = probe()
        assert task.zero_shot is True
        assert task.level == "mid_level"

    def test_it_reads_pooled_features_not_dense(self):
        """A global vector per image is what the comparison needs."""
        assert probe().uses_dense is False

    def test_fit_is_a_no_op_that_returns_self(self, planted):
        features, triplets = planted
        task = probe()
        assert task.fit(features, triplets) is task

    def test_it_is_not_retrieval(self):
        """Both are 'similarity'-flavoured and must stay separate tasks."""
        assert probe().level == "mid_level"
        assert visbench.get_probe("retrieval").level == "high_level"


class TestPredict:
    def test_it_picks_the_more_similar_candidate(self, planted):
        features, triplets = planted
        assert probe().predict(features, triplets).tolist() == [0, 1]

    def test_it_needs_the_triplets(self, planted):
        """Feature vectors alone do not say what is compared with what."""
        features, _ = planted
        with pytest.raises(ValueError, match="triplets"):
            probe().predict(features)

    def test_a_bare_tensor_works_like_a_feature_dict(self, planted):
        features, triplets = planted
        task = probe()
        assert torch.equal(
            task.predict(features, triplets), task.predict(features["pooled"], triplets)
        )

    def test_indices_outside_the_features_are_caught(self, planted):
        """Triplets and features from different datasets must not score silently."""
        features, _ = planted
        with pytest.raises(IndexError, match="outside"):
            probe().predict(features, torch.tensor([[0, 1, 99, 0]]))

    def test_malformed_triplets_are_refused(self, planted):
        features, _ = planted
        with pytest.raises(ValueError, match=r"\(T, 4\)"):
            probe().predict(features, torch.tensor([[0, 1, 2]]))

    def test_dense_only_features_are_refused(self):
        with pytest.raises((ValueError, KeyError, TypeError)):
            probe().predict({"dense": torch.randn(4, 8, 2, 2)}, torch.tensor([[0, 1, 2, 0]]))

    def test_cosine_ignores_magnitude(self):
        """Scaling a candidate must not change which one is preferred."""
        pooled = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        triplets = torch.tensor([[0, 1, 2, 0]])
        scaled = pooled.clone()
        scaled[1] *= 100.0

        task = probe()
        assert torch.equal(
            task.predict({"pooled": pooled}, triplets),
            task.predict({"pooled": scaled}, triplets),
        )


class TestEvaluate:
    def test_perfect_agreement(self, planted):
        features, triplets = planted
        metrics = probe().evaluate(features, triplets)

        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0

    def test_total_disagreement(self, planted):
        """Flipping every vote must give 0, not 1 — the sign is easy to invert."""
        features, triplets = planted
        flipped = triplets.clone()
        flipped[:, 3] = 1 - flipped[:, 3]

        assert probe().evaluate(features, flipped)["accuracy"] == 0.0

    def test_it_returns_a_flat_dict_of_floats(self, planted):
        features, triplets = planted
        metrics = probe().evaluate(features, triplets)
        assert all(isinstance(value, float) for value in metrics.values())

    def test_ties_are_reported_not_hidden(self):
        """Identical candidates leave the choice arbitrary; say how often."""
        pooled = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        triplets = torch.tensor([[0, 1, 2, 0]])

        assert probe().evaluate({"pooled": pooled}, triplets)["tie_rate"] == 1.0

    def test_no_ties_on_distinct_features(self, planted):
        features, triplets = planted
        assert probe().evaluate(features, triplets)["tie_rate"] == 0.0


class TestDescribe:
    def test_it_records_the_protocol(self):
        """Borrowed from the mid-level paper, and not probe3d's."""
        assert probe().describe()["task_params"]["protocol"] == "midvision_2afc"

    def test_it_records_the_vote_filter(self):
        """A score over near-unanimous triplets is not comparable to one over
        contested ones, so the threshold has to travel with the number."""
        assert probe(min_votes=6).describe()["task_params"]["min_votes"] == 6

    def test_min_votes_is_absent_rather_than_wrong_when_unset(self):
        assert probe().describe()["task_params"]["min_votes"] is None

    def test_it_records_the_similarity_function(self):
        assert probe().describe()["task_params"]["similarity"] == "cosine"


class TestThroughRun:
    def test_it_runs_end_to_end_and_writes_a_record(self, tmp_path, fake_vit, two_afc_folder):
        from visbench.cache import FeatureCache

        dataset = two_afc_folder(tmp_path / "data", triplets=6)
        results = tmp_path / "results.jsonl"

        result = visbench.run(
            fake_vit,
            MidLevelSimilarityTask(min_votes=6),
            dataset,
            cache=FeatureCache(root=tmp_path / "cache"),
            results=results,
        )

        assert result.record.task == "similarity"
        assert result.record.level == "mid_level"
        assert 0.0 <= result.metrics["accuracy"] <= 1.0
        assert results.exists()

    def test_the_record_names_the_triplet_count(self, tmp_path, two_afc_folder):
        """dataset_size counts images, which is three times as many."""
        dataset = two_afc_folder(tmp_path / "data", triplets=5)
        described = dataset.describe()

        assert described["num_triplets"] == 5
        assert described["dataset_size"] == 15

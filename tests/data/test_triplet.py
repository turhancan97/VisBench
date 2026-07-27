"""``TwoAFCDataset`` — triplets presented as a flat image dataset.

The design under test: a triplet is three images, but the feature cache works
one image at a time, so the dataset exposes *unique images* and puts the triplet
structure in ``labels()`` as indices into itself. That keeps the cache, the
fingerprint and ``visbench.run`` unchanged, and makes the pairing travel by
index rather than by iteration order.

The things that can go wrong quietly: reading the wrong CSV column, letting a
split filter through the wrong rows, and any subsetting that moves an image
without moving the triplets that point at it.
"""

import pytest
import torch

from visbench.data.triplet import NIGHTS_MIN_VOTES, TwoAFCDataset


class TestStructure:
    def test_length_counts_images_not_triplets(self, tmp_path, two_afc_folder):
        dataset = two_afc_folder(tmp_path / "d", triplets=4)

        assert len(dataset) == 12
        assert len(dataset.triplets) == 4

    def test_items_are_images_with_no_label(self, tmp_path, two_afc_folder):
        """A single image's 'label' is meaningless; the vote belongs to a triplet."""
        dataset = two_afc_folder(tmp_path / "d", triplets=2)
        image, label = dataset[0]

        assert label is None
        assert image.size == (32, 32)

    def test_triplet_indices_stay_in_range(self, tmp_path, two_afc_folder):
        dataset = two_afc_folder(tmp_path / "d", triplets=5)
        indices = dataset.triplets[:, :3]

        assert indices.min() >= 0
        assert indices.max() < len(dataset)

    def test_labels_are_the_triplets(self, tmp_path, two_afc_folder):
        dataset = two_afc_folder(tmp_path / "d", triplets=3)
        assert torch.equal(dataset.labels(), dataset.triplets)

    def test_shape_is_ref_left_right_vote(self, tmp_path, two_afc_folder):
        dataset = two_afc_folder(tmp_path / "d", triplets=3)
        assert dataset.triplets.shape == (3, 4)

    def test_repeated_images_are_extracted_once(self, tmp_path):
        """Two triplets sharing a reference must share its index, not duplicate it."""
        root = tmp_path / "d"
        (root / "ref").mkdir(parents=True)
        (root / "distort").mkdir(parents=True)
        from PIL import Image

        for name in ("r", "a", "b", "c"):
            Image.new("RGB", (8, 8), (10, 20, 30)).save(root / "ref" / f"{name}.png")
        rows = [
            "id,left_vote,right_vote,votes,ref_path,left_path,right_path,split,is_imagenet",
            "0,1,0,8,ref/r.png,ref/a.png,ref/b.png,test,TRUE",
            "1,0,1,8,ref/r.png,ref/a.png,ref/c.png,test,TRUE",
        ]
        (root / "data.csv").write_text("\n".join(rows) + "\n")

        dataset = TwoAFCDataset(root, split="test")
        assert len(dataset) == 4, "r and a are shared between the two triplets"
        assert dataset.triplets[0, 0] == dataset.triplets[1, 0]


class TestVotes:
    def test_the_vote_column_is_right_vote(self, tmp_path, two_afc_folder):
        """Read by name; the reference reads it positionally as iloc[idx, 2]."""
        dataset = two_afc_folder(tmp_path / "d", triplets=4)
        # The fixture alternates: even index -> left preferred (0), odd -> right (1).
        assert dataset.triplets[:, 3].tolist() == [0, 1, 0, 1]

    def test_low_vote_triplets_are_dropped(self, tmp_path, two_afc_folder):
        root = two_afc_folder(tmp_path / "d", triplets=4, min_votes_ok=False, construct=False)
        with pytest.raises(ValueError, match="agreeing votes"):
            TwoAFCDataset(root, split="test")

    def test_min_votes_default_matches_the_reference(self):
        assert NIGHTS_MIN_VOTES == 6

    def test_a_lower_threshold_admits_more(self, tmp_path, two_afc_folder):
        """Which is why the threshold is recorded: it moves every score."""
        root = two_afc_folder(tmp_path / "d", triplets=3, min_votes_ok=False, construct=False)
        relaxed = TwoAFCDataset(root, split="test", min_votes=1)
        assert len(relaxed.triplets) == 3


class TestSplits:
    def test_an_unknown_split_is_refused(self, tmp_path, two_afc_folder):
        two_afc_folder(tmp_path / "d", triplets=2)
        with pytest.raises(ValueError, match="Unknown split"):
            TwoAFCDataset(tmp_path / "d", split="testing")

    def test_imagenet_subsets_partition_the_test_split(self, tmp_path, two_afc_folder):
        """A backbone trained on ImageNet has seen those references."""
        two_afc_folder(tmp_path / "d", triplets=6)
        everything = TwoAFCDataset(tmp_path / "d", split="test")
        seen = TwoAFCDataset(tmp_path / "d", split="test_imagenet")
        unseen = TwoAFCDataset(tmp_path / "d", split="test_no_imagenet")

        assert len(seen.triplets) + len(unseen.triplets) == len(everything.triplets)
        assert len(seen.triplets) > 0 and len(unseen.triplets) > 0

    def test_an_empty_split_raises_rather_than_scoring_nothing(self, tmp_path, two_afc_folder):
        two_afc_folder(tmp_path / "d", triplets=2, split="test")
        with pytest.raises(ValueError, match="No triplets"):
            TwoAFCDataset(tmp_path / "d", split="train")


class TestMalformed:
    def test_a_missing_csv_says_what_is_needed(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="which candidate was preferred"):
            TwoAFCDataset(tmp_path / "empty")

    def test_a_missing_column_is_named(self, tmp_path):
        root = tmp_path / "d"
        root.mkdir()
        (root / "data.csv").write_text("id,votes,split\n0,8,test\n")

        with pytest.raises(ValueError, match="missing column"):
            TwoAFCDataset(root)

    def test_a_missing_root_is_refused(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            TwoAFCDataset(tmp_path / "nope")


class TestLimiting:
    def test_max_triplets_shortens_both_lists_together(self, tmp_path, two_afc_folder):
        two_afc_folder(tmp_path / "d", triplets=6)
        short = TwoAFCDataset(tmp_path / "d", split="test", max_triplets=2)

        assert len(short.triplets) == 2
        assert len(short) == 6, "images follow the triplets that survive"
        assert short.triplets[:, :3].max() < len(short)

    def test_subset_is_refused_with_a_pointer_to_the_alternative(self, tmp_path, two_afc_folder):
        """Slicing images would leave triplets pointing at moved indices."""
        dataset = two_afc_folder(tmp_path / "d", triplets=4)
        with pytest.raises(NotImplementedError, match="max_triplets"):
            dataset.subset(2)


class TestFingerprint:
    def test_it_covers_the_triplets_not_just_the_images(self, tmp_path, two_afc_folder):
        """Two splits can share images and ask entirely different questions."""
        two_afc_folder(tmp_path / "d", triplets=6)
        everything = TwoAFCDataset(tmp_path / "d", split="test")
        subset = TwoAFCDataset(tmp_path / "d", split="test_imagenet")

        assert everything.fingerprint() != subset.fingerprint()

    def test_the_vote_filter_changes_it(self, tmp_path, two_afc_folder):
        two_afc_folder(tmp_path / "d", triplets=4)
        strict = TwoAFCDataset(tmp_path / "d", split="test", min_votes=6)
        relaxed = TwoAFCDataset(tmp_path / "d", split="test", min_votes=1)

        assert strict.fingerprint() != relaxed.fingerprint()

    def test_it_is_stable(self, tmp_path, two_afc_folder):
        two_afc_folder(tmp_path / "d", triplets=4)
        first = TwoAFCDataset(tmp_path / "d", split="test")
        second = TwoAFCDataset(tmp_path / "d", split="test")

        assert first.fingerprint() == second.fingerprint()

    def test_describe_reports_both_counts(self, tmp_path, two_afc_folder):
        dataset = two_afc_folder(tmp_path / "d", triplets=4)
        described = dataset.describe()

        assert described["dataset_size"] == 12
        assert described["num_triplets"] == 4


class TestCacheIdentity:
    def test_it_is_offered_so_cached_images_are_not_decoded(self, tmp_path, two_afc_folder):
        dataset = two_afc_folder(tmp_path / "d", triplets=2)
        assert dataset.cache_identity(0) is not None

    def test_it_differs_between_images(self, tmp_path, two_afc_folder):
        dataset = two_afc_folder(tmp_path / "d", triplets=2)
        assert dataset.cache_identity(0) != dataset.cache_identity(1)

"""``BaseDataset.subset`` — the public way to shorten a split.

Before this existed, every example did it by reaching into the dataset: two
touched the private ``_labels``, and the four dense ones sliced three parallel
lists in step, each carrying the same comment explaining that dropping one would
pair a target with the wrong image. That hazard belongs in one tested place,
which is what these tests cover.
"""

import numpy as np
import pytest
from PIL import Image

from visbench.data import DenseFolderDataset, ImageFolderDataset
from visbench.data.pair_dataset import HomographyPairDataset


@pytest.fixture
def image_folder(tmp_path):
    """Two classes of four images, so per-class behaviour is visible."""
    for klass in ("a", "b"):
        (tmp_path / klass).mkdir()
        for index in range(4):
            Image.fromarray(np.full((16, 16, 3), 10 * index + 1, dtype=np.uint8)).save(
                tmp_path / klass / f"{index}.png"
            )
    return ImageFolderDataset(tmp_path, split="val")


@pytest.fixture
def dense_folder(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "depths").mkdir()
    for index in range(6):
        Image.fromarray(np.full((16, 16, 3), 10 * index + 1, dtype=np.uint8)).save(
            tmp_path / "images" / f"s{index}.png"
        )
        np.save(tmp_path / "depths" / f"s{index}.npy", np.full((16, 16), index + 1.0))
    return DenseFolderDataset(tmp_path, image_size=16)


class TestPrefix:
    def test_it_takes_the_first_n(self, image_folder):
        assert len(image_folder.subset(3)) == 3

    def test_it_clamps_rather_than_raising(self, image_folder):
        """`--limit` means "at most N", so asking for more is not an error."""
        assert len(image_folder.subset(100)) == len(image_folder)

    def test_zero_is_refused(self, image_folder):
        with pytest.raises(ValueError, match="n >= 1"):
            image_folder.subset(0)

    def test_the_original_is_untouched(self, image_folder):
        before = len(image_folder)
        image_folder.subset(2)
        assert len(image_folder) == before


class TestExplicitIndices:
    def test_it_selects_and_reorders(self, image_folder):
        chosen = image_folder.subset([7, 0])
        assert chosen.labels() == [1, 0]

    def test_an_out_of_range_index_raises(self, image_folder):
        """Skipping it would silently produce a shorter split than asked for."""
        with pytest.raises(IndexError, match="out of range"):
            image_folder.subset([0, 99])

    def test_an_empty_sequence_is_refused(self, image_folder):
        with pytest.raises(ValueError, match="empty"):
            image_folder.subset([])

    def test_a_negative_index_is_refused_not_wrapped(self, image_folder):
        """Python would read -1 as the last item; a split spec should not."""
        with pytest.raises(IndexError, match="out of range"):
            image_folder.subset([-1])


class TestParallelSequences:
    """The failure this method exists to prevent."""

    def test_every_parallel_list_is_reindexed_together(self, dense_folder):
        chosen = dense_folder.subset([4, 1])

        assert chosen.stems == ["s4", "s1"]
        assert [path.stem for path in chosen.image_paths] == ["s4", "s1"]
        assert [path.stem for path in chosen.target_paths] == ["s4", "s1"]

    def test_targets_still_match_their_images(self, dense_folder):
        """Depth s3 encodes the value 4; it must survive selection intact."""
        chosen = dense_folder.subset([3])
        assert chosen.target(0).unique().tolist() == [4.0]

    def test_labels_follow_the_selection(self, image_folder):
        assert image_folder.subset([0, 4]).labels() == [0, 1]


class TestFingerprint:
    def test_a_subset_is_distinguishable_from_the_full_split(self, image_folder):
        """Or a quick run's record would look exactly like a full one's."""
        assert image_folder.subset(3).fingerprint() != image_folder.fingerprint()

    def test_two_different_subsets_differ(self, image_folder):
        assert (
            image_folder.subset([0, 1]).fingerprint() != image_folder.subset([2, 3]).fingerprint()
        )

    def test_the_same_subset_is_stable(self, image_folder):
        assert image_folder.subset(3).fingerprint() == image_folder.subset(3).fingerprint()

    def test_dense_subsets_differ_too(self, dense_folder):
        assert dense_folder.subset(2).fingerprint() != dense_folder.fingerprint()


class TestPairDataset:
    """Delegates to its source, and warps are drawn from the position."""

    @pytest.fixture
    def pairs(self, tmp_path):
        for index in range(5):
            Image.fromarray(
                np.random.RandomState(index).randint(0, 255, (32, 32, 3), dtype=np.uint8)
            ).save(tmp_path / f"{index}.png")
        return HomographyPairDataset(tmp_path, image_size=32)

    def test_it_subsets_through_the_source(self, pairs):
        assert len(pairs.subset(2)) == 2

    def test_a_prefix_keeps_the_original_warps(self, pairs):
        """Positions 0..n-1 are unchanged, so a --limit run is a sub-experiment."""
        full = pairs.labels()[1]["homography"]
        assert pairs.subset(3).labels()[1]["homography"].equal(full)

    def test_a_reordered_subset_re_rolls_them(self, pairs):
        """Documented, not a bug: the warp comes from the position, not the image.

        Worth a test because the consequence is easy to miss — such a subset is
        not a sub-experiment of the original and should not be reported as one.
        """
        reordered = pairs.subset([3, 0])
        assert not reordered.labels()[0]["homography"].equal(pairs.labels()[3]["homography"])

    def test_the_original_is_untouched(self, pairs):
        pairs.subset(2)
        assert len(pairs) == 5


class TestUndeclared:
    def test_a_dataset_that_declares_nothing_says_so(self):
        """Better than silently returning an identical copy."""
        from visbench.data.base import BaseDataset

        class Bare(BaseDataset):
            def __len__(self):
                return 3

            def __getitem__(self, index):
                return None, None

        with pytest.raises(NotImplementedError, match="_parallel_attrs"):
            Bare().subset(2)

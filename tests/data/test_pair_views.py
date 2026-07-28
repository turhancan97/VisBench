"""Flattening a pair dataset into single views, and putting it back.

The whole point of :class:`PairViewDataset` is that the pairing survives the
round trip through a cache that has no idea pairs exist. What that costs if it
is wrong is not an error — it is a correspondence score computed by matching
some image against the wrong partner, which still produces a plausible number.
"""

import pytest
from PIL import Image

from visbench.data import PairViewDataset
from visbench.data.pair_dataset import HomographyPairDataset, PairDataset


@pytest.fixture
def folder(tmp_path):
    """Six visibly different images, flat."""
    root = tmp_path / "images"
    root.mkdir()
    for index in range(6):
        shade = 20 + 40 * index
        Image.new("RGB", (64, 64), (shade, 255 - shade, 128)).save(root / f"{index:02d}.png")
    return root


@pytest.fixture
def pairs(folder):
    return HomographyPairDataset(folder, image_size=64)


class TestFlattening:
    def test_two_views_per_pair(self, pairs):
        assert len(PairViewDataset(pairs)) == 2 * len(pairs)

    def test_views_interleave_pair_by_pair(self, pairs):
        """Item 2i and 2i+1 are pair i's two views, in that order."""
        views = PairViewDataset(pairs)
        for index in range(len(pairs)):
            image_0, image_1, _ = pairs[index]
            assert list(views[2 * index][0].getdata()) == list(image_0.getdata())
            assert list(views[2 * index + 1][0].getdata()) == list(image_1.getdata())

    def test_a_view_carries_no_label(self, pairs):
        """Geometry belongs to the pair, so a single view has nothing to report."""
        assert PairViewDataset(pairs)[0][1] is None

    def test_out_of_range_raises(self, pairs):
        views = PairViewDataset(pairs)
        with pytest.raises(IndexError, match="out of range"):
            views[len(views)]

    def test_it_refuses_an_ordinary_dataset(self, folder):
        from visbench.data import ImageFolderDataset

        with pytest.raises(TypeError, match="wraps a PairDataset"):
            PairViewDataset(ImageFolderDataset(folder, labeled=False))


class TestIdentity:
    """The reason this class exists at all: cheap, correct, per-view identities."""

    def test_the_two_views_of_a_pair_have_different_identities(self, pairs):
        views = PairViewDataset(pairs)
        assert views.cache_identity(0) != views.cache_identity(1)

    def test_identities_are_unique_across_the_split(self, pairs):
        views = PairViewDataset(pairs)
        identities = [views.cache_identity(index) for index in range(len(views))]
        assert len(set(identities)) == len(identities)

    def test_identity_tracks_the_warp(self, folder):
        """A different warp is different data, so the memo must not be reused."""
        one = PairViewDataset(HomographyPairDataset(folder, max_warp=0.2, image_size=64))
        other = PairViewDataset(HomographyPairDataset(folder, max_warp=0.4, image_size=64))
        assert one.cache_identity(1) != other.cache_identity(1)

    def test_a_base_pair_dataset_has_no_identity(self):
        """None is always safe — it costs a decode, it never serves wrong features."""

        class Bare(PairDataset):
            name = "bare"
            split = "test"

            def __len__(self):
                return 1

        assert Bare().view_identity(0, 0) is None
        assert PairViewDataset(Bare()).cache_identity(0) is None

    def test_fingerprint_is_the_pair_datasets_own(self, pairs):
        assert PairViewDataset(pairs).fingerprint() == pairs.fingerprint()


class TestRegroup:
    def test_it_restores_the_pairing(self):
        views = [f"v{index}" for index in range(6)]
        regrouped = PairViewDataset.regroup(views)
        assert len(regrouped) == 3
        assert list(regrouped) == [("v0", "v1"), ("v2", "v3"), ("v4", "v5")]

    def test_it_is_lazy(self):
        """Indexing must not materialise the whole split; that was the point."""

        class Counting:
            def __init__(self):
                self.reads = 0

            def __len__(self):
                return 8

            def __getitem__(self, index):
                self.reads += 1
                return index

        source = Counting()
        regrouped = PairViewDataset.regroup(source)
        assert len(regrouped) == 4
        assert source.reads == 0  # len() alone reads nothing
        assert regrouped[2] == (4, 5)
        assert source.reads == 2  # and one pair reads exactly two views

    def test_an_odd_number_of_views_raises(self):
        with pytest.raises(ValueError, match="even number of views"):
            PairViewDataset.regroup(["only", "three", "views"])

    def test_out_of_range_raises(self):
        regrouped = PairViewDataset.regroup(["a", "b"])
        with pytest.raises(IndexError, match="out of range"):
            regrouped[1]

    def test_slicing_is_refused(self):
        """Rather than returning a tuple of two views, which is what a slice
        would silently look like."""
        with pytest.raises(TypeError, match="index one pair"):
            PairViewDataset.regroup(["a", "b", "c", "d"])[0:2]

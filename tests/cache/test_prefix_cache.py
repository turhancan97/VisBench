"""The frozen-prefix cache — step 6b.

Two things have to hold, and both fail silently if they do not:

* a run served from the prefix cache reports **exactly** what a run that
  recomputed the frozen blocks reports. The saving is compute only, so any
  difference at all is a bug, not a tolerance;
* a prefix entry is never mistaken for a feature entry. They live in one
  directory tree and describe the same images, and serving one for the other
  produces a plausible number rather than an error.
"""

import torch

from visbench.cache import PREFIX_DIR, FeatureCache, PrefixCache, make_key, make_prefix_key

BACKBONE_KEY = "dinov2/dinov2_vits14/224/7764ea0f912e"


class TestKeys:
    def test_a_prefix_key_names_the_cut(self):
        """Two runs unfreezing different depths cut in different places, and
        their prefixes are not interchangeable."""
        assert make_prefix_key("img", BACKBONE_KEY, 10) != make_prefix_key("img", BACKBONE_KEY, 8)

    def test_a_prefix_key_cannot_collide_with_a_feature_key(self):
        """``prefix@10`` occupies the field a layer index would, and no layer
        index can render that way — so the namespaces are disjoint by
        construction rather than by convention."""
        prefix = make_prefix_key("img", BACKBONE_KEY, 10)
        for layer in (None, 0, 10, -1):
            for pooling in ("cls", "mean"):
                assert prefix != make_key("img", BACKBONE_KEY, layer, pooling)

    def test_a_negative_cut_raises(self):
        """A cut is a block count, and a negative one would key an entry that
        no lookup could ever produce."""
        for bad in (-1, "10", 1.0, True):
            try:
                make_prefix_key("img", BACKBONE_KEY, bad)
            except ValueError:
                continue
            raise AssertionError(f"cut={bad!r} should have raised")


class TestStore:
    def test_round_trip_preserves_the_tensor_and_the_grid(self, tmp_path):
        cache = PrefixCache(root=tmp_path)
        tokens = torch.randn(257, 384)
        key = make_prefix_key("img", BACKBONE_KEY, 10)

        cache.put(key, tokens, (16, 16))
        loaded = cache.get(key)

        assert loaded is not None
        got, grid = loaded
        assert torch.equal(got, tokens)
        assert grid == (16, 16)

    def test_the_grid_travels_with_the_activation(self, tmp_path):
        """Token count gives the number of patches, not their arrangement. A
        non-square input reconstructed from the count alone would be the wrong
        grid, which misaligns every feature map against its target."""
        cache = PrefixCache(root=tmp_path)
        key = make_prefix_key("img", BACKBONE_KEY, 10)
        cache.put(key, torch.randn(129, 384), (16, 8))
        loaded = cache.get(key)
        assert loaded is not None
        assert loaded[1] == (16, 8)

    def test_a_miss_returns_none(self, tmp_path):
        assert PrefixCache(root=tmp_path).get(make_prefix_key("nope", BACKBONE_KEY, 10)) is None

    def test_disabled_never_hits(self, tmp_path):
        """``enabled=False`` is how a run measures its own cost without a
        second code path."""
        cache = PrefixCache(root=tmp_path, enabled=False)
        key = make_prefix_key("img", BACKBONE_KEY, 10)
        cache.put(key, torch.randn(257, 384), (16, 16))
        assert cache.get(key) is None

    def test_a_corrupt_entry_is_a_miss_not_a_hit(self, tmp_path):
        """Self-healing: the put() that follows overwrites it."""
        cache = PrefixCache(root=tmp_path)
        key = make_prefix_key("img", BACKBONE_KEY, 10)
        cache.put(key, torch.randn(257, 384), (16, 16))
        cache._path(key).write_bytes(b"not a tensor")
        assert cache.get(key) is None

    def test_it_stores_on_cpu(self, tmp_path):
        """So a cache written on a GPU box is readable on a laptop."""
        cache = PrefixCache(root=tmp_path)
        key = make_prefix_key("img", BACKBONE_KEY, 10)
        cache.put(key, torch.randn(257, 384), (16, 16))
        loaded = cache.get(key)
        assert loaded is not None
        assert loaded[0].device.type == "cpu"


class TestItStaysOutOfTheFeatureCache:
    """The prefix store nests under the feature cache root for one directory
    to configure and clear. That convenience is only safe while the two never
    read each other's entries."""

    def test_prefix_entries_are_not_counted_as_features(self, tmp_path):
        features = FeatureCache(root=tmp_path)
        prefix = PrefixCache(root=tmp_path)

        prefix.put(make_prefix_key("img", BACKBONE_KEY, 10), torch.randn(257, 384), (16, 16))

        assert features.stats()["entries"] == 0, "a prefix was counted as a feature"
        assert prefix.stats()["entries"] == 1

    def test_it_lives_in_the_reserved_subdirectory(self, tmp_path):
        prefix = PrefixCache(root=tmp_path)
        assert prefix.root == tmp_path / PREFIX_DIR

    def test_clearing_features_leaves_prefixes_alone(self, tmp_path):
        """A whole-root rmtree would delete them while reporting a count that
        excluded them — the caller would be told it removed fewer things than
        it did."""
        features = FeatureCache(root=tmp_path)
        prefix = PrefixCache(root=tmp_path)
        key = make_prefix_key("img", BACKBONE_KEY, 10)
        prefix.put(key, torch.randn(257, 384), (16, 16))
        features.put(
            make_key("img", BACKBONE_KEY, None, "mean"),
            {"dense": torch.randn(8, 4, 4), "pooled": torch.randn(8), "grid_hw": (4, 4)},
        )

        removed = features.clear()

        assert removed == 1
        assert features.stats()["entries"] == 0
        assert prefix.get(key) is not None, "clearing features deleted a prefix"

    def test_the_prefix_cache_clears_itself(self, tmp_path):
        prefix = PrefixCache(root=tmp_path)
        key = make_prefix_key("img", BACKBONE_KEY, 10)
        prefix.put(key, torch.randn(257, 384), (16, 16))

        assert prefix.clear() == 1
        assert prefix.get(key) is None

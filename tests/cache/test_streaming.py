"""Reading cached features a batch at a time.

``extract_dataset`` stacks everything; ``materialise`` returns a reader that
loads from disk on demand. The two share one extraction path, so most of what
matters here is that they cannot disagree: same features, same cache entries,
and — the one that would fail silently — targets still paired with the right
image after the loader shuffles.
"""

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from visbench.cache import CachedFeatures, FeatureCache
from visbench.data import DenseFolderDataset


@pytest.fixture
def dense_dataset(tmp_path):
    """Ten images whose depth is a distinct constant, so pairing is checkable."""
    root = tmp_path / "data"
    (root / "images").mkdir(parents=True)
    (root / "depths").mkdir(parents=True)
    for index in range(10):
        depth = np.full((32, 32), 1.0 + index, dtype=np.float32)
        Image.fromarray(np.full((32, 32, 3), index * 20, dtype=np.uint8)).save(
            root / "images" / f"s{index:02d}.png"
        )
        np.save(root / "depths" / f"s{index:02d}.npy", depth)
    return DenseFolderDataset(root, image_size=32)


@pytest.fixture
def cache(tmp_path):
    return FeatureCache(root=tmp_path / "cache")


# -- the two paths must agree -------------------------------------------------


class TestEquivalenceWithExtractDataset:
    def test_same_features_single_layer(self, cache, fake_vit, dense_dataset):
        stacked = cache.extract_dataset(fake_vit, dense_dataset, keep="dense", pooling="mean")
        streamed = cache.materialise(fake_vit, dense_dataset, pooling="mean")

        for index in range(len(dense_dataset)):
            assert torch.equal(streamed[index], stacked["dense"][index])

    def test_same_features_multi_layer(self, cache, fake_vit, dense_dataset):
        stacked = cache.extract_dataset(
            fake_vit, dense_dataset, keep="dense", pooling="mean", layers=[3, 7]
        )
        streamed = cache.materialise(fake_vit, dense_dataset, pooling="mean", layers=[3, 7])

        for index in range(len(dense_dataset)):
            for depth, layer in enumerate(streamed[index]):
                assert torch.equal(layer, stacked["dense_layers"][depth][index])

    def test_they_share_cache_entries(self, cache, fake_vit, dense_dataset):
        """One extraction serves both readers — the cache's whole promise."""
        cache.extract_dataset(fake_vit, dense_dataset, keep="dense", pooling="mean")
        assert fake_vit.call_count == 1

        cache.materialise(fake_vit, dense_dataset, pooling="mean")
        assert fake_vit.call_count == 1, "materialise re-ran the backbone"

    def test_materialise_then_extract_is_a_hit(self, cache, fake_vit, dense_dataset):
        cache.materialise(fake_vit, dense_dataset, pooling="mean")
        cache.extract_dataset(fake_vit, dense_dataset, keep="dense", pooling="mean")
        assert fake_vit.call_count == 1

    def test_re_materialising_is_a_hit(self, cache, fake_vit, dense_dataset):
        cache.materialise(fake_vit, dense_dataset, pooling="mean", layers=[2, 5])
        cache.materialise(fake_vit, dense_dataset, pooling="mean", layers=[2, 5])
        assert fake_vit.call_count == 1


# -- what the reader hands back -----------------------------------------------


class TestReader:
    def test_length_and_shapes(self, cache, fake_vit, dense_dataset):
        reader = cache.materialise(fake_vit, dense_dataset, pooling="mean")
        assert len(reader) == 10
        assert reader[0].shape == (fake_vit.embed_dim, 4, 4)

    def test_the_batch_dimension_is_squeezed_off(self, cache, fake_vit, dense_dataset):
        """Entries are stored as (1, C, H, W); left alone, every batch would
        arrive as (B, 1, C, H, W) and no head would accept it."""
        reader = cache.materialise(fake_vit, dense_dataset, pooling="mean")
        assert reader[0].ndim == 3

    def test_multi_layer_items_are_lists(self, cache, fake_vit, dense_dataset):
        reader = cache.materialise(fake_vit, dense_dataset, pooling="mean", layers=[1, 4, 9])
        item = reader[0]
        assert isinstance(item, list) and len(item) == 3

    def test_channels_reports_per_layer_widths(self, cache, fake_vit, dense_dataset):
        single = cache.materialise(fake_vit, dense_dataset, pooling="mean")
        multi = cache.materialise(fake_vit, dense_dataset, pooling="mean", layers=[1, 4])
        assert single.channels == fake_vit.embed_dim
        assert multi.channels == [fake_vit.embed_dim] * 2

    def test_targets_are_returned_alongside(self, cache, fake_vit, dense_dataset):
        reader = cache.materialise(
            fake_vit, dense_dataset, pooling="mean", targets=dense_dataset.target
        )
        features, target = reader[0]
        assert features.shape == (fake_vit.embed_dim, 4, 4)
        assert target.shape == (32, 32)

    def test_without_targets_only_features_come_back(self, cache, fake_vit, dense_dataset):
        reader = cache.materialise(fake_vit, dense_dataset, pooling="mean")
        assert isinstance(reader[0], torch.Tensor)

    def test_a_deleted_entry_explains_itself(self, cache, fake_vit, dense_dataset):
        """Clearing the cache under a running job should not surface as a
        confusing None somewhere downstream."""
        reader = cache.materialise(fake_vit, dense_dataset, pooling="mean")
        cache.clear()
        with pytest.raises(RuntimeError, match="gone"):
            reader[0]


# -- pairing, which is the thing that would fail silently ---------------------


class TestPairing:
    def test_each_target_belongs_to_its_own_image(self, cache, fake_vit, dense_dataset):
        """Depth i is the constant 1+i, so a mispairing is arithmetic, not luck."""
        reader = cache.materialise(
            fake_vit, dense_dataset, pooling="mean", targets=dense_dataset.target
        )
        for index in range(len(dense_dataset)):
            _, target = reader[index]
            assert target.mean().item() == pytest.approx(1.0 + index)

    def test_pairing_survives_shuffling(self, cache, fake_vit, dense_dataset):
        """The reason features and targets are read by one index rather than
        stacked separately: a shuffled loader must not decouple them."""
        reader = cache.materialise(
            fake_vit, dense_dataset, pooling="mean", targets=dense_dataset.target
        )
        by_target = {
            round(reader[index][1].mean().item()): reader[index][0]
            for index in range(len(dense_dataset))
        }

        loader = DataLoader(reader, batch_size=3, shuffle=True, collate_fn=reader.collate)
        seen = 0
        for features, targets in loader:
            for feature, target in zip(features, targets, strict=True):
                expected = by_target[round(target.mean().item())]
                assert torch.equal(feature, expected)
                seen += 1
        assert seen == len(dense_dataset)

    def test_reader_order_matches_dataset_order(self, cache, fake_vit, dense_dataset):
        stacked = cache.extract_dataset(fake_vit, dense_dataset, keep="dense", pooling="mean")
        reader = cache.materialise(fake_vit, dense_dataset, pooling="mean")
        assert torch.equal(reader[3], stacked["dense"][3])


# -- through a DataLoader -----------------------------------------------------


class TestWithDataLoader:
    def test_batches_have_the_batch_dimension_back(self, cache, fake_vit, dense_dataset):
        reader = cache.materialise(
            fake_vit, dense_dataset, pooling="mean", targets=dense_dataset.target
        )
        loader = DataLoader(reader, batch_size=4, collate_fn=reader.collate)
        features, targets = next(iter(loader))
        assert features.shape == (4, fake_vit.embed_dim, 4, 4)
        assert targets.shape == (4, 32, 32)

    def test_multi_layer_batches_keep_their_layer_structure(self, cache, fake_vit, dense_dataset):
        """The default collation would transpose a list-of-layers into something
        no head accepts."""
        reader = cache.materialise(
            fake_vit, dense_dataset, pooling="mean", layers=[2, 6], targets=dense_dataset.target
        )
        loader = DataLoader(reader, batch_size=5, collate_fn=reader.collate)
        features, targets = next(iter(loader))

        assert isinstance(features, list) and len(features) == 2
        assert all(layer.shape == (5, fake_vit.embed_dim, 4, 4) for layer in features)
        assert targets.shape == (5, 32, 32)

    def test_shuffling_actually_reorders(self, cache, fake_vit, dense_dataset):
        reader = cache.materialise(
            fake_vit, dense_dataset, pooling="mean", targets=dense_dataset.target
        )
        generator = torch.Generator().manual_seed(0)
        loader = DataLoader(
            reader, batch_size=10, shuffle=True, collate_fn=reader.collate, generator=generator
        )
        first = next(iter(loader))[1].mean(dim=(1, 2))
        ordered = torch.arange(1.0, 11.0)
        assert not torch.equal(first, ordered)
        assert sorted(first.tolist()) == pytest.approx(ordered.tolist())

    def test_the_seed_governs_the_order(self, cache, fake_vit, dense_dataset):
        """Shuffling must stay inside the caller's seed, or the seed recorded
        next to the metrics stops describing the run."""
        reader = cache.materialise(
            fake_vit, dense_dataset, pooling="mean", targets=dense_dataset.target
        )

        def order(seed):
            loader = DataLoader(
                reader,
                batch_size=10,
                shuffle=True,
                collate_fn=reader.collate,
                generator=torch.Generator().manual_seed(seed),
            )
            return next(iter(loader))[1].mean(dim=(1, 2)).tolist()

        assert order(0) == order(0)
        assert order(0) != order(1)

    def test_without_targets_a_batch_is_just_features(self, cache, fake_vit, dense_dataset):
        reader = cache.materialise(fake_vit, dense_dataset, pooling="mean")
        loader = DataLoader(reader, batch_size=2, collate_fn=reader.collate)
        assert next(iter(loader)).shape == (2, fake_vit.embed_dim, 4, 4)


# -- validation ---------------------------------------------------------------


def test_empty_dataset_raises(cache, fake_vit):
    with pytest.raises(ValueError, match="empty dataset"):
        cache.materialise(fake_vit, [])


def test_layer_and_layers_together_are_refused(cache, fake_vit, dense_dataset):
    """The shared plan means both entry points reject this identically."""
    with pytest.raises(ValueError, match="not both"):
        cache.materialise(fake_vit, dense_dataset, layer=3, layers=[3, 7])


def test_it_is_a_torch_dataset(cache, fake_vit, dense_dataset):
    """So it can be handed to a DataLoader with num_workers, which is what the
    disk reads want."""
    from torch.utils.data import Dataset

    assert isinstance(cache.materialise(fake_vit, dense_dataset, pooling="mean"), Dataset)
    assert isinstance(cache.materialise(fake_vit, dense_dataset, pooling="mean"), CachedFeatures)

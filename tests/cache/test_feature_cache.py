"""Contract tests for the feature cache.

The cache is the piece most likely to fail silently, so these matter more than
their line count suggests:

* A second extraction of the same image hits the cache; the backbone forward
  runs exactly once (asserted via the call-counting fake backbone).
* Changing any key component — image, backbone key, layer, pooling — misses.
* Two identical images under different filenames share one entry.
* ``enabled=False`` always misses without changing call sites.
* An interrupted write leaves no readable entry (atomic writes).
"""

import pytest
import torch
from PIL import Image

from visbench.cache import FeatureCache, hash_image, make_key
from visbench.types import Pooling
from visbench.utils.image import load_image


def make_cache(tmp_path, **kwargs):
    return FeatureCache(root=tmp_path / "cache", **kwargs)


def features(value: float = 1.0):
    return {
        "dense": torch.full((1, 8, 4, 4), value),
        "pooled": torch.full((1, 8), value),
        "grid_hw": (4, 4),
    }


# -- round trip --------------------------------------------------------------


def test_put_then_get_round_trips(tmp_path):
    cache = make_cache(tmp_path)
    cache.put("k", features(3.0))
    got = cache.get("k")

    assert torch.equal(got["dense"], features(3.0)["dense"])
    assert torch.equal(got["pooled"], features(3.0)["pooled"])
    assert got["grid_hw"] == (4, 4)


def test_grid_hw_survives_as_a_tuple(tmp_path):
    """Stored as a list for weights_only=True; a task unpacking it needs a tuple."""
    cache = make_cache(tmp_path)
    cache.put("k", features())
    assert isinstance(cache.get("k")["grid_hw"], tuple)


def test_miss_returns_none(tmp_path):
    assert make_cache(tmp_path).get("absent") is None


def test_entries_are_stored_on_cpu(tmp_path):
    """A cache written on a GPU box must be readable on a laptop."""
    cache = make_cache(tmp_path)
    cache.put("k", features())
    assert cache.get("k")["dense"].device.type == "cpu"


# -- the central claim: one forward pass per image per backbone --------------


def test_second_extraction_hits_cache(tmp_path, fake_vit, solid_images):
    cache = make_cache(tmp_path)

    first = cache.extract_dataset(fake_vit, solid_images)
    assert fake_vit.call_count == 1

    second = cache.extract_dataset(fake_vit, solid_images)
    assert fake_vit.call_count == 1, "second pass re-ran the backbone"
    assert torch.equal(first["pooled"], second["pooled"])


def test_only_missing_images_are_recomputed(tmp_path, fake_vit, solid_images):
    cache = make_cache(tmp_path)
    cache.extract_dataset(fake_vit, solid_images[:2], batch_size=8)
    assert fake_vit.call_count == 1

    cache.extract_dataset(fake_vit, solid_images, batch_size=8)
    # One more batch, containing only the two new images.
    assert fake_vit.call_count == 2


def test_get_or_compute_computes_once(tmp_path):
    cache = make_cache(tmp_path)
    calls = []

    def compute():
        calls.append(1)
        return features()

    cache.get_or_compute("k", compute)
    cache.get_or_compute("k", compute)
    assert len(calls) == 1


def test_extract_dataset_preserves_order(tmp_path, fake_vit, solid_images):
    cache = make_cache(tmp_path)
    batched = cache.extract_dataset(fake_vit, solid_images, batch_size=2)
    one_at_a_time = torch.cat(
        [
            make_cache(tmp_path / str(i)).extract_dataset(fake_vit, [img])["pooled"]
            for i, img in enumerate(solid_images)
        ]
    )
    assert torch.allclose(batched["pooled"], one_at_a_time)


def test_extract_dataset_accepts_image_label_pairs(tmp_path, fake_vit, solid_images):
    cache = make_cache(tmp_path)
    labelled = [(img, i) for i, img in enumerate(solid_images)]
    assert cache.extract_dataset(fake_vit, labelled)["pooled"].shape[0] == 4


def test_empty_dataset_raises(tmp_path, fake_vit):
    import pytest

    with pytest.raises(ValueError, match="empty dataset"):
        make_cache(tmp_path).extract_dataset(fake_vit, [])


# -- streaming and memory ----------------------------------------------------


def test_dataset_is_consumed_lazily(tmp_path, fake_vit, solid_images):
    """A dataset that decodes on access must never be fully materialised.

    This is what bounds memory on a 50k-image run: at most ``batch_size``
    images are alive at once.
    """
    pulled = []

    def generator():
        for img in solid_images:
            pulled.append(len(pulled))
            yield img

    cache = make_cache(tmp_path)
    high_water = []

    original = fake_vit.preprocess

    def spy(images):
        high_water.append(len(pulled))
        return original(images)

    fake_vit.preprocess = spy
    cache.extract_dataset(fake_vit, generator(), batch_size=2)

    # First forward happens after only the first 2 images have been pulled.
    assert high_water[0] == 2
    assert len(pulled) == 4


def test_keep_pooled_skips_dense(tmp_path, fake_vit, solid_images):
    cache = make_cache(tmp_path)
    features = cache.extract_dataset(fake_vit, solid_images, keep="pooled")

    assert "pooled" in features
    assert "dense" not in features
    assert features["grid_hw"] == (4, 4)


def test_keep_dense_skips_pooled(tmp_path, fake_vit, solid_images):
    cache = make_cache(tmp_path)
    features = cache.extract_dataset(fake_vit, solid_images, keep="dense")

    assert "dense" in features
    assert "pooled" not in features


def test_store_defaults_to_keep(tmp_path, fake_vit, solid_images):
    """Storing dense for a task that never reads it turned a 20 MB cache into 5 GB."""
    cache = make_cache(tmp_path)
    cache.extract_dataset(fake_vit, solid_images, keep="pooled")

    entry = torch.load(next((tmp_path / "cache").rglob("*.pt")), weights_only=True)
    assert "pooled" in entry
    assert "dense" not in entry


def test_incomplete_entry_is_a_miss_not_a_broken_hit(tmp_path, fake_vit, solid_images):
    """A leaner cache must cost re-extraction, never a dict with a missing key."""
    cache = make_cache(tmp_path)
    cache.extract_dataset(fake_vit, solid_images, keep="pooled")
    assert fake_vit.call_count == 1

    both = cache.extract_dataset(fake_vit, solid_images, keep="both")
    assert fake_vit.call_count == 2, "pooled-only entries must not satisfy keep='both'"
    assert both["dense"].shape[0] == 4
    assert both["pooled"].shape[0] == 4


def test_store_both_serves_a_later_pooled_only_run(tmp_path, fake_vit, solid_images):
    """Storing more than you keep is allowed, and pays off on the next task."""
    cache = make_cache(tmp_path)
    cache.extract_dataset(fake_vit, solid_images, keep="pooled", store="both")
    assert fake_vit.call_count == 1

    cache.extract_dataset(fake_vit, solid_images, keep="both")
    assert fake_vit.call_count == 1


def test_unknown_store_raises(tmp_path, fake_vit, solid_images):
    with pytest.raises(ValueError, match="store must be one of"):
        make_cache(tmp_path).extract_dataset(fake_vit, solid_images, store="everything")


# -- resolving a cached image without decoding it ----------------------------


class TestIdentityMemo:
    """A cached run must not re-read the images.

    Measured on Imagenette before this existed: a fully cached run still cost
    ~113 s, almost all of it decoding 13,394 JPEGs to compute hashes the cache
    had already seen.
    """

    @pytest.fixture
    def folder(self, tmp_path):
        from visbench.data import ImageFolderDataset

        root = tmp_path / "imgs" / "cls"
        root.mkdir(parents=True)
        for i in range(4):
            Image.new("RGB", (64, 64), (i * 50, 10, 10)).save(root / f"{i}.png")
        return ImageFolderDataset(tmp_path / "imgs")

    def decode_count(self, monkeypatch):
        import visbench.data.image_folder as module

        calls = []
        original = module.load_image
        monkeypatch.setattr(
            module, "load_image", lambda path: (calls.append(path), original(path))[1]
        )
        return calls

    def test_cold_run_decodes(self, tmp_path, fake_vit, folder, monkeypatch):
        calls = self.decode_count(monkeypatch)
        make_cache(tmp_path).extract_dataset(fake_vit, folder, keep="pooled")
        assert len(calls) == 4

    def test_cached_run_decodes_nothing(self, tmp_path, fake_vit, folder, monkeypatch):
        cache = make_cache(tmp_path)
        cache.extract_dataset(fake_vit, folder, keep="pooled")

        calls = self.decode_count(monkeypatch)
        cache.extract_dataset(fake_vit, folder, keep="pooled")
        assert calls == [], "a fully cached run re-read the images"
        assert fake_vit.call_count == 1

    def test_results_are_identical_either_way(self, tmp_path, fake_vit, folder):
        cache = make_cache(tmp_path)
        first = cache.extract_dataset(fake_vit, folder, keep="pooled")
        second = cache.extract_dataset(fake_vit, folder, keep="pooled")
        assert torch.equal(first["pooled"], second["pooled"])

    def test_edited_file_is_not_served_from_the_memo(self, tmp_path, fake_vit, folder):
        """The memo keys on size and mtime, so changed bytes must miss."""
        from visbench.data import ImageFolderDataset

        cache = make_cache(tmp_path)
        before = cache.extract_dataset(fake_vit, folder, keep="pooled")

        target = folder.paths[0]
        Image.new("RGB", (64, 64), (7, 200, 7)).save(target)
        reloaded = ImageFolderDataset(target.parent.parent)
        after = cache.extract_dataset(fake_vit, reloaded, keep="pooled")

        assert not torch.equal(before["pooled"][0], after["pooled"][0])

    def test_copied_file_still_shares_the_entry(self, tmp_path, fake_vit, folder):
        """A new identity, same pixels: content addressing must still win.

        The memo is an optimisation over hashing, never a replacement for it.
        """
        from visbench.data import ImageFolderDataset

        cache = make_cache(tmp_path)
        cache.extract_dataset(fake_vit, folder, keep="pooled")
        entries_before = cache.stats()["entries"]

        copy_root = tmp_path / "copy" / "cls"
        copy_root.mkdir(parents=True)
        for path in folder.paths:
            copy_root.joinpath(path.name).write_bytes(path.read_bytes())

        cache.extract_dataset(fake_vit, ImageFolderDataset(tmp_path / "copy"), keep="pooled")
        assert fake_vit.call_count == 1, "identical pixels re-ran the backbone"
        assert cache.stats()["entries"] == entries_before

    def test_plain_image_list_still_works(self, tmp_path, fake_vit, solid_images):
        """No identity available: correctness must not depend on the memo."""
        cache = make_cache(tmp_path)
        first = cache.extract_dataset(fake_vit, solid_images, keep="pooled")
        second = cache.extract_dataset(fake_vit, solid_images, keep="pooled")

        assert fake_vit.call_count == 1
        assert torch.equal(first["pooled"], second["pooled"])

    def test_memo_does_not_count_as_a_feature_entry(self, tmp_path, fake_vit, folder):
        cache = make_cache(tmp_path)
        cache.extract_dataset(fake_vit, folder, keep="pooled")
        assert cache.stats()["entries"] == 4


def test_unknown_keep_raises(tmp_path, fake_vit, solid_images):
    import pytest

    with pytest.raises(ValueError, match="keep must be one of"):
        make_cache(tmp_path).extract_dataset(fake_vit, solid_images, keep="everything")


# -- key identity ------------------------------------------------------------


def test_key_components_all_affect_identity():
    base = dict(image_hash="abc", backbone_key="dinov2/vitb14/224", layer=None, pooling=Pooling.CLS)
    key = make_key(**base)

    assert make_key(**{**base, "image_hash": "def"}) != key
    assert make_key(**{**base, "backbone_key": "clip/vitb16/224"}) != key
    assert make_key(**{**base, "layer": 11}) != key
    assert make_key(**{**base, "pooling": Pooling.MEAN}) != key


def test_default_layer_is_distinct_from_layer_zero():
    """`None` and `0` must not collide, or v0.2 entries would poison v0.1 ones."""
    base = dict(image_hash="abc", backbone_key="b", pooling=Pooling.CLS)
    assert make_key(**base, layer=None) != make_key(**base, layer=0)


def test_pooling_change_misses(tmp_path, fake_vit, solid_images):
    cache = make_cache(tmp_path)
    cache.extract_dataset(fake_vit, solid_images, pooling=Pooling.CLS)
    assert fake_vit.call_count == 1

    cache.extract_dataset(fake_vit, solid_images, pooling=Pooling.MEAN)
    assert fake_vit.call_count == 2, "different pooling must not reuse the entry"


def test_backbone_key_with_separator_is_rejected():
    """A '|' in a backbone key would make the composite key ambiguous."""
    import pytest

    with pytest.raises(ValueError, match="must not contain"):
        make_key("abc", "dinov2|vitb14", None, Pooling.CLS)


# -- image hashing -----------------------------------------------------------


def test_identical_images_share_entry(tmp_path, solid_images):
    red = solid_images[0]
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    red.save(tmp_path / "a" / "first.png")
    red.save(tmp_path / "b" / "second.png")

    assert hash_image(load_image(tmp_path / "a" / "first.png")) == hash_image(
        load_image(tmp_path / "b" / "second.png")
    )


def test_different_images_hash_differently(solid_images):
    assert len({hash_image(img) for img in solid_images}) == len(solid_images)


def test_transposed_dimensions_hash_differently():
    """Raw bytes alone cannot tell a 2x3 image from a 3x2 one."""
    assert hash_image(Image.new("RGB", (2, 3))) != hash_image(Image.new("RGB", (3, 2)))


def test_hash_accepts_tensors():
    assert hash_image(torch.zeros(3, 4, 4)) != hash_image(torch.ones(3, 4, 4))


# -- disabled cache ----------------------------------------------------------


def test_disabled_cache_always_misses(tmp_path):
    cache = make_cache(tmp_path, enabled=False)
    cache.put("k", features())
    assert cache.get("k") is None


def test_disabled_cache_still_extracts(tmp_path, fake_vit, solid_images):
    """enabled=False changes performance, never call sites or results."""
    disabled = make_cache(tmp_path, enabled=False)
    enabled = make_cache(tmp_path, enabled=True)

    without = disabled.extract_dataset(fake_vit, solid_images)
    with_cache = enabled.extract_dataset(fake_vit, solid_images)
    assert torch.allclose(without["pooled"], with_cache["pooled"])


def test_disabled_cache_writes_nothing(tmp_path):
    cache = make_cache(tmp_path, enabled=False)
    cache.put("k", features())
    assert not (tmp_path / "cache").exists()


# -- durability --------------------------------------------------------------


def test_partial_write_is_not_a_hit(tmp_path):
    """A truncated file must read as a miss, never as corrupt features."""
    cache = make_cache(tmp_path)
    cache.put("k", features())

    entry = next((tmp_path / "cache").rglob("*.pt"))
    entry.write_bytes(entry.read_bytes()[: len(entry.read_bytes()) // 2])

    assert cache.get("k") is None


def test_failed_write_leaves_no_temp_file(tmp_path, monkeypatch):
    cache = make_cache(tmp_path)

    def explode(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr("visbench.cache.feature_cache.torch.save", explode)
    try:
        cache.put("k", features())
    except RuntimeError:
        pass

    assert list((tmp_path / "cache").rglob("*.tmp")) == []
    assert cache.get("k") is None


def test_corrupt_entry_self_heals(tmp_path):
    cache = make_cache(tmp_path)
    cache.put("k", features(1.0))
    next((tmp_path / "cache").rglob("*.pt")).write_bytes(b"garbage")

    cache.put("k", features(2.0))
    assert cache.get("k")["pooled"][0, 0] == 2.0


# -- maintenance -------------------------------------------------------------


def test_clear_by_backbone_leaves_others(tmp_path):
    cache = make_cache(tmp_path)
    cache.put(make_key("img", "dinov2/vitb14/224", None, Pooling.CLS), features())
    cache.put(make_key("img", "clip/vitb16/224", None, Pooling.CLS), features())

    assert cache.clear("dinov2/vitb14/224") == 1
    assert cache.stats()["entries"] == 1


def test_clear_all(tmp_path):
    cache = make_cache(tmp_path)
    cache.put("a", features())
    cache.put("b", features())
    assert cache.clear() == 2
    assert cache.stats()["entries"] == 0


def test_stats_counts_hits_and_misses(tmp_path):
    cache = make_cache(tmp_path)
    cache.get("absent")
    cache.put("k", features())
    cache.get("k")

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["entries"] == 1
    assert stats["bytes"] > 0


class TestFeatureModeKeying:
    """The modes produce different dense tensors from one forward pass."""

    def test_modes_do_not_share_entries(self, tmp_path, fake_vit, solid_images):
        """dense_cls_broadcast has twice the channels; serving one for the other
        would be a shape error at best and a wrong feature map at worst."""
        cache = make_cache(tmp_path)
        cache.extract_dataset(fake_vit, solid_images, keep="dense")
        assert fake_vit.call_count == 1

        cache.extract_dataset(
            fake_vit, solid_images, keep="dense", feature_mode="dense_cls_broadcast"
        )
        assert fake_vit.call_count == 2

    def test_broadcast_survives_a_cache_hit(self, tmp_path, fake_vit, solid_images):
        cache = make_cache(tmp_path)
        first = cache.extract_dataset(
            fake_vit, solid_images, keep="dense", feature_mode="dense_cls_broadcast"
        )
        second = cache.extract_dataset(
            fake_vit, solid_images, keep="dense", feature_mode="dense_cls_broadcast"
        )
        assert fake_vit.call_count == 1
        assert torch.equal(first["dense"], second["dense"])
        assert first["dense"].shape[1] == 2 * fake_vit.embed_dim

    def test_cls_survives_a_cache_hit(self, tmp_path, fake_vit, solid_images):
        """The bug this guards: cls is produced by extraction but was not stored,
        so it existed on a miss and vanished on the next hit."""
        cache = make_cache(tmp_path)
        first = cache.extract_dataset(
            fake_vit, solid_images, keep="dense", feature_mode="dense_plus_cls"
        )
        assert "cls" in first

        second = cache.extract_dataset(
            fake_vit, solid_images, keep="dense", feature_mode="dense_plus_cls"
        )
        assert fake_vit.call_count == 1, "should have been a hit"
        assert "cls" in second, "cls vanished on the cache hit"
        assert torch.equal(first["cls"], second["cls"])

    def test_key_includes_the_mode(self):
        base = dict(image_hash="abc", backbone_key="b", layer=None, pooling=Pooling.CLS)
        assert make_key(**base, feature_mode="dense_only") != make_key(
            **base, feature_mode="dense_plus_cls"
        )


class TestMultiLayerCaching:
    """Each layer gets its own entry, so overlapping requests share work."""

    def test_returns_one_stack_per_layer(self, tmp_path, fake_vit, solid_images):
        cache = make_cache(tmp_path)
        features = cache.extract_dataset(fake_vit, solid_images, layers=[3, 7], keep="dense")
        assert len(features["dense_layers"]) == 2
        assert features["layer_indices"] == [3, 7]
        assert all(dense.shape[0] == len(solid_images) for dense in features["dense_layers"])

    def test_dense_is_the_deepest_layer(self, tmp_path, fake_vit, solid_images):
        cache = make_cache(tmp_path)
        features = cache.extract_dataset(fake_vit, solid_images, layers=[3, 7], keep="dense")
        assert torch.equal(features["dense"], features["dense_layers"][-1])

    def test_one_forward_pass_for_every_layer(self, tmp_path, fake_vit, solid_images):
        cache = make_cache(tmp_path)
        cache.extract_dataset(fake_vit, solid_images, layers=[1, 5, 9], keep="dense")
        assert fake_vit.call_count == 1

    def test_a_repeat_run_is_a_hit(self, tmp_path, fake_vit, solid_images):
        cache = make_cache(tmp_path)
        first = cache.extract_dataset(fake_vit, solid_images, layers=[3, 7], keep="dense")
        second = cache.extract_dataset(fake_vit, solid_images, layers=[3, 7], keep="dense")
        assert fake_vit.call_count == 1
        for a, b in zip(first["dense_layers"], second["dense_layers"], strict=True):
            assert torch.equal(a, b)

    def test_widening_the_request_reuses_the_shared_layers(self, tmp_path, fake_vit, solid_images):
        """The reason each layer is keyed separately rather than the list as a
        whole: adding a layer must not re-extract the ones already stored."""
        cache = make_cache(tmp_path)
        first = cache.extract_dataset(fake_vit, solid_images, layers=[3, 7], keep="dense")
        widened = cache.extract_dataset(fake_vit, solid_images, layers=[3, 7, 11], keep="dense")
        assert torch.equal(first["dense_layers"][0], widened["dense_layers"][0])
        assert torch.equal(first["dense_layers"][1], widened["dense_layers"][1])

    def test_a_single_layer_run_reads_what_a_multi_layer_run_stored(
        self, tmp_path, fake_vit, solid_images
    ):
        cache = make_cache(tmp_path)
        multi = cache.extract_dataset(fake_vit, solid_images, layers=[3, 7], keep="dense")
        single = cache.extract_dataset(fake_vit, solid_images, layer=7, keep="dense")
        assert fake_vit.call_count == 1, "layer 7 was already stored"
        assert torch.equal(multi["dense_layers"][1], single["dense"])

    def test_negative_indices_hit_the_same_entries(self, tmp_path, fake_vit, solid_images):
        """[-9, -5] and [3, 7] name one pair of entries on a 12-block model."""
        cache = make_cache(tmp_path)
        cache.extract_dataset(fake_vit, solid_images, layers=[3, 7], keep="dense")
        relative = cache.extract_dataset(fake_vit, solid_images, layers=[-9, -5], keep="dense")
        assert fake_vit.call_count == 1
        assert relative["layer_indices"] == [3, 7]

    def test_a_single_layer_run_returns_no_layer_keys(self, tmp_path, fake_vit, solid_images):
        cache = make_cache(tmp_path)
        features = cache.extract_dataset(fake_vit, solid_images, keep="dense")
        assert "dense_layers" not in features
        assert "layer_indices" not in features

    def test_layer_and_layers_together_are_refused(self, tmp_path, fake_vit, solid_images):
        cache = make_cache(tmp_path)
        with pytest.raises(ValueError, match="not both"):
            cache.extract_dataset(fake_vit, solid_images, layer=3, layers=[3, 7])

    def test_multi_layer_with_pooled_only_is_refused(self, tmp_path, fake_vit, solid_images):
        """Pooled comes from one layer; the rest would be extracted and dropped."""
        cache = make_cache(tmp_path)
        with pytest.raises(ValueError, match="contradiction"):
            cache.extract_dataset(fake_vit, solid_images, layers=[3, 7], keep="pooled")

    def test_pooled_comes_from_the_deepest_layer(self, tmp_path, fake_vit, solid_images):
        cache = make_cache(tmp_path)
        multi = cache.extract_dataset(fake_vit, solid_images, layers=[3, 7], keep="both")
        single = cache.extract_dataset(fake_vit, solid_images, layer=7, keep="both")
        assert torch.equal(multi["pooled"], single["pooled"])

    def test_cls_survives_a_multi_layer_hit(self, tmp_path, fake_vit, solid_images):
        cache = make_cache(tmp_path)
        first = cache.extract_dataset(
            fake_vit, solid_images, layers=[3, 7], keep="dense", feature_mode="dense_plus_cls"
        )
        second = cache.extract_dataset(
            fake_vit, solid_images, layers=[3, 7], keep="dense", feature_mode="dense_plus_cls"
        )
        assert fake_vit.call_count == 1
        assert torch.equal(first["cls"], second["cls"])

    def test_cnn_stages_of_different_shapes_stack_independently(
        self, tmp_path, fake_cnn, solid_images
    ):
        """Each layer's grid is checked against itself, not against the others —
        a CNN's stages are meant to differ."""
        cache = make_cache(tmp_path)
        fake_cnn.preprocess = lambda images: torch.stack([torch.rand(3, 64, 64) for _ in images])
        features = cache.extract_dataset(fake_cnn, solid_images, layers=[0, 1, 2], keep="dense")
        grids = [tuple(dense.shape[-2:]) for dense in features["dense_layers"]]
        assert len(set(grids)) == 3


def test_pair_dataset_is_refused(tmp_path, fake_vit):
    """It yields (image_0, image_1, geometry); unpacking would take image_0 and
    silently drop the second view and the geometry."""
    import numpy as np

    from visbench.data.pair_dataset import HomographyPairDataset

    root = tmp_path / "pairs"
    root.mkdir()
    for i in range(2):
        Image.fromarray(np.random.RandomState(i).randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(
            root / f"{i}.png"
        )

    with pytest.raises(TypeError, match="does not take a PairDataset"):
        make_cache(tmp_path).extract_dataset(fake_vit, HomographyPairDataset(root, image_size=64))

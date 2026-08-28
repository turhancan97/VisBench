"""The torchvision and Hugging Face dataset bridges.

The load-bearing tests are the ``cache_identity`` ones. Three of the four
optional ``BaseDataset`` methods fail loudly when omitted; ``cache_identity``
fails *silently* — return ``None`` and every run re-decodes every image forever
while appearing to work. So these pin that it is never ``None``, that it is
stable across calls, and that it moves when the underlying content could have.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from visbench.data import HuggingFaceDataset, TorchvisionDataset
from visbench.data.bridges import to_pil


class FakeTV:
    """A minimal map-style dataset: in-memory arrays, a ``.targets`` list."""

    def __init__(self, n: int = 9, offset: int = 0, repr_tag: str = "FakeTV(train=True)"):
        # Fill colour encodes the class, so a linear probe on mean pixel value
        # can separate them — see the alignment test.
        self._items = [
            (np.full((8, 8, 3), (i % 3) * 80 + (offset % 5), np.uint8), i % 3) for i in range(n)
        ]
        self.targets = [i % 3 for i in range(n)]
        self._repr_tag = repr_tag

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i):
        array, label = self._items[i]
        return Image.fromarray(array), label

    def __repr__(self) -> str:
        return self._repr_tag


# -- to_pil ------------------------------------------------------------------


def test_to_pil_accepts_the_common_shapes():
    assert to_pil(Image.new("L", (4, 4))).mode == "RGB"
    assert to_pil(torch.rand(3, 5, 6)).size == (6, 5)  # CHW float -> HWC RGB
    assert to_pil(np.zeros((4, 4), np.uint8)).size == (4, 4)  # HW grey
    assert to_pil(np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)).mode == "RGB"


def test_to_pil_is_deterministic():
    array = np.random.randint(0, 255, (6, 6, 3), dtype=np.uint8)
    assert np.array_equal(np.asarray(to_pil(array)), np.asarray(to_pil(array)))


# -- TorchvisionDataset ----------------------------------------------------


def test_it_reads_length_labels_and_items():
    dataset = TorchvisionDataset(FakeTV(), split="test")
    assert len(dataset) == 9
    assert dataset.labels() == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    image, label = dataset[4]
    assert isinstance(image, Image.Image) and label == 1


def test_labels_do_not_decode_an_image():
    class Exploding(FakeTV):
        def __getitem__(self, i):
            raise AssertionError("labels() must not touch __getitem__")

    assert TorchvisionDataset(Exploding()).labels() == [0, 1, 2, 0, 1, 2, 0, 1, 2]


def test_cache_identity_is_never_none_and_is_stable():
    dataset = TorchvisionDataset(FakeTV())
    token = dataset.cache_identity(0)
    assert token is not None
    assert dataset.cache_identity(0) == token


def test_cache_identity_differs_between_different_datasets():
    a = TorchvisionDataset(FakeTV(repr_tag="A"))
    b = TorchvisionDataset(FakeTV(repr_tag="B"))
    assert a.cache_identity(0) != b.cache_identity(0)


def test_cache_identity_uses_the_file_path_for_the_imagefolder_family(tmp_path):
    (tmp_path / "cat").mkdir()
    path = tmp_path / "cat" / "a.png"
    Image.new("RGB", (8, 8)).save(path)

    class FakeImageFolder:
        samples = [(str(path), 0)]

        def __len__(self):
            return 1

        def __getitem__(self, i):
            return Image.open(self.samples[i][0]), 0

        def __repr__(self):
            return "FakeImageFolder"

    dataset = TorchvisionDataset(FakeImageFolder())
    identity = dataset.cache_identity(0)
    assert str(path.resolve()) in identity
    assert str(path.stat().st_size) in identity


def test_fingerprint_moves_with_the_labels():
    base = TorchvisionDataset(FakeTV())
    relabelled = TorchvisionDataset(FakeTV(), labels=[9] * 9)
    assert base.fingerprint() != relabelled.fingerprint()
    assert base.fingerprint() == TorchvisionDataset(FakeTV()).fingerprint()


def test_subset_and_balanced_subset():
    dataset = TorchvisionDataset(FakeTV(n=12))
    short = dataset.subset(3)
    assert len(short) == 3 and short.labels() == [0, 1, 2]
    assert short.fingerprint() != dataset.fingerprint()

    balanced = dataset.balanced_subset(1)
    assert len(balanced) == 3 and sorted(balanced.labels()) == [0, 1, 2]


def test_subset_reindexes_cache_identity_too():
    dataset = TorchvisionDataset(FakeTV(n=12))
    picked = dataset.subset([5, 2])
    # item 0 of the subset is source row 5 of the original
    assert picked.cache_identity(0) == dataset.cache_identity(5)


def test_describe_records_the_source():
    info = TorchvisionDataset(FakeTV(), split="test").describe()
    assert info["dataset_source"] == "torchvision:FakeTV"
    assert info["num_classes"] == 3


def test_it_falls_back_to_reading_labels_when_no_targets_attr():
    class NoTargets(FakeTV):
        def __init__(self):
            super().__init__()
            del self.targets

    assert TorchvisionDataset(NoTargets()).labels() == [0, 1, 2, 0, 1, 2, 0, 1, 2]


def test_a_non_map_style_object_is_refused():
    with pytest.raises(TypeError, match="map-style"):
        TorchvisionDataset(object())


def test_it_threads_features_and_labels_through_a_probe_in_step():
    """The bridge's job is index-order alignment; this is the check for it.

    Fill value = i and label = i % 3, so mean-pooled pixel colour separates the
    classes. If the bridge paired any image with another item's label the probe
    could not reach top1 > 0.5.
    """
    from visbench.tasks.high_level.classification import ClassificationTask

    torch.manual_seed(0)
    dataset = TorchvisionDataset(FakeTV(n=30), split="train")
    feats = torch.stack(
        [
            torch.tensor(np.asarray(dataset[i][0]), dtype=torch.float32).mean(dim=(0, 1))
            for i in range(len(dataset))
        ]
    )
    task = ClassificationTask(epochs=50, lr=0.1)
    task.fit({"pooled": feats}, dataset.labels())
    assert task.evaluate({"pooled": feats}, dataset.labels())["top1"] > 0.5


# -- HuggingFaceDataset --------------------------------------------------


@pytest.fixture
def hf_dataset():
    datasets = pytest.importorskip("datasets")
    rng = np.random.default_rng(0)
    rows = {
        "image": [
            Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)) for _ in range(9)
        ],
        "label": [i % 3 for i in range(9)],
    }
    features = datasets.Features(
        {"image": datasets.Image(), "label": datasets.ClassLabel(num_classes=3)}
    )
    return datasets.Dataset.from_dict(rows, features=features, split="test")


def test_hf_reads_columns_without_decoding(hf_dataset):
    dataset = HuggingFaceDataset(hf_dataset)
    assert len(dataset) == 9
    assert dataset.labels() == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    image, label = dataset[3]
    assert isinstance(image, Image.Image) and label == 0


def test_hf_cache_identity_is_the_fingerprint_plus_index(hf_dataset):
    dataset = HuggingFaceDataset(hf_dataset)
    assert dataset.cache_identity(0) == f"{hf_dataset._fingerprint}|0"
    assert dataset.cache_identity(0) is not None


def test_hf_cache_identity_moves_when_the_dataset_is_transformed(hf_dataset):
    a = HuggingFaceDataset(hf_dataset)
    b = HuggingFaceDataset(hf_dataset.shuffle(seed=1))
    assert a.cache_identity(0) != b.cache_identity(0)


def test_hf_autodetects_the_columns(hf_dataset):
    dataset = HuggingFaceDataset(hf_dataset)
    assert dataset.image_column == "image"
    assert dataset.label_column == "label"


def test_hf_rejects_a_bad_column(hf_dataset):
    with pytest.raises(ValueError, match="not in"):
        HuggingFaceDataset(hf_dataset, image_column="nope")


def test_hf_rejects_a_datasetdict():
    datasets = pytest.importorskip("datasets")
    dd = datasets.DatasetDict({"train": datasets.Dataset.from_dict({"x": [1]})})
    with pytest.raises(TypeError, match="single datasets.Dataset"):
        HuggingFaceDataset(dd)


def test_hf_subset_and_describe(hf_dataset):
    dataset = HuggingFaceDataset(hf_dataset)
    short = dataset.subset([4, 1])
    assert short.labels() == [1, 1]
    assert short.cache_identity(0) == dataset.cache_identity(4)
    assert dataset.describe()["dataset_source"].startswith("hf:")

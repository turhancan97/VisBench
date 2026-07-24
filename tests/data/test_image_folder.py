"""ImageFolderDataset — the v0.1 data path."""

import pytest
from PIL import Image

from visbench.data import ImageFolderDataset


@pytest.fixture
def folder(tmp_path):
    """root/<class>/<image>, deliberately created out of alphabetical order."""
    for class_name, count in [("dog", 2), ("cat", 3)]:
        directory = tmp_path / class_name
        directory.mkdir()
        for i in reversed(range(count)):
            Image.new("RGB", (32, 32), (i * 40, 0, 0)).save(directory / f"{i:02d}.png")
    return tmp_path


def test_indexes_classes_alphabetically(folder):
    dataset = ImageFolderDataset(folder)
    assert dataset.classes == ["cat", "dog"]
    assert len(dataset) == 5


def test_file_order_is_sorted_not_filesystem_order(folder):
    """Cached features are matched to labels by index, across machines."""
    dataset = ImageFolderDataset(folder)
    names = [path.name for path in dataset.paths]
    assert names == sorted(names[:3]) + sorted(names[3:])
    assert dataset.labels() == [0, 0, 0, 1, 1]


def test_getitem_returns_pil_and_label(folder):
    image, label = ImageFolderDataset(folder)[0]
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert label == 0


def test_labels_does_not_open_images(folder, monkeypatch):
    """Reading labels must not defeat the cache by decoding every file."""
    import visbench.data.image_folder as module

    monkeypatch.setattr(module, "load_image", lambda path: pytest.fail("labels() opened an image"))
    assert ImageFolderDataset(folder).labels() == [0, 0, 0, 1, 1]


def test_iteration_is_lazy(folder, monkeypatch):
    """Only what is consumed gets decoded — this is what bounds cache memory."""
    import visbench.data.image_folder as module

    opened = []
    original = module.load_image

    def counting(path):
        opened.append(path)
        return original(path)

    monkeypatch.setattr(module, "load_image", counting)

    dataset = ImageFolderDataset(folder)
    iterator = iter(dataset)
    assert opened == [], "constructing the dataset decoded images"

    next(iterator)
    assert len(opened) == 1, "one step pulled more than one image"


def test_unlabeled_flat_folder(tmp_path):
    for i in range(3):
        Image.new("RGB", (32, 32)).save(tmp_path / f"{i}.png")

    dataset = ImageFolderDataset(tmp_path, labeled=False)
    assert len(dataset) == 3
    assert dataset.classes == []
    assert dataset.labels() == [None, None, None]


def test_extensions_are_filtered(tmp_path):
    (tmp_path / "cls").mkdir()
    Image.new("RGB", (32, 32)).save(tmp_path / "cls" / "keep.png")
    (tmp_path / "cls" / "notes.txt").write_text("ignore me")

    assert len(ImageFolderDataset(tmp_path)) == 1


def test_missing_root_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        ImageFolderDataset(tmp_path / "absent")


def test_labeled_folder_without_subdirs_explains_itself(tmp_path):
    Image.new("RGB", (32, 32)).save(tmp_path / "loose.png")
    with pytest.raises(ValueError, match="labeled=False"):
        ImageFolderDataset(tmp_path)


def test_empty_folder_raises(tmp_path):
    (tmp_path / "cls").mkdir()
    with pytest.raises(ValueError, match="No images"):
        ImageFolderDataset(tmp_path)


def test_describe_feeds_the_result_record(folder):
    described = ImageFolderDataset(folder, split="val").describe()
    assert described["split"] == "val"
    assert described["size"] == 5
    assert described["num_classes"] == 2

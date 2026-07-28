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
    assert described["dataset_size"] == 5
    assert described["num_classes"] == 2
    assert described["dataset_fingerprint"]


class TestFingerprint:
    """What the fingerprint must and must not distinguish.

    Its job is telling datasets apart in a result record, cheaply. It reads
    ``stat()`` only — fingerprinting file contents would repeat, on every run,
    the I/O the feature cache exists to avoid.
    """

    def test_is_stable_across_instances(self, folder):
        assert ImageFolderDataset(folder).fingerprint() == ImageFolderDataset(folder).fingerprint()

    def test_does_not_decode_images(self, folder, monkeypatch):
        import visbench.data.image_folder as module

        monkeypatch.setattr(
            module, "load_image", lambda path: pytest.fail("fingerprint decoded an image")
        )
        assert ImageFolderDataset(folder).fingerprint()

    def test_added_image_changes_it(self, folder):
        before = ImageFolderDataset(folder).fingerprint()
        Image.new("RGB", (32, 32), (7, 7, 7)).save(folder / "cat" / "new.png")
        assert ImageFolderDataset(folder).fingerprint() != before

    def test_removed_image_changes_it(self, folder):
        before = ImageFolderDataset(folder).fingerprint()
        next((folder / "cat").glob("*.png")).unlink()
        assert ImageFolderDataset(folder).fingerprint() != before

    def test_relabelling_changes_it(self, folder):
        """Same images, different class assignment, is a different dataset."""
        before = ImageFolderDataset(folder).fingerprint()
        moved = next((folder / "cat").glob("*.png"))
        moved.rename(folder / "dog" / moved.name)
        assert ImageFolderDataset(folder).fingerprint() != before

    def test_replacing_with_a_different_size_changes_it(self, folder):
        before = ImageFolderDataset(folder).fingerprint()
        target = next((folder / "cat").glob("*.png"))
        Image.new("RGB", (256, 256), (1, 2, 3)).save(target)
        assert ImageFolderDataset(folder).fingerprint() != before

    def test_split_changes_it(self, folder):
        assert (
            ImageFolderDataset(folder, split="train").fingerprint()
            != ImageFolderDataset(folder, split="val").fingerprint()
        )

    def test_touching_a_file_does_not_change_it(self, folder):
        """mtime is not content; re-copying a dataset must not invalidate records."""
        before = ImageFolderDataset(folder).fingerprint()
        next((folder / "cat").glob("*.png")).touch()
        assert ImageFolderDataset(folder).fingerprint() == before

    def test_base_default_is_none_not_a_lie(self):
        """A subclass with no cheap fingerprint reports nothing, not something wrong."""
        from visbench.data.base import BaseDataset

        class Minimal(BaseDataset):
            def __len__(self):
                return 0

            def __getitem__(self, index):
                raise IndexError

        assert Minimal().fingerprint() is None


class TestBalancedSubset:
    """Shortening a labelled split without collapsing it to one class.

    The file list is grouped by class, so ``subset(n)`` — a prefix — takes every
    image from class 0. A single-class retrieval or classification run then
    scores 1.0 while measuring nothing, which looks like a triumph rather than a
    bug. Both examples that needed this carried their own copy of the fix.
    """

    def test_it_takes_n_from_each_class(self, folder):
        dataset = ImageFolderDataset(folder)
        limited = dataset.balanced_subset(1)
        assert len(limited) == len(dataset.classes)
        assert sorted(limited.labels()) == list(range(len(dataset.classes)))

    def test_a_prefix_would_not_have(self, folder):
        """The behaviour this exists to avoid, stated as a test."""
        dataset = ImageFolderDataset(folder)
        assert len(set(dataset.subset(1).labels())) == 1

    def test_it_clamps_rather_than_raising(self, folder):
        dataset = ImageFolderDataset(folder)
        assert len(dataset.balanced_subset(10_000)) == len(dataset)

    def test_it_leaves_the_original_alone(self, folder):
        dataset = ImageFolderDataset(folder)
        before = len(dataset)
        dataset.balanced_subset(1)
        assert len(dataset) == before

    def test_the_fingerprint_follows(self, folder):
        """So a limited run cannot be mistaken for a full one in the results."""
        dataset = ImageFolderDataset(folder)
        assert dataset.balanced_subset(1).fingerprint() != dataset.fingerprint()

    def test_zero_is_refused(self, folder):
        with pytest.raises(ValueError, match="needs n >= 1"):
            ImageFolderDataset(folder).balanced_subset(0)

    def test_unlabeled_degenerates_to_a_prefix(self, folder):
        """Every item shares the label None, so there is one 'class'."""
        one_class = next(child for child in sorted(folder.iterdir()) if child.is_dir())
        dataset = ImageFolderDataset(one_class, labeled=False)
        assert len(dataset.balanced_subset(1)) == 1

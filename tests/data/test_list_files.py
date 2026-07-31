"""`list_files` — the shared directory listing every folder dataset builds on.

It replaced `iterdir()` + `Path.is_file()` in three constructors, which was a
stat per entry and, over NFS, the slowest thing any of them did (see the
docstring for the measurements). That was a *performance* change, so what these
tests protect is that it changed nothing else: the same files, in the same
order, with directories and unmatched suffixes still excluded.
"""

from pathlib import Path

import pytest

from visbench.data.base import list_files


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    for name in ("b.jpg", "a.png", "c.JPEG", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "deep.jpg").write_bytes(b"x")
    return tmp_path


def test_returns_files_sorted_by_name(folder: Path):
    """Order is sorted, not filesystem order.

    Every folder dataset pairs its file list with labels or targets by index,
    and the cache is keyed on that order, so a listing that varied by machine
    would silently repoint supervision.
    """
    assert [p.name for p in list_files(folder)] == ["a.png", "b.jpg", "c.JPEG", "notes.txt"]


def test_extensions_filter_is_case_insensitive(folder: Path):
    """`c.JPEG` must match `.jpeg`; VOC and Imagenette both ship mixed case."""
    names = [p.name for p in list_files(folder, (".jpg", ".jpeg"))]
    assert names == ["b.jpg", "c.JPEG"]


def test_directories_are_excluded(folder: Path):
    """The reason the old code called `is_file()` at all.

    `scandir` answers this from the type `readdir` returned rather than by
    stat-ing, but the answer has to be the same one — a subdirectory appearing
    in an image list would fail later, at decode time, far from the cause.
    """
    assert "nested" not in [p.name for p in list_files(folder)]


def test_a_symlink_to_a_file_is_followed(folder: Path):
    """`readdir` reports a link as a link, so this is the one entry still stat-ed.

    Worth pinning: it is exactly where the scandir rewrite could have changed
    behaviour, since `DirEntry.is_file()` follows links by default and
    `is_file(follow_symlinks=False)` would not.
    """
    (folder / "link.jpg").symlink_to(folder / "b.jpg")
    assert "link.jpg" in [p.name for p in list_files(folder, (".jpg",))]


def test_a_broken_symlink_is_excluded(folder: Path):
    """A link to nothing is not a file, and must not reach a loader."""
    (folder / "broken.jpg").symlink_to(folder / "absent.jpg")
    assert "broken.jpg" not in [p.name for p in list_files(folder, (".jpg",))]


def test_matches_the_iterdir_implementation_it_replaced(folder: Path):
    """The equivalence the rewrite claimed, asserted rather than assumed."""
    extensions = (".jpg", ".jpeg", ".png")
    previous = sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in extensions
    )
    assert list_files(folder, extensions) == previous


def test_an_empty_directory_lists_nothing(tmp_path: Path):
    """Callers raise their own errors on this; it is not this function's job."""
    assert list_files(tmp_path) == []


def test_a_missing_directory_raises(tmp_path: Path):
    """Surfaces the OS error rather than returning an empty list.

    An empty list here would read as "a directory with no images", and the
    dataset's own message would then name the wrong problem.
    """
    with pytest.raises(FileNotFoundError):
        list_files(tmp_path / "absent")

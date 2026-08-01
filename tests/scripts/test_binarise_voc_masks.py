"""Binarising VOC label maps into masks the generic segmentation probe can read.

This guards a *silently wrong number*, which is the category CLAUDE.md says
belongs in the fast suite rather than behind `-m slow`. Pointing the binary
probe straight at VOC's `SegmentationClass` loads, trains and scores — against
masks that are wrong at every object boundary and have no void region at all.
Nothing raises, so only a test catches it.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visbench.data.dense import load_mask

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "binarise_voc_masks.py"


def _load_script():
    """Import the script by path — `scripts/` is not an installed package."""
    spec = importlib.util.spec_from_file_location("binarise_voc_masks", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


def write_palette_png(path: Path, indices: np.ndarray) -> None:
    """A VOC-style mode-P PNG whose raw bytes are class indices.

    The palette deliberately maps indices to *different* greys, which is what
    makes `convert("L")` destructive: index 15 becomes 147, and void 255 becomes
    a light grey that reads as foreground.
    """
    # frombytes rather than fromarray(mode="P"): the mode= parameter is
    # deprecated in Pillow 13, and a uint8 array would otherwise become "L",
    # which is the very conversion this fixture exists to avoid.
    raw = indices.astype(np.uint8)
    image = Image.frombytes("P", (raw.shape[1], raw.shape[0]), raw.tobytes())
    palette = []
    for value in range(256):
        grey = (value * 37) % 256  # arbitrary, but not the identity
        palette.extend([grey, grey, grey])
    image.putpalette(palette)
    image.save(path)


def test_binarise_merges_classes_and_keeps_void(script, tmp_path):
    source = tmp_path / "label.png"
    indices = np.array(
        [
            [0, 0, 1, 1],
            [0, 15, 15, 1],
            [255, 255, 20, 0],
            [0, 0, 0, 0],
        ]
    )
    write_palette_png(source, indices)

    mask = np.array(script.binarise(source))

    expected = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [255, 255, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(mask, expected)


def test_output_is_greyscale_not_palette(script, tmp_path):
    """`load_mask` calls convert("L"); on a mode-P output that would re-break it."""
    source = tmp_path / "label.png"
    write_palette_png(source, np.array([[0, 1], [15, 255]]))

    assert script.binarise(source).mode == "L"


def test_round_trips_through_load_mask(script, tmp_path):
    """The contract that matters: 0/1 foreground and -1 where VOC marked void."""
    source = tmp_path / "label.png"
    write_palette_png(source, np.array([[0, 1], [15, 255]]))

    written = tmp_path / "mask.png"
    script.binarise(source).save(written)

    mask = load_mask(written, ignore_index=255).numpy()
    np.testing.assert_array_equal(mask, np.array([[0.0, 1.0], [1.0, -1.0]]))


def test_reading_the_palette_file_directly_is_wrong(script, tmp_path):
    """The failure this script exists to prevent, pinned so it stays prevented.

    Reading `SegmentationClass` with `load_mask` resolves the palette. Two
    things go wrong at once and neither raises: void becomes a non-zero grey and
    is therefore counted as foreground, and `ignore_index=255` matches nothing
    because it is compared against the resolved value rather than the index.
    """
    source = tmp_path / "label.png"
    write_palette_png(source, np.array([[0, 1], [15, 255]]))

    converted = tmp_path / "correct.png"
    script.binarise(source).save(converted)

    correct = load_mask(converted, ignore_index=255).numpy()
    wrong = load_mask(source, ignore_index=255).numpy()

    assert (correct == -1).sum() == 1, "the correct mask keeps VOC's void region"
    assert (wrong == -1).sum() == 0, "the palette read loses void entirely"
    assert (wrong == 1).sum() > (correct == 1).sum(), "void is counted as foreground"

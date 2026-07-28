"""Fixtures for the CLI tests.

The CLI resolves backbones by *name*, so unlike the rest of the suite it cannot
be handed a fake object — the fake has to be in the registry. It is registered
once here, under a name no real backbone will ever claim.
"""

from pathlib import Path
from typing import NamedTuple

import pytest
from PIL import Image

import visbench
import visbench.cli
from tests.conftest import FakeViT

#: A real registration, kept for the whole session. Guarded because pytest may
#: import this module more than once (``-p no:cacheprovider``, xdist workers),
#: and register_backbone raises on a duplicate name by design.
FAKE_BACKBONE = "fake_cli_vit"
if FAKE_BACKBONE not in visbench.list_backbones():
    visbench.register_backbone(FAKE_BACKBONE, device="cpu")(FakeViT)


@pytest.fixture
def image_folder(tmp_path):
    """``root/{train,val}/<class>/*.png`` — three colour-separable classes."""
    root = tmp_path / "folder"
    palette = {"red": (200, 30, 30), "blue": (30, 30, 200), "green": (30, 200, 30)}
    for split, count in (("train", 4), ("val", 3)):
        for name, colour in palette.items():
            directory = root / split / name
            directory.mkdir(parents=True)
            for index in range(count):
                jitter = tuple(min(255, channel + index * 5) for channel in colour)
                Image.new("RGB", (64, 64), jitter).save(directory / f"{index:02d}.png")
    return root


@pytest.fixture
def flat_folder(tmp_path):
    """``root/test/*.png`` — unlabelled, which is what correspondence wants."""
    root = tmp_path / "flat" / "test"
    root.mkdir(parents=True)
    for index in range(4):
        shade = 20 + 50 * index
        Image.new("RGB", (64, 64), (shade, 255 - shade, 100)).save(root / f"{index:02d}.png")
    return root.parent


@pytest.fixture
def dense_folder(tmp_path):
    """``root/{train,val}/{images,masks}`` — paired by stem, as the CLI expects."""
    import numpy as np

    root = tmp_path / "dense"
    for split in ("train", "val"):
        images = root / split / "images"
        masks = root / split / "masks"
        images.mkdir(parents=True)
        masks.mkdir(parents=True)
        for index in range(3):
            Image.new("RGB", (64, 64), (40 * index, 90, 200 - 40 * index)).save(
                images / f"{index:02d}.png"
            )
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[: 16 + 8 * index] = 1
            np.save(masks / f"{index:02d}.npy", mask)
    return root


class CliResult(NamedTuple):
    """Everything one invocation produced.

    Both streams are captured together because ``capsys`` drains them together:
    reading stdout here and calling ``readouterr()`` again in the test for
    stderr would silently return an empty string, and an assertion against an
    empty string is one that can only pass by accident.
    """

    code: int
    out: str
    err: str


@pytest.fixture
def run_cli(capsys):
    """Invoke ``main`` and return its exit code with everything it printed."""

    def invoke(*argv: str) -> CliResult:
        code = visbench.cli.main(list(argv))
        captured = capsys.readouterr()
        return CliResult(code, captured.out, captured.err)

    return invoke


@pytest.fixture
def cache_dir(tmp_path) -> Path:
    return tmp_path / "cache"

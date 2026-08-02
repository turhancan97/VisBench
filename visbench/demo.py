"""A probe run that needs no dataset and no manual download.

Every example in ``examples/`` takes ``--data``, which means the shortest path
from ``pip install visbench`` to a number went through finding a dataset,
laying it out correctly and fetching a backbone. That is a long way to go
before knowing whether the library works at all.

This module removes that. :func:`synthesise` draws a small labelled folder of
geometric shapes, :func:`demo_backbone` wraps torchvision's ResNet-18 — a core
dependency, ~45 MB of weights — and ``visbench demo`` runs a real probe over
them through the ordinary :func:`visbench.run` path. No special-cased code: the
demo uses the same cache, the same probes and the same result records as any
other run, so what it demonstrates is the actual library.

**The task is deliberately not easy, and that is the point.** Colour carries no
information — foreground and background are drawn from the same base with a
small contrast offset, and the sign of that offset is random — so a backbone
that has learned nothing about geometry cannot do better than chance. Shapes are
rotated, scaled, positioned at random and buried in noise. A first pass without
those made a pretrained ResNet-18 score **1.0**, which is the saturation this
project rejects elsewhere: a demo printing a perfect score teaches nothing and
looks like a fixture. At the defaults it scores about **0.81 top-1 against a
0.25 chance baseline**, and raising ``--noise`` walks it down to chance — see
:func:`synthesise` for the measured curve.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

__all__ = ["SHAPES", "demo_backbone", "synthesise"]

#: The four classes. Distinguishable by geometry alone, which is the point —
#: they differ in outline, not in colour or size.
SHAPES = ("circle", "square", "triangle", "cross")


def _draw_shape(
    shape: str,
    size: int,
    rng: random.Random,
    contrast: int,
    noise: float,
) -> Image.Image:
    """One image of ``shape``, with everything except the geometry randomised."""
    base = rng.randrange(60, 196)
    background = tuple(max(0, min(255, base + rng.randrange(-12, 13))) for _ in range(3))
    # The sign is random, so "lighter than the background" is not a cue either.
    sign = 1 if rng.random() < 0.5 else -1
    foreground = tuple(
        max(0, min(255, c + sign * contrast + rng.randrange(-10, 11))) for c in background
    )

    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)

    span = rng.uniform(0.34, 0.52) * size
    cx = rng.uniform(span / 2 + 4, size - span / 2 - 4)
    cy = rng.uniform(span / 2 + 4, size - span / 2 - 4)
    box = (cx - span / 2, cy - span / 2, cx + span / 2, cy + span / 2)

    if shape == "circle":
        draw.ellipse(box, fill=foreground)
    elif shape == "square":
        draw.rectangle(box, fill=foreground)
    elif shape == "triangle":
        draw.polygon(
            [(cx, cy - span / 2), (cx + span / 2, cy + span / 2), (cx - span / 2, cy + span / 2)],
            fill=foreground,
        )
    elif shape == "cross":
        arm = span / 6
        draw.rectangle((cx - arm, cy - span / 2, cx + arm, cy + span / 2), fill=foreground)
        draw.rectangle((cx - span / 2, cy - arm, cx + span / 2, cy + arm), fill=foreground)
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"Unknown shape {shape!r}; expected one of {SHAPES}")

    # Image.Resampling rather than the Image.BILINEAR alias: the alias is
    # deprecated and Pillow's stubs no longer declare it, so mypy rejects it.
    image = image.rotate(
        rng.uniform(0, 360), resample=Image.Resampling.BILINEAR, fillcolor=background
    )

    if noise > 0:
        # Vectorised: a per-pixel Python loop took ~10 s for 96 images, which is
        # most of the demo's budget. One shared draw per pixel keeps the noise
        # achromatic, so it blurs the outline without adding a colour cue.
        array = np.asarray(image, dtype=np.float32)
        grain = np.random.default_rng(rng.randrange(2**32)).normal(0, noise, array.shape[:2])
        array = np.clip(array + grain[:, :, None], 0, 255).astype(np.uint8)
        image = Image.fromarray(array)

    return image.filter(ImageFilter.GaussianBlur(0.6))


def synthesise(
    root: str | Path,
    per_class: int = 20,
    image_size: int = 224,
    seed: int = 0,
    contrast: int = 30,
    noise: float = 45.0,
) -> Path:
    """Write ``root/{train,val}/<shape>/*.png`` and return ``root``.

    Deterministic given ``seed``, so two runs of the demo produce the same
    number and a user comparing against the documented one is comparing like
    with like.

    ``contrast`` and ``noise`` are the difficulty knobs, and they are chosen so
    the demo's number is *interpretable* rather than impressive. Measured on
    torchvision ResNet-18, 20 images per class per split:

    ======  ========  ======  ======
    noise   contrast  top1    mAP
    ======  ========  ======  ======
    28      40        0.975   0.771
    **45**  **30**    0.812   0.521
    60      24        0.550   0.365
    75      20        0.438   0.314
    90      16        0.312   0.301
    ======  ========  ======  ======

    The defaults are the middle row. The bottom row is chance (0.25), and the
    monotonic slide into it is the most useful thing the demo shows: the probe
    is responding to how recoverable the shape actually is, not to a fixture.
    An earlier default of ``noise=28`` scored 0.975 and taught nothing.
    """
    root = Path(root)
    rng = random.Random(seed)
    for split in ("train", "val"):
        for shape in SHAPES:
            directory = root / split / shape
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(per_class):
                image = _draw_shape(shape, image_size, rng, contrast, noise)
                image.save(directory / f"{index:03d}.png")
    return root


def demo_backbone(device: str | None = None) -> Any:
    """Torchvision's ImageNet ResNet-18, wrapped as a VisBench backbone.

    Torchvision rather than a registered name, deliberately: it is a **core**
    dependency, so the demo works on a bare ``pip install visbench`` with no
    extra. The weights are ~45 MB against DINOv2's ~1.7 GB, which is the
    difference between a demo someone runs and one they abandon.

    It also exercises :class:`~visbench.CustomBackbone`, so the first thing a
    reader sees is the path they would take for their own model.
    """
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    # Drop avgpool and fc: VisBench wants the last conv feature map, not logits.
    trunk = nn.Sequential(*list(model.children())[:-2])

    from visbench.backbones.custom import CustomBackbone

    return CustomBackbone(
        trunk,
        preprocess=weights.transforms(),
        name="resnet18_torchvision",
        weights_id="torchvision/IMAGENET1K_V1",
        device=device,
    )

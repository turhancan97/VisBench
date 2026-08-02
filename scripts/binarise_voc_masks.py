#!/usr/bin/env python3
"""Turn VOC ``SegmentationClass`` label maps into binary foreground masks.

The generic (binary) segmentation probe wants a folder of masks where non-zero
means foreground. VOC does not ship one — it ships 21-class label maps — and
the obvious shortcut of pointing the probe straight at ``SegmentationClass`` is
**silently wrong**, which is why this script exists rather than a flag.

``SegmentationClass`` PNGs are palette images (mode ``P``) whose raw bytes are
the class indices. :func:`load_mask` calls ``convert("L")``, because a binary
mask only cares whether a pixel is non-zero; on a palette file that resolves the
palette, so classes ``[0, 1, 15, 255]`` arrive as greys ``[0, 38, 147, 220]``.
Every object survives as "non-zero", so it looks like it worked — but VOC's void
255 also resolves to a light grey, i.e. *foreground*, and ``ignore_index=255``
never matches because it is compared against the resolved value. The result is
masks that are wrong along every object boundary, with nothing raising.

So the conversion goes through :func:`load_label_map`, which reads the file
without any mode conversion, and the output is written as mode ``L`` — real
greyscale, where ``load_mask``'s ``convert("L")`` is a no-op and
``ignore_index=255`` matches what is actually stored.

Output convention, chosen to match what the probe already reads:

===== ==========================================================
value  meaning
===== ==========================================================
0      background (VOC class 0)
1      foreground (VOC classes 1-20, all merged)
255    void — VOC's boundary outlines, scored as "no ground truth"
===== ==========================================================

Merging all 20 object classes into one is the point: this is the *generic*
segmentation probe, which asks whether a representation separates object from
background without naming the object. The semantic probe is the one that keeps
the classes, and it reads ``SegmentationClass`` directly.

Usage
-----
::

    python scripts/binarise_voc_masks.py \\
        --voc-root /shared/sets/datasets/pascal_voc_2021/VOCdevkit/VOC2012 \\
        --out-dir data/voc_binary_masks
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Import from the library rather than reimplementing the read, so this script
# cannot drift from the loader whose behaviour it is compensating for.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visbench.data.dense import load_label_map  # noqa: E402

#: VOC's void value, and what this script writes back out for those pixels.
VOID = 255

#: What load_label_map returns for a void pixel, having mapped it.
IGNORED = -1


def binarise(label_path: Path) -> Image.Image:
    """One VOC label map to a 0 / 1 / 255 greyscale mask.

    ``load_label_map`` has already mapped void to -1, so the three cases are
    disjoint and there is no ordering subtlety: negative is void, zero is
    background, anything else is one of the twenty object classes.
    """
    labels = load_label_map(label_path).numpy()

    mask = np.zeros(labels.shape, dtype=np.uint8)
    mask[labels > 0] = 1
    mask[labels == IGNORED] = VOID

    # A 2-D uint8 array becomes mode "L" on its own, and passing mode= is
    # deprecated in Pillow 13. "L" is the whole point regardless: the next
    # reader's convert("L") must be a no-op rather than a palette resolution.
    return Image.fromarray(mask)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--voc-root",
        type=Path,
        required=True,
        help="VOC2012 root, the directory containing SegmentationClass/",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="where to write the binary masks; created if absent",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rewrite masks that already exist (default: skip them)",
    )
    args = parser.parse_args()

    source = args.voc_root / "SegmentationClass"
    if not source.is_dir():
        parser.error(f"No SegmentationClass/ under {args.voc_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    foreground_total = 0.0
    void_total = 0.0

    paths = sorted(source.glob("*.png"))
    if not paths:
        parser.error(f"No PNGs in {source}")

    for index, path in enumerate(paths, start=1):
        destination = args.out_dir / path.name
        if destination.exists() and not args.overwrite:
            skipped += 1
            continue

        image = binarise(path)
        image.save(destination)
        written += 1

        array = np.array(image)
        foreground_total += float((array == 1).mean())
        void_total += float((array == VOID).mean())

        if index % 500 == 0:
            print(f"  {index}/{len(paths)}", flush=True)

    print(f"wrote {written} masks to {args.out_dir} ({skipped} already present)")
    if written:
        # Sanity figures rather than decoration. VOC is a foreground-sparse
        # dataset, so a foreground fraction near 1.0 would mean the palette was
        # resolved after all and every non-black grey counted as an object --
        # the exact failure this script exists to avoid, and one that is
        # invisible in any single image.
        print(f"mean foreground fraction: {foreground_total / written:.4f}")
        print(f"mean void fraction:       {void_total / written:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

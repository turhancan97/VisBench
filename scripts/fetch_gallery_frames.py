#!/usr/bin/env python3
"""Fetch the gallery's source photographs from Open Images, licence-checked.

    python scripts/fetch_gallery_frames.py        # writes assets/gallery_frames/

The docs gallery used to be drawn on generated scenes, and the reason was never
that shapes are pretty: it was that VisBench ships to PyPI under MIT while the
datasets its probes normally run on -- VOC, ImageNet, NYUv2, Taskonomy, NIGHTS
-- each restrict redistribution to some degree and none clearly grants it.
Committing their frames would put third-party imagery in an MIT package, which
is the line ``NOTICE`` already draws around probe3d's CC BY-NC code.

**Open Images is the way to have real photographs without crossing it.** Every
one of the 41,620 images in its validation split is CC BY 2.0 -- verified here
rather than believed, see :data:`ALLOWED_LICENCES` -- and its boxes and
instance masks are CC BY 4.0 human annotations. So the gallery can show a real
photograph *with real ground truth* for the probes whose targets are annotated,
which no amount of care with a generated scene can imitate.

CC BY has one obligation and it is not optional: **attribution**. This script
writes ``CREDITS.md`` beside the frames, one row per photograph naming the
author, the title, the licence and the original landing page. A frame whose
metadata is missing any of those is refused rather than fetched, because an
unattributable CC BY image is one this repository may not redistribute.

WHAT IS PINNED, AND WHY

The image IDs below are fixed, the way 8b pinned the corner probe's frame set.
A gallery that re-selected its own frames would redraw every figure on every
run, so a figure could never be compared against the one it replaced, and a
renderer bug would look like a different photograph. Selection is a decision
recorded here; fetching is the only thing that happens at run time.

The IDs were chosen against three requirements, checked before pinning:

- **real annotations**: each scene frame carries instance masks *and* boxes, so
  detection and both segmentation probes draw human labels rather than anything
  derived from the pixels;
- **visible structure**: texture and straight edges, so the derived probes
  (corner, edge) have something to find -- a frame of open sky would make a
  correct probe look broken;
- **no prominent faces**: the class groups are objects and animals. A
  documentation page is the wrong place to republish photographs of identifiable
  people, whatever the licence permits.
"""

import argparse
import csv
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Outside ``docs/``: these are *inputs* to the gallery, not pages of it. The
#: rendered figures live in ``docs/_static/gallery`` because Sphinx must be able
#: to reach them; nothing needs to reach these but this script and the renderer.
#: ``pyproject.toml`` excludes them from the sdist, as it already does the
#: figures.
DEFAULT_OUT = REPO / "assets" / "gallery_frames"

#: Metadata downloads are cached here between runs -- 40 MB of CSV to select a
#: dozen images, which is not something to re-fetch while iterating.
DEFAULT_CACHE = Path("/tmp/visbench_gallery_cache")

_OI = "https://storage.googleapis.com/openimages"
_SOURCES = {
    "images": f"{_OI}/2018_04/validation/validation-images-with-rotation.csv",
    "boxes": f"{_OI}/v5/validation-annotations-bbox.csv",
    "segments": f"{_OI}/v5/validation-annotations-object-segmentation.csv",
    "classes": f"{_OI}/v5/class-descriptions-boxable.csv",
    "masks": f"{_OI}/v5/validation-masks/validation-masks-0.zip",
}
_IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/validation/{}.jpg"

#: The only licences this repository may redistribute a photograph under.
#:
#: An allowlist rather than a denylist, and checked per image rather than once
#: for the split: "all of Open Images validation is CC BY 2.0" is true today and
#: is not a promise anyone made to this repository. A frame whose licence is not
#: listed here is refused, so the claim in ``CREDITS.md`` is something the
#: fetcher enforced rather than something a docstring asserts.
ALLOWED_LICENCES = {
    "https://creativecommons.org/licenses/by/2.0/",
    "https://creativecommons.org/publicdomain/zero/1.0/",
    "https://creativecommons.org/publicdomain/mark/1.0/",
}

#: Frames for the probes that draw a map over one image. Masks and boxes come
#: with each, so detection and both segmentations show human annotation.
SCENES = [
    "0ff2a9e5ed288222",  # zebras on grass -- two classes, strong texture
    "0cb5cae66bb9c4cd",  # alpaca and sheep -- two classes, soft geometry
    "09270cfd3d9cff39",  # leopard on rock -- high-frequency pattern
    "0d87b8c4ea2f4c79",  # guitars -- straight edges and corners
    "09b616a35758e42c",  # tomatoes in a bowl -- curved specular surfaces
    "0d80e9ef29a2fffe",  # hot air balloons -- large smooth regions
]

#: Frames for the four probes drawn as predictions rather than targets.
#:
#: Interiors, deliberately. Those probes' heads were fitted on NYUv2 rooms and
#: Taskonomy buildings, so a photograph filled by an animal's face is far outside
#: what they were trained to read and the prediction shows domain shift rather
#: than the probe. These need no annotation at all -- there is no target column
#: to fill -- which is why they are their own kind rather than more scenes.
CONTEXT_FRAMES = [
    "dea2fdb1a77ec6bb",  # bedroom: walls meeting at a corner, a receding floor
    "3f6e15958d41abf9",  # lounge: furniture at several depths
    "826690670ace3be4",  # office: planar desk, cabinet, blank wall
]

#: Frames for the probes whose answer is a choice among images. Four classes so
#: a linear probe has something to separate, four frames each so leave-one-out
#: retrieval ranks against three alternatives rather than one.
#:
#: Object and animal classes deliberately -- see the module docstring.
CLASS_FRAMES = {
    "car": ["001083f05db4352b", "001a794d1865ee47", "001a809ad40a2f84", "0022e32008e479cb"],
    "dog": ["0007d6cf88afaa4a", "0008e425fb49a2bf", "000c4d66ce89aa69", "00493fdf106b5fdf"],
    "airplane": ["0001eeaf4aed83f9", "007384da2ed0464f", "019a3d18cb357cf3", "01a10772cd4613bb"],
    "tree": ["001a78754e43abc5", "00437aa0ab4abf9d", "004a9412eb1a83e8", "00627adda9a83f9d"],
}

#: Longest side the frames are stored at. Small enough that two dozen
#: photographs are a reasonable thing to commit, large enough that the 224px
#: crop a real probe uses is not an upscale.
STORE_SIZE = 512


def _download(url: str, path: Path) -> Path:
    """Fetch ``url`` to ``path`` unless it is already there."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url.rsplit('/', 1)[-1]} ...", flush=True)
    with urllib.request.urlopen(url, timeout=300) as response:
        path.write_bytes(response.read())
    return path


def _metadata(cache: Path) -> dict:
    """The four CSVs and the mask archive, downloaded once and cached."""
    paths = {name: _download(url, cache / url.rsplit("/", 1)[-1]) for name, url in _SOURCES.items()}

    classes = {}
    with paths["classes"].open() as handle:
        for code, name in csv.reader(handle):
            classes[code] = name

    wanted = (
        set(SCENES)
        | set(CONTEXT_FRAMES)
        | {frame for frames in CLASS_FRAMES.values() for frame in frames}
    )

    images = {}
    with paths["images"].open() as handle:
        for row in csv.DictReader(handle):
            if row["ImageID"] in wanted:
                images[row["ImageID"]] = row

    boxes: dict[str, list] = {frame: [] for frame in wanted}
    with paths["boxes"].open() as handle:
        for row in csv.DictReader(handle):
            if row["ImageID"] in wanted:
                boxes[row["ImageID"]].append(
                    {
                        "label": classes.get(row["LabelName"], row["LabelName"]),
                        "xyxy_norm": [
                            float(row["XMin"]),
                            float(row["YMin"]),
                            float(row["XMax"]),
                            float(row["YMax"]),
                        ],
                        "group_of": row["IsGroupOf"] == "1",
                    }
                )

    segments: dict[str, list] = {frame: [] for frame in SCENES}
    with paths["segments"].open() as handle:
        for row in csv.DictReader(handle):
            if row["ImageID"] in segments:
                segments[row["ImageID"]].append(
                    {"path": row["MaskPath"], "label": classes.get(row["LabelName"], "?")}
                )

    return {"images": images, "boxes": boxes, "segments": segments, "masks": paths["masks"]}


def _check_licence(frame: str, row: dict) -> dict:
    """Refuse anything this repository may not redistribute, or may not credit.

    Both halves matter. A licence outside the allowlist cannot be shipped at
    all; a CC BY frame with no author or no landing page cannot be *attributed*,
    which is the one thing CC BY requires in return. Neither is a warning --
    a frame that reaches ``CREDITS.md`` with a blank author is a licence breach
    that renders perfectly.
    """
    licence = (row.get("License") or "").strip()
    if licence not in ALLOWED_LICENCES:
        raise SystemExit(
            f"{frame}: licence {licence!r} is not in the allowlist, so this repository "
            f"may not redistribute it. Allowed: {sorted(ALLOWED_LICENCES)}"
        )
    missing = [key for key in ("Author", "OriginalLandingURL") if not (row.get(key) or "").strip()]
    if missing:
        raise SystemExit(f"{frame}: CC BY requires attribution and {missing} is empty; refusing.")
    return {
        "id": frame,
        "title": (row.get("Title") or "").strip() or f"Open Images {frame}",
        "author": row["Author"].strip(),
        "licence": licence,
        "source": row["OriginalLandingURL"].strip(),
    }


def _store_image(frame: str, out: Path) -> tuple[int, int]:
    """Download one photograph, downscale it, and return the stored size."""
    from PIL import Image

    destination = out / "images" / f"{frame}.jpg"
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(_IMAGE_URL.format(frame), timeout=300) as response:
            image = Image.open(io.BytesIO(response.read())).convert("RGB")
        image.thumbnail((STORE_SIZE, STORE_SIZE), Image.LANCZOS)
        image.save(destination, quality=90)
    with Image.open(destination) as stored:
        return stored.size


def _store_masks(frame: str, segments: list, archive: Path, out: Path, size: tuple) -> list:
    """Extract this frame's instance masks, resized to the stored photograph.

    Open Images masks are full-frame binaries at their own resolution, so they
    are resampled to the stored image with **nearest neighbour** -- the rule
    every dense target in this codebase follows, and for the same reason:
    bilinear would invent boundary values that are neither foreground nor
    background, which is exactly the halo `DenseFolderDataset` exists to avoid.
    """
    from PIL import Image

    kept = []
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        for index, segment in enumerate(segments):
            if segment["path"] not in names:
                continue
            destination = out / "masks" / frame / f"{index}.png"
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(segment["path"]) as handle:
                    mask = Image.open(io.BytesIO(handle.read())).convert("L")
                mask.resize(size, Image.NEAREST).save(destination)
            kept.append({"label": segment["label"], "path": f"masks/{frame}/{index}.png"})
    return kept


def _write_credits(out: Path, credits: list) -> None:
    """The attribution CC BY requires, as a page a human can read."""
    lines = [
        "# Gallery photograph credits",
        "",
        "Every photograph used by the VisBench documentation gallery, with the",
        "attribution its licence requires. Generated by",
        "`scripts/fetch_gallery_frames.py` -- do not edit by hand.",
        "",
        "The photographs are from the [Open Images](https://storage.googleapis.com/",
        "openimages/web/index.html) validation split and remain under their own",
        "licences, which are **not** VisBench's MIT licence. The box and mask",
        "annotations drawn over them are Open Images' own, CC BY 4.0.",
        "",
        "Images are stored downscaled to a longest side of "
        f"{STORE_SIZE}px; no other modification is made.",
        "",
        "| photograph | author | licence | source |",
        "| --- | --- | --- | --- |",
    ]
    for entry in sorted(credits, key=lambda c: c["id"]):
        short = "CC BY 2.0" if "by/2.0" in entry["licence"] else entry["licence"]
        lines.append(
            f"| `{entry['id']}` {entry['title'][:40]} | {entry['author']} "
            f"| [{short}]({entry['licence']}) | [original]({entry['source']}) |"
        )
    (out / "CREDITS.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()

    print("reading Open Images metadata")
    meta = _metadata(args.cache)

    frames: dict[str, dict] = {}
    credits = []
    for frame in SCENES:
        row = meta["images"].get(frame)
        if row is None:
            raise SystemExit(f"{frame}: not in the validation metadata")
        credits.append(_check_licence(frame, row))
        size = _store_image(frame, args.out)
        frames[frame] = {
            "kind": "scene",
            "size": list(size),
            "boxes": [box for box in meta["boxes"][frame] if not box["group_of"]],
            "masks": _store_masks(frame, meta["segments"][frame], meta["masks"], args.out, size),
        }
        print(
            f"  {frame}  {size[0]}x{size[1]}  "
            f"{len(frames[frame]['boxes'])} boxes  {len(frames[frame]['masks'])} masks"
        )

    for frame in CONTEXT_FRAMES:
        row = meta["images"].get(frame)
        if row is None:
            raise SystemExit(f"{frame}: not in the validation metadata")
        credits.append(_check_licence(frame, row))
        size = _store_image(frame, args.out)
        frames[frame] = {"kind": "context", "size": list(size)}
    print(f"  context: {len(CONTEXT_FRAMES)} interiors")

    for label, members in CLASS_FRAMES.items():
        for frame in members:
            row = meta["images"].get(frame)
            if row is None:
                raise SystemExit(f"{frame}: not in the validation metadata")
            credits.append(_check_licence(frame, row))
            size = _store_image(frame, args.out)
            frames[frame] = {"kind": "class", "label": label, "size": list(size)}
        print(f"  class {label}: {len(members)} frames")

    (args.out / "frames.json").write_text(json.dumps(frames, indent=2, sort_keys=True) + "\n")
    _write_credits(args.out, credits)
    print(f"\n{len(frames)} frames -> {args.out}")
    print(f"credits -> {args.out / 'CREDITS.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Render one ``visbench show`` figure per probe, for the docs and the README.

    python scripts/render_gallery.py     # writes docs/_static/gallery/*.png

**Every frame is generated, and that is a licensing decision as much as a
convenience one.** VisBench ships to PyPI under MIT; the datasets its probes are
normally run on (VOC, ImageNet, NYUv2, Taskonomy, NIGHTS) each restrict
redistribution to some degree and none clearly grants it, so committing panels
containing their photographs would put third-party imagery in an MIT package.
That is the same line this project already took when it declined to vendor
probe3d's CC BY-NC code -- see ``NOTICE``.

Generating instead buys three things beyond the licence:

- **The gallery regenerates from one command with no downloads**, so the figures
  can never drift from what the code actually draws. The same argument as
  ``visbench demo``, and as 8b's pinned frame set for the corner probe.
- **The ground truth is exact**, so each panel shows its probe's convention
  rather than an approximation of it: sphere normals are analytic, depth comes
  from the z-order, masks and boxes are exact by construction.
- **Invalid pixels can be placed deliberately.** Each scene carries a "sensor
  hole" -- a region with no depth return -- so the magenta marker appears where
  it is supposed to, which a lucky real frame might not show at all.

What is *not* faked: every figure is produced by the real ``visbench show``
command over a real dataset class, through the real renderers. Only the pixels
are synthetic.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import visbench.cli

REPO = Path(__file__).resolve().parent.parent
#: Inside ``docs/`` on purpose. The Sphinx site must reference its images with
#: relative paths that stay within the source tree -- ``../assets/...`` is not
#: something Sphinx can follow, and MyST does not warn about it -- so the
#: gallery lives under ``_static``, which Sphinx copies to the built site. The
#: README then points at the same files by absolute URL, which is the rule
#: ``tests/test_readme.py`` enforces there.
DEFAULT_OUT = REPO / "docs" / "_static" / "gallery"

#: Scene size before the probe's resize and centre crop. Deliberately not
#: square: a viewer that mishandled the geometry would be visible immediately,
#: which is the one thing these figures must not hide.
HEIGHT, WIDTH = 260, 340

#: Class 0 is background and is a *real* class for segmentation, which is why
#: the label convention differs from depth's. Order fixes the VOC palette
#: colours the figures come out with.
CLASSES = ("background", "sphere", "box", "bar")

_PLANE_NEAR, _PLANE_FAR = 2.0, 7.0


def _scene(rng: np.random.Generator, dominant: int = 0) -> dict:
    """One frame with exact ground truth for every dense probe.

    Returns arrays that are the *definition* of the scene rather than an
    estimate of it: depth is the z-order, normals are analytic, and the label
    map, mask and boxes all fall out of which shape owns each pixel.
    """
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)

    # A ground plane receding with y, tilted, so depth and normals both vary.
    depth = _PLANE_NEAR + (_PLANE_FAR - _PLANE_NEAR) * (1.0 - ys / HEIGHT)
    tilt = np.arctan2(_PLANE_FAR - _PLANE_NEAR, HEIGHT)
    normal = np.zeros((3, HEIGHT, WIDTH))
    normal[1] = -np.sin(tilt)
    normal[2] = np.cos(tilt)

    labels = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    shade = 0.18 + 0.16 * (ys / HEIGHT)
    boxes: list[tuple[int, int, int, int, int]] = []

    # The first shape is forced to `dominant` and made the largest and nearest,
    # so the class label the folder layout derives from the scene actually
    # describes what a reader sees. Without it a "sphere" frame can be mostly
    # boxes, and the classification sheet then looks like a labelling bug.
    kinds = [dominant] if dominant else []
    kinds += [int(rng.integers(1, 4)) for _ in range(int(rng.integers(2, 5)))]

    for order, kind in enumerate(kinds):
        leading = order == 0 and dominant != 0
        cx, cy = rng.uniform(50, WIDTH - 50), rng.uniform(50, HEIGHT - 50)
        radius = rng.uniform(40, 52) if leading else rng.uniform(22, 36)
        z = float(rng.uniform(1.0, 1.3) if leading else rng.uniform(1.6, 3.4))

        if kind == 1:  # sphere: the one shape with a curved, analytic normal
            dx, dy = (xs - cx) / radius, (ys - cy) / radius
            inside = dx**2 + dy**2 <= 1.0
            nz = np.sqrt(np.clip(1.0 - dx**2 - dy**2, 0.0, 1.0))
            surface = z - radius / 100.0 * nz
            take = inside & (surface < depth)
            depth[take] = surface[take]
            normal[0][take], normal[1][take], normal[2][take] = dx[take], dy[take], nz[take]
            shade[take] = 0.5 + 0.45 * nz[take]
        else:  # a flat surface: box or bar, tilted slightly out of frontal
            half_w = radius if kind == 2 else radius * 1.6
            half_h = radius if kind == 2 else radius * 0.32
            inside = (np.abs(xs - cx) <= half_w) & (np.abs(ys - cy) <= half_h)
            take = inside & (z < depth)
            # A real surface is rarely exactly camera-facing, and if every flat
            # one here were, the whole normal map would be a single lavender
            # field with nothing to read. The tilt is small and analytic.
            facing = np.array([rng.uniform(-0.45, 0.45), rng.uniform(-0.45, 0.45), 1.0])
            facing /= np.linalg.norm(facing)
            depth[take] = z
            for axis in range(3):
                normal[axis][take] = facing[axis]
            shade[take] = 0.62 + 0.22 * float(facing[2]) + 0.1 * (kind == 3)

        labels[take] = kind
        if take.any():
            rows, columns = np.nonzero(take)
            boxes.append(
                (kind, int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max()))
            )

    # A depth sensor's dropout, placed on purpose. Every probe marks this
    # differently -- 0 for depth, the zero vector for normals, NaN for occlusion
    # edges -- and each figure shows its own marker as a result.
    hole = np.zeros((HEIGHT, WIDTH), dtype=bool)
    hy, hx = int(rng.integers(20, HEIGHT - 70)), int(rng.integers(20, WIDTH - 70))
    hole[hy : hy + 46, hx : hx + 58] = True

    # Low-frequency texture, not per-pixel noise. Two reasons, and the second
    # is the one that decides it: per-pixel noise creates gradient energy
    # everywhere, which would swamp the edge and corner targets with signal that
    # is not structure -- and it defeats PNG compression, taking the committed
    # gallery from ~1 MB to ~8 MB. Smooth variation gives the matcher something
    # to lock onto without either cost.
    coarse = rng.normal(0.0, 0.09, (HEIGHT // 16 + 2, WIDTH // 16 + 2))
    texture = np.asarray(Image.fromarray(coarse).resize((WIDTH, HEIGHT), Image.BICUBIC))
    grey = np.clip(shade + texture, 0.0, 1.0)
    tint = np.stack([grey * 1.0, grey * 0.94, grey * 0.86], axis=-1)
    rgb = (np.clip(tint, 0, 1) * 255).astype(np.uint8)

    depth[hole] = 0.0  # depth's convention: 0 is "no ground truth"
    normal[:, hole] = 0.0  # normals': the zero vector
    normal /= np.maximum(np.linalg.norm(normal, axis=0, keepdims=True), 1e-9)
    normal[:, hole] = 0.0

    # VOC marks object outlines 255 and excludes them from scoring. Reproducing
    # that here is what makes the semantic figure show the *ignore* convention
    # rather than only the palette -- and 255 is what `--ignore-index` maps to
    # -1, which is why 0 can stay a real class.
    outline = np.zeros_like(labels, dtype=bool)
    for shift in (-1, 1):
        for axis in (0, 1):
            outline |= np.roll(labels, shift, axis=axis) != labels
    labels_with_void = labels.astype(np.int32).copy()
    labels_with_void[outline] = 255

    return {
        "rgb": rgb,
        "dominant": dominant,
        "labels_void": labels_with_void,
        "depth": depth.astype(np.float32),
        "normal": normal.astype(np.float32),
        "labels": labels,
        "mask": (labels > 0).astype(np.uint8),
        "boxes": boxes,
        "hole": hole,
    }


def _magnitude_targets(scene: dict) -> dict:
    """The three magnitude maps, each honest about where its holes are.

    ``edge`` is an intensity gradient, so it is defined everywhere and **0 is a
    real reading**. ``occlusion_edge`` is a *depth* discontinuity, so it is
    undefined wherever the depth is, and carries ``NaN`` there -- the one
    out-of-band convention, and the reason it exists.
    """
    grey = scene["rgb"].mean(axis=-1) / 255.0
    gy, gx = np.gradient(grey)
    edge = np.hypot(gx, gy)

    depth = scene["depth"].astype(np.float64)
    dy, dx = np.gradient(depth)
    occlusion = np.log1p(50.0 * np.hypot(dx, dy))
    occlusion[scene["hole"]] = np.nan

    # A corner-ish response: the product of the two gradient directions, which
    # peaks where both vary. Not Shi-Tomasi -- the corner probe computes its own
    # target from the image, so this is only for keypoints2d.
    keypoints = np.abs(gx * gy) * 400.0

    return {
        "edge": (edge * 1000.0).astype(np.float32),
        "keypoints": (keypoints * 30.0).astype(np.float32),
        "occlusion": occlusion.astype(np.float32),
    }


def build_dataset(root: Path, frames: int = 8, seed: int = 0) -> None:
    """Write every layout the CLI's probes expect, from one set of scenes."""
    rng = np.random.default_rng(seed)
    scenes = [_scene(rng, dominant=1 + index % 3) for index in range(frames)]

    for split in ("train", "val"):
        for name in ("images", "depths", "normals", "masks", "labels", "edges", "kp", "occ"):
            (root / split / name).mkdir(parents=True, exist_ok=True)
        (root / split / "JPEGImages").mkdir(parents=True, exist_ok=True)
        (root / split / "Annotations").mkdir(parents=True, exist_ok=True)

        for index, scene in enumerate(scenes):
            stem = f"{index:03d}"
            magnitudes = _magnitude_targets(scene)
            Image.fromarray(scene["rgb"]).save(root / split / "images" / f"{stem}.png")
            Image.fromarray(scene["rgb"]).save(root / split / "JPEGImages" / f"{stem}.jpg")
            np.save(root / split / "depths" / f"{stem}.npy", scene["depth"])
            np.save(root / split / "normals" / f"{stem}.npy", scene["normal"])
            np.save(root / split / "masks" / f"{stem}.npy", scene["mask"])
            np.save(
                root / split / "labels" / f"{stem}.npy", scene["labels_void"].astype(np.float32)
            )
            np.save(root / split / "edges" / f"{stem}.npy", magnitudes["edge"])
            np.save(root / split / "kp" / f"{stem}.npy", magnitudes["keypoints"])
            np.save(root / split / "occ" / f"{stem}.npy", magnitudes["occlusion"])
            _write_voc_xml(root / split / "Annotations" / f"{stem}.xml", scene["boxes"])

    _build_taskonomy(root / "tasko", scenes)
    _build_labelled_folder(root / "folder", scenes)
    _build_flat_folder(root / "flat", scenes)
    _build_triplets(root / "triplets", scenes, rng)


def _build_taskonomy(root: Path, scenes: list) -> None:
    """Taskonomy's own layout: split lists, building-nested frames, 16-bit PNGs.

    The three magnitude probes are indexed from ``splits/<partition>_<split>.csv``
    rather than from two named folders, so they need this shape rather than the
    ``<split>/{images,targets}`` one. Targets are uint16 because
    ``load_edge_map`` reads them **without mode conversion** -- Taskonomy stores
    values well past 255 and ``convert("L")`` would quantise six bits away.
    """
    import csv

    building = "generated"
    domains = {
        "edge_texture": ("edge", 1000.0),
        "keypoints2d": ("keypoints", 30.0),
        "edge_occlusion": ("occlusion", 50.0),
    }
    (root / "splits").mkdir(parents=True, exist_ok=True)
    for name in ("rgb", "mask_valid", *domains):
        (root / name / building).mkdir(parents=True, exist_ok=True)

    rows = []
    for index, scene in enumerate(scenes):
        stem = f"point_{index}_view_0"
        rows.append({"building": building, "point": index, "view": 0})
        Image.fromarray(scene["rgb"]).save(root / "rgb" / building / f"{stem}_domain_rgb.png")

        magnitudes = _magnitude_targets(scene)
        for domain, (key, scale) in domains.items():
            values = np.nan_to_num(magnitudes[key], nan=0.0) * scale
            stored = np.clip(values, 0, 65535).astype(np.uint16)
            Image.fromarray(stored).save(root / domain / building / f"{stem}_domain_{domain}.png")

        # Named for depth whatever it masks -- Taskonomy derived one mask per
        # frame from the depth render and never renamed the file.
        valid = (~scene["hole"]).astype(np.uint8) * 255
        Image.fromarray(valid).save(
            root / "mask_valid" / building / f"{stem}_domain_depth_zbuffer.png"
        )

    for split in ("train", "val"):
        with (root / "splits" / f"tiny_{split}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["building", "point", "view"])
            writer.writeheader()
            writer.writerows(rows)


def _write_voc_xml(path: Path, boxes: list) -> None:
    body = "".join(
        f"<object><name>{CLASSES[kind]}</name><difficult>0</difficult><bndbox>"
        # VOC is 1-indexed and the loader subtracts 1, so write it VOC's way.
        f"<xmin>{x1 + 1}</xmin><ymin>{y1 + 1}</ymin>"
        f"<xmax>{x2 + 1}</xmax><ymax>{y2 + 1}</ymax></bndbox></object>"
        for kind, x1, y1, x2, y2 in boxes
    )
    path.write_text(
        f"<annotation><size><width>{WIDTH}</width><height>{HEIGHT}</height>"
        f"<depth>3</depth></size>{body}</annotation>"
    )


#: The three pooled-feature probes draw the source frame itself, so their
#: figures are as large as the files. Reduced here rather than in the renderer,
#: which must never resize -- that is the one rule ``visbench.viz`` keeps.
_TILE = (136, 104)


def _small(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb).resize(_TILE, Image.BICUBIC)


def _build_labelled_folder(root: Path, scenes: list) -> None:
    """``<split>/<class>/*.png``, labelled by which shape covers most pixels."""
    for split in ("train", "val"):
        for name in CLASSES[1:]:
            (root / split / name).mkdir(parents=True, exist_ok=True)
        for index, scene in enumerate(scenes):
            # By the shape the scene was *built* around, not by pixel count: a
            # bar is thin, so counting pixels lets a background box outvote the
            # subject and the split comes out 1-5 per class. That is a real
            # thing the balance footer would report, but it is not what this
            # figure is meant to show.
            name = CLASSES[scene["dominant"]]
            _small(scene["rgb"]).save(root / split / name / f"{index:03d}.png")


def _build_flat_folder(root: Path, scenes: list) -> None:
    """Unannotated images, which is all correspondence needs."""
    (root / "val").mkdir(parents=True, exist_ok=True)
    for index, scene in enumerate(scenes):
        Image.fromarray(scene["rgb"]).save(root / "val" / f"{index:03d}.png")


def _build_triplets(root: Path, scenes: list, rng: np.random.Generator) -> None:
    """A NIGHTS-shaped CSV: a reference and two candidates, one badly distorted.

    The human vote goes to the *less* distorted candidate and the side it sits
    on alternates, so the drawn figure shows the check the panel exists for --
    a vote read from the wrong column would put the marker on the noisy frame.
    """
    import csv

    (root / "ref").mkdir(parents=True, exist_ok=True)
    (root / "distort").mkdir(parents=True, exist_ok=True)
    rows = []
    for index, scene in enumerate(scenes):
        rgb = scene["rgb"].astype(np.float64)
        _small(scene["rgb"]).save(root / "ref" / f"{index}.png")
        for tag, noise in (("near", 6.0), ("far", 70.0)):
            noisy = np.clip(rgb + rng.normal(0, noise, rgb.shape), 0, 255).astype(np.uint8)
            _small(noisy).save(root / "distort" / f"{index}_{tag}.png")

        near_on_right = index % 2 == 0
        rows.append(
            {
                "id": index,
                "left_vote": 0 if near_on_right else 1,
                "right_vote": 1 if near_on_right else 0,
                "votes": 8,
                "ref_path": f"ref/{index}.png",
                "left_path": f"distort/{index}_{'far' if near_on_right else 'near'}.png",
                "right_path": f"distort/{index}_{'near' if near_on_right else 'far'}.png",
                "split": "test",
            }
        )
    with (root / "data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figures(root: Path, backbone: str) -> dict[str, list[str]]:
    """The ``visbench show`` invocation behind each figure, by probe name.

    Every figure is rendered at 160px rather than the 224 a real run uses. That
    is a repository-size decision and nothing else: thirteen pages of 224px
    tiles come to roughly 8 MB, which is not a reasonable thing to commit for
    documentation. Nothing about what the panels *show* depends on it.
    """
    scene, folder, tasko = str(root), str(root / "folder"), str(root / "tasko")
    size = ["--image-size", "160"]
    features = ["--backbone", backbone, "--device", "cpu", "--cache", str(root / "cache")]
    rows = ["--frames", "3"]

    return {
        "depth": ["depth", "--data", scene, "--max-depth", "8", *rows, *size],
        "surface_normal": ["surface_normal", "--data", scene, *rows, *size],
        "semantic_segmentation": [
            "semantic_segmentation",
            "--data",
            scene,
            "--target-dir",
            "labels",
            "--num-classes",
            "4",
            "--ignore-index",
            "255",
            *rows,
            *size,
        ],
        "generic_segmentation": ["generic_segmentation", "--data", scene, *rows, *size],
        "edge": ["edge", "--data", tasko, *rows, *size],
        "keypoints2d": ["keypoints2d", "--data", tasko, *rows, *size],
        "occlusion_edge": ["occlusion_edge", "--data", tasko, *rows, *size],
        "corner": ["corner", "--data", scene, *rows, *size],
        "correspondence": [
            "correspondence",
            "--data",
            str(root / "flat"),
            "--split",
            "val",
            "--frames",
            "2",
            *size,
            *features,
        ],
        "classification": [
            "classification",
            "--data",
            folder,
            "--frames",
            "9",
            "--columns",
            "5",
        ],
        "retrieval": [
            "retrieval",
            "--data",
            folder,
            "--neighbours",
            "4",
            *rows,
            *features,
        ],
        "similarity": [
            "similarity",
            "--data",
            str(root / "triplets"),
            "--split",
            "test",
            *rows,
        ],
    }


def _detection_figure(root: Path, out: Path) -> int:
    """Detection, through the Python API rather than the CLI.

    The only figure here not produced by ``visbench show``, and the reason is
    deliberate on the CLI's part: ``_detection_kwargs`` takes the class count
    from ``VOC_CLASSES`` rather than a flag, so "the loader's classes and the
    head's width are the same fact" and cannot disagree. That is right for a
    probe whose published numbers are VOC's, and it means a generated dataset
    with its own class names cannot be named from the shell. Labelling these
    shapes with real VOC names instead would put "chair" under a sphere, which
    is worse than a one-line exception here -- a figure that lies about what it
    shows is exactly what this package exists to prevent.
    """
    from visbench.data.detection import DetectionFolderDataset
    from visbench.viz import render_probe_panels

    dataset = DetectionFolderDataset(
        root / "val",
        image_dir="JPEGImages",
        annotation_dir="Annotations",
        split="val",
        image_size=160,
        classes=CLASSES[1:],
        include_difficult=True,
    )
    render_probe_panels(dataset, "detection", [0, 1, 2], None, dataset.classes).save(out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--backbone", default="resnet18", help="only two figures need one")
    parser.add_argument("--frames", type=int, default=8, help="scenes to generate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep", type=Path, default=None, help="keep the scenes here")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    scratch = args.keep or Path(tempfile.mkdtemp(prefix="visbench-gallery-"))
    scratch.mkdir(parents=True, exist_ok=True)

    print(f"generating {args.frames} scenes in {scratch}")
    build_dataset(scratch, frames=args.frames, seed=args.seed)

    failed = []
    print("  detection                -> detection.png")
    if _detection_figure(scratch, args.out / "detection.png") != 0:
        failed.append("detection")

    for probe, argv in figures(scratch, args.backbone).items():
        destination = args.out / f"{probe}.png"
        print(f"  {probe:24s} -> {destination.name}")
        # In process, not a subprocess: `main` returns an exit code rather than
        # calling sys.exit precisely so it can be driven this way, and it is the
        # same code path `visbench show` runs.
        try:
            # SystemExit as well as a non-zero return: argparse raises it for an
            # unrecognised flag, which would otherwise take the whole run down
            # and leave the remaining figures silently unwritten.
            if visbench.cli.main(["show", *argv, "--out", str(destination), "--quiet"]) != 0:
                failed.append(probe)
        except SystemExit:
            failed.append(probe)

    if args.keep is None:
        shutil.rmtree(scratch, ignore_errors=True)

    if failed:
        print(f"\nFAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"\n{len(list(args.out.glob('*.png')))} figures in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

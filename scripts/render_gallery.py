#!/usr/bin/env python3
"""Render one ``visbench show`` figure per probe, for the docs and the README.

    python scripts/fetch_gallery_frames.py   # once, to get the photographs
    python scripts/render_gallery.py         # writes docs/_static/gallery/*.png

**The frames are real photographs, and staying inside the licence is what
decides which ground truth each figure can show.** VisBench ships to PyPI under
MIT, so the datasets its probes normally run on -- VOC, ImageNet, NYUv2,
Taskonomy, NIGHTS -- cannot supply these pages: each restricts redistribution to
some degree and none clearly grants it. ``scripts/fetch_gallery_frames.py``
sources Open Images instead, every frame CC BY 2.0 and licence-checked at fetch
time, with the attribution CC BY requires written to ``CREDITS.md``.

That buys real photographs *with real human annotation* for the probes whose
targets are annotated, and exact ground truth for the probes that compute their
own. It does not buy metric geometry, so the sixteen probes are drawn three
ways and each figure says which it is:

- **exact ground truth, computed from the frame itself** -- ``corner`` runs the
  probe's own Shi-Tomasi generator, ``correspondence`` warps by a homography
  this script chooses, and ``classification``/``retrieval`` need only which
  folder a photograph is in. Nothing is approximated.
- **real human annotation** -- ``detection``, ``generic_segmentation`` and
  ``semantic_segmentation`` draw Open Images' own boxes and instance masks.
- **a prediction, labelled as one** -- ``depth``, ``surface_normal``,
  ``keypoints2d`` and ``occlusion_edge`` need sensor or reconstruction ground
  truth that no redistributable photograph carries. Rather than fabricate a
  target, these pages drop the target column entirely and show what a *published*
  VisBench probe predicts, pulled from the Hub. The column is headed
  ``prediction`` and the footer says there is no ground truth behind it.

``edge`` is the one in between and is called out on its own page: Taskonomy's
``edge_texture`` is itself computed from the RGB frame, so the same *kind* of
target is computed here -- an intensity-gradient magnitude, 0 meaning "no edge"
and nothing masked, which is the convention the probe is scored under. It is not
Taskonomy's generator and the figure does not claim to be.

What is *not* faked anywhere: every figure is produced by the real
``visbench show`` command over a real dataset class, through the real renderers.
"""

import argparse
import csv
import json
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

#: The photographs, fetched separately. Not under ``docs/`` because nothing in
#: the built site needs to reach them -- only this script does.
FRAMES = REPO / "assets" / "gallery_frames"

#: The published probes the four prediction-only figures are drawn from. These
#: are the heads ``visbench run --push-to`` uploaded, so the figures show what
#: this project actually shipped rather than something trained here and thrown
#: away.
PREDICTION_BACKBONE = "dinov2_vitb14"
PREDICTED = ("depth", "surface_normal", "keypoints2d", "occlusion_edge")

#: The four prediction figures are drawn at 224, not the 160 the rest use.
#: Not a preference: a trained head's ``output_size`` is fitted state, measured
#: from the first batch it ever saw, so these heads emit 224x224 whatever they
#: are fed. Rendered beside a 160 crop the two panels are different sizes *and*
#: different framings -- the page lays out ragged and the prediction silently
#: describes a wider view than the photograph next to it. Matching the head is
#: the only correct option; resizing either panel to fit is the one thing
#: ``visbench.viz`` may never do.
PREDICTION_SIZE = 224


#: Void, for the pixels that are genuinely unlabelled -- see ``_label_map``.
VOID = 255


def load_frames() -> dict:
    """The fetched photographs and their annotations, or an actionable error."""
    manifest = FRAMES / "frames.json"
    if not manifest.exists():
        raise SystemExit(
            f"No photographs at {FRAMES}.\n"
            "Run:  python scripts/fetch_gallery_frames.py\n"
            "They are not committed as a dataset; the manifest records which "
            "Open Images frames the gallery is pinned to."
        )
    return json.loads(manifest.read_text())


def scene_classes(frames: dict) -> list[str]:
    """Every class the scene frames' masks name, background first.

    One list across all frames rather than per frame, because a label map's
    integers must mean the same thing on every page -- and because
    ``--num-classes`` is one number for the whole run.
    """
    labels = {
        mask["label"]
        for value in frames.values()
        if value["kind"] == "scene"
        for mask in value["masks"]
    }
    return ["background", *sorted(labels)]


def _load_rgb(frame: str) -> np.ndarray:
    with Image.open(FRAMES / "images" / f"{frame}.jpg") as image:
        return np.asarray(image.convert("RGB"))


def _load_mask(path: str) -> np.ndarray:
    with Image.open(FRAMES / path) as mask:
        return np.asarray(mask.convert("L")) > 127


def _label_map(value: dict, classes: list[str], shape: tuple) -> np.ndarray:
    """A semantic label map from Open Images instance masks.

    0 is background and is a **real class**, which is the segmentation
    convention this codebase keeps -- an unlabelled pixel is not 0, it is
    :data:`VOID`.

    And some pixels here genuinely are unlabelled. Open Images annotates a box
    for every object but a mask for only some of them, so a region inside a box
    with no mask holds an object whose extent nobody recorded. Calling that
    background would train a probe to answer "background" over an animal; it is
    exactly what an ignore index is for, so it is marked void. That is real
    annotation structure rather than a hole placed to make the figure
    interesting -- and it is why the magenta marker appears on these pages at
    all.
    """
    height, width = shape
    labels = np.zeros(shape, dtype=np.int32)
    covered = np.zeros(shape, dtype=bool)

    for mask in value["masks"]:
        pixels = _load_mask(mask["path"])
        labels[pixels] = classes.index(mask["label"])
        covered |= pixels

    for box in value["boxes"]:
        x1, y1, x2, y2 = box["xyxy_norm"]
        region = (
            slice(int(y1 * height), int(y2 * height)),
            slice(int(x1 * width), int(x2 * width)),
        )
        # Only if this box has essentially no mask under it: a box that its own
        # instance mask already explains is not unlabelled.
        window = covered[region]
        if window.size and window.mean() < 0.05:
            labels[region] = np.where(covered[region], labels[region], VOID)

    return labels


def build_scenes(frames: dict, classes: list[str]) -> list[dict]:
    """One dict per scene frame, carrying only what is genuinely known."""
    scenes = []
    for frame, value in sorted(frames.items()):
        if value["kind"] != "scene":
            continue
        rgb = _load_rgb(frame)
        shape = rgb.shape[:2]

        foreground = np.zeros(shape, dtype=bool)
        for mask in value["masks"]:
            foreground |= _load_mask(mask["path"])

        scenes.append(
            {
                "id": frame,
                "rgb": rgb,
                "mask": foreground.astype(np.float32),
                "labels": _label_map(value, classes, shape),
                "boxes": value["boxes"],
                "size": shape,
            }
        )
    return scenes


def _edge_target(rgb: np.ndarray) -> np.ndarray:
    """An intensity-gradient magnitude -- the convention, not the generator.

    Taskonomy's ``edge_texture`` is computed from its RGB frame too, so this is
    the same *kind* of target on a frame that may be redistributed. It shares
    what the figure exists to show: a non-negative magnitude map where **0 is a
    real reading** ("no edge here") and nothing is masked, which is the third of
    this codebase's four validity conventions.
    """
    grey = rgb.mean(axis=-1) / 255.0
    gy, gx = np.gradient(grey)
    return (np.hypot(gx, gy) * 1000.0).astype(np.float32)


def build_dataset(root: Path, scenes: list, classes: list[str], frames: dict) -> None:
    """Write every layout the CLI's probes expect, from the real frames."""
    for split in ("train", "val"):
        for name in ("images", "masks", "labels", "JPEGImages", "Annotations"):
            (root / split / name).mkdir(parents=True, exist_ok=True)

        for index, scene in enumerate(scenes):
            stem = f"{index:03d}"
            image = Image.fromarray(scene["rgb"])
            image.save(root / split / "images" / f"{stem}.png")
            image.save(root / split / "JPEGImages" / f"{stem}.jpg")
            np.save(root / split / "masks" / f"{stem}.npy", scene["mask"])
            np.save(root / split / "labels" / f"{stem}.npy", scene["labels"].astype(np.float32))
            _write_voc_xml(
                root / split / "Annotations" / f"{stem}.xml", scene["boxes"], scene["size"]
            )

    _build_taskonomy(root / "tasko", scenes)
    _build_labelled_folder(root / "folder", frames)
    _build_flat_folder(root / "flat", scenes)
    _build_context_folder(root / "context", frames)
    _build_triplets(root / "triplets", scenes)


def _build_taskonomy(root: Path, scenes: list) -> None:
    """Taskonomy's own layout, for the one magnitude probe with a real target.

    ``edge`` is indexed from ``splits/<partition>_<split>.csv`` rather than from
    two named folders, so it needs this shape. The target is uint16 because
    ``load_edge_map`` reads it **without mode conversion** -- Taskonomy stores
    values well past 255 and ``convert("L")`` would quantise six bits away.

    ``keypoints2d`` and ``edge_occlusion`` are deliberately absent: both are
    Taskonomy targets derived from a 3D reconstruction, which a photograph does
    not carry, so they are drawn as predictions instead.
    """
    building = "gallery"
    (root / "splits").mkdir(parents=True, exist_ok=True)
    for name in ("rgb", "edge_texture"):
        (root / name / building).mkdir(parents=True, exist_ok=True)

    rows = []
    for index, scene in enumerate(scenes):
        stem = f"point_{index}_view_0"
        rows.append({"building": building, "point": index, "view": 0})
        Image.fromarray(scene["rgb"]).save(root / "rgb" / building / f"{stem}_domain_rgb.png")
        values = np.clip(_edge_target(scene["rgb"]) * 1000.0, 0, 65535).astype(np.uint16)
        Image.fromarray(values).save(
            root / "edge_texture" / building / f"{stem}_domain_edge_texture.png"
        )

    for split in ("train", "val"):
        with (root / "splits" / f"tiny_{split}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["building", "point", "view"])
            writer.writeheader()
            writer.writerows(rows)


def _write_voc_xml(path: Path, boxes: list, size: tuple) -> None:
    """VOC's annotation shape, from normalised Open Images boxes.

    ``size`` is this frame's own, not a constant: the photographs are not all
    the same shape, and a box scaled by the wrong height lands somewhere
    plausible rather than raising. That is the failure 6c-1 spent its care on,
    arriving through the figure that exists to reveal it.
    """
    height, width = size
    body = "".join(
        f"<object><name>{box['label']}</name><difficult>0</difficult><bndbox>"
        # VOC is 1-indexed and the loader subtracts 1, so write it VOC's way.
        f"<xmin>{int(box['xyxy_norm'][0] * width) + 1}</xmin>"
        f"<ymin>{int(box['xyxy_norm'][1] * height) + 1}</ymin>"
        f"<xmax>{int(box['xyxy_norm'][2] * width) + 1}</xmax>"
        f"<ymax>{int(box['xyxy_norm'][3] * height) + 1}</ymax></bndbox></object>"
        for box in boxes
    )
    path.write_text(
        f"<annotation><size><width>{width}</width><height>{height}</height>"
        f"<depth>3</depth></size>{body}</annotation>"
    )


#: The three pooled-feature probes draw the source frame itself, so their
#: figures are as large as the files. Reduced here rather than in the renderer,
#: which must never resize -- that is the one rule ``visbench.viz`` keeps.
_TILE = (136, 104)


def _small(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb).resize(_TILE, Image.BICUBIC)


def _build_labelled_folder(root: Path, frames: dict) -> None:
    """``<split>/<class>/*.png`` from the class frames, not the scene frames.

    A separate pool because these probes need *several photographs per class* --
    leave-one-out retrieval over four images ranks each against three
    alternatives, and a linear probe needs something to separate. The scene
    frames are one-per-subject by design.
    """
    for split in ("train", "val"):
        for frame, value in sorted(frames.items()):
            if value["kind"] != "class":
                continue
            folder = root / split / value["label"]
            folder.mkdir(parents=True, exist_ok=True)
            _small(_load_rgb(frame)).save(folder / f"{frame}.png")


def _build_flat_folder(root: Path, scenes: list) -> None:
    """Unannotated images, which is all correspondence needs."""
    (root / "val").mkdir(parents=True, exist_ok=True)
    for index, scene in enumerate(scenes):
        Image.fromarray(scene["rgb"]).save(root / "val" / f"{index:03d}.png")


def _build_context_folder(root: Path, frames: dict) -> None:
    """The interiors the four prediction figures are drawn on.

    Their own folder rather than more entries in ``flat``: nothing here carries
    an annotation, so putting them among the scene frames would shift every
    index the annotated figures select by and make the two sets silently
    coupled.
    """
    directory = root / "val" / "images"
    directory.mkdir(parents=True, exist_ok=True)
    for frame, value in sorted(frames.items()):
        if value["kind"] == "context":
            Image.fromarray(_load_rgb(frame)).save(directory / f"{frame}.png")


def _build_triplets(root: Path, scenes: list) -> None:
    """A NIGHTS-shaped CSV: a reference and two candidates, one badly distorted.

    The distortions are constructed rather than human-judged -- NIGHTS' own
    triplets are not redistributable here -- so the *photographs* are real and
    the *votes* are a fixture. The human vote goes to the less distorted
    candidate and the side it sits on alternates, which is what makes the
    figure's vote-balance footer meaningful: read from the wrong CSV column, the
    marker lands on the noisy frame and the footer reports it.
    """
    rng = np.random.default_rng(0)
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


def figures(root: Path, backbone: str, classes: list[str]) -> dict[str, list[str]]:
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
        "semantic_segmentation": [
            "semantic_segmentation",
            "--data",
            scene,
            "--target-dir",
            "labels",
            "--num-classes",
            str(len(classes)),
            "--ignore-index",
            str(VOID),
            *rows,
            *size,
        ],
        "generic_segmentation": ["generic_segmentation", "--data", scene, *rows, *size],
        "edge": ["edge", "--data", tasko, *rows, *size],
        "corner": ["corner", "--data", scene, *rows, *size],
        "orientation": ["orientation", "--data", scene, *rows, *size],
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
            "8",
            "--columns",
            "4",
        ],
        "scene_classification": [
            "scene_classification",
            "--data",
            folder,
            "--frames",
            "8",
            "--columns",
            "4",
        ],
        "fine_grained_classification": [
            "fine_grained_classification",
            "--data",
            folder,
            "--frames",
            "8",
            "--columns",
            "4",
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


def _detection_figure(root: Path, out: Path, classes: list[str]) -> int:
    """Detection, through the Python API rather than the CLI.

    The only figure here not produced by ``visbench show``, and the reason is
    deliberate on the CLI's part: ``_detection_kwargs`` takes the class count
    from ``VOC_CLASSES`` rather than a flag, so "the loader's classes and the
    head's width are the same fact" and cannot disagree. That is right for a
    probe whose published numbers are VOC's, and it means a dataset with its own
    class names cannot be named from the shell.
    """
    from visbench.data.detection import DetectionFolderDataset
    from visbench.viz import render_probe_panels

    dataset = DetectionFolderDataset(
        root / "val",
        image_dir="JPEGImages",
        annotation_dir="Annotations",
        split="val",
        image_size=160,
        classes=classes[1:],
        include_difficult=True,
    )
    render_probe_panels(dataset, "detection", [0, 1, 2], None, dataset.classes).save(out)
    return 0


def _prediction_figure(probe: str, root: Path, out: Path) -> int:
    """``image | prediction`` for a probe whose ground truth cannot ship.

    There is deliberately **no target column**. A figure with three columns
    where the middle one is invented would be worse than no figure at all --
    these pages exist to show what a target looks like, so a fabricated one
    teaches the wrong convention to exactly the reader who came here to learn
    it. Two columns and a footer saying why is the honest shape.

    The head is a *published* VisBench probe pulled from the Hub, so what is
    drawn is what this project shipped rather than something fitted here to make
    a nice picture.

    One consequence worth stating, because it is the one rule this drops: a
    prediction is normally drawn against the **target's** range, so a head
    predicting uniformly half the right magnitude cannot render as correct.
    With no target there is no such range and the prediction is drawn against
    its own. That is unavoidable rather than overlooked, and it is another
    reason the column is not labelled "target".
    """
    import visbench
    from visbench.cache import FeatureCache
    from visbench.data.derived import DerivedTargetDataset
    from visbench.hub import load_probe_from_hub
    from visbench.viz.colour import display_range, target_to_rgb
    from visbench.viz.panels import _SCALAR_KINDS, _as_target_form, render_panels
    from visbench.viz.styles import style_for

    repo = f"turhancan97/visbench-{probe}-{PREDICTION_BACKBONE}"
    backbone = visbench.get_backbone(PREDICTION_BACKBONE, device="cpu")
    probe_task = load_probe_from_hub(repo, backbone=backbone, task=visbench.get_probe(probe))

    # For its *geometry*, not its target: this is the one dataset class here
    # that yields a correctly resized and centre-cropped frame without needing
    # an annotation file. Its Shi-Tomasi target is ignored. Building the crop
    # by hand instead would be a viewer applying its own geometry, which is
    # the single thing `visbench.viz` exists to never do.
    dataset = DerivedTargetDataset(
        root / "context" / "val", split="val", image_size=PREDICTION_SIZE
    )
    indices = [0, 1, 2]
    frames = dataset.subset(indices)
    features = FeatureCache(root=root / "cache").extract_dataset(
        backbone,
        frames,
        pooling=probe_task.pooling,
        layers=probe_task.layers,
        feature_mode=probe_task.feature_mode,
    )
    # No labels: `visbench show` passes the dataset's, because detection needs
    # them -- but this dataset's targets are the corner generator's, and
    # SurfaceNormalTask validates what it is handed. All four probes here are
    # dense and predict from features alone.
    predictions = probe_task.predict(features)

    style = style_for(probe)
    rows = []
    for position in range(len(indices)):
        prediction = _as_target_form(predictions[position], style)
        # Only the scalar kinds take a range. A normal map is (3, H, W) and asking
        # for a span over it is a shape error, not merely waste -- the same trap
        # `_row` documents, arriving here because this function reimplements the
        # one decision it could not reuse.
        span = display_range(prediction) if style.kind in _SCALAR_KINDS else None
        rows.append(
            (
                str(indices[position]),
                [np.asarray(frames[position][0]), target_to_rgb(prediction, style, span)],
            )
        )

    footer = (
        f"prediction only, no ground truth: {probe} needs sensor or reconstruction geometry, "
        f"which no redistributable photograph carries. Drawn from the published {repo} head."
    )
    render_panels(rows, ["image", "prediction"], footer).save(out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--backbone", default="resnet18", help="only two figures need one")
    parser.add_argument("--keep", type=Path, default=None, help="keep the datasets here")
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="omit the four Hub-backed figures (they need network and the [hub] extra)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="PROBE",
        help=(
            "render just these probes. Exists so adding one figure does not "
            "rewrite the other fifteen: re-rendering is a fresh encode, so the "
            "committed bytes would change even where the picture did not, and "
            "the diff would then hide which figure the change was actually for."
        ),
    )
    args = parser.parse_args()

    frames = load_frames()
    classes = scene_classes(frames)
    scenes = build_scenes(frames, classes)
    print(f"{len(scenes)} scene frames, {len(classes) - 1} annotated classes: {classes[1:]}")

    args.out.mkdir(parents=True, exist_ok=True)
    scratch = args.keep or Path(tempfile.mkdtemp(prefix="visbench-gallery-"))
    scratch.mkdir(parents=True, exist_ok=True)
    build_dataset(scratch, scenes, classes, frames)

    def wanted(probe: str) -> bool:
        return args.only is None or probe in args.only

    failed = []
    if wanted("detection"):
        print("  detection                -> detection.png")
        if _detection_figure(scratch, args.out / "detection.png", classes) != 0:
            failed.append("detection")

    for probe, argv in figures(scratch, args.backbone, classes).items():
        if not wanted(probe):
            continue
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

    if not args.skip_predictions:
        for probe in PREDICTED:
            if not wanted(probe):
                continue
            destination = args.out / f"{probe}.png"
            print(f"  {probe:24s} -> {destination.name}  (Hub prediction)")
            try:
                if _prediction_figure(probe, scratch, destination) != 0:
                    failed.append(probe)
            except Exception as error:  # noqa: BLE001 - report and keep going
                print(f"    {type(error).__name__}: {error}", file=sys.stderr)
                failed.append(probe)

    if args.keep is None:
        shutil.rmtree(scratch, ignore_errors=True)

    if args.only is not None:
        known = {"detection", *figures(scratch, args.backbone, classes), *PREDICTED}
        unknown = sorted(set(args.only) - known)
        if unknown:
            print(f"\nNo such probe: {', '.join(unknown)}", file=sys.stderr)
            return 1

    if failed:
        print(f"\nFAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"\n{len(list(args.out.glob('*.png')))} figures in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

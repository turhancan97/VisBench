#!/usr/bin/env python3
"""The oracle gate — what a candidate dense target could score with perfect features.

Run this **before** building a new derived-target probe, beside the tail check
and the overlap check. It is the gate photometric superpixels would have failed,
and the reason that probe was built in full and thrown away the same day.

Every other check in the derived-target gauntlet is about the *target*: how
heavy its tail is (does the magnitude protocol transfer?) and how much it
overlaps a target that already ships (is it new evidence?). None of them asks
whether the target is **recoverable from patch features at all**. A dense probe
sees one feature vector per patch, so a signal finer than a patch is not merely
hard to predict, it is absent from the input — and a target made mostly of such
signal cannot rank backbones however good they are.

The oracle asks that directly, and it costs one pass over a split: pool the
target to the feature grid, upsample it back, and score it with the probe's own
metric. No backbone, no features, no fitted head. See
:meth:`visbench.tasks.dense_base.DenseTrainingTask.evaluate_oracle`, which this
is a thin driver for.

Usage::

    # the calibration set: every shipped magnitude/orientation target, plus the
    # rejected superpixel candidate, on the pinned corner frames
    python scripts/oracle_ceiling.py

    # one candidate, at two grids, on your own frames
    python scripts/oracle_ceiling.py --targets corner --frames my/images --grids 16 7

`--frames` defaults to ``data/corner_frames/val/images``, the pinned set
``scripts/stage_corner_frames.py`` builds and the `corner` and `orientation`
boards ran on. The three Taskonomy targets are read from ``--taskonomy``, and
the two families cover that same 600-frame set — **in different orders**: a
derived dataset sorts its stems alphabetically and a Taskonomy one follows the
official split list. That costs nothing here, because the oracle is a mean over
independent per-image scores and a mean does not care about order, but it does
mean a ``--limit`` below the full 600 takes a *different subset* per family and
the rows stop being one set of pixels. Hence the default. The Taskonomy rows are
skipped with a note if that copy is absent.

(Do not reuse this pairing-free convenience for the *overlap* check, which
compares two targets frame by frame: there the orders must be reconciled by
stem, or the correlation is between unrelated frames. Index-paired, the corner
and edge targets read 0.08 where they truly correlate at 0.53 — the "targets
travel by index" hazard, arriving through a script rather than a loader.)

**Reading the table.** The oracle is an achievable score, not a proven upper
bound (see ``evaluate_oracle``'s notes), so read it as a bar and not as a
denominator. Do not report a probe's score as a percentage of its oracle.
"""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from visbench.data.derived import DerivedTargetDataset, OrientationResponse, ShiTomasiResponse
from visbench.tasks.dense_base import DenseTrainingTask
from visbench.tasks.low_level.corner import CornerTask
from visbench.tasks.low_level.edge import EdgeTask
from visbench.tasks.low_level.keypoints import Keypoint2DTask
from visbench.tasks.low_level.orientation import OrientationTask
from visbench.tasks.magnitude_base import DenseMagnitudeTask
from visbench.tasks.mid_level.occlusion_edge import OcclusionEdgeTask

REPO = Path(__file__).resolve().parent.parent
DEFAULT_FRAMES = REPO / "data" / "corner_frames" / "val" / "images"
DEFAULT_TASKONOMY = "/shared/sets/datasets/taskonomy-dataset/taskonomy"

#: Rows in each pinned split, matching TASKONOMY_LIMIT in build_corpus.sh and
#: DEFAULT_LIMIT in stage_corner_frames.py. The default so that the derived and
#: the read targets cover one set of frames rather than two subsets of it.
PINNED_FRAMES = 600

#: Grids the corpus backbones actually hand a head at 224px: 16 is DINOv2's
#: ViT/14, 14 a ViT/16, 7 a ResNet's layer4. Two are enough to show whether a
#: target's oracle *falls* with the grid, which is the shape of the answer.
DEFAULT_GRIDS = (16, 7)


class _SuperpixelProbe(DenseMagnitudeTask):
    """The rejected superpixel probe, enough of it to be scored.

    Not registered and not importable from the package: this candidate was
    measured and thrown away, and it lives here only so the gate keeps a *known
    negative* to be calibrated against. A rejection criterion with no failing
    example beside its passing ones is a threshold nobody has tested.
    """

    name = "superpixel"
    level = "low_level"
    display_name = "Photometric superpixel boundaries"
    target_noun = "boundary maps"
    correlation_key = "superpixel_correlation"
    protocol = "visbench_slic_boundary_regression"


class SlicBoundaryResponse:
    """Boundary map of a SLIC partition — a reconstruction of the rejected target.

    The original was never committed, so this is rebuilt from its recorded
    description (SLIC partition, boundary map, no compression) and tuned to
    reproduce its published pre-measurements rather than guessed at: tail mass
    in the strongest 1% of pixels 0.055, per-image ``|r|`` with ``edge_texture``
    0.267. Those two are what say this is the same candidate; the oracle number
    is in any case insensitive to the details, because *any* few-pixel-wide
    boundary map is destroyed by pooling to a 14px grid.

    Shaped like :class:`~visbench.data.derived.ShiTomasiResponse` — callable on
    a PIL image already at the working geometry — so it drops into
    :class:`~visbench.data.derived.DerivedTargetDataset` unchanged.
    """

    operator = "slic_boundary"

    def __init__(self, n_segments: int = 100, compactness: float = 10.0) -> None:
        self.n_segments = n_segments
        self.compactness = compactness

    def __call__(self, image: Image.Image) -> torch.Tensor:
        from skimage.segmentation import find_boundaries, slic

        array = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        labels = slic(
            array,
            n_segments=self.n_segments,
            compactness=self.compactness,
            start_label=1,
            channel_axis=-1,
        )
        boundaries = find_boundaries(labels, mode="thick")
        return torch.from_numpy(boundaries.astype(np.float32))

    def describe(self) -> dict[str, Any]:
        return {
            "target_operator": self.operator,
            "target_n_segments": self.n_segments,
            "target_compactness": self.compactness,
            "target_boundary_mode": "thick",
        }

    def token(self) -> str:
        return f"slic{self.n_segments}c{int(self.compactness)}"


def _derived(generator: Any) -> Callable[[argparse.Namespace], Any]:
    def build(args: argparse.Namespace) -> Any:
        return DerivedTargetDataset(
            root=Path(args.frames).parent,
            image_dir=Path(args.frames).name,
            split="val",
            image_size=args.image_size,
            generator=generator,
            max_images=args.limit,
        )

    return build


def _taskonomy(domain: str) -> Callable[[argparse.Namespace], Any]:
    def build(args: argparse.Namespace) -> Any:
        from visbench.data.taskonomy import TaskonomyDataset

        return TaskonomyDataset(
            root=args.taskonomy,
            domain=domain,
            split="val",
            partition="tiny",
            image_size=args.image_size,
            max_images=args.limit,
        )

    return build


#: One row per target: the probe that scores it, the dataset that supplies it,
#: and whether it is a shipped probe (a passing example) or a known negative.
#: A candidate is added here for one run and removed again; nothing downstream
#: reads this table, which is why it may hold a probe that does not exist.
TARGETS: dict[str, tuple[DenseTrainingTask, Callable, str]] = {
    "corner": (CornerTask(), _derived(ShiTomasiResponse()), "ships"),
    "orientation": (OrientationTask(), _derived(OrientationResponse()), "ships"),
    "edge": (EdgeTask(), _taskonomy("edge_texture"), "ships"),
    "keypoints2d": (Keypoint2DTask(), _taskonomy("keypoints2d"), "ships"),
    "occlusion_edge": (OcclusionEdgeTask(), _taskonomy("edge_occlusion"), "ships"),
    "superpixel": (_SuperpixelProbe(), _derived(SlicBoundaryResponse()), "rejected"),
}

#: Headline metric per probe family. The oracle returns whatever
#: ``_batch_metrics`` returns, and reading whichever key sorted first is the
#: mistake ``HEADLINE_METRICS`` exists to prevent.
HEADLINE = {
    "orientation": ("orientation_error", "deg, lower better, 45 = chance"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--targets",
        nargs="+",
        default=sorted(TARGETS),
        help="Targets to measure; default is every row of TARGETS.",
    )
    parser.add_argument(
        "--frames", default=str(DEFAULT_FRAMES), help="Image folder for the derived targets."
    )
    parser.add_argument(
        "--taskonomy", default=DEFAULT_TASKONOMY, help="Taskonomy root, for the read targets."
    )
    parser.add_argument(
        "--grids",
        nargs="+",
        type=int,
        default=list(DEFAULT_GRIDS),
        help="Square feature grids to pool to. 16 = ViT/14 at 224, 7 = ResNet layer4.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=PINNED_FRAMES,
        help=(
            f"Frames to score; default {PINNED_FRAMES}, the whole pinned set. Below that, "
            "the derived and Taskonomy rows describe different subsets — see the module docstring."
        ),
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    unknown = [name for name in args.targets if name not in TARGETS]
    if unknown:
        print(f"Unknown target(s): {', '.join(unknown)}. Known: {', '.join(sorted(TARGETS))}")
        return 2

    print(f"Oracle over {args.limit} frames at {args.image_size}px, grids {args.grids}")
    print(f"frames: {args.frames}\n")

    width = max(len(name) for name in args.targets) + 2
    header = f"{'target':<{width}}{'source':<10}" + "".join(
        f"{'grid ' + str(g):>12}" for g in args.grids
    )
    print(header)
    print("-" * len(header))

    failed = []
    for name in args.targets:
        probe, build, origin = TARGETS[name]
        try:
            dataset = build(args)
        except Exception as error:  # noqa: BLE001 — a missing dataset is a skip, not a crash
            failed.append(f"{name}: {type(error).__name__}: {error}")
            continue

        if name in HEADLINE:
            key, note = HEADLINE[name]
        else:
            key, note = probe.correlation_key, "correlation, higher better"
        cells = []
        for grid in args.grids:
            metrics = probe.evaluate_oracle(dataset, grid, batch_size=args.batch_size)
            cells.append(f"{metrics[key]:>12.4f}")
        print(f"{name:<{width}}{origin:<10}" + "".join(cells) + f"   {note}")

    if failed:
        print("\nSkipped:")
        for line in failed:
            print(f"  {line}")
    print(
        "\nThe oracle is an achievable score, not a proven bound: read it as a bar,\n"
        "never as a denominator for a probe's score."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

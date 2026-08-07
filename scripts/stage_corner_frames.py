#!/usr/bin/env python3
"""Stage Taskonomy RGB frames as a flat image folder, for the corner probe.

The corner probe computes its target from the image, so it runs on *any* folder
of photographs and needs no dataset. That is the point of it — and it is also
why it cannot have a leaderboard board without this script.

**A derived-target probe has no canonical data of its own.** Two people's corner
numbers are comparable only if they ran the same images, and nothing in the
probe pins which images those are. So the corpus has to name a set, and the set
this names is the one the published numbers used: the first ``--limit`` rows of
Taskonomy's ``tiny`` split lists, which is exactly what ``edge``, ``keypoints2d``
and ``occlusion_edge`` read.

**Why those frames and not a prettier choice.** The corner target correlates
0.52 with ``edge_texture`` — higher than the 0.147 between the two Taskonomy
probes that already ship separately. The claim that earns the corner probe its
place is that the two nonetheless *rank backbones differently* (CLIP-B/16 is
first on edges and third on corners). That claim is only exact if both probes
saw the same pixels. Staging anything else would make the most interesting
number in ``docs/tasks.md`` unverifiable.

**Symlinks, not copies.** 1,200 frames at ~1 MB is not the reason; identity is.
``DerivedTargetDataset.cache_identity`` keys on path, size and mtime, and a
symlink reports the target's size and mtime, so a staged frame and the original
are the same cache entry. Copying would double the feature cache for no gain.

**The stem carries the provenance.** ``allensville__point_0_view_0`` rather than
a serial number, because a flat folder otherwise loses which building a frame
came from — and Taskonomy's splits are disjoint *by building*, which is the
whole reason a val number means anything here.

Usage::

    python scripts/stage_corner_frames.py --data /path/to/taskonomy
    python scripts/stage_corner_frames.py --dest data/corner_frames --limit 600

Idempotent: an existing correct link is left alone, so re-running costs a stat
per frame and changes nothing.
"""

import argparse
import sys
from pathlib import Path

from visbench.data.taskonomy import load_taskonomy_split

# The published corner numbers used 600/600, matching TASKONOMY_LIMIT in
# scripts/build_corpus.sh. Changing it does not make the numbers better, it
# makes them incomparable with everything already quoted -- and, because the
# frames are shared with the edge probe, it breaks the cross-probe comparison
# as well as the within-probe one.
DEFAULT_LIMIT = 600

# Taskonomy's splits are disjoint by building. `val` is what gets scored;
# `train` is what the head is fitted on.
SPLITS = ("train", "val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/shared/sets/datasets/taskonomy-dataset/taskonomy"),
        help="Taskonomy root, holding rgb/ and splits/",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/corner_frames"),
        help="where to write <split>/images/; this is what --data points at for the probe",
    )
    parser.add_argument("--partition", default="tiny", help="Taskonomy download size")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"frames per split (default {DEFAULT_LIMIT}, matching the published numbers)",
    )
    return parser.parse_args()


def stage_split(root: Path, dest: Path, split: str, partition: str, limit: int) -> int:
    """Link the first ``limit`` frames of one split into ``dest/<split>/images``."""
    rows = load_taskonomy_split(root, split, partition)[:limit]
    if len(rows) < limit:
        print(f"  ! only {len(rows)} rows in {partition}_{split}.csv, wanted {limit}")

    images = dest / split / "images"
    images.mkdir(parents=True, exist_ok=True)

    linked = 0
    for building, point, view in rows:
        source = root / "rgb" / building / f"point_{point}_view_{view}_domain_rgb.png"
        if not source.is_file():
            # Late and loud, naming the frame. The split lists name up to 272k
            # frames and are deliberately not stat-ed at construction (6d-1), so
            # a missing one surfaces here rather than never.
            raise FileNotFoundError(f"{source} is named by the split list but absent")

        link = images / f"{building}__point_{point}_view_{view}.png"
        if link.is_symlink():
            if link.readlink() == source:
                continue  # already correct; idempotent re-run
            link.unlink()
        elif link.exists():
            raise FileExistsError(f"{link} exists and is not a symlink; refusing to replace it")

        link.symlink_to(source)
        linked += 1

    return linked


def main() -> None:
    args = parse_args()

    if not (args.data / "rgb").is_dir():
        sys.exit(f"No rgb/ under {args.data}. Point --data at a Taskonomy root.")

    print(f"staging {args.partition} -> {args.dest}")
    for split in SPLITS:
        linked = stage_split(args.data, args.dest, split, args.partition, args.limit)
        total = len(list((args.dest / split / "images").iterdir()))
        print(f"  {split}: {linked} new link(s), {total} frames")

    print(f"\nNow: visbench run corner --data {args.dest} --split val --train-split train")
    print("The frames are the same ones the edge probe reads, which is what makes")
    print("a corner ranking and an edge ranking comparable.")


if __name__ == "__main__":
    main()

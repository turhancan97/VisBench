#!/usr/bin/env python
"""Pre-measure a depth-ordering target: its ceiling AND its floor.

    scripts/premeasure_ordering.py
    scripts/premeasure_ordering.py --data <nyuv2_new> --frames 120

Run before building any probe whose target is an *ordering* rather than a
measurement. It needs no backbone, no features and no fitted head -- one pass
over a split's targets -- and it is what rejected relative depth ordering
before a 12-backbone board was spent on it.

**Why it measures a shortcut baseline, which the oracle gate does not.**
`scripts/oracle_ceiling.py` asks what a probe could score if the features
contained the answer. It never asked what a *trivial* answer already scores,
and a candidate needs room between the two. Relative depth ordering cleared the
gate at a 94.0% oracle and was rejected anyway, because "the lower point in the
image is nearer" scores 65.2% with no features at all: the usable band was 0.157
wide, a third of it was grid size, and the three strongest backbones landed
0.0007 apart. See `results/controls/README.md` and the gauntlet section of
`visbench/tasks/low_level/README.md`.

The four things it reports, and what each one decided:

1. SATURATION -- the depth-ratio distribution of random pairs. If ordering is
   trivial, every backbone scores the same and the probe ranks nothing.
2. THE SHORTCUT -- "the lower point is nearer", which needs no features.
   **This is the check that did not previously exist.**
3. RECOVERABILITY -- the patch-mean oracle, at several grids, which is the gate
   restated for an ordinal metric.
4. THE SAMPLING RULE -- and the finding that a minimum depth-ratio threshold
   makes the task *easier*, not harder: it raises the shortcut faster than the
   ceiling and narrows the band. The widest band is at no threshold.

Nothing here is a metric and nothing is written back. It reads a split's target
maps and prints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

#: probe3d's NYUv2 copy, which is what `probe_depth` reads.
DEFAULT_DATA = Path("/shared/sets/datasets/vision/probing_3D/nyuv2_new")
#: The working resolution every VisBench probe uses.
RES = 224
#: Grids worth reporting: DINOv2/14 at 224, a ViT-B/16, and a ResNet's layer4.
GRIDS = [16, 14, 7]
#: Minimum depth ratios to sweep. 1.0 means "no threshold".
RATIOS = [1.0, 1.05, 1.10, 1.25, 1.50, 2.00]


def centre_crop_resize(depth: np.ndarray, size: int = RES) -> np.ndarray:
    """Nearest-neighbour, matching DenseFolderDataset's rule for targets."""
    h, w = depth.shape
    scale = size / min(h, w)
    nh, nw = max(size, int(round(h * scale))), max(size, int(round(w * scale)))
    yi = np.clip((np.arange(nh) / scale).astype(int), 0, h - 1)
    xi = np.clip((np.arange(nw) / scale).astype(int), 0, w - 1)
    out = depth[yi][:, xi]
    top, left = (nh - size) // 2, (nw - size) // 2
    return out[top : top + size, left : left + size]


def pool_upsample(depth: np.ndarray, grid: int) -> np.ndarray:
    """What a linear head reading a grid x grid map could emit: patch means,
    bilinearly upsampled -- the same bottleneck evaluate_oracle models."""
    size = depth.shape[0]
    step = size // grid
    usable = step * grid
    valid = np.isfinite(depth) & (depth > 0)
    filled = np.where(valid, depth, 0.0)[:usable, :usable]
    weight = valid.astype(float)[:usable, :usable]
    blocks = filled.reshape(grid, step, grid, step).sum(axis=(1, 3))
    counts = weight.reshape(grid, step, grid, step).sum(axis=(1, 3))
    means = np.divide(blocks, counts, out=np.full_like(blocks, np.nan), where=counts > 0)
    # bilinear back up, which is LinearHead's upsample
    ys = np.linspace(0, grid - 1, size)
    xs = np.linspace(0, grid - 1, size)
    y0 = np.clip(np.floor(ys).astype(int), 0, grid - 1)
    y1 = np.clip(y0 + 1, 0, grid - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, grid - 1)
    x1 = np.clip(x0 + 1, 0, grid - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    m = np.nan_to_num(means, nan=np.nanmean(means))
    top = m[y0][:, x0] * (1 - wx) + m[y0][:, x1] * wx
    bot = m[y1][:, x0] * (1 - wx) + m[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="the nyuv2_new root")
    parser.add_argument("--split", default="test")
    parser.add_argument("--target-dir", default="depths")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--pairs", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    N_PAIRS = args.pairs
    root = args.data / args.split / args.target_dir
    if not root.is_dir():
        raise SystemExit(f"No target maps at {root} -- pass --data")
    frames = sorted(root.glob("*.npy"))[: args.frames]
    print(f"{len(frames)} frames from {root}\n")

    stats = {
        r: {"n": 0, "oracle": {g: 0 for g in GRIDS}, "vertical": 0, "chance": 0} for r in RATIOS
    }
    ratio_all = []

    for path in frames:
        depth = centre_crop_resize(np.load(path).astype(np.float64))
        valid = np.isfinite(depth) & (depth > 0)
        ys, xs = np.nonzero(valid)
        if len(ys) < 100:
            continue
        pooled = {g: pool_upsample(depth, g) for g in GRIDS}

        pick = rng.integers(0, len(ys), size=(N_PAIRS, 2))
        ay, ax = ys[pick[:, 0]], xs[pick[:, 0]]
        by, bx = ys[pick[:, 1]], xs[pick[:, 1]]
        da, db = depth[ay, ax], depth[by, bx]
        ok = (da > 0) & (db > 0)
        ratio = np.maximum(da, db) / np.minimum(da, db)
        ratio_all.append(ratio[ok])
        truth = da < db  # is a nearer than b

        for r in RATIOS:
            keep = ok & (ratio >= r) & (da != db)
            if not keep.any():
                continue
            stats[r]["n"] += int(keep.sum())
            # The vertical shortcut: predict "the LOWER point in the image is nearer".
            stats[r]["vertical"] += int(((ay > by) == truth)[keep].sum())
            stats[r]["chance"] += int(keep.sum()) // 2
            for g in GRIDS:
                p = pooled[g]
                pa, pb = p[ay, ax], p[by, bx]
                stats[r]["oracle"][g] += int(((pa < pb) == truth)[keep].sum())

    ratio_all = np.concatenate(ratio_all)
    print("Depth-ratio distribution over random valid pairs:")
    for q in (10, 25, 50, 75, 90):
        print(f"  p{q:<3d} {np.percentile(ratio_all, q):.3f}")
    print()

    print("Ordinal accuracy by minimum depth ratio (the sampling rule):")
    print(
        f"{'min ratio':>10s} {'pairs kept':>11s} {'vertical':>9s} "
        + "".join(f"{'oracle@' + str(g):>11s}" for g in GRIDS)
    )
    for r in RATIOS:
        s = stats[r]
        if not s["n"]:
            continue
        frac = s["n"] / len(ratio_all)
        row = f"{r:10.2f} {frac * 100:10.1f}% {s['vertical'] / s['n'] * 100:8.1f}%"
        row += "".join(f"{s['oracle'][g] / s['n'] * 100:10.1f}%" for g in GRIDS)
        print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

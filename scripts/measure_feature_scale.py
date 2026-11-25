#!/usr/bin/env python
"""How large is each backbone's dense feature, as the linear heads receive it? (24b)

    python scripts/measure_feature_scale.py --out results/controls/feature_scale.json
    python scripts/measure_feature_scale.py --report    # read the committed file

`scripts/run_scale_control.py` showed that multiplying two columns' dense
features by a constant reorders `generic_segmentation`, `semantic_segmentation`
and `depth` -- the linear heads fit raw features at a fixed learning rate, and
that fit is not scale-invariant. It moved only those two columns. This measures
the quantity it varied, **feature scale, for every column on the board**, so the
write-up can say how far apart the corpus's columns already are on it, and
whether the dense boards' orderings track it.

**What is measured.** The last-layer `dense` map, `dense_only`, exactly what
`LinearHead` reads -- for the first `--images` frames of the `generic_segmentation`
val split, built through the CLI's own `build_parser` + `SPECS` from the command
`build_corpus.sh` would run, so the geometry is the board's and not a copy of
it. Three statistics per backbone: the std over every value (the one 24b's 8.66
ratio was quoted in), the mean per-channel std, and the mean L2 norm of a token.
A forward pass only: no head, no cache write.

**What it cannot say.** Scale is confounded with everything else that differs
between columns -- family, objective, grid. A correlation between scale and a
board's ordering is a reason to run the scale control on more columns, never a
substitute for it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"
#: A JSON list, not ``.jsonl``: every ``.jsonl`` under `results/controls/` is a
#: file of result records, and `test_controls_stay_out_of_the_corpus.py` reads
#: them as such. This is a measurement table with no record in it.
OUT = ROOT / "results" / "controls" / "feature_scale.json"

#: The board whose frames are measured.
PROBE = "generic_segmentation"

#: The dense boards whose head is `LinearHead` on the last-layer map -- the ones
#: a feature's scale reaches directly. `detection`/`instance_segmentation` have
#: their own heads and `relative_pose` reads pooled pairs, so they are excluded
#: rather than averaged in.
LINEAR_DENSE = (
    "depth",
    "surface_normal",
    "generic_segmentation",
    "semantic_segmentation",
    "scene_parsing",
    "edge",
    "keypoints2d",
    "occlusion_edge",
    "corner",
    "orientation",
)


def measure(backbones: list[str], images: int, device: str) -> list[dict]:
    """One row per backbone: the three scale statistics over `images` frames."""
    import torch

    import visbench
    from visbench.cli.datasets import SPECS
    from visbench.cli.main import build_parser

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_crop_control import corpus_command

    rows = []
    for name in backbones:
        argv = corpus_command(PROBE, name, ROOT / "unused.jsonl")
        args = build_parser().parse_args(argv)
        dataset = SPECS[PROBE].build(args).evaluate
        backbone = visbench.get_backbone(name, device=device)
        frames = [dataset[i][0] for i in range(min(images, len(dataset)))]
        with torch.no_grad():
            dense = torch.cat(
                [
                    backbone.extract_features(backbone.preprocess(frames[i : i + 16]))["dense"]
                    .float()
                    .cpu()
                    for i in range(0, len(frames), 16)
                ]
            )
        # `dense` is (B, C, H, W): one token per grid cell, C values each.
        tokens = dense.permute(0, 2, 3, 1).reshape(-1, dense.shape[1])
        rows.append(
            {
                "backbone": name,
                "images": len(frames),
                "channels": int(dense.shape[1]),
                "dense_shape": list(dense.shape[1:]),
                "std": float(tokens.std()),
                "channel_std": float(tokens.std(dim=0).mean()),
                "token_norm": float(tokens.norm(dim=1).mean()),
            }
        )
        print(f"{name:20s} std {rows[-1]['std']:8.4f}  shape {rows[-1]['dense_shape']}", flush=True)
    return rows


def board_backbones() -> list[str]:
    """The columns of the published `generic_segmentation` board."""
    from analyse_board_correlates import load_boards

    return sorted(load_boards(CORPUS)[PROBE])


def spearman(left: dict[str, float], right: dict[str, float]) -> float:
    from analyse_board_correlates import spearman as rho

    return rho(left, right)


def partial_spearman(x: dict[str, float], y: dict[str, float], z: dict[str, float]) -> float:
    """Spearman rho of `x` and `y` with `z` held fixed (partial correlation of ranks).

    Pure, over the backbones the three share. The question it answers is whether
    scale tracks a board *beyond* what the feature grid already explains, since
    the grid is the corpus's strongest single correlate of the dense boards.
    """
    keys = sorted(set(x) & set(y) & set(z))
    r_xy = spearman({k: x[k] for k in keys}, {k: y[k] for k in keys})
    r_xz = spearman({k: x[k] for k in keys}, {k: z[k] for k in keys})
    r_zy = spearman({k: z[k] for k in keys}, {k: y[k] for k in keys})
    return (r_xy - r_xz * r_zy) / math.sqrt((1 - r_xz**2) * (1 - r_zy**2))


def report(rows: list[dict]) -> None:
    from analyse_board_correlates import STRUCTURE, load_boards

    scale = {row["backbone"]: math.log(row["std"]) for row in rows}
    tokens = {name: float(STRUCTURE[name].tokens) for name in scale}
    print(f"  {'backbone':20s} {'std':>9s} {'chan std':>9s} {'tok norm':>9s}")
    for row in sorted(rows, key=lambda r: r["std"]):
        print(
            f"  {row['backbone']:20s} {row['std']:9.4f} {row['channel_std']:9.4f} "
            f"{row['token_norm']:9.2f}"
        )
    spread = max(r["std"] for r in rows) / min(r["std"] for r in rows)
    print(f"\n  largest / smallest std: {spread:.1f}x")

    boards = load_boards(CORPUS)  # sign-adjusted: higher is better on every board
    print(f"\n  rho(scale, tokens) across the columns: {spearman(scale, tokens):+.3f}")
    print(
        f"\n  Spearman rho against each board (higher = better), n={len(scale)}:\n"
        f"  {'board':24s} {'scale':>7s} {'tokens':>7s} {'scale|tokens':>13s}"
    )
    for task in LINEAR_DENSE:
        cells = {k: v for k, v in boards[task].items() if k in scale}
        print(
            f"  {task:24s} {spearman(scale, cells):+7.3f} {spearman(tokens, cells):+7.3f} "
            f"{partial_spearman(scale, cells, tokens):+13.3f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--images", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", action="store_true", help="read --out (or the default) only")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT / "scripts"))
    path = args.out or OUT
    if not args.report:
        rows = measure(board_backbones(), args.images, args.device)
        path.write_text(json.dumps(rows, indent=1) + "\n")
    report(json.loads(path.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

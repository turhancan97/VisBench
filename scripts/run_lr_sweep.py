#!/usr/bin/env python
"""Choose one learning rate per dense board for standardised features (25b-1, #172).

    python scripts/run_lr_sweep.py --plan                       # print the matrix
    python scripts/run_lr_sweep.py depth --backbone convnext_base \\
        --results results/controls/parts/lr_sweep/depth__convnext_base.jsonl

**Why.** 25a's control showed that with ``--standardize`` a constant rescaling
can no longer move a dense score, and also that standardised features
*underfit* at probe3d's schedule (10 epochs, lr 5e-4): ``train_loss`` rose on
five of seven boards and ``surface_normal`` went from 38 to 52-57 degrees. That
rate was tuned for raw features. Before any board is re-run standardised (25b-2),
each board needs a rate chosen for standardised features.

**The rule, as the maintainer set it** (2026-11-27):

- **One rate per board**, shared by every backbone, so the tuning cannot decide
  part of the ranking.
- **Chosen on a held-out slice of the training split**, never on the split a
  board reports. The slice is fixed: the same images for every backbone, drawn
  by :func:`holdout_indices` with a recorded seed and fraction, so anyone can
  re-derive the choice.
- **Swept on four backbones that span the feature-scale range**
  (:data:`BACKBONES`), over :data:`RATES`.
- **Only the learning rate moves.** Epochs, warmup, cosine decay, weight decay,
  head and batch size are each cell's published flags, read from
  ``build_corpus.sh`` as every control here does.
- **A raw arm per cell** (no standardisation, the published lr) on the same
  slice is the bar: a board whose best standardised rate still fits clearly
  worse than raw is reported, not replaced.

Every record is a control: its dataset is a training-split subset, so its
fingerprint matches no board, and the standardised ones also carry a ``+std``
protocol. **Never write these to the corpus.**
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

import visbench
from visbench.cache import FeatureCache
from visbench.cli.datasets import spec_for
from visbench.cli.main import build_parser

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_crop_control import CORPUS, corpus_command  # noqa: E402

#: The ten boards #172 is about.
PROBES = (
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

#: Four columns spanning the measured feature scale (`feature_scale.json`):
#: the smallest and largest ConvNeXt (0.357 and 3.304), a CNN (`resnet50`,
#: 0.339) and a ViT (`dinov2_vitb14`).
BACKBONES = ("convnext_base", "clip_convnext_base", "resnet50", "dinov2_vitb14")

#: Candidate rates for standardised features; 5e-4 is probe3d's.
RATES = (2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2)

#: The second pass (2026-11-27): five boards chose 1e-2, the top of
#: :data:`RATES`, and a choice at the edge of a grid is a limit of the search
#: rather than a finding, so the grid was extended upward on exactly those.
EXTENDED_RATES = (2e-2, 5e-2)
EXTENDED_PROBES = ("generic_segmentation", "depth", "keypoints2d", "corner", "surface_normal")

#: The published rate, used by the raw arm.
PUBLISHED_LR = 5e-4

HOLDOUT_FRACTION = 0.1
HOLDOUT_SEED = 0


def holdout_indices(n: int, fraction: float = HOLDOUT_FRACTION, seed: int = HOLDOUT_SEED):
    """``(fit, held)`` index lists over a training split of ``n`` items.

    A seeded permutation, so the slice is the same for every backbone, rate and
    machine. Both halves are returned sorted, which keeps each subset in the
    split's own order.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    held_count = max(1, round(n * fraction))
    if held_count >= n:
        raise ValueError(f"a split of {n} cannot spare {held_count} items to hold out")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator).tolist()
    return sorted(order[held_count:]), sorted(order[:held_count])


def arms(rates=RATES, raw: bool = True) -> list[tuple[str, float, bool]]:
    """``(label, lr, standardize)`` for every run in one (board, backbone) cell."""
    first = [("raw", PUBLISHED_LR, False)] if raw else []
    return first + [(f"std_lr{rate:g}", rate, True) for rate in rates]


def run_cell(
    probe: str,
    backbone: str,
    results: Path,
    only: str | None = None,
    planned: list[tuple[str, float, bool]] | None = None,
) -> int:
    """Every arm of one (board, backbone) cell, appended to ``results``."""
    argv = corpus_command(probe, backbone, results)
    args = build_parser().parse_args(argv)  # argv already starts with "run"
    spec = spec_for(probe)
    splits = spec.build(args)
    if splits.train is None:
        raise SystemExit(f"{probe} has no training split; it is not a trained probe")

    fit_idx, held_idx = holdout_indices(len(splits.train))
    fit, held = splits.train.subset(fit_idx), splits.train.subset(held_idx)
    print(
        f"=== {probe} / {backbone}: fitting on {len(fit)}, choosing on {len(held)} "
        f"(holdout seed {HOLDOUT_SEED}, fraction {HOLDOUT_FRACTION})",
        flush=True,
    )

    failed = 0
    for label, lr, standardize in planned if planned is not None else arms():
        if only and label != only:
            continue
        args.lr, args.standardize = lr, standardize
        probe_obj = visbench.get_probe(probe, **spec.probe_kwargs(args))
        notes = (
            f"25b-1 lr sweep: arm={label} lr={lr:g} standardize={standardize} "
            f"holdout_seed={HOLDOUT_SEED} holdout_fraction={HOLDOUT_FRACTION} "
            f"fit={len(fit)} held={len(held)}"
        )
        try:
            result = visbench.run(
                args.backbone,
                probe_obj,
                held,
                train_dataset=fit,
                cache=FeatureCache(root=args.cache, enabled=not args.no_cache),
                results=results,
                batch_size=args.batch_size,
                seed=args.seed,
                device=args.device,
                notes=notes,
            )
        except Exception as error:  # one arm failing must not abandon the rest
            failed += 1
            print(f"!!! {probe} / {backbone} / {label}: {error!r}", flush=True)
            continue
        print(f"--- {label}: {result.metrics}", flush=True)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("probe", nargs="?", choices=PROBES)
    parser.add_argument("--backbone", choices=BACKBONES)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--arm", help="run one arm only, e.g. raw or std_lr0.002")
    parser.add_argument("--plan", action="store_true", help="print the matrix; run nothing")
    parser.add_argument(
        "--extended",
        action="store_true",
        help="the second pass: only EXTENDED_RATES, no raw arm, EXTENDED_PROBES only",
    )
    args = parser.parse_args(argv)

    if args.plan:
        cells = [(p, b) for p in PROBES for b in BACKBONES]
        print(f"{len(cells)} cells x {len(arms())} arms = {len(cells) * len(arms())} runs")
        for label, lr, standardize in arms():
            print(f"  {label:12s} lr={lr:g} standardize={standardize}")
        return 0
    if not (args.probe and args.backbone and args.results):
        parser.error("give a probe, --backbone and --results, or --plan")
    if args.results.resolve() == CORPUS.resolve():
        parser.error("--results may not be the corpus; this is a control")
    args.results.parent.mkdir(parents=True, exist_ok=True)
    planned = None
    if args.extended:
        if args.probe not in EXTENDED_PROBES:
            parser.error(f"--extended covers {EXTENDED_PROBES} only")
        planned = arms(EXTENDED_RATES, raw=False)
    return run_cell(args.probe, args.backbone, args.results, only=args.arm, planned=planned)


if __name__ == "__main__":
    raise SystemExit(main())

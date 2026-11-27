#!/usr/bin/env python
"""Measure the #173 target-alignment fix on the real datasets, with no backbone.

    python scripts/measure_target_alignment.py --out results/controls/target_alignment.json

Before #173 every dense target was resized with torch's left-aligned
``interpolate(mode="nearest")`` while its image went through PIL's centre-aligned
bicubic resize. The fix resamples targets with PIL's own nearest-neighbour
(``target_resample="pil_nearest"``, now the default). This script measures,
for every board whose targets go through that resize:

1. **The legacy gate.** With ``--target-resample torch_nearest`` the evaluation
   split's fingerprint must equal the published corpus record's
   ``dataset_fingerprint``. That proves the old behaviour is still reachable and
   that every published cell stays reproducible.
2. **What moves.** The fraction of target pixels that differ between the two
   geometries, over the first ``--limit`` evaluation items. ``NaN`` holes count as
   equal to each other.
3. **The oracle ceiling, before and after**, on the three Taskonomy magnitude
   boards that declare one. It shows what the fix does to the best score a
   linear head could reach.

Each split is built through the CLI's own parser from the exact corpus command
in ``build_corpus.sh``, as every control here does, so the data is the board's.
``corner`` and ``orientation`` compute their target after the crop and never
resample, so they are absent by construction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import visbench
from visbench.cli.datasets import spec_for
from visbench.cli.main import build_parser
from visbench.results.writer import iter_records

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_crop_control import CORPUS, corpus_command  # noqa: E402

#: Every board whose targets go through the dense resize.
PROBES = (
    "depth",
    "surface_normal",
    "generic_segmentation",
    "semantic_segmentation",
    "scene_parsing",
    "edge",
    "keypoints2d",
    "occlusion_edge",
    "instance_segmentation",
)
ORACLE_PROBES = ("edge", "keypoints2d", "occlusion_edge")
#: Any column will do: the dataset does not depend on the backbone.
BACKBONE = "dinov2_vits14"


def _splits(probe: str, method: str):
    argv = corpus_command(probe, BACKBONE, Path("/dev/null"))
    args = build_parser().parse_args([*argv, "--target-resample", method])
    return args, spec_for(probe).build(args)


def published_fingerprint(probe: str) -> str:
    fingerprints = {r.dataset_fingerprint for r in iter_records(CORPUS) if r.task == probe}
    if len(fingerprints) != 1:
        raise SystemExit(f"{probe}: expected one published fingerprint, got {fingerprints}")
    return fingerprints.pop()


def _planes(target) -> list[torch.Tensor]:
    """The tensors a target is made of: one map, or an instance dict's maps."""
    if isinstance(target, dict):
        return [v for v in target.values() if isinstance(v, torch.Tensor) and v.dim() >= 2]
    return [torch.as_tensor(target)]


def differing_fraction(a, b) -> float:
    moved = total = 0
    for x, y in zip(_planes(a), _planes(b), strict=True):
        if x.shape != y.shape:
            # Instance masks are dropped when cropped away, so the counts can differ.
            return float("nan")
        x, y = x.float(), y.float()
        same = (x == y) | (torch.isnan(x) & torch.isnan(y))
        moved += int((~same).sum())
        total += same.numel()
    return moved / max(total, 1)


def measure(probe: str, limit: int, grid: int) -> dict:
    _, legacy = _splits(probe, "torch_nearest")
    args, aligned = _splits(probe, "pil_nearest")
    published = published_fingerprint(probe)
    row: dict = {
        "probe": probe,
        "legacy_fingerprint": legacy.evaluate.fingerprint(),
        "published_fingerprint": published,
        "legacy_reproduces_published": legacy.evaluate.fingerprint() == published,
        "aligned_fingerprint_differs": aligned.evaluate.fingerprint() != published,
    }
    count = min(limit, len(aligned.evaluate))
    fractions = [
        differing_fraction(legacy.evaluate.target(i), aligned.evaluate.target(i))
        for i in range(count)
    ]
    finite = [f for f in fractions if f == f]
    row["items"] = count
    row["mean_fraction_moved"] = sum(finite) / len(finite) if finite else None
    row["items_with_changed_instance_count"] = len(fractions) - len(finite)

    if probe in ORACLE_PROBES:
        task = visbench.get_probe(probe, **spec_for(probe).probe_kwargs(args))
        for name, splits in (("legacy", legacy), ("aligned", aligned)):
            subset = splits.evaluate.subset(count)
            row[f"oracle_{name}"] = task.evaluate_oracle(subset, grid)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("probes", nargs="*", default=list(PROBES))
    parser.add_argument("--limit", type=int, default=200, help="evaluation items per board")
    parser.add_argument("--grid", type=int, default=16, help="oracle grid (16 = a ViT-B/14)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    rows = []
    for probe in args.probes:
        row = measure(probe, args.limit, args.grid)
        rows.append(row)
        print(json.dumps(row, default=str), flush=True)
    failed = [r["probe"] for r in rows if not r["legacy_reproduces_published"]]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"rows": rows}, indent=2, default=str) + "\n")
    if failed:
        print(f"!!! legacy fingerprint does NOT reproduce the corpus for: {failed}")
        return 1
    print("legacy fingerprint reproduces the corpus on every board")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

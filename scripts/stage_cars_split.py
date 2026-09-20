#!/usr/bin/env python3
"""Stage a cleaned, pinned Stanford Cars split — because the copy here is not one.

`vehicle_classification` asks the same *shape* of question as
`fine_grained_classification` — subordinate categories inside one basic-level
class — on man-made rigid objects instead of birds. That earns it a board only
if the data it reads is something two people can reproduce, and the copy on this
machine is not:

* **train has 8,148 images against Stanford Cars' official 8,144**, and eleven
  of them are *the same image filed under two different class directories* —
  contradictory supervision, and the pairs sit on adjacent class indices
  (13/14, 124/125, 176/177), which looks like a boundary artefact in whoever
  built this copy rather than anything in the benchmark;
* **test has seven such pairs**, so those items are unanswerable by
  construction: whichever label a model predicts, one copy scores wrong and the
  achievable ceiling drops below 1.0 for a reason that is a property of this
  *copy*;
* `train_cars/187/00004.png` is a 514x323 **pure white** placeholder; and
* `train.txt` / `test.txt` index `train_cars_augmented/` (195,456 images), so
  they describe a different set entirely and must not be used as split lists.

So this script pins a split **VisBench defines and says it defines**, the way
`scripts/stage_corner_frames.py` pins the corner frame set. It is *not* the
official 8,144/8,041 split, and no number measured on it may be compared with
published Stanford Cars results. That sentence is the reason this file exists
rather than a `--data` flag pointing at the raw folder.

**What is excluded.** For a *cross-class* duplicate there is no way to tell
which of the two labels is right, so keeping either would be a guess that
silently supervises against it: **both** copies go. A *within-class* duplicate is
only redundant, so the first copy is kept and the rest go — a repeated
evaluation item would otherwise count twice in the metric. The blank goes for
the obvious reason. Every exclusion is named in the manifest with its reason,
and what is left satisfies one sentence: **no image appears twice, under any
label.**

**Symlinks, not copies.** `cache_identity` keys on path, size and mtime, and a
symlink reports its target's, so a staged image and the original share one cache
entry — the same argument the corner staging makes, and it matters more here
because the pre-measurement already extracted these 16k images.

**The manifest is committed and checked.** `data/cars_split_manifest.json` lists
every excluded file with its reason and the resulting counts;
`tests/scripts/test_stage_cars_split.py` recomputes the exclusions from the
dataset when it is present and compares, so a copy that is *differently* broken
cannot be staged as though it were this one.

Usage::

    python scripts/stage_cars_split.py --data /shared/sets/datasets/stanford_cars
    python scripts/stage_cars_split.py --audit          # report, stage nothing

Idempotent: an existing correct link is left alone.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = Path("/shared/sets/datasets/stanford_cars")
DEFAULT_DEST = ROOT / "data" / "cars_split"
MANIFEST = ROOT / "data" / "cars_split_manifest.json"

#: The raw directories, and the split names VisBench stages them under. `val`
#: rather than `test`, because every other folder probe here scores a split
#: named `val` and a second spelling would only invite `--split test` on one
#: probe and not the others.
SOURCES = {"train": "train_cars", "val": "test_cars"}

#: A file smaller than this is *checked* for blankness rather than assumed bad;
#: the threshold only decides what is worth decoding.
SMALL_BYTES = 5000


def digest(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def is_blank(path: Path) -> bool:
    """One or two distinct colours over the whole frame — a placeholder, not a car."""
    array = np.asarray(Image.open(path).convert("RGB"))
    return len(np.unique(array.reshape(-1, 3), axis=0)) <= 2


def audit(source: Path) -> dict:
    """What is wrong with one raw split, computed rather than remembered."""
    files = sorted(source.rglob("*.png"))
    by_hash: dict[str, list[Path]] = collections.defaultdict(list)
    for path in files:
        by_hash[digest(path)].append(path)

    duplicates = {h: paths for h, paths in by_hash.items() if len(paths) > 1}
    cross_class = {
        h: paths for h, paths in duplicates.items() if len({p.parent.name for p in paths}) > 1
    }
    within_class = {h: paths for h, paths in duplicates.items() if h not in cross_class}
    blanks = [p for p in files if p.stat().st_size < SMALL_BYTES and is_blank(p)]
    return {
        "files": files,
        "cross_class": cross_class,
        "within_class": within_class,
        "blanks": blanks,
    }


def excluded(report: dict) -> dict[Path, str]:
    """Every file to drop, with the reason that drops it."""
    drop: dict[Path, str] = {}
    for paths in report["cross_class"].values():
        classes = "/".join(sorted(p.parent.name for p in paths))
        for path in paths:
            # BOTH members: a cross-class duplicate gives no way to tell which
            # label is the right one, so keeping either is a guess.
            drop[path] = f"same image under classes {classes}"
    for paths in report["within_class"].values():
        # Not a contradiction, only redundancy -- so the FIRST copy is kept and
        # the rest go. A repeated evaluation item would otherwise count twice in
        # the metric, and the rule "no image appears twice, under any label" is
        # one sentence to state and one query to check.
        keeper, *rest = sorted(paths)
        for path in rest:
            drop[path] = f"duplicate of {keeper.parent.name}/{keeper.name}"
    for path in report["blanks"]:
        drop[path] = "blank image"
    return drop


def stage(source: Path, dest: Path, drop: dict[Path, str]) -> int:
    staged = 0
    for path in sorted(source.rglob("*.png")):
        if path in drop:
            continue
        link = dest / path.parent.name / path.name
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.readlink() == path:
                staged += 1
                continue
            link.unlink()
        elif link.exists():
            raise FileExistsError(f"{link} exists and is not a symlink; refusing to replace it")
        link.symlink_to(path)
        staged += 1
    return staged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--audit", action="store_true", help="report and stage nothing")
    args = parser.parse_args(argv)

    if not args.data.is_dir():
        print(f"no Stanford Cars copy at {args.data}", file=sys.stderr)
        return 1

    manifest: dict = {"source": str(args.data), "splits": {}}
    for split, folder in SOURCES.items():
        source = args.data / folder
        report = audit(source)
        drop = excluded(report)
        kept = len(report["files"]) - len(drop)
        print(f"=== {split} ({folder})")
        print(f"    raw {len(report['files'])} files")
        print(
            f"    cross-class duplicate groups {len(report['cross_class'])}"
            f" -> {sum(len(v) for v in report['cross_class'].values())} files dropped"
        )
        print(
            f"    within-class duplicate groups {len(report['within_class'])}"
            f" -> {sum(len(v) - 1 for v in report['within_class'].values())} files dropped"
        )
        print(f"    blank {len(report['blanks'])} -> dropped")
        print(f"    staged {kept}")
        manifest["splits"][split] = {
            "source_folder": folder,
            "raw_files": len(report["files"]),
            "kept": kept,
            "excluded": [
                {"path": str(path.relative_to(source)), "reason": drop[path]}
                for path in sorted(drop)
            ],
        }
        if not args.audit:
            args.dest.mkdir(parents=True, exist_ok=True)
            staged = stage(source, args.dest / split, drop)
            assert staged == kept, f"staged {staged} against {kept} expected"

    if args.audit:
        print("\n--audit: nothing staged")
        return 0

    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nstaged under {args.dest}")
    print(f"manifest    {args.manifest}")
    print("\nThis is a VisBench-defined split, NOT Stanford Cars' official")
    print("8,144/8,041 one. Numbers measured on it are not comparable with")
    print("published Cars results, and the probe's docs page says so.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

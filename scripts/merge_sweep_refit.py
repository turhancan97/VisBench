#!/usr/bin/env python
"""Replace the re-fitted columns of every committed seed sweep (23b).

    python scripts/merge_sweep_refit.py results/controls/parts/23b           # dry run
    python scripts/merge_sweep_refit.py results/controls/parts/23b --write

23a moved the nine timm columns to new ``.../crop1`` keys, so their rows in every
committed sweep re-fit a board that no longer exists. 23b re-swept those columns
only -- the other four kept their keys, so their rows still describe the
published board -- and this script splices the two together.

**Why it replaces rather than appends.** ``merge_controls.sh`` appends, which is
right for a re-run of the same configuration. Here the old rows are a different
configuration under a different key, and a sweep file holding both would give
one backbone two keys -- which ``pinned_to_keys`` refuses, because there would
be no single board the sweep was fitted against. So the old file is copied,
unchanged, to ``results/controls/pre23a/`` (the history: 20a-22b were computed
from it), and the live file keeps every row of an unchanged column plus the new
rows of a re-fitted one.

**What it refuses**, before writing anything: a staged part that is not a
``/crop1`` key or not a V100; a re-fitted column with any seed missing or twice;
a target whose result would not hold every backbone it held before, at every
seed it held before; and a result that still carries a key ``SUPERSEDED_KEYS``
retired, which is a column the re-fit missed. Each of those would leave a sweep
that silently describes fewer pairs than the board has.

Dry run by default; ``--write`` copies the archive (once -- an existing archive
is never overwritten) and rewrites the live files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from visbench.results.leaderboard import SUPERSEDED_KEYS

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ROOT / "results" / "controls"
ARCHIVE = CONTROLS / "pre23a"
CORRECTED = "/crop1"

#: Where a sweep for a probe lives. Two predate `seeds/` (see
#: `scripts/analyse_placements.py`): pose was the first sweep, and
#: `scene_classification`'s is held out for not reproducing its board.
SPECIAL = {
    "relative_pose": "pose_seeds.jsonl",
    "scene_classification": "scene_classification_seeds.jsonl",
}


def target_for(probe: str) -> Path:
    """The committed sweep file a probe's re-fit replaces rows in, relative to controls."""
    return Path(SPECIAL.get(probe, f"seeds/{probe}.jsonl"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def staged(parts: Path) -> dict[str, list[dict]]:
    """``{probe: records}`` from a staging directory, refusing anything uncorrected."""
    out: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(parts.glob("*.jsonl")):
        for record in read_jsonl(path):
            if not record["backbone_key"].endswith(CORRECTED):
                raise SystemExit(f"{path.name}: {record['backbone_key']} is not a corrected key")
            gpu = (record.get("hardware") or {}).get("gpu") or ""
            if "V100" not in gpu:
                raise SystemExit(f"{path.name}: ran on {gpu!r}, not a V100")
            out[record["task"]].append(record)
    return dict(out)


def splice(old: list[dict], new: list[dict]) -> list[dict]:
    """The old sweep with every re-fitted column's rows replaced by the new ones.

    Pure, so the refusals below are fast-testable. Old rows come first in their
    original order, so a diff of the live file shows the replaced columns only.
    """
    seeds_before: dict[str, set[int]] = defaultdict(set)
    for record in old:
        seeds_before[record["backbone"]].add(record["seed"])

    refit: dict[str, list[dict]] = defaultdict(list)
    for record in new:
        refit[record["backbone"]].append(record)
    for backbone, rows in refit.items():
        seeds = [r["seed"] for r in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{backbone}: a seed appears twice in the re-fit")
        if backbone in seeds_before and set(seeds) != seeds_before[backbone]:
            raise ValueError(
                f"{backbone}: re-fit seeds {sorted(seeds)} differ from the sweep's "
                f"{sorted(seeds_before[backbone])}"
            )
        if len({r["backbone_key"] for r in rows}) != 1:
            raise ValueError(f"{backbone}: the re-fit carries two keys")

    kept = [r for r in old if r["backbone"] not in refit]
    merged = kept + [r for b in refit for r in sorted(refit[b], key=lambda r: r["seed"])]

    seeds_after: dict[str, set[int]] = defaultdict(set)
    for record in merged:
        seeds_after[record["backbone"]].add(record["seed"])
    if dict(seeds_after) != dict(seeds_before):
        missing = sorted(set(seeds_before) - set(seeds_after))
        extra = sorted(set(seeds_after) - set(seeds_before))
        raise ValueError(f"the merge changes the sweep's shape: missing {missing}, extra {extra}")
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("parts", type=Path, help="staging directory holding only 23b's parts")
    parser.add_argument("--write", action="store_true", help="archive and rewrite; else dry run")
    args = parser.parse_args(argv)

    by_probe = staged(args.parts)
    if not by_probe:
        print(f"no staged records under {args.parts}", file=sys.stderr)
        return 1
    for probe in sorted(by_probe):
        relative = target_for(probe)
        live = CONTROLS / relative
        if not live.is_file():
            raise SystemExit(f"{probe}: no committed sweep at {live}")
        archived = ARCHIVE / relative
        old = read_jsonl(archived if archived.is_file() else live)
        try:
            merged = splice(old, by_probe[probe])
        except ValueError as error:
            raise SystemExit(f"{probe}: {error}") from None
        stale = sorted({r["backbone"] for r in merged if r["backbone_key"] in SUPERSEDED_KEYS})
        if stale:
            raise SystemExit(f"{probe}: still on a superseded key after the merge: {stale}")
        refitted = sorted({r["backbone"] for r in by_probe[probe]})
        print(
            f"{probe:28s} {len(old):3d} rows -> {len(merged):3d}; "
            f"{len(refitted)} columns re-fitted -> {relative}"
        )
        if args.write:
            if not archived.is_file():
                archived.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(live, archived)
            live.write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in merged), encoding="utf-8"
            )
    if not args.write:
        print("\n(dry run; pass --write to archive and rewrite)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

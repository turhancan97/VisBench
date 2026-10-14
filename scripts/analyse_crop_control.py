#!/usr/bin/env python
"""What the timm double crop does to the published boards (22c).

    python scripts/analyse_crop_control.py
    python scripts/analyse_crop_control.py --control results/controls/parts/crop/*.jsonl

Reads the corpus and the control written by ``scripts/run_crop_control.py`` and
answers three questions, in this order, because each is void without the one
before it:

1. **Does the gate reproduce?** Gate records re-ran a published name through
   the control's own code path. If they miss their corpus cells beyond the
   board's reproducibility floor, the control is not re-running what the board
   ran, and nothing below means anything.
2. **How far does each timm row move without the second crop?** Per board, the
   published score against the full-frame one, signed so that **positive means
   the published number understated the backbone**.
3. **Does the board's ordering change?** Each timm cell is replaced by its
   full-frame value -- the four DINOv2/open_clip rows already preprocess with a
   no-op, so they stand as published -- and the board is re-ranked: Spearman
   against the published order, the leader, and every row that moves.

It imports no visbench (the tables it needs come from
``analyse_board_correlates.py``, loaded by path), so it runs on a login node.
Nothing here is written back anywhere.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"
CONTROL = ROOT / "results" / "controls" / "crop_control.jsonl"
PARTS = ROOT / "results" / "controls" / "parts" / "crop"
SUFFIX = "_fullframe"

#: How close a gate cell must come to its published value. The linear dense
#: boards and the zero-shot ones reproduce to ~1e-5 or better (20b/20c); a gate
#: missing by more than this is a different configuration, not jitter.
GATE_TOLERANCE = 1e-4

#: The nine timm corpus columns, i.e. the rows a second crop touches. Restated
#: rather than read from `_VARIANTS` so this script needs no visbench; a test
#: pins the two equal.
TIMM_COLUMNS = (
    "convnext_base",
    "dino_vitb16",
    "dino_vitb8",
    "mae_vitb16",
    "resnet18",
    "resnet50",
    "sam_vitb16",
    "siglip_vitb16",
    "supervised_vitb16",
)


def _correlates():
    path = Path(__file__).resolve().parent / "analyse_board_correlates.py"
    spec = importlib.util.spec_from_file_location("analyse_board_correlates", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("analyse_board_correlates", module)
    spec.loader.exec_module(module)
    return module


correlates = _correlates()


def _records(paths: list[Path]) -> list[dict]:
    out = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def published() -> dict[tuple[str, str], dict]:
    """The pre-23a cell per (task, backbone); file order, so the newest wins.

    Records under a corrected key (``.../crop1``, 23a) are skipped, so this
    keeps meaning "what was published before the fix" after the corrected
    records are merged beside the old ones.
    """
    return {
        (r["task"], r["backbone"]): r
        for r in _records([CORPUS])
        if not r["backbone_key"].endswith("/crop1")
    }


def rerun_report(paths: list[Path], control: list[dict], cells: dict) -> bool:
    """Gate and summarise a staged 23a re-run before it is merged.

    Every record must carry a corrected key, a V100 and a fit. On the four
    boards 22c measured, each must reproduce the control's full-frame value --
    the same code path, so a miss means something else moved. Then, per board,
    what the correction changes: the old and new headline, and the leader.
    """
    rerun = _records(paths)
    full = {
        (r["task"], r["backbone"].removesuffix(SUFFIX)): r
        for r in control
        if r["backbone"].endswith(SUFFIX)
    }
    print(f"=== 23a re-run: {len(rerun)} staged records ===")
    ok = True
    for record in rerun:
        problems = []
        if not record["backbone_key"].endswith("/crop1"):
            problems.append(f"key {record['backbone_key']}")
        if "V100" not in ((record.get("hardware") or {}).get("gpu") or ""):
            problems.append(f"gpu {(record.get('hardware') or {}).get('gpu')}")
        if record["task"] not in ("retrieval", "correspondence", "similarity") and not record.get(
            "training"
        ):
            problems.append("no training block")
        if problems:
            ok = False
            print(f"  BAD {record['task']}/{record['backbone']}: {', '.join(problems)}")

    print("\n  the gate: re-run vs 22c's full-frame control, same code path")
    for record in sorted(rerun, key=lambda r: (r["task"], r["backbone"])):
        twin = full.get((record["task"], record["backbone"]))
        if twin is None:
            continue
        delta = headline(record) - headline(twin)
        passed = abs(delta) <= GATE_TOLERANCE
        ok &= passed
        if not passed:
            print(f"    MISS {record['task']:22s} {record['backbone']:18s} {delta:+.2e}")
    gated = sum(1 for r in rerun if (r["task"], r["backbone"]) in full)
    print(f"    {gated} records gated, {'all within' if ok else 'NOT all within'} {GATE_TOLERANCE}")

    print("\n  per board: what the correction changes")
    by_task: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in rerun:
        by_task[record["task"]][record["backbone"]] = record
    for task in sorted(by_task):
        board = {b: oriented(task, headline(r)) for (t, b), r in cells.items() if t == task}
        new = dict(board)
        for backbone, record in by_task[task].items():
            new[backbone] = oriented(task, headline(record))
        rho = correlates.spearman(board, new)
        lead_old, lead_new = max(board, key=board.get), max(new, key=new.get)
        moved = sum(1 for b in board if correlates.ranks(board)[b] != correlates.ranks(new)[b])
        change = "" if lead_old == lead_new else f"  leader {lead_old} -> {lead_new}"
        print(
            f"    {task:28s} rows {len(by_task[task])}/9  rho {rho:+.3f}  moved {moved:2d}{change}"
        )
    return ok


def headline(record: dict) -> float:
    return float(record["metrics"][correlates.HEADLINE_METRICS[record["task"]]])


def oriented(task: str, value: float) -> float:
    """Higher is better everywhere, so a positive delta always means 'gained'."""
    return -value if task in correlates.LOWER_IS_BETTER else value


def gate_report(gates: list[dict], cells: dict) -> bool:
    print("=== 1. the gate: published names through the control's path ===")
    if not gates:
        print("  no gate records -- run with --gate first; nothing below is checkable")
        return False
    ok = True
    for record in sorted(gates, key=lambda r: r["task"]):
        cell = cells[(record["task"], record["backbone"])]
        delta = headline(record) - headline(cell)
        same_key = record["backbone_key"] == cell["backbone_key"]
        passed = abs(delta) <= GATE_TOLERANCE and same_key
        ok &= passed
        print(
            f"  {record['task']:22s} {record['backbone']:16s} delta {delta:+.2e}  "
            f"key {'same' if same_key else 'DIFFERS'}  {'ok' if passed else 'FAILED'}"
        )
    print(f"  -> {'gate passed' if ok else 'GATE FAILED: the control is void'}")
    return ok


def board_report(full: list[dict], cells: dict) -> None:
    by_task: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in full:
        by_task[record["task"]][record["backbone"].removesuffix(SUFFIX)] = record

    print("\n=== 2./3. per board: the second crop's cost, and the ordering ===")
    print("  delta > 0 means the published (cropped) number UNDERSTATED the row\n")
    for task in sorted(by_task):
        rows = by_task[task]
        board = {b: headline(r) for (t, b), r in cells.items() if t == task}
        corrected = dict(board)
        print(f"--- {task} ({correlates.HEADLINE_METRICS[task]}), {len(rows)} timm rows")
        deltas = []
        for backbone in sorted(rows):
            before, after = board[backbone], headline(rows[backbone])
            delta = oriented(task, after) - oriented(task, before)
            deltas.append(delta)
            corrected[backbone] = after
            print(f"    {backbone:20s} {before:9.4f} -> {after:9.4f}   {delta:+.4f}")
        missing = sorted(set(TIMM_COLUMNS) - set(rows))
        gained = sum(1 for d in deltas if d > 0)
        print(
            f"    gained on {gained} of {len(deltas)}; mean {sum(deltas) / len(deltas):+.4f}"
            + (f"; NOT YET RUN: {missing}" if missing else "")
        )

        pub = {b: oriented(task, v) for b, v in board.items()}
        new = {b: oriented(task, v) for b, v in corrected.items()}
        rho = correlates.spearman(pub, new)
        lead_pub, lead_new = max(pub, key=pub.get), max(new, key=new.get)
        rank_pub = correlates.ranks(pub)
        rank_new = correlates.ranks(new)
        moved = sorted(
            (b for b in pub if rank_pub[b] != rank_new[b]),
            key=lambda b: rank_pub[b],
        )
        print(
            f"    ordering: rho(published, corrected) = {rho:+.3f}; leader "
            f"{lead_pub}" + ("" if lead_pub == lead_new else f" -> {lead_new}  ** CHANGES **")
        )
        for b in moved:
            print(f"      {b:20s} rank {rank_pub[b]:4.1f} -> {rank_new[b]:4.1f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--control",
        type=Path,
        nargs="*",
        default=None,
        help="control JSONL files; default: the committed crop_control.jsonl, else the parts",
    )
    parser.add_argument(
        "--rerun",
        type=Path,
        default=None,
        help="a staged 23a parts directory to gate and summarise before merging",
    )
    args = parser.parse_args(argv)
    if args.rerun is not None:
        staged = sorted(args.rerun.glob("*.jsonl"))
        if not staged:
            print(f"no staged records under {args.rerun}", file=sys.stderr)
            return 1
        return 0 if rerun_report(staged, _records([CONTROL]), published()) else 1
    if args.control:
        paths = args.control
    elif CONTROL.is_file():
        paths = [CONTROL]
    else:
        paths = sorted(PARTS.glob("*.jsonl"))
    if not paths:
        print(f"no control records under {PARTS}", file=sys.stderr)
        return 1
    records = _records(paths)
    cells = published()
    gates = [r for r in records if not r["backbone"].endswith(SUFFIX)]
    full = [r for r in records if r["backbone"].endswith(SUFFIX)]
    gate_ok = gate_report(gates, cells)
    if full:
        board_report(full, cells)
    if not gate_ok:
        print("\n(the per-board numbers above are not interpretable until the gate passes)")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

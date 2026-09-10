#!/usr/bin/env python3
"""Would re-running a board today land its records in the SAME board?

The corpus is append-only, so a board is re-run by appending twelve new records
beside the twelve they supersede and letting ``latest_per_backbone`` pick the
newest. That works only if the new records are *comparable* with the old ones:
``comparability_key`` groups on the dataset fingerprint, the split, the
protocol, the pooling **request**, and every entry of ``task_params`` and
``dataset_params``. If any of those has moved since the board ran -- a new
task parameter, a re-staged dataset, a changed default -- the re-run does not
join the board. It makes that board **two groups of twelve**, and
``render_tables.py::board_for`` refuses to render a task with more than one
comparability group, so the board disappears from `LEADERBOARD.md` rather than
disagreeing with it.

    scripts/preflight_regroup.py                       # every board in the corpus
    scripts/preflight_regroup.py depth surface_normal  # only these

It asks the question with **no GPU and no backbone**: each board's command comes
out of ``build_corpus.sh`` itself under ``DRY_RUN=1``, so there is no second copy
of the flags to drift, and every field of the key is computed through the same
``spec.build`` / ``get_probe`` path ``visbench run`` uses.

Two things it cannot do, both stated rather than papered over:

* A board whose task declares ``layers`` needs a backbone to resolve them
  against, since ``[-4, -1]`` names different blocks on a 12- and a 24-block
  ViT and the *resolved* list is what the key carries. Such a board is reported
  as unresolved rather than guessed at. No board in the corpus declares any
  today.
* ``ClassificationTask.num_classes`` is ``None`` until ``fit()`` infers it from
  the labels, and the runner describes the task *after* fitting. Read as-is it
  would report ``None`` against a corpus record saying 10, 365 or 200 and flag
  all three linear-probe boards as splitting when they do not, so the value
  ``fit()`` will infer is supplied here from the labelled folder itself.

Written for the schema-v8 ``training`` re-run, where eight boards produced
between v0.5.0 and v0.12.0 were re-run five releases later. It reported all
eight comparable, and the one thing it did prove non-trivially was that
regenerating ``data/voc_binary/`` reproduces ``generic_segmentation``'s
fingerprint exactly -- the fingerprint reads file names and sizes, not mtimes.
"""

from __future__ import annotations

import dataclasses
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import visbench  # noqa: E402
from visbench.cli.datasets import spec_for  # noqa: E402
from visbench.cli.main import build_parser  # noqa: E402
from visbench.results import ResultRecord, comparability_key, read_records  # noqa: E402

# The runner's own set, imported rather than copied. Whatever a dataset
# describes *beyond* these lands in `dataset_params`, which the comparability
# key groups on -- so a copy that drifted by one name would compute a different
# dict here than the run will, and this check would clear a board that splits.
# (This script already imports visbench, unlike the analyse_* scripts, which
# duplicate their tables to stay importable without torch.)
from visbench.runner import _RECORD_FIELDS  # noqa: E402

CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"
BUILD_CORPUS = ROOT / "scripts" / "build_corpus.sh"

#: The backbone the check runs as. Any one will do -- the key carries no
#: backbone -- and this is the one every board has a record for.
BACKBONE = "dinov2_vits14"


class NoCommand(RuntimeError):
    """`build_corpus.sh` produced no command for a probe, usually because the
    data it reads has to be staged first."""


def command_for(probe: str) -> list[str]:
    """The exact argv ``build_corpus.sh`` would run for ``probe``."""
    result = subprocess.run(
        ["bash", str(BUILD_CORPUS), probe],
        cwd=ROOT,
        env={"DRY_RUN": "1", "BACKBONES": BACKBONE, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    commands = [line for line in result.stdout.splitlines() if line.startswith("visbench run ")]
    if len(commands) != 1:
        # `build_corpus.sh` skips a probe whose staged data is absent -- the
        # binarised VOC masks, or the pinned corner/orientation frames -- and
        # says so on stderr. Passing that through is the difference between
        # "stage this folder" and "expected 1 command, got 0".
        skipped = [line for line in result.stderr.splitlines() if "SKIPPED" in line]
        raise NoCommand(
            f"{probe}: build_corpus.sh produced no command"
            + ("\n  " + "\n  ".join(skipped) if skipped else "")
        )
    return shlex.split(commands[0])[1:]


def fresh_record(argv: list[str]) -> ResultRecord:
    """What today's code would record, bar the numbers, for that command line."""
    args = build_parser().parse_args(argv)
    spec = spec_for(args.probe)
    splits = spec.build(args)
    task = visbench.get_probe(args.probe, **spec.probe_kwargs(args))

    if getattr(task, "num_classes", "absent") is None:
        labels = splits.train.labels() if splits.train is not None else splits.evaluate.labels()
        task.num_classes = int(max(labels)) + 1

    dataset_described = splits.evaluate.describe()
    described = {**dataset_described, **task.describe()}
    return ResultRecord(
        backbone=BACKBONE,
        backbone_key="preflight",
        task=described["task"],
        level=described["level"],
        dataset=described["dataset"],
        split=described["split"],
        dataset_size=described["dataset_size"],
        dataset_fingerprint=described["dataset_fingerprint"],
        pooling="preflight",
        pooling_requested=str(getattr(task.pooling, "value", task.pooling)),
        feature_mode=described["feature_mode"],
        # Left unresolved deliberately: resolving needs a backbone, and a
        # guessed value would compare equal to nothing.
        layers=None if task.layers is None else ["<unresolved>"],
        task_params=described["task_params"],
        finetune=task.finetune(),
        dataset_params={
            key: value for key, value in dataset_described.items() if key not in _RECORD_FIELDS
        },
        metrics={},
        timestamp="preflight",
        visbench_version=visbench.__version__,
    )


def main(argv: list[str] | None = None) -> int:
    wanted = list(argv if argv is not None else sys.argv[1:])

    latest: dict[str, ResultRecord] = {}
    for record in read_records(CORPUS):
        if record.backbone == BACKBONE:
            latest[record.task] = record
    if not wanted:
        wanted = sorted(latest)

    unknown = sorted(set(wanted) - set(latest))
    if unknown:
        print(f"No {BACKBONE} record in the corpus for {unknown}", file=sys.stderr)
        return 1

    splitting = []
    for probe in wanted:
        old = latest[probe]
        try:
            new = fresh_record(command_for(probe))
        except NoCommand as error:
            print(f"SKIP {probe:32s} {error}")
            continue
        old_key, new_key = comparability_key(old), comparability_key(new)
        if old_key == new_key:
            print(f"OK   {probe:32s} joins its board (corpus record from v{old.visbench_version})")
        else:
            splitting.append(probe)
            print(f"DIFF {probe:32s} would SPLIT the board -- board becomes unrenderable")
            for field in dataclasses.fields(new_key):
                before, after = getattr(old_key, field.name), getattr(new_key, field.name)
                if before != after:
                    print(
                        f"       {field.name}:\n         corpus: {before}\n         today : {after}"
                    )
        if new.layers is not None:
            print(f"     ?? {probe} declares layers; this check cannot resolve them")

    print()
    if splitting:
        print(f"{len(splitting)} board(s) would split: {splitting}")
        return 1
    print(f"all {len(wanted)} board(s) would take a re-run into the existing group")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

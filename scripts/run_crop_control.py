#!/usr/bin/env python
"""The double-crop control (22c): what the timm rows score without a second crop.

    python scripts/run_crop_control.py --check                       # no GPU, no runs
    python scripts/run_crop_control.py correspondence --backbones convnext_base \\
        --results results/controls/parts/crop__correspondence__convnext_base.jsonl
    python scripts/run_crop_control.py depth --gate --backbones resnet18 \\
        --results /tmp/gate.jsonl                                    # must reproduce

**What it measures.** Every VisBench dataset resizes and centre-crops to the
working size before a backbone sees an image. DINOv2 and open_clip then resize
and crop to that same size, which is a no-op. A timm backbone applies its
checkpoint's eval transform instead -- resize to ``size / crop_pct``, crop
``size`` -- so at ``crop_pct`` 0.875-0.95 it crops a second time, and its
features cover the central 87.5-95% of a frame whose dense target, and whose
correspondence homography, span all of it. Nine of the thirteen corpus columns
are timm. This re-runs them with ``crop_pct=1.0`` -- the same no-op the other
four already get -- under **the exact flags their published cells used**.

**Why not a flag on build_corpus.sh, and why in-process.** The full-frame rows
are registered here, in this process only, as ``<variant>_fullframe``: never in
``list_backbones()``, so no table, test or corpus array can pick them up. Each
cell's command is the one ``DRY_RUN=1 scripts/build_corpus.sh`` prints, with
only ``--backbone`` and ``--results`` changed, and it goes through
``visbench.cli.main`` -- so ``run()`` seeds before it constructs the backbone,
exactly as for the published cell. Constructing it here instead would put the
random init outside the seeded window and every trained number would move for
a reason no record carries.

**Since 23a the default is full-frame**, so this script now exists to
reproduce 22c rather than to change anything: ``--gate`` asks for the legacy
crop explicitly (``crop_pct=None`` under the variant's own name) and still
reproduces a pre-23a cell; the full-frame rows match the new default's key.

**The gate is free and the rest is void without it.** ``--gate`` runs the
published name through the same path; it must reproduce the corpus cell to
that board's reproducibility floor, or this script is not re-running what the
board ran.

Records go to ``--results``, which may not be the corpus: ``seed`` is not the
only field ``comparability_key`` ignores, and a ``_fullframe`` record carries a
backbone name no board has, so it would sit in the corpus as a fourteenth row.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

import visbench
from visbench.backbones.timm_backbone import _VARIANTS, TimmBackbone
from visbench.cli.main import main as cli_main
from visbench.registry import register_backbone

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "results" / "corpus" / "visbench.jsonl"
SUFFIX = "_fullframe"

#: The timm corpus columns. Read off `_VARIANTS` so a new timm column is
#: covered without an edit; `dinov2_vitb14_196` and the open_clip rows are
#: not timm and already preprocess with a no-op.
TIMM_COLUMNS = tuple(sorted(_VARIANTS))


def register_legacy(variant: str) -> str:
    """Register the pre-23a crop under the variant's own name, in-process only.

    Since 23a the published name preprocesses full-frame, so a gate that must
    reproduce a pre-23a cell has to ask for the checkpoint's crop explicitly.
    The record it writes carries the variant's name and the legacy key, exactly
    as the published cell does.
    """
    alias = f"{variant}_legacycrop"
    if alias not in visbench.list_backbones():
        register_backbone(alias, variant=variant, crop_pct=None, name=variant)(TimmBackbone)
    return alias


def register_fullframe(variant: str) -> str:
    """Register ``<variant>_fullframe`` in this process only, and return it."""
    name = f"{variant}{SUFFIX}"
    if name not in visbench.list_backbones():
        register_backbone(name, variant=variant, crop_pct=1.0, name=name)(TimmBackbone)
    return name


def corpus_command(probe: str, backbone: str, results: Path) -> list[str]:
    """The exact `visbench run` argv the corpus used for this cell.

    Read from `build_corpus.sh` itself rather than restated, because a second
    copy of the flags is how a `--limit` drifts and the control stops being a
    control of the published board.
    """
    env = dict(os.environ, DRY_RUN="1", BACKBONES=backbone, RESULTS=str(results))
    env["PATH"] = f"{ROOT / '.venv' / 'bin'}:{env.get('PATH', '')}"
    out = subprocess.run(
        [str(ROOT / "scripts" / "build_corpus.sh"), probe],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lines = [line for line in out.splitlines() if line.startswith("visbench run ")]
    if len(lines) != 1:
        raise SystemExit(f"expected one command for {probe}/{backbone}, got:\n{out}")
    argv = shlex.split(lines[0])[1:]
    if argv[argv.index("--results") + 1] != str(results):
        raise SystemExit(f"build_corpus.sh did not honour RESULTS for {probe}/{backbone}")
    return argv


def check(variants: tuple[str, ...]) -> int:
    """Construct each full-frame backbone and prove what it is before any run.

    Three things, each of which would make the control measure something else:
    preprocessing must be an exact no-op on an image already at the working
    size; the cache key must differ from the legacy one, or the pre-23a
    (cropped) features would be served from cache; and the legacy crop must
    still crop, or the gate could no longer reproduce a pre-23a cell.
    """
    cols = np.tile(np.arange(224, dtype=np.uint8)[None, :, None], (224, 1, 3))
    image = Image.fromarray(cols)
    failed = 0
    for variant in variants:
        full = visbench.get_backbone(register_fullframe(variant), device="cpu")
        default = visbench.get_backbone(register_legacy(variant), device="cpu")
        assert full.image_size == default.image_size == 224, (variant, full.image_size)
        same = full.preprocess(image)
        reference = default.preprocess(image)
        mean = np.array(full._transform.transforms[-1].mean)[:, None, None]
        std = np.array(full._transform.transforms[-1].std)[:, None, None]
        pixels = np.rint((same.numpy()[0] * std + mean) * 255)
        no_op = bool(np.array_equal(pixels.transpose(1, 2, 0), cols.astype(float)))
        keys_differ = full.cache_key() != default.cache_key()
        still_crops = not np.allclose(same.numpy(), reference.numpy())
        ok = no_op and keys_differ and still_crops and full.name.endswith(SUFFIX)
        failed += not ok
        print(
            f"{variant:20s} no-op={no_op} keys_differ={keys_differ} "
            f"legacy_still_crops={still_crops}  {full.cache_key()}"
        )
    print("check passed" if not failed else f"check FAILED on {failed}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("probes", nargs="*")
    parser.add_argument("--backbones", nargs="+", default=list(TIMM_COLUMNS))
    parser.add_argument("--results", type=Path)
    parser.add_argument(
        "--gate", action="store_true", help="run the pre-23a crop under the published name"
    )
    parser.add_argument("--check", action="store_true", help="verify the mechanism; run nothing")
    args = parser.parse_args(argv)

    unknown = sorted(set(args.backbones) - set(TIMM_COLUMNS))
    if unknown:
        parser.error(f"not timm corpus columns: {unknown}")
    if args.check:
        return check(tuple(args.backbones))
    if not args.probes or args.results is None:
        parser.error("give probes and --results, or --check")
    if args.results.resolve() == CORPUS.resolve():
        parser.error("--results may not be the corpus; this is a control")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    for probe in args.probes:
        for variant in args.backbones:
            cli_argv = corpus_command(probe, variant, args.results)
            alias = register_legacy(variant) if args.gate else register_fullframe(variant)
            cli_argv[cli_argv.index("--backbone") + 1] = alias
            print("=== " + " ".join(cli_argv), flush=True)
            code = cli_main(cli_argv)
            failed += code != 0
            print(f"=== exit {code}: {probe} / {variant}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

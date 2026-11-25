#!/usr/bin/env python
"""Does a dense board read a backbone's representation, or its feature scale? (24b)

    python scripts/run_scale_control.py --check
    python scripts/run_scale_control.py generic_segmentation orientation \\
        --results results/controls/scale_control.jsonl

**The question.** `clip_convnext_base` fits worst of all fourteen columns on
several dense boards, and its stage-4 features are about 8.7x smaller in scale
than its supervised sibling `convnext_base`'s (std 0.35 against 3.03 on VOC).
The image-level classification probes standardise features before fitting;
the dense heads fit raw features with a fixed learning rate, which is not
scale-invariant. So a low dense placement could be the optimiser rather than
the representation, and the record cannot say which.

**The control.** Refit the published configuration with the dense features
multiplied by a constant: `clip_convnext_base` *up* by the measured ratio, and
`convnext_base` *down* by it, the mirror. Multiplying features by a constant
removes no information a linear head could use -- the optimum is the same map
rescaled -- so any score that moves, moved because of how the fit reached it.

Each run is the cell's exact corpus command, read from `build_corpus.sh` as
`run_crop_control.py` does, with only the backbone swapped for a scaled alias
registered in this process. **The gate** (`--gate`) runs the same alias path at
scale 1: it must reproduce the published cell, or the scaled arms describe some
adjacent configuration rather than the board. The alias carries its own name and cache key
(`.../x<scale>`), so it can neither evict a corpus row nor be served the
unscaled features. **Never write these to the corpus**: a scaled backbone is a
model nobody would choose.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

import visbench
from visbench.backbones.timm_backbone import TimmBackbone
from visbench.cli.main import main as cli_main
from visbench.registry import register_backbone

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_crop_control import CORPUS, corpus_command  # noqa: E402

#: std of `convnext_base`'s stage-4 features over that of `clip_convnext_base`'s,
#: measured on the first 16 VOC2012 JPEGs at 224 (3.030 / 0.350).
RATIO = 8.66

#: (variant, scale): each column moved onto the other's feature scale.
ARMS = (("clip_convnext_base", RATIO), ("convnext_base", 1.0 / RATIO))


class ScaledTimm(TimmBackbone):
    """A timm backbone whose every extracted layer is multiplied by ``scale``."""

    def __init__(self, *args, scale: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scale = scale

    def _forward_features(self, image, layers):  # type: ignore[override]
        return [
            (tokens * self.scale, None if cls is None else cls * self.scale, grid)
            for tokens, cls, grid in super()._forward_features(image, layers)
        ]

    def cache_key(self) -> str:
        return f"{super().cache_key()}/x{self.scale:.4g}"


def register_scaled(variant: str, scale: float) -> str:
    """Register ``<variant>_x<scale>`` in this process only, and return it."""
    alias = f"{variant}_x{scale:.4g}"
    if alias not in visbench.list_backbones():
        defaults = {"image_size": 224} if variant == "clip_convnext_base" else {}
        register_backbone(alias, variant=variant, scale=scale, name=alias, **defaults)(ScaledTimm)
    return alias


def check() -> int:
    """Prove each alias is a pure rescaling of its variant before any run."""
    image = torch.rand(1, 3, 224, 224)
    failed = 0
    for variant, scale in ARMS:
        plain = visbench.get_backbone(variant, device="cpu")
        scaled = visbench.get_backbone(register_scaled(variant, scale), device="cpu")
        a = plain.extract_features(image)["dense"]
        b = scaled.extract_features(image)["dense"]
        rescaled = torch.allclose(a * scale, b, rtol=1e-5, atol=1e-6)
        keys_differ = plain.cache_key() != scaled.cache_key()
        names_differ = plain.name != scaled.name
        ok = rescaled and keys_differ and names_differ
        failed += not ok
        print(
            f"{variant:20s} x{scale:<8.4g} rescaled={rescaled} keys_differ={keys_differ} "
            f"names_differ={names_differ} -> {'OK' if ok else 'FAIL'}"
        )
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("probes", nargs="*")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--check", action="store_true", help="verify the mechanism; run nothing")
    parser.add_argument(
        "--arm",
        choices=[variant for variant, _ in ARMS],
        action="append",
        help="run only this column's arm (repeatable; default both)",
    )
    parser.add_argument(
        "--gate", action="store_true", help="run each arm at scale 1: must reproduce the corpus"
    )
    args = parser.parse_args(argv)
    if args.check:
        return check()
    if not args.probes or args.results is None:
        parser.error("give probes and --results, or --check")
    if args.results.resolve() == CORPUS.resolve():
        parser.error("--results may not be the corpus; this is a control")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    arms = [(v, 1.0 if args.gate else k) for v, k in ARMS if not args.arm or v in args.arm]
    for probe in args.probes:
        for variant, scale in arms:
            cli_argv = corpus_command(probe, variant, args.results)
            cli_argv[cli_argv.index("--backbone") + 1] = register_scaled(variant, scale)
            print("=== " + " ".join(cli_argv), flush=True)
            code = cli_main(cli_argv)
            failed += code != 0
            print(f"=== exit {code}: {probe} / {variant} x{scale:.4g}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

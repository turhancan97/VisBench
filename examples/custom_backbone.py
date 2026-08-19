"""Probe a model VisBench has never heard of.

``CustomBackbone`` is the escape hatch: any ``nn.Module`` you already have — a
fine-tuned checkpoint, an architecture from a paper's repo, something you
trained yourself — probed by the same thirteen probes as DINOv2 and CLIP, with
no addition to this package.

Run it with no arguments at all::

    python examples/custom_backbone.py

That synthesises the ``visbench demo`` shape dataset, so the script needs no
download beyond torchvision's ResNet-18 (~45 MB, a core dependency). Point it
at your own labelled folder instead with ``--data``, in the layout
``<data>/train/<class>/<image>`` and ``<data>/val/<class>/<image>``.

Three things it demonstrates, in the order they bite:

1. **Wrapping.** A module plus a preprocessing callable is the whole contract.
   ``preprocess`` is required rather than guessed, because normalisation
   constants that are wrong produce a quietly mediocre number, not an error.

2. **The cache keys on the weights.** A custom backbone has no upstream commit
   or pretrained tag to name, so ``hash_weights`` digests the parameters. That
   is what stops a fine-tuned checkpoint from being served its parent's cached
   features — the failure that would otherwise be invisible, since the shapes
   match and the numbers stay plausible. ``--finetune`` demonstrates it by
   perturbing the weights and showing the key move.

3. **Registering it, if you want a name.** ``CustomBackbone`` is constructed,
   never looked up, because a registry name cannot carry an ``nn.Module`` — so
   it is unreachable from the ``visbench`` CLI. Subclassing ``BaseBackbone``
   with ``@register_backbone`` gives it a name, and ``--register`` shows that.

**One measured caveat about trained probes, which is specific to this path.**
:func:`visbench.run` seeds *before* it constructs a backbone from a name, so a
backbone you build yourself is constructed outside the seeded window and the
probe head is initialised from a different RNG state. Measured on this script's
own demo data, wrapping timm's ResNet-18 whose features are **bit-identical**
(max abs difference 0.0) to ``run("resnet18")``'s:

===========================  ====================================  =======
path                         classification top-1, seeds 0-4       spread
===========================  ====================================  =======
``CustomBackbone``           0.9125 0.9125 0.9125 0.9187 0.9125     0.0062
``run("resnet18")``          0.9062 0.9125 0.9062 0.9062 0.9125     0.0063
===========================  ====================================  =======

Two things follow, and the first is the one people expect to go the other way.
**The custom path is perfectly reproducible** — the same number every time, and
unchanged by RNG consumed before ``run()`` is called, because construction
happens before the seed is set rather than after it. And the gap between the
two paths is 0.0062, *the same size as each path's own seed-to-seed spread*, so
it is RNG jitter and not a cost of wrapping. **Zero-shot probes are unaffected
entirely**: retrieval scores 0.603730 through both, bit for bit, because no
head is fitted.

So: a number from a wrapped model is comparable with another number from the
same wrapped model, and not with a registered backbone's trained number to the
last decimal. Register the backbone if you need that.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

import visbench
from visbench.backbones.custom import CustomBackbone, hash_weights
from visbench.cache import FeatureCache
from visbench.data import ImageFolderDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="root containing train/ and val/; default synthesises the demo shapes",
    )
    parser.add_argument("--probe", default="classification", help="see visbench.list_probes()")
    parser.add_argument("--per-class", type=int, default=40, help="demo images per class")
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="also probe a perturbed copy, to show the cache key follows the weights",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="also register a named subclass, reachable the way a built-in is",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64, help="extraction batch size")
    parser.add_argument("--device", default=None, help="cuda | cpu; default is best available")
    parser.add_argument("--cache", type=Path, default=Path(".visbench_cache"))
    return parser.parse_args()


def build_trunk() -> tuple[nn.Module, ResNet18_Weights]:
    """torchvision's ResNet-18 with its classifier removed.

    VisBench wants the last conv feature map, not logits — so ``avgpool`` and
    ``fc`` come off. A module that returns a pooled ``(B, C)`` vector is
    refused by name rather than reshaped, since guessing which axis was the
    grid is how a probe ends up measuring the wrong thing.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    return nn.Sequential(*list(model.children())[:-2]), weights


def wrap(trunk: nn.Module, weights: ResNet18_Weights, name: str, device: str | None):
    """The whole contract: a module and a way to turn a PIL image into a tensor."""
    return CustomBackbone(
        trunk,
        preprocess=weights.transforms(),
        name=name,
        device=device,
    )


def finetuned_copy(trunk: nn.Module, scale: float = 0.02) -> nn.Module:
    """A stand-in for a checkpoint you fine-tuned, without the fine-tuning.

    Perturbing the parameters is enough to make the point, which is about the
    *cache key* and not about the training: two checkpoints of one architecture
    are shape-identical and interchangeable to everything except the weights
    hash.
    """
    import copy

    clone = copy.deepcopy(trunk)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in clone.parameters():
            noise = torch.randn(parameter.shape, generator=generator)
            parameter.add_(noise * scale * parameter.std())
    return clone


def register_a_named_subclass() -> str:
    """The other path: a name, and therefore CLI reachability.

    ``CustomBackbone`` is constructed rather than looked up because a registry
    name is a string and cannot carry an ``nn.Module``. Subclassing is how a
    model gets a name of its own — the same path every built-in backbone takes,
    and the only one that reaches ``visbench run --backbone``.
    """
    name = "resnet18_example"
    if name in visbench.list_backbones():
        return name

    @visbench.register_backbone(name)
    class ExampleResNet18(CustomBackbone):
        def __init__(self, device: str | None = None, **kwargs) -> None:
            trunk, weights = build_trunk()
            super().__init__(
                trunk,
                preprocess=weights.transforms(),
                name=name,
                device=device,
                **kwargs,
            )

    return name


def probe(backbone, probe_name: str, splits, cache: FeatureCache, args) -> float:
    """One ordinary :func:`visbench.run` — nothing here is special-cased."""
    train, val = splits
    kwargs = {}
    if probe_name == "classification":
        kwargs = {"epochs": args.epochs, "lr": args.lr}
    result = visbench.run(
        backbone,
        probe_name,
        val,
        train_dataset=train if not visbench.get_probe(probe_name).zero_shot else None,
        cache=cache,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        **kwargs,
    )
    headline = next(iter(result.metrics))
    # `result.backbone` rather than the argument: `backbone` is a *name* on the
    # registered path, and run() is where a name becomes an object. Taking it
    # back off the result is the documented way to hold the exact object that
    # produced the number -- see RunResult.backbone.
    print(
        f"    {headline}={result.metrics[headline]:.4f}   cache key: {result.backbone.cache_key()}"
    )
    return result.metrics[headline]


def main() -> None:
    args = parse_args()

    root = args.data
    if root is None:
        from visbench.demo import synthesise

        root = Path(tempfile.mkdtemp(prefix="visbench-custom-"))
        print(f"no --data given; synthesising the demo shapes in {root}")
        synthesise(root, per_class=args.per_class, seed=args.seed)

    splits = (ImageFolderDataset(root / "train"), ImageFolderDataset(root / "val"))
    cache = FeatureCache(args.cache)
    trunk, weights = build_trunk()

    print("\n1. a plain nn.Module, wrapped")
    print(f"    weights hash: {hash_weights(trunk)}")
    probe(
        wrap(trunk, weights, "resnet18_torchvision", args.device), args.probe, splits, cache, args
    )

    if args.finetune:
        print("\n2. the same architecture, different weights")
        clone = finetuned_copy(trunk)
        print(f"    weights hash: {hash_weights(clone)}   <- moved, so the cache cannot collide")
        probe(
            wrap(clone, weights, "resnet18_torchvision", args.device),
            args.probe,
            splits,
            cache,
            args,
        )
        print("    Note the name is deliberately the SAME on both runs. The key still")
        print("    differs, because it is the weights that identify a custom backbone.")

    if args.register:
        name = register_a_named_subclass()
        print(f"\n3. registered as {name!r}, so it resolves by name like a built-in")
        print(f"    in visbench.list_backbones(): {name in visbench.list_backbones()}")
        probe(name, args.probe, splits, cache, args)
        print("    Constructed INSIDE run()'s seeded window, unlike the wrapped calls")
        print("    above -- see this file's docstring for what that is worth (0.006 top-1).")


if __name__ == "__main__":
    main()

"""Per-probe dataset construction for the CLI — v0.2, step 5j.

Everything the command line knows that :func:`visbench.run` does not lives
here. ``run()`` takes datasets already built; a shell user has a directory and a
probe name, and something has to know that ``depth`` expects two parallel
folders read by :func:`~visbench.data.dense.load_depth_map` while ``similarity``
expects a NIGHTS CSV and ``correspondence`` expects no annotation at all.

That knowledge is a *table*, not a hierarchy: one :class:`ProbeSpec` per
registered probe, holding the flags it accepts, how to turn them into splits,
and how to configure the probe. Adding a probe to the CLI means adding a row.

**The layouts here are conventions, not requirements.** The Python API accepts
any :class:`~visbench.data.base.BaseDataset`; the CLI has to guess, so it
documents its guess in ``--help`` and fails loudly when the folders are not
there rather than silently evaluating on whatever it found.
"""

import argparse
import functools
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from visbench.data import (
    DenseFolderDataset,
    DetectionFolderDataset,
    ImageFolderDataset,
    TaskonomyDataset,
)
from visbench.data.dense import load_label_map, load_mask, load_normal_map
from visbench.data.derived import DerivedTargetDataset, ShiTomasiResponse
from visbench.data.detection import VOC_CLASSES
from visbench.data.pair_dataset import HomographyPairDataset
from visbench.data.triplet import TwoAFCDataset

__all__ = [
    "ProbeSpec",
    "Splits",
    "CLASSIFICATION_SCHEDULE_DEFAULTS",
    "SCHEDULE_DEFAULTS",
    "SPECS",
    "showable_probes",
    "spec_for",
    "supported_probes",
]


class Splits(NamedTuple):
    """What ``run()`` needs: the split that is scored, and the one that trains.

    ``train`` is ``None`` for a zero-shot probe, which is what ``run()`` demands
    — passing one there raises rather than being ignored.
    """

    evaluate: Any
    train: Any = None


@dataclass(frozen=True)
class ProbeSpec:
    """One row of the table: how the CLI turns flags into a run for this probe."""

    #: One line under the subcommand in ``visbench run --help``.
    summary: str
    #: Expected directory layout, shown in the subcommand's own ``--help``.
    layout: str
    #: Adds this probe's flags to its subparser.
    add_arguments: Callable[[argparse.ArgumentParser], None]
    #: Builds the splits from parsed arguments.
    build: Callable[[argparse.Namespace], Splits]
    #: Keyword arguments forwarded to :func:`visbench.get_probe`.
    probe_kwargs: Callable[[argparse.Namespace], dict]
    #: Adds the flags ``visbench show`` needs: the ones :attr:`build` reads, plus
    #: the ones that decide the head's *shape* (needed to load a saved probe),
    #: and none of the ones that decide its training. ``None`` means this probe
    #: has no spatial target to draw, and ``show`` refuses it by name.
    #:
    #: A separate callable rather than reusing :attr:`add_arguments`: that would
    #: put ``--epochs`` and ``--lr`` in ``visbench show depth --help``, promising
    #: something they cannot do. Splitting the flag groups instead is the move
    #: step 6d-1 already made to this file, for the same reason.
    show_arguments: Callable[[argparse.ArgumentParser], None] | None = None


# -- shared flag groups ------------------------------------------------------


def _split_flags(parser: argparse.ArgumentParser, evaluate: str, train: str | None) -> None:
    """``--split``, and ``--train-split`` for a probe that trains."""
    parser.add_argument(
        "--split", default=evaluate, help=f"subdirectory of --data to score (default: {evaluate})"
    )
    if train is not None:
        parser.add_argument(
            "--train-split", default=train, help=f"subdirectory to train on (default: {train})"
        )


def _viewing(
    view_flags: Callable[[argparse.ArgumentParser], None],
    defaults: dict[str, Any] | None = None,
) -> Callable:
    """Turn a probe's non-schedule flags into its ``show_arguments``.

    Every probe's ``add_arguments`` is exactly ``<view flags> + _schedule_flags``,
    so this composes ``show``'s surface from the *same* callable rather than a
    parallel copy. That matters more than the tidiness: a second copy could
    build a different dataset than ``run`` would from the same command line, and
    a viewer that draws data the probe did not see is worse than no viewer.
    """

    def add(parser: argparse.ArgumentParser) -> None:
        view_flags(parser)
        parser.set_defaults(**(SCHEDULE_DEFAULTS if defaults is None else defaults))

    return add


def _dense_view_flags(parser: argparse.ArgumentParser, target_dir: str) -> None:
    """Everything a dense probe needs *except* how its head is trained.

    Split in two when the edge probe arrived (step 6d-1): Taskonomy is indexed
    from a split list rather than from two named folders, so it wants the head
    half without ``--image-dir``, ``--target-dir`` or ``--stems``, which would
    appear in its ``--help`` promising something they cannot do.

    Split again for ``visbench show``, which draws data and loads a head but
    trains nothing — so it wants precisely this and none of
    :func:`_schedule_flags`. Every probe below is therefore
    ``<view flags> + _schedule_flags``, and :func:`_viewing` reuses the first
    half rather than restating it.
    """
    _dense_folder_flags(parser, target_dir)
    _image_size_flag(parser)
    _head_shape_flags(parser)


def _dense_folder_flags(parser: argparse.ArgumentParser, target_dir: str) -> None:
    """Where the images and targets live. Everything ``_dense_splits`` reads."""
    parser.add_argument("--image-dir", default="images", help="image folder inside a split")
    parser.add_argument(
        "--target-dir",
        default=target_dir,
        help=f"target folder inside a split (default: {target_dir})",
    )
    # A real benchmark names its splits in a file, not by foldering its images:
    # VOC ships 17k images beside 2.9k segmentation labels and lists split
    # membership in ImageSets/Segmentation/*.txt. Passing one of these switches
    # the layout — --data is then the dataset root itself, with no <split>
    # subdirectory, because the file *is* the split.
    parser.add_argument(
        "--stems",
        type=Path,
        default=None,
        help="file listing the scored split's filename stems, one per line "
        "(e.g. VOC's ImageSets/Segmentation/val.txt). Makes --data the root directly.",
    )
    parser.add_argument(
        "--train-stems",
        type=Path,
        default=None,
        help="the same for the training split; required alongside --stems",
    )


def _image_size_flag(parser: argparse.ArgumentParser) -> None:
    """The working resolution.

    A *data* flag, despite having lived in the head group: it decides the resize
    and centre crop the dataset applies, so anything that builds a dataset needs
    it and anything that only trains a head does not.
    """
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="working resolution; must be a multiple of the backbone's patch size",
    )


def _head_shape_flags(parser: argparse.ArgumentParser) -> None:
    """What the head *is*, as opposed to how it was trained.

    These are what ``visbench show --predict-from`` also needs: a saved artifact
    is only loadable onto a head of the same shape, and ``load_probe`` checks
    the layers among its four identity fields.
    """
    parser.add_argument("--head", default="linear", help="linear | dpt; see `visbench list heads`")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="backbone depths to read, e.g. --layers 2 5 8 11; dpt needs several",
    )
    parser.add_argument("--hidden-dim", type=int, default=512, help="DPT width")


def _schedule_flags(parser: argparse.ArgumentParser) -> None:
    """How the head is trained. Nothing here changes what a dataset yields.

    Every default is mirrored in :data:`SCHEDULE_DEFAULTS`, so a command that
    builds a probe without training it can satisfy ``probe_kwargs`` without
    offering flags it would ignore. A test pins the two against each other.
    """
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=8,
        help="batch size for training the head, distinct from --batch-size (extraction)",
    )
    # v0.3. Off by default, and the record says which a number came from: a
    # fine-tuned score and a frozen one answer different questions, so the two
    # must never be compared without knowing which is which.
    parser.add_argument(
        "--finetune-blocks",
        type=int,
        default=0,
        help="unfreeze this many trailing backbone blocks and train them with the head "
        "(default: 0, a frozen probe). Bypasses the feature cache, so it runs 1.3-2.2x "
        "slower than a cached frozen run on VOC, more so the larger the backbone; "
        "DINOv2 only for now",
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=None,
        help="learning rate for the unfrozen blocks (default: --lr / 100). Only valid "
        "with --finetune-blocks",
    )
    parser.add_argument(
        "--no-prefix-cache",
        action="store_true",
        help="recompute the frozen blocks every epoch instead of caching their output "
        "(step 6b). The prefix cache is on by default and cannot change a metric -- the "
        "blocks below the cut never train -- so this is for measuring what it saves, and "
        "for trading disk against time on a small split",
    )


#: :func:`_schedule_flags`' defaults, for a command that constructs a probe
#: without training it. ``visbench show --predict-from`` loads a head someone
#: else's run already fitted, so every one of these is irrelevant to it — but
#: ``probe_kwargs`` reads them all, and forking that per command is how the two
#: copies drift. ``parser.set_defaults(**SCHEDULE_DEFAULTS)`` puts them on
#: ``args`` without putting them in ``--help``.
SCHEDULE_DEFAULTS: dict[str, Any] = {
    "epochs": 10,
    "lr": 5e-4,
    "train_batch_size": 8,
    "finetune_blocks": 0,
    "backbone_lr": None,
    "no_prefix_cache": False,
}


#: ``_classification_flags``' own schedule defaults, which are not the dense
#: ones -- a linear probe on pooled features trains for 200 epochs at 1e-2, not
#: 10 at 5e-4. A single shared table would silently give ``show`` a probe built
#: with the wrong ones.
CLASSIFICATION_SCHEDULE_DEFAULTS: dict[str, Any] = {
    "epochs": 200,
    "lr": 1e-2,
    "train_batch_size": 256,
}


def _dense_probe_kwargs(args: argparse.Namespace) -> dict:
    return {
        "head": args.head,
        "layers": args.layers,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "lr": args.lr,
        # The head's training batch, not the extraction batch --batch-size sets.
        "batch_size": args.train_batch_size,
        "device": args.device,
        "finetune_blocks": args.finetune_blocks,
        "backbone_lr": args.backbone_lr,
    }


def _dense_splits(
    args: argparse.Namespace,
    target_loader: Callable[[Path], Any] | None = None,
    **extra: Any,
) -> Splits:
    """Both halves of a dense run.

    Two layouts, and the presence of ``--stems`` is what chooses between them:
    without it, splits are directories (``<data>/<split>/{images,targets}``);
    with it, ``<data>`` is the root and the file lists the split's members.
    There is no third mode where both apply — a stems file *is* a split, so
    also descending into a directory named after one would be asking the same
    question twice and getting to disagree with itself.
    """
    if (args.stems is None) != (args.train_stems is None):
        raise ValueError(
            "Pass --stems and --train-stems together, or neither. One split named by a "
            "file and the other by a directory would silently mix two layouts."
        )

    def load(split: str, listing: Path | None) -> DenseFolderDataset:
        stems = None
        if listing is not None:
            if not listing.is_file():
                raise FileNotFoundError(f"No split list at {listing}")
            stems = listing.read_text().split()
            # Truncated here rather than through subset(), since the stem list is
            # what the dataset is built from.
            if args.limit is not None:
                stems = stems[: args.limit]
        dataset = DenseFolderDataset(
            args.data if listing is not None else args.data / split,
            image_dir=args.image_dir,
            target_dir=args.target_dir,
            split=split,
            image_size=args.image_size,
            target_loader=target_loader,
            stems=stems,
            **extra,
        )
        if listing is not None or args.limit is None:
            return dataset
        # subset() reindexes the three parallel lists together. Slicing one by
        # hand would pair a target with the wrong image and still train.
        return dataset.subset(args.limit)

    return Splits(
        evaluate=load(args.split, args.stems),
        train=load(args.train_split, args.train_stems),
    )


def _folder_split(args: argparse.Namespace, split: str) -> ImageFolderDataset:
    """One labelled image folder, limited **per class** rather than by prefix."""
    root = args.data / split if split else args.data
    dataset = ImageFolderDataset(root, split=split or "all")
    return dataset if args.limit is None else dataset.balanced_subset(args.limit)


# -- high level --------------------------------------------------------------


def _classification_view_flags(parser: argparse.ArgumentParser) -> None:
    _split_flags(parser, evaluate="val", train="train")
    parser.add_argument("--pooling", default=None, help="cls | mean; default is the backbone's")
    # A head-shape flag, not a schedule one: the standardiser is fitted state
    # that travels in the artifact, so a head saved with it does not load onto a
    # probe built without.
    parser.add_argument("--standardize", action="store_true", help="normalise features first")


def _classification_flags(parser: argparse.ArgumentParser) -> None:
    _classification_view_flags(parser)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--train-batch-size", type=int, default=256)


def _classification_splits(args: argparse.Namespace) -> Splits:
    return Splits(
        evaluate=_folder_split(args, args.split),
        train=_folder_split(args, args.train_split),
    )


def _classification_kwargs(args: argparse.Namespace) -> dict:
    kwargs: dict[str, Any] = {
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.train_batch_size,
        "standardize": args.standardize,
        "device": args.device,
    }
    if args.pooling is not None:
        kwargs["pooling"] = args.pooling
    # num_classes is left to the probe, which reads it from the labels it is
    # fitted on. Passing a count derived from the *eval* split would be wrong
    # for any split missing a class.
    return kwargs


def _retrieval_flags(parser: argparse.ArgumentParser) -> None:
    _split_flags(parser, evaluate="val", train=None)
    parser.add_argument("--pooling", default=None, help="cls | mean; default is the backbone's")
    parser.add_argument("--metric", default="cosine", choices=("cosine", "l2"))
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 5, 10])


def _retrieval_kwargs(args: argparse.Namespace) -> dict:
    kwargs: dict[str, Any] = {"metric": args.metric, "topk": tuple(args.topk)}
    if args.pooling is not None:
        kwargs["pooling"] = args.pooling
    return kwargs


def _semantic_view_flags(parser: argparse.ArgumentParser) -> None:
    _split_flags(parser, evaluate="val", train="train")
    _dense_view_flags(parser, target_dir="labels")
    parser.add_argument(
        "--num-classes",
        type=int,
        required=True,
        help="including background; required because a wrong value does not raise",
    )
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=255,
        help="raw value marking unlabelled pixels; -1 to disable (default: 255)",
    )


def _semantic_flags(parser: argparse.ArgumentParser) -> None:
    _semantic_view_flags(parser)
    _schedule_flags(parser)


def _semantic_splits(args: argparse.Namespace) -> Splits:
    ignore = None if args.ignore_index < 0 else args.ignore_index
    # No max_target: it marks out-of-range sensor readings invalid, and against
    # class indices it would erase whole categories.
    return _dense_splits(args, functools.partial(load_label_map, ignore_index=ignore))


def _detection_flags(parser: argparse.ArgumentParser) -> None:
    _detection_view_flags(parser)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--train-batch-size", type=int, default=8)


def _detection_view_flags(parser: argparse.ArgumentParser) -> None:
    """Detection's own flag set, minus the schedule.

    Not built from :func:`_head_shape_flags`: this probe's head defaults differ
    (``detection`` rather than ``linear``, ``hidden_dim=0`` rather than 512) and
    it reads no ``--layers``. The decoding settings belong here too — they shape
    what ``predict`` emits, so a panel drawn under different ones is not the
    panel the score came from.
    """
    _split_flags(parser, evaluate="val", train="train")
    parser.add_argument("--image-dir", default="JPEGImages", help="image folder (VOC's name)")
    parser.add_argument(
        "--annotation-dir", default="Annotations", help="box XML folder (VOC's name)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="working resolution; must be a multiple of the backbone's patch size. Passed to "
        "the dataset AND the probe from this one flag, because box targets are absolute "
        "pixels and two different values would silently misplace every cell",
    )
    parser.add_argument(
        "--stems",
        type=Path,
        default=None,
        help="file listing the scored split's stems (e.g. VOC's ImageSets/Main/val.txt). "
        "Makes --data the dataset root directly",
    )
    parser.add_argument(
        "--train-stems",
        type=Path,
        default=None,
        help="the same for the training split; required alongside --stems",
    )
    parser.add_argument("--head", default="detection", help="see `visbench list heads`")
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=0,
        help="width of an optional 3x3 stem; 0 (default) keeps the head linear, which is "
        "what makes a difference between backbones a difference between representations",
    )
    parser.add_argument(
        "--min-box-size",
        type=float,
        default=1.0,
        help="drop boxes smaller than this after the centre crop",
    )
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)


def _detection_splits(args: argparse.Namespace) -> Splits:
    """Both halves of a detection run.

    **The two splits differ in one setting, and it is not a slip.** The scored
    split keeps ``difficult`` objects because VOC's protocol *ignores* a
    detection that matches one rather than counting it wrong — worth 4.3 mAP on
    VOC val (step 6c-2). The training split drops them, because an object the
    annotators judged unreasonable to require is not supervision anyone wants.
    ``DetectionTask`` drops them from assignment as well, so this is belt and
    braces rather than the only guard.
    """
    if (args.stems is None) != (args.train_stems is None):
        raise ValueError(
            "Pass --stems and --train-stems together, or neither. One split named by a "
            "file and the other by a directory would silently mix two layouts."
        )

    def load(split: str, listing: Path | None, include_difficult: bool) -> DetectionFolderDataset:
        stems = None
        if listing is not None:
            if not listing.is_file():
                raise FileNotFoundError(f"No split list at {listing}")
            stems = listing.read_text().split()
            if args.limit is not None:
                stems = stems[: args.limit]
        dataset = DetectionFolderDataset(
            args.data if listing is not None else args.data / split,
            image_dir=args.image_dir,
            annotation_dir=args.annotation_dir,
            split=split,
            stems=stems,
            image_size=args.image_size,
            include_difficult=include_difficult,
            min_box_size=args.min_box_size,
        )
        if listing is not None or args.limit is None:
            return dataset
        # subset() reindexes the three parallel lists together; slicing one by
        # hand would pair an image with another image's boxes.
        return dataset.subset(args.limit)

    return Splits(
        evaluate=load(args.split, args.stems, include_difficult=True),
        train=load(args.train_split, args.train_stems, include_difficult=False),
    )


def _detection_kwargs(args: argparse.Namespace) -> dict:
    return {
        # From the dataset's class list rather than a flag: the loader's classes
        # and the head's width are the same fact, and a flag would let them
        # disagree without raising.
        "num_classes": len(VOC_CLASSES),
        "image_size": args.image_size,
        "head": args.head,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.train_batch_size,
        "score_threshold": args.score_threshold,
        "nms_iou": args.nms_iou,
        "max_detections": args.max_detections,
        "device": args.device,
    }


# -- mid level ---------------------------------------------------------------


def _depth_view_flags(parser: argparse.ArgumentParser) -> None:
    _split_flags(parser, evaluate="val", train="train")
    _dense_view_flags(parser, target_dir="depths")
    parser.add_argument(
        "--target-scale",
        type=float,
        default=1.0,
        help="divisor for integer files; 1000 for mm PNGs",
    )
    parser.add_argument("--max-depth", type=float, default=10.0, help="beyond this is invalid")
    # A head-shape flag, not a schedule one: DepthTask emits one channel per bin,
    # so a saved head only loads onto a probe built with the same count.
    parser.add_argument("--n-bins", type=int, default=256)


def _depth_flags(parser: argparse.ArgumentParser) -> None:
    _depth_view_flags(parser)
    _schedule_flags(parser)


def _depth_splits(args: argparse.Namespace) -> Splits:
    return _dense_splits(args, target_scale=args.target_scale, max_target=args.max_depth)


def _normal_view_flags(parser: argparse.ArgumentParser) -> None:
    _split_flags(parser, evaluate="val", train="train")
    _dense_view_flags(parser, target_dir="normals")
    parser.add_argument(
        "--normal-source",
        default=None,
        help="how the normals were derived (e.g. geonet); recorded, since sources disagree",
    )
    # Shapes the head: the uncertainty-aware loss adds a fourth output channel
    # for kappa, so a head fitted with it does not load onto one built without.
    parser.add_argument(
        "--plain-loss",
        action="store_true",
        help="drop probe3d's uncertainty-aware angular loss; makes numbers incomparable",
    )


def _normal_flags(parser: argparse.ArgumentParser) -> None:
    _normal_view_flags(parser)
    _schedule_flags(parser)


def _generic_seg_view_flags(parser: argparse.ArgumentParser) -> None:
    _split_flags(parser, evaluate="val", train="train")
    _dense_view_flags(parser, target_dir="masks")
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=-1,
        help="raw value marking unlabelled pixels; negative disables (default: disabled)",
    )


def _generic_seg_flags(parser: argparse.ArgumentParser) -> None:
    _generic_seg_view_flags(parser)
    _schedule_flags(parser)


def _generic_seg_splits(args: argparse.Namespace) -> Splits:
    ignore = None if args.ignore_index < 0 else args.ignore_index
    # No max_target for a mask: it would erase the foreground class.
    return _dense_splits(args, functools.partial(load_mask, ignore_index=ignore))


def _taskonomy_flags(parser: argparse.ArgumentParser, *, domain: str) -> None:
    """Taskonomy is indexed from its own split lists, so no folder flags here.

    ``--domain`` is restricted to the single domain this probe's ``protocol``
    describes, and that is deliberate rather than a placeholder. In v0.4.0 the
    edge probe accepted any domain the dataset would load, so
    ``visbench run edge --domain keypoints2d`` produced a keypoint number
    recorded as ``visbench_edge_regression`` — the exact mislabelling the
    ``protocol`` field exists to prevent. The flag is kept, restricted, so that
    a probe growing a second honest domain has somewhere to put it.
    """
    _taskonomy_view_flags(parser, domain=domain)
    _schedule_flags(parser)


def _taskonomy_view_flags(parser: argparse.ArgumentParser, *, domain: str) -> None:
    """The split, resolution, partition and domain — plus the head's shape."""
    _split_flags(parser, evaluate="val", train="train")
    _image_size_flag(parser)
    _head_shape_flags(parser)
    parser.add_argument(
        "--partition",
        default="tiny",
        help="Taskonomy download size naming the split files, splits/<partition>_<split>.csv "
        "(tiny, medium, fullplus). Part of what the number means, so it is recorded",
    )
    parser.add_argument(
        "--domain",
        default=domain,
        choices=(domain,),
        help=f"target domain directory (default and only: {domain}). Restricted to what this "
        "probe's recorded protocol describes",
    )


def _taskonomy_splits(args: argparse.Namespace) -> Splits:
    """Both halves, from the official building-disjoint split lists.

    ``--limit`` reaches the constructor as ``max_images`` rather than becoming a
    ``subset()`` afterwards. The tiny train list is 272,296 rows, so building
    every path only to discard all but a few hundred is pure waste — and unlike
    a labelled image folder, a prefix here is not one class, because the rows
    are already interleaved across buildings.
    """
    common = {
        "root": args.data,
        "domain": args.domain,
        "partition": args.partition,
        "image_size": args.image_size,
        "max_images": args.limit,
    }
    return Splits(
        evaluate=TaskonomyDataset(split=args.split, **common),
        train=TaskonomyDataset(split=args.train_split, **common),
    )


def _corner_flags(parser: argparse.ArgumentParser) -> None:
    """One image folder, and the operator that turns it into a target.

    No ``--target-dir``: there is no target folder, which is the point of this
    probe. What replaces it is the operator's parameterisation, and every one of
    these flags changes the number, so all of them land in ``dataset_params``
    and split the comparability groups on their own.
    """
    _corner_view_flags(parser)
    _schedule_flags(parser)


def _corner_view_flags(parser: argparse.ArgumentParser) -> None:
    """Everything ``_corner_splits`` reads, the operator's settings included."""
    _split_flags(parser, evaluate="val", train="train")
    parser.add_argument("--image-dir", default="images", help="image folder inside a split")
    _image_size_flag(parser)
    _head_shape_flags(parser)
    parser.add_argument(
        "--corner-sigma",
        type=float,
        default=2.0,
        help="Gaussian window of the structure tensor, in pixels — the scale the "
        "operator sees, not a smoothing nicety (default: 2.0)",
    )
    parser.add_argument(
        "--corner-transform",
        default="log1p",
        choices=("log1p", "none"),
        help="compression applied to the raw response. The raw target has 27%% of its "
        "mass in its strongest 1%% of pixels, which is the regime where this "
        "protocol stops ranking backbones (default: log1p)",
    )
    parser.add_argument(
        "--corner-scale",
        type=float,
        default=1e4,
        help="multiplier applied before the compression; sets both the tail and the "
        "target's magnitude, and an L1 loss needs the latter of order 1 (default: 1e4)",
    )


def _corner_splits(args: argparse.Namespace) -> Splits:
    generator = ShiTomasiResponse(
        sigma=args.corner_sigma,
        transform=args.corner_transform,
        scale=args.corner_scale,
    )
    common = {
        "image_dir": args.image_dir,
        "image_size": args.image_size,
        "generator": generator,
        # A prefix is safe here in a way it is not on a labelled folder: nothing
        # about the ordering correlates with the target, because the target is
        # computed per image rather than read from a per-class layout.
        "max_images": args.limit,
    }

    def load(split: str) -> DerivedTargetDataset:
        return DerivedTargetDataset(root=args.data / split, split=split, **common)

    return Splits(evaluate=load(args.split), train=load(args.train_split))


def _correspondence_flags(parser: argparse.ArgumentParser) -> None:
    _split_flags(parser, evaluate="", train=None)
    parser.add_argument("--max-warp", type=float, default=0.2, help="corner shift, 0-0.5")
    parser.add_argument("--num-corr", type=int, default=1000)
    parser.add_argument("--ratio", type=float, default=0.9, help="Lowe ratio threshold")
    # Pixels, because a patch width is a property of the backbone -- see
    # CorrespondenceTask. Ranking two backbones in patch units asks each a
    # different question and inverted this board before v0.6.1.
    parser.add_argument("--units", default="pixel", choices=("patch", "pixel"))
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help="default: 1 2 5 10 pixels, or 0.5 1 2 4 patch widths",
    )
    parser.add_argument(
        "--image-size", type=int, default=224, help="must match the backbone's input size"
    )


def _correspondence_splits(args: argparse.Namespace) -> Splits:
    dataset = HomographyPairDataset(
        args.data / args.split if args.split else args.data,
        split=args.split or "all",
        max_warp=args.max_warp,
        seed=args.seed,
        image_size=args.image_size,
    )
    # A prefix, so each pair keeps the warp it had in the full set — the warps
    # are drawn from the seed and the position, not from the image.
    return Splits(evaluate=dataset if args.limit is None else dataset.subset(args.limit))


def _correspondence_kwargs(args: argparse.Namespace) -> dict:
    return {
        "num_corr": args.num_corr,
        "ratio_threshold": args.ratio,
        "thresholds": tuple(args.thresholds) if args.thresholds else None,
        "threshold_units": args.units,
    }


def _similarity_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split",
        default="test",
        help="train | val | test | test_imagenet | test_no_imagenet (default: test)",
    )
    parser.add_argument(
        "--min-votes", type=int, default=6, help="drop triplets humans disagreed about"
    )
    parser.add_argument("--annotations", default="data.csv", help="CSV filename under --data")


def _similarity_splits(args: argparse.Namespace) -> Splits:
    # max_triplets, not subset(): the triplets index into the image list, so
    # slicing the images alone would silently repoint every one of them.
    return Splits(
        evaluate=TwoAFCDataset(
            args.data,
            split=args.split,
            min_votes=args.min_votes,
            max_triplets=args.limit,
            annotations=args.annotations,
        )
    )


# -- the table ---------------------------------------------------------------

_FOLDER_LAYOUT = "<data>/<split>/<class>/*.jpg"
_DENSE_LAYOUT = "<data>/<split>/{images,%s}/  (paired by filename stem)"

SPECS: dict[str, ProbeSpec] = {
    "classification": ProbeSpec(
        summary="linear probe on pooled features, scored top-1/top-5",
        layout=_FOLDER_LAYOUT,
        add_arguments=_classification_flags,
        build=_classification_splits,
        probe_kwargs=_classification_kwargs,
        show_arguments=_viewing(
            _classification_view_flags, defaults=CLASSIFICATION_SCHEDULE_DEFAULTS
        ),
    ),
    "retrieval": ProbeSpec(
        summary="zero-shot leave-one-out ranking over pooled features",
        layout=_FOLDER_LAYOUT,
        add_arguments=_retrieval_flags,
        build=lambda args: Splits(evaluate=_folder_split(args, args.split)),
        probe_kwargs=_retrieval_kwargs,
        # Zero-shot: every flag shapes the ranking, so none is dropped.
        show_arguments=_retrieval_flags,
    ),
    "semantic_segmentation": ProbeSpec(
        summary="multi-class per-pixel probe; reports mIoU both reductions",
        layout=_DENSE_LAYOUT % "labels",
        add_arguments=_semantic_flags,
        build=_semantic_splits,
        probe_kwargs=lambda args: {
            **_dense_probe_kwargs(args),
            "num_classes": args.num_classes,
        },
        show_arguments=_viewing(_semantic_view_flags),
    ),
    "detection": ProbeSpec(
        summary="anchor-free single-scale box probe, scored mAP@50 the VOC way",
        layout="<data>/<split>/{JPEGImages,Annotations}/  (or --stems on a VOC root)",
        add_arguments=_detection_flags,
        build=_detection_splits,
        probe_kwargs=_detection_kwargs,
        show_arguments=_viewing(_detection_view_flags),
    ),
    "depth": ProbeSpec(
        summary="probe3d's 256-bin depth protocol",
        layout=_DENSE_LAYOUT % "depths",
        add_arguments=_depth_flags,
        build=_depth_splits,
        probe_kwargs=lambda args: {
            **_dense_probe_kwargs(args),
            "max_depth": args.max_depth,
            "n_bins": args.n_bins,
        },
        show_arguments=_viewing(_depth_view_flags),
    ),
    "surface_normal": ProbeSpec(
        summary="probe3d's angular protocol, uncertainty-aware loss",
        layout=_DENSE_LAYOUT % "normals",
        add_arguments=_normal_flags,
        build=lambda args: _dense_splits(args, load_normal_map),
        probe_kwargs=lambda args: {
            **_dense_probe_kwargs(args),
            "uncertainty_aware": not args.plain_loss,
            "normal_source": args.normal_source,
        },
        show_arguments=_viewing(_normal_view_flags),
    ),
    "generic_segmentation": ProbeSpec(
        summary="binary foreground probe; foreground IoU is the number to quote",
        layout=_DENSE_LAYOUT % "masks",
        add_arguments=_generic_seg_flags,
        build=_generic_seg_splits,
        probe_kwargs=_dense_probe_kwargs,
        show_arguments=_viewing(_generic_seg_view_flags),
    ),
    "edge": ProbeSpec(
        summary="dense edge-magnitude regression; quote edge_correlation",
        layout="<data>/rgb/<building>/*.png, <data>/edge_texture/..., <data>/splits/*.csv",
        add_arguments=functools.partial(_taskonomy_flags, domain="edge_texture"),
        build=_taskonomy_splits,
        probe_kwargs=_dense_probe_kwargs,
        show_arguments=_viewing(functools.partial(_taskonomy_view_flags, domain="edge_texture")),
    ),
    "corner": ProbeSpec(
        # The only probe here that needs no target folder: the target is
        # computed from the images, so any image folder will do.
        summary="dense corner-response regression on a derived target; quote corner_correlation",
        layout="<data>/<split>/images/*.jpg   (no targets: they are computed)",
        add_arguments=_corner_flags,
        build=_corner_splits,
        probe_kwargs=_dense_probe_kwargs,
        show_arguments=_viewing(_corner_view_flags),
    ),
    "keypoints2d": ProbeSpec(
        summary="dense 2D keypoint-response regression; quote keypoint_correlation",
        layout="<data>/rgb/<building>/*.png, <data>/keypoints2d/..., <data>/splits/*.csv",
        add_arguments=functools.partial(_taskonomy_flags, domain="keypoints2d"),
        build=_taskonomy_splits,
        probe_kwargs=_dense_probe_kwargs,
        show_arguments=_viewing(functools.partial(_taskonomy_view_flags, domain="keypoints2d")),
    ),
    "occlusion_edge": ProbeSpec(
        # The one Taskonomy probe here whose target has holes; mask_valid/ is
        # required on disk and the dataset refuses to run without it.
        summary="dense occlusion-edge regression; quote occlusion_edge_correlation",
        layout=(
            "<data>/rgb/<building>/*.png, <data>/edge_occlusion/..., "
            "<data>/mask_valid/..., <data>/splits/*.csv"
        ),
        add_arguments=functools.partial(_taskonomy_flags, domain="edge_occlusion"),
        build=_taskonomy_splits,
        probe_kwargs=_dense_probe_kwargs,
        show_arguments=_viewing(functools.partial(_taskonomy_view_flags, domain="edge_occlusion")),
    ),
    "correspondence": ProbeSpec(
        summary="zero-shot dense matching under a known homography",
        layout="<data>/<split>/*.jpg  (no annotation: warps are generated)",
        add_arguments=_correspondence_flags,
        build=_correspondence_splits,
        probe_kwargs=_correspondence_kwargs,
        # Every flag here shapes the *matches*, so all of them survive into
        # `show`: there is no schedule half to drop, because nothing trains.
        show_arguments=_correspondence_flags,
    ),
    "similarity": ProbeSpec(
        summary="zero-shot 2AFC against human perceptual judgements",
        layout="<data>/data.csv + the ref/ and distort/ trees it names",
        add_arguments=_similarity_flags,
        build=_similarity_splits,
        probe_kwargs=lambda args: {"min_votes": args.min_votes},
        # Zero-shot, as correspondence is: nothing here is a training setting.
        show_arguments=_similarity_flags,
    ),
}


def supported_probes() -> list[str]:
    """Probe names the CLI can run, sorted."""
    return sorted(SPECS)


def showable_probes() -> list[str]:
    """Probe names ``visbench show`` can draw, sorted.

    A subset of :func:`supported_probes`: a probe with no spatial target has
    nothing to put in a panel. Kept in step with
    :func:`visbench.viz.show_probes`, which holds the *drawing* half of the same
    question, by a test asserting the two sets are equal — the two tables live in
    different packages precisely so the Python API needs no CLI, and that is
    exactly the arrangement that lets one grow a row the other lacks.
    """
    return sorted(name for name, spec in SPECS.items() if spec.show_arguments is not None)


def spec_for(probe: str) -> ProbeSpec:
    """The spec for ``probe``, with an error naming what the CLI does support.

    A probe registered in :func:`visbench.list_probes` but absent here is a real
    gap, not a user error — the message says so rather than implying the name
    was wrong.
    """
    if probe not in SPECS:
        import visbench

        known = ", ".join(supported_probes())
        if probe in visbench.list_probes():
            raise KeyError(
                f"{probe!r} is a registered probe but the CLI does not know what data it "
                f"expects, so it cannot build a dataset for it. Use the Python API. "
                f"CLI-supported: {known}"
            )
        raise KeyError(f"Unknown probe {probe!r}. Available: {known}")
    return SPECS[probe]

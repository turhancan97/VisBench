"""Adapters for datasets VisBench did not ship a loader for.

Two tiers already exist and both work without this module: a **folder layout**
needs no code at all (NYUv2 joined the corpus with none), and **anything else**
is a :class:`~visbench.data.base.BaseDataset` subclass with two methods. What
was missing is the shortest path when the data already lives inside a
``torch.utils.data.Dataset`` or a Hugging Face ``datasets.Dataset`` — which is
where most published benchmarks are handed out.

Both bridges are thin: they present the wrapped dataset through VisBench's
interface and add nothing. The wrapped dataset still yields the pixels; the
backbone's ``preprocess`` still does the tensor conversion.

**The one method a bridge must not skip is ``cache_identity``.** Four
:class:`BaseDataset` methods are optional and three of them fail *loudly* when
omitted — a task with no ``labels()`` raises, a record with no ``fingerprint()``
carries ``None``. ``cache_identity`` fails *silently*: return ``None`` and every
run re-decodes every image, forever, while appearing to work. That is the
``view_identity`` bug — a mechanism tested and correct for a year while a caller
passed bare PIL images and paid a full decode on every "cached" run. So both
classes here derive a real per-item token, and both lean on the same property
to do it cheaply: **the wrapped dataset is immutable in index order.**

- A ``datasets.Dataset`` carries a ``_fingerprint`` that changes on any
  transform, so ``(fingerprint, row index)`` uniquely names a row's content.
- A ``torchvision`` dataset has no such hash, but its ``__repr__`` states its
  root, split and download flags, and its samples are fixed once constructed;
  when it exposes file paths (the ``ImageFolder`` family) those are used
  directly, and otherwise a digest of the repr plus length stands in. That is
  weaker — two different downloads of the same class with the same repr would
  collide — and it is documented on the class rather than hidden.

Pass ``identity=`` / ``labels=`` to override either when the defaults cannot see
what they need.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image

from visbench.data.base import BaseDataset

__all__ = ["TorchvisionDataset", "HuggingFaceDataset", "to_pil"]


def to_pil(value: Any) -> Image.Image:
    """Coerce one dataset item's image to an RGB :class:`PIL.Image.Image`.

    Accepts a PIL image, a ``(H, W[, C])`` uint8 array, or a float tensor/array
    in ``[0, 1]`` or ``[0, 255]``. A ``CHW`` tensor is moved to ``HWC`` first,
    the layout ``torchvision`` transforms produce. The result must be
    deterministic — the feature cache hashes decoded pixels — so there is no
    resizing or normalisation here.
    """
    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)

    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[2] not in (1, 3):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]

    if array.dtype != np.uint8:
        top = float(array.max()) if array.size else 1.0
        scale = 255.0 if top <= 1.0 + 1e-6 else 1.0
        array = np.clip(np.round(array.astype(np.float64) * scale), 0, 255).astype(np.uint8)

    return Image.fromarray(array).convert("RGB")


class _IndexedBridge(BaseDataset):
    """Shared machinery: an index vector that :meth:`subset` reindexes.

    Both bridges keep ``_indices`` — positions into the wrapped dataset — and a
    parallel ``_labels`` list, so ``subset`` and ``balanced_subset`` work
    without the wrapped dataset needing to support slicing.
    """

    _parallel_attrs = ("_indices", "_labels")

    _indices: list[int]
    _labels: list[Any]

    def __len__(self) -> int:
        return len(self._indices)

    def labels(self) -> list:
        """Labels in index order, without decoding an image."""
        return list(self._labels)


class TorchvisionDataset(_IndexedBridge):
    """Wrap any map-style ``torch.utils.data.Dataset`` yielding ``(image, target)``.

    ::

        from torchvision.datasets import CIFAR10
        raw = CIFAR10("./data", train=False, download=True)
        dataset = TorchvisionDataset(raw, split="test")
        visbench.run("dinov2_vits14", "classification", dataset, train_dataset=...)

    Parameters
    ----------
    dataset:
        The wrapped dataset. ``dataset[i]`` must return ``(image, target)`` with
        no random transform in the path — see :func:`to_pil` for accepted image
        types.
    name:
        Recorded identifier; defaults to the wrapped class name, lowercased.
    split:
        ``"train"`` / ``"test"`` / ``"val"``, recorded and used by the CLI.
    labels:
        Explicit label sequence. When omitted, ``dataset.targets`` /
        ``dataset.labels`` is used if present; otherwise every item is read once
        to collect its target (a one-time decode, warned about is not — this is
        the documented cost of a dataset that hides its labels).
    identity:
        ``callable(source_index) -> str`` overriding :meth:`cache_identity`.

    Notes
    -----
    **``cache_identity`` for a dataset that keeps its images in memory** (CIFAR,
    MNIST) is ``"<repr digest>|<index>"``. That is correct as long as the
    wrapped dataset is content-stable in index order, which ``torchvision``
    guarantees, but it *cannot* tell two different downloads apart if their
    reprs match. For the ``ImageFolder`` family (``.samples`` / ``.imgs``) the
    file path, size and mtime are used instead, exactly as
    :class:`~visbench.data.image_folder.ImageFolderDataset` does.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        name: str | None = None,
        split: str = "",
        labels: Sequence[Any] | None = None,
        identity: Callable[[int], str] | None = None,
    ) -> None:
        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            raise TypeError(
                "TorchvisionDataset needs a map-style dataset with __len__ and __getitem__, "
                f"got {type(dataset).__name__}"
            )
        self.dataset = dataset
        self.name = name or type(dataset).__name__.lower()
        self.split = split
        self._identity_override = identity
        self._indices = list(range(len(dataset)))

        self._samples = getattr(dataset, "samples", None) or getattr(dataset, "imgs", None)

        if labels is not None:
            resolved = list(labels)
        elif getattr(dataset, "targets", None) is not None:
            resolved = list(dataset.targets)
        elif getattr(dataset, "labels", None) is not None:
            resolved = list(dataset.labels)
        elif self._samples is not None:
            resolved = [label for _, label in self._samples]
        else:
            resolved = [dataset[i][1] for i in range(len(dataset))]
        if len(resolved) != len(dataset):
            raise ValueError(f"labels= has {len(resolved)} entries for a dataset of {len(dataset)}")
        self._labels = [_coerce_label(label) for label in resolved]

        self._repr_digest = hashlib.sha256(
            f"{type(dataset).__module__}.{type(dataset).__name__}|{dataset!r}|{len(dataset)}".encode()
        ).hexdigest()[:16]

    def __getitem__(self, index: int) -> tuple[Image.Image, Any]:
        source = self._indices[index]
        image, _ = self.dataset[source]
        return to_pil(image), self._labels[index]

    def cache_identity(self, index: int) -> str:
        source = self._indices[index]
        if self._identity_override is not None:
            return self._identity_override(source)
        if self._samples is not None:
            from pathlib import Path

            path = Path(self._samples[source][0])
            stat = path.stat()
            return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        return f"{self._repr_digest}|{source}"

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(f"{self.name}|{self.split}|{len(self._indices)}|{self._repr_digest}".encode())
        # Labels, not pixels: cheap, and enough to tell a relabelled or
        # reordered split from the original.
        digest.update("|".join(str(label) for label in self._labels).encode())
        return digest.hexdigest()[:16]

    def describe(self) -> dict:
        info = super().describe()
        info["dataset_source"] = f"torchvision:{type(self.dataset).__name__}"
        info["num_classes"] = len({label for label in self._labels if label is not None})
        return info


class HuggingFaceDataset(_IndexedBridge):
    """Wrap a Hugging Face ``datasets.Dataset``.

    ::

        from datasets import load_dataset
        raw = load_dataset("cifar100", split="test")
        dataset = HuggingFaceDataset(raw, image_column="img", label_column="fine_label")

    Needs the optional dependency: ``pip install visbench[datasets]``.

    Parameters
    ----------
    dataset:
        A ``datasets.Dataset`` (not a ``DatasetDict`` — pass one split).
    image_column, label_column:
        Column names. When omitted, the first ``Image`` feature and the first
        ``ClassLabel`` feature are used, and a dataset with none of either
        raises rather than guessing.
    name, split:
        Default to the dataset's own ``info.dataset_name`` and ``split``.

    Notes
    -----
    ``cache_identity`` is ``"<dataset._fingerprint>|<row index>"``. A
    ``datasets.Dataset`` is immutable — every ``map`` / ``filter`` / ``cast``
    returns a new dataset with a new fingerprint — so this names a row's exact
    content and is stable across processes and machines. This is the clean case
    the ``torchvision`` bridge only approximates.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        image_column: str | None = None,
        label_column: str | None = None,
        name: str | None = None,
        split: str | None = None,
    ) -> None:
        try:
            import datasets as hf_datasets
        except ImportError as error:  # pragma: no cover - exercised via the CLI
            raise ImportError(
                "HuggingFaceDataset needs the 'datasets' package: pip install visbench[datasets]"
            ) from error

        if not isinstance(dataset, hf_datasets.Dataset):
            raise TypeError(
                "HuggingFaceDataset wraps a single datasets.Dataset split, got "
                f"{type(dataset).__name__}"
                + (
                    " — index it by split first, e.g. load_dataset(name)['train']"
                    if isinstance(dataset, hf_datasets.DatasetDict)
                    else ""
                )
            )

        self.dataset = dataset
        features = dataset.features

        if image_column is None:
            image_column = _first_feature(features, hf_datasets.Image)
            if image_column is None:
                raise ValueError(f"no Image column found in {list(features)}; pass image_column=")
        elif image_column not in features:
            raise ValueError(f"image_column={image_column!r} is not in {list(features)}")
        self.image_column = image_column

        if label_column is None:
            label_column = _first_feature(features, hf_datasets.ClassLabel)
        elif label_column not in features:
            raise ValueError(f"label_column={label_column!r} is not in {list(features)}")
        self.label_column = label_column

        self.name = name or (getattr(dataset.info, "dataset_name", None) or "hf_dataset")
        self.split = split if split is not None else (str(dataset.split) if dataset.split else "")
        self._indices = list(range(len(dataset)))

        if label_column is not None:
            column = dataset.with_format(None)[label_column]
            self._labels = [_coerce_label(label) for label in column]
        else:
            self._labels = [None] * len(dataset)

    def __getitem__(self, index: int) -> tuple[Image.Image, Any]:
        source = self._indices[index]
        row = self.dataset[source]
        return to_pil(row[self.image_column]), self._labels[index]

    def cache_identity(self, index: int) -> str:
        return f"{self.dataset._fingerprint}|{self._indices[index]}"

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            f"{self.name}|{self.split}|{len(self._indices)}|{self.dataset._fingerprint}".encode()
        )
        digest.update("|".join(str(label) for label in self._labels).encode())
        return digest.hexdigest()[:16]

    def describe(self) -> dict:
        info = super().describe()
        info["dataset_source"] = f"hf:{self.name}"
        info["num_classes"] = len({label for label in self._labels if label is not None})
        return info


def _coerce_label(label: Any) -> Any:
    """A scalar tensor/array label becomes a plain ``int``; everything else passes."""
    if isinstance(label, torch.Tensor):
        return label.item() if label.ndim == 0 else label.tolist()
    if isinstance(label, np.generic):
        return label.item()
    return label


def _first_feature(features: Any, feature_type: type) -> str | None:
    for column, feature in features.items():
        if isinstance(feature, feature_type):
            return column
    return None

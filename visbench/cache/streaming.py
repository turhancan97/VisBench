"""Reading cached features a batch at a time — v0.2.

:meth:`FeatureCache.extract_dataset` stacks every feature map and hands back one
tensor. That is right for pooled features and impossible for dense ones: 24k
NYUv2 images at DINOv2-B is roughly 19 GB, and their depth targets another
~4.8 GB on top.

The cache already writes one file per image per layer, so the fix is not to
store anything differently — only to read it back on demand.
:class:`CachedFeatures` is an ordinary ``torch.utils.data.Dataset`` over those
files, so a task can hand it to a ``DataLoader`` and get batching, per-epoch
shuffling and background workers without any of that being written here.

Random access, not a generator, is the point. Training reshuffles every epoch,
and a generator yielding batches in dataset order can only shuffle *within* a
batch — which is not the same thing, and would quietly make a probe worse than
the representation it is meant to measure.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import Dataset

__all__ = ["CachedFeatures"]


class CachedFeatures(Dataset):
    """Per-image features (and their targets) loaded from the cache on demand.

    Built by :meth:`FeatureCache.materialise`, not directly — it needs the key
    list that extraction produced.

    Each item is ``(features, target)`` when targets were supplied, else just
    ``features``. ``features`` is a ``(C, H, W)`` map, or a list of them under a
    multi-layer request, matching what a head expects once the loader has added
    the batch dimension.
    """

    def __init__(
        self,
        cache: Any,
        keys: list[list[str]],
        layer_indices: list[int],
        multi: bool,
        grid_hw: tuple[int, int],
        targets: Callable[[int], Any] | None = None,
    ) -> None:
        self.cache = cache
        self.keys = keys
        self.layer_indices = layer_indices
        self.multi = multi
        self.grid_hw = grid_hw
        self.targets = targets

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Any:
        """Load image ``index`` from disk.

        Entries are stored with a leading batch dimension of 1, which is
        squeezed off here so the loader's collation puts it back — otherwise
        every batch would arrive as ``(B, 1, C, H, W)``.
        """
        maps = []
        for depth, key in enumerate(self.keys[index]):
            entry = self.cache.get(key, require=("dense",))
            if entry is None:
                raise RuntimeError(
                    f"Cache entry for item {index}, layer {self.layer_indices[depth]} is "
                    "gone. It was written during extraction, so the cache directory was "
                    "cleared or a file was deleted while this run was using it. Re-run to "
                    "re-extract."
                )
            maps.append(entry["dense"].squeeze(0))

        features: torch.Tensor | list = maps if self.multi else maps[0]
        if self.targets is None:
            return features
        return features, self.targets(index)

    @property
    def channels(self) -> int | list[int]:
        """Feature width per layer, for building a head that fits.

        Read from the first item rather than recorded at extraction: a head is
        built once, so one extra file read is cheaper than another field that
        could disagree with what is actually on disk.
        """
        first = self[0]
        maps = first[0] if self.targets is not None else first
        if self.multi:
            return [layer.shape[0] for layer in maps]
        return int(maps.shape[0])

    @staticmethod
    def collate(batch: list) -> Any:
        """Stack a batch, handling the multi-layer case the default cannot.

        ``torch.utils.data``'s default collation turns a list of tensors into a
        batch, but a multi-layer item is a *list* of tensors, which it would
        transpose into something no head accepts. Here the layer structure is
        preserved and each layer stacked across the batch.
        """
        first = batch[0]
        has_targets = isinstance(first, tuple)
        features = [item[0] for item in batch] if has_targets else list(batch)

        if isinstance(features[0], list):
            stacked: Any = [
                torch.stack([item[depth] for item in features]) for depth in range(len(features[0]))
            ]
        else:
            stacked = torch.stack(features)

        if not has_targets:
            return stacked
        return stacked, torch.stack([torch.as_tensor(item[1]) for item in batch])

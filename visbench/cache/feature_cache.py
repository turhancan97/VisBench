"""Disk-backed feature cache.

Mandatory in v0.1, not an optional speed-up bolted on later (CLAUDE.md,
"Feature cache"). Every task reads through this; the backbone forward pass runs
**at most once per image per backbone**.
"""

import functools
import hashlib
import itertools
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Callable, Optional, cast

import torch

from visbench.cache.keys import SEPARATOR, hash_image, make_key
from visbench.types import FeatureDict, FeatureMode, Pooling

__all__ = ["FeatureCache", "DEFAULT_CACHE_DIR"]

#: Default location, relative to the working directory. In .gitignore.
DEFAULT_CACHE_DIR = Path(".visbench_cache")

_ENTRY_SUFFIX = ".pt"

#: Which outputs :meth:`FeatureCache.extract_dataset` accumulates in memory,
#: and which it writes to disk.
_KEEP_CHOICES = ("both", "pooled", "dense")

#: Subdirectory holding the identity -> content-hash memo. Underscore-prefixed
#: so it cannot collide with a sanitised backbone key.
_IDENTITY_DIR = "_identity"


def _parts(choice: str) -> tuple[str, ...]:
    """Map a ``keep``/``store`` choice to the feature keys it covers."""
    return ("dense", "pooled") if choice == "both" else (choice,)


def _identified_items(dataset: Iterable) -> Iterator[tuple[Optional[str], Callable[[], Any]]]:
    """Yield ``(identity, load)`` pairs, deferring the image load.

    Two shapes of input reach here. An indexed dataset exposing
    ``cache_identity`` is walked by index, so ``load`` is only called for items
    the cache could not already resolve — that is what avoids decoding a
    cached image. A plain iterable of images has no identity and is already
    materialised item by item, so ``load`` just hands the image back.

    Returning a thunk rather than the image is the whole point: it lets the
    caller decide, per item, whether decoding is necessary at all.
    """
    from visbench.data.pair_dataset import PairDataset

    if isinstance(dataset, PairDataset):
        # A pair dataset yields (image_0, image_1, geometry). The unpacking
        # below would take image_0 and drop the second view and the geometry
        # without a word, handing back features for half the data. Refuse
        # instead: this is the shape of silence the cache is built to avoid.
        raise TypeError(
            "extract_dataset does not take a PairDataset: it yields "
            "(image_0, image_1, geometry), and only the first view would be "
            "extracted. Extract each view yourself — see examples/correspond.py."
        )

    indexed = (
        hasattr(dataset, "cache_identity")
        and hasattr(dataset, "__len__")
        and hasattr(dataset, "__getitem__")
    )

    if indexed:
        indexed_dataset = cast(Any, dataset)
        for index in range(len(indexed_dataset)):
            yield (
                indexed_dataset.cache_identity(index),
                functools.partial(_load_indexed, indexed_dataset, index),
            )
        return

    for item in dataset:
        image = item[0] if isinstance(item, (tuple, list)) else item
        yield None, functools.partial(_identity_fn, image)


def _load_indexed(dataset: Any, index: int) -> Any:
    item = dataset[index]
    return item[0] if isinstance(item, (tuple, list)) else item


def _identity_fn(image: Any) -> Any:
    return image


def _chunks(items: Iterable, size: int) -> Iterator[list]:
    """Yield lists of at most ``size`` items, pulling from ``items`` lazily.

    Distinct from :func:`visbench.utils.device.batched`, which slices a
    ``Sequence`` and therefore needs the whole thing in memory first. Here the
    point is precisely that it never is.
    """
    if size < 1:
        raise ValueError(f"batch_size must be >= 1, got {size}")
    iterator = iter(items)
    while True:
        chunk = list(itertools.islice(iterator, size))
        if not chunk:
            return
        yield chunk


class FeatureCache:
    """Key-value store mapping cache keys to extracted features on disk.

    Stores ``dense``, ``pooled`` and ``grid_hw`` together, since they come from
    one forward pass and splitting them would allow a partial hit that still
    requires re-running the backbone.

    Entries are stored on CPU regardless of the extraction device, so a cache
    written on a GPU box is readable on a laptop.
    """

    def __init__(self, root: Optional[Path] = None, enabled: bool = True) -> None:
        """Open (creating if needed) the cache directory.

        ``enabled=False`` turns every lookup into a miss without changing call
        sites — useful for tests and for measuring true extraction cost.
        """
        self.root = Path(root) if root is not None else DEFAULT_CACHE_DIR
        self.enabled = enabled
        self._hits = 0
        self._misses = 0
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    # -- paths ---------------------------------------------------------------

    def _path(self, key: str) -> Path:
        """Map a key to its file: ``<root>/<backbone>/<shard>/<digest>.pt``.

        The backbone directory is what makes :meth:`clear` per-backbone cheap,
        and the two-character shard keeps any single directory from collecting
        every entry in a large dataset.
        """
        backbone_key = key.split(SEPARATOR, 1)[0]
        # "/" is legal in a backbone key ("dinov2/dinov2_vitb14/224") but would
        # nest directories arbitrarily; flatten it to one level.
        safe_backbone = backbone_key.replace("/", "__").replace(os.sep, "__")
        digest = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self.root / safe_backbone / digest[:2] / f"{digest}{_ENTRY_SUFFIX}"

    # -- single entries ------------------------------------------------------

    def get(
        self, key: str, require: tuple[str, ...] = ("dense", "pooled")
    ) -> Optional[FeatureDict]:
        """Return the cached entry, or ``None`` on a miss.

        ``require`` names the parts the caller needs. An entry written by a
        run that stored only ``pooled`` is a **miss** for a caller that needs
        ``dense`` — returning it would silently hand back a feature dict with
        a missing key, which fails much further from the cause.
        """
        if not self.enabled:
            self._misses += 1
            return None

        path = self._path(key)
        if not path.exists():
            self._misses += 1
            return None

        try:
            entry = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # A truncated or corrupt file counts as a miss, never as a hit. The
            # subsequent put() overwrites it, so the cache self-heals.
            self._misses += 1
            return None

        if any(part not in entry for part in require):
            self._misses += 1
            return None

        self._hits += 1
        result: FeatureDict = {"grid_hw": tuple(entry["grid_hw"])}
        for part in ("dense", "pooled", "cls"):
            if part in entry:
                result[part] = entry[part]  # type: ignore[literal-required]
        return result

    def put(self, key: str, features: FeatureDict, store: str = "both") -> None:
        """Write an entry. Writes atomically so an interrupted run cannot leave
        a half-written file that later reads as a corrupt hit.

        ``store`` selects which parts reach disk. Dense features are roughly
        250x the size of pooled ones — 390 KB against 1.5 KB for DINOv2 ViT-S
        at 224 — so writing them for a task that only reads pooled vectors
        costs gigabytes and buys nothing.
        """
        if not self.enabled:
            return

        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            # Saved as a list: weights_only=True unpickles lists but not tuples
            # of arbitrary origin. Restored to a tuple on read.
            "grid_hw": list(features["grid_hw"]),
        }
        for part in _parts(store):
            payload[part] = features[part].detach().cpu()  # type: ignore[literal-required]
        # feature_mode="dense_plus_cls" returns the global vector separately.
        # Without this it would survive extraction and vanish on the next cache
        # hit — the caller would get a dict missing a key it explicitly asked
        # for, which is exactly the kind of silence `require` exists to prevent.
        if "cls" in features:
            payload["cls"] = features["cls"].detach().cpu()

        # Same directory as the target, so os.replace stays on one filesystem
        # and is therefore atomic.
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                torch.save(payload, handle)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    # -- identity memo -------------------------------------------------------

    def _identity_path(self, identity: str) -> Path:
        digest = hashlib.sha256(identity.encode()).hexdigest()[:32]
        return self.root / _IDENTITY_DIR / digest[:2] / f"{digest}.txt"

    def _remembered_hash(self, identity: Optional[str]) -> Optional[str]:
        """Content hash this identity last produced, or ``None``.

        This is a memo of a computation, never a substitute for it: a hit here
        still yields the same content-derived key the decode would have.
        """
        if identity is None or not self.enabled:
            return None
        path = self._identity_path(identity)
        try:
            return path.read_text().strip() or None
        except OSError:
            return None

    def _remember_hash(self, identity: Optional[str], image_hash: str) -> None:
        if identity is None or not self.enabled:
            return
        path = self._identity_path(identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(image_hash)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def get_or_compute(self, key: str, compute: Callable[[], FeatureDict]) -> FeatureDict:
        """Return the cached entry, computing and storing it on a miss."""
        cached = self.get(key)
        if cached is not None:
            return cached
        features = compute()
        self.put(key, features)
        return features

    # -- datasets ------------------------------------------------------------

    def extract_dataset(
        self,
        backbone: Any,
        dataset: Iterable,
        pooling: str = Pooling.DEFAULT,
        layer: Optional[int] = None,
        batch_size: int = 32,
        keep: str = "both",
        store: Optional[str] = None,
        feature_mode: str = FeatureMode.DENSE_ONLY,
    ) -> FeatureDict:
        """Extract (or load) features for a whole dataset.

        The main entry point tasks use: batches only the cache misses through
        the backbone, then returns features for the full dataset in dataset
        order.

        ``dataset`` may yield PIL images or ``(image, label)`` pairs; labels are
        ignored here, since a task reads them from the dataset directly. It is
        consumed lazily, one batch at a time — a dataset that decodes on
        ``__getitem__`` never holds more than ``batch_size`` images in memory,
        which is what makes a 50k-image run possible at all.

        When the dataset implements :meth:`BaseDataset.cache_identity`, an
        already-cached image is resolved from that token alone and **never
        decoded**. Measured on Imagenette (13,394 images), that is the
        difference between a fully cached run costing ~113 s and costing
        almost nothing.

        Parameters
        ----------
        keep:
            Which outputs to accumulate in memory and return: ``"both"``,
            ``"pooled"`` or ``"dense"``. ``dense`` is the memory risk — 5k
            images at 16x16x768 is roughly 4 GB in fp32 — so a task needing
            only pooled features should say so.
        feature_mode:
            Passed through to :meth:`BaseBackbone.extract_features` and part of
            the cache key, since the modes produce different ``dense`` tensors
            from the same forward pass.
        store:
            Which outputs to write to disk; defaults to ``keep``. Dense
            features are ~250x the size of pooled ones, so storing them for a
            task that never reads them turned a 20 MB pooled cache into 5 GB.
            An entry stored without a part is treated as a miss by a later run
            that needs it, so a leaner cache costs re-extraction, never
            silently wrong features.
        """
        if keep not in _KEEP_CHOICES:
            raise ValueError(f"keep must be one of {_KEEP_CHOICES}, got {keep!r}")
        if store is None:
            store = keep
        if store not in _KEEP_CHOICES:
            raise ValueError(f"store must be one of {_KEEP_CHOICES}, got {store!r}")

        backbone_key = backbone.cache_key()
        layers = None if layer is None else [layer]
        required = _parts(keep)
        if feature_mode == FeatureMode.DENSE_PLUS_CLS:
            required = required + ("cls",)
        key_of = functools.partial(
            make_key,
            backbone_key=backbone_key,
            layer=layer,
            pooling=pooling,
            feature_mode=feature_mode,
        )

        dense_chunks: list[torch.Tensor] = []
        pooled_chunks: list[torch.Tensor] = []
        cls_chunks: list[torch.Tensor] = []
        grids: set = set()
        count = 0

        for batch_items in _chunks(_identified_items(dataset), batch_size):
            entries: list[Optional[FeatureDict]] = []
            keys: list[Optional[str]] = []

            # Pass 1: resolve anything whose content hash we already know for
            # this exact file. A hit here never touches the image.
            for identity, _ in batch_items:
                image_hash = self._remembered_hash(identity)
                if image_hash is None:
                    keys.append(None)
                    entries.append(None)
                    continue
                key = key_of(image_hash)
                keys.append(key)
                entries.append(self.get(key, require=required))

            # Pass 2: decode whatever pass 1 could not resolve, and hash it.
            # This may still hit — a copied file has a new identity but the
            # same pixels, which is exactly what content addressing is for.
            images: dict[int, Any] = {}
            for index, (identity, load) in enumerate(batch_items):
                if entries[index] is not None:
                    continue
                image = load()
                images[index] = image
                if keys[index] is None:
                    image_hash = hash_image(image)
                    self._remember_hash(identity, image_hash)
                    resolved_key = key_of(image_hash)
                    keys[index] = resolved_key
                    entries[index] = self.get(resolved_key, require=required)

            missing = [i for i, entry in enumerate(entries) if entry is None]
            if missing:
                # Pass 2 assigned a key for every entry it could not resolve,
                # so nothing still missing has a None key.
                assert all(keys[i] is not None for i in missing)
                tensor_batch = backbone.preprocess([images[i] for i in missing])
                features = backbone.extract_features(
                    tensor_batch, pooling=pooling, layers=layers, feature_mode=feature_mode
                )
                for position, index in enumerate(missing):
                    single: FeatureDict = {
                        "dense": features["dense"][position : position + 1].cpu(),
                        "pooled": features["pooled"][position : position + 1].cpu(),
                        "grid_hw": features["grid_hw"],
                    }
                    if "cls" in features:
                        single["cls"] = features["cls"][position : position + 1].cpu()
                    self.put(keys[index], single, store=store)  # type: ignore[arg-type]
                    entries[index] = single

            for entry in entries:
                assert entry is not None  # every miss was filled above
                grids.add(entry["grid_hw"])
                count += 1
                if keep in ("both", "dense"):
                    dense_chunks.append(entry["dense"])
                if keep in ("both", "pooled"):
                    pooled_chunks.append(entry["pooled"])
                if "cls" in entry:
                    cls_chunks.append(entry["cls"])

        if count == 0:
            raise ValueError("Cannot extract features from an empty dataset")
        if len(grids) > 1:
            raise ValueError(
                f"Dataset produced more than one dense grid shape ({sorted(grids)}); "
                "features cannot be stacked. Use a fixed input resolution."
            )

        result: FeatureDict = {"grid_hw": grids.pop()}  # type: ignore[typeddict-item]
        if keep in ("both", "dense"):
            result["dense"] = torch.cat(dense_chunks)
        if keep in ("both", "pooled"):
            result["pooled"] = torch.cat(pooled_chunks)
        if cls_chunks:
            result["cls"] = torch.cat(cls_chunks)
        return result

    # -- maintenance ---------------------------------------------------------

    def clear(self, backbone_key: Optional[str] = None) -> int:
        """Delete cached entries, optionally only those for one backbone.

        Returns the number of entries removed.
        """
        if backbone_key is None:
            targets = [self.root]
        else:
            safe = backbone_key.replace("/", "__").replace(os.sep, "__")
            targets = [self.root / safe]

        removed = 0
        for target in targets:
            if not target.exists():
                continue
            removed += sum(1 for _ in target.rglob(f"*{_ENTRY_SUFFIX}"))
            shutil.rmtree(target)

        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
        return removed

    def stats(self) -> dict:
        """Entry count, on-disk size, and hit/miss counts for this session."""
        entries = list(self.root.rglob(f"*{_ENTRY_SUFFIX}")) if self.root.exists() else []
        return {
            "entries": len(entries),
            "bytes": sum(path.stat().st_size for path in entries),
            "hits": self._hits,
            "misses": self._misses,
            "enabled": self.enabled,
            "root": str(self.root),
        }

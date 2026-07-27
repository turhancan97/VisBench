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
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from visbench.cache.keys import SEPARATOR, hash_image, make_key
from visbench.types import FeatureDict, FeatureMode, LayerSpec, Pooling

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


def _identified_items(dataset: Iterable) -> Iterator[tuple[str | None, Callable[[], Any]]]:
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


@dataclass(frozen=True)
class _Plan:
    """One resolved extraction request.

    Everything that decides correctness — which layer indices a key names, what
    counts as a hit, what gets written — computed once and shared by both entry
    points, so the in-memory and streaming paths cannot drift apart on it.
    """

    backbone_key: str
    indices: list[int]
    request: LayerSpec
    multi: bool
    pooling: str
    feature_mode: str
    required_per_layer: list[tuple[str, ...]]
    store_per_layer: list[str | None]

    def keys_for(self, image_hash: str) -> list[str]:
        """One cache key per requested layer, for this image."""
        return [
            make_key(
                image_hash,
                backbone_key=self.backbone_key,
                layer=index,
                pooling=self.pooling,
                feature_mode=self.feature_mode,
            )
            for index in self.indices
        ]


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

    def __init__(self, root: Path | None = None, enabled: bool = True) -> None:
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

    def get(self, key: str, require: tuple[str, ...] = ("dense", "pooled")) -> FeatureDict | None:
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

    def _remembered_hash(self, identity: str | None) -> str | None:
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

    def _remember_hash(self, identity: str | None, image_hash: str) -> None:
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

    # -- planning and walking, shared by both entry points -------------------

    def _plan(
        self,
        backbone: Any,
        pooling: str,
        layer: int | None,
        layers: LayerSpec,
        keep: str,
        store: str,
        feature_mode: str,
        streaming: bool,
    ) -> "_Plan":
        """Resolve one request into concrete keys, requirements and storage.

        Shared by :meth:`extract_dataset` and :meth:`materialise` so the two
        cannot drift on the questions that decide correctness: which layer
        indices a key names, what counts as a hit, and what gets written.
        """
        if layer is not None and layers is not None:
            raise ValueError(
                f"Pass layer= or layers=, not both (got layer={layer}, layers={layers}). "
                "Nothing here can check that they agree."
            )
        multi = layers is not None
        if multi and not streaming and keep == "pooled":
            raise ValueError(
                f"layers={layers} with keep='pooled' is a contradiction: the pooled "
                "vector comes from one layer, so the rest would be extracted and "
                "discarded. Use layer= for the depth you want pooled from."
            )

        request: LayerSpec = layers if multi else (None if layer is None else [layer])
        # Resolved here, not just inside the backbone, because the resolved
        # index is what goes in the key: layers=[-1] and layers=[11] name one
        # entry on a 12-block model, and must not occupy two.
        indices = backbone.resolve_layers(request)

        required_per_layer: list[tuple[str, ...]]
        store_per_layer: list[str | None]
        if streaming:
            # A streaming reader loads `dense` from every layer, so every entry
            # must hold it — including the deepest, which the in-memory path is
            # allowed to keep pooled-only.
            required_per_layer = [("dense",)] * len(indices)
            store_per_layer = ["dense"] * len(indices)
        else:
            # Only the deepest layer carries `pooled` and `cls` — they describe
            # `dense`, and `dense` is the last requested layer. A shallower entry
            # holding a `pooled` from some other run's layer choice would be a
            # different vector under a name that does not say so.
            last_required = _parts(keep)
            required_per_layer = [("dense",)] * (len(indices) - 1) + [last_required]
            # A shallower entry has no pooled vector to write, so it is stored as
            # dense-only — or not at all, when this run is storing pooled alone.
            shallow_store = "dense" if store in ("both", "dense") else None
            store_per_layer = [shallow_store] * (len(indices) - 1) + [store]

        if feature_mode == FeatureMode.DENSE_PLUS_CLS:
            required_per_layer[-1] = required_per_layer[-1] + ("cls",)
            if streaming:
                store_per_layer[-1] = "both"

        return _Plan(
            backbone_key=backbone.cache_key(),
            indices=indices,
            request=request,
            multi=multi,
            pooling=pooling,
            feature_mode=feature_mode,
            required_per_layer=required_per_layer,
            store_per_layer=store_per_layer,
        )

    def _walk(
        self,
        backbone: Any,
        dataset: Iterable,
        plan: "_Plan",
        batch_size: int,
    ) -> Iterator[list[tuple[list[str], list[FeatureDict]]]]:
        """Yield ``(keys, entries)`` per image, one batch at a time.

        The whole extraction loop lives here: identity resolution, content
        hashing, batching misses through the backbone, and writing entries. It
        **yields** rather than accumulating, so a caller that only wants the
        keys never holds a feature tensor at all.
        """
        for batch_items in _chunks(_identified_items(dataset), batch_size):
            entries: list[list[FeatureDict] | None] = []
            keys: list[list[str] | None] = []

            # Pass 1: resolve anything whose content hash we already know for
            # this exact file. A hit here never touches the image.
            for identity, _ in batch_items:
                image_hash = self._remembered_hash(identity)
                if image_hash is None:
                    keys.append(None)
                    entries.append(None)
                    continue
                batch_keys = plan.keys_for(image_hash)
                keys.append(batch_keys)
                entries.append(self._load_entry(batch_keys, plan))

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
                    resolved_keys = plan.keys_for(image_hash)
                    keys[index] = resolved_keys
                    entries[index] = self._load_entry(resolved_keys, plan)

            missing = [i for i, entry in enumerate(entries) if entry is None]
            if missing:
                # Pass 2 assigned keys for every entry it could not resolve,
                # so nothing still missing has a None key.
                assert all(keys[i] is not None for i in missing)
                tensor_batch = backbone.preprocess([images[i] for i in missing])
                features = backbone.extract_features(
                    tensor_batch,
                    pooling=plan.pooling,
                    layers=plan.request,
                    feature_mode=plan.feature_mode,
                )
                per_layer = features.get("dense_layers") or [features["dense"]]
                for position, index in enumerate(missing):
                    image_keys = keys[index]
                    assert image_keys is not None
                    single_layers: list[FeatureDict] = []
                    for depth, dense in enumerate(per_layer):
                        single: FeatureDict = {
                            "dense": dense[position : position + 1].cpu(),
                            "grid_hw": tuple(dense.shape[-2:]),  # type: ignore[typeddict-item]
                        }
                        if depth == len(per_layer) - 1:
                            single["pooled"] = features["pooled"][position : position + 1].cpu()
                            if "cls" in features:
                                single["cls"] = features["cls"][position : position + 1].cpu()
                        layer_store = plan.store_per_layer[depth]
                        if layer_store is not None:
                            self.put(image_keys[depth], single, store=layer_store)
                        single_layers.append(single)
                    entries[index] = single_layers

            resolved: list[tuple[list[str], list[FeatureDict]]] = []
            for image_keys, entry in zip(keys, entries, strict=True):
                assert image_keys is not None and entry is not None
                resolved.append((image_keys, entry))
            yield resolved

    def _load_entry(self, keys: list[str], plan: "_Plan") -> list[FeatureDict] | None:
        """All layers, or ``None`` — a partial hit is a miss.

        One forward pass produces every layer, so re-running it to fill a gap
        costs nothing beyond what the missing layer already required.
        """
        found = []
        for key, require in zip(keys, plan.required_per_layer, strict=True):
            entry = self.get(key, require=require)
            if entry is None:
                return None
            found.append(entry)
        return found

    # -- entry points --------------------------------------------------------

    def materialise(
        self,
        backbone: Any,
        dataset: Iterable,
        pooling: str = Pooling.DEFAULT,
        layer: int | None = None,
        layers: LayerSpec = None,
        batch_size: int = 32,
        feature_mode: str = FeatureMode.DENSE_ONLY,
        targets: Callable[[int], Any] | None = None,
    ) -> Any:
        """Ensure every entry is on disk, and return a reader over them.

        The streaming counterpart to :meth:`extract_dataset`. It runs the same
        extraction, but keeps **nothing** in memory: it returns a
        :class:`~visbench.cache.streaming.CachedFeatures`, which loads each
        image's features from disk on demand and can be handed straight to a
        ``torch.utils.data.DataLoader``.

        This is what lets a dense task train on a dataset larger than RAM.
        ``extract_dataset`` stacks every feature map — 24k NYUv2 images at
        DINOv2-B is roughly 19 GB — which is fine for pooled features and a
        hard wall for dense ones.

        Entries are always stored with ``dense``, since that is what the reader
        loads; ``keep`` and ``store`` have no meaning here because nothing is
        kept.

        Parameters
        ----------
        targets:
            ``index -> target`` for the supervision paired with each image,
            typically :meth:`~visbench.data.dense.DenseFolderDataset.target`.
            Passed here rather than stacked separately so features and targets
            are read by the same index and cannot drift apart — and because a
            dense split's targets are themselves too large to stack (24k NYUv2
            maps at 224 square is another ~4.8 GB).
        """
        from visbench.cache.streaming import CachedFeatures

        plan = self._plan(
            backbone,
            pooling=pooling,
            layer=layer,
            layers=layers,
            keep="dense",
            store="dense",
            feature_mode=feature_mode,
            streaming=True,
        )

        keys: list[list[str]] = []
        grids: list[set] = [set() for _ in plan.indices]
        for batch in self._walk(backbone, dataset, plan, batch_size):
            for image_keys, entry in batch:
                keys.append(image_keys)
                for depth, layer_entry in enumerate(entry):
                    grids[depth].add(layer_entry["grid_hw"])
            # `entry` goes out of scope here: the tensors it holds are the only
            # thing standing between this and the memory ceiling it exists to
            # remove, so nothing above may keep a reference to them.

        if not keys:
            raise ValueError("Cannot extract features from an empty dataset")
        self._check_grids(grids, plan)

        return CachedFeatures(
            cache=self,
            keys=keys,
            layer_indices=plan.indices,
            multi=plan.multi,
            grid_hw=grids[-1].pop(),
            targets=targets,
        )

    def extract_dataset(
        self,
        backbone: Any,
        dataset: Iterable,
        pooling: str = Pooling.DEFAULT,
        layer: int | None = None,
        layers: LayerSpec = None,
        batch_size: int = 32,
        keep: str = "both",
        store: str | None = None,
        feature_mode: str = FeatureMode.DENSE_ONLY,
    ) -> FeatureDict:
        """Extract (or load) features for a whole dataset, stacked in memory.

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

        The *result* is still held whole in memory, which is fine for pooled
        features and a hard wall for dense ones. Use :meth:`materialise` for a
        dense task on anything large.

        Parameters
        ----------
        layer / layers:
            Which backbone depth(s) to read. ``layer`` is the single-layer
            form; ``layers`` takes a list and adds ``dense_layers`` /
            ``layer_indices`` to the result, for a multiscale head. Passing
            both raises — they would have to agree, and nothing here could
            check that they did.

            Every layer gets **its own cache entry**, keyed on the resolved
            index. Widening ``[3, 7]`` to ``[3, 7, 11]`` therefore re-extracts
            one layer rather than all three, and a later single-layer run at
            layer 7 reads what the multi-layer run already stored.
        keep:
            Which outputs to accumulate in memory and return: ``"both"``,
            ``"pooled"`` or ``"dense"``. ``dense`` is the memory risk — 5k
            images at 16x16x768 is roughly 4 GB in fp32 — and a multi-layer
            request multiplies it by the number of layers, so a task needing
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

        plan = self._plan(
            backbone,
            pooling=pooling,
            layer=layer,
            layers=layers,
            keep=keep,
            store=store,
            feature_mode=feature_mode,
            streaming=False,
        )

        # One accumulator per requested layer; the single-layer case is the
        # list-of-one, so both paths run the same code rather than branching.
        dense_chunks: list[list[torch.Tensor]] = [[] for _ in plan.indices]
        pooled_chunks: list[torch.Tensor] = []
        cls_chunks: list[torch.Tensor] = []
        grids: list[set] = [set() for _ in plan.indices]
        count = 0

        for batch in self._walk(backbone, dataset, plan, batch_size):
            for _, entry in batch:
                count += 1
                for depth, layer_entry in enumerate(entry):
                    grids[depth].add(layer_entry["grid_hw"])
                    if keep in ("both", "dense"):
                        dense_chunks[depth].append(layer_entry["dense"])
                deepest = entry[-1]
                if keep in ("both", "pooled"):
                    pooled_chunks.append(deepest["pooled"])
                if "cls" in deepest:
                    cls_chunks.append(deepest["cls"])

        if count == 0:
            raise ValueError("Cannot extract features from an empty dataset")
        self._check_grids(grids, plan)

        result: FeatureDict = {"grid_hw": grids[-1].pop()}  # type: ignore[typeddict-item]
        if keep in ("both", "dense"):
            stacked = [torch.cat(chunks) for chunks in dense_chunks]
            result["dense"] = stacked[-1]
            if plan.multi:
                result["dense_layers"] = stacked
        if keep in ("both", "pooled"):
            result["pooled"] = torch.cat(pooled_chunks)
        if cls_chunks:
            result["cls"] = torch.cat(cls_chunks)
        if plan.multi:
            result["layer_indices"] = plan.indices
        return result

    @staticmethod
    def _check_grids(grids: list[set], plan: "_Plan") -> None:
        """Every image must produce one grid shape per layer, or nothing stacks."""
        for depth, seen in enumerate(grids):
            if len(seen) > 1:
                where = f" at layer {plan.indices[depth]}" if plan.multi else ""
                raise ValueError(
                    f"Dataset produced more than one dense grid shape{where} "
                    f"({sorted(seen)}); features cannot be stacked. Use a fixed "
                    "input resolution."
                )

    # -- maintenance ---------------------------------------------------------

    def clear(self, backbone_key: str | None = None) -> int:
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

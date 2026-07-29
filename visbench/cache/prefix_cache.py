"""Frozen-prefix activation cache — step 6b.

Fine-tuning unfreezes the last ``n`` blocks of a backbone. Everything *below*
that cut stays frozen for the whole run, so its output is a pure function of
the image and weights that do not change — exactly the property the feature
cache relies on, and the reason step 6a could not use that cache while this one
can.

What is stored is the token sequence after ``cut`` blocks: **before** the norm,
before pooling, before any grid reshaping. That is deliberately less processed
than a :class:`~visbench.cache.FeatureCache` entry, because it has to serve
every pooling and feature mode a task might ask for from one tensor.

Why this is a separate class rather than a mode on ``FeatureCache``:

* the entries are not interchangeable, and a mix-up is silent — a prefix
  resumed as though it were features, or features handed to a resumption, both
  produce plausible numbers rather than errors;
* the two have different key shapes (a prefix has no pooling or feature mode),
  and widening ``make_key`` to carry fields that are meaningless for one of its
  two uses is how a key stops describing what it names;
* ``FeatureCache`` is on the frozen path that every published VisBench number
  came from. This is on the fine-tuning path. Keeping the code apart is the
  same instinct as keeping the *numbers* apart in the result record.

Sizing, measured on VOC at 224px: one prefix is 386 KB on DINOv2-S/14 and
772 KB on DINOv2-B/14 — within a few percent of a dense feature entry, since
it is the same tokens one layer earlier. So this trades the frozen blocks'
forward compute for a per-file read of the size the frozen path already pays,
which is what the 6a measurements say is the term that matters.
"""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from visbench.cache.feature_cache import DEFAULT_CACHE_DIR, PREFIX_DIR
from visbench.cache.keys import SEPARATOR

__all__ = ["PrefixCache"]

_ENTRY_SUFFIX = ".pt"


class PrefixCache:
    """Disk store mapping a prefix key to one frozen activation.

    Entries are stored on CPU regardless of the compute device, matching
    :class:`~visbench.cache.FeatureCache`, so a cache written on a GPU box is
    readable elsewhere.
    """

    def __init__(self, root: Path | None = None, enabled: bool = True) -> None:
        """Open (creating if needed) the prefix directory under ``root``.

        ``root`` is the *feature cache* root; the prefix entries go in
        :data:`PREFIX_DIR` beneath it. ``enabled=False`` turns every lookup
        into a miss without changing call sites, which is what makes the
        "measure it without the cache" comparison a flag rather than a
        different code path.
        """
        base = Path(root) if root is not None else DEFAULT_CACHE_DIR
        self.root = base / PREFIX_DIR
        self.enabled = enabled
        self._hits = 0
        self._misses = 0
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        """``<root>/_prefix/<backbone>/<shard>/<digest>.pt``, as features are laid out."""
        backbone_key = key.split(SEPARATOR, 1)[0]
        safe_backbone = backbone_key.replace("/", "__").replace(os.sep, "__")
        digest = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self.root / safe_backbone / digest[:2] / f"{digest}{_ENTRY_SUFFIX}"

    def get(self, key: str) -> tuple[torch.Tensor, tuple[int, int]] | None:
        """Return ``(tokens, grid_hw)``, or ``None`` on a miss.

        ``grid_hw`` travels *with* the activation rather than being recomputed
        from the token count. Token count alone gives the number of patches,
        not their arrangement, so a non-square input would be reconstructed as
        the wrong grid — the misalignment class of bug the correspondence task
        already paid for once.
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
            # Truncated or corrupt reads as a miss, never a hit; the put() that
            # follows overwrites it, so the cache self-heals.
            self._misses += 1
            return None

        if "tokens" not in entry or "grid_hw" not in entry:
            self._misses += 1
            return None

        self._hits += 1
        height, width = entry["grid_hw"]
        return entry["tokens"], (int(height), int(width))

    def put(self, key: str, tokens: torch.Tensor, grid_hw: tuple[int, int]) -> None:
        """Write one activation, atomically.

        The tensor is detached before it reaches disk. It is produced under
        ``no_grad`` in the one caller that exists, but a cached tensor carrying
        a graph would pin the whole forward pass in memory for the lifetime of
        the entry, so this does not rely on the caller having done it.
        """
        if not self.enabled:
            return

        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "tokens": tokens.detach().cpu(),
            # A list, not a tuple: weights_only=True unpickles lists but not
            # arbitrary tuples. Restored to a tuple on read.
            "grid_hw": [int(grid_hw[0]), int(grid_hw[1])],
        }

        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                torch.save(payload, handle)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def stats(self) -> dict:
        """Entry count, on-disk size, and this session's hit/miss counts."""
        entries = list(self.root.rglob(f"*{_ENTRY_SUFFIX}")) if self.root.exists() else []
        return {
            "entries": len(entries),
            "bytes": sum(path.stat().st_size for path in entries),
            "hits": self._hits,
            "misses": self._misses,
            "enabled": self.enabled,
            "root": str(self.root),
        }

    def clear(self, backbone_key: str | None = None) -> int:
        """Delete every entry, or only one backbone's. Returns the count removed."""
        if not self.root.exists():
            return 0
        if backbone_key is None:
            targets = list(self.root.rglob(f"*{_ENTRY_SUFFIX}"))
        else:
            safe = backbone_key.replace("/", "__").replace(os.sep, "__")
            targets = list((self.root / safe).rglob(f"*{_ENTRY_SUFFIX}"))
        for path in targets:
            path.unlink(missing_ok=True)
        return len(targets)

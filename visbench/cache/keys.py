"""Cache key construction.

Keys are ``(image_hash, backbone_name, layer, pooling)`` per CLAUDE.md. Getting
this wrong means silently serving stale or mismatched features, so hashing is
isolated here and tested directly.
"""

from typing import Optional

__all__ = ["hash_image", "make_key"]


def hash_image(image) -> str:
    """Content hash of a single image.

    Hashes decoded pixel content rather than the file path, so the same image
    under two filenames hits the same cache entry, and an edited file under an
    unchanged filename does not.
    """
    raise NotImplementedError


def make_key(
    image_hash: str,
    backbone_key: str,
    layer: Optional[int],
    pooling: str,
) -> str:
    """Build the cache key for one (image, backbone, layer, pooling) combination.

    ``backbone_key`` comes from :meth:`BaseBackbone.cache_key` and must encode
    the weights and input resolution, not just the model family.

    ``layer`` is ``None`` for the v0.1 single-layer path; the field exists now
    so v0.2 multi-layer entries do not collide with v0.1 ones.
    """
    raise NotImplementedError

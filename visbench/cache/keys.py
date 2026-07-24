"""Cache key construction.

Keys are ``(image_hash, backbone_name, layer, pooling)`` per CLAUDE.md. Getting
this wrong means silently serving stale or mismatched features, so hashing is
isolated here and tested directly.
"""

import hashlib
from typing import Optional

import numpy as np
import torch
from PIL import Image

__all__ = ["hash_image", "make_key"]

#: 128 bits of digest. Short enough for a readable filename, far past any
#: realistic collision risk for a per-user feature cache.
_DIGEST_CHARS = 32

#: Field separator in a composite key. Backbone keys are sanitised against it
#: so the key can be split back apart unambiguously.
SEPARATOR = "|"


def hash_image(image) -> str:
    """Content hash of a single image.

    Hashes decoded pixel content rather than the file path, so the same image
    under two filenames hits the same cache entry, and an edited file under an
    unchanged filename does not.

    Accepts a PIL image or a tensor/array; the dimensions and mode are folded
    into the digest, because raw bytes alone cannot distinguish a 2x3 image
    from a 3x2 one.
    """
    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
        header = f"PIL|{rgb.mode}|{rgb.width}x{rgb.height}"
        payload = rgb.tobytes()
    elif isinstance(image, torch.Tensor):
        array = image.detach().cpu().contiguous().numpy()
        header = f"tensor|{array.dtype}|{'x'.join(map(str, array.shape))}"
        payload = array.tobytes()
    elif isinstance(image, np.ndarray):
        array = np.ascontiguousarray(image)
        header = f"array|{array.dtype}|{'x'.join(map(str, array.shape))}"
        payload = array.tobytes()
    else:
        raise TypeError(
            f"Cannot hash {type(image).__name__}; expected a PIL image, torch tensor or numpy array"
        )

    digest = hashlib.sha256()
    digest.update(header.encode())
    digest.update(payload)
    return digest.hexdigest()[:_DIGEST_CHARS]


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
    so v0.2 multi-layer entries do not collide with v0.1 ones. It is rendered
    as ``"-"``, which is distinct from ``"0"`` — otherwise "the default layer"
    and "layer 0" would share an entry.

    The backbone key comes first so :meth:`FeatureCache.clear` can select a
    whole backbone by prefix.
    """
    for field, value in (("backbone_key", backbone_key), ("pooling", pooling)):
        if SEPARATOR in value:
            raise ValueError(f"{field} must not contain {SEPARATOR!r}: {value!r}")
    layer_field = "-" if layer is None else str(layer)
    return SEPARATOR.join([backbone_key, layer_field, pooling, image_hash])

"""Cache key construction.

Keys are ``(image_hash, backbone_name, layer, pooling)`` per CLAUDE.md. Getting
this wrong means silently serving stale or mismatched features, so hashing is
isolated here and tested directly.
"""

import hashlib

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
    layer: int | None,
    pooling: str,
    feature_mode: str = "dense_only",
) -> str:
    """Build the cache key for one (image, backbone, layer, pooling, mode) entry.

    ``backbone_key`` comes from :meth:`BaseBackbone.cache_key` and must encode
    the weights and input resolution, not just the model family.

    ``layer`` is ``None`` for the v0.1 single-layer path; the field exists now
    so v0.2 multi-layer entries do not collide with v0.1 ones. It is rendered
    as ``"-"``, which is distinct from ``"0"`` — otherwise "the default layer"
    and "layer 0" would share an entry.

    ``feature_mode`` is in the key because the three modes produce genuinely
    different ``dense`` tensors from one forward pass — ``dense_cls_broadcast``
    has twice the channels of ``dense_only`` — and serving one for the other
    would be a shape error at best and a wrong feature map at worst.

    The consequence is that sweeping modes over one dataset stores each
    separately. That is the honest trade: the alternative is caching the
    primitive grid plus CLS and re-assembling on read, which saves disk at the
    cost of the cache no longer holding what it hands back.

    The backbone key comes first so :meth:`FeatureCache.clear` can select a
    whole backbone by prefix.
    """
    for field, value in (
        ("backbone_key", backbone_key),
        ("pooling", pooling),
        ("feature_mode", feature_mode),
    ):
        if SEPARATOR in value:
            raise ValueError(f"{field} must not contain {SEPARATOR!r}: {value!r}")
    layer_field = "-" if layer is None else str(layer)
    return SEPARATOR.join([backbone_key, layer_field, pooling, feature_mode, image_hash])

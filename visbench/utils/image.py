"""Image loading and normalisation constants."""

from pathlib import Path

from PIL import Image, ImageOps

__all__ = ["load_image", "IMAGENET_MEAN", "IMAGENET_STD", "CLIP_MEAN", "CLIP_STD"]

#: Standard ImageNet normalisation, used by DINOv2 among others.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: CLIP uses its own constants — applying ImageNet's instead is a silent
#: accuracy loss, not an error, so keep them explicitly separate.
CLIP_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)


def load_image(path: Path) -> Image.Image:
    """Load an image as RGB PIL, applying EXIF orientation.

    EXIF handling matters for the feature cache: the same photo loaded with and
    without orientation applied hashes differently and gives different features.
    """
    with Image.open(path) as img:
        # exif_transpose before convert: the orientation tag lives on the file,
        # and convert("RGB") drops it.
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")

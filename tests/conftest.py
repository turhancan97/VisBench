"""Shared test fixtures.

The fake backbones here exist so the :class:`BaseBackbone` contract and the
feature cache can be tested without downloading weights. A test that needs the
real DINOv2 checkpoint is marked ``slow`` and lives beside these.
"""

from pathlib import Path

import pytest
import torch
from PIL import Image

from visbench.backbones.base import BaseBackbone
from visbench.data.triplet import TwoAFCDataset


class FakeViT(BaseBackbone):
    """ViT-shaped backbone with deterministic features and a call counter.

    Features are derived from the input, not random, so a cached value and a
    recomputed one can be compared for equality.
    """

    has_cls_token = True

    def __init__(self, device: str | None = "cpu", embed_dim: int = 8, patch_size: int = 16):
        super().__init__(device)
        self.name = "fake_vit"
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = 64
        #: Incremented on every forward, so tests can assert the cache actually
        #: prevented a second extraction.
        self.call_count = 0
        # One real parameter, so freeze/eval assertions have something to check.
        self.proj = torch.nn.Linear(3, embed_dim)
        self._finalize()

    @property
    def num_layers(self):
        """As many blocks as a real ViT-B, so layer indices behave the same."""
        return 12

    def _forward_features(self, image, layers):
        self.call_count += 1
        b, _, h, w = image.shape
        grid_hw = (h // self.patch_size, w // self.patch_size)
        n = grid_hw[0] * grid_hw[1]
        # Deterministic in the input: mean colour per image, projected, then
        # varied per token so mean-pooling and CLS differ.
        base = self.proj(image.mean(dim=(2, 3)))
        offsets = torch.arange(n, dtype=base.dtype).view(1, n, 1)
        outputs = []
        for index in layers:
            # Offset by layer so two depths are never accidentally equal —
            # a multi-layer test that passes on identical stages proves nothing.
            patch_tokens = base.unsqueeze(1) + offsets + float(index)
            outputs.append((patch_tokens, base * 2.0 + float(index), grid_hw))
        return outputs

    def preprocess(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        tensors = []
        for img in images:
            resized = img.convert("RGB").resize((self.image_size, self.image_size))
            # bytearray, not bytes: frombuffer warns on a read-only buffer.
            array = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
            tensors.append(
                array.view(self.image_size, self.image_size, 3).permute(2, 0, 1).float() / 255
            )
        return torch.stack(tensors)

    def cache_key(self) -> str:
        return f"fake_vit/{self.embed_dim}/{self.image_size}"


class FakeCNN(BaseBackbone):
    """CNN-shaped backbone: no CLS token, conv map flattened to tokens.

    Present to prove the base class needs no ``if is_vit`` branch — the same
    ``extract_features`` serves both families.
    """

    has_cls_token = False

    def __init__(self, device: str | None = "cpu", embed_dim: int = 8):
        super().__init__(device)
        self.name = "fake_cnn"
        self.embed_dim = embed_dim
        self.patch_size = None
        self.image_size = 64
        self.conv = torch.nn.Conv2d(3, embed_dim, kernel_size=8, stride=8)
        # Two shallower stages, narrower and at higher resolution, so the
        # multi-layer path is exercised against a real CNN's shape behaviour:
        # stages differ in *both* width and stride, unlike a ViT's blocks.
        self.stage0 = torch.nn.Conv2d(3, embed_dim // 4, kernel_size=2, stride=2)
        self.stage1 = torch.nn.Conv2d(3, embed_dim // 2, kernel_size=4, stride=4)
        self._finalize()

    @property
    def num_layers(self):
        return 3

    def _forward_features(self, image, layers):
        stages = [self.stage0, self.stage1, self.conv]
        outputs = []
        for index in layers:
            feature_map = stages[index](image)
            _, _, h, w = feature_map.shape
            # The flatten a real CNN subclass performs; CLS is None.
            outputs.append((feature_map.flatten(2).transpose(1, 2), None, (h, w)))
        return outputs

    def preprocess(self, images):
        raise NotImplementedError("Not needed for these tests")

    def cache_key(self) -> str:
        return f"fake_cnn/{self.embed_dim}/{self.image_size}"


@pytest.fixture
def fake_vit():
    return FakeViT()


@pytest.fixture
def fake_cnn():
    return FakeCNN()


@pytest.fixture
def solid_images():
    """Four distinguishable 64x64 images."""
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)]
    return [Image.new("RGB", (64, 64), colour) for colour in colours]


@pytest.fixture
def two_afc_folder():
    """Build a NIGHTS-shaped 2AFC folder: a CSV plus the images it names.

    The candidate humans "preferred" is drawn to match the reference's colour,
    so the vote is learnable from features rather than arbitrary — a probe that
    scores at chance on this has a real problem.
    """

    def build(root, triplets=6, split="test", votes=8, min_votes_ok=True, construct=True):
        root = Path(root)
        (root / "ref").mkdir(parents=True)
        (root / "distort").mkdir(parents=True)

        rows = ["id,left_vote,right_vote,votes,ref_path,left_path,right_path,split,is_imagenet"]
        for index in range(triplets):
            shade = 20 + 30 * (index % 7)
            near = (shade, shade, shade)
            far = (255 - shade, 0, 255)
            # Alternate which side is the match, so a probe that always answers
            # one way scores 50% rather than passing by accident.
            right_is_match = index % 2 == 1
            left_colour = far if right_is_match else near
            right_colour = near if right_is_match else far

            Image.new("RGB", (32, 32), near).save(root / "ref" / f"{index:03d}.png")
            Image.new("RGB", (32, 32), left_colour).save(root / "distort" / f"{index:03d}_0.png")
            Image.new("RGB", (32, 32), right_colour).save(root / "distort" / f"{index:03d}_1.png")

            rows.append(
                f"{index},{int(not right_is_match)},{int(right_is_match)},"
                f"{votes if min_votes_ok else 2},"
                f"ref/{index:03d}.png,distort/{index:03d}_0.png,distort/{index:03d}_1.png,"
                f"{split},{'TRUE' if index % 3 == 0 else 'FALSE'}"
            )

        (root / "data.csv").write_text("\n".join(rows) + "\n")
        # construct=False returns the folder only, for the cases that need to
        # assert on how construction *fails*.
        return TwoAFCDataset(root, split=split) if construct else root

    return build

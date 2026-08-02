"""The zero-setup demo.

`visbench demo` is the first thing a new user runs, so its failure mode is not
"a test goes red" — it is someone concluding the library does not work and
leaving. These tests cover the generator and the command; the real torchvision
backbone is exercised in the slow suite.
"""

import pytest
from PIL import Image

from visbench.demo import SHAPES, synthesise


class TestSynthesise:
    def test_it_writes_the_layout_the_probes_expect(self, tmp_path):
        root = synthesise(tmp_path / "shapes", per_class=3, image_size=64)

        for split in ("train", "val"):
            for shape in SHAPES:
                files = sorted((root / split / shape).glob("*.png"))
                assert len(files) == 3, f"{split}/{shape}"

    def test_images_are_the_requested_size_and_rgb(self, tmp_path):
        root = synthesise(tmp_path / "shapes", per_class=1, image_size=64)
        image = Image.open(next((root / "train" / "circle").glob("*.png")))
        assert image.size == (64, 64)
        assert image.mode == "RGB"

    def test_it_is_deterministic(self, tmp_path):
        """A documented demo number is only meaningful if the data repeats."""
        first = synthesise(tmp_path / "a", per_class=2, image_size=64, seed=7)
        second = synthesise(tmp_path / "b", per_class=2, image_size=64, seed=7)

        a = (first / "train" / "square" / "000.png").read_bytes()
        b = (second / "train" / "square" / "000.png").read_bytes()
        assert a == b

    def test_a_different_seed_gives_different_data(self, tmp_path):
        first = synthesise(tmp_path / "a", per_class=2, image_size=64, seed=1)
        second = synthesise(tmp_path / "b", per_class=2, image_size=64, seed=2)

        a = (first / "train" / "square" / "000.png").read_bytes()
        b = (second / "train" / "square" / "000.png").read_bytes()
        assert a != b

    @pytest.mark.parametrize("knob", ["noise", "contrast"])
    def test_the_difficulty_knobs_change_the_data(self, tmp_path, knob):
        """A parameter that is accepted and does nothing is the 6d-1 failure.

        `target_scale` was recorded, fingerprinted and ignored for a whole
        release, and the sweep that should have caught it returned four
        identical rows. When a generator takes a numeric parameter, test that
        changing it changes the bytes.
        """
        base = {"per_class": 1, "image_size": 64, "seed": 0}
        low = synthesise(tmp_path / "low", **base, **{knob: 5})
        high = synthesise(tmp_path / "high", **base, **{knob: 60})

        a = (low / "train" / "circle" / "000.png").read_bytes()
        b = (high / "train" / "circle" / "000.png").read_bytes()
        assert a != b, f"{knob} had no effect on the generated image"

    def test_colour_does_not_identify_the_class(self, tmp_path):
        """The point of the whole design.

        If mean colour separated the shapes, the demo would measure a colour
        lookup and report it as shape recognition — the same shortcut that makes
        a saturated benchmark useless.
        """
        import numpy as np

        root = synthesise(tmp_path / "shapes", per_class=12, image_size=64, seed=0)
        means = {}
        for shape in SHAPES:
            values = [
                np.asarray(Image.open(p), dtype=float).mean()
                for p in sorted((root / "train" / shape).glob("*.png"))
            ]
            means[shape] = (float(np.mean(values)), float(np.std(values)))

        spread_between = float(np.std([m for m, _ in means.values()]))
        spread_within = float(np.mean([s for _, s in means.values()]))
        assert spread_between < spread_within, (
            f"class mean brightness separates the shapes ({means}); colour must not be a shortcut"
        )

    def test_an_unknown_shape_is_refused(self, tmp_path):
        import random

        from visbench.demo import _draw_shape

        with pytest.raises(ValueError, match="Unknown shape"):
            _draw_shape("hexagon", 32, random.Random(0), 30, 0.0)


@pytest.mark.slow
class TestDemoOnRealWeights:
    """The half the fake backbone cannot cover: torchvision's ResNet-18."""

    def test_the_demo_backbone_produces_features(self):
        import torch

        from visbench.demo import demo_backbone

        backbone = demo_backbone(device="cpu")
        features = backbone.extract_features(torch.rand(1, 3, 224, 224))
        assert features["dense"].shape[:2] == (1, 512)
        assert features["pooled"].shape == (1, 512)

    def test_the_documented_score_still_holds(self, tmp_path):
        """The number in the docstring and the README is a claim; check it.

        Loose bounds on purpose — the claim is "clearly above chance and clearly
        not saturated", which is what makes the demo worth printing. A tight
        assertion here would fail on a torchvision weight refresh and say
        nothing useful.
        """
        import visbench
        from visbench.cache import FeatureCache
        from visbench.data import ImageFolderDataset
        from visbench.demo import demo_backbone

        root = synthesise(tmp_path / "shapes", per_class=20, seed=0)
        backbone = demo_backbone(device="cpu")
        result = visbench.run(
            backbone,
            "classification",
            ImageFolderDataset(root / "val", split="val"),
            train_dataset=ImageFolderDataset(root / "train", split="train"),
            cache=FeatureCache(root=tmp_path / "cache"),
            device="cpu",
            seed=0,
        )
        top1 = result.metrics["top1"]
        assert 0.5 < top1 < 0.97, f"demo top1 drifted to {top1}; chance is 0.25"

    def test_more_noise_drives_the_score_toward_chance(self, tmp_path):
        """The demo's real lesson, and the check that it measures something.

        A probe whose score does not move when the signal is destroyed is not
        measuring the signal.
        """
        import visbench
        from visbench.cache import FeatureCache
        from visbench.data import ImageFolderDataset
        from visbench.demo import demo_backbone

        backbone = demo_backbone(device="cpu")
        scores = []
        for noise, contrast in ((45.0, 30), (90.0, 16)):
            root = synthesise(
                tmp_path / f"n{noise}", per_class=20, seed=0, noise=noise, contrast=contrast
            )
            result = visbench.run(
                backbone,
                "classification",
                ImageFolderDataset(root / "val", split="val"),
                train_dataset=ImageFolderDataset(root / "train", split="train"),
                cache=FeatureCache(root=tmp_path / "cache"),
                device="cpu",
                seed=0,
            )
            scores.append(result.metrics["top1"])

        assert scores[0] > scores[1], f"harder data did not score lower: {scores}"
        assert scores[1] < 0.5, f"the hard setting should approach chance, got {scores[1]}"

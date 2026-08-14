"""Geometric correspondence — zero-shot dense feature matching. v0.1.

The only v0.1 task that uses dense features, which makes it the real test of
the dense path in :meth:`BaseBackbone.extract_features`.

Matching logic is conceptually the same as vismatch
(https://github.com/gmberton/vismatch), applied to raw backbone features
instead of dedicated matcher networks. Evaluation protocol follows probe3d
(El Banani et al., CVPR 2024, arXiv:2404.08476), implemented from its
published description rather than its code, which is CC BY-NC (see NOTICE).
"""

from typing import Any

import torch

from visbench.data.pair_dataset import apply_homography
from visbench.metrics.correspondence import (
    correspondence_recall,
    error_auc,
    nn_match,
    ratio_test,
)
from visbench.registry import register_task
from visbench.tasks.base import BaseTask
from visbench.types import FeatureMode, MetricsDict

__all__ = ["CorrespondenceTask", "patch_centers", "patch_spacing"]


def patch_centers(grid_hw: tuple[int, int], size: tuple[int, int]) -> torch.Tensor:
    """Pixel xy centre of every patch in a ``grid_hw`` grid over a ``size`` image.

    A patch is a region, not a point, so its centre sits at ``(i + 0.5)`` cells
    in — not at ``i``. Using the corner instead biases every reported error by
    half a patch, which at 14px patches is a 7px offset that would swamp a
    1px threshold.
    """
    grid_h, grid_w = grid_hw
    width, height = size
    ys = (torch.arange(grid_h, dtype=torch.float64) + 0.5) * height / grid_h
    xs = (torch.arange(grid_w, dtype=torch.float64) + 0.5) * width / grid_w
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)


def patch_spacing(grid_hw: tuple[int, int], size: tuple[int, int]) -> float:
    """Mean distance in pixels between neighbouring patch centres.

    The quantisation floor of the whole protocol: no match can be more precise
    than this, whatever the features are. Averaged over the two axes, which are
    equal for a square input and close to it otherwise.
    """
    grid_h, grid_w = grid_hw
    width, height = size
    return float((width / grid_w + height / grid_h) / 2)


@register_task("correspondence")
class CorrespondenceTask(BaseTask):
    """Match dense features between image pairs and score against known geometry.

    Given a pair of views of the same scene/object, nearest-neighbour match
    patch features (with a ratio test to reject ambiguous matches), then score
    the surviving matches against ground-truth geometry.
    """

    level = "mid_level"
    feature_mode = FeatureMode.DENSE_ONLY
    zero_shot = True
    uses_dense = True
    #: Pairs plus geometry, not images plus labels. :func:`visbench.run` reads
    #: this to extract both views of every pair and hand them back paired.
    uses_pairs = True

    def __init__(
        self,
        num_corr: int = 1000,
        ratio_threshold: float = 0.9,
        thresholds: tuple | None = None,
        threshold_units: str = "pixel",
    ) -> None:
        """Configure how many correspondences to keep and the error thresholds scored.

        Parameters
        ----------
        threshold_units:
            ``"pixel"`` (default) measures error in raw pixels of the input
            frame; ``"patch"`` in multiples of this backbone's patch spacing.

            **Pixels are the default because they are the only unit two
            backbones can be compared in, and this was wrong until v0.6.1.**
            A patch width is a *property of the backbone*: at 224px it is 14px
            on DINOv2/14, 16px on CLIP ViT-B/16 and 32px on ViT-B/32 or a
            ResNet's final stage. So ``recall@1p`` asks a coarse-grid backbone
            to land within 32px and a fine-grid one within 14px, then prints
            both under one name.

            Measured on 200 Imagenette pairs, that inverts the board outright:

            ==============  ==============  ================
            backbone        ``recall@1p``   ``recall@5px``
            ==============  ==============  ================
            resnet18        **0.8927**      0.0973
            dinov2_vits14   0.7834          **0.3049**
            ==============  ==============  ================

            First place and last place swap. Nothing about the patch-unit
            number is a lie *within* one backbone — it is a real answer to
            "how close, relative to what this grid can resolve" — but it is not
            the question a leaderboard row is read as answering.

            The quantisation floor that motivated patch units is real and is
            handled the honest way instead: :meth:`evaluate_ceiling` reports
            what perfect matching on this grid would score, and
            :meth:`BaseTask.context_metrics` carries it beside every result. A
            7x7 grid has a ``ceiling_recall@5px`` of ~0.10 against ~0.41 for a
            16x16 one, which *states* the disadvantage rather than normalising
            it away.

            Use ``"patch"`` deliberately, for one backbone against itself or to
            reproduce a pre-v0.6.1 number. Do not rank two backbones with it.
        thresholds:
            Defaults to ``(1, 2, 5, 10)`` pixels, or ``(0.5, 1, 2, 4)`` patch
            widths, matching the chosen unit.
        """
        if num_corr < 1:
            raise ValueError(f"num_corr must be >= 1, got {num_corr}")
        if not 0 < ratio_threshold <= 1:
            raise ValueError(f"ratio_threshold must be in (0, 1], got {ratio_threshold}")
        if threshold_units not in ("patch", "pixel"):
            raise ValueError(f"threshold_units must be 'patch' or 'pixel', got {threshold_units!r}")
        if thresholds is None:
            thresholds = (0.5, 1, 2, 4) if threshold_units == "patch" else (1, 2, 5, 10)
        if not thresholds:
            raise ValueError("thresholds must contain at least one value")

        self.name = "correspondence"
        self.num_corr = num_corr
        self.ratio_threshold = ratio_threshold
        self.threshold_units = threshold_units
        self.thresholds = tuple(sorted(thresholds))
        self._unit = "p" if threshold_units == "patch" else "px"

    def fit(self, features: Any, labels: Any | None = None) -> "CorrespondenceTask":
        """No-op — correspondence is zero-shot. Returns ``self``."""
        return self

    # -- matching ------------------------------------------------------------

    @staticmethod
    def _as_dense(features: Any) -> torch.Tensor:
        """Accept a FeatureDict or a bare tensor, return ``(C, H, W)``."""
        dense = features["dense"] if isinstance(features, dict) else features
        if not isinstance(dense, torch.Tensor):
            raise TypeError(f"Expected dense features as a tensor, got {type(dense).__name__}")
        if dense.ndim == 4:
            if len(dense) != 1:
                raise ValueError(
                    f"Expected one image's dense features, got a batch of {len(dense)}. "
                    "Correspondence matches pairs one at a time."
                )
            dense = dense[0]
        if dense.ndim != 3:
            raise ValueError(f"Expected dense features (C, H, W), got {tuple(dense.shape)}")
        return dense

    def match(self, features_0: Any, features_1: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Match one pair. Returns ``(indices_0, indices_1)`` into the flattened grids.

        Patches are matched from view 0 into view 1, filtered by the ratio
        test, then the most confident ``num_corr`` are kept. Confidence is the
        nearest-neighbour distance: keeping the *best* matches rather than an
        arbitrary slice is what makes the number comparable across backbones
        that reject different amounts.
        """
        dense_0 = self._as_dense(features_0)
        dense_1 = self._as_dense(features_1)

        tokens_0 = dense_0.flatten(1).T  # (N, C)
        tokens_1 = dense_1.flatten(1).T

        distances, indices = nn_match(tokens_0, tokens_1, k=2)
        keep = ratio_test(distances, self.ratio_threshold)

        kept_source = torch.nonzero(keep, as_tuple=True)[0]
        if len(kept_source) == 0:
            empty = torch.zeros(0, dtype=torch.long)
            return empty, empty

        kept_distance = distances[kept_source, 0]
        order = kept_distance.argsort()[: self.num_corr]
        selected = kept_source[order]
        # Returned on CPU regardless of where the features live. All the
        # geometry downstream is float64 on CPU — homographies need the
        # precision and are tiny — and indexing a CPU tensor with CUDA indices
        # raises. Features straight off a GPU backbone would otherwise fail
        # here while the same call through the cache, which returns CPU
        # tensors, works.
        return selected.cpu(), indices[selected, 0].cpu()

    def predict(self, features: Any) -> Any:
        """Return matched patch-index pairs and their similarity scores.

        ``features`` is a sequence of ``(features_0, features_1)`` pairs.
        """
        return [self.match(pair[0], pair[1]) for pair in features]

    def match_details(self, features_0: Any, features_1: Any, geometry: dict) -> dict:
        """Every kept match as points and errors, for one pair.

        The per-match view of what :meth:`evaluate` pools. Returned as pixel
        coordinates in the working frame, so a caller can draw them straight
        onto the images the dataset yielded:

        ``source``
            ``(M, 2)`` patch centres in view 0 that survived the ratio test.
        ``target``
            ``(M, 2)`` centres in view 1 they were matched to.
        ``expected``
            ``(M, 2)`` where the geometry says each ``source`` point *should*
            have landed in view 1. ``target`` minus this is the error vector.
        ``errors_px``
            ``(M,)`` distances in pixels, whatever ``threshold_units`` is.
        ``errors``
            ``(M,)`` the same in the configured unit — what is scored.

        Split out of :meth:`_pair_errors`, which now calls it, so a rendered
        panel and a reported number cannot disagree about which matches were
        kept or how far off they were. Recomputing this alongside the scorer
        would put a second copy of the geometry one edit away from drifting,
        and a *drawing* that disagrees with the score is worse than none: it
        would vouch for the number.
        """
        if "homography" not in geometry:
            raise KeyError(
                "v0.1 scores correspondence against a homography; geometry dict has "
                f"keys {sorted(geometry)}. Depth-and-pose geometry arrives in v0.2."
            )

        source_idx, target_idx = self.match(features_0, features_1)
        empty = torch.zeros(0, dtype=torch.float64)
        if len(source_idx) == 0:
            points = torch.zeros(0, 2, dtype=torch.float64)
            return {
                "source": points,
                "target": points.clone(),
                "expected": points.clone(),
                "errors_px": empty,
                "errors": empty.clone(),
            }

        size = geometry["size"]
        grid_0 = self._grid_hw(features_0)
        grid_1 = self._grid_hw(features_1)

        centres_0 = patch_centers(grid_0, size)[source_idx]
        centres_1 = patch_centers(grid_1, size)[target_idx]

        # Where view 0's patch centres *should* land in view 1.
        expected = apply_homography(geometry["homography"], centres_0)
        errors_px = (centres_1 - expected).norm(dim=1)
        return {
            "source": centres_0,
            "target": centres_1,
            "expected": expected,
            "errors_px": errors_px,
            "errors": self._scale(errors_px, grid_1, size),
        }

    # -- scoring -------------------------------------------------------------

    def _pair_errors(self, features_0: Any, features_1: Any, geometry: dict) -> torch.Tensor:
        """Error of every kept match for one pair, in the configured unit."""
        return self.match_details(features_0, features_1, geometry)["errors"]

    def _scale(self, errors: torch.Tensor, grid_hw: tuple, size: tuple) -> torch.Tensor:
        """Convert pixel errors to the configured unit.

        Done per pair, before pooling: two pairs at different resolutions have
        different patch spacings, so converting afterwards would mix scales.
        """
        if self.threshold_units == "pixel":
            return errors
        return errors / patch_spacing(grid_hw, size)

    @staticmethod
    def _grid_hw(features: Any) -> tuple[int, int]:
        if isinstance(features, dict) and "grid_hw" in features:
            return tuple(features["grid_hw"])
        dense = CorrespondenceTask._as_dense(features)
        return (dense.shape[1], dense.shape[2])

    def evaluate(self, features: Any, labels: Any | None = None) -> MetricsDict:
        """Return recall at each pixel threshold plus the error AUC.

        ``features`` is a sequence of ``(features_0, features_1)`` pairs and
        ``labels`` the matching sequence of geometry dicts.

        Errors from every pair are pooled before scoring, so a pair that
        survives more matches contributes more — matching probe3d, and
        avoiding a per-pair mean in which a pair with two lucky matches counts
        as much as one with two hundred.
        """
        if labels is None:
            raise ValueError("Correspondence needs geometry to score against; got None")

        # Only materialised when it has to be. A streamed pair sequence already
        # has a length, and listing it would load every dense map in the split
        # into memory at once — undoing the streaming that put it on disk.
        pairs = features if hasattr(features, "__len__") else list(features)
        geometries = labels if hasattr(labels, "__len__") else list(labels)
        if len(pairs) != len(geometries):
            raise ValueError(f"Got {len(pairs)} feature pairs for {len(geometries)} geometries")
        if not pairs:
            raise ValueError("Cannot evaluate correspondence on an empty set of pairs")

        errors = torch.cat(
            [
                self._pair_errors(pair[0], pair[1], geometry)
                for pair, geometry in zip(pairs, geometries, strict=True)
            ]
        )

        metrics: MetricsDict = dict(correspondence_recall(errors, self.thresholds, unit=self._unit))
        metrics.update(error_auc(errors, self.thresholds, unit=self._unit))
        metrics["num_matches"] = float(len(errors))
        return metrics

    def evaluate_ceiling(self, features: Any, labels: Any | None = None) -> MetricsDict:
        """Scores a *perfect* matcher would get, given patch quantisation.

        Matches can only land on patch centres, so a target point that falls
        between them cannot be hit exactly. With a 16x16 grid over a 224px
        image the spacing is 14px, which makes ``recall@1px`` almost
        unreachable no matter how good the features are.

        For each match the task actually kept, this computes the error of the
        *closest patch centre to the true target* — the best that match could
        possibly have scored. Report it beside :meth:`evaluate` or the raw
        numbers invite the wrong conclusion: a low ``recall@1px`` usually means
        the grid is coarse, not that the backbone failed.

        Crucially this is restricted to the **same matches** :meth:`evaluate`
        scores, not to every patch. Scoring all patches instead makes the two
        numbers averages over different populations, and the "ceiling" stops
        being a bound: the ratio test keeps distinctive patches, which are also
        the easier ones to localise, so a real run reported 127% of its own
        ceiling. Restricted this way, ``score <= ceiling`` holds by
        construction — the match chosen is one of the candidates the minimum
        is taken over.
        """
        if labels is None:
            raise ValueError("Correspondence needs geometry to score against; got None")

        errors = []
        # strict=True: a pair scored against another pair's geometry is the
        # misalignment that once put recall@1px at 0.003, and short labels would
        # silently score a prefix of the split instead.
        for pair, geometry in zip(features, labels, strict=True):
            source_idx, _ = self.match(pair[0], pair[1])
            if len(source_idx) == 0:
                continue

            size = geometry["size"]
            centres_0 = patch_centers(self._grid_hw(pair[0]), size)[source_idx]
            candidates = patch_centers(self._grid_hw(pair[1]), size)

            expected = apply_homography(geometry["homography"], centres_0)
            best = torch.cdist(expected, candidates).min(dim=1).values
            errors.append(self._scale(best, self._grid_hw(pair[1]), size))

        if not errors:
            return {
                **correspondence_recall(torch.zeros(0), self.thresholds, unit=self._unit),
                **error_auc(torch.zeros(0), self.thresholds, unit=self._unit),
            }

        pooled = torch.cat(errors)
        metrics: MetricsDict = dict(correspondence_recall(pooled, self.thresholds, unit=self._unit))
        metrics.update(error_auc(pooled, self.thresholds, unit=self._unit))
        return metrics

    def context_metrics(self, features: Any, labels: Any | None = None) -> MetricsDict:
        """:meth:`evaluate_ceiling`, prefixed ``ceiling_``, for the result record.

        CLAUDE.md's rule is that the ceiling is reported beside every score, and
        until now only ``examples/correspond.py`` obeyed it — a run through
        :func:`visbench.run` logged ``recall@1px`` with nothing saying that 0.015
        was the most it could have been.

        This re-runs the matching :meth:`evaluate` already did. That cost is
        real and is the price of the two numbers being separately callable; it
        is unchanged from what the example has always paid.
        """
        return {
            f"ceiling_{name}": value
            for name, value in self.evaluate_ceiling(features, labels).items()
        }

    def describe(self) -> dict:
        described = super().describe()
        described["task_params"] = {
            "num_corr": self.num_corr,
            "ratio_threshold": self.ratio_threshold,
            "thresholds": list(self.thresholds),
            "threshold_units": self.threshold_units,
        }
        return described

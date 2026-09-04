"""Relative depth ordering — **a control, not a probe. It did not earn a board.**

`depth` asks for metres. This asks only **which of two points is nearer**, so
every monotone transform of the prediction is the same answer and neither scale
nor shift is supervised or scored. It was built as a candidate probe, measured
against the board it subclasses, and **rejected on that measurement** -- the
third rejection in this project and the first for failing to *rank* rather than
for failing to be recoverable.

**Why it is not a probe**

Over the four backbones both readouts ran on the whole NYUv2 split:

===================  =========  ==========
backbone             ordinal    `depth` d1
===================  =========  ==========
``dinov2_vits14``      0.8407      0.7652
``mae_vitb16``         0.8400      0.6945
``clip_vitb16``        0.7757      0.6321
``resnet50``           0.7523      0.5395
===================  =========  ==========

**Spearman between the two readouts is +1.000.** Identical ordering, half the
spread (0.0884 against 0.2257), and a smallest adjacent gap of **0.0007** --
``dinov2_vits14`` against ``mae_vitb16``, which `depth` separates by 0.0624.
A probe that cannot separate two backbones the existing one separates by a
hundredfold is measuring less than what already ships. That is 6d-2's
occlusion-edge test applied to a readout instead of a target.

**What it is instead, and why that is worth keeping**

The rejection is itself a finding **about the published `depth` board**: a
readout that discards scale entirely reproduces the metric ranking exactly, so
that board is not ranking backbones by metric accuracy. It is ranking them by
ordering plus feature resolution. See ``results/controls/README.md`` and
``CORPUS_FINDINGS.md``.

So the class stays, unregistered, and its records live in
``results/controls/relative_depth.jsonl``. Unregistered is load-bearing: a
registered probe is pinned by test to have a corpus board, a CLI row, a
`TARGET_STYLES` entry and a committed gallery figure, and this has earned none
of the four. Construct it directly -- ``RelativeDepthTask()`` -- the way
``CustomBackbone`` is used.

**What the pre-measurement said, and the one thing it under-weighted**

Reproduced by ``scripts/premeasure_ordering.py``, on 120 NYUv2 test frames at
224px. It cleared the oracle gate comfortably and still should not have been
built, which is the transferable lesson:

===========  ==========  ============  ===========
min ratio    pairs kept  vertical      oracle@16
===========  ==========  ============  ===========
1.00 (none)      99.2%        65.2%        94.0%
1.25             49.4%        74.9%        99.3%
2.00             14.8%        83.7%        99.9%
===========  ==========  ============  ===========

**Ordering survives the patch bottleneck** -- 94.0% at 16x16, 89.5% at a
ResNet's 7x7 -- so unlike photometric superpixels the signal is not destroyed by
the bottleneck a dense probe reads through.

**But an image-coordinate shortcut scores 65.2%.** Indoor depth increases with
height in the frame, so "the lower point is nearer" beats chance by 15 points
with no features at all. That is why
:func:`~visbench.metrics.dense.ordinal_metrics` reports ``ordinal_vertical``
beside every score.

**A minimum depth-ratio threshold makes the task easier, not harder** -- it
raises the shortcut faster than the ceiling. The widest band is at no threshold,
which is the opposite of the intuition, so pairs are sampled unrestricted.

**The band is what was under-weighted.** 0.7056 to 0.8627 on the shipped runs is
**0.157 wide**, and the four backbones' own ceilings differ by 0.055 of that --
so a third of the available room is grid size before a representation is
consulted. `corner` ranks fine at a comparable 0.83 ceiling because its trivial
floor is near zero. **The oracle gate measures a candidate's ceiling and nothing
measured its floor**; that check now exists in
``visbench/tasks/low_level/README.md``'s gauntlet, added by this rejection.

The ranking loss and ordinal evaluation follow Chen et al., "Single-Image Depth
Perception in the Wild" (NeurIPS 2016, arXiv:1604.03901). Not their metric:
WHDR is computed on human-annotated DIW pairs weighted by annotator agreement,
neither of which NYUv2 carries, so the record claims
``visbench_relative_depth`` and never DIW's protocol.
"""

import torch

from visbench.metrics.dense import ordinal_metrics
from visbench.tasks.dense_base import DenseTrainingTask
from visbench.types import MetricsDict

__all__ = ["RelativeDepthTask", "relative_depth_loss"]


def relative_depth_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pairs_per_image: int = 2000,
    seed: int = 0,
) -> torch.Tensor:
    """Pairwise ranking loss over sampled point pairs, Chen et al.'s formulation.

    For an ordered pair where ``a`` is nearer than ``b``, the loss is
    ``log(1 + exp(score_a - score_b))`` -- zero when the score at ``a`` is far
    below the score at ``b``, growing linearly when the order is wrong. **Lower
    score means nearer**, and that convention is fixed here rather than chosen
    by the metric; :func:`~visbench.metrics.dense.ordinal_metrics` reads it back
    the same way.

    Pairs with equal ground-truth depth are dropped rather than given an
    equality term. Chen et al. include one because DIW's annotators can answer
    "about the same"; a rendered depth map's exact ties are quantisation, not a
    judgement, and supervising them would train the probe to flatten real
    structure.

    ``softplus`` rather than a hand-written ``log1p(exp(...))``: the second
    overflows to ``inf`` once a confident pair's margin passes about 88 in
    float32, and every later loss in the epoch is then ``nan`` while the run
    reports a plausible-looking 0.5 accuracy. That is the same failure mode as
    detection's unclamped distance ``exp``.
    """
    pred = pred.squeeze(1) if pred.ndim == 4 else pred
    target = target.squeeze(1) if target.ndim == 4 else target
    total = pred.new_zeros(())
    counted = 0

    for index in range(pred.shape[0]):
        gt = target[index]
        valid = torch.nonzero(gt > 0, as_tuple=False)
        if valid.shape[0] < 2:
            continue
        # Seeded per image so a re-run draws the same pairs; the generator is
        # on CPU because torch.randint with a generator requires the two to
        # agree on device, and the indices are used to gather either way.
        generator = torch.Generator().manual_seed(seed + index)
        pick = torch.randint(0, valid.shape[0], (pairs_per_image, 2), generator=generator)
        a, b = valid[pick[:, 0]], valid[pick[:, 1]]
        depth_a, depth_b = gt[a[:, 0], a[:, 1]], gt[b[:, 0], b[:, 1]]
        keep = depth_a != depth_b
        if not bool(keep.any()):
            continue

        score_a = pred[index][a[:, 0], a[:, 1]]
        score_b = pred[index][b[:, 0], b[:, 1]]
        # Orient every kept pair so the first element is the nearer one, then
        # one term covers both directions.
        nearer_first = depth_a < depth_b
        margin = torch.where(nearer_first, score_a - score_b, score_b - score_a)
        total = total + torch.nn.functional.softplus(margin[keep]).sum()
        counted += int(keep.sum())

    if counted == 0:
        return pred.new_zeros(()) * pred.sum()
    return total / counted


class RelativeDepthTask(DenseTrainingTask):
    """Which of two points is nearer, from one image, with no notion of scale."""

    level = "mid_level"
    display_name = "Relative depth ordering"
    target_noun = "target depth maps"
    target_channels = 1
    protocol = "visbench_relative_depth"

    def __init__(
        self,
        head: str = "linear",
        layers: list[int] | None = None,
        hidden_dim: int = 512,
        epochs: int = 10,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 8,
        warmup_epochs: float = 1.5,
        pairs_per_image: int = 2000,
        pair_seed: int = 0,
        head_kwargs: dict | None = None,
        device: str | None = None,
        finetune_blocks: int = 0,
        backbone_lr: float | None = None,
    ) -> None:
        """Configure the probe; the head is built lazily in :meth:`fit`.

        Parameters
        ----------
        head, layers, epochs, lr, warmup_epochs:
            See :class:`~visbench.tasks.dense_base.DenseTrainingTask`. The
            schedule is probe3d's, unchanged, because it is the one every other
            dense probe here uses and a different one would make this board
            incomparable with them for a reason unrelated to the protocol.
        pairs_per_image:
            Point pairs drawn per image, for both the loss and the metric.
        pair_seed:
            Base seed for pair sampling. Offset by the image's index within the
            batch, so an image is always scored on the same pairs.
        """
        super().__init__(
            head=head,
            layers=layers,
            hidden_dim=hidden_dim,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            warmup_epochs=warmup_epochs,
            head_kwargs=head_kwargs,
            device=device,
            finetune_blocks=finetune_blocks,
            backbone_lr=backbone_lr,
        )
        self.name = "relative_depth"
        if pairs_per_image < 1:
            raise ValueError(f"pairs_per_image must be >= 1, got {pairs_per_image}")
        self.pairs_per_image = pairs_per_image
        self.pair_seed = pair_seed

    @property
    def out_channels(self) -> int:
        """One unitless score per pixel. Not metres, and not bins.

        `depth` needs 256 bins because a *metric* linear probe otherwise
        collapses to the dataset mean. A ranking loss has no such pull: it only
        ever sees differences, so the constant that a mean-predictor would
        learn cancels and one channel is enough.
        """
        return 1

    def _activate(self, raw: torch.Tensor) -> torch.Tensor:
        """The identity, and it must stay so.

        Any monotone map of the score is the same answer, so an activation adds
        nothing -- and the two obvious ones actively hurt. A sigmoid saturates,
        which flattens the gradient for exactly the confident pairs the loss
        wants to keep separating; a ReLU makes every negative score equal, so
        every pair among them ties and the ordering is destroyed rather than
        merely compressed. `edge`'s ``_activate`` is the identity for the
        parallel reason and a test pins it.
        """
        return raw

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return relative_depth_loss(
            pred, target, pairs_per_image=self.pairs_per_image, seed=self.pair_seed
        )

    def _batch_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> MetricsDict:
        """``{"ordinal_accuracy", "ordinal_vertical"}``, per image.

        ``ordinal_vertical`` is a **diagnostic, never a score**: it is what an
        image-coordinate prior scores on the same pairs, and it says nothing
        about a backbone. It is here because 65.2% of it comes free on NYUv2.
        """
        return ordinal_metrics(
            pred, target, pairs_per_image=self.pairs_per_image, seed=self.pair_seed
        )

    def oracle_prediction(self, targets: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        """The patch-mean depth map, which orders correctly wherever it survives.

        Opting in is right here for the reason `depth` does **not** opt in: this
        probe's ``_activate`` is the identity, so the pooled-and-upsampled
        target is directly a prediction on the same scale the metric reads.
        `depth`'s activation is a bin expectation, whose pooled value is a
        different quantity from the one its head emits.

        Measured before the probe was written: **94.0%** on a 16x16 grid, 89.5%
        on a ResNet's 7x7. Ranking those two against each other without the
        ceiling beside them invites attributing 4.5 points of grid to a
        representation.
        """
        return self._averaging_oracle(targets, grid_hw)

    def context_metrics(self, features: object, labels: object | None = None) -> MetricsDict:
        """The accuracy's ceiling only -- a diagnostic has no ceiling.

        The base implementation prefixes ``ceiling_`` onto **every** key the
        oracle's metrics returned, which is right when they are all scores. Here
        one of them is not: ``ordinal_vertical`` is the image-coordinate
        baseline, and it does not depend on the prediction at all, so
        ``ceiling_ordinal_vertical`` came out bit-identical to
        ``ordinal_vertical`` -- a second copy of one number under a name
        claiming to bound it. Caught by reading a run's output, not by a test,
        which is the argument for proving a probe end to end on real data.
        """
        return {
            name: value
            for name, value in super().context_metrics(features, labels).items()
            if name != "ceiling_ordinal_vertical"
        }

    def _task_params(self) -> dict:
        """``protocol`` is this library's, not probe3d's and not DIW's.

        probe3d has no ordinal task, and DIW's WHDR needs human-annotated pairs
        weighted by agreement -- neither of which NYUv2 carries. Only the
        optimiser schedule is borrowed, and that is recorded under ``optimizer``
        already.
        """
        return {
            "protocol": self.protocol,
            "loss": "pairwise_ranking",
            "activation": "identity",
            "pairs_per_image": self.pairs_per_image,
            "pair_seed": self.pair_seed,
        }

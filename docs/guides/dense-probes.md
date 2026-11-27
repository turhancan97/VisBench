# Dense probes

Nine of the twenty probes predict something per *pixel*. They share almost
everything — feature sources, batching, head construction, the optimiser
schedule, the training loop, per-image metric averaging — and a new one supplies
four methods.

## Subclass `DenseTrainingTask`

```python
class MyTask(DenseTrainingTask):
    level, name = "low_level", "my_probe"
    display_name, target_noun, target_channels = "My probe", "targets", 1

    @property
    def out_channels(self) -> int: ...       # channels the head emits
    def _activate(self, raw): ...            # raw output -> prediction
    def _loss(self, pred, target): ...       # both (B, C, H, W)
    def _batch_metrics(self, pred, target):  # PER-IMAGE averages
        ...
```

**`_batch_metrics` must return per-image averages.** That is what lets
`evaluate` weight each batch by its size and recover the whole-split number.
And per-image rather than pooled over the split, because pooling every pixel
lets uneven hole coverage silently reweight the dataset.

**`_activate` is applied in the loss, the metrics *and* `predict`**, so those
three can never disagree about what the head's output means.

Read `DepthTask` (224 lines), `SurfaceNormalTask` (299),
`GenericSegmentationTask` (173) and `SemanticSegmentationTask` (186) before
writing a fifth. Between them they show a scalar target and a vector one; a
bin-expectation activation, a normalising one and a sigmoid; a protocol borrowed
wholesale from probe3d and one that borrows only its schedule.

## Heads

```bash
visbench run depth --data ... --head dpt --layers 2 5 8 11
```

`LinearHead` is a 1x1 convolution per patch and a bilinear upsample.
`DPTHead` fuses several backbone depths top-down and refuses a single feature
map rather than duplicating it — a DPT fed one layer is not multiscale, and
reporting its score as a DPT number would misdescribe the architecture.

**Report the linear number when comparing representations.** A deeper head can
compensate for a weak feature map and narrow the very gap a probe exists to
measure. Measured, not asserted: across five probes and nine ViTs a DPT head
reorders **24 of 174 separable pairs** and changes the leader on two of five
boards.

### `--standardize`: take feature scale out of the fit

```bash
visbench run depth --data ... --standardize
```

The heads are fitted with AdamW at a fixed learning rate, which is not
scale-invariant, and backbones' features differ in scale by almost 10x. A
control showed what that costs: multiplying two columns' features by a constant
— which removes nothing a linear head could use — **reversed their order** on
three boards. `--standardize` (`standardize=True` in Python) maps every feature
channel to mean 0 and standard deviation 1 before the head, with statistics
computed once on the **training** split and reused for evaluation. With it on, a
constant rescaling of the features cannot move the score.

It is **off by default**, and every published dense board was measured with it
off. A run with it on records `standardize: true` and a `+std` suffix on its
`protocol` (`probe3d+std`), so it never shares a ranking with an unstandardised
number. The statistics travel inside a saved probe, and it is refused together
with `--finetune-blocks`, since fine-tuning moves the features they describe.

### `--target-resample`: where a target lands

Every dense dataset resizes and centre-crops the image and its target together.
Since #173 the target is resampled with PIL's nearest-neighbour, the same pixel
geometry as the image's resize, so it lands exactly on the image.
`--target-resample torch_nearest` restores the old left-aligned geometry, which
every dense number published before #173 used; it reproduces their dataset
fingerprints exactly. The two geometries get different fingerprints, so they
never share a board.

## Features stream

Dense features are ~250x the size of pooled ones — 24k NYUv2 images at DINOv2-B
is about 19 GB. `run()` streams automatically for a probe that declares
`uses_dense`: 10.8 GB peak RSS in memory against 1.7 GB streaming, for 0.63 GB
of features.

`CachedFeatures` is **random-access, not a generator**, because training
reshuffles every epoch and a generator can only shuffle *within* a batch.

## Before you build one: the gauntlet

A target that is distinctive is not necessarily a target a probe can rank
backbones with, and this project has rejected three candidates on that
distinction. In cost order, all of them cheaper than a per-backbone board:

1. **Run the oracle gate.** `scripts/oracle_ceiling.py` asks what the probe
   could score *if the features contained the answer*. A dense probe sees one
   feature vector per patch, so signal finer than a patch is absent from its
   input rather than merely hard to predict. The four shipped magnitude targets
   score 0.53-0.83 at a 16x16 grid; photometric superpixels scored **0.25**, and
   was built anyway because this check did not yet exist.
2. **Measure the floor, not only the ceiling.** Name the cheapest shortcut a
   head could learn — an image coordinate, a per-image constant, the dataset
   mean — and measure it on the samples the metric will use. Relative depth
   ordering cleared the gate at a **94.0%** oracle and was rejected anyway,
   because "the lower point in the image is nearer" scores **65.2%** with no
   features at all. **A ceiling of 0.9 above a floor of 0.7 is a worse probe
   than a ceiling of 0.6 above a floor of 0.**
3. **Check the tail.** A target with too much mass in its strongest 1% of pixels
   scores badly and ranks nothing — `edge_occlusion` at 46% is the case where
   L1 and Pearson pull apart.
4. **Check the overlap with what already ships.** DoG-blob detection was
   rejected without a probe run: its target correlated **0.51** with `corner`,
   as redundant with an existing probe as `corner` is with `edge`.
5. **A correlated target can still rank differently, and that is the
   criterion.** `corner` correlates 0.52 with `edge` and earned its place
   because the two nonetheless order backbones differently. Ask about the spread
   over the full set, never about one pair.

The write-ups are in
[`visbench/tasks/low_level/README.md`](https://github.com/turhancan97/VisBench/blob/main/visbench/tasks/low_level/README.md).

# Reading a board

Every board on this site is generated from
[`results/corpus/visbench.jsonl`](https://github.com/turhancan97/VisBench/blob/main/results/corpus/visbench.jsonl), and a
fast test fails if any of them drifts from the records. So the numbers are
right. What follows is about **what they mean**, which is a different question
and the one this project has got wrong more often.

The evidence for each claim below, and the earlier published reading each one
corrected, is in
[`CORPUS_FINDINGS.md`](https://github.com/turhancan97/VisBench/blob/main/CORPUS_FINDINGS.md).
`scripts/analyse_board_correlates.py` reproduces the correlational ones.

## "Which backbone is best" is not a well-formed question

`mae_vitb16` is first on **five** of the twenty boards and last on **five**. A
summary that picks a winner is discarding the result — the whole point of
nineteen boards is that a representation is good *at things*, not good.

## A count is a fact about the corpus, not about a backbone

Three of MAE's counts have moved without its features changing: twice because a
backbone column was added, once because a whole *board* was. **Re-read a count
off [`LEADERBOARD.md`](https://github.com/turhancan97/VisBench/blob/main/LEADERBOARD.md)** rather than out of prose —
including the paragraph above this one. Every count that has gone stale in this
project went stale exactly that way: carried forward through a release that
added something.

## A ceiling travels beside a score, and must never be ranked on

`ceiling_*` says what the feature grid made **available**, not what the
backbone recovered. A match can only land on a patch centre, so a 7x7 grid
cannot place one within 5px more than ~10% of the time whatever its features
are, against ~41% on a 16x16 grid.

Because a ceiling *falls* with the grid, ranking on it would rank feature
resolution directly. Same for the training diagnostics: a probe that fits its
training data perfectly has said nothing yet about a backbone — on CUB every
backbone reaches `train_top1` 1.0000, including the one that comes last.

## What the training diagnostics *do* answer

A low score has two opposite readings, and `training` is what separates them. An
unconverged head **understates** a backbone; a converged head that still scores
badly says the representation does not carry the answer. `GenericSegmentationTask`
on 80 training images reads 0.16 IoU at the defaults and 0.87 at `epochs=40` on
*identical* features — the same number, and opposite conclusions.

So before reading a low cell as a fact about a backbone, read its `train_loss`
(and `train_top1`, where the probe reports one).
[`scripts/analyse_training_diagnostics.py`](https://github.com/turhancan97/VisBench/blob/main/scripts/analyse_training_diagnostics.py)
does that for every board at once. Two things it will tell you that are easy to
misread on your own: a board can be **saturated**, where every head fits its
training data perfectly and the whole spread is generalisation; and on a dense
board the highest training loss tends to belong to the **coarsest feature
grid**, which is the grid rather than a failure to converge.

`train_loss` is comparable *within* a board and meaningless across boards — each
is a different loss on a different target.

## Quote an objective gap against the *recipe* gap on the same board

`sam_vitb16` and `supervised_vitb16` share architecture, data, labels and
input normalisation, and differ **only** in training recipe. On seven of the
boards that gap is more than a third of the whole objective spread. So a gap
between two training objectives means nothing until you know how large a gap two
runs of the *same* objective produce — and nothing in a record tells you which
board you are on.

## Feature resolution is the strongest correlate of every dense board

And it is *not* what DINOv2's lead is made of, which is the part that took a
control to establish. Holding weights fixed and cutting DINOv2-B from 256 to 196
tokens costs under 3% on all five dense boards, and it keeps its lead over the
whole ViT-B/16 pack on both boards it led.

**Check who leads a board before explaining their lead.** On the other three
boards DINOv2-B never led, so there was nothing to explain — a first reading of
the correlation table treated all five as confounded.

### But never read a linear board as evidence that resolution does not help

That control could only *lower* a grid. Raising one — `dino_vitb8` against
`dino_vitb16`, same objective, data, width and depth, 784 tokens against 196 —
moves the published boards on this site by a rounding error or the wrong way,
and moves the same five probes **with a DPT head** by one to two orders of
magnitude more, improving 5 of 5. The ceilings say why: they rise every time,
and the share a linear head recovers of them falls every time.

The effect is not confined to that pair. Across all ten ViTs in the corpus
(49 to 784 tokens), the correlation between token count and score is
**stronger under a DPT head on 4 of 5 low-level boards** — mean +0.388 against
+0.701, and `keypoints2d` +0.096 against +0.775 — and it is largest exactly
where the linear head recovers least of its own oracle.

So feature resolution is causal about what a representation **carries** and
close to invisible in what these boards **report**. The boards are linear on
purpose (see above); the price is that this is one thing they cannot show you.

## Two boards are close to in-distribution recall, not transfer

`convnext_base` and `supervised_vitb16` are ImageNet-1k supervised, and
Imagenette's classes are ImageNet-1k wnids. Their high-level numbers are
measuring memory as much as transfer.

## The high-level tier is two clusters, not one

`classification` and `retrieval` (image-level categorisation) barely
correlate with `detection`, `semantic_segmentation`,
`scene_classification` and `fine_grained_classification` (localised or
spatial-context prediction).

The replication is what makes this a property rather than an accident of one
dataset: **two probes that are mechanically object classification with a
different folder both land in the localised cluster.**
`fine_grained_classification`'s strongest partner in the whole corpus is
`detection` at **+0.860**, against +0.343 with the object board it subclasses.

Treat `high_level` as a folder, not a quantity to average over. Mid- and
low-level cohere.

## Two decimals that are not there

- **Quote `detection` to three decimals, not four.** GPU non-determinism a
  discrete metric can see, only on the two 16x16-grid backbones. Nothing to fix.
- **Quote a DPT number to three decimals.** A DPT head reproduces to ~3e-3
  where a linear head manages ~1e-7.
- **`orientation`'s two middle rows are not separable.** Its metric is a
  coherence-weighted `acos`, which is ill-conditioned at its endpoints, and it
  drifts 1.8e-03 where the other four boards drift 1e-7.
- **Quote `relative_pose` to whole degrees, and treat rows closer than about
  one degree as ties.** This is the largest of the four by far, and it is not
  the metric — it is the fit. Its 1,536-dimensional MLP lands in a different
  place for any disturbance at all: a second extraction of the same frames
  moves it about a degree, and so does simply changing the seed. Against one
  cache at one seed it reproduces bit for bit, and every recorded field — seed,
  fingerprint, `hardware` — is identical across the two runs that differ, so
  there is no field to check instead of counting digits.

  **Do not order two adjacent rows from the gap alone.** All thirteen were
  re-fitted at five seeds, and a gap threshold turns out to be the wrong
  instrument: pairs 1.6 degrees apart are solidly ordered, while two pairs the
  board shows as near-level come out *the other way round* across seeds —
  `dinov2_vitb14` ahead of `dino_vitb16` by 0.80, and `clip_vitb32` ahead of
  `resnet18` by 2.08, where the board has both losing by about a tenth. The
  board is a correct record of one seed; `results/controls/pose_seeds.jsonl`
  says which pairs survive a re-fit.

## n = 13

Every correlation above has wide error bars, and coefficients quoted from
before `dino_vitb8` were measured at n=12.

## What may sit in one table at all

`comparability_key` decides, and it is stricter than "same probe": task, level,
`protocol`, dataset and fingerprint, split, requested pooling, feature mode and
resolved layers must all match. Two consequences:

- **A second dataset under an existing probe name does not merge into its
  board — it makes that board unrenderable.** `board_for` refuses a task with
  more than one comparability group.
- **`results/controls/` is rankable and still excluded.** Those records pass
  `comparability_key` against the boards they explain and answer a *different*
  question: the corpus says what a backbone scores, a control says what changes
  when one thing about one backbone moves.

Never rank or average across `finetune` either. Frozen asks what a
representation already carries; fine-tuned asks what it can be adapted into.
Every published VisBench number is frozen.

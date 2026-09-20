# Roadmap, build order and future directions

This is the project's own record of how it was built and where it might go. It
is here rather than in the README because it answers "what is the plan", not
"how do I use this" — see the [documentation home](index.md) for the latter.

## Build order

This is a multi-month roadmap, built one reviewed step at a time.

- [x] **1. Scaffold** — every folder and module, docstrings and stubs, no logic
- [x] **2.** `BaseBackbone` + feature cache + DINOv2, with tests
- [x] **3.** `BaseTask` + one task end-to-end on a local image folder
- [x] **4.** Next task, then next backbone — all three v0.1 tasks, both v0.1
      backbones, `uv.lock`, and the `run()` entry point
- [x] **5a.** ResNet/timm backbone — the first non-ViT, validating the CNN half
      of `BaseBackbone`
- [x] **5b.** custom `nn.Module` backbones, and pluggable heads (linear + DPT)
- [x] **5c.** multi-layer extraction — `layers=[...]` through every backbone
      and the cache, so the DPT head has something real to consume
- [x] **5d.** depth estimation — the first dense task, end to end on probe3d's
      protocol: dense dataset, metrics, loss, pluggable head
- [x] **5e.** streaming features from disk, so a dense task can run a dataset
      larger than memory
- [x] **5f.** surface normals — probe3d's angular protocol, reusing the dense
      dataset, the streaming path and the shared `DenseTrainingTask`
- [x] **5g.** generic (binary) segmentation — the first dense task whose
      protocol is not probe3d's, and the first target where 0 is a label rather
      than a hole
- [x] **5h.** semantic (multi-class) segmentation — the high-level counterpart
      to 5g, on the same base class, with a class-index target and mIoU under
      both reductions
- [x] **5i.** mid-level image similarity — zero-shot 2AFC against human
      judgement, deliberately distinct from high-level retrieval
- [x] **5j.** the CLI, a thin wrapper over `visbench.run()` — which also
      taught `run()` to cover correspondence, the one task it had never been
      able to express
- [x] **6a.** opt-in fine-tuning — unfreeze the last N backbone blocks, with the
      feature cache out of the path and the result record saying which a number
      came from
- [x] **6b.** cache the *frozen prefix*, so a fine-tuned run recomputes only the
      blocks it is training
- [x] **6c.** detection, in three parts and in this order: the box dataset, then
      the VOC metric, then the head — so the head is judged by a scorer that was
      already cross-checked against `VOCevaldet.m`
- [x] **6d.** the first low-level task — edge detection on Taskonomy, filling a
      folder that had been a documented placeholder since v0.1
- [x] **6d-2.** `mask_valid/`, so the reconstruction-derived Taskonomy domains
      can be read at all; 2D keypoints and occlusion edges on the lifted
      `DenseMagnitudeTask`
- [x] **6e.** the leaderboard and probe sharing, in five parts: the
      comparability rules as pure functions, a committed record corpus covering
      every probe against every backbone, the renderer that generates the
      published tables from it, a serialised probe artifact carrying the
      backbone identity beside the weights, and the Hub transport behind a
      `[hub]` extra
- [x] **6f.** correspondence scored in pixels rather than patch widths — a unit
      change that inverted the published board
- [x] **7a–7e.** the contributor-facing surface, shipping no new number:
      `visbench demo`, the README reorganised around a reader with `docs/` split
      out of it, `CONTRIBUTING.md` and the issue/PR templates, the Sphinx site
      and its workflow, and citation metadata with a DOI
- [x] **8a–8b.** corner detection — the first probe whose target is *computed*
      from the image rather than downloaded, then put into the record corpus on
      a frame set a script pins and reconstructs
- [x] **9a–9d.** `visbench show` — the panel viewer, then the correspondence
      pair renderer, then the three probes whose answer is a choice among images
      rather than a map, then the generated docs gallery. Every probe is
      drawable, and `show_probes() == list_probes()` is asserted
- [x] **10a.** three more backbones — `TimmBackbone` learns to read a ViT, which
      added ConvNeXt-B, MAE ViT-B/16 and SigLIP-GAP ViT-B/16 in one change
- [x] **10b.** the corpus at 13 probes x 9 backbones — 117 records, and the
      first time the three tiers visibly separate: at nine backbones MAE was
      first on six boards and last on four (five and three at twelve — a count
      over a corpus is a fact about that corpus)
- [x] **10c.** a supervised ViT-B/16 — the same architecture and the same
      pretraining set as MAE, differing only in objective, so the corpus gets
      its first controlled experiment: 130 records, and supervised takes every
      high-level board while MAE takes every low-level one. The tidy version of
      that claim — "the winner changes exactly at the tier boundary" — is wrong,
      because mid-level similarity crosses it; count tiers from `record.level`,
      not from which boards feel semantic
- [x] **11a.** the documentation gallery drawn on real photographs — Open
      Images frames under a per-image licence check, rather than generated
      scenes
- [x] **10d.** DINO ViT-B/16 — a third value of the objective variable, which
      answers what the supervised/MAE pair could not: high-level structure comes
      from a semantic training signal, not from labels
- [x] **10e.** a recipe control — the same objective trained two ways, which
      supplies the denominator every objective claim needs, and refutes 10d's
      semantic-segmentation evidence on arrival
- [x] **backlog: `scene_classification`** — scene category on the object
      `classification` linear-probe path (Places365), a new probe *name* rather
      than a dataset flag; ranks backbones almost independently of the object
      board (Spearman +0.15)
- [x] **backlog: `orientation`** — gradient orientation, the fourth low-level
      probe and the second derived from the frame, but the first whose target
      is a direction; DoG-blob was rejected first for overlapping 0.51 with
      `corner`
- [x] **backlog: `fine_grained_classification`** — subordinate category on the
      same linear-probe path (CUB-200-2011), the third distinct question on that
      one implementation; a new probe *name* rather than a dataset flag, for the
      reason `scene_classification` is. Its twelve-backbone board landed the same
      day and *replicates* `scene_classification`'s surprise: it correlates
      +0.832 with `detection` and only +0.322 with the object board it
      subclasses
- [x] **backlog: the oracle gate** — the derived-target gauntlet asks whether a
      candidate target is *distinctive*; it never asked whether it is
      *recoverable* from patch features. `evaluate_oracle` asks that with no
      backbone and no fitted head, calibrated so the four shipped magnitude
      targets pass at 0.53-0.83 and the rejected superpixel candidate fails at
      0.25
- [x] **12a-1, 12a-2: BSDS500** — the dataset with every annotator's boundary
      map, and an ODS/OIS/AP implementation written from the paper that
      reproduces the published human agreement (0.8030 against 0.80). **The
      probe was refused by the oracle gate** and the line is closed at two
      steps; see the note under "Already partly answered"
- [x] **the corpus carries its ceilings** — the five low-level boards re-run so
      every record emits `ceiling_*` beside its score and the schema-v8
      `training` block, 60 records, each value produced by a run rather than
      backfilled. Four boards reproduced to ~1e-7 relative; `orientation` did
      not, and its metric is recorded as ill-conditioned
- [x] **relative depth ordering, built and rejected** — the third rejection and
      the first for failing to *rank* rather than to be recoverable. It also
      added the gauntlet check that was missing: the oracle gate measures a
      candidate's **ceiling** and nothing measured its **floor**, and an
      image-coordinate shortcut took 65.2% of this one's 0.157-wide band before
      a representation was consulted. `scripts/premeasure_ordering.py` is the
      worked example
- [x] **the oracle gate is a bar, not a bound** — `results/controls/dpt_head.jsonl`
      measures what a DPT head reaches against the linear oracle the gate
      computes. The gate models a linear head exactly, so it is a bar for the
      head VisBench reports rather than a bound on what is achievable
- [x] **the corpus carries its fit diagnostics** — the eight trained boards
      that predated schema v8 re-run so every trained record says how its fit
      went, 96 cells. It doubled as a reproducibility audit and the corpus
      passed: **93 of 96 cells reproduced the value they were re-running**, and
      the boards that appeared not to had been produced by a node that was
      failing while reporting success — which the new field is what caught,
      since every bad cell carried a worse fit beside its worse score. It also
      closes the split control's one open question (the published `detection`
      board does underfit relative to the full split, 12/12) and refutes a
      published claim of its own: the detection board's two CLIP rows are a
      tie, not a verified ordering
- [x] **the DPT control, widened to the whole corpus** — nine ViTs in
      `dpt_head.jsonl` and three CNNs in `dpt_head_cnn.jsonl`, two
      comparability groups answering two questions. The gate bounds **eight of
      nine ViTs**: only `mae_vitb16` exceeds it, twice in 45 cells, so the
      exception is backbone-specific rather than a property of decoders. Two
      ViT boards change leader and 24 of 174 separable pairs reorder; on the
      CNNs, where a DPT run changes the bottleneck as well as the head, three
      of five boards change leader and two invert outright. It still does not
      reopen BSDS500
- [x] **13a.** the documentation site restructured — four long pages become
      guides, one page per probe, and the API reference `conf.py` had been
      configured for since v0.7.0. ~5,198 lines of docstring went through
      docutils for the first time and nine source files had real defects, so
      `scripts/check_docstrings.py` now renders every one in the fast suite: a
      docstring convention nothing renders is a guess, not a convention
- [x] **14a-1 to 14a-4.** `instance_segmentation`, the **seventeenth probe** —
      which *object* a pixel belongs to, over the same 1,449 VOC images
      `semantic_segmentation` scores. `InstanceHead` is a `DetectionHead` plus
      **one 1x1 convolution**, so the only learned thing between features and
      mask is that convolution; mask AP is the detection protocol with the
      overlap swapped, one matcher rather than two. Seventeen probes against
      twelve backbones, **204 board cells**
- [x] **the split control** — the new board ranks with the *mid-level geometry*
      boards rather than its own tier, and this traced that to the **split**
      rather than the probe: run `detection` on the same 1464/1449 images and
      it changes cluster too (`occlusion_edge` +0.483 to **+0.965**). So a
      cluster is a property of a board **as configured**, and the standing rule
      is now never to quote one as a property of a *task*. No published number
      moves — the reading does
- [x] **schema v9: a record says what it ran on** — every other field describes
      the experiment and none described the machine, which cost real work twice
      in two days. `hardware` is recorded and **never grouped**: keying it would
      give every GPU its own group and split the corpus by age as well as by
      machine, while the measured answer is that such cells reproduce to six
      decimals
- [x] **the grid claim, measured** — the correlation between feature resolution
      and a board's *fit* was asserted from two boards and is now measured over
      all eleven that read a grid (11 of 11, mean rho −0.681). Fixing the tie
      handling it exposed corrected three published coefficients, and every
      published board-*pair* number survived unchanged
- [x] **16a-1.** NAVI and the pose geometry — the dataset and metric halves of
      relative camera pose, before any head, which is the order detection was
      built in. The pair set is the protocol, so the shipped class was checked
      against the pre-measurement rather than trusted: 1,740 / 6,477 / 50,519
      pairs and a no-feature floor of 66.94 and 66.85 degrees, all exact. It
      also corrected the pre-measurement — 327 of NAVI's 8,217 frames carry a
      half-turn EXIF tag whose camera pose describes the *turned* image, so
      reading them as stored supervises 4.0% of the release against its own
      negation
- [x] **16a-2.** the pose head and the probe that fits it — `PoseHead` and
      `RelativePoseTask`, **the first board here whose head is not a linear map
      of the features** — `DPTHead` is nonlinear but is a control, and every
      head a published board uses is an affine layer or a 1x1 convolution. The departure is measured rather than assumed: a single affine
      map underfits this task at 40x the MLP's training loss and produces an
      ordering that nearly inverts it, so `hidden_dims=()` stays reachable as
      the control and the record says which head produced a number. The
      no-feature floor travels with the score as `floor_*`, the training pair
      count is in `task_params` because the score has not converged in it, and
      the fitted floor is in `probe_state()` — the one piece of state that
      does not change a prediction, and so the easiest to lose. Proved
      end to end on NAVI against the parked pre-measurement — `mae_vitb16`
      21.84 deg and `clip_vitb16` 40.50 against a no-feature floor of **66.85,
      the parked value to the digit**, which is what says the shipped dataset
      drew the same pairs
- [x] **16a-3.** the eighteenth probe registered, with its thirteen-backbone
      board, its `visbench show` renderer, its gallery figure and its page.
      Registration turned **16 tests red at once**, each naming a table that has
      to agree with `list_probes()` — which is the argument for that guard:
      none of them can be found by reading the code that adds the probe. The
      board is the first here close to *orthogonal* to the high-level tier
      (mean rho +0.099 against it, +0.701 against low-level), and it must be
      read to whole degrees: two extractions of the same features differ by
      ~1e-5 and thirty epochs of this head turn that into about a degree
- [x] **19a.** `scene_parsing` — NYUv2-40 as the **nineteenth** probe, the same
      implementation `semantic_segmentation` runs asking a different question:
      forty classes of *stuff* in a room against twenty-one objects on a
      background. The label convention was measured rather than assumed (255 is
      void, 0 is `wall`), so it needed no loader and no fifth validity rule. Its
      board ranks with the **semantic** boards (+0.553 mean against high-level,
      +0.136 against low-level) *including* against the two boards that read its
      identical frames — the question dominating the data, which complements the
      split control

- [x] **21a.** `vehicle_classification`, the **twentieth** probe — which car
      model, not which car. The fourth question on one linear-probe
      implementation, measured against the relative-depth standard before it was
      built (+0.863 against CUB, where a dozen existing pairs are more
      correlated) and **separating granularity from what this family actually
      shares**: its closest partner is the *place* board at +0.901, not the other
      subordinate one. `siglip_vitb16` leads where it is fifth on CUB. The
      Stanford Cars copy here is not the official split — eleven train images are
      the same photograph under two labels — so the board runs a pinned, cleaned
      8,125/8,026 split and says it is not comparable with published Cars numbers

- [x] **20d.** why one board would not reproduce — `scene_classification`'s
      held-out sweep explained by **amplification** rather than by hardware:
      across a thousand-fold perturbation range its most interpolating row moves
      by the same ~0.005 at every size while its most stable row moves 150x
      less, so a disturbance too small to identify suffices. The board gets a
      verdict anyway (8 of 12 pairs ordered), and the two configurations differ
      on only 2 of 78 pairs, both already tied

- [x] **20c.** every trained board swept — twelve more at five seeds, which
      **corrects 20b's own conclusion**: `surface_normal`, a linear board,
      reverses a pair, so reversals are rare and marginal rather than a property
      of the pose head. 119 of 180 adjacent pairs ordered, 58 tied, 3 reversed,
      and on 10 of 15 boards no gap threshold could sort the pairs.
      `scene_classification` is held out, its published cells not reproducing on
      this silicon with a different fit — the case schema v8 exists to make
      legible. No published number moves
- [x] **20b.** the seed sweep generalised — `relative_pose` was the only board
      ever re-fitted at several seeds, and the only one whose head is not a
      linear map, so its two reversed pairs might have been a property of that
      head. Three boards unlike it were swept: `classification`, `corner` and
      `detection`, thirteen backbones at five seeds each. **The reversals do not
      generalise** — none of the three has one — but **twenty of the
      forty-eight adjacent pairs across the four swept boards cannot be
      ordered**, and on every new board the largest unordered gap exceeds the
      smallest ordered one, so no gap threshold can sort them. No published
      number moves

## Roadmap

**v0.1** — prove the abstraction. DINOv2 + CLIP. Zero-shot or linear-probe-on-cached-features only; no fine-tuning, no dense training loops. Deferred: CLI, custom backbones, ResNet/timm, multi-layer extraction.

**v0.2** — ResNet/timm + custom backbones *(done)*, pluggable heads (linear + DPT) *(done)*, multi-layer extraction *(done)*, depth estimation *(done)*, surface normals *(done)*, generic (binary) segmentation *(done)*, semantic segmentation *(done)*, mid-level similarity *(done)*, CLI *(done)*.

**v0.3** — opt-in fine-tuning of the last N blocks *(done)*, prefix caching *(done)*, detection *(done)*.

**v0.4** — edge detection, the first low-level task *(done)*.

**v0.5** — Taskonomy's `mask_valid/` and the four domains it unblocks *(done)*, 2D keypoint detection *(done)*, occlusion-edge detection *(done)*.

**v0.6** — the leaderboard: comparability rules, a committed record corpus, and
generated tables *(done)*; HF Hub probe sharing *(done)*. **v0.6.1** corrects
the correspondence board it shipped, which was ranked upside down by a
backbone-dependent threshold unit.

**v0.7** — the contributor-facing surface: `visbench demo`, the reorganised
README, `CONTRIBUTING.md`, the documentation site, and a citable DOI *(done)*.
Every measurement v0.6.1 reported, v0.7.0 reports identically.

**v0.8** — corner detection, the first probe whose target is computed from the
image rather than downloaded *(done)*.

**v0.9** — `visbench show`: every probe drawable, four renderers, and a
generated docs gallery *(done)*. It adds no probe and moves no number.

**v0.10** — four more backbones *(done)*: ConvNeXt-B, MAE ViT-B/16 and
SigLIP-GAP ViT-B/16 through a `TimmBackbone` that now reads a ViT's own
structure, then a supervised ViT-B/16 that turns the corpus into a controlled
experiment. Thirteen probes against ten backbones, 130 records. The gallery
moves to real photographs in the same release.

**v0.11** — two more backbones *(done)*, and with them the corpus gets its
controls: DINO ViT-B/16 completes an objective family three wide on one
architecture and one pretraining set, and a SAM-trained ViT-B/16 varies only
the *recipe*, supplying the denominator an objective gap has to be quoted
against. Thirteen probes against twelve backbones, 156 records. The control
refuted half of its own family's published claim on arrival. It also ships
`examples/custom_backbone.py`, the escape hatch that had been documented and
never demonstrated.

**v0.12** — two more probes *(done)*: `scene_classification` on Places365 and
`orientation`, the first derived target that is a *direction* rather than a
magnitude and so the first that could not reuse `DenseMagnitudeTask`. Fifteen
probes against twelve backbones, 180 records. It also closes the
library-surface backlog with the `torchvision` / Hugging Face dataset bridges.

**v0.13** — a sixteenth probe *(done)*: `fine_grained_classification` on
CUB-200-2011, the third distinct question asked by one linear-probe
implementation, which **replicates** v0.12's most surprising result rather than
merely adding to it. Sixteen probes against twelve backbones, 192 records.
Schema moves to **v8** — additively, so no published number changes — because a
trained run had been discarding the one diagnostic that separates an
underfitting probe from a weak representation.

**v0.14** — the release that learned to say no *(done)*. It ships no probe and
moves no number. Two candidates were built or scoped and then **refused on
measurement**, the second by the **oracle gate** this release adds: it asks
what a probe could score if the features contained the answer, needing no
backbone and no fitted head. BSDS500's dataset and a validated ODS/OIS/AP
metric ship — reproducing the published human agreement at 0.8030 against 0.80
— and its probe does not, at a 0.4193 ODS ceiling against Canny's published
0.60. Every dense probe that declares an oracle now reports a ceiling beside
its score.

**v0.15** — the release that measures its own gate *(done)*. No probe is added
and every published ordering is unchanged. The five low-level boards are
**re-run** so each record carries its `ceiling_*` beside its score — computable
from the target and the grid alone, so backfilling would have been easy and
would have put numbers in records no run produced. The DPT control measures
what the gate was described as bounding, and the word "ceiling" turns out to
have been too strong: it is a bar for the *linear* head VisBench reports, not a
bound on what is achievable.

**v0.16** — the release that changes no number and rewrites where the numbers
live *(done)*. The documentation site goes from four long pages to 42, the API
reference `conf.py` had been configured for since v0.7.0 finally exists, and
the README is an arrival path again. `depth`, `surface_normal` and
`generic_segmentation` get a reference page and a generated board for the first
time. Relative depth ordering was built and **rejected** in the same window.
**v0.16.1** corrects what the gallery figures *say* — the panels themselves
were right — and draws `depth` as a ramp rather than in grey.

**v0.17** — a seventeenth probe, and a control that changed how a board may be
read *(done)*. `instance_segmentation` asks which *object* a pixel belongs to,
over the same 1,449 VOC images `semantic_segmentation` scores: two boards on
identical pixels answering different questions. Seventeen probes against twelve
backbones, **204 board cells**. The split control is the more consequential
half — a cluster is a property of a board **as configured**, so never quote one
as a property of a task.

**v0.18** — the release where the records learned to describe themselves
*(done)*. Two fields, no new probe, and no measurement moves: every trained
board records how its *fit* went (schema v8's `training`, backfilled by
re-running 96 cells) and every run records what *machine* it ran on (schema
**v9**'s `hardware`, recorded and never grouped). Getting the first into the
corpus doubled as a reproducibility audit five releases on, and **the corpus
reproduces in full, 96 of 96** — catching, on the way, a cluster node returning
plausible wrong numbers while reporting success, which only the new fit
diagnostics could tell. The corpus goes from 264 records to **360**, still 204
board cells.

**Next** — there is no committed next step. NYUv2 scene parsing is **done**
(19a), and relative camera pose is **done**
(16a-1 to 16a-3): an eighteenth probe, its board, and the linear control
committed beside it, because it is the first board here whose head is not a
linear map and the honest form of "we used a bigger head" is a measurement of
what the smaller one did.

The rest of what follows is a candidate pool. Its cheap end is exhausted:
superpixels was built and rejected, DoG-blob rejected on overlap, relative
depth ordering rejected for failing to rank, BSDS500 refused by the gate,
instance segmentation shipped, and no optical-flow set is on this machine.

## Future directions

A candidate pool, not a commitment. VisBench is built one reviewed step at a
time, so these are ordered by *what they would cost*, not by preference — and
several are cheap only because the machinery they need already exists.

Anything here is open to contribution. `visbench/tasks/dense_base.py` supplies
everything a trained dense probe needs bar four methods, and
`visbench/tasks/magnitude_base.py` supplies the rest when the target is a
magnitude map.

### Cheapest — targets derived from the image itself

No new dataset. Taskonomy's `edge_texture` is already a target *computed from
the RGB frame*, and the same is true of these, so a target generator plus a
task subclass is most of the work. `corner` (v0.8) is the worked example for a
magnitude target and `orientation` for a vector one — read
`visbench/data/derived.py` and `visbench/tasks/low_level/{corner,orientation}.py`
before starting one of these.

| Task | Level | Note |
|---|---|---|
| Local orientation / gradient fields | low | **Implemented** as `orientation` — a 2-channel `(cos 2θ, sin 2θ)` field with coherence-weighted angular error; the first derived target that could not reuse `DenseMagnitudeTask` |
| ~~Superpixel / texture segmentation~~ | low | **Rejected after building it** — the SLIC boundary target passed every pre-measurement and then scored 0.021–0.043 correlation on three backbones, against 0.18–0.65 for every shipped low-level probe. A 1px partition boundary is not recoverable from patch features |
| Blob detection (DoG, LoG) | low | **Rejected** — the pre-measurement found its target correlates 0.51 with `corner`, as redundant with an existing probe as `corner` is with `edge` |

Four things a derived target has to establish before it is worth shipping, none
of which a probe run reveals on its own. The first is what the superpixel
rejection cost; the rest cost real time on the `corner` and `orientation`
probes:

1. **Run the oracle gate** — `python scripts/oracle_ceiling.py --targets <yours>`
   — which asks what the probe could score *if the features contained the
   answer*. A dense probe sees one feature vector per patch, so signal finer
   than a patch is absent from its input rather than merely hard to predict, and
   a target made of it cannot rank backbones however distinctive it is. Every
   shipped magnitude target scores 0.53–0.83 on a 16x16 grid; photometric
   superpixels scores 0.25, and was built anyway because this check did not yet
   exist. It needs no backbone and no fitted head, so it costs one pass over a
   split.
2. **Check the response's tail** before assuming the magnitude protocol
   transfers. A target with too much mass in its strongest 1% of pixels scores
   badly and ranks nothing. (An angle has no tail, so `orientation` skipped the
   compression — confirmed by the pre-measurement.)
3. **Check the overlap with what already ships, *before building*.** Cornerness
   correlates 0.52 with `edge_texture`; DoG blob correlated 0.51 with `corner`
   and was dropped; `orientation`'s `|r|` with both is under 0.09, because it
   measures phase. This costs an afternoon of correlations, not a probe run per
   backbone.
4. **A correlated target still earns its place if it *ranks* differently**, and
   that — not the absolute score — is the criterion.

Compute the target *after* the crop. There is then no second geometry and no
resampling of the response, which deletes the alignment hazard every other
dense probe has to test for.

### Reachable with data already common

| Task | Level | Note |
|---|---|---|
| ~~Instance segmentation~~ | high | **Done** — `instance_segmentation` on VOC2012, the seventeenth probe, with its twelve-backbone board. COCO was measured and refused: at a 16x16 grid the median VOC instance covers 16.92 patches against COCO's 2.27, and zero of 3207 VOC val instance pairs share a grid cell, which is what makes a per-patch head viable |
| ~~Fine-grained recognition (CUB-200-2011)~~ | high | **Done** — `fine_grained_classification`, a distinct probe on the linear-probe path, with its twelve-backbone board. Subordinate categories where the object board asks a basic-level question, which is why the object board is saturated and this one spans 0.87 to 0.47 |
| ~~Scene classification (Places365)~~ | high | **Done** — `scene_classification`, a distinct probe on the linear-probe path, with its twelve-backbone board. Ranks backbones almost independently of the object board (Spearman +0.15) |
| ~~Relative depth ordering~~ | mid | **Built and rejected** — it cleared the oracle gate at a 94.0% ceiling and then reproduced the `depth` board's ordering at Spearman **+1.000** over five backbones, at 38% of its spread, with two backbones 0.0007 apart that `depth` separates by 0.0707. Kept as a control, because the rejection is a finding about `depth`: discarding scale leaves its ranking unchanged. See `results/controls/README.md` |
| Intrinsic image decomposition (albedo vs shading) | mid | Classic Marr-style separation of appearance from geometry and lighting. Ground truth is scarce outside synthetic data |
| Room / scene layout estimation | mid | Floor–wall–ceiling boundaries |
| ~~Semantic segmentation on NYUv2 (40 classes)~~ | high | **Done** — shipped as `scene_parsing`, the nineteenth probe, with its thirteen-backbone board. Named for the question rather than the dataset, per this project's convention. The label convention was measured before any code: 255 is void (88.5% of border pixels against 13.7% of interior ones) and 0 is `wall`, so it is VOC's shape and needed no new loader |
| Vanishing point / line detection | low | Published as a Taskonomy domain |
| Color constancy / illuminant estimation | low | Needs measured illuminant ground truth |

### Harder — new machinery, not just a new dataset

| Task | Level | Note |
|---|---|---|
| Optical flow | low | Needs image pairs and a flow head. `PairViewDataset` already expresses the pairing; the head is the real cost |
| ~~Relative camera pose (essential / fundamental matrix)~~ | mid | **Done** — probe3d's pairwise 7D regression on NAVI, which is on this machine. It reads *pooled* features, so it is cheaper than this table implies. Multi-partner sampling settled it: at eight partners per training anchor every one of four ViT-B/16s clears the no-feature floor by 24 deg or more and every adjacent pair is separated by 5-7x the widest seed range. The dataset and the metric ship (16a-1); the head and the task follow. Two decisions were taken first, both protocol rather than tuning — the pair count is **pinned** at eight partners, and the board departs from `hidden_dim=0` for probe3d's MLP with the linear run kept as a control. See `visbench/tasks/low_level/README.md` |
| Multi-view stereo / point-cloud / mesh recovery | mid | Multi-view input and a non-raster output; the largest departure from every probe here |
| Motion / video object segmentation | mid | Grouping by motion, not identity. Needs video input, which nothing here consumes yet — **the machinery is the blocker, not the data**: DAVIS is on this machine, 90 sequences with 2016 and 2017 split lists |
| Action / activity recognition | high | Same video constraint |
| Amodal completion of occluded boundaries | mid | Targets extend beyond visible evidence |
| Symmetry / repeated-structure detection | mid | Annotation is scarce |
| Object counting without recognition | mid | Scalar-per-image target rather than dense or categorical |
| Attribute recognition (material, function) | high | Multi-label rather than multi-class |
| Panoptic segmentation | high | Instance segmentation plus stuff classes |
| Image captioning / VQA | high | Needs a language decoder — a different kind of probe from anything here |
| Zero-shot / open-vocabulary classification | high | Only meaningful for backbones with a text tower, so it does not rank the full set |
| Animal identification | high | Fine-grained recognition at individual rather than species level |

### Already partly answered

Worth naming so they are not re-scoped from scratch:

- **Edge / contour detection** is implemented (v0.4) as dense magnitude
  regression on Taskonomy. What is missing is **BSDS500's** ODS/OIS/AP boundary
  protocol, which is a bipartite matching after non-maximum suppression and a
  step of its own — not a dataset swap. **The dataset and the metric both ship**
  (`scripts/fetch_bsds500.py`, `visbench.data.BSDS500Dataset`,
  `visbench.metrics.boundary`), and the metric reproduces BSDS500's published
  human agreement — ODS 0.8030 against the published 0.80.

  **There is deliberately no BSDS probe.** The oracle gate puts a linear
  probe's ceiling at **0.4193 ODS** on the 16x16 feature grid every corpus
  backbone produces at 224px, against published detectors at 0.60-0.79. A board
  whose ceiling sits below the weakest classical baseline could not be compared
  with the literature, which is the only reason to add this dataset rather than
  reuse the Taskonomy edge probe.

  **The first of the two routes that could have reopened it has been measured,
  and it does not.** The gate models a linear head exactly, so a progressive
  decoder was the obvious way it might understate what is achievable — and it
  does: `results/controls/dpt_head.jsonl` puts a DPT head at 54-104% of the
  linear oracle across five probes and **nine** ViTs, exceeding it in 2 of 45
  cells — both `mae_vitb16`. But scaling 0.4193 by the best ratio observed
  anywhere (1.038) gives ~0.435 ODS, still below the weakest classical
  baseline, so **the closure survives the correction to its premise.** The remaining route is to run at
  32x32 tokens or finer, which only DINOv2 can do — and a board only DINOv2 can
  appear on ranks nothing. Both are written up in
  `visbench/tasks/low_level/README.md`.
- **Occlusion-edge detection** (v0.5) already covers the depth-discontinuity
  half of contour detection, at mid level.
- **Corner detection** is implemented (v0.8) as Shi-Tomasi cornerness computed
  from the RGB frame — λ_min rather than Harris's `R`, because it is
  non-negative by construction and has no `k` to record.
- **Gradient orientation** is implemented as `orientation` — a coherence-weighted
  `(cos 2θ, sin 2θ)` field, also computed from the frame, the first derived
  target that is a direction rather than a magnitude. **Blob detection (DoG)
  was rejected** by its pre-measurement: the target correlates 0.51 with
  `corner`.
- **Texture / reflectance** overlaps with intrinsic decomposition above.
  Taskonomy ships no reflectance domain, so `mask_valid/` did not unblock it.

## Library surface — candidates that ship no new number

Everything above adds a probe. These do not: they are the parts of the library
someone reaches for *around* a measurement, and each one is a gap found by
asking what a new user would try and failing to find it. v0.7 (7a–7e) is the
precedent — a release that changed no number and was worth shipping anyway.

None of these is a defect. Every one is reachable today by writing Python; what
is missing is the shortest path to it.

### Looking at what a probe saw and what it predicted — **done**

Shipped as `visbench show`, the first visualisation anywhere in the package. It
writes a grid of image / target / prediction panels to a file for the nine
probes with a spatial target — the eight dense ones plus `detection` — and
measures nothing.

**The argument for it was the project's own bug history, not convenience.** Two
of the most expensive failures here were geometry misalignments that stayed
invisible because nothing ever rendered a target next to its image:

- the correspondence misalignment that scored `recall@1px = 0.003`
- VOC's palette PNGs read through `convert("L")`, turning classes
  `[0, 1, 15, 255]` into `[0, 38, 147, 220]` — which loads, trains and scores
  against labels that mean nothing

Both were found by reading code. Both are obvious in one frame of output.

The two open decisions were settled as: **PIL and numpy only**, no matplotlib
and no dependency change; and **a saved artifact rather than training on the
spot**, which is what `visbench run --save-probe` was added to produce. The
stated hazard — that a viewer applying its own resize or colour-map is a second
geometry — is guarded by a test pinning the image panel byte-for-byte against
`np.asarray(dataset[i][0])`.

See [looking at a probe](guides/visualising.md) for the three rules it keeps and why invalid
pixels are magenta.

**The pair renderer for `correspondence` followed immediately**, and step 9c
covered the last three, so **every probe is drawable** and
`show_probes() == list_probes()` is asserted. Two frames with the matches between them is a
different layout from a panel grid, and it is the probe whose historical bug is
quoted above — which is why it reports **coherence**, the mean resultant length
of the error directions. Measured with ResNet-18 features: 0.29-0.40 when the
geometry is right, 0.98-1.00 when the homography is in the wrong pixel frame,
while the median error alone cannot separate "broken" from "hopeless".

`classification`, `retrieval` and `similarity` have no spatial target, so they
draw the *decision* instead — a contact sheet, a query with its neighbours, a
triplet with the human vote marked. Each states a diagnostic as a figure the way
coherence does: **class balance**, which catches a split collapsed to one class
scoring 1.0, and **vote balance**, which catches a vote read from the wrong CSV
column.

A rendered figure for every probe now ships in the README and on the docs site,
drawn by `scripts/render_gallery.py` on **real photographs** — Open Images
validation frames, CC BY 2.0, committed under `assets/gallery_frames/` so the
gallery still rebuilds from one command with no downloads. Fetching them is a
separate one-off step (`scripts/fetch_gallery_frames.py`).

**The licence rule that made the first gallery synthetic was satisfied by
better sourcing, not waived.** VOC, ImageNet, NYUv2, Taskonomy and NIGHTS each
restrict redistribution and appear nowhere in this repository; Open Images
grants it, and the grant is verified *per frame* rather than inherited from
that sentence — a frame whose metadata carries no author or landing page is
refused, because an unattributable CC BY image is one this repository may not
ship. `CREDITS.md` is generated beside the frames and a test fails if a
committed photograph has no credit.

Real annotation cost one property and improved another. Ground truth is now
what a human marked wherever it is annotated, rather than what a script
constructed — but **invalid pixels are no longer placed on purpose**, since
real annotation has holes only where it has them. Four probes cannot have a
target column at all: `depth`, `surface_normal`, `keypoints2d` and
`occlusion_edge` need sensor or reconstruction geometry no redistributable
photograph carries, so they render `image | prediction` from a published Hub
head and say so in the footer.

### Bringing a dataset VisBench has never heard of — **done**

Three tiers now:

- **Folder layouts, no code.** `ImageFolderDataset` (class subdirectories or
  flat), `DenseFolderDataset` (`images/` + `targets/`, with `stems=` for an
  official split list), `DetectionFolderDataset` (VOC-style XML). NYUv2 joined
  the corpus with *no code change* because its layout already matched.
- **A `torch.utils.data` dataset or a Hugging Face `datasets.Dataset`** wraps in
  `TorchvisionDataset` / `HuggingFaceDataset` (`visbench.data`). Both are thin
  adapters over `BaseDataset`. The image-level CLI probes take
  `--dataset torchvision:CIFAR10` / `--dataset hf:cifar100` in place of
  `--data <path>`; `datasets` is a `[datasets]` extra, `torchvision` is already
  a core dependency.
- **Anything else, by subclassing `BaseDataset`** — two abstract methods,
  `__len__` and `__getitem__`.

**The trap the bridges had to avoid.** `BaseDataset` has four optional methods
beyond the two abstract ones, and skipping them fails *silently*: `labels()`
(supervised probes have no targets), `cache_identity()` (every run re-decodes
every image — the memo cannot recognise the file), `fingerprint()` (records
cannot tell your dataset from another), `describe()` (`dataset_params` comes out
empty). `cache_identity` is the worst: return `None` and everything still works,
just slowly, forever — the `view_identity` failure, a mechanism tested and
correct for a year while a caller passed bare PIL images and paid a full decode
on every "cached" run.

Both bridges supply a real `cache_identity` by leaning on one property: **the
wrapped dataset is immutable in index order.** A `datasets.Dataset` carries a
`_fingerprint` that changes on any transform, so `(fingerprint, row index)`
names a row's content exactly. A `torchvision` dataset has no such hash, so the
`ImageFolder` family uses the file path and everything else a digest of the
`__repr__` (which states root, split and download flags) plus the index — weaker,
and documented on the class rather than hidden.

### Probing a model VisBench has never heard of

This is the best-supported of the three and mostly needs *showing*, not
building. `CustomBackbone` is the documented escape hatch and its docstring
already names the case — a fine-tuned checkpoint, an architecture VisBench has
never heard of, something from a paper's repo:

```python
backbone = CustomBackbone(my_model, preprocess=my_transform, name="mine")
visbench.run(backbone, "retrieval", dataset)
```

`hash_weights()` keys the cache on the parameters themselves, so a fine-tuned
checkpoint automatically gets a different cache entry from the model it was
fine-tuned from. `register_backbone` is a top-level export for anyone who would
rather have a registry name.

**`examples/custom_backbone.py` closes the first of these** — it wraps
torchvision's ResNet-18, shows the cache key moving when the weights change,
and registers a named subclass, all through ordinary `visbench.run()` calls on
generated data that needs no download.

What remains:

| gap | note |
| --- | --- |
| Not reachable from the CLI | `--backbone` is a registry name, and a string cannot carry an `nn.Module`. Registering a named `BaseBackbone` subclass is the workaround, and the example demonstrates it |
| Fine-tuning does not apply | `finetune_blocks` is DINOv2-only by design and raises elsewhere |

**One thing the example measured that was not previously written down**: a
wrapped model is constructed *before* `run()` seeds, while a registry name is
constructed *after*, so a trained probe's head is initialised from a different
RNG state on the two paths. Measured on bit-identical features (max absolute
difference 0.0), classification top-1 came out 0.9125 wrapped against 0.9062
named — and each path's own spread across five seeds is 0.0062, so this is RNG
jitter rather than a cost of wrapping. The wrapped path is *perfectly*
reproducible, which is the opposite of what the hazard sounds like. Zero-shot
probes are identical bit for bit, since no head is fitted.

### If these are picked up, this order

Ordered by cost against what they prevent, not by preference:

1. ~~**`examples/custom_backbone.py`**~~ — **done.** It closed a gap between
   what the docs promised and what was demonstrated, and measuring it turned up
   the RNG-position note above.
2. ~~**`visbench show`**~~ — **done** in v0.9 (steps 9a-9d).
3. ~~**The dataset bridges**~~ — **done.** `TorchvisionDataset` /
   `HuggingFaceDataset` plus `--dataset torchvision:… | hf:…` on the image-level
   probes. Dense/pair/triplet probes stay folder-only for now — an HF dataset
   carrying a dense target is a much larger surface (per-probe target-column
   plumbing, loader/dtype selection, the four validity conventions).

**The whole library-surface backlog is now closed.**

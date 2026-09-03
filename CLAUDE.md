# VisBench

Unified library for probing vision backbones (DINOv2, CLIP, custom) across
high-level, mid-level, and eventually low-level computer vision tasks, through
a `get_backbone()` / `get_probe()` API. Sibling project to vismatch (image
matching - https://github.com/gmberton/vismatch), same ergonomic philosophy, applied to representation probing
instead of matching.

**Distribution**: this ships as a pip-installable Python package on
[PyPI](https://pypi.org/) (`pip install visbench`), not just a research repo.
Packaging conventions (pyproject.toml, semantic versioning, a lockfile) apply
from v0.1 onward, not bolted on later.

**Prior art to credit explicitly, not re-derive**:
- [probe3d](https://arxiv.org/abs/2404.08476) (El Banani et al., CVPR 2024) —
  reuse its evaluation protocols for depth, surface normal, and correspondence.
- Chen, Marks & Cheng, ["Probing the Mid-level Vision Capabilities of
  Self-Supervised Learning"](https://arxiv.org/abs/2411.17474) — the task
  categorization below follows this paper directly.

---

## Critical: build order — read this before writing any code

Do not implement all tasks, backbones, and versions in one pass. This is a
multi-month roadmap; **each session completes one step and stops for review**,
not racing ahead. If asked to "build VisBench" or "continue", re-confirm which
step is next rather than attempting the whole roadmap in one session.

| Step | What | Status |
| --- | --- | --- |
| 1 | Scaffold every folder and module, docstrings + stubs, no logic | done |
| 2 | `BaseBackbone` + feature cache + DINOv2, with tests | done |
| 3 | `BaseTask` + one task (retrieval) end to end on a local folder | done |
| 4 | All three v0.1 tasks, both v0.1 backbones, `uv.lock`, `run()` | done |
| 5a | ResNet/timm backbone — first non-ViT, validates the CNN half | done |
| 5b | Custom `nn.Module` backbones, and pluggable heads (linear + DPT) | done |
| 5c | Multi-layer extraction through every backbone and the cache | done |
| 5d | Depth estimation — first dense task, full probe3d protocol | done |
| 5e | Streaming features from disk, for splits larger than memory | done |
| 5f | Surface normals + the shared `DenseTrainingTask` | done |
| 5g | Generic (binary) segmentation | done |
| 5h | High-level semantic (multi-class) segmentation | done |
| 5i | Mid-level image similarity | done |
| 5j | The CLI — last, once the dense-task Python API has settled | done |
| 6a | Fine-tuning: unfreeze last N blocks, cache out of the path, DINOv2 only, proved on VOC segmentation | done |
| 6b | Cache the frozen prefix — works, saves 21%, and found the real bottleneck | done |
| 6c-1 | Detection: the box dataset and VOC loader | done |
| 6c-2 | Detection: `average_precision`, mAP@50, mAP@50:95 | done |
| 6c-3 | Detection: the head, against a metric already trusted | done |
| 6d-0 | Dataset listing: `scandir`, not a stat per file | done |
| 6d-1 | Edge detection — the first low-level task, on Taskonomy | done |
| 6d-2 | `mask_valid`, keypoints2d + occlusion_edge, `DenseMagnitudeTask` | done |
| 6e-1 | Leaderboard: the comparability rules, as pure functions | done |
| 6e-2 | Leaderboard: regenerate a record corpus for all twelve probes | done |
| 6e-3 | Leaderboard: render it, and generate the README tables from records | done |
| 6e-4 | Hub: serialise a trained head, with the backbone identity beside it | done |
| 6e-5 | Hub: push/pull through `huggingface_hub`, behind a `[hub]` extra | done |
| 6f | Correspondence: score in pixels — the unit that inverted the board | done |
| 7a | `visbench demo` — a real probe run that needs no dataset | done |
| 7b | Reorganise the README around a reader; split `docs/` out of it | done |
| 7c | `CONTRIBUTING.md`, issue and PR templates, tests that keep them true | done |
| 7d | The documentation site: Sphinx, the theme, and the `Docs` workflow | done |
| 7e | Citation metadata: `CITATION.cff`, `.zenodo.json`, and a DOI | done |
| 8a | Corner detection — the first target computed rather than downloaded | done |
| 8b | Corner in the corpus: a pinned frame set, and a generated board | done |
| 9a | `visbench show` — the panel viewer, and `run --save-probe` | done |
| 9b | `visbench show correspondence` — the pair renderer, and coherence | done |
| 9c | `show` for the last three probes; every probe drawable | done |
| 9d | The rendered gallery: one figure per probe, generated not photographed | done |
| 10a | Three more backbones: `TimmBackbone` learns to read a ViT | done |
| 10b | The corpus at 13 probes x 9 backbones | done |
| 10c | Supervised ViT-B/16: the corpus's first controlled experiment | done |
| 11a | The gallery on real photographs, licence-checked | done |
| 10d | `dino_vitb16`: the objective family becomes three wide | done |
| 10e | `sam_vitb16`, a recipe control — the denominator an objective gap needs | done |
| 12a-1 | BSDS500: the dataset, and several people's answers per image | done |
| 12a-2 | BSDS500: ODS/OIS/AP, reproducing the published human agreement | done |
| 12a-3 | BSDS500: the probe — **refused by the oracle gate**, line closed | n/a |

**A closed step's full write-up lives in
[`ENGINEERING_LOG.md`](ENGINEERING_LOG.md), not here.** That file is the archive
of what each step measured, rejected and decided, and it is where every `6x`
label in this document resolves. This file carries the *rule*; the log carries
the derivation behind it. Read the log before touching the code a step built —
several of those decisions look wrong until you read the numbers.

**When you finish a step, its narrative goes to `CHANGELOG.md`, its derivation
to `ENGINEERING_LOG.md`, and anything about *what a board means* to
[`CORPUS_FINDINGS.md`](CORPUS_FINDINGS.md); add at most a few lines here.**
This file is loaded into every session's context whole, so it has a budget, and
it has now been over that budget **twice**: 203k on 2026-08-20, split three ways
to get back under it, and 151k on 2026-09-03, trimmed to 134k by moving the
per-release upload records and the v0.1/v0.2 scope to `ENGINEERING_LOG.md` and
collapsing board write-ups that `CORPUS_FINDINGS.md` already carried in full.
**Both times the growth was retrospective narrative, not rules** — so when it
next crosses, look for a closed step's write-up or a per-release paragraph
before touching "decisions already paid for", which is 70k of the file and is
what the file is for. **Read `CORPUS_FINDINGS.md` before quoting any board** —
this file keeps only the claims, and several of them corrected an earlier
published reading.

Steps 7a-7e ship no new probe, backbone or metric. They are the **contributor-
facing surface**: the shortest path from `pip install` to a number, where the
reference material lives, how someone outside this file learns the rules, and
how the work is cited. All five are **v0.7.0**, the release that changes no
number — every measurement v0.6.1 reported, v0.7.0 reports identically.

---

## Current state

**v0.1 through v0.8 are all complete, and every numbered step in the build table
is done.** Every task, all three backbone families, the CLI, fine-tuning,
detection, the low-level probes, the leaderboard and probe sharing. Everything
below exists, is tested, and is on `main`. v0.4.0 filled the low-level tier;
v0.5.0 was the `mask_valid` release; **v0.6.0 is the leaderboard release**
(2026-08-02), **v0.6.1 corrects the correspondence board it shipped ranked
upside down** — see step 6f — v0.7.0 is the contributor-facing release that
changes no number, and **v0.8.0 (2026-08-07) is the corner probe**, steps 8a and
8b together.

**v0.9.0 through v0.14.0 are all shipped**, and each one's narrative is in
`CHANGELOG.md`, its derivation in [`ENGINEERING_LOG.md`](ENGINEERING_LOG.md)
and its upload record in that file's "Release history": v0.9.0 was
`visbench show` (9a-9d) plus the Hub work below; v0.10.0 was three timm
backbones, a supervised ViT-B/16 and the gallery on real photographs (10a-10c,
11a); v0.11.0 was `dino_vitb16`, the `sam_vitb16` recipe control and
`examples/custom_backbone.py` (10d, 10e); v0.12.0-v0.13.0 were the
`scene_classification`, `orientation` and `fine_grained_classification` probes
and their boards; v0.14.0 is the oracle gate. **The corpus is 192 records
across sixteen boards**, twelve backbones a board.

Two standing consequences of that history, both of which have already cost a
published claim:

- **Read a count off `LEADERBOARD.md`, never out of prose.** Every count that
  has gone stale in this project went stale by being carried forward through a
  release that added a column or a board. The generated tables have always been
  right; only the prose around them drifts, which is the half no test reads.
- **Before quoting any board, read
  [`CORPUS_FINDINGS.md`](CORPUS_FINDINGS.md)** — in particular the
  `sam_vitb16`, `dino_vitb16` and `mae_vitb16` entries. 10e's recipe control
  refuted half of 10d's own published claim before either shipped, and the
  three-tier separation 10b announced no longer holds for *high-level*.

**There is no `next` step.** The remaining work is the candidate task backlog
further down this file — and the cheapest items there need no new dataset at
all. **Two of those are done.**

**Three probes shipped after v0.11.0 and each has a 12-backbone board**, all
2026-08-28 unless stated. Their board readings are in
[`CORPUS_FINDINGS.md`](CORPUS_FINDINGS.md) and their per-probe reference in
`docs/tasks.md`; what matters here is what each one *is*:

- **`scene_classification`** (14th, probe 2026-08-27) — scene category on the
  same linear-probe path as object `classification`, on `places365_standard`,
  read with no loader code.
- **`orientation`** (15th) — local gradient orientation, the fourth low-level
  task and the second computed from the frame, but the first whose target is a
  *direction*, so the first that could not reuse `DenseMagnitudeTask`. Target is
  `(cos 2θ, sin 2θ)` with its length set to the coherence; `orientation_error`
  is degrees of coherence-weighted angular error, halved so 45 is chance. It
  reuses `corner`'s pinned `data/corner_frames/` set. **DoG-blob was the first
  candidate for this slot and was rejected** at 0.51 overlap with `corner`.
- **`fine_grained_classification`** (16th) — subordinate category on the same
  path again, on **CUB-200-2011**, the official 5994/5794 split read from
  `vision/CUB-200/images_train_test/`. `probe_fine_grained_classification` runs
  the whole official split with no `--limit`, which is what makes the board
  comparable to the published CUB literature.

Three probes now share one implementation and ask three questions — basic-level,
place, and subordinate — and each is a distinct probe *name* for the reason in
"decisions already paid for", which the second and third instances confirmed
edit for edit.

**Two findings from those boards are load-bearing enough to state here**, both
expanded in `CORPUS_FINDINGS.md`: the two image-level classification probes
rank with the *localised* cluster (`detection`, `semantic_segmentation`) rather
than with the object board they subclass — `fine_grained_classification`
correlates **+0.860 with `detection`** against +0.343 with `classification` —
and `orientation`'s board is **not** independent even though its target is,
ranking like `keypoints2d` (rho +0.95), `corner` (+0.82) and `edge` (+0.79).


**There is a second backlog beside the candidate-task one**, the
library-surface one, which ships no new number the way v0.7 did; it is in the same part of this file and in
`docs/roadmap.md`. **Its first item is done as of 2026-08-14**: step 9a shipped
`visbench show`, the panel viewer, plus `visbench run --save-probe` to feed its
prediction column, **9b added the correspondence pair renderer**, and **9c
covered the last three** — so `visbench show` is valid on *every* probe and
`show_probes() == list_probes()` is asserted — and **9d added the rendered
gallery**: one generated figure per probe in the README and on the docs site. See the build table and the five
bullets in "decisions already paid for". **The library-surface backlog is now
closed** (2026-08-28): the **dataset bridges** shipped —
`TorchvisionDataset` / `HuggingFaceDataset` in `visbench.data`, plus
`--dataset torchvision:… | hf:…` on the three image-level probes — after
`visbench show` (9a-9d) and `examples/custom_backbone.py` (2026-08-19). The
candidate-task backlog is what remains, and **fine-grained recognition came
off it on 2026-08-28** (`fine_grained_classification`, CUB-200-2011).
**Photometric superpixels was built and rejected the same day** — it scored
0.021-0.043 after passing every pre-measurement; see the bullet below and
`visbench/tasks/low_level/README.md`. **The gauntlet gained the oracle gate that
rejection was missing on 2026-09-01** — `scripts/oracle_ceiling.py`, calibrated
so the four shipped magnitude targets pass at 0.53-0.83 and the rejected one
fails at 0.25. It ships no probe and moves no number. That leaves BSDS500 edge
and optical flow,
**both of which need a download first** — and **the BSDS500 line is closed at
two steps** (12a-1 and 12a-2, 2026-09-01): the dataset and a validated
ODS/OIS/AP metric ship, reproducing the published human ODS of 0.80 at
**0.8030**, and **the probe was refused by the oracle gate**. A linear probe on
the 16x16 grid every corpus backbone produces has a ceiling of **0.4193 ODS**,
below Canny's published 0.60, so a board could not be compared with the
literature — which was the only reason to add BSDS rather than reuse `edge`.
The write-up and the two routes that could reopen it are in
`visbench/tasks/low_level/README.md`. Nothing cheap remains. Re-confirm what is wanted before
starting anything; do not assume its order is a plan. **The one thread that was open — why `detection`
alone fails to reproduce — is closed**: it is GPU non-determinism made visible
by a discrete metric, it was never a bug, and detection reproduces to *three*
decimals rather than four. **Settled 2026-08-14 on all six backbones**: only
the two 16x16-grid rows (DINOv2) drift; CLIP-B/16, CLIP-B/32 and both ResNets
are bit-identical and match their corpus records to every digit, so it tracks
the feature grid rather than the width, the architecture or the probe. See
[`CORPUS_FINDINGS.md`](CORPUS_FINDINGS.md) for the table. **There is no open lead
here any more.**

**Step 8a added the thirteenth probe**, `corner` — the first whose target is
computed from the image rather than downloaded, so `visbench/data/derived.py`
is a new kind of dataset here and the low-level tier now has three entries.
**8b put it in the corpus**: six records on a pinned frame set, a generated
board replacing 8a's hand-written table, and `scripts/stage_corner_frames.py`
to reconstruct the frames. **Both shipped together as v0.8.0**, which is what is
on PyPI.

**Between v0.6.1 and v0.7.0 the work was contributor-facing, not measurement**
(7a-7e).
`visbench demo` runs a real probe on generated images with no dataset and no
extras; the README is 397 lines instead of 1,020, with the per-probe reference
in `docs/tasks.md` and the roadmap in `docs/roadmap.md`; `CONTRIBUTING.md` is
the public version of the rules this file keeps; and `docs/` is now a Sphinx
site deployed to GitHub Pages by a **third workflow**, `docs.yml`. **The
generated tables moved with the reference material** — they are in
`docs/tasks.md` now, not the README, and `scripts/render_tables.py` takes a
list of marked files. 7e added `CITATION.cff` and `.zenodo.json`, so a release
is archived with a DOI — and **the DOI now exists**: v0.7.0 was released on
GitHub, Zenodo archived it, and the concept DOI is
[10.5281/zenodo.21822684](https://doi.org/10.5281/zenodo.21822684). **v0.7.0
reached PyPI on 2026-08-06**, verified out of the wheel; see below.

**What v0.6.0 changed, in one paragraph.** `results/corpus/visbench.jsonl` is a
committed corpus — 72 records when it shipped, 78 after 8b added `corner`, and
117 after 10b added the three timm backbones, and **130 since 10c added a
supervised ViT** — of every probe against every
backbone, one comparability group each and every group holding all ten.
`visbench/results/leaderboard.py`
holds the rules for which records may be ranked together and
`visbench/results/render.py` turns an answer into markdown; **ten** marker-
delimited tables and `LEADERBOARD.md` are generated from the corpus, and a
**fast** test fails if either drifts. (Nine of those were in the README when
v0.6.0 shipped; 7b moved them to `docs/tasks.md`, and corner's joined them in
8b.) `visbench/hub/` serialises a
trained head with the backbone identity beside it and moves it to and from the
Hugging Face Hub behind a `[hub]` extra. Schema was **v7** at that release — `pooling_requested`, because keying
comparability on resolved pooling could never rank a CNN against a ViT.

**v0.3's numbered steps are all done: 6a (fine-tuning), 6b
(prefix caching) and all of 6c (detection).** Dense probes take `finetune_blocks=N` /
`--finetune-blocks N`, DINOv2 only, recorded under schema v6's `finetune` field.
Proved on VOC at two scales: 0.7758 against the frozen 0.7328 on DINOv2-S, and
0.7992 against 0.7533 on DINOv2-B. 6b caches the frozen blocks below the cut in
a separate `PrefixCache`, cutting a fine-tuned ViT-B run from ~345 s to ~272 s
with the mIoU unchanged to four decimals. 6c added the box dataset, the VOC
metric and an anchor-free single-scale detection probe, in that order. **The
schema is still v6 — detection needed no bump**, because `task_params` and
`dataset_params` are both open dicts and the protocol, the decoding settings and
`include_difficult` all land in them.

**6d-1 added the first low-level task**, so all three levels of the taxonomy now
have entries and `visbench/tasks/low_level/` is no longer a placeholder. Edge
detection is dense magnitude regression on Taskonomy's `edge_texture`, scored by
per-image Pearson correlation and recorded as `visbench_edge_regression` — not
BSDS500's, which is a correspondence metric and a step of its own.

**6d-2 wired up `mask_valid/` and added two more probes**, released as v0.5.0.
Four of the six Taskonomy domains that were refused are now supported — `depth_zbuffer`, `normal`, `edge_occlusion`, `keypoints3d` —
each declaring how it marks an invalid pixel rather than exposing a mask.
`keypoints2d` (low-level) and `occlusion_edge` (mid-level) joined `edge` on a
lifted `DenseMagnitudeTask`. **Still schema v6**: twelve probes, and the new
`target_transform` / `invalid` / `masked` settings land in `dataset_params`,
which is what that field was added for. The next step is **6d-3+** — another
task, or the HF Hub / leaderboard work.

Registered names — `visbench.list_backbones()`, `list_probes()`,
`visbench.heads.list_heads()`:

```text
backbones  dinov2_vits14, dinov2_vitb14, clip_vitb16, clip_vitb32,
           resnet18, resnet50, convnext_base, mae_vitb16, siglip_vitb16,
           supervised_vitb16, dino_vitb16, sam_vitb16,
           dinov2_vitb14_196 (a resolution control, not a corpus column)
           (+ CustomBackbone, unregistered)
probes     classification, scene_classification,
           fine_grained_classification, retrieval, correspondence,
           depth, surface_normal, generic_segmentation, semantic_segmentation,
           similarity, detection, edge, keypoints2d, occlusion_edge, corner,
           orientation
heads      linear, dpt, detection
```

The CLI exposes all sixteen probes: `visbench list`, `visbench run <probe>`,
`visbench cache stats|clear`, plus `visbench demo` (7a) and **`visbench show
<probe>` (9a)**. A test asserts the CLI's table and `list_probes()` are the same
set, so a probe cannot ship unreachable from a shell by accident. Since 9c
`show` is valid on *every* probe and `show_probes() == list_probes()` is
asserted.

**`visbench run --push-to REPO_ID` publishes the head it just trained**
(`--public` overrides the private default), and `scripts/build_corpus.sh` takes
`PUSH_TO` / `PUSH_PUBLIC` so a whole board is published from the file that
already holds every probe's flags. **Publishing from the run, not from a second
script, is the design**: a head is only meaningful against the features it was
fitted on, and the run's flags are what fitted them — a separate publish step is
a second copy of every dataset flag, free to drift, and a head trained under
drifted flags uploads, loads and scores without complaint. The CLI refuses a
zero-shot probe *before* the run; `save_probe` would raise the same thing after
it, having spent the whole run to do so. `scripts/publish_collection.py` groups
the pushed repositories into one collection, dry-run unless `--create`.

**Twenty trained heads are published and public**, as of 2026-08-07: the ten
trained probes against DINOv2-S/14 and DINOv2-B/14, one repository per pair at
`turhancan97/visbench-<probe>-<backbone>`, gathered in a collection whose URL is
quoted in `README.md` and `docs/hub.md`. **Read it from one of those two files
rather than reconstructing it** — a Hub collection slug carries a generated hash
suffix and cannot be derived from its title. The three zero-shot probes are
deliberately absent. Two operational notes that each cost an attempt: the Hub
caps a collection description at **150 characters** and rejects a longer one
with a 400 naming neither the limit nor the value (asserted at import in the
script, so it fails before the network call), and creating a collection needs
**`collection.write`** on the token — `repo.write` alone pushes models happily
and then 403s.

Republishing the board is `PUSH_TO=... PUSH_PUBLIC=1 scripts/build_corpus.sh`.
**Point `RESULTS=` at a scratch file, never `results/corpus/visbench.jsonl`**,
so the run can be diffed against the corpus instead of replacing the reference
it would be checked against. That diff is what caught the seeding bug below, and
it is the only reason the bug was found at all. Do not pipe a long publishing
run through `tail`: it buffers, so a run that is killed part-way leaves no log,
and the Hub then has to be queried to find out what actually shipped — which
happened, and is recoverable only because each record names its own pair.

**The current release is `0.14.0`** (2026-09-01), and every release from
v0.6.0 onward is recorded paragraph by paragraph in
[`ENGINEERING_LOG.md`](ENGINEERING_LOG.md) under "Release history" — byte
counts, wheel digests, which commit the tag resolves to, and what each
`__version__`/`SCHEMA_VERSION` import read back. Read it there rather than
carrying it here; only the standing rules below stay in this file.

**What 0.14.0 is**: the release that ships no probe. The corpus is unchanged at
16 probes x 12 backbones, 192 records, and schema stays at v8. What it adds is
the **oracle gate**, the **ceiling beside every dense score**, and **BSDS500's
dataset plus a validated ODS/OIS/AP metric** reproducing the published human
agreement at 0.8030 against 0.80 — and it is the release in which the gate
*refused* a probe, BSDS's own, at a 0.4193 ODS ceiling. `scipy` became a
declared core dependency.

**The concept DOI is `10.5281/zenodo.21822684`**, unchanged across all eight
version DOIs and now resolving to v0.14.0. That is the point of quoting it
rather than a version DOI; `tests/test_citation.py` rejects any other Zenodo
DOI in `README.md`, `docs/index.md` and `CITATION.cff`, because pasting one
over it is the realistic mistake and it freezes every citation at one release.


**Releasing — the standing rules, learned across v0.6.0 through v0.11.0.** The
release-by-release detail is in [`ENGINEERING_LOG.md`](ENGINEERING_LOG.md),
under "Release history"; what recurs is:

- **Tag before building**, so the artifact is built from the tagged commit.
  v0.10.0 was the first release whose tag and wheel agree exactly and v0.11.0
  the second, which is what that order buys.
- **Never move a tag to close a gap.** A PyPI version can never be re-uploaded
  and a Zenodo archive is permanent, so a moved tag would disagree with both.
  Three releases up to v0.10.0 had `main` one docs-only commit ahead of the
  tag, which is precisely the benign pattern that trains you to stop checking;
  such a gap reaches PyPI with the next release.
- **Clear `dist/` first.** `twine upload dist/*` uploads everything in the
  directory, so a stale artifact from the previous release is attempted too,
  PyPI refuses a version that already exists, and twine aborts the batch —
  failing on the *old* version before it reaches the new one, which reads as a
  problem with the new one.
- **Keep twine current.** Hatchling emits `Metadata-Version: 2.5`, which twine
  6.2.0 rejects as "not a valid metadata version" — the builder outran the
  checker, and `pip install -U twine` (7.0.0) is the fix; PyPI's server accepts
  2.5 without complaint. Metadata 2.5 also writes environment markers with
  **single** quotes (`extra == 'hub'`) where earlier versions used double, so a
  check keyed on `extra == "hub"` silently matches nothing and reports every
  extra as empty — which looks exactly like the extras having lost their
  dependencies. Match either quote.
- **Verify out of the published wheel, by import.** Download it from the JSON
  API, check its SHA256 against PyPI's digest, extract it, put it *first* on
  `sys.path` and **import** it, with an assert on `visbench.__file__` so the
  editable checkout cannot answer in its place. That last step is the one worth
  copying: without it the check passes on a machine where the package is
  already installed, whatever the wheel contains.
- **When a release's content is a default value, read it back through an
  import, not out of the source text** — source inspection cannot rule out a
  runtime override. That is how v0.6.1's `threshold_units="pixel"` was
  confirmed, and it has been re-read through every release since.
- **`.venv/` has no `pip`** — it is uv-managed, so `pip download` and
  `python -m pip` both fail there. Fetch the artifact with `urllib` and unpack
  it with `zipfile` (there is no `unzip` on this machine either). That keeps
  `.venv/` matching what CI has, which is the reason not to install `pip` into
  it to make the check easier.
- **`pip download` can report "No matching distribution found" for a version
  the JSON API already lists.** That is pip's cached index page, not a failed
  upload; `--no-cache-dir` resolves it. Check the JSON API before concluding an
  upload did not happen.

**Publishing needs the maintainer's credentials and is theirs to
run** — never attempt it, and do not assume a tag means a release went out, or
that `main` matches what is installable; check
[PyPI](https://pypi.org/project/visbench/) rather than this paragraph if it
matters. A version number on PyPI
can never be reused, so anything that renders wrong ships until the next
release: anything wrong in the README ships with it. Two separate checks cover
that, and neither replaces the other. **Rendering** is CI's `build` job —
`twine check dist/*` runs `readme_renderer`, which is what PyPI itself uses, so
a README that fails to render cannot reach a tag. **Relative paths** are
invisible to that check: they are valid markdown, render without complaint, and
merely point nowhere once the page is served from `pypi.org` rather than
GitHub. `tests/test_readme.py` is the guard for those, in the fast suite —
every link and image must be absolute. Do not "tidy" one back to relative;
point it at `.../blob/main/...`, or `raw.githubusercontent.com` for an image.
**`docs/` carries the opposite rule and the same file tests both** (7b/7d): it
is a Sphinx source tree, not package metadata, so its links must be *relative*
and must resolve, and none may escape the tree with `../` — which Sphinx cannot
follow and MyST does not warn about, so `-W` would not catch one.
Result schema is at **v8** (`training` added 2026-08-28; `pooling_requested`
in 6e-2b; `finetune` was 6a; `dataset_params` was 5j) and is **additive only**: never remove or repurpose
a field, or old records stop being readable.

### Layout worth knowing before editing

```text
visbench/
  backbones/     base.py (resolve_layers, _assemble), dinov2, clip,
                 timm_backbone, custom, pooling.py (feature modes)
  cache/         feature_cache.py (_Plan/_walk, extract_dataset, materialise)
                 streaming.py (CachedFeatures — a torch Dataset over the cache)
                 prefix_cache.py (PrefixCache — frozen prefixes, 6b; nests in
                   _prefix/ under the same root and is never counted as features)
  cli/           main.py (build_parser + the three commands),
                 datasets.py (ProbeSpec table: flags -> datasets, per probe)
  data/          detection.py (DetectionFolderDataset, load_voc_boxes,
                   VOC_CLASSES — boxes transform, they do not resample)
                 image_folder (+ balanced_subset), pair_dataset
                 (PairDataset, HomographyPairDataset, PairViewDataset),
                 triplet.py (TwoAFCDataset — NIGHTS-style 2AFC), dense.py
                 (DenseFolderDataset + stems= for official splits,
                  _init_geometry() — the crop, shared with taskonomy.py,
                  load_depth_map, load_normal_map, load_mask, load_label_map,
                  load_edge_map)
                 taskonomy.py (TaskonomyDataset — building-nested, indexed from
                   splits/*.csv; subclasses DenseFolderDataset for geometry only.
                   _DOMAIN_SPECS: per-domain loader, scale, invalid convention,
                   log_transform. load_valid_mask — mask_valid/, 6d-2)
                 derived.py (ShiTomasiResponse + OrientationResponse +
                   DerivedTargetDataset — the target computed from the image,
                   after the crop; 8a. OrientationResponse is a 2-ch direction)
                 bridges.py (TorchvisionDataset + HuggingFaceDataset — wrap a
                   torch/HF dataset; cache_identity from index-order immutability)
                 bsds.py (BSDS500Dataset — every annotator's boundary map, 4-9
                   per image; native resolution, NO resize or crop; target() is
                   the consensus mean and is NOT the scoring ground truth)
                 base.py (BaseDataset, list_files — scandir, never a stat/entry;
                   balanced_subset lives here now, not on ImageFolderDataset)
  heads/         base.py (register_head/build_head), linear.py, dpt.py,
                 detection.py (DetectionHead — cls + box branches, focal prior)
  metrics/       classification, retrieval, correspondence, similarity,
                 boundary.py (BSDS500's ODS/OIS/AP — thin_boundaries,
                   correspond_pixels (exact min-cost max-cardinality;
                   sparse, pads the SMALLER side), image_counts,
                   boundary_metrics. Reproduces published human ODS)
                 dense.py
                 (+ magnitude_metrics — per-image Pearson, masks NaN;
                    edge_metrics is it under the published key;
                    orientation_metrics — coherence-weighted angular error, deg)
                 detection.py (box_iou, average_precision, detection_metrics —
                   VOC protocol, dataset-level, difficult ignored not dropped)
  tasks/         base.py (BaseTask)
                 dense_base.py (DenseTrainingTask — shared by every dense probe;
                   pool_to_grid + evaluate_oracle — the recoverability gate,
                   opted into per probe, no backbone and no fitted head)
                 magnitude_base.py (DenseMagnitudeTask — edge/keypoints2d/
                   occlusion_edge; identity activation, masked L1, correlation)
                 schedule.py (warmup_cosine/check_schedule — probe3d's schedule,
                   shared by DenseTrainingTask and DetectionTask)
                 high_level/  classification, retrieval, semantic_segmentation,
                              detection (anchor-free, single-scale, 6c-3)
                 mid_level/   correspondence, depth, surface_normal,
                              generic_segmentation, similarity, occlusion_edge
                 low_level/   edge (6d-1), keypoints (Keypoint2DTask, 6d-2),
                              corner (CornerTask, 8a — derived target),
                              orientation (OrientationTask — derived, a
                              direction not a magnitude; own _activate/_loss)
  results/       schema.py (ResultRecord, SCHEMA_VERSION), writer.py
  viz/           styles.py (TargetStyle + the listed TARGET_STYLES table —
                   one row per drawable probe, style_for() raises otherwise)
                 colour.py (DisplayRange/display_range, target_to_rgb,
                   voc_palette, INVALID_RGB — pure, no I/O)
                 panels.py (render_probe_panels, render_panels, draw_boxes —
                   pastes at the dataset's own resolution, never resizes; 9a)
                 matches.py (render_match_panels, draw_matches, error_coherence
                   — the pair renderer; two views and the errors between; 9b)
                 gallery.py (annotate, render_sheet/_retrieval_panels/
                   _triplet_panels, class_balance, vote_balance — the probes
                   whose answer is a choice among images, not a map; 9c)
  demo.py        generated shapes + CustomBackbone(resnet18) — `visbench demo`
  runner.py      visbench.run() — the one call the CLI wraps
examples/        custom_backbone (the escape hatch + the RNG note),
                 classify, retrieve, correspond, depth, normals, segment,
                 segment_semantic, similarity, detect, edges, keypoints,
                 occlusion_edges, corners, orientation, save_probe (local),
                 push_probe (the Hub round trip; never uploads without --push),
                 show_panels (the viewer; no backbone without --predict-from)
docs/            conf.py, index.md, tasks.md (the per-probe reference AND the
                 ten generated tables), hub.md (sharing a trained probe),
                 show.md (the panel viewer + all 13 figures), roadmap.md,
                 _static/custom.css, _static/gallery/*.png (the figures —
                   inside docs/ because Sphinx cannot follow ../, and excluded
                   from the sdist)
                 — a Sphinx source tree; `_build/` is gitignored
.github/         workflows/{ci,slow,docs}.yml, ISSUE_TEMPLATE/, PR template
CITATION.cff     GitHub's cite button + what Zenodo archives (7e)
.zenodo.json     the deposit metadata Zenodo prefers over the .cff
```

### The CLI — add a probe by adding a row

`visbench/cli/datasets.py` holds one `ProbeSpec` per probe: its summary, the
folder layout it expects, the flags it adds, how those become `Splits`, and the
kwargs its constructor takes. That is a **table, not a hierarchy** — the nine
probes share flag *groups* (`_dense_flags`, `_split_flags`) but not a class
tree, because what they have in common is a set of options, not behaviour.

Two things the CLI must keep doing, both already tested:

- **Build the probe as an object, never by name with kwargs.** `run()` owns
  `batch_size` (extraction) and `device` (the backbone's), and every dense probe
  takes constructor arguments of those names meaning something else. Passing
  them through `run(**task_kwargs)` is a `TypeError`. The CLI keeps them apart
  as `--batch-size` and `--train-batch-size`.
- **`--limit` is per class on a labelled folder** (`balanced_subset`), by
  triplet on `TwoAFCDataset` (`max_triplets=`), by stem on a dense split, and a
  prefix on pairs. A plain prefix of an Imagenette split is entirely class 0 and
  scores 1.0 while measuring nothing.

### `DenseTrainingTask` — subclass this for a new dense task

`visbench/tasks/dense_base.py` holds everything a trained dense probe needs:
feature sources (in-memory dict *or* streaming `CachedFeatures`, normalised to
one indexable source), batching, head construction, the AdamW + warmup/cosine
schedule, the training loop, batch-wise `predict`/`evaluate`, and per-image
metric averaging. A subclass supplies only:

- `out_channels` — how many channels the head emits
- `_activate(raw)` — raw head output → prediction (applied in loss, metrics
  *and* `predict`, so those three can never disagree)
- `_loss(pred, target)` — both `(B, C, H, W)`
- `_batch_metrics(pred, target)` — must return **per-image averages**, which is
  what lets `evaluate` weight each batch by size and recover the split number
- `target_channels`, `display_name`, `target_noun`, `level`, `name`
- `target_dtype` if the target is not a float measurement — `long` for class
  indices, which is the one place a classification target leaves the path the
  other three share
- optionally `_task_params()` (extra `task_params` for the record) and
  `_on_epoch_start()` (per-epoch diagnostic hook)

`DepthTask` is 224 lines, `SurfaceNormalTask` 299,
`GenericSegmentationTask` 173 and `SemanticSegmentationTask` 186 because of
this — read them before writing a fifth. Between them they show a scalar target and a vector one; a
bin-expectation activation, a normalising one and a sigmoid; a protocol borrowed
wholesale from probe3d and one that only borrows its schedule. The base was
lifted out of a *working* `DepthTask` when the second task arrived, not
designed up front; extend it the same way, from a case that already runs.

### Decisions already paid for — do not re-litigate or re-derive

- **Fetch probe3d's real source before implementing any of its protocols.**
  Reconstructing depth from memory would have produced scalar regression
  instead of the 256-bin expectation, which is a materially different probe.
- **Not all of probe3d is MIT.** `evals/utils/metrics.py`, `losses.py` and
  `probes.py` are safe to follow. `evals/utils/correspondence.py` and
  `evals/models/croco_models/` are **CC BY-NC** and must never be copied — see
  `NOTICE`, which is the consolidated record.
- **Dense geometry**: image and target must survive the *same* resize and crop,
  applied by the dataset, and targets resample **nearest-neighbour**. Bilinear
  averages across depth discontinuities and turns a hole's zeros into a halo of
  plausible wrong values the valid mask no longer excludes. The correspondence
  task already paid for a misalignment bug once (recall@1px = 0.003).
- **Validity convention**: a pixel is invalid where the target is 0 (depth) or
  zero-length (normals). Cap out-of-range values by *marking them invalid*, not
  clamping — clamping trains and scores against a wall of fabricated values.
  **Label maps are the exception and shift by one**: for segmentation 0 is a
  real class (background) and an unlabelled pixel is *negative*. Reusing the
  depth convention there would discard every background pixel and train the
  probe to answer foreground everywhere. `SemanticSegmentationTask` inherits
  this: `IGNORE_INDEX = -1`, and it is what `cross_entropy(ignore_index=)` and
  the confusion matrix both mask on, so loss and metric drop the same pixels.
  **Edge maps are the third case and have no invalid value at all** (6d-1): 0
  means "no edge", a real reading covering most of a frame, so `edge_metrics`
  masks nothing. **`NaN` is the fourth, and exists because the third has no
  spare value** (6d-2): a magnitude map derived from Taskonomy's 3D
  reconstruction *does* have holes, and they hold a plain 0 that is
  indistinguishable from a real reading, so validity has to travel out of band.
  `NaN` is also the loud choice — it makes an unmasked loss `NaN` on the first
  step, where a fabricated 0 trains quietly and merely scores badly. Four
  targets, four conventions — check which one a new dense task needs rather
  than inheriting the nearest, and note that `depth_zbuffer` and `normal` on
  Taskonomy needed *none* of the new machinery because their existing in-band
  sentinels turned out to match `mask_valid/` pixel for pixel.
- **A label map must be read without mode conversion, and this is silent when
  wrong.** VOC's `SegmentationClass` PNGs are palette images (mode `P`) whose
  raw bytes *are* the class indices; `convert("L")` resolves the palette and
  turns classes `[0, 1, 15, 255]` into `[0, 38, 147, 220]`, which loads, trains
  and scores against labels that mean nothing. `load_label_map` therefore never
  converts, while `load_mask` must (it only asks "non-zero?"). The two cannot
  share that step, and **`load_mask` is wrong on a palette file** — VOC's void
  255 resolves to a light grey, i.e. foreground, and `ignore_index=255` never
  matches because it compares against the resolved value. Binarise
  `load_label_map` instead.
- **Two mIoUs, and they disagree by about 5 points.** Dataset-level (one
  confusion matrix over the split, ratios taken once) is what VOC, ADE20K and
  Cityscapes define and the only one comparable to published numbers;
  per-image-then-averaged is this codebase's rule everywhere else. Measured on
  VOC val with DINOv2-S: 0.732 against 0.683; with DINOv2-B, 0.753 against
  0.712. `SemanticSegmentationTask`
  reports both under distinct names and overrides `evaluate` to do it, because
  no weighted mean of per-batch ratios equals the ratio of the sums. Do not
  collapse them to one number.
- **Not every dense task gets to borrow probe3d.** It has no binary or semantic
  segmentation task, and no edge task, so `GenericSegmentationTask`,
  `SemanticSegmentationTask` and `EdgeTask` keep only its *optimiser* schedule
  and record `protocol: "visbench_binary_seg"` / `"visbench_semantic_seg"` /
  `"visbench_edge_regression"`. Do not let a record claim `"probe3d"` for a loss
  and metric that paper never defined; the whole value of the field is that it
  says what a number is comparable to. The same applies to *other* papers'
  protocols: the edge probe must not claim BSDS500's, which is a correspondence
  metric it does not implement.
- **The ten-epoch schedule assumes NYUv2-sized data.** Measured on 80 training
  images: 0.16 IoU at the defaults, 0.87 at `epochs=40, lr=5e-3`, identical
  features. That is underfitting, not a weak representation, and `train_loss`
  is what separates the two. Do not tune the defaults away from probe3d's —
  say so in the example instead, which `examples/segment.py` does. **At real
  scale the defaults are fine**: 1464 VOC training images reach 0.73 mIoU at
  ten epochs with `train_loss` 0.19, so the schedule is not the problem, small
  splits are.
- **Bigger is not better on every task, and this is the point of the library.**
  DINOv2-S beats DINOv2-B on mid-level similarity (0.870 vs 0.858), low-level
  edges (0.4558 vs 0.4481) and low-level keypoints (0.2356 vs 0.2248) while
  losing to it on semantic segmentation (0.732 vs 0.753), detection (0.213 vs
  0.262), occlusion edges (0.2924 vs 0.3167) and Taskonomy depth (d1 0.5832 vs
  0.5986). Do not "sanity check" a new task by asking whether the larger model
  won. **And a task can disagree with itself**: on Taskonomy normals DINOv2-S
  wins on mean angular error while DINOv2-B wins on the 11.25° threshold, so
  quoting one and dropping the other manufactures a result.

- **What the corpus says is in
  [`CORPUS_FINDINGS.md`](CORPUS_FINDINGS.md), and you must read it before
  quoting any board.** Eight findings moved there on 2026-08-20 when this file
  passed the context limit it is loaded under. The claims are below; the
  evidence, the numbers and the readings each one corrected are in that file.
  `scripts/analyse_board_correlates.py` reproduces the correlational ones.

  - **"Which backbone is best" is not a well-formed question against this
    corpus.** `mae_vitb16` is first on six of the sixteen boards and last on
    four. A summary that picks a winner is discarding the result.
  - **A count over a corpus is a fact about that corpus, not about a
    backbone.** Three of MAE's counts have now moved without its features
    changing — twice because a column was added, once because a *board* was.
    Re-read counts off `LEADERBOARD.md`.
  - **Quote an objective gap against the *recipe* gap on the same board, never
    against zero.** `sam_vitb16` and `supervised_vitb16` share architecture,
    data, labels and normalisation and differ only in training recipe; on seven
    of the thirteen boards that gap is more than a third of the whole objective
    spread. Nothing in a record says which board you are on.
  - **The semantic segmentation board separates neither training objectives nor
    feature resolution**, which every other dense board ranks by. Do not
    present it as evidence about an objective.
  - **The high-level tier is two clusters, not one** — `classification`/
    `retrieval` (image-level categorisation) and `detection`/
    `semantic_segmentation`/`scene_classification`/`fine_grained_classification`
    (localised / spatial-context prediction) — that barely correlate with each
    other. **Two probes that are mechanically object classification with a
    different folder both land in the *localised* cluster**, which is the
    replication that makes this a property of the cluster rather than a fact
    about Places365: `fine_grained_classification`'s strongest partner in the
    whole corpus is `detection` at **+0.860**, against +0.343 with the object
    board it subclasses. The tier-mean-vs-
    cross-tier sign has flipped both ways with corpus composition (below the
    line at 13 boards, marginally above at 14) and is noise; the two clusters
    are the stable finding. `scene_classification` is image-level classification
    yet lands with the localised cluster (+0.72 with `detection`, −0.22 with
    `retrieval`). Treat `high_level` as a folder, not a quantity to average
    over. Mid- and low-level cohere. This is not the taxonomy being wrong.
  - **That clustering is not a shared-dataset artefact**, checked: the two
    boards reading the *same 1449 images* agree least of the three VOC pairs,
    and Imagenette's three probes average +0.128.
  - **Quote `detection` to three decimals, not four.** It is GPU
    non-determinism a discrete metric can see, only on the two 16x16-grid
    backbones, and there is nothing to fix.
  - **Two backbones' high-level scores are close to in-distribution recall**,
    not transfer: `convnext_base` and `supervised_vitb16` are ImageNet-1k
    supervised and Imagenette's classes are ImageNet-1k wnids.
  - **Feature resolution is the strongest correlate of every dense board, and
    it is not what DINOv2's lead is made of.** Holding weights fixed and
    cutting DINOv2-B from 256 to 196 tokens costs under 3% on all five dense
    boards, and it keeps its lead over the whole ViT-B/16 pack on both boards
    it led — 21% of the `generic_segmentation` gap and 7% of the `depth` one.
    On the other three boards DINOv2-B never led, so there was nothing to
    explain. **Check who leads a board before explaining their lead.**
  - **A control is rankable and still must not be listed beside the corpus.**
    `results/controls/` holds records that pass `comparability_key` against
    their board and answer a different question from it — the corpus says what
    a backbone scores, a control says what changes when one thing about one
    backbone moves. Nothing there feeds a generated table.
  - **n=12.** Every correlation above has wide error bars.

- **A trained probe records how its fit went, and the reason it took until
  schema v8 is worth keeping** (2026-08-28). Every trained probe computed
  `train_loss` — and the classification family `train_top1` — printed it to a
  log line, and dropped it before the record. So the corpus could not answer
  the one question this file says matters most about a low score: whether the
  probe **underfitted**, which *understates* a backbone, or whether the
  representation genuinely does not carry the answer. Those are opposite
  conclusions from the same number, and the binary-segmentation bullet above is
  the proof — 0.16 IoU at the defaults against 0.87 at `epochs=40`, identical
  features.

  Found by friction, not by audit: the CUB write-up could claim "does not
  underfit" for the six backbones run by hand and could not check it for the
  six run on the cluster, because their `train_top1` existed only in a Slurm
  log. 156 trained records, none of them able to answer it.

  **`training` is a separate field, not entries in `metrics`.** `metrics` is
  what `evaluate()` returned about the *evaluation* split, and every leaderboard
  path reads it; a training number there is one the ranking code can only refuse.
  `DIAGNOSTIC_METRICS` does exist and would have worked, which is why this was a
  real choice rather than an obvious one — it was rejected because it blurs what
  `metrics` means and widens the key-collision surface the `ceiling_` guard
  exists for. **An open dict**, for `task_params`' stated reason: a future
  probe's own diagnostic must not force another bump. **`None`, not `{}`,** for
  the three zero-shot probes — no fit happened, which is a different statement
  from "trained and reported nothing", and it is what every pre-v8 record
  carries by absence, exactly as `finetune` does.

  **Never rank on it.** A probe that fits its training data perfectly has said
  nothing yet about a backbone; on CUB every backbone reaches `train_top1`
  1.0000, including the one that comes last.

- **`run()` seeds before it constructs a *name*, so a `CustomBackbone` is
  constructed outside the seeded window — and that makes it more reproducible,
  not less** (`examples/custom_backbone.py`, 2026-08-19). The standing bullet
  below says to pass a backbone's name and take the object back off
  `RunResult.backbone`. On the custom path that advice is unavailable: a
  registry name cannot carry an `nn.Module`, so the caller must construct.

  Measured on generated data with features that are **bit-identical** (max
  absolute difference 0.0), classification top-1:

  | path | seeds 0-4 | spread |
  | --- | --- | --- |
  | `CustomBackbone` | 0.9125 0.9125 0.9125 0.9187 0.9125 | 0.0062 |
  | `run("resnet18")` | 0.9062 0.9125 0.9062 0.9062 0.9125 | 0.0063 |

  **The wrapped path is perfectly reproducible**, and unchanged by RNG consumed
  before `run()` is called — because construction happens *before* `set_seed`
  rather than after it, so nothing the caller did can reach the head. The
  between-path gap of 0.0062 is **the same size as each path's own seed-to-seed
  spread**, so it is jitter rather than a cost of wrapping. Zero-shot probes are
  identical bit for bit (retrieval 0.603730 both), since no head is fitted.

  So a wrapped model's trained number is comparable with another number from the
  same wrapped model, and not with a registered backbone's to the last decimal.
  **Do not read this as "the custom path is unreliable"** — that is the
  intuition it was written to correct.

- **`dgx1` was degraded on 2026-08-19 and it does not fail like a broken node**
  (found while running 10c). A job submitted there hung for 30 minutes with no
  output and no error; the first read was that the new backbone was broken.
  `mae_vitb16` — which had run fine five days earlier — hung identically, and
  forcing HF offline changed nothing, which ruled out both the registration and
  the network. `python -X importtime` is what named it: on dgx1 **`import torch`
  spends 1–1.6 seconds per submodule** and never finishes inside 300 s, while
  dgx2 imports the same venv in 2.4 s. Nothing in Slurm reports the node as
  unhealthy, so it accepts work and starves it.

  Submit with **`--exclude=dgx1`** until this clears, and when a job on this
  cluster hangs with an empty log, time an `import torch` on the node before
  suspecting the code. The venv is still only valid on `dgx1`/`dgx2`, so the
  usable set is currently one node.

- **The corpus matrix is defined in two files, and one of them was short by a
  probe for a whole release** (10b). `slurm/corpus.sbatch`'s `PROBES` had twelve
  entries where `scripts/build_corpus.sh`'s `ALL_PROBES` has thirteen: `corner`
  shipped in 8a/8b, was added to one file and not the other, and was therefore
  **unschedulable by the array job** through v0.9.0 however wide the `--array`.

  **The guard that exists to prevent exactly this could not see it.** It sizes
  the matrix as `len(PROBES) * len(BACKBONES)` and refuses a range that does not
  match — but it reads its *own* short list, so a twelve-probe matrix is
  self-consistent and the corpus it produces looks complete, because every group
  present still holds every backbone. **A guard that derives its expectation
  from the same data it is checking is not a guard.** The check has to come from
  somewhere else, which is what `tests/scripts/test_corpus_scripts.py` is: the
  two shell arrays against each other *and* against `list_probes()`.

  The same guard also accepted any range ending at the right index —
  `--array=3-38` on a 39-task matrix passed while omitting the first probe
  entirely, and a strided range passed however much it skipped. It reads
  `SLURM_ARRAY_TASK_COUNT` as well now; an endpoint cannot see a hole in the
  middle.

- **Ask whether a new probe *ranks*, not whether its number is high.** A low
  absolute score can be by design — detection's 0.21 mAP is, because a
  single-scale head has no pyramid. What is never acceptable is failing to
  separate two backbones, and that is the question a suspiciously low number
  should prompt. 6d-2 nearly shipped an occlusion-edge probe scoring 0.088 with
  an S-versus-B gap of 0.0035; the score alone looked like "a hard task", and
  the gap is what showed it was measuring nothing.
- **A second dataset for an existing probe needs a new probe *name*, not a
  flag** (`scene_classification`, 2026-08-27). `scripts/render_tables.py::board_for`
  renders exactly one table per task and **refuses a task with more than one
  comparability group**, and `comparability_key` groups by dataset name and
  fingerprint — so a Places365 record under `task="classification"` does not
  merge with the Imagenette board, it makes that board *unrenderable*. Scene
  classification is therefore `SceneClassificationTask(ClassificationTask)` with
  `self.name = "scene_classification"` and a `protocol` string, exactly as
  `corner` is a renamed `DenseMagnitudeTask`. Adding the name means touching
  every fixed table that a `test_show_command.py` assertion pins equal to
  `list_probes()`: `_REGISTRATION_MODULES`, `HEADLINE_METRICS`, the CLI `SPECS`
  row, `TARGET_STYLES`, both corpus-script probe arrays, the gallery `figures()`
  map and a committed `docs/_static/gallery/<name>.png`, and
  `analyse_board_correlates.py`'s copied `HEADLINE_METRICS`. Then the board step
  adds `analyse_board_correlates.py`'s `SOURCE_IMAGES` (hand-written, a test
  fails without it), a `docs/tasks.md` board marker, and — because a fourteenth
  high-level board shifts every tier and source correlation — a deliberate
  rewrite of the tier-coherence test and finding. A probe that is "just a
  dataset swap" is still a dozen small edits plus a corpus analysis, because
  the name is load-bearing everywhere and the board is not inert.

  **Confirmed a second time by `fine_grained_classification`** (CUB-200-2011,
  2026-08-28), which followed that edit list exactly and needed nothing not on
  it — so treat the list as complete rather than re-deriving it. Three probes
  now share one implementation and ask three questions, which is the point:
  `classification` is basic-level, `scene_classification` is place, and
  `fine_grained_classification` is subordinate. A test pins all three
  identities *and* that their three `protocol` strings are distinct, since the
  failure mode is two of them collapsing into one board.

  One thing that list did *not* cover, found by the sixteenth probe: **the
  gallery had a flat 4 MB size budget, and a new figure failed it by existing.**
  The gallery was at 4.03 MB with a perfectly ordinary 220 KB page, and these
  are photographs — ~80k distinct colours, so lossless re-encoding buys under
  1% and there was nothing to shrink. The guard was also not expressing what it
  documents: a *total* budget lets one page rendered at four times its intended
  size pass while there is slack, then fires on someone else's reasonable
  figure later. It is per figure now (`MAX_FIGURE_BYTES`), with the total scaled
  by `len(list_probes())`. **Raising a budget to make a guard pass is usually
  wrong; check first whether the budget was measuring the right thing.**
- **The NIGHTS ImageNet split is a contamination check, not a subset.**
  `test_imagenet` and `test_no_imagenet` partition the test set by whether the
  reference image came from ImageNet. DINOv2-S scores 0.882 against 0.854 across
  them; quoting the combined 0.870 without that gap overstates how much of it is
  perceptual alignment.
- **A triplet task is a flat image dataset plus indices, not a widened cache.**
  `TwoAFCDataset` presents unique images and puts the triplet structure in
  `labels()` as `(ref, left, right, vote)` indices into itself. That keeps the
  cache, the fingerprint and `run()` unchanged, extracts a shared image once,
  and makes the pairing travel by index. It is also why `subset()` is refused
  there — slicing images would silently repoint every triplet — and why
  `max_triplets=` exists on the constructor instead.
- **The mid-level similarity protocol is zero-shot, despite what its own paper's
  README says.** `evaluate_model_percepture.py` builds a test loader, freezes
  the backbone and compares two cosine similarities; nothing is trained. The
  README says otherwise. Follow the code, and do not add a head "to match the
  description".
- **Read a CSV by column name.** The reference reads the vote as `iloc[idx, 2]`
  and the paths as `4`/`5`/`6`. Reordering the file would silently score
  against the wrong column, and the failure looks like a mediocre number rather
  than an error.
- **A pair task is a flat image dataset plus interleaving, not a widened cache.**
  Same resolution as the triplet one. `PairViewDataset` presents a
  `PairDataset`'s two views as `2N` single images — item `2i` and `2i+1` are
  pair `i` — so extraction, batching, streaming and the identity memo need no
  change, and `regroup` restores the pairing. `run()` forks on the task's
  declared `uses_pairs`, one explicit branch rather than a general "dataset
  adapter" mechanism built to fit a single case. **Both directions stay lazy**:
  materialising the pairs would pull a whole split of dense features back into
  memory, undoing the streaming that had just written them to disk.
- **`view_identity` is the reason a cached correspondence run is fast, and it
  had no caller for a year.** The two views of a pair come from one file and
  would otherwise share a `cache_identity`, so the memo would serve view 1 the
  features of view 0 — trivially perfect matches, no error. It existed and was
  tested from v0.1; `examples/correspond.py` passed the cache a bare list of PIL
  images, which has no identity, so a fully cached run still decoded, cropped and
  warped everything. 16.4 s cold against 8.2 s warm on 200 pairs, once `run()`
  used it. **A declared-but-uncalled mechanism is the same failure as the
  QuickGELU guard**: it passes its own tests forever while doing nothing.
- **A ceiling travels with its score, through `BaseTask.context_metrics` — for
  correspondence and, since 2026-09-01, for every dense probe that declares an
  oracle.** A match can only land on a patch centre, so a
  coarse grid has a hard floor on achievable precision: `ceiling_recall@5px` is
  ~0.10 on a 7x7 grid against ~0.41 on a 16x16 one. The score alone therefore
  says the wrong thing. Measured on 200
  Imagenette pairs, DINOv2-S: `recall@5px` 0.3049 against a ceiling of 0.4123.

  **The dense probes have the same problem and now say so.** The head reads one
  feature vector per patch, so part of every dense target is out of reach before
  the backbone is chosen, and how much varies by *backbone*: `corner`'s ceiling
  is 0.8316 on a 16x16 grid and 0.6685 on a ResNet's 7x7. Ranking those two
  against each other silently invites a reader to attribute a grid difference to
  a representation. `edge`, `keypoints2d`, `occlusion_edge`, `corner` and
  `orientation` emit `ceiling_*`; every other dense probe still returns `{}`,
  because pooling a class-index or bin-expectation target is meaningless.

  Everything else is unchanged: `run()` refuses a context key that
  collides with a score, since they share one flat dict, and both prefix
  `ceiling_`. **Never rank or average on a ceiling** — it says what was
  available, not what was recovered, and since it falls with the grid, ranking
  on it would rank feature resolution directly.

  **The corpus carries them since 2026-09-01.** The five boards were re-run —
  60 records, every value produced by a run rather than backfilled, which is the
  distinction that matters: a number in a record no run produced would be a
  fabrication however easy it is to compute. The old lines stay (the corpus is
  append-only) and `latest_per_backbone` picks the new ones, so the file is 252
  lines for 16 boards x 12 backbones. Those records also gained the schema-v8
  `training` block, because they predated it. Schema is untouched by the
  ceilings themselves: they are keys inside `metrics`, not a new field.

  **Four of the five boards reproduced to ~1e-7 relative; `orientation` did
  not.** See its entry in [`CORPUS_FINDINGS.md`](CORPUS_FINDINGS.md) — its
  metric is ill-conditioned and two of its rows are not separable.
- **Correspondence thresholds are in *pixels*, and normalising them by patch
  spacing is the bug v0.6.1 fixed — do not reinstate it.** Patch widths look
  like the natural unit (they are the quantisation floor) and are fine within
  one backbone. Across backbones a patch is 14px on DINOv2/14 and 32px on a
  ResNet, so `recall@1p` asks each a different question. It inverted the
  published board: `resnet18` 0.8927 against DINOv2-S 0.7834 in patch widths,
  and 0.0973 against 0.3049 in pixels — first and last place swapped. The floor
  is stated by `ceiling_`, not divided out.
- **The dataset half of a run gets `dataset_params`, like the task half gets
  `task_params`** (schema v5). Filled from whatever `describe()` returns beyond
  the record's own fields, so `max_warp`, `image_size` and `num_triplets` land
  there without a per-setting column. Before this they changed the fingerprint
  and nothing else — two runs were distinguishable only as "not the same data".
- **Shorten a labelled folder with `balanced_subset(n)`, not `subset(n)`.**
  The file list is grouped by class, so a prefix is entirely class 0 and a
  single-class retrieval scores 1.0 while measuring nothing. Two examples
  carried their own copy of this before it became a method.
- **Shorten a split with `dataset.subset()`, never by slicing its attributes.**
  A dense dataset carries three index-parallel lists and slicing one alone pairs
  a target with the wrong image, silently, since every later step still sees
  equal lengths. Subclasses declare `_parallel_attrs`; the base reindexes them
  together and the fingerprint follows, so a limited run cannot be mistaken for
  a full one. The CLI (5j) needs exactly this — do not reinvent it there.
- **Per image, then averaged.** Never pool every pixel of the split; that lets
  uneven hole coverage silently reweight the dataset.
- **Dense features stream.** ~250x the size of pooled ones (24k NYUv2 images at
  DINOv2-B is ~19 GB). `run()` streams automatically for `uses_dense` tasks.
  Measured: 10.8 GB peak RSS in memory vs 1.7 GB streaming for 0.63 GB of
  features. `CachedFeatures` is random-access, not a generator — training
  reshuffles every epoch and a generator can only shuffle *within* a batch.
- **Targets travel by index, never by iteration order**, or they drift from
  their features the moment a loader shuffles. Silent failure: it still trains.
- **probe3d's uncertainty-aware angular loss can switch itself off** near chance
  accuracy. `SurfaceNormalTask.fit` documents the measured dynamics, detects it
  and warns. The loss is deliberately left as probe3d wrote it — silently
  substituting the plain one would make VisBench's numbers incomparable with
  the published ones, which is the only reason to borrow a protocol at all.
- **mypy's `python_version` tracks the newest syntax any *dependency stub* uses,
  not the package's floor.** It is pinned to 3.12 in `pyproject.toml` because
  numpy 2.x uses PEP 695 `type` statements. Do not "fix" it down to the floor.
- **A fine-tuned number and a frozen one are different measurements, and the
  record is what keeps them apart.** Frozen asks what a representation already
  carries; fine-tuned asks what it can be adapted into. Every published VisBench
  number is frozen. Schema v6's `finetune` field is `None` for those and a dict
  otherwise — never rank or average across it. The trainable forward pass is a
  **separate entry point** (`extract_features_trainable`), not a flag on
  `extract_features`, because the cache depends on getting detached tensors and
  a keyword defaulting to the safe value puts the expensive mistake one typo
  away. The unfrozen backbone **stays in `eval()`**: train mode would start
  BatchNorm updating and dropout firing, moving a fine-tuned number for two
  reasons at once with one of them unrecorded.
- **A wall clock is not a metric — repeat it before concluding anything from
  it.** Every score in this codebase is deterministic and reproduces to four
  decimals across runs, which makes it tempting to treat a `duration_seconds`
  from the same record as equally solid. It is not: 6a timed one frozen/
  fine-tuned pair, got 252 s against 238 s, and recorded "fine-tuning is not
  slower" in three files and a merged PR. Re-running the identical commands
  gave 156 s and 126 s frozen against 200 s fine-tuned — same metrics to the
  digit, opposite conclusion. The machine is shared. Run a timing at least
  twice, and prefer the *repeat* to the first, since the first also pays for
  whatever the page cache had evicted.
- **Constructing a backbone draws from the global RNG, and `run()` seeds
  *before* it constructs.** So `run("dinov2_vits14", ...)` and
  `run(get_backbone("dinov2_vits14"), ...)` fit the head from different RNG
  states and produce different trained numbers, with every recorded field —
  seed included — identical. **Pass the name; take the object back off
  `RunResult.backbone`.** Building one outside `run()` puts its random init
  (DINOv2 and timm both initialise randomly before loading the state dict)
  outside the seeded window.

  This shipped in `--push-to`, which built the backbone early so it could hand
  the same object to `push_probe`. Found by publishing a full board and diffing
  it against the corpus: 20 of 26 records differed and **the 6 that reproduced
  were exactly the zero-shot probes**, which train no head. That signature —
  trained probes all move, zero-shot ones all reproduce, no recorded field
  explains it — means seeding, not a version regression, and it was misread as
  the latter first because the corpus was written under an older version.

  **The obvious regression test for this is vacuous, and mutation testing is
  the only thing that says so.** Comparing a pushed run's metrics against an
  unpushed one looks decisive and is not: the CLI fixtures are three
  colour-separable classes, so both sides read 1.0 however badly the RNG is
  threaded. Pin the backbone's *weights* against a freshly seeded
  construction — that is what the seed decides and what actually moved.

  **The whole board was re-run and republished after the fix, and 24 of the 26
  DINOv2 records reproduce the corpus exactly** — including all three rankings
  the bug had inverted (`edge` S, `keypoints2d` S, `corner` B). The two
  exceptions are both `detection`, and only in the fourth decimal: `map_50`
  0.2291 against the corpus's 0.2285 on ViT-S, 0.2897 against 0.2895 on ViT-B,
  with the S-versus-B ordering unmoved. That is a property of the detection
  probe and not of the seeding fix — **diagnosed 2026-08-13, see the next
  bullet.**

- **A backbone that changes its configuration must change its *name*, or a
  leaderboard silently deletes the row it was built to be compared against**
  (the resolution control, 2026-08-21). `latest_per_backbone` keys on
  `record.backbone` and keeps the newest, which is right for a re-run and
  catastrophic for a reconfiguration: `DINOv2.__init__` set `self.name =
  variant`, so the same weights at 196px reported `dinov2_vitb14` and would
  have evicted the corpus's 224px number from all five boards it touched. Not a
  wrong row — a **deleted** one, with no rendered field saying the
  configuration had moved.

  Three things now stop it, and the middle one is the guard: `DINOv2` takes a
  `name=`; **`latest_per_backbone` raises when one name arrives under two
  `backbone_key`s**, the posture `METRIC_DIRECTIONS` and `style_for` already
  take; and `register_backbone`/`register_task` take `name` positional-only, so
  a decorator parameter cannot shadow a constructor argument of the same name.

  The general rule: **pass a distinct name whenever you change `image_size`,
  `hub_ref` or `checkpoint`.** `backbone_key` already separates the cache
  correctly — it carried the resolution all along — so the cache was never at
  risk and the *record* was. Two mechanisms that look like one.

- **Verify with the exact commands CI runs** (below). A local env with extra
  packages installed will pass checks that CI fails.
- **A guard whose only test is `slow` is a guard CI never runs.** `addopts`
  deselects `slow`, and CI runs a plain `pytest`, so the entire
  weight-downloading suite is invisible to it. The CLIP QuickGELU check filtered
  on a phrase open_clip has never emitted and was dead code for its whole life;
  its test existed, failed correctly, and never ran. When a check exists to stop
  a *silently wrong number*, give it a test in the fast suite — extract the
  logic to a pure helper if that is what it takes.

- **The Python floor is 3.10 because DINOv2 requires it, and that was the
  cheaper of two bad options.** The pinned `HUB_REF` uses `float | None` at
  class-body scope, which 3.9 evaluates at import and rejects — so DINOv2, six
  of seven `examples/`, and every slow test were broken on the declared floor
  (#1). The alternative, repinning `HUB_REF` to a 3.9-compatible commit, would
  have **invalidated every cached DINOv2 feature on every machine**, since
  `HUB_REF` feeds `cache_key()`. Raising the floor keeps the ref and therefore
  the caches: verified identical keys before and after. Do not lower it back
  without checking DINOv2 still imports.

- **The demo's score is deliberately not 1.0, and colour deliberately carries no
  information** (7a). A first pass with fixed colours and centred shapes scored a
  flat 1.0 — the saturation this project rejects everywhere else, and the reason
  Imagenette classification was refused as fine-tuning's proof. Foreground and
  background now share a base and the contrast offset's *sign* is random, so a
  colour shortcut cannot be reported as shape recognition; a fast test asserts
  colour does not separate the classes. `--noise` walks top-1 from 0.975 to
  0.312 across 28→90, and a slow test pins that slide rather than only the
  headline number: a probe whose score does not move when the signal is
  destroyed is not measuring the signal. Nothing in the demo is special-cased —
  same `run()`, same cache, same record — so "fixing" it with a bespoke path
  would make it stop demonstrating the library.

- **A derived target is the cheapest kind to add and the easiest to fool
  yourself with** (8a). Corner detection computes its target from the RGB frame,
  so it needs no dataset — and three things had to be measured before it was
  worth shipping, none of which a probe run would have revealed on its own.

  **Check the tail, and check it before writing the task.** Every raw corner
  response was more concentrated than `edge_occlusion`'s 0.46, the case that
  scored 0.088 and ranked nothing: Harris `R` clipped at 0 is 0.52, `|R|` is
  0.33, Shi-Tomasi's λ_min is 0.27. `log1p(1e4·λ_min)` brings it to 0.089 with a
  frame mean of 0.593, which satisfies 6d-2's tail rule and 6d-1's order-1 rule
  at one setting. **Shi-Tomasi rather than Harris** because λ_min is
  non-negative by construction and has no `k`.

  **Check the overlap with what already ships, which nothing in the codebase
  previously asked for.** The corner target correlates **0.52** with
  `edge_texture` and 0.27 with `keypoints2d`, where those two correlate **0.147**
  with each other — so the new target is more redundant with an existing one
  than the two existing ones are with each other. The overlap is *intrinsic*: it
  holds at 0.46–0.54 across eight transforms including near-linear ones, because
  a corner is a pixel whose gradient is large in two directions and an edge map
  is gradient magnitude. A first pass blamed the `log1p` for it and was wrong.

  **A correlated target can still rank differently, and that is the criterion.**
  Spread over six backbones is 0.1603 against edge's 0.1136, and CLIP-B/16 is
  first on edges and third on corners. Had the ordering matched, the probe
  should not have shipped.

  **Do not read one pair as a failure to rank.** DINOv2-S and B differ by 0.0014
  here, which looks like the occlusion-edge failure and is not: that probe was
  flat across *all six*, and the edge probe's own top two differ by 0.0007. Ask
  about the spread over the full set.

  **Computing the target after the crop deletes the alignment hazard rather
  than testing for it.** There is no second geometry and no resampling of the
  response — the single strongest property of this class of target, and the
  reason `DerivedTargetDataset` does not subclass `DenseFolderDataset`.

- **The gauntlet asks whether a target is distinctive; it never asked whether
  it is *recoverable*. Photometric superpixels is what that cost** (built and
  rejected 2026-08-28). SLIC boundary regression passed every gate — tail 0.055
  against `edge_occlusion`'s 0.46, overlap with `edge_texture` 0.267 per image
  against the 0.52 `corner` shipped with, cross-image `|r|` 0.044 (below the
  edge target's own 0.060, so the boundaries followed the image and not SLIC's
  seeding lattice), and two rival formulations rejected on the overlap rule.
  Then it scored **0.0434 / 0.0209 / 0.0238** on DINOv2-S, CLIP-B/16 and
  ResNet-50, where the *weakest* shipped low-level probe (`keypoints2d`) scores
  0.179-0.236 and `corner` scores 0.492-0.651. Spread 0.023, with ResNet-50
  "beating" CLIP by 0.003, and `train_loss` **lowest** for the worst scorers —
  the heads learned the mean boundary density and nothing about location.

  **The missing check was an oracle, and it now ships** (2026-09-01).
  `DenseTrainingTask.evaluate_oracle` pools the target to the feature grid,
  upsamples it back and scores it with the probe's own metric — what a perfect
  backbone would make available, since a dense probe sees one feature vector per
  patch and signal finer than a patch is *absent from its input* rather than
  merely hard to predict. No backbone, no features, no fitted head, so it costs
  one pass over a split rather than a board.
  `CorrespondenceTask.evaluate_ceiling` is the same idea, arrived at the same
  way. **Run `scripts/oracle_ceiling.py` before writing the next derived
  task**, and see the "oracle gate" section of
  `visbench/tasks/low_level/README.md` for the numbers.

  **The bar, calibrated against this rejection**, over the pinned 600 val frames
  at a 16x16 grid: the four shipped magnitude targets score 0.53–0.83 and
  photometric superpixels scores **0.25**. At a ResNet's 7x7 grid, 0.43–0.67
  against 0.11. Three things about it that are not obvious:

  - **A probe opts in**, `TARGET_STYLES`-style, and every other dense probe
    raises. Pooling is the right bottleneck only for a target that averages —
    the mean of classes 1 and 15 is class 8 — and a silently defaulting oracle
    would return a confident number about nothing, which is worse than none for
    a gate whose job is to stop work.
  - **The upsample is bilinear because `LinearHead`'s is**, so the gate is never
    more permissive than the heads it protects. Even a target built from hard
    grid cells scores ~0.88 rather than 1.0.
  - **It is a bar, never a denominator.** Unlike `evaluate_ceiling` it is an
    achievable score rather than a proven bound, and the ratio does not
    discriminate anyway: `corner` reaches 80% of its oracle and `keypoints2d`
    41%, and both rank backbones fine.
  - **It models a *linear* head exactly, and a DPT head can beat it —
    measured, 2026-09-01.** `LinearHead` is a 1x1 convolution per patch plus a
    bilinear upsample, which is literally what the oracle computes; a DPT head
    decodes progressively and places structure *within* a patch. Across five
    probes x two backbones a DPT head reaches **70-104%** of the oracle and
    **exceeds it in two of ten cases** (`results/controls/dpt_head.jsonl`). So
    it is a bar for the head VisBench reports, **not a bound on what is
    achievable** — do not call it a ceiling a better head cannot pass.

    **It does not reopen BSDS500**: scaling that line's 0.4193 linear ceiling by
    the best ratio observed (1.037) gives ~0.435 ODS, still below Canny's 0.60,
    so the closure survives the correction to its premise.

    **And a head is not a neutral magnifying glass.** On `occlusion_edge` the
    DPT run *reverses* the linear board's top two. That is the demonstration
    behind the standing rule to report the linear number when comparing
    representations.

  **It has now refused something** (2026-09-01). The BSDS500 probe was not built
  because the gate put a linear probe's ceiling at **0.4193 ODS** on the 16x16
  grid every corpus backbone produces, against published detectors at 0.60-0.79
  and human agreement at 0.80. That cost one 60-second run instead of a
  12-backbone board. **Do not read that 0.42 against the 0.25 that rejected
  superpixels** — one is ODS and the other Pearson correlation, they are not
  comparable, and an earlier draft made exactly that mistake.

  **A pooled-resolution overlap check is not that test, and nearly became a
  false veto.** The boundary map reads 0.267 against `edge` at full resolution
  and 0.684 pooled to a 16x16 grid, which looked decisive — until it was
  calibrated: the shipped `corner` target reads **0.781** there and its board
  ranks backbones differently from `edge` anyway. **Calibrate a new rejection
  criterion against something that already passed before letting it reject
  anything.**

  What survived: `DerivedTargetDataset` memoises computed targets now
  (`MEMO_LIMIT`), because `CachedFeatures.__getitem__` calls
  `dataset.target(index)` on every access — a ten-epoch streaming run had been
  recomputing every target ten times, which `corner` and `orientation` were
  both paying.

- **The overlap check is a veto, and `orientation` is the probe that proves it
  earns its keep** (2026-08-28). DoG-blob detection was the obvious next derived
  probe — the scale-space counterpart to `corner`. Its pre-measurement (the same
  afternoon of correlations `corner` established): tail@1% ≈ 0.084 raw, so *no
  compression needed*, which passed. But per-image `|r|` with `edge_texture` was
  0.50 and **with `corner` 0.51** — as redundant with an existing probe as
  `corner` is with `edge`. Rejected without a probe run: the check exists so you
  do not spend a per-backbone board to discover redundancy.

  **Structure-tensor orientation was the alternative and it pre-measures
  clean.** `|r|` under 0.09 with both `corner` and `edge`, because it measures
  *phase* and no other probe does. Its target is a *direction* — the unit vector
  `(cos 2θ, sin 2θ)`, the angle taken mod π so the double angle handles the wrap
  — with the coherence `(λ_max−λ_min)/(λ_max+λ_min)` folded into its length. So
  it is the first derived probe that could **not** reuse `DenseMagnitudeTask`:
  it needs a 2-channel L2-normalising `_activate`, a coherence-weighted angular
  `_loss`, and `orientation_metrics` (degrees, `orientation_error` halved so 45
  is chance). Coherence is a **weight, not a mask** — only 1.4% of Taskonomy
  tiny val pixels fall below 0.1 — folded into the target length exactly as a
  zero-length normal marks an invalid pixel, so the loss and metric both weight
  by `target.norm(dim=1)`. An angle has **no tail**, so the compression `corner`
  needed is absent here; the pre-measurement confirmed that before the task was
  written. Proved end to end on DINOv2-S: `orientation_error` 35° against the
  45° floor on 40 training frames. The 12-backbone board is the next step and
  reuses `corner`'s pinned `data/corner_frames/` set.

  Viz: `orientation` is drawn in **colour**, not greyscale — a new `"orientation"`
  `Kind` whose `_orientation` colouriser maps `2θ` to hue and coherence to
  brightness (inline HSV→RGB, no new dependency), so a flat patch reads as black
  rather than a confident wrong colour.

- **A viewer that applies its own geometry is worse than no viewer** (9a). This
  is the single rule `visbench/viz/` exists to keep, and it inverts the usual
  cost/benefit: a panel's entire evidential content is whether the image and the
  target line up, so a viewer that resizes for layout, re-reads the source file
  or re-crops can make a *misaligned pipeline look fine and a correct one look
  broken*. It is guaranteed by pasting `np.asarray(dataset[i][0])` unchanged,
  which is cheap only because dense datasets already yield a PIL image at the
  working resolution rather than a normalised tensor — there is nothing to
  invert. A fast test pins the image panel byte-for-byte.

  **Four validity conventions, one listed table, no fallback.** The four
  conventions in the bullet above are invisible in a tensor's shape or dtype, so
  `TARGET_STYLES` is keyed per probe and `style_for` raises on an unlisted one —
  the posture `METRIC_DIRECTIONS` takes, for the same reason. A "scalar map,
  mask the zeros" default is right for depth and silently wrong for the four
  probes where 0 is a real reading, and it *renders*: the panel comes out
  looking like a target full of holes. There is a test per convention.

  **A prediction is drawn against the target's range, not its own.** Scaling
  each panel to its own extremes is the obvious implementation and it hides the
  most common way a regression head is wrong: a prediction uniformly half the
  target's magnitude renders identically to a correct one. The test asserts both
  halves — that the shared range separates them, *and* that independent ranges
  do not — because only the second one fails if someone "simplifies" it back.

  **Magenta for invalid, chosen because no colouriser here can produce it**:
  greyscale has no hue, `(n + 1) / 2` cannot reach it for a unit vector, and
  VOC's palette does not contain it. A test asserts that, so a future colouriser
  cannot quietly make the marker ambiguous.

  One thing it is deliberately **not**: it does not train. That is
  `run --save-probe`, added alongside, because `--push-to` needed a Hub account
  and the prediction column otherwise had no CLI-producible input.
  `correspondence` was out of scope for 9a and is covered by 9b, below.

- **The three probes with no spatial target draw their *decision*, and each
  states the diagnostic its own history calls for** (9c). `classification`,
  `retrieval` and `similarity` have nothing to lay beside the image at the same
  resolution, which is why they were skipped in 9a. What they have is a choice —
  which class, which neighbours, which candidate — and drawing it closes the
  last gap: `show_probes() == list_probes()` is now asserted, so a new probe
  cannot ship undrawable.

  **`class_balance` is the prefix bug as a figure.** `subset(n)` on a labelled
  folder takes a prefix and the file list is grouped by class, so an Imagenette
  prefix is entirely class 0 and the run scores 1.0 while measuring nothing —
  which is why `balanced_subset` exists. The footer reads `1 class, ... any
  score here is an artefact` whichever frames were drawn, so the diagnosis does
  not depend on the sample. **Frames are therefore picked spread across the
  split for the class-grouped kinds**, not as a prefix: drawing the first four
  rows would reproduce the artefact the sheet exists to reveal and look like a
  bug in the viewer.

  **`vote_balance` is the CSV-column bug as a figure.** NIGHTS presents the two
  candidates in arbitrary order, so the human vote sits near 50%; far from it
  means the vote was read from the wrong field, which otherwise surfaces only
  as a mediocre accuracy. Both are **diagnostics, never scores**, like
  `error_coherence`.

  **Retrieval loads the whole split whatever `--frames` says.** Leave-one-out
  retrieval over four images ranks each against three alternatives, so
  shortening the split does not shorten the drawing — it destroys what is being
  drawn. `--limit` became an explicit `show` flag for this: it is *how much to
  load*, distinct from `--frames`, *how many rows to draw*.

  **`--backbone` now defaults to `None`** and is demanded only where something
  must be computed — `correspondence` and `retrieval`, whose content is the
  features, and anywhere `--predict-from` is passed. It is checked *before* the
  split is indexed, which on a real dataset is the slow part.

  **Classification keeps its own schedule defaults**
  (`CLASSIFICATION_SCHEDULE_DEFAULTS`: 200 epochs at 1e-2, not the dense
  probes' 10 at 5e-4). One shared table would hand `show` a probe built with
  the wrong ones, and `load_probe` would then refuse a head that is fine.

  **Two bugs were found by rendering a page, not by a test**, which is the
  argument for this package arriving inside it: PIL's built-in bitmap font has
  no glyph for an em dash or an ellipsis and draws an empty box, so every
  caption this package writes is **ASCII** and a test asserts it; and a fixture
  whose vote column held a raw tally rather than 0/1 read as "humans chose
  right in 0%" — caught by the footer figure that exists for exactly that.

- **For correspondence it is the *shape* of the errors that diagnoses the bug,
  not their size — and that is now a number, not an impression** (9b).
  `error_coherence` is the mean resultant length of the error directions: 1.0
  when every match is wrong the same way, ~0 when they scatter. Measured on
  224px homography pairs with ResNet-18 features:

  | geometry | median error | coherence |
  | --- | --- | --- |
  | correct | 10.2 px, 22.6 px | **0.40, 0.29** |
  | homography in the wrong pixel frame | 293.9 px, 226.6 px | **0.98, 1.00** |

  **The median cannot make this call and the coherence can.** A weak backbone
  and a broken pipeline produce overlapping medians; only the direction
  distribution separates them, which is exactly the discrimination that took
  reading the code to make when `recall@1px = 0.003` first appeared. It is a
  **diagnostic, never a score** — it says nothing about a backbone, it is not
  recorded, and it must not reach a leaderboard.

  **`match_details` is on the task, and `_pair_errors` calls it.** The panel and
  the number therefore come from one code path by construction. A renderer that
  recomputed the geometry would put a *drawing that vouches for a wrong number*
  one edit away, which is worse than no drawing: the whole value of a panel is
  that it is independent evidence about the same computation, not about a
  parallel one.

  **Matches are sampled evenly, never from the front.** `match()` returns them
  sorted by descending similarity, so a prefix draws the most confident few and
  shows a systematically better picture than the score describes. Evenly rather
  than randomly so the same pair draws the same way twice.

  Correspondence is the one drawable probe that **always needs a backbone** —
  the matches are the thing being looked at and do not exist until features do —
  and it is also zero-shot, so `--predict-from` is refused by name rather than
  ignored. Its `show_arguments` is its `add_arguments`: nothing about a match is
  a training setting, so there was no schedule half to drop.

- **`TimmBackbone` reads a model's own structure; it used to assume a CNN's**
  (10a). `has_cls_token` and `patch_size` were *class* attributes declaring
  "CNN" for everything, which is why timm ViTs were refused outright — a false
  `has_cls_token` discards the CLS token while the record claims there was none
  to keep. Read per instance from `num_prefix_tokens` and `patch_embed`, any
  timm ViT becomes usable *and honest*, which is what added ConvNeXt-B, MAE
  ViT-B/16 and SigLIP-GAP ViT-B/16 in one change rather than three.

  **`default` pooling is read from timm's `global_pool`, not inferred from
  whether a CLS token exists.** The base class's "CLS if there is one, mean
  otherwise" is a good default and only a proxy: a ViT can carry a CLS token and
  still be trained to average. MAE reports `token` and SigLIP-GAP reports `avg`,
  so `default` means different things for two models of identical shape — each
  matching what the model hands its own classifier, which is the rule the
  ResNets already followed.

  **SigLIP is the `_gap_` variant deliberately.** Canonical SigLIP pools with an
  `AttentionPoolLatent` (`global_pool='map'`) — a *trained module*, not a
  reduction over tokens, so it cannot be a pooling mode over features the cache
  stores. `describe_transformer` refuses `map` by name and says which sibling to
  use. Do not "add a map mode" without deciding first that a pooling mode may
  carry weights.

  **ConvNeXt breaks the "pooled is what the model hands its classifier" rule,
  and the exception is documented rather than smoothed over.** Its head is
  `avg -> LayerNorm2d`, so the model's own vector is `norm(mean(x))` while this
  class returns `mean(x)` — max absolute difference 27.5 on one frame. Both
  invariants cannot hold: LayerNorm across channels does not commute with a
  spatial mean. The one kept is the structural one — **`pooled` is always a
  reduction of `dense`** — because the cache stores dense features and every
  pooling task reduces them. A test pins which four backbones match their own
  head and that ConvNeXt does not, in both directions.

  **The guards have fast tests, which is the point of `describe_transformer`
  being a module-level function.** Every timm backbone test needs real weights
  and is marked `slow`, which CI does not run; the three decisions here each
  produce a silently wrong number rather than an error, so the logic takes a
  stub and is tested without a download.

- **The docs gallery is real photographs now, and the licence rule that made it
  generated is unchanged — it was satisfied by better sourcing, not waived**
  (9d, replaced 2026-08-19). The original bullet said "do not improve the
  gallery by swapping in real frames", and that instruction was right about
  every source it had in view: VOC, ImageNet, NYUv2, Taskonomy and NIGHTS each
  restrict redistribution, none clearly grants it, and committing their frames
  would put third-party imagery in an MIT package — the line `NOTICE` already
  takes on probe3d's CC BY-NC code. **Those five are still forbidden and still
  appear nowhere in this repository.**

  What changed is that a source exists which does grant it. Open Images'
  validation split is **CC BY 2.0 for all 41,620 images**, with CC BY 4.0 human
  boxes and instance masks, so `scripts/fetch_gallery_frames.py` fetches from
  there and `render_gallery.py` draws on the result.
  **The licence is verified per frame rather than inherited from that
  sentence**: an allowlist at fetch time, and a refusal for any frame whose
  metadata carries no author or landing page, because an unattributable CC BY
  image is one this repository may not redistribute. `CREDITS.md` is generated
  beside the frames and `tests/test_gallery_licences.py` fails if a committed
  photograph has no credit — CC BY compliance is the kind of obligation that
  rots silently, since the page renders correctly either way.

  **The three properties generating used to buy were each paid for
  differently**, and one was genuinely lost:

  - *rebuilds with no downloads* — kept, by committing the frames
    (`assets/gallery_frames/`, 1.5 MB). Fetching is a separate one-off command.
  - *exact ground truth* — kept where the target is computed from the frame
    (`corner` runs the probe's own generator, `correspondence` warps by a chosen
    homography) and **replaced by something better** where it is annotated:
    `detection` and both segmentations now show what a human marked rather than
    what a script constructed.
  - *invalid pixels placed on purpose* — **weakened**. Real annotation has holes
    where it has them. The magenta marker survives because Open Images boxes an
    object it does not always mask, and a boxed region with no mask under it is
    genuinely unlabelled — which is what an ignore index is for. That is real
    structure rather than a placed hole, and it is rarer.

  **Four probes cannot have a target column at all and must not be given one.**
  `depth`, `surface_normal`, `keypoints2d` and `occlusion_edge` need sensor or
  reconstruction geometry no redistributable photograph carries. They render
  `image | prediction` from a *published* Hub head, footer saying so. A
  three-column figure with an invented middle column would teach the wrong
  convention to precisely the reader who came to learn it, which is worse than
  no figure. Two details cost an attempt each: a trained head's `output_size` is
  **fitted state**, so these heads emit 224x224 whatever they are fed and the
  figure must be rendered at 224 or the panels differ in size *and framing*; and
  they are drawn on **interiors**, because the heads were fitted on NYUv2 rooms
  and a photograph filled by an animal's face shows domain shift rather than the
  probe.

  **The figures live under `docs/_static/`, not `assets/`.** Sphinx cannot
  follow a relative path that escapes its source tree and MyST does not warn
  about it, so `-W` would not catch `../assets/...` — the site would simply have
  holes. The README points at the same files through
  `raw.githubusercontent.com`, which is the absolute-URL rule
  `tests/test_readme.py` already enforces. They are excluded from the sdist in
  `pyproject.toml`, which they would otherwise nearly triple.

  **Rendering the gallery found three real bugs that the whole test suite had
  missed**, which is the argument for this package landing on itself a third
  time. `_row` computed a display range for *every* kind, and a normal map's
  validity mask is `(H, W)` against a `(3, H, W)` target — so the first
  three-channel page ever rendered raised, having shipped in 9a. `render_panels`
  could not lay out a **ragged** final row, which a contact sheet produces
  whenever the tile count is not a multiple of `--columns`. And a long footer
  ran off the page edge, silently truncating the *legend* — the one line that
  says how to read the panel. All three now have tests; the first is
  `TestEveryPanelKindRenders`, which renders a full page per panel kind and is
  the coverage whose absence let it through.

- **`show` and `run` compose their flags from one callable, and that is a
  correctness property rather than tidiness** (9a). `ProbeSpec.show_arguments`
  is built by `_viewing(<probe>_view_flags)`, where every probe's
  `add_arguments` is exactly `<view flags> + _schedule_flags`. A parallel copy
  could build a *different dataset* than `run` would from the same command line,
  and a viewer that draws data the probe did not see is the failure mode this
  whole feature exists to catch, arriving through the feature itself.
  `SCHEDULE_DEFAULTS` supplies what `probe_kwargs` reads without putting
  `--epochs` in `visbench show depth --help`. **`--image-size` moved from the
  head group to the data group** in that re-cut, where it belongs: it decides
  the resize and centre crop. `run`'s surface is unchanged flag for flag and
  default for default — verified by diffing the parsed surface of all thirteen
  subcommands before and after, and pinned by
  `test_run_flags_are_unchanged_by_the_split`.

- **`DetectionTask.grid_hw` was fitted state outside the head and was not in
  `probe_state()`** (9a), so a saved detection probe loaded back and raised
  "this probe has not been fitted" on `predict`. Latent since v0.6.0 and
  reachable only through the Hub artifact path, so no measurement moved. It is
  the case `probe_state` was added for — the same one `ClassificationTask`'s
  standardiser was — and detection was simply missed when it arrived. **Check
  any probe that learns something outside `self.head`**, and note that the
  standing instruction to do so did not prevent this one: the check has to
  happen when the *probe* is written, not only when the artifact module is.

- **A probe that runs on any folder cannot have a leaderboard without a chosen
  folder** (8a). This is the cost of a derived target and it is not obvious from
  the API: two people's corner numbers are comparable only if they ran the same
  images, and nothing in the probe pins which. **The set chosen is Taskonomy
  tiny, the first 600 rows of each split list — the same frames `probe_edge`
  reads** — and `scripts/stage_corner_frames.py` is what makes them readable,
  symlinking the building-nested RGB frames into the flat `<split>/images/`
  layout `DerivedTargetDataset` expects. `build_corpus.sh` skips the probe with
  an actionable message if that folder is absent, as it already did for
  `generic_segmentation`'s binarised masks.

  **Shared frames are the point, not a convenience.** The corner target
  correlates 0.52 with `edge_texture`; the claim that earns the probe its place
  is that the two nonetheless rank backbones differently. That is exact only on
  identical pixels, so the staging is verified **set-equal** to the edge
  probe's 600 rather than assumed equal.

  **Symlinks, not copies**: `cache_identity` keys on path, size and mtime, and a
  symlink reports its target's, so a staged frame and the original share one
  feature-cache entry instead of doubling the cache.

  The cost of getting this wrong was demonstrated rather than argued: 8a's
  numbers were produced on an ad-hoc staging that was never committed, so the
  six published figures had **no surviving records** — 6e-2's exact failure,
  recurring on the newest probe two steps after that step ended it. The
  regenerated corpus reproduces all six to four decimals, which is what
  retired the hand-written table.

- **The docs build runs `-W` with `nitpicky = False`, and both halves are
  load-bearing** (7d). Many of the 371 cross-references are bare (`` :meth:`fit`
  ``) and resolve only from the owning class's context; with nitpicky off those
  render as literal text and emit nothing, so `-W` can stay fatal and still
  catch what matters — a broken toctree, a missing image, a malformed directive.
  Turning nitpicky on without rewriting those references would make every build
  red. `autosummary_generate` is `False` for a related reason: recursive
  generation walks `visbench.backbones.__all__`, which names CLIP and
  `TimmBackbone`, served by a module `__getattr__` that imports the optional
  extra on attribute access.

- **A `-W` docs build must tolerate an unreachable intersphinx inventory, and
  the filter has two details that each cost an attempt** (7d). intersphinx
  fetches five `objects.inv` over the network on every cold build; a
  `ConnectionResetError` reaching `docs.python.org` is logged as a warning,
  which `-W` turns into a failed deploy — it did, on the first push to `main`,
  minutes after the same commit passed on its PR. Losing intersphinx degrades
  gracefully by itself (nitpicky is off, so those references become plain text),
  so the *warning* is the only real problem, and it carries no `type=`, which is
  why `suppress_warnings` cannot target it. The filter in `docs/conf.py`
  therefore matches that one message, and: it goes on the **handlers, not the
  logger** (Sphinx emits from per-module child loggers, and a parent's filters
  never see a propagated record — only its handlers do), and it is inserted at
  **position 0, not appended** (Sphinx implements `-W` as a filter on the same
  handler, so anything added after it never runs — appending looks correct and
  does nothing). It prints a note to stderr rather than dropping the failure
  silently. Verified three ways, and the third is the one that matters: a broken
  toctree still fails, so the filter did not disable the guard.

- **A DOI is permanent and an archived release cannot be edited, which is what
  makes citation metadata a correctness problem** (7e). `CITATION.cff` naming
  the previous version is silently wrong in the expensive direction: GitHub
  renders the button, Zenodo mints the DOI, and someone citing v0.6.1 gets an
  archive of v0.7.0's code. So the cited version is tested against
  `visbench.__version__`, exactly as `uv lock --check` tests the lockfile
  against a bump — **a release commit now moves three things, not two**:
  `__init__.py`, `uv.lock` and `CITATION.cff`. Two further traps are pinned by
  the same file: **Zenodo reads `.zenodo.json` in preference to `CITATION.cff`**,
  so a divergence between them surfaces only once the archive is published
  under a title nobody chose; and the ORCID is written **two incompatible ways**
  — CFF wants the resolvable `https://orcid.org/...` URL, Zenodo wants the bare
  identifier, and neither accepts the other's form.
- **The concept DOI is what the README and `CITATION.cff` carry, never a
  version DOI** (7e). Zenodo mints both per release: the version DOI names one
  archive forever, the concept DOI always resolves to the newest. Someone
  citing "VisBench" wants the latter. The former is what a *paper reporting
  measured numbers* should pin, because a VisBench number is reproducible only
  against the release that produced it — which is the same reason every record
  carries its schema, pooling and protocol. **The minted concept DOI is
  `10.5281/zenodo.21822684`**, and it is quoted in three files: `CITATION.cff`,
  the README (badge and BibTeX) and `docs/index.md`. `tests/test_citation.py`
  pins that literal and *additionally rejects any other* `10.5281/zenodo.\d+`
  in those files, because the realistic mistake is pasting a version DOI over
  it from a Zenodo archive page — which resolves, renders, and looks entirely
  correct while freezing every citation at one release. **`.zenodo.json` is
  deliberately excluded from that check and carries no `doi` key**: it is the
  deposit's *input*, so a `doi` there claims a pre-reserved identifier rather
  than recording the minted one.
- **The optional-extra trap has now been hit twice, by the same person, two
  steps apart.** v0.6.0's hub tests needed `huggingface_hub` at monkeypatch time
  and CI installs `.[dev]` only; 7c's issue-template test needed PyYAML, present
  locally via timm and absent from `.[dev]`. The second was caught *before
  pushing* by blocking the import the way `CONTRIBUTING.md` now documents — the
  `find_spec` recipe there, and in 6e-5's section of the engineering log. Run it
  whenever a test touches `clip`, `timm`, `hub` or `yaml`; the five verification
  commands cannot catch this, because they run in the environment that has
  everything. PyYAML is now a declared `dev` dependency.

**The bullets below were lifted out of the v0.3 step write-ups when those moved
to [`ENGINEERING_LOG.md`](ENGINEERING_LOG.md). Each states the rule; the log has
the measurement behind it, under the step named in brackets.**

- **Boxes are `xyxy`, absolute post-transform pixels, 0-indexed — convert at the
  loader boundary and nowhere else** (6c). VOC stores `xyxy` and is *1*-indexed,
  so the loader subtracts 1. Both halves are silent when wrong: a swapped pair
  loads, trains and scores, and an absolute box is meaningless without the
  resolution it refers to — it must be transformed alongside its image, by hand,
  because a box does not resample. Rescale by the **achieved** ratio, not the
  nominal one.
- **VOC's `difficult` objects are *ignored*, never dropped from the ground
  truth** (6c-2). Measured on oracle predictions: ignoring them scores 1.0000,
  dropping them 0.9567 — **4.3 mAP**, and the wrong one is *lower*, so it reads
  as a weak detector rather than a scoring bug. Only the first may claim VOC's
  protocol. AP is also the one **dataset-level** metric here; "per image, then
  averaged" does not apply to a ranking.
- **GIoU, not IoU loss** (6c-3). Plain IoU is flat at 1.0 for every disjoint
  pair, so it has no gradient in the state every box starts in. With it: focal
  loss and its `-log((1-0.01)/0.01)` prior bias, and the distance `exp` clamped
  at 8 — unclamped it reaches `inf` in one step, and every later loss is `nan`
  while the run reports 0.0 mAP as though the features were useless.
- **A magnitude probe's `_activate` is the identity, and both ways of imposing
  non-negativity destroy it** (6d-1). On features that encode the answer,
  ceiling 1.0: ReLU **0.0000**, softplus **-0.9851**, identity **0.9997**.
  Non-negativity is learned from the targets. Reinstating a rectifier is the
  natural tidy-up and costs only a mediocre score, so a test pins it.
- **Scale the target, not the learning rate** (6d-1). `target_scale` is 1000
  rather than the container's 65535, because L1's gradient is `sign(...)` and
  does not shrink to match a small target: 0.047 → 0.456 `edge_correlation`
  across that sweep, plateauing once the target is order 1. **`depth_zbuffer` is
  the exception and *raises* if changed** — its 512 is what puts the target in
  metres, and `depth_metrics` reports RMSE in whatever unit it is handed.
- **Every Taskonomy mask file is named `..._domain_depth_zbuffer.png` whatever
  it masks** (6d-2). Build mask paths from the `mask_valid` directory with that
  suffix hard-coded, never from the requested domain. Read one as depth and you
  get a map of 0 and 255 that loads, trains and scores.
- **`weights_only=True` on every artifact load, and nothing may enter the
  payload that needs unpickling to reconstruct** (6e-4). These are fetched from
  a hub, so an unrestricted `torch.load` is arbitrary code execution; a test
  asserts the artifact still loads under it. Relatedly `push_probe` defaults to
  `private=True`, because a push is not reversible the way a local write is.
- **`HEADLINE_METRICS` and `METRIC_DIRECTIONS` are listed tables, and an
  unlisted entry raises** (6e-1/6e-3). A board ordered by whichever metric
  sorted first asserts a ranking nobody chose — and `mean`/`median` are angular
  *error*, so a heuristic reading them as scores ranks that board upside down
  and the output reads as a finding rather than a bug.
- **Cluster: locate the repo with `$SLURM_SUBMIT_DIR`, never
  `${BASH_SOURCE[0]}`** (6e-2). `sbatch` copies the script to `/var/spool/`, so
  the script is not in the repo. `/tmp` is node-local, so an `--output` path
  there leaves **no log at all** — the one failure that gives you nothing to
  read. And `build_corpus.sh` needs `.venv/bin` on `PATH`, not just
  `.venv/bin/python`, or a missing environment is reported as a failed probe.

### Open issues — read before assuming a red suite is your fault

**Every issue below is closed; the tracker was empty as of 2026-08-06.** The
fast suite is **1824 tests** and green on 2026-09-01, after the oracle gate
(1784 through the `fine_grained_classification` probe and schema v8), as
are all three lint steps and the `-W` docs build — all five run on `dgx1` via
`sbatch`, since `.venv/bin/python` does not resolve on a `dgxh100` login shell.
`uv lock --check` was green on 2026-08-06 and no dependency has moved since; the
90 slow tests were green on `main` on 2026-08-14 (up from 79: 10a's timm
ViT tests had never run in CI, since `slow.yml` does not run on pull requests). If
anything is red for you, that is new — do not go looking for a known cause here.

The entries are kept because each one records a *class* of failure this
codebase has actually shipped, and the next one will rhyme with them.

- **[#2] CI never ran `-m slow`** — fixed. `.github/workflows/slow.yml` runs it
  on every push to `main`, nightly at 03:00 UTC, and on demand, with the
  downloaded weights cached against `HUB_REF`. It is **not** part of the gating
  CI workflow and does not run on pull requests, so a 1.7 GB download never
  blocks ordinary work. If you add a check that guards a *silently wrong
  number*, it still belongs in the fast suite — this catches the ones that can
  only be caught with real weights, a day later at worst, not instead.
- **[#4] `zip(strict=)`** — done. `B905` is enforced, not ignored: 12 sites take
  `strict=True`, and `zip(resolved, resolved[1:])` in `backbones/base.py` takes
  `strict=False` because pairing a list with its own tail is meant to be ragged.
  Most of the 12 are backstops for invariants already enforced a few lines
  above, but one was a real hole: `CorrespondenceTask.evaluate_ceiling` never
  length-checked its arguments, so nine geometries against ten pairs scored nine
  and reported the number as covering the split. `evaluate` had always checked.
  **When you add a `zip` over two things paired by index, `strict=True` is the
  default** — the cost is nothing and the failure it prevents still trains.
- **[#1] DINOv2 on 3.9** — fixed by raising the floor; see above.
- **[#3] CLIP QuickGELU guard** — fixed; see the bullet above.

`CHANGELOG.md` under `[Unreleased]` is the full record of what each step
added and why. Since 7b the user-facing view is **split three ways**:
`README.md` is the arrival path (demo, install, what it is, the CLI), while the
per-probe reference and the measured numbers are in `docs/tasks.md` and the
roadmap and backlog in `docs/roadmap.md`. `CONTRIBUTING.md` is the public
version of the conventions this file keeps. All are kept current per step —
update them in the same commit as the code, not afterwards, and put a new
probe's measured numbers in `docs/tasks.md` rather than the README.

**Update this file at the end of every step, in that same commit.** The build
table, the registered names, the layout block, the v0.2 checklist and the
"decisions already paid for" list are how the next session knows where the work
stands and what it must not re-derive; a step that ships code without updating
them has left the next session to rediscover its findings the expensive way.
A step that *measures* something — a new backbone column, a corpus analysis —
also updates `CORPUS_FINDINGS.md`, and re-reads its counts off
`LEADERBOARD.md` rather than carrying the old prose forward. Every count that
has gone stale in this project went stale exactly that way.

---

## Architecture

### `BaseBackbone`

- One method, `.extract_features(image, pooling="default", layers=None,
  feature_mode="dense_only")`, returning a `FeatureDict`:
  `{"dense", "pooled", "grid_hw", "cls", "dense_layers", "layer_indices"}`.
- Returns **both** the dense spatial features and a pooled single vector from
  the same call — tasks pick whichever they need, backbones never expose
  separate methods per use case.
- Subclasses implement `_forward_features(image, layers) -> list[LayerOutput]`,
  one `(patch_tokens, cls_or_None, grid_hw)` per requested depth, from **one**
  forward pass. `resolve_layers()` on the base turns `None`/negatives into
  absolute indices and enforces strictly increasing order — order is meaningful,
  since DPT reads the first as coarsest.
- `dense`, `pooled` and `cls` always describe the **last** requested layer, so a
  multi-layer call is a strict superset of a single-layer one and a task reading
  only `dense` is unaffected. Multi-layer maps live under `dense_layers`, a
  separate key rather than `dense` sometimes being a list.
- Same method signature for every backbone type (ViT or CNN) even though the
  internals differ completely — see CNN vs ViT handling below.

### `BaseTask` (a.k.a. probe)

- `.fit(features, labels)` — no-op for zero-shot tasks (retrieval,
  correspondence).
- `.evaluate(features, labels) -> dict` — always returns a flat metrics dict,
  never prints results directly (see structured logging below).
- `.predict(features)`.
- **Pooling strategy is chosen here, not on the backbone.** A task passes
  `pooling="cls"` or `pooling="mean"` (etc.) into `extract_features()`; the
  backbone just executes whatever is asked. This keeps backbones dumb and
  interchangeable, and keeps the "what representation does this task need"
  decision in one place. Same for `feature_mode` and `layers`.
- A task **declares** `uses_dense`, rather than it being inferred from the
  task's level: it tells the cache which half of the extraction to keep and
  whether to stream, and dense features are ~250x larger, so guessing would be
  an expensive guess.
- `describe()` returns the metadata for the result record and always includes
  `task_params`, empty by default — a caller building a record should never
  have to ask whether a particular task has hyperparameters.
- `run()` records `pooling` **resolved** (`"default"` means CLS on a ViT and
  mean on a CNN, so the literal word does not say what produced the number).

### Feature cache

Mandatory in v0.1, not an optional speed-up added later. Disk-backed
key-value store, one file per (image, layer), keyed by
`backbone_key | layer | pooling | feature_mode | image_hash`. Every task reads
from the cache; the backbone forward pass runs at most once per image per
backbone. Two front doors:

- `extract_dataset(...)` stacks everything and returns one `FeatureDict`.
  Right for pooled features; impossible for dense ones.
- `materialise(...)` runs the same extraction, keeps nothing in memory, and
  returns a `CachedFeatures` — an ordinary `torch.utils.data.Dataset` over the
  files already on disk, so a `DataLoader` supplies batching, shuffling and
  workers. Pass `targets=dataset.target` to pair supervision by index.

---

## Feature extraction design — the most important decision in this codebase

Handle this consistently; don't improvise per-backbone.

### Default pooling rules
- ViT backbones with a CLS token → default single-vector representation is
  the **CLS token**.
- CNNs, and any backbone without a CLS token → default is **mean-pooling**
  over the dense feature map / patch tokens.
- Either default can be overridden per task call via the `pooling` argument.

### Dense-task feature modes (all three implemented and reachable through
`extract_features(feature_mode=...)`; mode 1 is the default)

1. **`dense_only`** (default) — just the spatial grid of patch/conv
   features, no CLS involved.
2. **`dense_cls_broadcast`** — the CLS token is broadcast spatially and
   concatenated onto every patch location, increasing channel dim uniformly
   across the grid.
3. **`dense_plus_cls`** — the dense grid and a single global CLS vector are
   kept **separate** and both handed to the task head, which decides how to
   fuse them (e.g. only at a bottleneck, or as a global conditioning vector),
   rather than broadcasting CLS into every spatial location.

Modes 2 and 3 are opt-in — a task must explicitly request them. The cache keys
on the mode, and `dense_plus_cls` returns the global vector under a separate
`cls` key. `DPTHead` is the consumer these were built for.

### CNN vs ViT handling
- **CNNs**: "dense features" = the last conv feature map before global
  pooling (e.g. `layer4` output of a ResNet).
- **ViTs**: "dense features" = the patch token grid, reshaped from
  `(num_patches, dim)` to `(H, W, dim)` using the model's known patch size and
  input resolution.
- Both are exposed through the **identical** `.extract_features()` signature
  and return shape, even though the internal extraction logic is completely
  different per architecture family.

### Multi-layer extraction
Declared in the interface from v0.1, **wired up in v0.2 (step 5c)** for every
backbone: `layers=[2, 5, 8, 11]` returns one map per depth from a single
forward pass. `visbench.run()` carries a task's declared `layers` into
extraction, and the result record stores them **resolved** against that
backbone's depth — `[-4, -1]` names different blocks on a 12- and a 24-block
ViT, so an unresolved record does not say what produced the number.

---

## Task categorization

Tasks are organized into three levels, following Chen, Marks & Cheng
(arXiv:2411.17474):

```text
tasks/
  high_level/   classification, semantic (multi-class) segmentation, detection
  mid_level/    generic (binary) object segmentation, depth estimation,
                surface normal estimation, geometric correspondence,
                mid-level image similarity, occlusion-edge detection (6d-2)
  low_level/    edge detection (v0.4), 2D keypoint detection (6d-2)
                — still scope only: optical flow, texture/reflectance,
                image quality
```

The occlusion-edge and texture-edge probes **share every line of their
implementation and sit one tier apart**, which is the cleanest statement of what
the tiers mean that this codebase has: recovering a depth discontinuity needs
scene geometry, recovering an intensity one does not. Taskonomy has no
reflectance domain, so the texture/reflectance row is *not* unblocked by 6d-2's
mask work — see `visbench/tasks/low_level/README.md`.

- **High-level** = semantic/category understanding.
- **Mid-level** = geometry and generic structure prior to semantic labeling —
  this is the paper's core contribution area, and it's where VisBench should
  be strongest relative to existing tools.
- **Mid-level image similarity is a distinct task class from high-level
  (semantic) retrieval** — mid-level similarity judges perceptual/geometric
  resemblance between candidates and a reference (scene layout, geometry),
  not category membership. Do not merge these two into one task even though
  both are "similarity"-flavored.
- **Low-level** = signal-level properties recoverable without naming an object.
  Was a README describing future scope only until step 6d-1 (v0.4), which added
  edge detection. The folder's own README now separates what is implemented from
  what is still scope, and is the place to look before starting a second one.

---

## v0.1 and v0.2 — **both COMPLETE**

Their scope, their hard boundaries and the numbers each task was proved on are
in [`ENGINEERING_LOG.md`](ENGINEERING_LOG.md) under "v0.1 and v0.2". Nothing
there constrains new work: v0.1's "no fine-tuning, no dense-prediction training
loops" boundary was lifted by v0.2, and every task, backbone and head those two
releases listed exists and is tested. The rules they established that *do* still
constrain new work were lifted into "decisions already paid for" above.


## Backlogs — what could come next

The v0.3 build steps that used to sit here — 6a-6f: fine-tuning, prefix
caching, detection, the low-level probes, the leaderboard and the Hub — are
archived in [`ENGINEERING_LOG.md`](ENGINEERING_LOG.md), which is where every
`6x` label in this file resolves. Read it before touching the code one of those
steps built; the rules that still constrain new work were lifted back into
"decisions already paid for" above.

### The candidate task backlog — and what is actually on this machine

`docs/roadmap.md` has the public version of this list, grouped by cost — it was
in the README until 7b moved it. What follows
is the part a contributor cannot see: **which of these have data on this
machine**, checked on 2026-08-01 rather than assumed. A candidate whose dataset
is absent is not cheap, however simple its protocol.

**`/shared/sets/datasets/` has a `vision/` subdirectory, and a top-level listing
does not see into it.** 96 more datasets live there, including ones a first pass
recorded as absent. Check both levels before concluding anything is missing —
this note exists because the first version of this section did not, and said
Places365 and NIGHTS were absent when both are on disk.

**Verified present at the top level:** `ADE20K` (`ADEChallengeData2016`), `COCO`
(`annotations/` has `instances_*`, `captions_*`, `person_keypoints_*` — **no
panoptic and no stuff**), `cub_200_2011`, `stanford_cars`, `stanford_dogs`, many
ImageNet variants, `Imagenette`.

**Verified present under `vision/`:** `nights` (`data.csv`, `ref/`, `distort/` —
this is what the `similarity` probe reads), `places365_standard` (`train/`,
`val/`, `categories_places365.txt`), `SUN397`, `mit67_indoor_scenes`,
`caltech101`, `country211`, `CUB-200`, `oxford_flowers102`. **Scene
classification was a dataset-swap on the existing linear-probe path** and
shipped 2026-08-27 as the `scene_classification` probe on `places365_standard`
(a new probe *name* rather than a flag — see the "decisions already paid for"
bullet). Its 12-backbone corpus board landed 2026-08-28. **Fine-grained
recognition shipped the same way on 2026-08-28** as
`fine_grained_classification`, on **CUB-200-2011** — and the copy to use is
`vision/CUB-200/images_train_test/`, which already holds the official
5994/5794 split as `train/<class>/` + `val/<class>/`. Two traps in that
directory: the top-level `cub_200_2011/CUB_200_2011` is **permission-denied**,
and `test/` is a **symlink to `val/`**, so naming `val` is naming the official
test set and `--split test` would index the same files under a different path
and so a different fingerprint. Stanford Cars (`train_cars`/`test_cars`, 196
numeric class dirs) is the same folder shape and still open; Stanford Dogs and
Flowers102 are **not** — both keep their splits in `.mat` files and so need
loader code, which is a different cost class from a folder swap.

**Verified absent, both levels:** any optical-flow set (Sintel, KITTI,
FlyingChairs), NYUv2, any intrinsic-image set (IIW, SAW, MIT intrinsic).
`bsds300` is still the MAF density-estimation benchmark, not BSDS500 (its
`bsds300.hdf5` sits beside `gas` and `hepmass`) — see 6d-1. **BSDS500 itself is
no longer absent**: Berkeley is unreachable from this machine (`www2.eecs`
times out, the old host 403s) while the network is otherwise fine, so
`scripts/fetch_bsds500.py` reads the `BIDS/BSDS500` GitHub mirror at a pinned
commit into gitignored `data/bsds500/`.
`davis` exists but holds two sequences of derived output (`dpt/`,
`epipolar_error*`), not the DAVIS annotations, so it is not a video-segmentation
benchmark.

**The Taskonomy copy on disk carries eight domains only**: `depth_zbuffer`,
`edge_occlusion`, `edge_texture`, `keypoints2d`, `keypoints3d`, `normal`,
`principal_curvature`, `reshading`, plus `rgb` and `mask_valid`. Taskonomy
*publishes* `vanishing_point`, `room_layout`, `segment_unsup2d/25d` and
`point_matching`, and none of them are here. So the roadmap items that look like
free Taskonomy wins — vanishing points, room layout, superpixel segmentation —
each need a download first, and are not in the same cost class as 6d-1 and 6d-2
were.

**The cheapest items on the list need no dataset at all, and that is the useful
observation.** `edge_texture` is a target Taskonomy *computed from the RGB
frame*, and so are these. **`corner` (8a) and `orientation` (2026-08-28) are
done**; **DoG blobs was rejected** for overlapping 0.51 with `corner`;
**photometric superpixels** is the one that remains derivable from any image
folder already here. A magnitude target is a generator plus a
`DenseMagnitudeTask` subclass; a vector one (`orientation`) needs its own small
task base, which `visbench/tasks/low_level/orientation.py` now provides as the
second worked example.

Three hazards to carry into any of them, all paid for:

- **Check the tail before assuming the magnitude protocol transfers.** A corner
  response is spikier than an edge response, and `edge_occlusion` at 46% mass in
  its strongest 1% of pixels is the case where L1 and Pearson pull apart and the
  probe stops ranking backbones. (An *angle* has no tail — `orientation` needed
  no compression, confirmed by the pre-measurement.)
- **Check the overlap with what already ships, before building.** DoG blob was
  vetoed on this: 0.51 with `corner`. `orientation` passed it: under 0.09 with
  both `corner` and `edge`, because it measures phase. One afternoon of
  correlations, not a probe run per backbone.
- **A derived target is only as honest as its generator, and `protocol` must say
  which generator.** "Harris corners" is a family, not a definition — the
  k parameter, the window, the smoothing and the non-maximum suppression all
  move the target. A record claiming a bare `"harris"` says less than it looks.

### The library-surface backlog — closed 2026-08-28

Added 2026-08-14, after a read of what a new user would reach for and not find.
All three shipped: `visbench show` (9a-9d), `examples/custom_backbone.py`
(2026-08-19), and the **dataset bridges** (2026-08-28, below). `docs/roadmap.md`
has the public version. **None of these was a defect** — each was already
reachable by writing Python; what was missing was the shortest path. v0.7 is the
precedent for shipping a release that changes no number.

**The dataset bridges, as shipped.** `TorchvisionDataset` and
`HuggingFaceDataset` in `visbench/data/bridges.py` — thin `BaseDataset`
adapters over a `torch.utils.data` dataset / a `datasets.Dataset`. `torchvision`
is a core dep so its bridge imports at module scope; `datasets` is a `[datasets]`
extra, imported lazily inside `HuggingFaceDataset.__init__` and `_build_hf`, so
`import visbench` never needs it (and it is in `dev` too, or the bridge tests
skip in CI — the optional-extra trap, pre-empted this time). On the CLI,
`classification` / `retrieval` / `scene_classification` take
`--dataset torchvision:CIFAR10` / `--dataset hf:cifar100:name=cifar100` in place
of `--data` (a mutually-exclusive group; `resolve_named_dataset` in
`cli/datasets.py` parses `scheme:name:key=value…`). **Image-level probes only** —
a dense/pair/triplet probe with `--dataset` raises with a message, because an HF
dataset carrying a dense target is a much larger surface (per-probe
target-column plumbing, loader/dtype selection, the four validity conventions).

**`cache_identity` is the method a bridge must not skip, and both get it right by
leaning on index-order immutability.** Return `None` there and every run
re-decodes every image forever while appearing to work — the `view_identity`
failure. A `datasets.Dataset` carries a `_fingerprint` that changes on any
transform, so `f"{fingerprint}|{row}"` names a row's content exactly. A
`torchvision` dataset has no such hash: the `ImageFolder` family
(`.samples`/`.imgs`) uses the file path + size + mtime like
`ImageFolderDataset`, everything else a sha256 of `repr(dataset)` (which states
root, split, download flags) + length + index. The `repr` digest is weaker —
two different downloads with matching reprs would collide — and that is
documented on the class, not hidden. `describe()` adds `dataset_source`
(`"torchvision:CIFAR10"` / `"hf:<name>"`) so a bridge record lands in its own
comparability group rather than merging with a folder board.

**`balanced_subset` moved to `BaseDataset`.** It only needs `labels()` and
`subset()`, both of which the bridges have, so the CLI's per-class `--limit`
works on them for free. `ImageFolderDataset` lost its copy; the method is
otherwise unchanged.

**All three shipped in that cost order**: `examples/custom_backbone.py`
(hours), `visbench show` (the only one that guarded a silently wrong number),
the dataset bridges (largest). The pre-bridge reasoning for each — why a viewer
was the one that guarded a wrong number, what the two tiers of custom-dataset
support already covered, and why `CustomBackbone` needed showing rather than
building — is in `docs/roadmap.md` under "Library surface". What remains is the
candidate-task backlog.

---

## Engineering conventions

- PyTorch, Python 3.10+. Optional extras: `clip` (open_clip), `timm`, `hub`,
  `datasets` (the HuggingFace bridge), `docs`, `dev`.
  **A backbone whose extra is missing is still registered and still listed** —
  both CLIP and timm import their dependency lazily inside `__init__`, so the
  registration module imports cleanly and `_REGISTRATION_MODULES`' skip logic
  never fires for them. Constructing one raises `ImportError: ... pip install
  visbench[clip]`, which is *better* than the registry raising "Unknown
  backbone", so do not "fix" it by moving the imports to module scope. Use
  `registry.missing_extra(name)` to ask without importing; the CLI's `list`
  marks them. This was documented backwards until the v0.2.0 wheel test, where
  a core-only install listed all six backbones under a footer promising it
  would not.
- Pin exact dependency versions via `uv.lock` — this is a reproducible
  benchmark library, not a moving-target research repo.
- Write tests alongside every new module; don't defer testing to "later."
- Every task run logs a structured JSON record — backbone, task, dataset,
  pooling mode, feature mode, layers, metrics, timestamp — under one
  **additive-only** schema, so leaderboard tooling never needs a retrofit.
  Bump `SCHEMA_VERSION` when adding a field; never remove or repurpose one.
  A *trained* run also records `training` (v8) — how the fit itself went, which
  is what separates an underfitting probe from a weak representation, and is
  never something to rank on.
- Package for PyPI from v0.1: `pyproject.toml`, semantic versioning,
  `pip install visbench` as the eventual target install path.
- Cite prior art in code comments and docs wherever an evaluation protocol is
  borrowed, not just in the README. `NOTICE` is the consolidated list.

### Verifying — use these exact commands

The project venv is `.venv/` — Python 3.10.12, the supported floor, with
`visbench` installed editable and all extras present. **Use it.** Another
interpreter on the machine will not have `visbench` importable (examples fail
with `ModuleNotFoundError`) and may have different dependency versions.

```bash
source .venv/bin/activate       # or call .venv/bin/<tool> directly

pytest                                              # 1824 fast tests
pytest -m slow                                      # 79, real DINOv2/CLIP weights
ruff check visbench/ tests/ conftest.py examples/ scripts/
ruff format --check visbench/ tests/ conftest.py examples/ scripts/
mypy visbench/ examples/ --ignore-missing-imports   # reads [tool.mypy], py 3.12
```

CI runs all five: the four fast ones gate every push and pull request, and
`-m slow` runs in a separate workflow on pushes to `main` and nightly. A local
environment with extra packages installed will pass checks that CI fails, so do
not substitute your own invocations — particularly for mypy, which reads
`python_version` from `pyproject.toml` and checks nothing useful if you
override it.

**CI gates two more jobs the five commands do not cover, and a release touches
both.** `lock` runs `uv lock --check`, and `build` runs `python -m build` +
`twine check dist/*`.

**And a third workflow can red-X a pull request** (7d): `docs.yml` builds the
Sphinx site on every PR and deploys from `main`. It is not part of `ci.yml` —
`tests/test_contributing.py` asserts `ci.yml`'s job set *exactly*, and docs are
not a gating concern for a code change — so the docs build is a sixth command,
run when `docs/` or a docstring changes:

```bash
pip install -e ".[all,docs]"    # the docs extra: sphinx, furo, myst, copybutton
sphinx-build -b html -W --keep-going docs docs/_build/html
```

`-W` makes every warning fatal, so a page with a hole in it never publishes;
`--keep-going` reports all of them in one run. `tests/test_contributing.py`
pins that exact command against the one the workflow runs, so the guide and CI
cannot drift.

- **A version bump requires `uv lock` — and, since 7e, `CITATION.cff`.**
  `uv.lock` pins visbench *itself*, so editing `version` in `pyproject.toml`
  desynchronises it and `lock` fails while all five local commands pass — which
  is exactly what happened on the v0.3.0 PR. Re-lock in the same commit as the
  bump and confirm the diff is the one line: anything more means dependencies
  moved too, which is a separate decision and not part of a release.
  `CITATION.cff`'s `version` and `date-released` move with it, enforced by
  `tests/test_citation.py` in the fast suite.
- **`twine check` is the only local proxy for how PyPI will render the README.**
  Neither `build` nor `twine` is in `.venv/`; install them into a throwaway venv
  rather than the project one, so `.venv/` keeps matching what CI has.

Both suites and all three lint steps must be clean before a commit. Prove a
new task end to end on a real backbone via its `examples/` script, not only
against the fake backbones in `tests/conftest.py`; the toy backbones cannot
show a training-dynamics problem, and one has already been found that way.

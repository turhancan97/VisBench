# VisBench

The name is **Vis**(ion) **Bench**(mark), and both halves are a scope
commitment worth keeping in view when deciding what belongs here. *Vis*: the
subject is a vision backbone's features — a text tower is only ever a means of
building the visual one, which is why zero-shot open-vocabulary classification
sits in the backlog rather than the corpus (it cannot rank a backbone without
one). *Bench*: the deliverable is a **comparable** number, not a score — which
is what `comparability_key`, the `protocol` field, the resolved `pooling` and
`layers`, and the refusal to rank across `finetune` all exist to protect.

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
| 1-4 | The abstraction, the cache, DINOv2, and the three v0.1 tasks end to end | done |
| 5a-5j | v0.2: timm/custom backbones, pluggable heads, multi-layer, every dense task, streaming, and the CLI | done |
| 6a-6f | v0.3: fine-tuning, the prefix cache, detection, the low-level probes, the leaderboard, the Hub, and correspondence in pixels | done |
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
| 13a | The documentation site restructured: guides, 16 probe pages, an API reference | done |
| 14a-1 | Instance segmentation: the VOC dataset and its instance loader | done |
| 14a-2 | Instance segmentation: mask AP, by making VOC's matching IoU-agnostic | done |
| 14a-3 | Instance segmentation: the head, proved end to end on DINOv2-S | done |
| 14a-4 | Instance segmentation: the 12-backbone board and the probe's own page | done |
| 15a | `dino_vitb8`: the grid control, raised rather than lowered | done |
| 15b | The grid finding at n=10: DPT-vs-linear across every ViT | done |
| 16a-1 | Relative camera pose: the NAVI pair set and the pose metric | done |
| 16a-2 | Pose: `PoseHead` + the task, proved against the pre-measurement | done |
| 16a-3 | Pose: registration, the 13-backbone board, the viewer, the docs page | done |
| 19a | `scene_parsing`: NYUv2-40 as the nineteenth probe, and its board | done |
| 19b | The pose board's noise, measured: a trigger rather than a dose | done |
| 20a | The pose seed sweep: which adjacent rows are actually ordered | done |

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
it has now been over that budget **four times** — 203k, 151k, 158k and **150k
on 2026-09-17**, trimmed to ~134k by compressing eleven
retrospective blocks to the rules they left behind: the corpus-claim list (the
largest single block in the file, and a restatement of `CORPUS_FINDINGS.md` in
the same detail), the pose write-up, 9a, 9d, 10a, the DPT control, the
optional-extra trap, the split control, the second-dataset edit list, the
closed library-surface backlog, the derived-target gauntlet and schema v8.
**10a was the only one whose derivation was not already filed elsewhere**, so
its full text moved to `ENGINEERING_LOG.md`; the rest were pointed at the file
that already owned them.

**Every time, the growth was retrospective narrative rather than rules** — so
when it next crosses, look for a closed step's write-up, a per-release
paragraph, or a claim restated here in the same detail as the file that owns
it, before touching "decisions already paid for", which is the bulk of this
file and is what it is for. **Each move keeps the rule here and sends the
derivation there**; nothing is deleted, and the owning file says so.

**Read `CORPUS_FINDINGS.md` before quoting any board** —
this file keeps only the claims, and several of them corrected an earlier
published reading.

Steps 7a-7e ship no new probe, backbone or metric. They are the **contributor-
facing surface**: the shortest path from `pip install` to a number, where the
reference material lives, how someone outside this file learns the rules, and
how the work is cited. All five are **v0.7.0**, the release that changes no
number — every measurement v0.6.1 reported, v0.7.0 reports identically.

---

## Current state

**Everything through v0.23.0 is shipped**, and every numbered step in the build
table before 14a is done — every task, all three backbone families, the CLI,
fine-tuning, detection, the low-level probes, the leaderboard and probe
sharing. Each release's narrative is in `CHANGELOG.md`, its derivation in
[`ENGINEERING_LOG.md`](ENGINEERING_LOG.md) and its upload record in that file's
"Release history"; read them there rather than carrying a recap here. The two
that changed **no measurement** are worth knowing by name because they are the
precedent for shipping one: **v0.7.0** (contributor-facing) and **v0.16.0**
(documentation). **v0.6.1** is the other one to know — it corrects a
correspondence board that shipped ranked upside down; see step 6f.

**The corpus file is 405 records resolving to 247 board cells** — nineteen
boards, thirteen backbones a board. The two numbers differ because the corpus
is **append-only** and re-runs have appended beside records they supersede:
0.15.0 re-ran the five low-level boards for their `ceiling_*`, the
schema-v8 `training` re-run (2026-09-10) re-ran the eight trained boards that
predated that field, with three cells of it landing on 2026-09-11, and
`dino_vitb8`'s two smoke-test cells were re-run inside its own array.
`latest_per_backbone` picks the newest. Quote 247 for
coverage and 405 only for the file, and re-read both off `LEADERBOARD.md` and
`wc -l` rather than from here.

Two standing consequences of that history, both of which have already cost a
published claim:

- **Read a count off `LEADERBOARD.md`, never out of prose.** Every count that
  has gone stale in this project went stale by being carried forward through a
  release that added a column or a board. The generated tables have always been
  right; only the prose around them drifts, which is the half no test reads.
  **`tests/test_docs_counts.py` now reads it** (2026-09-14): the present-tense
  probe and board totals in seven user-facing files are pinned against
  `list_probes()` and `LEADERBOARD.md`'s rendered sections. It is keyed to the
  *idioms* that carry a total, not to any number word before "probes", because
  a subset count is ordinary prose — and `CHANGELOG.md`, `ENGINEERING_LOG.md`
  and `docs/roadmap.md` are excluded by name, since a count stated *as of a
  release* is correct history and rewriting it would falsify the record.
  **The board-*cell* total and the README's `**Status: vX.Y.Z**` line are pinned
  too** (0.23.0), after the front page shipped all three stale to PyPI: the
  board idiom had been anchored on `boards, twelve` and stopped matching when a
  thirteenth backbone made that line read `boards, thirteen backbones each`.
  **An idiom anchored on a count beside the one it checks is anchored on
  something that moves.**
- **"It reproduces across two caches" is evidence only if the caches differ**
  (19a). `scene_parsing`'s board cell and a local run agreed bit for bit across
  two cache roots, and the write-up nearly quoted that as a contrast with
  `relative_pose`'s ~1 degree movement — but the two caches' *features* were
  identical here (0.00e+00), because both extractions used the same batch size
  on the same GPU model. Pose's differed by 1.1e-05 because one side extracted
  at batch 64 against the other's 32. **Check the inputs before reporting that
  the outputs agree**, or the claim is that identical inputs give identical
  outputs.
- **A guard on a total is not a guard on the count beside it** (16a-3). The
  docs-count test pinned "the seventeen boards" and said nothing about "first on
  six of them", so v0.20.0 shipped a leader count its own new backbone had made
  wrong, and three pretraining coefficients on a probe page stayed at their
  twelve-backbone values. Both were found only because a new probe forced those
  lines to be edited. `tests/test_docs_counts.py` now recomputes the leader
  count from the corpus; a coefficient cannot be guarded that way, so **re-run
  `scripts/analyse_board_correlates.py` whenever a column is added** rather than
  carrying its output forward.
- **Before quoting any board, read
  [`CORPUS_FINDINGS.md`](CORPUS_FINDINGS.md)** — in particular the
  `sam_vitb16`, `dino_vitb16` and `mae_vitb16` entries. 10e's recipe control
  refuted half of 10d's own published claim before either shipped, and the
  three-tier separation 10b announced no longer holds for *high-level*.

**The relative-camera-pose line is closed** (16a-1 to 16a-3, 2026-09-17): an
eighteenth probe, its thirteen-backbone board, and the linear control committed
beside it. Its two protocol decisions are not to be reopened as implementation
details: **the pair count is pinned at eight partners per training anchor**, and
**the board uses probe3d's MLP** — the first VisBench board whose head is not a
linear map (`DPTHead` is nonlinear and is a *control*; every head a published
board uses is an affine layer or a 1x1 convolution). Two standing consequences:
**read that board to whole degrees**, since that head's run-to-run scatter is
about a degree (against one cache at one seed it reproduces bit for bit), and it
is the first board close to *orthogonal* to the high-level tier, so a pose
number is near-independent evidence about a backbone. **The scatter is a
trigger, not a dose** (19b, `results/controls/README.md`): it was published as
"1e-5 in the features, amplified by thirty epochs into 0.93 degrees", and
perturbing the features across a *thousand-fold* range moves the score by the
same amount at every size, while changing only the seed moves it as much again —
0.72 on `mae_vitb16` and **2.23** on `clip_vitb16`. **So do not order two
adjacent rows from the gap** (20a): all thirteen were re-fitted at five seeds
into `results/controls/pose_seeds.jsonl`, every seed-0 row reproduces its
published cell exactly, and **two of the five pairs the tie list names come out
the other way round** — `dinov2_vitb14` over `dino_vitb16` by 0.80 and
`clip_vitb32` over `resnet18` by **2.08**, against published gaps of 0.04 and
0.12 in the opposite direction. **A gap threshold is the wrong instrument and
widening it would be worse**, since pairs 1.58 apart are solidly ordered while
the reversed pair is 0.12 apart; only 11% of the seed variance is common-mode,
so the gap is one draw of a quantity whose spread is unrelated to it. No
published number moves. Everything else is the candidate task backlog
further down this file; its cheap end is exhausted,
and **three candidates were built and rejected** — photometric
superpixels (0.021-0.043, which bought the oracle gate), DoG blobs (0.51 overlap
with `corner`) and **relative depth ordering** (2026-09-04, the first rejected
for failing to *rank* rather than to be recoverable: it cleared the gate at
94.0% and then reproduced the `depth` board at Spearman **+1.000**, kept
unregistered as a five-record control, and its lesson is the gauntlet's floor
rule below). **The BSDS500 line is closed at two steps** (12a-1/12a-2): the
dataset and a validated ODS/OIS/AP metric ship, reproducing the published human
ODS of 0.80 at **0.8030**, and the probe was **refused by the oracle gate** —
its write-up and the two routes that could reopen it are in
`visbench/tasks/low_level/README.md`. **Instance segmentation on VOC shipped**
(14a-1 to 14a-4, 2026-09-08) as the seventeenth probe and board and was the
exception to "nothing cheap remains". Re-confirm what is wanted before starting
anything; do not assume this order is a plan.

**The one thread that was open — why `detection` alone fails to reproduce — is
closed**: GPU non-determinism made visible by a discrete metric, never a bug,
and detection reproduces to *three* decimals rather than four. See
[`CORPUS_FINDINGS.md`](CORPUS_FINDINGS.md) for the table; **there is no open
lead here any more.**

**Three probes shipped after v0.11.0 and each has a 12-backbone board**, with
their readings in [`CORPUS_FINDINGS.md`](CORPUS_FINDINGS.md) and their reference
on their own page under `docs/probes/`. `scene_classification` (14th, place) and
`fine_grained_classification` (16th, subordinate, the official CUB-200-2011
5994/5794 split run with no `--limit`, which is what makes it comparable to the
published CUB literature) share the object `classification` linear-probe path
and ask a different question each — a distinct probe *name* for the reason in
"decisions already paid for", confirmed edit for edit the second and third time.
`orientation` (15th) is the fourth low-level task and the second computed from
the frame, the first whose target is a *direction*: `(cos 2θ, sin 2θ)` with its
length set to the coherence, `orientation_error` in degrees of
coherence-weighted angular error halved so 45 is chance, on `corner`'s pinned
`data/corner_frames/` set.

**Two findings from those boards are load-bearing enough to state here**, both
expanded in `CORPUS_FINDINGS.md`: the two image-level classification probes
rank with the *localised* cluster (`detection`, `semantic_segmentation`) rather
than with the object board they subclass — `fine_grained_classification`
correlates **+0.832 with `detection`** against +0.322 with `classification` —
and `orientation`'s board is **not** independent even though its target is,
ranking like `keypoints2d` (rho +0.95), `corner` (+0.82) and `edge` (+0.79).

**The library-surface backlog is closed** (2026-08-28) — `visbench show`,
`examples/custom_backbone.py` and the dataset bridges; see its section below.

**`corner` (8a/8b) is the first probe whose target is computed from the image
rather than downloaded**, which is why `visbench/data/derived.py` exists and
why `scripts/stage_corner_frames.py` has to pin the frame set — see the
derived-target rules below.

**v0.7.0 was contributor-facing, not measurement** (7a-7e), and what survives
it: `visbench demo` runs a real probe on generated images with no dataset and
no extras; `CONTRIBUTING.md` is the public version of the rules this file
keeps; `docs/` is a Sphinx site deployed by a **third workflow**, `docs.yml`;
the generated tables live with the reference material rather than in the README,
and `scripts/render_tables.py` takes a list of marked files; and
`CITATION.cff` + `.zenodo.json` are what get a release archived with a DOI.

**What v0.6.0 through v0.5.0 left behind, in one paragraph** — the narratives
are in `CHANGELOG.md` and the derivations in `ENGINEERING_LOG.md`; these are
the facts still load-bearing. `results/corpus/visbench.jsonl` is a committed,
**append-only** corpus of every probe against every backbone;
`visbench/results/leaderboard.py` holds the rules for which records may be
ranked together, `visbench/results/render.py` turns an answer into markdown,
and every marker-delimited table plus `LEADERBOARD.md` is generated from the
corpus with a **fast** test failing if either drifts. `visbench/hub/`
serialises a trained head with the backbone identity beside it, behind a
`[hub]` extra. Dense probes take `finetune_blocks=N` / `--finetune-blocks N`,
**DINOv2 only**, recorded under schema v6's `finetune` field, with the frozen
blocks below the cut cached separately in `PrefixCache`. Four Taskonomy domains
declare how they mark an invalid pixel rather than exposing a mask —
`depth_zbuffer`, `normal`, `edge_occlusion`, `keypoints3d` — and the
`target_transform` / `invalid` / `masked` settings land in `dataset_params`,
which is what that field was added for. Edge detection is dense magnitude
regression on `edge_texture` recorded as `visbench_edge_regression`, **not**
BSDS500's, which is a correspondence metric and was a step of its own.

Registered names — `visbench.list_backbones()`, `list_probes()`,
`visbench.heads.list_heads()`:

```text
backbones  dinov2_vits14, dinov2_vitb14, clip_vitb16, clip_vitb32,
           resnet18, resnet50, convnext_base, mae_vitb16, siglip_vitb16,
           supervised_vitb16, dino_vitb16, sam_vitb16, dino_vitb8,
           dinov2_vitb14_196 (a resolution control, not a corpus column)
           (+ CustomBackbone, unregistered)
probes     classification, scene_classification,
           fine_grained_classification, retrieval, correspondence,
           depth, surface_normal, generic_segmentation, semantic_segmentation,
           similarity, detection, instance_segmentation, edge,
           keypoints2d, occlusion_edge, corner, orientation, relative_pose,
           scene_parsing
heads      linear, dpt, detection, instance, pose
```

The CLI exposes all nineteen probes: `visbench list`, `visbench run <probe>`,
`visbench cache stats|clear`, plus `visbench demo` (7a) and **`visbench show
<probe>` (9a)**. A test asserts the CLI's table and `list_probes()` are the same
set, so a probe cannot ship unreachable from a shell by accident. Since 9c
`show` is valid on *every* probe and `show_probes() == list_probes()` is
asserted.

**`visbench run --push-to REPO_ID` publishes the head it just trained**
(`--public` overrides the private default), and `scripts/build_corpus.sh` takes
`PUSH_TO` / `PUSH_PUBLIC` so a whole board publishes from the file that already
holds every probe's flags. **Publishing from the run, not a second script, is
the design**: a head is only meaningful against the features it was fitted on,
and the run's flags are what fitted them, so a separate publish step is a second
copy of every dataset flag, free to drift — and a head trained under drifted
flags uploads, loads and scores without complaint. The CLI refuses a zero-shot
probe *before* the run rather than after spending it, and
`scripts/publish_collection.py` groups the pushed repositories into one
collection, dry-run unless `--create`.

**Twenty trained heads are published and public**, as of 2026-08-07: ten probes
against DINOv2-S/14 and DINOv2-B/14, one repository per pair at
`turhancan97/visbench-<probe>-<backbone>`, in a collection whose URL is quoted
in `README.md` and `docs/guides/sharing.md` — **read it from one of those two
files rather than reconstructing it**, since a Hub collection slug carries a
generated hash suffix. That is ten of the fourteen probes that train a head:
`scene_classification`, `fine_grained_classification`, `orientation` and
`instance_segmentation` all shipped after the push. The three zero-shot probes
are deliberately absent, which is a different reason.

Republishing the board is `PUSH_TO=... PUSH_PUBLIC=1 scripts/build_corpus.sh`.
**Point `RESULTS=` at a scratch file, never `results/corpus/visbench.jsonl`**,
so the run can be diffed against the corpus instead of replacing the reference
it would be checked against — that diff is the only reason the seeding bug below
was ever found. And **do not pipe a long publishing run through `tail`**: it
buffers, so a run killed part-way leaves no log and the Hub has to be queried to
find out what shipped.

**Eighteen releases are archived, `0.23.0` the newest** (2026-09-18). Byte
counts, wheel digests, the commit each tag resolves to and what every
`__version__`/`SCHEMA_VERSION` import read back are recorded release by release
in [`ENGINEERING_LOG.md`](ENGINEERING_LOG.md) under "Release history"; read them
there. `0.23.0` is the **eleventh running with no gap** between tag, wheel, release
and `main`; `0.18.0` is the only release to move the schema (v8 -> v9), so a
v0.17.0 install cannot read a record written by it while older records read
fine here. **`0.19.0` shipped without its wheel check and without its
release-history entry** — both were done retrospectively on 2026-09-16 — and
**`0.21.0` and `0.22.0` both stopped with PyPI complete and no GitHub
release**, so Zenodo had archived nothing while `CITATION.cff` already named the
new version. Twice running makes it structural rather than a slip: the upload is
the step that feels like the finish line, CI went green long before, and
**nothing anywhere turns red when the archive never happens**. `0.23.0` is the
rule's first test and it held — the release was cut as a step of its own and
confirmed with `gh release view`, and Zenodo had archived it a minute later.

Four rules those releases left behind, beyond the standing list below:

- **When a change deletes a file the README names, the clock starts** (0.16.0).
  A PyPI version can never be re-uploaded, so dead links on the front page are
  only fixed by the *next* release.
- **Read Zenodo's API before claiming an archive is wrong** (0.16.1): a
  `CITATION.cff`/`.zenodo.json` divergence is half wrong and half fine, since
  Zenodo prefers the latter and GitHub's cite button the former. And **when a
  docstring states a property as absolute, check whether the test states it that
  way too, and prefer the test's wording.**
- **Both archive abstracts are hand-checked whenever a probe ships** (0.17.0).
  `CITATION.cff` and `.zenodo.json` each state the probe count in prose, and
  `tests/test_citation.py` compares the two files' *titles* only. Record
  22681020 reads "Seventeen probes span"; three older archives say "Fifteen",
  which is what 0.16.1 was spent discovering and cannot be fixed after the fact.
- **An empty `uv lock` diff after a version bump is a symptom, not a pass**
  (0.18.0, and it nearly shipped a wrong artifact). The version lives in **two**
  literals — `pyproject.toml` and `visbench/__init__.py` — and `uv.lock` pins
  only the former, so bumping `__init__.py` alone produces no diff, which looks
  like success and means the wheel would build at the *old* version while
  `__version__` reports the new one. A bump always moves the lockfile.

**The concept DOI is `10.5281/zenodo.21822684`**, unchanged across all
**eighteen** version DOIs and now resolving to v0.23.0 (confirmed against
Zenodo's API on 2026-09-18 — record 21822684 redirects to `id` 22836524, which
reports `metadata.version` `v0.23.0` — rather than assumed).

**That API 302s, and without `-L` the check lies to you**: `curl` alone returns
a 229-byte HTML redirect page and the JSON parse fails with a traceback that
reads as Zenodo serving garbage. Use `curl -sL`. That is the point of quoting it
rather than a version DOI; `tests/test_citation.py` rejects any other Zenodo
DOI in `README.md`, `docs/index.md` and `CITATION.cff`, because pasting one
over it is the realistic mistake and it freezes every citation at one release.

**Releasing — the standing rules, learned across v0.6.0 through v0.16.0.** The
release-by-release detail is in [`ENGINEERING_LOG.md`](ENGINEERING_LOG.md),
under "Release history"; what recurs is:

- **Cut the GitHub release, and check it exists.** The tag archives nothing:
  the *release* is what Zenodo watches, so without it `CITATION.cff` names a
  version no archive holds and the concept DOI still resolves to the previous
  one. Missed on `0.21.0` **and** `0.22.0`, both times with PyPI already
  complete, and nothing turns red either time; held on `0.23.0`, cut as its own
  step. `gh release view vX.Y.Z` is the check, and it is the last step rather
  than an afterthought.
- **Advance `date-released` if the calendar has rolled since the last
  release.** Zenodo publishes it as the deposit's date and a deposit cannot be
  edited afterwards; `tests/test_citation.py` only checks that it parses, so a
  day-old value passes every gate and is then permanent. `0.22.0` was cut on
  the 18th carrying the 17th.
- **Tag before building**, so the artifact is built from the tagged commit.
  v0.10.0 was the first release whose tag and wheel agree exactly, and v0.14.0
  through v0.17.0 are the five with no gap at all, which is what that order
  buys.
- **`git tag -a`, annotated.** `v0.12.0` is the only lightweight tag in this
  project's history and it is the reason `gh api .../git/tags/{sha}` 404s on
  it: there is no tag *object* to fetch, only a ref. Nothing about the release
  breaks, and one API call you will reach for during verification stops
  working.
- **When a verification check imports a symbol, read where that symbol lives
  first.** v0.15.0's wheel check reached for `show_probes` in
  `visbench.cli.main`; it is in `visbench.viz.styles`. It failed loudly, which
  is the right outcome — but a check that imports from a guessed path can only
  ever fail, and a real regression and a wrong import look identical in the
  traceback. The same failure in a different tool: `gh release view --json` has
  `publishedAt`, not `createdAt`, and two guessed queries failed before the real
  field list was read.
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
  dependencies. Match either quote. **And the pin that matters beside twine is
  `packaging`**: the conda env has twine 7.0.0 beside `packaging` 23.2, which
  has no `packaging.errors`, so `twine upload` dies in an import before it
  reaches the network, while the throwaway venv's 26.3 works. "Conda's twine is
  broken" was the wrong diagnosis **three times** — 0.21.0 hit it again, with
  the identical traceback. Upload from the venv that built and checked the
  artifacts.
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
Result schema is at **v9** (`hardware` added 2026-09-11; `training`
2026-08-28; `pooling_requested` in 6e-2b; `finetune` was 6a; `dataset_params`
was 5j) and is **additive only**: never remove or repurpose
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
                 instance.py (VOCInstanceDataset + load_instance_map — per-
                   instance masks from VOC SegmentationObject, class read EXACTLY
                   from SegmentationClass, boxes DERIVED from the cropped mask;
                   void 255 travels as `ignore`, 14a-1)
                 bridges.py (TorchvisionDataset + HuggingFaceDataset — wrap a
                   torch/HF dataset; cache_identity from index-order immutability)
                 bsds.py (BSDS500Dataset — every annotator's boundary map, 4-9
                   per image; native resolution, NO resize or crop; target() is
                   the consensus mean and is NOT the scoring ground truth)
                 navi.py (NaviPoseDataset — unique frames + pairing BY INDEX,
                   the TwoAFCDataset move; the sampler is protocol and is in
                   the fingerprint, val is always 1 partner, mm -> m here)
                 base.py (BaseDataset, list_files — scandir, never a stat/entry;
                   balanced_subset lives here now, not on ImageFolderDataset)
  heads/         base.py (register_head/build_head), linear.py, dpt.py,
                 pose.py (PoseHead — BatchNorm + 512/256/128 over a CONCATENATED
                   pooled PAIR; the first head a BOARD here uses that is not
                   a linear map, and the only one reading pooled vectors; 16a-2)
                 detection.py (DetectionHead — cls + box branches, focal prior)
                 instance.py (InstanceHead — a DetectionHead plus ONE 1x1 conv
                   for masks, reachable as mask_logits(); one module so both
                   branches ride head.state_dict(), 14a-3)
  metrics/       classification, retrieval, correspondence, similarity,
                 instance.py (instance_metrics -> mask_map_50/mask_map_50_95,
                   mask_average_precision, masks_from_instance_map — the
                   DETECTION protocol with the overlap swapped, 14a-2)
                 boundary.py (BSDS500's ODS/OIS/AP — thin_boundaries,
                   correspond_pixels (exact min-cost max-cardinality;
                   sparse, pads the SMALLER side), image_counts,
                   boundary_metrics. Reproduces published human ODS)
                 pose.py (quaternion maths, geodesic rotation error and the
                   no-feature FLOOR — `floor_*`, the `ceiling_*` convention;
                   acos is ill-conditioned at 0, so its own noise is 0.03 deg)
                 dense.py
                 (+ magnitude_metrics — per-image Pearson, masks NaN;
                    edge_metrics is it under the published key;
                    orientation_metrics — coherence-weighted angular error, deg)
                 detection.py (box_iou + mask_iou, average_precision(shapes=),
                   sweep_average_precision, detection_metrics — VOC protocol,
                   dataset-level, difficult ignored not dropped. SHAPE_KINDS is
                   the listed geometry table the matcher reads; 14a-2)
  tasks/         base.py (BaseTask)
                 dense_base.py (DenseTrainingTask — shared by every dense probe;
                   pool_to_grid + evaluate_oracle — the recoverability gate,
                   opted into per probe, no backbone and no fitted head)
                 magnitude_base.py (DenseMagnitudeTask — edge/keypoints2d/
                   occlusion_edge; identity activation, masked L1, correlation)
                 schedule.py (warmup_cosine/check_schedule — probe3d's schedule,
                   shared by DenseTrainingTask and DetectionTask)
                 high_level/  classification, retrieval, semantic_segmentation,
                              scene_parsing (SceneParsingTask — NYUv2-40 on the
                                semantic implementation; 0 is `wall` and 255 is
                                void, MEASURED; num_classes defaults because the
                                probe is named for a label set; 19a),
                              detection (anchor-free, single-scale, 6c-3),
                              instance_segmentation (DetectionTask + RoIAlign +
                                a mask BCE; 14a-4)
                 mid_level/   correspondence, depth, surface_normal,
                              generic_segmentation, similarity, occlusion_edge,
                              pose (RelativePoseTask — pooled pairs by index;
                                the FLOOR travels as floor_*, and train_pairs
                                is in task_params because the score has not
                                converged in it; 16a-2)
                 low_level/   edge (6d-1), keypoints (Keypoint2DTask, 6d-2),
                              corner (CornerTask, 8a — derived target),
                              orientation (OrientationTask — derived, a
                              direction not a magnitude; own _activate/_loss)
  results/       schema.py (ResultRecord, SCHEMA_VERSION), writer.py
  viz/           styles.py (TargetStyle + the listed TARGET_STYLES table —
                   one row per drawable probe, style_for() raises otherwise)
                 colour.py (DisplayRange/display_range, target_to_rgb,
                   voc_palette, INVALID_RGB — pure, no I/O)
                 panels.py (render_probe_panels, render_panels, draw_boxes,
                   draw_instances (one colour per instance, magenta = VOC void) —
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
docs/            conf.py, index.md (the landing page: sphinx-design cards)
                 getting-started/ installation, quickstart, cli
                 guides/      backbones, datasets, dense-probes, visualising
                              (was show.md), sharing (was hub.md),
                              reading-a-board (the CORPUS_FINDINGS rules)
                 probes/      overview, leaderboard, and ONE PAGE PER PROBE
                              under {high,mid,low}-level/ — each carries its
                              gallery figure and its generated board marker.
                              `render_tables.py` derives the file list from
                              `list_probes()`, so a new probe needs its page
                 api/         15 pages of autodoc over 84 of 87 modules. Every
                              directive sits in an ```{eval-rst} fence, NEVER a
                              ```{automodule} one — MyST re-parses a directive
                              body as markdown, so autodoc's generated RST would
                              render as literal text WITH NO WARNING
                 roadmap.md, _static/custom.css, _static/gallery/*.png (the
                   figures — inside docs/ because Sphinx cannot follow ../, and
                   excluded from the sdist)
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
  quoting any board.** Only the *rules* are below — every number, the evidence,
  and the earlier published reading each one corrected are in that file, and
  `scripts/analyse_board_correlates.py` reproduces the correlational ones.
  **Trimmed to rules on 2026-09-17**; nothing was lost, because each claim's
  derivation already lived in that file, in `results/controls/README.md`, or in
  `scripts/merge_corpus.sh`.

  - **"Which backbone is best" is not well-formed against this corpus.**
    `mae_vitb16` is first on **five** of the eighteen boards and last on four —
    a count that was six at twelve backbones and four at thirteen, which is the
    next rule rather than an aside.
  - **A count over a corpus is a fact about that corpus, not a backbone.**
    Re-read counts off `LEADERBOARD.md`.
  - **Quote an objective gap against the *recipe* gap on the same board, never
    against zero** — `sam_vitb16` against `supervised_vitb16`, and nothing in a
    record says which board you are on.
  - **`semantic_segmentation` and `detection` are the two boards that rank by
    neither objective nor grid.** The test is whether another property *beats*
    the grid, never whether the grid coefficient is small — and a coefficient's
    rank among boards is a fact about which boards are in the corpus.
  - **The high-level tier is two clusters, not one**, and two probes that are
    mechanically object classification land in the *localised* one. Treat
    `high_level` as a folder, not a quantity to average over. **Never quote a
    cluster as a property of a *task*** — the split control showed it is a
    property of a board as configured. Not a shared-dataset artefact, checked.
  - **Quote `detection` and `instance_segmentation` to three decimals**, and
    treat `clip_vitb16`/`clip_vitb32` as tied. GPU non-determinism a discrete
    metric can see; there is nothing to fix.
  - **`convnext_base` and `supervised_vitb16` high-level scores are close to
    in-distribution recall**, not transfer — ImageNet-1k supervised on
    Imagenette's ImageNet-1k wnids.
  - **Feature resolution correlates with nearly every dense board, is causal
    about what a representation carries, and the LINEAR board cannot see it**
    (`dino_vitb8`, 2026-09-16). Quadrupling the grid at fixed objective, data,
    width and depth moves the published linear boards by a rounding error and
    the same probes under a DPT head by one to two orders of magnitude more —
    over one sibling pair, and over all ten ViTs (stronger under DPT on 4 of 5,
    largest where the linear head recovers least of its oracle).
    **Never quote a linear board as evidence that resolution does not help**;
    the first draft of this finding did exactly that, and its headline here
    contradicted its own body until 2026-09-16. `analyse_dpt_control.py --grid`
    reprints it. Separately: **check who leads a board before explaining their
    lead.**
  - **The `depth` board is not ranking by metric accuracy** — it ranks ordering
    plus feature resolution and reports it in metres. Not a defect; it
    reproduces probe3d's protocol.
  - **A control is rankable and still must not be listed beside the corpus**
    (`results/controls/`, nothing there feeds a generated table). And
    **`results/corpus/parts/` is an archive, not a queue** — `merge_corpus.sh`
    dedups by exact JSON line, which cannot see a record deliberately kept
    *out*; merge from a staging directory holding only your step's parts.
  - **n=13.** Every correlation above has wide error bars.

- **A trained probe records how its fit went** (schema v8, 2026-08-28; the
  derivation is in `ENGINEERING_LOG.md`). Every trained probe computed
  `train_loss` — and the classification family `train_top1` — printed it to a
  log line and dropped it before the record, so 156 trained records could not
  answer the question this file says matters most about a low score: whether the
  probe **underfitted**, which *understates* a backbone, or whether the
  representation genuinely does not carry the answer. Opposite conclusions from
  the same number. **`training` is a separate field, not entries in `metrics`**,
  because `metrics` is what `evaluate()` returned about the *evaluation* split
  and a training number there is one the ranking code can only refuse; an open
  dict, and **`None` rather than `{}`** for the three zero-shot probes, since no
  fit happened is a different statement from trained and reported nothing.
  **Never rank on it** — on CUB every backbone reaches `train_top1` 1.0000,
  including the one that comes last.

- **A re-run replaces a published record only where it *reproduces* it**
  (the schema-v8 `training` re-run, 2026-09-10; the evidence is in
  `CORPUS_FINDINGS.md`). The corpus is append-only and `latest_per_backbone`
  takes the newest, so re-running a board silently republishes whatever the
  re-run produced. Ninety-three of ninety-six cells reproduced, and the three
  that did not were held out of the corpus rather than merged, because merging
  them would have dropped `convnext_base` below `resnet50` on the CUB board — **a
  ranking change caused by a variable no record carried**. Not picking the
  convenient number; refusing to let an unrecorded variable move a board.

  **Closed 2026-09-11**: `dgx2` left `DRAIN`, the three were re-run on a V100,
  **all three reproduce their published value exactly**, and every board now
  carries `training`. Their A100 records stay in
  `results/controls/hardware_a100.jsonl` because the three-way comparison is
  the finding: same code, data and seed, two silicons, and `convnext_base`
  reaches `train_top1` **1.000000** on a V100 against **0.989156** twice on an
  A100. **The disagreement is a fit that does not interpolate, not a metric
  that wobbles** — which is why the fit diagnostics, not the score, tell the
  two apart.

- **A record says what it ran on, and that field must never be keyed**
  (schema v9, 2026-09-11). Every other field describes the experiment; none
  described the machine, and the degraded node below is what that cost — a
  failing node identified only by diffing every value against the corpus, and
  a cross-silicon disagreement whose evidence lived in a shell history.
  `hardware` is `{device, torch, gpu}`, filled from the **backbone's resolved
  device** rather than the `device` argument, which is `None` on the common
  path and ignored when a constructed backbone is passed — the same reason
  `pooling` is recorded resolved.

  **It is recorded and never grouped**, like `duration_seconds` and `training`.
  Putting it in `comparability_key` would give every GPU its own group, so a
  board whose twelve cells came off two nodes would stop rendering — the
  failure `board_for` raises on — and the measured answer is that such cells
  reproduce to six decimals, so they belong on one board. Three tests pin this,
  including that a pre-v9 record (`hardware: None`) still groups with a v9 one,
  since keying on it would also split the corpus by *age*.

  **`None` dates a record rather than describing a run.** Unlike `finetune` and
  `training`, there is no run for which "no hardware" is the right answer.

- **A degraded node returns plausible wrong numbers, and only the fit
  diagnostics catch it** (same re-run). `dgx2` produced twenty cells before
  entering `DRAIN`; seven of its eight `scene_classification` cells were wrong
  by up to −0.0102, with the same seed, fingerprint and `task_params` as the
  records they disagreed with, and every one carried a visibly *worse fit*
  beside its worse score — which is what identified them. Re-running on healthy
  hardware reproduced all twelve exactly. **A saturated board cannot reveal
  this**: `classification` came off the same faulty node bit-identical, because
  top-1 ~0.99 with `train_top1` 1.0 has no margin left to flip. Check `training`
  before believing a re-run that disagrees, and prefer a board that is *not*
  saturated when you want a canary.

- **`run()` seeds before it constructs a *name*, so a `CustomBackbone` is
  constructed outside the seeded window — and that makes it more reproducible,
  not less** (`examples/custom_backbone.py`, 2026-08-19). On the custom path the
  standing "pass the name" advice is unavailable: a registry name cannot carry an
  `nn.Module`, so the caller must construct. Measured on bit-identical features,
  classification top-1 over seeds 0-4: the wrapped path spreads 0.0062 and
  `run("resnet18")` spreads 0.0063 — **the wrapped path is perfectly
  reproducible** and unchanged by RNG consumed before `run()`, because
  construction happens *before* `set_seed` rather than after it. The
  between-path gap is the same size as each path's own seed-to-seed spread, so
  it is jitter rather than a cost of wrapping; zero-shot probes are identical bit
  for bit. So a wrapped model's number is comparable with another from the same
  wrapped model, and not with a registered backbone's to the last decimal. **Do
  not read this as "the custom path is unreliable"** — that is the intuition it
  was written to correct.

- **The cluster's node map, measured rather than inferred.** The usable set is
  `dgx1`, `dgx2` and `dgxa100` — **`dgxh100` cannot run `.venv` at all**, since
  it ships `/usr/bin/python3.12` and no 3.10, so `.venv/bin/python` is a dead
  symlink there and a job dies in one second with "cannot execute: required file
  not found". Its `--qos=quick` refusal (a misleading `QOSMaxGRESPerJob`, fixed
  by `--qos=normal --gres=gpu:1`) is a *scheduling* obstacle in front of that,
  and clearing it only buys the right to fail on the node — so a QoS fix is not
  evidence a node is usable. `dgxa100` **is** usable, checked by importing torch
  and visbench there rather than inferred from its Ubuntu 24.04: it ships
  `/usr/bin/python3.10` beside 3.12.

  **`dgx1` is degraded and does not fail like a broken node** (found running
  10c). It accepts work and starves it — uniformly **~600x slow**, `import
  torch` returning in **1376.7 s** against ~2 s on dgx2 — and nothing in Slurm
  reports it unhealthy, so a job hangs with an empty log and the first read is
  that your new code is broken. "It hangs" is the wrong summary and invites a
  longer walltime: given three hours it imported, then spent **4.5 hours on one
  CUB cell** that takes 80 s on `dgxa100` and timed out having written no
  record. **Submit with `--exclude=dgx1`**, and when a job here hangs with no
  output, time an `import torch` on the node before suspecting the code.

  **`dgx2` can go `DRAIN` mid-run** (it did, "Kill task failed"), which leaves
  `dgxa100` as the only healthy node — and it is the one whose silicon differs.
  An A100 has TF32 where a V100 has none, so see the reproducibility entry in
  `CORPUS_FINDINGS.md` before putting corpus records on one: TF32 moves a dense
  board by ~1e-6, but three `fine_grained_classification` cells do not reproduce
  across the two. **Do not read `torch.backends.cudnn.allow_tf32` as evidence of
  the hardware** — it is `True` by default and reads `True` on a V100 too, where
  there is no TF32 unit for it to enable; the question is settled by
  `get_device_name` or by measuring the effect.

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
  flag** (`scene_classification` 2026-08-27, confirmed edit for edit by
  `fine_grained_classification`; the full edit list is in `ENGINEERING_LOG.md`).
  `board_for` renders one table per task and **refuses a task with more than one
  comparability group**, and `comparability_key` groups by dataset name and
  fingerprint — so a second dataset under the same task name does not merge with
  the old board, it makes that board **unrenderable**. The name is load-bearing
  in roughly a dozen fixed tables that a test pins equal to `list_probes()`
  (`_REGISTRATION_MODULES`, `HEADLINE_METRICS`, the CLI `SPECS` row,
  `TARGET_STYLES`, both corpus-script arrays, the gallery map and its committed
  figure, `analyse_board_correlates.py`'s copies), plus a `docs/probes/` page
  and a corpus analysis, because **a new board is not inert**. Treat that list
  as complete rather than re-deriving it. Three probes now share one
  implementation and ask three questions — basic-level, place, subordinate — and
  a test pins all three identities *and* that their `protocol` strings differ,
  since the failure mode is two collapsing into one board. One thing the list
  did not cover: **a guard's budget should express what it documents** — the
  gallery's flat 4 MB total let an oversized page pass while there was slack and
  then fired on someone else's reasonable figure, so it is per figure now.
  **Raising a budget to make a guard pass is usually wrong; check first whether
  the budget was measuring the right thing.**

- **The instance probe is `DetectionTask` plus a mask branch, and every
  decision in it is about staying attributable** (14a-3/14a-4). `InstanceHead`
  is a `DetectionHead` plus **one 1x1 convolution** over RoI-aligned features,
  class-**agnostic**, where Mask R-CNN's branch is four 3x3 convolutions and a
  deconvolution on an FPN. RoIAlign carries no parameters, so the only learned
  thing between features and mask is that convolution — the argument behind
  `LinearHead` and `hidden_dim=0`. The board ranks (spread **0.2148** on
  `mask_map_50`, reproducing no other board's ordering) and is **quoted to three
  decimals**, like `detection`.

  Four things not to re-derive. **Both branches live in one module**, or a saved
  probe loads its boxes and predicts blank masks — 9a's `grid_hw` bug, one
  artifact round trip later. **The mask branch trains on ground-truth boxes and
  predicts on detected ones** (`mask_train_boxes: "ground_truth"`), with
  `box_map_50` reported beside `mask_map_50` so a low score is attributable to
  outlines or to localisation. **Target and prediction go through the same
  RoIAlign** — cropping the ground truth by hand puts them on two sampling
  grids, the `recall@1px = 0.003` failure — and the mask bias starts at **zero,
  not the focal prior**, since half a RoI's pixels are foreground. And
  **collecting a split's mask predictions costs 5.4 GB**, feasible only as
  `bool` rather than `float32`; a first draft of that docstring said "~5 MB",
  which was per-image arithmetic labelled as a total — check a memory claim by
  multiplying it out.

- **`instance_segmentation` is a high-level board whose four strongest
  partners are all mid-level — and the mask branch is not why** (14a-4). Mean
  rho **+0.821** against mid-level against **+0.238** against its own tier: the
  sharpest case yet of `high_level` being a folder rather than a quantity. The
  obvious explanation — "mask AP measures outlines, so it ranks with geometry" —
  was checked and is wrong, because the record carries `box_map_50` from the
  same runs and the two halves agree at **+0.986**, so the box half alone ranks
  with the geometry cluster while inheriting every line from `detection`. **The
  split control settled the rest** (2026-09-09) — see the next bullet.

- **A board's cluster membership is partly a property of its *split*, and
  `detection` is the proof** (the split control; every number in
  `results/controls/README.md`, reprinted by `scripts/analyse_split_control.py`,
  with the reading in `CORPUS_FINDINGS.md`). Run `detection` on the instance
  probe's segmentation images instead of its own 600 — same probe, head, losses,
  matcher and metric — and it **changes cluster**. **Never quote a cluster as a
  property of a *task***; it is a property of a board as configured, and **no
  published number moves** — what was contingent was always the reading. It also
  showed the published board underfits relative to the full split (12/12, a
  question only v8's `training` could answer), and **retired an earlier
  finding's argument while the conclusion survived on different evidence**, so
  do not quote the old ordering. **The control as first written down was
  impossible** — checking beat assuming, and inverting it was the better
  experiment anyway.

- **Mask AP is the detection protocol with the overlap swapped, and the
  sharing is enforced by a listed table** (14a-2). `average_precision` takes
  `shapes="boxes"|"masks"`, keys of `SHAPE_KINDS`, and reads the annotation key,
  the coercion and the overlap from that row — one implementation of
  `VOCevaldet.m`'s matching rather than two, so mask AP is comparable with this
  codebase's own box AP. **The guard is the point of the table**: annotations
  carrying the *other* geometry are refused by name, because masks scored as
  boxes read an absent key, coerce to empty and report **0.0** — a silent wrong
  number that looks like a detector finding nothing. Three things not to
  re-derive: **box AP is bit-identical after the refactor**, checked over 4800
  values on 400 random splits, twice, because `detection` is a published board
  and "the tests still pass" is not that claim; **the sweep is an optimisation,
  not an approximation**, and a test pins the swept and naive paths equal on
  both geometries; and **rectangle masks score exactly as their boxes**, which
  is what makes any divergence between the two paths fail on a number. Keys are
  prefixed `mask_` so they cannot sit beside detection's `map_50` meaning
  something else.

- **Instance segmentation is feasible on VOC and not on COCO, and the deciding
  number is grid collisions** (14a-1). At a 16x16 grid the median VOC instance
  covers **16.92 patches** against COCO's 2.27, and **zero of 3207 val instance
  pairs share a grid cell** — no patch is contested, which is what makes a
  per-patch head viable. The instance *index* is only annotation order and can
  never be a stable output channel. Oracle mask mAP@50 **0.6666** against a
  connected-components floor of 0.1376 mean IoU. VOC2012 ships those masks on
  the *same* official 1464/1449 splits `semantic_segmentation` reads.

  Three things the loader must keep doing, each silent when wrong. **An
  instance's class comes from `SegmentationClass` at that instance's pixels and
  the lookup is EXACT, not a vote** — all 6934 instances carry one class at
  purity 1.000000, so a mixed instance means the two files disagree and
  `target()` raises. **The class is read before the crop**, so an instance
  cropped away resolves and is dropped rather than raising. And **boxes are
  derived from the cropped mask**, which deletes the rescale-and-shift hazard
  `data/detection.py` guards rather than re-testing it.

- **Every dense target sits a sub-pixel from its image, in the shipped
  loaders** (found in 14a-1, not fixed). `DenseFolderDataset` resamples targets
  with `torch.nn.functional.interpolate(mode="nearest")`, which is
  left-aligned, while the *image* is resized by PIL, which is centre-aligned.
  The two crops of one VOC map differ on **1.7%** of pixels (1.5% for a
  semantic map), always at object boundaries, and on 28 of 60 images a
  one-pixel roll fits better than none — a sub-pixel offset rather than a
  shift. `VOCInstanceDataset` keeps the shared convention **deliberately**: the
  new probe reads the same 1449 images as `semantic_segmentation`, and
  diverging for one probe would trade comparability for a fractional
  alignment gain. Measured cost on what it is for: oracle mask mAP@50 0.6666
  through the torch path against 0.6638 through a PIL one. **Changing it is a
  decision about five published boards, not about one loader** — so it is
  recorded here and left alone.

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
  oracle.** A match can only land on a patch centre, so a coarse grid has a hard
  floor on achievable precision — `ceiling_recall@5px` is ~0.10 on a 7x7 grid
  against ~0.41 on a 16x16 one — and the score alone therefore says the wrong
  thing. The dense probes have the same problem: the head reads one feature
  vector per patch, so part of every dense target is out of reach before the
  backbone is chosen, and how much varies by *backbone* (`corner`'s ceiling is
  0.8316 on a 16x16 grid and 0.6685 on a ResNet's 7x7), which silently invites a
  reader to attribute a grid difference to a representation. `edge`,
  `keypoints2d`, `occlusion_edge`, `corner` and `orientation` emit `ceiling_*`;
  every other dense probe returns `{}`, because pooling a class-index or
  bin-expectation target is meaningless. `run()` refuses a context key that
  collides with a score, since they share one flat dict, and both prefix
  `ceiling_`. **Never rank or average on a ceiling** — it says what was
  available, not what was recovered, and since it falls with the grid, ranking
  on it would rank feature resolution directly.

  **The corpus carries them because the five boards were re-run**, not because
  the values were backfilled — a number in a record no run produced would be a
  fabrication however easy it is to compute. They are keys inside `metrics`, so
  the schema is untouched. **Four of the five boards reproduced to ~1e-7
  relative; `orientation` did not** — see its entry in
  [`CORPUS_FINDINGS.md`](CORPUS_FINDINGS.md), where its metric is
  ill-conditioned and two of its rows are not separable.

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

- **A range over three draws is not a noise estimate, and a board's reading
  rule was built on one** (19b; the numbers are in `results/controls/README.md`).
  `pose_protocol.jsonl` reports seed ranges of 0.21 and 0.41 for `mae_vitb16`
  and `clip_vitb16`; re-measuring the same two backbones over the same three
  seeds gives **0.72 and 2.23**. Both were computed correctly — the statistic is
  just unstable, and this file already says `spread / noise` has misled in
  *both* directions. **Quote a three-draw range as one sample of a noisy
  statistic, and prefer a bar several independent measurements agree on.** The
  same step's other half: **a perturbation study is how you tell a trigger from
  a dose**, and is cheap, since it needs no new data and no new backbone.

- **A threshold on a gap cannot say whether two rows are ordered; a paired
  re-fit can** (20a; the table is in `results/controls/README.md`). When runs
  are repeatable, **re-run both rows under the same seeds and test the paired
  difference** rather than comparing their gap to a noise figure — the gap is
  one draw, and on the pose board pairs 1.58 apart are solidly ordered while a
  pair 0.12 apart is **reversed by 2.08**. Widening the threshold to the
  measured noise would have called two ordered pairs tied and still missed the
  reversal. Only 11% of that board's seed variance is common-mode, so the two
  rows move independently and no function of the gap can carry the answer.
  **The gate that makes such a sweep mean anything is free**: the corpus cells
  were run at seed 0, so the sweep's own seed-0 rows must reproduce them, and
  all thirteen did at delta 0.0.
- **Constructing a backbone draws from the global RNG, and `run()` seeds
  *before* it constructs.** So `run("dinov2_vits14", ...)` and
  `run(get_backbone("dinov2_vits14"), ...)` fit the head from different RNG
  states and produce different trained numbers, with every recorded field —
  seed included — identical. **Pass the name; take the object back off
  `RunResult.backbone`.** Building one outside `run()` puts its random init
  (DINOv2 and timm both initialise randomly before loading the state dict)
  outside the seeded window.

  Found by publishing a full board and diffing it against the corpus: 20 of 26
  records differed and **the 6 that reproduced were exactly the zero-shot
  probes**, which train no head. That signature — trained probes all move,
  zero-shot ones all reproduce, no recorded field explains it — means seeding
  rather than a version regression, and it was misread as the latter first.

  **The obvious regression test for this is vacuous.** Comparing a pushed run's
  metrics against an unpushed one looks decisive and is not: the CLI fixtures
  are three colour-separable classes, so both sides read 1.0 however badly the
  RNG is threaded. Pin the backbone's *weights* against a freshly seeded
  construction — that is what the seed decides and what actually moved. The
  board was re-run after the fix and 24 of 26 DINOv2 records reproduce the
  corpus exactly, the two exceptions being `detection` in the fourth decimal.

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
  yourself with** (8a; the numbers are in `visbench/tasks/low_level/README.md`).
  It needs no dataset, and three things had to be measured before it was worth
  shipping, none of which a probe run would have revealed. **Check the tail
  before writing the task** — `edge_occlusion`'s concentration is the case that
  scored 0.088 and ranked nothing, so `corner` uses `log1p` and **Shi-Tomasi
  rather than Harris**, λ_min being non-negative by construction with no `k`.
  **Check the overlap with what already ships**: the corner target correlates
  0.52 with `edge_texture`, *intrinsically*, holding across eight transforms —
  a first pass blamed the `log1p` and was wrong. **A correlated target can still
  rank differently, and that is the criterion**; had the ordering matched, it
  should not have shipped, and **one pair is not a failure to rank** — ask about
  the spread over the full set. **Computing the target after the crop deletes
  the alignment hazard rather than testing for it**, which is why
  `DerivedTargetDataset` does not subclass `DenseFolderDataset`.

- **The gauntlet asks whether a target is distinctive; it never asked whether
  it is *recoverable*. Photometric superpixels is what that cost** (built and
  rejected 2026-08-28). SLIC boundary regression passed every gate and then
  scored **0.0434 / 0.0209 / 0.0238** on DINOv2-S, CLIP-B/16 and ResNet-50,
  where the weakest shipped low-level probe scores 0.179-0.236 — spread 0.023,
  and `train_loss` **lowest** for the worst scorers, so the heads had learned
  the mean boundary density and nothing about location.

  **The missing check was an oracle, and it now ships** (2026-09-01).
  `DenseTrainingTask.evaluate_oracle` pools the target to the feature grid,
  upsamples it back and scores it with the probe's own metric — what a perfect
  backbone would make available, since a dense probe sees one feature vector per
  patch and signal finer than a patch is *absent from its input* rather than
  merely hard to predict. No backbone, no features, no fitted head, so it costs
  one pass over a split rather than a board.
  `CorrespondenceTask.evaluate_ceiling` is the same idea, arrived at the same
  way. **Run `scripts/oracle_ceiling.py` before writing the next derived
  task**; the numbers are in the "oracle gate" section of
  `visbench/tasks/low_level/README.md`, and the bar it calibrates is the four
  shipped magnitude targets at 0.53–0.83 against superpixels' **0.25** on a
  16x16 grid (0.43–0.67 against 0.11 at a ResNet's 7x7). Five things about it
  that are not obvious:

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
  - **It measures a candidate's ceiling and nothing measured its floor, which
    is the gap relative depth ordering cost** (2026-09-04). Relative depth
    **cleared the gate at a 94.0% oracle and was rejected anyway**: "the lower
    point in the image is nearer" scores **65.2%** with no features at all, so
    the usable band was 0.157 wide and the three strongest backbones landed
    **0.0007** apart. `corner` ranks fine at a comparable 0.83 ceiling *because
    its trivial floor is near zero*. **A ceiling of 0.9 above a floor of 0.7 is
    a worse probe than a ceiling of 0.6 above a floor of 0.** Name the cheapest
    shortcut — an image coordinate, a per-image constant, the dataset mean —
    and measure it on the samples the metric will use;
    `scripts/premeasure_ordering.py` is the worked example.
  - **It models a *linear* head exactly, and it is a bar for the head VisBench
    reports rather than a bound on what is achievable** (the DPT control; every
    number is in `results/controls/README.md`, which
    `scripts/analyse_dpt_control.py` reprints). A DPT head reaches 54-104% of
    the oracle across five probes and nine ViTs, exceeding it only for
    `mae_vitb16` — **not a property of decoders**, which is the reading the
    n=2 version invited. **A head is not a neutral magnifying glass**: two of
    five ViT boards change leader, 24 of 174 separable pairs reorder, and on
    the CNNs `convnext_base` goes first to last — the demonstration behind
    reporting the *linear* number when comparing representations. **A DPT
    number is good to three decimals** where the linear boards reproduce at
    ~1e-7, so count a reordering only over pairs both boards separate by more
    than that. **A CNN's DPT run moves the oracle too**, so only the
    DPT/linear *gain* is comparable there and the ViT group is the clean
    control.

  **It has now refused something** (2026-09-01). The BSDS500 probe was not built
  because the gate put a linear probe's ceiling at **0.4193 ODS** on the 16x16
  grid every corpus backbone produces, against published detectors at 0.60-0.79
  and human agreement at 0.80 — one 60-second run instead of a 12-backbone
  board. Scaling it by the best DPT ratio seen anywhere (1.038) gives ~0.435,
  so the DPT result does not reopen it. **Do not read that 0.42 against the 0.25
  that rejected superpixels** — one is ODS and the other Pearson correlation,
  and an earlier draft made exactly that mistake.

  **A pooled-resolution overlap check nearly became a false veto**: the
  boundary map reads 0.267 against `edge` at full resolution and 0.684 pooled to
  a 16x16 grid, which looked decisive until the shipped `corner` target read
  **0.781** there and its board ranks differently from `edge` anyway.
  **Calibrate a new rejection criterion against something that already passed
  before letting it reject anything.**

  What survived: `DerivedTargetDataset` memoises computed targets
  (`MEMO_LIMIT`), because `CachedFeatures.__getitem__` calls
  `dataset.target(index)` on every access — a ten-epoch streaming run was
  recomputing every target ten times, which `corner` and `orientation` both
  paid.

- **The overlap check is a veto, and `orientation` is the probe that proves it
  earns its keep** (2026-08-28). DoG-blob detection was the obvious next derived
  probe and **was rejected without a probe run**: its tail passed, but per-image
  `|r|` was 0.50 with `edge_texture` and **0.51 with `corner`** — as redundant
  with an existing probe as `corner` is with `edge`. The check exists so you do
  not spend a per-backbone board to discover redundancy.

  **Structure-tensor orientation was the alternative and it pre-measures
  clean**: `|r|` under 0.09 with both `corner` and `edge`, because it measures
  *phase* and no other probe does. Its target is a *direction* — `(cos 2θ, sin
  2θ)`, the angle mod π so the double angle handles the wrap, with the coherence
  folded into its length — so it is the first derived probe that could **not**
  reuse `DenseMagnitudeTask`: a 2-channel L2-normalising `_activate`, a
  coherence-weighted angular `_loss`, and `orientation_metrics` in degrees.
  Coherence is a **weight, not a mask** (only 1.4% of pixels fall below 0.1),
  folded into the target length exactly as a zero-length normal marks an invalid
  pixel, so loss and metric both weight by `target.norm(dim=1)`. An angle has
  **no tail**, so `corner`'s compression is absent here — confirmed by the
  pre-measurement before the task was written. It is drawn in **colour**, a
  `Kind` of its own mapping `2θ` to hue and coherence to brightness, so a flat
  patch reads as black rather than a confident wrong colour.

- **A viewer that applies its own geometry is worse than no viewer** (9a; the
  measurements and the four bugs are in `ENGINEERING_LOG.md`). A panel's entire
  evidential content is whether the image and the target line up, so a viewer
  that resizes, re-reads the source or re-crops can make a *misaligned pipeline
  look fine and a correct one look broken*. Guaranteed by pasting
  `np.asarray(dataset[i][0])` unchanged; a fast test pins it byte-for-byte.
  Four rules: **four validity conventions, one listed table, no fallback**
  (`TARGET_STYLES`, `style_for` raises — a "mask the zeros" default is silently
  wrong for the four probes where 0 is real, and it *renders*); **a prediction
  is drawn against the target's range, not its own**, and the test asserts both
  halves, since only the second fails if someone simplifies it back; **magenta
  marks invalid** because no colouriser here can produce it, asserted as a
  distance; and **captions are ASCII** (PIL's built-in font draws an empty box
  for an em dash). It deliberately does not train — that is `run --save-probe`.

- **The three probes with no spatial target draw their *decision*** (9c).
  `classification`, `retrieval` and `similarity` have nothing to lay beside the
  image, so they draw the choice, and `show_probes() == list_probes()` is
  asserted so a new probe cannot ship undrawable. Four rules survive.

  **`class_balance` and `vote_balance` are the prefix bug and the CSV-column
  bug as figures**, and both are **diagnostics, never scores**, like
  `error_coherence`. **Frames are picked spread across the split** for the
  class-grouped kinds, since a prefix would reproduce the artefact the sheet
  exists to reveal. **Retrieval loads the whole split whatever `--frames`
  says** — leave-one-out over four images ranks each against three, so `--limit`
  (how much to load) is distinct from `--frames` (how many rows to draw). And
  **classification keeps its own schedule defaults**
  (`CLASSIFICATION_SCHEDULE_DEFAULTS`, 200 epochs at 1e-2), or `show` builds a
  probe with the wrong ones and `load_probe` refuses a head that is fine.

  Two bugs there were found by **rendering a page, not by a test**: PIL's
  built-in font has no glyph for an em dash or ellipsis and draws an empty box,
  so every caption here is **ASCII** and a test asserts it; and a fixture whose
  vote column held a raw tally read as "humans chose right in 0%", caught by the
  footer figure that exists for exactly that.

- **For correspondence it is the *shape* of the errors that diagnoses the bug,
  not their size** (9b). `error_coherence` is the mean resultant length of the
  error directions: measured on 224px homography pairs with ResNet-18, a
  *correct* geometry gives median error 10.2/22.6 px at coherence **0.40/0.29**,
  and the homography in the wrong pixel frame gives 293.9/226.6 px at
  **0.98/1.00**. **The median cannot make that call and the coherence can** — a
  weak backbone and a broken pipeline produce overlapping medians. It is a
  **diagnostic, never a score**: not recorded, and it must not reach a
  leaderboard.

  Two rules from it. **`match_details` is on the task and the renderer calls
  it**, so the panel and the number come from one code path by construction; a
  renderer that recomputed the geometry would put a drawing that vouches for a
  wrong number one edit away. And **matches are sampled evenly, never from the
  front** — `match()` returns them sorted by descending similarity, so a prefix
  draws the most confident few and shows a better picture than the score
  describes.

- **`TimmBackbone` reads a model's own structure; it used to assume a CNN's**
  (10a; derivation in `ENGINEERING_LOG.md`). `has_cls_token` and `patch_size`
  were *class* attributes declaring "CNN" for everything, so timm ViTs were
  refused — and a false `has_cls_token` discards the CLS token while the record
  claims there was none to keep. Read per instance, any timm ViT becomes usable
  *and honest*. Three decisions inside it, each producing a silently wrong
  number rather than an error — which is why `describe_transformer` is a
  module-level function with **fast** tests over a stub: **`default` pooling is
  read from timm's `global_pool`**, not inferred from whether a CLS token
  exists (MAE reports `token`, SigLIP-GAP `avg`); **SigLIP is the `_gap_`
  variant deliberately**, since canonical SigLIP pools with a *trained module*
  that cannot be a pooling mode over cached features, and `map` is refused by
  name; and **ConvNeXt breaks the "pooled is what the model hands its
  classifier" rule**, documented rather than smoothed over, because the
  invariant kept is structural — **`pooled` is always a reduction of `dense`**.

- **The docs gallery is real photographs, and the licence rule was satisfied by
  better sourcing rather than waived** (9d/11a; full write-up in
  `ENGINEERING_LOG.md`). **VOC, ImageNet, NYUv2, Taskonomy and NIGHTS all
  restrict redistribution and appear nowhere in this repository**; Open Images
  validation is CC BY 2.0, the frames are committed, and **the licence is
  verified per frame rather than inherited** because CC BY compliance rots
  silently — the page renders correctly either way. **Four probes cannot have a
  target column and must not be given one** (`depth`, `surface_normal`,
  `keypoints2d`, `occlusion_edge`): they need sensor geometry no redistributable
  photograph carries, so they render `image | prediction` from a published head
  with a footer saying so. **The figures live under `docs/_static/`**, since
  Sphinx cannot follow a path escaping its source tree and `-W` would not catch
  it. **Every gallery bug so far was found by looking at the output, never by a
  test** — so when a page cannot be rendered in the fast suite, make what it
  *says* a pure function.

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
  reads** — and `scripts/stage_corner_frames.py` makes them readable by
  symlinking the building-nested RGB frames into the flat `<split>/images/`
  layout `DerivedTargetDataset` expects, with `build_corpus.sh` skipping the
  probe with an actionable message if that folder is absent. **Shared frames are
  the point, not a convenience**: the claim that earns `corner` its place is
  that it ranks differently from `edge` despite a 0.52 target correlation, which
  is exact only on identical pixels — so the staging is verified **set-equal**
  to the edge probe's 600 rather than assumed equal. **Symlinks, not copies**,
  since `cache_identity` keys on path, size and mtime and a symlink reports its
  target's, so a staged frame shares one cache entry with the original. The cost
  of getting this wrong was demonstrated rather than argued: 8a's numbers were
  produced on an ad-hoc staging that was never committed, so its six published
  figures had **no surviving records** — 6e-2's exact failure, two steps after
  that step ended it.

- **A `{automodule}` fence in MyST produces garbage and emits no warning**
  (13a). This is the single most expensive thing the API reference could have
  got wrong, and `-W` cannot catch it. MyST renders a directive's body with
  `nested_parse` **as markdown**, so autodoc's generated reST — `.. py:class::`,
  `:param x:`, tables, roles — arrives as literal paragraph text. It builds
  clean and the page is nonsense. **Every autodoc directive on this site sits
  in an ```` ```{eval-rst} ```` fence**, which routes through a real
  `RSTParser`. Two rules follow: never put a reST section title inside such a
  fence (it is a standalone parse, so the title is legal and splices in at the
  wrong level), and write every heading as a markdown `##` *outside* the fence.

- **Docstrings had been written for an API reference for six steps and none of
  them had ever been rendered** (13a). ~5,198 lines of numpydoc went through
  docutils for the first time and nine source files had real defects — malformed
  simple tables, a `#:` block whose `History/-----` reached docutils as a
  section title (fatal under `-W`), `Returns`/`Raises` sections whose free prose
  had no type line so napoleon read *the prose* as the type, and dead
  cross-references. **A docstring convention nothing renders is not a
  convention, it is a guess** — `scripts/check_docstrings.py` runs each one
  through napoleon and docutils in the fast suite, in ~1s with no Sphinx
  *build*, by indenting the result three spaces under a dummy directive
  (un-nested, a section title is legal and docutils says nothing). It documents
  what it **cannot** reach and a test asserts that limit. Its `sphinx` import
  was the optional-extra trap for the third time — see that bullet below.

- **Only prose drifts, and the one measured number with no generated table
  drifted three ways** (13a). The pooling-mismatch pair is quoted by hand in
  the CLI help, an example, two guides, an API page and both archives. The
  README said 0.9540/0.9830, `docs/hub.md` said 0.9620/0.9895 and `docs/show.md`
  said 0.9620/0.9820; the record in `ENGINEERING_LOG.md` says **0.9620 against
  0.9820** and a test pins the card at 0.9820. Two of the three published
  values were wrong and nothing failed. `tests/test_docs_numbers.py` reads the
  pair **out of the engineering log** — not out of its own source, which would
  be a fourth place to drift — and fails any file that quotes it differently.

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
  the filter has two details that each cost an attempt** (7d). A
  `ConnectionResetError` fetching an `objects.inv` is logged as a warning, which
  `-W` turns into a failed deploy — it did, on the first push to `main`, minutes
  after the same commit passed on its PR. Losing intersphinx degrades gracefully
  (nitpicky is off), so the *warning* is the only real problem, and it carries
  no `type=`, so `suppress_warnings` cannot target it. The filter in
  `docs/conf.py` matches that one message, and it goes on the **handlers, not
  the logger** (Sphinx emits from child loggers, and a parent's filters never
  see a propagated record), inserted at **position 0, not appended** (`-W` is
  itself a filter on the same handler, so anything appended after it never runs
  — which looks correct and does nothing). Verified by checking that a broken
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
- **The optional-extra trap has now been hit four times** (the four instances
  are in `ENGINEERING_LOG.md`). CI installs `.[dev]`, so a fast test that needs
  `clip`, `timm`, `hub`, `yaml`, `datasets` or `sphinx` is green locally and red
  on every pull request. **Run the `find_spec` import blocker `CONTRIBUTING.md`
  documents whenever a fast test touches one**; the six verification commands
  cannot catch this, because they run in the environment that has everything.
  **Having run the docs build is not coverage for Sphinx** — that build installs
  `.[all,docs]` and the test suite does not, which is how the third one reached
  a PR. **Declare it; do not skip it**: a `pytest.importorskip` makes the suite
  green and leaves the guard uninstalled in exactly the environment that gates
  every PR. **The fourth instance is not a package but `.venv/` itself**, so the
  family is wider — **a test must not depend on anything that exists because of
  how this machine is set up**; build the stub the test needs itself.

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
fast suite **collects 2304 tests**, green on 2026-09-18 along with all three
lint steps, mypy and the `-W` docs build. The slow suite is **116** since
16a-1, whose own slow test was run then; the other 115 were last green on
`main` on 2026-09-11. Earlier fast counts, for dating a claim: 2301 at 20a, 2296 at 19b, 2281 at v0.21.0, 2222 at 16a-1,
2261 at the pose board, 2160 at the grid finding, 2158 at
the control guard, 2144 at
`dino_vitb8`, 2126 at the docs-count guard, 2122 at the 0.18.0 release, 2113 at
the v8 `training` re-run, 2082 at the 0.17.0 release, 2071 at the instance
board, 1824 at the oracle gate. Keep this list
short — it is one of the places this file accretes.

**Quote the collected count, not "N passed", because the skip count is a fact
about the environment rather than the suite.** Measured on one commit
(`06b85bb`, 1930 collected): **1927 passed + 3 skipped on CI, 1922 + 8 on
`.venv`, 1930 + 0 on a machine with every extra installed.** So a remembered
"(8 skipped)" reads as universal and is wrong two ways out of three, and the
three environments cannot all be right about "N passed".
`pytest --collect-only` is the number that means something.

Run the checks via `sbatch` (`.venv/bin/python` does not resolve on a `dgxh100`
login shell) and **`--exclude=dgx1`**, per the cluster's node map above; the
usable node is `dgx2`. `uv lock --check` is green as of 2026-09-05 — 13a moved
it twice, adding `sphinx-design` to `docs` and `sphinx` to `dev`, which is the
first dependency change since 2026-08-06. If anything is red for you, that is
new — do not go looking for a known cause here.

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
- **[#3] CLIP QuickGELU guard** — fixed; see above.

`CHANGELOG.md` is the full record of what each step added and why. Since 7b the user-facing view is **split three ways**:
`README.md` is the arrival path (demo, install, what it is, the CLI), while the
reference material lives on the docs site and the roadmap and backlog in
`docs/roadmap.md`. `CONTRIBUTING.md` is the public version of the conventions
this file keeps. All are kept current per step — update them in the same commit
as the code, not afterwards, and put a new probe's measured numbers on its own
`docs/probes/` page rather than in the README.

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

`docs/roadmap.md` has the public version of this list, grouped by cost. What
follows is the part a contributor cannot see: **which of these have data on this
machine**, checked on 2026-08-01 rather than assumed. A candidate whose dataset
is absent is not cheap, however simple its protocol.

**`/shared/sets/datasets/` has a `vision/` subdirectory, and a top-level listing
does not see into it.** 96 more datasets live there, including ones a first pass
recorded as absent — this note exists because that pass said Places365 and
NIGHTS were missing when both are on disk. Check both levels.

**Present at the top level:** `ADE20K` (`ADEChallengeData2016`), `COCO`
(`instances_*`, `captions_*`, `person_keypoints_*` — **no panoptic and no
stuff**), `cub_200_2011`, `stanford_cars`, `stanford_dogs`, many ImageNet
variants, `Imagenette`. **Present under `vision/`:** `nights` (what `similarity`
reads), `places365_standard`, `SUN397`, `mit67_indoor_scenes`, `caltech101`,
`country211`, `CUB-200`, `oxford_flowers102`. Two of those became probes as
dataset swaps on the linear-probe path — `scene_classification` on
`places365_standard` and `fine_grained_classification` on
**`vision/CUB-200/images_train_test/`**, which already holds the official
5994/5794 split as `train/<class>/` + `val/<class>/`. Two traps there: the
top-level `cub_200_2011/CUB_200_2011` is **permission-denied**, and `test/` is a
**symlink to `val/`**, so naming `val` names the official test set while
`--split test` would index the same files under a different path and so a
different fingerprint. Stanford Cars is the same folder shape and still open;
Stanford Dogs and Flowers102 are **not**, since both keep their splits in
`.mat` files — a different cost class from a folder swap.

**Absent at both levels:** any optical-flow set (Sintel, KITTI, FlyingChairs),
any intrinsic-image set (IIW, SAW, MIT intrinsic). `bsds300` is the MAF
density-estimation benchmark, not BSDS500; **BSDS500 itself is reachable only
through a mirror**, since Berkeley times out from this machine while the network
is otherwise fine, so `scripts/fetch_bsds500.py` reads the `BIDS/BSDS500` GitHub
mirror at a pinned commit into gitignored `data/bsds500/`.

**Two entries this survey recorded as absent are present, and both were found
by listing rather than by the note being doubted** (2026-09-14). The lesson is
the one the `vision/` subdirectory already taught: **a nested path defeats the
listing that produced this section, so re-check before quoting it as a
blocker.**

- **NYUv2 is here, twice, in two cost classes.**
  `vision/probing_3D/nyuv2_new/{train,test}/` is the **cheap** one — the
  canonical **795/654** labeled split already extracted as stem-matched folders
  `images/` (png), `depths/`, `normals/` (both `.npy`) and
  `segmentation_nyu40/` (png). That is `DenseFolderDataset`'s layout, and
  `.npy` needs no new loader: `_TARGET_SUFFIXES` has carried it since 5d and
  `load_depth_map`/`load_normal_map` take it at face value. **A first draft of
  this bullet said it cost a loader** — the claim was written from "every dense
  target in the corpus is a PNG", which is true and says nothing about what the
  loaders accept. Check the suffix table, not the datasets.
  `vision/probing_3D/nyuv2/` is the raw form
  (`nyu_depth_v2_labeled.mat` 2.9 GB, `nyuv2_snorm_all.pkl` 8.9 GB,
  `labels40.mat`) — the `.mat` cost class that puts Stanford Dogs and
  Flowers102 out of "folder swap" reach. `nyu_geonet/` is a third, per-frame
  `.mat`.
- **DAVIS is here.** `vision/DAVIS` has `JPEGImages/480p` and
  `Annotations/480p` over **90 sequences** with `ImageSets/{2016,2017}/
  {train,val}.txt`. It is the *top-level* `davis` that holds two sequences of
  derived output — the two are different directories, and the old note named
  the wrong one. Video still needs machinery nothing here has; what changed is
  that the data is no longer the blocker.

**The Taskonomy copy on disk carries eight domains only**: `depth_zbuffer`,
`edge_occlusion`, `edge_texture`, `keypoints2d`, `keypoints3d`, `normal`,
`principal_curvature`, `reshading`, plus `rgb` and `mask_valid`. Taskonomy
*publishes* `vanishing_point`, `room_layout`, `segment_unsup2d/25d` and
`point_matching`, and none are here — so the roadmap items that look like free
Taskonomy wins each need a download first.

**The cheapest items need no dataset at all, and that is the useful
observation**: `edge_texture` is a target Taskonomy computed from the RGB frame,
and so are these. `corner` and `orientation` are done, DoG blobs was rejected,
and **photometric superpixels** is the one that remains derivable from any image
folder already here. A magnitude target is a generator plus a
`DenseMagnitudeTask` subclass; a vector one needs its own small task base, which
`visbench/tasks/low_level/orientation.py` provides as the second worked example.

**Relative camera pose ranks, and it needs a nonlinear head** (measured
2026-09-14, **corrected 2026-09-15**, and **being built since 16a-1**;
`scripts/premeasure_pose.py`, and the full derivation — every pair count, the
floor, the curve and the traps — in `visbench/tasks/low_level/README.md`,
trimmed from here 2026-09-17). It reads *pooled* features, so it is far cheaper
than the roadmap's "harder" tier implies, and NAVI is on this machine. Six
rules survive, the last two from 16a-1:

- **`LinearHead` cannot express it** — it underfits, and its residual ordering
  nearly inverts the MLP's at the top. **So a pose board departs from
  `hidden_dim=0` deliberately or not at all**, which is a decision about what
  VisBench's boards mean, not an implementation detail.
- **A pose board must pin its pair count as protocol**, the way `corner` pins
  its frame set. The curve never flattens, the pairing rule runs out first, and
  the spread *compresses* as data grows — so part of a small-sample gap is
  **data efficiency**, not representation.
- **Before concluding a probe fails to separate, check the training-pair count
  against the head's input width.** The first reading said the opposite of the
  corrected one and was **overfitting at n≈d**; a candidate measured where the
  head cannot generalise looks exactly like one that does not rank.
- **Quote the per-row margin over the floor and the adjacent gaps, never the
  spread** — `spread / noise` has now misled in *both* directions. And reading
  a range as signal without its floor is what the floor rule exists to stop; it
  was done while writing the script that measures the floor.
- **A pre-measurement that assembles its own data stops predicting the probe.**
  The script opened NAVI's frames without their EXIF orientation, and **327 of
  8,217 carry a half-turn tag whose camera pose describes the *turned* image**
  — checked against the untagged frames of the one mixed scene, not assumed —
  so 4.0% of the release was supervised against its own negation and the parked
  tables were measured on those pixels. `NaviPoseDataset` owns the pairing now
  and the script reads it.
- **A context metric qualifies a score from either side, and the leaderboard
  has to know both** (16a-3). `CONTEXT_PREFIX` was the single string
  `"ceiling_"`; `relative_pose` emits `floor_*`, and an unknown prefix does not
  raise — `metric_direction` refuses the name, `shared_metrics` skips what it
  cannot direct, and **the floor silently does not appear on the board**, which
  is the number the probe is read against. It is `CONTEXT_PREFIXES` now. Same
  family: a *parametrised* metric name (`rotation_acc@30deg`) is directed by its
  stem, and an exact name nobody listed is dropped rather than refused.
- **A correction is read against the noise, exactly like a score** (16a-2).
  The EXIF fix was right — 4.0% of NAVI was supervised against its own negation
  — and it moves a pose number **−1.38 on `mae_vitb16` against seed ranges near
  1.0, and −0.02 on `clip_vitb16`**, i.e. not beyond noise. Quoting the first
  row alone as "worth 1.4 degrees" is the floor rule's mistake wearing a
  correction's clothes. What *did* show is per-epoch reshuffling: same
  direction on both rows and a seed range cut 3-5x. And **`train_loss` from two
  code paths is two statistics** — the pre-measurement reports the last
  training batch in train mode, the task a full split in eval mode.
- **A metric can be unable to validate the conversion feeding it.** Scored
  through `rotation_error_deg`, Shepperd's branch and the naive one read 0.0485
  against 0.0560 deg and both fail a 1e-3 tolerance, because `acos` of the
  trace is ill-conditioned at zero error — its own noise floor is 0.028 deg on
  a quaternion against *itself*. On components they are 1.2e-07 against
  1.2e-04, three orders apart. **Validate a conversion on components; the angle
  is for scoring a probe.**

Three hazards to carry into any of them, all paid for: **check the tail** before
assuming the magnitude protocol transfers (`edge_occlusion` at 46% of its mass
in the strongest 1% of pixels is where L1 and Pearson pull apart and the probe
stops ranking); **check the overlap with what already ships** before building,
which is one afternoon of correlations rather than a board; and **a derived
target is only as honest as its generator, so `protocol` must say which** —
"Harris corners" is a family, not a definition, and the k parameter, the window,
the smoothing and the non-maximum suppression all move the target.

### The library-surface backlog — closed 2026-08-28

All three shipped — `visbench show` (9a-9d), `examples/custom_backbone.py` and
the **dataset bridges** — and **none was a defect**: each was already reachable
by writing Python, so the backlog changed no number, as v0.7 did.
`docs/roadmap.md` has the public version and `ENGINEERING_LOG.md` the full
write-up. What the bridges left as rules: `datasets` is an extra imported
lazily and **declared in `dev` too**, or the bridge tests skip in CI;
`--dataset torchvision:… / hf:…` replaces `--data` on the three **image-level**
probes only, since a dense, pair or triplet target is a much larger surface;
**`cache_identity` is the method a bridge must not skip** — return `None` and
every run re-decodes every image forever while appearing to work, so both lean
on index-order immutability; and `describe()` adds `dataset_source` so a bridge
record lands in its own comparability group rather than merging with a folder
board.

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
  never something to rank on. Every run records `hardware` (v9) — what it
  executed on, which is neither ranked nor **grouped**: it says what produced a
  number, not whether two may be compared.
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

pytest                                              # 2301 fast tests
pytest -m slow                                      # 116, real DINOv2/CLIP weights
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

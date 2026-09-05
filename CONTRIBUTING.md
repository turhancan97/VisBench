# Contributing to VisBench

Thanks for looking. This document is the short version of how the project is
built and what a change has to satisfy before it lands.

VisBench measures things, so most of its rules exist to stop it reporting a
number that means something other than what it says. That is the thread running
through everything below: **the failure this codebase guards against is not a
crash, it is a plausible wrong answer.**

## Getting set up

```bash
git clone https://github.com/turhancan97/VisBench && cd VisBench
uv sync --all-extras          # exact locked versions — what CI and the numbers used
# or, without uv:
pip install -e ".[dev,clip,timm,hub,datasets]"
```

Python 3.10 or newer. The floor is 3.10 because DINOv2's pinned revision uses
syntax 3.9 rejects.

Check it works — this needs no dataset and no large download:

```bash
visbench demo
```

## The checks

CI runs five commands. **Run them exactly as written**, from the project
environment, before opening a pull request:

```bash
pytest                                              # fast tests
pytest -m slow                                      # downloads real weights
ruff check visbench/ tests/ conftest.py examples/ scripts/
ruff format --check visbench/ tests/ conftest.py examples/ scripts/
mypy visbench/ examples/ --ignore-missing-imports
```

`mypy` reads its settings from `pyproject.toml`; overriding them locally checks
nothing useful. Two more jobs gate CI that these do not cover:

- `uv lock --check` — **a dependency or version change means `uv lock` in the
  same commit.** `uv.lock` pins VisBench itself, so bumping `version` without
  re-locking passes all five commands and fails CI.
- `python -m build && twine check dist/*` — the README is PyPI metadata.
  Install `build` and `twine` into a throwaway venv, not the project one, so
  `.venv/` keeps matching what CI has.

### Documentation

The site at <https://turhancan97.github.io/VisBench/> is built by Sphinx from
`docs/` and deploys on merge to `main`. A **third workflow**,
`.github/workflows/docs.yml`, builds it on every pull request — so a broken docs
build fails review rather than `main`.

```bash
sphinx-build -b html -W --keep-going docs docs/_build/html
```

`-W` makes warnings fatal. If you add a page, add it to a `toctree`, or the
build fails with an unreferenced-document warning.

### Fast tests and slow tests

`pytest` deselects `slow` by default, and CI's gating workflow runs the fast
suite only. The slow suite runs on pushes to `main` and nightly.

**A guard whose only test is `slow` is a guard CI never runs.** If a check
exists to prevent a *silently wrong number*, it belongs in the fast suite — pull
the logic into a pure helper if that is what it takes. This is not hypothetical:
a CLIP configuration guard filtered on a warning phrase the library never
emitted, and was dead code for its entire life while its test passed.

### If you touch an optional extra

`clip`, `timm`, `hub`, `datasets` and `docs` are optional. `datasets` and
`sphinx` are also in `dev` (the HuggingFace-bridge tests and the docstring guard
need them at import time), but `clip`/`timm`/`hub` are not and **CI installs
`.[dev]` only** — a test that imports one of those passes locally and fails in
CI. Reproduce CI's environment by blocking the import:

```python
# /tmp/blockhub.py
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "huggingface_hub":
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None
sys.meta_path.insert(0, _Block())
```

```bash
PYTHONPATH=/tmp pytest -p blockhub
```

Inject a stub module rather than skipping the test — a skip leaves the code path
untested in exactly the install most people have. If the test cannot work
without the package (a guard that genuinely needs `docutils`, say), **add the
package to `dev`** rather than skipping: a guard that disappears in the
environment CI uses is the "only tested under `slow`" failure with a new label.

**Running the docs build does not cover this.** That build installs
`.[all,docs]`; the test job does not. A fast test that imports `sphinx` can pass
every local check, including the `-W` build, and still be red on the pull
request — which has happened.

## What a change needs

**Tests alongside the code, not afterwards.** Every module here has them.

**A test that fails when the thing it guards is removed.** Worth checking
directly: delete the guard, run the suite, confirm it goes red, put it back.
Several tests in this repository passed while asserting nothing — one checked
that `` `num_matches` `` appeared *somewhere on the rendered page*, which the
surrounding prose satisfied whether or not the column existed.

**A numeric parameter must be tested by changing it.** A dataset option was
accepted, recorded and folded into the fingerprint while having no effect on the
data for a whole release. Nothing raised; the sweep that should have caught it
returned identical rows.

**Prove a new probe on a real backbone**, through its `examples/` script, not
only against the fake backbones in `tests/conftest.py`. Toy backbones cannot
show a training-dynamics problem, and one has already been found that way.

## Adding a probe

1. Subclass `BaseTask`, or `DenseTrainingTask` if it trains a dense head —
   `visbench/tasks/dense_base.py` already has the optimiser, schedule, batching
   and metric averaging. Read `DepthTask` and `GenericSegmentationTask` before
   writing a new one.
2. Register it with `@register_task("name")`.
3. Add a row to the `ProbeSpec` table in `visbench/cli/datasets.py`. That table
   is flat on purpose: probes share flag *groups*, not a class hierarchy.
4. Add an entry to `HEADLINE_METRICS` in `visbench/results/render.py`, and a
   direction for each new metric in `visbench/results/leaderboard.py`. Both
   raise on a missing entry rather than guessing.
5. Add a `TargetStyle` row to `TARGET_STYLES` in `visbench/viz/styles.py` and a
   `show_arguments` callable to its `ProbeSpec`, so `visbench show` can draw it.
   **This is not optional**: a test asserts `show_probes() == list_probes()`, so
   a probe without a style fails the suite rather than shipping undrawable.

   If your target is a **spatial map**, the row has to state its *validity
   convention* explicitly. There are already four (`0`, the zero vector,
   negative, `NaN`) plus *nothing at all* where 0 is a real reading, and none is
   visible in a tensor's shape. Picking the wrong one renders: the panel comes
   out looking like a target full of holes.

   If it is not a map — a class, a ranking, a preference — give it a kind in
   `COMPOSITE_KINDS` and a renderer, as `visbench/viz/gallery.py` does for the
   three probes that choose. Prefer stating the failure your probe can hide as a
   **footer figure** over trusting the reader's eye: `class_balance`,
   `vote_balance` and `error_coherence` are the three that exist, each of them a
   silent failure from this project's history turned into a number. None is a
   score and none is recorded.

   Captions are drawn with PIL's built-in bitmap font, which has no glyph for an
   em dash or an ellipsis — **keep them ASCII**, which a test enforces.
6. Add an `examples/` script and a row in
   [`docs/probes/`](docs/probes/overview.md).

**Do not claim another paper's protocol unless you implemented it.** The
`protocol` field exists so a reader knows what a number is comparable to. If you
borrowed only the optimiser schedule, say `visbench_*`, not `probe3d`.

**Ask whether your probe *ranks*, not whether its score is high.** A low
absolute number can be by design — the detection probe scores ~0.21 mAP because
a single-scale head has no pyramid. What is never acceptable is failing to
separate two backbones. A probe scoring 0.088 with a small-versus-base gap of
0.0035 looked like "a hard task" and was measuring nothing.

## Numbers, tables and the corpus

`results/corpus/visbench.jsonl` is a committed set of result records.
`LEADERBOARD.md` and the board on each page under `docs/probes/` are
**generated from it**:

```bash
scripts/render_tables.py            # regenerate
scripts/render_tables.py --check    # what the test runs
```

**Do not edit a generated table by hand** — the next regeneration overwrites it,
and a fast test fails if one drifts from the records.

Two rules the leaderboard enforces, worth knowing before you add a metric:

- **Metric directions are listed, never inferred from the name.** `mean` and
  `median` are angular error in degrees, where lower is better. A heuristic
  reading "mean" as a score ranks that board upside down, and the output reads
  as a finding rather than a bug.
- **A threshold must mean the same thing on every row.** Correspondence was
  scored in patch widths until v0.6.1 — and a patch is 14px on one backbone and
  32px on another, so each was asked a different question. It inverted the
  board: first place and last place swapped when the unit changed.

## Result records

Every run writes a JSON record under one **additive-only** schema. Never remove
or repurpose a field; add one and bump `SCHEMA_VERSION`, so old records stay
readable.

Two fields are easy to confuse and must not be merged. `metrics` is what
`evaluate()` returned about the **evaluation** split, and every leaderboard code
path reads it. `training` is how the **fit** went — `train_loss` for every
trained probe, plus `train_top1` for the classification family — and is `None`
for the three zero-shot probes, which fit nothing. It exists because a low score
has two opposite readings, an unconverged probe or a weak representation, and
only the training numbers separate them. **Never rank on `training`**: a probe
that fits its training data perfectly has said nothing yet about a backbone.

If you add a probe that trains something, override `training_summary()`. The
three existing implementations (`DenseTrainingTask`, `ClassificationTask`,
`DetectionTask`) cover every probe that ships, so a new dense or linear probe
inherits it for free; a probe with its own training loop does not, and a test
over `list_probes()` is what catches that.

## Style

Line length and formatting are `ruff`'s; run `ruff format`. Beyond that:

Comments explain **why**, especially where the obvious choice is wrong. Much of
this codebase's value is in notes like "clamping here would train against a wall
of fabricated values" — a future reader who does not know that will helpfully
add the clamp.

## Pull requests

- One reviewable change per PR.
- Say what you measured, not only what you wrote. If a number moved, show both.
- Update `CHANGELOG.md` under `[Unreleased]` in the same commit as the code.
- Report honestly. If a test fails or you skipped a step, say so — a PR that
  overstates its evidence costs more to review than one that admits a gap.

Releases and PyPI uploads are the maintainer's.

## Questions

Open an [issue](https://github.com/turhancan97/VisBench/issues). A question that
turns out to be a documentation gap is a useful bug report.

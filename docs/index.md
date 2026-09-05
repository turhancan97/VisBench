# VisBench

**Probe any vision backbone across high-, mid- and low-level computer vision
tasks.**

The name is literal: **Vis**(ion) **Bench**(mark). Both halves are a scope
commitment. *Vis* — the subject is a vision backbone's features, not a
multimodal or language model's behaviour. *Bench* — the output is a
**comparable** number rather than a score: every run writes a record of what
produced it, and explicit rules decide which records may be ranked together.

VisBench answers one question with as little ceremony as possible: *what does
this vision backbone actually encode?* Sixteen probes, three backbone families,
one `run()` call, and a result record that says exactly how every number was
produced.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.2em` Getting started
:link: getting-started/quickstart
:link-type: doc

`pip install visbench && visbench demo` runs a real probe on a real pretrained
backbone in thirty seconds, with no dataset and no configuration.
+++
Installation · Quickstart · The command line
:::

:::{grid-item-card} {octicon}`beaker;1.2em` The probes
:link: probes/overview
:link-type: doc

Sixteen probes across three levels. Each page states what it measures, the data
it expects, its `protocol` string, and its twelve-backbone board.
+++
High level · Mid level · Low level · Leaderboard
:::

:::{grid-item-card} {octicon}`book;1.2em` Guides
:link: guides/backbones
:link-type: doc

The parts you compose yourself — a custom backbone, your own dataset, a new
dense probe — and the rules each one silently depends on.
+++
Backbones · Datasets · Dense probes · Reading a board
:::

:::{grid-item-card} {octicon}`code;1.2em` API reference
:link: api/index
:link-type: doc

Every public class, function and attribute, generated from the docstrings, with
a link to the source beside each one.
+++
84 modules across 15 pages
:::

::::

## What makes it different

**One extraction method.** `extract_features()` returns the dense grid *and* a
pooled vector from one forward pass. ViTs and CNNs share the exact signature and
return shape despite completely different internals.

**Tasks choose pooling, backbones don't.** The "what representation does this
task need" decision lives in one place.

**The cache is not optional.** Disk-backed and keyed on the weights, so the
backbone forward pass runs at most once per image per backbone.

**Numbers come with the conditions that produced them.** Every run writes a
record under one additive-only schema, and the comparability rules decide which
records may be ranked together at all — because the failure this library guards
against is not a crash, it is a plausible wrong answer.

## Elsewhere in the repository

| | |
| --- | --- |
| Sixteen probes against twelve backbones, ranked | [LEADERBOARD.md](https://github.com/turhancan97/VisBench/blob/main/LEADERBOARD.md) |
| What each board means, and the readings it corrected | [CORPUS_FINDINGS.md](https://github.com/turhancan97/VisBench/blob/main/CORPUS_FINDINGS.md) |
| Setting up, the checks, and how to add a probe | [CONTRIBUTING.md](https://github.com/turhancan97/VisBench/blob/main/CONTRIBUTING.md) |
| What each step measured, rejected and decided | [ENGINEERING_LOG.md](https://github.com/turhancan97/VisBench/blob/main/ENGINEERING_LOG.md) |
| How it was built, and what might come next | {doc}`Roadmap <roadmap>` |

## Citing VisBench

If VisBench contributed to work you are publishing, please cite it — GitHub's
**Cite this repository** button generates APA and BibTeX from
[CITATION.cff](https://github.com/turhancan97/VisBench/blob/main/CITATION.cff),
which carries the concept DOI
[10.5281/zenodo.21822684](https://doi.org/10.5281/zenodo.21822684).

If you are reporting numbers, cite the *version you ran*. A VisBench record
carries the schema, the resolved pooling, the layers and the protocol behind
every number, which is what makes it reproducible — against that release.

```{toctree}
:hidden:
:caption: Getting started

getting-started/installation
getting-started/quickstart
getting-started/cli
```

```{toctree}
:hidden:
:caption: Guides

guides/backbones
guides/datasets
guides/dense-probes
guides/visualising
guides/sharing
guides/reading-a-board
```

```{toctree}
:hidden:
:caption: Probes

probes/overview
```

```{toctree}
:hidden:
:caption: Project

roadmap
```

```{toctree}
:hidden:
:caption: API

api/index
```

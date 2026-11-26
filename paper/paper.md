---
title: 'VisBench: comparable probing of frozen vision backbones across high-, mid- and low-level tasks'
tags:
  - Python
  - PyTorch
  - computer vision
  - representation learning
  - probing
  - benchmark
  - foundation models
authors:
  - name: Turhan Can Kargın
    orcid: 0000-0002-6751-4773
    affiliation: 1
affiliations:
  - name: Independent Researcher, Poland
    index: 1
date: 26 November 2026
bibliography: paper.bib
---

# Summary

Modern computer vision systems are built on *backbones*: large networks,
pretrained once, whose internal features are reused for many downstream tasks.
Choosing among them is hard, because a backbone is not good or bad in general:
the features that make it recognise object categories well are not necessarily
the ones that recover depth, surface orientation or edges. VisBench is a
Python library for asking, systematically, what a frozen backbone's features
encode. It fits small *probes* (usually a single linear map) on cached
features for twenty tasks spanning three levels of vision: high-level
semantics (object, scene and fine-grained categories, detection,
segmentation), mid-level geometry (depth, surface normals, correspondence,
camera pose) and low-level signal (edges, keypoints, corners, orientation),
following the taxonomy of @chen2025probing. Every run writes a structured
record stating exactly what produced its number, and explicit rules decide
which records may be ranked together. VisBench is installable from PyPI
(`pip install visbench`), exposes a three-call Python API and a command-line
interface, and is archived on Zenodo [@visbench].

![Each backbone's rank (1 = best of fourteen) on each probe; rows ordered by
mean rank (in brackets). Rankings change from probe to probe, so VisBench
reports one board per probe, never an aggregate; seed re-fits mark which
adjacent rows are tied.\label{fig:ranks}](figure.png)

# Statement of need

Comparisons of pretrained backbones are difficult to reuse. Evaluation code
is usually written for one paper, and the protocol details that decide a
number, such as how features are pooled, which layers are read, how images
are resized and cropped, which head is fitted and on what schedule, are
rarely recorded beside it. Two numbers that look comparable often are not,
and nothing in a results table says so. VisBench was built around that
failure. Its intended users are researchers who evaluate a new backbone,
study what pretraining objectives produce, or need a reproducible baseline
for a frozen representation.

The library makes the protocol part of the result. A run is one call,
`visbench.run(backbone, probe, dataset)`, where the backbone is either a
registered model (DINO, DINOv2, CLIP, SigLIP, MAE, SAM, supervised ViTs,
ResNets and ConvNeXts) or any PyTorch module wrapped in a few lines. The
record it writes carries the probe's `protocol` string, the pooling and
layers *as resolved* for that backbone, whether any block was fine-tuned,
a fingerprint of the exact dataset split, how the probe's own fit went, and
the hardware it ran on. A comparability key built from these fields
determines which records a leaderboard may place side by side, so a number
produced under a different protocol cannot silently join a ranking. Where a
published protocol exists it is reimplemented and cited rather than
approximated: depth, surface normals and correspondence follow
@elbanani2024probing.

# State of the field

Several resources compare vision backbones, but none combines frozen probing
across all three levels with a record of comparability. The evaluation code
of @elbanani2024probing measures 3D awareness through depth, normals and
correspondence for a fixed set of models; it is research code organised
around that paper's experiments, and parts of it are released under a
non-commercial licence, so VisBench reimplements its protocols from the
permissively licensed metric and loss code instead of vendoring them.
@chen2025probing probe mid-level vision capabilities of self-supervised
models, and VisBench adopts their task categorisation. The Visual Task
Adaptation Benchmark [@zhai2019vtab] measures transfer through fine-tuning
on a suite of mostly image-level tasks, and Battle of the Backbones
[@goldblum2023battle] compares pretrained models across many tasks and
training regimes as a single large study. Model
libraries such as timm [@wightman2019timm] and OpenCLIP [@ilharco2021openclip]
provide the backbones but not their evaluation. Contributing to any of these
would not have produced what was needed: a packaged library with one
interface over heterogeneous backbones and twenty probes, a feature cache
shared across tasks, and a result schema whose comparability rules are
enforced by the code that renders rankings.

# Software design

A backbone implements one method, `extract_features`, that returns both the
dense spatial feature map and a pooled vector from a single forward pass,
with identical shapes for convolutional networks and vision transformers;
several layers can be read in the same pass. The probe, not the backbone,
chooses the pooling, the layers and whether a class token is broadcast to the
dense grid, which keeps backbones interchangeable. Extracted features are
written to a disk cache keyed by backbone configuration, layer, pooling mode
and image content, so each image passes through a backbone at most once
across all probes; dense features are streamed from the cache as a PyTorch
dataset rather than held in memory.

Probes share a small interface (`fit`, `evaluate`, `predict`) and a common
training base for dense tasks, so a new dense probe supplies only its output
activation, loss and metric. Heads are pluggable: a linear head is the
default, with DPT [@ranftl2021vision], detection, instance-segmentation and
pose heads where a task requires them. Results are written under an
additive-only schema, now at version 9, so records written by earlier
releases remain readable. Two further mechanisms guard the numbers a user
reads. A recoverability check scores a candidate dense target after pooling
it to the feature grid, before any backbone is run, which rejects targets
that no per-patch probe could recover. Seed re-fits test whether two adjacent
rows of a ranking are ordered or tied by their paired difference across
seeds, rather than by the size of their gap.

The command-line interface exposes every probe (`visbench run`), renders
image, target and prediction panels (`visbench show`), and runs a complete
probe on generated images with no download (`visbench demo`). A trained head
can be published to the Hugging Face Hub together with the identity of the
backbone it was fitted on, and is loaded with `weights_only` deserialisation
and refused against any other backbone. The package is tested by more than 2,500
fast tests in continuous integration, plus a slow suite that runs on real
pretrained weights, and is documented with guides, one page per probe and an
API reference.

# Research impact statement

VisBench is in research use. It produced the depth-estimation and
image-classification baselines against which SpaRRTa [@kargin2026sparrta], a
benchmark for spatial-relation recognition in visual foundation models
published in the *International Journal of Computer Vision*, compares
spatial-relation recognition (§4.3), and that article cites VisBench. Beyond that use, the library provides reproducible materials
of its own: a committed corpus covering 280 board cells (twenty probes across
fourteen backbones), each regenerable from the flags in its own record,
five-seed re-fits of every trained board, and 32 trained probe heads published
on the Hugging Face Hub.

# AI usage disclosure

<!-- TODO(author): this section is an affirmation in your name. Check every
statement, especially the last paragraph, before submitting. -->

VisBench was developed with substantial assistance from generative AI. Code,
tests, documentation, analysis scripts and cluster job scripts were written
with Claude Code (Anthropic), using the models Claude Opus 4.8, Claude Opus 5,
Claude Opus 5.5 and Claude Sonnet 5, as recorded in the `Co-Authored-By`
trailers of 271 of the repository's first 440 commits.

This manuscript was drafted with Claude Code (Claude Opus 5.5) from the
author's outline and facts; the author revised the text and verified every
statement in it.

The author framed the problem and its scope, chose the task taxonomy, the
backbones and the published protocols to follow, and set the design rules
that the software enforces, including the record schema, the comparability
rules and the requirement that a re-run replaces a published result only
where it reproduces it. Development proceeded as a sequence of scoped steps,
each delivered as a pull request that the author reviewed and merged, and
the experiments behind every published number were run on a computing
cluster under the author's direction and checked by the author. The author reviewed, modified and validated
all AI-generated content, and takes responsibility for it.

# Acknowledgements

This work received no funding.

# References

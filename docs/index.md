# VisBench

**Probe any vision backbone across high-, mid- and low-level computer vision
tasks.**

VisBench answers one question with as little ceremony as possible: *what does
this vision backbone actually encode?* Twelve probes, three backbone families,
one `run()` call, and a result record that says exactly how every number was
produced.

## Try it in thirty seconds

No dataset, no configuration, no large download:

```bash
pip install visbench
visbench demo
```

```text
drawing 20 images per class for 4 shapes...
loading resnet18 (torchvision, ~45 MB on first run)...
running the classification probe...

  top1         0.8125

  chance is 0.25 — the shapes differ in outline only.
```

That is a real probe on a real pretrained backbone, through the same code path
every other run uses. The images are generated: four shapes with colour, size,
position and rotation randomised, so only geometry identifies a class.

The number is deliberately not 1.0. Turn the difficulty up and watch it fall
toward chance — a probe whose score does not move when you destroy the signal is
not measuring the signal.

## Where to start

| | |
| --- | --- |
| Every probe, the data it expects, and what it has measured | [The probes](tasks.md) |
| How it was built, and what might come next | [Roadmap](roadmap.md) |
| Twelve probes against six backbones, ranked | [LEADERBOARD.md](https://github.com/turhancan97/VisBench/blob/main/LEADERBOARD.md) |
| Setting up, the checks, and how to add a probe | [CONTRIBUTING.md](https://github.com/turhancan97/VisBench/blob/main/CONTRIBUTING.md) |

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

```{toctree}
:hidden:
:caption: The probes

tasks
```

```{toctree}
:hidden:
:caption: Project

roadmap
```

# Sharing probes through the Hugging Face Hub

A trained probe is a small module — often a single linear layer — fitted on
frozen features. It is cheap to train and easy to share, and that is exactly
what makes it dangerous to share carelessly: **a probe head is only meaningful
against the exact features it was fitted on**, and almost every way of getting
that wrong is shape-compatible.

Measured on real DINOv2-S weights over Imagenette (200 images per class): a
linear head fitted on CLS tokens, then fed *mean-pooled* tokens from the same
backbone, scores **0.9620 against 0.9895**. It does not crash. It does not
produce garbage. It produces a number nobody would question.

So VisBench does not distribute weights. It distributes weights *plus the
identity they are valid against*, and refuses a load that does not match.

## Use one without training

Trained heads for the DINOv2 backbones are published as a collection:

**[huggingface.co/turhancan97 → collections](https://huggingface.co/turhancan97/collections)**

```python
import visbench
from visbench.hub import load_probe_from_hub

backbone = visbench.get_backbone("dinov2_vits14")
probe = load_probe_from_hub("turhancan97/visbench-depth-dinov2_vits14", backbone=backbone)
```

One repository per (probe, backbone) pair — a head fitted on one backbone is
refused against any other, so pairing them in one repository would publish a
file that is wrong for half its contents. The three zero-shot probes are not
published: retrieval, correspondence and mid-level similarity train nothing,
and the backbone alone reproduces their numbers.

## Install

```bash
pip install visbench[hub]
```

Saving and loading a probe to a **local file needs no extra** — `huggingface_hub`
is imported inside the two functions that talk to the network, so a core install
can still write, read and inspect artifacts.

## What you can do

| Action | Call | Needs the extra |
| --- | --- | --- |
| Write a fitted probe to a file | `save_probe(task, path, backbone=...)` | no |
| Load one back, identity checked | `load_probe(path, backbone=...)` | no |
| Inspect what would be written | `probe_metadata(task, backbone)` | no |
| Render the model card | `probe_card(task, backbone, repo_id)` | no |
| Upload probe + card | `push_probe(task, repo_id, backbone=...)` | yes |
| Download and load | `load_probe_from_hub(repo_id, backbone=...)` | yes |

All six are exported from `visbench.hub`.

## The shortest round trip

```python
import visbench
from visbench.hub import push_probe, load_probe_from_hub

backbone = visbench.get_backbone("dinov2_vits14")
probe = visbench.get_probe("classification")
probe.fit(train_features, train_labels)

# private by default; pass private=False deliberately
push_probe(probe, "you/imagenette-probe", backbone=backbone, metrics={"top1": 0.982})
```

and, on someone else's machine:

```python
backbone = visbench.get_backbone("dinov2_vits14")
probe = load_probe_from_hub("you/imagenette-probe", backbone=backbone, revision="a1b2c3d")
scores = probe.evaluate(features, labels)
```

[`examples/push_probe.py`](https://github.com/turhancan97/VisBench/blob/main/examples/push_probe.py)
runs all of this on real weights, and **does not upload unless you pass
`--push`** — the default prints the card and the identity block so you can see
what would go out. Its `--pull` mode does the other half.
[`examples/save_probe.py`](https://github.com/turhancan97/VisBench/blob/main/examples/save_probe.py)
covers the local file and demonstrates the mismatch above.

## Publishing a set of them

The command line pushes what it just trained:

```bash
visbench run depth --data ... --push-to you/visbench-depth-dinov2_vits14 --public
```

That flag exists so the artifact and the record come from **one** command. A
head is only meaningful against the features it was fitted on, and those are
decided by the run's flags — a separate publishing step is free to drift from
them, and a head trained under drifted flags uploads, loads and scores without
complaint.

For a whole board,
[`scripts/build_corpus.sh`](https://github.com/turhancan97/VisBench/blob/main/scripts/build_corpus.sh)
already holds every probe's flags in one place, so it takes an owner:

```bash
PUSH_TO=you PUSH_PUBLIC=1 scripts/build_corpus.sh          # every probe
PUSH_TO=you DRY_RUN=1 scripts/build_corpus.sh              # print, run nothing
```

and
[`scripts/publish_collection.py`](https://github.com/turhancan97/VisBench/blob/main/scripts/publish_collection.py)
groups the results into one collection (`--create` to do it; dry run otherwise).

**Push public for anything you intend to share.** The default is private, which
is right for a push you are still checking and wrong for a collection: private
repositories render the collection page empty to everyone but you, which reads
as broken rather than as a permissions choice.

## What is checked on load

Four fields must agree between the artifact and the backbone you load it
against. Each one is its own way to be silently wrong:

| Field | The failure it prevents |
| --- | --- |
| `backbone_key` | The weights and resolution. A fine-tuned DINOv2-S and its parent share a name, width, pooling rule, feature mode and depth — this is the *only* thing that differs |
| `pooling` | Resolved, so `cls` and `mean` cannot be confused. The 0.9620-against-0.9820 case, which raises nothing on its own |
| `feature_mode` | `dense_cls_broadcast` doubles the channel width, so this usually raises anyway — "usually" is doing a lot of work when the head is a 1x1 convolution |
| `layers` | A head fitted on block 11 and fed block 5: right shape, wrong depth |

A mismatch raises `IncompatibleProbe`. Provenance — metrics, timestamps, notes —
is recorded but never gates a load.

`strict=False` warns and loads instead. Deliberately probing how far a head
transfers is a legitimate experiment; doing it *silently* is not, and a number
produced that way is comparable with nothing, because `run()` records the
backbone actually used and nothing would say the head came from elsewhere.

## Safety, and the two defaults worth knowing

**Downloaded probes are read with `weights_only=True`.** An unrestricted
`torch.load` on a file from a stranger's repository is arbitrary code execution.
`load_probe_from_hub` is `load_probe` with a download in front of it — there is
deliberately no second loading path, because a separate one is how one of them
ends up without this guard, and a downloaded probe is precisely the one that
needs it.

**A push is private by default.** It is not reversible the way a local write is:
once a repository is public it may already have been fetched, and deleting it
does not unpublish what was taken. Public is a decision, not a default someone
discovers afterwards.

**Pin a revision for anything you quote.** A Hub repository is mutable, so
`main` today and `main` next month are not promised to be the same weights.

## What is refused, and why

- **Zero-shot probes.** Retrieval, correspondence and mid-level similarity train
  nothing, so an artifact would hold no weights — the backbone alone reproduces
  them.
- **Unfitted probes.** An artifact holding an untrained head cannot be told
  apart from one holding a head that learned nothing.
- **Artifacts from a newer `ARTIFACT_VERSION`**, rather than guessing at a
  layout that may have moved. That number is separate from the result schema's
  `SCHEMA_VERSION`; the two version unrelated things and move for unrelated
  reasons.

The refusals run **before** anything is created, so a rejected push never leaves
an empty repository behind.

## The card

`push_probe` generates a model card from `probe_metadata` — the same source the
artifact itself uses, so the page and the file cannot disagree. It carries the
backbone, the backbone key, the resolved and requested pooling, the feature
mode, the layers, any reported metrics, and the hyperparameters the probe was
fitted with.

A bare `.pt` on a model page does not tell a visitor the one thing they have to
know, which is that the weights belong to exactly one backbone.

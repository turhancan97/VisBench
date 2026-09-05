# The command line

Installing the package puts a `visbench` command on your path. It is a thin
wrapper over `visbench.run()` — same cache, same result records, same numbers.

```bash
visbench demo                       # a real probe on generated data, no setup
visbench list                       # backbones, probes and heads that exist
visbench run retrieval --data /path/to/imagenette2 --split val
visbench run classification --dataset torchvision:CIFAR10 --split test
visbench show depth --data /path/to/nyuv2 --out panels.png
visbench cache stats
```

Each probe is its own subcommand, because they do not take the same data.
`visbench run depth --help` shows the folder layout depth expects and only
depth's flags:

```bash
# mid-level geometry, zero-shot, no annotation needed
visbench run correspondence --data /path/to/images --split val --limit 200

# a dense probe: <data>/<split>/{images,masks}, paired by filename stem
visbench run generic_segmentation --data /path/to/data --epochs 40 --lr 5e-3

# an official split list instead of split directories — how real benchmarks
# express one. Passing --stems makes --data the dataset root itself.
visbench run semantic_segmentation --data VOCdevkit/VOC2012 \
    --image-dir JPEGImages --target-dir SegmentationClass \
    --stems ImageSets/Segmentation/val.txt \
    --train-stems ImageSets/Segmentation/train.txt \
    --num-classes 21 --backbone dinov2_vits14

# detection reads the same way, from ImageSets/Main
visbench run detection --data VOCdevkit/VOC2012 \
    --stems ImageSets/Main/val.txt \
    --train-stems ImageSets/Main/train.txt \
    --backbone dinov2_vits14
```

That last one reports `miou 0.733` on VOC val, against the 0.732 the Python API
records for the same backbone — which is the check that matters for a wrapper.

Following [Chen, Marks & Cheng (arXiv:2411.17474)](https://arxiv.org/abs/2411.17474):

| Level | Tasks | Status |
|---|---|---|
| **High-level** — semantic / category | classification, retrieval | v0.1 |
| | semantic (multi-class) segmentation | v0.2 |
| | detection (anchor-free, single-scale) | v0.3 |
| **Mid-level** — geometry & generic structure | geometric correspondence | v0.1 |
| | depth, surface normals, generic (binary) segmentation, mid-level similarity | v0.2 |
| | occlusion-edge detection | v0.5 |
| **Low-level** — signal properties | edge detection (dense magnitude regression) | v0.4 |
| | 2D keypoint detection | v0.5 |
| | optical flow, texture, IQA | [scope only](https://github.com/turhancan97/VisBench/blob/main/visbench/tasks/low_level/README.md) |

Mid-level is where VisBench aims to be strongest relative to existing tooling.
Note that **mid-level image similarity and high-level retrieval are separate
tasks** — one judges perceptual/geometric resemblance, the other category
membership.

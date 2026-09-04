# `generic_segmentation`

**Binary figure-ground — is this pixel part of an object at all?**

The same 1449 VOC images as {doc}`semantic_segmentation </probes/high-level/semantic_segmentation>`, at the same resolution with the same linear head
and the same schedule, differing **only** in whether the target has 2 classes
or 21. That makes the pair a control, and they behave oppositely: this board
ranks by feature-grid area at **+0.958**, the semantic one at +0.545 — so that
difference is a property of the target, not of the data or the protocol.

```{figure} /_static/gallery/generic_segmentation.png
:alt: generic_segmentation — image, target and prediction

What `visbench show generic_segmentation` draws. {doc}`How to read it </show>`.
```

## Things that will bite

  elsewhere. On VOC they sit five points apart. Quote `miou` against published
  numbers, and say which one you mean.
- **Label maps are read without mode conversion, and getting this wrong is
  silent.** VOC's PNGs are palette images whose raw bytes are the class indices;
  resolving the palette turns classes `[0, 1, 15]` into `[0, 38, 147]`, which
  trains and scores perfectly happily against labels that mean nothing. Use
  `load_label_map`, not `load_mask`, for anything multi-class — including

## Its board

The twelve-backbone board is in [`LEADERBOARD.md`](https://github.com/turhancan97/VisBench/blob/main/LEADERBOARD.md).

## Run it

[`examples/segment.py`](https://github.com/turhancan97/VisBench/blob/main/examples/segment.py) is the whole path, end to end, on a real backbone.

```bash
python examples/segment.py --data /path/to/dataset
```

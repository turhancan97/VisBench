# `depth`

**Monocular metric depth, via probe3d's 256-bin expectation.**

Bins rather than one number: regressing a scalar per pixel pushes a linear head
towards the dataset's mean depth almost everywhere, and predicting a
distribution lets a *linear* map express a multi-modal belief. That is most of
why probe3d's linear probe is a fair baseline rather than a straw man.

**This board is not ranking by metric accuracy.** A readout that discards scale
and shift entirely reproduces its ordering at Spearman **+1.000**, so what it
ranks is ordering plus feature resolution — reported in metres. Not a defect:
it reproduces probe3d's protocol, which is the only reason its numbers compare
to anything. See `results/controls/relative_depth.jsonl`.

```{figure} /_static/gallery/depth.png
:alt: depth — image, target and prediction

What `visbench show depth` draws. {doc}`How to read it </show>`.
```

## Data layout

Images and per-pixel targets paired by filename stem under `train/` and `val/`.

```bash
python examples/depth.py --data /path/to/dataset --target-scale 1000
```

**`--target-scale` is load-bearing.** Depth datasets ship millimetres in a
16-bit container, so 1000 is right for a PNG distribution and **wrong** for
`.npy` files already in metres — pass 1.0 there. `depth_metrics` reports RMSE
in whatever unit it is handed, so the mistake produces a superb-looking number
that means nothing.

## Its board

The twelve-backbone board is in [`LEADERBOARD.md`](https://github.com/turhancan97/VisBench/blob/main/LEADERBOARD.md); this page does not duplicate it.

## Run it

[`examples/depth.py`](https://github.com/turhancan97/VisBench/blob/main/examples/depth.py) is the whole path, end to end, on a real backbone.

```bash
python examples/depth.py --data /path/to/dataset
```

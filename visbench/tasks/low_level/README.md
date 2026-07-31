# Low-level tasks

Low-level = signal-level properties recoverable largely without object or scene
understanding.

This folder was a documented placeholder from v0.1 until **step 6d-1**, which
added its first entry.

## Implemented

| Task | Module | Dataset | Protocol |
|---|---|---|---|
| Edge detection | `edge.py` (`EdgeTask`) | Taskonomy `edge_texture` | `visbench_edge_regression` |

Dense edge-magnitude regression: one channel, identity activation, L1 loss,
scored by per-image Pearson correlation (plus RMSE and MAE). Quote
`edge_correlation` — it is invariant to scale and offset, so a probe predicting
the split's mean everywhere scores 0 on it while still achieving a small RMSE.

**It is not BSDS500's protocol**, and a record saying so is the point of the
`protocol` field. BSDS is the canonical edge benchmark, but ODS/OIS/AP matches
predicted edge pixels to several annotators' by bipartite correspondence after
non-maximum suppression, swept over thresholds. Borrowing a protocol is only
worth it if borrowed exactly — see `NOTICE`, and the depth probe's 256-bin
expectation, which a from-memory reconstruction would have turned into scalar
regression. Adding BSDS properly is a step of its own.

## Still candidates

| Task | Notes |
|---|---|
| Optical flow | Needs image pairs and a flow head. `PairViewDataset` already expresses the pairing (see correspondence), so the flow head is the real cost. No flow dataset is assumed present. |
| Texture / reflectance | Intrinsic-image decomposition; ground truth is scarce outside synthetic data. Taskonomy ships `principal_curvature` and `reshading`, but both are derived from its 3D reconstruction and need `mask_valid/` wired up first — see `TaskonomyDataset`, which refuses them by name rather than scoring against reprojection holes. |
| Image quality assessment | No-reference IQA against human MOS ratings. Closest in shape to mid-level similarity, which is zero-shot; IQA is not. |
| Edge detection on BSDS500 | The correspondence metric above, as a second protocol beside the Taskonomy one rather than a replacement. |

Contributions welcome — see the head interface in `visbench/heads/`, which is a
deliberate extension point, and `visbench/tasks/dense_base.py`, which supplies
everything a trained dense probe needs bar four methods.

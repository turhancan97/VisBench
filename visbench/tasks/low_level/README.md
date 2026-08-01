# Low-level tasks

Low-level = signal-level properties recoverable largely without object or scene
understanding.

This folder was a documented placeholder from v0.1 until **step 6d-1**, which
added its first entry. **Step 6d-2** added the second, and moved the
reconstruction-derived Taskonomy domains from "refused" to "supported" by wiring
up `mask_valid/`.

## Implemented

| Task | Module | Dataset | Protocol |
|---|---|---|---|
| Edge detection | `edge.py` (`EdgeTask`) | Taskonomy `edge_texture` | `visbench_edge_regression` |
| 2D keypoint detection | `keypoints.py` (`Keypoint2DTask`) | Taskonomy `keypoints2d` | `visbench_keypoint2d_regression` |

Both are dense magnitude regression and share
`visbench/tasks/magnitude_base.py` (`DenseMagnitudeTask`): one channel, identity
activation, L1 loss, scored by per-image Pearson correlation (plus RMSE and MAE).
Quote the correlation — it is invariant to scale and offset, so a probe
predicting the split's mean everywhere scores 0 on it while still achieving a
small RMSE.

They are two probes rather than one because an edge response fires along
intensity *contours* and a keypoint response at *corners and blobs*, and a
backbone can be good at one and weak at the other. Distinct metric keys and
distinct `protocol` strings are what stop the two numbers being pooled.

**Neither is BSDS500's protocol**, and a record saying so is the point of the
`protocol` field. BSDS is the canonical edge benchmark, but ODS/OIS/AP matches
predicted edge pixels to several annotators' by bipartite correspondence after
non-maximum suppression, swept over thresholds. Borrowing a protocol is only
worth it if borrowed exactly — see `NOTICE`, and the depth probe's 256-bin
expectation, which a from-memory reconstruction would have turned into scalar
regression. Adding BSDS properly is a step of its own.

### The mid-level twin

`visbench/tasks/mid_level/occlusion_edge.py` (`OcclusionEdgeTask`) shares every
line of this implementation and reads Taskonomy's `edge_occlusion` instead. It
is **mid-level** because an occlusion edge is a depth discontinuity and
recovering one needs scene geometry, where a texture edge is an intensity
discontinuity and does not. Running both on the same frames is about as clean a
comparison of the two tiers as VisBench offers.

It is also the cautionary case for anyone adding a third magnitude probe: its
target has 46% of its total mass in the strongest 1% of pixels, against ~0.10 for
the two above, and at that tail the L1 loss and the Pearson metric pull in
opposite directions. The linear-target probe scored 0.088 and could not tell
DINOv2-S from DINOv2-B; under a `log1p` target it scores 0.29/0.32 and ranks
them. **Check the tail before assuming this protocol transfers.**

## Still candidates

| Task | Notes |
|---|---|
| Optical flow | Needs image pairs and a flow head. `PairViewDataset` already expresses the pairing (see correspondence), so the flow head is the real cost. No flow dataset is assumed present. |
| Texture / reflectance | Intrinsic-image decomposition; ground truth is scarce outside synthetic data. **Taskonomy does not ship a reflectance domain**, so this is not unblocked by 6d-2. `principal_curvature` and `reshading` are present and are still refused, but no longer for want of a mask — see below. |
| Image quality assessment | No-reference IQA against human MOS ratings. Closest in shape to mid-level similarity, which is zero-shot; IQA is not. |
| Edge detection on BSDS500 | The correspondence metric above, as a second protocol beside the Taskonomy one rather than a replacement. |
| Corner / blob detection (Harris, DoG) | The classical-target counterpart to `keypoints2d`'s learned response maps. **Needs no dataset** — see below. |
| Local orientation / gradient fields | Structure-tensor or HOG-style. Vector-valued, so closer in shape to surface normals than to the magnitude probes. Needs no dataset. |
| Superpixel / texture segmentation | Grouping by local photometric similarity alone, with no depth or figure-ground reasoning. Needs no dataset. |
| Color constancy / illuminant estimation | A per-image scalar/vector target rather than a dense one, and it needs measured illuminant ground truth. |
| Vanishing point / line detection | Published as a Taskonomy domain, but **not in the copy on this machine** — that download carries eight domains and this is not one. |

### The three that need no dataset

`edge_texture` is a target Taskonomy *computed from the RGB frame*. Corner and
blob responses, structure-tensor orientation and photometric superpixels are all
equally derivable, deterministically, from any image folder already present —
so each is a target generator plus a `DenseMagnitudeTask` subclass, both of
which already exist, rather than a data acquisition step. That makes them the
cheapest entries here by a wide margin.

Two cautions specific to a derived target:

- **Check the tail first.** A corner response is spikier than an edge response,
  and the occlusion-edge case above is what happens when L1 and Pearson pull
  apart: the probe scores low *and stops ranking backbones*.
- **`protocol` must name the generator, not the family.** "Harris corners" is
  not a definition — the k parameter, window, smoothing and non-maximum
  suppression all move the target, so two runs claiming `"harris"` need not be
  comparable. That is the same failure the field exists to prevent.

### Why `principal_curvature` and `reshading` are still refused

Not the mask — that is wired up as of 6d-2 and these two would get it. Each is
blocked on a **task** decision instead, and `TaskonomyDataset` refuses them by
name with the reason rather than reporting an unknown domain:

- `principal_curvature` packs two principal curvatures plus an unused channel
  into RGB, and which is which, and in what units, is not established here.
  Reading it as a magnitude map would train and score against a quantity this
  library cannot name.
- `reshading` is a re-rendered shading image, so its target is an RGB frame
  rather than a per-pixel measurement, and no probe here consumes one.

Contributions welcome — see the head interface in `visbench/heads/`, which is a
deliberate extension point, `visbench/tasks/dense_base.py`, which supplies
everything a trained dense probe needs bar four methods, and
`visbench/tasks/magnitude_base.py`, which supplies the rest if the target is a
magnitude map.

# Low-level tasks

Low-level = signal-level properties recoverable largely without object or scene
understanding.

This folder was a documented placeholder from v0.1 until **step 6d-1**, which
added its first entry. **Step 6d-2** added the second, and moved the
reconstruction-derived Taskonomy domains from "refused" to "supported" by wiring
up `mask_valid/`. The third, corner detection, is the first probe in VisBench
whose target is **computed from the image** rather than read from disk, so it
runs on any image folder with no download. Gradient orientation is the fourth —
also computed from the image, but a *direction* rather than a magnitude.

## Implemented

| Task | Module | Dataset | Protocol |
|---|---|---|---|
| Edge detection | `edge.py` (`EdgeTask`) | Taskonomy `edge_texture` | `visbench_edge_regression` |
| 2D keypoint detection | `keypoints.py` (`Keypoint2DTask`) | Taskonomy `keypoints2d` | `visbench_keypoint2d_regression` |
| Corner detection | `corner.py` (`CornerTask`) | **any image folder** — the target is computed | `visbench_shi_tomasi_regression` |
| Gradient orientation | `orientation.py` (`OrientationTask`) | **any image folder** — the target is computed | `visbench_structure_tensor_orientation_regression` |

The first three are dense magnitude regression and share
`visbench/tasks/magnitude_base.py` (`DenseMagnitudeTask`): one channel, identity
activation, L1 loss, scored by per-image Pearson correlation (plus RMSE and MAE).
Quote the correlation — it is invariant to scale and offset, so a probe
predicting the split's mean everywhere scores 0 on it while still achieving a
small RMSE.

They are three probes rather than one because an edge response fires along
intensity *contours* and a keypoint response at *corners and blobs*, and a
backbone can be good at one and weak at the other. Distinct metric keys and
distinct `protocol` strings are what stop the numbers being pooled.

**Gradient orientation is the odd one out.** Its target is a *direction*, not a
magnitude, so it cannot use `DenseMagnitudeTask`: it predicts a 2-channel unit
vector `(cos 2θ, sin 2θ)` (the angle is mod π, so the double angle handles the
wrap), the loss is a coherence-weighted angular error, and the metric
`orientation_error` is degrees of angular error, not a correlation. See
`orientation.py`. It measures phase — per-image `|r|` with the corner and edge
targets is under 0.09, where those two sit at 0.53 — which is exactly the gap
the DoG-blob candidate could not fill (it landed at 0.51 with `corner`).

## Corner detection, and the target that is computed rather than read

`corner.py` is the first probe here whose target needs **no dataset**. It is
Shi-Tomasi cornerness — the smaller eigenvalue of the Gaussian-windowed
structure tensor — computed from the RGB frame at read time by
`visbench/data/derived.py`. Any folder of photographs runs it.

Three things were settled by measurement while building it, over 60 Taskonomy
val frames at 224px.

**The operator is Shi-Tomasi, not Harris, and the tail is why.** Share of target
mass in the strongest 1% of pixels, against the ~0.10 of the two probes above
and the 0.46 that stopped `edge_occlusion` ranking:

| response | tail@1% |
|---|---|
| Harris `R`, clipped at 0 | 0.52 |
| Harris `\|R\|` | 0.33 |
| Shi-Tomasi λ_min | **0.27** |

All three are too concentrated raw. λ_min starts closest, is non-negative by
construction, and has no `k` — one fewer free parameter making "Harris corners"
a family rather than a definition. `log1p(1e4·λ_min)` lands the tail at **0.089**
and the frame mean at **0.593**, satisfying 6d-2's tail criterion and 6d-1's
"an L1 target must be of order 1" at the same setting.

**The target overlaps with the edge target, measurably, and this is stated
rather than hidden.** Per-image correlation:

| target pair | mean r |
|---|---|
| `edge_texture` vs `keypoints2d` — two probes shipped separately | **0.147** |
| corner vs `keypoints2d` | 0.271 |
| **corner vs `edge_texture`** | **0.519** |

So this target is *more* redundant with edges than the two Taskonomy probes are
with each other. The overlap is intrinsic, not an artifact of the compression:
it holds at 0.46–0.54 across eight transforms including near-linear ones,
because a corner is a pixel whose gradient is large in two directions and an
edge map is gradient magnitude. **A corner score and an edge score are not
independent evidence about a backbone.**

**Correlated targets can still rank differently, and that is what earns the
probe its place.** 600 train / 600 val, linear head, ten epochs, on the *same
frames* the edge probe uses so only the target differs:

| backbone | `corner_correlation` | rank | `edge_correlation` | rank |
|---|---|---|---|---|
| dinov2_vitb14 | **0.6526** | 1 | 0.4481 | 3 |
| dinov2_vits14 | 0.6512 | 2 | 0.4558 | 2 |
| clip_vitb16 | 0.6227 | 3 | **0.4565** | 1 |
| clip_vitb32 | 0.5367 | 4 | 0.3834 | 4 |
| resnet18 | 0.5014 | 5 | 0.3430 | 6 |
| resnet50 | 0.4923 | 6 | 0.3549 | 5 |

Spread **0.1603**, wider than the edge probe's 0.1136, and the ordering is not
the same one: **CLIP-B/16 is first on edges and third on corners**, and the two
ResNets swap. That is the "a backbone can be good at one and weak at the other"
claim demonstrated between two probes whose targets correlate at 0.52.

Those six numbers are in the committed corpus as of **8b**, and the frames they
ran on are pinned there: `scripts/stage_corner_frames.py` symlinks the first 600
rows of each Taskonomy `tiny` split list into the flat `<split>/images/` layout
a derived-target dataset reads. **That is the same 600 the edge rows above use,
verified set-equal rather than assumed** — which is the only thing that makes
the side-by-side ranking in this table a comparison rather than a coincidence.
A derived target has no data of its own, so a board for one has to name a set;
running the probe still needs no download.

**Do not read the DINOv2-S/B gap of 0.0014 as a failure to rank.** It nearly was
read that way here. The occlusion-edge case that "scored 0.088 and could not
separate two backbones" was flat *across all six*, which is what made it
meaningless; the edge probe's own top two differ by 0.0007. Ask about the spread
over the full set, not one pair.

**None of the three is BSDS500's protocol**, and a record saying so is the point of the
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
| Superpixel / texture segmentation | Grouping by local photometric similarity alone, with no depth or figure-ground reasoning. Needs no dataset. |
| Color constancy / illuminant estimation | A per-image scalar/vector target rather than a dense one, and it needs measured illuminant ground truth. |
| Vanishing point / line detection | Published as a Taskonomy domain, but **not in the copy on this machine** — that download carries eight domains and this is not one. |

### Derived targets — what is done and what remains

`visbench/data/derived.py` is the machinery `corner` left behind. A magnitude
derived target is a `ShiTomasiResponse`-shaped class plus a `DenseMagnitudeTask`
subclass; a vector one — `OrientationResponse` — needs its own small task base,
which `orientation.py` now provides. **Photometric superpixels** is the one that
remains.

Four cautions, the first three paid for by `corner` and `orientation`:

- **Check the tail first.** Every raw corner response was spikier than
  `edge_occlusion`'s 0.46, the case that stopped ranking. A compression is not
  optional, and which one is a measurement. (An *angle* has no tail — the
  orientation target needed no compression, which the pre-measurement confirmed
  before the task was written.)
- **Check the overlap with what already ships, before building.** It costs one
  afternoon of correlations rather than a probe run per backbone. **DoG-blob
  detection was rejected on exactly this**: its target correlated 0.50 with the
  edge target and **0.51 with `corner`**, i.e. as redundant with an existing
  probe as `corner` is with `edge`. `orientation` passed the same check with
  `|r|` under 0.09 against both, because it measures phase.
- **A correlated target can still rank differently** — that is what earned
  `corner` its place despite 0.52 with edge (spread 0.1603 vs 0.1136, CLIP-B/16
  first on edges and third on corners). It is a reason to build and measure, not
  a reason to skip the overlap check.
- **`protocol` must name the generator, not the family.** "Harris corners" is
  not a definition — the k parameter, window, smoothing and non-maximum
  suppression all move the target. Naming the operator is half the fix; the
  other half is that every setting travels in `dataset_params`, so two sigmas
  split into two comparability groups without anyone noticing.

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

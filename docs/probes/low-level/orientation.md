# `orientation`

**Local gradient orientation — the first derived target that is a *direction*.**

`(cos 2θ, sin 2θ)` with its length set to the coherence, so it could not reuse
`DenseMagnitudeTask`: it needs a 2-channel L2-normalising activation and a
coherence-weighted angular loss. It measures **phase**, and no other probe here
does.

```{figure} /_static/gallery/orientation.png
:alt: orientation — image, target and prediction

What `visbench show orientation` draws. {doc}`How to read it </show>`.
```

[`examples/orientation.py`](https://github.com/turhancan97/VisBench/blob/main/examples/orientation.py).
The fourth low-level probe and the second whose target is computed from the
frame — but the first whose target is a *direction*: the local orientation
structure runs, read as `2θ = atan2(2·Ixy, Ixx − Iyy)` from the same
Gaussian-windowed structure tensor whose smaller eigenvalue is the corner
response.

```bash
visbench run orientation --data /path/to/any/images --limit 600
```

The angle is defined **modulo π** — an edge and its reverse run the same way —
so the target is the unit vector `(cos 2θ, sin 2θ)`, which is single-valued
under that wrap, with its length set to the **coherence**
`(λ_max − λ_min) / (λ_max + λ_min)`. Loss and metric both weight by that length,
so a flat isotropic patch contributes ~0 rather than being masked out by a
threshold nobody chose. Only 1.4% of Taskonomy tiny val pixels fall below
coherence 0.1. The metric, `orientation_error`, is the coherence-weighted mean
angular error in degrees, halved into `[0, 90]` (45 is chance).

**It measures phase, which no other probe here does.** Per-image `|r|` with the
`edge_texture` target is 0.07 and with `corner` 0.08, where `corner` and `edge`
themselves sit at 0.53 — so an orientation score is close to independent
evidence about a backbone, unlike a corner score beside an edge score. Its board
uses the same pinned Taskonomy frames as `corner` and `edge`; `--orientation-sigma`
travels in `dataset_params` and splits the comparability groups on its own.

## Its board

<!-- visbench:board task=orientation metrics=orientation_error,d1,d2 heading=3 -->
### orientation

| backbone | `orientation_error` | `d1` | `d2` | `ceiling_d1` | `ceiling_d2` | `ceiling_orientation_error` |
| --- | --- | --- | --- | --- | --- | --- |
| `mae_vitb16` | **18.8206** | **0.5820** | **0.7268** | 0.7047 | 0.8450 | 12.1822 |
| `dino_vitb16` | 21.4352 | 0.5244 | 0.6850 | 0.7047 | 0.8450 | 12.1822 |
| `sam_vitb16` | 21.7203 | 0.5231 | 0.6811 | 0.7047 | 0.8450 | 12.1822 |
| `dinov2_vits14` | 22.1286 | 0.4962 | 0.6688 | 0.7321 | 0.8652 | 11.0211 |
| `supervised_vitb16` | 24.2851 | 0.4608 | 0.6355 | 0.7047 | 0.8450 | 12.1822 |
| `dinov2_vitb14` | 24.5740 | 0.4646 | 0.6312 | 0.7321 | 0.8652 | 11.0211 |
| `convnext_base` | 28.2284 | 0.4194 | 0.5780 | 0.5699 | 0.7348 | 18.5669 |
| `clip_vitb16` | 28.2988 | 0.4097 | 0.5740 | 0.7047 | 0.8450 | 12.1822 |
| `resnet18` | 28.9932 | 0.4074 | 0.5672 | 0.5699 | 0.7348 | 18.5669 |
| `clip_vitb32` | 29.9416 | 0.3859 | 0.5494 | 0.5699 | 0.7348 | 18.5669 |
| `resnet50` | 29.9725 | 0.3956 | 0.5526 | 0.5699 | 0.7348 | 18.5669 |
| `siglip_vitb16` | 31.1453 | 0.3759 | 0.5317 | 0.7047 | 0.8450 | 12.1822 |

Ordered by `orientation_error`, which **disagrees with `d1`, `d2`, `median`, `rmse`** — this task does not rank its backbones the same way twice, so the row order is one of several defensible ones.

<sub>orientation on val/val, protocol=visbench_structure_tensor_orientation_regression, frozen [38bd953b]</sub>
<!-- /visbench:board -->

`orientation_error` is degrees, lower is better; chance is 45. The board spans
18.8° to 31.2° — every backbone is well clear of chance, and the ordering is
unlike any other low-level board: `mae_vitb16` leads (as it does across the
low-level tier), but the image-text ViTs `siglip_vitb16` and `clip_vitb32` are
*last*, where a semantic board puts them near the top, and DINOv2-S beats
DINOv2-B. See `CORPUS_FINDINGS.md` for what that does to the low-level tier.

## Run it

[`examples/orientation.py`](https://github.com/turhancan97/VisBench/blob/main/examples/orientation.py) is the whole path, end to end, on a real backbone.

```bash
python examples/orientation.py --data /path/to/dataset
```

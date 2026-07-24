# Low-level tasks — future scope

Nothing is implemented here, and nothing will be before **v0.3** — possibly not
even then, without contributor bandwidth. This folder documents intent so the
three-level taxonomy is complete and visible in the structure.

Low-level = signal-level properties recoverable largely without object or scene
understanding.

## Candidate tasks

| Task | Notes |
|---|---|
| Edge detection | Cheapest entry point; BSDS500 protocol is well established. |
| Optical flow | Needs image pairs and a flow head — closer in cost to a v0.2 dense task than to a probe. |
| Texture / reflectance | Intrinsic-image decomposition; ground truth is scarce outside synthetic data. |
| Image quality assessment | No-reference IQA against human MOS ratings. |

## Why nothing here yet

The v0.1 boundary is zero/near-zero-training tasks, and every candidate above
needs a trained dense head. Edge detection is the likely first entry, since it
reuses the v0.2 dense-head machinery almost unchanged.

Contributions welcome — see the head interface in `visbench/heads/`, which is a
deliberate extension point.

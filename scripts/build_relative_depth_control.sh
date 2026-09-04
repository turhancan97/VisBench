#!/usr/bin/env bash
#
# Run the relative-depth control: what a ranking-only readout of NYUv2 scores,
# against the metric `depth` board over the same frames.
#
#   scripts/build_relative_depth_control.sh
#   BACKBONES="dinov2_vits14" scripts/build_relative_depth_control.sh
#   DRY_RUN=1 scripts/build_relative_depth_control.sh
#
# WHAT THIS IS, AND WHAT IT IS NOT
#
# `RelativeDepthTask` is **not a registered probe**. It was built as a candidate
# and rejected on measurement: Spearman between its ranking and `depth`'s is
# +1.000 over these four backbones, with half the spread and a smallest
# adjacent gap of 0.0007 where `depth` manages 0.0624. A probe that cannot
# separate two backbones the existing one separates by a hundredfold has not
# earned a board. See the module docstring for the numbers.
#
# The result is kept as a control because it says something about a board that
# DOES ship: `depth` is not ranking backbones by metric accuracy, since
# discarding scale entirely leaves the ranking unchanged. It is ranking them by
# ordering plus feature resolution.
#
# WHY IT CANNOT USE build_corpus.sh
#
# That script drives `visbench run <probe>`, which resolves a name through the
# registry -- and this task is deliberately absent from it, so a registered
# probe's obligations (a corpus board, a CLI row, a TARGET_STYLES entry, a
# committed gallery figure) cannot be acquired by accident. So this constructs
# the task as an object and calls `visbench.run()` directly, which is the
# `CustomBackbone` path.
#
# The dataset flags below must stay equal to `probe_depth`'s in
# build_corpus.sh. That is the whole control: only the readout differs, so a
# board that changed the data as well could not say which of the two moved a
# rank.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

NYU=${NYU:-/shared/sets/datasets/vision/probing_3D/nyuv2_new}
RESULTS=${RESULTS:-results/controls/relative_depth.jsonl}
BACKBONES=${BACKBONES:-"dinov2_vits14 dinov2_vitb14 clip_vitb16 mae_vitb16 resnet50"}
DRY_RUN=${DRY_RUN:-}

if [[ ! -d "$NYU" ]]; then
  echo "No NYUv2 at $NYU -- set NYU=/path/to/nyuv2_new" >&2
  exit 1
fi
mkdir -p "$(dirname "$RESULTS")"

for backbone in $BACKBONES; do
  echo "=== relative_depth / $backbone"
  if [[ -n "$DRY_RUN" ]]; then
    echo "python scripts/run_relative_depth.py --backbone $backbone --data $NYU --results $RESULTS"
    continue
  fi
  python scripts/run_relative_depth.py \
    --backbone "$backbone" --data "$NYU" --results "$RESULTS" \
    ${VISBENCH_CACHE:+--cache "$VISBENCH_CACHE"} \
    || echo "!!! FAILED: relative_depth / $backbone" >&2
done

echo
echo "Records appended to $RESULTS"

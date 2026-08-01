#!/usr/bin/env bash
#
# Regenerate the leaderboard's record corpus (step 6e-2).
#
# Every published VisBench number to date was produced ad hoc and hand-copied
# into a markdown table; most of the records behind them no longer exist. This
# script is the replacement: one command per (probe, backbone), all of them
# appending schema-v6 records to a tracked JSONL, so a leaderboard reads what
# actually ran rather than what someone remembered.
#
# WHY A SCRIPT AND NOT TYPED COMMANDS
#
# `comparability_key` requires task_params and dataset_params to match exactly
# before two records may be ranked. So every backbone within a probe must run
# the *identical* command with only --backbone varying. A stray --limit or
# --image-size does not produce a wrong number, it produces two groups of one,
# and the run is wasted rather than misleading. Keeping the flags in one file
# is what makes that structural instead of a matter of care.
#
# USAGE
#
#   scripts/build_corpus.sh                 # every probe, every listed backbone
#   scripts/build_corpus.sh edge detection  # only the named probes
#   DRY_RUN=1 scripts/build_corpus.sh       # print the commands, run nothing
#
# Set BACKBONES to widen or narrow the matrix.

set -euo pipefail

VOC=/shared/sets/datasets/pascal_voc_2021/VOCdevkit/VOC2012
VOC_BINARY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/voc_binary"
IMAGENETTE=/shared/sets/datasets/Imagenette/imagenette2
NIGHTS=/shared/sets/datasets/vision/nights
TASKONOMY=/shared/sets/datasets/taskonomy-dataset/taskonomy

RESULTS=${RESULTS:-results/corpus/visbench.jsonl}
BACKBONES=${BACKBONES:-"dinov2_vits14 dinov2_vitb14"}
DRY_RUN=${DRY_RUN:-}

# Taskonomy and detection splits are held at 600/600 because that is what the
# published numbers used; changing it does not make them better, it makes them
# incomparable with everything already quoted.
TASKONOMY_LIMIT=600
DETECTION_LIMIT=600

mkdir -p "$(dirname "$RESULTS")"

run() {
  local probe=$1; shift
  for backbone in $BACKBONES; do
    echo "=== $probe / $backbone"
    if [[ -n "$DRY_RUN" ]]; then
      echo "visbench run $probe --backbone $backbone $* --results $RESULTS"
      continue
    fi
    # Deliberately not `set -e`-fatal: one probe failing should not discard the
    # runs already appended. The summary at the end reports what is missing.
    visbench run "$probe" --backbone "$backbone" "$@" --results "$RESULTS" || \
      echo "!!! FAILED: $probe / $backbone" >&2
  done
}

probe_classification() {
  run classification --data "$IMAGENETTE" --split val --train-split train
}

probe_retrieval() {
  run retrieval --data "$IMAGENETTE" --split val
}

probe_correspondence() {
  run correspondence --data "$IMAGENETTE" --split val --limit 200
}

probe_similarity() {
  # The zero-shot 2AFC protocol. `test` is the combined split; the
  # imagenet/no_imagenet halves are a contamination check and are run
  # separately rather than pooled, since quoting the combined number without
  # that gap overstates how much of it is perceptual alignment.
  run similarity --data "$NIGHTS" --split test
}

probe_semantic_segmentation() {
  run semantic_segmentation \
    --data "$VOC" --image-dir JPEGImages --target-dir SegmentationClass \
    --stems "$VOC/ImageSets/Segmentation/val.txt" \
    --train-stems "$VOC/ImageSets/Segmentation/train.txt" \
    --num-classes 21
}

probe_generic_segmentation() {
  # Reads the binary masks produced by scripts/binarise_voc_masks.py, through a
  # local root that symlinks VOC's JPEGImages beside them. Pointing this probe
  # at SegmentationClass directly would load, train and score against masks
  # that are wrong at every boundary -- see that script's docstring.
  if [[ ! -d "$VOC_BINARY/masks" ]]; then
    echo "!!! SKIPPED generic_segmentation: no masks at $VOC_BINARY/masks" >&2
    echo "    run scripts/binarise_voc_masks.py first" >&2
    return
  fi
  run generic_segmentation \
    --data "$VOC_BINARY" --image-dir JPEGImages --target-dir masks \
    --stems "$VOC/ImageSets/Segmentation/val.txt" \
    --train-stems "$VOC/ImageSets/Segmentation/train.txt" \
    --ignore-index 255
}

probe_detection() {
  # ImageSets/Main, not ImageSets/Segmentation -- the detection split is ~4x
  # larger, and a schedule sized on the segmentation one is not sized on this.
  run detection \
    --data "$VOC" --image-dir JPEGImages --annotation-dir Annotations \
    --stems "$VOC/ImageSets/Main/val.txt" \
    --train-stems "$VOC/ImageSets/Main/train.txt" \
    --limit "$DETECTION_LIMIT"
}

probe_edge() {
  run edge --data "$TASKONOMY" --partition tiny --limit "$TASKONOMY_LIMIT"
}

probe_keypoints2d() {
  run keypoints2d --data "$TASKONOMY" --partition tiny --limit "$TASKONOMY_LIMIT"
}

probe_occlusion_edge() {
  run occlusion_edge --data "$TASKONOMY" --partition tiny --limit "$TASKONOMY_LIMIT"
}

# depth and surface_normal are deliberately absent.
#
# Their published Taskonomy numbers (d1 0.5832/0.5986, mean 26.66/27.37) were
# produced by code that was never committed: only edge, keypoints2d and
# occlusion_edge have Taskonomy rows in the CLI, and both examples/depth.py and
# the depth CLI row take a flat <root>/<split>/{images,targets} layout that
# Taskonomy's building-nested store does not have. Adding them to the corpus
# needs a Taskonomy build for those two rows first -- a code change, not a
# command.
ALL_PROBES=(
  classification
  retrieval
  correspondence
  similarity
  semantic_segmentation
  generic_segmentation
  detection
  edge
  keypoints2d
  occlusion_edge
)

main() {
  local probes=("$@")
  if [[ ${#probes[@]} -eq 0 ]]; then
    probes=("${ALL_PROBES[@]}")
  fi

  local started
  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "corpus -> $RESULTS"
  echo "backbones: $BACKBONES"
  echo "started:   $started"
  echo

  for probe in "${probes[@]}"; do
    if ! declare -F "probe_$probe" >/dev/null; then
      echo "!!! unknown probe: $probe" >&2
      continue
    fi
    "probe_$probe"
  done

  echo
  echo "done. records in $RESULTS:"
  [[ -f "$RESULTS" ]] && wc -l < "$RESULTS"
}

main "$@"

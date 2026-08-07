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
#
# PUBLISHING THE TRAINED HEADS
#
#   PUSH_TO=your-hf-name scripts/build_corpus.sh depth edge
#
# appends `--push-to your-hf-name/visbench-<probe>-<backbone>` to every command,
# so the same run that produces a record also uploads the head it trained. The
# repositories are **private** unless PUSH_PUBLIC=1.
#
# Publishing from this script rather than a second one is deliberate: a probe
# head is only meaningful against the exact features it was fitted on, and the
# flags here are what fitted them. A separate publish script would be a second
# copy of every dataset flag, free to drift from the one that produced the
# numbers in the corpus -- and a head trained under drifted flags uploads,
# loads and scores without complaint.
#
# Zero-shot probes are skipped when pushing: retrieval, correspondence and
# similarity train nothing, so there is no head to share.
#
#   PUSH_TO=you DRY_RUN=1 scripts/build_corpus.sh   # see exactly what would go out

set -euo pipefail

VOC=/shared/sets/datasets/pascal_voc_2021/VOCdevkit/VOC2012
VOC_BINARY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/voc_binary"
CORNER_FRAMES="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/corner_frames"
IMAGENETTE=/shared/sets/datasets/Imagenette/imagenette2
NIGHTS=/shared/sets/datasets/vision/nights
TASKONOMY=/shared/sets/datasets/taskonomy-dataset/taskonomy
NYU=/shared/sets/datasets/vision/probing_3D/nyuv2_new

RESULTS=${RESULTS:-results/corpus/visbench.jsonl}
BACKBONES=${BACKBONES:-"dinov2_vits14 dinov2_vitb14"}
DRY_RUN=${DRY_RUN:-}

# Hub owner to publish the trained heads under; empty means publish nothing.
# Uploading is opt-in for the same reason `visbench run` needs --push-to: a
# push is not reversible the way a local write is.
PUSH_TO=${PUSH_TO:-}
PUSH_PUBLIC=${PUSH_PUBLIC:-}

# Where the feature cache lives. Left unset, `visbench run` defaults to
# ./.visbench_cache, which on this machine is under a 60 GB NFS home quota that
# is already 90% full -- and dense features are ~0.8 MB per image per backbone,
# so a full corpus run can exhaust it mid-array. Hitting a quota does not fail
# cleanly: it surfaces as a truncated cache write or a half-written record, and
# is then reported as some unrelated exception several steps later.
#
# Point it at a filesystem with room. The path is not recorded in a result, so
# it cannot affect comparability -- two runs differing only in cache location
# produce identical records.
VISBENCH_CACHE=${VISBENCH_CACHE:-}

# Taskonomy and detection splits are held at 600/600 because that is what the
# published numbers used; changing it does not make them better, it makes them
# incomparable with everything already quoted.
TASKONOMY_LIMIT=600
DETECTION_LIMIT=600

mkdir -p "$(dirname "$RESULTS")"

# Probes that fit nothing. `visbench run --push-to` refuses these before it
# runs, which is the right behaviour for a typed command and the wrong one for
# a loop: a hard failure here would abandon the probes after it.
ZERO_SHOT="retrieval correspondence similarity"

run() {
  local probe=$1; shift
  local cache_args=()
  [[ -n "$VISBENCH_CACHE" ]] && cache_args=(--cache "$VISBENCH_CACHE")

  local push_args=()
  if [[ -n "$PUSH_TO" ]]; then
    if [[ " $ZERO_SHOT " == *" $probe "* ]]; then
      echo "--- $probe: zero-shot, nothing to push (still recording it)" >&2
    else
      push_args=(--push-to "PLACEHOLDER")
      [[ -n "$PUSH_PUBLIC" ]] && push_args+=(--public)
    fi
  fi

  for backbone in $BACKBONES; do
    echo "=== $probe / $backbone"
    # The repo id carries both halves of the identity, because that is what a
    # visitor needs before the weights mean anything -- and one repo per pair,
    # since a head fitted on one backbone is refused against any other.
    [[ ${#push_args[@]} -gt 0 ]] && push_args[1]="$PUSH_TO/visbench-$probe-$backbone"

    if [[ -n "$DRY_RUN" ]]; then
      echo "visbench run $probe --backbone $backbone $* ${cache_args[*]} ${push_args[*]} --results $RESULTS"
      continue
    fi
    # Deliberately not `set -e`-fatal: one probe failing should not discard the
    # runs already appended. The summary at the end reports what is missing.
    visbench run "$probe" --backbone "$backbone" "$@" \
      "${cache_args[@]}" "${push_args[@]}" --results "$RESULTS" || \
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
  # --units pixel is the default since v0.6.1, and stated here anyway because
  # it is the whole comparability of this board. A patch width is a property of
  # the backbone -- 14px on DINOv2/14, 32px on a ResNet -- so scoring in patch
  # widths asks each backbone to hit a different target and prints the answers
  # under one name. It inverted this board: resnet18 read 0.8927 against
  # dinov2_vits14's 0.7834 on recall@1p, and 0.0973 against 0.3049 on recall@5px.
  run correspondence --data "$IMAGENETTE" --split val --limit 200 --units pixel
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

probe_corner() {
  # The one probe here whose target is COMPUTED rather than read, so it has no
  # dataset of its own -- and therefore no board without a chosen one. Two
  # people's corner numbers are comparable only if they ran the same images.
  #
  # The set chosen is Taskonomy tiny, the same 600/600 frames probe_edge reads,
  # staged as a flat image folder by scripts/stage_corner_frames.py. That is not
  # a convenience: the corner target correlates 0.52 with edge_texture, and the
  # claim that earns this probe its place is that the two nonetheless rank
  # backbones differently. That claim is exact only on shared pixels.
  #
  # Every generator setting is left at its default and travels in
  # dataset_params, so a run at a different --corner-sigma lands in a different
  # comparability group rather than being ranked against these.
  if [[ ! -d "$CORNER_FRAMES/val/images" ]]; then
    echo "!!! SKIPPED corner: no frames at $CORNER_FRAMES/val/images" >&2
    echo "    run scripts/stage_corner_frames.py first" >&2
    return
  fi
  run corner --data "$CORNER_FRAMES" --split val --train-split train
}

probe_depth() {
  # NYUv2, not Taskonomy. probe3d's own copy, whose layout happens to be exactly
  # the <root>/<split>/{images,targets} one the CLI already expects, so these
  # two probes need no code change after all.
  #
  # --target-scale 1.0 is load-bearing. The flag exists because depth datasets
  # ship millimetres in a 16-bit image container, and NYUv2's PNG distribution
  # uses 1000 -- but these are .npy already in metres. Passing 1000 here would
  # divide a 3-metre reading down to 3 millimetres, and depth_metrics reports
  # RMSE in whatever unit it is handed, so the number would look superb and
  # mean nothing.
  run depth --data "$NYU" --split test --train-split train \
    --image-dir images --target-dir depths \
    --target-scale 1.0 --max-depth 10.0
}

probe_surface_normal() {
  # The normals here are DENSE: not one zero-length vector in the 40 frames
  # sampled, including across the ~28% of pixels where the depth map has no
  # ground truth at all. So load_normal_map's validity rule marks nothing, and
  # this probe is scored everywhere, including on GeoNet's filled geometry.
  # That is what probe3d's own files support -- no mask ships beside them -- but
  # it means the number is not restricted to measured surfaces, and it is not
  # comparable with a masked normals probe such as the Taskonomy one.
  run surface_normal --data "$NYU" --split test --train-split train \
    --image-dir images --target-dir normals
}

ALL_PROBES=(
  classification
  retrieval
  correspondence
  similarity
  semantic_segmentation
  generic_segmentation
  detection
  depth
  surface_normal
  edge
  keypoints2d
  occlusion_edge
  corner
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

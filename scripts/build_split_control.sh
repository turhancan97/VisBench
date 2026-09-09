#!/usr/bin/env bash
#
# The split control: `detection` on the *instance* probe's images.
#
# WHAT QUESTION THIS ANSWERS
#
# 14a-4 found that `instance_segmentation` ranks with the mid-level geometry
# boards (occlusion_edge +0.958, surface_normal +0.930, depth +0.902) and not
# with its own high-level tier (semantic_segmentation +0.378, retrieval -0.217),
# mean +0.821 against mid-level and +0.238 against high. The obvious reading --
# "mask AP measures outlines, hence geometry" -- was tested and is wrong: the
# `box_map_50` half of the same runs agrees with the mask half at +0.986 and is
# topped by `occlusion_edge` too. So the box half ALONE ranks with geometry,
# while `detection` -- the same head, the same focal/GIoU losses, the same VOC
# matcher, the same metric -- sits at +0.804 with `semantic_segmentation`.
#
# Two probes, one implementation, two clusters. What differs is the data, and
# this control decomposes that.
#
# THE CONTROL AS FIRST WRITTEN DOWN WAS IMPOSSIBLE
#
# CORPUS_FINDINGS.md named it as "the instance probe's own head on
# ImageSets/Main at --limit 600". That cannot be run: `SegmentationObject`
# exists for 2913 images only, so **141 of the first 600 Main train stems have
# an instance mask** and the other 459 have no target at all. Checked, not
# assumed -- and the direction has to invert.
#
# `Annotations/` carries **17125 XMLs**, covering every image in the devkit
# including all 2913 segmentation ones. So the runnable control moves the
# *published* probe onto the *new* probe's images, rather than the reverse. That
# is also the better experiment: `detection` is the board with a published
# reading, so moving it is a change whose baseline is already understood.
#
# THE THREE-WAY DECOMPOSITION
#
#   A  full      detection, Segmentation stems, 1464 train / 1449 val
#   B  limit600  detection, Segmentation stems, --limit 600
#
# against the two boards already in the corpus:
#
#   C  detection            Main stems, --limit 600        (the published board)
#   D  instance_segmentation Segmentation stems, full      (14a-4's board)
#
#   A vs C  same head, same boxes, same protocol; different images AND size
#             -> is it the data at all?
#   A vs B  same images, same boxes; different training size
#             -> is it "how many" rather than "which"?
#   A vs D  same images, same size; differs only in where the boxes come from
#           (VOC's hand-drawn XML vs derived from the instance mask) and in the
#           mask branch riding alongside
#             -> is it something about the instance probe rather than its data?
#
# Those three comparisons exhaust the difference between C and D. Whichever
# pair moves the ranking is the answer, and a null result everywhere is itself
# informative: it would mean the effect is in box provenance, which is the one
# thing none of these vary on its own.
#
# WHY THIS IS A CONTROL AND NOT A BOARD
#
# `comparability_key` groups on the dataset name and fingerprint, so a
# `task=detection` record over the Segmentation split does NOT merge with the
# published detection board -- it makes that board **unrenderable**, because
# `board_for` refuses a task carrying more than one comparability group. That is
# the `scene_classification` lesson (a second dataset needs a new probe *name*,
# not a flag), and it is why these records go to `results/controls/` where
# nothing feeds a generated table. Keeping the two scripts apart is what makes
# that structural rather than a matter of remembering.
#
# The flags below are copied from build_corpus.sh's `probe_detection` and
# `probe_instance_segmentation` and must stay equal to them apart from the
# split list and the limit: the whole point is that only the data moved.
#
# USAGE
#
#   scripts/build_split_control.sh                     # both configs, 12 backbones
#   CONFIG=full scripts/build_split_control.sh          # one config
#   BACKBONES="dinov2_vits14 resnet18" scripts/build_split_control.sh
#   DRY_RUN=1 scripts/build_split_control.sh
#
# `slurm/split_control.sbatch` drives one (config, backbone) per array task and
# merges afterwards, for the same NFS-has-no-atomic-O_APPEND reason the corpus
# array does.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOC=/shared/sets/datasets/pascal_voc_2021/VOCdevkit/VOC2012

# The corpus detection board's own limit. Config B holds it so that A vs B
# varies size and nothing else.
DETECTION_LIMIT=600

RESULTS=${RESULTS:-results/controls/detection_split.jsonl}
DRY_RUN=${DRY_RUN:-}
VISBENCH_CACHE=${VISBENCH_CACHE:-}

# All twelve corpus backbones. A Spearman correlation against the published
# boards is only comparable to the +0.804/+0.650 figures this control exists to
# explain if it is taken over the same twelve rows -- a three-backbone rho would
# be a different statistic wearing the same name.
BACKBONE_LIST=(
  dinov2_vits14 dinov2_vitb14 clip_vitb16 clip_vitb32
  resnet18 resnet50 convnext_base mae_vitb16
  siglip_vitb16 supervised_vitb16 dino_vitb16 sam_vitb16
)

if [[ -n ${BACKBONES:-} ]]; then
  BACKBONE_LIST=($BACKBONES)
fi

ALL_CONFIGS=(full limit600)
CONFIGS=("${CONFIG:-${ALL_CONFIGS[@]}}")
if [[ -n ${CONFIG:-} ]]; then
  CONFIGS=("$CONFIG")
fi

mkdir -p "$(dirname "$RESULTS")"

if [[ ! -f "$VOC/ImageSets/Segmentation/val.txt" ]]; then
  echo "!!! No VOC segmentation split list at $VOC/ImageSets/Segmentation/" >&2
  exit 1
fi

# One run. `--stems`/`--train-stems` name ImageSets/Segmentation, which is what
# makes this the instance probe's images; everything else is probe_detection's.
run_one() {
  local config=$1 backbone=$2
  local limit_args=()
  [[ $config == limit600 ]] && limit_args=(--limit "$DETECTION_LIMIT")
  local cache_args=()
  [[ -n "$VISBENCH_CACHE" ]] && cache_args=(--cache "$VISBENCH_CACHE")

  echo "=== detection / $backbone / segmentation-split / $config"
  if [[ -n "$DRY_RUN" ]]; then
    echo "visbench run detection --backbone $backbone --data $VOC" \
      "--image-dir JPEGImages --annotation-dir Annotations" \
      "--stems $VOC/ImageSets/Segmentation/val.txt" \
      "--train-stems $VOC/ImageSets/Segmentation/train.txt" \
      "${limit_args[*]} ${cache_args[*]} --results $RESULTS"
    return
  fi
  # Not fatal: one failure should not discard the runs already appended.
  visbench run detection --backbone "$backbone" \
    --data "$VOC" --image-dir JPEGImages --annotation-dir Annotations \
    --stems "$VOC/ImageSets/Segmentation/val.txt" \
    --train-stems "$VOC/ImageSets/Segmentation/train.txt" \
    "${limit_args[@]}" "${cache_args[@]}" --results "$RESULTS" \
    || echo "!!! FAILED: detection / $backbone / $config" >&2
}

echo "control -> $RESULTS"
echo "configs:   ${CONFIGS[*]}"
echo "backbones: ${BACKBONE_LIST[*]}"
echo

for config in "${CONFIGS[@]}"; do
  for backbone in "${BACKBONE_LIST[@]}"; do
    run_one "$config" "$backbone"
  done
done

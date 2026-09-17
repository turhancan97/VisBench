#!/usr/bin/env bash
#
# The pose linear control: `relative_pose` with one affine map instead of an MLP.
#
# WHAT QUESTION THIS ANSWERS
#
# Every other VisBench board is quoted with the least expressive head that can
# express the task, because then a gap between two backbones is a gap between
# two *representations* and anything deeper can compensate for a weak feature
# vector. In practice that has always come out a linear map: an affine layer,
# a LinearHead, or the 1x1 convolutions DetectionHead and InstanceHead are
# built from. `relative_pose` is the first board here that departs from it.
#
# The departure was measured before it was made, on four ViT-B/16s: the linear
# head underfits at train_loss 0.0725-0.0732 -- flat across all four and 40x the
# MLP's -- clears the no-feature floor by 2.7-3.5 degrees against the MLP's
# 24.5-42.6, and produces a residual ordering that does not reproduce the MLP's
# and nearly inverts its top two. This runs that comparison over all thirteen
# corpus backbones, so the published board ships with the measurement of what
# the smaller head would have said rather than a sentence about it.
#
# WHY THIS IS A CONTROL AND NOT A BOARD
#
# `comparability_key` reads `task_params`, and the head is in it -- so these
# records form their own group and could not be listed beside the published
# board even if that were desirable. They go to results/controls/ where nothing
# feeds a generated table, for the reason every file there does: the corpus
# answers "what does this backbone score", and a row whose head differs invites
# exactly the reading it was built to prevent.
#
# COST
#
# Nearly free *after* the board and expensive before it, and what decides which
# is the CACHE, not the machine: the features are the same ones, so a warm cache
# makes each cell a couple of minutes of training while a cold one is about an
# hour of JPEG decode per backbone. The corpus array writes to
# /shared/results/common/kargin/visbench_cache (slurm/corpus.sbatch exports
# VISBENCH_CACHE), so run this against that root rather than a local one.
#
# USAGE
#
#   VISBENCH_CACHE=/shared/results/common/kargin/visbench_cache \
#     scripts/build_pose_linear_control.sh
#   BACKBONES="dinov2_vits14 resnet18" scripts/build_pose_linear_control.sh
#   DRY_RUN=1 scripts/build_pose_linear_control.sh
#
# On the cluster it is one job rather than an array -- thirteen cells of pure
# training is under an hour, where the board itself is thirteen hours of decode:
#
#   sbatch -p dgx --gpus=1 --cpus-per-task=8 --mem=64G --time=02:00:00 \
#     --exclude=dgx1 --output=logs/pose_linear_%j.log \
#     --wrap '. .venv/bin/activate && \
#       VISBENCH_CACHE=/shared/results/common/kargin/visbench_cache \
#       scripts/build_pose_linear_control.sh'
#
# `.` rather than `source`: --wrap hands the string to **sh**, not bash, and
# `source` is a bashism -- the job then fails in one second with
# "source: not found", which reads like a missing venv rather than a missing
# shell builtin.

set -euo pipefail

NAVI=${NAVI:-/shared/sets/datasets/vision/probing_3D/navi_v1}
RESULTS=${RESULTS:-results/controls/pose_linear.jsonl}
DRY_RUN=${DRY_RUN:-}
VISBENCH_CACHE=${VISBENCH_CACHE:-}

# All thirteen corpus backbones. A comparison against the published board is
# only comparable to it over the same rows -- a four-backbone ordering would be
# a different statistic wearing the same name.
BACKBONE_LIST=(
  dinov2_vits14 dinov2_vitb14 clip_vitb16 clip_vitb32
  resnet18 resnet50 convnext_base mae_vitb16
  siglip_vitb16 supervised_vitb16 dino_vitb16 sam_vitb16
  dino_vitb8
)

if [[ -n ${BACKBONES:-} ]]; then
  BACKBONE_LIST=($BACKBONES)
fi

mkdir -p "$(dirname "$RESULTS")"

if [[ ! -d "$NAVI" ]]; then
  echo "!!! No NAVI release at $NAVI" >&2
  exit 1
fi

# One run. Every flag is probe_relative_pose's from build_corpus.sh -- the same
# pairs, the same seed, the same pair count -- except the head. That is the
# whole point: only the function class moves.
run_one() {
  local backbone=$1
  local cache_args=()
  [[ -n "$VISBENCH_CACHE" ]] && cache_args=(--cache "$VISBENCH_CACHE")

  echo "=== relative_pose / $backbone / linear head"
  if [[ -n "$DRY_RUN" ]]; then
    echo "visbench run relative_pose --backbone $backbone --data $NAVI" \
      "--hidden-dims '' ${cache_args[*]} --results $RESULTS"
    return
  fi
  # Not fatal: one failure should not discard the runs already appended.
  visbench run relative_pose --backbone "$backbone" --data "$NAVI" \
    --hidden-dims "" "${cache_args[@]}" --results "$RESULTS" \
    || echo "!!! FAILED: relative_pose / $backbone / linear" >&2
}

echo "control -> $RESULTS"
echo "backbones: ${BACKBONE_LIST[*]}"
echo

for backbone in "${BACKBONE_LIST[@]}"; do
  run_one "$backbone"
done

echo
echo "done -> $RESULTS"
echo "These are NOT corpus records: the head is in task_params and therefore in"
echo "the comparability key, so they form their own group. See"
echo "results/controls/README.md."

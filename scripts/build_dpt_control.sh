#!/usr/bin/env bash
#
# Run the DPT-head control: the five low-level probes with a progressive
# decoder instead of a linear one, against the oracle gate that models a linear
# head exactly.
#
# WHY THIS SCRIPT EXISTS AT ALL
#
# The first ten records of `results/controls/dpt_head.jsonl` were produced by
# typed commands that were never committed. That is 8a's failure exactly -- six
# published corner figures whose records did not survive -- and it recurred on
# the newest control two releases after the step that was supposed to have
# ended it. Widening the control from 2 backbones to 12 is the moment to fix
# it: the flags live here now, so a later re-run cannot drift from the one that
# produced the numbers.
#
# WHY NOT build_corpus.sh
#
# `--head dpt --layers ...` must never reach a corpus record by accident. The
# head and the layers are both in `comparability_key`, so a DPT record under
# `task=edge` does not merge with the linear board -- it makes that board
# *unrenderable*, since `board_for` refuses a task with more than one group.
# Keeping the two scripts apart is what makes that structural rather than a
# matter of remembering. `results/controls/` is the same separation on disk.
#
# TWO GROUPS, AND THEY TEST DIFFERENT THINGS
#
# A ViT reads *blocks*: twelve of them, all one width and, crucially, all one
# grid. So `2 5 8 11` hands DPT four maps at the SAME resolution, and any gain
# over a linear head comes from depth and from decoding rather than from finer
# spatial input. Exceeding the oracle there means the decoder placed structure
# *within* a patch, which is the claim being tested.
#
# A CNN reads feature *stages*, at different widths AND different grids --
# 56/28/14/7 on a ResNet at 224px. So DPT reads genuinely finer spatial input
# than the final grid the oracle pools to, and exceeding the oracle is the
# expected outcome rather than a surprising one. The CNN group therefore
# answers the weaker question "does a pyramid recover more than a linear read
# of the last stage", not "can a decoder beat a per-patch bound".
#
# Do not merge the two readings. `layers` is in `comparability_key` anyway, so
# they are necessarily separate groups and separate files.
#
# THE LAYER SPEC IS PER BACKBONE, WHICH THE FIRST DRAFT GOT WRONG
#
# `num_layers` counts what timm's `feature_info` exposes, and that is not
# uniform across CNNs: a ResNet has **5** entries (the stem at index 0, then
# layer1..layer4), while ConvNeXt has **4** (no separate stem entry). So
# `1 2 3 4` is a ResNet's last four stages and is OUT OF RANGE on ConvNeXt,
# which was caught by a pre-check before an array burned on it rather than by
# reading the numbers afterwards.
#
# Each CNN therefore gets *its own last four stages*: `1 2 3 4` for a ResNet,
# `0 1 2 3` for ConvNeXt. Both include the deepest stage, which is the one a
# linear probe reads and the one the oracle's grid is taken from. Levelling
# them to a single spec was the other option and is worse: `0 1 2 3` on a
# ResNet drops layer4 entirely, so the control would compare a DPT head that
# never saw the final stage against a linear head that read only that stage.
#
# USAGE
#
#   scripts/build_dpt_control.sh                    # the ViT group, all 5 probes
#   GROUP=cnn scripts/build_dpt_control.sh          # the CNN group
#   scripts/build_dpt_control.sh edge corner        # only the named probes
#   DRY_RUN=1 GROUP=cnn scripts/build_dpt_control.sh
#
# BACKBONES narrows a group to the named backbones (each keeps its own layer
# spec); RESULTS says where the records go. `slurm/dpt_control.sbatch` drives one (probe, backbone) per array
# task and merges afterwards, for the same NFS reason the corpus array does.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORNER_FRAMES="$REPO/data/corner_frames"
TASKONOMY=/shared/sets/datasets/taskonomy-dataset/taskonomy

# Held at 600/600, the same as the linear boards these are compared against.
# A control that ran different frames would answer nothing.
TASKONOMY_LIMIT=600

GROUP=${GROUP:-vit}
DRY_RUN=${DRY_RUN:-}
VISBENCH_CACHE=${VISBENCH_CACHE:-}

# One entry per backbone, "name=layer spec". The spec travels WITH the name
# because it is not uniform (see the note above), and pairing them here is what
# stops a loop from applying one model's spec to another.
case "$GROUP" in
  vit)
    # The nine transformers in the corpus. All twelve-block -- verified off the
    # instance rather than assumed, since `sam_vitb16` is a timm SAM variant
    # and the dino/supervised/siglip rows are all vit_base_patch16 rebadged.
    SPECS=(
      "dinov2_vits14=2 5 8 11"
      "dinov2_vitb14=2 5 8 11"
      "clip_vitb16=2 5 8 11"
      "clip_vitb32=2 5 8 11"
      "mae_vitb16=2 5 8 11"
      "siglip_vitb16=2 5 8 11"
      "supervised_vitb16=2 5 8 11"
      "dino_vitb16=2 5 8 11"
      "sam_vitb16=2 5 8 11"
    )
    RESULTS=${RESULTS:-results/controls/dpt_head.jsonl}
    ;;
  cnn)
    # Each one's own last four stages. ResNet exposes 5 (stem at 0), ConvNeXt 4.
    SPECS=(
      "resnet18=1 2 3 4"
      "resnet50=1 2 3 4"
      "convnext_base=0 1 2 3"
    )
    RESULTS=${RESULTS:-results/controls/dpt_head_cnn.jsonl}
    ;;
  *)
    echo "GROUP must be vit or cnn, got '$GROUP'" >&2
    exit 1
    ;;
esac

# BACKBONES narrows the list to the named ones, keeping each one's own spec --
# which is what the Slurm array uses to run a single (probe, backbone) task.
if [[ -n ${BACKBONES:-} ]]; then
  filtered=()
  for want in $BACKBONES; do
    found=
    for spec in "${SPECS[@]}"; do
      [[ ${spec%%=*} == "$want" ]] && filtered+=("$spec") && found=1 && break
    done
    if [[ -z $found ]]; then
      echo "'$want' is not in the $GROUP group" >&2
      echo "  group holds: ${SPECS[*]%%=*}" >&2
      exit 1
    fi
  done
  SPECS=("${filtered[@]}")
fi

mkdir -p "$(dirname "$RESULTS")"

# The five probes with an oracle. Every other dense probe returns {} from
# `evaluate_oracle` -- pooling a class-index or bin-expectation target is
# meaningless -- so there is nothing for a DPT run to be measured against.
ALL_PROBES=(
  edge
  keypoints2d
  occlusion_edge
  corner
  orientation
)

run() {
  local probe=$1; shift
  local cache_args=()
  [[ -n "$VISBENCH_CACHE" ]] && cache_args=(--cache "$VISBENCH_CACHE")

  for spec in "${SPECS[@]}"; do
    local backbone=${spec%%=*}
    local layers=${spec#*=}
    echo "=== $probe / $backbone / dpt / layers $layers"
    if [[ -n "$DRY_RUN" ]]; then
      echo "visbench run $probe --backbone $backbone --head dpt --layers $layers $* ${cache_args[*]} --results $RESULTS"
      continue
    fi
    # Not fatal: one failure should not discard the runs already appended.
    visbench run "$probe" --backbone "$backbone" \
      --head dpt --layers $layers "$@" \
      "${cache_args[@]}" --results "$RESULTS" || \
      echo "!!! FAILED: $probe / $backbone" >&2
  done
}

# The flags below are copied from build_corpus.sh's own probe functions and must
# stay equal to them apart from --head/--layers: the whole point of the control
# is that only the head changed. tests/scripts/test_dpt_control.sh pins that.
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
  if [[ ! -d "$CORNER_FRAMES/val/images" ]]; then
    echo "!!! SKIPPED corner: no frames at $CORNER_FRAMES/val/images" >&2
    echo "    run scripts/stage_corner_frames.py first" >&2
    return
  fi
  run corner --data "$CORNER_FRAMES" --split val --train-split train
}

probe_orientation() {
  if [[ ! -d "$CORNER_FRAMES/val/images" ]]; then
    echo "!!! SKIPPED orientation: no frames at $CORNER_FRAMES/val/images" >&2
    echo "    run scripts/stage_corner_frames.py first" >&2
    return
  fi
  run orientation --data "$CORNER_FRAMES" --split val --train-split train
}

if [[ $# -gt 0 ]]; then
  PROBES=("$@")
else
  PROBES=("${ALL_PROBES[@]}")
fi

for probe in "${PROBES[@]}"; do
  if ! declare -F "probe_${probe}" >/dev/null; then
    echo "No such probe in this control: ${probe}" >&2
    echo "  available: ${ALL_PROBES[*]}" >&2
    exit 1
  fi
  "probe_${probe}"
done

echo
echo "Records appended to $RESULTS"

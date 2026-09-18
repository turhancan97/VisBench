#!/usr/bin/env bash
#
# The pose seed sweep: every corpus backbone re-fitted at several seeds.
#
# WHAT QUESTION THIS ANSWERS
#
# `relative_pose` is quoted to whole degrees, and five adjacent pairs on its
# board are called ties, because the fit's run-to-run scatter is about a degree.
# 19b measured that scatter properly for the first time and found two things
# that make the rule worth re-deriving over thirteen rows rather than two:
#
#   * the scatter is a TRIGGER rather than a dose -- perturbing the features
#     across a thousand-fold range moves the score by the same amount at every
#     size, and changing only the seed moves it as much again; and
#   * the two backbones it could measure disagreed by 3x, at 0.72 and 2.23
#     degrees, and a third of that gap is the *statistic*: a range over three
#     draws is one sample of something noisy, which is exactly how the published
#     tie list came to be calibrated against the smaller of two such samples.
#
# So the tie list currently rests on n=2, one of which may be an outlier. This
# sweep re-fits all thirteen at SEEDS seeds so the board can state its noise the
# way the low-level boards state a ceiling -- from a measurement, over every row
# it applies to.
#
# WHY THIS IS A CONTROL AND NOT A BOARD
#
# Unlike the linear control these records are the *published configuration* --
# same head, same pairs, same flags -- so they land in the identical
# comparability group as the corpus cells and `latest_per_backbone` would
# happily let a seed-3 run evict a published seed-0 one. That is precisely why
# they must not go near `results/corpus/`: a board is one seed by construction,
# and thirteen backbones at five seeds is sixty-five rankable rows describing
# thirteen models. `resolution.jsonl` is kept out for the same reason.
#
# THE SEED-0 ROW IS THE HARNESS CHECK, AND IT IS FREE
#
# Every corpus pose cell was run at the default seed 0, so this sweep's own
# seed-0 runs must reproduce the published values exactly. They are not a
# separate validation step: if they disagree, the sweep is measuring something
# other than the board and the other seeds mean nothing. Check them first.
#
# COST
#
# Pure training against a warm cache -- the features already exist, so this is
# thirteen loads and SEEDS*13 fits, roughly two hours at five seeds. Against a
# COLD cache it is about an hour of JPEG decode per backbone, so point
# VISBENCH_CACHE at the root the corpus array wrote:
# /shared/results/common/kargin/visbench_cache.
#
# USAGE
#
#   VISBENCH_CACHE=/shared/results/common/kargin/visbench_cache \
#     scripts/build_pose_seed_sweep.sh
#   SEEDS=3 BACKBONES="mae_vitb16 clip_vitb16" scripts/build_pose_seed_sweep.sh
#   DRY_RUN=1 scripts/build_pose_seed_sweep.sh
#
#   sbatch -p dgx --gpus=1 --cpus-per-task=8 --mem=64G --time=06:00:00 \
#     --exclude=dgx1 --output=logs/pose_seeds_%j.log \
#     --wrap '. .venv/bin/activate && \
#       VISBENCH_CACHE=/shared/results/common/kargin/visbench_cache \
#       scripts/build_pose_seed_sweep.sh'
#
# `.` rather than `source`: --wrap hands the string to **sh**, and `source` is a
# bashism that fails in one second looking like a missing venv.

set -euo pipefail

NAVI=${NAVI:-/shared/sets/datasets/vision/probing_3D/navi_v1}
RESULTS=${RESULTS:-results/controls/pose_seeds.jsonl}
DRY_RUN=${DRY_RUN:-}
VISBENCH_CACHE=${VISBENCH_CACHE:-}

# Five rather than three deliberately. The whole finding this answers is that a
# three-draw range is an unstable statistic, so measuring the fix with three
# draws would reproduce the problem it exists to correct.
SEEDS=${SEEDS:-5}

# All thirteen corpus backbones, in the board's own order.
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

# One run. Every flag is probe_relative_pose's from build_corpus.sh, unchanged
# -- the same pairs, the same head, the same pair count. Only --seed moves,
# which is the one thing this measures.
run_one() {
  local backbone=$1
  local seed=$2
  local cache_args=()
  [[ -n "$VISBENCH_CACHE" ]] && cache_args=(--cache "$VISBENCH_CACHE")

  echo "=== relative_pose / $backbone / seed $seed"
  if [[ -n "$DRY_RUN" ]]; then
    echo "visbench run relative_pose --backbone $backbone --data $NAVI" \
      "--seed $seed ${cache_args[*]} --results $RESULTS"
    return
  fi
  # Not fatal: one failure should not discard the cells already appended.
  visbench run relative_pose --backbone "$backbone" --data "$NAVI" \
    --seed "$seed" "${cache_args[@]}" --results "$RESULTS" \
    || echo "!!! FAILED: relative_pose / $backbone / seed $seed" >&2
}

echo "seed sweep -> $RESULTS"
echo "backbones: ${BACKBONE_LIST[*]}"
echo "seeds: 0..$((SEEDS - 1))"
echo

# Backbone outer, seed inner: a backbone's features are loaded once per run
# either way, but this order means a sweep killed part-way leaves whole rows
# rather than one seed of everything, and a whole row is analysable.
for backbone in "${BACKBONE_LIST[@]}"; do
  for ((seed = 0; seed < SEEDS; seed++)); do
    run_one "$backbone" "$seed"
  done
done

echo
echo "done -> $RESULTS"
echo "These are NOT corpus records. They carry the published configuration, so"
echo "they land in the SAME comparability group as the board and would evict it."
echo "Check the seed-0 rows against results/corpus/visbench.jsonl first: they"
echo "must reproduce it exactly, or the sweep is not measuring this board."

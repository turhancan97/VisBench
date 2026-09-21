#!/bin/bash
#
# Re-fit a published board at several seeds: one array task per (probe,
# backbone), SEEDS runs inside each (20b).
#
#   mkdir -p logs results/controls/parts results/controls/seeds
#   sbatch --array=0-194%4 slurm/seed_sweep.sbatch
#   scripts/merge_controls.sh
#   python scripts/analyse_seeds.py corner
#
# 15 probes x 13 backbones = 195 tasks, five seeds each. The array range is NOT
# derived from the lists below and cannot be: #SBATCH directives are read before
# any of this runs, so --array must be passed to match, and the guard below is
# what turns a mismatch into an error rather than a quietly short sweep.
#
# The first three probes are the 20b pilot and were run at indices 0-38. That
# ordering is kept so those indices still name the same (probe, backbone) pairs;
# the remaining twelve were run as
#
#   VISBENCH_PARTIAL=1 sbatch --array=39-194%4 slurm/seed_sweep.sbatch
#
# against a fifteen-probe list that included `scene_classification`; the indices
# above are the list as it stands. `vehicle_classification` was appended at the
# END for the same reason -- 21b ran it as
#
#   VISBENCH_PARTIAL=1 sbatch --array=182-194%4 slurm/seed_sweep.sbatch
#
# and every earlier index still names the pair it named before.
#
# which is what VISBENCH_PARTIAL is for -- a deliberate gap, here "the boards
# that were not already swept", rather than a short matrix nobody noticed.
#
# WHAT THIS MEASURES, AND WHY IT IS NOT A BOARD
#
# Every corpus cell is one fit at seed 0, and a board's adjacent rows are then
# ordered by their gap. 20a re-fitted `relative_pose` at five seeds and found two
# of its five published "tie" pairs come out the OTHER WAY ROUND -- so the gap is
# one draw of a quantity whose spread is unrelated to it. These sweeps ask the
# same question of boards that were never measured that way.
#
# The records carry the published configuration exactly and differ only in
# `seed`, which `comparability_key` does not read. They are therefore rankable
# rows inside the published board's own group and MUST NOT reach the corpus:
# `latest_per_backbone` would hand the board to whichever seed was written last.
# `scripts/build_corpus.sh` refuses SEEDS>1 against a corpus path, and
# `tests/results/test_seed_sweeps.py` checks the committed files.
#
# dgx, NOT dgxh100 -- the venv interpreter does not exist on the H100 image; see
# slurm/corpus.sbatch for the full note.
#SBATCH -p dgx
#SBATCH --gpus=1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --job-name=visbench-seed-sweep
#SBATCH --time=06:00:00
#SBATCH --array=0-194%4
#SBATCH --exclude=dgx1
#SBATCH --output=logs/seed_sweep_%A_%a.log

set -euo pipefail

export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/shared/results/common/kargin/.cache}
export HF_HOME=${HF_HOME:-/shared/results/common/kargin/.cache/huggingface}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

# The same shared cache the corpus array uses. A sweep re-fits features that
# already exist, so a warm cache is most of why this is affordable: measured on
# `corner`, the first seed of a backbone costs 88 s and each later one 40 s.
export VISBENCH_CACHE=${VISBENCH_CACHE:-/shared/results/common/kargin/visbench_cache}

REPO=${VISBENCH_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO"

if [[ ! -f pyproject.toml || ! -d .venv ]]; then
  echo "Not a VisBench checkout with a .venv: $REPO" >&2
  exit 1
fi

# Five rather than three deliberately: 19b showed a three-draw range is an
# unstable statistic, so measuring separability with three draws would reproduce
# the problem this exists to correct.
SEEDS=${SEEDS:-5}

# Every trained board whose sweep is usable, which is all of them bar two.
#
#   * `relative_pose`'s sweep predates this directory and lives in
#     `results/controls/pose_seeds.jsonl` (20a).
#   * `scene_classification` is ABSENT DELIBERATELY. It was swept with the rest
#     and eleven of its thirteen seed-0 rows do not reproduce their published
#     cells -- deterministically, on identical features, with a *different* fit.
#     Its records are kept at `results/controls/scene_classification_seeds.jsonl`
#     rather than here, because a sweep that does not reproduce its board
#     describes some adjacent configuration and must not be read as that board's
#     noise. See results/controls/README.md. To re-sweep it (on other silicon,
#     say):
#
#       SEEDS=5 RESULTS=results/controls/scene_classification_seeds.jsonl \
#         scripts/build_corpus.sh scene_classification
#
# The three zero-shot probes -- retrieval, correspondence, similarity -- are
# absent by construction: they fit no head, so a seed cannot move them, and a
# test derives that from the corpus rather than trusting this comment.
#
# The first three are the 20b pilot, kept at the front so their finished array
# indices keep naming the same pairs. Every probe here must have a committed
# sweep file and vice versa, and the backbones must be the ones on the board
# being swept -- tests/scripts/test_seed_sweep_scripts.py pins both, because a
# sweep short of a backbone is invisible afterwards.
PROBES=(
  classification
  corner
  detection
  orientation
  keypoints2d
  occlusion_edge
  edge
  generic_segmentation
  scene_parsing
  depth
  semantic_segmentation
  fine_grained_classification
  surface_normal
  instance_segmentation
  vehicle_classification
)
BACKBONE_LIST=(
  dinov2_vits14 dinov2_vitb14 clip_vitb16 clip_vitb32
  resnet18 resnet50 convnext_base mae_vitb16
  siglip_vitb16 supervised_vitb16 dino_vitb16 sam_vitb16
  dino_vitb8
)

expected=$(( ${#PROBES[@]} * ${#BACKBONE_LIST[@]} ))
index=${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array job}

if (( index >= expected )); then
  echo "Array index ${index} is outside the ${expected}-task matrix" >&2
  echo "  ${#PROBES[@]} probes x ${#BACKBONE_LIST[@]} backbones" >&2
  echo "  submit with --array=0-$(( expected - 1 ))" >&2
  exit 1
fi

# COUNT as well as the endpoint: a range ending at expected-1 can still omit
# tasks in the middle, and a strided range no endpoint check can see. A short
# sweep is invisible afterwards -- every probe present would still hold every
# backbone, exactly the failure the corpus array shipped for a whole release.
if [[ -z ${VISBENCH_PARTIAL:-} ]] \
  && [[ -n ${SLURM_ARRAY_TASK_MAX:-} ]] \
  && { (( SLURM_ARRAY_TASK_MAX != expected - 1 )) \
    || (( ${SLURM_ARRAY_TASK_COUNT:-expected} != expected )); }; then
  echo "!!! --array covers ${SLURM_ARRAY_TASK_COUNT:-?} tasks ending at ${SLURM_ARRAY_TASK_MAX}," >&2
  echo "!!! but the matrix is ${expected} tasks" >&2
  echo "!!!   ${#PROBES[@]} probes x ${#BACKBONE_LIST[@]} backbones" >&2
  echo "Submit --array=0-$(( expected - 1 )), or set VISBENCH_PARTIAL=1 if the" >&2
  echo "gap is deliberate (a smoke test, or resubmitting a few failed tasks)." >&2
  exit 1
fi

probe=${PROBES[$(( index / ${#BACKBONE_LIST[@]} ))]}
backbone=${BACKBONE_LIST[$(( index % ${#BACKBONE_LIST[@]} ))]}

# One file per task, merged afterwards: NFS has no atomic O_APPEND, so two tasks
# finishing together can interleave lines, and iter_records raises on the first
# bad line -- a corrupted sweep is worse than none.
RESULTS="results/controls/parts/seeds_${probe}__${backbone}.jsonl"
mkdir -p "$(dirname "$RESULTS")"

source .venv/bin/activate

if ! python --version >/dev/null 2>&1; then
  echo "The venv interpreter does not run on $(hostname)." >&2
  echo "  .venv/bin/python -> $(readlink -f .venv/bin/python 2>/dev/null || echo '<broken>')" >&2
  exit 1
fi

echo "=============================================="
echo "task ${index}: ${probe} / ${backbone} / seeds 0..$(( SEEDS - 1 ))"
echo "node:     $(hostname)"
echo "results:  ${RESULTS}"
echo "=============================================="

# build_corpus.sh, not a sweep script of its own: the sweep has to re-fit the
# PUBLISHED flags, and those live there. A second copy would drift, and a sweep
# measured under drifted flags is a noise figure for some adjacent probe.
SEEDS="$SEEDS" BACKBONES="$backbone" RESULTS="$RESULTS" \
  VISBENCH_CACHE="$VISBENCH_CACHE" \
  scripts/build_corpus.sh "$probe"

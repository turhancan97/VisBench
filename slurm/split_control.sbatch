#!/bin/bash
#
# Run the split control as a Slurm array: one task per (config, backbone).
#
#   mkdir -p logs results/controls/parts
#   sbatch --array=0-23%4 slurm/split_control.sbatch
#   scripts/merge_controls.sh
#
# 2 configs x 12 backbones = 24 tasks. The array range is NOT derived from the
# lists below and cannot be: #SBATCH directives are read before any of this
# runs, so --array must be passed to match, and the guard below is what turns a
# mismatch into an error rather than a quietly short control.
#
# See scripts/build_split_control.sh for what the control asks and why it is a
# control rather than a board. dgx, NOT dgxh100 -- the venv interpreter does not
# exist on the H100 image; see slurm/corpus.sbatch for the full note.
#SBATCH -p dgx
#SBATCH --gpus=1
#SBATCH --qos=quick
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --job-name=visbench-split-control
#SBATCH --time=01:00:00
#SBATCH --array=0-23%4
#SBATCH --output=logs/split_control_%A_%a.log

set -euo pipefail

export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/shared/results/common/kargin/.cache}
export HF_HOME=${HF_HOME:-/shared/results/common/kargin/.cache/huggingface}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

# The same shared cache the corpus array uses, on /shared/results rather than
# /home: /home is under an NFS quota that a dense run exhausts, and a truncated
# cache write surfaces later as an unrelated exception. Sharing it across tasks
# is safe -- both caches write via mkstemp + os.replace -- and the cache path is
# never recorded, so it cannot affect comparability.
export VISBENCH_CACHE=${VISBENCH_CACHE:-/shared/results/common/kargin/visbench_cache}

REPO=${VISBENCH_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO"

if [[ ! -f pyproject.toml || ! -d .venv ]]; then
  echo "Not a VisBench checkout with a .venv: $REPO" >&2
  exit 1
fi

# Must stay equal to build_split_control.sh's own lists;
# tests/scripts/test_split_control_scripts.py pins that -- the corpus array was
# short by a probe for a whole release because two files defined one matrix.
CONFIGS=(full limit600)
BACKBONE_LIST=(
  dinov2_vits14 dinov2_vitb14 clip_vitb16 clip_vitb32
  resnet18 resnet50 convnext_base mae_vitb16
  siglip_vitb16 supervised_vitb16 dino_vitb16 sam_vitb16
)

expected=$(( ${#CONFIGS[@]} * ${#BACKBONE_LIST[@]} ))
index=${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array job}

if (( index >= expected )); then
  echo "Array index ${index} is outside the ${expected}-task matrix" >&2
  echo "  ${#CONFIGS[@]} configs x ${#BACKBONE_LIST[@]} backbones" >&2
  echo "  submit with --array=0-$(( expected - 1 ))" >&2
  exit 1
fi

# COUNT as well as the endpoint: a range ending at expected-1 can still omit
# tasks in the middle, and a strided range no endpoint check can see.
if [[ -z ${VISBENCH_PARTIAL:-} ]] \
  && [[ -n ${SLURM_ARRAY_TASK_MAX:-} ]] \
  && { (( SLURM_ARRAY_TASK_MAX != expected - 1 )) \
    || (( ${SLURM_ARRAY_TASK_COUNT:-expected} != expected )); }; then
  echo "!!! --array covers ${SLURM_ARRAY_TASK_COUNT:-?} tasks ending at ${SLURM_ARRAY_TASK_MAX}," >&2
  echo "!!! but the matrix is ${expected} tasks" >&2
  echo "!!!   ${#CONFIGS[@]} configs x ${#BACKBONE_LIST[@]} backbones" >&2
  echo "!!! A short control is invisible afterwards: every config present would" >&2
  echo "!!! still hold every backbone." >&2
  echo "Submit --array=0-$(( expected - 1 )), or set VISBENCH_PARTIAL=1 if the" >&2
  echo "gap is deliberate (a smoke test, or resubmitting a few failed tasks)." >&2
  exit 1
fi

config=${CONFIGS[$(( index / ${#BACKBONE_LIST[@]} ))]}
backbone=${BACKBONE_LIST[$(( index % ${#BACKBONE_LIST[@]} ))]}

# One file per task, merged afterwards: NFS has no atomic O_APPEND, so two
# tasks finishing together can interleave lines and a corrupted control is
# worse than none -- iter_records raises on the first bad line.
RESULTS="results/controls/parts/detection_split__${config}__${backbone}.jsonl"
mkdir -p "$(dirname "$RESULTS")"

source .venv/bin/activate

if ! python --version >/dev/null 2>&1; then
  echo "The venv interpreter does not run on $(hostname)." >&2
  echo "  .venv/bin/python -> $(readlink -f .venv/bin/python 2>/dev/null || echo '<broken>')" >&2
  exit 1
fi

echo "=============================================="
echo "task ${index}: detection / ${backbone} / ${config}"
echo "node:     $(hostname)"
echo "results:  ${RESULTS}"
echo "=============================================="

CONFIG="$config" BACKBONES="$backbone" RESULTS="$RESULTS" \
  VISBENCH_CACHE="$VISBENCH_CACHE" \
  scripts/build_split_control.sh

#!/bin/bash
#
# Run the DPT-head control as a Slurm array: one task per (probe, backbone).
#
# Submit with:
#   mkdir -p logs results/controls/parts
#   sbatch --array=0-44%4 slurm/dpt_control.sbatch                 # 5 x 9 ViTs
#   VISBENCH_GROUP=cnn sbatch --array=0-14%3 slurm/dpt_control.sbatch  # 5 x 3 CNNs
#   scripts/merge_controls.sh                                      # once done
#
# The array range is NOT derived from the lists below and cannot be: #SBATCH
# directives are read before any of this runs. The guard below is what turns a
# mismatched range into an error instead of a quietly short control, which is
# the same reasoning -- and the same guard -- as slurm/corpus.sbatch.
#
# dgx, NOT dgxh100: `.venv/bin/python` is a symlink to /usr/bin/python3.10 and
# an H100 node ships only 3.12, so every console script fails naming the
# *script* rather than the missing interpreter.
#
# --exclude=dgx1 is the default here, and it is not superstition. dgx1 accepts
# work and starves it: `import torch` takes 1-1.6 s PER SUBMODULE there and
# never finishes, while dgx2 imports the same venv in 6 s. Nothing in Slurm
# reports the node as unhealthy. Re-confirmed 2026-09-04, when a pre-check job
# sat on dgx1 for six minutes without printing its first line. Set
# VISBENCH_NODES= to override once it clears.
#SBATCH -p dgx
#SBATCH --gpus=1
#SBATCH --qos=quick
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclude=dgx1
#SBATCH --job-name=visbench-dpt
#SBATCH --time=00:45:00
#SBATCH --output=logs/dpt_%A_%a.log

set -euo pipefail

export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/shared/results/common/kargin/.cache}
export HF_HOME=${HF_HOME:-/shared/results/common/kargin/.cache/huggingface}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

# The same shared cache the corpus array uses, on /shared/results rather than
# the quota'd home. A DPT run reads FOUR layers where the linear boards read
# one, so this is ~4x the dense features per backbone -- the one respect in
# which this control is more expensive than the board it is compared against.
export VISBENCH_CACHE=${VISBENCH_CACHE:-/shared/results/common/kargin/visbench_cache}

# SLURM_SUBMIT_DIR, never ${BASH_SOURCE[0]}: sbatch copies this script to
# /var/spool/slurmd/job<id>/, so BASH_SOURCE points outside the repo.
REPO=${VISBENCH_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO"

if [[ ! -f pyproject.toml || ! -d .venv ]]; then
  echo "Not a VisBench checkout with a .venv: $REPO" >&2
  echo "Submit from the repo root, or set VISBENCH_REPO=/path/to/VisBench" >&2
  exit 1
fi

# The five probes that declare an oracle. Every other dense probe returns {}
# from `evaluate_oracle`, so there is nothing to measure a DPT head against.
# Kept equal to build_dpt_control.sh's ALL_PROBES by
# tests/scripts/test_dpt_control_scripts.py -- the corpus array was short by a
# probe for a whole release because two files disagreed and each was
# self-consistent.
PROBES=(
  edge
  keypoints2d
  occlusion_edge
  corner
  orientation
)

GROUP=${VISBENCH_GROUP:-vit}
case "$GROUP" in
  vit) BACKBONE_LIST=(dinov2_vits14 dinov2_vitb14 clip_vitb16 clip_vitb32 \
                      mae_vitb16 siglip_vitb16 supervised_vitb16 dino_vitb16 sam_vitb16) ;;
  cnn) BACKBONE_LIST=(resnet18 resnet50 convnext_base) ;;
  *)   echo "VISBENCH_GROUP must be vit or cnn, got '$GROUP'" >&2; exit 1 ;;
esac

expected=$(( ${#PROBES[@]} * ${#BACKBONE_LIST[@]} ))
index=${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array job}

if (( index >= expected )); then
  echo "Array index ${index} is outside the ${expected}-task matrix" >&2
  echo "  ${#PROBES[@]} probes x ${#BACKBONE_LIST[@]} backbones (group: $GROUP)" >&2
  echo "  submit with --array=0-$(( expected - 1 ))" >&2
  exit 1
fi

# COUNT as well as the endpoint, so a strided or offset range cannot hide a hole
# in the middle. VISBENCH_PARTIAL is the claim that a short range is deliberate.
if [[ -z ${VISBENCH_PARTIAL:-} ]] \
  && [[ -n ${SLURM_ARRAY_TASK_MAX:-} ]] \
  && { (( SLURM_ARRAY_TASK_MAX != expected - 1 )) \
    || (( ${SLURM_ARRAY_TASK_COUNT:-expected} != expected )); }; then
  echo "!!! --array covers ${SLURM_ARRAY_TASK_COUNT:-?} tasks ending at ${SLURM_ARRAY_TASK_MAX}," >&2
  echo "!!! but the ${GROUP} matrix is ${expected} tasks" >&2
  echo "!!!   ${#PROBES[@]} probes x ${#BACKBONE_LIST[@]} backbones" >&2
  echo "Submit --array=0-$(( expected - 1 )), or set VISBENCH_PARTIAL=1 if the" >&2
  echo "gap is deliberate (a smoke test, or resubmitting a few failed tasks)." >&2
  exit 1
fi

probe=${PROBES[$(( index / ${#BACKBONE_LIST[@]} ))]}
backbone=${BACKBONE_LIST[$(( index % ${#BACKBONE_LIST[@]} ))]}

# One file per task, merged afterwards. NOT one shared JSONL: this repo is on
# NFSv4, which has no atomic O_APPEND, so two tasks finishing together can
# interleave lines -- and a corrupted JSONL raises on the first bad line with
# no way to tell which runs were lost.
mkdir -p results/controls/parts
RESULTS="results/controls/parts/${GROUP}__${probe}__${backbone}.jsonl"

source .venv/bin/activate

# Prove the interpreter runs on THIS node before spending anything. A venv is a
# symlink to a system python, so it is only valid where that version ships.
if ! python --version >/dev/null 2>&1; then
  echo "The venv interpreter does not run on $(hostname)." >&2
  echo "  .venv/bin/python -> $(readlink -f .venv/bin/python 2>/dev/null || echo '<broken>')" >&2
  echo "  this node has: $(ls /usr/bin/python3.* 2>/dev/null | tr '\n' ' ')" >&2
  exit 1
fi

echo "=============================================="
echo "task ${index}: ${probe} / ${backbone} / dpt (${GROUP})"
echo "node:     $(hostname)"
echo "results:  ${RESULTS}"
echo "cache:    ${VISBENCH_CACHE}"
nvidia-smi -L
echo "=============================================="

# The flags stay in build_dpt_control.sh. Two places defining them is how a
# --limit drifts and splits one comparability group into two groups of one.
GROUP="$GROUP" BACKBONES="$backbone" RESULTS="$RESULTS" \
  ./scripts/build_dpt_control.sh "$probe"

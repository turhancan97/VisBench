#!/bin/bash
#
# The 25b-1 learning-rate sweep (#172) as a Slurm array: one task per
# (board, backbone) cell, each running every arm of scripts/run_lr_sweep.py
# (one raw arm at the published lr, six standardised rates).
#
#   mkdir -p logs results/controls/parts/lr_sweep
#   sbatch --array=0-39%8 slurm/lr_sweep.sbatch
#
# EXTENDED=1 is the second pass: the five boards that chose the grid's top rate,
# at 2e-2 and 5e-2 only, into parts/lr_sweep_ext/ (20 tasks):
#
#   EXTENDED=1 sbatch --array=0-19%8 slurm/lr_sweep.sbatch
#
# The flags live in build_corpus.sh, read through the script; nothing here
# restates one. Parts go to their own directory, never the shared
# results/controls/parts/ that merge_controls.sh treats as a queue.
#
# dgx, not dgxh100 (no python3.10 there). V100s, because every corpus cell ran
# on one. --exclude=dgx2: its NFS stalled every task on 2026-11-25.
#SBATCH -p dgx
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclude=dgx2
#SBATCH --job-name=visbench-lrsweep
#SBATCH --time=04:00:00
#SBATCH --output=logs/lrsweep_%A_%a.log

set -euo pipefail

export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/shared/results/common/kargin/.cache}
export HF_HOME=${HF_HOME:-/shared/results/common/kargin/.cache/huggingface}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1
export VISBENCH_CACHE=${VISBENCH_CACHE:-/shared/results/common/kargin/visbench_cache}

# SLURM_SUBMIT_DIR, never ${BASH_SOURCE[0]}: sbatch runs a copy from /var/spool.
REPO=${VISBENCH_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO"
if [[ ! -f pyproject.toml || ! -d .venv ]]; then
  echo "Not a VisBench checkout with a .venv: $REPO" >&2
  exit 1
fi

PROBES=(depth surface_normal generic_segmentation semantic_segmentation scene_parsing
        edge keypoints2d occlusion_edge corner orientation)
extended_flag=()
parts=results/controls/parts/lr_sweep
if [[ -n ${EXTENDED:-} ]]; then
  PROBES=(generic_segmentation depth keypoints2d corner surface_normal)
  extended_flag=(--extended)
  parts=results/controls/parts/lr_sweep_ext
fi
BACKBONES=(convnext_base clip_convnext_base resnet50 dinov2_vitb14)

expected=$(( ${#PROBES[@]} * ${#BACKBONES[@]} ))
index=${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array job}
if (( index >= expected )); then
  echo "Array index ${index} is outside the ${expected}-task matrix" >&2
  exit 1
fi
if [[ -z ${VISBENCH_PARTIAL:-} ]] \
  && [[ -n ${SLURM_ARRAY_TASK_MAX:-} ]] \
  && { (( SLURM_ARRAY_TASK_MAX != expected - 1 )) \
    || (( ${SLURM_ARRAY_TASK_COUNT:-expected} != expected )); }; then
  echo "!!! --array covers ${SLURM_ARRAY_TASK_COUNT:-?} tasks; the matrix is ${expected}" >&2
  echo "Submit --array=0-$(( expected - 1 )), or set VISBENCH_PARTIAL=1 if deliberate." >&2
  exit 1
fi

probe=${PROBES[$(( index / ${#BACKBONES[@]} ))]}
backbone=${BACKBONES[$(( index % ${#BACKBONES[@]} ))]}

# One file per task: NFSv4 has no atomic O_APPEND.
mkdir -p "$parts"
RESULTS="${parts}/${probe}__${backbone}.jsonl"

source .venv/bin/activate
if ! python --version >/dev/null 2>&1; then
  echo "The venv interpreter does not run on $(hostname)." >&2
  exit 1
fi

echo "task ${index}: ${probe} / ${backbone} on $(hostname) -> ${RESULTS}"
nvidia-smi -L
python scripts/run_lr_sweep.py "$probe" --backbone "$backbone" --results "$RESULTS" \
  "${extended_flag[@]}"

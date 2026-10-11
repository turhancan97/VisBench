#!/bin/bash
#
# The double-crop control (22c) as a Slurm array: one task per (probe, backbone).
#
#   mkdir -p logs results/controls/parts/crop
#   sbatch --array=0-35%4 slurm/crop_control.sbatch                 # 4 probes x 9 timm rows
#   VISBENCH_GATE=1 VISBENCH_PARTIAL=1 sbatch --array=0-3 slurm/crop_control.sbatch
#                                    # the gate: published name, must reproduce the corpus
#
# The flags live in build_corpus.sh, read through scripts/run_crop_control.py;
# nothing here restates one. Parts go to their own directory, never the shared
# results/controls/parts/ that merge_controls.sh treats as a queue.
#
# dgx, not dgxh100 (no python3.10 there), and --exclude=dgx1 (degraded: it
# accepts work and starves it). See CLAUDE.md's node map.
#SBATCH -p dgx
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclude=dgx1
#SBATCH --job-name=visbench-crop
#SBATCH --time=00:45:00
#SBATCH --output=logs/crop_%A_%a.log

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

# Kept equal to run_crop_control.py's reading of _VARIANTS by
# tests/scripts/test_crop_control.py.
PROBES=(correspondence depth generic_segmentation edge)
BACKBONE_LIST=(convnext_base dino_vitb16 dino_vitb8 mae_vitb16 resnet18 resnet50 \
               sam_vitb16 siglip_vitb16 supervised_vitb16)
MODE=fullframe
if [[ -n ${VISBENCH_GATE:-} ]]; then
  # The gate re-runs the PUBLISHED name on one backbone per probe, through the
  # identical path. It must reproduce the corpus, or nothing else here counts.
  MODE=gate
  BACKBONE_LIST=(convnext_base)
fi

expected=$(( ${#PROBES[@]} * ${#BACKBONE_LIST[@]} ))
index=${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array job}
if (( index >= expected )); then
  echo "Array index ${index} is outside the ${expected}-task ${MODE} matrix" >&2
  exit 1
fi
if [[ -z ${VISBENCH_PARTIAL:-} ]] \
  && [[ -n ${SLURM_ARRAY_TASK_MAX:-} ]] \
  && { (( SLURM_ARRAY_TASK_MAX != expected - 1 )) \
    || (( ${SLURM_ARRAY_TASK_COUNT:-expected} != expected )); }; then
  echo "!!! --array covers ${SLURM_ARRAY_TASK_COUNT:-?} tasks; the ${MODE} matrix is ${expected}" >&2
  echo "Submit --array=0-$(( expected - 1 )), or set VISBENCH_PARTIAL=1 if deliberate." >&2
  exit 1
fi

probe=${PROBES[$(( index / ${#BACKBONE_LIST[@]} ))]}
backbone=${BACKBONE_LIST[$(( index % ${#BACKBONE_LIST[@]} ))]}

# One file per task: NFSv4 has no atomic O_APPEND.
mkdir -p results/controls/parts/crop
RESULTS="results/controls/parts/crop/${MODE}__${probe}__${backbone}.jsonl"

source .venv/bin/activate
if ! python --version >/dev/null 2>&1; then
  echo "The venv interpreter does not run on $(hostname)." >&2
  exit 1
fi

echo "task ${index}: ${probe} / ${backbone} / ${MODE} on $(hostname) -> ${RESULTS}"
nvidia-smi -L

gate_flag=()
[[ $MODE == gate ]] && gate_flag=(--gate)
python scripts/run_crop_control.py "$probe" --backbones "$backbone" \
  --results "$RESULTS" "${gate_flag[@]}"

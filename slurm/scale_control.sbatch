#!/bin/bash
#
# The feature-scale control (24b) as a Slurm array: one task per (mode, probe, arm).
#
#   mkdir -p logs results/controls/parts/scale
#   sbatch --array=0-27%6 slurm/scale_control.sbatch
#
# Tasks 0-13 are the scaled arms, 14-27 the gate (each arm at scale 1 through the
# identical alias path, which must reproduce the corpus cell). The flags live in
# build_corpus.sh, read through scripts/run_scale_control.py; nothing here
# restates one. Parts go to their own directory, never the shared
# results/controls/parts/ that merge_controls.sh treats as a queue.
#
# dgx, not dgxh100 (no python3.10 there). V100s, because every corpus cell this
# gates against ran on one. --exclude=dgx2: on 2026-11-25 its NFS left every task
# in D state for 45 min with the GPU idle (dgx1 imported torch in 9 s); the node
# map in CLAUDE.md, which says the reverse, predates that.
#SBATCH -p dgx
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclude=dgx2
#SBATCH --job-name=visbench-scale
#SBATCH --time=01:30:00
#SBATCH --output=logs/scale_%A_%a.log

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

# The linear dense boards clip_convnext_base sits at or near the foot of.
PROBES=(generic_segmentation orientation semantic_segmentation depth corner edge surface_normal)
ARM_LIST=(clip_convnext_base convnext_base)
MODES=(scaled gate)

per_mode=$(( ${#PROBES[@]} * ${#ARM_LIST[@]} ))
expected=$(( per_mode * ${#MODES[@]} ))
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

mode=${MODES[$(( index / per_mode ))]}
rest=$(( index % per_mode ))
probe=${PROBES[$(( rest / ${#ARM_LIST[@]} ))]}
arm=${ARM_LIST[$(( rest % ${#ARM_LIST[@]} ))]}

# One file per task: NFSv4 has no atomic O_APPEND.
mkdir -p results/controls/parts/scale
RESULTS="results/controls/parts/scale/${mode}__${probe}__${arm}.jsonl"

source .venv/bin/activate
if ! python --version >/dev/null 2>&1; then
  echo "The venv interpreter does not run on $(hostname)." >&2
  exit 1
fi

echo "task ${index}: ${probe} / ${arm} / ${mode} on $(hostname) -> ${RESULTS}"
nvidia-smi -L

gate_flag=()
[[ $mode == gate ]] && gate_flag=(--gate)
python scripts/run_scale_control.py "$probe" --arm "$arm" \
  --results "$RESULTS" "${gate_flag[@]}"

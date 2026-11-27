#!/bin/bash
#
# Is this dgx node healthy enough to run on? (#194)
#
#   sbatch -w dgx1 slurm/node_check.sbatch
#   sbatch -w dgx2 slurm/node_check.sbatch
#   cat logs/node_check_<jobid>.log
#
# The node map in CLAUDE.md has inverted once already: dgx1 was ~600x slow in
# 2026-09, and on 2026-11-25 dgx2's NFS left every task in D state for 45 min.
# Nothing in Slurm reports either as unhealthy, and a starved job looks exactly
# like broken code -- an empty log. So before a long array, time an import on
# each candidate and pass the slow one as --exclude on the sbatch command line.
# A healthy node imports torch in seconds; a degraded one takes minutes or never
# finishes, which is why this job has a short time limit.
#SBATCH -p dgx
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=visbench-node-check
#SBATCH --time=00:15:00
#SBATCH --output=logs/node_check_%j.log

set -uo pipefail
REPO=${VISBENCH_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO"
echo "node: $(hostname)"
nvidia-smi -L || true
SECONDS=0
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "import torch took ${SECONDS} s"
SECONDS=0
ls "${VISBENCH_CACHE:-/shared/results/common/kargin/visbench_cache}" >/dev/null
echo "listing the feature cache took ${SECONDS} s"

#!/bin/bash
#
# Build the leaderboard corpus as a Slurm array: one task per (probe, backbone).
#
# Submit with:
#   mkdir -p logs results/corpus/parts
#   sbatch slurm/corpus.sbatch
#   scripts/merge_corpus.sh          # once the array finishes
#
# Narrow it with the usual array syntax, e.g. --array=0-3 to smoke-test:
#   sbatch --array=0-3 slurm/corpus.sbatch
#
# dgx, NOT dgxh100.
#
# The node images differ: dgxh100 runs Ubuntu 24.04 with only
# /usr/bin/python3.12, while the login node and the dgx nodes run 22.04 with
# python3.10. `.venv/bin/python` is a symlink to /usr/bin/python3.10, so on an
# H100 node the interpreter simply is not there and every console script fails
# with "cannot execute: required file not found" -- which names the *script*,
# not the missing interpreter, and reads like a corrupt install.
#
# Staying on 22.04 keeps the corpus on the exact environment every published
# VisBench number came from. Building a second venv against 3.12 would work and
# is a supported version, but it is a different environment -- possibly a
# different torch build -- and "half the corpus ran somewhere else" is not a
# variable worth introducing into the first set of records anyone will rank.
#SBATCH -p dgx
#SBATCH --gpus=1
#SBATCH --qos=quick
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --job-name=visbench-corpus
#SBATCH --time=00:45:00
#SBATCH --array=0-23%4
#SBATCH --output=logs/corpus_%A_%a.log

set -euo pipefail

# XDG_CACHE_HOME is how torch.hub finds the already-downloaded DINOv2 weights at
# /shared/results/common/kargin/.cache/torch/hub. Without it every task re-downloads
# ~1.7 GB into a fresh path, concurrently, and 24 simultaneous writes to one hub
# directory is also how that cache gets corrupted rather than merely slow.
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/shared/results/common/kargin/.cache}
export HF_HOME=${HF_HOME:-/shared/results/common/kargin/.cache/huggingface}

# Match the thread pool to what was actually allocated. Left unset, torch sizes
# it from the node's core count -- 128 threads on a DGX -- and the tasks
# oversubscribe each other into being slower than running serially.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# So a tailed log shows progress rather than a buffer flushed at exit.
export PYTHONUNBUFFERED=1

# The feature cache goes on /shared/results, not in the repo.
#
# /home is under a 60 GB NFS quota that is already ~90% full, and a full corpus
# run adds several GB of dense features (~0.8 MB per image per ViT-B backbone).
# Exhausting a quota mid-array does not fail cleanly -- it surfaces as a
# truncated cache write and is reported as an unrelated exception later.
# /shared/results is a different filesystem (gmumresults, ~2 TB free) with no
# per-user quota.
#
# Sharing one cache across all 24 tasks is safe and wanted: FeatureCache and
# PrefixCache both write via mkstemp + os.replace, which is atomic on NFS within
# a directory, so the worst concurrent case is two tasks computing the same
# entry and one replace winning. The cache path is never recorded in a result,
# so this cannot affect comparability.
export VISBENCH_CACHE=${VISBENCH_CACHE:-/shared/results/common/kargin/visbench_cache}

# Locate the repo.
#
# NOT from ${BASH_SOURCE[0]}: sbatch copies the script to
# /var/spool/slurmd/job<id>/slurm_script before running it, so BASH_SOURCE
# points into the spool directory and `dirname .../..` resolves to
# /var/spool/slurmd. That is what "line 94: ../.venv/bin/activate: No such file
# or directory" was. SLURM_SUBMIT_DIR is the directory sbatch was invoked from,
# which is the repo root when submitted as documented.
REPO=${VISBENCH_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO"

# Fail loudly and early rather than 40 lines later with something obscure. The
# venv matters specifically: another interpreter on this machine does not have
# visbench importable, so the run would die inside the CLI instead of here.
if [[ ! -f pyproject.toml || ! -d .venv ]]; then
  echo "Not a VisBench checkout with a .venv: $REPO" >&2
  echo "Submit from the repo root, or set VISBENCH_REPO=/path/to/VisBench" >&2
  exit 1
fi

PROBES=(
  classification
  retrieval
  correspondence
  similarity
  semantic_segmentation
  generic_segmentation
  detection
  depth
  surface_normal
  edge
  keypoints2d
  occlusion_edge
)
BACKBONE_LIST=(dinov2_vits14 dinov2_vitb14)

index=${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array job}
probe=${PROBES[$(( index / ${#BACKBONE_LIST[@]} ))]}
backbone=${BACKBONE_LIST[$(( index % ${#BACKBONE_LIST[@]} ))]}

# One results file per task, merged afterwards.
#
# NOT one shared JSONL, even though ResultWriter opens with "a" and its
# docstring says concurrent cluster runs can append safely. That holds on a
# local POSIX filesystem for writes under PIPE_BUF. This repository lives on
# NFSv4 (s11.storage:/srv/nfs/gmumhome), and **NFS has no atomic O_APPEND** --
# the client writes at an offset it believes is EOF, so two tasks finishing
# together can interleave or overwrite each other's lines. A corrupted corpus is
# worse than no corpus: iter_records raises on the first bad line, and which
# runs were lost is not recoverable from what remains.
# mkdir -p results/corpus/parts
RESULTS="results/corpus/parts/${probe}__${backbone}.jsonl"

source .venv/bin/activate

# Prove the interpreter actually runs on THIS node before doing anything
# expensive. A venv is a symlink to a system python, so it is only valid on
# nodes whose image ships that version -- and when it is not, the failure
# surfaces as "cannot execute: required file not found" against the console
# script, which points at the wrong file entirely.
if ! python --version >/dev/null 2>&1; then
  echo "The venv interpreter does not run on $(hostname)." >&2
  echo "  .venv/bin/python -> $(readlink -f .venv/bin/python 2>/dev/null || echo '<broken>')" >&2
  echo "  this node has: $(ls /usr/bin/python3.* 2>/dev/null | tr '\n' ' ')" >&2
  echo "Submit to a partition whose image matches the venv (see -p above)." >&2
  exit 1
fi

echo "=============================================="
echo "task ${index}: ${probe} / ${backbone}"
echo "node:     $(hostname)"
echo "results:  ${RESULTS}"
echo "cache:    ${VISBENCH_CACHE}"
nvidia-smi -L
echo "=============================================="

# BACKBONES is what build_corpus.sh loops over, so pinning it to one name makes
# this task a single run. The flags themselves stay in that script -- two places
# defining them is how a --limit drifts and silently splits a comparability
# group into two groups of one.
BACKBONES="$backbone" RESULTS="$RESULTS" ./scripts/build_corpus.sh "$probe"

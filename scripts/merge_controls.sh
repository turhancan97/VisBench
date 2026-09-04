#!/usr/bin/env bash
#
# Merge the per-task JSONL files the DPT-control array wrote into the tracked
# control files.
#
#   scripts/merge_controls.sh
#
# The array writes one file per (group, probe, backbone) rather than appending
# to a shared one, because this repository is on NFS and NFS has no atomic
# O_APPEND -- see slurm/dpt_control.sbatch. This is the other half of that
# decision.
#
# **It merges; it does not replace.** merge_corpus.sh's docstring records why:
# a `cat parts/* > target` rebuild was right exactly once, and would since have
# deleted every board produced before the most recent array. The same applies
# here -- the ViT file already holds the two records the control shipped with.
#
# Idempotent: lines are deduplicated exactly, so running it twice adds nothing.
# A genuine *re-run* writes a new timestamp and so a new line, kept beside the
# old one; that is the append-only design, and `latest_per_backbone` picks the
# newest.
#
# Records are routed by the `group` prefix in each part's filename, because the
# two groups are two comparability keys and must not land in one file.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PARTS=${PARTS:-results/controls/parts}

if [[ ! -d "$PARTS" ]]; then
  echo "No parts directory at $PARTS -- has the array run?" >&2
  exit 1
fi

merge_group() {
  local group=$1 target=$2
  local found=("$PARTS"/"$group"__*.jsonl)
  if [[ ! -e ${found[0]} ]]; then
    echo "--- $group: no parts, nothing to merge"
    return
  fi

  local before=0
  [[ -f "$target" ]] && before=$(grep -c '' "$target")

  local tmp
  tmp=$(mktemp)
  # Existing lines first so their order is preserved; dedup keeps first sight.
  [[ -f "$target" ]] && cat "$target" >>"$tmp"
  cat "${found[@]}" >>"$tmp"
  awk 'NF && !seen[$0]++' "$tmp" >"$target"
  rm -f "$tmp"

  local after
  after=$(grep -c '' "$target")
  echo "--- $group: ${#found[@]} parts, $before -> $after lines in $target"
}

merge_group vit results/controls/dpt_head.jsonl
merge_group cnn results/controls/dpt_head_cnn.jsonl

echo
echo "Parts left in $PARTS; delete them once the merge looks right."

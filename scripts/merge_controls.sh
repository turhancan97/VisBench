#!/usr/bin/env bash
#
# Merge the per-task JSONL files a control array wrote into the tracked control
# files. Two controls use it: the DPT-head one and the split one.
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
# The split control (2026-09-09): `detection` on the instance probe's images.
# Its two configs DO share a group here, unlike the DPT ones -- they differ only
# in `--limit`, so `dataset_size` tells them apart inside one file and
# `comparability_key` keeps them out of the published board either way.
merge_group detection_split results/controls/detection_split.jsonl

# A sweep that does NOT reproduce its own board is held out by hand, and its
# parts stay in the archive like everyone else's -- so deriving the probe list
# from the parts present routes it straight back into seeds/ as though it were
# publishable. That is `results/corpus/parts/ is an archive, not a queue` in a
# second file: a merge dedups by exact line, which cannot see a record kept
# *out* on purpose. `scene_classification` (20c/20d) lives at
# results/controls/scene_classification_seeds.jsonl and must never appear under
# seeds/, where every consumer reads it as a board's measured noise.
#
# Refused here rather than left to `tests/results/test_seed_sweeps.py`. That
# test does catch it -- the gate fails on the first seed-0 row -- but only after
# the file is written, and a merge should not depend on a later test to undo it.
HELD_OUT_SWEEPS=" scene_classification "

# The seed sweeps (20b): one file per swept probe, under results/controls/seeds/.
# Derived from the parts present rather than listed, because this table would
# otherwise have to be edited every time another board is swept -- and a probe
# missing from it would leave its parts unmerged while the merge reported
# success, which is the corpus array's own failure in a different file.
for part in "$PARTS"/seeds_*__*.jsonl; do
  [[ -e $part ]] || break
  probe=$(basename "$part"); probe=${probe#seeds_}; probe=${probe%%__*}
  [[ " ${swept_probes:-} " == *" $probe "* ]] && continue
  swept_probes="${swept_probes:-} $probe"
  if [[ $HELD_OUT_SWEEPS == *" $probe "* ]]; then
    echo "--- seeds_$probe: HELD OUT, not merged into seeds/ (see results/controls/README.md)"
    continue
  fi
  mkdir -p results/controls/seeds
  merge_group "seeds_$probe" "results/controls/seeds/$probe.jsonl"
done

echo
echo "Parts left in $PARTS; delete them once the merge looks right."

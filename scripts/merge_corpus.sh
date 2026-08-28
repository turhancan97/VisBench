#!/usr/bin/env bash
#
# Merge the per-task JSONL files a Slurm array wrote into the tracked corpus.
#
# The array writes one file per (probe, backbone) rather than appending to a
# shared one, because this repository is on NFS and NFS has no atomic O_APPEND
# -- see the comment in slurm/corpus.sbatch. This is the other half of that
# decision.
#
#   scripts/merge_corpus.sh
#
# **It merges into the corpus; it does not replace it.** The original version
# rebuilt the corpus by `cat parts/*.jsonl > corpus`, which was right exactly
# once -- at 6e-2, when the whole matrix was produced by one array. The corpus
# has been widened one board at a time ever since, so `parts/` holds only the
# most recent board and a rebuild would delete every board before it. Checked
# on 2026-08-28: 12 orientation parts against a 180-record, 15-board corpus, so
# a rebuild would have dropped 168 records and 14 boards. That is silent -- the
# generated tables would simply render fewer boards, and every board still
# present would still hold every backbone.
#
# Still idempotent: records are deduplicated by their exact JSON line, so
# running it twice adds nothing. A *re-run* of a (probe, backbone) writes a new
# timestamp and so a new line, which is kept beside the old one -- that is the
# corpus's append-only design, and `latest_per_backbone` is what picks the
# newest.
#
# REBUILD=1 restores the old behaviour, for the case it was written for: the
# whole matrix in `parts/` and a corpus to be replaced wholesale.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PARTS=${PARTS:-results/corpus/parts}
CORPUS=${CORPUS:-results/corpus/visbench.jsonl}

if [[ ! -d "$PARTS" ]]; then
  echo "No parts directory at $PARTS -- has the array run?" >&2
  exit 1
fi

shopt -s nullglob
files=("$PARTS"/*.jsonl)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "No .jsonl parts in $PARTS" >&2
  exit 1
fi

# Sorted so the corpus has a stable order and its diff is readable when a single
# probe is re-run.
printf '%s\n' "${files[@]}" | sort | xargs cat > "$CORPUS.parts"

if [[ -n ${REBUILD:-} ]]; then
  # The 6e-2 behaviour, kept for the case it was written for: the whole matrix
  # is in parts/ and the corpus is meant to be replaced.
  echo "REBUILD=1: replacing $CORPUS with ${#files[@]} parts"
  mv "$CORPUS.parts" "$CORPUS"
else
  # Existing records first, then any part line not already present. Keying on
  # the exact line makes a second merge a no-op, while a genuine re-run -- which
  # carries a new timestamp -- is a new line and is kept, because the corpus is
  # append-only and latest_per_backbone is what resolves duplicates.
  if [[ -f "$CORPUS" ]]; then
    awk 'NR==FNR { seen[$0]=1; next } !($0 in seen)' "$CORPUS" "$CORPUS.parts" \
      > "$CORPUS.new"
    added=$(wc -l < "$CORPUS.new")
    cat "$CORPUS" "$CORPUS.new" > "$CORPUS.tmp"
    rm -f "$CORPUS.new"
    echo "merged ${#files[@]} parts -> $CORPUS  (+${added} new records)"
  else
    cp "$CORPUS.parts" "$CORPUS.tmp"
    echo "created $CORPUS from ${#files[@]} parts"
  fi
  mv "$CORPUS.tmp" "$CORPUS"
  rm -f "$CORPUS.parts"
fi

wc -l < "$CORPUS" | xargs echo "records:"

# Validate rather than trust. Every line must parse as a schema-readable record,
# and the whole point of the corpus is that a leaderboard can group it -- so
# report the groups too. A part that failed mid-run leaves a truncated final
# line, and this is where that shows up rather than three steps later.
source .venv/bin/activate 2>/dev/null || true
python3 - "$CORPUS" <<'PY'
import sys
from collections import Counter

from visbench.results import group_comparable, read_records

path = sys.argv[1]
records = read_records(path)
print(f"parsed:  {len(records)} records, schema v{min(r.schema_version for r in records)}"
      f"-v{max(r.schema_version for r in records)}")

by_task = Counter(r.task for r in records)
missing = [t for t, n in by_task.items() if n < 2]
print(f"probes:  {len(by_task)}")
for task, count in sorted(by_task.items()):
    print(f"  {task:24s} {count}")

groups = group_comparable(records)
print(f"groups:  {len(groups)} comparability groups")
rankable = sum(1 for g in groups.values() if len({r.backbone for r in g}) > 1)
print(f"         {rankable} with more than one backbone (i.e. actually rankable)")

if missing:
    print(f"\nWARNING: only one record for {sorted(missing)} -- nothing to rank there")
PY

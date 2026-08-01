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
# Idempotent: it rebuilds the corpus from the parts every time rather than
# appending, so running it twice does not double every record.

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
printf '%s\n' "${files[@]}" | sort | xargs cat > "$CORPUS.tmp"
mv "$CORPUS.tmp" "$CORPUS"

echo "merged ${#files[@]} parts -> $CORPUS"
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

#!/bin/bash
# Join the rendered scenes, in script.md's order, into one video.
#
#   sbatch -p cpu --cpus-per-task=2 --mem=4G --time=00:20:00 \
#          -o logs/video_%j.log video/assemble.sh
#
# Every scene must already be rendered at the same quality (QUALITY, default
# 1080p60); a missing one is an error, never a shorter video.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
E=${VIDEO_ENV:-/shared/results/common/kargin/.cache/envs/visbench-video}
export PATH="$E/bin:$PATH"
RES=${RES:-1080p60}
OUT=video/build/visbench_explainer.mp4
LIST=video/build/concat.txt
: > "$LIST"
for scene in $(python -c "import sys; sys.path.insert(0, 'video'); from narrate import scenes; print(' '.join(scenes()))"); do
  class=$(sed -n 's/^class \([A-Za-z]*\)(NarratedScene).*/\1/p' "video/scenes/$scene.py")
  mp4="video/build/media/videos/$scene/$RES/$class.mp4"
  [[ -f $mp4 ]] || { echo "missing $mp4 -- render $scene at $RES first" >&2; exit 1; }
  echo "file '$PWD/$mp4'" >> "$LIST"
done
ffmpeg -loglevel error -y -f concat -safe 0 -i "$LIST" -c:v copy -c:a aac -b:a 192k "$OUT"
ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT" | xargs printf "wrote $OUT (%.1f s)\n"

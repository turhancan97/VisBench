#!/bin/bash
# Render one or more scenes end to end: data -> narration -> Manim -> stills.
#
#   sbatch -p cpu --cpus-per-task=4 --mem=16G --time=01:00:00 \
#          -o logs/video_%j.log video/render.sh twist
#
# Runs on a compute node, never the login node. The environment is a separate
# conda env (see video/README.md), not the project .venv.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
E=${VIDEO_ENV:-/shared/results/common/kargin/.cache/envs/visbench-video}
export PATH="$E/bin:$PATH" HF_HOME=/shared/results/common/kargin/.cache/huggingface
unset HF_TOKEN   # the stale token shadows the cached login and 401s public downloads
QUALITY=${QUALITY:--qh}   # -ql for a fast draft, -qh for 1080p60

# data.json is committed and regenerated with the project's Python
# (`PYTHONPATH=. python video/extract_data.py`), since it imports visbench and
# this environment deliberately has no torch.
test -f video/data.json || { echo "video/data.json missing; run extract_data.py" >&2; exit 1; }
for scene in "$@"; do
  python video/narrate.py "$scene" ${VOICE:+--voice "$VOICE"}
  class=$(python - "$scene" <<'PY'
import re, sys
src = open(f"video/scenes/{sys.argv[1]}.py").read()
print(re.search(r"^class (\w+)\(NarratedScene\)", src, re.M).group(1))
PY
)
  (cd video/scenes && manim $QUALITY --media_dir ../build/media "$scene.py" "$class")
  mp4=$(ls -t video/build/media/videos/"$scene"/*/"$class".mp4 | head -1)
  mkdir -p video/build/stills/"$scene"
  # One still at the end of every beat, for checking the layout by eye.
  python - "$scene" "$mp4" <<'PY'
import json, subprocess, sys
scene, mp4 = sys.argv[1], sys.argv[2]
t = json.load(open(f"video/build/audio/{scene}/timing.json"))
end = 0.0
for i, d in enumerate(t["durations"]):
    end += d + t["gap"]
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{end - 0.15:.2f}", "-i", mp4,
                    "-frames:v", "1", f"video/build/stills/{scene}/beat{i}.png"], check=True)
print(f"{scene}: {mp4}  ({end:.1f}s, {len(t['durations'])} stills)")
PY
done

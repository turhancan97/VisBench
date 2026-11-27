"""No Slurm file hard-codes which node to avoid (#194).

The cluster's node health has inverted once already: `dgx1` was ~600x slow in
2026-09 and every file excluded it, then on 2026-11-25 `dgx2` was the one
starving jobs while `dgx1` imported torch in 9 s. A hard-coded exclusion is
right on the day it is written and silently wrong after, and a job sent to a
starved node looks exactly like broken code -- an empty log. The rule instead:
time an import with `slurm/node_check.sbatch`, then pass `--exclude` on the
command line.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLURM = sorted((ROOT / "slurm").glob("*.sbatch"))


def test_there_are_slurm_files_to_check():
    assert len(SLURM) >= 7


def test_no_slurm_file_pins_or_excludes_a_node():
    pinned = re.compile(r"^#SBATCH\s+(--exclude|-x|--nodelist|-w)\b", re.MULTILINE)
    offenders = [p.name for p in SLURM if pinned.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_the_node_check_exists_and_times_an_import():
    text = (ROOT / "slurm" / "node_check.sbatch").read_text(encoding="utf-8")
    assert "import torch" in text
    assert "#SBATCH -p dgx" in text

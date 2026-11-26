"""The paper's figure: each backbone's rank on each of the twenty boards.

    python3 paper/make_figure.py        # writes paper/figure.png

Reads ``video/data.json``, which ``video/extract_data.py`` generates from the
committed corpus. Regenerate that file first whenever the corpus changes, so
the figure and the leaderboard cannot disagree. Needs only matplotlib, not
visbench or torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "video" / "data.json").read_text())
OUT = Path(__file__).resolve().parent / "figure.png"

LEVELS = ("high_level", "mid_level", "low_level")
LEVEL_NAME = {"high_level": "high-level", "mid_level": "mid-level", "low_level": "low-level"}
LEVEL_COLOUR = {"high_level": "#8A6BDD", "mid_level": "#55A398", "low_level": "#EAB34D"}
BLUE, PALE = "#507AFF", "#EEF1F8"


def main() -> None:
    ranks, levels = DATA["ranks"], DATA["levels"]
    rows = sorted(DATA["mean_rank"], key=DATA["mean_rank"].get)
    columns = [p for level in LEVELS for p in levels if levels[p] == level]
    n = len(rows)
    grid = [[ranks[p][b] for p in columns] for b in rows]

    cmap = LinearSegmentedColormap.from_list("rank", [BLUE, PALE])
    fig, ax = plt.subplots(figsize=(10, 3.9), dpi=200)
    ax.imshow(grid, cmap=cmap, vmin=1, vmax=n, aspect="auto")
    for i, row in enumerate(grid):
        for j, rank in enumerate(row):
            ax.text(
                j, i, str(rank), ha="center", va="center", fontsize=6.5,
                color="white" if rank <= n / 2 else "#1E2028",
            )

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right", rotation_mode="anchor", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"{b}  ({DATA['mean_rank'][b]:.1f})" for b in rows], fontsize=7
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # One coloured band over each level's columns.
    start = 0
    for level in LEVELS:
        width = sum(levels[p] == level for p in columns)
        ax.plot(
            [start - 0.45, start + width - 0.55], [-0.85, -0.85],
            color=LEVEL_COLOUR[level], linewidth=4, solid_capstyle="butt", clip_on=False,
        )
        ax.text(
            start + (width - 1) / 2, -1.2, LEVEL_NAME[level], ha="center",
            va="bottom", fontsize=8, color=LEVEL_COLOUR[level], weight="bold",
        )
        if start:
            ax.axvline(start - 0.5, color="white", linewidth=2)
        start += width
    ax.set_ylim(n - 0.5, -0.5)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

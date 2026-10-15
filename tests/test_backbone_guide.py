"""Every backbone the package registers is named in the backbones guide.

`docs/guides/backbones.md` lists the registry by family, by hand. It went a
whole release without `dino_vitb8` (15a) and nothing failed, because no test
read it — the same drift `tests/test_docs_counts.py` exists for, on a list
rather than a count.
"""

from __future__ import annotations

from pathlib import Path

import visbench
from visbench.registry import _BACKBONES

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "guides" / "backbones.md"


def _shipped() -> list[str]:
    """Names registered by the package itself.

    The registry is global and other tests register fakes into it, so "every
    name in `list_backbones()`" depends on test order; the defining module does
    not.
    """
    names = visbench.list_backbones()  # imports every registration module first
    return sorted(n for n in names if _BACKBONES[n][0].__module__.startswith("visbench."))


def test_the_guard_reaches_the_corpus_columns():
    assert {"dinov2_vits14", "clip_vitb16", "convnext_base", "dino_vitb8"} <= set(_shipped())


def test_every_registered_backbone_is_in_the_guide():
    text = GUIDE.read_text(encoding="utf-8")
    missing = [name for name in _shipped() if f"`{name}`" not in text]
    assert not missing, f"docs/guides/backbones.md does not name: {missing}"

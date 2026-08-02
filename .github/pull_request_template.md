## What this changes

<!-- One or two sentences. -->

## What you measured

<!--
Not only what you wrote. If a number moved, show both sides — VisBench's whole
job is reporting numbers that mean what they say, so a change that touches one
should say what it did to it.
-->

## Checks

- [ ] `pytest`
- [ ] `pytest -m slow`, or "not affected"
- [ ] `ruff check` / `ruff format --check` / `mypy`
- [ ] `uv lock` re-run, if dependencies or the version changed
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] New guards have a test that **fails when the guard is removed**

<!--
If something is red or skipped, say so here rather than leaving it unticked. A
PR that admits a gap is cheaper to review than one that overstates its evidence.
-->

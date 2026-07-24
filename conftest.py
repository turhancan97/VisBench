"""Root conftest.

Its presence at the repo root makes pytest prepend that root to ``sys.path``,
so ``import visbench`` resolves from a plain ``pytest`` invocation in a fresh
clone — without it, only ``python -m pytest`` or an editable install works,
which is a confusing first failure for a new contributor.
"""

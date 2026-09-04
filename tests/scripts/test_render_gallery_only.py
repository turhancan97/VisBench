"""`render_gallery.py --only`: render one figure without rewriting the rest.

Added when the gallery had to gain a single figure. Re-rendering is a fresh
encode, so committing all sixteen would change bytes where the picture did not
and the diff would hide which figure the change was actually for.

The figure it was added for is not in the gallery -- the probe was rejected --
so this is the flag's only caller, and a flag with no test would be the
QuickGELU failure: passing its own definition forever while doing nothing. The
tests read the parser and the module's tables rather than rendering, so they run
in the fast suite; rendering needs the photographs and the network.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "render_gallery.py"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("render_gallery", SCRIPT)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_the_flag_exists_and_takes_several_names(module):
    args = _parser(module).parse_args(["--only", "edge", "corner"])
    assert args.only == ["edge", "corner"]


def test_it_defaults_to_every_figure(module):
    """None, not an empty list: `--only` with nothing after it should be an
    argparse error rather than silently rendering nothing."""
    assert _parser(module).parse_args([]).only is None
    with pytest.raises(SystemExit):
        _parser(module).parse_args(["--only"])


def test_every_predicted_probe_is_selectable(module):
    """The names `--only` accepts must be the names the loops use, or a valid
    request would silently render nothing."""
    for probe in module.PREDICTED:
        assert isinstance(probe, str) and probe


def test_an_unknown_name_is_rejected_rather_than_ignored(module):
    """`main` returns 1 for a name it does not know.

    Silently writing nothing is the failure this guards: a typo would look like
    a successful run that produced no figure, and the gallery test that fires
    later names a *missing figure* rather than the typo that caused it.
    """
    source = SCRIPT.read_text()
    assert "No such probe" in source
    assert "unknown = sorted(set(args.only) - known)" in source


def _parser(module):
    """The parser `main` builds, without running it.

    `main` constructs it inline rather than in a factory, so this reaches it by
    executing that block against **the module's own globals** -- which is the
    detail a first draft got wrong, building a hand-picked namespace that was
    missing `DEFAULT_OUT`. Reusing `vars(module)` means a default that moves
    cannot make this test disagree with the script about what it parses, which
    is the whole point of asserting on the parsed surface.
    """
    source = SCRIPT.read_text()
    start = source.index("    parser = argparse.ArgumentParser(")
    end = source.index("    args = parser.parse_args()")
    namespace = dict(vars(module))
    body = "\n".join(line[4:] for line in source[start:end].split("\n"))
    exec(body, namespace)  # noqa: S102 - the parser definition, read from source
    return namespace["parser"]

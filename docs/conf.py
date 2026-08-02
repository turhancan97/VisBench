"""Sphinx configuration for the VisBench documentation site.

The docstrings were written for this before it existed: numpydoc sections, 371
``:meth:``/``:func:``/``:class:`` cross-references and 462 ``#:`` attribute
comments across the package. Almost everything here is about letting that
render as it already is, rather than reshaping it.

Build:

    sphinx-build -b html -W --keep-going docs docs/_build/html

``-W`` makes warnings fatal, which is what keeps a page with a hole in it from
being published. See ``nitpicky`` below for the one class of warning that is
deliberately *not* enabled yet.
"""

from importlib.metadata import version as _version

# -- Project ----------------------------------------------------------------

project = "VisBench"
author = "Turhan Can Kargın"
copyright = "2026, Turhan Can Kargın"

# Read from the installed package rather than hardcoded. The version already
# lives in pyproject.toml and visbench/__init__.py, and uv.lock pins it a third
# time; a fourth copy here would be one more thing to forget on a release.
release = _version("visbench")
version = ".".join(release.split(".")[:2])

# -- Extensions -------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "myst_parser",
    "sphinx_copybutton",
]

# Deliberately absent: sphinx.ext.autosectionlabel. With 74 modules and ~25
# prose pages it produces duplicate-label warnings, and `-W` turns those fatal.

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- autodoc ----------------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}

# File order in this package is deliberate -- modules are written to be read
# top to bottom -- so preserve it rather than sorting alphabetically.
autodoc_member_order = "bysource"

# Parameters are documented numpydoc-style with a name and prose but no type,
# so the signature is the only place a type appears. Moving types into the
# description would duplicate nothing useful.
autodoc_typehints = "signature"
autodoc_typehints_format = "short"
autoclass_content = "both"
add_module_names = False
python_use_unqualified_type_names = True

# Inert in CI, which installs `.[all,docs]`. Present so a contributor on a core
# install can still build: all three are imported inside function bodies, never
# at module scope, so mocking them cannot reach a signature or a base class.
autodoc_mock_imports = ["open_clip", "timm", "huggingface_hub"]

# The autosummary tables in docs/api/ are navigation only; the content comes
# from explicit `automodule` blocks. Generating stubs would additionally walk
# `visbench.backbones.__all__`, which names CLIP and TimmBackbone -- served by a
# module `__getattr__` that imports the optional extra on attribute access.
autosummary_generate = False

# -- napoleon ---------------------------------------------------------------
#
# napoleon rather than the numpydoc extension: the docstrings use numpydoc
# sections (41 `Parameters`, 13 `Notes`, 6 `Returns`, 6 `Raises`) but not its
# conventions for types, and numpydoc's validation would start emitting
# warnings against 5,198 lines of docstrings straight into the `-W` gate.

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_preprocess_types = True

# -- Cross-references -------------------------------------------------------
#
# `default_role` is deliberately NOT set to "py:obj". There are ~355
# single-backtick spans in the package, most of them prose rather than object
# names, and every one would become a cross-reference attempt.

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    # docs.pytorch.org, not pytorch.org/docs -- the latter redirects and the
    # redirect has been unreliable.
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
intersphinx_timeout = 15


def _tolerate_unreachable_inventories(app):
    """Stop a network hiccup from failing a `-W` build.

    intersphinx fetches five ``objects.inv`` over the network on every cold
    build. When one is unreachable Sphinx logs a warning, and under ``-W`` that
    warning fails the build — which is what happened on the first deploy: a
    ``ConnectionResetError`` reaching ``docs.python.org`` turned a perfectly
    good site into a red X.

    Losing intersphinx degrades the site gracefully by itself: with
    ``nitpicky = False`` the affected references simply render as plain text.
    So the *warning* is the only real problem, and it carries no ``type=``,
    which means ``suppress_warnings`` cannot target it — hence a filter on the
    message.

    Deliberately narrow: it matches the inventory-fetch message only. Reference
    resolution warnings still surface, which is what matters when ``nitpicky``
    is eventually turned on.
    """
    import logging as _logging
    import sys as _sys

    class _DropUnreachableInventories(_logging.Filter):
        def filter(self, record: _logging.LogRecord) -> bool:
            if "failed to reach any of the inventories" not in record.getMessage():
                return True
            # Not silently: the site is still built, but its links to Python,
            # NumPy and PyTorch are now plain text, and a reader of the build
            # log should be able to see why.
            print(
                "NOTE: intersphinx could not fetch an inventory; external type "
                "links will render as plain text in this build.",
                file=_sys.stderr,
            )
            return False

    # Attached to the handlers rather than the logger: Sphinx emits from
    # per-module child loggers, and a parent logger's filters are not applied to
    # a propagated record — only its handlers are.
    #
    # Inserted at position 0, not appended. Sphinx implements ``-W`` as a filter
    # on the same handler that *raises* when it sees a warning, so a filter
    # added after it never runs.
    for handler in _logging.getLogger("sphinx").handlers:
        handler.filters.insert(0, _DropUnreachableInventories())


def setup(app):
    _tolerate_unreachable_inventories(app)

# Left False on purpose. Many of the 371 cross-references are bare -- `:meth:`
# `fit``, `:meth:`evaluate`` -- and resolve only from the owning class's own
# context. With nitpicky off an unresolved reference renders as literal text and
# emits no warning, so `-W` can stay on and catch the failures that matter: a
# broken toctree, a missing image, a malformed directive. Turning this on is a
# separate piece of work, together with qualifying those refs at the source.
nitpicky = False

# -- MyST -------------------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist", "attrs_inline"]
myst_heading_anchors = 3

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

# -- HTML -------------------------------------------------------------------

html_theme = "furo"
html_title = f"VisBench {release}"

# ../assets is the source of truth for the logos -- the README's
# raw.githubusercontent.com badges point straight at it -- so it is copied in at
# build time rather than duplicated under _static/.
html_static_path = ["_static", "../assets"]
html_css_files = ["custom.css"]
# Resolved against this directory, and checked before html_static_path is
# copied -- so it must name the real source file, not its destination.
html_favicon = "../assets/visbench-icon-light.svg"

# Brand palette: #3A7EAB blue, #CF4832 accent, #D1D3D4 grey, with #6FA8CC as the
# blue's dark-mode variant -- all four taken from the logo SVGs rather than
# invented.
#
# `color-brand-content` (body links) is #2F6A91, not #3A7EAB. The brand blue is
# 4.42:1 against white, just under the 4.5:1 WCAG AA threshold for body text;
# the darkened tint is 5.85:1 and visually the same colour. #3A7EAB is still the
# brand colour everywhere contrast is not the constraint. #D1D3D4 is 1.50:1 and
# is therefore a border and surface colour only, never text.
html_theme_options = {
    "light_logo": "visbench-logo-light.svg",
    "dark_logo": "visbench-logo-dark.svg",
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/turhancan97/VisBench/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#3A7EAB",
        "color-brand-content": "#2F6A91",
        "color-api-name": "#CF4832",
        "color-api-pre-name": "#3A7EAB",
        "color-background-border": "#D1D3D4",
    },
    "dark_css_variables": {
        "color-brand-primary": "#6FA8CC",
        "color-brand-content": "#6FA8CC",
        "color-api-name": "#CF4832",
        "color-api-pre-name": "#6FA8CC",
    },
}

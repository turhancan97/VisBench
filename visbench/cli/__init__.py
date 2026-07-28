"""The ``visbench`` command line — v0.2, step 5j.

Deliberately last. A CLI freezes an API into strings that users put in shell
scripts, and every one of the eight probes changed shape at least once while it
was being built — depth gained streaming, normals produced the base class the
segmentation tasks inherit, semantic segmentation added ``target_dtype``. Naming
those flags before they settled would have meant either breaking them later or
carrying the wrong ones forever.

Three commands::

    visbench list                       what backbones, probes and heads exist
    visbench run <probe> --data ...     one probe, one backbone, one dataset
    visbench cache stats | clear        inspect or drop extracted features

``run`` is a thin wrapper over :func:`visbench.run`, which was written to be
exactly that. Everything the CLI knows that the Python API does not is dataset
*construction* — which folder layout a probe expects, which loader reads its
targets — and that lives in :mod:`visbench.cli.datasets`.
"""

from visbench.cli.main import build_parser, main

__all__ = ["main", "build_parser"]

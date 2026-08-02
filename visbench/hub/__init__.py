"""Sharing trained probe heads.

Two halves. :mod:`visbench.hub.artifact` (6e-4) serialises a fitted head
together with the backbone identity that makes it meaningful, and refuses to
load it against anything else. :mod:`visbench.hub.remote` (6e-5) moves that file
to and from the Hugging Face Hub, and adds no rules of its own — a downloaded
probe goes through the same :func:`load_probe`, identity checks included.

``huggingface_hub`` is imported **inside** the functions that need it, so
importing this package, saving a probe and loading one from a local path all
work in a core install. Only push and pull require ``pip install visbench[hub]``,
and the error says so.
"""

from visbench.hub.artifact import (
    ARTIFACT_VERSION,
    IncompatibleProbe,
    load_probe,
    probe_metadata,
    save_probe,
)
from visbench.hub.remote import (
    PROBE_FILENAME,
    load_probe_from_hub,
    probe_card,
    push_probe,
)

__all__ = [
    "ARTIFACT_VERSION",
    "PROBE_FILENAME",
    "IncompatibleProbe",
    "load_probe",
    "load_probe_from_hub",
    "probe_card",
    "probe_metadata",
    "push_probe",
    "save_probe",
]

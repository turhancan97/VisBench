"""Sharing trained probe heads.

Step 6e-4 is the local half: serialising a fitted head together with the
backbone identity that makes it meaningful, and refusing to load it against
anything else. Step 6e-5 adds push/pull through ``huggingface_hub``, behind a
``[hub]`` extra — this package deliberately has no network dependency yet, so
saving and loading a probe works in a core install.
"""

from visbench.hub.artifact import (
    ARTIFACT_VERSION,
    IncompatibleProbe,
    load_probe,
    probe_metadata,
    save_probe,
)

__all__ = [
    "ARTIFACT_VERSION",
    "IncompatibleProbe",
    "load_probe",
    "probe_metadata",
    "save_probe",
]

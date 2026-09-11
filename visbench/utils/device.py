"""Device selection and batching helpers."""

import torch

__all__ = ["describe_hardware", "resolve_device"]


def resolve_device(device: str | None = None) -> str:
    """Resolve ``None`` to the best available device (cuda > mps > cpu).

    Explicit values pass through unchanged, so a caller can always force cpu.
    """
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_hardware(device: str | None = None) -> dict:
    """What a run executed on, for the result record's ``hardware`` field.

    Every other field of a record describes the *experiment*; none described
    the machine, and that absence cost real work twice in two days — once when
    a node started returning plausible wrong numbers while reporting success,
    and once when three cells disagreed across two GPU generations. Both became
    detective work because "which machine" was not written down anywhere.

    Returns ``device`` (resolved, so ``None`` never reaches a record),
    ``torch``, and ``gpu`` when there is one to name. The dict is open, for the
    reason ``task_params`` is: a future detail must not force another schema
    bump.

    **What this is not for.** It must never reach ``comparability_key``.
    Grouping on it would put every GPU in its own group and make a board with
    mixed provenance unrenderable, and the point is the opposite: two runs on
    different machines usually *are* comparable — measured, to six decimals.
    This says what produced a number, not whether two may be ranked together.

    Never raises. Provenance that can break a run is worse than none, so a
    device that cannot be queried is recorded as the plain resolved string.
    """
    resolved = resolve_device(device)
    hardware: dict = {"device": resolved, "torch": torch.__version__}
    if resolved.startswith("cuda"):
        try:
            # "cuda" means whichever device is current; "cuda:2" names one.
            index = (
                torch.cuda.current_device() if resolved == "cuda" else torch.device(resolved).index
            )
            hardware["gpu"] = torch.cuda.get_device_name(index)
        except Exception:  # pragma: no cover - a CUDA string on a box without one
            pass
    return hardware

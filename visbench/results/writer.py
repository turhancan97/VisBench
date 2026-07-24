"""Writing and reading structured result records.

Tasks never print results themselves; the run harness builds a
:class:`ResultRecord` from the returned metrics dict and writes it here.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Optional

from visbench.results.schema import ResultRecord

__all__ = ["ResultWriter", "read_records"]


class ResultWriter:
    """Append-only JSON Lines writer, one record per line.

    JSONL rather than one JSON file so concurrent runs on a cluster can append
    without clobbering each other, and so a partial file is still readable.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        """Open the results file for appending, creating parent dirs as needed."""
        raise NotImplementedError

    def write(self, record: ResultRecord) -> None:
        """Append one record, flushing so a crashed run keeps completed results."""
        raise NotImplementedError

    def __enter__(self) -> "ResultWriter":
        raise NotImplementedError

    def __exit__(self, *exc) -> None:
        raise NotImplementedError


def read_records(path: Path) -> list[ResultRecord]:
    """Load all records from a JSONL file, skipping blank lines."""
    raise NotImplementedError


def iter_records(path: Path) -> Iterator[ResultRecord]:
    """Stream records one at a time, for result files too large to hold in memory."""
    raise NotImplementedError

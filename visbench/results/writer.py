"""Writing and reading structured result records.

Tasks never print results themselves; the run harness builds a
:class:`ResultRecord` from the returned metrics dict and writes it here.
"""

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

from visbench.results.schema import ResultRecord

__all__ = ["ResultWriter", "read_records", "iter_records", "DEFAULT_RESULTS_PATH"]

#: Default results file, relative to the working directory. In .gitignore.
DEFAULT_RESULTS_PATH = Path("results/visbench.jsonl")


class ResultWriter:
    """Append-only JSON Lines writer, one record per line.

    JSONL rather than one JSON file so concurrent runs on a cluster can append
    without clobbering each other, and so a partial file is still readable.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        """Open the results file for appending, creating parent dirs as needed."""
        self.path = Path(path) if path is not None else DEFAULT_RESULTS_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered append. "a" is what makes concurrent writers safe: each
        # write positions at the current end of file, so two runs interleave
        # lines rather than overwriting each other.
        self._handle = open(self.path, "a", encoding="utf-8")

    def write(self, record: ResultRecord) -> None:
        """Append one record, flushing so a crashed run keeps completed results."""
        line = json.dumps(record.to_dict(), sort_keys=True)
        if "\n" in line:
            raise ValueError("Serialised record contains a newline; would corrupt the JSONL file")
        self._handle.write(line + "\n")
        self._handle.flush()
        # flush() only reaches the OS buffer; fsync is what survives a node
        # dying mid-run, which on a cluster is the case worth surviving.
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_records(path: Path) -> list[ResultRecord]:
    """Load all records from a JSONL file, skipping blank lines."""
    return list(iter_records(path))


def iter_records(path: Path) -> Iterator[ResultRecord]:
    """Stream records one at a time, for result files too large to hold in memory."""
    path = Path(path)
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number} is not valid JSON") from exc
            yield ResultRecord.from_dict(payload)

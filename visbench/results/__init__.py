"""Structured result logging.

Named ``results`` rather than ``logging`` to avoid shadowing the stdlib module.
"""

from visbench.results.schema import SCHEMA_VERSION, ResultRecord
from visbench.results.writer import ResultWriter, read_records

__all__ = ["ResultRecord", "ResultWriter", "read_records", "SCHEMA_VERSION"]

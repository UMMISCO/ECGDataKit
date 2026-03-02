"""Batch processing utilities for parsing multiple ECG files."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser


def _parse_single(file_path: Path) -> ECGRecord:
    """Parse a single file (top-level function for pickling in multiprocessing)."""
    return FileParser().parse(file_path)


def parse_batch(
    files: list[Path | str],
    max_workers: int | None = None,
) -> Iterator[ECGRecord]:
    """Parse multiple ECG files in parallel.

    Parameters
    ----------
    files : list[Path | str]
        Paths to ECG files.
    max_workers : int | None
        Maximum number of worker processes. Defaults to CPU count.

    Yields
    ------
    ECGRecord
        Parsed records in the same order as the input files.
    """
    paths = [Path(f) for f in files]
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        yield from pool.map(_parse_single, paths)

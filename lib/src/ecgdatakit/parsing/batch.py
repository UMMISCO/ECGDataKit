"""Batch processing utilities for parsing multiple ECG files."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser

_ON_ERROR = ("raise", "warn", "ignore")


class BatchParseWarning(UserWarning):
    """A file could not be parsed during :func:`parse_batch`."""


def _parse_single(
    file_path: Path, auto_scale: bool, units: str,
) -> tuple[ECGRecord | None, list[tuple[type[Warning], str]], BaseException | None]:
    """Parse one file in a worker and return the record, its warnings and any error."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            record = FileParser().parse(file_path, auto_scale=auto_scale, units=units)
            error = None
        except Exception as e:
            record, error = None, e
    return record, [(w.category, str(w.message)) for w in caught], error


def parse_batch(
    files: list[Path | str],
    max_workers: int | None = None,
    auto_scale: bool = True,
    units: str = "mV",
    on_error: str = "raise",
) -> Iterator[ECGRecord]:
    """Parse multiple ECG files in parallel.

    Parameters
    ----------
    files : list[Path | str]
        Paths to ECG files.
    max_workers : int | None
        Maximum number of worker processes. Defaults to CPU count.
    auto_scale, units
        Passed to :meth:`FileParser.parse`.
    on_error : str
        What to do when a file fails: ``"raise"`` (default) re-raises the
        error, ``"warn"`` emits a :class:`BatchParseWarning` naming the file
        and continues, ``"ignore"`` skips it silently. Skipped files yield
        nothing, use ``record.raw_metadata["filepath"]`` to match records.

    Yields
    ------
    ECGRecord
        Parsed records in the same order as the input files.
    """
    if on_error not in _ON_ERROR:
        raise ValueError(f"on_error must be one of {_ON_ERROR}, got {on_error!r}")
    paths = [Path(f) for f in files]
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_parse_single, p, auto_scale, units) for p in paths]
        for path, future in zip(paths, futures):
            record, caught, error = future.result()
            for category, message in caught:
                warnings.warn(f"{path.name}: {message}", category, stacklevel=2)
            if error is not None:
                if on_error == "raise":
                    raise error
                if on_error == "warn":
                    warnings.warn(
                        f"{path.name}: {type(error).__name__}: {error}",
                        BatchParseWarning,
                        stacklevel=2,
                    )
                continue
            yield record

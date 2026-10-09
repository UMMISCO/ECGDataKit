"""Batch processing utilities for parsing multiple ECG files."""

from __future__ import annotations

import os
import warnings
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser

_ON_ERROR = ("raise", "warn", "ignore")
_EXECUTORS = ("process", "serial")

_Result = tuple[ECGRecord | None, list[tuple[type[Warning], str]], BaseException | None]


class BatchParseWarning(UserWarning):
    """A file could not be parsed during :func:`parse_batch`."""


def _parse_single(file_path: Path, auto_scale: bool, units: str) -> _Result:
    """Parse one file and return the record, its warnings and any error."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            record = FileParser().parse(file_path, auto_scale=auto_scale, units=units)
            error = None
        except Exception as e:
            record, error = None, e
    return record, [(w.category, str(w.message)) for w in caught], error


def _run_isolated(path: Path, auto_scale: bool, units: str) -> _Result:
    """Parse *path* alone in a fresh worker, to tell whether it kills its worker."""
    with ProcessPoolExecutor(max_workers=1) as pool:
        try:
            return pool.submit(_parse_single, path, auto_scale, units).result()
        except BrokenProcessPool as e:
            return None, [], e


def _process_results(
    paths: list[Path], max_workers: int | None, auto_scale: bool, units: str,
) -> Iterator[tuple[Path, _Result]]:
    """Yield (path, result) in input order with a bounded number of pending files."""
    pool = ProcessPoolExecutor(max_workers=max_workers)
    window = 2 * (max_workers or os.cpu_count() or 1)  # records waiting in memory at most
    queue: deque[tuple[Path, Future]] = deque()
    todo = iter(paths)

    def refill() -> None:
        for path in todo:
            queue.append((path, pool.submit(_parse_single, path, auto_scale, units)))
            if len(queue) >= window:
                break

    try:
        refill()
        while queue:
            path, future = queue.popleft()
            try:
                result = future.result()
            except BrokenProcessPool:
                # A worker died (e.g. killed when out of memory). Rerun this
                # file alone to know if it is the cause, then restart the
                # pool for the files still pending.
                pending = [p for p, _ in queue]
                queue.clear()
                pool.shutdown(wait=True, cancel_futures=True)
                result = _run_isolated(path, auto_scale, units)
                pool = ProcessPoolExecutor(max_workers=max_workers)
                todo = iter(pending + list(todo))
            del future  # the executor keeps no reference, so the record can be freed
            refill()
            yield path, result
            del result
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def parse_batch(
    files: Iterable[Path | str],
    max_workers: int | None = None,
    auto_scale: bool = True,
    units: str = "mV",
    on_error: str = "raise",
    executor: str = "process",
) -> Iterator[ECGRecord]:
    """Parse multiple ECG files, in parallel worker processes by default.

    Records are yielded one at a time in input order. At most
    ``2 * max_workers`` parsed records wait in memory, and a record is
    freed as soon as the caller drops it. Leaving the loop early (``break``
    or an error) cancels the files not started yet.

    Each record is sent back from its worker process by pickling, which
    costs time and a transient copy of the samples. For long recordings
    (24 h Holter) this transfer is slower than parsing itself; use
    ``executor="serial"`` there, or parse with :class:`FileParser` in a loop.

    With ``executor="process"`` the calling script must be importable
    without side effects: on macOS and Windows worker processes re-import
    the main module, so call ``parse_batch`` under
    ``if __name__ == "__main__":``.

    Parameters
    ----------
    files : iterable of Path | str
        Paths to ECG files.
    max_workers : int | None
        Maximum number of worker processes. Defaults to CPU count.
        Ignored with ``executor="serial"``.
    auto_scale, units
        Passed to :meth:`FileParser.parse`.
    on_error : str
        What to do when a file fails: ``"raise"`` (default) re-raises the
        error, ``"warn"`` emits a :class:`BatchParseWarning` naming the file
        and continues, ``"ignore"`` skips it silently. Skipped files yield
        nothing, use ``record.raw_metadata["filepath"]`` to match records.
        A file whose worker process dies (``BrokenProcessPool``, e.g. out of
        memory) counts as a failed file; the other files are still parsed.
        An ``ImportError`` (missing optional dependency, e.g. ``pydicom``
        for DICOM files) is always raised, whatever *on_error* says.
    executor : str
        ``"process"`` (default) parses in a process pool, ``"serial"``
        parses in the calling process, one file after the other.

    Yields
    ------
    ECGRecord
        Parsed records in the same order as the input files.
    """
    if on_error not in _ON_ERROR:
        raise ValueError(f"on_error must be one of {_ON_ERROR}, got {on_error!r}")
    if executor not in _EXECUTORS:
        raise ValueError(f"executor must be one of {_EXECUTORS}, got {executor!r}")
    paths = [Path(f) for f in files]
    if executor == "serial":
        results: Iterator[tuple[Path, _Result]] = (
            (p, _parse_single(p, auto_scale, units)) for p in paths
        )
    else:
        results = _process_results(paths, max_workers, auto_scale, units)
    try:
        for path, (record, caught, error) in results:
            for category, message in caught:
                warnings.warn(f"{path.name}: {message}", category, stacklevel=2)
            if error is not None:
                # A missing optional dependency is an environment problem,
                # not a bad file: stop at once whatever on_error says
                if on_error == "raise" or isinstance(error, ImportError):
                    raise error
                if on_error == "warn":
                    warnings.warn(
                        f"{path.name}: {type(error).__name__}: {error}",
                        BatchParseWarning,
                        stacklevel=2,
                    )
                continue
            yield record
            del record
    finally:
        close = getattr(results, "close", None)
        if close is not None:
            close()

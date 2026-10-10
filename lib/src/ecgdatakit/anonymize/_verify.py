"""Check of each anonymized file before it is published.

Both files are read with ecgdatakit. The anonymized file must be read by
the same parser, hold the same samples bit for bit (leads, median beats,
enhanced leads), have only the pseudonym in the patient name and ID fields,
and give every other field the raw value with the identity text replaced.
"""

from __future__ import annotations

import threading
import warnings
from pathlib import Path

import numpy as np

from ecgdatakit.anonymize._identity import Replacer
from ecgdatakit.anonymize.formats._base import is_data
from ecgdatakit.parsing.parser import FileParser

_PATIENT_FIELDS = ("patient_id", "first_name", "last_name")


class VerificationError(Exception):
    """The anonymized file does not match its raw file."""


# Warning filters are global to the process: parses are not run in parallel
_PARSE_LOCK = threading.Lock()


def _parse(path: Path):
    with _PARSE_LOCK, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FileParser().parse(path, auto_scale=False)


def _replace_all(value, replacer: Replacer):
    if isinstance(value, str):
        return replacer.replace_value(value, is_data(value))
    if isinstance(value, list):
        return [_replace_all(v, replacer) for v in value]
    if isinstance(value, dict):
        return {k: _replace_all(v, replacer) for k, v in value.items()}
    return value


def _same_signals(a: list, b: list, what: str) -> None:
    if len(a) != len(b):
        raise VerificationError(f"{what}: {len(a)} in the raw file, {len(b)} after")
    for la, lb in zip(a, b):
        if la.label != lb.label or la.sampling_rate != lb.sampling_rate:
            raise VerificationError(f"{what}: lead {la.label} differs after anonymization")
        if not np.array_equal(np.asarray(la.samples), np.asarray(lb.samples), equal_nan=True):
            raise VerificationError(f"{what}: samples of lead {la.label} differ after anonymization")


def verify(raw: Path, anonymized: Path, replacer: Replacer, patient_code: str,
           kept: list[str] | None = None) -> None:
    """Raise VerificationError unless *anonymized* holds the same ECG as *raw*.

    Same parser, same samples bit for bit, patient name and ID fields hold
    only the pseudonym, and every other field equals the raw value with the
    identity text replaced.
    """
    try:
        before = _parse(raw)
    except Exception as e:  # noqa: BLE001 - reported in the catalog
        raise VerificationError(f"raw file cannot be read: {type(e).__name__}: {e}") from e
    try:
        after = _parse(anonymized)
    except Exception as e:  # noqa: BLE001
        raise VerificationError(f"anonymized file cannot be read: {type(e).__name__}: {e}") from e
    if before.file_format.parser != after.file_format.parser:
        raise VerificationError(
            f"read by {after.file_format.parser} after anonymization, "
            f"{before.file_format.parser} before")
    _same_signals(before.leads, after.leads, "leads")
    _same_signals(before.median_beats, after.median_beats, "median beats")
    _same_signals(before.leads_enhanced, after.leads_enhanced, "enhanced leads")

    old = before.to_dict(include_samples=False)
    new = after.to_dict(include_samples=False)
    kept = set(kept or [])
    for key in _PATIENT_FIELDS:
        value = str(new["patient"].pop(key) or "")
        old["patient"].pop(key)
        if key == "patient_id" and value.strip() in kept:
            continue
        if value and set(value.replace("^", " ").split()) - {patient_code}:
            raise VerificationError(f"patient {key} still holds other text than the pseudonym")
    # Values set by the library, not read from the file, must not change
    library_set = [("source_format",), ("file_format", "name"), ("file_format", "parser")]
    for path in library_set:
        before_value, after_value = _pop(old, path), _pop(new, path)
        if before_value != after_value:
            raise VerificationError(f"{'.'.join(path)} changed after anonymization")
    expected = _replace_all(old, replacer)
    if expected != new:
        diff = _first_difference(expected, new)
        raise VerificationError(f"field changed beyond the identity values: {diff}")


def _pop(data: dict, path: tuple[str, ...]):
    for key in path[:-1]:
        data = data.get(key, {})
    return data.pop(path[-1], None)


def _first_difference(a, b, path: str = "") -> str:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            if a.get(key) != b.get(key):
                return _first_difference(a.get(key), b.get(key), f"{path}.{key}")
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return _first_difference(x, y, f"{path}[{i}]")
    return path.lstrip(".") or "(root)"

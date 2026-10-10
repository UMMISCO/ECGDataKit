"""Pseudonym codes.

Patient codes are random 12-character shortuuid strings (57-letter
alphabet, about 71 bits). ECG codes are random UUIDs (version 4), written
as is in place of the ECG ID; DICOM, which only accepts digits and dots,
gets the same UUID in its standard UID form (``2.25.`` followed by the
UUID as an integer). A code already used in the dataset is drawn again.
"""

from __future__ import annotations

import uuid

CODE_LENGTH = 12


def _shortuuid():
    try:
        import shortuuid
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Anonymization needs shortuuid: pip install 'ecgdatakit[anonymize]'"
        ) from e
    return shortuuid


def new_code(used: set[str]) -> str:
    """A random code not in *used* (the code is added to *used*)."""
    generator = _shortuuid().ShortUUID()
    for _ in range(1000):
        code = generator.random(length=CODE_LENGTH)
        if code not in used:
            used.add(code)
            return code
    raise RuntimeError("Could not draw a unique pseudonym code")  # pragma: no cover


def new_ecg_code(used: set[str]) -> str:
    """A random UUID not in *used* (the UUID is added to *used*)."""
    for _ in range(1000):
        code = str(uuid.uuid4())
        if code not in used:
            used.add(code)
            return code
    raise RuntimeError("Could not draw a unique ECG code")  # pragma: no cover


def code_dicom_uid(code: str) -> str:
    """DICOM UID form of the ECG code (2.25 root + the UUID as an integer)."""
    return f"2.25.{uuid.UUID(code).int}"

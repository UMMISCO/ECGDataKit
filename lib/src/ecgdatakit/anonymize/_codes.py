"""Pseudonym codes.

Codes are random 12-character shortuuid strings (57-letter alphabet, about
71 bits), drawn again if already used in the dataset. ECG IDs that must keep
their type get a value derived from the ECG code: a UUID (HL7 aECG) or a
DICOM UID under the 2.25 root.
"""

from __future__ import annotations

import uuid

CODE_LENGTH = 12
_NAMESPACE = uuid.UUID("6f1c3c8e-2d4b-5a7e-9c1d-8b3e4f5a6b7c")


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


def code_uuid(code: str) -> str:
    """UUID written in place of an ECG ID that must stay a UUID."""
    return str(uuid.uuid5(_NAMESPACE, code))


def code_dicom_uid(code: str) -> str:
    """DICOM UID (2.25 + UUID as an integer) written in place of an ECG UID."""
    return f"2.25.{uuid.uuid5(_NAMESPACE, code).int}"

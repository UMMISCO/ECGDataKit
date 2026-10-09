"""Decoding of waveform samples stored as text in XML exports."""

from __future__ import annotations

import base64
import binascii
import re

import numpy as np

_INT_LIST = re.compile(r"[+-]?\d+(?:[\s,;]+[+-]?\d+)*[\s,;]*")
_SEPARATOR = re.compile(r"[\s,;]")
_WHITESPACE = re.compile(r"\s+")


def decode_int16_text(text: str) -> tuple[np.ndarray, str]:
    """Decode waveform text holding int16 samples.

    Two encodings are recognised: a list of integers separated by commas,
    semicolons or whitespace, and Base64 of little-endian int16 values.
    Integer lists are detected first (a string of digits with separators is
    never valid sample data in Base64 form), so ``"1000,2000"`` is not
    misread as Base64. A single signed integer of at most 6 digits (the
    int16 range) is one sample; a longer run of digits is read as Base64.

    Returns ``(samples, encoding)`` where *encoding* is ``"int_list"`` or
    ``"base64_int16le"``. Raises ``ValueError`` when the text is neither.
    """
    stripped = text.strip()
    if not stripped:
        return np.array([], dtype=np.float64), ""

    if _INT_LIST.fullmatch(stripped) and (
        _SEPARATOR.search(stripped) or stripped[0] in "+-" or len(stripped) <= 6
    ):
        values = [int(v) for v in re.split(r"[\s,;]+", stripped) if v]
        return np.array(values, dtype=np.float64), "int_list"

    compact = _WHITESPACE.sub("", stripped)
    try:
        raw = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"invalid Base64 waveform data: {e}") from e
    if len(raw) % 2:
        raise ValueError(
            f"Base64 waveform data decodes to {len(raw)} bytes, "
            "not a whole number of int16 samples"
        )
    return np.frombuffer(raw, dtype="<i2").astype(np.float64), "base64_int16le"

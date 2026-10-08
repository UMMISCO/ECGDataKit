"""XLI decompression codec for Sierra ECG waveform data.

The XLI format stores ECG leads as individually compressed chunks.  Each chunk
has an 8-byte header (4 bytes compressed size, 2 bytes unused, 2 bytes start
value) followed by LZW-compressed delta-encoded 16-bit samples.

Decoding pipeline per chunk:
  1. LZW decompress the chunk body (10-bit codes).
  2. Unpack the bytes into signed 16-bit delta codes (high bytes in the first
     half of the buffer, low bytes in the second half).
  3. Reverse the second-order delta encoding to recover the sample values.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ecgdatakit.exceptions import CorruptedFileError
from ecgdatakit.parsing.codecs.lzw import LzwDecoder

_HEADER_SIZE = 8


def xli_decode(data: bytes, labels: list[str] | None = None) -> list[npt.NDArray[np.int32]]:
    """Decode XLI-compressed waveform data into per-lead sample arrays.

    Parameters
    ----------
    data : bytes
        Raw XLI-compressed waveform blob (typically Base64-decoded).
    labels : list[str], optional
        Lead labels. Unused, kept for backward compatibility.

    Returns
    -------
    list[NDArray[int32]]
        One array of samples per chunk, in file order.

    Raises
    ------
    CorruptedFileError
        If a chunk header or chunk body is truncated or invalid.
    """
    samples: list[npt.NDArray[np.int32]] = []
    offset = 0
    while offset < len(data):
        if len(data) - offset < _HEADER_SIZE:
            raise CorruptedFileError(
                f"XLI chunk {len(samples) + 1}: truncated header "
                f"({len(data) - offset} of {_HEADER_SIZE} bytes)"
            )
        header = data[offset: offset + _HEADER_SIZE]
        offset += _HEADER_SIZE

        size = int.from_bytes(header[0:4], byteorder="little", signed=True)
        start = int.from_bytes(header[6:8], byteorder="little", signed=True)
        if size <= 0 or size > len(data) - offset:
            raise CorruptedFileError(
                f"XLI chunk {len(samples) + 1}: invalid size {size} "
                f"({len(data) - offset} bytes left)"
            )
        chunk = data[offset: offset + size]
        offset += size

        buffer = LzwDecoder(chunk, bits=10).read_all()
        if len(buffer) < 4:
            raise CorruptedFileError(
                f"XLI chunk {len(samples) + 1}: only {len(buffer)} bytes after decompression"
            )
        samples.append(xli_decode_deltas(buffer, start))

    return samples


def xli_decode_deltas(buffer: bytes | list[int], first: int) -> npt.NDArray[np.int32]:
    """Reverse the delta encoding to recover original sample values."""
    deltas = xli_unpack(buffer).tolist()
    if len(deltas) < 2:
        raise CorruptedFileError("XLI chunk holds fewer than two delta codes")
    x = deltas[0]
    y = deltas[1]
    last = first
    for i in range(2, len(deltas)):
        z = (y + y) - x - last
        last = deltas[i] - 64
        deltas[i] = z
        x = y
        y = z
    return np.asarray(deltas, dtype=np.int32)


def xli_unpack(buffer: bytes | list[int]) -> npt.NDArray[np.int16]:
    """Unpack a byte buffer into signed 16-bit integers.

    The first half of *buffer* holds the high bytes and the second half the
    low bytes. An odd-length buffer is padded with a zero low byte.
    """
    raw = np.frombuffer(bytes(buffer), dtype=np.uint8)
    if len(raw) % 2 == 1:
        raw = np.append(raw, np.uint8(0))
    half = len(raw) // 2
    words = (raw[:half].astype(np.uint16) << 8) | raw[half:].astype(np.uint16)
    return words.view(np.int16)

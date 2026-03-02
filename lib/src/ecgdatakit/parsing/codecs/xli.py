"""XLI decompression codec for Sierra ECG waveform data.

The XLI format stores ECG leads as individually compressed chunks.  Each chunk
has an 8-byte header (4 bytes compressed size, 2 bytes unused, 2 bytes start
value) followed by LZW-compressed delta-encoded 16-bit samples.

Decoding pipeline per chunk:
  1. LZW decompress the chunk body.
  2. Unpack the raw bytes into signed 16-bit integers.
  3. Reverse the delta encoding to recover the original sample values.
"""

from typing import List

import numpy as np
import numpy.typing as npt

from ecgdatakit.parsing.codecs.lzw import LzwDecoder


def xli_decode(data: bytes, labels: list[str]) -> list[npt.NDArray[np.int16]]:
    """Decode XLI-compressed waveform data into per-lead sample arrays.

    Parameters
    ----------
    data : bytes
        Raw XLI-compressed waveform blob (typically Base64-decoded).
    labels : list[str]
        Lead labels (used only to determine expected lead count).

    Returns
    -------
    list[NDArray[int16]]
        One array of samples per lead.
    """
    samples: list[npt.NDArray[np.int16]] = []
    offset = 0
    while offset < len(data):
        header = data[offset: offset + 8]
        offset += 8

        size = int.from_bytes(header[0:4], byteorder="little", signed=True)
        start = int.from_bytes(header[6:], byteorder="little", signed=True)
        chunk = data[offset: offset + size]
        offset += size

        decoder = LzwDecoder(chunk, bits=10)

        buffer: list[int] = []
        while -1 != (b := decoder.read()):
            buffer.append(b & 0xFF)

        if len(buffer) % 2 == 1:
            buffer.append(0)

        deltas = xli_decode_deltas(buffer, start)
        samples.append(deltas)

    return samples


def xli_decode_deltas(buffer: list[int], first: int) -> npt.NDArray[np.int16]:
    """Reverse the delta encoding to recover original sample values."""
    deltas = xli_unpack(buffer)
    x = int(deltas[0])
    y = int(deltas[1])
    last = first
    for i in range(2, len(deltas)):
        z = (y + y) - x - last
        last = int(deltas[i]) - 64
        deltas[i] = z
        x = y
        y = z
    return deltas


def xli_unpack(buffer: list[int]) -> npt.NDArray[np.int16]:
    """Unpack a byte buffer into signed 16-bit integers."""
    half = len(buffer) // 2
    unpacked: npt.NDArray[np.int16] = np.zeros(half, dtype=np.int16)
    for i in range(half):
        joined_bytes = (((buffer[i] << 8) | buffer[half + i]) << 16) >> 16
        unpacked[i] = np.array(joined_bytes).astype(np.int16)
    return unpacked

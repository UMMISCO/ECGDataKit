"""LZW decompression codec for Sierra ECG waveform data.

The LZW (Lempel-Ziv-Welch) algorithm is a dictionary-based compression scheme.
This implementation reads variable-width codepoints from a bitstream and builds
a string table on the fly, producing the original uncompressed byte sequence.
"""

from array import array
from typing import MutableSequence


class LzwDecoder:
    """Decodes LZW-compressed byte streams.

    Parameters
    ----------
    buffer : bytes
        The compressed data.
    bits : int
        Codepoint bit width (e.g. 10 for Sierra ECG XLI compression).
    """

    def __init__(self, buffer: bytes, bits: int) -> None:
        self.buffer = buffer
        self.bits = bits
        self.max_code = (1 << bits) - 2
        self.offset = 0

        self.bit_count = 0
        self.bit_buffer = 0

        self.previous: MutableSequence[int] = array("B", [])
        self.next_code = 256
        self.strings: dict[int, MutableSequence[int]] = {
            code: array("B", [code]) for code in range(256)
        }

        self.current: MutableSequence[int] | None = None
        self.position = 0

    def read(self) -> int:
        """Read the next decompressed byte. Returns -1 at end of stream."""
        if self.current is None or self.position == len(self.current):
            self.current = self._read_next_string()
            self.position = 0

        if len(self.current) > 0:
            byte = self.current[self.position] & 0xFF
            self.position += 1
            return byte

        return -1

    def read_bytes(self, count: int) -> MutableSequence[int]:
        """Read ``count`` decompressed bytes."""
        return array("B", [self.read() for _ in range(count)])

    def _read_next_string(self) -> MutableSequence[int]:
        code = self._read_codepoint()
        if 0 <= code <= self.max_code:
            data: MutableSequence[int]
            if code not in self.strings:
                data = self.previous[:]
                data.append(self.previous[0])
                self.strings[code] = data
            else:
                data = self.strings[code]

            if len(self.previous) > 0 and self.next_code <= self.max_code:
                next_data = self.previous[:]
                next_data.append(data[0])
                self.strings[self.next_code] = next_data
                self.next_code += 1

            self.previous = data
            return data

        return array("B", [])

    def _read_codepoint(self) -> int:
        while self.bit_count <= 24:
            if self.offset < len(self.buffer):
                next_byte = self.buffer[self.offset]
                self.offset += 1
                self.bit_buffer |= ((next_byte & 0xFF) << (24 - self.bit_count)) & 0xFFFFFFFF
                self.bit_count += 8
            elif self.bit_count < self.bits:
                return -1
            else:
                break

        code = (self.bit_buffer >> (32 - self.bits)) & 0x0000FFFF
        self.bit_buffer = ((self.bit_buffer & 0xFFFFFFFF) << self.bits) & 0xFFFFFFFF
        self.bit_count -= self.bits

        return code

"""Tests for LZW and XLI codecs."""

from __future__ import annotations

import numpy as np
import pytest

from ecgdatakit.parsing.codecs.lzw import LzwDecoder
from ecgdatakit.parsing.codecs.xli import xli_decode, xli_decode_deltas, xli_unpack


class TestLzwDecoder:
    def test_empty_buffer(self):
        decoder = LzwDecoder(b"", bits=10)
        assert decoder.read() == -1

    def test_single_byte_roundtrip(self):
        """A single raw byte (codepoint < 256) should decode to itself."""
        # Encode codepoint 65 ('A') in 10 bits, left-aligned in 2 bytes
        # 65 = 0b0001000001, 10-bit left-aligned in 16 bits = 0b0001000001_000000
        code = 65
        bits = 10
        val = code << (16 - bits)
        buf = val.to_bytes(2, "big")
        decoder = LzwDecoder(buf, bits=bits)
        assert decoder.read() == 65

    def test_read_bytes(self):
        # Encode two codepoints: 72 ('H') and 105 ('i')
        bits = 10
        cp1 = 72 << (32 - bits)
        cp2 = 105 << (32 - 2 * bits)
        combined = cp1 | cp2
        buf = combined.to_bytes(4, "big")
        decoder = LzwDecoder(buf, bits=bits)
        result = decoder.read_bytes(2)
        assert list(result) == [72, 105]


class TestXliUnpack:
    def test_basic_unpack(self):
        # 4 bytes -> 2 int16 values
        buffer = [0x00, 0x01, 0x64, 0xC8]
        result = xli_unpack(buffer)
        assert len(result) == 2
        assert result.dtype == np.int16


class TestXliDecodeDeltasShape:
    def test_output_shape(self):
        buffer = [0] * 20
        result = xli_decode_deltas(buffer, first=0)
        assert len(result) == 10
        assert result.dtype == np.int16

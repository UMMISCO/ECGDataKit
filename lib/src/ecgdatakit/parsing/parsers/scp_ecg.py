"""SCP-ECG (Standard Communications Protocol for computer-assisted ECG) parser.

Reference: EN 1064 / ISO 11073-91064. A record is a 6-byte preamble (CRC,
record size) followed by sections, each with a 16-byte header (CRC, ID,
length, section version, protocol version, reserved). Section 0 lists the
other sections. All integers are little-endian.
"""

from __future__ import annotations

import binascii
import math
import struct
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ecgdatakit.exceptions import (
    ChecksumWarning,
    CorruptedFileError,
    UnsupportedFormatError,
)
from ecgdatakit.models import (
    ECGRecord,
    FileFormatInfo,
    Interpretation,
    Lead,
    derive_is_raw,
)
from ecgdatakit.parsing.helpers import (
    decode_text,
    fill_signal_summary,
    normalize_lead_label,
    unique_labels,
)
from ecgdatakit.parsing.parser import Parser

_SECTION_HEADER = 16
_UNDEFINED = 29999
_DEFAULT_HUFFMAN = 19999

# Section 3 lead identification codes
_LEAD_NAMES = [
    "Unspecified", "I", "II", "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V2R", "V3R", "V4R", "V5R", "V6R", "V7R", "X", "Y", "Z", "CC5",
    "CM5", "LA", "RA", "LL", "fI", "fE", "fC", "fA", "fM", "fF",
    "fH", "I-cal", "II-cal", "V1-cal", "V2-cal", "V3-cal", "V4-cal", "V5-cal", "V6-cal", "V7-cal",
    "V2R-cal", "V3R-cal", "V4R-cal", "V5R-cal", "V6R-cal", "V7R-cal", "X-cal", "Y-cal", "Z-cal", "CC5-cal",
    "CM5-cal", "LA-cal", "RA-cal", "LL-cal", "fI-cal", "fE-cal", "fC-cal", "fA-cal", "fM-cal", "fF-cal",
    "fH-cal", "III", "aVR", "aVL", "aVF", "-aVR", "V8", "V9", "V8R", "V9R",
    "D", "A", "J", "Defib", "ExtPace", "A1", "A2", "A3", "A4", "V8-cal",
    "V9-cal", "V8R-cal", "V9R-cal", "D-cal", "A-cal", "J-cal",
]

_MANUFACTURERS = [
    "", "Burdick", "Cambridge", "Compumed", "Datamed", "Fukuda", "Hewlett-Packard",
    "Marquette Electronics", "Mortara Instruments", "Nihon Kohden", "Okin", "Quinton",
    "Siemens", "Spacelabs", "Telemed", "Hellige", "ESA-OTE", "Schiller",
    "Picker-Schwarzer", "et medical devices", "Zwönitz",
]

# Section 1 tag 14 byte 16 (language support code) to Python codec
_LANGUAGE_CODECS = {
    0x03: "iso8859_2", 0x0B: "iso8859_4", 0x13: "iso8859_5", 0x1B: "iso8859_6",
    0x23: "iso8859_7", 0x2B: "iso8859_8", 0x33: "iso8859_11", 0x3B: "iso8859_15",
    0x07: "utf-16-le", 0x0F: "shift_jis", 0x17: "euc_jp", 0x1F: "euc_jp",
    0x27: "gb2312", 0x2F: "cp949", 0x37: "utf-8",
}
# ISO 10646 (code 0x07) is a 2-byte encoding: text ends at a 2-byte NUL.
# Without a byte order mark it is read little-endian, like every SCP field.
_WIDE_CODECS = {"utf-16-le"}

_RACE = {1: "Caucasian", 2: "Black", 3: "Oriental"}
_SEX = {1: "M", 2: "F"}
_AGE_UNITS = {1: 1.0, 2: 1 / 12, 3: 7 / 365.25, 4: 1 / 365.25, 5: 1 / 8766}
_HEIGHT_TO_CM = {1: 1.0, 2: 2.54, 3: 0.1}
_WEIGHT_TO_KG = {1: 1.0, 2: 0.001, 3: 0.45359237, 4: 0.028349523125}
_ELECTRODES_12 = {
    1: "standard 12-lead", 2: "Mason-Likar, individual electrodes",
    3: "Mason-Likar, one pad", 4: "all electrodes on one pad",
    5: "12-lead derived from Frank XYZ", 6: "non-standard",
}
_ELECTRODES_XYZ = {
    1: "Frank", 2: "McFee-Parungao", 3: "cube", 4: "bipolar uncorrected",
    5: "pseudo-orthogonal", 6: "derived from 12-lead",
}
_SPIKE_TYPES = {1: "no trigger", 2: "ventricular", 3: "atrial"}
_SPIKE_SOURCES = {1: "internal", 2: "external"}
_PACEMAKER_HISTORY_CODE = 42

# Fixed value sizes of Section 1 tags, used to recover from wrong tag lengths
_FIXED_TAG_SIZES = {
    4: 3, 5: 4, 6: 3, 7: 3, 8: 1, 9: 1, 11: 2, 12: 2, 24: 1,
    25: 4, 26: 3, 27: 2, 28: 2, 29: 1, 33: 2,
}

# Section 10 per-lead measurements, in file order (ms, uV, uV/s)
_LEAD_MEASUREMENTS = [
    "p_duration", "pr_interval", "qrs_duration", "qt_interval", "q_duration",
    "r_duration", "s_duration", "r_prime_duration", "s_prime_duration",
    "q_amplitude", "r_amplitude", "s_amplitude", "r_prime_amplitude",
    "s_prime_amplitude", "j_point_amplitude", "p_plus_amplitude",
    "p_minus_amplitude", "t_plus_amplitude", "t_minus_amplitude", "st_slope",
    "p_morphology", "t_morphology", "iso_qrs_onset", "iso_qrs_end",
    "intrinsicoid_deflection", "quality", "st_j20", "st_j60", "st_j80",
    "st_rr16", "st_rr8",
]


def _u16(data: bytes, pos: int) -> int:
    return struct.unpack_from("<H", data, pos)[0]


def _i16(data: bytes, pos: int) -> int:
    return struct.unpack_from("<h", data, pos)[0]


def _u32(data: bytes, pos: int) -> int:
    return struct.unpack_from("<I", data, pos)[0]


def _crc(data: bytes) -> int:
    """CRC-CCITT (polynomial 0x1021, initial value 0xFFFF)."""
    return binascii.crc_hqx(data, 0xFFFF)


def _date(raw: bytes) -> datetime | None:
    if len(raw) < 4:
        return None
    year, month, day = _u16(raw, 0), raw[2], raw[3]
    try:
        return datetime(year, month, day) if year else None
    except ValueError:
        return None


def _time(raw: bytes) -> tuple[int, int, int] | None:
    if len(raw) < 3 or raw[0] > 23 or raw[1] > 59 or raw[2] > 59:
        return None
    return raw[0], raw[1], raw[2]


def _decode(value: bytes, encoding: str | None) -> str:
    """Decode a NUL-terminated SCP text in the Section 1 language encoding."""
    if encoding in _WIDE_CODECS:
        if value[:2] in (b"\xff\xfe", b"\xfe\xff"):
            encoding = "utf-16"  # the byte order mark decides
        end = next((i for i in range(0, len(value) - 1, 2) if value[i:i + 2] == b"\x00\x00"), None)
        value = value[:end] if end is not None else value[: len(value) // 2 * 2]
    else:
        value = value.split(b"\x00", 1)[0]
    return decode_text(value, encoding).strip()


def _lead_label(code: int) -> str:
    if code < len(_LEAD_NAMES):
        return normalize_lead_label(_LEAD_NAMES[code])
    return f"Lead{code}"


# ---------------------------------------------------------------------------
# Huffman decoding (Section 2)
# ---------------------------------------------------------------------------


@dataclass
class _HuffmanCode:
    prefix_bits: int
    total_bits: int
    mode: int  # 1 = data code, 0 = switch to table `value`
    value: int
    code: int  # prefix bits, first transmitted bit is the most significant


@dataclass
class _HuffmanTable:
    codes: dict[tuple[int, int], _HuffmanCode]
    max_prefix: int


def _make_table(codes: list[_HuffmanCode]) -> _HuffmanTable:
    return _HuffmanTable(
        codes={(c.prefix_bits, c.code): c for c in codes},
        max_prefix=max((c.prefix_bits for c in codes), default=0),
    )


def _default_huffman_table() -> _HuffmanTable:
    codes = [_HuffmanCode(1, 1, 1, 0, 0b0)]
    for magnitude in range(1, 9):
        ones = (1 << magnitude) - 1  # magnitude leading ones
        for sign, value in ((0, magnitude), (1, -magnitude)):
            bits = magnitude + 2
            codes.append(_HuffmanCode(bits, bits, 1, value, (ones << 2) | sign))
    # 1111111110 + 8-bit value, 1111111111 + 16-bit value
    codes.append(_HuffmanCode(10, 18, 1, 0, 0b1111111110))
    codes.append(_HuffmanCode(10, 26, 1, 0, 0b1111111111))
    return _make_table(codes)


_DEFAULT_TABLE = _default_huffman_table()


def _parse_huffman_tables(body: bytes) -> list[_HuffmanTable]:
    """Parse Section 2. Returns the default table for code 19999."""
    if len(body) < 2:
        raise CorruptedFileError("SCP Section 2 is too short")
    count = _u16(body, 0)
    if count == _DEFAULT_HUFFMAN:
        return [_DEFAULT_TABLE]
    if count == 0:
        raise CorruptedFileError("SCP Section 2 declares no Huffman table")
    pos = 2
    tables = []
    for t in range(count):
        if pos + 2 > len(body):
            raise CorruptedFileError(f"SCP Section 2 truncated in Huffman table {t + 1}")
        ncodes = _u16(body, pos)
        pos += 2
        if pos + 9 * ncodes > len(body):
            raise CorruptedFileError(f"SCP Section 2 truncated in Huffman table {t + 1}")
        codes = []
        for _ in range(ncodes):
            prefix, total, mode = body[pos], body[pos + 1], body[pos + 2]
            value = _i16(body, pos + 3)
            stored = _u32(body, pos + 5)
            pos += 9
            if prefix > total:
                prefix, total = total, prefix
            if not 1 <= prefix <= 32:
                raise CorruptedFileError(f"SCP Section 2: invalid Huffman prefix length {prefix}")
            # The base code is stored bit-reversed (first bit in the LSB)
            code = int(f"{stored & ((1 << prefix) - 1):0{prefix}b}"[::-1], 2)
            codes.append(_HuffmanCode(prefix, total, mode, value, code))
        tables.append(_make_table(codes))
    return tables


def _huffman_decode(data: bytes, count: int, tables: list[_HuffmanTable], what: str) -> np.ndarray:
    """Decode *count* values from a Huffman bit stream (MSB first)."""
    nbits = len(data) * 8
    bits = bin(int.from_bytes(data, "big") | (1 << nbits))[3:] if data else ""
    out = np.empty(count, dtype=np.int64)
    table = tables[0]
    pos = 0
    n = 0
    while n < count:
        code = 0
        length = 0
        entry = None
        while entry is None:
            if pos >= nbits:
                raise CorruptedFileError(
                    f"SCP {what}: Huffman data ends after {n} of {count} samples"
                )
            code = (code << 1) | (bits[pos] == "1")
            pos += 1
            length += 1
            entry = table.codes.get((length, code))
            if entry is None and length >= table.max_prefix:
                raise CorruptedFileError(f"SCP {what}: invalid Huffman code at bit {pos}")
        if entry.mode == 0:
            if not 1 <= entry.value <= len(tables):
                raise CorruptedFileError(f"SCP {what}: switch to unknown Huffman table {entry.value}")
            table = tables[entry.value - 1]
            continue
        extra = entry.total_bits - entry.prefix_bits
        value = entry.value
        if extra:
            if pos + extra > nbits:
                raise CorruptedFileError(
                    f"SCP {what}: Huffman data ends after {n} of {count} samples"
                )
            raw = int(bits[pos:pos + extra], 2)
            if raw >= 1 << (extra - 1):
                raw -= 1 << extra
            value += raw
            pos += extra
        out[n] = value
        n += 1
    return out


def _undo_differences(values: np.ndarray, mode: int) -> np.ndarray:
    """Reconstruct amplitudes from first (1) or second (2) differences.

    Samples are 16-bit: like the reference decoders, the reconstruction
    wraps around so that encoders using 16-bit arithmetic decode exactly.
    """
    if mode == 0 or len(values) == 0:
        return values
    if mode == 1:
        out = np.cumsum(values)
    elif len(values) < 3:
        out = values.copy()
    else:
        # x[n] = d[n] + 2 x[n-1] - x[n-2], with x[0] = d[0] and x[1] = d[1]
        slope = np.empty(len(values) - 1, dtype=np.int64)
        slope[0] = values[1] - values[0]
        slope[1:] = slope[0] + np.cumsum(values[2:])
        out = np.concatenate(([values[0]], values[0] + np.cumsum(slope)))
    return out.astype(np.int16).astype(np.int64)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


@dataclass
class _Section:
    id: int
    offset: int
    length: int
    version: int
    protocol: int

    def body(self, data: bytes) -> bytes:
        return data[self.offset + _SECTION_HEADER: self.offset + self.length]


@dataclass
class _LeadDef:
    start: int
    end: int
    code: int

    @property
    def count(self) -> int:
        return self.end - self.start + 1


@dataclass
class _QRS:
    type: int
    start: int
    fiducial: int
    end: int


@dataclass
class _Context:
    data: bytes = b""
    sections: dict[int, _Section] = field(default_factory=dict)
    warnings: list[tuple[str, type[Warning]]] = field(default_factory=list)
    encoding: str | None = None
    protocol: int = 0
    tables: list[_HuffmanTable] | None = None
    leads: list[_LeadDef] = field(default_factory=list)
    lead_flags: int = 0
    ref_length_ms: int = 0
    ref_fiducial: int = 0
    qrs_count: int | None = None
    qrs: list[_QRS] = field(default_factory=list)
    protected: list[tuple[int, int]] = field(default_factory=list)
    confirming_physician: str = ""

    def warn(self, message: str, category: type[Warning] = UserWarning) -> None:
        self.warnings.append((message, category))


class SCPECGParser(Parser):
    """Parser for SCP-ECG binary ECG files (EN 1064 / ISO 11073-91064)."""

    FORMAT_NAME = "SCP-ECG"
    FORMAT_DESCRIPTION = "Standard Communications Protocol for ECG (EN 1064 / ISO 11073-91064)"
    FILE_EXTENSIONS = [".scp"]
    PRIORITY = 30

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        if len(header) < 32:
            return False
        record_size = _u32(header, 2)
        sec0_id = _u16(header, 8)
        sec0_len = _u32(header, 10)
        if sec0_id != 0 or sec0_len < 26 or (sec0_len - _SECTION_HEADER) % 10:
            return False
        if record_size < 6 + sec0_len:
            return False
        # The first pointer describes Section 0 itself, which starts at byte 7
        return _u16(header, 22) == 0 and _u32(header, 28) == 7

    def parse(self, file_path: Path) -> ECGRecord:
        data = Path(file_path).read_bytes()
        ctx = _Context()
        try:
            record = self._parse(data, ctx, file_path)
        except (struct.error, IndexError) as exc:
            raise CorruptedFileError(f"SCP-ECG: truncated or inconsistent data ({exc})") from exc
        for message, category in ctx.warnings:
            warnings.warn(message, category, stacklevel=2)
        return record

    # -- structure ---------------------------------------------------------

    def _parse(self, data: bytes, ctx: _Context, file_path: Path) -> ECGRecord:
        if len(data) < 6 + _SECTION_HEADER:
            raise CorruptedFileError(f"File too small for SCP-ECG: {len(data)} bytes")
        record_size = _u32(data, 2)
        if record_size > len(data):
            raise CorruptedFileError(
                f"SCP-ECG record declares {record_size} bytes but the file has {len(data)}"
            )
        if record_size < 6 + _SECTION_HEADER:
            raise CorruptedFileError(f"SCP-ECG record size {record_size} is too small")
        data = data[:record_size]
        ctx.data = data

        record = ECGRecord(source_format="scp_ecg")
        raw = record.raw_metadata
        raw["filepath"] = str(file_path)
        raw["file_size"] = record_size

        self._read_pointers(ctx)
        raw["checksum_valid"] = self._verify_crcs(ctx)
        raw["sections_found"] = sorted(ctx.sections)
        sec0 = ctx.sections[0]
        raw["section_versions"] = {i: s.version for i, s in ctx.sections.items()}
        ctx.protocol = sec0.protocol

        if 2 in ctx.sections:
            ctx.tables = _parse_huffman_tables(ctx.sections[2].body(data))
            raw["huffman_tables"] = (
                "default" if ctx.tables[0] is _DEFAULT_TABLE else len(ctx.tables)
            )
        if 6 not in ctx.sections and 12 in ctx.sections:
            raise UnsupportedFormatError(
                "SCP-ECG: long-term rhythm data (Section 12, SCP-ECG v3) is not supported"
            )
        if 3 not in ctx.sections:
            raise CorruptedFileError("SCP-ECG: Section 3 (lead definition) is missing")
        self._parse_section3(ctx)
        if 4 in ctx.sections:
            self._parse_section4(ctx, raw)

        sec1 = self._parse_section1(ctx, record) if 1 in ctx.sections else {}
        if not ctx.protocol and sec1.get("protocol"):
            ctx.protocol = sec1["protocol"]
        if ctx.protocol:
            record.file_format = FileFormatInfo(version=f"{ctx.protocol / 10:.1f}")

        median = None
        if 5 in ctx.sections:
            median = self._parse_section5(ctx, record)
        if 6 not in ctx.sections:
            raise CorruptedFileError("SCP-ECG: Section 6 (rhythm data) is missing")
        self._parse_section6(ctx, record, median)

        if 7 in ctx.sections:
            self._parse_section7(ctx, record)
        if ctx.qrs_count is not None:
            record.measurements.qrs_count = ctx.qrs_count
        if 10 in ctx.sections:
            self._parse_section10(ctx, raw)
        self._parse_statements(ctx, record)
        if 9 in ctx.sections:
            raw["section9"] = ctx.sections[9].body(data)
        if 12 in ctx.sections and ctx.protocol >= 30:
            # SCP-ECG v3 Section 12 (long-term rhythm data) next to Section 6
            raw["section12_length"] = ctx.sections[12].length
            ctx.warn("SCP-ECG: Section 12 (long-term rhythm data) is present but not loaded")
        vendor = {
            i: s.length for i, s in ctx.sections.items()
            if i > 12 or (i == 12 and ctx.protocol < 30)
        }
        if vendor:
            raw["vendor_sections"] = vendor

        leads = record.leads
        if leads:
            longest = max(leads, key=lambda lead: len(lead.samples))
            record.recording.duration = timedelta(
                seconds=len(longest.samples) / longest.sampling_rate
            )
        fill_signal_summary(record)
        return record

    def _read_pointers(self, ctx: _Context) -> None:
        data = ctx.data
        sec0_id, sec0_len = _u16(data, 8), _u32(data, 10)
        if sec0_id != 0:
            raise CorruptedFileError(f"SCP-ECG: expected Section 0 at byte 7, found section {sec0_id}")
        if sec0_len < _SECTION_HEADER or 6 + sec0_len > len(data):
            raise CorruptedFileError(f"SCP-ECG: invalid Section 0 length {sec0_len}")
        ctx.sections[0] = _Section(0, 6, sec0_len, data[14], data[15])
        pos = 6 + _SECTION_HEADER
        end = 6 + sec0_len
        while pos + 10 <= end:
            sid, length, index = _u16(data, pos), _u32(data, pos + 2), _u32(data, pos + 6)
            pos += 10
            if sid == 0 or length == 0:
                continue
            offset = index - 1
            if index == 0 or offset + length > len(data):
                raise CorruptedFileError(
                    f"SCP-ECG: Section {sid} ({length} bytes at byte {index}) "
                    f"lies outside the {len(data)}-byte record"
                )
            if sid > 11:
                # Manufacturer sections are kept as found, their headers vary
                ctx.sections[sid] = _Section(sid, offset, length, 0, 0)
                continue
            if length < _SECTION_HEADER:
                raise CorruptedFileError(f"SCP-ECG: Section {sid} is shorter than its header")
            header_id, header_len = _u16(data, offset + 2), _u32(data, offset + 4)
            if header_id != sid:
                raise CorruptedFileError(
                    f"SCP-ECG: Section 0 points to section {sid} but the header says {header_id}"
                )
            if header_len != length:
                ctx.warn(
                    f"SCP-ECG: Section {sid} length is {header_len} in its header and "
                    f"{length} in Section 0, using {min(header_len, length)}"
                )
                length = max(_SECTION_HEADER, min(header_len, length))
            ctx.sections[sid] = _Section(sid, offset, length, data[offset + 8], data[offset + 9])

    def _verify_crcs(self, ctx: _Context) -> bool:
        data = ctx.data
        bad = []
        if _crc(data[2:]) != _u16(data, 0):
            bad.append("record")
        for sid, sec in sorted(ctx.sections.items()):
            if sid > 11:
                continue
            chunk = data[sec.offset: sec.offset + sec.length]
            if _crc(chunk[2:]) != _u16(chunk, 0):
                bad.append(f"Section {sid}")
        if bad:
            ctx.warn(f"SCP-ECG CRC mismatch in: {', '.join(bad)}", ChecksumWarning)
        return not bad

    # -- Section 1 ---------------------------------------------------------

    @staticmethod
    def _split_tags(body: bytes, fixed: bool) -> list[tuple[int, bytes]] | None:
        """Split Section 1 into (tag, value). Returns None if the walk fails."""
        tags = []
        pos = 0
        while pos + 3 <= len(body):
            tag, length = body[pos], _u16(body, pos + 1)
            if fixed and tag in _FIXED_TAG_SIZES:
                length = _FIXED_TAG_SIZES[tag]
            pos += 3
            if tag == 255:
                return tags
            if pos + length > len(body):
                return None
            tags.append((tag, body[pos: pos + length]))
            pos += length
        return tags  # no terminator, but the walk ended on the section end

    def _parse_section1(self, ctx: _Context, record: ECGRecord) -> dict:
        body = ctx.sections[1].body(ctx.data)
        tags = self._split_tags(body, fixed=False)
        if tags is None:
            # Some writers store a wrong length for fixed-size tags
            tags = self._split_tags(body, fixed=True)
            if tags is None:
                raise CorruptedFileError("SCP-ECG: Section 1 tag lengths overrun the section")
            ctx.warn("SCP-ECG: Section 1 has a wrong tag length, "
                     "used the fixed sizes defined by the standard")

        # The language code (tag 14) decides how every text tag is decoded
        for tag, value in tags:
            if tag == 14 and len(value) > 16 and value[16] & 0x01:
                ctx.encoding = _LANGUAGE_CODECS.get(value[16], "latin-1")

        def text(value: bytes) -> str:
            return _decode(value, ctx.encoding)

        patient = record.patient
        rec = record.recording
        device = rec.device
        filters = rec.acquisition.filters
        raw = record.raw_metadata
        extra: dict = {}
        out: dict = {}
        age_years = None
        acq_date = acq_time = None
        manufacturer_tags: dict[int, str] = {}
        history: list[str] = []
        comments: list[str] = []

        for tag, value in tags:
            if tag == 0:
                patient.last_name = text(value)
            elif tag == 1:
                patient.first_name = text(value)
            elif tag == 2:
                patient.patient_id = text(value)
            elif tag == 3:
                extra["second_last_name"] = text(value)
            elif tag == 4 and len(value) >= 3:
                age, unit = _u16(value, 0), value[2]
                if age and unit in _AGE_UNITS:
                    age_years = int(age * _AGE_UNITS[unit])
            elif tag == 5:
                patient.birth_date = _date(value)
            elif tag == 6 and len(value) >= 3:
                height, unit = _u16(value, 0), value[2]
                if height and unit in _HEIGHT_TO_CM:
                    patient.height = round(height * _HEIGHT_TO_CM[unit], 1)
                elif height:
                    extra["height_unspecified_unit"] = height
            elif tag == 7 and len(value) >= 3:
                weight, unit = _u16(value, 0), value[2]
                if weight and unit in _WEIGHT_TO_KG:
                    patient.weight = round(weight * _WEIGHT_TO_KG[unit], 2)
                elif weight:
                    extra["weight_unspecified_unit"] = weight
            elif tag == 8 and value:
                patient.sex = _SEX.get(value[0], "U")
            elif tag == 9 and value:
                patient.race = _RACE.get(value[0], "" if value[0] == 0 else f"code {value[0]}")
            elif tag == 10 and value:
                drug = text(value[3:]) if len(value) > 3 else ""
                if not drug and len(value) >= 3:
                    drug = f"table {value[0]} class {value[1]} code {value[2]}"
                if drug:
                    patient.medications.append(drug)
            elif tag in (11, 12) and len(value) >= 2:
                pressure = _u16(value, 0)
                if pressure:
                    key = "systolic_bp_mmhg" if tag == 11 else "diastolic_bp_mmhg"
                    extra[key] = pressure
            elif tag == 13:
                history.append(text(value))
            elif tag in (14, 15) and len(value) >= 36:
                info = self._device_id(value, text)
                if tag == 14:
                    out["protocol"] = info.pop("protocol_revision")
                    device.manufacturer = info.pop("manufacturer")
                    device.model = info.pop("model")
                    device.serial_number = info.pop("serial_number")
                    device.software_version = info.pop("software")
                    mains = info.pop("mains_frequency")
                    if mains:
                        rec.acquisition.signal.acsetting = mains
                    extra["acquiring_device"] = info
                else:
                    extra["analyzing_device"] = info
            elif tag == 16:
                device.institution = text(value)
            elif tag == 17:
                extra["analyzing_institution"] = text(value)
            elif tag == 18:
                device.department = text(value)
            elif tag == 19:
                extra["analyzing_department"] = text(value)
            elif tag == 20:
                rec.referring_physician = text(value)
            elif tag == 21:
                ctx.confirming_physician = text(value)
            elif tag == 22:
                rec.technician = text(value)
            elif tag == 23:
                rec.room = text(value)
            elif tag == 24 and value:
                extra["stat_code"] = value[0]
            elif tag == 25:
                acq_date = _date(value)
            elif tag == 26:
                acq_time = _time(value)
            elif tag == 27 and len(value) >= 2:
                filters.highpass = _u16(value, 0) / 100.0 or None
            elif tag == 28 and len(value) >= 2:
                filters.lowpass = float(_u16(value, 0)) or None
            elif tag == 29 and value:
                bitmap = value[0]
                filters.notch = 60.0 if bitmap & 0x01 else 50.0 if bitmap & 0x02 else None
                filters.notch_active = bool(bitmap & 0x03)
                filters.artifact_filter = bool(bitmap & 0x04)
                extra["filter_bitmap"] = bitmap
            elif tag == 30:
                comments.append(text(value))
            elif tag == 31:
                extra["sequence_number"] = text(value)
            elif tag == 32 and value:
                codes = list(value[1:]) if value[0] == 0 else []
                extra.setdefault("medical_history_codes", []).append(
                    {"table": value[0], "codes": list(value[1:])}
                )
                if _PACEMAKER_HISTORY_CODE in codes:
                    patient.has_pacemaker = True
            elif tag == 33 and len(value) >= 2:
                placement = [
                    _ELECTRODES_12.get(value[0], ""),
                    f"XYZ {_ELECTRODES_XYZ[value[1]]}" if value[1] in _ELECTRODES_XYZ else "",
                ]
                rec.acquisition.signal.electrode_placement = ", ".join(p for p in placement if p)
            elif tag == 34 and len(value) >= 4:
                offset = _i16(value, 0)
                if offset != 0x7FFF and abs(offset) <= 14 * 60:
                    out["tz"] = timezone(timedelta(minutes=offset))
                if len(value) > 4:
                    extra["time_zone_description"] = text(value[4:])
            elif tag == 35:
                history.append(text(value))
            elif tag >= 200:
                manufacturer_tags[tag] = text(value) if value else ""
            else:
                extra.setdefault("unknown_tags", {})[tag] = value

        patient.clinical_history = "\n".join(h for h in history if h)
        if comments and any(comments):
            record.annotations["comments"] = "\n".join(c for c in comments if c)
        if acq_date is not None:
            if acq_time is not None:
                rec.date = acq_date.replace(
                    hour=acq_time[0], minute=acq_time[1], second=acq_time[2],
                    tzinfo=out.get("tz"),
                )
            else:
                extra["acquisition_date"] = acq_date.date().isoformat()
        elif acq_time is not None:
            # A time of day without a date: kept as stored, no date invented
            extra["acquisition_time"] = "{:02d}:{:02d}:{:02d}".format(*acq_time)
        # Stored age (0 means not set), else derived from the date of birth
        if age_years:
            patient.age = age_years
        elif patient.birth_date is not None and acq_date is not None:
            born, on = patient.birth_date, acq_date
            age = on.year - born.year - ((on.month, on.day) < (born.month, born.day))
            patient.age = age if age >= 0 else None
        if manufacturer_tags:
            extra["manufacturer_tags"] = manufacturer_tags
        raw.update(extra)
        return out

    @staticmethod
    def _device_id(value: bytes, text) -> dict:
        """Decode the machine ID of tags 14 and 15."""
        strings = value[36:].split(b"\x00")
        strings += [b""] * 5
        code = value[7]
        manufacturer = text(strings[4])
        if not manufacturer and code < len(_MANUFACTURERS):
            manufacturer = _MANUFACTURERS[code]
        return {
            "institution_number": _u16(value, 0),
            "department_number": _u16(value, 2),
            "device_id": _u16(value, 4),
            "device_type": {0: "cart", 1: "system or host"}.get(value[6], str(value[6])),
            "manufacturer_code": code,
            "manufacturer": manufacturer,
            "model": text(value[8:14]),
            "protocol_revision": value[14],
            "compatibility_level": value[15],
            "language_code": value[16],
            "capabilities": value[17],
            "mains_frequency": {1: 50, 2: 60}.get(value[18]),
            "analysis_program": text(strings[0]),
            "serial_number": text(strings[1]),
            "software": text(strings[2]),
            "scp_implementation": text(strings[3]),
        }

    # -- Sections 3 and 4 --------------------------------------------------

    def _parse_section3(self, ctx: _Context) -> None:
        body = ctx.sections[3].body(ctx.data)
        if len(body) < 2:
            raise CorruptedFileError("SCP-ECG: Section 3 is too short")
        count, ctx.lead_flags = body[0], body[1]
        if count == 0:
            raise CorruptedFileError("SCP-ECG: Section 3 defines no leads")
        if 2 + 9 * count > len(body):
            raise CorruptedFileError(
                f"SCP-ECG: Section 3 declares {count} leads but holds {(len(body) - 2) // 9}"
            )
        for i in range(count):
            pos = 2 + 9 * i
            lead = _LeadDef(_u32(body, pos), _u32(body, pos + 4), body[pos + 8])
            if lead.start < 1 or lead.end < lead.start:
                raise CorruptedFileError(
                    f"SCP-ECG: lead {i + 1} has invalid sample range {lead.start}-{lead.end}"
                )
            ctx.leads.append(lead)

    def _parse_section4(self, ctx: _Context, raw: dict) -> None:
        body = ctx.sections[4].body(ctx.data)
        if len(body) < 6:
            raise CorruptedFileError("SCP-ECG: Section 4 is too short")
        ctx.ref_length_ms, ctx.ref_fiducial, nqrs = _u16(body, 0), _u16(body, 2), _u16(body, 4)
        ctx.qrs_count = nqrs or None
        pos = 6
        if pos + 14 * nqrs > len(body):
            if ctx.lead_flags & 0x01:
                raise CorruptedFileError("SCP-ECG: Section 4 subtraction zones are truncated")
            ctx.warn("SCP-ECG: Section 4 QRS subtraction zones are truncated, ignored")
            return
        for _ in range(nqrs):
            ctx.qrs.append(_QRS(_u16(body, pos), _u32(body, pos + 2),
                                _u32(body, pos + 6), _u32(body, pos + 10)))
            pos += 14
        if pos + 8 * nqrs <= len(body):
            for _ in range(nqrs):
                ctx.protected.append((_u32(body, pos), _u32(body, pos + 4)))
                pos += 8
        raw["reference_beat_length_ms"] = ctx.ref_length_ms
        raw["qrs_locations"] = [
            {"type": q.type, "start": q.start, "fiducial": q.fiducial, "end": q.end}
            for q in ctx.qrs
        ]
        if ctx.protected:
            raw["qrs_protected_zones"] = list(ctx.protected)

    # -- Sections 5 and 6 --------------------------------------------------

    def _decode_leads(
        self, ctx: _Context, sid: int, counts: list[int] | None,
    ) -> tuple[int, int, int, int, list[np.ndarray]]:
        """Decode the per-lead data of Section 5 or 6.

        Returns (AVM, sample interval, difference mode, byte 6, leads).
        *counts* gives the expected samples per lead, None to use all bytes.
        """
        what = f"Section {sid}"
        body = ctx.sections[sid].body(ctx.data)
        nleads = len(ctx.leads)
        if len(body) < 6 + 2 * nleads:
            raise CorruptedFileError(f"SCP-ECG: {what} is too short")
        avm, interval, diff, flag = _u16(body, 0), _u16(body, 2), body[4], body[5]
        if diff not in (0, 1, 2):
            raise CorruptedFileError(f"SCP-ECG: {what} has unknown difference encoding {diff}")
        sizes = [_u16(body, 6 + 2 * i) for i in range(nleads)]
        pos = 6 + 2 * nleads
        if pos + sum(sizes) > len(body):
            raise CorruptedFileError(
                f"SCP-ECG: {what} lead data ({sum(sizes)} bytes) exceeds the section"
            )
        leads = []
        for i, size in enumerate(sizes):
            chunk = body[pos: pos + size]
            pos += size
            count = counts[i] if counts is not None else None
            if ctx.tables is not None:
                if count is None:
                    raise CorruptedFileError(f"SCP-ECG: {what} sample count is unknown")
                values = _huffman_decode(chunk, count, ctx.tables, what)
            else:
                if count is None:
                    count = size // 2
                if 2 * count > size:
                    raise CorruptedFileError(
                        f"SCP-ECG: {what} lead {i + 1} holds {size // 2} samples, expected {count}"
                    )
                values = np.frombuffer(chunk, dtype="<i2", count=count).astype(np.int64)
            leads.append(_undo_differences(values, diff))
        return avm, interval, diff, flag, leads

    @staticmethod
    def _make_lead(
        label: str, samples: np.ndarray, rate: int, avm: float, file_avm: float | None = None,
    ) -> Lead:
        """*avm* scales the samples; *file_avm* is the multiplier as written."""
        file_avm = avm if file_avm is None else file_avm
        if avm > 0:
            resolution, unit = avm / 1000.0, "uV"
        else:
            resolution, unit = 1.0, ""
        is_raw = derive_is_raw(resolution, 0.0, unit)
        return Lead(
            label=label,
            samples=samples.astype(np.float64),
            sampling_rate=rate,
            resolution=resolution,
            resolution_unit=unit,
            units="" if is_raw else unit,
            is_raw=is_raw,
            adc_resolution=float(file_avm) if file_avm > 0 else 0.0,
            adc_resolution_unit="nV" if file_avm > 0 else "",
        )

    @staticmethod
    def _rate(interval: int, what: str) -> int:
        if interval == 0:
            raise CorruptedFileError(f"SCP-ECG: {what} sample interval is 0")
        return round(1_000_000 / interval)

    def _labels(self, ctx: _Context) -> list[str]:
        return unique_labels([_lead_label(lead.code) for lead in ctx.leads])

    def _parse_section5(self, ctx: _Context, record: ECGRecord) -> dict | None:
        body = ctx.sections[5].body(ctx.data)
        if len(body) < 6:
            raise CorruptedFileError("SCP-ECG: Section 5 is too short")
        interval = _u16(body, 2)
        rate = self._rate(interval, "Section 5")
        counts = None
        if ctx.ref_length_ms:
            n = round(ctx.ref_length_ms * 1000 / interval)
            if n < 1:
                ctx.warn(f"SCP-ECG: reference beat length {ctx.ref_length_ms} ms is shorter "
                         "than one sample, median beats skipped")
                return None
            counts = [n] * len(ctx.leads)
        elif ctx.tables is not None:
            ctx.warn("SCP-ECG: Section 5 is Huffman encoded but the reference beat "
                     "length (Section 4) is missing, median beats skipped")
            return None
        avm, interval, diff, _, beats = self._decode_leads(ctx, 5, counts)
        if avm == 0:
            ctx.warn("SCP-ECG: Section 5 amplitude multiplier is 0, median beats left unscaled")
        record.median_beats = [
            self._make_lead(label, beat, rate, avm)
            for label, beat in zip(self._labels(ctx), beats)
        ]
        record.raw_metadata["median_encoding"] = ("amplitudes", "first_difference", "second_difference")[diff]
        record.raw_metadata["median_sample_interval_us"] = interval
        return {"avm": avm, "interval": interval, "beats": beats}

    def _parse_section6(self, ctx: _Context, record: ECGRecord, median: dict | None) -> None:
        body = ctx.sections[6].body(ctx.data)
        if len(body) < 6:
            raise CorruptedFileError("SCP-ECG: Section 6 is too short")
        interval = _u16(body, 2)
        rate = self._rate(interval, "Section 6")
        # Byte 6 is the bimodal flag up to version 2.x, reserved from 3.0
        bimodal = bool(body[5]) and ctx.protocol < 30
        subtraction = bool(ctx.lead_flags & 0x01)
        if bimodal and (median is None or median["interval"] != interval):
            raise UnsupportedFormatError(
                "SCP-ECG: bimodal compression (decimated rhythm data) is not supported"
            )
        avm, interval, diff, _, leads = self._decode_leads(
            ctx, 6, [lead.count for lead in ctx.leads]
        )
        res_avm = avm
        if subtraction:
            leads, res_avm = self._add_reference_beats(ctx, leads, avm, median, interval)
        if res_avm == 0:
            ctx.warn("SCP-ECG: Section 6 amplitude multiplier is 0, leads left unscaled")
        labels = self._labels(ctx)
        for label, lead_def, samples in zip(labels, ctx.leads, leads):
            lead = self._make_lead(label, samples, rate, res_avm, avm)
            if lead_def.start != 1:
                lead.annotations["start_sample"] = str(lead_def.start)
            record.leads.append(lead)

        sig = record.recording.acquisition.signal
        sig.sampling_rate = rate
        sig.signal_signed = True
        sig.number_channels_allocated = len(ctx.leads)
        sig.number_channels_valid = len(record.leads)
        steps = []
        if ctx.tables is not None:
            steps.append("huffman")
        if subtraction:
            steps.append("reference beat subtraction")
        if bimodal:
            steps.append("bimodal")
        sig.compression = "+".join(steps) or "none"
        sig.data_encoding = ("amplitudes", "first_difference", "second_difference")[diff]
        if ctx.tables is None:
            sig.bits_per_sample = 16
        if ctx.lead_flags & 0x04:
            record.raw_metadata["simultaneous_leads"] = len(ctx.leads)
        else:
            record.raw_metadata["simultaneous_leads"] = (ctx.lead_flags >> 3) & 0x1F
        record.raw_metadata["rhythm_avm_nv"] = avm
        record.raw_metadata["rhythm_sample_interval_us"] = interval

    def _add_reference_beats(
        self, ctx: _Context, residuals: list[np.ndarray], avm: int,
        median: dict | None, interval: int,
    ) -> tuple[list[np.ndarray], int]:
        """Add reference beat type 0 back into the Section 6 residuals."""
        if median is None or not ctx.qrs:
            raise CorruptedFileError(
                "SCP-ECG: reference beat subtraction is used but Section 4 or 5 is missing"
            )
        if median["interval"] != interval:
            raise UnsupportedFormatError(
                "SCP-ECG: reference beat and rhythm data have different sample rates"
            )
        ref_avm = median["avm"]
        if avm == 0 or ref_avm == 0:
            raise CorruptedFileError("SCP-ECG: amplitude multiplier 0 with reference beat subtraction")
        # Express both in a common integer unit of gcd(AVM5, AVM6) nV
        common = math.gcd(avm, ref_avm)
        out = []
        for lead_def, residual, beat in zip(ctx.leads, residuals, median["beats"]):
            signal = residual * (avm // common)
            beat = beat * (ref_avm // common)
            for qrs in ctx.qrs:
                if qrs.type != 0:
                    continue
                first = max(qrs.start, lead_def.start, qrs.fiducial - ctx.ref_fiducial + 1)
                last = min(qrs.end, lead_def.end,
                           qrs.fiducial - ctx.ref_fiducial + len(beat))
                if last < first:
                    continue
                r0 = first - lead_def.start
                m0 = ctx.ref_fiducial - qrs.fiducial + first - 1
                signal[r0: r0 + last - first + 1] += beat[m0: m0 + last - first + 1]
            out.append(signal)
        return out, common

    # -- Sections 7, 8, 10, 11 ---------------------------------------------

    def _parse_section7(self, ctx: _Context, record: ECGRecord) -> None:
        body = ctx.sections[7].body(ctx.data)
        if len(body) < 6:
            raise CorruptedFileError("SCP-ECG: Section 7 is too short")
        nblocks, nspikes = body[0], body[1]
        rr, pp = _u16(body, 2), _u16(body, 4)
        pos = 6
        if pos + 16 * nblocks + 10 * nspikes > len(body):
            raise CorruptedFileError("SCP-ECG: Section 7 measurement blocks are truncated")
        gm = record.measurements
        raw = record.raw_metadata

        def ok(value: int) -> bool:
            return value not in (_UNDEFINED, 0xFFFF)

        def axis(value: int) -> int | None:
            return None if abs(value) in (999, _UNDEFINED) else value

        if ok(rr) and rr:
            gm.rr_interval = rr
        if ok(pp) and pp:
            raw["pp_interval_ms"] = pp
        blocks = []
        for _ in range(nblocks):
            p_on, p_off, qrs_on, qrs_off, t_off = struct.unpack_from("<5H", body, pos)
            axes = struct.unpack_from("<3h", body, pos + 10)
            blocks.append((p_on, p_off, qrs_on, qrs_off, t_off, *axes))
            pos += 16
        if blocks:
            # Section 7 stores wave boundaries, not intervals: they are reported
            # as stored and PR/QRS/QT are not computed from them
            p_on, p_off, qrs_on, qrs_off, t_off, p_ax, qrs_ax, t_ax = blocks[0]
            gm.p_axis, gm.qrs_axis, gm.t_axis = axis(p_ax), axis(qrs_ax), axis(t_ax)
            names = ("p_onset", "p_offset", "qrs_onset", "qrs_offset", "t_offset")
            record.annotations["wave_boundaries_ms"] = ", ".join(
                f"{n}={v}" for n, v in zip(names, (p_on, p_off, qrs_on, qrs_off, t_off)) if ok(v)
            )
            raw["measurement_blocks"] = [
                dict(zip(("p_onset", "p_offset", "qrs_onset", "qrs_offset", "t_offset",
                          "p_axis", "qrs_axis", "t_axis"), b))
                for b in blocks
            ]

        spikes = []
        for i in range(nspikes):
            time_ms, amplitude = _u16(body, pos + 4 * i), _i16(body, pos + 4 * i + 2)
            info = pos + 4 * nspikes + 6 * i
            spikes.append({
                "time_ms": time_ms,
                "amplitude_uv": amplitude,
                "type": _SPIKE_TYPES.get(body[info], "unknown"),
                "source": _SPIKE_SOURCES.get(body[info + 1], "unknown"),
                "qrs_index": _u16(body, info + 2),
                "pulse_width_us": _u16(body, info + 4),
            })
        pos += 10 * nspikes
        if spikes:
            # Detected spikes are device output, not a patient attribute
            raw["pacemaker_spikes"] = spikes

        # Optional parts (QRS types, additional measurements): read when complete
        if pos + 2 <= len(body):
            nqrs = _u16(body, pos)
            if pos + 2 + nqrs <= len(body):
                raw["qrs_types"] = list(body[pos + 2: pos + 2 + nqrs])
                pos += 2 + nqrs
                if record.measurements.qrs_count is None and ctx.qrs_count is None:
                    gm.qrs_count = nqrs
                if pos + 7 <= len(body):
                    vent, atrial, qtc, formula = struct.unpack_from("<3HB", body, pos)
                    if ok(vent) and vent:
                        gm.heart_rate = vent
                    if ok(atrial) and atrial:
                        raw["atrial_rate_bpm"] = atrial
                    if ok(qtc) and qtc:
                        if formula == 1:
                            gm.qtc_bazett = qtc
                        else:
                            raw["qtc_ms"] = qtc
                            raw["qtc_formula_code"] = formula
                            record.annotations["qtc"] = f"{qtc} ms (formula code {formula})"

    @staticmethod
    def _lead_blocks(body: bytes, count: int, with_header: bool) -> list[tuple[int, bytes]] | None:
        blocks = []
        pos = 4
        for _ in range(count):
            if pos + 4 > len(body):
                return None
            code, size = _u16(body, pos), _u16(body, pos + 2)
            data_size = size - 4 if with_header else size
            if data_size < 0 or pos + 4 + data_size > len(body):
                return None
            blocks.append((code, body[pos + 4: pos + 4 + data_size]))
            pos += 4 + data_size
        return blocks

    def _parse_section10(self, ctx: _Context, raw: dict) -> None:
        body = ctx.sections[10].body(ctx.data)
        if len(body) < 4:
            raise CorruptedFileError("SCP-ECG: Section 10 is too short")
        count = _u16(body, 0)
        blocks = self._lead_blocks(body, count, with_header=False)
        if blocks is None:
            # Some writers count the 4-byte block header in the block length
            blocks = self._lead_blocks(body, count, with_header=True)
            if blocks is None:
                raise CorruptedFileError("SCP-ECG: Section 10 is truncated")
        labels = unique_labels([_lead_label(code) for code, _ in blocks])
        result = {}
        for label, (_, block) in zip(labels, blocks):
            values = {}
            for i, name in enumerate(_LEAD_MEASUREMENTS[: len(block) // 2]):
                value = _i16(block, 2 * i)
                if value not in (_UNDEFINED, -32768):
                    values[name] = value
            result[label] = values
        raw["lead_measurements"] = result

    def _read_statements(self, ctx: _Context, sid: int) -> Interpretation:
        """Section 8 (textual statements) or 11 (universal statement codes)."""
        body = ctx.sections[sid].body(ctx.data)
        what = f"Section {sid}"
        if len(body) < 9:
            raise CorruptedFileError(f"SCP-ECG: {what} is too short")
        interp = Interpretation()
        interp.source = {1: "confirmed", 2: "overread"}.get(body[0], "machine")
        day, clock = _date(body[1:5]), _time(body[5:8])
        if day is not None and clock is not None:
            interp.interpretation_date = day.replace(hour=clock[0], minute=clock[1], second=clock[2])
        count = body[8]
        pos = 9
        for _ in range(count):
            if pos + 3 > len(body):
                raise CorruptedFileError(f"SCP-ECG: {what} statements are truncated")
            size = _u16(body, pos + 1)
            value = body[pos + 3: pos + 3 + size]
            if len(value) < size:
                raise CorruptedFileError(f"SCP-ECG: {what} statements are truncated")
            pos += 3 + size
            if sid == 11:
                value = value[1:]  # statement type byte
            parts = self._split_text(value, ctx.encoding)
            line = " ".join(p for p in parts if p)
            if line:
                interp.statements.append((line, ""))
        if interp.source != "machine":
            interp.interpreter = ctx.confirming_physician
        return interp

    @staticmethod
    def _split_text(value: bytes, encoding: str | None) -> list[str]:
        """Split NUL-separated statement texts."""
        if encoding in _WIDE_CODECS:
            parts, start = [], 0
            for i in range(0, len(value) - 1, 2):
                if value[i:i + 2] == b"\x00\x00":
                    parts.append(value[start:i])
                    start = i + 2
            parts.append(value[start:])
            return [_decode(p, encoding) for p in parts]
        return [decode_text(p, encoding).strip() for p in value.split(b"\x00")]

    def _parse_statements(self, ctx: _Context, record: ECGRecord) -> None:
        found = [self._read_statements(ctx, sid) for sid in (8, 11) if sid in ctx.sections]
        found = [i for i in found if i.statements]
        if ctx.confirming_physician and not any(i.source != "machine" for i in found):
            record.raw_metadata["confirming_physician"] = ctx.confirming_physician
        if not found:
            return
        expert = [i for i in found if i.source != "machine"]
        machine = [i for i in found if i.source == "machine"]
        ordered = expert + machine
        record.interpretation = ordered.pop(0)
        if expert and machine:
            ordered.remove(machine[0])
            record.annotations["machine_interpretation"] = "\n".join(
                s for s, _ in machine[0].statements
            )
        if ordered:  # both sections from the same source
            record.raw_metadata["universal_statements"] = [s for s, _ in ordered[0].statements]

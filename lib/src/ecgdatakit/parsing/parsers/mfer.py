"""MFER (Medical waveform Format Encoding Rules) parser.

Reference: MFER Part 1 v1.05 (basic encoding rules, equivalent to
ISO 22077-1), Part 3-1 (standard 12-lead ECG) and Part 3-2 (long-term ECG),
https://www.mfer.org/en/document.htm

A file is a sequence of TLV definitions read in order. Waveform data
(MWF_WAV) forms a frame laid out as SEQ sequences x CHN channels x BLK
samples, with per-channel overrides from MWF_ATT. Samples are the stored
values; ``physical = (sample + MWF_OFF) * MWF_SEN``. Samples equal to the
MWF_NUL value, and samples declared by the frame but absent from the data,
are NaN.
"""

from __future__ import annotations

import codecs
import copy
import math
import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ecgdatakit.exceptions import CorruptedFileError, UnsupportedFormatError
from ecgdatakit.models import (
    DeviceInfo,
    ECGRecord,
    FilterSettings,
    Lead,
    PatientInfo,
    RecordingInfo,
    SignalCharacteristics,
    derive_is_raw,
)
from ecgdatakit.parsing.helpers import decode_text, fill_signal_summary, unique_labels
from ecgdatakit.parsing.parser import Parser

# Tag codes (MFER Part 1, Table 5-64)
MWF_ZRO = 0x00
MWF_BLE = 0x01
MWF_VER = 0x02
MWF_TXC = 0x03
MWF_BLK = 0x04
MWF_CHN = 0x05
MWF_SEQ = 0x06
MWF_PNT = 0x07
MWF_WFM = 0x08
MWF_LDN = 0x09
MWF_DTP = 0x0A
MWF_IVL = 0x0B
MWF_SEN = 0x0C
MWF_OFF = 0x0D
MWF_CMP = 0x0E
MWF_IPD = 0x0F
MWF_FLT = 0x11
MWF_NUL = 0x12
MWF_INF = 0x15
MWF_NTE = 0x16
MWF_MAN = 0x17
MWF_WAV = 0x1E
MWF_ATT = 0x3F
MWF_PRE = 0x40
MWF_EVT = 0x41
MWF_VAL = 0x42
MWF_SKW = 0x43
MWF_CND = 0x44
MWF_RPT = 0x45
MWF_SIG = 0x46
MWF_SET = 0x67
MWF_END = 0x80
MWF_PNM = 0x81
MWF_PID = 0x82
MWF_AGE = 0x83
MWF_SEX = 0x84
MWF_TIM = 0x85
MWF_MSS = 0x86
MWF_UID = 0x87
MWF_MAP = 0x88

_KNOWN_TAGS = {
    MWF_BLE, MWF_VER, MWF_TXC, MWF_BLK, MWF_CHN, MWF_SEQ, MWF_PNT, MWF_WFM,
    MWF_LDN, MWF_DTP, MWF_IVL, MWF_SEN, MWF_OFF, MWF_CMP, MWF_IPD, MWF_FLT,
    MWF_NUL, MWF_INF, MWF_NTE, MWF_MAN, MWF_WAV, MWF_ATT, MWF_PRE, MWF_EVT,
    MWF_VAL, MWF_SKW, MWF_CND, MWF_RPT, MWF_SIG, MWF_SET, MWF_PNM, MWF_PID,
    MWF_AGE, MWF_SEX, MWF_TIM, MWF_MSS, MWF_UID, MWF_MAP,
}

# Definitions that describe the frame layout or sampling of a channel
_CHANNEL_ATTRS = {
    MWF_BLK: "blk", MWF_SEQ: "seq", MWF_WFM: "wfm", MWF_LDN: "ldn",
    MWF_DTP: "dtp", MWF_IVL: "ivl", MWF_SEN: "sen", MWF_OFF: "off", MWF_NUL: "nul",
}

# MWF_DTP data types (Table 5-6); 9 (AHA compression) is not supported
_DATA_TYPES = {0: "i2", 1: "u2", 2: "i4", 3: "u1", 4: "u2", 5: "i1", 6: "u4", 7: "f4", 8: "f8"}
_DATA_ENCODINGS = {
    0: "int16", 1: "uint16", 2: "int32", 3: "uint8", 4: "status16",
    5: "int8", 6: "uint32", 7: "float32", 8: "float64",
}
_DTP_STATUS = 4

# MWF_SEN unit codes (Table 5-4)
_SEN_UNITS = {
    0: "V", 1: "mmHg", 2: "Pa", 3: "cmH2O", 4: "mmHg/s", 5: "dyne", 6: "N",
    7: "%", 8: "degC", 9: "/min", 10: "/s", 11: "Ohm", 12: "A", 13: "rpm",
    14: "W", 15: "dB", 16: "kg", 17: "J", 18: "dyne s m-2 cm-5", 19: "L",
    20: "L/s", 21: "L/min", 22: "cd",
}

# MWF_WFM waveform types (Table 5-14)
_WAVEFORM_TYPES = {
    1: "Standard 12-lead ECG", 2: "Long-term ECG", 3: "Vectorcardiogram",
    4: "Stress ECG", 5: "Intracardiac ECG", 6: "Body surface ECG",
    7: "Ventricular late potential", 8: "Body surface late potential",
    9: "Dominant beat ECG", 20: "Long-term monitoring", 21: "Monitoring",
    25: "Power spectrum", 26: "Trend", 30: "Sound", 31: "Pulse wave",
    40: "Resting EEG", 41: "Evoked potential", 42: "EEG frequency analysis",
    43: "Long-term EEG",
}
_WFM_DOMINANT_BEAT = 9

# MWF_LDN lead codes (Part 1 Table 5-17, Part 3-2 Tables 5-21 and 5-22)
_LEAD_CODES = {
    1: "I", 2: "II", 3: "V1", 4: "V2", 5: "V3", 6: "V4", 7: "V5", 8: "V6",
    9: "V7", 11: "V3R", 12: "V4R", 13: "V5R", 14: "V6R", 15: "V7R",
    16: "X", 17: "Y", 18: "Z", 19: "CC5", 20: "CM5",
    31: "NASA", 32: "CB4", 33: "CB5", 34: "CB6",
    61: "III", 62: "aVR", 63: "aVL", 64: "aVF", 65: "-aVR",
    66: "V8", 67: "V9", 68: "V8R", 69: "V9R", 70: "D", 71: "A", 72: "J",
    143: "Blood pressure", 160: "Impedance respiration", 175: "SpO2",
    4160: "Status", 4161: "Body position", 4162: "Body motion",
    4163: "Respiration", 4166: "ECG1", 4167: "ECG2", 4168: "ECG3", 4169: "ECG4",
}
_LDN_STATUS = 4160

# MWF_TXC character sets (Table 5-34) mapped to Python codecs
_TEXT_CODECS = {
    "ascii": "ascii", "ansi x3.4": "ascii", "iso 646": "ascii",
    "utf-8": "utf-8", "utf8": "utf-8", "unicode": "utf-8",
    "jis x 0201": "shift_jis", "jis x 0208": "iso2022_jp", "jis x 0212": "iso2022_jp_1",
    "rfc 1468": "iso2022_jp", "iso 2022": "iso2022_jp", "iso-2022-jp": "iso2022_jp",
    "iso-ir 100": "latin-1",
}

_SEX = {1: "M", 2: "F"}



def _read_length(data: bytes, pos: int, end: int) -> tuple[int | None, int]:
    """Decode a length field. Returns (length or None if indefinite, new pos)."""
    if pos >= end:
        raise CorruptedFileError(f"MFER: length field missing at offset {pos}")
    first = data[pos]
    if first < 0x80:
        return first, pos + 1
    if first == 0x80:
        return None, pos + 1
    count = first & 0x7F
    if count > 4:
        raise CorruptedFileError(
            f"MFER: length field at offset {pos} uses {count} bytes (at most 4 allowed)"
        )
    if pos + 1 + count > end:
        raise CorruptedFileError(f"MFER: truncated length field at offset {pos}")
    return int.from_bytes(data[pos + 1:pos + 1 + count], "big"), pos + 1 + count


def _read_tag(data: bytes, pos: int, end: int) -> tuple[int, int | None, int]:
    """Read a tag (and the channel number of MWF_ATT). Returns (tag, channel, pos)."""
    tag = data[pos]
    pos += 1
    channel = None
    if tag == MWF_ATT:
        channel = 0
        while True:
            if pos >= end:
                raise CorruptedFileError("MFER: truncated channel number in MWF_ATT")
            byte = data[pos]
            pos += 1
            channel = (channel << 7) | (byte & 0x7F)
            if not byte & 0x80:
                break
    return tag, channel, pos


def _skip_indefinite(data: bytes, pos: int, end: int) -> tuple[int, int]:
    """Return (content end, position after End-of-Contents) of indefinite content."""
    while pos + 1 < end:
        if data[pos] == 0 and data[pos + 1] == 0:
            return pos, pos + 2
        if data[pos] == MWF_ZRO:
            pos += 1
            continue
        _, _, pos = _read_tag(data, pos, end)
        length, pos = _read_length(data, pos, end)
        if length is None:
            _, pos = _skip_indefinite(data, pos, end)
        else:
            pos += length
    raise CorruptedFileError("MFER: indefinite-length definition has no End-of-Contents")


def _iter_tlv(data: bytes, pos: int, end: int):
    """Yield (tag, channel, value_start, value_end) until MWF_END or *end*."""
    while pos < end:
        if data[pos] == MWF_ZRO:
            pos += 1
            continue
        tag, channel, pos = _read_tag(data, pos, end)
        if tag == MWF_END:
            return
        length, pos = _read_length(data, pos, end)
        if length is None:
            value_end, next_pos = _skip_indefinite(data, pos, end)
        else:
            value_end = next_pos = pos + length
            if value_end > end:
                raise CorruptedFileError(
                    f"MFER: tag 0x{tag:02X} at offset {pos} declares {length} bytes, "
                    f"only {end - pos} remain"
                )
        yield tag, channel, pos, value_end
        pos = next_pos


def _text_codec(name: str) -> str | None:
    key = name.strip().lower()
    part = re.match(r"iso[ _-]?8859(?:[ _-]?(\d+))?", key)
    if part:
        # ISO 8859 parts 1-9 are listed; a bare "ISO 8859" means part 1
        return f"iso8859_{part.group(1) or 1}"
    for prefix, codec in _TEXT_CODECS.items():
        if key.startswith(prefix):
            return codec
    try:
        return codecs.lookup(key).name
    except (LookupError, ValueError):
        return None


class _State:
    """Definitions in force while reading (parent and per-channel)."""

    def __init__(self) -> None:
        self.big_endian = True
        self.encoding: str | None = None
        self.channels = 1
        self.parent: dict = {}
        self.child: dict[int, dict] = {}
        self.pointer: int | None = None

    def effective(self, channel: int) -> dict:
        defs = dict(self.parent)
        if channel != 0:
            defs.pop("ldn", None)  # a parent lead name only applies to channel 1
        defs.update(self.child.get(channel, {}))
        return defs


class MFERParser(Parser):
    """Parser for MFER binary waveform files (ISO 22077-1)."""

    FORMAT_NAME = "MFER"
    FORMAT_DESCRIPTION = "Medical waveform Format Encoding Rules (ISO 22077)"
    FILE_EXTENSIONS = [".mwf", ".mfer", ".mfr"]
    PRIORITY = 90

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        """Accept a MWF_PRE preamble or a chain of valid MFER definitions."""
        if len(header) >= 6 and header[0] == MWF_PRE and header[2:6] == b"MFR ":
            return True
        end = len(header)
        pos = 0
        count = 0
        core = False
        reached_data = False
        try:
            while pos < end and count < 64:
                if header[pos] == MWF_ZRO:
                    pos += 1
                    continue
                tag, _, pos = _read_tag(header, pos, end)
                if tag == MWF_END:
                    break
                if tag not in _KNOWN_TAGS:
                    return False
                length, pos = _read_length(header, pos, end)
                if length is None:
                    if tag not in (MWF_ATT, MWF_SET):
                        return False
                    count += 1
                    core = True
                    break
                value = header[pos:pos + length]
                if not MFERParser._plausible(tag, length, value):
                    return False
                count += 1
                core = core or tag in (
                    MWF_BLE, MWF_BLK, MWF_CHN, MWF_SEQ, MWF_DTP, MWF_IVL, MWF_SEN, MWF_ATT,
                )
                if tag == MWF_WAV:
                    reached_data = True
                    break
                if pos + length > end:
                    break
                pos += length
        except CorruptedFileError:
            return False
        return core and (count >= 3 or (reached_data and count >= 2))

    @staticmethod
    def _plausible(tag: int, length: int, value: bytes) -> bool:
        """Check fixed-size definitions while sniffing."""
        if tag == MWF_BLE:
            return length == 1 and value[:1] in (b"\x00", b"\x01")
        if tag == MWF_VER:
            return length == 3
        if tag == MWF_DTP:
            return length <= 1 and (not value or value[0] <= 9)
        if tag in (MWF_BLK, MWF_CHN, MWF_SEQ, MWF_PNT):
            return length <= 4
        if tag in (MWF_IVL, MWF_SEN):
            return length == 0 or 3 <= length <= 6
        if tag == MWF_SEX:
            return length == 1
        return True

    def parse(self, file_path: Path) -> ECGRecord:
        # Parsing state lives on the instance: work on a copy so that one
        # parser object can be shared between threads
        return copy.copy(self)._parse(file_path)

    def _parse(self, file_path: Path) -> ECGRecord:
        with open(file_path, "rb") as f:
            data = f.read()
        if not data:
            raise CorruptedFileError("MFER: file is empty")

        self._warnings: list[str] = []
        self._record = ECGRecord(source_format="mfer")
        self._meta = self._record.raw_metadata
        self._meta["filepath"] = str(file_path)
        self._patient = PatientInfo()
        self._recording = RecordingInfo()
        self._device = DeviceInfo()
        self._filters = FilterSettings()
        self._comments: list[str] = []
        self._frames: list[dict] = []
        self._age: tuple[int | None, int | None] = (None, None)
        self._state = _State()

        self._walk(data, 0, len(data), None)
        if not self._frames:
            raise CorruptedFileError("MFER: no waveform data (MWF_WAV) found")

        record = self._record
        record.patient = self._patient
        record.recording = self._recording
        record.recording.device = self._device
        record.recording.acquisition.filters = self._filters
        if self._comments:
            record.annotations["comment"] = "\n".join(self._comments)
        self._build_leads()
        self._finish_patient()

        fill_signal_summary(record)
        for message in self._warnings:
            warnings.warn(message, stacklevel=3)
        return record

    # ------------------------------------------------------------------
    # Definitions
    # ------------------------------------------------------------------

    def _uint(self, value: bytes) -> int:
        return int.from_bytes(value, "big" if self._state.big_endian else "little")

    def _sint(self, value: bytes) -> int:
        return int.from_bytes(value, "big" if self._state.big_endian else "little", signed=True)

    def _str(self, value: bytes) -> str:
        return decode_text(value, self._state.encoding).strip("\x00 ")

    def _walk(self, data: bytes, pos: int, end: int, channel: int | None) -> None:
        for tag, att_channel, start, stop in _iter_tlv(data, pos, end):
            value = data[start:stop]
            if tag == MWF_ATT:
                if channel is not None:
                    raise CorruptedFileError("MFER: nested MWF_ATT channel definition")
                if att_channel < self._state.channels:
                    self._walk(data, start, stop, att_channel)
                else:
                    self._warnings.append(
                        f"MFER: attributes for channel {att_channel + 1} ignored, "
                        f"only {self._state.channels} channel(s) defined"
                    )
            elif tag == MWF_SET:
                self._walk(data, start, stop, channel)
            elif tag in _CHANNEL_ATTRS:
                self._define(tag, value, channel)
            elif tag == MWF_WAV:
                self._add_frame(value)
            else:
                self._header(tag, value, channel)

    def _define(self, tag: int, value: bytes, channel: int | None) -> None:
        """Store a sampling/frame definition for the parent or one channel."""
        state = self._state
        target = state.parent if channel is None else state.child.setdefault(channel, {})
        name = _CHANNEL_ATTRS[tag]
        if not value:
            target.pop(name, None)  # zero length resets to the default/parent
            return
        if tag in (MWF_BLK, MWF_SEQ):
            if len(value) > 4:
                raise CorruptedFileError(f"MFER: {name.upper()} longer than 4 bytes")
            target[name] = self._uint(value)
        elif tag == MWF_DTP:
            dtp = value[0]
            if dtp == 9:
                raise UnsupportedFormatError("MFER: AHA compressed data (MWF_DTP 9) is not supported")
            if dtp not in _DATA_TYPES:
                raise CorruptedFileError(f"MFER: unknown data type {dtp} (MWF_DTP)")
            target[name] = dtp
        elif tag in (MWF_IVL, MWF_SEN):
            if len(value) < 3:
                raise CorruptedFileError(f"MFER: {name.upper()} needs unit, exponent and mantissa")
            exponent = int.from_bytes(value[1:2], "big", signed=True)
            target[name] = (value[0], exponent, self._sint(value[2:]))
        elif tag in (MWF_OFF, MWF_NUL):
            target[name] = (value, state.big_endian)
        elif tag == MWF_WFM:
            code = self._uint(value[:2])
            target[name] = code
            if channel is None:
                self._meta["waveform_type"] = code
                description = self._str(value[2:]) if len(value) > 2 else ""
                self._device.acquisition_type = description or _WAVEFORM_TYPES.get(code, "")
        elif tag == MWF_LDN:
            code = value[0] if len(value) == 1 else self._uint(value[:2])
            text = self._str(value[2:]) if len(value) > 2 else ""
            target[name] = (code, text)

    def _header(self, tag: int, value: bytes, channel: int | None) -> None:
        state = self._state
        meta = self._meta
        if tag == MWF_BLE:
            if not value:
                state.big_endian = True
            elif value[:1] not in (b"\x00", b"\x01"):
                raise CorruptedFileError(f"MFER: invalid byte order value {value.hex()}")
            else:
                state.big_endian = value[0] == 0
        elif tag == MWF_CHN and channel is None:
            if not value:
                state.channels = 1
            else:
                if len(value) > 4:
                    raise CorruptedFileError("MFER: CHN longer than 4 bytes")
                state.channels = self._uint(value)
                if state.channels <= 0:
                    raise CorruptedFileError("MFER: number of channels is 0")
            state.child = {}  # a channel count resets channel attributes
        elif tag == MWF_VER:
            self._record.file_format.version = ".".join(str(b) for b in value)
        elif tag == MWF_TXC:
            name = value.decode("ascii", errors="replace").strip("\x00 ")
            meta["character_set"] = name
            state.encoding = _text_codec(name)
        elif tag == MWF_PNT:
            state.pointer = self._sint(value) if value else None
        elif tag == MWF_CMP:
            code = self._uint(value[:2]) if value else 0
            if code != 0:
                raise UnsupportedFormatError(f"MFER: compressed content (MWF_CMP code {code}) is not supported")
        elif tag == MWF_FLT:
            self._add_filter(self._str(value))
        elif tag == MWF_NTE:
            text = self._str(value)
            if text:
                self._comments.append(text)
        elif tag == MWF_MAN:
            parts = self._str(value).split("^") + ["", "", "", ""]
            self._device.manufacturer, self._device.model = parts[0].strip(), parts[1].strip()
            self._device.software_version, self._device.serial_number = parts[2].strip(), parts[3].strip()
        elif tag == MWF_PRE:
            meta["preamble"] = self._str(value)
        elif tag in (MWF_EVT, MWF_INF):
            entry = self._event(value)
            if channel is not None:
                entry["channel"] = channel + 1
            meta.setdefault("events" if tag == MWF_EVT else "info", []).append(entry)
        elif tag == MWF_VAL:
            code = self._uint(value[:2]) if len(value) >= 2 else None
            point = self._sint(value[2:6]) if len(value) >= 6 else None
            text = self._str(value[6:]) if len(value) > 6 else ""
            meta.setdefault("values", []).append({"code": code, "point": point, "value": text})
            if code is not None and text:
                self._record.annotations[f"value_{code}"] = text.replace("^", " ").strip()
        elif tag == MWF_SKW:
            meta.setdefault("sample_skew_ns", []).append(self._uint(value) if value else 0)
        elif tag == MWF_IPD:
            meta.setdefault("interpolation", []).append(value.hex())
        elif tag in (MWF_CND, MWF_RPT, MWF_SIG, MWF_MAP):
            meta.setdefault(f"tag_0x{tag:02x}", []).append(value.hex())
        elif tag == MWF_MSS:
            meta.setdefault("messages", []).append(self._str(value))
        elif tag == MWF_UID:
            meta["uid"] = self._str(value)
        elif tag == MWF_PNM:
            self._patient_name(self._str(value))
        elif tag == MWF_PID:
            text = self._str(value)
            parts = text.split("^")
            self._patient.patient_id = parts[0].strip()
            if len(parts) > 1:
                meta["patient_id_components"] = parts
        elif tag == MWF_AGE:
            self._patient_age(value)
        elif tag == MWF_SEX:
            if value:
                self._patient.sex = _SEX.get(value[0], "U")
        elif tag == MWF_TIM:
            self._measurement_time(value)

    def _event(self, value: bytes) -> dict:
        entry: dict = {}
        if len(value) >= 2:
            entry["code"] = self._uint(value[:2])
        if len(value) >= 6:
            entry["start"] = self._sint(value[2:6])
        if len(value) >= 10:
            entry["duration"] = self._sint(value[6:10])
        if len(value) > 10:
            entry["description"] = self._str(value[10:])
        return entry

    def _add_filter(self, text: str) -> None:
        """Decode MWF_FLT strings such as ``"HPF=0.05"`` or ``"BEF=50^Hum filter"``."""
        if not text:
            self._filters = FilterSettings()  # an empty MWF_FLT resets the filters
            return
        self._meta.setdefault("filters", []).append(text)
        kind, _, rest = text.partition("=")
        try:
            freq = float(rest.split("^")[0].strip().rstrip("Hz").strip())
        except ValueError:
            return
        kind = kind.strip().upper()
        if kind == "HPF":
            self._filters.highpass = freq or None
        elif kind == "LPF":
            self._filters.lowpass = freq or None
        elif kind == "BEF":
            self._filters.notch = freq or None
            self._filters.notch_active = True if freq else None

    def _patient_name(self, text: str) -> None:
        # Recommended form: family^family phonetic^given^given phonetic^middle^middle phonetic
        parts = [p.strip() for p in text.split("^")]
        self._patient.last_name = parts[0]
        if len(parts) > 2:
            self._patient.first_name = parts[2]
        phonetic = [p for p in parts[1::2] if p]
        if phonetic:
            self._meta["patient_name_phonetic"] = " ".join(phonetic)
        if len(parts) > 4 and parts[4]:
            self._meta["patient_middle_name"] = parts[4]

    def _patient_age(self, value: bytes) -> None:
        years = value[0] if len(value) >= 1 else None
        days = self._uint(value[1:3]) if len(value) >= 3 else None
        if len(value) >= 7:
            year = self._uint(value[3:5])
            try:
                if year:
                    self._patient.birth_date = datetime(year, value[5], value[6])
            except ValueError:
                self._warnings.append(f"MFER: invalid birth date {year}-{value[5]}-{value[6]}")
        self._age = (years, days)

    def _measurement_time(self, value: bytes) -> None:
        if len(value) < 4:
            return
        year = self._uint(value[:2])
        fields = list(value[2:7])
        try:
            if len(value) < 7:
                self._meta["acquisition_date"] = datetime(year, fields[0], fields[1]).date().isoformat()
                return
            micro = 0
            if len(value) >= 9:
                micro += self._uint(value[7:9]) * 1000
            if len(value) >= 11:
                micro += self._uint(value[9:11])
            self._recording.date = datetime(year, *fields[:5], min(micro, 999_999))
        except ValueError:
            self._warnings.append(f"MFER: invalid measurement time {value.hex()}")

    def _finish_patient(self) -> None:
        dob = self._patient.birth_date
        acquired = self._recording.date
        years, days = self._age
        if years is not None and (years or days):
            self._patient.age = years
        elif dob is not None and acquired is not None:
            age = acquired.year - dob.year - ((acquired.month, acquired.day) < (dob.month, dob.day))
            self._patient.age = age if age >= 0 else None
        if not self._patient.sex:
            self._patient.sex = "U"

    # ------------------------------------------------------------------
    # Waveform frames
    # ------------------------------------------------------------------

    def _add_frame(self, wav: bytes) -> None:
        state = self._state
        if state.channels > max(len(wav), 1):
            raise CorruptedFileError(
                f"MFER: {state.channels} channels declared for {len(wav)} bytes of waveform data"
            )
        defs = [state.effective(ch) for ch in range(state.channels)]
        order = ">" if state.big_endian else "<"
        dtypes = [np.dtype(_DATA_TYPES[d.get("dtp", 0)]).newbyteorder(order) for d in defs]
        blks = [d.get("blk", 1) for d in defs]
        if any(b <= 0 for b in blks):
            raise CorruptedFileError("MFER: data block length (MWF_BLK) is 0")
        bytes_per_seq = sum(b * dt.itemsize for b, dt in zip(blks, dtypes))

        declared = [d.get("seq") for d in defs]
        if all(s is None for s in declared):
            n_seq = math.ceil(len(wav) / bytes_per_seq)
            seqs = [n_seq] * len(defs)
        else:
            fallback = math.ceil(len(wav) / bytes_per_seq)
            seqs = [s if s is not None else fallback for s in declared]

        channels, missing, omitted = self._decode_frame(wav, blks, dtypes, seqs)
        if missing:
            self._warnings.append(
                f"MFER: waveform data shorter than the frame definition, "
                f"{missing} missing sample(s) set to NaN"
            )
        if omitted:
            self._warnings.append(
                f"MFER: frame declares {omitted} sequence(s) with no waveform data, "
                "they are left out"
            )

        pointer = state.pointer
        expected = sum(len(c) for c in self._frames[-1]["channels"][:1]) if self._frames else 0
        if self._frames:
            expected += self._frames[-1]["pointer"]
        if pointer is not None and self._frames and pointer != expected:
            self._warnings.append(
                f"MFER: frame {len(self._frames) + 1} starts at pointer {pointer}, "
                f"expected {expected}; frames are concatenated without a gap"
            )
        self._frames.append({
            "defs": defs,
            "dtypes": dtypes,
            "channels": channels,
            "pointer": pointer if pointer is not None else expected,
            "wfm": state.parent.get("wfm"),
        })
        state.pointer = None

    @staticmethod
    def _decode_frame(wav: bytes, blks: list[int], dtypes: list[np.dtype], seqs: list[int]):
        """Split frame data into channels: for each sequence, each channel's block.

        Returns (channels, missing samples set to NaN, absent sequences omitted).
        """
        n_ch = len(blks)
        pieces: list[list[np.ndarray]] = [[] for _ in range(n_ch)]
        missing = omitted = 0
        pos = 0
        start_seq = 0
        # Channels may stop early when their SEQ is overridden (Fig. 5-19)
        for stop_seq in sorted(set(seqs)):
            active = [c for c in range(n_ch) if seqs[c] > start_seq]
            count = stop_seq - start_seq
            start_seq = stop_seq
            if count <= 0 or not active:
                continue
            record_size = sum(blks[c] * dtypes[c].itemsize for c in active)
            if record_size > len(wav) + (1 << 20):
                raise CorruptedFileError(
                    f"MFER: frame block of {record_size} bytes is far larger "
                    f"than the {len(wav)} bytes of waveform data"
                )
            record_dtype = np.dtype([(f"c{c}", dtypes[c], (blks[c],)) for c in active])
            available = -(-(len(wav) - pos) // record_dtype.itemsize)
            if available < count:
                # Sequences with no data at all are left out instead of NaN-filled
                omitted += count - max(available, 0)
                count = max(available, 0)
            if count == 0:
                continue
            size = record_dtype.itemsize * count
            chunk = wav[pos:pos + size]
            partial = len(chunk) < size
            if partial:
                # Absent data is processed as "no data" (Part 1 5.2.3(7))
                chunk = chunk + b"\x00" * (size - len(chunk))
            arr = np.frombuffer(chunk, dtype=record_dtype, count=count)
            for c in active:
                values = arr[f"c{c}"].reshape(-1).astype(np.float64)
                if partial:
                    field_offset = record_dtype.fields[f"c{c}"][1]
                    ends = ((np.arange(count) * record_dtype.itemsize)[:, None] + field_offset
                            + (np.arange(blks[c]) + 1) * dtypes[c].itemsize).reshape(-1)
                    absent = ends > len(wav) - pos
                    values[absent] = np.nan
                    missing += int(absent.sum())
                pieces[c].append(values)
            pos += size
        channels = [np.concatenate(p) if p else np.empty(0) for p in pieces]
        return channels, missing, omitted

    def _build_leads(self) -> None:
        record = self._record
        meta = self._meta
        rhythm = [f for f in self._frames if f["wfm"] != _WFM_DOMINANT_BEAT]
        median = [f for f in self._frames if f["wfm"] == _WFM_DOMINANT_BEAT]
        meta["frames"] = [
            {"channels": len(f["channels"]), "samples": len(f["channels"][0]) if f["channels"] else 0,
             "pointer": f["pointer"], "waveform_type": f["wfm"]}
            for f in self._frames
        ]

        groups = []
        for frames in (rhythm, median):
            if not frames:
                groups.append(([], []))
                continue
            # Only consecutive frames with the same channel layout, sampling
            # and scaling form one recording; joining frames that redefine
            # them would apply the first frame's rate and scale to the others
            key = self._frame_key(frames[0])
            run = 1
            while run < len(frames) and self._frame_key(frames[run]) == key:
                run += 1
            if run < len(frames):
                self._warnings.append(
                    f"MFER: {len(frames) - run} frame(s) after frame "
                    f"{next(i for i, f in enumerate(self._frames) if f is frames[run - 1]) + 1} "
                    "redefine the channels, "
                    "sampling or scaling and are not loaded (see raw_metadata['frames'])"
                )
            groups.append(self._frames_to_leads(frames[:run]))

        (leads, rates), (beats, _) = groups
        record.median_beats = beats
        # Pacing bits in a status channel are device detections, not a patient
        # attribute: the channel stays in raw_metadata["status_channels"]
        statuses = meta.get("status_channels", {})
        record.leads = leads
        if not any(len(lead.samples) for lead in leads + beats):
            if any(len(v) for v in statuses.values()):
                raise UnsupportedFormatError(
                    "MFER: the file holds only status channels, no waveform leads"
                )
            raise CorruptedFileError("MFER: waveform frames contain no samples")

        ref = leads or beats
        defs = rhythm[0]["defs"] if rhythm else median[0]["defs"]
        dtp = defs[0].get("dtp", 0) if defs else 0
        record.recording.acquisition.signal = SignalCharacteristics(
            sampling_rate=ref[0].sampling_rate if ref else 0,
            bits_per_sample=np.dtype(_DATA_TYPES[dtp]).itemsize * 8,
            signal_signed=_DATA_TYPES[dtp][0] in "if",
            number_channels_allocated=len(defs),
            number_channels_valid=len(leads),
            data_encoding=_DATA_ENCODINGS[dtp],
            compression="none",
        )
        if leads and rates[0] > 0:
            meta["sampling_rates"] = rates
            duration = timedelta(seconds=len(leads[0].samples) / rates[0])
            record.recording.duration = duration

    @staticmethod
    def _frame_key(frame: dict) -> tuple:
        """Definitions that must match for frames to be concatenated."""
        names = ("dtp", "ivl", "sen", "off", "nul", "ldn", "wfm")
        return tuple(tuple(d.get(n) for n in names) for d in frame["defs"])

    def _frames_to_leads(self, frames: list[dict]) -> tuple[list[Lead], list[float]]:
        """Concatenate frames per channel; returns the leads and exact rates (Hz)."""
        if not frames:
            return [], []
        base = frames[0]
        leads: list[Lead] = []
        labels: list[str] = []
        rates: list[float] = []
        for ch, defs in enumerate(base["defs"]):
            samples = np.concatenate([f["channels"][ch] for f in frames])
            dtype = base["dtypes"][ch]
            if "nul" in defs:
                null = self._typed_value(defs["nul"], dtype)
                if null is not None:
                    samples[samples == null] = np.nan

            code, text = defs.get("ldn", (None, ""))
            label = _LEAD_CODES.get(code, "") if code is not None else ""
            label = label or text or (f"Code{code}" if code else f"Ch{ch + 1}")

            if defs.get("dtp") == _DTP_STATUS or code == _LDN_STATUS:
                self._meta.setdefault("status_channels", {})[label] = samples
                continue

            if defs.get("ivl") is None:
                # 1000 Hz is the MFER default when MWF_IVL is absent
                self._meta["sampling_rate_stated"] = False
                msg = "MFER: no sampling interval (MWF_IVL), using the MFER default of 1000 Hz"
                if msg not in self._warnings:
                    self._warnings.append(msg)
            rate = self._sampling_rate(defs.get("ivl"))
            if round(rate) < 1:
                raise UnsupportedFormatError(
                    f"MFER: sampling rate {rate:g} Hz is below 1 Hz and cannot be "
                    "represented as an integer sampling rate"
                )
            resolution, unit, adc, adc_unit = self._scale(defs.get("sen"))
            offset_counts = self._typed_value(defs["off"], dtype) if "off" in defs else 0.0
            annotations: dict[str, str] = {}
            if unit:
                offset = float(offset_counts or 0.0) * resolution
            else:
                offset = 0.0
                if offset_counts:
                    annotations["offset_counts"] = f"{offset_counts:g}"
            if defs.get("wfm") is not None and defs.get("wfm") != base["wfm"]:
                annotations["waveform_type"] = str(defs["wfm"])
            if code is not None:
                annotations["lead_code"] = str(code)
            is_raw = derive_is_raw(resolution, offset, unit)
            leads.append(Lead(
                label=label,
                samples=samples,
                sampling_rate=int(round(rate)),
                resolution=resolution,
                resolution_unit=unit,
                offset=offset,
                units="" if is_raw else unit,
                is_raw=is_raw,
                adc_resolution=adc,
                adc_resolution_unit=adc_unit,
                annotations=annotations,
            ))
            labels.append(label)
            rates.append(rate)
        for lead, unique in zip(leads, unique_labels(labels)):
            lead.label = unique
        return leads, rates

    @staticmethod
    def _typed_value(defn: tuple[bytes, bool], dtype: np.dtype) -> float | None:
        raw, big = defn
        dt = dtype.newbyteorder(">" if big else "<")
        if len(raw) < dt.itemsize:
            return None
        return float(np.frombuffer(raw[:dt.itemsize], dtype=dt)[0])

    @staticmethod
    def _sampling_rate(ivl) -> float:
        """MWF_IVL to Hz (default 1000 Hz)."""
        if ivl is None:
            return 1000.0
        unit, exponent, mantissa = ivl
        value = mantissa * 10.0 ** exponent
        if unit not in (0, 1):
            raise UnsupportedFormatError(f"MFER: sampling unit code {unit} is not a time base")
        rate = value if unit == 0 else (1.0 / value if value > 0 else 0.0)
        if not 1e-6 <= rate <= 1e9:
            raise CorruptedFileError(f"MFER: invalid sampling interval/frequency (MWF_IVL {ivl})")
        return rate

    @staticmethod
    def _scale(sen) -> tuple[float, str, float, str]:
        """MWF_SEN to (resolution, resolution_unit, adc_resolution, adc_unit)."""
        if sen is None:
            return 1.0, "", 0.0, ""
        unit, exponent, mantissa = sen
        value = mantissa * 10.0 ** exponent
        if unit != 0:
            # Not a voltage: keep the scale for reference only
            return 1.0, "", value, _SEN_UNITS.get(unit, f"unit {unit}")
        for name, base in (("nV", -9), ("uV", -6), ("mV", -3), ("V", 0)):
            if exponent <= base or name == "V":
                adc = mantissa * 10.0 ** (exponent - base)
                adc_unit = name
                break
        return round(value * 1e6, 12), "uV", adc, adc_unit

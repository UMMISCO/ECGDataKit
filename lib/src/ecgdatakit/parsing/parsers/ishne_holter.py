"""ISHNE Holter binary format parser.

Reference: http://thew-project.org/papers/Badilini.ISHNE.Holter.Standard.pdf
"""

from __future__ import annotations

import datetime
import os
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from crccheck.crc import Crc16CcittFalse
from ecgdatakit.exceptions import CorruptedFileError
from ecgdatakit.models import (
    DeviceInfo,
    ECGRecord,
    Lead,
    PatientInfo,
    RecordingInfo,
    SignalCharacteristics,
)
from ecgdatakit.parsing.parser import Parser

_MAGIC_ECG = b"ISHNE1.0"

_FIXED_HEADER_SIZE = 522
_HEADER_RECORD_OFFSET = 10

_LEAD_SPECS: dict[int, str] = {
    -9: "absent", 0: "unknown", 1: "generic",
    2: "X",  3: "Y",  4: "Z",
    5: "I",  6: "II",  7: "III",
    8: "aVR", 9: "aVL", 10: "aVF",
    11: "V1", 12: "V2", 13: "V3",
    14: "V4", 15: "V5", 16: "V6",
    17: "ES", 18: "AS", 19: "AI",
}

pm_codes = {
    0: 'none',
    1: 'unknown type',
    2: 'single chamber unipolar',
    3: 'dual chamber unipolar',
    4: 'single chamber bipolar',
    5: 'dual chamber bipolar',
}

#----------------------------------------------------------------------------------------------
# Buffer field readers (little-endian)
#----------------------------------------------------------------------------------------------

def _i16(buf: bytes, ptr: int) -> int:
    return struct.unpack_from("<h", buf, ptr)[0]


def _i32(buf: bytes, ptr: int) -> int:
    return struct.unpack_from("<i", buf, ptr)[0]


def _u16(buf: bytes, ptr: int) -> int:
    return struct.unpack_from("<H", buf, ptr)[0]


def _text(buf: bytes, ptr: int, size: int) -> str:
    return buf[ptr:ptr + size].split(b"\x00")[0].decode("ascii", errors="replace")


def _date(buf: bytes, ptr: int) -> datetime.date | None:
    day, month, year = (_i16(buf, ptr + 2 * i) for i in range(3))
    try:
        return datetime.date(year, month, day)
    except ValueError:
        return None


def _time(buf: bytes, ptr: int) -> datetime.time | None:
    hour, minute, second = (_i16(buf, ptr + 2 * i) for i in range(3))
    try:
        return datetime.time(hour, minute, second)
    except ValueError:
        return None

#----------------------------------------------------------------------------------------------
# Parsed sub-structures
#----------------------------------------------------------------------------------------------

@dataclass
class _Layout:
    """Sizes/offsets from the fixed header that locate the rest of the file."""
    checksum: int
    var_block_size: int
    ecg_size: int  # declared sample count (see _read_signal for caveat)
    var_block_offset: int
    ecg_block_offset: int


@dataclass
class _LeadMeta:
    nleads: int
    spec: list[int]
    quality: list[int]
    ampl_res: list[int]  # amplitude resolution per lead, in nV/LSB
    sampling_rate: int
    pacemaker_code: int

#----------------------------------------------------------------------------------------------
# Parser
#----------------------------------------------------------------------------------------------

class ISHNEHolterParser(Parser):
    """Parser for ISHNE Holter binary ECG files."""

    FORMAT_NAME = "ISHNE Holter"
    FORMAT_DESCRIPTION = "ISHNE Holter standard binary format"
    FILE_EXTENSIONS = [".ecg"]

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        return header[:8] == _MAGIC_ECG

    def parse(self, file_path: Path) -> ECGRecord:
        filename = str(file_path)

        header = self._read_fixed_header(filename)
        layout = self._parse_layout(header)
        self._verify_checksum(filename, layout)

        lead_meta = self._parse_lead_metadata(header)

        record = ECGRecord(source_format="ishne_holter")
        record.patient = self._parse_patient(header)
        record.recording = self._parse_recording(header)

        signal = self._read_signal(filename, layout, lead_meta.nleads)
        record.leads = self._build_leads(signal, lead_meta)
        # The base ISHNE .ecg format carries no median/representative beats
        record.median_beats = []
        self._finalize_recording(record, lead_meta)

        var_block_hex = self._read_variable_block(filename, layout)
        self._build_metadata(record, filename, layout, lead_meta, var_block_hex)
        return record

    @staticmethod
    def _read_fixed_header(filename: str) -> bytes:
        size = os.path.getsize(filename)
        if size < _FIXED_HEADER_SIZE:
            raise CorruptedFileError(f"File too small to be ISHNE Holter: {size} bytes")
        with open(filename, "rb") as f:
            buf = f.read(_FIXED_HEADER_SIZE)
        if len(buf) < _FIXED_HEADER_SIZE:
            raise CorruptedFileError("Truncated ISHNE header")
        if buf[:8] != _MAGIC_ECG:
            raise CorruptedFileError(f"Bad ISHNE magic number: {buf[:8]!r}")
        return buf

    @staticmethod
    def _parse_layout(buf: bytes) -> _Layout:
        return _Layout(
            checksum=_u16(buf, 8),
            var_block_size=_i32(buf, 10),
            ecg_size=_i32(buf, 14),
            var_block_offset=_i32(buf, 18),
            ecg_block_offset=_i32(buf, 22),
        )

    def _verify_checksum(self, filename: str, layout: _Layout) -> None:
        stored = layout.checksum
        if stored == 0:
            return  # 0 = writer did not compute a checksum, nothing to verify

        end = layout.ecg_block_offset
        size = os.path.getsize(filename)
        if end <= _HEADER_RECORD_OFFSET or end > size:
            warnings.warn(f"Cannot verify checksum: bad ecg_block_offset={end}", stacklevel=2)
            return

        with open(filename, "rb") as f:
            f.seek(_HEADER_RECORD_OFFSET)
            block = f.read(end - _HEADER_RECORD_OFFSET)
        computed = int(Crc16CcittFalse.calc(block))
        if computed != stored:
            msg = (
                f"ISHNE checksum mismatch: stored={stored:#06x} "
                f"computed={computed:#06x}"
            )
            #raise ChecksumError(msg)
            warnings.warn(msg, stacklevel=2)

    def _parse_patient(self, buf: bytes) -> PatientInfo:
        patient = PatientInfo()
        patient.first_name = _text(buf, 28, 40)
        patient.last_name = _text(buf, 68, 40)
        patient.patient_id = _text(buf, 108, 20)
        patient.sex = {1: "M", 2: "F"}.get(_i16(buf, 128), "U")
        birth = _date(buf, 132)
        if birth is not None:
            patient.birth_date = datetime.datetime.combine(birth, datetime.time.min)
        return patient

    def _parse_recording(self, buf: bytes) -> RecordingInfo:
        recording = RecordingInfo()
        rec_date = _date(buf, 138)
        start_time = _time(buf, 150)
        if rec_date is not None and start_time is not None:
            recording.date = datetime.datetime.combine(rec_date, start_time)
        recording.device = DeviceInfo(model=_text(buf, 232, 40))
        return recording

    def _parse_lead_metadata(self, buf: bytes) -> _LeadMeta:
        nleads = _i16(buf, 156)
        if nleads <= 0 or nleads > 12:
            raise CorruptedFileError(f"Invalid lead count: {nleads}")
        return _LeadMeta(
            nleads=nleads,
            spec=[_i16(buf, 158 + 2 * i) for i in range(12)],
            quality=[_i16(buf, 182 + 2 * i) for i in range(12)],
            ampl_res=[_i16(buf, 206 + 2 * i) for i in range(12)],
            pacemaker_code=_i16(buf, 230),
            sampling_rate=_i16(buf, 272),
        )

    def _read_signal(self, filename: str, layout: _Layout, nleads: int) -> np.ndarray:
        with open(filename, "rb") as f:
            f.seek(layout.ecg_block_offset)
            data_bytes = f.read()
        data_bytes = data_bytes[: len(data_bytes) // 2 * 2]
        raw = np.frombuffer(data_bytes, dtype="<i2")

        available = raw.size
        expected = layout.ecg_size
        # ecg_size is sometimes stored as total samples, sometimes per-lead.
        # Accept either and only warn if it matches neither.
        if expected > 0 and available not in (expected, expected * nleads):
            warnings.warn(
                f"ISHNE sample-count mismatch: header={expected} "
                f"(or x{nleads} leads), found {available}",
                stacklevel=2,
            )

        usable = (available // nleads) * nleads  # drop any trailing partial frame
        return np.reshape(raw[:usable], (nleads, usable // nleads), order="F")

    def _build_leads(self, signal: np.ndarray, meta: _LeadMeta) -> list[Lead]:
        leads: list[Lead] = []
        for i in range(meta.nleads):
            res_nv = meta.ampl_res[i]
            has_res = res_nv > 0 # is resolution present ?
            res_unit = "uV" if has_res else ""
            label = _LEAD_SPECS.get(meta.spec[i], f"Lead {i + 1}")
            leads.append(Lead(
                label=label,
                samples=signal[i].astype(np.float64),
                sampling_rate=meta.sampling_rate,
                resolution=(res_nv / 1_000.0) if has_res else 1.0,
                resolution_unit=res_unit,
                offset=0.0,
                adc_resolution=float(res_nv),
                adc_resolution_unit="nV" if has_res else "",
                quality=meta.quality[i],
                # Samples stay as raw ADC counts whenever a scale factor exists.
                units="" if has_res else res_unit,
                is_raw=has_res,
            ))
        return leads

    def _finalize_recording(self, record: ECGRecord, meta: _LeadMeta) -> None:
        sr = meta.sampling_rate
        if record.leads and sr > 0:
            n_samples = len(record.leads[0].samples)
            record.recording.duration = datetime.timedelta(seconds=n_samples / sr)

    def _read_variable_block(self, filename: str, layout: _Layout) -> str:
        if layout.var_block_size <= 0:
            return ""
        with open(filename, "rb") as f:
            f.seek(layout.var_block_offset)
            return f.read(layout.var_block_size).hex()

    def _build_metadata(self, record: ECGRecord, filename: str, layout: _Layout, meta: _LeadMeta, var_block_hex: str,) -> None:
        record.recording.acquisition.signal = SignalCharacteristics(
            sampling_rate=meta.sampling_rate,
            bits_per_sample=16,
            signal_signed=True,
            number_channels_allocated=meta.nleads,
            number_channels_valid=len(record.leads),
            data_encoding="int16",
            compression="none",
        )

        raw = record.raw_metadata
        raw["filepath"] = filename
        raw["var_block_size"] = layout.var_block_size
        raw["ecg_size"] = layout.ecg_size
        raw["checksum"] = layout.checksum
        raw["lead_quality"] = meta.quality[: meta.nleads]
        raw["pacemaker_code"] = meta.pacemaker_code
        raw["recorder_type"] = record.recording.device.model
        if var_block_hex:
            raw["variable_block"] = var_block_hex
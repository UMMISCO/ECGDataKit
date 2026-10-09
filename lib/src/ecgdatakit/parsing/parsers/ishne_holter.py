"""ISHNE Holter binary format parser.

Reference: Badilini F. The ISHNE Holter Standard Output File Format.
Ann Noninvasive Electrocardiol. 1998;3(3):263-266.
"""

from __future__ import annotations

import binascii
import datetime
import os
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ecgdatakit.exceptions import ChecksumWarning, CorruptedFileError
from ecgdatakit.models import (
    DeviceInfo,
    ECGRecord,
    FileFormatInfo,
    Lead,
    PatientInfo,
    RecordingInfo,
    SignalCharacteristics,
    derive_is_raw,
)
from ecgdatakit.parsing.helpers import unique_labels
from ecgdatakit.parsing.parser import Parser

_MAGIC_ECG = b"ISHNE1.0"

_FIXED_HEADER_SIZE = 522  # magic (8) + checksum (2) + fixed block (512)
_HEADER_RECORD_OFFSET = 10

# Spec Table 1. Codes 0 (unknown) and 1 (generic bipolar) carry no lead name.
_LEAD_SPECS: dict[int, str] = {
    2: "X",  3: "Y",  4: "Z",
    5: "I",  6: "II",  7: "III",
    8: "aVR", 9: "aVL", 10: "aVF",
    11: "V1", 12: "V2", 13: "V3",
    14: "V4", 15: "V5", 16: "V6",
    17: "ES", 18: "AS", 19: "AI",
}

_ABSENT = -9  # spec value for nonpresent leads

# Spec Table 2
_QUALITY_CODES: dict[int, str] = {
    0: "unrated",
    1: "good",
    2: "intermittent noise",
    3: "frequent noise",
    4: "intermittent disconnection",
    5: "frequent disconnection",
}

_PACEMAKER_CODES: dict[int, str] = {
    1: "unknown type",
    2: "single chamber unipolar",
    3: "dual chamber unipolar",
    4: "single chamber bipolar",
    5: "dual chamber bipolar",
}

_RACE_CODES: dict[int, str] = {1: "Caucasian", 2: "Black", 3: "Oriental"}

# Sample value proposed by the spec to flag lead fault
_FAULT_SAMPLE = -32768

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
    # Spec strings use the extended 256-char set. Latin-1 decodes every byte losslessly.
    return buf[ptr:ptr + size].split(b"\x00")[0].decode("latin-1").strip()


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
    ecg_size: int  # declared samples per lead, often nominal (see _read_signal)
    var_block_offset: int
    ecg_block_offset: int
    data_offset: int = 0  # validated start of ECG data, set by _resolve_data_offset
    checksum_valid: bool | None = None  # None when no checksum is stored


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
    PRIORITY = 10

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        return header[:8] == _MAGIC_ECG

    def parse(self, file_path: Path) -> ECGRecord:
        filename = str(file_path)

        header = self._read_fixed_header(filename)
        layout = self._parse_layout(header)
        self._resolve_data_offset(filename, layout)
        self._verify_checksum(filename, layout)

        lead_meta = self._parse_lead_metadata(header)

        record = ECGRecord(source_format="ishne_holter")
        record.patient = self._parse_patient(header, lead_meta)
        record.recording = self._parse_recording(header)
        birth, rec_date = record.patient.birth_date, record.recording.date
        if birth is not None and rec_date is not None:
            # The format has no age field, derive it from the dates
            record.patient.age = rec_date.year - birth.year - (
                (rec_date.month, rec_date.day) < (birth.month, birth.day)
            )
        record.file_format = FileFormatInfo(
            version=str(_i16(header, 26)),
            creation_date=_date(header, 144),
        )

        signal = self._read_signal(filename, layout, lead_meta.nleads)
        record.leads = self._build_leads(signal, lead_meta)
        # The base ISHNE .ecg format carries no median/representative beats
        record.median_beats = []
        self._finalize_recording(record, lead_meta)

        comments = self._read_variable_block(filename, layout)
        if comments:
            record.annotations["comments"] = comments
        self._build_metadata(record, filename, header, layout, lead_meta, signal)
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

    @staticmethod
    def _resolve_data_offset(filename: str, layout: _Layout) -> None:
        # The spec fixes the variable block at 522 and the ECG data right after it.
        # Like ECG-Kit, trust 522 + variable block size over the stored ECG offset.
        size = os.path.getsize(filename)
        if layout.var_block_size < 0:
            raise CorruptedFileError(
                f"Invalid ISHNE header: variable block size is negative ({layout.var_block_size})"
            )
        offset = _FIXED_HEADER_SIZE + layout.var_block_size
        if offset > size:
            raise CorruptedFileError(
                f"Invalid ISHNE header: variable block of {layout.var_block_size} bytes "
                f"ends at byte {offset}, past the end of the file ({size} bytes)"
            )
        if layout.var_block_offset != _FIXED_HEADER_SIZE:
            warnings.warn(
                f"ISHNE: variable block offset is {layout.var_block_offset}, "
                f"spec requires {_FIXED_HEADER_SIZE}. Using {_FIXED_HEADER_SIZE}",
                stacklevel=3,
            )
        if layout.ecg_block_offset != offset:
            warnings.warn(
                f"ISHNE: ECG block offset is {layout.ecg_block_offset}, expected {offset} "
                f"(522 + variable block size). Using {offset}",
                stacklevel=3,
            )
        layout.data_offset = offset

    def _verify_checksum(self, filename: str, layout: _Layout) -> None:
        stored = layout.checksum
        if stored == 0:
            return  # 0 = writer did not compute a checksum, nothing to verify

        with open(filename, "rb") as f:
            f.seek(_HEADER_RECORD_OFFSET)
            block = f.read(layout.data_offset - _HEADER_RECORD_OFFSET)
        computed = binascii.crc_hqx(block, 0xFFFF)  # CRC-CCITT, init 0xFFFF
        # Spec does not fix the byte order of the stored CRC, accept both
        swapped = ((computed & 0xFF) << 8) | (computed >> 8)
        layout.checksum_valid = stored in (computed, swapped)
        if not layout.checksum_valid:
            msg = (
                f"ISHNE checksum mismatch: stored={stored:#06x} "
                f"computed={computed:#06x}"
            )
            #raise ChecksumError(msg)
            warnings.warn(msg, ChecksumWarning, stacklevel=3)

    def _parse_patient(self, buf: bytes, meta: _LeadMeta) -> PatientInfo:
        patient = PatientInfo()
        patient.first_name = _text(buf, 28, 40)
        patient.last_name = _text(buf, 68, 40)
        patient.patient_id = _text(buf, 108, 20)
        patient.sex = {1: "M", 2: "F"}.get(_i16(buf, 128), "U")
        patient.race = _RACE_CODES.get(_i16(buf, 130), "")
        birth = _date(buf, 132)
        if birth is not None:
            patient.birth_date = datetime.datetime.combine(birth, datetime.time.min)
        code = meta.pacemaker_code
        if code == 0:
            patient.has_pacemaker = False
        elif code in _PACEMAKER_CODES:
            patient.has_pacemaker = True
            patient.pacemaker_type = _PACEMAKER_CODES[code]
        return patient

    def _parse_recording(self, buf: bytes) -> RecordingInfo:
        recording = RecordingInfo()
        rec_date = _date(buf, 138)
        start_time = _time(buf, 150)
        if rec_date is not None and start_time is not None:
            recording.date = datetime.datetime.combine(rec_date, start_time)
        # Spec defines this field as "analog" or "digital", many devices store their model here
        recorder = _text(buf, 232, 40)
        if recorder.lower() in ("analog", "digital"):
            recording.device = DeviceInfo(acquisition_type=recorder.lower())
        else:
            recording.device = DeviceInfo(model=recorder)
        return recording

    def _parse_lead_metadata(self, buf: bytes) -> _LeadMeta:
        nleads = _i16(buf, 156)
        if nleads <= 0 or nleads > 12:
            raise CorruptedFileError(
                f"Invalid ISHNE header: lead count is {nleads}, spec allows 1 to 12"
            )
        sampling_rate = _i16(buf, 272)
        if sampling_rate <= 0:
            raise CorruptedFileError(
                f"Invalid ISHNE header: sampling rate is {sampling_rate} Hz, must be positive"
            )
        return _LeadMeta(
            nleads=nleads,
            spec=[_i16(buf, 158 + 2 * i) for i in range(12)],
            quality=[_i16(buf, 182 + 2 * i) for i in range(12)],
            ampl_res=[_i16(buf, 206 + 2 * i) for i in range(12)],
            pacemaker_code=_i16(buf, 230),
            sampling_rate=sampling_rate,
        )

    def _read_signal(self, filename: str, layout: _Layout, nleads: int) -> np.ndarray:
        with open(filename, "rb") as f:
            f.seek(layout.data_offset)
            data_bytes = f.read()
        raw = np.frombuffer(data_bytes, dtype="<i2", count=len(data_bytes) // 2)
        if len(data_bytes) % 2:
            warnings.warn(
                "ISHNE: odd number of data bytes, the last byte is ignored. "
                "File may be truncated",
                stacklevel=3,
            )

        available = raw.size
        usable = (available // nleads) * nleads  # whole multiplexed frames only
        # The spec defines ecg_size as samples per lead, but writers differ and some
        # store a fixed maximum, so it is not checked here. Both counts are kept in
        # raw_metadata. Only an incomplete final frame is reported as truncation.
        dropped = available - usable
        if dropped:
            warnings.warn(
                f"ISHNE: incomplete final frame, dropped {dropped} trailing "
                f"sample(s) not divisible by {nleads} leads. File may be truncated",
                stacklevel=3,
            )
        if usable == 0:
            raise CorruptedFileError(
                f"ISHNE file holds no complete sample frame for {nleads} leads"
            )
        return np.reshape(raw[:usable], (nleads, usable // nleads), order="F")

    def _build_leads(self, signal: np.ndarray, meta: _LeadMeta) -> list[Lead]:
        leads: list[Lead] = []
        labels = unique_labels([self._lead_label(meta.spec[i], i) for i in range(meta.nleads)])
        for i in range(meta.nleads):
            res_nv = meta.ampl_res[i]
            has_res = res_nv > 0 # is resolution present ?
            res_unit = "uV" if has_res else ""
            resolution = (res_nv / 1_000.0) if has_res else 1.0
            # Physical only when the unit is known and scaling is a no-op
            # (e.g. 1000 nV = 1.0 uV/count). Leads with a real scale factor,
            # or with no resolution at all, stay raw ADC counts.
            is_raw = derive_is_raw(resolution, 0.0, res_unit)
            leads.append(Lead(
                label=labels[i],
                samples=signal[i].astype(np.float64),
                sampling_rate=meta.sampling_rate,
                resolution=resolution,
                resolution_unit=res_unit,
                offset=0.0,
                adc_resolution=float(res_nv),
                adc_resolution_unit="nV" if has_res else "",
                quality=meta.quality[i],
                units="" if is_raw else res_unit,
                is_raw=is_raw,
            ))
        return leads

    @staticmethod
    def _lead_label(code: int, index: int) -> str:
        if code == _ABSENT:
            warnings.warn(
                f"ISHNE: stored lead {index + 1} has spec code -9 (not present)",
                stacklevel=4,
            )
        return _LEAD_SPECS.get(code, f"Lead {index + 1}")

    def _finalize_recording(self, record: ECGRecord, meta: _LeadMeta) -> None:
        if record.leads:
            n_samples = len(record.leads[0].samples)
            record.recording.duration = datetime.timedelta(seconds=n_samples / meta.sampling_rate)

    def _read_variable_block(self, filename: str, layout: _Layout) -> str:
        # Free-text comments in the extended 256-char set, per spec
        if layout.var_block_size == 0:
            return ""
        with open(filename, "rb") as f:
            f.seek(_FIXED_HEADER_SIZE)
            block = f.read(layout.var_block_size)
        return block.split(b"\x00")[0].decode("latin-1").strip()

    def _build_metadata(
        self,
        record: ECGRecord,
        filename: str,
        header: bytes,
        layout: _Layout,
        meta: _LeadMeta,
        signal: np.ndarray,
    ) -> None:
        record.recording.acquisition.signal = SignalCharacteristics(
            sampling_rate=meta.sampling_rate,
            bits_per_sample=16,
            signal_signed=True,
            number_channels_allocated=meta.nleads,
            number_channels_valid=len(record.leads),
            data_encoding="int16",
            compression="none",
        )

        quality = meta.quality[: meta.nleads]
        raw = record.raw_metadata
        raw["filepath"] = filename
        raw["var_block_size"] = layout.var_block_size
        raw["ecg_size"] = layout.ecg_size
        raw["samples_per_lead"] = signal.shape[1]
        raw["checksum"] = layout.checksum
        raw["checksum_valid"] = layout.checksum_valid
        raw["lead_spec"] = meta.spec[: meta.nleads]
        raw["lead_quality"] = quality
        raw["lead_quality_desc"] = [_QUALITY_CODES.get(q, "") for q in quality]
        raw["lead_fault_samples"] = [int(np.count_nonzero(row == _FAULT_SAMPLE)) for row in signal]
        raw["pacemaker_code"] = meta.pacemaker_code
        raw["recorder_type"] = _text(header, 232, 40)
        raw["proprietary"] = _text(header, 274, 80)
        raw["copyright"] = _text(header, 354, 80)
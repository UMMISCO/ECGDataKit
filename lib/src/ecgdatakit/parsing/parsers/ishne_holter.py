"""ISHNE Holter binary format parser.

Reference: http://thew-project.org/papers/Badilini.ISHNE.Holter.Standard.pdf
"""

from __future__ import annotations

import datetime
import os
import struct
from pathlib import Path

import numpy as np

from ecgdatakit.exceptions import ChecksumError, CorruptedFileError
from ecgdatakit.models import DeviceInfo, ECGRecord, FilterSettings, Lead, PatientInfo, RecordingInfo, SignalCharacteristics
from ecgdatakit.parsing.parser import Parser

_MAGIC_ECG = b"ISHNE1.0"
_MAGIC_ANN = b"ANN  1.0"


def _get_val(filename: str, ptr: int, datatype: type | str) -> object:
    """Read a single value of the given dtype at file offset *ptr*."""
    with open(filename, "rb") as f:
        f.seek(ptr, os.SEEK_SET)
        val = np.fromfile(f, dtype=datatype, count=1)
        return val[0]


def _get_short_int(filename: str, ptr: int) -> int:
    return int(_get_val(filename, ptr, np.int16))


def _get_long_int(filename: str, ptr: int) -> int:
    return int(_get_val(filename, ptr, np.int32))


def _get_datetime(filename: str, offset: int, time: bool = False) -> datetime.date | datetime.time | None:
    a, b, c = [_get_short_int(filename, offset + 2 * i) for i in range(3)]
    try:
        if time:
            return datetime.time(a, b, c)
        return datetime.date(c, b, a)
    except ValueError:
        return None


_LEAD_SPECS: dict[int, str] = {
    -9: "absent", 0: "unknown", 1: "generic",
    2: "X",  3: "Y",  4: "Z",
    5: "I",  6: "II",  7: "III",
    8: "aVR", 9: "aVL", 10: "aVF",
    11: "V1", 12: "V2", 13: "V3",
    14: "V4", 15: "V5", 16: "V6",
    17: "ES", 18: "AS", 19: "AI",
}


class ISHNEHolterParser(Parser):
    """Parser for ISHNE Holter binary ECG files."""

    FORMAT_NAME = "ISHNE Holter"
    FORMAT_DESCRIPTION = "ISHNE Holter standard binary format"
    FILE_EXTENSIONS = [".ecg", ".ann"]

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        return header[:8] in (_MAGIC_ECG, _MAGIC_ANN)

    def parse(self, file_path: Path) -> ECGRecord:
        filename = str(file_path)
        file_size = os.path.getsize(filename)
        if file_size < 522:
            raise CorruptedFileError(f"File too small to be ISHNE Holter: {file_size} bytes")

        record = ECGRecord(source_format="ishne_holter")

        magic = _get_val(filename, 0, "S8")
        is_annfile = (magic == _MAGIC_ANN)
        checksum = _get_val(filename, 8, np.uint16)

        var_block_size = _get_long_int(filename, 10)
        ecg_size = _get_long_int(filename, 14)
        var_block_offset = _get_long_int(filename, 18)
        ecg_block_offset = _get_long_int(filename, 22)

        nleads = _get_short_int(filename, 156)
        sr = _get_short_int(filename, 272)

        lead_spec = [_get_short_int(filename, 158 + i * 2) for i in range(12)]
        lead_quality = [_get_short_int(filename, 182 + i * 2) for i in range(12)]
        ampl_res = [_get_short_int(filename, 206 + i * 2) for i in range(12)]

        patient = PatientInfo()
        first_name = _get_val(filename, 28, "S40")
        if isinstance(first_name, bytes):
            patient.first_name = first_name.split(b"\x00")[0].decode("ascii", errors="replace")
        last_name = _get_val(filename, 68, "S40")
        if isinstance(last_name, bytes):
            patient.last_name = last_name.split(b"\x00")[0].decode("ascii", errors="replace")
        patient_id = _get_val(filename, 108, "S20")
        if isinstance(patient_id, bytes):
            patient.patient_id = patient_id.split(b"\x00")[0].decode("ascii", errors="replace")

        sex_code = _get_short_int(filename, 128)
        patient.sex = "M" if sex_code == 1 else "F" if sex_code == 2 else "U"

        birth_date = _get_datetime(filename, 132)
        if isinstance(birth_date, datetime.date):
            patient.birth_date = datetime.datetime.combine(birth_date, datetime.time.min)

        record.patient = patient

        recording = RecordingInfo()

        record_date = _get_datetime(filename, 138)
        start_time = _get_datetime(filename, 150, time=True)
        if isinstance(record_date, datetime.date) and isinstance(start_time, datetime.time):
            recording.date = datetime.datetime.combine(record_date, start_time)

        device_raw = _get_val(filename, 232, "S40")
        if isinstance(device_raw, bytes):
            device_name = device_raw.split(b"\x00")[0].decode("ascii", errors="replace")
        else:
            device_name = ""

        record.recording = recording
        record.recording.device = DeviceInfo(model=device_name)

        if nleads <= 0 or nleads > 12:
            raise CorruptedFileError(f"Invalid lead count: {nleads}")

        with open(filename, "rb") as f:
            f.seek(ecg_block_offset, os.SEEK_SET)
            data = np.fromfile(f, dtype=np.int16)

        total = len(data)
        samples_per_lead = total // nleads
        data = data[: samples_per_lead * nleads]
        data = np.reshape(data, (nleads, samples_per_lead), order="F")

        for i in range(nleads):
            spec = lead_spec[i] if i < len(lead_spec) else 0
            res_nv = ampl_res[i] if i < len(ampl_res) else 1

            label = _LEAD_SPECS.get(spec, f"Lead {i + 1}")
            samples = data[i].astype(np.float64)

            lq = lead_quality[i] if i < len(lead_quality) else None
            raw_res = float(res_nv)
            res = raw_res / 1_000.0 if res_nv > 0 else 1.0
            res_unit = "uV" if res_nv > 0 else ""
            raw = res != 1.0
            record.leads.append(Lead(
                label=label,
                samples=samples,
                sample_rate=sr,
                resolution=res,
                resolution_unit=res_unit,
                adc_resolution=raw_res,
                adc_resolution_unit="nV" if res_nv > 0 else "",
                quality=lq,
                units="" if raw else res_unit,
                is_raw=raw,
            ))

        if record.leads:
            duration_s = len(record.leads[0].samples) / sr if sr > 0 else 0
            recording.duration = datetime.timedelta(seconds=duration_s)

        pacemaker_code = _get_short_int(filename, 274)
        recorder_type = _get_short_int(filename, 276)

        variable_block_hex = ""
        if var_block_size > 0:
            with open(filename, "rb") as f:
                f.seek(var_block_offset, os.SEEK_SET)
                variable_block_hex = f.read(var_block_size).hex()

        record.recording.acquisition.signal = SignalCharacteristics(
            sample_rate=sr,
            bits_per_sample=16,
            signal_signed=True,
            number_channels_allocated=nleads,
            number_channels_valid=len(record.leads),
            data_encoding="int16",
            compression="none",
        )

        record.raw_metadata["filepath"] = filename
        record.raw_metadata["var_block_size"] = var_block_size
        record.raw_metadata["ecg_size"] = ecg_size
        record.raw_metadata["checksum"] = int(checksum)
        record.raw_metadata["is_annfile"] = is_annfile
        record.raw_metadata["lead_quality"] = lead_quality[:nleads]
        record.raw_metadata["pacemaker_code"] = pacemaker_code
        record.raw_metadata["recorder_type"] = recorder_type
        if variable_block_hex:
            record.raw_metadata["variable_block"] = variable_block_hex

        return record

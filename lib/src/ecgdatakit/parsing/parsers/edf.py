"""EDF/EDF+ (European Data Format) parser.

Reference: https://www.edfplus.info/specs/edf.html
           https://www.edfplus.info/specs/edfplus.html
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ecgdatakit.exceptions import CorruptedFileError
from ecgdatakit.models import (
    DeviceInfo,
    ECGRecord,
    FilterSettings,
    Lead,
    PatientInfo,
    RecordingInfo,
    SignalCharacteristics,
)
from ecgdatakit.parsing.parser import Parser

_LEAD_LABELS = {
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
}


def _clean_str(raw: str) -> str:
    """Strip null bytes and whitespace from a fixed-width EDF string."""
    return raw.replace("\x00", "").strip()


def _parse_edf_date(date_str: str, time_str: str) -> datetime | None:
    """Parse EDF header date (dd.mm.yy) and time (hh.mm.ss)."""
    date_str = _clean_str(date_str)
    time_str = _clean_str(time_str)
    if not date_str or not time_str:
        return None
    try:
        day, month, year = date_str.split(".")
        hour, minute, second = time_str.split(".")
        year_int = int(year)
        if year_int >= 85:
            year_int += 1900
        else:
            year_int += 2000
        return datetime(year_int, int(month), int(day),
                        int(hour), int(minute), int(second))
    except (ValueError, IndexError):
        return None


def _parse_edfplus_patient(patient_str: str) -> PatientInfo:
    """Parse EDF+ patient identification field.

    Format: patient_code sex birthdate name [additional]
    """
    info = PatientInfo()
    patient_str = _clean_str(patient_str)
    if not patient_str:
        return info

    parts = patient_str.split()
    if len(parts) >= 1 and parts[0] != "X":
        info.patient_id = parts[0]
    if len(parts) >= 2:
        sex = parts[1].upper()
        if sex == "M":
            info.sex = "M"
        elif sex == "F":
            info.sex = "F"
        else:
            info.sex = "U"
    if len(parts) >= 3 and parts[2] != "X":
        try:
            info.birth_date = datetime.strptime(parts[2], "%d-%b-%Y")
        except ValueError:
            pass
    if len(parts) >= 4 and parts[3] != "X":
        name = parts[3]
        if "_" in name:
            name_parts = name.split("_", 1)
            info.last_name = name_parts[0]
            info.first_name = name_parts[1]
        else:
            info.last_name = name

    return info


def _parse_edfplus_recording(recording_str: str) -> dict[str, str]:
    """Parse EDF+ recording identification field.

    Format: ``Startdate DD-MMM-YYYY investigator_id equipment_id technician_id``

    Returns a dict with keys ``investigator``, ``equipment``, ``technician``.
    Values that are ``X`` (unknown) are returned as empty strings.
    """
    result: dict[str, str] = {
        "investigator": "",
        "equipment": "",
        "technician": "",
    }
    cleaned = _clean_str(recording_str)
    if not cleaned:
        return result

    parts = cleaned.split()
    if len(parts) < 2 or parts[0] != "Startdate":
        return result

    if len(parts) >= 3 and parts[2] != "X":
        result["investigator"] = parts[2]
    if len(parts) >= 4 and parts[3] != "X":
        result["equipment"] = parts[3]
    if len(parts) >= 5 and parts[4] != "X":
        result["technician"] = parts[4]

    return result


def _parse_tal_annotations(raw_bytes: bytes) -> list[dict]:
    """Parse EDF+ TAL (Time-stamped Annotation List) data.

    TAL format:  ``+onset\x15duration\x14annotation\x14\x00``
    Multiple TALs may be concatenated.  The onset may be preceded by ``+``
    or ``-``.  Duration is optional and separated by ``\x15`` (NAK).  Each
    annotation text is terminated by ``\x14`` (DC4), and TALs are
    separated by ``\x00``.

    Returns a list of dicts with keys ``onset``, ``duration``, and ``text``.
    """
    annotations: list[dict] = []
    text = raw_bytes.decode("latin-1")

    tals = text.split("\x00")
    for tal in tals:
        if not tal or tal == "\x14":
            continue
        m = re.match(
            r"([+\-]\d+(?:\.\d*)?)"
            r"(?:\x15(\d+(?:\.\d*)?))?",
            tal,
        )
        if not m:
            continue
        onset = float(m.group(1))
        duration = float(m.group(2)) if m.group(2) else 0.0
        remainder = tal[m.end():]
        parts = remainder.split("\x14")
        for part in parts:
            part = part.strip()
            if part:
                annotations.append({
                    "onset": onset,
                    "duration": duration,
                    "text": part,
                })

    return annotations


def _parse_prefiltering(prefilter_str: str) -> FilterSettings:
    """Extract filter frequencies from an EDF prefiltering string.

    Recognises common patterns such as:
    - ``HP:0.05Hz`` or ``HP:0.05 Hz``  (highpass)
    - ``LP:150Hz``  (lowpass)
    - ``N:50Hz`` or ``Notch:50Hz``     (notch)
    """
    fs = FilterSettings()
    s = _clean_str(prefilter_str).upper()
    if not s:
        return fs

    hp = re.search(r"HP[:\s]*(\d+(?:\.\d+)?)\s*HZ", s)
    if hp:
        fs.highpass = float(hp.group(1))

    lp = re.search(r"LP[:\s]*(\d+(?:\.\d+)?)\s*HZ", s)
    if lp:
        fs.lowpass = float(lp.group(1))

    notch = re.search(r"(?:N|NOTCH)[:\s]*(\d+(?:\.\d+)?)\s*HZ", s)
    if notch:
        fs.notch = float(notch.group(1))
        fs.notch_active = True

    return fs


class EDFParser(Parser):
    """Parser for EDF and EDF+ ECG files."""

    FORMAT_NAME = "EDF/EDF+"
    FORMAT_DESCRIPTION = "European Data Format (EDF and EDF+)"
    FILE_EXTENSIONS = [".edf"]

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        if len(header) < 256:
            return False
        version = header[:8]
        if version != b"0       ":
            return False
        try:
            nr = int(header[236:244].decode("ascii").strip())
            ns = int(header[252:256].decode("ascii").strip())
            return nr >= -1 and ns > 0
        except (ValueError, UnicodeDecodeError):
            return False

    def parse(self, file_path: Path) -> ECGRecord:
        with open(file_path, "rb") as f:
            raw = f.read()

        if len(raw) < 256:
            raise CorruptedFileError(f"File too small for EDF header: {len(raw)} bytes")

        version = raw[0:8].decode("ascii")
        patient_id_str = raw[8:88].decode("ascii", errors="replace")
        recording_id_str = raw[88:168].decode("ascii", errors="replace")
        start_date = raw[168:176].decode("ascii", errors="replace")
        start_time = raw[176:184].decode("ascii", errors="replace")
        header_bytes = int(raw[184:192].decode("ascii").strip())
        reserved = raw[192:236].decode("ascii", errors="replace")
        num_records = int(raw[236:244].decode("ascii").strip())
        record_duration = float(raw[244:252].decode("ascii").strip())
        num_signals = int(raw[252:256].decode("ascii").strip())

        if num_signals <= 0:
            raise CorruptedFileError(f"Invalid signal count: {num_signals}")

        reserved_clean = _clean_str(reserved)
        is_edfplus = reserved_clean.startswith("EDF+")
        source_fmt = "edf_plus" if is_edfplus else "edf"

        record = ECGRecord(source_format=source_fmt)

        if is_edfplus:
            record.patient = _parse_edfplus_patient(patient_id_str)
        else:
            pid = _clean_str(patient_id_str)
            record.patient = PatientInfo(patient_id=pid)

        offset = 256
        ns = num_signals

        def _read_field(length: int) -> list[str]:
            nonlocal offset
            vals = []
            for i in range(ns):
                vals.append(raw[offset + i * length: offset + (i + 1) * length]
                            .decode("ascii", errors="replace"))
            offset += ns * length
            return vals

        labels = _read_field(16)
        transducer_types = _read_field(80)
        physical_dims = _read_field(8)
        physical_mins = _read_field(8)
        physical_maxs = _read_field(8)
        digital_mins = _read_field(8)
        digital_maxs = _read_field(8)
        prefilterings = _read_field(80)
        samples_per_record = _read_field(8)
        _reserved_signals = _read_field(32)

        phys_min = [float(_clean_str(v)) for v in physical_mins]
        phys_max = [float(_clean_str(v)) for v in physical_maxs]
        dig_min = [float(_clean_str(v)) for v in digital_mins]
        dig_max = [float(_clean_str(v)) for v in digital_maxs]
        samps_per_rec = [int(_clean_str(v)) for v in samples_per_record]

        recording_date = _parse_edf_date(start_date, start_time)
        total_duration = num_records * record_duration if num_records > 0 else 0.0

        sample_rates = []
        for spr in samps_per_rec:
            sr = int(spr / record_duration) if record_duration > 0 else 0
            sample_rates.append(sr)

        global_sr = sample_rates[0] if sample_rates else 0

        recording = RecordingInfo(
            date=recording_date,
            sample_rate=global_sr,
            duration=timedelta(seconds=total_duration) if total_duration > 0 else None,
        )
        if recording_date and total_duration > 0:
            recording.end_date = recording_date + timedelta(seconds=total_duration)

        record.recording = recording

        if is_edfplus:
            rec_id_fields = _parse_edfplus_recording(recording_id_str)
            if rec_id_fields["equipment"]:
                record.device = DeviceInfo(model=rec_id_fields["equipment"])
            if rec_id_fields["technician"]:
                record.recording.technician = rec_id_fields["technician"]
            if rec_id_fields["investigator"]:
                record.recording.referring_physician = rec_id_fields["investigator"]

        data_offset = header_bytes
        if data_offset > len(raw):
            raise CorruptedFileError("Header bytes offset exceeds file size")

        annotation_indices = set()
        for i, label in enumerate(labels):
            if "annotation" in _clean_str(label).lower():
                annotation_indices.add(i)

        leads: list[Lead] = []
        for sig_idx in range(ns):
            if sig_idx in annotation_indices:
                continue

            label = _clean_str(labels[sig_idx])
            dmin = dig_min[sig_idx]
            dmax = dig_max[sig_idx]
            pmin = phys_min[sig_idx]
            pmax = phys_max[sig_idx]

            if dmax == dmin:
                gain = 1.0
                offset_val = 0.0
            else:
                gain = (pmax - pmin) / (dmax - dmin)
                offset_val = pmin - gain * dmin

            spr = samps_per_rec[sig_idx]
            total_samples = num_records * spr
            all_samples = np.empty(total_samples, dtype=np.float64)

            pos = data_offset
            for rec in range(num_records):
                skip = sum(samps_per_rec[:sig_idx]) * 2
                sig_start = pos + skip
                sig_end = sig_start + spr * 2
                if sig_end > len(raw):
                    break
                digital = np.frombuffer(raw[sig_start:sig_end], dtype="<i2")
                all_samples[rec * spr: rec * spr + len(digital)] = (
                    digital.astype(np.float64) * gain + offset_val
                )
                if sig_idx == 0:
                    pass
            all_samples = np.empty(total_samples, dtype=np.float64)
            pos = data_offset
            for rec in range(num_records):
                rec_offset = pos
                for s in range(ns):
                    block_size = samps_per_rec[s] * 2
                    if s == sig_idx:
                        end = rec_offset + block_size
                        if end > len(raw):
                            break
                        digital = np.frombuffer(
                            raw[rec_offset:end], dtype="<i2"
                        )
                        samples_converted = digital.astype(np.float64) * gain + offset_val
                        dest_start = rec * spr
                        dest_end = dest_start + len(samples_converted)
                        all_samples[dest_start:dest_end] = samples_converted
                    rec_offset += block_size
                pos = rec_offset

            leads.append(Lead(
                label=label,
                samples=all_samples[:total_samples],
                sample_rate=sample_rates[sig_idx],
                units=_clean_str(physical_dims[sig_idx]),
                transducer=_clean_str(transducer_types[sig_idx]),
                prefiltering=_clean_str(prefilterings[sig_idx]),
            ))

        record.leads = leads

        if is_edfplus and annotation_indices:
            all_annotations: list[dict] = []
            for ann_idx in sorted(annotation_indices):
                spr = samps_per_rec[ann_idx]
                block_bytes = spr * 2
                ann_raw = bytearray()
                pos = data_offset
                for rec in range(num_records):
                    rec_offset = pos
                    for s in range(ns):
                        bs = samps_per_rec[s] * 2
                        if s == ann_idx:
                            end = rec_offset + bs
                            if end <= len(raw):
                                ann_raw.extend(raw[rec_offset:end])
                        rec_offset += bs
                    pos = rec_offset
                all_annotations.extend(_parse_tal_annotations(bytes(ann_raw)))
            if all_annotations:
                record.raw_metadata["edf_annotations"] = all_annotations

        parsed_filters: list[FilterSettings] = []
        for sig_idx in range(ns):
            if sig_idx in annotation_indices:
                continue
            pf = _clean_str(prefilterings[sig_idx])
            if pf:
                parsed_filters.append(_parse_prefiltering(pf))

        if parsed_filters:
            ref = parsed_filters[0]
            consistent = all(
                f.highpass == ref.highpass
                and f.lowpass == ref.lowpass
                and f.notch == ref.notch
                for f in parsed_filters
            )
            if consistent and (ref.highpass is not None
                               or ref.lowpass is not None
                               or ref.notch is not None):
                record.filters = ref

        record.signal = SignalCharacteristics(
            bits_per_sample=16,
            signal_signed=True,
            number_channels_allocated=num_signals,
            number_channels_valid=len(record.leads),
            data_encoding="int16",
            compression="none",
        )

        record.raw_metadata["filepath"] = str(file_path)
        record.raw_metadata["recording_id"] = _clean_str(recording_id_str)
        record.raw_metadata["num_records"] = num_records
        record.raw_metadata["record_duration"] = record_duration
        record.raw_metadata["is_edfplus"] = is_edfplus

        return record

"""EDF/EDF+ (European Data Format) parser.

Reference: https://www.edfplus.info/specs/edf.html
           https://www.edfplus.info/specs/edfplus.html
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ecgdatakit.exceptions import CorruptedFileError, UnsupportedFormatError
from ecgdatakit.models import (
    DeviceInfo,
    ECGRecord,
    FileFormatInfo,
    FilterSettings,
    Lead,
    PatientInfo,
    RecordingInfo,
    SignalCharacteristics,
    _normalize_unit,
    derive_is_raw,
)
from ecgdatakit.parsing.helpers import (
    STANDARD_LEADS,
    decode_text,
    fill_signal_summary,
    normalize_lead_label,
    unique_labels,
)
from ecgdatakit.parsing.parser import Parser

_ANNOTATION_LABEL = "EDF Annotations"
_MONTHS = {
    m: i for i, m in enumerate(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
         "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)
}
_TAL_RE = re.compile(r"^([+\-]\d+(?:\.\d*)?)(?:\x15(\d+(?:\.\d*)?))?\x14")


def _clean_str(raw: str) -> str:
    """Strip null bytes and whitespace from a fixed-width EDF string."""
    return raw.replace("\x00", "").strip()


def _parse_edf_date(date_str: str, time_str: str) -> datetime | None:
    """Parse EDF header date (dd.mm.yy) and time (hh.mm.ss).

    Years 85-99 are 1985-1999 and 00-84 are 2000-2084 (EDF+ clipping rule).
    """
    date_str = _clean_str(date_str)
    time_str = _clean_str(time_str)
    if not date_str or not time_str:
        return None
    try:
        day, month, year = date_str.split(".")
        hour, minute, second = time_str.split(".")
        year_int = int(year)
        year_int += 1900 if year_int >= 85 else 2000
        return datetime(year_int, int(month), int(day),
                        int(hour), int(minute), int(second))
    except (ValueError, IndexError):
        return None


def _parse_edfplus_date(text: str) -> datetime | None:
    """Parse an EDF+ ``dd-MMM-yyyy`` date (English month, any case)."""
    parts = text.split("-")
    if len(parts) != 3 or parts[1].upper() not in _MONTHS:
        return None
    try:
        return datetime(int(parts[2]), _MONTHS[parts[1].upper()], int(parts[0]))
    except ValueError:
        return None


def _edfplus_name(name: str) -> tuple[str, str]:
    """Split an EDF+ name subfield into (last, first).

    Spaces are written as ``_``. The spec does not order the name, so it is
    only split when written ``Last,First``; otherwise it is kept whole.
    """
    name = name.replace("_", " ").strip()
    if "," in name:
        last, _, first = name.partition(",")
        return last.strip(), first.strip()
    return name, ""


def _parse_edfplus_patient(patient_str: str) -> tuple[PatientInfo, list[str]]:
    """Parse the EDF+ local patient identification field.

    Format: ``code sex birthdate name [additional subfields]``, ``X`` for
    unknown. Returns the patient and the additional subfields.
    """
    info = PatientInfo()
    parts = _clean_str(patient_str).split()
    if len(parts) >= 1 and parts[0] != "X":
        info.patient_id = parts[0].replace("_", " ")
    if len(parts) >= 2:
        info.sex = {"M": "M", "F": "F"}.get(parts[1].upper(), "U")
    if len(parts) >= 3 and parts[2] != "X":
        info.birth_date = _parse_edfplus_date(parts[2])
    if len(parts) >= 4 and parts[3] != "X":
        info.last_name, info.first_name = _edfplus_name(parts[3])
    return info, parts[4:]


def _parse_edfplus_recording(recording_str: str) -> dict:
    """Parse the EDF+ local recording identification field.

    Format: ``Startdate dd-MMM-yyyy admin_code technician equipment [...]``
    (``X`` for unknown). Unknown values are returned as empty strings.
    """
    result: dict = {
        "startdate": None,
        "admin_code": "",
        "technician": "",
        "equipment": "",
        "additional": [],
    }
    parts = _clean_str(recording_str).split()
    if len(parts) < 2 or parts[0] != "Startdate":
        return result
    if parts[1] != "X":
        result["startdate"] = _parse_edfplus_date(parts[1])
    for key, index in (("admin_code", 2), ("technician", 3), ("equipment", 4)):
        if len(parts) > index and parts[index] != "X":
            result[key] = parts[index].replace("_", " ")
    result["additional"] = parts[5:]
    return result


def _parse_tals(raw_bytes: bytes) -> list[tuple[float, float | None, list[str]]]:
    """Parse the TALs of one annotation signal block of one data record.

    TAL format: ``+onset[\\x15duration]\\x14text\\x14[text\\x14...]\\x00``.
    Texts are UTF-8. Returns (onset, duration, texts) per TAL, in order.
    """
    tals: list[tuple[float, float | None, list[str]]] = []
    for chunk in raw_bytes.split(b"\x00"):
        if not chunk:
            continue
        text = decode_text(chunk)
        m = _TAL_RE.match(text)
        if not m:
            continue
        onset = float(m.group(1))
        duration = float(m.group(2)) if m.group(2) else None
        texts = [t.strip() for t in text[m.end():].split("\x14")]
        tals.append((onset, duration, [t for t in texts if t]))
    return tals


def _parse_prefiltering(prefilter_str: str) -> FilterSettings:
    """Extract filter frequencies from an EDF prefiltering string.

    Recognises ``HP:0.05Hz``, ``LP:150Hz`` and ``N:50Hz`` / ``Notch:50Hz``.
    """
    fs = FilterSettings()
    s = _clean_str(prefilter_str).upper()
    if not s:
        return fs

    hp = re.search(r"HP[:\s]*(\d+(?:\.\d+)?|\.\d+)\s*HZ", s)
    if hp and float(hp.group(1)):
        fs.highpass = float(hp.group(1))

    lp = re.search(r"LP[:\s]*(\d+(?:\.\d+)?|\.\d+)\s*HZ", s)
    if lp and float(lp.group(1)):
        fs.lowpass = float(lp.group(1))

    notch = re.search(r"(?:NOTCH|\bN)[:\s]*(\d+(?:\.\d+)?)\s*HZ", s)
    if notch and float(notch.group(1)):
        fs.notch = float(notch.group(1))
        fs.notch_active = True

    return fs


class EDFParser(Parser):
    """Parser for EDF and EDF+ ECG files."""

    FORMAT_NAME = "EDF/EDF+"
    FORMAT_DESCRIPTION = "European Data Format (EDF and EDF+)"
    FILE_EXTENSIONS = [".edf"]
    PRIORITY = 20

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        if len(header) < 256:
            return False
        if header[:8] != b"0       ":
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
        self._warnings: list[str] = []
        record = self._parse(raw)
        record.raw_metadata["filepath"] = str(file_path)
        for msg in self._warnings:
            warnings.warn(msg, UserWarning, stacklevel=2)
        fill_signal_summary(record)
        return record

    # ------------------------------------------------------------------

    @staticmethod
    def _number(text: str, what: str, kind: type = float):
        value = _clean_str(text)
        try:
            return kind(value)
        except ValueError:
            raise CorruptedFileError(f"EDF header: invalid {what} {value!r}") from None

    def _parse(self, raw: bytes) -> ECGRecord:
        if len(raw) < 256:
            raise CorruptedFileError(f"File too small for EDF header: {len(raw)} bytes")
        if raw[:8] != b"0       ":
            if raw[:8] == b"\xffBIOSEMI":
                raise UnsupportedFormatError("BDF (24-bit BioSemi) files are not supported")
            raise CorruptedFileError(f"EDF header: invalid version {raw[:8]!r}")

        def text(start: int, end: int) -> str:
            return decode_text(raw[start:end])

        patient_id_str = text(8, 88)
        recording_id_str = text(88, 168)
        header_bytes = self._number(text(184, 192), "header size", int)
        reserved = _clean_str(text(192, 236))
        num_records = self._number(text(236, 244), "number of data records", int)
        record_duration = self._number(text(244, 252), "data record duration")
        ns = self._number(text(252, 256), "number of signals", int)

        if ns <= 0:
            raise CorruptedFileError(f"Invalid signal count: {ns}")
        if record_duration < 0:
            raise CorruptedFileError(f"Invalid data record duration: {record_duration}")
        expected_header = 256 * (ns + 1)
        if len(raw) < expected_header:
            raise CorruptedFileError(
                f"EDF header truncated: {ns} signals need {expected_header} bytes, "
                f"file has {len(raw)}"
            )
        if header_bytes != expected_header:
            self._warnings.append(
                f"EDF header size field is {header_bytes}, expected {expected_header} "
                f"for {ns} signals; using {expected_header}"
            )

        is_edfplus = reserved.startswith("EDF+")
        edf_type = reserved[:5] if reserved[:5] in ("EDF+C", "EDF+D") else (
            "EDF+" if is_edfplus else "EDF")
        record = ECGRecord(source_format="edf_plus" if is_edfplus else "edf")
        record.file_format = FileFormatInfo(version=edf_type)

        # Signal headers: each field is stored for all signals in turn
        pos = 256

        def field(length: int) -> list[str]:
            nonlocal pos
            values = [text(pos + i * length, pos + (i + 1) * length) for i in range(ns)]
            pos += ns * length
            return [_clean_str(v) for v in values]

        labels = field(16)
        transducers = field(80)
        dims = field(8)
        phys_min = [self._number(v, "physical minimum") for v in field(8)]
        phys_max = [self._number(v, "physical maximum") for v in field(8)]
        dig_min = [self._number(v, "digital minimum", int) for v in field(8)]
        dig_max = [self._number(v, "digital maximum", int) for v in field(8)]
        prefilterings = field(80)
        spr = [self._number(v, "samples per data record", int) for v in field(8)]
        if any(n <= 0 for n in spr):
            raise CorruptedFileError(f"EDF header: invalid samples per data record {spr}")

        # Data records
        record_samples = sum(spr)
        available = (len(raw) - expected_header) // (2 * record_samples)
        if num_records == -1:
            self._warnings.append(
                "EDF number of data records is -1 (file not closed by the "
                f"recorder); using the {available} complete records found"
            )
            num_records = available
        elif num_records < 0:
            raise CorruptedFileError(f"Invalid number of data records: {num_records}")
        elif available < num_records:
            if available == 0:
                raise CorruptedFileError(
                    f"EDF file truncated: header declares {num_records} data "
                    "records, none is complete"
                )
            self._warnings.append(
                f"EDF file truncated: header declares {num_records} data records, "
                f"only {available} are complete; reading those"
            )
            num_records = available
        data = np.frombuffer(
            raw, dtype="<i2", count=num_records * record_samples, offset=expected_header,
        ).reshape(num_records, record_samples)
        starts = np.concatenate([[0], np.cumsum(spr)])

        annotation_idx = [i for i, label in enumerate(labels) if label == _ANNOTATION_LABEL]
        signal_idx = [i for i in range(ns) if i not in annotation_idx]
        if signal_idx and record_duration <= 0:
            raise CorruptedFileError(
                "EDF data record duration is 0 but the file has ordinary signals"
            )

        # Leads
        leads: list[Lead] = []
        exact_rates: dict[str, float] = {}
        names = unique_labels([normalize_lead_label(labels[i]) for i in signal_idx])
        for label, i in zip(names, signal_idx):
            samples = data[:, starts[i]:starts[i + 1]].ravel().astype(np.float64)
            rate = spr[i] / record_duration
            if rate != round(rate):
                exact_rates[label] = rate
            dim = dims[i]
            unit = (_normalize_unit(dim) or dim) if dim else ""
            dmin, dmax, pmin, pmax = dig_min[i], dig_max[i], phys_min[i], phys_max[i]
            if dmax <= dmin or pmax == pmin:
                self._warnings.append(
                    f"EDF signal {label!r}: invalid range (digital {dmin}..{dmax}, "
                    f"physical {pmin}..{pmax}); samples left uncalibrated"
                )
                gain, offset, unit = 1.0, 0.0, ""
            else:
                gain = (pmax - pmin) / (dmax - dmin)
                offset = pmin - gain * dmin
            is_raw = derive_is_raw(gain, offset, unit)
            leads.append(Lead(
                label=label,
                samples=samples,
                sampling_rate=round(rate),
                resolution=gain,
                resolution_unit=unit,
                offset=offset,
                units="" if is_raw else unit,
                is_raw=is_raw,
                adc_resolution=gain,
                adc_resolution_unit=dim,
                transducer=transducers[i],
                prefiltering=prefilterings[i],
                annotations={
                    "physical_minimum": f"{pmin:g}",
                    "physical_maximum": f"{pmax:g}",
                    "digital_minimum": str(dmin),
                    "digital_maximum": str(dmax),
                },
            ))
        record.leads = leads

        # Patient and recording identification
        rec_fields = _parse_edfplus_recording(recording_id_str) if is_edfplus else None
        start = _parse_edf_date(text(168, 176), text(176, 184))
        startdate = rec_fields["startdate"] if rec_fields else None
        if startdate is not None:
            # The EDF+ startdate carries the 4-digit year (needed after 2084)
            time_str = _clean_str(text(176, 184))
            try:
                hour, minute, second = (int(v) for v in time_str.split("."))
                start = startdate.replace(hour=hour, minute=minute, second=second)
            except ValueError:
                start = None

        if is_edfplus:
            record.patient, extra_patient = _parse_edfplus_patient(patient_id_str)
            if extra_patient:
                record.raw_metadata["patient_additional"] = extra_patient
        else:
            record.patient = PatientInfo(patient_id=_clean_str(patient_id_str))

        # Annotations and record start times (EDF+)
        record_onsets: list[float] = []
        events: list[dict] = []
        for rec in range(num_records):
            first = True
            for i in annotation_idx:
                block = data[rec, starts[i]:starts[i + 1]].tobytes()
                for onset, duration, texts in _parse_tals(block):
                    if first and is_edfplus:
                        # First TAL of each record keeps time: its onset is the record start
                        record_onsets.append(onset)
                        first = False
                        if not texts:
                            continue
                    for t in texts:
                        events.append({"onset": onset, "duration": duration, "text": t})

        if start is not None and record_onsets:
            # Sub-second start time (onset of the first data record)
            start = start + timedelta(seconds=record_onsets[0])

        recording = RecordingInfo(date=start)
        data_seconds = num_records * record_duration
        if data_seconds > 0:
            recording.duration = timedelta(seconds=data_seconds)
        if len(record_onsets) == num_records and num_records:
            expected = record_onsets[0] + record_duration * np.arange(num_records)
            gaps = not np.allclose(record_onsets, expected, atol=1e-6)
            if gaps:
                self._warnings.append(
                    "EDF+D file has gaps between data records; leads are the "
                    "records joined end to end, record start times are in "
                    "raw_metadata['record_onsets']"
                )
            record.raw_metadata["record_onsets"] = record_onsets

        if rec_fields:
            if rec_fields["equipment"]:
                recording.device = DeviceInfo(model=rec_fields["equipment"])
            recording.technician = rec_fields["technician"]
            if rec_fields["admin_code"]:
                record.raw_metadata["admin_code"] = rec_fields["admin_code"]
            if rec_fields["additional"]:
                record.raw_metadata["recording_additional"] = rec_fields["additional"]
        record.recording = recording

        p = record.patient
        if p.birth_date and recording.date and p.age is None:
            d0, d1 = p.birth_date, recording.date
            p.age = d1.year - d0.year - ((d1.month, d1.day) < (d0.month, d0.day))

        if events:
            record.raw_metadata["edf_annotations"] = events
            # Tab separated: onset (s), duration (s, empty if none), text
            record.annotations["edf_annotations"] = "\n".join(
                f"{e['onset']:+g}\t{'' if e['duration'] is None else format(e['duration'], 'g')}"
                f"\t{e['text']}"
                for e in events
            )

        # Filters, when every signal declares the same ones
        parsed_filters = [_parse_prefiltering(prefilterings[i]) for i in signal_idx
                          if prefilterings[i]]
        if parsed_filters:
            ref = parsed_filters[0]
            consistent = all(
                (f.highpass, f.lowpass, f.notch) == (ref.highpass, ref.lowpass, ref.notch)
                for f in parsed_filters
            )
            if consistent and len(parsed_filters) == len(signal_idx) and (
                    ref.highpass is not None or ref.lowpass is not None
                    or ref.notch is not None):
                record.recording.acquisition.filters = ref

        # Record-level rate: shared rate of the ECG leads (else of all leads)
        ecg_rates = {lead.sampling_rate for lead in leads
                     if lead.label.split("_")[0] in STANDARD_LEADS}
        rates = ecg_rates or {lead.sampling_rate for lead in leads}
        record.recording.acquisition.signal = SignalCharacteristics(
            sampling_rate=rates.pop() if len(rates) == 1 else 0,
            bits_per_sample=16,
            signal_signed=True,
            number_channels_allocated=ns,
            number_channels_valid=len(leads),
            data_encoding="int16",
            compression="none",
        )

        raw_meta = record.raw_metadata
        raw_meta["recording_id"] = _clean_str(recording_id_str)
        raw_meta["num_records"] = num_records
        raw_meta["record_duration"] = record_duration
        raw_meta["is_edfplus"] = is_edfplus
        if exact_rates:
            raw_meta["exact_sampling_rates"] = exact_rates
        return record

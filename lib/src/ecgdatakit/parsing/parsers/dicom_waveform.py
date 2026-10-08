"""DICOM Waveform ECG parser.

Parses DICOM Part 10 files containing ECG waveform data (PS3.3 C.10.9).
Requires pydicom (optional dependency).

Samples are the encoded values of Waveform Data (mu-law and A-law are
expanded to linear PCM). Physical values follow PS3.3 C.10.9.1.4.2:
``physical = sample * sensitivity * correction_factor + baseline``, in the
Channel Sensitivity Units. Samples equal to Waveform Padding Value are set
to NaN.
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ecgdatakit.exceptions import CorruptedFileError, ECGDataKitError, MissingElementError
from ecgdatakit.models import (
    DeviceInfo,
    ECGRecord,
    FilterSettings,
    GlobalMeasurements,
    Interpretation,
    Lead,
    PatientInfo,
    RecordingInfo,
    SignalCharacteristics,
    _normalize_unit,
    derive_is_raw,
)
from ecgdatakit.parsing.helpers import fill_signal_summary, normalize_lead_label, unique_labels
from ecgdatakit.parsing.parser import Parser


def _import_pydicom():
    """Lazily import pydicom, raising a helpful error if missing."""
    try:
        import pydicom
        return pydicom
    except ImportError:
        raise ImportError(
            "pydicom is required for DICOM waveform parsing. "
            'Install it with: pip install "ecgdatakit[dicom]"'
        )


# Waveform Sample Interpretation per Waveform Bits Allocated (PS3.3 C.10.9.1.5)
_SAMPLE_DTYPES: dict[tuple[int, str], str] = {
    (8, "SB"): "i1", (8, "UB"): "u1", (8, "MB"): "u1", (8, "AB"): "u1",
    (16, "SS"): "i2", (16, "US"): "u2",
    (32, "SL"): "i4", (32, "UL"): "u4",
    (64, "SV"): "i8", (64, "UV"): "u8",
}

_DEFAULT_INTERPRETATION = {8: "SB", 16: "SS", 32: "SL", 64: "SV"}


def _ulaw_table() -> np.ndarray:
    """ITU-T G.711 mu-law to 16-bit linear PCM."""
    out = np.empty(256, dtype=np.int32)
    for code in range(256):
        u = ~code & 0xFF
        t = (((u & 0x0F) << 3) + 0x84) << ((u & 0x70) >> 4)
        out[code] = (0x84 - t) if u & 0x80 else (t - 0x84)
    return out


def _alaw_table() -> np.ndarray:
    """ITU-T G.711 A-law to 16-bit linear PCM."""
    out = np.empty(256, dtype=np.int32)
    for code in range(256):
        a = code ^ 0x55
        t = (a & 0x0F) << 4
        seg = (a & 0x70) >> 4
        if seg == 0:
            t += 8
        else:
            t = (t + 0x108) << (seg - 1)
        out[code] = t if a & 0x80 else -t
    return out


_EXPANSION = {"MB": _ulaw_table(), "AB": _alaw_table()}

# SCP-ECG / ISO/IEEE 11073-10102 lead codes, shared by MDC "2:N" and
# SCPECG "5.6.3-9-N" coded channel sources (PS3.16 CID 3001).
_LEAD_CODES: dict[int, str] = {
    1: "I", 2: "II", 3: "V1", 4: "V2", 5: "V3", 6: "V4", 7: "V5", 8: "V6",
    9: "V7", 10: "V2R", 11: "V3R", 12: "V4R", 13: "V5R", 14: "V6R", 15: "V7R",
    16: "X", 17: "Y", 18: "Z", 19: "CC5", 20: "CM5",
    61: "III", 62: "aVR", 63: "aVL", 64: "aVF", 65: "-aVR",
    66: "V8", 67: "V9", 68: "V8R", 69: "V9R",
}

# Global measurement codes (PS3.16 CID 3226-3229 MDC codes, MDC reference
# ids, LOINC heart rate and legacy SCPECG codes).
_MEASUREMENT_CODES: dict[str, str] = {
    "2:16770": "heart_rate",
    "8867-4": "heart_rate",
    "MDC_ECG_HEART_RATE": "heart_rate",
    "2:16168": "rr_interval",
    "2:16000": "rr_interval",
    "MDC_ECG_TIME_PD_RR": "rr_interval",
    "MDC_ECG_TIME_PD_RR_GL": "rr_interval",
    "2:15872": "pr_interval",
    "MDC_ECG_TIME_PD_PR": "pr_interval",
    "MDC_ECG_TIME_PD_PR_GL": "pr_interval",
    "2:16156": "qrs_duration",
    "MDC_ECG_TIME_PD_QRS": "qrs_duration",
    "MDC_ECG_TIME_PD_QRS_GL": "qrs_duration",
    "2:16160": "qt_interval",
    "MDC_ECG_TIME_PD_QT": "qt_interval",
    "MDC_ECG_TIME_PD_QT_GL": "qt_interval",
    "2:15880": "qtc_bazett",
    "MDC_ECG_TIME_PD_QTcB": "qtc_bazett",
    "2:15892": "qtc_fridericia",
    "MDC_ECG_TIME_PD_QTcF": "qtc_fridericia",
    "2:16128": "p_axis",
    "MDC_ECG_ANGLE_P_FRONT": "p_axis",
    "2:16132": "qrs_axis",
    "MDC_ECG_ANGLE_QRS_FRONT": "qrs_axis",
    "2:16136": "t_axis",
    "MDC_ECG_ANGLE_T_FRONT": "t_axis",
    "5.10.2.1-1": "heart_rate",
    "5.10.3-1": "rr_interval",
    "5.10.3-2": "pr_interval",
    "5.10.3-3": "qrs_duration",
    "5.10.3-4": "qt_interval",
}

_MEDIAN_GROUP_WORDS = ("MEDIAN", "REPRESENTATIVE", "AVERAGE")


def _text(value) -> str:
    """Return a DICOM value as a stripped string ("" when absent)."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        return "; ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


def _person(value) -> str:
    """Format a DICOM Person Name as ``"Given Middle Family"``."""
    if value is None:
        return ""
    values = value if type(value).__name__ == "MultiValue" else [value]
    names = []
    for pn in values:
        parts = [
            getattr(pn, "name_prefix", ""), getattr(pn, "given_name", ""),
            getattr(pn, "middle_name", ""), getattr(pn, "family_name", ""),
            getattr(pn, "name_suffix", ""),
        ]
        text = " ".join(p.strip() for p in parts if p and p.strip())
        names.append(text or str(pn).replace("^", " ").strip())
    return "; ".join(n for n in names if n)


def _float(value) -> float | None:
    try:
        return float(value) if value is not None and str(value).strip() != "" else None
    except (TypeError, ValueError):
        return None


def _first_code(item, name: str):
    seq = item.get(name) if item is not None else None
    return seq[0] if seq else None


def _code_parts(code_item) -> tuple[str, str, str]:
    """Return (scheme, value, meaning) of a code sequence item."""
    if code_item is None:
        return "", "", ""
    value = _text(code_item.get("CodeValue")) or _text(code_item.get("LongCodeValue"))
    return (
        _text(code_item.get("CodingSchemeDesignator")),
        value,
        _text(code_item.get("CodeMeaning")),
    )


def _lead_from_code(scheme: str, value: str) -> str:
    """Map a coded channel source to a standard lead label ("" if unknown)."""
    number = None
    if value.startswith("2:") or (scheme == "MDC" and value.isdigit()):
        number = value.split(":")[-1]
    elif value.startswith("5.6.3-9-"):
        number = value.rsplit("-", 1)[-1]
    elif value.upper().startswith("MDC_ECG_LEAD_"):
        label = normalize_lead_label(value[len("MDC_ECG_LEAD_"):])
        return label if label in _LEAD_CODES.values() else ""
    if number is not None and number.isdigit():
        return _LEAD_CODES.get(int(number), "")
    return ""


def _lead_from_meaning(meaning: str) -> str:
    """Clean a channel source meaning such as ``"aVR, augmented voltage, right"``."""
    text = meaning.split(",")[0].split("(")[0].strip()
    return normalize_lead_label(text) if text else ""


def _parse_da(value) -> date | None:
    text = _text(value)
    if len(text) >= 8 and text[:8].isdigit():
        try:
            return datetime.strptime(text[:8], "%Y%m%d").date()
        except ValueError:
            return None
    return None


def _parse_tm(value) -> tuple[int, int, int, int] | None:
    """Parse a DICOM TM (``HHMMSS.FFFFFF``, minutes/seconds optional)."""
    text = _text(value).replace(":", "")
    if len(text) < 2 or not text[:2].isdigit():
        return None
    main, _, frac = text.partition(".")
    try:
        hour = int(main[0:2])
        minute = int(main[2:4]) if len(main) >= 4 else 0
        second = int(main[4:6]) if len(main) >= 6 else 0
        micro = int((frac + "000000")[:6]) if frac.isdigit() else 0
        datetime(2000, 1, 1, hour, minute, min(second, 59), micro)
    except ValueError:
        return None
    return hour, minute, min(second, 59), micro


def _parse_tz(value) -> timezone | None:
    text = _text(value)
    if len(text) == 5 and text[0] in "+-" and text[1:].isdigit():
        delta = timedelta(hours=int(text[1:3]), minutes=int(text[3:5]))
        return timezone(-delta if text[0] == "-" else delta)
    return None


def _parse_dt(value, default_tz: timezone | None) -> datetime | None:
    """Parse a DICOM DT; returns None when it carries no time of day."""
    text = _text(value)
    if len(text) < 10 or not text[:8].isdigit():
        return None
    tz = default_tz
    for sign in "+-":
        idx = text.find(sign, 8)
        if idx != -1:
            tz = _parse_tz(text[idx:]) or tz
            text = text[:idx]
            break
    day = _parse_da(text[:8])
    tm = _parse_tm(text[8:])
    if day is None or tm is None:
        return None
    return datetime(day.year, day.month, day.day, *tm, tzinfo=tz)


class DICOMWaveformParser(Parser):
    """Parser for DICOM Waveform ECG files."""

    FORMAT_NAME = "DICOM Waveform"
    FORMAT_DESCRIPTION = "DICOM Waveform ECG (Part 10, requires pydicom)"
    FILE_EXTENSIONS = [".dcm", ".dicom"]
    PRIORITY = 10

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        if len(header) < 132:
            return False
        return header[128:132] == b"DICM"

    def parse(self, file_path: Path) -> ECGRecord:
        pydicom = _import_pydicom()

        try:
            ds = pydicom.dcmread(str(file_path))
        except Exception as e:
            raise CorruptedFileError(f"Failed to read DICOM file: {e}") from e

        # pydicom decodes element values lazily, so malformed values only
        # fail when they are accessed
        try:
            return self._parse_dataset(ds, file_path)
        except (ECGDataKitError, Warning):
            raise
        except Exception as e:
            raise CorruptedFileError(
                f"DICOM: cannot decode an element: {type(e).__name__}: {e}"
            ) from e

    def _parse_dataset(self, ds, file_path: Path) -> ECGRecord:
        record = ECGRecord(source_format="dicom_waveform")
        meta = record.raw_metadata
        meta["filepath"] = str(file_path)

        tz = _parse_tz(ds.get("TimezoneOffsetFromUTC"))
        record.recording = self._read_recording(ds, tz, meta)
        record.recording.device = self._read_device(ds)
        record.patient = self._read_patient(ds, record.recording.date, meta)
        record.file_format.version, record.file_format.creation_date = self._read_file_format(ds)

        little_endian = self._is_little_endian(ds)
        groups = []
        for index, wf in enumerate(ds.get("WaveformSequence") or []):
            group = self._read_group(wf, index, little_endian)
            if group is not None:
                groups.append(group)
        if not groups:
            raise MissingElementError(
                "DICOM file has no usable Waveform Sequence multiplex group "
                "(5400,0100) with channels, samples and Waveform Data"
            )
        meta["multiplex_groups"] = [
            {
                "label": g["label"],
                "channels": len(g["leads"]),
                "samples": g["num_samples"],
                "sampling_frequency": g["sampling_frequency"],
                "sample_interpretation": g["interpretation"],
            }
            for g in groups
        ]

        median_groups = [g for g in groups if g["is_median"]]
        rhythm_groups = [g for g in groups if not g["is_median"]]
        rhythm = rhythm_groups[0] if rhythm_groups else None
        median = median_groups[0] if median_groups else None
        skipped = len(groups) - (rhythm is not None) - (median is not None)
        if skipped:
            warnings.warn(
                f"DICOM file has {skipped} additional multiplex group(s) that "
                "are not loaded (see raw_metadata['multiplex_groups'])",
                stacklevel=3,
            )

        if median is not None:
            record.median_beats = median["leads"]
        if rhythm is not None:
            record.leads = rhythm["leads"]
            record.recording.acquisition.filters = rhythm["filters"]
            record.recording.acquisition.signal = rhythm["signal"]
            if rhythm["channel_filters"]:
                meta["channel_filters"] = rhythm["channel_filters"]
            rate = rhythm["sampling_frequency"]
            if rate > 0:
                record.recording.duration = timedelta(seconds=rhythm["num_samples"] / rate)
        elif median is not None:
            record.recording.acquisition.filters = median["filters"]
            record.recording.acquisition.signal = median["signal"]

        channel_labels = {
            (g["index"], i + 1): lead.label for g in groups for i, lead in enumerate(g["leads"])
        }
        self._read_annotations(ds, record, channel_labels)

        for key, attr in (
            ("sop_class_uid", "SOPClassUID"),
            ("sop_instance_uid", "SOPInstanceUID"),
            ("study_instance_uid", "StudyInstanceUID"),
            ("series_instance_uid", "SeriesInstanceUID"),
            ("accession_number", "AccessionNumber"),
            ("modality", "Modality"),
            ("study_description", "StudyDescription"),
            ("series_description", "SeriesDescription"),
            ("study_id", "StudyID"),
            ("waveform_originality", "WaveformOriginality"),
            ("performing_physician", "PerformingPhysicianName"),
            ("medical_alerts", "MedicalAlerts"),
            ("allergies", "Allergies"),
            ("patient_comments", "PatientComments"),
            ("specific_character_set", "SpecificCharacterSet"),
        ):
            value = ds.get(attr)
            text = _person(value) if attr.endswith("PhysicianName") else _text(value)
            if text:
                meta[key] = text
        sop_class = ds.get("SOPClassUID")
        if sop_class is not None and getattr(sop_class, "name", ""):
            meta["sop_class_name"] = sop_class.name
        file_meta = getattr(ds, "file_meta", None)
        if file_meta is not None and file_meta.get("TransferSyntaxUID"):
            meta["transfer_syntax_uid"] = str(file_meta.TransferSyntaxUID)
        context = self._read_acquisition_context(ds)
        if context:
            meta["acquisition_context"] = context

        fill_signal_summary(record)
        return record

    # ------------------------------------------------------------------
    # Header sections
    # ------------------------------------------------------------------

    @staticmethod
    def _is_little_endian(ds) -> bool:
        file_meta = getattr(ds, "file_meta", None)
        uid = file_meta.get("TransferSyntaxUID") if file_meta is not None else None
        if uid is not None:
            try:
                return bool(uid.is_little_endian)
            except (AttributeError, ValueError):
                pass
        encoding = getattr(ds, "original_encoding", None)
        if encoding and encoding[1] is not None:
            return bool(encoding[1])
        return True

    @staticmethod
    def _read_file_format(ds) -> tuple[str, date | None]:
        version = ""
        file_meta = getattr(ds, "file_meta", None)
        raw_version = file_meta.get("FileMetaInformationVersion") if file_meta is not None else None
        if isinstance(raw_version, (bytes, bytearray)) and raw_version:
            version = str(int.from_bytes(raw_version, "big"))
        created = _parse_da(ds.get("InstanceCreationDate")) or _parse_da(ds.get("ContentDate"))
        return version, created

    def _read_patient(self, ds, acquired: datetime | None, meta: dict) -> PatientInfo:
        info = PatientInfo()
        info.patient_id = _text(ds.get("PatientID"))

        name = ds.get("PatientName")
        if name is not None:
            info.last_name = (getattr(name, "family_name", "") or "").strip()
            info.first_name = (getattr(name, "given_name", "") or "").strip()
            if not info.last_name and not info.first_name:
                info.last_name = str(name).strip()
            middle = (getattr(name, "middle_name", "") or "").strip()
            if middle:
                meta["patient_middle_name"] = middle

        sex = _text(ds.get("PatientSex")).upper()
        info.sex = sex if sex in ("M", "F") else "U"

        dob = _parse_da(ds.get("PatientBirthDate"))
        if dob is not None:
            info.birth_date = datetime(dob.year, dob.month, dob.day)
        age = _text(ds.get("PatientAge")).upper()
        if len(age) == 4 and age[:3].isdigit() and age[3] in "DWMY":
            n = int(age[:3])
            info.age = {"Y": n, "M": n // 12, "W": n // 52, "D": n // 365}[age[3]]
        elif dob is not None and acquired is not None:
            years = acquired.year - dob.year - (
                (acquired.month, acquired.day) < (dob.month, dob.day)
            )
            info.age = years if years >= 0 else None

        info.weight = _float(ds.get("PatientWeight"))
        size = _float(ds.get("PatientSize"))
        info.height = round(size * 100, 1) if size else None
        info.race = _text(ds.get("EthnicGroup"))
        info.clinical_history = _text(ds.get("AdditionalPatientHistory"))
        return info

    def _read_recording(self, ds, tz: timezone | None, meta: dict) -> RecordingInfo:
        info = RecordingInfo()
        candidates = [_parse_dt(ds.get("AcquisitionDateTime"), tz)]
        for date_attr, time_attr in (
            ("AcquisitionDate", "AcquisitionTime"),
            ("ContentDate", "ContentTime"),
            ("StudyDate", "StudyTime"),
        ):
            day = _parse_da(ds.get(date_attr))
            tm = _parse_tm(ds.get(time_attr))
            if day is not None and tm is not None:
                candidates.append(datetime(day.year, day.month, day.day, *tm, tzinfo=tz))
        info.date = next((c for c in candidates if c is not None), None)
        if info.date is None:
            # A date without a time of day: keep it, do not invent midnight
            for attr in ("AcquisitionDateTime", "AcquisitionDate", "ContentDate", "StudyDate"):
                day = _parse_da(ds.get(attr))
                if day is not None:
                    meta["acquisition_date"] = day.isoformat()
                    break

        info.referring_physician = _person(ds.get("ReferringPhysicianName"))
        info.technician = _person(ds.get("OperatorsName"))
        info.location = _text(ds.get("CurrentPatientLocation"))
        return info

    def _read_device(self, ds) -> DeviceInfo:
        info = DeviceInfo()
        info.manufacturer = _text(ds.get("Manufacturer"))
        info.model = _text(ds.get("ManufacturerModelName"))
        info.name = _text(ds.get("StationName"))
        info.software_version = _text(ds.get("SoftwareVersions"))
        info.institution = _text(ds.get("InstitutionName"))
        info.department = _text(ds.get("InstitutionalDepartmentName"))
        info.serial_number = _text(ds.get("DeviceSerialNumber"))
        sop_class = ds.get("SOPClassUID")
        name = getattr(sop_class, "name", "") if sop_class is not None else ""
        if name and name != str(sop_class):
            info.acquisition_type = name.replace(" Storage", "")
        return info

    @staticmethod
    def _read_acquisition_context(ds) -> list[dict]:
        items = []
        for item in ds.get("AcquisitionContextSequence") or []:
            _, _, name = _code_parts(_first_code(item, "ConceptNameCodeSequence"))
            _, _, value = _code_parts(_first_code(item, "ConceptCodeSequence"))
            entry = {"concept": name}
            if value:
                entry["value"] = value
            numeric = item.get("NumericValue")
            if numeric is not None:
                entry["numeric_value"] = _text(numeric)
            text = _text(item.get("TextValue"))
            if text:
                entry["text"] = text
            items.append(entry)
        return items

    # ------------------------------------------------------------------
    # Waveform data
    # ------------------------------------------------------------------

    def _read_group(self, wf, index: int, little_endian: bool) -> dict | None:
        """Decode one multiplex group, or return None when it holds no data."""
        try:
            num_channels = int(wf.get("NumberOfWaveformChannels") or 0)
            num_samples = int(wf.get("NumberOfWaveformSamples") or 0)
            bits = int(wf.get("WaveformBitsAllocated") or 0)
        except (TypeError, ValueError) as e:
            raise CorruptedFileError(f"DICOM multiplex group {index}: invalid size attribute: {e}") from e
        data = wf.get("WaveformData")
        if data is None or num_channels <= 0 or num_samples <= 0:
            return None

        interpretation = _text(wf.get("WaveformSampleInterpretation")).upper()
        if not interpretation:
            interpretation = _DEFAULT_INTERPRETATION.get(bits, "")
        dtype_code = _SAMPLE_DTYPES.get((bits, interpretation))
        if dtype_code is None:
            raise CorruptedFileError(
                f"DICOM multiplex group {index}: unsupported Waveform Bits Allocated "
                f"{bits} with Sample Interpretation {interpretation!r}"
            )
        dtype = np.dtype(dtype_code).newbyteorder("<" if little_endian else ">")
        expected = num_channels * num_samples * dtype.itemsize
        if len(data) < expected:
            raise CorruptedFileError(
                f"DICOM multiplex group {index}: Waveform Data has {len(data)} bytes, "
                f"expected {expected} ({num_samples} samples x {num_channels} channels "
                f"x {dtype.itemsize} bytes)"
            )
        codes = np.frombuffer(bytes(data[:expected]), dtype=dtype).reshape(num_samples, num_channels)

        padding = None
        pad_raw = wf.get("WaveformPaddingValue")
        if isinstance(pad_raw, (bytes, bytearray)) and len(pad_raw) >= dtype.itemsize:
            padding = np.frombuffer(bytes(pad_raw[:dtype.itemsize]), dtype=dtype)[0]

        if interpretation in _EXPANSION:
            values = _EXPANSION[interpretation][codes].astype(np.float64)
        else:
            values = codes.astype(np.float64)
        if padding is not None:
            values[codes == padding] = np.nan

        rate_exact = _float(wf.get("SamplingFrequency")) or 0.0
        rate = int(round(rate_exact))
        label = _text(wf.get("MultiplexGroupLabel"))

        channel_defs = list(wf.get("ChannelDefinitionSequence") or [])
        leads: list[Lead] = []
        labels: list[str] = []
        filter_rows: list[dict] = []
        for ch in range(num_channels):
            ch_def = channel_defs[ch] if ch < len(channel_defs) else None
            lead, lead_label, filt = self._read_channel(ch_def, ch, values[:, ch], rate)
            leads.append(lead)
            labels.append(lead_label)
            filter_rows.append(filt)
        for lead, unique in zip(leads, unique_labels(labels)):
            lead.label = unique

        filters = FilterSettings()
        first = filter_rows[0] if filter_rows else {}
        filters.highpass = first.get("highpass")
        filters.lowpass = first.get("lowpass")
        filters.notch = first.get("notch")
        if filters.notch is not None:
            filters.notch_active = True
        channel_filters = []
        if any(row != first for row in filter_rows):
            channel_filters = [dict(row, label=lead.label) for row, lead in zip(filter_rows, leads)]

        stored = wf.get("WaveformBitsStored")
        sig = SignalCharacteristics(
            sampling_rate=rate,
            bits_per_sample=int(stored) if stored else bits,
            signal_signed=interpretation in ("SB", "SS", "SL", "SV", "MB", "AB"),
            number_channels_allocated=num_channels,
            number_channels_valid=len(leads),
            data_encoding={"MB": "mu-law", "AB": "a-law"}.get(interpretation, "pcm"),
            compression="none",
        )

        upper = label.upper()
        return {
            "index": index + 1,
            "label": label,
            "is_median": any(word in upper for word in _MEDIAN_GROUP_WORDS),
            "leads": leads,
            "filters": filters,
            "channel_filters": channel_filters,
            "signal": sig,
            "num_samples": num_samples,
            "sampling_frequency": rate_exact,
            "interpretation": interpretation,
        }

    def _read_channel(self, ch_def, ch: int, samples: np.ndarray, rate: int) -> tuple[Lead, str, dict]:
        annotations: dict[str, str] = {}
        label = ""
        filt: dict = {}
        sensitivity = correction = baseline = None
        unit_norm = None
        unit_text = ""

        if ch_def is not None:
            scheme, value, meaning = _code_parts(_first_code(ch_def, "ChannelSourceSequence"))
            label = _lead_from_code(scheme, value) or _lead_from_meaning(meaning)
            if value:
                annotations["source_code"] = f"{scheme}:{value}" if scheme else value
            if meaning:
                annotations["source_meaning"] = meaning
            channel_label = _text(ch_def.get("ChannelLabel"))
            if channel_label:
                annotations["channel_label"] = channel_label
                label = label or normalize_lead_label(channel_label)
            status = _text(ch_def.get("ChannelStatus"))
            if status:
                annotations["channel_status"] = status

            sensitivity = _float(ch_def.get("ChannelSensitivity"))
            correction = _float(ch_def.get("ChannelSensitivityCorrectionFactor"))
            baseline = _float(ch_def.get("ChannelBaseline"))
            unit_item = _first_code(ch_def, "ChannelSensitivityUnitsSequence")
            _, unit_value, unit_meaning = _code_parts(unit_item)
            unit_text = unit_value or unit_meaning
            for candidate in (unit_value, unit_meaning):
                if candidate and _normalize_unit(candidate):
                    unit_norm = _normalize_unit(candidate)
                    break

            for key, attr in (
                ("highpass", "FilterLowFrequency"),
                ("lowpass", "FilterHighFrequency"),
                ("notch", "NotchFilterFrequency"),
            ):
                freq = _float(ch_def.get(attr))
                if freq:
                    filt[key] = freq
            skew = _float(ch_def.get("ChannelSampleSkew"))
            if skew:
                annotations["sample_skew_ms"] = str(skew)

        if sensitivity is not None and unit_norm is not None:
            resolution = sensitivity * (correction if correction is not None else 1.0)
            resolution_unit = unit_norm
            offset = baseline or 0.0
        else:
            # No voltage unit: the scale cannot be expressed, keep it for reference
            resolution, resolution_unit, offset = 1.0, "", 0.0
            if correction is not None and correction != 1.0:
                annotations["sensitivity_correction_factor"] = str(correction)
            if baseline:
                annotations["channel_baseline"] = str(baseline)

        is_raw = derive_is_raw(resolution, offset, resolution_unit)
        lead = Lead(
            label=label or f"Ch{ch + 1}",
            samples=np.ascontiguousarray(samples),
            sampling_rate=rate,
            resolution=resolution,
            resolution_unit=resolution_unit,
            offset=offset,
            units="" if is_raw else resolution_unit,
            is_raw=is_raw,
            adc_resolution=sensitivity or 0.0,
            adc_resolution_unit=unit_text if sensitivity is not None else "",
            prefiltering=", ".join(
                f"{name} {filt[key]:g} Hz"
                for key, name in (("highpass", "HP"), ("lowpass", "LP"), ("notch", "notch"))
                if key in filt
            ),
            annotations=annotations,
        )
        return lead, lead.label, filt

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    def _read_annotations(self, ds, record: ECGRecord, channel_labels: dict) -> None:
        """Split the Waveform Annotation Sequence into measurements,
        statements and time-referenced (fiducial) annotations."""
        items = list(ds.get("WaveformAnnotationSequence") or [])
        for wf in ds.get("WaveformSequence") or []:
            items.extend(wf.get("WaveformAnnotationSequence") or [])
        if not items:
            return

        statements: list[tuple[str, str]] = []
        measurements: dict[str, int] = {}
        fiducials: list[dict] = []
        for item in items:
            scheme, code, name = _code_parts(_first_code(item, "ConceptNameCodeSequence"))
            _, _, concept = _code_parts(_first_code(item, "ConceptCodeSequence"))
            text = _text(item.get("UnformattedTextValue"))
            numeric = _float(item.get("NumericValue"))
            _, unit_value, unit_meaning = _code_parts(_first_code(item, "MeasurementUnitsCodeSequence"))
            unit = unit_value or unit_meaning
            channels = self._referenced_channels(item, channel_labels)

            positions = None
            for attr in ("ReferencedSamplePositions", "ReferencedTimeOffsets", "ReferencedDateTime"):
                if item.get(attr) is not None:
                    positions = (attr, _text(item.get(attr)))
                    break
            if positions is not None:
                entry = {"concept": concept or name or text, positions[0]: positions[1]}
                if code:
                    entry["code"] = code
                if numeric is not None:
                    entry["value"] = numeric
                    entry["unit"] = unit
                if channels:
                    entry["channels"] = channels
                fiducials.append(entry)
                continue

            if numeric is not None:
                field_name = None if channels else self._measurement_field(code, name)
                converted = self._convert_measurement(field_name, numeric, unit)
                if field_name and converted is not None:
                    measurements[field_name] = converted
                else:
                    key = name or code or "measurement"
                    if channels:
                        key = f"{key} ({', '.join(channels)})"
                    record.annotations[key] = f"{numeric:g} {unit}".strip()
                continue

            if text:
                statements.append((text, ""))
            elif concept:
                statements.append((concept, ""))
            elif name:
                statements.append((name, ""))

        if statements:
            record.interpretation = Interpretation(statements=statements, source="machine")
        if fiducials:
            record.raw_metadata["waveform_annotations"] = fiducials
        meas = GlobalMeasurements()
        for field_name, value in measurements.items():
            setattr(meas, field_name, value)
        record.measurements = meas

    @staticmethod
    def _referenced_channels(item, channel_labels: dict) -> list[str]:
        refs = item.get("ReferencedWaveformChannels")
        if refs is None:
            return []
        values = list(refs) if type(refs).__name__ == "MultiValue" or isinstance(refs, list) else [refs]
        result = []
        for i in range(0, len(values) - 1, 2):
            key = (int(values[i]), int(values[i + 1]))
            if key[1] == 0:
                result.append(f"group {key[0]}")
            else:
                result.append(channel_labels.get(key, f"group {key[0]} channel {key[1]}"))
        return result

    @staticmethod
    def _measurement_field(code: str, meaning: str) -> str | None:
        lower = meaning.lower()
        if "framingham" in lower or "hodges" in lower:
            return None
        if "bazett" in lower:
            return "qtc_bazett"
        if "fridericia" in lower or "fredericia" in lower:
            return "qtc_fridericia"
        return _MEASUREMENT_CODES.get(code) or _MEASUREMENT_CODES.get(meaning)

    @staticmethod
    def _convert_measurement(field_name: str | None, value: float, unit: str) -> int | None:
        """Convert to the unit of the GlobalMeasurements field (ms, bpm, deg)."""
        if field_name is None:
            return None
        unit = unit.strip()
        if field_name.endswith("_axis"):
            if unit in ("rad",):
                value = float(np.degrees(value))
            elif unit not in ("", "deg", "°", "degree", "degrees"):
                return None
        elif field_name == "heart_rate":
            if unit not in ("", "/min", "{beats}/min", "{beat}/min", "1/min", "bpm", "beats/min"):
                return None
        else:
            factors = {"": 1.0, "ms": 1.0, "s": 1000.0, "us": 0.001}
            if unit not in factors:
                return None
            value *= factors[unit]
        return int(round(value))

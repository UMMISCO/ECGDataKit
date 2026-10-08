"""GE MAC 2000 XML format parser (beta).

No public specification or sample of the MAC 2000 XML export is available.
The element names follow the GE MUSE conventions (PatientDemographics,
TestDemographics, Waveform/LeadData/WaveFormData, LeadAmplitudeUnitsPerBit,
RestingECGMeasurements, Diagnosis/DiagnosisStatement/StmtText), which is
how this parser reads them. Distinct from the GE MUSE XML parser, which
handles ``<RestingECG>`` documents.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from xml.parsers.expat import ExpatError

import xmltodict

from ecgdatakit.exceptions import CorruptedFileError, MissingElementError
from ecgdatakit.models import (
    DeviceInfo,
    ECGRecord,
    FileFormatInfo,
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
from ecgdatakit.parsing.helpers import (
    fill_signal_summary,
    normalize_lead_label,
    unique_labels,
)
from ecgdatakit.parsing.helpers.waveform_text import decode_int16_text
from ecgdatakit.parsing.helpers.xml import find_tag
from ecgdatakit.parsing.parser import Parser

_ROOT_TAGS = ("mac2000", "mac_2000", "ge_mac", "macexport")

_DATE_FORMATS = ("%m-%d-%Y", "%m/%d/%Y", "%Y-%m-%d", "%Y%m%d", "%d-%b-%Y")
_TIME_FORMATS = ("%H:%M:%S", "%H:%M")

_GE_UNITS = {"MICROVOLTS": "uV", "MILLIVOLTS": "mV", "NANOVOLTS": "nV", "VOLTS": "V"}

_HEIGHT_TAGS = (("PatientHeightCM", 1.0), ("HeightCM", 1.0),
                ("PatientHeightIN", 2.54), ("HeightIN", 2.54))
_WEIGHT_TAGS = (("PatientWeightKG", 1.0), ("WeightKG", 1.0),
                ("PatientWeightLBS", 0.45359237), ("WeightLBS", 0.45359237))
_UNIT_TO_CM = {"CM": 1.0, "M": 100.0, "IN": 2.54, "INCHES": 2.54}
_UNIT_TO_KG = {"KG": 1.0, "G": 0.001, "LB": 0.45359237, "LBS": 0.45359237}

_MEASUREMENTS = {
    "VentricularRate": "heart_rate",
    "PRInterval": "pr_interval",
    "QRSDuration": "qrs_duration",
    "QTInterval": "qt_interval",
    "QTCorrected": None,  # formula not stated, kept in annotations only
    "QTcFrederica": "qtc_fridericia",
    "QTCFredericia": "qtc_fridericia",
    "PAxis": "p_axis",
    "RAxis": "qrs_axis",
    "TAxis": "t_axis",
    "RRInterval": "rr_interval",
    "QRSCount": "qrs_count",
    "NumQRS": "qrs_count",
}


def _text(value) -> str:
    """Text of an xmltodict value (string, element with attributes, or list)."""
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, dict):
        value = value.get("#text")
    return str(value).strip() if value is not None else ""


def _child(node, tag: str):
    """Direct child *tag* of *node* (case-insensitive), or ``None``."""
    if not isinstance(node, dict):
        return None
    lower = tag.lower()
    for key, value in node.items():
        if key.lower() == lower:
            return value
    return None


def _number(value) -> float | None:
    text = _text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _parse_date(text: str) -> datetime | None:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    return None


def _with_time(day: datetime | None, time_str: str) -> datetime | None:
    """Combine a date with a time; ``None`` when the time is missing or invalid."""
    if day is None or not time_str:
        return None
    for fmt in _TIME_FORMATS:
        try:
            t = datetime.strptime(time_str.strip(), fmt)
            return day.replace(hour=t.hour, minute=t.minute, second=t.second)
        except ValueError:
            continue
    return None


def _years_between(start: datetime, end: datetime) -> int:
    return end.year - start.year - ((end.month, end.day) < (start.month, start.day))


class GEMAC2000Parser(Parser):
    """Parser for GE MAC 2000 XML ECG files (beta)."""

    FORMAT_NAME = "GE MAC 2000"
    FORMAT_DESCRIPTION = "GE MAC 2000 XML ECG format (beta)"
    FILE_EXTENSIONS = [".xml"]
    PRIORITY = 50

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        upper = header.decode("utf-8", errors="ignore").upper()
        return any(f"<{tag.upper()}" in upper for tag in _ROOT_TAGS)

    def parse(self, file_path: Path) -> ECGRecord:
        try:
            with open(file_path, "rb") as f:
                doc = xmltodict.parse(f.read())
        except ExpatError as e:
            raise CorruptedFileError(f"Malformed GE MAC 2000 XML in {file_path}: {e}") from e

        root = next((v for k, v in doc.items() if k.lower() in _ROOT_TAGS), None)
        if not isinstance(root, dict):
            raise CorruptedFileError(f"No recognized root element in {file_path}")

        record = ECGRecord(source_format="ge_mac2000")
        raw = record.raw_metadata

        test = self._first(root, ("TestDemographics", "AcquisitionInfo", "RecordingInfo"))
        record.recording = self._read_recording(test, raw)
        record.recording.device = self._read_device(root, test)
        record.recording.acquisition.filters = self._read_filters(root)
        record.patient = self._read_patient(root, record.recording.date, raw)

        default_rate = _number(_child(test, "SampleRate")) or _number(_child(test, "SampleBase"))
        record.leads, record.median_beats, encodings = self._read_waveforms(
            root, default_rate, file_path)
        if not record.leads:
            raise CorruptedFileError(f"No rhythm waveform data in {file_path}")

        lead = record.leads[0]
        record.recording.duration = timedelta(seconds=lead.samples.size / lead.sampling_rate)

        record.annotations, record.measurements = self._read_measurements(root)
        record.interpretation = self._read_interpretation(root, record.annotations)

        record.recording.acquisition.signal = SignalCharacteristics(
            sampling_rate=lead.sampling_rate,
            signal_signed=True,
            number_channels_valid=len(record.leads),
            data_encoding=encodings.pop() if len(encodings) == 1 else "",
            compression="none",
        )

        version = _text(find_tag(root, "MuseVersion")) or _text(root.get("@version"))
        if version:
            record.file_format = FileFormatInfo(version=version)

        fill_signal_summary(record)
        return record

    @staticmethod
    def _first(root: dict, tags: tuple[str, ...]):
        for tag in tags:
            node = find_tag(root, tag)
            if isinstance(node, list):
                node = node[0]
            if isinstance(node, dict):
                return node
        return None

    def _read_patient(self, root: dict, acq_date: datetime | None, raw: dict) -> PatientInfo:
        info = PatientInfo()
        demo = self._first(root, ("PatientDemographics", "PatientInfo", "Patient"))
        if demo is None:
            return info

        def get(*tags: str) -> str:
            for tag in tags:
                value = _text(_child(demo, tag))
                if value:
                    return value
            return ""

        info.patient_id = get("PatientID", "ID")
        info.first_name = get("PatientFirstName", "FirstName")
        info.last_name = get("PatientLastName", "LastName")

        sex = get("Gender", "Sex").upper()
        if sex:
            info.sex = "M" if sex in ("M", "MALE") else "F" if sex in ("F", "FEMALE") else "U"

        dob = get("DateofBirth", "DateOfBirth", "DOB")
        if dob:
            info.birth_date = _parse_date(dob)

        age = _number(get("PatientAge", "Age"))
        units = get("AgeUnits").upper()
        if age is not None:
            if units in ("", "Y", "YEARS"):
                info.age = int(age)
            else:
                raw["age"] = {"value": age, "units": units}
        if info.age is None and info.birth_date is not None and acq_date is not None:
            info.age = _years_between(info.birth_date, acq_date)

        for tag, factor in _HEIGHT_TAGS:
            value = _number(_child(demo, tag))
            if value:
                info.height = round(value * factor, 2)
                break
        else:
            info.height = self._with_unit(_child(demo, "Height") or _child(demo, "PatientHeight"),
                                          _UNIT_TO_CM, "height", raw)
        for tag, factor in _WEIGHT_TAGS:
            value = _number(_child(demo, tag))
            if value:
                info.weight = round(value * factor, 3)
                break
        else:
            info.weight = self._with_unit(_child(demo, "Weight") or _child(demo, "PatientWeight"),
                                          _UNIT_TO_KG, "weight", raw)

        race = get("Race", "Ethnicity")
        if race:
            info.race = race
        return info

    @staticmethod
    def _with_unit(node, table: dict, key: str, raw: dict) -> float | None:
        """Convert a value whose unit is given by a ``units`` attribute."""
        value = _number(node)
        if not value:
            return None
        unit = ""
        if isinstance(node, dict):
            unit = str(node.get("@units") or node.get("@Units") or node.get("@unit") or "").upper()
        if unit in table:
            return round(value * table[unit], 3)
        raw[key] = {"value": value, "units": unit}
        return None

    def _read_recording(self, test, raw: dict) -> RecordingInfo:
        info = RecordingInfo()
        if test is None:
            return info

        def get(*tags: str) -> str:
            for tag in tags:
                value = _text(_child(test, tag))
                if value:
                    return value
            return ""

        date_str = get("AcquisitionDate", "Date")
        time_str = get("AcquisitionTime", "Time")
        day = _parse_date(date_str) if date_str else None
        info.date = _with_time(day, time_str)
        if day is not None and info.date is None:
            # No time of day in the file: keep the date without inventing one
            raw["acquisition_date"] = day.date().isoformat()

        info.location = get("LocationName", "Location")
        info.room = get("RoomID", "Room")
        info.technician = get("TechnicianID", "OperatorID", "Technician",
                              "AcquisitionTechID")
        info.referring_physician = get("ReferringPhysician", "ReferringMDLastName",
                                       "OrderingPhysician", "OrderingMDLastName",
                                       "AttendingPhysician")
        return info

    @staticmethod
    def _read_device(root: dict, test) -> DeviceInfo:
        def get(*tags: str) -> str:
            for node in (test, root):
                for tag in tags:
                    value = _text(find_tag(node, tag)) if node is not None else ""
                    if value:
                        return value
            return ""

        return DeviceInfo(
            manufacturer="GE",
            model=get("DeviceModel", "AcquisitionDevice") or "MAC 2000",
            software_version=get("SoftwareVersion", "AcquisitionSoftwareVersion"),
            serial_number=get("SerialNumber", "DeviceSerialNumber"),
            institution=get("SiteName", "Institution"),
            department=get("Department", "DepartmentName"),
        )

    @staticmethod
    def _read_filters(root: dict) -> FilterSettings:
        filters = FilterSettings()
        node = find_tag(root, "FilterSettings")
        if isinstance(node, list):
            node = node[0]
        if not isinstance(node, dict):
            node = root

        def hz(tag: str) -> float | None:
            value = _number(find_tag(node, tag))
            return value if value else None

        filters.highpass = hz("HighPassFilter")
        filters.lowpass = hz("LowPassFilter")
        filters.notch = hz("NotchFilter") or hz("ACFilter")
        if filters.notch:
            filters.notch_active = True
        return filters

    def _read_waveforms(self, root: dict, default_rate: float | None, file_path: Path):
        rhythm: list[Lead] = []
        median: list[Lead] = []
        encodings: set[str] = set()

        waveforms = []
        for tag in ("Waveform", "WaveformData", "Leads"):
            waveforms = [w for w in _as_list(find_tag(root, tag)) if isinstance(w, dict)]
            if waveforms:
                break

        for wf in waveforms:
            wf_type = _text(_child(wf, "WaveformType")).lower()
            rate = (_number(_child(wf, "SampleBase")) or _number(_child(wf, "SampleRate"))
                    or default_rate)
            if not rate or rate <= 0:
                raise MissingElementError(f"No sampling rate (SampleBase) in {file_path}")

            # Waveform-level scale: direct children only, per-lead values override
            wf_scale = (_number(_child(wf, "LeadAmplitudeUnitsPerBit"))
                        or _number(_child(wf, "AmplitudeResolution")))
            wf_unit = _text(_child(wf, "LeadAmplitudeUnits"))

            nodes = []
            for tag in ("LeadData", "Lead", "Channel"):
                nodes = [n for n in _as_list(_child(wf, tag)) if isinstance(n, dict)]
                if nodes:
                    break

            leads: list[Lead] = []
            for i, node in enumerate(nodes):
                label = (_text(_child(node, "LeadID")) or _text(node.get("@Name"))
                         or _text(node.get("@name")) or _text(node.get("@NAME"))
                         or _text(_child(node, "Name")) or f"CH{i + 1}")
                data = (_text(_child(node, "WaveFormData")) or _text(_child(node, "Data"))
                        or _text(node.get("@Data")) or _text(node.get("@DATA"))
                        or _text(node.get("#text")))
                try:
                    samples, encoding = decode_int16_text(data)
                except ValueError as e:
                    raise CorruptedFileError(f"Lead {label} in {file_path}: {e}") from e
                if samples.size == 0:
                    raise CorruptedFileError(f"Lead {label} in {file_path} has no waveform data")
                encodings.add(encoding)

                scale = _number(_child(node, "LeadAmplitudeUnitsPerBit")) or wf_scale
                unit_text = _text(_child(node, "LeadAmplitudeUnits")) or wf_unit
                unit = ""
                if scale:
                    # GE amplitudes are in microvolts unless stated otherwise
                    unit = (_GE_UNITS.get(unit_text.upper()) or _normalize_unit(unit_text)
                            or "uV") if unit_text else "uV"
                resolution = scale or 1.0
                is_raw = derive_is_raw(resolution, 0.0, unit)
                leads.append(Lead(
                    label=normalize_lead_label(label),
                    samples=samples,
                    sampling_rate=int(round(rate)),
                    resolution=resolution,
                    resolution_unit=unit,
                    units="" if is_raw else unit,
                    is_raw=is_raw,
                    adc_resolution=scale or 0.0,
                    adc_resolution_unit=unit,
                ))
            for lead, unique in zip(leads, unique_labels([l.label for l in leads])):
                lead.label = unique

            if wf_type == "median":
                median = leads
            elif wf_type == "rhythm" or not rhythm:
                rhythm = leads
        return rhythm, median, encodings

    @staticmethod
    def _read_measurements(root: dict) -> tuple[dict[str, str], GlobalMeasurements]:
        annotations: dict[str, str] = {}
        m = GlobalMeasurements()
        node = None
        for tag in ("RestingECGMeasurements", "Measurements", "OriginalRestingECGMeasurements"):
            node = find_tag(root, tag)
            if isinstance(node, list):
                node = node[0]
            if isinstance(node, dict):
                break
        if not isinstance(node, dict):
            return annotations, m
        for key, attr in _MEASUREMENTS.items():
            value = _text(_child(node, key))
            if not value:
                continue
            annotations[key.lower()] = value
            num = _number(value)
            if num is not None and attr and getattr(m, attr) is None:
                setattr(m, attr, int(round(num)))
        return annotations, m

    @staticmethod
    def _statements(node) -> list[tuple[str, str]]:
        """Statements of a Diagnosis-like node (DiagnosisStatement/StmtText)."""
        out: list[tuple[str, str]] = []
        for item in _as_list(node):
            if isinstance(item, str):
                out.extend((s.strip(), "") for s in item.splitlines() if s.strip())
                continue
            if not isinstance(item, dict):
                continue
            stmts = _child(item, "DiagnosisStatement") or _child(item, "Statement")
            if stmts is not None:
                for stmt in _as_list(stmts):
                    if isinstance(stmt, dict):
                        parts = _as_list(_child(stmt, "StmtText") or _child(stmt, "Text")
                                         or stmt.get("#text"))
                        text = " ".join(_text(p) for p in parts if _text(p))
                    else:
                        text = _text(stmt)
                    if text:
                        out.append((text, ""))
                continue
            text = _text(_child(item, "StmtText") or _child(item, "Text") or item.get("#text"))
            out.extend((s.strip(), "") for s in text.splitlines() if s.strip())
        return out

    def _read_interpretation(self, root: dict, annotations: dict[str, str]) -> Interpretation:
        overread = find_tag(root, "OverreadDiagnosis")
        original = find_tag(root, "OriginalDiagnosis")
        diagnosis = find_tag(root, "Diagnosis")
        if diagnosis is None:
            diagnosis = find_tag(root, "Interpretation")

        expert_node, expert_source = None, ""
        if overread is not None and self._statements(overread):
            expert_node, expert_source = overread, "overread"
            machine_node = original if original is not None else diagnosis
        elif original is not None and diagnosis is not None and self._statements(diagnosis):
            # MUSE convention: Diagnosis is the edited/confirmed version of
            # OriginalDiagnosis (the device statements)
            expert_node, expert_source = diagnosis, "confirmed"
            machine_node = original
        else:
            machine_node = original if original is not None else diagnosis

        machine = self._statements(machine_node) if machine_node is not None else []
        interp = Interpretation()
        if expert_node is not None:
            interp.statements = self._statements(expert_node)
            interp.source = expert_source
            interp.interpreter = " ".join(p for p in (
                _text(find_tag(root, "OverreaderFirstName")) or _text(find_tag(root, "EditorFirstName")),
                _text(find_tag(root, "OverreaderLastName")) or _text(find_tag(root, "EditorLastName")),
            ) if p) or _text(find_tag(root, "ConfirmedBy"))
            edit_date = _parse_date(_text(find_tag(root, "EditDate")) or "")
            interp.interpretation_date = _with_time(edit_date, _text(find_tag(root, "EditTime")))
            if machine:
                annotations["machine_interpretation"] = "\n".join(s for s, _ in machine)
        elif machine:
            interp.statements = machine
            interp.source = "machine"

        severity = _text(find_tag(root, "Severity")) or _text(find_tag(root, "DiagnosisSeverity"))
        if severity and interp.statements:
            interp.severity = severity.upper()
        return interp

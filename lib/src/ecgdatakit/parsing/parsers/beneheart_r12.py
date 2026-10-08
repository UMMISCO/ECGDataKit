"""Mindray BeneHeart R12 XML format parser (beta).

No public description of the BeneHeart R12 native XML export exists (the
device can also export HL7 aECG, which the HL7 aECG parser handles). The
element names below are inferred; amplitude scaling is applied only when
the file states a resolution, otherwise samples stay raw counts.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from xml.parsers.expat import ExpatError

import xmltodict

from ecgdatakit.exceptions import CorruptedFileError, MissingElementError
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
from ecgdatakit.parsing.helpers import (
    STANDARD_LEADS,
    fill_signal_summary,
    normalize_lead_label,
    unique_labels,
)
from ecgdatakit.parsing.helpers.waveform_text import decode_int16_text
from ecgdatakit.parsing.helpers.xml import find_tag
from ecgdatakit.parsing.parser import Parser

_ROOT_TAGS = ("beneheartr12", "mindrayecg", "ecgdata")

_UNIT_TO_CM = {"CM": 1.0, "M": 100.0, "IN": 2.54, "INCH": 2.54}
_UNIT_TO_KG = {"KG": 1.0, "G": 0.001, "LB": 0.45359237, "LBS": 0.45359237}

_MEASUREMENTS = (
    ("heart_rate", ("VentricularRate", "HeartRate")),
    ("pr_interval", ("PRInterval",)),
    ("qrs_duration", ("QRSDuration",)),
    ("qt_interval", ("QTInterval",)),
    ("qtc_bazett", ("QTcBazett",)),
    ("qtc_fridericia", ("QTcFridericia", "QTcFredericia")),
    ("p_axis", ("PAxis",)),
    ("qrs_axis", ("QRSAxis", "RAxis")),
    ("t_axis", ("TAxis",)),
    ("rr_interval", ("RRInterval", "RR")),
    ("qrs_count", ("QRSCount", "NumQRS")),
)

_STATEMENT_TAGS = ("DiagnosisStatement", "Statement", "StmtText", "Text", "Description")


def _text(value) -> str:
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, dict):
        value = value.get("#text")
    return str(value).strip() if value is not None else ""


def _child(node, tag: str):
    """Direct child or attribute *tag* of *node* (case-insensitive)."""
    if not isinstance(node, dict):
        return None
    lower = tag.lower()
    for key, value in node.items():
        if key.lower() in (lower, "@" + lower):
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


def _parse_dob(text: str) -> datetime | None:
    """Parse ISO-like dates; d/m vs m/d only when unambiguous."""
    text = text.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if m:
        a, b, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            if a > 12 >= b:
                return datetime(year, b, a)
            if b > 12 >= a:
                return datetime(year, a, b)
        except ValueError:
            return None
    return None


def _years_between(start: datetime, end: datetime) -> int:
    return end.year - start.year - ((end.month, end.day) < (start.month, start.day))


class BeneHeartR12Parser(Parser):
    """Parser for Mindray BeneHeart R12 XML ECG files (beta)."""

    FORMAT_NAME = "Mindray BeneHeart R12"
    FORMAT_DESCRIPTION = "Mindray BeneHeart R12 XML ECG format (beta)"
    FILE_EXTENSIONS = [".xml"]
    PRIORITY = 50

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        upper = header.decode("utf-8", errors="ignore").upper()
        if "<BENEHEARTR12" in upper or "<MINDRAYECG" in upper:
            return True
        return "<ECGDATA" in upper and "MINDRAY" in upper

    def parse(self, file_path: Path) -> ECGRecord:
        try:
            with open(file_path, "rb") as f:
                doc = xmltodict.parse(f.read())
        except ExpatError as e:
            raise CorruptedFileError(f"Malformed BeneHeart XML in {file_path}: {e}") from e

        root = next((v for k, v in doc.items() if k.lower() in _ROOT_TAGS), None)
        if not isinstance(root, dict):
            raise CorruptedFileError(f"No recognized root element in {file_path}")

        record = ECGRecord(source_format="beneheart_r12")
        raw = record.raw_metadata

        acq = self._first(root, ("AcquisitionInfo", "RecordingInfo", "TestInfo"))
        record.recording = self._read_recording(acq, raw)
        record.patient = self._read_patient(root, record.recording.date, raw)
        record.leads, encodings = self._read_leads(root, acq, file_path)
        if not record.leads:
            raise CorruptedFileError(f"No lead waveform data in {file_path}")
        record.recording.device = self._read_device(root, acq)
        record.recording.acquisition.filters = self._read_filters(root)
        record.interpretation, record.measurements, extra = self._read_annotations(root)
        record.annotations.update(extra)

        clinical: dict[str, str] = {}
        for tag in ("ClinicalInfo", "OrderInfo", "Indication"):
            node = find_tag(root, tag)
            if isinstance(node, list):
                node = node[0]
            if isinstance(node, dict):
                for key, value in node.items():
                    text = _text(value)
                    if text and not key.startswith("#"):
                        clinical[key.lstrip("@")] = text
            elif _text(node):
                clinical[tag] = _text(node)
        if clinical:
            raw["clinical_info"] = clinical
            record.patient.clinical_history = "\n".join(f"{k}: {v}" for k, v in clinical.items())

        lead = record.leads[0]
        record.recording.duration = timedelta(seconds=lead.samples.size / lead.sampling_rate)

        record.recording.acquisition.signal = SignalCharacteristics(
            sampling_rate=lead.sampling_rate,
            signal_signed=True,
            number_channels_valid=len(record.leads),
            data_encoding=encodings.pop() if len(encodings) == 1 else "",
            compression="none",
        )

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
        demo = self._first(root, ("PatientInfo", "Patient", "Demographics"))
        if demo is None:
            return info

        def get(*tags: str) -> str:
            for tag in tags:
                value = _text(_child(demo, tag))
                if value:
                    return value
            return ""

        info.patient_id = get("PatientID", "ID")
        info.first_name = get("FirstName", "GivenName")
        info.last_name = get("LastName", "FamilyName")

        sex = get("Sex", "Gender").upper()
        if sex:
            info.sex = "M" if sex in ("M", "MALE") else "F" if sex in ("F", "FEMALE") else "U"

        dob = get("DateOfBirth", "BirthDate", "DOB")
        if dob:
            info.birth_date = _parse_dob(dob)
            if info.birth_date is None:
                raw["birth_date"] = dob

        age = _number(_child(demo, "Age"))
        if age is not None:
            info.age = int(age)
        elif info.birth_date is not None and acq_date is not None:
            info.age = _years_between(info.birth_date, acq_date)

        info.height = self._with_unit(_child(demo, "Height") or _child(demo, "PatientHeight"),
                                      _UNIT_TO_CM, "height", raw)
        info.weight = self._with_unit(_child(demo, "Weight") or _child(demo, "PatientWeight"),
                                      _UNIT_TO_KG, "weight", raw)
        race = get("Race", "Ethnicity")
        if race:
            info.race = race
        return info

    @staticmethod
    def _with_unit(node, table: dict, key: str, raw: dict) -> float | None:
        """Convert a value whose unit is given by a unit attribute; else keep it raw."""
        value = _number(node)
        if not value:
            return None
        unit = ""
        if isinstance(node, dict):
            unit = _text(_child(node, "Unit") or _child(node, "Units")).upper()
        if unit in table:
            return round(value * table[unit], 3)
        raw[key] = {"value": value, "units": unit}
        return None

    def _read_recording(self, acq, raw: dict) -> RecordingInfo:
        info = RecordingInfo()
        if acq is None:
            return info
        date_str = _text(_child(acq, "AcquisitionDate") or _child(acq, "Date"))
        time_str = _text(_child(acq, "AcquisitionTime") or _child(acq, "Time"))
        if not date_str:
            return info
        try:
            if "T" in date_str:
                info.date = datetime.fromisoformat(date_str)
                return info
        except ValueError:
            pass
        day = None
        for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"):
            try:
                day = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
        if day is None:
            raw["acquisition_date"] = date_str
            return info
        for fmt in ("%H:%M:%S", "%H%M%S", "%H:%M"):
            try:
                t = datetime.strptime(time_str, fmt)
                info.date = day.replace(hour=t.hour, minute=t.minute, second=t.second)
                break
            except ValueError:
                continue
        if info.date is None:
            # No time of day in the file: keep the date without inventing one
            raw["acquisition_date"] = day.date().isoformat()
        return info

    @staticmethod
    def _resolution(node) -> tuple[float | None, str]:
        """Resolution and its unit from a node's children or attributes."""
        for tag in ("Resolution", "AmplitudeResolution"):
            value = _child(node, tag)
            res = _number(value)
            if res:
                unit = ""
                if isinstance(value, dict):
                    unit = _text(_child(value, "Unit") or _child(value, "Units"))
                unit = unit or _text(_child(node, "ResolutionUnit") or _child(node, "Unit")
                                     or _child(node, "Units"))
                return res, (_normalize_unit(unit) or "") if unit else ""
        return None, ""

    def _read_leads(self, root: dict, acq, file_path: Path) -> tuple[list[Lead], set]:
        container = self._first(root, ("Leads", "WaveformData", "Channels"))
        if container is None:
            return [], set()

        rate = (_number(_child(container, "SampleRate")) or _number(_child(container, "SamplingRate"))
                or _number(_child(acq, "SampleRate")) or _number(_child(acq, "SamplingRate"))
                or _number(find_tag(root, "SampleRate")) or _number(find_tag(root, "SamplingRate")))
        if not rate or rate <= 0:
            raise MissingElementError(f"No sampling rate in {file_path}")

        default_res, default_unit = self._resolution(container)
        if default_res is None:
            default_res, default_unit = self._resolution(acq)

        entries: list[tuple[str, object, dict | None]] = []
        nodes = _as_list(_child(container, "Lead")) or _as_list(_child(container, "Channel"))
        if nodes:
            for i, node in enumerate(nodes):
                if not isinstance(node, dict):
                    entries.append((f"CH{i + 1}", node, None))
                    continue
                label = (_text(_child(node, "Name")) or _text(_child(node, "Label"))
                         or f"CH{i + 1}")
                data = (_text(_child(node, "Data")) or _text(node.get("#text"))
                        or _text(_child(node, "Samples")))
                entries.append((label, data, node))
        else:
            # One child element per lead, named after the lead
            for label in STANDARD_LEADS:
                node = _child(container, label)
                if node is not None:
                    entries.append((label, _text(node), node if isinstance(node, dict) else None))

        leads: list[Lead] = []
        encodings: set[str] = set()
        for label, data, node in entries:
            try:
                samples, encoding = decode_int16_text(_text(data))
            except ValueError as e:
                raise CorruptedFileError(f"Lead {label} in {file_path}: {e}") from e
            if samples.size == 0:
                raise CorruptedFileError(f"Lead {label} in {file_path} has no waveform data")
            encodings.add(encoding)

            res, unit = self._resolution(node) if node is not None else (None, "")
            if res is None:
                res, unit = default_res, default_unit
            resolution = res or 1.0
            is_raw = derive_is_raw(resolution, 0.0, unit)
            leads.append(Lead(
                label=normalize_lead_label(label),
                samples=samples,
                sampling_rate=int(round(rate)),
                resolution=resolution,
                resolution_unit=unit,
                units="" if is_raw else unit,
                is_raw=is_raw,
                adc_resolution=res or 0.0,
                adc_resolution_unit=unit,
            ))
        for lead, unique in zip(leads, unique_labels([l.label for l in leads])):
            lead.label = unique
        return leads, encodings

    @staticmethod
    def _item_statements(item) -> list[str]:
        """Statement texts of one Diagnosis-like element (or plain string)."""
        if isinstance(item, str):
            return [s.strip() for s in item.splitlines() if s.strip()]
        if not isinstance(item, dict):
            return []
        out: list[str] = []
        for tag in _STATEMENT_TAGS:
            for sub in _as_list(_child(item, tag)):
                text = _text(sub)
                if not text and isinstance(sub, dict):
                    text = _text(_child(sub, "StmtText") or _child(sub, "Text"))
                if text:
                    out.append(text)
        if not out:
            text = _text(item.get("#text"))
            if text:
                out.append(text)
        return out

    def _read_annotations(
        self, root: dict,
    ) -> tuple[Interpretation, GlobalMeasurements, dict[str, str]]:
        interp = Interpretation()
        statements: list[tuple[str, str]] = []
        seen: set[str] = set()
        for tag in ("Diagnosis", "Interpretation", "AnalysisResult", "MachineInterpretation"):
            for item in _as_list(find_tag(root, tag)):
                for text in self._item_statements(item):
                    if text not in seen:
                        seen.add(text)
                        statements.append((text, ""))
        if statements:
            interp.statements = statements
            interp.source = "machine"
            severity = _text(find_tag(root, "Severity")) or _text(find_tag(root, "DiagnosisSeverity"))
            if severity:
                interp.severity = severity.upper()

        measurements = GlobalMeasurements()
        node = self._first(root, ("Measurements", "GlobalMeasurements",
                                  "RestingECGMeasurements", "OriginalRestingECGMeasurements"))
        if node is not None:
            for attr, tags in _MEASUREMENTS:
                for tag in tags:
                    value = _number(_child(node, tag))
                    if value is not None:
                        setattr(measurements, attr, int(round(value)))
                        break
        extra: dict[str, str] = {}
        if node is not None:
            # QTc without a stated formula is not filed as Bazett or Fridericia
            qtc = _number(_child(node, "QTCorrected"))
            if qtc is not None:
                extra["qtc"] = f"{qtc:g} ms (formula not stated)"
        return interp, measurements, extra

    @staticmethod
    def _read_device(root: dict, acq) -> DeviceInfo:
        sources = [src for src in (acq, root) if isinstance(src, dict)]

        def get(*tags: str) -> str:
            for src in sources:
                for tag in tags:
                    value = _text(_child(src, tag))
                    if value:
                        return value
            return ""

        return DeviceInfo(
            manufacturer=get("Manufacturer") or "Mindray",
            model=get("DeviceModel", "DeviceName", "Device", "Model") or "BeneHeart R12",
            serial_number=get("SerialNumber"),
            software_version=get("SoftwareVersion", "FirmwareVersion"),
            institution=get("Institution", "Hospital"),
            department=get("Department"),
        )

    @staticmethod
    def _read_filters(root: dict) -> FilterSettings:
        filters = FilterSettings()
        node = find_tag(root, "FilterSettings")
        if isinstance(node, list):
            node = node[0]
        if not isinstance(node, dict):
            node = root

        def hz(*tags: str) -> float | None:
            for tag in tags:
                value = _number(find_tag(node, tag))
                if value:
                    return value
            return None

        filters.highpass = hz("HighPass", "HighPassFilter")
        filters.lowpass = hz("LowPass", "LowPassFilter")
        filters.notch = hz("NotchFilter", "Notch")
        if filters.notch:
            filters.notch_active = True
        return filters

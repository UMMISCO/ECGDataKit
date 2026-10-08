"""GE MUSE XML format parser.

Parses ``RestingECG`` exports from the GE Healthcare MUSE system (restecg.dtd).
Waveforms are Base64 little-endian signed integers per lead
(``LeadSampleSize`` bytes, normally 2), scaled by
``LeadAmplitudeUnitsPerBit`` in ``LeadAmplitudeUnits``. MUSE usually stores
8 leads (I, II, V1-V6); only the stored leads are returned.
"""

from __future__ import annotations

import base64
import binascii
import re
import warnings
import xml.etree.ElementTree as ET
import zlib
from datetime import date, datetime
from pathlib import Path

import numpy as np

from ecgdatakit.exceptions import ChecksumWarning, CorruptedFileError
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
    derive_is_raw,
)
from ecgdatakit.parsing.helpers import (
    fill_signal_summary,
    normalize_lead_label,
    unique_labels,
)
from ecgdatakit.parsing.helpers.xml import parse_xml_root
from ecgdatakit.parsing.parser import Parser

_AMPLITUDE_UNITS = {
    "MICROVOLTS": "uV", "MICROVOLT": "uV", "UV": "uV",
    "MILLIVOLTS": "mV", "MILLIVOLT": "mV", "MV": "mV",
    "VOLTS": "V", "VOLT": "V", "V": "V",
    "NANOVOLTS": "nV", "NANOVOLT": "nV", "NV": "nV",
}

_AGE_UNITS_TO_YEARS = {
    "YEARS": 1.0, "MONTHS": 1 / 12, "WEEKS": 7 / 365.25, "DAYS": 1 / 365.25,
    "HOURS": 1 / 8766,
}

_MEASUREMENT_MAP = {
    "VentricularRate": "heart_rate",
    "PRInterval": "pr_interval",
    "QRSDuration": "qrs_duration",
    "QTInterval": "qt_interval",
    "QTCorrected": None,  # formula not stated, kept in annotations only
    "QTcFrederica": "qtc_fridericia",
    "PAxis": "p_axis",
    "RAxis": "qrs_axis",
    "TAxis": "t_axis",
    "QRSCount": "qrs_count",
}



def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return " ".join("".join(el.itertext()).split())


def _get(node: ET.Element | None, tag: str) -> str:
    """Text of the direct child *tag* of *node* (case-insensitive)."""
    if node is None:
        return ""
    el = node.find(tag)
    if el is None:
        low = tag.lower()
        el = next((c for c in node if c.tag.lower() == low), None)
    return _text(el)


def _num(text: str) -> float | None:
    try:
        return float(text.replace(",", "."))
    except (ValueError, AttributeError):
        return None


def _parse_date(text: str) -> date | None:
    """Parse MUSE dates (MM-DD-YYYY by default, other common layouts)."""
    for fmt in ("%m-%d-%Y", "%m/%d/%Y", "%Y-%m-%d", "%Y%m%d", "%d-%b-%Y"):
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _parse_datetime(date_text: str, time_text: str) -> datetime | None:
    """Combine a MUSE date and time; ``None`` when the time is missing."""
    d = _parse_date(date_text) if date_text else None
    if d is None or not time_text:
        return None
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            t = datetime.strptime(time_text.strip(), fmt).time()
            return datetime.combine(d, t)
        except ValueError:
            continue
    return None


class GEMuseXMLParser(Parser):
    """Parser for GE MUSE XML ECG exports."""

    FORMAT_NAME = "GE MUSE XML"
    FORMAT_DESCRIPTION = "GE MUSE ECG management system XML export"
    FILE_EXTENSIONS = [".xml"]
    PRIORITY = 50

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        upper = header.decode("utf-8", errors="ignore").upper()
        if "<RESTINGECGDATA" in upper:  # Philips Sierra
            return False
        return re.search(r"<RESTINGECG[\s>]", upper) is not None or "<MUSEINFO" in upper

    def parse(self, file_path: Path) -> ECGRecord:
        self._warnings: list[tuple[str, type[Warning]]] = []
        raw = Path(file_path).read_bytes()
        try:
            root = parse_xml_root(raw)
        except ET.ParseError as e:
            raise CorruptedFileError(f"Malformed XML in {file_path}: {e}") from e
        if root.tag.lower() != "restingecg":
            raise CorruptedFileError(f"Missing RestingECG root element in {file_path}")

        record = ECGRecord(source_format="ge_muse_xml")
        meta = record.raw_metadata
        meta["filepath"] = str(file_path)
        demo = root.find("PatientDemographics")
        test = root.find("TestDemographics")

        record.recording = self._read_recording(test, root, meta)
        record.patient = self._read_patient(demo, record.recording.date)
        record.leads, record.median_beats = self._read_waveforms(root, record, file_path)
        if not record.leads:
            raise CorruptedFileError(f"No rhythm waveform in {file_path}")
        record.measurements = self._read_measurements(root, record)
        self._read_interpretation(root, test, record)
        self._read_extras(root, record)

        rates = {lead.sampling_rate for lead in record.leads}
        record.recording.acquisition.signal = SignalCharacteristics(
            sampling_rate=rates.pop() if len(rates) == 1 else 0,
            bits_per_sample=8 * meta.get("sample_size", 2),
            signal_signed=True,
            number_channels_valid=len(record.leads),
            number_channels_allocated=meta.get("stored_rhythm_leads", len(record.leads)),
            data_encoding=f"base64_int{8 * meta.get('sample_size', 2)}le",
            compression="none",
        )
        fill_signal_summary(record)

        record.file_format = FileFormatInfo(
            version=_get(root.find("MuseInfo"), "MuseVersion"),
            creation_date=_parse_date(_get(test, "EditDate")) if _get(test, "EditDate") else None,
        )

        for message, category in self._warnings:
            warnings.warn(message, category, stacklevel=2)
        return record

    # ------------------------------------------------------------------
    # Demographics
    # ------------------------------------------------------------------

    def _read_patient(self, demo: ET.Element | None, acquired: datetime | None) -> PatientInfo:
        info = PatientInfo()
        if demo is None:
            return info
        info.patient_id = _get(demo, "PatientID")
        info.first_name = _get(demo, "PatientFirstName")
        info.last_name = _get(demo, "PatientLastName")

        gender = _get(demo, "Gender").upper()
        if gender:
            info.sex = "M" if gender in ("M", "MALE") else "F" if gender in ("F", "FEMALE") else "U"
        dob = _get(demo, "DateofBirth")
        if dob:
            d = _parse_date(dob)
            info.birth_date = datetime.combine(d, datetime.min.time()) if d else None
        info.race = _get(demo, "Race")

        age = _num(_get(demo, "PatientAge"))
        factor = _AGE_UNITS_TO_YEARS.get(_get(demo, "AgeUnits").upper() or "YEARS")
        if age is not None and factor:
            info.age = int(age * factor)
        elif info.birth_date and acquired:
            a, b = acquired.date(), info.birth_date.date()
            info.age = a.year - b.year - ((a.month, a.day) < (b.month, b.day))

        height = _num(_get(demo, "HeightCM"))
        if height is None and _num(_get(demo, "HeightIN")) is not None:
            height = round(_num(_get(demo, "HeightIN")) * 2.54, 1)
        weight = _num(_get(demo, "WeightKG"))
        if weight is None and _num(_get(demo, "WeightLBS")) is not None:
            weight = round(_num(_get(demo, "WeightLBS")) * 0.45359237, 1)
        info.height = height or None
        info.weight = weight or None
        return info

    def _read_recording(self, test: ET.Element | None, root: ET.Element, meta: dict) -> RecordingInfo:
        info = RecordingInfo()
        if test is None:
            return info
        acq_date, acq_time = _get(test, "AcquisitionDate"), _get(test, "AcquisitionTime")
        info.date = _parse_datetime(acq_date, acq_time)
        if info.date is None and acq_date:
            meta["acquisition_date"] = f"{acq_date} {acq_time}".strip()

        info.location = _get(test, "LocationName")
        info.room = _get(test, "RoomID")
        tech = " ".join(p for p in (_get(test, "AcquisitionTechFirstName"),
                                    _get(test, "AcquisitionTechLastName")) if p)
        info.technician = tech or _get(test, "AcquisitionTechID")
        ref = " ".join(p for p in (_get(test, "ReferringMDFirstName"),
                                   _get(test, "ReferringMDLastName")) if p)
        info.referring_physician = ref or _get(test, "ReferringMDID")

        dev: DeviceInfo = info.device
        dev.model = _get(test, "AcquisitionDevice")
        dev.software_version = _get(test, "AcquisitionSoftwareVersion")
        dev.name = _get(test, "CartNumber")
        dev.serial_number = _get(root.find("PharmaData"), "PharmaCartID")
        dev.institution = _get(test, "SiteName")
        dev.acquisition_type = _get(test, "DataType")

        for tag in ("Site", "Location", "Status", "EditListStatus", "Priority",
                    "AnalysisSoftwareVersion", "TestType", "TestReason", "SecondaryID",
                    "OrderingMDID", "OrderingMDLastName", "OrderingMDFirstName",
                    "EditDate", "EditTime", "XMLSourceVersion"):
            value = _get(test, tag)
            if value:
                meta[tag] = value
        return info

    # ------------------------------------------------------------------
    # Waveforms
    # ------------------------------------------------------------------

    def _read_waveforms(
        self, root: ET.Element, record: ECGRecord, file_path: Path
    ) -> tuple[list[Lead], list[Lead]]:
        found: dict[str, list[Lead]] = {}
        meta = record.raw_metadata
        crc_results: list[bool] = []
        for wf in root.findall("Waveform"):
            kind = _get(wf, "WaveformType").lower() or "rhythm"
            if kind in found:
                self._warnings.append(
                    (f"Additional {kind} waveform ignored in {Path(file_path).name}", UserWarning)
                )
                continue
            base = _num(_get(wf, "SampleBase"))
            exponent = _num(_get(wf, "SampleExponent")) or 0
            if not base or base <= 0:
                raise CorruptedFileError(f"{kind} waveform has no valid SampleBase in {file_path}")
            rate = int(round(base * 10 ** exponent))

            leads: list[Lead] = []
            for ld in wf.findall("LeadData"):
                lead = self._read_lead(ld, rate, kind, file_path, crc_results)
                if lead is not None:
                    leads.append(lead)
            if kind == "rhythm":
                meta["stored_rhythm_leads"] = len(leads)
                self._read_filters(wf, record.recording.acquisition.filters)
                pace = wf.find("PaceSpikes")
                if pace is not None:
                    # Detected spikes are device output, not a patient attribute
                    meta["pace_spikes"] = {c.tag: _text(c) for c in pace}
            for lead, label in zip(leads, unique_labels([lead.label for lead in leads])):
                lead.label = label
            found[kind] = leads

        meta["checksum_valid"] = all(crc_results) if crc_results else None
        if getattr(self, "_sample_size", None):
            meta["sample_size"] = self._sample_size
        return found.get("rhythm", []), found.get("median", [])

    def _read_lead(
        self, ld: ET.Element, rate: int, kind: str, file_path: Path, crc_results: list[bool]
    ) -> Lead | None:
        lead_id = _get(ld, "LeadID")
        data = _get(ld, "WaveFormData")
        if not lead_id:
            raise CorruptedFileError(f"LeadData without LeadID in {kind} waveform of {file_path}")
        if not data:
            raise CorruptedFileError(f"Lead {lead_id} ({kind}) has no WaveFormData in {file_path}")
        try:
            payload = base64.b64decode("".join(data.split()), validate=True)
        except (binascii.Error, ValueError) as e:
            raise CorruptedFileError(f"Invalid Base64 in lead {lead_id} ({kind}) of {file_path}") from e

        size = int(_num(_get(ld, "LeadSampleSize")) or 2)
        if size not in (1, 2, 4) or len(payload) % size:
            raise CorruptedFileError(
                f"Lead {lead_id} ({kind}): {len(payload)} bytes is not a whole number "
                f"of {size}-byte samples in {file_path}"
            )
        expected_bytes = _num(_get(ld, "LeadByteCountTotal"))
        expected_samples = _num(_get(ld, "LeadSampleCountTotal"))
        if expected_bytes is not None and int(expected_bytes) != len(payload):
            raise CorruptedFileError(
                f"Lead {lead_id} ({kind}): LeadByteCountTotal {int(expected_bytes)} "
                f"but {len(payload)} bytes decoded in {file_path}"
            )
        samples = np.frombuffer(payload, dtype=f"<i{size}").astype(np.float64)
        self._sample_size = size
        if expected_samples is not None and int(expected_samples) != len(samples):
            raise CorruptedFileError(
                f"Lead {lead_id} ({kind}): LeadSampleCountTotal {int(expected_samples)} "
                f"but {len(samples)} samples decoded in {file_path}"
            )

        crc = _get(ld, "LeadDataCRC32")
        if crc.isdigit():
            ok = (zlib.crc32(payload) & 0xFFFFFFFF) == int(crc)
            crc_results.append(ok)
            if not ok:
                self._warnings.append(
                    (f"CRC32 mismatch in lead {lead_id} ({kind}) of {Path(file_path).name}",
                     ChecksumWarning)
                )

        scale_text = _get(ld, "LeadAmplitudeUnitsPerBit")
        unit_text = _get(ld, "LeadAmplitudeUnits")
        scale = _num(scale_text)
        unit = _AMPLITUDE_UNITS.get(unit_text.upper(), "") if unit_text else ""
        if scale is not None and not unit_text:
            unit = "uV"  # restecg.dtd default when LeadAmplitudeUnits is absent
        resolution = scale if scale is not None else 1.0
        if scale is None:
            unit = ""
        raw = derive_is_raw(resolution, 0.0, unit)

        flags = {}
        for tag in ("LeadOff", "BaselineSway", "ExcessiveACNoise", "MuscleNoise",
                    "LeadOffsetFirstSample", "FirstSampleBaseline", "LeadTimeOffset"):
            value = _get(ld, tag)
            if value and value.upper() not in ("FALSE", "0"):
                flags[tag] = value
        return Lead(
            label=normalize_lead_label(lead_id),
            samples=samples,
            sampling_rate=rate,
            resolution=resolution,
            resolution_unit=unit,
            units="" if raw else unit,
            is_raw=raw,
            adc_resolution=scale if scale is not None else 0.0,
            adc_resolution_unit=unit_text if scale is not None else "",
            annotations=flags,
        )

    @staticmethod
    def _read_filters(wf: ET.Element, fs: FilterSettings) -> None:
        hp = _num(_get(wf, "HighPassFilter"))
        lp = _num(_get(wf, "LowPassFilter"))
        if hp:
            fs.highpass = hp / 100.0  # stored in 0.01 Hz
        if lp:
            fs.lowpass = lp
        ac_values = [_text(el) for el in wf.findall("ACFilter") if _text(el)]
        for value in ac_values:
            freq = _num(value)
            if freq:
                fs.notch = freq
                fs.notch_active = True
                break
            if value.upper() in ("NONE", "OFF"):
                fs.notch_active = False

    # ------------------------------------------------------------------
    # Measurements and interpretation
    # ------------------------------------------------------------------

    def _read_measurements(self, root: ET.Element, record: ECGRecord) -> GlobalMeasurements:
        gm = GlobalMeasurements()
        current = root.find("RestingECGMeasurements")
        original = root.find("OriginalRestingECGMeasurements")
        primary = current if current is not None else original
        if primary is not None:
            for child in primary:
                value = _text(child)
                if not value:
                    continue
                field = _MEASUREMENT_MAP.get(child.tag)
                number = _num(value)
                if field and number is not None:
                    setattr(gm, field, int(round(number)))
                elif child.tag not in ("ECGSampleBase", "ECGSampleExponent"):
                    record.annotations[child.tag] = value
        if current is not None and original is not None:
            orig = {c.tag: _text(c) for c in original if _text(c)}
            cur = {c.tag: _text(c) for c in current if _text(c)}
            if orig != cur:
                record.raw_metadata["original_measurements"] = orig

        rr = _num(_get(root.find("QRSTimesTypes"), "GlobalRR"))
        if rr is None:
            rr = _num(_get(root.find("PharmaData"), "PharmaRRinterval"))
        if rr:
            gm.rr_interval = int(round(rr))
        return gm

    def _read_interpretation(self, root: ET.Element, test: ET.Element | None, record: ECGRecord) -> None:
        current = self._statements(root.find("Diagnosis"))
        original = self._statements(root.find("OriginalDiagnosis"))
        status = _get(test, "Status").upper()
        overreader = " ".join(p for p in (_get(test, "OverreaderFirstName"),
                                          _get(test, "OverreaderLastName")) if p)
        overreader = overreader or _get(test, "OverreaderID")
        expert = status == "CONFIRMED" or bool(overreader)

        interp = Interpretation()
        interp.statements = [(s, "") for s in current]
        if expert and current:
            interp.source = "confirmed" if status == "CONFIRMED" else "overread"
            interp.interpreter = overreader
            interp.interpretation_date = _parse_datetime(_get(test, "EditDate"), _get(test, "EditTime"))
            machine = original or []
            if machine:
                record.annotations["machine_interpretation"] = "\n".join(machine)
        elif current:
            interp.source = "machine"
            if original and original != current:
                record.annotations["original_diagnosis"] = "\n".join(original)
        elif original:
            interp.statements = [(s, "") for s in original]
            interp.source = "machine"
        record.interpretation = interp

    @staticmethod
    def _statements(diag: ET.Element | None) -> list[str]:
        """Diagnosis lines; fragments are joined until a statement flagged ENDSLINE."""
        if diag is None:
            return []
        items = []
        for stmt in diag.findall("DiagnosisStatement"):
            flags = {_text(f).upper() for f in stmt.findall("StmtFlag")}
            text = _get(stmt, "StmtText")
            items.append((text, "ENDSLINE" in flags))
        if not any(end for _, end in items):
            return [t for t, _ in items if t]
        lines, current = [], []
        for text, end in items:
            if text:
                current.append(text)
            if end and current:
                lines.append(" ".join(current))
                current = []
        if current:
            lines.append(" ".join(current))
        return lines

    @staticmethod
    def _read_extras(root: ET.Element, record: ECGRecord) -> None:
        meta = record.raw_metadata
        order = root.find("Order")
        if order is not None:
            values = {c.tag: _text(c) for c in order if _text(c)}
            if values:
                meta["order"] = values
            reason = values.get("ReasonForTest") if values else None
            if reason:
                record.annotations["reason_for_test"] = reason
        questions = {}
        for q in root.findall("ExtraQuestions/ExtraQuestion"):
            if _get(q, "Answer"):
                questions[_get(q, "Question")] = _get(q, "Answer")
        if questions:
            meta["extra_questions"] = questions
        qrs = root.find("QRSTimesTypes")
        if qrs is not None:
            beats = [
                {"number": _get(b, "Number"), "type": _get(b, "Type"), "time": _get(b, "Time")}
                for b in qrs.findall("QRS")
            ]
            if beats:
                meta["qrs_times"] = beats
        for tag in ("MeasurementMatrix", "PharmaData", "IntervalMeasurements",
                    "AmplitudeMeasurements"):
            el = root.find(tag)
            if el is None:
                continue
            if len(el):
                meta[tag] = {c.tag: _text(c) for c in el if not len(c)} or _text(el)
            elif _text(el):
                meta[tag] = _text(el)

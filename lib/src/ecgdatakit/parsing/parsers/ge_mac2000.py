"""GE MAC 2000 XML format parser.

Parses ECG exports from GE MAC 2000 devices.
Distinct from GE MUSE XML format.

Note: This parser is marked as beta due to limited public documentation.
Falls back to common GE XML patterns.
"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xmltodict

from ecgdatakit.exceptions import CorruptedFileError
from ecgdatakit.parsing.helpers.xml import find_tag, read_path
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
)
from ecgdatakit.parsing.parser import Parser


def _decode_lead_data(data_str: str) -> np.ndarray:
    """Decode lead data — try Base64, then comma-separated integers."""
    data_str = data_str.strip()
    if not data_str:
        return np.array([], dtype=np.float64)

    try:
        raw = base64.b64decode(data_str)
        if len(raw) >= 2:
            signal = np.frombuffer(raw, dtype="<i2")
            return signal.astype(np.float64)
    except Exception:
        pass

    try:
        values = [int(v.strip()) for v in data_str.split(",") if v.strip()]
        return np.array(values, dtype=np.float64)
    except ValueError:
        pass

    try:
        values = [int(v.strip()) for v in data_str.split() if v.strip()]
        return np.array(values, dtype=np.float64)
    except ValueError:
        pass

    return np.array([], dtype=np.float64)


class GEMAC2000Parser(Parser):
    """Parser for GE MAC 2000 XML ECG files (beta)."""

    FORMAT_NAME = "GE MAC 2000"
    FORMAT_DESCRIPTION = "GE MAC 2000 XML ECG format (beta)"
    FILE_EXTENSIONS = [".xml"]

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        try:
            text = header.decode("utf-8", errors="ignore")
            upper = text.upper()
            if "<MAC2000" in upper or "<MAC_2000" in upper or "<GE_MAC" in upper:
                return True
            if "<MACEXPORT" in upper and "MAC" in upper:
                return True
            return False
        except Exception:
            return False

    def parse(self, file_path: Path) -> ECGRecord:
        with open(file_path, "rb") as f:
            doc = xmltodict.parse(f.read())

        root = None
        for key in ("MAC2000", "MAC_2000", "GE_MAC", "MacExport"):
            if key in doc:
                root = doc[key]
                break
        if root is None:
            for key in doc:
                k = key.lower()
                if k in ("mac2000", "mac_2000", "ge_mac", "macexport"):
                    root = doc[key]
                    break

        if root is None:
            raise CorruptedFileError(f"No recognized root element in {file_path}")

        record = ECGRecord(source_format="ge_mac2000")

        record.patient = self._read_patient(root)
        record.recording = self._read_recording(root)
        record.recording.device = self._read_device(root)
        record.recording.acquisition.filters = self._read_filters(root)
        record.leads = self._read_leads(root)
        if record.leads and record.recording.acquisition.signal.sample_rate == 0:
            record.recording.acquisition.signal.sample_rate = record.leads[0].sample_rate
        if record.leads and record.recording.acquisition.signal.sample_rate > 0:
            duration_s = len(record.leads[0].samples) / record.recording.acquisition.signal.sample_rate
            record.recording.duration = timedelta(seconds=duration_s)

        record.annotations = self._read_annotations(root)
        record.measurements = self._read_measurements(record.annotations)
        record.interpretation = self._read_interpretation(root)

        record.recording.acquisition.signal = SignalCharacteristics(
            sample_rate=record.recording.acquisition.signal.sample_rate,
            signal_signed=True,
            number_channels_valid=len(record.leads),
            data_encoding="base64_int16le",
            compression="none",
        )

        record.raw_metadata["filepath"] = str(file_path)

        return record

    def _read_patient(self, root: dict) -> PatientInfo:
        info = PatientInfo()

        demo = (
            find_tag(root, "PatientDemographics") or
            find_tag(root, "PatientInfo") or
            find_tag(root, "Patient")
        )
        if demo is None:
            return info

        if isinstance(demo, list):
            demo = demo[0]

        info.patient_id = self._get_text(demo, "PatientID") or self._get_text(demo, "ID")
        info.first_name = self._get_text(demo, "PatientFirstName") or self._get_text(demo, "FirstName")
        info.last_name = self._get_text(demo, "PatientLastName") or self._get_text(demo, "LastName")

        sex = self._get_text(demo, "Gender") or self._get_text(demo, "Sex")
        if sex:
            s = sex.upper()
            info.sex = "M" if s in ("M", "MALE") else "F" if s in ("F", "FEMALE") else "U"

        dob = self._get_text(demo, "DateofBirth") or self._get_text(demo, "DateOfBirth") or self._get_text(demo, "DOB")
        if dob:
            for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%Y%m%d", "%d-%b-%Y"):
                try:
                    info.birth_date = datetime.strptime(dob.strip(), fmt)
                    break
                except ValueError:
                    continue

        age_str = self._get_text(demo, "PatientAge") or self._get_text(demo, "Age")
        if age_str and age_str.isdigit():
            info.age = int(age_str)

        height_str = self._get_text(demo, "Height") or self._get_text(demo, "PatientHeight")
        if height_str:
            try:
                info.height = float(height_str)
            except (ValueError, TypeError):
                pass

        weight_str = self._get_text(demo, "Weight") or self._get_text(demo, "PatientWeight")
        if weight_str:
            try:
                info.weight = float(weight_str)
            except (ValueError, TypeError):
                pass

        race_str = self._get_text(demo, "Race") or self._get_text(demo, "Ethnicity")
        if race_str:
            info.race = race_str

        return info

    def _read_recording(self, root: dict) -> RecordingInfo:
        info = RecordingInfo()

        test = (
            find_tag(root, "TestDemographics") or
            find_tag(root, "AcquisitionInfo") or
            find_tag(root, "RecordingInfo")
        )
        if test is not None:
            if isinstance(test, list):
                test = test[0]

            date_str = self._get_text(test, "AcquisitionDate") or self._get_text(test, "Date")
            time_str = self._get_text(test, "AcquisitionTime") or self._get_text(test, "Time")
            if date_str:
                for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%Y%m%d"):
                    try:
                        dt = datetime.strptime(date_str.strip(), fmt)
                        if time_str:
                            try:
                                t = datetime.strptime(time_str.strip(), "%H:%M:%S")
                                dt = dt.replace(hour=t.hour, minute=t.minute, second=t.second)
                            except ValueError:
                                pass
                        info.date = dt
                        break
                    except ValueError:
                        continue

            sr_str = self._get_text(test, "SampleRate") or self._get_text(test, "SampleBase")
            if sr_str:
                try:
                    info.acquisition.signal.sample_rate = int(float(sr_str))
                except ValueError:
                    pass

            info.location = self._get_text(test, "Location")
            info.room = self._get_text(test, "Room")

            tech = (
                self._get_text(test, "TechnicianID") or
                self._get_text(test, "OperatorID") or
                self._get_text(test, "Technician")
            )
            if tech:
                info.technician = tech

            ref_phys = (
                self._get_text(test, "ReferringPhysician") or
                self._get_text(test, "OrderingPhysician") or
                self._get_text(test, "AttendingPhysician")
            )
            if ref_phys:
                info.referring_physician = ref_phys

        return info

    def _read_leads(self, root: dict) -> list[Lead]:
        leads: list[Lead] = []

        sample_rate = 500
        sr_node = find_tag(root, "SampleRate") or find_tag(root, "SampleBase")
        if sr_node is not None:
            try:
                sample_rate = int(str(sr_node))
            except ValueError:
                pass

        waveform = find_tag(root, "Waveform") or find_tag(root, "WaveformData") or find_tag(root, "Leads")
        if waveform is not None:
            if isinstance(waveform, list):
                waveform = waveform[0]

            lead_nodes = (
                find_tag(waveform, "LeadData") or
                find_tag(waveform, "Lead") or
                find_tag(waveform, "Channel")
            )

            # Try waveform-level amplitude scale (GE convention)
            wf_scale = 1.0
            wf_amp = (
                find_tag(waveform, "LeadAmplitudeUnitsPerBit") or
                find_tag(waveform, "AcquiredAmplitudeResolution") or
                find_tag(waveform, "AmplitudeResolution")
            )
            if wf_amp is not None:
                try:
                    wf_scale = float(str(wf_amp))
                except (ValueError, TypeError):
                    pass

            if lead_nodes is not None:
                if isinstance(lead_nodes, dict):
                    lead_nodes = [lead_nodes]
                if isinstance(lead_nodes, list):
                    for node in lead_nodes:
                        if isinstance(node, dict):
                            label = (
                                self._get_text(node, "LeadID") or
                                node.get("@Name") or node.get("@name") or
                                node.get("@NAME") or
                                self._get_text(node, "Name") or "?"
                            )
                            data_str = (
                                self._get_text(node, "WaveFormData") or
                                self._get_text(node, "Data") or
                                node.get("@Data") or node.get("@DATA") or
                                node.get("#text") or ""
                            )
                            # Per-lead scale overrides waveform-level scale
                            scale = wf_scale
                            amp_str = self._get_text(node, "LeadAmplitudeUnitsPerBit")
                            if amp_str:
                                try:
                                    scale = float(amp_str)
                                except (ValueError, TypeError):
                                    pass
                            samples = _decode_lead_data(data_str)
                            if len(samples) > 0:
                                leads.append(Lead(
                                    label=label,
                                    samples=samples,
                                    sample_rate=sample_rate,
                                    resolution=scale,
                                    units="uV" if scale != 1.0 else "",
                                    is_raw=scale != 1.0,
                                ))

        return leads

    def _read_annotations(self, root: dict) -> dict[str, str]:
        annotations: dict[str, str] = {}

        measurements = (
            find_tag(root, "Measurements") or
            find_tag(root, "RestingECGMeasurements")
        )
        if measurements is not None:
            if isinstance(measurements, list):
                measurements = measurements[0]
            for key in ("VentricularRate", "PRInterval", "QRSDuration",
                        "QTInterval", "QTCorrected", "PAxis", "RAxis", "TAxis",
                        "RRInterval", "QRSCount", "NumQRS"):
                val = self._get_text(measurements, key)
                if val:
                    annotations[key.lower()] = val

        diag = find_tag(root, "Diagnosis") or find_tag(root, "Interpretation")
        if diag is not None:
            if isinstance(diag, dict):
                text = self._get_text(diag, "Statement") or self._get_text(diag, "Text")
                if text:
                    annotations["diagnosis"] = text
            elif isinstance(diag, str):
                annotations["diagnosis"] = diag

        return annotations

    def _read_device(self, root: dict) -> DeviceInfo:
        model = (
            self._get_text(root, "DeviceModel")
            or self._get_text(root, "AcquisitionDevice")
            or "MAC 2000"
        )
        software_version = (
            self._get_text(root, "SoftwareVersion")
            or self._get_text(root, "AcquisitionSoftwareVersion")
        )
        serial_number = (
            self._get_text(root, "SerialNumber")
            or self._get_text(root, "DeviceSerialNumber")
        )
        return DeviceInfo(
            manufacturer="GE",
            model=model,
            software_version=software_version,
            serial_number=serial_number,
        )

    def _read_filters(self, root: dict) -> FilterSettings:
        filters = FilterSettings()

        fs = find_tag(root, "FilterSettings")
        if fs is not None:
            if isinstance(fs, list):
                fs = fs[0]
            if isinstance(fs, dict):
                hp = self._get_text(fs, "HighPassFilter")
                lp = self._get_text(fs, "LowPassFilter")
                nf = self._get_text(fs, "NotchFilter")
                if hp:
                    try:
                        filters.highpass = float(hp)
                    except (ValueError, TypeError):
                        pass
                if lp:
                    try:
                        filters.lowpass = float(lp)
                    except (ValueError, TypeError):
                        pass
                if nf:
                    try:
                        filters.notch = float(nf)
                        filters.notch_active = True
                    except (ValueError, TypeError):
                        pass
                return filters

        hp = self._get_text(root, "HighPassFilter")
        lp = self._get_text(root, "LowPassFilter")
        nf = self._get_text(root, "NotchFilter")
        if hp:
            try:
                filters.highpass = float(hp)
            except (ValueError, TypeError):
                pass
        if lp:
            try:
                filters.lowpass = float(lp)
            except (ValueError, TypeError):
                pass
        if nf:
            try:
                filters.notch = float(nf)
                filters.notch_active = True
            except (ValueError, TypeError):
                pass

        return filters

    def _read_measurements(self, annotations: dict[str, str]) -> GlobalMeasurements:
        m = GlobalMeasurements()

        def _int(key: str) -> int | None:
            val = annotations.get(key, "")
            if not val:
                return None
            try:
                return int(val)
            except (ValueError, TypeError):
                return None

        m.heart_rate = _int("ventricularrate")
        m.pr_interval = _int("printerval")
        m.qrs_duration = _int("qrsduration")
        m.qt_interval = _int("qtinterval")
        m.qtc_bazett = _int("qtcorrected")
        m.p_axis = _int("paxis")
        m.qrs_axis = _int("raxis")
        m.t_axis = _int("taxis")
        m.rr_interval = _int("rrinterval")
        m.qrs_count = _int("qrscount") or _int("numqrs")

        return m

    def _read_interpretation(self, root: dict) -> Interpretation:
        interp = Interpretation()

        overread = find_tag(root, "OverreadDiagnosis")
        if overread is not None:
            statements = self._extract_statements(overread)
            if statements:
                interp.statements = statements
                interp.source = "overread"
                return interp

        original = find_tag(root, "OriginalDiagnosis")
        if original is not None:
            statements = self._extract_statements(original)
            if statements:
                interp.statements = statements
                interp.source = "machine"
                return interp

        diag = find_tag(root, "Diagnosis") or find_tag(root, "Interpretation")
        if diag is not None:
            statements = self._extract_statements(diag)
            if statements:
                interp.statements = statements
                interp.source = "machine"

        return interp

    def _extract_statements(self, node: dict | list | str | None) -> list[tuple[str, str]]:
        """Extract diagnosis statement tuples from a node."""
        if node is None:
            return []
        if isinstance(node, str):
            return [(s.strip(), "") for s in node.split("\n") if s.strip()]
        if isinstance(node, list):
            node = node[0]
        if isinstance(node, dict):
            stmts_node = find_tag(node, "Statement") or find_tag(node, "DiagnosisStatement")
            if stmts_node is not None:
                return self._flatten_statements(stmts_node)
            text = self._get_text(node, "Text") or self._get_text(node, "StmtText")
            if text:
                return [(s.strip(), "") for s in text.split("\n") if s.strip()]
        return []

    @staticmethod
    def _flatten_statements(node: dict | list | str | None) -> list[tuple[str, str]]:
        """Flatten Statement nodes into a list of (text, "") tuples."""
        if node is None:
            return []
        if isinstance(node, str):
            return [(s.strip(), "") for s in node.split("\n") if s.strip()]
        if isinstance(node, dict):
            text = node.get("StmtText") or node.get("Text") or node.get("#text") or ""
            if isinstance(text, dict):
                text = text.get("#text", "")
            text = str(text).strip()
            return [(text, "")] if text else []
        if isinstance(node, list):
            out: list[tuple[str, str]] = []
            for item in node:
                if isinstance(item, str):
                    s = item.strip()
                    if s:
                        out.append((s, ""))
                elif isinstance(item, dict):
                    text = item.get("StmtText") or item.get("Text") or item.get("#text") or ""
                    if isinstance(text, dict):
                        text = text.get("#text", "")
                    s = str(text).strip()
                    if s:
                        out.append((s, ""))
            return out
        return []

    @staticmethod
    def _get_text(node: dict | None, tag: str) -> str:
        if node is None:
            return ""
        val = find_tag(node, tag)
        if val is None:
            return ""
        if isinstance(val, list):
            val = val[0]
        if isinstance(val, dict):
            return val.get("#text", "")
        return str(val).strip()

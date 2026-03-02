"""Mindray BeneHeart R12 XML format parser.

Parses ECG exports from Mindray BeneHeart R12 devices.

Note: This parser is marked as beta due to limited public documentation.
Structure is inferred from device export samples and Mindray SDK docs.
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


class BeneHeartR12Parser(Parser):
    """Parser for Mindray BeneHeart R12 XML ECG files (beta)."""

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        try:
            text = header.decode("utf-8", errors="ignore")
            upper = text.upper()
            if "<BENEHEARTR12" in upper or "<MINDRAYECG" in upper:
                return True
            if "<ECGDATA" in upper and "MINDRAY" in upper:
                return True
            return False
        except Exception:
            return False

    def parse(self, file_path: Path) -> ECGRecord:
        with open(file_path, "rb") as f:
            doc = xmltodict.parse(f.read())

        root = None
        for key in ("BeneHeartR12", "MindrayECG", "ECGData"):
            if key in doc:
                root = doc[key]
                break
        if root is None:
            for key in doc:
                if key.lower() in ("beneheartr12", "mindrayecg", "ecgdata"):
                    root = doc[key]
                    break

        if root is None:
            raise CorruptedFileError(f"No recognized root element in {file_path}")

        record = ECGRecord(source_format="beneheart_r12")

        record.patient = self._read_patient(root)
        record.recording = self._read_recording(root)
        record.leads = self._read_leads(root)
        record.device = self._read_device(root)
        record.filters = self._read_filters(root)
        record.interpretation, record.measurements = self._read_annotations(root)
        clinical_info: dict[str, str] = {}
        for tag in ("ClinicalInfo", "OrderInfo", "Indication"):
            node = find_tag(root, tag)
            if node is not None:
                if isinstance(node, list):
                    node = node[0]
                if isinstance(node, dict):
                    for k, v in node.items():
                        if isinstance(v, str):
                            clinical_info[k] = v
                        elif isinstance(v, dict) and "#text" in v:
                            clinical_info[k] = v["#text"]
                elif isinstance(node, str) and node.strip():
                    clinical_info[tag] = node.strip()
        if clinical_info:
            record.raw_metadata["clinical_info"] = clinical_info

        if record.leads and record.recording.sample_rate == 0:
            record.recording.sample_rate = record.leads[0].sample_rate
        if record.leads and record.recording.sample_rate > 0:
            duration_s = len(record.leads[0].samples) / record.recording.sample_rate
            record.recording.duration = timedelta(seconds=duration_s)

        record.raw_metadata["filepath"] = str(file_path)

        return record

    def _read_patient(self, root: dict) -> PatientInfo:
        info = PatientInfo()

        demo = find_tag(root, "PatientInfo") or find_tag(root, "Patient") or find_tag(root, "Demographics")
        if demo is None:
            return info

        if isinstance(demo, list):
            demo = demo[0]

        info.patient_id = self._get_text(demo, "PatientID") or self._get_text(demo, "ID")
        info.first_name = self._get_text(demo, "FirstName") or self._get_text(demo, "GivenName")
        info.last_name = self._get_text(demo, "LastName") or self._get_text(demo, "FamilyName")

        sex = self._get_text(demo, "Sex") or self._get_text(demo, "Gender")
        if sex:
            s = sex.upper()
            info.sex = "M" if s in ("M", "MALE", "1") else "F" if s in ("F", "FEMALE", "2") else "U"

        dob = self._get_text(demo, "DateOfBirth") or self._get_text(demo, "BirthDate") or self._get_text(demo, "DOB")
        if dob:
            for fmt in ("%Y%m%d", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
                try:
                    info.birth_date = datetime.strptime(dob.strip(), fmt)
                    break
                except ValueError:
                    continue

        age_str = self._get_text(demo, "Age")
        if age_str and age_str.isdigit():
            info.age = int(age_str)

        return info

    def _read_recording(self, root: dict) -> RecordingInfo:
        info = RecordingInfo()

        acq = find_tag(root, "AcquisitionInfo") or find_tag(root, "RecordingInfo") or find_tag(root, "TestInfo")
        if acq is not None:
            if isinstance(acq, list):
                acq = acq[0]

            date_str = self._get_text(acq, "AcquisitionDate") or self._get_text(acq, "Date")
            time_str = self._get_text(acq, "AcquisitionTime") or self._get_text(acq, "Time")
            if date_str:
                for fmt in ("%Y%m%d", "%Y-%m-%d", "%m/%d/%Y"):
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

            sr_str = self._get_text(acq, "SampleRate") or self._get_text(acq, "SamplingRate")
            if sr_str:
                try:
                    info.sample_rate = int(float(sr_str))
                except ValueError:
                    pass

            info.device = self._get_text(acq, "Device") or self._get_text(acq, "Model") or "BeneHeart R12"

        if not info.device:
            info.device = "BeneHeart R12"

        return info

    def _read_leads(self, root: dict) -> list[Lead]:
        leads: list[Lead] = []

        lead_container = (
            find_tag(root, "Leads") or
            find_tag(root, "WaveformData") or
            find_tag(root, "Channels")
        )

        sample_rate = 500
        sr_node = find_tag(root, "SampleRate") or find_tag(root, "SamplingRate")
        if sr_node is not None:
            try:
                sample_rate = int(str(sr_node))
            except ValueError:
                pass

        if lead_container is not None:
            if isinstance(lead_container, list):
                lead_container = lead_container[0]

            lead_nodes = find_tag(lead_container, "Lead") or find_tag(lead_container, "Channel")
            if lead_nodes is None:
                for label in ("I", "II", "III", "aVR", "aVL", "aVF",
                              "V1", "V2", "V3", "V4", "V5", "V6"):
                    data = find_tag(lead_container, label)
                    if data is not None:
                        samples = _decode_lead_data(str(data))
                        if len(samples) > 0:
                            leads.append(Lead(
                                label=label,
                                samples=samples,
                                sample_rate=sample_rate,
                            ))
            else:
                if isinstance(lead_nodes, dict):
                    lead_nodes = [lead_nodes]
                if isinstance(lead_nodes, list):
                    for node in lead_nodes:
                        if isinstance(node, dict):
                            label = (
                                node.get("@Name") or node.get("@name") or
                                node.get("@NAME") or node.get("@Label") or
                                node.get("@label") or
                                self._get_text(node, "Name") or
                                self._get_text(node, "Label") or "?"
                            )
                            data_str = (
                                node.get("@Data") or node.get("@data") or
                                node.get("@DATA") or
                                node.get("#text") or
                                self._get_text(node, "Data") or
                                self._get_text(node, "Samples") or ""
                            )
                            samples = _decode_lead_data(data_str)
                            if len(samples) > 0:
                                leads.append(Lead(
                                    label=label,
                                    samples=samples,
                                    sample_rate=sample_rate,
                                ))

        return leads

    def _read_annotations(self, root: dict) -> tuple[Interpretation, GlobalMeasurements]:
        interp = Interpretation()
        interp.source = "machine"
        measurements = GlobalMeasurements()

        statements: list[str] = []
        for tag in ("Diagnosis", "Interpretation", "AnalysisResult",
                     "MachineInterpretation"):
            node = find_tag(root, tag)
            if node is None:
                continue
            if isinstance(node, dict):
                node = [node]
            if isinstance(node, list):
                for item in node:
                    if isinstance(item, dict):
                        for stmt_tag in ("DiagnosisStatement", "Statement",
                                         "StmtText", "Text", "Description"):
                            stmt_node = find_tag(item, stmt_tag)
                            if stmt_node is not None:
                                if isinstance(stmt_node, list):
                                    for s in stmt_node:
                                        text = s.get("#text", "") if isinstance(s, dict) else str(s).strip()
                                        if text:
                                            statements.append(text)
                                elif isinstance(stmt_node, dict):
                                    text = stmt_node.get("#text", "")
                                    if text:
                                        statements.append(text)
                                elif isinstance(stmt_node, str) and stmt_node.strip():
                                    statements.append(stmt_node.strip())
                        if not statements:
                            text = (
                                item.get("@TEXT") or item.get("@Text") or
                                item.get("@DESCRIPTION") or item.get("#text", "")
                            )
                            if text and isinstance(text, str) and text.strip():
                                statements.append(text.strip())
                    elif isinstance(item, str) and item.strip():
                        statements.append(item.strip())

        for sev_tag in ("Severity", "DiagnosisSeverity"):
            sev = self._get_text(root, sev_tag)
            if sev:
                interp.severity = sev.upper()
                break

        interp.statements = statements

        meas_node = None
        for tag in ("Measurements", "GlobalMeasurements",
                     "RestingECGMeasurements", "OriginalRestingECGMeasurements"):
            meas_node = find_tag(root, tag)
            if meas_node is not None:
                break

        if meas_node is not None:
            if isinstance(meas_node, list):
                meas_node = meas_node[0]
            if isinstance(meas_node, dict):
                hr = self._get_text(meas_node, "VentricularRate") or self._get_text(meas_node, "HeartRate")
                if hr:
                    try:
                        measurements.heart_rate = int(float(hr))
                    except (ValueError, TypeError):
                        pass

                pr = self._get_text(meas_node, "PRInterval")
                if pr:
                    try:
                        measurements.pr_interval = int(float(pr))
                    except (ValueError, TypeError):
                        pass

                qrs = self._get_text(meas_node, "QRSDuration")
                if qrs:
                    try:
                        measurements.qrs_duration = int(float(qrs))
                    except (ValueError, TypeError):
                        pass

                qt = self._get_text(meas_node, "QTInterval")
                if qt:
                    try:
                        measurements.qt_interval = int(float(qt))
                    except (ValueError, TypeError):
                        pass

                qtc = self._get_text(meas_node, "QTCorrected") or self._get_text(meas_node, "QTcBazett")
                if qtc:
                    try:
                        measurements.qtc_bazett = int(float(qtc))
                    except (ValueError, TypeError):
                        pass

                p_axis = self._get_text(meas_node, "PAxis")
                if p_axis:
                    try:
                        measurements.p_axis = int(float(p_axis))
                    except (ValueError, TypeError):
                        pass

                qrs_axis = self._get_text(meas_node, "QRSAxis") or self._get_text(meas_node, "RAxis")
                if qrs_axis:
                    try:
                        measurements.qrs_axis = int(float(qrs_axis))
                    except (ValueError, TypeError):
                        pass

                t_axis = self._get_text(meas_node, "TAxis")
                if t_axis:
                    try:
                        measurements.t_axis = int(float(t_axis))
                    except (ValueError, TypeError):
                        pass

        return interp, measurements

    def _read_device(self, root: dict) -> DeviceInfo:
        dev = DeviceInfo()
        dev.model = "BeneHeart R12"

        acq = find_tag(root, "AcquisitionInfo") or find_tag(root, "RecordingInfo")
        sources = [acq, root] if acq is not None else [root]

        for src in sources:
            if isinstance(src, list):
                src = src[0]
            if not isinstance(src, dict):
                continue

            if not dev.model or dev.model == "BeneHeart R12":
                name = self._get_text(src, "DeviceName") or self._get_text(src, "DeviceModel")
                if name:
                    dev.model = name

            if not dev.manufacturer:
                dev.manufacturer = self._get_text(src, "Manufacturer")

            if not dev.serial_number:
                dev.serial_number = self._get_text(src, "SerialNumber")

            if not dev.software_version:
                dev.software_version = (
                    self._get_text(src, "SoftwareVersion")
                    or self._get_text(src, "FirmwareVersion")
                )

        if not dev.manufacturer:
            dev.manufacturer = "Mindray"

        return dev

    def _read_filters(self, root: dict) -> FilterSettings:
        filters = FilterSettings()

        filt_node = find_tag(root, "FilterSettings")
        if filt_node is None:
            filt_node = root

        if isinstance(filt_node, list):
            filt_node = filt_node[0]

        if isinstance(filt_node, dict):
            hp = self._get_text(filt_node, "HighPass") or self._get_text(filt_node, "HighPassFilter")
            if hp:
                try:
                    filters.highpass = float(hp)
                except (ValueError, TypeError):
                    pass

            lp = self._get_text(filt_node, "LowPass") or self._get_text(filt_node, "LowPassFilter")
            if lp:
                try:
                    filters.lowpass = float(lp)
                except (ValueError, TypeError):
                    pass

            nf = self._get_text(filt_node, "NotchFilter") or self._get_text(filt_node, "Notch")
            if nf:
                try:
                    filters.notch = float(nf)
                except (ValueError, TypeError):
                    pass

        return filters

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

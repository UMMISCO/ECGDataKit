"""GE MUSE XML format parser.

Parses ECG exports from GE Healthcare MUSE system.
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


def _decode_waveform_data(base64_str: str, dtype: str = "<i2") -> np.ndarray:
    """Decode Base64-encoded waveform data to float64 array."""
    raw = base64.b64decode(base64_str)
    signal = np.frombuffer(raw, dtype=dtype)
    return signal.astype(np.float64)


class GEMuseXMLParser(Parser):
    """Parser for GE MUSE XML ECG exports."""

    FORMAT_NAME = "GE MUSE XML"
    FORMAT_DESCRIPTION = "GE MUSE ECG management system XML export"
    FILE_EXTENSIONS = [".xml"]

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        try:
            text = header.decode("utf-8", errors="ignore")
            upper = text.upper()
            if "<RESTINGECG>" in upper or "<RESTINGECG " in upper:
                if "<RESTINGECGDATA" in upper:
                    return False
                return True
            if "<MUSEINFO" in upper:
                return True
            return False
        except Exception:
            return False

    def parse(self, file_path: Path) -> ECGRecord:
        with open(file_path, "rb") as f:
            doc = xmltodict.parse(f.read())

        root = doc.get("RestingECG") or doc.get("restingecg") or doc.get("Restingecg")
        if root is None:
            for key in doc:
                if key.lower() == "restingecg":
                    root = doc[key]
                    break
        if root is None:
            raise CorruptedFileError(f"Missing RestingECG root element in {file_path}")

        record = ECGRecord(source_format="ge_muse_xml")

        record.patient = self._read_patient(root)
        record.recording = self._read_recording(root)
        record.recording.device = self._read_device(root)
        record.recording.acquisition.filters = self._read_filters(root)
        rhythm_leads, median_leads = self._read_leads(root)
        record.leads = rhythm_leads
        record.median_beats = median_leads

        if record.leads and record.recording.acquisition.signal.sampling_rate == 0:
            record.recording.acquisition.signal.sampling_rate = record.leads[0].sampling_rate

        annotations, measurements, interpretation = self._read_annotations(root)
        record.annotations = annotations
        record.measurements = measurements
        record.interpretation = interpretation

        order_info = self._read_order_info(root)
        if order_info:
            record.raw_metadata["order_info"] = order_info
            # Promote referring physician to structured field
            if not record.recording.referring_physician:
                ref = order_info.get("referring_physician", "")
                if ref:
                    record.recording.referring_physician = ref

        # Extract technician from TestDemographics
        test_demo = find_tag(root, "TestDemographics")
        if test_demo is not None:
            if isinstance(test_demo, list):
                test_demo = test_demo[0]
            tech = self._get_text(test_demo, "TechnicianID") or self._get_text(test_demo, "OperatorID")
            if tech:
                record.recording.technician = tech

        # Count all lead data elements for channels_allocated
        all_lead_count = 0
        waveform_nodes = find_tag(root, "Waveform")
        if waveform_nodes is not None:
            if isinstance(waveform_nodes, dict):
                waveform_nodes = [waveform_nodes]
            for wf_node in waveform_nodes:
                ld_nodes = find_tag(wf_node, "LeadData")
                if ld_nodes is not None:
                    if isinstance(ld_nodes, dict):
                        all_lead_count += 1
                    elif isinstance(ld_nodes, list):
                        all_lead_count += len(ld_nodes)

        record.recording.acquisition.signal = SignalCharacteristics(
            sampling_rate=record.recording.acquisition.signal.sampling_rate,
            bits_per_sample=16,
            signal_signed=True,
            number_channels_valid=len(record.leads),
            number_channels_allocated=all_lead_count or len(record.leads),
            data_encoding="base64_int16le",
            compression="none",
        )

        record.raw_metadata["filepath"] = str(file_path)

        return record

    def _read_patient(self, root: dict) -> PatientInfo:
        info = PatientInfo()

        demo = find_tag(root, "PatientDemographics")
        if demo is None:
            return info

        if isinstance(demo, list):
            demo = demo[0]

        info.patient_id = self._get_text(demo, "PatientID")
        info.first_name = self._get_text(demo, "PatientFirstName")
        info.last_name = self._get_text(demo, "PatientLastName")

        gender = self._get_text(demo, "Gender")
        if gender:
            g = gender.upper()
            info.sex = "M" if g in ("M", "MALE") else "F" if g in ("F", "FEMALE") else "U"

        dob_str = self._get_text(demo, "DateofBirth")
        if dob_str:
            info.birth_date = self._parse_date(dob_str)

        age_str = self._get_text(demo, "PatientAge")
        if age_str and age_str.isdigit():
            info.age = int(age_str)

        height_str = self._get_text(demo, "PatientHeightCM")
        if height_str:
            try:
                info.height = float(height_str)
            except ValueError:
                pass

        weight_str = self._get_text(demo, "PatientWeightKG")
        if weight_str:
            try:
                info.weight = float(weight_str)
            except ValueError:
                pass

        race_str = self._get_text(demo, "Race")
        if race_str:
            info.race = race_str

        return info

    def _read_recording(self, root: dict) -> RecordingInfo:
        info = RecordingInfo()

        test_demo = find_tag(root, "TestDemographics")
        if test_demo is None:
            return info

        if isinstance(test_demo, list):
            test_demo = test_demo[0]

        date_str = self._get_text(test_demo, "AcquisitionDate")
        time_str = self._get_text(test_demo, "AcquisitionTime")
        if date_str:
            dt = self._parse_date(date_str)
            if dt and time_str:
                try:
                    t = datetime.strptime(time_str, "%H:%M:%S")
                    dt = dt.replace(hour=t.hour, minute=t.minute, second=t.second)
                except ValueError:
                    pass
            info.date = dt

        info.location = (
            self._get_text(test_demo, "Location")
            or self._get_text(test_demo, "SiteName")
        )
        info.room = self._get_text(test_demo, "Room")

        return info

    def _read_device(self, root: dict) -> DeviceInfo:
        dev = DeviceInfo()

        test_demo = find_tag(root, "TestDemographics")
        if test_demo is not None:
            if isinstance(test_demo, list):
                test_demo = test_demo[0]
            dev.model = self._get_text(test_demo, "AcquisitionDevice")
            dev.software_version = self._get_text(
                test_demo, "AcquisitionSoftwareVersion"
            )
            dev.manufacturer = self._get_text(test_demo, "ManufacturerID")

        return dev

    def _read_leads(self, root: dict) -> tuple[list[Lead], list[Lead]]:
        """Parse waveform data and return ``(rhythm_leads, median_leads)``."""
        rhythm_leads: list[Lead] = []
        median_leads: list[Lead] = []

        waveform_nodes = find_tag(root, "Waveform")
        if waveform_nodes is None:
            return rhythm_leads, median_leads

        if isinstance(waveform_nodes, dict):
            waveform_nodes = [waveform_nodes]

        for wf_node in waveform_nodes:
            wf_type = self._get_text(wf_node, "WaveformType")

            sampling_rate_str = self._get_text(wf_node, "SampleBase")
            sampling_rate = (
                int(sampling_rate_str)
                if sampling_rate_str and sampling_rate_str.isdigit()
                else 500
            )

            lead_data_nodes = find_tag(wf_node, "LeadData")
            if lead_data_nodes is None:
                continue

            if isinstance(lead_data_nodes, dict):
                lead_data_nodes = [lead_data_nodes]

            wf_leads: list[Lead] = []
            for ld in lead_data_nodes:
                label = self._get_text(ld, "LeadID")
                if not label:
                    continue

                waveform_data = self._get_text(ld, "WaveFormData")
                if not waveform_data:
                    continue

                try:
                    samples = _decode_waveform_data(waveform_data.strip())
                except Exception:
                    continue

                scale = 1.0
                amp_units = self._get_text(ld, "LeadAmplitudeUnitsPerBit")
                if amp_units:
                    try:
                        scale = float(amp_units)
                    except ValueError:
                        pass

                raw = scale != 1.0
                res_unit = "uV" if raw else ""
                wf_leads.append(Lead(
                    label=label,
                    samples=samples,
                    sampling_rate=sampling_rate,
                    resolution=scale,
                    resolution_unit=res_unit,
                    units="" if raw else res_unit,
                    is_raw=raw,
                ))

            if wf_type and wf_type.lower() == "rhythm":
                rhythm_leads = wf_leads
            elif wf_type and wf_type.lower() == "median":
                median_leads = wf_leads
            else:
                if not rhythm_leads:
                    rhythm_leads = wf_leads

        return rhythm_leads, median_leads

    def _read_annotations(
        self, root: dict
    ) -> tuple[dict[str, str], GlobalMeasurements, Interpretation]:
        annotations: dict[str, str] = {}
        interp = Interpretation()

        def _extract_statements(node) -> list[str]:
            stmts: list[str] = []
            if node is None:
                return stmts
            if isinstance(node, dict):
                node = [node]
            if isinstance(node, list):
                for d in node:
                    stmt = find_tag(d, "DiagnosisStatement")
                    if stmt is not None:
                        if isinstance(stmt, dict):
                            stmt = [stmt]
                        if isinstance(stmt, list):
                            for s in stmt:
                                text = self._get_text(s, "StmtText")
                                if text:
                                    stmts.append((text, ""))
            return stmts

        overread = find_tag(root, "OverreadDiagnosis")
        if overread is None:
            overread = find_tag(root, "OverreadConfirmation")

        if overread is not None:
            overread_stmts = _extract_statements(overread)
            if overread_stmts:
                interp.statements = overread_stmts
                interp.source = "overread"
                annotations["diagnosis"] = "; ".join(s[0] for s in overread_stmts)
                physician = (
                    self._get_text(
                        overread if isinstance(overread, dict) else {},
                        "OverreaderLastName",
                    )
                    or self._get_text(
                        overread if isinstance(overread, dict) else {},
                        "EditorLastName",
                    )
                )
                if physician:
                    first = self._get_text(
                        overread if isinstance(overread, dict) else {},
                        "OverreaderFirstName",
                    ) or self._get_text(
                        overread if isinstance(overread, dict) else {},
                        "EditorFirstName",
                    )
                    interp.interpreter = (
                        f"{first} {physician}".strip() if first else physician
                    )

        diag = find_tag(root, "Diagnosis")
        original_stmts = _extract_statements(diag)
        if original_stmts:
            if not interp.statements:
                interp.statements = original_stmts
                interp.source = "machine"
                annotations["diagnosis"] = "; ".join(s[0] for s in original_stmts)
            else:
                annotations["original_diagnosis"] = "; ".join(s[0] for s in original_stmts)

        for left, _right in interp.statements:
            s_up = left.upper()
            if "NORMAL" in s_up and "ABNORMAL" not in s_up:
                interp.severity = "NORMAL"
            elif "ABNORMAL" in s_up:
                interp.severity = "ABNORMAL"
            elif "BORDERLINE" in s_up:
                interp.severity = "BORDERLINE"

        gm = GlobalMeasurements()

        meas_node = find_tag(root, "OriginalRestingECGMeasurements")
        if meas_node is None:
            meas_node = find_tag(root, "RestingECGMeasurements")

        if meas_node is not None:
            if isinstance(meas_node, list):
                meas_node = meas_node[0]

            mapping = {
                "VentricularRate": "heart_rate",
                "PRInterval": "pr_interval",
                "QRSDuration": "qrs_duration",
                "QTInterval": "qt_interval",
                "QTCorrected": "qtc_bazett",
                "PAxis": "p_axis",
                "RAxis": "qrs_axis",
                "TAxis": "t_axis",
                "QTCFredericia": "qtc_fridericia",
                "RRInterval": "rr_interval",
                "QRSCount": "qrs_count",
            }

            for xml_key, attr in mapping.items():
                val = self._get_text(meas_node, xml_key)
                if val:
                    annotations[xml_key.lower()] = val
                    try:
                        setattr(gm, attr, int(val))
                    except (ValueError, TypeError):
                        pass

        return annotations, gm, interp

    def _read_filters(self, root: dict) -> FilterSettings:
        """Extract filter settings from Waveform elements."""
        fs = FilterSettings()

        waveform_nodes = find_tag(root, "Waveform")
        if waveform_nodes is None:
            return fs

        if isinstance(waveform_nodes, dict):
            waveform_nodes = [waveform_nodes]

        for wf_node in waveform_nodes:
            hp = self._get_text(wf_node, "HighPassFilter")
            lp = self._get_text(wf_node, "LowPassFilter")
            nf = self._get_text(wf_node, "NotchFilterFrequency")
            gen = self._get_text(wf_node, "FilterSetting")

            if hp or lp or nf or gen:
                if hp:
                    try:
                        fs.highpass = float(hp)
                    except ValueError:
                        pass
                if lp:
                    try:
                        fs.lowpass = float(lp)
                    except ValueError:
                        pass
                if nf:
                    try:
                        fs.notch = float(nf)
                        fs.notch_active = True
                    except ValueError:
                        pass
                if gen and gen.lower() in ("on", "true", "yes"):
                    fs.artifact_filter = True
                elif gen and gen.lower() in ("off", "false", "no"):
                    fs.artifact_filter = False
                break

        return fs

    def _read_order_info(self, root: dict) -> dict:
        """Extract OrderInfo section into a dict for raw_metadata."""
        order: dict[str, str] = {}

        order_node = find_tag(root, "OrderInfo")
        if order_node is None:
            return order

        if isinstance(order_node, list):
            order_node = order_node[0]
        if not isinstance(order_node, dict):
            return order

        for tag, key in (
            ("OrderingPhysician", "ordering_physician"),
            ("OrderNumber", "order_number"),
            ("Reason", "reason"),
            ("OrderPriority", "priority"),
            ("ReferringPhysician", "referring_physician"),
        ):
            val = self._get_text(order_node, tag)
            if val:
                order[key] = val

        return order

    @staticmethod
    def _get_text(node: dict | None, tag: str) -> str:
        """Get text content of a tag from an xmltodict node."""
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

    @staticmethod
    def _parse_date(date_str: str) -> datetime | None:
        """Parse date strings in common GE MUSE formats."""
        for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", "%Y%m%d", "%d-%b-%Y"):
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        return None

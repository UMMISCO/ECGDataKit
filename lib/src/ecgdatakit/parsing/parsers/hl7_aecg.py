"""HL7 Annotated ECG (aECG) XML format parser."""

from __future__ import annotations

from datetime import datetime
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

LEAD_NAMES = [
    "MDC_ECG_LEAD_I", "MDC_ECG_LEAD_II", "MDC_ECG_LEAD_III",
    "MDC_ECG_LEAD_AVR", "MDC_ECG_LEAD_AVL", "MDC_ECG_LEAD_AVF",
    "MDC_ECG_LEAD_V1", "MDC_ECG_LEAD_V2", "MDC_ECG_LEAD_V3",
    "MDC_ECG_LEAD_V4", "MDC_ECG_LEAD_V5", "MDC_ECG_LEAD_V6",
    "MDC_ECG_LEAD_IIc",
]


def _parse_str_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    if "." in value:
        value = value.split(".")[0]
    try:
        return datetime.strptime(value, "%Y%m%d%H%M%S")
    except ValueError:
        return None


class HL7aECGParser(Parser):
    """Parser for HL7 Annotated ECG (aECG) XML files."""

    FORMAT_NAME = "HL7 aECG"
    FORMAT_DESCRIPTION = "HL7 Annotated ECG XML format (FDA submission standard)"
    FILE_EXTENSIONS = [".xml"]

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        try:
            text = header.decode("utf-8", errors="ignore")
            return b"<AnnotatedECG" in header or "<AnnotatedECG" in text
        except Exception:
            return False

    def parse(self, file_path: Path) -> ECGRecord:
        with open(file_path, "rb") as f:
            doc = xmltodict.parse(f.read())

        if "AnnotatedECG" not in doc:
            raise CorruptedFileError(f"Missing AnnotatedECG root element in {file_path}")

        record = ECGRecord(source_format="hl7_aecg")
        record.patient = self._read_patient(doc)
        record.recording = self._read_recording(doc)
        record.leads = self._read_leads(doc)
        record.device = self._read_device(doc)
        measurements, interpretation = self._read_annotations(doc)
        record.measurements = measurements
        record.interpretation = interpretation
        # Signal characteristics
        record.signal = SignalCharacteristics(
            data_encoding="decimal",
            compression="none",
            number_channels_valid=len(record.leads),
            number_channels_allocated=len(record.leads),
        )

        # Extract device serial number from assignedAuthor/id
        author_node = find_tag(doc, "author")
        if author_node is not None:
            if isinstance(author_node, list):
                author_node = author_node[0]
            assigned = find_tag(author_node, "assignedAuthor")
            if assigned is not None:
                if isinstance(assigned, list):
                    assigned = assigned[0]
                id_node = find_tag(assigned, "id")
                if id_node is not None:
                    if isinstance(id_node, list):
                        id_node = id_node[0]
                    if isinstance(id_node, dict):
                        ext = read_path(id_node, "@extension")
                        if ext and not record.device.serial_number:
                            record.device.serial_number = str(ext)

        # Extract custodian organization as institution
        custodian = find_tag(doc, "custodian")
        if custodian is not None:
            if isinstance(custodian, list):
                custodian = custodian[0]
            org_name = find_tag(custodian, "name")
            if org_name is not None and not record.device.institution:
                if isinstance(org_name, str):
                    record.device.institution = org_name
                elif isinstance(org_name, list) and org_name:
                    record.device.institution = str(org_name[0])

        # Extract patient race
        subject_node = find_tag(doc, "subject")
        if subject_node is not None:
            race_node = find_tag(subject_node, "raceCode")
            if race_node is not None:
                if isinstance(race_node, list):
                    race_node = race_node[0]
                if isinstance(race_node, dict):
                    race_val = read_path(race_node, "@displayName") or read_path(race_node, "@code")
                    if race_val:
                        record.patient.race = str(race_val)

        record.raw_metadata["filepath"] = str(file_path)

        return record

    def _read_patient(self, doc: dict) -> PatientInfo:
        info = PatientInfo()
        node = find_tag(doc, "subject")
        if node is None:
            return info

        id_node = find_tag(node, "id")
        if id_node is not None:
            if isinstance(id_node, list):
                id_node = id_node[0]
            if isinstance(id_node, dict):
                pid = read_path(id_node, "@extension") or read_path(id_node, "@root") or ""
                info.patient_id = str(pid)

        name_node = find_tag(node, "name")
        if name_node is not None:
            if isinstance(name_node, list):
                name_node = name_node[0]
            if isinstance(name_node, dict):
                family = find_tag(name_node, "family")
                given = find_tag(name_node, "given")
                if family:
                    info.last_name = str(family) if isinstance(family, str) else str(
                        family[0] if isinstance(family, list) else family
                    )
                if given:
                    info.first_name = str(given) if isinstance(given, str) else str(
                        given[0] if isinstance(given, list) else given
                    )

        gender_node = find_tag(node, "administrativeGenderCode")
        if gender_node is not None:
            code = read_path(gender_node, "@code")
            if code == "M":
                info.sex = "M"
            elif code == "F":
                info.sex = "F"
            else:
                info.sex = "U"

        birth_node = find_tag(node, "birthTime")
        if birth_node is not None:
            raw = read_path(birth_node, "@value")
            if raw and len(raw) >= 8:
                try:
                    info.birth_date = datetime.strptime(raw[:8], "%Y%m%d")
                except ValueError:
                    pass

        return info

    def _read_recording(self, doc: dict) -> RecordingInfo:
        info = RecordingInfo()
        info.date = _parse_str_datetime(
            read_path(doc, "AnnotatedECG/effectiveTime/low/@value")
        )
        end = _parse_str_datetime(
            read_path(doc, "AnnotatedECG/effectiveTime/high/@value")
        )
        info.end_date = end
        if info.date and end:
            info.duration = end - info.date

        increment_node = find_tag(doc, "increment")
        if increment_node is not None:
            if isinstance(increment_node, list):
                increment_node = increment_node[0]
            inc_val = None
            if isinstance(increment_node, dict):
                inc_val = read_path(increment_node, "value/@value") or read_path(
                    increment_node, "@value"
                )
            elif isinstance(increment_node, str):
                inc_val = increment_node
            if inc_val is not None:
                try:
                    increment_us = float(inc_val)
                    if increment_us > 0:
                        info.sample_rate = int(round(1_000_000 / increment_us))
                except (ValueError, ZeroDivisionError):
                    pass

        return info

    def _read_leads(self, doc: dict) -> list[Lead]:
        leads: list[Lead] = []
        series_node = find_tag(doc, "series")
        if series_node is None:
            return leads

        components = read_path(series_node, "component")
        if components is None:
            return leads

        if isinstance(components, list):
            for comp in components:
                leads.extend(self._parse_series(comp))
        else:
            leads.extend(self._parse_series(components))

        return leads

    def _parse_series(self, component: dict) -> list[Lead]:
        leads: list[Lead] = []
        nodes = read_path(component, "sequenceSet/component")
        if nodes is None:
            return leads

        if not isinstance(nodes, list):
            nodes = [nodes]

        for node in nodes:
            code = read_path(node, "sequence/code/@code")
            if code not in LEAD_NAMES:
                continue

            digits_str = read_path(node, "sequence/value/digits")
            if digits_str is None:
                continue

            raw_digits = digits_str.split(" ")
            signal = []
            for raw in raw_digits:
                elem = raw.strip()
                if elem:
                    signal.append(int(float(elem)))

            scale = 1.0
            scale_node = find_tag(node, "scale")
            if scale_node is not None:
                if isinstance(scale_node, list):
                    scale_node = scale_node[0]
                if isinstance(scale_node, dict):
                    s_val = read_path(scale_node, "@value")
                    if s_val is not None:
                        try:
                            scale = float(s_val)
                        except ValueError:
                            pass

            origin = 0.0
            origin_node = find_tag(node, "origin")
            if origin_node is not None:
                if isinstance(origin_node, list):
                    origin_node = origin_node[0]
                if isinstance(origin_node, dict):
                    o_val = read_path(origin_node, "@value")
                    if o_val is not None:
                        try:
                            origin = float(o_val)
                        except ValueError:
                            pass

            sample_rate = 0
            inc_node = find_tag(node, "increment")
            if inc_node is not None:
                if isinstance(inc_node, list):
                    inc_node = inc_node[0]
                inc_val = None
                if isinstance(inc_node, dict):
                    inc_val = read_path(inc_node, "@value")
                elif isinstance(inc_node, str):
                    inc_val = inc_node
                if inc_val is not None:
                    try:
                        increment_us = float(inc_val)
                        if increment_us > 0:
                            sample_rate = int(round(1_000_000 / increment_us))
                    except (ValueError, ZeroDivisionError):
                        pass

            label = code.replace("MDC_ECG_LEAD_", "").replace("AV", "aV")
            samples = np.array(signal, dtype=np.float64) * scale + origin
            leads.append(Lead(
                label=label,
                samples=samples,
                sample_rate=sample_rate,
                resolution=scale,
            ))

        return leads

    def _read_device(self, doc: dict) -> DeviceInfo:
        info = DeviceInfo()

        author_node = find_tag(doc, "author")
        if author_node is None:
            return info

        if isinstance(author_node, list):
            author_node = author_node[0]

        assigned = find_tag(author_node, "assignedAuthor")
        if assigned is None:
            return info

        if isinstance(assigned, list):
            assigned = assigned[0]

        device_node = find_tag(assigned, "assignedAuthoringDevice")
        if device_node is not None:
            if isinstance(device_node, list):
                device_node = device_node[0]

            model_name = find_tag(device_node, "manufacturerModelName")
            if model_name is not None:
                info.model = str(model_name) if isinstance(model_name, str) else str(
                    model_name[0] if isinstance(model_name, list) else
                    (read_path(model_name, "#text") or model_name)
                )

            software = find_tag(device_node, "softwareName")
            if software is not None:
                info.software_version = str(software) if isinstance(software, str) else str(
                    software[0] if isinstance(software, list) else
                    (read_path(software, "#text") or software)
                )

        org_node = find_tag(assigned, "representedOrganization")
        if org_node is not None:
            if isinstance(org_node, list):
                org_node = org_node[0]
            org_name = find_tag(org_node, "name")
            if org_name is not None:
                info.manufacturer = str(org_name) if isinstance(org_name, str) else str(
                    org_name[0] if isinstance(org_name, list) else
                    (read_path(org_name, "#text") or org_name)
                )

        return info

    _MEASUREMENT_MAP: dict[str, str] = {
        "MDC_ECG_HEART_RATE": "heart_rate",
        "MDC_ECG_TIME_PD_RR": "rr_interval",
        "MDC_ECG_TIME_PD_PR": "pr_interval",
        "MDC_ECG_TIME_PD_QRS": "qrs_duration",
        "MDC_ECG_TIME_PD_QT": "qt_interval",
        "MDC_ECG_TIME_PD_QTc": "qtc_bazett",
        "MDC_ECG_TIME_PD_QTcb": "qtc_bazett",
        "MDC_ECG_TIME_PD_QTcf": "qtc_fridericia",
        "MDC_ECG_ANGLE_P_FRONT": "p_axis",
        "MDC_ECG_ANGLE_QRS_FRONT": "qrs_axis",
        "MDC_ECG_ANGLE_T_FRONT": "t_axis",
        "MDC_ECG_QRS_COUNT": "qrs_count",
    }

    def _read_annotations(
        self, doc: dict
    ) -> tuple[GlobalMeasurements, Interpretation]:
        measurements = GlobalMeasurements()
        interpretation = Interpretation()

        subject_of_nodes = find_tag(doc, "subjectOf")
        if subject_of_nodes is None:
            return measurements, interpretation

        if not isinstance(subject_of_nodes, list):
            subject_of_nodes = [subject_of_nodes]

        for subject_of in subject_of_nodes:
            annotations = find_tag(subject_of, "annotation")
            if annotations is None:
                continue
            if not isinstance(annotations, list):
                annotations = [annotations]

            for ann in annotations:
                if not isinstance(ann, dict):
                    continue
                self._process_annotation(ann, measurements, interpretation)

        return measurements, interpretation

    def _process_annotation(
        self,
        ann: dict,
        measurements: GlobalMeasurements,
        interpretation: Interpretation,
    ) -> None:
        code = read_path(ann, "code/@code") or ""
        display_name = read_path(ann, "code/@displayName") or ""

        value_node = read_path(ann, "value")
        ann_value: str | None = None
        if value_node is not None:
            if isinstance(value_node, dict):
                ann_value = read_path(value_node, "@value") or read_path(value_node, "#text")
            elif isinstance(value_node, str):
                ann_value = value_node

        if code == "MDC_ECG_INTERPRETATION" or "INTERPRETATION" in code.upper():
            text = ann_value or display_name
            if text:
                interpretation.statements.append(text)
                interpretation.source = "machine"
            self._collect_nested_interpretation(ann, interpretation)
        elif code in self._MEASUREMENT_MAP:
            field_name = self._MEASUREMENT_MAP[code]
            if ann_value is not None:
                try:
                    setattr(measurements, field_name, int(round(float(ann_value))))
                except (ValueError, TypeError):
                    pass
        elif code in ("MDC_ECG_INTERPRETATION_SEVERITY", "MDC_ECG_RESULT"):
            if ann_value:
                interpretation.severity = ann_value.upper()

        components = read_path(ann, "component")
        if components is not None:
            if not isinstance(components, list):
                components = [components]
            for comp in components:
                nested = find_tag(comp, "annotation")
                if nested is None:
                    continue
                if not isinstance(nested, list):
                    nested = [nested]
                for nested_ann in nested:
                    if isinstance(nested_ann, dict):
                        self._process_annotation(
                            nested_ann, measurements, interpretation
                        )

    def _collect_nested_interpretation(
        self, ann: dict, interpretation: Interpretation
    ) -> None:
        """Collect interpretation text from nested component annotations."""
        components = read_path(ann, "component")
        if components is None:
            return
        if not isinstance(components, list):
            components = [components]

        for comp in components:
            nested = find_tag(comp, "annotation")
            if nested is None:
                continue
            if not isinstance(nested, list):
                nested = [nested]
            for nested_ann in nested:
                if not isinstance(nested_ann, dict):
                    continue
                val_node = read_path(nested_ann, "value")
                text = None
                if val_node is not None:
                    if isinstance(val_node, dict):
                        text = read_path(val_node, "@value") or read_path(val_node, "#text")
                    elif isinstance(val_node, str):
                        text = val_node
                if text and text not in interpretation.statements:
                    interpretation.statements.append(text)
